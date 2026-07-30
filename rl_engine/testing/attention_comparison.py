# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-GPU WS2 attention cross-implementation comparison harness.

This module compares logically equivalent attention materializations before CP
communication is introduced.  The full path is the training-style reference.
The chunked-query and paged-KV paths emulate rollout-style prefill layouts on a
single device while preserving global causal positions and attention-domain LSE.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from rl_engine.testing.reference_ops import selected_logprobs_reference

MergeBackend = Literal["rl_kernel", "transformer_engine"]

_TE_CONTEXT_PARALLEL_MODULE = (
    "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"
)


class TransformerEngineUnavailable(RuntimeError):
    """Raised when the optional Transformer Engine oracle cannot be imported."""


@dataclass(frozen=True)
class AttentionComparisonInputs:
    """Inputs shared by every single-GPU attention comparison path."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    causal: bool = True
    scale: float | None = None
    key_padding_mask: torch.Tensor | None = None
    lm_head_weight: torch.Tensor | None = None
    target_ids: torch.Tensor | None = None
    active_token_mask: torch.Tensor | None = None
    output_dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class AttentionPathResult:
    """One materialized attention path result."""

    name: str
    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DriftStats:
    """Shape-aware absolute drift summary."""

    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    active_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p95_abs": self.p95_abs,
            "p99_abs": self.p99_abs,
            "active_count": self.active_count,
        }


@dataclass(frozen=True)
class AttentionPathDrift:
    """Candidate-vs-reference drift for one attention path."""

    candidate_name: str
    out: DriftStats
    lse: DriftStats
    dlogp: DriftStats | None
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "out": self.out.to_dict(),
            "lse": self.lse.to_dict(),
            "dlogp": None if self.dlogp is None else self.dlogp.to_dict(),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AttentionComparisonReport:
    """Structured report for PR2 single-GPU attention attribution."""

    reference_name: str
    drifts: tuple[AttentionPathDrift, ...]
    unavailable: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_name": self.reference_name,
            "drifts": [drift.to_dict() for drift in self.drifts],
            "unavailable": list(self.unavailable),
        }


@dataclass(frozen=True)
class _PartialAttentionState:
    out: torch.Tensor
    lse: torch.Tensor
    block_start: int
    block_end: int


def compare_single_gpu_attention(
    inputs: AttentionComparisonInputs,
    *,
    query_chunk_size: int | None = None,
    kv_page_size: int | None = None,
    include_transformer_engine: bool = False,
) -> AttentionComparisonReport:
    """Compare full attention with chunked/paged single-GPU materializations.

    If ``lm_head_weight`` and ``target_ids`` are provided, the report also
    includes active-token selected-logprob drift using the #207 convention:
    candidate logp minus reference logp.
    """

    _validate_comparison_inputs(inputs)
    reference = run_full_attention(inputs)
    candidates = [
        run_chunked_query_attention(inputs, query_chunk_size=query_chunk_size),
        run_paged_kv_attention(inputs, kv_page_size=kv_page_size, merge_backend="rl_kernel"),
    ]
    unavailable: list[str] = []
    if include_transformer_engine:
        try:
            candidates.append(
                run_paged_kv_attention(
                    inputs,
                    kv_page_size=kv_page_size,
                    merge_backend="transformer_engine",
                )
            )
        except TransformerEngineUnavailable as exc:
            unavailable.append(f"transformer_engine_paged_kv: {exc}")

    drifts = tuple(_compare_path(candidate, reference, inputs) for candidate in candidates)
    return AttentionComparisonReport(
        reference_name=reference.name,
        drifts=drifts,
        unavailable=tuple(unavailable),
    )


def run_full_attention(inputs: AttentionComparisonInputs) -> AttentionPathResult:
    """Training-style full-sequence attention with exported attention-domain LSE."""

    out, lse = _attention_with_lse(
        inputs.q,
        inputs.k,
        inputs.v,
        causal=inputs.causal,
        scale=inputs.scale,
        key_padding_mask=inputs.key_padding_mask,
        q_start=0,
        k_start=0,
        total_query_len=inputs.q.size(2),
        total_kv_len=inputs.k.size(2),
        output_dtype=inputs.output_dtype,
    )
    return AttentionPathResult(
        name="full_prefill",
        out=out,
        lse=lse,
        provenance={
            "attention_mode": "prefill",
            "materialization": "full_sequence",
            "lse_domain": "attention",
        },
    )


def run_chunked_query_attention(
    inputs: AttentionComparisonInputs,
    *,
    query_chunk_size: int | None,
) -> AttentionPathResult:
    """Rollout-style chunked prefill replay over full KV on one device."""

    sq = inputs.q.size(2)
    chunk_size = (
        sq if query_chunk_size is None else _positive_int(query_chunk_size, "query_chunk_size")
    )
    out_chunks: list[torch.Tensor] = []
    lse_chunks: list[torch.Tensor] = []
    chunk_bounds = _chunk_bounds(sq, chunk_size)
    for q_start, q_end in chunk_bounds:
        out, lse = _attention_with_lse(
            inputs.q[:, :, q_start:q_end, :],
            inputs.k,
            inputs.v,
            causal=inputs.causal,
            scale=inputs.scale,
            key_padding_mask=inputs.key_padding_mask,
            q_start=q_start,
            k_start=0,
            total_query_len=sq,
            total_kv_len=inputs.k.size(2),
            output_dtype=inputs.output_dtype,
        )
        out_chunks.append(out)
        lse_chunks.append(lse)

    return AttentionPathResult(
        name="chunked_prefill",
        out=torch.cat(out_chunks, dim=2),
        lse=torch.cat(lse_chunks, dim=2),
        provenance={
            "attention_mode": "chunked_prefill",
            "materialization": "query_chunks",
            "query_chunk_size": chunk_size,
            "chunk_bounds": [list(bound) for bound in chunk_bounds],
            "lse_domain": "attention",
        },
    )


def run_paged_kv_attention(
    inputs: AttentionComparisonInputs,
    *,
    kv_page_size: int | None,
    merge_backend: MergeBackend = "rl_kernel",
) -> AttentionPathResult:
    """Rollout-style paged-KV prefill replay with explicit LSE merge."""

    skv = inputs.k.size(2)
    page_size = skv if kv_page_size is None else _positive_int(kv_page_size, "kv_page_size")
    states: list[_PartialAttentionState] = []
    page_bounds = _chunk_bounds(skv, page_size)
    for k_start, k_end in page_bounds:
        key_mask = (
            None if inputs.key_padding_mask is None else inputs.key_padding_mask[:, k_start:k_end]
        )
        out, lse = _attention_with_lse(
            inputs.q,
            inputs.k[:, :, k_start:k_end, :],
            inputs.v[:, :, k_start:k_end, :],
            causal=inputs.causal,
            scale=inputs.scale,
            key_padding_mask=key_mask,
            q_start=0,
            k_start=k_start,
            total_query_len=inputs.q.size(2),
            total_kv_len=skv,
            output_dtype=torch.float32,
        )
        states.append(
            _PartialAttentionState(
                out=out,
                lse=lse,
                block_start=k_start,
                block_end=k_end,
            )
        )

    out, lse = _merge_partial_states(states, backend=merge_backend)
    return AttentionPathResult(
        name=f"{merge_backend}_paged_kv",
        out=out.to(inputs.output_dtype),
        lse=lse,
        provenance={
            "attention_mode": "prefill",
            "materialization": "paged_kv",
            "kv_page_size": page_size,
            "kv_page_bounds": [list(bound) for bound in page_bounds],
            "merge_backend": merge_backend,
            "merge_order": "global_block_index",
            "lse_domain": "attention",
        },
    )


def transformer_engine_context_parallel_available() -> bool:
    """Return whether the optional TE context-parallel helper module imports."""

    try:
        _load_te_context_parallel()
    except TransformerEngineUnavailable:
        return False
    return True


def _compare_path(
    candidate: AttentionPathResult,
    reference: AttentionPathResult,
    inputs: AttentionComparisonInputs,
) -> AttentionPathDrift:
    dlogp = None
    if inputs.lm_head_weight is not None and inputs.target_ids is not None:
        candidate_logp = _selected_logps_from_attention(candidate.out, inputs)
        reference_logp = _selected_logps_from_attention(reference.out, inputs)
        dlogp = _drift_stats(candidate_logp, reference_logp, mask=inputs.active_token_mask)

    return AttentionPathDrift(
        candidate_name=candidate.name,
        out=_drift_stats(candidate.out, reference.out),
        lse=_drift_stats(candidate.lse, reference.lse),
        dlogp=dlogp,
        provenance=candidate.provenance,
    )


def _attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float | None,
    key_padding_mask: torch.Tensor | None,
    q_start: int,
    k_start: int,
    total_query_len: int,
    total_kv_len: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_qkv(q, k, v)
    if key_padding_mask is not None:
        if key_padding_mask.shape != (q.size(0), k.size(2)):
            raise ValueError("key_padding_mask must have shape [B, local_skv]")
        if key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be bool")

    qf, kf, vf = q.float(), k.float(), v.float()
    hq, sq, dim = qf.shape[1], qf.shape[2], qf.shape[3]
    hkv, skv = kf.shape[1], kf.shape[2]
    if hkv != hq:
        repeat = hq // hkv
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)

    scale_value = scale if scale is not None else 1.0 / math.sqrt(dim)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale_value
    if causal:
        query_offset = total_kv_len - total_query_len
        q_pos = torch.arange(sq, device=q.device) + q_start + query_offset
        k_pos = torch.arange(skv, device=q.device) + k_start
        scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    finite_lse = torch.isfinite(lse)
    weights = torch.exp(scores - lse.unsqueeze(-1))
    weights = torch.where(finite_lse.unsqueeze(-1), weights, torch.zeros_like(weights))
    out = torch.matmul(weights, vf)
    return out.to(output_dtype), lse


def _merge_partial_states(
    states: list[_PartialAttentionState],
    *,
    backend: MergeBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not states:
        raise ValueError("at least one partial state is required")
    ordered = sorted(states, key=lambda state: (state.block_start, state.block_end))
    _validate_partial_states(ordered)
    if backend == "rl_kernel":
        return _merge_partial_states_rl_kernel(ordered)
    if backend == "transformer_engine":
        return _merge_partial_states_transformer_engine(ordered)
    raise ValueError(f"unsupported merge backend: {backend}")


def _merge_partial_states_rl_kernel(
    states: list[_PartialAttentionState],
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_out = states[0].out.float()
    merged_lse = states[0].lse.float()
    for state in states[1:]:
        next_lse = torch.logaddexp(merged_lse, state.lse.float())
        finite = torch.isfinite(next_lse)
        weight_prev = torch.where(
            finite,
            torch.exp(merged_lse - next_lse),
            torch.zeros_like(next_lse),
        )
        weight_next = torch.where(
            finite,
            torch.exp(state.lse.float() - next_lse),
            torch.zeros_like(next_lse),
        )
        merged_out = (
            weight_prev.unsqueeze(-1) * merged_out + weight_next.unsqueeze(-1) * state.out.float()
        )
        merged_lse = next_lse
    return merged_out, merged_lse


def _merge_partial_states_transformer_engine(
    states: list[_PartialAttentionState],
) -> tuple[torch.Tensor, torch.Tensor]:
    te_cp = _load_te_context_parallel()
    merged_out = states[0].out.float()
    merged_lse = states[0].lse.float()
    for state in states[1:]:
        previous_lse = merged_lse
        merged_lse = previous_lse.clone()
        te_cp.flash_attn_fwd_softmax_lse_correction(merged_lse, state.lse.float())
        merged_out = te_cp.flash_attn_fwd_out_correction_init(
            merged_out,
            merged_lse,
            previous_lse,
            seq_dim=2,
        )
        te_cp.flash_attn_fwd_out_correction(
            merged_out,
            state.out.float(),
            merged_lse,
            state.lse.float(),
            seq_dim=2,
        )
    return merged_out, merged_lse


def _load_te_context_parallel() -> Any:
    try:
        return importlib.import_module(_TE_CONTEXT_PARALLEL_MODULE)
    except (ImportError, OSError, RuntimeError) as exc:
        raise TransformerEngineUnavailable(str(exc)) from exc


def _selected_logps_from_attention(
    out: torch.Tensor,
    inputs: AttentionComparisonInputs,
) -> torch.Tensor:
    if inputs.lm_head_weight is None or inputs.target_ids is None:
        raise ValueError("lm_head_weight and target_ids are required for dlogp drift")
    batch, heads, seq, dim = out.shape
    hidden = out.transpose(1, 2).reshape(batch, seq, heads * dim)
    if inputs.lm_head_weight.shape[1] != hidden.size(-1):
        raise ValueError(
            "lm_head_weight hidden dimension must equal Hq * D; "
            f"got {inputs.lm_head_weight.shape[1]} and {hidden.size(-1)}"
        )
    logits = torch.matmul(hidden.float(), inputs.lm_head_weight.float().transpose(0, 1))
    return selected_logprobs_reference(
        logits,
        inputs.target_ids,
        mask=inputs.active_token_mask,
        output_dtype=torch.float32,
    )


def _drift_stats(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> DriftStats:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match "
            f"reference shape {tuple(reference.shape)}"
        )
    diff = (candidate.float() - reference.float()).abs()
    values = _active_values(diff, mask)
    active_count = int(values.numel())
    if active_count == 0:
        return DriftStats(0.0, 0.0, 0.0, 0.0, 0)
    return DriftStats(
        max_abs=float(values.max().item()),
        mean_abs=float(values.mean().item()),
        p95_abs=float(torch.quantile(values, 0.95).item()),
        p99_abs=float(torch.quantile(values, 0.99).item()),
        active_count=active_count,
    )


def _active_values(diff: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return diff.reshape(-1)
    if mask.shape == diff.shape:
        return diff[mask.to(device=diff.device, dtype=torch.bool)]
    if mask.ndim == 2 and diff.ndim == 4 and mask.shape == (diff.size(0), diff.size(2)):
        expanded = mask[:, None, :, None].expand_as(diff)
        return diff[expanded.to(device=diff.device, dtype=torch.bool)]
    if mask.ndim == 2 and diff.ndim == 3 and mask.shape == (diff.size(0), diff.size(2)):
        expanded = mask[:, None, :].expand_as(diff)
        return diff[expanded.to(device=diff.device, dtype=torch.bool)]
    raise ValueError(f"mask shape {tuple(mask.shape)} cannot select diff shape {tuple(diff.shape)}")


def _validate_comparison_inputs(inputs: AttentionComparisonInputs) -> None:
    _validate_qkv(inputs.q, inputs.k, inputs.v)
    if inputs.key_padding_mask is not None:
        if inputs.key_padding_mask.shape != (inputs.q.size(0), inputs.k.size(2)):
            raise ValueError("key_padding_mask must have shape [B, Skv]")
        if inputs.key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be bool")
    if (inputs.lm_head_weight is None) != (inputs.target_ids is None):
        raise ValueError("lm_head_weight and target_ids must be provided together")
    if inputs.target_ids is not None and inputs.target_ids.shape != (
        inputs.q.size(0),
        inputs.q.size(2),
    ):
        raise ValueError("target_ids must have shape [B, Sq]")
    if inputs.active_token_mask is not None:
        if inputs.active_token_mask.shape != (inputs.q.size(0), inputs.q.size(2)):
            raise ValueError("active_token_mask must have shape [B, Sq]")
        if inputs.active_token_mask.dtype != torch.bool:
            raise ValueError("active_token_mask must be bool")


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have matching shape")
    if q.size(0) != k.size(0) or q.size(3) != k.size(3):
        raise ValueError("q, k, and v must share batch size and head dim")
    if q.size(1) % k.size(1) != 0:
        raise ValueError(f"Hq={q.size(1)} must be divisible by Hkv={k.size(1)}")


def _validate_partial_states(states: list[_PartialAttentionState]) -> None:
    first = states[0]
    previous_end = first.block_end
    for state in states[1:]:
        if state.out.shape != first.out.shape or state.lse.shape != first.lse.shape:
            raise ValueError("all partial states must have matching shapes")
        if state.block_start < previous_end:
            raise ValueError("partial state block ranges must not overlap")
        previous_end = state.block_end


def _chunk_bounds(length: int, chunk_size: int) -> list[tuple[int, int]]:
    if length <= 0:
        raise ValueError("sequence length must be positive")
    bounds: list[tuple[int, int]] = []
    cursor = 0
    while cursor < length:
        end = min(cursor + chunk_size, length)
        bounds.append((cursor, end))
        cursor = end
    return bounds


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


__all__ = [
    "AttentionComparisonInputs",
    "AttentionComparisonReport",
    "AttentionPathDrift",
    "AttentionPathResult",
    "DriftStats",
    "TransformerEngineUnavailable",
    "compare_single_gpu_attention",
    "run_chunked_query_attention",
    "run_full_attention",
    "run_paged_kv_attention",
    "transformer_engine_context_parallel_available",
]
