# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic context-parallel attention reference.

This module is the correctness-first WS2 reference for CP-aware standard
softmax attention. It intentionally stays in PyTorch and uses fp32 partial
states so fused CUDA/Triton backends can validate their CP/LSE merge semantics
against a small, inspectable implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp


@dataclass(frozen=True)
class AttentionPartialState:
    """One KV block's attention state before deterministic LSE merge.

    ``out`` is already normalized within the local KV block and has shape
    ``[B, Hq, Sq, D]``. ``lse`` is the local attention-domain log-sum-exp with
    shape ``[B, Hq, Sq]``. ``block_start`` / ``block_end`` are logical global KV
    positions and define the canonical merge order.
    """

    out: torch.Tensor
    lse: torch.Tensor
    block_start: int
    block_end: int

    def __post_init__(self) -> None:
        if self.out.ndim != 4:
            raise ValueError("partial attention out must have shape [B, Hq, Sq, D]")
        if self.lse.shape != self.out.shape[:3]:
            raise ValueError("partial attention lse must have shape [B, Hq, Sq]")
        if self.block_start < 0:
            raise ValueError("block_start must be non-negative")
        if self.block_end < self.block_start:
            raise ValueError("block_end must be >= block_start")


def merge_attention_partial_states(
    states: Sequence[AttentionPartialState],
) -> AttentionPartialState:
    """Merge CP/chunk partial states in logical block order.

    The merge is the online-softmax/LSE merge used by attention, not a plain
    sum. The input order is deliberately ignored: states are sorted by logical
    ``block_start`` so the result depends on global block indices rather than
    arrival order.
    """

    if not states:
        raise ValueError("at least one attention partial state is required")

    ordered = sorted(states, key=lambda item: (item.block_start, item.block_end))
    _validate_merge_shapes_and_ranges(ordered)

    merged = ordered[0]
    merged_out = merged.out.float()
    merged_lse = merged.lse.float()
    for state in ordered[1:]:
        merged_out, merged_lse = _merge_two_states(
            merged_out,
            merged_lse,
            state.out.float(),
            state.lse.float(),
        )

    return AttentionPartialState(
        out=merged_out,
        lse=merged_lse,
        block_start=ordered[0].block_start,
        block_end=ordered[-1].block_end,
    )


class DeterministicCPAttentionReferenceOp:
    """Correctness-first CP attention reference for prefill and chunked prefill.

    The op emulates CP by splitting query and KV sequence dimensions into
    logical CP shards. Each query shard computes one partial attention state per
    KV block, then merges those states in fixed global-block order using fp32
    LSE arithmetic. ``forward`` returns the input dtype after the final write;
    ``forward_fp32`` keeps the fp32 merged output.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute CP attention with fp32 accumulation and final input-dtype write."""

        out, _ = self.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            output_dtype=q.dtype,
        )
        return out

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute CP attention with fp32 accumulation and fp32 output."""

        out, _ = self.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )
        return out

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(out, lse)`` for the CP reference path.

        ``lse`` is always fp32 and in the attention domain. ``out`` is fp32
        until the final write, then downcast to ``output_dtype``. When omitted,
        ``output_dtype`` defaults to the input dtype.
        """

        out, lse = self._forward_impl(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )
        out = out.to(q.dtype if output_dtype is None else output_dtype)
        return out, lse

    def forward_fp32_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fp32 ``(out, lse)`` for the CP reference path."""

        return self.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            output_dtype=torch.float32,
        )

    def local_partial_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        q_start: int,
        k_start: int,
        total_kv_len: int,
        total_query_len: Optional[int] = None,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
    ) -> AttentionPartialState:
        """Compute one query shard against one logical KV block.

        ``query_position_offsets`` and ``key_position_offsets`` are optional
        per-batch-row base positions. They let the reference express varlen or
        packed metadata while retaining the dense [B, H, S, D] tensor layout.
        """

        _validate_qkv(q, k, v)
        if q_start < 0 or k_start < 0:
            raise ValueError("q_start and k_start must be non-negative")
        if total_kv_len < k_start + k.size(2):
            raise ValueError("total_kv_len must cover the local KV block")
        if total_query_len is None:
            total_query_len = q.size(2)
        if total_query_len < q_start + q.size(2):
            raise ValueError("total_query_len must cover the local query block")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (q.size(0), k.size(2)):
                raise ValueError("local key_padding_mask must have shape [B, local_skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("local key_padding_mask must be bool")
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            q.size(0),
            q.device,
            default=total_kv_len - total_query_len,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            q.size(0),
            q.device,
            default=0,
            name="key_position_offsets",
        )

        ctx = NativeAttentionOp._strict_fp32_math(q.device.type)
        with ctx:
            qf = q.float()
            kf = k.float()
            vf = v.float()
            hq, sq, dim = qf.shape[1], qf.shape[2], qf.shape[3]
            hkv, skv = kf.shape[1], kf.shape[2]
            if hkv != hq:
                repeat = hq // hkv
                kf = kf.repeat_interleave(repeat, dim=1)
                vf = vf.repeat_interleave(repeat, dim=1)

            if skv == 0:
                zero_dep = _zero_dependency(qf, kf, vf)
                return AttentionPartialState(
                    out=torch.zeros(q.size(0), hq, sq, dim, device=q.device, dtype=torch.float32)
                    + zero_dep,
                    lse=torch.full(
                        (q.size(0), hq, sq),
                        float("-inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    + zero_dep,
                    block_start=k_start,
                    block_end=k_start,
                )

            scale_value = scale if scale is not None else (1.0 / math.sqrt(dim))
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale_value
            if causal:
                query_base = query_offsets[:, None] + q_start
                key_base = key_offsets[:, None] + k_start
                q_pos = torch.arange(sq, device=q.device, dtype=torch.long) + query_base
                k_pos = torch.arange(skv, device=q.device, dtype=torch.long) + key_base
                causal_mask = k_pos[:, None, :] > q_pos[:, :, None]
                scores = scores.masked_fill(causal_mask[:, None, :, :], float("-inf"))
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

            lse = torch.logsumexp(scores, dim=-1)
            finite_lse = torch.isfinite(lse)
            weights = torch.exp(scores - lse.unsqueeze(-1))
            weights = torch.where(finite_lse.unsqueeze(-1), weights, torch.zeros_like(weights))
            out = torch.matmul(weights, vf)
            return AttentionPartialState(
                out=out,
                lse=lse,
                block_start=k_start,
                block_end=k_start + skv,
            )

    def _forward_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: Optional[float],
        key_padding_mask: Optional[torch.Tensor],
        query_position_offsets: Optional[torch.Tensor],
        key_position_offsets: Optional[torch.Tensor],
        cp_world_size: int,
        kv_chunk_size: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_qkv(q, k, v)
        if cp_world_size < 1:
            raise ValueError("cp_world_size must be >= 1")
        if kv_chunk_size is not None and kv_chunk_size < 1:
            raise ValueError("kv_chunk_size must be >= 1 when provided")

        batch, hq, sq, dim = q.shape
        skv = k.size(2)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, skv):
                raise ValueError("key_padding_mask must have shape [B, Skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            batch,
            q.device,
            default=skv - sq,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            batch,
            q.device,
            default=0,
            name="key_position_offsets",
        )

        q_bounds = _split_bounds(sq, cp_world_size)
        kv_bounds = _kv_block_bounds(skv, cp_world_size, kv_chunk_size)
        out_chunks: list[torch.Tensor] = []
        lse_chunks: list[torch.Tensor] = []
        for q_start, q_end in q_bounds:
            if q_start == q_end:
                continue
            q_block = q[:, :, q_start:q_end, :]
            states = [
                self.local_partial_state(
                    q_block,
                    k[:, :, k_start:k_end, :],
                    v[:, :, k_start:k_end, :],
                    q_start=q_start,
                    k_start=k_start,
                    total_kv_len=skv,
                    total_query_len=sq,
                    causal=causal,
                    scale=scale,
                    key_padding_mask=(
                        None if key_padding_mask is None else key_padding_mask[:, k_start:k_end]
                    ),
                    query_position_offsets=query_offsets,
                    key_position_offsets=key_offsets,
                )
                for k_start, k_end in kv_bounds
                if k_start != k_end
            ]
            if states:
                merged = merge_attention_partial_states(states)
                out_chunks.append(merged.out)
                lse_chunks.append(merged.lse)
            else:
                zero_dep = _zero_dependency(q_block.float(), k.float(), v.float())
                out_chunks.append(
                    torch.zeros(batch, hq, q_end - q_start, dim, device=q.device) + zero_dep
                )
                lse_chunks.append(
                    torch.full(
                        (batch, hq, q_end - q_start),
                        float("-inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    + zero_dep
                )

        if not out_chunks:
            zero_dep = _zero_dependency(q.float(), k.float(), v.float())
            return (
                torch.empty(batch, hq, 0, dim, device=q.device, dtype=torch.float32) + zero_dep,
                torch.empty(batch, hq, 0, device=q.device, dtype=torch.float32) + zero_dep,
            )
        return torch.cat(out_chunks, dim=2), torch.cat(lse_chunks, dim=2)


def _merge_two_states(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_lse = torch.logaddexp(lse_a, lse_b)
    finite = torch.isfinite(merged_lse)
    weight_a = torch.where(finite, torch.exp(lse_a - merged_lse), torch.zeros_like(merged_lse))
    weight_b = torch.where(finite, torch.exp(lse_b - merged_lse), torch.zeros_like(merged_lse))
    merged_out = weight_a.unsqueeze(-1) * out_a + weight_b.unsqueeze(-1) * out_b
    return merged_out, merged_lse


def _validate_merge_shapes_and_ranges(states: Sequence[AttentionPartialState]) -> None:
    first = states[0]
    previous_end = first.block_end
    for state in states[1:]:
        if state.out.shape != first.out.shape or state.lse.shape != first.lse.shape:
            raise ValueError("all partial states must have matching out/lse shapes")
        if state.block_start < previous_end:
            raise ValueError("partial state block ranges must not overlap")
        previous_end = state.block_end


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.size(0) != k.size(0) or q.size(3) != k.size(3):
        raise ValueError("q, k, and v must share batch size and head dim")
    if q.size(1) % k.size(1) != 0:
        raise ValueError(f"Hq={q.size(1)} not divisible by Hkv={k.size(1)} (GQA group)")


def _zero_dependency(*tensors: torch.Tensor) -> torch.Tensor:
    total = torch.tensor(0.0, device=tensors[0].device)
    for tensor in tensors:
        total = total + tensor.sum()
    return total * 0.0


def _normalize_position_offsets(
    offsets: Optional[torch.Tensor],
    batch: int,
    device: torch.device,
    *,
    default: int,
    name: str,
) -> torch.Tensor:
    if offsets is None:
        return torch.full((batch,), default, dtype=torch.long, device=device)
    if offsets.ndim != 1 or offsets.numel() != batch:
        raise ValueError(f"{name} must have shape [B]")
    if torch.is_floating_point(offsets) or torch.is_complex(offsets) or offsets.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer positions")
    return offsets.to(device=device, dtype=torch.long)


def _split_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(length, parts)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        width = base + (1 if index < extra else 0)
        end = start + width
        bounds.append((start, end))
        start = end
    return bounds


def _kv_block_bounds(
    length: int,
    cp_world_size: int,
    kv_chunk_size: Optional[int],
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for start, end in _split_bounds(length, cp_world_size):
        if kv_chunk_size is None:
            bounds.append((start, end))
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + kv_chunk_size, end)
            bounds.append((cursor, chunk_end))
            cursor = chunk_end
    return bounds


CPAttentionReferenceOp = DeterministicCPAttentionReferenceOp

__all__ = [
    "AttentionPartialState",
    "CPAttentionReferenceOp",
    "DeterministicCPAttentionReferenceOp",
    "merge_attention_partial_states",
]
