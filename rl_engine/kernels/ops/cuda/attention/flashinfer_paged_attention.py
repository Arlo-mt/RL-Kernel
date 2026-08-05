# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""FlashInfer paged-attention candidate for WS2 PR7.

This module is intentionally opt-in.  It adapts RL-Kernel's
``[B, H, S, D]`` attention tensors and PR6-style paged-KV metadata to
FlashInfer's paged attention wrappers, while recording the three PR7 contract
choices that affect rollout/training alignment:

* Qwen3-exact RoPE fused into attention through ``ROPE_LLAMA``;
* split-KV policy, with auto split rejected when batch invariance is required;
* LSE export and provenance for downstream drift reports.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
)

RoPEState = Literal["pre_rope", "post_rope"]
FlashInferAttentionMode = Literal["prefill", "decode"]
SplitKVMode = Literal["disabled", "fixed", "auto"]

_FLASHINFER_MODULE = "flashinfer"


class FlashInferUnavailable(RuntimeError):
    """Raised when FlashInfer cannot be imported or lacks required symbols."""


@dataclass(frozen=True)
class FlashInferRoPEFusionConfig:
    """Qwen3 RoPE settings used when FlashInfer performs RoPE inside attention."""

    pos_encoding_mode: str = "ROPE_LLAMA"
    rope_theta: float = 1_000_000.0
    rope_scale: float = 1.0
    rotary_dim: int | None = None
    q_rope_state: RoPEState = "pre_rope"
    k_cache_rope_state: RoPEState = "pre_rope"

    def validate(self, head_dim: int) -> None:
        if self.pos_encoding_mode != "ROPE_LLAMA":
            raise ValueError("PR7 RoPE fusion requires FlashInfer pos_encoding_mode='ROPE_LLAMA'")
        if float(self.rope_theta) != 1_000_000.0:
            raise ValueError("Qwen3-8B RoPE fusion requires rope_theta=1_000_000.0")
        if float(self.rope_scale) != 1.0:
            raise ValueError("Qwen3-8B RoPE fusion requires rope_scale=1.0")
        rotary_dim = head_dim if self.rotary_dim is None else int(self.rotary_dim)
        if rotary_dim != head_dim:
            raise ValueError("FlashInfer PR7 candidate supports full-head Qwen3 RoPE only")
        if self.q_rope_state != "pre_rope" or self.k_cache_rope_state != "pre_rope":
            raise ValueError(
                "FlashInfer ROPE_LLAMA attention fusion expects pre-RoPE Q and pre-RoPE K cache; "
                "post-RoPE tensors would be rotated twice"
            )

    def provenance(self, head_dim: int) -> dict[str, Any]:
        rotary_dim = head_dim if self.rotary_dim is None else int(self.rotary_dim)
        return {
            "rope_fusion": True,
            "rope_fusion_boundary": "flashinfer_attention_kernel",
            "pos_encoding_mode": self.pos_encoding_mode,
            "rope_backend": "flashinfer",
            "rope_theta": float(self.rope_theta),
            "rope_scale": float(self.rope_scale),
            "rotary_dim": rotary_dim,
            "rope_layout": "qwen3_rotate_half_non_interleaved",
            "q_rope_state": self.q_rope_state,
            "k_cache_rope_state": self.k_cache_rope_state,
        }


@dataclass(frozen=True)
class FlashInferSplitKVPolicy:
    """Split-KV policy surfaced in PR7 provenance and FlashInfer plan calls."""

    mode: SplitKVMode = "disabled"
    fixed_split_size: int | None = None

    @classmethod
    def disabled(cls) -> "FlashInferSplitKVPolicy":
        return cls(mode="disabled")

    @classmethod
    def fixed(cls, fixed_split_size: int) -> "FlashInferSplitKVPolicy":
        return cls(mode="fixed", fixed_split_size=fixed_split_size)

    @classmethod
    def auto(cls) -> "FlashInferSplitKVPolicy":
        return cls(mode="auto")

    def validate(self, *, require_batch_invariant: bool) -> None:
        if self.mode not in {"disabled", "fixed", "auto"}:
            raise ValueError(f"unsupported split-KV policy: {self.mode}")
        if self.mode == "fixed":
            if self.fixed_split_size is None or self.fixed_split_size <= 0:
                raise ValueError("fixed split-KV policy requires fixed_split_size > 0")
        elif self.fixed_split_size is not None:
            raise ValueError("fixed_split_size is only valid for fixed split-KV policy")
        if require_batch_invariant and self.mode == "auto":
            raise ValueError(
                "FlashInfer auto split-KV is not a PR7 batch-invariant candidate; "
                "use disabled split-KV or a fixed split size"
            )

    def plan_kwargs(self) -> dict[str, Any]:
        if self.mode == "disabled":
            return {"disable_split_kv": True}
        if self.mode == "fixed":
            assert self.fixed_split_size is not None
            return {"fixed_split_size": int(self.fixed_split_size), "disable_split_kv": False}
        return {"disable_split_kv": False}

    def provenance(self, *, require_batch_invariant: bool) -> dict[str, Any]:
        if self.mode == "disabled":
            split_kv_policy = "disabled"
            invariant = "strict_candidate"
        elif self.mode == "fixed":
            split_kv_policy = f"fixed:{self.fixed_split_size}"
            invariant = "candidate_fixed_split"
        else:
            split_kv_policy = "auto"
            invariant = "unsupported_when_required"
        return {
            "split_kv_policy": split_kv_policy,
            "fixed_split_size": self.fixed_split_size,
            "disable_split_kv": self.mode == "disabled",
            "batch_invariant_required": bool(require_batch_invariant),
            "batch_invariant_claim": invariant,
        }


@dataclass(frozen=True)
class FlashInferPagedAttentionConfig:
    """Runtime knobs for the opt-in FlashInfer paged attention candidate."""

    mode: FlashInferAttentionMode = "prefill"
    causal: bool = True
    kv_layout: str = "NHD"
    softmax_scale: float | None = None
    return_lse: bool = True
    require_batch_invariant: bool = True
    workspace_size_bytes: int = 128 * 1024 * 1024
    rope: FlashInferRoPEFusionConfig = field(default_factory=FlashInferRoPEFusionConfig)
    split_kv: FlashInferSplitKVPolicy = field(default_factory=FlashInferSplitKVPolicy.disabled)
    cp_comm_plan: AttentionCPCommunicationPlan = field(
        default_factory=lambda: AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        )
    )
    require_cp_comm: bool = False

    def validate(self, *, head_dim: int, query_len: int) -> None:
        if self.mode not in {"prefill", "decode"}:
            raise ValueError("mode must be 'prefill' or 'decode'")
        if self.mode == "decode" and query_len != 1:
            raise ValueError("BatchDecodeWithPagedKVCacheWrapper requires Sq == 1")
        if self.kv_layout != "NHD":
            raise ValueError("PR7 FlashInfer adapter currently supports kv_layout='NHD' only")
        if not self.return_lse:
            raise ValueError("PR7 requires attention-domain LSE export")
        if self.workspace_size_bytes <= 0:
            raise ValueError("workspace_size_bytes must be positive")
        self.rope.validate(head_dim)
        self.split_kv.validate(require_batch_invariant=self.require_batch_invariant)
        self.cp_comm_plan.validate()
        if self.cp_comm_plan.status != "interface_only":
            raise ValueError(
                "PR7 FlashInfer scaffold only exposes the CP communication interface; "
                "real custom CUDA AG/RS execution is not wired yet"
            )
        if self.require_cp_comm:
            raise ValueError(
                "requested CP communication cannot execute in PR7: the custom CUDA AG/RS "
                "communication operators are interface-only in this scaffold"
            )


@dataclass(frozen=True)
class FlashInferPagedKVPlan:
    """FlashInfer paged-KV tensors derived from PR6-style metadata."""

    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    kv_seq_lens: torch.Tensor
    seq_lens_q: torch.Tensor
    page_size: int
    physical_page_count_per_batch: int
    logical_block_counts: tuple[int, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "page_size": self.page_size,
            "physical_page_count_per_batch": self.physical_page_count_per_batch,
            "logical_block_counts": list(self.logical_block_counts),
            "qo_indptr": self.qo_indptr.detach().cpu().tolist(),
            "paged_kv_indptr": self.paged_kv_indptr.detach().cpu().tolist(),
            "paged_kv_indices": self.paged_kv_indices.detach().cpu().tolist(),
            "paged_kv_last_page_len": self.paged_kv_last_page_len.detach().cpu().tolist(),
            "kv_seq_lens": self.kv_seq_lens.detach().cpu().tolist(),
            "seq_lens_q": self.seq_lens_q.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class FlashInferAttentionResult:
    """Output of the FlashInfer PR7 candidate."""

    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


def build_flashinfer_paged_kv_plan(
    metadata: Any,
    *,
    batch_size: int,
    query_len: int,
    cache_capacity: int,
    device: torch.device,
) -> FlashInferPagedKVPlan:
    """Convert PR6-style paged metadata to FlashInfer page table tensors."""

    page_size = _positive_int(int(metadata.page_size), "page_size")
    if cache_capacity % page_size != 0:
        raise ValueError("physical KV cache capacity must be divisible by page_size")
    physical_page_count = cache_capacity // page_size
    if metadata.kv_seq_lens.shape != (batch_size,):
        raise ValueError("kv_seq_lens must have shape [B]")
    if metadata.block_table.ndim != 2 or metadata.block_table.size(0) != batch_size:
        raise ValueError("block_table must have shape [B, max_blocks]")

    qo_indptr = [0]
    paged_kv_indptr = [0]
    paged_kv_indices: list[int] = []
    paged_kv_last_page_len: list[int] = []
    kv_seq_lens: list[int] = []
    seq_lens_q: list[int] = []
    logical_block_counts: list[int] = []
    for batch_index in range(batch_size):
        seq_len = _positive_int(int(metadata.kv_seq_lens[batch_index].item()), "kv_seq_len")
        if seq_len > cache_capacity:
            raise ValueError("kv_seq_len must not exceed cache capacity")
        block_count = (seq_len + page_size - 1) // page_size
        if block_count > metadata.block_table.size(1):
            raise ValueError("block_table does not contain enough logical KV blocks")
        kv_seq_lens.append(seq_len)
        seq_lens_q.append(query_len)
        logical_block_counts.append(block_count)
        qo_indptr.append(qo_indptr[-1] + query_len)
        paged_kv_indptr.append(paged_kv_indptr[-1] + block_count)
        last_len = ((seq_len - 1) % page_size) + 1
        paged_kv_last_page_len.append(last_len)
        for logical_block in range(block_count):
            local_page = int(metadata.block_table[batch_index, logical_block].item())
            if local_page < 0 or local_page >= physical_page_count:
                raise ValueError("block_table contains an out-of-range physical page")
            paged_kv_indices.append(batch_index * physical_page_count + local_page)
        _validate_metadata_logical_positions(
            metadata,
            batch_index=batch_index,
            seq_len=seq_len,
            page_size=page_size,
            block_count=block_count,
            device=device,
        )

    return FlashInferPagedKVPlan(
        qo_indptr=torch.tensor(qo_indptr, device=device, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(paged_kv_indptr, device=device, dtype=torch.int32),
        paged_kv_indices=torch.tensor(paged_kv_indices, device=device, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(
            paged_kv_last_page_len,
            device=device,
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.tensor(kv_seq_lens, device=device, dtype=torch.int32),
        seq_lens_q=torch.tensor(seq_lens_q, device=device, dtype=torch.int32),
        page_size=page_size,
        physical_page_count_per_batch=physical_page_count,
        logical_block_counts=tuple(logical_block_counts),
    )


def materialize_flashinfer_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``[B, Hkv, P*page, D]`` caches to FlashInfer NHD pages."""

    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have matching shape")
    if k_cache.ndim != 4:
        raise ValueError("k_cache and v_cache must have shape [B, Hkv, cache_capacity, D]")
    batch, heads, cache_capacity, head_dim = k_cache.shape
    if cache_capacity % page_size != 0:
        raise ValueError("cache capacity must be divisible by page_size")
    page_count = cache_capacity // page_size
    k_pages = (
        k_cache.contiguous()
        .reshape(batch, heads, page_count, page_size, head_dim)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * page_count, page_size, heads, head_dim)
        .contiguous()
    )
    v_pages = (
        v_cache.contiguous()
        .reshape(batch, heads, page_count, page_size, head_dim)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * page_count, page_size, heads, head_dim)
        .contiguous()
    )
    return k_pages, v_pages


class FlashInferQwen3PagedAttentionOp:
    """Opt-in FlashInfer paged attention backend candidate for #235 PR7."""

    op_class = "attention"

    def __init__(self, *, flashinfer_module: Any | None = None) -> None:
        self._flashinfer_module = flashinfer_module

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        *,
        config: FlashInferPagedAttentionConfig | None = None,
    ) -> FlashInferAttentionResult:
        return self.forward(q, k_cache, v_cache, metadata, config=config)

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        *,
        config: FlashInferPagedAttentionConfig | None = None,
    ) -> FlashInferAttentionResult:
        """Run FlashInfer paged attention and return RL-Kernel shaped tensors.

        Args:
            q: pre-RoPE query tensor, ``[B, Hq, Sq, D]``.
            k_cache: pre-RoPE paged key cache, ``[B, Hkv, cache_capacity, D]``.
            v_cache: paged value cache, ``[B, Hkv, cache_capacity, D]``.
            metadata: PR6 ``DecodeKVCacheMetadata``-compatible object.
            config: PR7 FlashInfer backend knobs.
        """

        _validate_qkv_cache(q, k_cache, v_cache)
        cfg = FlashInferPagedAttentionConfig() if config is None else config
        batch_size, q_heads, query_len, head_dim = q.shape
        kv_heads = k_cache.size(1)
        cfg.validate(head_dim=head_dim, query_len=query_len)
        if self._flashinfer_module is None and q.device.type != "cuda":
            raise FlashInferUnavailable("FlashInfer PR7 candidate requires CUDA tensors")

        plan = build_flashinfer_paged_kv_plan(
            metadata,
            batch_size=batch_size,
            query_len=query_len,
            cache_capacity=k_cache.size(2),
            device=q.device,
        )
        q_flat = q.transpose(1, 2).reshape(batch_size * query_len, q_heads, head_dim).contiguous()
        k_pages, v_pages = materialize_flashinfer_paged_kv_cache(
            k_cache,
            v_cache,
            page_size=plan.page_size,
        )
        wrapper = self._make_wrapper(cfg, q)
        self._plan_wrapper(
            wrapper,
            cfg,
            plan,
            q_dtype=q.dtype,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_len=query_len,
        )
        out_flat, lse_flat = self._run_wrapper(wrapper, q_flat, (k_pages, v_pages), cfg)
        out = _restore_out(out_flat, batch_size=batch_size, query_len=query_len)
        lse = _restore_lse(
            lse_flat,
            batch_size=batch_size,
            query_len=query_len,
            q_heads=q_heads,
        )
        provenance = {
            "attention_backend": "flashinfer",
            "requested_backend": "flashinfer_qwen3_rope_paged_attention",
            "actual_backend": f"flashinfer_batch_{cfg.mode}_paged_kv",
            "attention_mode": cfg.mode,
            "materialization": "flashinfer_rope_llama_paged_kv",
            "kv_layout": cfg.kv_layout,
            "causal": cfg.causal,
            "softmax_scale": cfg.softmax_scale,
            "lse_domain": "attention",
            "lse_exported": True,
            "accum_dtype": "flashinfer_internal",
            "downcast_at": "flashinfer_output",
            "fallback": False,
            "fallback_reason": None,
            "paged_kv_policy": "flashinfer_page_table",
        }
        provenance.update(cfg.rope.provenance(head_dim))
        provenance.update(
            cfg.split_kv.provenance(require_batch_invariant=cfg.require_batch_invariant)
        )
        provenance.update(cfg.cp_comm_plan.provenance())
        provenance["cp_comm_required"] = cfg.require_cp_comm
        provenance.update(plan.provenance())
        return FlashInferAttentionResult(out=out, lse=lse, provenance=provenance)

    def _load_flashinfer(self) -> Any:
        if self._flashinfer_module is not None:
            return self._flashinfer_module
        try:
            self._flashinfer_module = importlib.import_module(_FLASHINFER_MODULE)
        except (ImportError, OSError, RuntimeError) as exc:
            raise FlashInferUnavailable(str(exc)) from exc
        return self._flashinfer_module

    def _make_wrapper(self, cfg: FlashInferPagedAttentionConfig, q: torch.Tensor) -> Any:
        module = self._load_flashinfer()
        namespace_name = "decode" if cfg.mode == "decode" else "prefill"
        class_name = (
            "BatchDecodeWithPagedKVCacheWrapper"
            if cfg.mode == "decode"
            else "BatchPrefillWithPagedKVCacheWrapper"
        )
        namespace = getattr(module, namespace_name, None)
        wrapper_cls = getattr(namespace, class_name, None) if namespace is not None else None
        if wrapper_cls is None:
            raise FlashInferUnavailable(f"flashinfer.{namespace_name}.{class_name} is unavailable")

        workspace = torch.zeros(cfg.workspace_size_bytes, dtype=torch.uint8, device=q.device)
        try:
            return wrapper_cls(workspace, kv_layout=cfg.kv_layout)
        except TypeError:
            try:
                return wrapper_cls(float_workspace_buffer=workspace, kv_layout=cfg.kv_layout)
            except TypeError as exc:
                raise FlashInferUnavailable(
                    f"could not instantiate flashinfer.{namespace_name}.{class_name}"
                ) from exc

    @staticmethod
    def _plan_wrapper(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
        plan: FlashInferPagedKVPlan,
        *,
        q_dtype: torch.dtype,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        query_len: int,
    ) -> None:
        plan_kwargs = {
            "qo_indptr": plan.qo_indptr,
            "paged_kv_indptr": plan.paged_kv_indptr,
            "paged_kv_indices": plan.paged_kv_indices,
            "paged_kv_last_page_len": plan.paged_kv_last_page_len,
            "indptr": plan.paged_kv_indptr,
            "indices": plan.paged_kv_indices,
            "last_page_len": plan.paged_kv_last_page_len,
            "num_qo_heads": q_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "head_dim_qk": head_dim,
            "page_size": plan.page_size,
            "causal": cfg.causal,
            "pos_encoding_mode": cfg.rope.pos_encoding_mode,
            "rope_scale": float(cfg.rope.rope_scale),
            "rope_theta": float(cfg.rope.rope_theta),
            "q_data_type": q_dtype,
            "kv_data_type": q_dtype,
            "o_data_type": q_dtype,
            "data_type": q_dtype,
            "seq_lens": plan.kv_seq_lens,
            "seq_lens_q": plan.seq_lens_q,
            "q_len_per_req": query_len,
        }
        scale = cfg.softmax_scale
        if scale is not None:
            plan_kwargs["softmax_scale"] = float(scale)
            plan_kwargs["sm_scale"] = float(scale)
        plan_kwargs.update(cfg.split_kv.plan_kwargs())
        _call_with_supported_kwargs(wrapper.plan, plan_kwargs)

    @staticmethod
    def _run_wrapper(
        wrapper: Any,
        q_flat: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        cfg: FlashInferPagedAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(wrapper, "run_return_lse"):
            result = wrapper.run_return_lse(q_flat, paged_kv_cache)
        else:
            result = _call_with_supported_kwargs(
                wrapper.run,
                {"q": q_flat, "paged_kv_cache": paged_kv_cache, "return_lse": cfg.return_lse},
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise FlashInferUnavailable("FlashInfer PR7 candidate must return (out, lse)")
        out_flat, lse_flat = result
        return out_flat, lse_flat


def flashinfer_qwen3_paged_attention_available() -> bool:
    """Return whether the FlashInfer paged attention wrappers are importable."""

    try:
        module = FlashInferQwen3PagedAttentionOp()._load_flashinfer()
        prefill = getattr(
            getattr(module, "prefill", None),
            "BatchPrefillWithPagedKVCacheWrapper",
            None,
        )
        decode = getattr(
            getattr(module, "decode", None),
            "BatchDecodeWithPagedKVCacheWrapper",
            None,
        )
        if not callable(prefill) or not callable(decode):
            return False
    except FlashInferUnavailable:
        return False
    return True


def _validate_metadata_logical_positions(
    metadata: Any,
    *,
    batch_index: int,
    seq_len: int,
    page_size: int,
    block_count: int,
    device: torch.device,
) -> None:
    if not hasattr(metadata, "global_token_positions"):
        return
    global_token_positions = metadata.global_token_positions
    if global_token_positions.ndim != 2 or global_token_positions.size(0) <= batch_index:
        raise ValueError("global_token_positions must have shape [B, cache_capacity]")
    logical_positions: list[int] = []
    physical_slots: list[int] = []
    for logical_block in range(block_count):
        local_page = int(metadata.block_table[batch_index, logical_block].item())
        token_count = min(page_size, seq_len - logical_block * page_size)
        for page_offset in range(token_count):
            physical_slots.append(local_page * page_size + page_offset)
            logical_positions.append(logical_block * page_size + page_offset)
    slot_index = torch.tensor(physical_slots, device=device, dtype=torch.long)
    expected = torch.tensor(logical_positions, device=device, dtype=global_token_positions.dtype)
    actual = global_token_positions[batch_index, slot_index]
    if not torch.equal(actual, expected):
        raise ValueError(
            "block_table/global_token_positions must reconstruct logical positions "
            "0..kv_seq_len-1 exactly"
        )
    if hasattr(metadata, "key_position_ids"):
        key_positions = metadata.key_position_ids[batch_index, slot_index]
        if not torch.equal(key_positions, expected.to(dtype=key_positions.dtype)):
            raise ValueError("key_position_ids must match cached global token positions")


def _validate_qkv_cache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("q, k_cache, and v_cache must have shape [B, H, S, D]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have matching shape")
    if q.size(0) != k_cache.size(0) or q.size(3) != k_cache.size(3):
        raise ValueError("q and KV cache must share batch size and head_dim")
    if q.size(1) % k_cache.size(1) != 0:
        raise ValueError("Q head count must be divisible by KV head count")


def _restore_out(out_flat: torch.Tensor, *, batch_size: int, query_len: int) -> torch.Tensor:
    if out_flat.ndim != 3:
        raise FlashInferUnavailable("FlashInfer output must have shape [B*Sq, Hq, D]")
    _, q_heads, head_dim = out_flat.shape
    expected = batch_size * query_len
    if out_flat.size(0) != expected:
        raise FlashInferUnavailable(
            f"FlashInfer output first dim must be B*Sq={expected}, got {out_flat.size(0)}"
        )
    return out_flat.reshape(batch_size, query_len, q_heads, head_dim).transpose(1, 2).contiguous()


def _restore_lse(
    lse_flat: torch.Tensor,
    *,
    batch_size: int,
    query_len: int,
    q_heads: int,
) -> torch.Tensor:
    expected_tokens = batch_size * query_len
    if lse_flat.shape == (expected_tokens, q_heads):
        return lse_flat.reshape(batch_size, query_len, q_heads).transpose(1, 2).contiguous()
    if lse_flat.shape == (q_heads, expected_tokens):
        return lse_flat.transpose(0, 1).reshape(batch_size, query_len, q_heads).transpose(1, 2)
    raise FlashInferUnavailable(
        "FlashInfer LSE must have shape [B*Sq, Hq] or [Hq, B*Sq]; " f"got {tuple(lse_flat.shape)}"
    )


def _call_with_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return fn(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in parameters}
    missing_required = [
        name
        for name, param in parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and name not in supported
    ]
    if missing_required:
        raise FlashInferUnavailable(
            f"{getattr(fn, '__qualname__', fn)} missing supported arguments: "
            f"{', '.join(missing_required)}"
        )
    return fn(**supported)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


__all__ = [
    "FlashInferAttentionMode",
    "FlashInferAttentionResult",
    "FlashInferPagedAttentionConfig",
    "FlashInferPagedKVPlan",
    "FlashInferQwen3PagedAttentionOp",
    "FlashInferRoPEFusionConfig",
    "FlashInferSplitKVPolicy",
    "FlashInferUnavailable",
    "SplitKVMode",
    "build_flashinfer_paged_kv_plan",
    "flashinfer_qwen3_paged_attention_available",
    "materialize_flashinfer_paged_kv_cache",
]
