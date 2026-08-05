# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import types

import pytest
import torch

from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPCommunicationUnavailable,
    AttentionCPMergedState,
    AttentionCPPartialState,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
    sort_attention_cp_partial_states,
)
from rl_engine.kernels.ops.cuda.attention.flashinfer_paged_attention import (
    FlashInferPagedAttentionConfig,
    FlashInferQwen3PagedAttentionOp,
    FlashInferRoPEFusionConfig,
    FlashInferSplitKVPolicy,
    FlashInferUnavailable,
    build_flashinfer_paged_kv_plan,
    materialize_flashinfer_paged_kv_cache,
)
from rl_engine.testing.attention_comparison import DecodeKVCacheMetadata


class _FakeFlashInferWrapper:
    instances: list["_FakeFlashInferWrapper"] = []

    def __init__(self, workspace_buffer, *, kv_layout):
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self.plan_kwargs = None
        self.run_q = None
        self.run_cache = None
        self.instances.append(self)

    def plan(self, **kwargs):
        self.plan_kwargs = kwargs

    def run_return_lse(self, q, paged_kv_cache):
        self.run_q = q
        self.run_cache = paged_kv_cache
        out = torch.zeros_like(q)
        lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32, device=q.device)
        return out, lse


def _fake_flashinfer():
    _FakeFlashInferWrapper.instances = []
    return types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_FakeFlashInferWrapper,
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_FakeFlashInferWrapper,
        ),
    )


def _metadata(*, batch: int = 2, query_len: int = 1) -> DecodeKVCacheMetadata:
    page_size = 2
    cache_capacity = 6
    positions = torch.arange(cache_capacity, dtype=torch.long).repeat(batch, 1)
    return DecodeKVCacheMetadata(
        cache_position=torch.full((batch, query_len), cache_capacity - 1, dtype=torch.long),
        kv_seq_lens=torch.full((batch,), cache_capacity, dtype=torch.long),
        block_table=torch.tensor([[0, 1, 2]] * batch, dtype=torch.long),
        global_token_positions=positions,
        query_position_ids=torch.full((batch, query_len), cache_capacity - 1, dtype=torch.long),
        key_position_ids=positions.clone(),
        page_size=page_size,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )


def _qkv(*, batch: int = 2, query_len: int = 1):
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(batch, 4, query_len, 8, generator=gen)
    k = torch.randn(batch, 2, 6, 8, generator=gen)
    v = torch.randn(batch, 2, 6, 8, generator=gen)
    return q, k, v


def _partial_state(global_block_index: int) -> AttentionCPPartialState:
    return AttentionCPPartialState(
        out=torch.full((1, 2, 1, 4), float(global_block_index)),
        lse=torch.full((1, 2, 1), float(global_block_index), dtype=torch.float32),
        block=AttentionCPBlockMetadata(
            global_block_index=global_block_index,
            kv_block_start=global_block_index * 2,
            kv_block_end=global_block_index * 2 + 2,
            owner_cp_rank=global_block_index % 2,
            owner_tp_rank=0,
        ),
    )


def test_flashinfer_pr7_prefill_adapter_passes_qwen3_rope_and_splitk_policy():
    q, k, v = _qkv(query_len=2)
    metadata = _metadata(query_len=2)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    result = op(
        q,
        k,
        v,
        metadata,
        config=FlashInferPagedAttentionConfig(
            mode="prefill",
            workspace_size_bytes=1024,
            split_kv=FlashInferSplitKVPolicy.fixed(4),
        ),
    )

    wrapper = _FakeFlashInferWrapper.instances[-1]
    assert wrapper.kv_layout == "NHD"
    assert wrapper.run_q.shape == (q.size(0) * q.size(2), q.size(1), q.size(3))
    assert result.out.shape == q.shape
    assert result.lse.shape == q.shape[:3]
    assert result.provenance["actual_backend"] == "flashinfer_batch_prefill_paged_kv"
    assert result.provenance["rope_fusion_boundary"] == "flashinfer_attention_kernel"
    assert result.provenance["pos_encoding_mode"] == "ROPE_LLAMA"
    assert result.provenance["rope_theta"] == 1_000_000.0
    assert result.provenance["rope_scale"] == 1.0
    assert result.provenance["split_kv_policy"] == "fixed:4"
    assert result.provenance["batch_invariant_claim"] == "candidate_fixed_split"
    assert result.provenance["tp_world_size"] == 2
    assert result.provenance["cp_world_size"] == 2
    assert result.provenance["cp_comm_backend"] == "cuda_ag_rs"
    assert result.provenance["cp_comm_status"] == "interface_only"
    assert result.provenance["cp_comm_pattern"] == "ag_rs"
    assert result.provenance["cp_comm_compute_communication"] == "decoupled"
    assert result.provenance["cp_comm_merge_order"] == "global_block_index"
    assert result.provenance["cp_comm_accum_dtype"] == "fp32"
    assert result.provenance["cp_comm_return_lse"] is True
    assert result.provenance["cp_comm_contract"] == "partial_out_lse_global_block_index"
    assert result.provenance["cp_comm_required"] is False

    plan = wrapper.plan_kwargs
    assert plan["qo_indptr"].tolist() == [0, 2, 4]
    assert plan["paged_kv_indptr"].tolist() == [0, 3, 6]
    assert plan["paged_kv_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    assert plan["paged_kv_last_page_len"].tolist() == [2, 2]
    assert plan["pos_encoding_mode"] == "ROPE_LLAMA"
    assert plan["rope_theta"] == 1_000_000.0
    assert plan["rope_scale"] == 1.0
    assert plan["q_data_type"] == q.dtype
    assert plan["kv_data_type"] == q.dtype
    assert plan["fixed_split_size"] == 4
    assert plan["disable_split_kv"] is False


def test_flashinfer_pr7_decode_adapter_can_disable_splitk_for_strict_candidate():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    result = op(
        q,
        k,
        v,
        metadata,
        config=FlashInferPagedAttentionConfig(
            mode="decode",
            workspace_size_bytes=1024,
            split_kv=FlashInferSplitKVPolicy.disabled(),
        ),
    )

    wrapper = _FakeFlashInferWrapper.instances[-1]
    assert result.provenance["actual_backend"] == "flashinfer_batch_decode_paged_kv"
    assert result.provenance["split_kv_policy"] == "disabled"
    assert result.provenance["batch_invariant_claim"] == "strict_candidate"
    assert wrapper.plan_kwargs["disable_split_kv"] is True


def test_flashinfer_pr7_rejects_auto_splitk_when_batch_invariance_is_required():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="auto split-KV"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                split_kv=FlashInferSplitKVPolicy.auto(),
            ),
        )


def test_flashinfer_pr7_rejects_required_cp_comm_until_cuda_ag_rs_ops_exist():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="AG/RS communication operators are interface-only"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                require_cp_comm=True,
            ),
        )


def test_flashinfer_pr7_rejects_implemented_cp_comm_status_in_scaffold():
    config = FlashInferPagedAttentionConfig(
        cp_comm_plan=AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
            status="implemented",
        )
    )

    with pytest.raises(ValueError, match="only exposes the CP communication interface"):
        config.validate(head_dim=8, query_len=1)


def test_flashinfer_pr7_rejects_post_rope_inputs_for_rope_llama_fusion():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="rotated twice"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                rope=FlashInferRoPEFusionConfig(q_rope_state="post_rope"),
            ),
        )


def test_attention_cp_partial_states_sort_by_global_block_index():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
    )

    ordered = sort_attention_cp_partial_states(
        (_partial_state(3), _partial_state(1), _partial_state(2)),
        plan=plan,
    )

    assert [state.block.global_block_index for state in ordered] == [1, 2, 3]


def test_attention_cp_partial_states_reject_duplicate_global_block_index():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
    )

    with pytest.raises(ValueError, match="duplicate global_block_index"):
        sort_attention_cp_partial_states(
            (_partial_state(1), _partial_state(1)),
            plan=plan,
        )


def test_cuda_ag_rs_attention_cp_comm_is_interface_only():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
    )
    communication = CUDAAGRSAttentionCPCommunication()

    with pytest.raises(AttentionCPCommunicationUnavailable, match="CUDA AG"):
        communication.all_gather_partial_states((_partial_state(0),), plan)

    merged = AttentionCPMergedState(
        out=torch.zeros(1, 2, 1, 4),
        lse=torch.zeros(1, 2, 1, dtype=torch.float32),
    )
    with pytest.raises(AttentionCPCommunicationUnavailable, match="CUDA RS"):
        communication.reduce_scatter_merged_state(merged, plan)


def test_flashinfer_pr7_real_backend_requires_cuda_before_importing_flashinfer():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp()

    with pytest.raises(FlashInferUnavailable, match="requires CUDA"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(mode="decode", workspace_size_bytes=1024),
        )


def test_flashinfer_pr7_plan_and_cache_materialization_follow_logical_page_order():
    q, k, v = _qkv(batch=1, query_len=1)
    positions = torch.full((1, 6), -1, dtype=torch.long)
    positions[:, 4:6] = torch.tensor([0, 1], dtype=torch.long)
    positions[:, 0:2] = torch.tensor([2, 3], dtype=torch.long)
    positions[:, 2:4] = torch.tensor([4, 5], dtype=torch.long)
    metadata = DecodeKVCacheMetadata(
        cache_position=torch.tensor([[5]], dtype=torch.long),
        kv_seq_lens=torch.tensor([6], dtype=torch.long),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.long),
        global_token_positions=positions,
        query_position_ids=torch.tensor([[5]], dtype=torch.long),
        key_position_ids=positions.clone(),
        page_size=2,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )

    plan = build_flashinfer_paged_kv_plan(
        metadata,
        batch_size=1,
        query_len=1,
        cache_capacity=k.size(2),
        device=q.device,
    )
    k_pages, v_pages = materialize_flashinfer_paged_kv_cache(k, v, page_size=2)

    assert plan.paged_kv_indices.tolist() == [2, 0, 1]
    torch.testing.assert_close(k_pages[2], k[0, :, 4:6, :].transpose(0, 1))
    torch.testing.assert_close(v_pages[0], v[0, :, 0:2, :].transpose(0, 1))


def test_flashinfer_pr7_plan_rejects_position_metadata_mismatch():
    q, k, _ = _qkv(batch=1, query_len=1)
    metadata = DecodeKVCacheMetadata(
        cache_position=torch.tensor([[5]], dtype=torch.long),
        kv_seq_lens=torch.tensor([6], dtype=torch.long),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.long),
        global_token_positions=torch.arange(6, dtype=torch.long).unsqueeze(0),
        query_position_ids=torch.tensor([[5]], dtype=torch.long),
        key_position_ids=torch.arange(6, dtype=torch.long).unsqueeze(0),
        page_size=2,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )

    with pytest.raises(ValueError, match="reconstruct logical positions"):
        build_flashinfer_paged_kv_plan(
            metadata,
            batch_size=1,
            query_len=1,
            cache_capacity=k.size(2),
            device=q.device,
        )
