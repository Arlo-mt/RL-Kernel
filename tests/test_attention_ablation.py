# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
    SplitKVSpec,
)
from rl_engine.kernels.ops.pytorch.attention.ablation import (
    BACKEND_ID,
    REFERENCE_BACKEND_ID,
    AttentionAblationOp,
)


def _contract(*, split_kv: SplitKVSpec | None = None) -> AttentionContract:
    sharding = ShardingSpec(
        tp_rank=0,
        tp_world_size=1,
        cp_rank=0,
        cp_world_size=1,
        global_q_heads=2,
        global_kv_heads=1,
        local_q_head_start=0,
        local_q_heads=2,
        local_kv_head_start=0,
        local_kv_heads=1,
        global_sequence_length=4,
        local_sequence_length=4,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, 4),
    )
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=4,
        head_dim=4,
        causal=True,
        causal_offsets=(0,),
        sharding=sharding,
        reduction=ReductionSpec(),
        split_kv=split_kv or SplitKVSpec.disabled(),
    )


def _qkv():
    torch.manual_seed(0)
    return (
        torch.randn(1, 2, 4, 4, dtype=torch.bfloat16),
        torch.randn(1, 1, 4, 4, dtype=torch.bfloat16),
        torch.randn(1, 1, 4, 4, dtype=torch.bfloat16),
    )


def test_attention_wrapper_has_unified_result_and_provenance():
    q, k, v = _qkv()
    result = AttentionAblationOp()(q, k, v, contract=_contract())

    assert result.backend_id == REFERENCE_BACKEND_ID
    assert result.deterministic
    assert result.out.shape == q.shape
    assert result.lse is not None
    assert result.lse.dtype is torch.float32
    assert result.provenance["semantic_operator"] == "attention"
    assert result.provenance["split_kv"]["mode"] == "disabled"
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_CORE_ID
    assert result.provenance["strict_schedule"] == STRICT_ATTENTION_SCHEDULE_ID
    assert result.readback()["out_shape"] == list(q.shape)


def test_attention_wrapper_supports_explicit_injected_backend():
    q, k, v = _qkv()

    class FakeBackend:
        backend_id = "test.attention.backend"

        def forward_with_lse(self, q, k, v, *, causal, scale):
            del k, v, causal, scale
            return q.clone(), torch.zeros(q.shape[:3], dtype=torch.float32)

    result = AttentionAblationOp()(
        q,
        k,
        v,
        contract=_contract(),
        backend=FakeBackend(),
        deterministic=False,
    )
    assert result.backend_id == "test.attention.backend"
    assert torch.equal(result.out, q)


def test_deterministic_attention_rejects_runtime_split_kv_auto():
    q, k, v = _qkv()
    contract = _contract(split_kv=SplitKVSpec.auto(strict_consistency=False))
    with pytest.raises(AttentionContractError, match="Split-KV=auto"):
        AttentionAblationOp()(q, k, v, contract=contract)


def test_deterministic_native_backend_requires_explicit_native_callable():
    q, k, v = _qkv()
    with pytest.raises(AttentionContractError, match="native Attention backend"):
        AttentionAblationOp()(q, k, v, contract=_contract(), backend="native")


def test_attention_wrapper_can_return_dq_dk_dv_from_reference_backend():
    q, k, v = _qkv()
    result = AttentionAblationOp()(
        q,
        k,
        v,
        contract=_contract(),
        return_gradients=True,
        dout=torch.ones_like(q),
    )

    assert result.dq is not None and result.dq.shape == q.shape
    assert result.dk is not None and result.dk.shape == k.shape
    assert result.dv is not None and result.dv.shape == v.shape


def test_attention_backend_is_registered_for_pr230_semantic_resolution():
    from rl_engine.kernels.registry import kernel_registry
    from rl_engine.kernels.semantic_registry import OperatorRequirements

    session = kernel_registry.semantic.session()
    resolution = session.resolve(
        semantic_op="attention",
        requested_backend=BACKEND_ID,
        target="training",
        requirements=OperatorRequirements(
            device="cpu",
            dtype="bfloat16",
            topology={"world_size": 1, "tensor_parallel_size": 1, "context_parallel_size": 1},
            alignment_properties={"deterministic": True},
        ),
    )
    instance = session.instantiate(resolution)
    assert isinstance(instance, AttentionAblationOp)
    provenance = session.instance_provenance(resolution, instance)
    assert provenance.backend_id == BACKEND_ID


def test_strict_wrapper_is_bitwise_invariant_to_batch_shape():
    q, k, v = _qkv()
    noise_q, noise_k, noise_v = _qkv()
    contract = _contract()
    batch_contract = AttentionContract(
        role=contract.role,
        mode=contract.mode,
        dtype=contract.dtype,
        batch_size=2,
        query_sequence_length=contract.query_sequence_length,
        head_dim=contract.head_dim,
        causal=contract.causal,
        causal_offsets=(0, 0),
        sharding=contract.sharding,
        reduction=contract.reduction,
        split_kv=contract.split_kv,
    )
    single = AttentionAblationOp()(q, k, v, contract=contract)
    batched = AttentionAblationOp()(
        torch.cat((q, noise_q), dim=0),
        torch.cat((k, noise_k), dim=0),
        torch.cat((v, noise_v), dim=0),
        contract=batch_contract,
    )
    assert torch.equal(single.out[0], batched.out[0])
    assert torch.equal(single.lse[0], batched.lse[0])
