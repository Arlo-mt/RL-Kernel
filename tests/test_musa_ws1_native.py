import pytest
import torch

from rl_engine.alignment.qwen3_dense import load_profile_ops
from rl_engine.kernels.ops.musa.ws1 import (
    MusaAttentionOp,
    MusaBatchInvariantLogpOp,
    MusaEmbeddingOp,
    MusaLogpOp,
    MusaRMSNormOp,
    MusaRoPEOp,
    MusaSiLUOp,
    MusaSwiGLUOp,
)
from rl_engine.kernels.ops.pytorch.activation.swiglu import (
    NativeSiLUOp,
    NativeSwiGLUOp,
)
from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp
from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import (
    NativeBatchInvariantLogpOp,
)
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp
from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp
from rl_engine.kernels.ops.pytorch.rotary_embedding.rope import NativeRoPEOp


def _musa_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


pytestmark = pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")


def test_cuda_bf16_profile_resolves_all_nodes_to_native_musa():
    profile = load_profile_ops("cuda_bf16", device=torch.device("musa"))
    assert set(profile.ops) == {
        "embedding",
        "rms_norm",
        "det_gemm",
        "qk_norm",
        "rope",
        "attention",
        "swiglu",
        "silu",
        "lm_head",
        "logprob",
        "batch_invariant_logp",
    }
    assert all(
        value["actual_backend"] == "musa" for value in profile.provenance.values()
    )
    assert all(
        value["candidate_path"].startswith("rl_engine.kernels.ops.musa.")
        for value in profile.provenance.values()
    )


def test_native_musa_elementwise_and_logprob_candidates_match_reference():
    torch.manual_seed(17)
    x = torch.randn(2, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    rms = MusaRMSNormOp()(x, weight)
    rms_ref = NativeRMSNormOp().forward(x, weight)
    assert torch.allclose(rms.float(), rms_ref.float(), atol=0.02, rtol=0.02)

    positions = torch.arange(2, device="musa", dtype=torch.long)
    rope = MusaRoPEOp().forward(x.reshape(1, 1, 2, 128), positions)
    rope_ref = NativeRoPEOp().forward(x.reshape(1, 1, 2, 128), positions)
    assert torch.allclose(rope.float(), rope_ref.float(), atol=0.02, rtol=0.02)

    logits = torch.randn(2, 19, device="musa", dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([1, 7], device="musa", dtype=torch.long)
    logp = MusaLogpOp().forward(logits, target)
    logp_ref = NativeLogpOp().forward(logits, target)
    assert torch.allclose(logp.float(), logp_ref.float(), atol=0.02, rtol=0.02)

    batch_logp = MusaBatchInvariantLogpOp().apply(logits, target)
    batch_ref = NativeBatchInvariantLogpOp().apply(logits, target)
    assert torch.allclose(batch_logp.float(), batch_ref.float(), atol=0.02, rtol=0.02)


def test_native_musa_attention_and_activation_candidates_match_reference():
    torch.manual_seed(23)
    q = torch.randn(1, 2, 3, 128, device="musa", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 3, 128, device="musa", dtype=torch.bfloat16)
    v = torch.randn(1, 1, 3, 128, device="musa", dtype=torch.bfloat16)
    mask = torch.ones(1, 3, device="musa", dtype=torch.bool)
    out = MusaAttentionOp().forward_fp32(q, k, v, key_padding_mask=mask)
    ref = NativeAttentionOp().forward_fp32(q, k, v, key_padding_mask=mask)
    assert torch.allclose(out.float(), ref.float(), atol=0.03, rtol=0.03)

    x = torch.randn(32, device="musa", dtype=torch.bfloat16)
    gate = torch.randn(32, device="musa", dtype=torch.bfloat16)
    up = torch.randn(32, device="musa", dtype=torch.bfloat16)
    assert torch.allclose(
        MusaSiLUOp().forward(x).float(),
        NativeSiLUOp().forward(x).float(),
        atol=0.02,
        rtol=0.02,
    )
    assert torch.allclose(
        MusaSwiGLUOp().forward(gate, up).float(),
        NativeSwiGLUOp().forward(gate, up).float(),
        atol=0.02,
        rtol=0.02,
    )
