# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch


def _sm90_linear_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        return (
            _EXT_AVAILABLE
            and hasattr(_C, "embedding_sm90_forward")
            and hasattr(_C, "lm_head_sm90_forward")
            and torch.cuda.get_device_capability()[0] == 9
        )
    except Exception:
        return False


requires_sm90_linear = pytest.mark.skipif(
    not _sm90_linear_available(),
    reason=(
        "SM90 embedding/lm_head kernels require an H200/Hopper-class GPU and "
        "KERNEL_ALIGN_FORCE_SM90=1."
    ),
)


def test_sm90_embedding_wrapper_calls_extension_symbol(monkeypatch):
    from rl_engine.kernels.ops.cuda.linear import embedding as embedding_module

    calls = []

    class FakeExtension:
        @staticmethod
        def embedding_sm90_forward(token_ids, weight):
            calls.append(("forward", token_ids, weight))
            return torch.empty((*token_ids.shape, weight.size(1)), dtype=weight.dtype)

        @staticmethod
        def embedding_sm90_forward_fp32(token_ids, weight):
            calls.append(("forward_fp32", token_ids, weight))
            return torch.empty((*token_ids.shape, weight.size(1)), dtype=torch.float32)

    monkeypatch.setattr(embedding_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(embedding_module, "_C", FakeExtension)
    monkeypatch.setattr(
        embedding_module.SM90EmbeddingOp,
        "_can_use_sm90",
        staticmethod(lambda token_ids, weight: True),
    )

    op = embedding_module.SM90EmbeddingOp()
    token_ids = torch.tensor([[1, 2], [3, 4]])
    weight = torch.randn(8, 5)

    assert op.forward(token_ids, weight).shape == (2, 2, 5)
    assert op.forward_fp32(token_ids, weight).dtype == torch.float32
    assert [name for name, *_ in calls] == ["forward", "forward_fp32"]


def test_sm90_lm_head_wrapper_calls_extension_symbol(monkeypatch):
    from rl_engine.kernels.ops.cuda.linear import lm_head as lm_head_module

    calls = []

    class FakeExtension:
        @staticmethod
        def lm_head_sm90_forward(hidden, weight, bias):
            calls.append(("forward", hidden, weight, bias))
            return torch.empty((*hidden.shape[:-1], weight.size(0)), dtype=hidden.dtype)

        @staticmethod
        def lm_head_sm90_forward_fp32(hidden, weight, bias):
            calls.append(("forward_fp32", hidden, weight, bias))
            return torch.empty((*hidden.shape[:-1], weight.size(0)), dtype=torch.float32)

    monkeypatch.setattr(lm_head_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(lm_head_module, "_C", FakeExtension)
    monkeypatch.setattr(
        lm_head_module.SM90LMHeadOp,
        "_can_use_sm90",
        staticmethod(lambda hidden, weight, bias: True),
    )

    op = lm_head_module.SM90LMHeadOp()
    hidden = torch.randn(2, 3, 7)
    weight = torch.randn(11, 7)
    bias = torch.randn(11)

    assert op.forward(hidden, weight, bias=bias).shape == (2, 3, 11)
    assert op.forward_fp32(hidden, weight, bias=bias).dtype == torch.float32
    assert [name for name, *_ in calls] == ["forward", "forward_fp32"]


@requires_sm90_linear
def test_sm90_embedding_forward_matches_direct_gather_on_h200_hopper():
    from rl_engine.kernels.ops.cuda.linear.embedding import SM90EmbeddingOp

    device = torch.device("cuda")
    token_ids = torch.tensor([[0, 7, 3], [5, 1, 7]], device=device)
    weight = torch.randn(11, 13, device=device, dtype=torch.bfloat16)

    out = SM90EmbeddingOp().forward(token_ids, weight)
    out_fp32 = SM90EmbeddingOp().forward_fp32(token_ids, weight)

    assert torch.equal(out, weight[token_ids])
    assert torch.equal(out_fp32, weight[token_ids].float())


@requires_sm90_linear
def test_sm90_lm_head_forward_is_slice_batch_invariant_on_h200_hopper():
    from rl_engine.kernels.ops.cuda.linear.lm_head import SM90LMHeadOp

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(150)
    hidden = torch.randn(4, 5, 64, device=device, dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(37, 64, device=device, dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(37, device=device, dtype=torch.bfloat16, generator=generator)
    op = SM90LMHeadOp()

    full = op.forward(hidden, weight, bias=bias)
    full_fp32 = op.forward_fp32(hidden, weight, bias=bias)

    assert torch.equal(op.forward(hidden[2:3], weight, bias=bias), full[2:3])
    assert torch.equal(op.forward_fp32(hidden[2:3], weight, bias=bias), full_fp32[2:3])
