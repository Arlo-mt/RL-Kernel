# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.musa.loss.linear_ratio_kl import MusaLinearRatioKLOp
from rl_engine.kernels.registry import KernelRegistry


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def _reference(hidden, weight, action, mask, old_logp, reference_logp, bias):
    logits = torch.nn.functional.linear(
        hidden.float(),
        weight.float(),
        None if bias is None else bias.float(),
    )
    safe_action = action.masked_fill(~mask.bool(), 0)
    policy_logp = (
        torch.log_softmax(logits, dim=-1).gather(-1, safe_action.unsqueeze(-1)).squeeze(-1)
    )
    delta = (policy_logp - old_logp.float()).masked_fill(~mask.bool(), 0.0)
    diff = (reference_logp.float() - policy_logp).masked_fill(~mask.bool(), 0.0)
    return torch.exp(delta), torch.exp(diff) - diff - 1.0


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize("with_bias", [False, True])
def test_musa_linear_ratio_kl_matches_reference_forward_and_backward(with_bias):
    from rl_engine import _C

    assert hasattr(_C, "fused_linear_ratio_kl_musa_forward")
    assert hasattr(_C, "fused_linear_ratio_kl_musa_backward")
    torch.manual_seed(2300)
    hidden = (torch.randn(2, 3, 32, device="musa", dtype=torch.bfloat16) * 0.05).requires_grad_()
    weight = (torch.randn(65, 32, device="musa", dtype=torch.bfloat16) * 0.05).requires_grad_()
    bias = (
        (torch.randn(65, device="musa", dtype=torch.bfloat16) * 0.05).requires_grad_()
        if with_bias
        else None
    )
    action = torch.tensor([[0, 17, -100], [64, 5, -100]], device="musa")
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], device="musa", dtype=torch.int32)

    with torch.no_grad():
        reference_policy_logp = (
            torch.log_softmax(
                torch.nn.functional.linear(
                    hidden.float(),
                    weight.float(),
                    None if bias is None else bias.float(),
                ),
                dim=-1,
            )
            .gather(-1, action.masked_fill(mask == 0, 0).unsqueeze(-1))
            .squeeze(-1)
        )
        old_logp = reference_policy_logp + torch.linspace(-0.1, 0.1, 6, device="musa").view(2, 3)
        reference_logp = reference_policy_logp + torch.linspace(0.05, -0.05, 6, device="musa").view(
            2, 3
        )

    ratio, kl = MusaLinearRatioKLOp()(
        hidden,
        weight,
        action,
        mask,
        old_logp,
        reference_logp,
        bias,
    )

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_bias = bias.detach().clone().requires_grad_(True) if bias is not None else None
    expected_ratio, expected_kl = _reference(
        reference_hidden,
        reference_weight,
        action,
        mask,
        old_logp,
        reference_logp,
        reference_bias,
    )
    assert torch.allclose(ratio, expected_ratio, atol=1e-4, rtol=1e-4)
    assert torch.allclose(kl, expected_kl, atol=1e-4, rtol=1e-4)
    assert torch.equal(ratio[mask == 0], torch.ones_like(ratio[mask == 0]))
    assert torch.equal(kl[mask == 0], torch.zeros_like(kl[mask == 0]))

    grad_ratio = torch.linspace(-0.75, 1.25, 6, device="musa").view(2, 3)
    grad_kl = torch.linspace(1.5, -0.5, 6, device="musa").view(2, 3)
    torch.autograd.backward((ratio, kl), (grad_ratio, grad_kl))
    torch.autograd.backward((expected_ratio, expected_kl), (grad_ratio, grad_kl))
    assert torch.allclose(hidden.grad.float(), reference_hidden.grad.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(weight.grad.float(), reference_weight.grad.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(hidden.grad[mask == 0], torch.zeros_like(hidden.grad[mask == 0]))
    if bias is not None:
        assert torch.allclose(bias.grad.float(), reference_bias.grad.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_linear_ratio_kl_rejects_active_out_of_range_action():
    hidden = torch.randn(2, 8, device="musa", dtype=torch.bfloat16)
    weight = torch.randn(7, 8, device="musa", dtype=torch.bfloat16)
    action = torch.tensor([0, 7], device="musa")
    mask = torch.ones(2, device="musa", dtype=torch.int32)
    values = torch.zeros(2, device="musa")
    with pytest.raises(ValueError, match="active positions"):
        MusaLinearRatioKLOp()(hidden, weight, action, mask, values, values)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_linear_ratio_kl_registry_dispatch():
    op = KernelRegistry().get_op("linear_ratio_kl", device="musa")
    assert op.__class__.__name__ == "MusaLinearRatioKLOp"
