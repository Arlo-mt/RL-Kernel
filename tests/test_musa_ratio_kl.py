# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.musa.loss.ratio_kl import MusaRatioKLOp
from rl_engine.kernels.ops.pytorch.loss.ratio_kl import NativeRatioKLOp
from rl_engine.kernels.registry import KernelRegistry


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        pytest.param(torch.float32, 2e-5, 2e-5, id="fp32"),
        pytest.param(torch.float16, 2e-3, 2e-3, id="fp16"),
        pytest.param(torch.bfloat16, 2e-2, 2e-2, id="bf16"),
    ],
)
def test_musa_ratio_kl_matches_reference_forward_and_backward(dtype, atol, rtol):
    from rl_engine import _C

    assert hasattr(_C, "ratio_kl_musa_forward")
    assert hasattr(_C, "ratio_kl_musa_backward")
    torch.manual_seed(2100)
    policy = (
        torch.randn(3, 5, 257, device="musa", dtype=dtype) * 0.25
    ).requires_grad_()
    reference = torch.randn_like(policy) * 0.25
    action = torch.randint(0, 257, (3, 5), device="musa", dtype=torch.long)
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1],
        ],
        device="musa",
        dtype=torch.int32,
    )
    action = action.clone()
    action[mask == 0] = 9999
    old_logps = torch.randn(3, 5, device="musa", dtype=torch.float32) * 0.1
    grad_ratio = torch.linspace(-0.75, 1.25, 15, device="musa").view(3, 5)
    grad_kl = torch.linspace(1.5, -0.5, 15, device="musa").view(3, 5)

    ratio, kl = MusaRatioKLOp()(policy, reference, action, mask, old_logps)
    reference_policy = policy.detach().clone().requires_grad_(True)
    expected_ratio, expected_kl = NativeRatioKLOp()(
        reference_policy, reference, action, mask, old_logps
    )
    assert torch.allclose(ratio, expected_ratio, atol=atol, rtol=rtol)
    assert torch.allclose(kl, expected_kl, atol=atol, rtol=rtol)
    assert torch.equal(ratio[mask == 0], torch.ones_like(ratio[mask == 0]))
    assert torch.equal(kl[mask == 0], torch.zeros_like(kl[mask == 0]))

    torch.autograd.backward((ratio, kl), (grad_ratio, grad_kl))
    torch.autograd.backward(
        (expected_ratio, expected_kl),
        (grad_ratio, grad_kl),
    )
    assert policy.grad is not None
    assert reference_policy.grad is not None
    assert torch.allclose(
        policy.grad.float(),
        reference_policy.grad.to(dtype).float(),
        atol=atol,
        rtol=rtol,
    )
    assert torch.equal(
        policy.grad[mask == 0],
        torch.zeros_like(policy.grad[mask == 0]),
    )


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_ratio_kl_rejects_active_out_of_range_action():
    policy = torch.randn(2, 7, device="musa")
    reference = torch.randn_like(policy)
    action = torch.tensor([0, 7], device="musa")
    mask = torch.ones(2, device="musa", dtype=torch.int32)
    old_logps = torch.zeros(2, device="musa")
    with pytest.raises(ValueError, match="active positions"):
        MusaRatioKLOp()(policy, reference, action, mask, old_logps)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_ratio_kl_registry_dispatch():
    op = KernelRegistry().get_op("ratio_kl", device="musa")
    assert op.__class__.__name__ == "MusaRatioKLOp"
