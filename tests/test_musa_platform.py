# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.musa.loss import MusaLinearLogpOp, MusaLogpOp
from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp
from rl_engine.kernels.registry import KernelRegistry, OpBackend
from rl_engine.platforms.device import is_musa_available


def test_musa_registry_uses_platform_adapters():
    registry = KernelRegistry()

    assert registry._priority_map["musa"]["logp"] == [OpBackend.MUSA_LOGP]
    assert registry._priority_map["musa"]["linear_logp"] == [OpBackend.MUSA_LINEAR_LOGP]
    assert registry._priority_map["musa"]["det_gemm"] == []


def test_musa_adapters_preserve_portable_pytorch_fallback():
    logits = torch.randn(2, 8)
    targets = torch.randint(0, 8, (2,))
    hidden = torch.randn(2, 8)
    weight = torch.randn(8, 8)

    torch.testing.assert_close(MusaLogpOp()(logits, targets), NativeLogpOp()(logits, targets))
    torch.testing.assert_close(
        MusaLinearLogpOp()(hidden, weight, targets),
        NativeLinearLogpOp()(hidden, weight, targets),
    )


@pytest.mark.skipif(not is_musa_available(), reason="MUSA is not available")
def test_musa_logp_and_linear_logp_forward_backward():
    device = torch.device("musa")
    registry = KernelRegistry()
    logits = torch.randn(4, 17, device=device, requires_grad=True)
    targets = torch.randint(0, 17, (4,), device=device)

    logp = registry.get_op("logp", device=device)(logits, targets)
    logp.sum().backward()

    hidden = torch.randn(4, 16, device=device, requires_grad=True)
    weight = torch.randn(17, 16, device=device, requires_grad=True)
    linear_logp = registry.get_op("linear_logp", device=device)(hidden, weight, targets)
    linear_logp.sum().backward()

    assert logp.device.type == "musa"
    assert linear_logp.device.type == "musa"
    assert logits.grad is not None
    assert hidden.grad is not None
    assert weight.grad is not None
