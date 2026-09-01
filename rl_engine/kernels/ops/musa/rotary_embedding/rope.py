# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol


def _cos_sin(positions, half, theta, device):
    inv = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    freqs = positions.to(device=device, dtype=torch.float32).reshape(-1, 1) * inv
    return freqs.cos().contiguous(), freqs.sin().contiguous()


class _MusaRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positions, theta):
        if positions.dim() != 1:
            raise NotImplementedError("MUSA native RoPE currently supports positions [S]")
        half = x.size(-1) // 2
        cos, sin = _cos_sin(positions, half, theta, x.device)
        x_2d = x.contiguous().reshape(-1, x.size(-1))
        output = _C.rope_apply(x_2d, cos, sin, 1.0).reshape_as(x)
        ctx.save_for_backward(cos, sin)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        grad = grad_output.contiguous().reshape(-1, grad_output.size(-1))
        return _C.rope_apply(grad, cos, sin, -1.0).reshape_as(grad_output), None, None


class MusaRoPEOp:
    def __init__(self):
        require_musa_symbol("rope_apply")

    def __call__(self, x, positions, *, theta=1_000_000.0):
        return self.forward(x, positions, theta=theta)

    def forward(self, x, positions, *, theta=1_000_000.0):
        check_musa(x)
        return _MusaRoPEFunction.apply(x, positions, float(theta))

    def forward_fp32(self, x, positions, *, theta=1_000_000.0):
        check_musa(x)
        return _MusaRoPEFunction.apply(x.float(), positions, float(theta))
