# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol


class _MusaRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x_2d = x.reshape(-1, x.size(-1)).contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x_2d)
        rstd = torch.empty(x_2d.size(0), device=x.device, dtype=torch.float32)
        _C.rmsnorm_forward(x_2d, weight, y, rstd, float(eps))
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.input_shape = tuple(x.shape)
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        dy = grad_output.reshape(-1, x.size(-1)).contiguous()
        dx = torch.empty_like(x)
        _C.rmsnorm_backward_dx(dy, x, weight, rstd, dx.reshape_as(x))
        dw = (dy.float() * x.float() * rstd[:, None]).sum(dim=0).to(weight.dtype)
        return dx.reshape(ctx.input_shape), dw, None


class RMSNormOp:
    def __init__(self):
        require_musa_symbol("rmsnorm_forward")
        require_musa_symbol("rmsnorm_backward_dx")

    def __call__(self, x, weight, *, eps=1e-6):
        return self.forward(x, weight, eps=eps)

    def forward(self, x, weight, *, eps=1e-6):
        check_musa(x, weight)
        return _MusaRMSNormFunction.apply(x, weight, eps)

    def forward_fp32(self, x, weight, *, eps=1e-6):
        check_musa(x, weight)
        return _MusaRMSNormFunction.apply(x.float(), weight.float(), eps)
