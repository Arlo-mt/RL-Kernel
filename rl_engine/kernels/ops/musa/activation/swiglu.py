# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol
from rl_engine.kernels.ops.base import _C


class _MusaSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _C.silu_forward(x.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return _C.silu_backward(grad_output.contiguous(), x)


class SiLUOp:
    def __init__(self):
        require_musa_symbol("silu_forward")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        check_musa(x)
        return _MusaSiLUFunction.apply(x)

    def forward_fp32(self, x):
        check_musa(x)
        return _MusaSiLUFunction.apply(x.float())


class _MusaSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return _C.swiglu_forward(gate.contiguous(), up.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        d_gate, d_up = _C.swiglu_backward(grad_output.contiguous(), gate, up)
        return d_gate, d_up


class SwiGLUOp:
    def __init__(self):
        require_musa_symbol("swiglu_forward")

    def __call__(self, gate, up):
        return self.forward(gate, up)

    def forward(self, gate, up):
        check_musa(gate, up)
        return _MusaSwiGLUFunction.apply(gate, up)

    def forward_fp32(self, gate, up):
        check_musa(gate, up)
        return _MusaSwiGLUFunction.apply(gate.float(), up.float())
