# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
from typing import Optional

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol


class _MusaLMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, output_dtype):
        output = _C.lm_head_fp32(hidden, weight, bias)
        ctx.save_for_backward(hidden, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        return output.to(output_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias = ctx.saved_tensors
        grad = grad_output.float().reshape(-1, grad_output.size(-1))
        hidden_2d = hidden.float().reshape(-1, hidden.size(-1))
        grad_hidden = (grad @ weight.float()).reshape_as(hidden)
        grad_weight = grad.t() @ hidden_2d
        grad_bias = grad.sum(dim=0) if ctx.has_bias else None
        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), grad_bias, None


class MusaLMHeadOp:
    def __init__(self):
        require_musa_symbol("lm_head_fp32")

    def __call__(self, hidden, weight, *, bias: Optional[torch.Tensor] = None):
        return self.forward(hidden, weight, bias=bias)

    def forward(self, hidden, weight, *, bias: Optional[torch.Tensor] = None):
        check_musa(hidden, weight)
        if bias is not None:
            check_musa(bias)
        return _MusaLMHeadFunction.apply(hidden, weight, bias, hidden.dtype)

    def forward_fp32(self, hidden, weight, *, bias: Optional[torch.Tensor] = None):
        check_musa(hidden, weight)
        if bias is not None:
            check_musa(bias)
        return _C.lm_head_fp32(hidden, weight, bias)
