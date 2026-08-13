# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton deterministic LM head built on the pinned no-split-K GEMM."""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.triton.matmul.det_gemm import _triton_gemm


class _TritonLMHeadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias):
        ctx.save_for_backward(hidden, weight, bias if bias is not None else hidden.new_empty(0))
        ctx.has_bias = bias is not None
        flat = hidden.reshape(-1, hidden.size(-1)).contiguous()
        out = _triton_gemm(flat, weight.t().contiguous())
        if bias is not None:
            out = out + bias
        return out.reshape(*hidden.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, weight.size(0)).float()
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).float()
        grad_hidden = grad_2d.matmul(weight.float()).reshape_as(hidden).to(hidden.dtype)
        grad_weight = grad_2d.transpose(0, 1).matmul(hidden_2d).to(weight.dtype)
        grad_bias = grad_2d.sum(0).to(bias.dtype) if ctx.has_bias else None
        return grad_hidden, grad_weight, grad_bias


class TritonLMHeadOp:
    op_class = "reduction"
    is_batch_invariant = True

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(hidden, weight, bias=bias)

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("TritonLMHeadOp requires BF16 hidden and weight")
        return _TritonLMHeadFn.apply(hidden, weight, bias)

    def parameter_vjp_contributions_fp32(self, *, hidden, weight, grad_output, bias=None):
        del weight, bias
        rows_h = hidden.reshape(-1, hidden.size(-1)).float()
        rows_g = grad_output.reshape(-1, grad_output.size(-1)).float()
        return {"weight": rows_g[:, :, None] * rows_h[:, None, :]}
