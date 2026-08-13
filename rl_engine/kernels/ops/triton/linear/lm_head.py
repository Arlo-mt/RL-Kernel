# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton deterministic LM head built on the pinned no-split-K GEMM."""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.triton.matmul.det_gemm import deterministic_gemm_triton


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
        flat = hidden.reshape(-1, hidden.size(-1)).contiguous()
        out = deterministic_gemm_triton(flat, weight.t().contiguous())
        if bias is not None:
            out = out + bias
        return out.reshape(*hidden.shape[:-1], weight.size(0))

    def parameter_vjp_contributions_fp32(self, *, hidden, weight, grad_output, bias=None):
        del weight, bias
        rows_h = hidden.reshape(-1, hidden.size(-1)).float()
        rows_g = grad_output.reshape(-1, grad_output.size(-1)).float()
        return {"weight": rows_g[:, :, None] * rows_h[:, None, :]}
