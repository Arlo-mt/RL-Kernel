# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any, Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.loss.linear_logp import (
    should_use_tensor_parallel_linear_logp,
    tensor_parallel_linear_logp,
)


class _MusaLinearLogpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        target_1d = target_ids.reshape(-1).to(device=hidden.device, dtype=torch.long).contiguous()
        bias_arg = bias if bias.numel() else None
        logp, lse = _C.fused_linear_logp_musa_forward(
            hidden_2d, weight.contiguous(), target_1d, bias_arg
        )
        ctx.save_for_backward(
            hidden_2d,
            weight.contiguous(),
            target_1d,
            lse,
            bias,
        )
        ctx.has_bias = bool(bias.numel())
        ctx.input_shape = tuple(hidden.shape)
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden, weight, target, lse, bias = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _C.fused_linear_logp_musa_backward(
            grad_logp.reshape(-1).contiguous(),
            hidden,
            weight,
            target,
            lse,
            bias if ctx.has_bias else None,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        has_hidden_grad = grad_hidden is not None and grad_hidden.numel() > 0
        has_weight_grad = grad_weight is not None and grad_weight.numel() > 0
        has_bias_grad = grad_bias is not None and grad_bias.numel() > 0
        return (
            grad_hidden.reshape(ctx.input_shape) if has_hidden_grad else None,
            grad_weight if has_weight_grad else None,
            grad_bias if has_bias_grad else None,
            None,
        )


class MusaLinearLogpOp:
    """Native MUSA fused linear selected-token log-probability operator."""

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or _C is None:
            raise RuntimeError("MUSA linear_logp requires the compiled extension")
        required = ("fused_linear_logp_musa_forward", "fused_linear_logp_musa_backward")
        missing = [name for name in required if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(f"MUSA linear_logp extension is missing: {', '.join(missing)}")

    @staticmethod
    def _validate(hidden, weight, target_ids, bias):
        if hidden.device.type != "musa" or weight.device.type != "musa":
            raise ValueError("MUSA linear_logp requires MUSA hidden and weight tensors")
        if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("MUSA linear_logp currently supports BF16 hidden and weight tensors")
        if hidden.ndim < 2 or weight.ndim != 2:
            raise ValueError("expected hidden [..., K] and weight [V, K]")
        if hidden.size(-1) != weight.size(1):
            raise ValueError("hidden and weight dimensions do not match")
        if tuple(hidden.shape[:-1]) != tuple(target_ids.shape):
            raise ValueError("target_ids must match hidden leading dimensions")
        if bias is not None and (bias.device != hidden.device or bias.numel() != weight.size(0)):
            raise ValueError("bias must be a MUSA tensor with one value per vocabulary row")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        self._validate(hidden, lm_head_weight, target_ids, bias)
        if should_use_tensor_parallel_linear_logp(
            tp_group, int(vocab_start_index), global_vocab_size, lm_head_weight.size(0)
        ):
            return tensor_parallel_linear_logp(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )
        bias_arg = bias if bias is not None else hidden.new_empty(0)
        return _MusaLinearLogpFunction.apply(hidden, lm_head_weight, bias_arg, target_ids)

    apply = __call__
