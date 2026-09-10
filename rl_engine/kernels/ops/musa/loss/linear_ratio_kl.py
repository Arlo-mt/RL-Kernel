# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional, Tuple

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


class _MusaLinearRatioKLFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        target,
        mask,
        old_logp,
        reference_logp,
        bias,
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        target_1d = target.reshape(-1).to(device=hidden.device, dtype=torch.long)
        mask_1d = mask.reshape(-1).to(device=hidden.device, dtype=torch.int32)
        safe_target = torch.where(mask_1d != 0, target_1d, torch.zeros_like(target_1d))
        old_1d = old_logp.reshape(-1).to(device=hidden.device, dtype=torch.float32)
        reference_1d = reference_logp.reshape(-1).to(device=hidden.device, dtype=torch.float32)
        bias_arg = bias if bias.numel() else None
        ratio, kl, diff, policy_lse = _C.fused_linear_ratio_kl_musa_forward(
            hidden_2d,
            weight.contiguous(),
            safe_target.contiguous(),
            mask_1d.contiguous(),
            old_1d.contiguous(),
            reference_1d.contiguous(),
            bias_arg,
        )
        ctx.save_for_backward(
            hidden_2d,
            weight,
            safe_target,
            mask_1d,
            ratio,
            diff,
            policy_lse,
            bias,
        )
        ctx.hidden_shape = tuple(hidden.shape)
        ctx.output_shape = tuple(target.shape)
        ctx.has_bias = bool(bias.numel())
        return ratio.view(ctx.output_shape), kl.view(ctx.output_shape)

    @staticmethod
    def backward(ctx, grad_ratio, grad_kl):
        hidden, weight, target, mask, ratio, diff, policy_lse, bias = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _C.fused_linear_ratio_kl_musa_backward(
            grad_ratio.reshape(-1).to(torch.float32).contiguous(),
            grad_kl.reshape(-1).to(torch.float32).contiguous(),
            hidden,
            weight,
            target,
            mask,
            ratio,
            diff,
            policy_lse,
            bias if ctx.has_bias else None,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[6],
        )
        return (
            grad_hidden.reshape(ctx.hidden_shape) if grad_hidden is not None else None,
            grad_weight,
            None,
            None,
            None,
            None,
            grad_bias,
        )


class MusaLinearRatioKLOp:
    """MUSA fused policy projection, importance ratio, and k3 KL."""

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or _C is None:
            raise RuntimeError("MUSA linear_ratio_kl requires the compiled extension")
        required = (
            "fused_linear_ratio_kl_musa_forward",
            "fused_linear_ratio_kl_musa_backward",
        )
        missing = [name for name in required if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(f"MUSA linear_ratio_kl extension is missing: {', '.join(missing)}")

    def __call__(
        self,
        hidden: torch.Tensor,
        policy_weight: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_logps: torch.Tensor,
        reference_logps: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden.device.type != "musa" or policy_weight.device.type != "musa":
            raise ValueError("MUSA linear_ratio_kl requires MUSA hidden and weight")
        if hidden.dtype != torch.bfloat16 or policy_weight.dtype != torch.bfloat16:
            raise TypeError("MUSA linear_ratio_kl currently supports BF16 hidden and weight")
        if hidden.ndim < 2 or policy_weight.ndim != 2:
            raise ValueError("expected hidden [..., H] and policy_weight [V, H]")
        if hidden.size(-1) != policy_weight.size(1):
            raise ValueError("hidden and policy weight dimensions do not match")
        expected = tuple(hidden.shape[:-1])
        for name, tensor in (
            ("action_ids", action_ids),
            ("attention_mask", attention_mask),
            ("old_logps", old_logps),
            ("reference_logps", reference_logps),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} shape must match hidden leading dimensions")
        if bias is not None:
            if (
                bias.device != hidden.device
                or bias.dtype != hidden.dtype
                or bias.shape != (policy_weight.size(0),)
            ):
                raise ValueError("bias must be a MUSA BF16 tensor with shape [V]")
        mask = attention_mask.to(torch.bool)
        invalid = mask & ((action_ids < 0) | (action_ids >= policy_weight.size(0)))
        if bool(invalid.any().item()):
            raise ValueError(
                f"action_ids at active positions must be in [0, {policy_weight.size(0)})"
            )
        bias_arg = bias if bias is not None else hidden.new_empty(0)
        return _MusaLinearRatioKLFunction.apply(
            hidden,
            policy_weight,
            action_ids,
            attention_mask,
            old_logps,
            reference_logps,
            bias_arg,
        )

    apply = __call__
