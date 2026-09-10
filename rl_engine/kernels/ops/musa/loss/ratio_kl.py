# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Tuple

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


class _MusaRatioKLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, policy_logits, ref_logits, action_ids, attention_mask, old_logps):
        expected = tuple(policy_logits.shape[:-1])
        if ref_logits.shape != policy_logits.shape:
            raise ValueError("ref_logits shape must match policy_logits shape")
        for name, tensor in (
            ("action_ids", action_ids),
            ("attention_mask", attention_mask),
            ("old_logps", old_logps),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match "
                    f"policy_logits.shape[:-1] {expected}"
                )
        vocab = policy_logits.size(-1)
        policy = policy_logits.contiguous().view(-1, vocab)
        reference = ref_logits.contiguous().view(-1, vocab)
        mask = (
            attention_mask.contiguous()
            .view(-1)
            .to(device=policy.device, dtype=torch.int32)
        )
        action = (
            action_ids.contiguous().view(-1).to(device=policy.device, dtype=torch.long)
        )
        invalid = mask.to(torch.bool) & ((action < 0) | (action >= vocab))
        if bool(invalid.any().item()):
            raise ValueError(f"action_ids at active positions must be in [0, {vocab})")
        safe_action = torch.where(mask != 0, action, torch.zeros_like(action))
        old = (
            old_logps.contiguous()
            .view(-1)
            .to(device=policy.device, dtype=torch.float32)
        )

        ratio, kl, diff, policy_logz = _C.ratio_kl_musa_forward(
            policy, reference, safe_action, mask, old
        )
        ctx.save_for_backward(policy, safe_action, mask, ratio, diff, policy_logz)
        ctx.policy_shape = tuple(policy_logits.shape)
        ctx.output_shape = expected
        return ratio.view(expected), kl.view(expected)

    @staticmethod
    def backward(ctx, grad_ratio, grad_kl):
        policy, action, mask, ratio, diff, policy_logz = ctx.saved_tensors
        grad_policy = _C.ratio_kl_musa_backward(
            policy,
            action,
            mask,
            ratio,
            diff,
            policy_logz,
            grad_ratio.contiguous().view(-1).to(torch.float32),
            grad_kl.contiguous().view(-1).to(torch.float32),
        )
        return grad_policy.view(ctx.policy_shape), None, None, None, None


class MusaRatioKLOp:
    """Native MUSA policy-ratio and k3 KL operator."""

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or _C is None:
            raise RuntimeError("MUSA ratio_kl requires the compiled extension")
        required = ("ratio_kl_musa_forward", "ratio_kl_musa_backward")
        missing = [name for name in required if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                f"MUSA ratio_kl extension is missing: {', '.join(missing)}"
            )

    def __call__(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if policy_logits.device.type != "musa" or ref_logits.device.type != "musa":
            raise ValueError("MUSA ratio_kl requires MUSA policy and reference logits")
        if policy_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("MUSA ratio_kl supports FP16, BF16, and FP32 logits")
        if ref_logits.dtype != policy_logits.dtype:
            raise TypeError("policy_logits and ref_logits dtypes must match")
        return _MusaRatioKLFunction.apply(
            policy_logits, ref_logits, action_ids, attention_mask, old_logps
        )
