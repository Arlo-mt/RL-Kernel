# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.musa.loss.ratio_kl import MusaRatioKLOp


class MusaGRPOLossOp:
    """Native MUSA GRPO loss op."""

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or _C is None:
            raise RuntimeError("MUSA grpo_loss requires the compiled extension")
        required = ("grpo_group_advantages_musa",)
        missing = [name for name in required if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                f"MUSA grpo_loss extension is missing: {', '.join(missing)}"
            )
        self._ratio_kl = MusaRatioKLOp()

    def __call__(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        old_logps: torch.Tensor,
        rewards: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        clip_eps: float = 0.2,
        beta: float = 0.0,
        samples_per_prompt: Optional[int] = None,
        group_boundaries: Optional[Sequence[int] | torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(
            policy_logits,
            ref_logits,
            action_ids,
            old_logps,
            rewards,
            completion_mask,
            clip_eps=clip_eps,
            beta=beta,
            samples_per_prompt=samples_per_prompt,
            group_boundaries=group_boundaries,
            eps=eps,
        )

    @staticmethod
    def _build_bounds(
        num_sequences: int,
        device: torch.device,
        samples_per_prompt: Optional[int],
        group_boundaries: Optional[Sequence[int] | torch.Tensor],
    ) -> torch.Tensor:
        provided = [spec is not None for spec in (samples_per_prompt, group_boundaries)]
        if sum(provided) != 1:
            raise ValueError(
                "Provide exactly one of samples_per_prompt or group_boundaries."
            )

        if samples_per_prompt is not None:
            if samples_per_prompt < 2:
                raise ValueError(
                    "samples_per_prompt must be at least 2 for group normalization."
                )
            if num_sequences % samples_per_prompt != 0:
                raise ValueError(
                    f"num_sequences ({num_sequences}) must be divisible by "
                    f"samples_per_prompt ({samples_per_prompt})."
                )
            return torch.arange(
                0,
                num_sequences + 1,
                samples_per_prompt,
                device=device,
                dtype=torch.int32,
            )

        bounds = torch.as_tensor(group_boundaries, device=device, dtype=torch.int32)
        if bounds.ndim != 1 or bounds.numel() < 2:
            raise ValueError(
                "group_boundaries must be a 1D tensor of length num_groups + 1."
            )
        sizes = bounds[1:] - bounds[:-1]
        if int(bounds[0].item()) != 0 or int(bounds[-1].item()) != num_sequences:
            raise ValueError(
                "group_boundaries must start at 0 and end at num_sequences."
            )
        if bool((sizes < 1).any().item()):
            raise ValueError("each group must contain at least one sequence.")
        return bounds

    def group_advantages(
        self,
        rewards: torch.Tensor,
        *,
        samples_per_prompt: Optional[int] = None,
        group_boundaries: Optional[Sequence[int] | torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if rewards.device.type != "musa":
            raise RuntimeError("MusaGRPOLossOp requires MUSA tensors.")
        flat = rewards.reshape(-1).to(torch.float32).contiguous()
        bounds = self._build_bounds(
            flat.numel(), flat.device, samples_per_prompt, group_boundaries
        )
        return _C.grpo_group_advantages_musa(flat, bounds, float(eps))

    @staticmethod
    def expand_advantages(
        sample_advantages: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        bool_mask = completion_mask.bool()
        expanded = sample_advantages.reshape(-1, 1).expand_as(bool_mask).clone()
        return expanded.masked_fill(~bool_mask, 0.0)

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, bool_mask: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        masked = values.masked_fill(~bool_mask, 0.0)
        denom = bool_mask.sum().to(values.dtype).clamp_min(eps)
        return masked.sum() / denom

    def apply(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        old_logps: torch.Tensor,
        sample_advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        clip_eps: float = 0.2,
        beta: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if policy_logits.device.type != "musa":
            raise RuntimeError("MusaGRPOLossOp requires MUSA tensors.")
        if completion_mask.ndim != 2:
            raise ValueError(
                "completion_mask must be 2D [num_sequences, completion_len]."
            )

        ratio, kl_terms = self._ratio_kl(
            policy_logits, ref_logits, action_ids, completion_mask, old_logps
        )
        bool_mask = completion_mask.bool()
        adv = self.expand_advantages(sample_advantages, completion_mask).float()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss_terms = -torch.minimum(unclipped, clipped)

        policy_loss = self._masked_mean(policy_loss_terms, bool_mask)
        kl = self._masked_mean(kl_terms, bool_mask)
        return policy_loss + beta * kl, policy_loss, kl

    def forward(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        old_logps: torch.Tensor,
        rewards: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        clip_eps: float = 0.2,
        beta: float = 0.0,
        samples_per_prompt: Optional[int] = None,
        group_boundaries: Optional[Sequence[int] | torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_advantages = self.group_advantages(
            rewards,
            samples_per_prompt=samples_per_prompt,
            group_boundaries=group_boundaries,
            eps=eps,
        )
        return self.apply(
            policy_logits,
            ref_logits,
            action_ids,
            old_logps,
            sample_advantages,
            completion_mask,
            clip_eps=clip_eps,
            beta=beta,
        )
