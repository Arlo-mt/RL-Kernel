# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict H100 QK-Norm and RoPE handoff for WS2 Attention.

This module intentionally has no runtime-native fallback.  A caller either runs
the RL-Kernel CUDA operators and records their identities, or the experiment
fails before Attention executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor


QK_RMSNORM_BACKEND_ID = "rlkernel.cuda.rmsnorm"
ROPE_BACKEND_ID = "rlkernel.cuda.rope_sm90"
MANDATED_ATTENTION_PREPROCESS_BACKENDS: Mapping[str, str] = MappingProxyType(
    {
        "qk_rmsnorm": QK_RMSNORM_BACKEND_ID,
        "rope": ROPE_BACKEND_ID,
    }
)


@dataclass(frozen=True)
class AttentionPreprocessResult:
    """Post-QK-Norm, post-RoPE tensors plus executed backend evidence."""

    q: Tensor
    k: Tensor
    backend_ids: Mapping[str, str]
    fallback: bool
    device_capability: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_ids", MappingProxyType(dict(self.backend_ids)))

    def evidence(self) -> dict[str, Any]:
        return {
            "backends": dict(self.backend_ids),
            "fallback": self.fallback,
            "device_capability": list(self.device_capability),
        }

    def readback_fields(self) -> dict[str, Any]:
        """Keyword fields consumed by ``AttentionRuntimeReadback``."""

        return {
            "preprocess_backends": dict(self.backend_ids),
            "preprocess_fallback": self.fallback,
        }


class H100AttentionPreprocessor:
    """Apply RL-Kernel CUDA QK-Norm then RoPE without silent fallback."""

    def __init__(self, device: torch.device | str | int | None = None) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("H100AttentionPreprocessor requires an available CUDA runtime")

        current_device = torch.cuda.current_device()
        self.device = torch.device("cuda", current_device)
        if device is not None:
            self.device = (
                torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            )
        if self.device.type != "cuda":
            raise RuntimeError(f"H100AttentionPreprocessor requires CUDA, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", current_device)

        capability = torch.cuda.get_device_capability(self.device)
        self.device_capability: tuple[int, int] = (int(capability[0]), int(capability[1]))
        if self.device_capability[0] != 9:
            raise RuntimeError(
                "H100AttentionPreprocessor requires Hopper SM90; "
                f"got sm_{self.device_capability[0]}{self.device_capability[1]}"
            )

        # Import only after the hardware gate so CPU tools can inspect the module.
        from rl_engine.kernels.ops.cuda.norm.rmsnorm import RMSNormCudaOp
        from rl_engine.kernels.ops.cuda.rotary_embedding.rope import RoPESM90Op

        self.rmsnorm = RMSNormCudaOp()
        self.rope = RoPESM90Op()
        actual_backends = {
            "qk_rmsnorm": self.rmsnorm.backend_id,
            "rope": self.rope.backend_id,
        }
        if actual_backends != dict(MANDATED_ATTENTION_PREPROCESS_BACKENDS):
            raise RuntimeError(f"unexpected Attention preprocess backends: {actual_backends}")
        self.backend_ids = MappingProxyType(actual_backends)

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        positions: Tensor,
        *,
        eps: float = 1.0e-6,
        theta: float = 1_000_000.0,
    ) -> AttentionPreprocessResult:
        return self.forward(
            q,
            k,
            q_weight,
            k_weight,
            positions,
            eps=eps,
            theta=theta,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        positions: Tensor,
        *,
        eps: float = 1.0e-6,
        theta: float = 1_000_000.0,
    ) -> AttentionPreprocessResult:
        _validate_inputs(q, k, q_weight, k_weight, positions, self.device)
        q_norm = self.rmsnorm(q, q_weight, eps=eps)
        k_norm = self.rmsnorm(k, k_weight, eps=eps)
        return AttentionPreprocessResult(
            q=self.rope(q_norm, positions, theta=theta),
            k=self.rope(k_norm, positions, theta=theta),
            backend_ids=self.backend_ids,
            fallback=False,
            device_capability=self.device_capability,
        )


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    positions: Tensor,
    device: torch.device,
) -> None:
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must use [B, H, S, D] layout")
    if q.shape[0] != k.shape[0] or q.shape[-2:] != k.shape[-2:]:
        raise ValueError("q and k must have the same batch, sequence, and head dimensions")
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        raise TypeError("the frozen H100 Attention experiment requires BF16 q and k")
    if q.device != device or k.device != device:
        raise ValueError(f"q and k must both be on the configured device {device}")
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if weight.shape != (q.shape[-1],):
            raise ValueError(f"{name} must have shape ({q.shape[-1]},)")
        if weight.device != device or weight.dtype is not torch.bfloat16:
            raise ValueError(f"{name} must be BF16 on {device}")
    if positions.device != device:
        raise ValueError(f"positions must be on {device}")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must use int32 or int64 global token indices")
    expected = (q.shape[-2],) if positions.dim() == 1 else (q.shape[0], q.shape[-2])
    if positions.dim() not in (1, 2) or tuple(positions.shape) != expected:
        raise ValueError(f"positions must have shape [S] or [B, S], expected {expected}")


__all__ = [
    "AttentionPreprocessResult",
    "H100AttentionPreprocessor",
    "MANDATED_ATTENTION_PREPROCESS_BACKENDS",
    "QK_RMSNORM_BACKEND_ID",
    "ROPE_BACKEND_ID",
]
