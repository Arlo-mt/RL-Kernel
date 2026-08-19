# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bitwise-bound QK-Norm and RoPE handoff for WS2 Attention.

Megatron/TE and vLLM/FlashInfer remain the first-choice implementations.  They
are admitted only after the same-input H100 probe is bitwise identical to the
deterministic RL-Kernel path.  A failed probe (or an unavailable native backend)
selects the common RL-Kernel path for both sides and records why the fallback
was taken.  The caller must pass this readback to the cross-config binder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

QK_RMSNORM_BACKEND_ID = "rlkernel.cuda.rmsnorm"
ROPE_BACKEND_ID = "rlkernel.cuda.rope_sm90"
NATIVE_QK_RMSNORM_BACKEND_ID = "native.qk_rmsnorm"
NATIVE_ROPE_BACKEND_ID = "native.rope"
PREPROCESS_POLICY_ID = "ws2.attention.preprocess.v2"
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
    fallback_reason: str | None = None
    probe_id: str = ""
    policy_id: str = PREPROCESS_POLICY_ID

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
            "preprocess_fallback_reason": self.fallback_reason,
            "preprocess_probe_id": self.probe_id,
            "preprocess_policy_id": self.policy_id,
        }


class H100AttentionPreprocessor:
    """Apply native QK-Norm/RoPE when the H100 probe passes.

    ``native_qk_norm`` and ``native_rope`` are framework-owned callables.  They
    are intentionally injected instead of importing TE/vLLM here, so the same
    policy can be used by both runtimes.  The deterministic callables default to
    RL-Kernel's CUDA operators and are always run to establish the probe oracle.
    """

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        *,
        native_qk_norm: Callable[..., Tensor] | None = None,
        native_rope: Callable[..., Tensor] | None = None,
        native_qk_norm_backend_id: str = NATIVE_QK_RMSNORM_BACKEND_ID,
        native_rope_backend_id: str = NATIVE_ROPE_BACKEND_ID,
        policy_id: str = PREPROCESS_POLICY_ID,
    ) -> None:
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
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        self.native_qk_norm = native_qk_norm
        self.native_rope = native_rope
        self.native_qk_norm_backend_id = native_qk_norm_backend_id
        self.native_rope_backend_id = native_rope_backend_id
        self.policy_id = policy_id

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
        q_norm_det = self.rmsnorm(q, q_weight, eps=eps)
        k_norm_det = self.rmsnorm(k, k_weight, eps=eps)
        q_det = _apply_deterministic_rope(self.rope, q_norm_det, positions, theta)
        k_det = _apply_deterministic_rope(self.rope, k_norm_det, positions, theta)

        native_qk_norm = self.native_qk_norm
        native_rope = self.native_rope
        if native_qk_norm is not None and native_rope is not None:
            try:
                q_norm_native = native_qk_norm(q, q_weight, eps=eps)
                k_norm_native = native_qk_norm(k, k_weight, eps=eps)
                q_native = native_rope(q_norm_native, positions, theta=theta)
                k_native = native_rope(k_norm_native, positions, theta=theta)
                probe_id = _probe_id(q, k, q_weight, k_weight, positions, eps, theta)
                if (
                    torch.equal(q_norm_native, q_norm_det)
                    and torch.equal(k_norm_native, k_norm_det)
                    and torch.equal(q_native, q_det)
                    and torch.equal(k_native, k_det)
                ):
                    return AttentionPreprocessResult(
                        q=q_native,
                        k=k_native,
                        backend_ids=MappingProxyType(
                            {
                                "qk_rmsnorm": self.native_qk_norm_backend_id,
                                "rope": self.native_rope_backend_id,
                            }
                        ),
                        fallback=False,
                        device_capability=self.device_capability,
                        probe_id=probe_id,
                        policy_id=self.policy_id,
                    )
                fallback_reason = "native_preprocess_bitwise_probe_failed"
            except Exception as exc:  # framework backend failures must fail over together
                fallback_reason = f"native_preprocess_unavailable:{type(exc).__name__}"
                probe_id = _probe_id(q, k, q_weight, k_weight, positions, eps, theta)
        else:
            fallback_reason = "native_preprocess_not_supplied"
            probe_id = _probe_id(q, k, q_weight, k_weight, positions, eps, theta)
        return AttentionPreprocessResult(
            q=q_det,
            k=k_det,
            backend_ids=MANDATED_ATTENTION_PREPROCESS_BACKENDS,
            fallback=True,
            device_capability=self.device_capability,
            fallback_reason=fallback_reason,
            probe_id=probe_id,
            policy_id=self.policy_id,
        )


def _apply_deterministic_rope(
    rope: Callable[..., Tensor],
    x: Tensor,
    positions: Tensor,
    theta: float,
) -> Tensor:
    """Adapt per-sample positions to the CUDA RoPE operator's 1-D contract."""

    if positions.dim() == 1:
        return rope(x, positions, theta=theta)
    return torch.cat(
        [rope(x[index : index + 1], positions[index], theta=theta) for index in range(x.shape[0])],
        dim=0,
    )


def _probe_id(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    positions: Tensor,
    eps: float,
    theta: float,
) -> str:
    payload = {
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "q_dtype": str(q.dtype),
        "k_dtype": str(k.dtype),
        "weight_dtype": str(q_weight.dtype),
        "positions_shape": list(positions.shape),
        "positions_sha256": hashlib.sha256(positions.detach().cpu().numpy().tobytes()).hexdigest(),
        "eps": float(eps),
        "theta": float(theta),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


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
    "NATIVE_QK_RMSNORM_BACKEND_ID",
    "NATIVE_ROPE_BACKEND_ID",
    "PREPROCESS_POLICY_ID",
]
