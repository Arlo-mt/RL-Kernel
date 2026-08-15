# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Custom CUDA RoPE op for SM90 (GPT-NeoX rotate-half), matching NativeRoPEOp.

cos/sin are built in fp32 with the exact reference math and passed to a small
CUDA kernel (``_C.rope_apply_sm90``) that does the per-position rotation. Backward
reuses the same kernel with the sine negated (RoPE is an orthogonal rotation, so
``grad_x = grad_out * cos - rotate_half(grad_out) * sin``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


def _build_cos_sin(positions: Tensor, half: int, theta: float, device: torch.device):
    """fp32 cos/sin rows, identical math to NativeRoPEOp."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = positions.to(device=device, dtype=torch.float32).reshape(-1, 1)
    freqs = pos * inv_freq  # [S, half]
    return freqs.cos().contiguous(), freqs.sin().contiguous()


class _RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        D = x.shape[-1]
        if D % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {D}")
        if positions.dim() not in (1, 2):
            raise ValueError("positions must have shape [S] or [B, S]")
        S = positions.shape[-1]
        if S == 0:
            raise ValueError("positions must not be empty")
        if x.shape[-2] != S:
            raise ValueError(f"x sequence length {x.shape[-2]} does not match positions length {S}")
        x_2d = x.contiguous().reshape(-1, D)
        n_rows = x_2d.shape[0]
        if positions.dim() == 2:
            batch = positions.shape[0]
            if x.dim() < 3 or x.shape[0] != batch:
                raise ValueError(
                    f"x batch size {x.shape[0]} does not match positions batch size {batch}"
                )
            rows_per_token = n_rows // (batch * S)
            if rows_per_token * batch * S != n_rows:
                raise ValueError("x rows are incompatible with [B, S] positions")
            # The CUDA kernel accepts one fp32 cos/sin row per flattened x row.
            # Expanding positions preserves arbitrary global/zigzag indices while
            # keeping the arithmetic inside the precompiled deterministic kernel.
            kernel_positions = (
                positions[:, None, :].expand(batch, rows_per_token, S).contiguous().reshape(-1)
            )
        else:
            kernel_positions = positions
        if n_rows % kernel_positions.numel() != 0:
            raise ValueError(
                f"row count {n_rows} not divisible by position rows "
                f"{kernel_positions.numel()}; "
                "expected a [..., S, D] contiguous layout."
            )
        cos, sin = _build_cos_sin(kernel_positions, D // 2, float(theta), x.device)
        ctx.save_for_backward(cos, sin)
        out = _C.rope_apply_sm90(x_2d, cos, sin, 1.0)
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            D = grad_out.shape[-1]
            g_2d = grad_out.contiguous().reshape(-1, D)
            # Inverse rotation: same kernel with the sine negated.
            grad_x = _C.rope_apply_sm90(g_2d, cos, sin, -1.0).reshape(grad_out.shape)
        # Inputs: x, positions, theta.
        return grad_x, None, None


class RoPESM90Op:
    """Custom CUDA RoPE op for SM90 (GPT-NeoX rotate-half), differentiable w.r.t. ``x``.

    Qwen3 defaults: theta=1e6, head_dim=128, full-dimension rotation. cos/sin are
    computed in fp32 from ``positions`` and ``theta`` (matching the reference); the
    rotation runs in the precompiled ``_C.rope_apply_sm90`` CUDA kernel.
    """

    op_class = "elementwise"
    backend_id = "rlkernel.cuda.rope_sm90"

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "rope_apply_sm90"):
            raise RuntimeError(
                "CUDA RoPE kernel 'rope_apply_sm90' is not compiled into _C. "
                "Rebuild the extension with 'KERNEL_ALIGN_FORCE_SM90=1 pip install -e .'."
            )
        logger.info("Successfully linked to precompiled _C.rope_apply_sm90 kernel.")

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        if x.device.type != "cuda":
            raise RuntimeError(f"RoPESM90Op requires a CUDA tensor, got device '{x.device}'.")
        return _RoPEFunction.apply(x, positions, theta)
