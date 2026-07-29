# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.linear.embedding import NativeEmbeddingOp
from rl_engine.utils.logger import logger

_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _is_hopper(device: torch.device) -> bool:
    try:
        return torch.cuda.get_device_capability(device)[0] == 9
    except Exception:
        return False


class _SM90EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids: torch.Tensor, weight: torch.Tensor, output_fp32: bool):
        ctx.save_for_backward(token_ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.output_fp32 = bool(output_fp32)
        if output_fp32:
            return _C.embedding_sm90_forward_fp32(token_ids, weight.contiguous())
        return _C.embedding_sm90_forward(token_ids, weight.contiguous())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (token_ids,) = ctx.saved_tensors
        grad_weight = None
        if ctx.needs_input_grad[1]:
            ids = token_ids.reshape(-1).to(device=grad_output.device, dtype=torch.long)
            hidden_size = int(ctx.weight_shape[1])
            grad_rows = grad_output.reshape(ids.numel(), hidden_size)
            valid = (ids >= 0) & (ids < int(ctx.weight_shape[0]))
            if not bool(valid.all().item()):
                ids = ids[valid]
                grad_rows = grad_rows[valid]
            grad_weight = torch.zeros(
                ctx.weight_shape,
                device=grad_output.device,
                dtype=grad_rows.dtype,
            )
            grad_weight.index_add_(0, ids, grad_rows)
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return None, grad_weight, None


class SM90EmbeddingOp(torch.nn.Module):
    """Single-card SM90 batch-invariant CUDA embedding op."""

    op_class = "elementwise"
    is_batch_invariant = True

    def __init__(self) -> None:
        super().__init__()
        if not _EXT_AVAILABLE or not hasattr(_C, "embedding_sm90_forward"):
            raise RuntimeError(
                "embedding_sm90_forward is not compiled into the extension. "
                "Rebuild on Hopper with KERNEL_ALIGN_FORCE_SM90=1."
            )
        self._fallback = NativeEmbeddingOp()
        logger.info("Successfully linked to precompiled _C.embedding_sm90_forward kernel.")

    def forward(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not self._can_use_sm90(token_ids, weight):
            return self._fallback.forward(token_ids, weight)
        return _SM90EmbeddingFunction.apply(token_ids, weight, False)

    def forward_fp32(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not self._can_use_sm90(token_ids, weight):
            return self._fallback.forward_fp32(token_ids, weight)
        return _SM90EmbeddingFunction.apply(token_ids, weight, True)

    @staticmethod
    def _can_use_sm90(token_ids: torch.Tensor, weight: torch.Tensor) -> bool:
        return (
            token_ids.is_cuda
            and weight.is_cuda
            and token_ids.device == weight.device
            and _is_hopper(weight.device)
            and weight.dim() == 2
            and weight.dtype in _SUPPORTED_DTYPES
        )
