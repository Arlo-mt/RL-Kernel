# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MUSA native fused selected-token log-probability backend."""

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


class _MusaFusedLogpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        token_ids_1d = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        output = _C.fused_logp(logits_2d, token_ids_1d)
        ctx.save_for_backward(logits_2d, token_ids_1d)
        ctx.input_shape = tuple(logits.shape)
        ctx.input_dtype = logits.dtype
        return output.reshape(logits.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, token_ids = ctx.saved_tensors
        probs = torch.softmax(logits.float(), dim=-1)
        rows = torch.arange(logits.size(0), device=logits.device)
        probs[rows, token_ids] -= 1.0
        grad = -grad_output.reshape(-1, 1).float() * probs
        return grad.to(ctx.input_dtype).reshape(ctx.input_shape), None


class FusedLogpOp:
    """Generic MUSA fused LogP; backward uses a portable PyTorch formula."""

    is_fused_logp = True

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_logp"):
            raise RuntimeError("MUSA fused_logp extension is unavailable")

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if logits.device.type != "musa":
            raise RuntimeError(f"FusedLogpOp requires a MUSA tensor, got {logits.device}")
        return _MusaFusedLogpFunction.apply(logits, token_ids)
