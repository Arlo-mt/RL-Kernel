# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol


class _MusaDeterministicLogpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, token_ids: torch.Tensor):
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        ids = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        output = _C.deterministic_logp_fp32(logits_2d, ids)
        ctx.save_for_backward(logits_2d, ids)
        ctx.input_shape = tuple(logits.shape)
        ctx.input_dtype = logits.dtype
        return output.reshape(logits.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        logits, ids = ctx.saved_tensors
        probs = torch.softmax(logits.float(), dim=-1)
        rows = torch.arange(logits.size(0), device=logits.device)
        probs[rows, ids] -= 1.0
        grad = -grad_output.reshape(-1, 1).float() * probs
        return grad.to(ctx.input_dtype).reshape(ctx.input_shape), None


class MusaDeterministicLogpOp:
    """Batch-invariant selected log-probability using the native MUSA kernel."""

    is_batch_invariant = True

    def __init__(self):
        require_musa_symbol("deterministic_logp_fp32")

    def __call__(self, logits, token_ids):
        return self.apply_fp32(logits, token_ids)

    def apply(self, logits, token_ids):
        return self.apply_fp32(logits, token_ids)

    def apply_fp32(self, logits, token_ids):
        check_musa(logits, token_ids)
        return _MusaDeterministicLogpFunction.apply(logits, token_ids)

    forward = apply_fp32
    forward_fp32 = apply_fp32
    online_fp32 = apply_fp32

    def out(self, logits, token_ids, output):
        output.copy_(self.apply_fp32(logits, token_ids).to(output.dtype))
        return output

    def indexed_fp32(self, logits, token_ids, row_indices):
        output = torch.zeros(logits.shape[:-1], device=logits.device, dtype=torch.float32)
        return self.indexed_out(logits, token_ids, row_indices, output)

    def indexed_out(self, logits, token_ids, row_indices, output):
        full = self.apply_fp32(logits, token_ids).reshape(-1)
        indices = row_indices.reshape(-1).to(device=full.device, dtype=torch.long)
        output.reshape(-1).index_copy_(0, indices, full.index_select(0, indices).to(output.dtype))
        return output

    online_out = out
    online_indexed_fp32 = indexed_fp32
    online_indexed_out = indexed_out
