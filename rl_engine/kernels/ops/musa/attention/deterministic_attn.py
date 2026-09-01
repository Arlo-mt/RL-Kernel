# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import math

import torch

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol


class _MusaDeterministicAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale, mask):
        mask_arg = mask if mask.numel() else None
        out, lse, probs = _C.deterministic_attention_fp32(
            q.contiguous(), k.contiguous(), v.contiguous(), causal, float(scale), mask_arg
        )
        ctx.save_for_backward(q, k, v, probs, mask)
        ctx.has_mask = mask.numel() != 0
        ctx.causal = causal
        ctx.scale = scale
        return out.to(q.dtype), lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        q, k, v, probs, mask = ctx.saved_tensors
        mask_arg = mask if ctx.has_mask else None
        dq, dk, dv = _C.deterministic_attention_backward(
            grad_out.contiguous(), q, k, v, probs, ctx.causal, float(ctx.scale), mask_arg
        )
        return dq, dk, dv, None, None, None


class MusaDeterministicAttentionOp:
    def __init__(self):
        require_musa_symbol("deterministic_attention_fp32")
        require_musa_symbol("deterministic_attention_backward")

    def __call__(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        return self.forward(q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask)

    def _apply(self, q, k, v, causal, scale, key_padding_mask):
        check_musa(q, k, v)
        if key_padding_mask is not None:
            check_musa(key_padding_mask)
        resolved_scale = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
        mask = (
            key_padding_mask.contiguous()
            if key_padding_mask is not None
            else torch.empty(0, device=q.device, dtype=torch.bool)
        )
        return _MusaDeterministicAttentionFunction.apply(
            q, k, v, causal, resolved_scale, mask
        )

    def forward(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        out, _lse = self._apply(q, k, v, causal, scale, key_padding_mask)
        return out

    def forward_with_lse(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        return self._apply(q, k, v, causal, scale, key_padding_mask)

    def forward_fp32(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        check_musa(q, k, v)
        if key_padding_mask is not None:
            check_musa(key_padding_mask)
        resolved_scale = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
        mask = (
            key_padding_mask.contiguous()
            if key_padding_mask is not None
            else torch.empty(0, device=q.device, dtype=torch.bool)
        )
        out, lse, _probs = _C.deterministic_attention_fp32(
            q.contiguous(), k.contiguous(), v.contiguous(),
            causal, float(resolved_scale), mask if mask.numel() else None
        )
        return out, lse
