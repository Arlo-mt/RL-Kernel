# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Wrappers for the native MUSA kernels exposed by ``rl_engine._C``."""

from __future__ import annotations

import math
from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


def _require(symbol: str) -> None:
    if not _EXT_AVAILABLE or _C is None or not hasattr(_C, symbol):
        raise RuntimeError(f"MUSA native symbol {symbol!r} is unavailable")


def _check_musa(*tensors: torch.Tensor) -> None:
    if any(t.device.type != "musa" for t in tensors):
        raise RuntimeError("MUSA native kernels require MUSA tensors")


class _MusaLogpFunction(torch.autograd.Function):
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
    is_batch_invariant = True

    def __init__(self):
        _require("deterministic_logp_fp32")

    def __call__(self, logits, token_ids):
        return self.apply_fp32(logits, token_ids)

    def apply(self, logits, token_ids):
        return self.apply_fp32(logits, token_ids)

    def apply_fp32(self, logits, token_ids):
        _check_musa(logits, token_ids)
        return _MusaLogpFunction.apply(logits, token_ids)

    forward = apply_fp32
    forward_fp32 = apply_fp32
    online_fp32 = apply_fp32

    def indexed_fp32(self, logits, token_ids, row_indices):
        output = torch.zeros(logits.shape[:-1], device=logits.device, dtype=torch.float32)
        return self.indexed_out(logits, token_ids, row_indices, output)

    def indexed_out(self, logits, token_ids, row_indices, output):
        full = self.apply_fp32(logits, token_ids)
        output.reshape(-1).index_copy_(
            0,
            row_indices.reshape(-1).to(device=output.device, dtype=torch.long),
            full.reshape(-1).index_select(
                0, row_indices.reshape(-1).to(device=full.device, dtype=torch.long)
            ).to(output.dtype),
        )
        return output

    out = indexed_out
    online_indexed_fp32 = indexed_fp32
    online_indexed_out = indexed_out


class _MusaSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _C.silu_forward(x.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return _C.silu_backward(grad_output.contiguous(), x)


class MusaSiLUOp:
    def __init__(self):
        _require("silu_forward")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        _check_musa(x)
        return _MusaSiLUFunction.apply(x)

    def forward_fp32(self, x):
        _check_musa(x)
        return _MusaSiLUFunction.apply(x.float())


class _MusaSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return _C.swiglu_forward(gate.contiguous(), up.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        d_gate, d_up = _C.swiglu_backward(grad_output.contiguous(), gate, up)
        return d_gate, d_up


class MusaSwiGLUOp:
    def __init__(self):
        _require("swiglu_forward")

    def __call__(self, gate, up):
        return self.forward(gate, up)

    def forward(self, gate, up):
        _check_musa(gate, up)
        return _MusaSwiGLUFunction.apply(gate, up)

    def forward_fp32(self, gate, up):
        _check_musa(gate, up)
        return _MusaSwiGLUFunction.apply(gate.float(), up.float())


class _MusaRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x_2d = x.reshape(-1, x.size(-1)).contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x_2d)
        rstd = torch.empty(x_2d.size(0), device=x.device, dtype=torch.float32)
        _C.rmsnorm_forward(x_2d, weight, y, rstd, float(eps))
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.input_shape = tuple(x.shape)
        ctx.eps = eps
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        dy = grad_output.reshape_as(x).reshape(-1, x.size(-1)).contiguous()
        dx = torch.empty_like(x)
        _C.rmsnorm_backward_dx(dy, x, weight, rstd, dx)
        dw = (dy.float() * x.float() * rstd[:, None]).sum(dim=0).to(weight.dtype)
        return dx.reshape(ctx.input_shape), dw, None


class MusaRMSNormOp:
    def __init__(self):
        _require("rmsnorm_forward")
        _require("rmsnorm_backward_dx")

    def __call__(self, x, weight, *, eps=1e-6):
        return self.forward(x, weight, eps=eps)

    def forward(self, x, weight, *, eps=1e-6):
        _check_musa(x, weight)
        return _MusaRMSNormFunction.apply(x, weight, eps)

    def forward_fp32(self, x, weight, *, eps=1e-6):
        _check_musa(x, weight)
        return _MusaRMSNormFunction.apply(x.float(), weight.float(), eps)


class _MusaEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, weight):
        ids = token_ids.reshape(-1).to(dtype=torch.long).contiguous()
        output = _C.embedding_fp32(ids, weight.contiguous()).to(weight.dtype)
        ctx.save_for_backward(ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.output_shape = tuple(token_ids.shape) + (weight.size(1),)
        return output.reshape(ctx.output_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (ids,) = ctx.saved_tensors
        grad_weight = torch.zeros(
            ctx.weight_shape, device=grad_output.device, dtype=ctx.weight_dtype
        )
        grad_weight.index_add_(0, ids, grad_output.reshape(-1, ctx.weight_shape[1]))
        return None, grad_weight


class MusaEmbeddingOp:
    def __init__(self):
        _require("embedding_fp32")

    def __call__(self, token_ids, weight):
        return self.forward(token_ids, weight)

    def forward(self, token_ids, weight):
        _check_musa(token_ids, weight)
        return _MusaEmbeddingFunction.apply(token_ids, weight)

    def forward_fp32(self, token_ids, weight):
        _check_musa(token_ids, weight)
        return _C.embedding_fp32(token_ids.reshape(-1).long(), weight).reshape(
            *token_ids.shape, weight.size(1)
        )


class _MusaLMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, output_dtype):
        output = _C.lm_head_fp32(hidden, weight, bias)
        ctx.save_for_backward(hidden, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        ctx.output_dtype = output_dtype
        return output.to(output_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias = ctx.saved_tensors
        grad = grad_output.float().reshape(-1, grad_output.size(-1))
        hidden_2d = hidden.float().reshape(-1, hidden.size(-1))
        grad_hidden = (grad @ weight.float()).reshape_as(hidden)
        grad_weight = grad.t() @ hidden_2d
        grad_bias = grad.sum(dim=0) if ctx.has_bias else None
        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), grad_bias, None


class MusaLMHeadOp:
    def __init__(self):
        _require("lm_head_fp32")

    def __call__(self, hidden, weight, *, bias=None):
        return self.forward(hidden, weight, bias=bias)

    def forward(self, hidden, weight, *, bias=None):
        _check_musa(hidden, weight)
        if bias is not None:
            _check_musa(bias)
        return _MusaLMHeadFunction.apply(hidden, weight, bias, hidden.dtype)

    def forward_fp32(self, hidden, weight, *, bias=None):
        _check_musa(hidden, weight)
        return _C.lm_head_fp32(hidden, weight, bias)


def _cos_sin(positions, half, theta, device):
    inv = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    freqs = positions.to(device=device, dtype=torch.float32).reshape(-1, 1) * inv
    return freqs.cos().contiguous(), freqs.sin().contiguous()


class _MusaRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positions, theta):
        if positions.dim() != 1:
            raise NotImplementedError("MUSA native RoPE currently supports positions [S]")
        half = x.size(-1) // 2
        cos, sin = _cos_sin(positions, half, theta, x.device)
        x_2d = x.contiguous().reshape(-1, x.size(-1))
        output = _C.rope_apply(x_2d, cos, sin, 1.0).reshape_as(x)
        ctx.save_for_backward(cos, sin)
        ctx.seq_len = positions.numel()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        grad = grad_output.contiguous().reshape(-1, grad_output.size(-1))
        return _C.rope_apply(grad, cos, sin, -1.0).reshape_as(grad_output), None, None


class MusaRoPEOp:
    def __init__(self):
        _require("rope_apply")

    def __call__(self, x, positions, *, theta=1_000_000.0):
        return self.forward(x, positions, theta=theta)

    def forward(self, x, positions, *, theta=1_000_000.0):
        _check_musa(x)
        return _MusaRoPEFunction.apply(x, positions, float(theta))

    def forward_fp32(self, x, positions, *, theta=1_000_000.0):
        _check_musa(x)
        return _MusaRoPEFunction.apply(x.float(), positions, float(theta))


class _MusaAttentionFunction(torch.autograd.Function):
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
        _require("deterministic_attention_fp32")
        _require("deterministic_attention_backward")

    def __call__(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        return self.forward(q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask)

    def forward(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        _check_musa(q, k, v)
        resolved_scale = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
        mask = (
            key_padding_mask.contiguous()
            if key_padding_mask is not None
            else torch.empty(0, device=q.device, dtype=torch.bool)
        )
        out, _lse = _MusaAttentionFunction.apply(q, k, v, causal, resolved_scale, mask)
        return out

    def forward_with_lse(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        _check_musa(q, k, v)
        resolved_scale = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
        mask = (
            key_padding_mask.contiguous()
            if key_padding_mask is not None
            else torch.empty(0, device=q.device, dtype=torch.bool)
        )
        return _MusaAttentionFunction.apply(q, k, v, causal, resolved_scale, mask)


class MusaDetGemmOp:
    def __init__(self):
        _require("det_gemm_rowwise_fwd_fp32")

    def __call__(self, a, b):
        _check_musa(a, b)
        return _C.det_gemm_rowwise_fwd_fp32(a, b)

    forward = __call__
    forward_fp32 = __call__
