# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native MUSA candidates for every node in the CUDA WS1 profile."""

from __future__ import annotations

import math

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.musa.loss.linear_logp import MusaLinearLogpOp
from rl_engine.kernels.ops.musa.matmul.det_gemm import MusaDetGemmOp


def _require(*symbols: str) -> None:
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError("MUSA WS1 candidates require the compiled rl_engine._C extension")
    missing = [symbol for symbol in symbols if not hasattr(_C, symbol)]
    if missing:
        raise RuntimeError(f"MUSA WS1 extension is missing: {', '.join(missing)}")


class _MusaEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, weight, output_fp32):
        output = _C.musa_embedding_forward(token_ids.contiguous(), weight.contiguous())[0]
        ctx.save_for_backward(token_ids.contiguous())
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        return output.float() if output_fp32 else output

    @staticmethod
    def backward(ctx, grad_output):
        (token_ids,) = ctx.saved_tensors
        ids = token_ids.reshape(-1).to(dtype=torch.long).contiguous()
        rows = grad_output.reshape(-1, ctx.weight_shape[1]).contiguous()
        order = torch.argsort(ids, stable=True)
        grad_weight = _C.musa_embedding_backward(
            ids.index_select(0, order),
            rows.index_select(0, order),
            int(ctx.weight_shape[0]),
            int(ctx.weight_shape[1]),
        ).to(ctx.weight_dtype)
        record_backward(
            "embedding",
            kernel_id="rl_engine._C.musa_embedding_backward",
            impl="musa_native_embedding",
            family="musa",
        )
        return None, grad_weight, None


class MusaEmbeddingOp:
    op_class = "elementwise"
    is_batch_invariant = True
    backward_impl = "musa_native_embedding"

    def __init__(self):
        _require("musa_embedding_forward", "musa_embedding_backward")

    def forward(self, token_ids, weight):
        return _MusaEmbeddingFunction.apply(token_ids, weight, False)

    def forward_fp32(self, token_ids, weight):
        return _MusaEmbeddingFunction.apply(token_ids, weight, True)


class _MusaRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, output_fp32):
        output, rstd = _C.musa_rmsnorm_forward(
            x.contiguous(), weight.contiguous(), float(eps), bool(output_fp32)
        )
        ctx.save_for_backward(x.contiguous(), weight.contiguous(), rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        grad_x, grad_weight = _C.musa_rmsnorm_backward(
            grad_output.to(x.dtype).contiguous(), x, weight, rstd
        )
        record_backward(
            "rms_norm",
            kernel_id="rl_engine._C.musa_rmsnorm_backward",
            impl="musa_native_rmsnorm",
            family="musa",
        )
        return grad_x, grad_weight, None, None


class MusaRMSNormOp:
    op_class = "reduction"
    backward_impl = "musa_native_rmsnorm"

    def __init__(self):
        _require("musa_rmsnorm_forward", "musa_rmsnorm_backward")

    def __call__(self, x, weight, *, eps=1e-6):
        return self.forward(x, weight, eps=eps)

    def forward(self, x, weight, *, eps=1e-6):
        return _MusaRMSNormFunction.apply(x, weight, eps, False)

    def forward_fp32(self, x, weight, *, eps=1e-6):
        return _MusaRMSNormFunction.apply(x, weight, eps, True)


def _rope_table(x, positions, theta):
    dim = x.shape[-1]
    half = dim // 2
    if positions.dim() == 1:
        x_2d = x.contiguous().reshape(-1, dim)
        table_positions = positions
    elif positions.dim() == 2 and x.dim() == 4:
        x_2d = x.permute(1, 0, 2, 3).contiguous().reshape(-1, dim)
        table_positions = positions.reshape(-1)
    elif positions.dim() == 2 and x.dim() == 3:
        x_2d = x.contiguous().reshape(-1, dim)
        table_positions = positions.reshape(-1)
    else:
        raise ValueError("RoPE expects positions [S]/[B,S] and x [...,S,D]")
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, half, dtype=torch.float32, device=x.device)
            / half
        )
    )
    frequencies = table_positions.to(torch.float32).reshape(-1, 1) * inv_freq
    return (
        x_2d,
        frequencies.cos().contiguous(),
        frequencies.sin().contiguous(),
    )


def _restore_rope(output, shape, positions_dim):
    if positions_dim == 1 or len(shape) != 4:
        return output.reshape(shape)
    batch, heads, sequence, dim = shape
    return output.reshape(heads, batch, sequence, dim).permute(1, 0, 2, 3).contiguous()


class _MusaRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positions, theta):
        x_2d, cos, sin = _rope_table(x, positions, float(theta))
        ctx.save_for_backward(cos, sin)
        ctx.shape = tuple(x.shape)
        ctx.positions_dim = positions.dim()
        output = _C.musa_rope_apply(x_2d, cos, sin, 1.0)
        return _restore_rope(output, ctx.shape, ctx.positions_dim)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        if ctx.positions_dim == 2 and len(ctx.shape) == 4:
            grad_2d = (
                grad_output.permute(1, 0, 2, 3)
                .contiguous()
                .reshape(-1, ctx.shape[-1])
            )
        else:
            grad_2d = grad_output.contiguous().reshape(-1, ctx.shape[-1])
        grad_x = _restore_rope(
            _C.musa_rope_apply(grad_2d, cos, sin, -1.0),
            ctx.shape,
            ctx.positions_dim,
        )
        record_backward(
            "rope",
            kernel_id="rl_engine._C.musa_rope_apply",
            impl="musa_native_rope",
            family="musa",
        )
        return grad_x, None, None


class MusaRoPEOp:
    op_class = "elementwise"
    backward_impl = "musa_native_rope"

    def __init__(self):
        _require("musa_rope_apply")

    def forward(self, x, positions, *, theta=1_000_000.0):
        return _MusaRoPEFunction.apply(x, positions, theta)

    def forward_fp32(self, x, positions, *, theta=1_000_000.0):
        return _MusaRoPEFunction.apply(x.float(), positions, theta)


class _MusaSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = _C.musa_silu_forward(x.contiguous())
        ctx.save_for_backward(x.contiguous())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = _C.musa_silu_backward(grad_output.contiguous(), x)
        record_backward(
            "silu",
            kernel_id="rl_engine._C.musa_silu_backward",
            impl="musa_native_silu",
            family="musa",
        )
        return grad


class MusaSiLUOp:
    op_class = "elementwise"
    backward_impl = "musa_native_silu"

    def __init__(self):
        _require("musa_silu_forward", "musa_silu_backward")

    def forward(self, x):
        return _MusaSiLUFunction.apply(x)

    def forward_fp32(self, x):
        return _MusaSiLUFunction.apply(x.float())


class _MusaSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        output = _C.musa_swiglu_forward(gate.contiguous(), up.contiguous())
        ctx.save_for_backward(gate.contiguous(), up.contiguous())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        grads = tuple(_C.musa_swiglu_backward(grad_output.contiguous(), gate, up))
        record_backward(
            "swiglu",
            kernel_id="rl_engine._C.musa_swiglu_backward",
            impl="musa_native_swiglu",
            family="musa",
        )
        return grads


class MusaSwiGLUOp:
    op_class = "elementwise"
    backward_impl = "musa_native_swiglu"

    def __init__(self):
        _require("musa_swiglu_forward", "musa_swiglu_backward")

    def forward(self, gate, up):
        return _MusaSwiGLUFunction.apply(gate, up)

    def forward_fp32(self, gate, up):
        return _MusaSwiGLUFunction.apply(gate.float(), up.float())


class _MusaLogpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index):
        output = _C.musa_selected_logp_forward(
            logits.contiguous(), target.contiguous(), int(ignore_index)
        )
        ctx.save_for_backward(logits.contiguous(), target.contiguous())
        ctx.ignore_index = int(ignore_index)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, target = ctx.saved_tensors
        grad = _C.musa_selected_logp_backward(
            grad_output.to(logits.dtype).contiguous(),
            logits,
            target,
            ctx.ignore_index,
        )
        record_backward(
            "logp",
            kernel_id="rl_engine._C.musa_selected_logp_backward",
            impl="musa_native_logp",
            family="musa",
        )
        return grad, None, None


class MusaLogpOp:
    op_class = "logprob"
    backward_impl = "musa_native_logp"

    def __init__(self):
        _require("musa_selected_logp_forward", "musa_selected_logp_backward")

    def __call__(self, logits, token_ids):
        return self.forward(logits, token_ids)

    def forward(self, logits, token_ids):
        return _MusaLogpFunction.apply(logits, token_ids, -100)

    def forward_fp32(self, logits, token_ids):
        return _MusaLogpFunction.apply(logits.float(), token_ids, -100)

    apply = forward
    apply_fp32 = forward_fp32


def _musa_batch_invariant_forward(logits, target_ids, ignore_index):
    lead_shape = logits.shape[:-1]
    logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
    target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
    logp, lse = _C.musa_batch_invariant_logp_forward(
        logits_2d, target_1d, int(ignore_index)
    )
    return logp.reshape(lead_shape), lse.reshape(lead_shape)


class _MusaBatchInvariantLogpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target_ids, ignore_index):
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        target_1d = target_ids.reshape(-1).to(
            device=logits.device, dtype=torch.long
        ).contiguous()
        logp, lse = _C.musa_batch_invariant_logp_forward(
            logits_2d, target_1d, int(ignore_index)
        )
        ctx.save_for_backward(logits_2d, target_1d, lse)
        ctx.input_shape = tuple(logits.shape)
        ctx.ignore_index = int(ignore_index)
        return logp.reshape(logits.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        logits, target_ids, lse = ctx.saved_tensors
        grad = _C.musa_batch_invariant_logp_backward(
            grad_output.reshape(-1).contiguous(),
            logits,
            target_ids,
            lse,
            ctx.ignore_index,
        )
        record_backward(
            "batch_invariant_logp",
            kernel_id="rl_engine._C.musa_batch_invariant_logp_backward",
            impl="musa_native_batch_invariant_logp",
            family="musa",
        )
        return grad.reshape(ctx.input_shape), None, None

class MusaBatchInvariantLogpOp:
    op_class = "logprob"
    is_batch_invariant = True
    backward_impl = "musa_native_batch_invariant_logp"

    def __init__(self):
        _require(
            "musa_batch_invariant_logp_forward",
            "musa_batch_invariant_logp_backward",
        )

    def __call__(self, logits, target_ids, ignore_index=-100, *, validate=True):
        return self.apply(logits, target_ids, ignore_index=ignore_index, validate=validate)

    def apply(self, logits, target_ids, ignore_index=-100, *, validate=True):
        self._validate_shapes(logits, target_ids)
        if validate:
            valid = target_ids[target_ids != ignore_index]
            if valid.numel() and bool(
                ((valid < 0) | (valid >= logits.size(-1))).any().item()
            ):
                raise ValueError("target_ids contains an out-of-range value")
        return _MusaBatchInvariantLogpFunction.apply(logits, target_ids, ignore_index)

    def forward_with_lse(self, logits, target_ids, ignore_index=-100, *, validate=True):
        self._validate_shapes(logits, target_ids)
        if validate:
            valid = target_ids[target_ids != ignore_index]
            if valid.numel() and bool(
                ((valid < 0) | (valid >= logits.size(-1))).any().item()
            ):
                raise ValueError("target_ids contains an out-of-range value")
        return _musa_batch_invariant_forward(logits, target_ids, ignore_index)

    @staticmethod
    def _validate_shapes(logits, target_ids):
        if logits.dim() < 2 or logits.shape[:-1] != target_ids.shape:
            raise ValueError("logits leading shape must match target_ids")


class _MusaAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale, mask, output_fp32):
        mask_arg = mask if mask.numel() else None
        output, _lse, probs = _C.musa_attention_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            bool(causal),
            float(scale),
            mask_arg,
            bool(output_fp32),
        )
        ctx.save_for_backward(q.contiguous(), k.contiguous(), v.contiguous(), probs)
        ctx.scale = float(scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, probs = ctx.saved_tensors
        grad_q, grad_k, grad_v = _C.musa_attention_backward(
            grad_output.to(q.dtype).contiguous(), q, k, v, probs, ctx.scale
        )
        record_backward(
            "attention",
            kernel_id="rl_engine._C.musa_attention_backward",
            impl="musa_native_attention",
            family="musa",
        )
        return grad_q, grad_k, grad_v, None, None, None, None


class MusaAttentionOp:
    op_class = "attention"
    backward_impl = "musa_native_attention"

    def __init__(self):
        _require("musa_attention_forward", "musa_attention_backward")

    @staticmethod
    def _call(q, k, v, causal, scale, key_padding_mask, output_fp32):
        mask = (
            key_padding_mask.contiguous()
            if key_padding_mask is not None
            else torch.empty(0, device=q.device, dtype=torch.bool)
        )
        return _MusaAttentionFunction.apply(q, k, v, causal, scale, mask, output_fp32)

    def forward(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        resolved = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        return self._call(q, k, v, causal, resolved, key_padding_mask, False)

    def forward_fp32(self, q, k, v, *, causal=True, scale=None, key_padding_mask=None):
        resolved = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        return self._call(q, k, v, causal, resolved, key_padding_mask, True)


class _MusaLMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, output_fp32):
        output = _C.musa_lm_head_fp32(hidden.contiguous(), weight.contiguous())
        ctx.save_for_backward(hidden.contiguous(), weight.contiguous())
        return output if output_fp32 else output.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight = ctx.saved_tensors
        grad = grad_output.reshape(-1, weight.size(0)).to(torch.bfloat16).contiguous()
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).to(torch.bfloat16).contiguous()
        grad_hidden = (
            _C.det_gemm_fwd(grad, weight.to(torch.bfloat16).contiguous())
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight = (
            _C.det_gemm_fwd(grad.t().contiguous(), hidden_2d)
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "lm_head",
            kernel_id="rl_engine._C.det_gemm_fwd+rl_engine._C.det_gemm_fwd",
            impl="musa_native_lm_head",
            family="musa",
        )
        if grad_hidden is not None:
            grad_hidden = grad_hidden.reshape_as(hidden)
        return grad_hidden, grad_weight, None


class MusaLMHeadOp:
    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "musa_det_gemm"

    def __init__(self):
        self._gemm = MusaDetGemmOp()

    def _project(self, hidden, weight, bias, output_fp32):
        flat = hidden.reshape(-1, hidden.shape[-1])
        if not hasattr(_C, "musa_lm_head_fp32"):
            raise RuntimeError("MUSA LM head requires musa_lm_head_fp32")
        out = _MusaLMHeadFunction.apply(hidden, weight, bool(output_fp32))
        out = out.reshape(*hidden.shape[:-1], weight.shape[0])
        if bias is not None:
            out = out + bias.reshape(*((1,) * (out.ndim - 1)), -1)
        return out

    def __call__(self, hidden, weight, *, bias=None):
        return self.forward(hidden, weight, bias=bias)

    def forward(self, hidden, weight, *, bias=None):
        return self._project(hidden, weight, bias, False)

    def forward_fp32(self, hidden, weight, *, bias=None):
        return self._project(hidden, weight, bias, True)


__all__ = [
    "MusaAttentionOp",
    "MusaBatchInvariantLogpOp",
    "MusaEmbeddingOp",
    "MusaLMHeadOp",
    "MusaLinearLogpOp",
    "MusaLogpOp",
    "MusaRMSNormOp",
    "MusaRoPEOp",
    "MusaSiLUOp",
    "MusaSwiGLUOp",
]
