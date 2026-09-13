# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


class _MusaDetGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, output_fp32: bool):
        ctx.save_for_backward(a, b)
        ctx.output_fp32 = bool(output_fp32)
        if ctx.output_fp32:
            return _C.musa_rowwise_gemm_fp32(a.float(), b.float())
        return _C.det_gemm_fwd(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        if ctx.output_fp32:
            grad32 = grad_output.contiguous().float()
            a32 = a.contiguous().float()
            b32 = b.contiguous().float()
            grad_a = (
                _C.musa_rowwise_gemm_fp32(grad32, b32.t().contiguous()).to(a.dtype)
                if ctx.needs_input_grad[0]
                else None
            )
            grad_b = (
                _C.musa_rowwise_gemm_fp32(a32.t().contiguous(), grad32).to(b.dtype)
                if ctx.needs_input_grad[1]
                else None
            )
            record_backward(
                "det_gemm",
                kernel_id="rl_engine._C.musa_rowwise_gemm_fp32",
                impl="musa_rowwise_fp32_accum_det_gemm",
                family="musa",
            )
            return grad_a, grad_b, None
        grad = grad_output.to(a.dtype).contiguous()
        grad_a = _C.det_gemm_da(grad, b) if ctx.needs_input_grad[0] else None
        grad_b = _C.det_gemm_db(a, grad) if ctx.needs_input_grad[1] else None
        record_backward(
            "det_gemm",
            kernel_id="rl_engine._C.det_gemm_da+rl_engine._C.det_gemm_db",
            impl="musa_det_gemm",
            family="musa",
        )
        return grad_a, grad_b, None


class _MusaDetLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, weight: torch.Tensor):
        ctx.save_for_backward(a, weight)
        return _C.det_gemm_fwd_rhs_transposed(a, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, weight = ctx.saved_tensors
        grad = grad_output.to(a.dtype).contiguous()
        grad_a = _C.det_gemm_fwd(grad, weight) if ctx.needs_input_grad[0] else None
        grad_weight = (
            _C.det_gemm_db_transposed(a, grad) if ctx.needs_input_grad[1] else None
        )
        record_backward(
            "lm_head",
            kernel_id="rl_engine._C.det_gemm_fwd+rl_engine._C.det_gemm_db_transposed",
            impl="musa_det_gemm_lm_head",
            family="musa",
        )
        return grad_a, grad_weight


class MusaDetGemmOp:
    """Deterministic fixed-order GEMM for MUSA BF16 tensors."""

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or _C is None:
            raise RuntimeError("MUSA det_gemm requires the compiled extension")
        required = (
            "det_gemm_fwd",
            "det_gemm_fwd_fp32",
            "musa_rowwise_gemm_fp32",
            "det_gemm_fwd_rhs_transposed",
            "det_gemm_da",
            "det_gemm_db",
            "det_gemm_db_transposed",
        )
        missing = [name for name in required if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(f"MUSA det_gemm extension is missing: {', '.join(missing)}")

    @staticmethod
    def _check(a: torch.Tensor, b: torch.Tensor) -> None:
        if a.device.type != "musa" or b.device.type != "musa":
            raise ValueError("MUSA det_gemm requires MUSA tensors")
        if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("MUSA det_gemm currently supports BF16/FP32 tensors")

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check(a, b)
        if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
            raise ValueError("det_gemm expects A[M,K] and B[K,N]")
        return _MusaDetGemmFunction.apply(a.contiguous(), b.contiguous(), False)

    def forward_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check(a, b)
        if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
            raise ValueError("det_gemm expects A[M,K] and B[K,N]")
        return _MusaDetGemmFunction.apply(a.contiguous(), b.contiguous(), True)

    forward_accum_fp32 = forward_fp32

    def linear(self, a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        self._check(a, weight)
        if a.ndim != 2 or weight.ndim != 2 or a.size(1) != weight.size(1):
            raise ValueError("linear expects A[M,K] and weight[N,K]")
        return _MusaDetLinearFunction.apply(a.contiguous(), weight.contiguous())


def deterministic_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return MusaDetGemmOp()(a, b)
