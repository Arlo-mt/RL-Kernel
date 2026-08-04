# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""SiLU / SwiGLU tests: native gold + CUDA / Triton candidates vs ground truth.

Covers:
- Native correctness (fp32 formula, dtype path, shape guard)
- Axis A batch invariance (slice + padding, forward + backward)
- CUDA / Triton forward+backward vs NativeSiLUOp / NativeSwiGLUOp (issue #108 harness)
- Registry dispatch + OP_SPECS candidate paths
"""

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.kernels.gtest.op_checks import run_operator_suite
from rl_engine.kernels.gtest.operator_specs import (
    make_candidate,
    make_operator_case,
    operator_names,
)
from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSiLUOp, NativeSwiGLUOp
from rl_engine.kernels.ops.triton.activation.swiglu import TritonSiLUOp, TritonSwiGLUOp
from rl_engine.kernels.registry import kernel_registry

try:
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    from rl_engine.kernels.ops.cuda.activation.swiglu import SiLUCudaOp, SwiGLUCudaOp

    _HAS_CUDA_ACTIVATION = (
        _EXT_AVAILABLE and hasattr(_C, "silu_forward") and hasattr(_C, "swiglu_forward")
    )
except ImportError:  # pragma: no cover - extension may be missing in CPU-only builds.
    _HAS_CUDA_ACTIVATION = False
    SiLUCudaOp = None  # type: ignore[misc, assignment]
    SwiGLUCudaOp = None  # type: ignore[misc, assignment]

# Qwen3-8B SwiGLU intermediate dim (gate/up_proj output width).
_INTERMEDIATE = 12288


# Shared helper
def _rand(shape, *, seed, dtype=torch.float32, device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=gen, dtype=torch.float32)
    return t.to(device=device, dtype=dtype)


def _dtype_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    # Matches elementwise row of tolerance_contract.json (issue #108).
    if dtype is torch.float32:
        return 1e-5, 1e-5
    if dtype is torch.float16:
        return 1e-3, 1e-3
    if dtype is torch.bfloat16:
        return 2e-2, 1.6e-2
    raise ValueError(f"unsupported dtype: {dtype}")


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_cuda_activation = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_CUDA_ACTIVATION),
    reason="CUDA SiLU/SwiGLU extension is not available",
)


# ---------------------------------------------------------------------------
# Native gold (PyTorch reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_native_silu_matches_fp32_reference(dtype: torch.dtype):
    x = torch.linspace(-6.0, 6.0, 33, dtype=dtype).reshape(3, 11)

    fp32_reference = x.float() * torch.sigmoid(x.float())
    result = NativeSiLUOp().forward(x)

    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    assert torch.equal(NativeSiLUOp().forward_fp32(x), fp32_reference)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_native_swiglu_matches_fp32_reference(dtype: torch.dtype):
    gate = torch.linspace(-4.0, 4.0, 48, dtype=dtype).reshape(2, 3, 8)
    up = torch.linspace(0.5, 2.0, 48, dtype=dtype).reshape(2, 3, 8)

    fp32_reference = gate.float() * torch.sigmoid(gate.float()) * up.float()
    result = NativeSwiGLUOp().forward(gate, up)

    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    assert torch.equal(NativeSwiGLUOp().forward_fp32(gate, up), fp32_reference)


def test_native_swiglu_rejects_mismatched_shape():
    gate = torch.randn(2, 3)
    up = torch.randn(2, 4)

    with pytest.raises(ValueError, match="share shape"):
        NativeSwiGLUOp().forward(gate, up)


# Axis A -- batch invariance, bitwise (the WS1 "aligned" property).
# A row's output must not depend on how many rows share the batch.
def test_silu_batch_invariance_slice():
    op = NativeSiLUOp()
    x = _rand((8, 32, _INTERMEDIATE), seed=2)
    full = op.forward_fp32(x)  # compute on full batch...
    assert torch.equal(op.forward_fp32(x[:1]), full[:1])  # ...then slice
    assert torch.equal(op.forward_fp32(x[3:5]), full[3:5])


def test_swiglu_batch_invariance_slice():
    op = NativeSwiGLUOp()
    gate = _rand((8, 32, _INTERMEDIATE), seed=3)
    up = _rand((8, 32, _INTERMEDIATE), seed=4)
    full = op.forward_fp32(gate, up)
    assert torch.equal(op.forward_fp32(gate[:1], up[:1]), full[:1])
    assert torch.equal(op.forward_fp32(gate[3:5], up[3:5]), full[3:5])


def test_silu_batch_invariance_with_padding():
    """Padding extra rows must not perturb the real rows (bitwise)."""
    op = NativeSiLUOp()
    x = _rand((4, _INTERMEDIATE), seed=5)
    padded = torch.cat([x, _rand((6, _INTERMEDIATE), seed=99)], dim=0)
    assert torch.equal(op.forward_fp32(padded)[:4], op.forward_fp32(x))


def test_swiglu_batch_invariance_with_padding():
    op = NativeSwiGLUOp()
    gate = _rand((4, _INTERMEDIATE), seed=6)
    up = _rand((4, _INTERMEDIATE), seed=7)
    pad_gate = torch.cat([gate, _rand((6, _INTERMEDIATE), seed=98)], dim=0)
    pad_up = torch.cat([up, _rand((6, _INTERMEDIATE), seed=97)], dim=0)
    assert torch.equal(op.forward_fp32(pad_gate, pad_up)[:4], op.forward_fp32(gate, up))


# Purity -- inputs not mutated in-place
def test_silu_inputs_not_mutated():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=8)
    xc = x.clone()
    op.forward(x)
    op.forward_fp32(x)
    assert torch.equal(x, xc)


def test_swiglu_inputs_not_mutated():
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=9)
    up = _rand((2, _INTERMEDIATE), seed=10)
    gc, uc = gate.clone(), up.clone()
    op.forward(gate, up)
    op.forward_fp32(gate, up)
    assert torch.equal(gate, gc) and torch.equal(up, uc)


# Gradient flows (fp32 autograd = backward golden source)
def test_silu_gradient_flows():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=11).requires_grad_(True)
    op.forward_fp32(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_swiglu_gradient_flows():
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=12).requires_grad_(True)
    up = _rand((2, _INTERMEDIATE), seed=13).requires_grad_(True)
    op.forward_fp32(gate, up).sum().backward()
    assert torch.isfinite(gate.grad).all() and torch.isfinite(up.grad).all()


def test_silu_backward_batch_invariance_slice():
    """Axis A: Gradients must be bitwise identical regardless of batch size."""
    op = NativeSiLUOp()

    x_full = _rand((8, 32, _INTERMEDIATE), seed=1).requires_grad_(True)
    out_full = op.forward_fp32(x_full)

    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)

    grad_full_sliced = x_full.grad[:1].clone()

    x_slice = _rand((8, 32, _INTERMEDIATE), seed=1)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(x_slice)
    out_slice.backward(dy_full[:1])

    assert torch.equal(x_slice.grad, grad_full_sliced)


def test_swiglu_backward_batch_invariance_slice():
    """Axis A: Gradients must be bitwise identical regardless of batch size."""
    op = NativeSwiGLUOp()

    gate_full = _rand((8, 32, _INTERMEDIATE), seed=1).requires_grad_(True)
    up_full = _rand((8, 32, _INTERMEDIATE), seed=2).requires_grad_(True)
    out_full = op.forward_fp32(gate_full, up_full)

    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)

    grad_gate_full_sliced = gate_full.grad[:1].clone()
    grad_up_full_sliced = up_full.grad[:1].clone()

    gate_slice = _rand((8, 32, _INTERMEDIATE), seed=1)[:1].detach().requires_grad_(True)
    up_slice = _rand((8, 32, _INTERMEDIATE), seed=2)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(gate_slice, up_slice)

    out_slice.backward(dy_full[:1])

    assert torch.equal(gate_slice.grad, grad_gate_full_sliced)
    assert torch.equal(up_slice.grad, grad_up_full_sliced)


def test_registry_dispatches_native_activation_ops_on_cpu():
    assert isinstance(kernel_registry.get_op("silu", device="cpu"), NativeSiLUOp)
    assert isinstance(kernel_registry.get_op("swiglu", device="cpu"), NativeSwiGLUOp)


# ---------------------------------------------------------------------------
# CUDA / Triton candidates vs native gold (RMSNorm-style)
# ---------------------------------------------------------------------------


def _silu_impls():
    impls = ["triton"]
    if _HAS_CUDA_ACTIVATION:
        impls.append("cuda")
    return impls


def _make_silu_op(impl: str):
    if impl == "cuda":
        return SiLUCudaOp()
    if impl == "triton":
        return TritonSiLUOp()
    raise ValueError(impl)


def _make_swiglu_op(impl: str):
    if impl == "cuda":
        return SwiGLUCudaOp()
    if impl == "triton":
        return TritonSwiGLUOp()
    raise ValueError(impl)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64),
        (8, 256),
        (4, 32, 512),
        (2, 8, _INTERMEDIATE),  # Qwen3-8B intermediate width
    ],
)
def test_cuda_triton_silu_matches_native_forward_and_backward(impl, dtype, shape):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SiLU extension is not available")

    native = NativeSiLUOp()
    cand = _make_silu_op(impl)

    x_cpu = _rand(shape, seed=0, dtype=torch.float32)
    dy_cpu = _rand(shape, seed=1, dtype=torch.float32)

    x_ref = x_cpu.to(dtype).float().detach().requires_grad_(True)
    dy_ref = dy_cpu.to(dtype).float()
    y_ref = native.forward_fp32(x_ref)
    y_ref.backward(dy_ref)

    x_gpu = x_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)
    y_gpu = cand.forward(x_gpu)
    y_gpu.backward(dy_gpu)

    atol, rtol = _dtype_tolerance(dtype)
    torch.testing.assert_close(y_gpu.detach().cpu().float(), y_ref.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(x_gpu.grad.detach().cpu().float(), x_ref.grad, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64),
        (8, 256),
        (4, 32, 512),
        (2, 8, _INTERMEDIATE),
    ],
)
def test_cuda_triton_swiglu_matches_native_forward_and_backward(impl, dtype, shape):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    native = NativeSwiGLUOp()
    cand = _make_swiglu_op(impl)

    gate_cpu = _rand(shape, seed=2, dtype=torch.float32)
    up_cpu = _rand(shape, seed=3, dtype=torch.float32)
    dy_cpu = _rand(shape, seed=4, dtype=torch.float32)

    gate_ref = gate_cpu.to(dtype).float().detach().requires_grad_(True)
    up_ref = up_cpu.to(dtype).float().detach().requires_grad_(True)
    dy_ref = dy_cpu.to(dtype).float()
    y_ref = native.forward_fp32(gate_ref, up_ref)
    y_ref.backward(dy_ref)

    gate_gpu = gate_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    up_gpu = up_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)
    y_gpu = cand.forward(gate_gpu, up_gpu)
    y_gpu.backward(dy_gpu)

    atol, rtol = _dtype_tolerance(dtype)
    torch.testing.assert_close(y_gpu.detach().cpu().float(), y_ref.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        gate_gpu.grad.detach().cpu().float(), gate_ref.grad, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        up_gpu.grad.detach().cpu().float(), up_ref.grad, atol=atol, rtol=rtol
    )


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_silu_batch_invariance_bitwise(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SiLU extension is not available")

    op = _make_silu_op(impl)
    x = _rand((8, 32, 256), seed=5, dtype=torch.bfloat16, device="cuda")
    full = op.forward(x)
    assert torch.equal(op.forward(x[:1]), full[:1])
    assert torch.equal(op.forward(x[3:5]), full[3:5])


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_batch_invariance_bitwise(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    op = _make_swiglu_op(impl)
    gate = _rand((8, 32, 256), seed=6, dtype=torch.bfloat16, device="cuda")
    up = _rand((8, 32, 256), seed=7, dtype=torch.bfloat16, device="cuda")
    full = op.forward(gate, up)
    assert torch.equal(op.forward(gate[:1], up[:1]), full[:1])
    assert torch.equal(op.forward(gate[3:5], up[3:5]), full[3:5])


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_silu_deterministic_repeat(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SiLU extension is not available")

    op = _make_silu_op(impl)
    x = _rand((32, 1024), seed=8, dtype=torch.bfloat16, device="cuda")
    dy = _rand((32, 1024), seed=9, dtype=torch.bfloat16, device="cuda")

    def _run():
        x_r = x.detach().clone().requires_grad_(True)
        y = op.forward(x_r)
        y.backward(dy)
        return y.detach(), x_r.grad.detach()

    y0, dx0 = _run()
    torch.cuda.synchronize()
    for _ in range(5):
        y, dx = _run()
        torch.cuda.synchronize()
        assert torch.equal(y0, y)
        assert torch.equal(dx0, dx)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_rejects_mismatched_shape(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    op = _make_swiglu_op(impl)
    gate = torch.randn(2, 3, device="cuda")
    up = torch.randn(2, 4, device="cuda")
    with pytest.raises(ValueError, match="share shape"):
        op.forward(gate, up)


# ---------------------------------------------------------------------------
# Issue #108 ground-truth harness (OP_SPECS + check_operator path)
# ---------------------------------------------------------------------------


def _spec_args(op: str, **overrides) -> argparse.Namespace:
    values = dict(
        op=op,
        candidate="pytorch",
        arch_key=None,
        batch=2,
        seq=4,
        vocab=17,
        seed=123,
        input_mode="random",
        constant_value=0.5,
        token_value=3,
        normalized_dim=8,
        k_dim=8,
        n_dim=8,
        theta=1.0e6,
        eps=1.0e-6,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_silu_swiglu_registered_in_op_specs():
    assert "silu" in operator_names()
    assert "swiglu" in operator_names()


def test_silu_pytorch_candidate_suite_passes_issue_108_helper():
    args = _spec_args("silu", candidate="pytorch")
    report = run_operator_suite(
        "silu",
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, torch.float32, torch.device("cpu"))],
        check_grad=True,
    )
    assert report.passed


def test_swiglu_pytorch_candidate_suite_passes_issue_108_helper():
    args = _spec_args("swiglu", candidate="pytorch")
    report = run_operator_suite(
        "swiglu",
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, torch.float32, torch.device("cpu"))],
        check_grad=True,
    )
    assert report.passed


@requires_cuda
@pytest.mark.parametrize("candidate", ["triton", "cuda"])
@pytest.mark.parametrize("op_name", ["silu", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_silu_swiglu_cuda_triton_issue_108_harness(candidate, op_name, dtype):
    if candidate == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA activation extension is not available")

    args = _spec_args(op_name, candidate=candidate, batch=2, seq=8)
    device = torch.device("cuda")
    report = run_operator_suite(
        op_name,
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, dtype, device)],
        check_grad=True,
    )
    assert report.passed, (
        f"{op_name}/{candidate}/{dtype} failed against gold: "
        f"{report.candidates[0].cases[0].outputs}"
    )
