// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <musa_runtime.h>
#include <torch/extension.h>
#include <ATen/Functions.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <type_traits>

namespace {

constexpr int kBlockSize = 256;
constexpr int kBf16TreeLeaf = 32;

__device__ __forceinline__ int midpoint_merge_count(int leaf, int leaf_count) {
    int lo = 0;
    int hi = leaf_count;
    int count = 0;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (leaf < mid) {
            hi = mid;
            count = 0;
        } else {
            lo = mid;
            ++count;
        }
    }
    return count;
}

template <bool TransposeA, bool TransposeB>
__device__ __forceinline__ c10::BFloat16 bf16_dot_tree(
    const c10::BFloat16* a,
    const c10::BFloat16* b,
    int row,
    int column,
    int m,
    int n,
    int k) {
    const int leaf_count = (k + kBf16TreeLeaf - 1) / kBf16TreeLeaf;
    c10::BFloat16 stack[16];
    int stack_size = 0;
    c10::BFloat16 value = static_cast<c10::BFloat16>(0.0f);
    for (int leaf = 0; leaf < leaf_count; ++leaf) {
        const int begin = leaf * kBf16TreeLeaf;
        const int end = min(begin + kBf16TreeLeaf, k);
        float accumulator = 0.0f;
        for (int inner = begin; inner < end; ++inner) {
            const int a_index = TransposeA ? inner * m + row : row * k + inner;
            const int b_index = TransposeB ? column * k + inner : inner * n + column;
            accumulator += static_cast<float>(a[a_index]) * static_cast<float>(b[b_index]);
        }
        value = static_cast<c10::BFloat16>(accumulator);
        const int merge_count = midpoint_merge_count(leaf, leaf_count);
        for (int merge = 0; merge < merge_count; ++merge) {
            const c10::BFloat16 lhs = stack[stack_size - 1];
            value = static_cast<c10::BFloat16>(
                static_cast<float>(lhs) + static_cast<float>(value));
            --stack_size;
        }
        if (leaf + 1 < leaf_count) {
            stack[stack_size++] = value;
        }
    }
    return value;
}

template <
    typename input_t,
    typename output_t,
    bool TransposeA,
    bool TransposeB,
    bool TransposeOutput>
__global__ void det_gemm_kernel(
    const input_t* a,
    const input_t* b,
    output_t* c,
    int m,
    int n,
    int k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * n) {
        return;
    }
    const int row = index / n;
    const int column = index % n;
    const int output_index = TransposeOutput ? column * m + row : index;
    if constexpr (
        std::is_same<input_t, c10::BFloat16>::value &&
        std::is_same<output_t, c10::BFloat16>::value) {
        // Match CUDA SM90 det_gemm: FP32 leaves of 32 K values are rounded to
        // BF16, then combined by the fixed midpoint tree. No split-K path is
        // allowed to change this order.
        c[output_index] = bf16_dot_tree<TransposeA, TransposeB>(
            a, b, row, column, m, n, k);
    } else {
        float accumulator = 0.0f;
        for (int inner = 0; inner < k; ++inner) {
            const int a_index = TransposeA ? inner * m + row : row * k + inner;
            const int b_index = TransposeB ? column * k + inner : inner * n + column;
            accumulator = fmaf(
                static_cast<float>(a[a_index]),
                static_cast<float>(b[b_index]),
                accumulator);
        }
        c[output_index] = static_cast<output_t>(accumulator);
    }
}

template <typename input_t, typename output_t>
void launch(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    int m,
    int n,
    int k,
    bool transpose_a,
    bool transpose_b,
    bool transpose_output) {
    const int blocks = (m * n + kBlockSize - 1) / kBlockSize;
    auto stream = at::musa::getCurrentMUSAStream();
    if (transpose_a) {
        if (transpose_output) {
            det_gemm_kernel<input_t, output_t, true, false, true>
                <<<blocks, kBlockSize, 0, stream>>>(
                    a.data_ptr<input_t>(),
                    b.data_ptr<input_t>(),
                    output.data_ptr<output_t>(),
                    m,
                    n,
                    k);
        } else {
            det_gemm_kernel<input_t, output_t, true, false, false>
                <<<blocks, kBlockSize, 0, stream>>>(
                    a.data_ptr<input_t>(),
                    b.data_ptr<input_t>(),
                    output.data_ptr<output_t>(),
                    m,
                    n,
                    k);
        }
    } else if (transpose_b) {
        det_gemm_kernel<input_t, output_t, false, true, false>
            <<<blocks, kBlockSize, 0, stream>>>(
                a.data_ptr<input_t>(),
                b.data_ptr<input_t>(),
                output.data_ptr<output_t>(),
                m,
                n,
                k);
    } else {
        det_gemm_kernel<input_t, output_t, false, false, false>
            <<<blocks, kBlockSize, 0, stream>>>(
                a.data_ptr<input_t>(),
                b.data_ptr<input_t>(),
                output.data_ptr<output_t>(),
                m,
                n,
                k);
    }
}

void check_inputs(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(
        a.device().type() == c10::kPrivateUse1 &&
            b.device().type() == c10::kPrivateUse1,
        "det_gemm requires MUSA tensors");
    TORCH_CHECK(a.device() == b.device(), "det_gemm tensors must share a device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm expects 2-D tensors");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "det_gemm dtypes must match");
  TORCH_CHECK(
      a.scalar_type() == at::ScalarType::BFloat16 ||
          a.scalar_type() == at::ScalarType::Float,
      "MUSA det_gemm currently supports bfloat16/float inputs");
}

template <typename input_t, typename output_t>
torch::Tensor dispatch(
    torch::Tensor a,
    torch::Tensor b,
    int m,
    int n,
    int k,
    bool transpose_a,
    bool transpose_b,
    bool transpose_output,
    bool output_fp32) {
    a = a.contiguous();
    b = b.contiguous();
    auto options = output_fp32 ? a.options().dtype(torch::kFloat) : a.options();
    auto output = torch::empty(
        transpose_output ? std::vector<int64_t>{n, m}
                         : std::vector<int64_t>{m, n},
        options);
    if (m == 0 || n == 0) {
        return output;
    }
    launch<input_t, output_t>(
        a, b, output, m, n, k, transpose_a, transpose_b, transpose_output);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor dispatch_any(
    torch::Tensor a,
    torch::Tensor b,
    int m,
    int n,
    int k,
    bool transpose_a,
    bool transpose_b,
    bool transpose_output,
    bool output_fp32) {
    if (a.scalar_type() == at::ScalarType::BFloat16) {
        if (output_fp32) {
            return dispatch<c10::BFloat16, float>(
                a, b, m, n, k, transpose_a, transpose_b, transpose_output, true);
        }
        return dispatch<c10::BFloat16, c10::BFloat16>(
            a, b, m, n, k, transpose_a, transpose_b, transpose_output, false);
    }
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Float, "unsupported det_gemm dtype");
    return dispatch<float, float>(
        a, b, m, n, k, transpose_a, transpose_b, transpose_output, output_fp32);
}

}  // namespace

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd: K mismatch");
    return dispatch_any(a, b, a.size(0), b.size(1), a.size(1), false, false, false, false);
}

torch::Tensor det_gemm_fwd_fp32(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd_fp32: K mismatch");
    return dispatch_any(
        a.to(torch::kFloat), b.to(torch::kFloat),
        a.size(0), b.size(1), a.size(1), false, false, false, true);
}

torch::Tensor det_gemm_fwd_rhs_transposed(torch::Tensor a, torch::Tensor bt) {
    check_inputs(a, bt);
    TORCH_CHECK(bt.size(1) == a.size(1), "det_gemm_fwd_rhs_transposed: K mismatch");
    return dispatch_any(a, bt, a.size(0), bt.size(0), a.size(1), false, true, false, false);
}

torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b) {
    check_inputs(dc, b);
    TORCH_CHECK(b.size(1) == dc.size(1), "det_gemm_da: N mismatch");
    return dispatch_any(dc, b, dc.size(0), b.size(0), dc.size(1), false, true, false, false);
}

torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc) {
    check_inputs(a, dc);
    TORCH_CHECK(a.size(0) == dc.size(0), "det_gemm_db: M mismatch");
    return dispatch_any(a, dc, a.size(1), dc.size(1), a.size(0), true, false, false, false);
}

torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc) {
    check_inputs(a, dc);
    TORCH_CHECK(a.size(0) == dc.size(0), "det_gemm_db_transposed: M mismatch");
    return dispatch_any(a, dc, a.size(1), dc.size(1), a.size(0), true, false, true, false);
}
