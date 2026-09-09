// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <musa_runtime.h>
#include <torch/extension.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <cfloat>

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b);
torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc);

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float block_reduce_max(float value) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, 32));
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? partial[lane] : -FLT_MAX;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, 32));
        }
    }
    if (threadIdx.x == 0) {
        partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset, 32);
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? partial[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset, 32);
        }
    }
    if (threadIdx.x == 0) {
        partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

template <typename scalar_t>
__device__ __forceinline__ float dot_row(
    const scalar_t* hidden_row,
    const scalar_t* weight_row,
    int hidden_dim) {
    float value = 0.0f;
    for (int index = 0; index < hidden_dim; ++index) {
        value += static_cast<float>(hidden_row[index]) *
            static_cast<float>(weight_row[index]);
    }
    return value;
}

template <typename scalar_t>
__global__ void linear_logp_forward_kernel(
    const scalar_t* hidden,
    const scalar_t* weight,
    const scalar_t* bias,
    const int64_t* target,
    scalar_t* logp,
    float* lse,
    int rows,
    int vocab,
    int hidden_dim,
    bool has_bias) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const scalar_t* hidden_row = hidden + static_cast<size_t>(row) * hidden_dim;
    float row_max = -FLT_MAX;
    for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
        float value = dot_row(hidden_row, weight + static_cast<size_t>(column) * hidden_dim, hidden_dim);
        if (has_bias) {
            value += static_cast<float>(bias[column]);
        }
        row_max = fmaxf(row_max, value);
    }
    row_max = block_reduce_max(row_max);

    float row_sum = 0.0f;
    for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
        float value = dot_row(hidden_row, weight + static_cast<size_t>(column) * hidden_dim, hidden_dim);
        if (has_bias) {
            value += static_cast<float>(bias[column]);
        }
        row_sum += expf(value - row_max);
    }
    row_sum = block_reduce_sum(row_sum);

    __shared__ float selected_logit;
    if (threadIdx.x == 0) {
        const int64_t target_id = target[row];
        selected_logit = dot_row(
            hidden_row,
            weight + static_cast<size_t>(target_id) * hidden_dim,
            hidden_dim);
        if (has_bias) {
            selected_logit += static_cast<float>(bias[target_id]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float row_lse = row_max + logf(row_sum);
        logp[row] = static_cast<scalar_t>(selected_logit - row_lse);
        lse[row] = row_lse;
    }
}

template <typename scalar_t>
__global__ void linear_logp_dlogits_kernel(
    const scalar_t* hidden,
    const scalar_t* weight,
    const scalar_t* bias,
    const int64_t* target,
    const scalar_t* grad_logp,
    float* dlogits,
    int rows,
    int vocab,
    int hidden_dim,
    bool has_bias) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const scalar_t* hidden_row = hidden + static_cast<size_t>(row) * hidden_dim;
    float row_max = -FLT_MAX;
    for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
        float value = dot_row(hidden_row, weight + static_cast<size_t>(column) * hidden_dim, hidden_dim);
        if (has_bias) {
            value += static_cast<float>(bias[column]);
        }
        row_max = fmaxf(row_max, value);
    }
    row_max = block_reduce_max(row_max);
    float row_sum = 0.0f;
    for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
        float value = dot_row(hidden_row, weight + static_cast<size_t>(column) * hidden_dim, hidden_dim);
        if (has_bias) {
            value += static_cast<float>(bias[column]);
        }
        row_sum += expf(value - row_max);
    }
    row_sum = block_reduce_sum(row_sum);

    const float upstream = static_cast<float>(grad_logp[row]);
    const int64_t target_id = target[row];
    for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
        float value = dot_row(hidden_row, weight + static_cast<size_t>(column) * hidden_dim, hidden_dim);
        if (has_bias) {
            value += static_cast<float>(bias[column]);
        }
        const float probability = expf(value - row_max) / row_sum;
        const float one_hot = column == target_id ? 1.0f : 0.0f;
        dlogits[static_cast<size_t>(row) * vocab + column] =
            upstream * (one_hot - probability);
    }
}

}  // namespace

std::vector<torch::Tensor> fused_linear_logp_musa_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias) {
    const int rows = hidden.size(0);
    const int vocab = weight.size(0);
    const int hidden_dim = hidden.size(1);
    const bool has_bias = bias.has_value();
    auto logp = torch::empty({rows}, hidden.options());
    auto lse = torch::empty({rows}, hidden.options().dtype(torch::kFloat));
    if (rows == 0) {
        return {logp, lse};
    }
    const auto bias_tensor = has_bias ? *bias : hidden;
    auto stream = at::musa::getCurrentMUSAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        hidden.scalar_type(),
        "musa_fused_linear_logp_forward",
        [&] {
            linear_logp_forward_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
                hidden.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias_tensor.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(),
                logp.data_ptr<scalar_t>(),
                lse.data_ptr<float>(),
                rows,
                vocab,
                hidden_dim,
                has_bias);
        });
    C10_MUSA_KERNEL_LAUNCH_CHECK();
    return {logp, lse};
}

std::vector<torch::Tensor> fused_linear_logp_musa_backward(
    torch::Tensor grad_logp,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::Tensor lse,
    torch::optional<torch::Tensor> bias,
    bool compute_grad_hidden,
    bool compute_grad_weight,
    bool compute_grad_bias) {
    const int rows = hidden.size(0);
    const int vocab = weight.size(0);
    const int hidden_dim = hidden.size(1);
    const bool has_bias = bias.has_value();
    auto dlogits = torch::empty({rows, vocab}, hidden.options().dtype(torch::kFloat));
    auto bias_tensor = has_bias ? *bias : hidden;
    if (rows > 0) {
        auto stream = at::musa::getCurrentMUSAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            hidden.scalar_type(),
            "musa_fused_linear_logp_dlogits",
            [&] {
                linear_logp_dlogits_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
                    hidden.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias_tensor.data_ptr<scalar_t>(),
                    target.data_ptr<int64_t>(),
                    grad_logp.data_ptr<scalar_t>(),
                    dlogits.data_ptr<float>(),
                    rows,
                    vocab,
                    hidden_dim,
                    has_bias);
            });
        C10_MUSA_KERNEL_LAUNCH_CHECK();
    }
    auto dlogits_low = dlogits.to(hidden.scalar_type());
    auto grad_hidden = compute_grad_hidden ? det_gemm_fwd(dlogits_low, weight) : torch::Tensor();
    auto grad_weight =
        compute_grad_weight ? det_gemm_db_transposed(hidden, dlogits_low) : torch::Tensor();
    auto grad_bias = compute_grad_bias && has_bias ? dlogits.sum(0).to(bias->scalar_type())
                                                   : torch::Tensor();
    return {grad_hidden, grad_weight, grad_bias};
}
