// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

#include <ATen/Functions.h>
#include <musa_runtime.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

namespace {

constexpr int64_t kVocabTile = 32768;

void check_inputs(
    const torch::Tensor& hidden,
    const torch::Tensor& weight,
    const torch::Tensor& target,
    const torch::optional<torch::Tensor>& bias) {
    TORCH_CHECK(
        hidden.device().type() == c10::kPrivateUse1 &&
            weight.device().type() == c10::kPrivateUse1 &&
            target.device().type() == c10::kPrivateUse1,
        "linear_logp requires MUSA tensors");
    TORCH_CHECK(
        hidden.device() == weight.device() && hidden.device() == target.device(),
        "linear_logp tensors must share a device");
    TORCH_CHECK(
        hidden.dim() == 2 && weight.dim() == 2 && target.dim() == 1,
        "expected hidden [N, H], weight [V, H], and target [N]");
    TORCH_CHECK(
        hidden.scalar_type() == at::ScalarType::BFloat16 &&
            weight.scalar_type() == at::ScalarType::BFloat16,
        "MUSA linear_logp currently supports BF16 hidden and weight");
    TORCH_CHECK(target.scalar_type() == at::ScalarType::Long, "target must be int64");
    TORCH_CHECK(
        hidden.size(1) == weight.size(1) && target.size(0) == hidden.size(0),
        "linear_logp shape mismatch");
    if (bias.has_value()) {
        TORCH_CHECK(
            bias->device() == hidden.device() &&
                bias->scalar_type() == weight.scalar_type() &&
                bias->dim() == 1 && bias->size(0) == weight.size(0),
            "bias must be a MUSA BF16 tensor with one value per vocabulary row");
    }
    if (target.numel() > 0) {
        TORCH_CHECK(
            target.min().item<int64_t>() >= 0 &&
                target.max().item<int64_t>() < weight.size(0),
            "target ids must be within the vocabulary range");
    }
}

torch::Tensor project_tile(
    const torch::Tensor& hidden,
    const torch::Tensor& weight,
    const torch::optional<torch::Tensor>& bias,
    int64_t vocab_start,
    int64_t vocab_count) {
    auto weight_tile = weight.narrow(0, vocab_start, vocab_count).t();
    auto logits = at::mm(hidden, weight_tile);
    if (bias.has_value()) {
        logits.add_(bias->narrow(0, vocab_start, vocab_count));
    }
    return logits;
}

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
    value = threadIdx.x < 8 ? partial[lane] : -INFINITY;
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
__global__ void merge_vocab_tile_kernel(
    const scalar_t* logits,
    const int64_t* target,
    float* row_max,
    float* row_sum,
    float* selected,
    int rows,
    int tile_size,
    int vocab_start) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float value_max = -INFINITY;
    for (int column = threadIdx.x; column < tile_size; column += blockDim.x) {
        value_max = fmaxf(value_max, static_cast<float>(logits[row * tile_size + column]));
    }
    const float tile_max = block_reduce_max(value_max);
    if (threadIdx.x == 0) {
        const float old_max = row_max[row];
        const float new_max = fmaxf(old_max, tile_max);
        row_sum[row] = row_sum[row] * expf(old_max - new_max);
        row_max[row] = new_max;
    }
    __syncthreads();

    float tile_sum = 0.0f;
    for (int column = threadIdx.x; column < tile_size; column += blockDim.x) {
        const float value = static_cast<float>(logits[row * tile_size + column]);
        tile_sum += expf(value - row_max[row]);
        if (target[row] == vocab_start + column) {
            selected[row] = value;
        }
    }
    tile_sum = block_reduce_sum(tile_sum);
    if (threadIdx.x == 0) {
        row_sum[row] += tile_sum;
    }
}

}  // namespace

std::vector<torch::Tensor> fused_linear_logp_musa_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias) {
    check_inputs(hidden, weight, target, bias);
    const int64_t rows = hidden.size(0);
    const int64_t vocab = weight.size(0);
    auto row_max = torch::full({rows}, -INFINITY, hidden.options().dtype(torch::kFloat));
    auto row_sum = torch::zeros({rows}, hidden.options().dtype(torch::kFloat));
    auto selected = torch::zeros({rows}, hidden.options().dtype(torch::kFloat));

    for (int64_t vocab_start = 0; vocab_start < vocab; vocab_start += kVocabTile) {
        const int64_t vocab_count = std::min(kVocabTile, vocab - vocab_start);
        auto logits = project_tile(hidden, weight, bias, vocab_start, vocab_count);
        auto stream = at::musa::getCurrentMUSAStream();
        merge_vocab_tile_kernel<c10::BFloat16><<<rows, 256, 0, stream>>>(
            logits.data_ptr<c10::BFloat16>(),
            target.data_ptr<int64_t>(),
            row_max.data_ptr<float>(),
            row_sum.data_ptr<float>(),
            selected.data_ptr<float>(),
            rows,
            vocab_count,
            vocab_start);
        C10_MUSA_KERNEL_LAUNCH_CHECK();
    }
    auto lse = row_max + row_sum.log();
    return {selected - lse, lse};
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
    check_inputs(hidden, weight, target, bias);
    TORCH_CHECK(
        grad_logp.device() == hidden.device() && grad_logp.dim() == 1 &&
            grad_logp.size(0) == hidden.size(0),
        "grad_logp must have one value per token");
    const int64_t rows = hidden.size(0);
    const int64_t vocab = weight.size(0);
    auto hidden_f = hidden.to(torch::kFloat).contiguous();
    auto weight_f = weight.to(torch::kFloat).contiguous();
    torch::optional<torch::Tensor> bias_f;
    if (bias.has_value()) {
        bias_f = bias->to(torch::kFloat).contiguous();
    }
    auto grad_f = grad_logp.to(torch::kFloat).contiguous();
    auto logits = at::mm(hidden.to(torch::kFloat), weight.to(torch::kFloat).t());
    if (bias.has_value()) {
        logits.add_(bias->to(torch::kFloat));
    }
    auto probabilities = at::_softmax(logits, 1, false);
    auto one_hot = torch::zeros_like(probabilities);
    one_hot.scatter_(1, target.unsqueeze(1), 1.0);
    auto dlogits = (one_hot - probabilities) * grad_f.unsqueeze(1);

    torch::Tensor grad_hidden;
    torch::Tensor grad_weight;
    torch::Tensor grad_bias;
    if (compute_grad_hidden) {
        grad_hidden = at::mm(dlogits, weight.to(torch::kFloat)).to(hidden.scalar_type());
    }
    if (compute_grad_weight) {
        grad_weight = at::mm(dlogits.t(), hidden.to(torch::kFloat)).to(weight.scalar_type());
    }
    if (compute_grad_bias && bias.has_value()) {
        grad_bias = dlogits.sum(0).to(bias->scalar_type());
    }
    return {grad_hidden, grad_weight, grad_bias};
}
