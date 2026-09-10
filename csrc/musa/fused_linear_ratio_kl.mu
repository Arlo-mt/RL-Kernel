// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <musa_runtime.h>
#include <torch/extension.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <ATen/Functions.h>
#include <cfloat>

namespace {

constexpr int kRowBlockSize = 1024;
constexpr int kElementBlockSize = 256;
constexpr int64_t kVocabTile = 65536;

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
  value = threadIdx.x < (kRowBlockSize / 32) ? partial[lane] : -FLT_MAX;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, 32));
    }
  }
  if (threadIdx.x == 0) {
    partial[0] = value;
  }
  __syncthreads();
  const float result = partial[0];
  __syncthreads();
  return result;
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
  value = threadIdx.x < (kRowBlockSize / 32) ? partial[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset, 32);
    }
  }
  if (threadIdx.x == 0) {
    partial[0] = value;
  }
  __syncthreads();
  const float result = partial[0];
  __syncthreads();
  return result;
}

__global__ void merge_policy_tile_kernel(const float *logits,
                                         const int64_t *target, float *row_max,
                                         float *row_sum, float *selected,
                                         int rows, int tile_size,
                                         int vocab_start) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float local_max = -FLT_MAX;
  for (int column = threadIdx.x; column < tile_size; column += blockDim.x) {
    local_max = fmaxf(local_max, logits[row * tile_size + column]);
  }
  const float tile_max = block_reduce_max(local_max);
  if (threadIdx.x == 0) {
    const float old_max = row_max[row];
    const float new_max = fmaxf(old_max, tile_max);
    row_sum[row] *= expf(old_max - new_max);
    row_max[row] = new_max;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int column = threadIdx.x; column < tile_size; column += blockDim.x) {
    const float value = logits[row * tile_size + column];
    local_sum += expf(value - row_max[row]);
    if (target[row] == vocab_start + column) {
      selected[row] = value;
    }
  }
  const float tile_sum = block_reduce_sum(local_sum);
  if (threadIdx.x == 0) {
    row_sum[row] += tile_sum;
  }
}

__global__ void linear_ratio_kl_epilogue_kernel(
    const float *row_max, const float *row_sum, const float *selected,
    const int32_t *mask, const float *old_logp, const float *reference_logp,
    float *ratio, float *kl, float *diff, float *policy_lse, int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  if (mask[row] == 0) {
    ratio[row] = 1.0f;
    kl[row] = 0.0f;
    diff[row] = 0.0f;
    policy_lse[row] = 0.0f;
    return;
  }
  const float lse = row_max[row] + logf(row_sum[row]);
  const float policy_logp = selected[row] - lse;
  const float delta = policy_logp - old_logp[row];
  const float d = reference_logp[row] - policy_logp;
  ratio[row] = expf(delta);
  kl[row] = expf(d) - d - 1.0f;
  diff[row] = d;
  policy_lse[row] = lse;
}

__global__ void linear_ratio_kl_dlogits_kernel(const float *logits,
                                               const float *policy_lse,
                                               const float *coefficient,
                                               const int64_t *target,
                                               float *dlogits, int rows,
                                               int tile_size, int vocab_start) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * tile_size;
  if (index >= total) {
    return;
  }
  const int row = index / tile_size;
  const int column = index % tile_size;
  const float probability = expf(logits[index] - policy_lse[row]);
  float gradient = -coefficient[row] * probability;
  if (target[row] == vocab_start + column) {
    gradient += coefficient[row];
  }
  dlogits[index] = gradient;
}

__global__ void linear_ratio_kl_coefficient_kernel(
    const float *ratio, const float *diff, const float *grad_ratio,
    const float *grad_kl, const int32_t *mask, float *coefficient, int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  coefficient[row] = mask[row] == 0
                         ? 0.0f
                         : grad_ratio[row] * ratio[row] +
                               grad_kl[row] * (1.0f - expf(diff[row]));
}

torch::Tensor project_tile(const torch::Tensor &hidden,
                           const torch::Tensor &weight,
                           const torch::optional<torch::Tensor> &bias,
                           int64_t vocab_start, int64_t vocab_count) {
  auto weight_tile = weight.narrow(0, vocab_start, vocab_count).t();
  auto logits = torch::empty({hidden.size(0), vocab_count},
                             hidden.options().dtype(torch::kFloat));
  at::mm_out(logits, hidden, weight_tile);
  if (bias.has_value()) {
    logits.add_(bias->narrow(0, vocab_start, vocab_count).to(torch::kFloat));
  }
  return logits;
}

} // namespace

std::vector<torch::Tensor> fused_linear_ratio_kl_musa_forward(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor target,
    torch::Tensor mask, torch::Tensor old_logp, torch::Tensor reference_logp,
    torch::optional<torch::Tensor> bias) {
  const int rows = hidden.size(0);
  const int64_t vocab = weight.size(0);
  auto float_options = hidden.options().dtype(torch::kFloat);
  auto row_max = torch::full({rows}, -INFINITY, float_options);
  auto row_sum = torch::zeros({rows}, float_options);
  auto selected = torch::zeros({rows}, float_options);
  for (int64_t vocab_start = 0; vocab_start < vocab;
       vocab_start += kVocabTile) {
    const int64_t vocab_count = std::min(kVocabTile, vocab - vocab_start);
    auto logits = project_tile(hidden, weight, bias, vocab_start, vocab_count);
    auto stream = at::musa::getCurrentMUSAStream();
    merge_policy_tile_kernel<<<rows, kRowBlockSize, 0, stream>>>(
        logits.data_ptr<float>(), target.data_ptr<int64_t>(),
        row_max.data_ptr<float>(), row_sum.data_ptr<float>(),
        selected.data_ptr<float>(), rows, vocab_count, vocab_start);
    C10_MUSA_KERNEL_LAUNCH_CHECK();
  }

  auto ratio = torch::empty({rows}, float_options);
  auto kl = torch::empty({rows}, float_options);
  auto diff = torch::empty({rows}, float_options);
  auto policy_lse = torch::empty({rows}, float_options);
  auto stream = at::musa::getCurrentMUSAStream();
  linear_ratio_kl_epilogue_kernel<<<(rows + kElementBlockSize - 1) /
                                        kElementBlockSize,
                                    kElementBlockSize, 0, stream>>>(
      row_max.data_ptr<float>(), row_sum.data_ptr<float>(),
      selected.data_ptr<float>(), mask.data_ptr<int32_t>(),
      old_logp.data_ptr<float>(), reference_logp.data_ptr<float>(),
      ratio.data_ptr<float>(), kl.data_ptr<float>(), diff.data_ptr<float>(),
      policy_lse.data_ptr<float>(), rows);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return {ratio, kl, diff, policy_lse};
}

std::vector<torch::Tensor> fused_linear_ratio_kl_musa_backward(
    torch::Tensor grad_ratio, torch::Tensor grad_kl, torch::Tensor hidden,
    torch::Tensor weight, torch::Tensor target, torch::Tensor mask,
    torch::Tensor ratio, torch::Tensor diff, torch::Tensor policy_lse,
    torch::optional<torch::Tensor> bias, bool compute_grad_hidden,
    bool compute_grad_weight, bool compute_grad_bias) {
  const int rows = hidden.size(0);
  const int64_t vocab = weight.size(0);
  auto hidden_f = hidden.to(torch::kFloat).contiguous();
  auto weight_f = weight.to(torch::kFloat).contiguous();
  auto grad_hidden_f = compute_grad_hidden
                           ? torch::zeros({rows, hidden.size(1)},
                                          hidden.options().dtype(torch::kFloat))
                           : torch::Tensor();
  auto grad_weight_f = compute_grad_weight
                           ? torch::empty({vocab, hidden.size(1)},
                                          hidden.options().dtype(torch::kFloat))
                           : torch::Tensor();
  auto grad_bias_f =
      compute_grad_bias && bias.has_value()
          ? torch::empty({vocab}, hidden.options().dtype(torch::kFloat))
          : torch::Tensor();
  auto coefficient =
      torch::empty({rows}, hidden.options().dtype(torch::kFloat));
  auto stream = at::musa::getCurrentMUSAStream();
  linear_ratio_kl_coefficient_kernel<<<(rows + kElementBlockSize - 1) /
                                           kElementBlockSize,
                                       kElementBlockSize, 0, stream>>>(
      ratio.data_ptr<float>(), diff.data_ptr<float>(),
      grad_ratio.data_ptr<float>(), grad_kl.data_ptr<float>(),
      mask.data_ptr<int32_t>(), coefficient.data_ptr<float>(), rows);
  C10_MUSA_KERNEL_LAUNCH_CHECK();

  for (int64_t vocab_start = 0; vocab_start < vocab;
       vocab_start += kVocabTile) {
    const int64_t vocab_count = std::min(kVocabTile, vocab - vocab_start);
    auto logits = project_tile(hidden, weight, bias, vocab_start, vocab_count);
    auto dlogits = torch::empty({rows, vocab_count},
                                hidden.options().dtype(torch::kFloat));
    const int total = rows * vocab_count;
    linear_ratio_kl_dlogits_kernel<<<(total + kElementBlockSize - 1) /
                                         kElementBlockSize,
                                     kElementBlockSize, 0, stream>>>(
        logits.data_ptr<float>(), policy_lse.data_ptr<float>(),
        coefficient.data_ptr<float>(), target.data_ptr<int64_t>(),
        dlogits.data_ptr<float>(), rows, vocab_count, vocab_start);
    C10_MUSA_KERNEL_LAUNCH_CHECK();

    if (compute_grad_hidden) {
      grad_hidden_f.add_(
          at::mm(dlogits, weight_f.narrow(0, vocab_start, vocab_count)));
    }
    if (compute_grad_weight) {
      auto grad_weight_tile = grad_weight_f.narrow(0, vocab_start, vocab_count);
      at::mm_out(grad_weight_tile, dlogits.t(), hidden_f);
    }
    if (compute_grad_bias && bias.has_value()) {
      grad_bias_f.narrow(0, vocab_start, vocab_count).copy_(dlogits.sum(0));
    }
  }

  torch::Tensor grad_hidden;
  torch::Tensor grad_weight;
  torch::Tensor grad_bias;
  if (compute_grad_hidden) {
    grad_hidden = grad_hidden_f.to(hidden.scalar_type());
  }
  if (compute_grad_weight) {
    grad_weight = grad_weight_f.to(weight.scalar_type());
  }
  if (compute_grad_bias && bias.has_value()) {
    grad_bias = grad_bias_f.to(bias->scalar_type());
  }
  return {grad_hidden, grad_weight, grad_bias};
}
