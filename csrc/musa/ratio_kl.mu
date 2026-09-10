// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <musa_runtime.h>
#include <torch/extension.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <cfloat>

namespace {

constexpr int kMaxWarps = 32;

template <int BlockSize>
__device__ __forceinline__ void block_reduce_max_pair(float &left,
                                                      float &right) {
  __shared__ float partial_left[kMaxWarps];
  __shared__ float partial_right[kMaxWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    left = fmaxf(left, __shfl_down_sync(0xffffffffu, left, offset, 32));
    right = fmaxf(right, __shfl_down_sync(0xffffffffu, right, offset, 32));
  }
  if (lane == 0) {
    partial_left[warp] = left;
    partial_right[warp] = right;
  }
  __syncthreads();

  left = threadIdx.x < (BlockSize / 32) ? partial_left[lane] : -FLT_MAX;
  right = threadIdx.x < (BlockSize / 32) ? partial_right[lane] : -FLT_MAX;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      left = fmaxf(left, __shfl_down_sync(0xffffffffu, left, offset, 32));
      right = fmaxf(right, __shfl_down_sync(0xffffffffu, right, offset, 32));
    }
  }
  if (threadIdx.x == 0) {
    partial_left[0] = left;
    partial_right[0] = right;
  }
  __syncthreads();
  left = partial_left[0];
  right = partial_right[0];
  __syncthreads();
}

template <int BlockSize>
__device__ __forceinline__ void block_reduce_sum_pair(float &left,
                                                      float &right) {
  __shared__ float partial_left[kMaxWarps];
  __shared__ float partial_right[kMaxWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    left += __shfl_down_sync(0xffffffffu, left, offset, 32);
    right += __shfl_down_sync(0xffffffffu, right, offset, 32);
  }
  if (lane == 0) {
    partial_left[warp] = left;
    partial_right[warp] = right;
  }
  __syncthreads();

  left = threadIdx.x < (BlockSize / 32) ? partial_left[lane] : 0.0f;
  right = threadIdx.x < (BlockSize / 32) ? partial_right[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      left += __shfl_down_sync(0xffffffffu, left, offset, 32);
      right += __shfl_down_sync(0xffffffffu, right, offset, 32);
    }
  }
  if (threadIdx.x == 0) {
    partial_left[0] = left;
    partial_right[0] = right;
  }
  __syncthreads();
  left = partial_left[0];
  right = partial_right[0];
  __syncthreads();
}

template <typename scalar_t, int BlockSize>
__global__ void
ratio_kl_forward_kernel(const scalar_t *policy, const scalar_t *reference,
                        const int64_t *action, const int32_t *mask,
                        const float *old_logp, float *ratio, float *kl,
                        float *diff, float *policy_logz, int rows, int vocab) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  if (mask[row] == 0) {
    if (threadIdx.x == 0) {
      ratio[row] = 1.0f;
      kl[row] = 0.0f;
      diff[row] = 0.0f;
      policy_logz[row] = 0.0f;
    }
    return;
  }

  const size_t row_offset = static_cast<size_t>(row) * vocab;
  float max_policy = -FLT_MAX;
  float max_reference = -FLT_MAX;
  for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
    max_policy =
        fmaxf(max_policy, static_cast<float>(policy[row_offset + column]));
    max_reference = fmaxf(max_reference,
                          static_cast<float>(reference[row_offset + column]));
  }
  block_reduce_max_pair<BlockSize>(max_policy, max_reference);

  float sum_policy = 0.0f;
  float sum_reference = 0.0f;
  for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
    sum_policy +=
        expf(static_cast<float>(policy[row_offset + column]) - max_policy);
    sum_reference += expf(static_cast<float>(reference[row_offset + column]) -
                          max_reference);
  }
  block_reduce_sum_pair<BlockSize>(sum_policy, sum_reference);

  if (threadIdx.x == 0) {
    const int64_t action_id = action[row];
    const float logz_policy = max_policy + logf(sum_policy);
    const float logz_reference = max_reference + logf(sum_reference);
    const float selected_policy =
        static_cast<float>(policy[row_offset + action_id]) - logz_policy;
    const float selected_reference =
        static_cast<float>(reference[row_offset + action_id]) - logz_reference;
    const float delta = selected_policy - old_logp[row];
    const float d = selected_reference - selected_policy;
    ratio[row] = expf(delta);
    kl[row] = expf(d) - d - 1.0f;
    diff[row] = d;
    policy_logz[row] = logz_policy;
  }
}

template <typename scalar_t, int BlockSize>
__global__ void
ratio_kl_backward_kernel(const scalar_t *policy, const int64_t *action,
                         const int32_t *mask, const float *ratio,
                         const float *diff, const float *policy_logz,
                         const float *grad_ratio, const float *grad_kl,
                         scalar_t *grad_policy, int rows, int vocab) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const size_t row_offset = static_cast<size_t>(row) * vocab;
  if (mask[row] == 0) {
    for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
      grad_policy[row_offset + column] = static_cast<scalar_t>(0.0f);
    }
    return;
  }

  const float coefficient =
      grad_ratio[row] * ratio[row] + grad_kl[row] * (1.0f - expf(diff[row]));
  const int64_t action_id = action[row];
  const float logz = policy_logz[row];
  for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
    const float probability =
        expf(static_cast<float>(policy[row_offset + column]) - logz);
    const float one_hot = column == action_id ? 1.0f : 0.0f;
    grad_policy[row_offset + column] =
        static_cast<scalar_t>(coefficient * (one_hot - probability));
  }
}

} // namespace

std::vector<torch::Tensor> ratio_kl_musa_forward_impl(torch::Tensor policy,
                                                      torch::Tensor reference,
                                                      torch::Tensor action,
                                                      torch::Tensor mask,
                                                      torch::Tensor old_logp) {
  const int rows = static_cast<int>(policy.size(0));
  const int vocab = static_cast<int>(policy.size(1));
  auto float_options = policy.options().dtype(torch::kFloat);
  auto ratio = torch::empty({rows}, float_options);
  auto kl = torch::empty({rows}, float_options);
  auto diff = torch::empty({rows}, float_options);
  auto policy_logz = torch::empty({rows}, float_options);
  if (rows == 0) {
    return {ratio, kl, diff, policy_logz};
  }

  auto stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, policy.scalar_type(),
      "musa_ratio_kl_forward", [&] {
        if (vocab <= 32768) {
          ratio_kl_forward_kernel<scalar_t, 256><<<rows, 256, 0, stream>>>(
              policy.data_ptr<scalar_t>(), reference.data_ptr<scalar_t>(),
              action.data_ptr<int64_t>(), mask.data_ptr<int32_t>(),
              old_logp.data_ptr<float>(), ratio.data_ptr<float>(),
              kl.data_ptr<float>(), diff.data_ptr<float>(),
              policy_logz.data_ptr<float>(), rows, vocab);
        } else {
          ratio_kl_forward_kernel<scalar_t, 512><<<rows, 512, 0, stream>>>(
              policy.data_ptr<scalar_t>(), reference.data_ptr<scalar_t>(),
              action.data_ptr<int64_t>(), mask.data_ptr<int32_t>(),
              old_logp.data_ptr<float>(), ratio.data_ptr<float>(),
              kl.data_ptr<float>(), diff.data_ptr<float>(),
              policy_logz.data_ptr<float>(), rows, vocab);
        }
      });
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return {ratio, kl, diff, policy_logz};
}

torch::Tensor
ratio_kl_musa_backward_impl(torch::Tensor policy, torch::Tensor action,
                            torch::Tensor mask, torch::Tensor ratio,
                            torch::Tensor diff, torch::Tensor policy_logz,
                            torch::Tensor grad_ratio, torch::Tensor grad_kl) {
  const int rows = static_cast<int>(policy.size(0));
  const int vocab = static_cast<int>(policy.size(1));
  auto grad_policy = torch::empty_like(policy);
  if (rows == 0) {
    return grad_policy;
  }

  auto stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, policy.scalar_type(),
      "musa_ratio_kl_backward", [&] {
        if (vocab < 32768) {
          ratio_kl_backward_kernel<scalar_t, 256><<<rows, 256, 0, stream>>>(
              policy.data_ptr<scalar_t>(), action.data_ptr<int64_t>(),
              mask.data_ptr<int32_t>(), ratio.data_ptr<float>(),
              diff.data_ptr<float>(), policy_logz.data_ptr<float>(),
              grad_ratio.data_ptr<float>(), grad_kl.data_ptr<float>(),
              grad_policy.data_ptr<scalar_t>(), rows, vocab);
        } else {
          ratio_kl_backward_kernel<scalar_t, 512><<<rows, 512, 0, stream>>>(
              policy.data_ptr<scalar_t>(), action.data_ptr<int64_t>(),
              mask.data_ptr<int32_t>(), ratio.data_ptr<float>(),
              diff.data_ptr<float>(), policy_logz.data_ptr<float>(),
              grad_ratio.data_ptr<float>(), grad_kl.data_ptr<float>(),
              grad_policy.data_ptr<scalar_t>(), rows, vocab);
        }
      });
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return grad_policy;
}
