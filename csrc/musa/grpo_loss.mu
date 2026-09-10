// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <musa_runtime.h>
#include <torch/extension.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ void block_reduce_sum_pair(float &left,
                                                      float &right) {
  __shared__ float partial_left[32];
  __shared__ float partial_right[32];
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

  left = threadIdx.x < (kBlockSize / 32) ? partial_left[lane] : 0.0f;
  right = threadIdx.x < (kBlockSize / 32) ? partial_right[lane] : 0.0f;
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

__global__ void group_advantages_kernel(const float *rewards,
                                        const int32_t *bounds,
                                        float *advantages,
                                        float eps,
                                        int groups) {
  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }

  const int start = bounds[group];
  const int end = bounds[group + 1];
  const int size = end - start;
  if (size <= 0) {
    return;
  }

  float local_sum = 0.0f;
  float local_sq_sum = 0.0f;
  for (int offset = threadIdx.x; offset < size; offset += blockDim.x) {
    const float value = rewards[start + offset];
    local_sum += value;
    local_sq_sum += value * value;
  }

  block_reduce_sum_pair(local_sum, local_sq_sum);
  const float inv_count = 1.0f / static_cast<float>(size);
  const float mean = local_sum * inv_count;
  const float variance = fmaxf(local_sq_sum * inv_count - mean * mean, 0.0f);
  const float std_value = fmaxf(sqrtf(variance), eps);

  for (int offset = threadIdx.x; offset < size; offset += blockDim.x) {
    advantages[start + offset] = (rewards[start + offset] - mean) / std_value;
  }
}

} // namespace

torch::Tensor grpo_group_advantages_musa_impl(torch::Tensor rewards,
                                              torch::Tensor bounds,
                                              double eps) {
  const int groups = static_cast<int>(bounds.numel() - 1);
  auto advantages = torch::empty_like(rewards);
  if (groups == 0 || rewards.numel() == 0) {
    return advantages;
  }

  auto stream = at::musa::getCurrentMUSAStream();
  group_advantages_kernel<<<groups, kBlockSize, 0, stream>>>(
      rewards.data_ptr<float>(), bounds.data_ptr<int32_t>(),
      advantages.data_ptr<float>(), static_cast<float>(eps), groups);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return advantages;
}
