// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Portable native MUSA implementations for the CUDA WS1 profile.
// These kernels deliberately avoid SM90-only TMA, WGMMA, and PTX features.

#include <torch/extension.h>

#include <ATen/Functions.h>
#include <musa_runtime.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kHeadDim = 128;
constexpr int kBatchInvariantThreads = 128;
constexpr int kBatchInvariantTile = 4096;
constexpr float kLogpNegInf = -3.4028234663852886e38F;

template <typename T>
__device__ __forceinline__ float as_float(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template <int BlockSize>
__device__ __forceinline__ float block_sum(float value) {
  static_assert(BlockSize % 32 == 0, "BlockSize must be a warp multiple");
  __shared__ float shared[BlockSize / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset, 32);
  }
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = threadIdx.x < (BlockSize / 32) ? shared[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset, 32);
    }
  }
  return value;
}

template <int BlockSize>
__device__ __forceinline__ float block_max(float value) {
  static_assert(BlockSize % 32 == 0, "BlockSize must be a warp multiple");
  __shared__ float shared[BlockSize / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, 32));
  }
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = threadIdx.x < (BlockSize / 32) ? shared[lane] : kLogpNegInf;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, 32));
    }
  }
  return value;
}
__device__ __forceinline__ float block_sum_128(float value) {
  __shared__ float shared[128];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ float block_max_128(float value) {
  __shared__ float shared[128];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ float silu_f32(float value) {
  const float sigmoid = 1.0f / (1.0f + expf(-value));
  return value * sigmoid;
}

__device__ __forceinline__ float silu_grad_f32(float value) {
  const float sigmoid = 1.0f / (1.0f + expf(-value));
  return sigmoid * (1.0f + value * (1.0f - sigmoid));
}

void check_musa(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.device().type() == c10::kPrivateUse1,
      name,
      " must be a MUSA tensor");
}

void check_same_device(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const char* name) {
  check_musa(lhs, name);
  check_musa(rhs, name);
  TORCH_CHECK(lhs.device() == rhs.device(), name, " tensors must share a device");
}

int blocks_for(int64_t elements) {
  return static_cast<int>((elements + kThreads - 1) / kThreads);
}

int rmsnorm_threads(int64_t hidden) {
  if (hidden <= 64) return 64;
  if (hidden <= 128) return 128;
  if (hidden <= 256) return 256;
  return 512;
}

template <typename input_t, typename output_t>
__global__ void embedding_forward_kernel(
    const int64_t* ids,
    const input_t* weight,
    output_t* output,
    int64_t rows,
    int64_t hidden) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * hidden) return;
  int64_t row = index / hidden;
  int64_t column = index % hidden;
  output[index] = from_float<output_t>(as_float(weight[ids[row] * hidden + column]));
}

template <typename scalar_t, typename output_t>
__global__ void lm_head_forward_kernel(
    const scalar_t* hidden,
    const scalar_t* weight,
    const float* bias,
    output_t* output,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size) {
  int64_t index = blockIdx.x;
  if (index >= rows * vocab_size) return;
  int64_t row = index / vocab_size;
  int64_t vocab = index % vocab_size;
  const scalar_t* hidden_row = hidden + row * hidden_size;
  const scalar_t* weight_row = weight + vocab * hidden_size;
  float accumulator = 0.0f;
  for (int64_t k = threadIdx.x; k < hidden_size; k += blockDim.x) {
    accumulator = fmaf(as_float(hidden_row[k]), as_float(weight_row[k]), accumulator);
  }
  accumulator = block_sum<256>(accumulator);
  if (threadIdx.x == 0) {
    if (bias != nullptr) accumulator += bias[vocab];
    output[index] = from_float<output_t>(accumulator);
  }
}

// Alternative forward-only reduction used for MUSA's large-K FP32 rowwise
// GEMM. Each lane owns a contiguous K chunk before the fixed block tree,
// which is a portable substitute for the CUDA path's strided lane mapping.
template <typename grad_t>
__global__ void embedding_backward_kernel(
    const int64_t* ids,
    const grad_t* grad_output,
    grad_t* grad_weight,
    int64_t rows,
    int64_t vocab,
    int64_t hidden) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vocab * hidden) return;
  int64_t vocab_row = index / hidden;
  int64_t column = index % hidden;
  float sum = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    if (ids[row] == vocab_row) {
      sum += as_float(grad_output[row * hidden + column]);
    }
  }
  grad_weight[index] = from_float<grad_t>(sum);
}

template <typename input_t, typename weight_t, typename output_t>
__global__ void rmsnorm_forward_kernel(
    const input_t* x,
    const weight_t* weight,
    output_t* y,
    float* rstd,
    int64_t rows,
    int64_t hidden,
    float eps) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  int tid = threadIdx.x;
  const input_t* x_row = x + row * hidden;
  float local_sum = 0.0f;
  for (int64_t column = tid; column < hidden; column += blockDim.x) {
    float value = as_float(x_row[column]);
    local_sum += value * value;
  }
  extern __shared__ float shared[];
  shared[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  float inv = rsqrtf(shared[0] / static_cast<float>(hidden) + eps);
  if (tid == 0) rstd[row] = inv;
  __syncthreads();
  output_t* y_row = y + row * hidden;
  for (int64_t column = tid; column < hidden; column += blockDim.x) {
    y_row[column] = from_float<output_t>(
        as_float(x_row[column]) * inv * as_float(weight[column]));
  }
}

template <typename input_t, typename weight_t, typename output_t>
__global__ void rmsnorm_backward_dx_kernel(
    const input_t* x,
    const weight_t* weight,
    const float* rstd,
    const output_t* grad_y,
    output_t* grad_x,
    int64_t rows,
    int64_t hidden) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  int tid = threadIdx.x;
  const input_t* x_row = x + row * hidden;
  const output_t* gy_row = grad_y + row * hidden;
  output_t* gx_row = grad_x + row * hidden;
  float local_dot = 0.0f;
  float inv = rstd[row];
  for (int64_t column = tid; column < hidden; column += blockDim.x) {
    local_dot +=
        as_float(gy_row[column]) * as_float(weight[column]) * as_float(x_row[column]);
  }
  extern __shared__ float shared[];
  shared[tid] = local_dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  float coefficient =
      shared[0] * inv * inv * inv / static_cast<float>(hidden);
  for (int64_t column = tid; column < hidden; column += blockDim.x) {
    float value = inv * as_float(gy_row[column]) * as_float(weight[column])
        - as_float(x_row[column]) * coefficient;
    gx_row[column] = from_float<output_t>(value);
  }
}

template <typename input_t, typename grad_t, typename weight_t>
__global__ void rmsnorm_backward_dw_kernel(
    const input_t* x,
    const float* rstd,
    const grad_t* grad_y,
    weight_t* grad_weight,
    int64_t rows,
    int64_t hidden) {
  int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= hidden) return;
  float sum = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    sum += as_float(grad_y[row * hidden + column]) *
        as_float(x[row * hidden + column]) * rstd[row];
  }
  grad_weight[column] = from_float<weight_t>(sum);
}

__device__ __forceinline__ int64_t rope_position(
    const int64_t* positions,
    int64_t batch,
    int64_t sequence,
    int64_t b,
    int64_t s,
    bool batched) {
  return positions[batched ? b * sequence + s : s];
}

template <typename scalar_t>
__global__ void rope_forward_kernel(
    const scalar_t* x,
    const int64_t* positions,
    scalar_t* output,
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t dim,
    double theta,
    bool batched) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * heads * sequence * dim) return;
  int64_t d = index % dim;
  int64_t s = (index / dim) % sequence;
  int64_t h = (index / (sequence * dim)) % heads;
  int64_t b = index / (heads * sequence * dim);
  int64_t half = dim / 2;
  int64_t pair = d < half ? d : d - half;
  float angle = static_cast<float>(rope_position(positions, batch, sequence, b, s, batched)) *
      powf(static_cast<float>(theta), -static_cast<float>(pair) / static_cast<float>(half));
  float c = cosf(angle);
  float sine = sinf(angle);
  int64_t mate = d < half ? d + half : d - half;
  float value = as_float(x[index]);
  float rotated = -as_float(x[
      ((b * heads + h) * sequence + s) * dim + mate]);
  if (d >= half) rotated = -rotated;
  output[index] = from_float<scalar_t>(value * c + rotated * sine);
}

template <typename scalar_t>
__global__ void rope_backward_kernel(
    const scalar_t* grad_output,
    const int64_t* positions,
    scalar_t* grad_x,
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t dim,
    double theta,
    bool batched) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * heads * sequence * dim) return;
  int64_t d = index % dim;
  int64_t s = (index / dim) % sequence;
  int64_t h = (index / (sequence * dim)) % heads;
  int64_t b = index / (heads * sequence * dim);
  int64_t half = dim / 2;
  int64_t pair = d < half ? d : d - half;
  float angle = static_cast<float>(rope_position(positions, batch, sequence, b, s, batched)) *
      powf(static_cast<float>(theta), -static_cast<float>(pair) / static_cast<float>(half));
  float c = cosf(angle);
  float sine = sinf(angle);
  int64_t mate = d < half ? d + half : d - half;
  float value = as_float(grad_output[index]) * c;
  if (d < half) {
    value += as_float(grad_output[
        ((b * heads + h) * sequence + s) * dim + mate]) * sine;
  } else {
    value -= as_float(grad_output[
        ((b * heads + h) * sequence + s) * dim + mate]) * sine;
  }
  grad_x[index] = from_float<scalar_t>(value);
}

template <typename scalar_t>
__global__ void rope_apply_kernel(
    const scalar_t* x,
    const float* cos,
    const float* sin,
    scalar_t* output,
    int64_t rows,
    int sequence,
    int half,
    float sin_sign) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * half;
  if (index >= total) return;
  int64_t row = index / half;
  int i = static_cast<int>(index % half);
  int seq = static_cast<int>(row % sequence);
  float c = cos[seq * half + i];
  float s = sin[seq * half + i] * sin_sign;
  int64_t base = row * (2LL * half);
  float x1 = as_float(x[base + i]);
  float x2 = as_float(x[base + i + half]);
  output[base + i] = from_float<scalar_t>(x1 * c - x2 * s);
  output[base + i + half] = from_float<scalar_t>(x2 * c + x1 * s);
}

template <typename scalar_t>
__global__ void silu_forward_kernel(
    const scalar_t* x,
    scalar_t* output,
    int64_t elements) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float value = as_float(x[index]);
  output[index] = from_float<scalar_t>(silu_f32(value));
}

template <typename scalar_t>
__global__ void silu_backward_kernel(
    const scalar_t* x,
    const scalar_t* grad_output,
    scalar_t* grad_x,
    int64_t elements) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float value = as_float(x[index]);
  grad_x[index] = from_float<scalar_t>(
      as_float(grad_output[index]) * silu_grad_f32(value));
}

template <typename scalar_t>
__global__ void swiglu_forward_kernel(
    const scalar_t* gate,
    const scalar_t* up,
    scalar_t* output,
    int64_t elements) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float g = as_float(gate[index]);
  output[index] = from_float<scalar_t>(silu_f32(g) * as_float(up[index]));
}

template <typename scalar_t>
__global__ void swiglu_backward_kernel(
    const scalar_t* gate,
    const scalar_t* up,
    const scalar_t* grad_output,
    scalar_t* grad_gate,
    scalar_t* grad_up,
    int64_t elements) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float g = as_float(gate[index]);
  float u = as_float(up[index]);
  float grad = as_float(grad_output[index]);
  grad_gate[index] = from_float<scalar_t>(grad * u * silu_grad_f32(g));
  grad_up[index] = from_float<scalar_t>(grad * silu_f32(g));
}

template <typename scalar_t, int BlockSize>
__global__ void selected_logp_forward_kernel(
    const scalar_t* logits,
    const int64_t* target,
    scalar_t* output,
    int64_t rows,
  int64_t vocab,
  int64_t ignore_index) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* values = logits + row * vocab;
  float local_max = kLogpNegInf;
  for (int64_t column = threadIdx.x; column < vocab; column += blockDim.x) {
    local_max = fmaxf(local_max, as_float(values[column]));
  }
  float max_value = block_max<BlockSize>(local_max);
  __shared__ float row_max;
  if (threadIdx.x == 0) row_max = max_value;
  __syncthreads();
  float local_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < vocab; column += blockDim.x) {
    local_sum += expf(as_float(values[column]) - row_max);
  }
  float sum = block_sum<BlockSize>(local_sum);
  __shared__ float row_sum;
  if (threadIdx.x == 0) row_sum = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    int64_t id = target[row];
    bool valid = id != ignore_index && id >= 0 && id < vocab;
    float selected = valid ? as_float(values[id]) : 0.0f;
    output[row] = from_float<scalar_t>(
        valid ? selected - row_max - logf(row_sum) : 0.0f);
  }
}

template <typename scalar_t>
__global__ void selected_logp_backward_kernel(
    const scalar_t* logits,
    const int64_t* target,
    const scalar_t* grad_output,
    scalar_t* grad_logits,
    int64_t rows,
    int64_t vocab,
    int64_t ignore_index) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * vocab) return;
  int64_t row = index / vocab;
  int64_t column = index % vocab;
  const scalar_t* values = logits + row * vocab;
  float max_value = -1e20f;
  for (int64_t j = 0; j < vocab; ++j) {
    max_value = fmaxf(max_value, as_float(values[j]));
  }
  float sum = 0.0f;
  for (int64_t j = 0; j < vocab; ++j) {
    sum += expf(as_float(values[j]) - max_value);
  }
  int64_t id = target[row];
  float probability = expf(as_float(values[column]) - max_value) / sum;
  float indicator = (id != ignore_index && id == column) ? 1.0f : 0.0f;
  float grad = (id == ignore_index || id < 0 || id >= vocab)
      ? 0.0f
      : as_float(grad_output[row]) * (indicator - probability);
  grad_logits[index] = from_float<scalar_t>(grad);
}

template <typename scalar_t>
__global__ void attention_qk_kernel(
    const scalar_t* q,
    const scalar_t* k,
    float* scores,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim,
    float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * query_heads * query_length * key_length;
  if (index >= total) return;
  int64_t key_pos = index % key_length;
  int64_t row = index / key_length;
  int64_t q_pos = row % query_length;
  int64_t hq = (row / query_length) % query_heads;
  int64_t b = row / (query_heads * query_length);
  int64_t kv_head = hq / (query_heads / kv_heads);
  const scalar_t* q_row = q + ((b * query_heads + hq) * query_length + q_pos) * dim;
  const scalar_t* k_row =
      k + ((b * kv_heads + kv_head) * key_length + key_pos) * dim;
  float dot = 0.0f;
  for (int64_t d = 0; d < dim; ++d) {
    dot += as_float(q_row[d]) * as_float(k_row[d]);
  }
  scores[index] = dot * scale;
}

__global__ void attention_softmax_kernel(
    float* scores,
    float* lse,
    const bool* mask,
    int64_t query_heads,
    int64_t query_length,
    int64_t key_length,
    bool causal) {
  int64_t row = blockIdx.x;
  int64_t b = row / (query_heads * query_length);
  int64_t q_pos = row % query_length;
  float* values = scores + row * key_length;
  int64_t causal_limit = causal ? key_length - query_length + q_pos + 1 : key_length;
  __shared__ float shared_max[kThreads];
  float local_max = -INFINITY;
  for (int64_t key_pos = threadIdx.x; key_pos < key_length; key_pos += blockDim.x) {
    bool valid = key_pos < causal_limit;
    if (valid && mask != nullptr) valid = mask[b * key_length + key_pos];
    if (!valid) {
      values[key_pos] = -INFINITY;
    } else {
      local_max = fmaxf(local_max, values[key_pos]);
    }
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] =
          fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float row_max = shared_max[0];
  __shared__ float shared_sum[kThreads];
  float local_sum = 0.0f;
  for (int64_t key_pos = threadIdx.x; key_pos < key_length; key_pos += blockDim.x) {
    bool valid = key_pos < causal_limit;
    if (valid && mask != nullptr) valid = mask[b * key_length + key_pos];
    float value = valid ? expf(values[key_pos] - row_max) : 0.0f;
    values[key_pos] = value;
    local_sum += value;
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    __syncthreads();
  }
  float row_sum = shared_sum[0];
  if (threadIdx.x == 0) lse[row] = row_sum == 0.0f ? -INFINITY : row_max + logf(row_sum);
  for (int64_t key_pos = threadIdx.x; key_pos < key_length; key_pos += blockDim.x) {
    values[key_pos] = row_sum == 0.0f ? 0.0f : values[key_pos] / row_sum;
  }
}

__global__ void attention_softmax_serial_kernel(
    float* scores,
    float* lse,
    const bool* mask,
    int64_t query_heads,
    int64_t query_length,
    int64_t key_length,
    bool causal) {
  const int64_t row = blockIdx.x;
  if (threadIdx.x != 0) return;
  const int64_t batch = row / (query_heads * query_length);
  const int64_t query_pos = row % query_length;
  float* values = scores + row * key_length;
  const int64_t causal_limit =
      causal ? key_length - query_length + query_pos + 1 : key_length;
  float row_max = -INFINITY;
  for (int64_t key_pos = 0; key_pos < key_length; ++key_pos) {
    const bool valid = key_pos < causal_limit &&
        (mask == nullptr || mask[batch * key_length + key_pos]);
    if (valid) row_max = fmaxf(row_max, values[key_pos]);
    else values[key_pos] = -INFINITY;
  }
  float row_sum = 0.0f;
  for (int64_t key_pos = 0; key_pos < key_length; ++key_pos) {
    const float probability = isinf(values[key_pos])
        ? 0.0f
        : expf(values[key_pos] - row_max);
    values[key_pos] = probability;
    row_sum += probability;
  }
  if (row_sum == 0.0f) {
    lse[row] = -INFINITY;
    return;
  }
  lse[row] = row_max + logf(row_sum);
  for (int64_t key_pos = 0; key_pos < key_length; ++key_pos) {
    values[key_pos] /= row_sum;
  }
}

template <typename scalar_t, typename output_t>
__global__ void attention_pv_kernel(
    const float* probabilities,
    const scalar_t* v,
    output_t* output,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * query_heads * query_length * dim) return;
  int64_t d = index % dim;
  int64_t q_pos = (index / dim) % query_length;
  int64_t hq = (index / (query_length * dim)) % query_heads;
  int64_t b = index / (query_heads * query_length * dim);
  int64_t kv_head = hq / (query_heads / kv_heads);
  int64_t row = (b * query_heads + hq) * query_length + q_pos;
  const scalar_t* v_base = v + (b * kv_heads + kv_head) * key_length * dim;
  float sum = 0.0f;
  for (int64_t key_pos = 0; key_pos < key_length; ++key_pos) {
    sum += probabilities[row * key_length + key_pos] *
        as_float(v_base[key_pos * dim + d]);
  }
  output[index] = from_float<output_t>(sum);
}

template <typename scalar_t>
__global__ void attention_backward_dq_kernel(
    const scalar_t* grad_output,
    const scalar_t* k,
    const scalar_t* v,
    const float* probabilities,
    scalar_t* grad_q,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim,
    float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * query_heads * query_length * dim) return;
  int64_t d = index % dim;
  int64_t q_pos = (index / dim) % query_length;
  int64_t hq = (index / (query_length * dim)) % query_heads;
  int64_t b = index / (query_heads * query_length * dim);
  int64_t kv_head = hq / (query_heads / kv_heads);
  int64_t row = (b * query_heads + hq) * query_length + q_pos;
  float sum_pd = 0.0f;
  for (int64_t key_pos = 0; key_pos < key_length; ++key_pos) {
    const scalar_t* v_row = v + ((b * kv_heads + kv_head) * key_length + key_pos) * dim;
    float dp = 0.0f;
    for (int64_t j = 0; j < dim; ++j) {
      dp += as_float(grad_output[index - d + j]) * as_float(v_row[j]);
    }
    sum_pd += probabilities[row * key_length + key_pos] * dp;
  }
  float result = 0.0f;
  for (int64_t key_pos = 0; key_pos < key_length; ++key_pos) {
    float probability = probabilities[row * key_length + key_pos];
    const scalar_t* v_row = v + ((b * kv_heads + kv_head) * key_length + key_pos) * dim;
    const scalar_t* k_row = k + ((b * kv_heads + kv_head) * key_length + key_pos) * dim;
    float dp = 0.0f;
    for (int64_t j = 0; j < dim; ++j) {
      dp += as_float(grad_output[index - d + j]) * as_float(v_row[j]);
    }
    result += probability * (dp - sum_pd) * as_float(k_row[d]) * scale;
  }
  grad_q[index] = from_float<scalar_t>(result);
}

template <typename scalar_t>
__global__ void attention_dp_kernel(
    const scalar_t* grad_output,
    const scalar_t* v,
    float* dp,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * query_heads * query_length * key_length;
  if (index >= total) return;
  int64_t key_pos = index % key_length;
  int64_t row = index / key_length;
  int64_t q_pos = row % query_length;
  int64_t hq = (row / query_length) % query_heads;
  int64_t b = row / (query_heads * query_length);
  int64_t kv_head = hq / (query_heads / kv_heads);
  const scalar_t* grad_row =
      grad_output + ((b * query_heads + hq) * query_length + q_pos) * dim;
  const scalar_t* v_row =
      v + ((b * kv_heads + kv_head) * key_length + key_pos) * dim;
  float accumulator = 0.0f;
  for (int64_t d = 0; d < dim; ++d) {
    accumulator += as_float(grad_row[d]) * as_float(v_row[d]);
  }
  dp[index] = accumulator;
}

__global__ void attention_softmax_backward_kernel(
    float* dp_ds,
    const float* probabilities,
    int64_t key_length) {
  int64_t row = blockIdx.x;
  float* ds_row = dp_ds + row * key_length;
  const float* p_row = probabilities + row * key_length;
  __shared__ float shared[kThreads];
  float local = 0.0f;
  for (int64_t key_pos = threadIdx.x; key_pos < key_length; key_pos += blockDim.x) {
    local += ds_row[key_pos] * p_row[key_pos];
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  float delta = shared[0];
  for (int64_t key_pos = threadIdx.x; key_pos < key_length; key_pos += blockDim.x) {
    ds_row[key_pos] = p_row[key_pos] * (ds_row[key_pos] - delta);
  }
}

template <typename scalar_t>
__global__ void attention_dq_ds_kernel(
    const float* ds,
    const scalar_t* k,
    scalar_t* grad_q,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim,
    float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * query_heads * query_length * dim) return;
  int64_t d = index % dim;
  int64_t q_pos = (index / dim) % query_length;
  int64_t hq = (index / (query_length * dim)) % query_heads;
  int64_t b = index / (query_heads * query_length * dim);
  int64_t kv_head = hq / (query_heads / kv_heads);
  int64_t row = (b * query_heads + hq) * query_length + q_pos;
  const scalar_t* k_base = k + (b * kv_heads + kv_head) * key_length * dim;
  float accumulator = 0.0f;
  for (int64_t key_pos = 0; key_pos < key_length; ++key_pos) {
    accumulator += ds[row * key_length + key_pos] *
        as_float(k_base[key_pos * dim + d]);
  }
  grad_q[index] = from_float<scalar_t>(scale * accumulator);
}

template <typename scalar_t>
__global__ void attention_dk_ds_kernel(
    const float* ds,
    const scalar_t* q,
    scalar_t* grad_k,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim,
    float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * kv_heads * key_length * dim) return;
  int64_t d = index % dim;
  int64_t key_pos = (index / dim) % key_length;
  int64_t kv_head = (index / (key_length * dim)) % kv_heads;
  int64_t b = index / (kv_heads * key_length * dim);
  int64_t group = query_heads / kv_heads;
  float accumulator = 0.0f;
  for (int64_t local = 0; local < group; ++local) {
    int64_t hq = kv_head * group + local;
    for (int64_t q_pos = 0; q_pos < query_length; ++q_pos) {
      float ds_value = ds[
          ((b * query_heads + hq) * query_length + q_pos) * key_length + key_pos];
      float q_value = as_float(
          q[((b * query_heads + hq) * query_length + q_pos) * dim + d]);
      accumulator += ds_value * q_value;
    }
  }
  grad_k[index] = from_float<scalar_t>(scale * accumulator);
}

template <typename scalar_t>
__global__ void attention_dv_p_kernel(
    const float* probabilities,
    const scalar_t* grad_output,
    scalar_t* grad_v,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * kv_heads * key_length * dim) return;
  int64_t d = index % dim;
  int64_t key_pos = (index / dim) % key_length;
  int64_t kv_head = (index / (key_length * dim)) % kv_heads;
  int64_t b = index / (kv_heads * key_length * dim);
  int64_t group = query_heads / kv_heads;
  float accumulator = 0.0f;
  for (int64_t local = 0; local < group; ++local) {
    int64_t hq = kv_head * group + local;
    for (int64_t q_pos = 0; q_pos < query_length; ++q_pos) {
      float probability = probabilities[
          ((b * query_heads + hq) * query_length + q_pos) * key_length + key_pos];
      float grad = as_float(
          grad_output[((b * query_heads + hq) * query_length + q_pos) * dim + d]);
      accumulator += probability * grad;
    }
  }
  grad_v[index] = from_float<scalar_t>(accumulator);
}

template <typename scalar_t>
__global__ void attention_backward_dkv_kernel(
    const scalar_t* grad_output,
    const scalar_t* q,
    const scalar_t* v,
    const float* probabilities,
    scalar_t* grad_k,
    scalar_t* grad_v,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t dim,
    float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= batch * kv_heads * key_length * dim) return;
  int64_t d = index % dim;
  int64_t key_pos = (index / dim) % key_length;
  int64_t kv_head = (index / (key_length * dim)) % kv_heads;
  int64_t b = index / (kv_heads * key_length * dim);
  float dk = 0.0f;
  float dv = 0.0f;
  int64_t group = query_heads / kv_heads;
  for (int64_t h = 0; h < group; ++h) {
    int64_t hq = kv_head * group + h;
    for (int64_t q_pos = 0; q_pos < query_length; ++q_pos) {
      int64_t row = (b * query_heads + hq) * query_length + q_pos;
      float probability = probabilities[row * key_length + key_pos];
      int64_t q_index = ((b * query_heads + hq) * query_length + q_pos) * dim;
      int64_t go_index = q_index;
      const scalar_t* v_row = v + ((b * kv_heads + kv_head) * key_length + key_pos) * dim;
      float dp = 0.0f;
      float sum_pd = 0.0f;
      for (int64_t j = 0; j < dim; ++j) {
        dp += as_float(grad_output[go_index + j]) * as_float(v_row[j]);
      }
      for (int64_t other = 0; other < key_length; ++other) {
        float p = probabilities[row * key_length + other];
        const scalar_t* other_v =
            v + ((b * kv_heads + kv_head) * key_length + other) * dim;
        float other_dp = 0.0f;
        for (int64_t j = 0; j < dim; ++j) {
          other_dp += as_float(grad_output[go_index + j]) * as_float(other_v[j]);
        }
        sum_pd += p * other_dp;
      }
      const scalar_t* q_row = q + q_index;
      dk += probability * (dp - sum_pd) * as_float(q_row[d]) * scale;
      dv += probability * as_float(grad_output[go_index + d]);
    }
  }
  grad_k[index] = from_float<scalar_t>(dk);
  grad_v[index] = from_float<scalar_t>(dv);
}

template <typename scalar_t>
__global__ void batch_invariant_logp_kernel(
    const scalar_t* logits,
    const int64_t* target,
    float* output,
    float* output_lse,
    int64_t rows,
    int64_t vocab,
    int64_t ignore_index) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* values = logits + row * vocab;
  __shared__ float tile_max_shared;
  __shared__ float tile_sum_shared;
  float row_max = -INFINITY;
  float row_sum = 0.0f;
  const int64_t v_aligned = (vocab / 256) * 256;
  for (int64_t tile_start = 0; tile_start < v_aligned; tile_start += kBatchInvariantTile) {
    const int64_t tile_end = min(
        tile_start + static_cast<int64_t>(kBatchInvariantTile), v_aligned);
    float local_max = -INFINITY;
    for (int64_t column = threadIdx.x + tile_start; column < tile_end;
         column += kBatchInvariantThreads) {
      local_max = fmaxf(local_max, as_float(values[column]));
    }
    const float tile_max = block_max_128(local_max);
    if (threadIdx.x == 0) tile_max_shared = tile_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int64_t column = threadIdx.x + tile_start; column < tile_end;
         column += kBatchInvariantThreads) {
      local_sum += expf(as_float(values[column]) - tile_max_shared);
    }
    const float tile_sum = block_sum_128(local_sum);
    if (threadIdx.x == 0) tile_sum_shared = tile_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
      const float new_max = fmaxf(row_max, tile_max_shared);
      row_sum = row_sum * expf(row_max - new_max) +
          tile_sum_shared * expf(tile_max_shared - new_max);
      row_max = new_max;
    }
    __syncthreads();
  }
  if (v_aligned < vocab) {
    float local_max = -INFINITY;
    for (int64_t column = v_aligned + threadIdx.x; column < vocab;
         column += kBatchInvariantThreads) {
      local_max = fmaxf(local_max, as_float(values[column]));
    }
    const float tail_max = block_max_128(local_max);
    if (threadIdx.x == 0) tile_max_shared = tail_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int64_t column = v_aligned + threadIdx.x; column < vocab;
         column += kBatchInvariantThreads) {
      local_sum += expf(as_float(values[column]) - tile_max_shared);
    }
    const float tail_sum = block_sum_128(local_sum);
    if (threadIdx.x == 0) tile_sum_shared = tail_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
      const float new_max = fmaxf(row_max, tile_max_shared);
      row_sum = row_sum * expf(row_max - new_max) +
          tile_sum_shared * expf(tile_max_shared - new_max);
      row_max = new_max;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float lse = row_max + logf(row_sum);
    const int64_t id = target[row];
    const bool valid = id != ignore_index && id >= 0 && id < vocab;
    output[row] = valid ? as_float(values[id]) - lse : 0.0f;
    output_lse[row] = lse;
  }
}

template <typename scalar_t>
__global__ void batch_invariant_logp_backward_kernel(
    const scalar_t* logits,
    const int64_t* target,
    const float* lse,
    const float* grad_output,
    scalar_t* grad_logits,
    int64_t rows,
    int64_t vocab,
    int64_t ignore_index) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (index >= rows * vocab) return;
  const int64_t row = index / vocab;
  const int64_t column = index % vocab;
  const int64_t id = target[row];
  const bool valid = id != ignore_index && id >= 0 && id < vocab;
  const float probability = expf(as_float(logits[index]) - lse[row]);
  const float indicator = valid && id == column ? 1.0f : 0.0f;
  const float value = valid ? grad_output[row] * (indicator - probability) : 0.0f;
  grad_logits[index] = from_float<scalar_t>(value);
}
template <typename scalar_t>
void launch_embedding_forward(
    const torch::Tensor& ids,
    const torch::Tensor& weight,
    torch::Tensor& output) {
  auto stream = at::musa::getCurrentMUSAStream();
  embedding_forward_kernel<scalar_t, scalar_t><<<blocks_for(ids.numel()), kThreads, 0, stream>>>(
      ids.data_ptr<int64_t>(),
      weight.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      ids.numel(),
      weight.size(1));
}

}  // namespace

std::vector<torch::Tensor> musa_embedding_forward(
    torch::Tensor token_ids,
    torch::Tensor weight) {
  check_same_device(token_ids, weight, "embedding");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "embedding ids must be int64");
  TORCH_CHECK(weight.dim() == 2, "embedding weight must be [V,H]");
  auto flat_ids = token_ids.contiguous().reshape({-1});
  auto output = torch::empty(
      {flat_ids.numel(), weight.size(1)}, weight.options());
  auto stream = at::musa::getCurrentMUSAStream();
  if (weight.scalar_type() == at::kBFloat16) {
    embedding_forward_kernel<c10::BFloat16, c10::BFloat16>
        <<<blocks_for(flat_ids.numel() * weight.size(1)), kThreads, 0, stream>>>(
            flat_ids.data_ptr<int64_t>(),
            weight.contiguous().data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(),
            flat_ids.numel(),
            weight.size(1));
  } else {
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "embedding supports BF16/FP32");
    embedding_forward_kernel<float, float>
        <<<blocks_for(flat_ids.numel() * weight.size(1)), kThreads, 0, stream>>>(
            flat_ids.data_ptr<int64_t>(),
            weight.contiguous().data_ptr<float>(),
            output.data_ptr<float>(),
            flat_ids.numel(),
            weight.size(1));
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  auto shape = token_ids.sizes().vec();
  shape.push_back(weight.size(1));
  return {output.reshape(shape)};
}

torch::Tensor musa_rowwise_gemm_fp32(
    torch::Tensor a,
    torch::Tensor b) {
  check_same_device(a, b, "rowwise gemm");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "rowwise gemm expects 2-D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "rowwise gemm K mismatch");
  auto a_float = a.to(torch::kFloat).contiguous();
  auto weight_float = b.transpose(0, 1).to(torch::kFloat).contiguous();
  auto output = torch::empty(
      {a.size(0), b.size(1)}, a.options().dtype(torch::kFloat));
  const int64_t total = a.size(0) * b.size(1);
  if (total == 0) return output;
  auto stream = at::musa::getCurrentMUSAStream();
  lm_head_forward_kernel<float, float>
      <<<static_cast<int>(total), kThreads, 0, stream>>>(
          a_float.data_ptr<float>(),
          weight_float.data_ptr<float>(),
          nullptr,
          output.data_ptr<float>(),
          a.size(0),
          a.size(1),
          b.size(1));
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor musa_lm_head_fp32(
    torch::Tensor hidden,
    torch::Tensor weight) {
  check_same_device(hidden, weight, "lm head");
  TORCH_CHECK(hidden.dim() >= 2 && weight.dim() == 2, "invalid lm head shapes");
  TORCH_CHECK(hidden.size(-1) == weight.size(1), "lm head hidden mismatch");
  auto hidden_2d = hidden.reshape({-1, hidden.size(-1)}).to(torch::kFloat).contiguous();
  auto weight_2d = weight.to(torch::kFloat).contiguous();
  auto output = torch::empty(
      {hidden_2d.size(0), weight.size(0)}, hidden.options().dtype(torch::kFloat));
  auto stream = at::musa::getCurrentMUSAStream();
  int64_t total = hidden_2d.size(0) * weight.size(0);
  lm_head_forward_kernel<float, float>
      <<<static_cast<int>(total), kThreads, 0, stream>>>(
          hidden_2d.data_ptr<float>(),
          weight_2d.data_ptr<float>(),
          nullptr,
          output.data_ptr<float>(),
          hidden_2d.size(0),
          hidden_2d.size(1),
          weight.size(0));
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  auto shape = hidden.sizes().vec();
  shape.back() = weight.size(0);
  return output.reshape(shape);
}

torch::Tensor musa_embedding_backward(
    torch::Tensor token_ids,
    torch::Tensor grad_output,
    int64_t vocab,
    int64_t hidden) {
  check_same_device(token_ids, grad_output, "embedding backward");
  auto ids = token_ids.contiguous().reshape({-1});
  auto grads = grad_output.contiguous().reshape({ids.numel(), hidden});
  auto result = torch::zeros({vocab, hidden}, grad_output.options());
  auto stream = at::musa::getCurrentMUSAStream();
  if (grad_output.scalar_type() == at::kBFloat16) {
    embedding_backward_kernel<c10::BFloat16>
        <<<blocks_for(vocab * hidden), kThreads, 0, stream>>>(
            ids.data_ptr<int64_t>(),
            grads.data_ptr<c10::BFloat16>(),
            result.data_ptr<c10::BFloat16>(),
            ids.numel(),
            vocab,
            hidden);
  } else {
    embedding_backward_kernel<float>
        <<<blocks_for(vocab * hidden), kThreads, 0, stream>>>(
            ids.data_ptr<int64_t>(),
            grads.data_ptr<float>(),
            result.data_ptr<float>(),
            ids.numel(),
            vocab,
            hidden);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return result;
}

std::vector<torch::Tensor> musa_rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps,
    bool output_fp32) {
  check_same_device(x, weight, "rmsnorm");
  TORCH_CHECK(x.dim() >= 2 && weight.dim() == 1, "invalid RMSNorm shapes");
  int64_t hidden = x.size(-1);
  int64_t rows = x.numel() / hidden;
  auto flat = x.contiguous().reshape({rows, hidden});
  auto w = weight.contiguous();
  auto output = output_fp32 ? torch::empty(flat.sizes(), x.options().dtype(torch::kFloat))
                            : torch::empty_like(flat);
  auto rstd = torch::empty({rows}, x.options().dtype(torch::kFloat));
  auto stream = at::musa::getCurrentMUSAStream();
  int threads = rmsnorm_threads(hidden);
  int shared_bytes = threads * static_cast<int>(sizeof(float));
  if (x.scalar_type() == at::kBFloat16) {
    if (weight.scalar_type() == at::kBFloat16) {
      if (output_fp32) {
        rmsnorm_forward_kernel<c10::BFloat16, c10::BFloat16, float>
          <<<rows, threads, shared_bytes, stream>>>(
              flat.data_ptr<c10::BFloat16>(), w.data_ptr<c10::BFloat16>(),
              output.data_ptr<float>(), rstd.data_ptr<float>(), rows, hidden, eps);
      } else {
        rmsnorm_forward_kernel<c10::BFloat16, c10::BFloat16, c10::BFloat16>
          <<<rows, threads, shared_bytes, stream>>>(
              flat.data_ptr<c10::BFloat16>(), w.data_ptr<c10::BFloat16>(),
              output.data_ptr<c10::BFloat16>(), rstd.data_ptr<float>(), rows, hidden, eps);
      }
    } else {
      TORCH_CHECK(weight.scalar_type() == at::kFloat, "RMSNorm weight must be BF16/FP32");
      if (output_fp32) {
        rmsnorm_forward_kernel<c10::BFloat16, float, float>
            <<<rows, threads, shared_bytes, stream>>>(
                flat.data_ptr<c10::BFloat16>(), w.data_ptr<float>(),
                output.data_ptr<float>(), rstd.data_ptr<float>(), rows, hidden, eps);
      } else {
        rmsnorm_forward_kernel<c10::BFloat16, float, c10::BFloat16>
            <<<rows, threads, shared_bytes, stream>>>(
                flat.data_ptr<c10::BFloat16>(), w.data_ptr<float>(),
                output.data_ptr<c10::BFloat16>(), rstd.data_ptr<float>(), rows, hidden, eps);
      }
    }
  } else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "RMSNorm supports BF16/FP32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "FP32 RMSNorm requires FP32 weight");
    rmsnorm_forward_kernel<float, float, float>
        <<<rows, threads, shared_bytes, stream>>>(
            flat.data_ptr<float>(), w.data_ptr<float>(), output.data_ptr<float>(),
            rstd.data_ptr<float>(), rows, hidden, eps);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return {output.reshape(x.sizes().vec()), rstd};
}

std::vector<torch::Tensor> musa_rmsnorm_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor rstd) {
  check_same_device(grad_output, x, "rmsnorm backward");
  int64_t hidden = x.size(-1);
  int64_t rows = x.numel() / hidden;
  auto flat_x = x.contiguous().reshape({rows, hidden});
  auto flat_grad = grad_output.contiguous().reshape({rows, hidden});
  auto dx = torch::empty_like(flat_x);
  auto dw = torch::empty_like(weight);
  auto stream = at::musa::getCurrentMUSAStream();
  int threads = rmsnorm_threads(hidden);
  int shared_bytes = threads * static_cast<int>(sizeof(float));
  if (x.scalar_type() == at::kBFloat16) {
    if (weight.scalar_type() == at::kBFloat16) {
      rmsnorm_backward_dx_kernel<c10::BFloat16, c10::BFloat16, c10::BFloat16>
          <<<rows, threads, shared_bytes, stream>>>(
              flat_x.data_ptr<c10::BFloat16>(), weight.data_ptr<c10::BFloat16>(),
              rstd.data_ptr<float>(), flat_grad.data_ptr<c10::BFloat16>(),
              dx.data_ptr<c10::BFloat16>(), rows, hidden);
      rmsnorm_backward_dw_kernel<c10::BFloat16, c10::BFloat16, c10::BFloat16>
          <<<blocks_for(hidden), kThreads, 0, stream>>>(
              flat_x.data_ptr<c10::BFloat16>(), rstd.data_ptr<float>(),
              flat_grad.data_ptr<c10::BFloat16>(), dw.data_ptr<c10::BFloat16>(), rows, hidden);
    } else {
      TORCH_CHECK(weight.scalar_type() == at::kFloat, "RMSNorm weight must be BF16/FP32");
      rmsnorm_backward_dx_kernel<c10::BFloat16, float, c10::BFloat16>
          <<<rows, threads, shared_bytes, stream>>>(
              flat_x.data_ptr<c10::BFloat16>(), weight.data_ptr<float>(),
              rstd.data_ptr<float>(), flat_grad.data_ptr<c10::BFloat16>(),
              dx.data_ptr<c10::BFloat16>(), rows, hidden);
      rmsnorm_backward_dw_kernel<c10::BFloat16, c10::BFloat16, float>
          <<<blocks_for(hidden), kThreads, 0, stream>>>(
              flat_x.data_ptr<c10::BFloat16>(), rstd.data_ptr<float>(),
              flat_grad.data_ptr<c10::BFloat16>(), dw.data_ptr<float>(), rows, hidden);
    }
  } else {
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "FP32 RMSNorm requires FP32 weight");
    rmsnorm_backward_dx_kernel<float, float, float>
        <<<rows, threads, shared_bytes, stream>>>(
            flat_x.data_ptr<float>(), weight.data_ptr<float>(), rstd.data_ptr<float>(),
            flat_grad.data_ptr<float>(), dx.data_ptr<float>(), rows, hidden);
    rmsnorm_backward_dw_kernel<float, float>
        <<<blocks_for(hidden), kThreads, 0, stream>>>(
            flat_x.data_ptr<float>(), rstd.data_ptr<float>(), flat_grad.data_ptr<float>(),
            dw.data_ptr<float>(), rows, hidden);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return {dx.reshape(x.sizes().vec()), dw};
}

torch::Tensor musa_rope_forward(
    torch::Tensor x,
    torch::Tensor positions,
    double theta) {
  check_same_device(x, positions, "rope");
  TORCH_CHECK(x.dim() == 4 && x.size(-1) % 2 == 0, "rope expects [B,H,S,D]");
  auto output = torch::empty_like(x);
  bool batched = positions.dim() == 2;
  auto stream = at::musa::getCurrentMUSAStream();
  int64_t batch = x.size(0), heads = x.size(1), sequence = x.size(2), dim = x.size(3);
  if (x.scalar_type() == at::kBFloat16) {
    rope_forward_kernel<c10::BFloat16>
        <<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.contiguous().data_ptr<c10::BFloat16>(), positions.contiguous().data_ptr<int64_t>(),
            output.data_ptr<c10::BFloat16>(), batch, heads, sequence, dim, theta, batched);
  } else {
    rope_forward_kernel<float>
        <<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.contiguous().data_ptr<float>(), positions.contiguous().data_ptr<int64_t>(),
            output.data_ptr<float>(), batch, heads, sequence, dim, theta, batched);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor musa_rope_apply(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    double sin_sign) {
  check_same_device(x, cos, "rope apply");
  check_same_device(x, sin, "rope apply");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "rope apply expects contiguous [rows,D]");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
              "rope cos/sin must be FP32");
  TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes(), "invalid rope tables");
  int64_t rows = x.size(0);
  int half = static_cast<int>(x.size(1) / 2);
  int sequence = static_cast<int>(cos.size(0));
  TORCH_CHECK(cos.size(1) == half && rows % sequence == 0, "rope table shape mismatch");
  auto output = torch::empty_like(x);
  auto stream = at::musa::getCurrentMUSAStream();
  int64_t total = rows * half;
  if (x.scalar_type() == at::kBFloat16) {
    rope_apply_kernel<c10::BFloat16>
        <<<blocks_for(total), kThreads, 0, stream>>>(
            x.data_ptr<c10::BFloat16>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
            output.data_ptr<c10::BFloat16>(), rows, sequence, half,
            static_cast<float>(sin_sign));
  } else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "rope apply supports BF16/FP32");
    rope_apply_kernel<float>
        <<<blocks_for(total), kThreads, 0, stream>>>(
            x.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
            output.data_ptr<float>(), rows, sequence, half,
            static_cast<float>(sin_sign));
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor musa_rope_backward(
    torch::Tensor grad_output,
    torch::Tensor positions,
    double theta) {
  check_same_device(grad_output, positions, "rope backward");
  auto output = torch::empty_like(grad_output);
  bool batched = positions.dim() == 2;
  auto stream = at::musa::getCurrentMUSAStream();
  int64_t batch = grad_output.size(0);
  int64_t heads = grad_output.size(1);
  int64_t sequence = grad_output.size(2);
  int64_t dim = grad_output.size(3);
  if (grad_output.scalar_type() == at::kBFloat16) {
    rope_backward_kernel<c10::BFloat16>
        <<<blocks_for(grad_output.numel()), kThreads, 0, stream>>>(
            grad_output.contiguous().data_ptr<c10::BFloat16>(),
            positions.contiguous().data_ptr<int64_t>(),
            output.data_ptr<c10::BFloat16>(),
            batch, heads, sequence, dim, theta, batched);
  } else {
    rope_backward_kernel<float>
        <<<blocks_for(grad_output.numel()), kThreads, 0, stream>>>(
            grad_output.contiguous().data_ptr<float>(),
            positions.contiguous().data_ptr<int64_t>(),
            output.data_ptr<float>(),
            batch, heads, sequence, dim, theta, batched);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor musa_silu_forward(torch::Tensor x) {
  check_musa(x, "silu");
  auto output = torch::empty_like(x);
  auto stream = at::musa::getCurrentMUSAStream();
  if (x.scalar_type() == at::kBFloat16) {
    silu_forward_kernel<c10::BFloat16>
        <<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.contiguous().data_ptr<c10::BFloat16>(), output.data_ptr<c10::BFloat16>(), x.numel());
  } else {
    silu_forward_kernel<float>
        <<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.contiguous().data_ptr<float>(), output.data_ptr<float>(), x.numel());
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor musa_silu_backward(torch::Tensor grad_output, torch::Tensor x) {
  check_same_device(grad_output, x, "silu backward");
  auto output = torch::empty_like(x);
  auto stream = at::musa::getCurrentMUSAStream();
  if (x.scalar_type() == at::kBFloat16) {
    silu_backward_kernel<c10::BFloat16>
        <<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.contiguous().data_ptr<c10::BFloat16>(),
            grad_output.contiguous().data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(), x.numel());
  } else {
    silu_backward_kernel<float>
        <<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.contiguous().data_ptr<float>(), grad_output.contiguous().data_ptr<float>(),
            output.data_ptr<float>(), x.numel());
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor musa_swiglu_forward(torch::Tensor gate, torch::Tensor up) {
  check_same_device(gate, up, "swiglu");
  TORCH_CHECK(gate.sizes() == up.sizes(), "SwiGLU inputs must share shape");
  auto output = torch::empty_like(gate);
  auto stream = at::musa::getCurrentMUSAStream();
  if (gate.scalar_type() == at::kBFloat16) {
    swiglu_forward_kernel<c10::BFloat16>
        <<<blocks_for(gate.numel()), kThreads, 0, stream>>>(
            gate.contiguous().data_ptr<c10::BFloat16>(),
            up.contiguous().data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(), gate.numel());
  } else {
    swiglu_forward_kernel<float>
        <<<blocks_for(gate.numel()), kThreads, 0, stream>>>(
            gate.contiguous().data_ptr<float>(), up.contiguous().data_ptr<float>(),
            output.data_ptr<float>(), gate.numel());
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> musa_swiglu_backward(
    torch::Tensor grad_output,
    torch::Tensor gate,
    torch::Tensor up) {
  check_same_device(grad_output, gate, "swiglu backward");
  auto grad_gate = torch::empty_like(gate);
  auto grad_up = torch::empty_like(up);
  auto stream = at::musa::getCurrentMUSAStream();
  if (gate.scalar_type() == at::kBFloat16) {
    swiglu_backward_kernel<c10::BFloat16>
        <<<blocks_for(gate.numel()), kThreads, 0, stream>>>(
            gate.contiguous().data_ptr<c10::BFloat16>(),
            up.contiguous().data_ptr<c10::BFloat16>(),
            grad_output.contiguous().data_ptr<c10::BFloat16>(),
            grad_gate.data_ptr<c10::BFloat16>(), grad_up.data_ptr<c10::BFloat16>(), gate.numel());
  } else {
    swiglu_backward_kernel<float>
        <<<blocks_for(gate.numel()), kThreads, 0, stream>>>(
            gate.contiguous().data_ptr<float>(), up.contiguous().data_ptr<float>(),
            grad_output.contiguous().data_ptr<float>(),
            grad_gate.data_ptr<float>(), grad_up.data_ptr<float>(), gate.numel());
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return {grad_gate, grad_up};
}

torch::Tensor musa_selected_logp_forward(
    torch::Tensor logits,
    torch::Tensor target,
    int64_t ignore_index) {
  check_same_device(logits, target, "logp");
  auto flat = logits.contiguous().reshape({-1, logits.size(-1)});
  auto ids = target.contiguous().reshape({-1});
  auto output = torch::empty({flat.size(0)}, logits.options());
  auto stream = at::musa::getCurrentMUSAStream();
  if (logits.scalar_type() == at::kBFloat16) {
    if (flat.size(1) <= 128) {
      selected_logp_forward_kernel<c10::BFloat16, 128>
          <<<flat.size(0), 128, 0, stream>>>(
              flat.data_ptr<c10::BFloat16>(), ids.data_ptr<int64_t>(),
              output.data_ptr<c10::BFloat16>(), flat.size(0), flat.size(1), ignore_index);
    } else if (flat.size(1) <= 4096) {
      selected_logp_forward_kernel<c10::BFloat16, 256>
          <<<flat.size(0), 256, 0, stream>>>(
              flat.data_ptr<c10::BFloat16>(), ids.data_ptr<int64_t>(),
              output.data_ptr<c10::BFloat16>(), flat.size(0), flat.size(1), ignore_index);
    } else {
      selected_logp_forward_kernel<c10::BFloat16, 512>
          <<<flat.size(0), 512, 0, stream>>>(
              flat.data_ptr<c10::BFloat16>(), ids.data_ptr<int64_t>(),
              output.data_ptr<c10::BFloat16>(), flat.size(0), flat.size(1), ignore_index);
    }
  } else {
    if (flat.size(1) <= 128) {
      selected_logp_forward_kernel<float, 128>
          <<<flat.size(0), 128, 0, stream>>>(
              flat.data_ptr<float>(), ids.data_ptr<int64_t>(), output.data_ptr<float>(),
              flat.size(0), flat.size(1), ignore_index);
    } else if (flat.size(1) <= 4096) {
      selected_logp_forward_kernel<float, 256>
          <<<flat.size(0), 256, 0, stream>>>(
              flat.data_ptr<float>(), ids.data_ptr<int64_t>(), output.data_ptr<float>(),
              flat.size(0), flat.size(1), ignore_index);
    } else {
      selected_logp_forward_kernel<float, 512>
          <<<flat.size(0), 512, 0, stream>>>(
              flat.data_ptr<float>(), ids.data_ptr<int64_t>(), output.data_ptr<float>(),
              flat.size(0), flat.size(1), ignore_index);
    }
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  auto shape = logits.sizes().vec();
  shape.pop_back();
  return output.reshape(shape);
}

torch::Tensor musa_selected_logp_backward(
    torch::Tensor grad_output,
    torch::Tensor logits,
    torch::Tensor target,
    int64_t ignore_index) {
  check_same_device(grad_output, logits, "logp backward");
  auto flat = logits.contiguous().reshape({-1, logits.size(-1)});
  auto ids = target.contiguous().reshape({-1});
  auto grad = grad_output.contiguous().reshape({-1});
  auto output = torch::empty_like(flat);
  auto stream = at::musa::getCurrentMUSAStream();
  if (logits.scalar_type() == at::kBFloat16) {
    selected_logp_backward_kernel<c10::BFloat16>
        <<<blocks_for(flat.numel()), kThreads, 0, stream>>>(
            flat.data_ptr<c10::BFloat16>(), ids.data_ptr<int64_t>(),
            grad.data_ptr<c10::BFloat16>(), output.data_ptr<c10::BFloat16>(),
            flat.size(0), flat.size(1), ignore_index);
  } else {
    selected_logp_backward_kernel<float>
        <<<blocks_for(flat.numel()), kThreads, 0, stream>>>(
            flat.data_ptr<float>(), ids.data_ptr<int64_t>(), grad.data_ptr<float>(),
            output.data_ptr<float>(), flat.size(0), flat.size(1), ignore_index);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output.reshape(logits.sizes().vec());
}

std::vector<torch::Tensor> musa_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double scale,
    torch::optional<torch::Tensor> mask,
    bool output_fp32) {
  check_same_device(q, k, "attention");
  check_same_device(q, v, "attention");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "attention expects 4-D tensors");
  TORCH_CHECK(q.size(3) == kHeadDim, "attention head dim must be 128");
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto probs = torch::empty(
      {q.size(0), q.size(1), q.size(2), k.size(2)},
      q.options().dtype(torch::kFloat));
  auto lse = torch::empty(
      {q.size(0), q.size(1), q.size(2)},
      q.options().dtype(torch::kFloat));
  auto output = output_fp32 ? torch::empty(q.sizes(), q.options().dtype(torch::kFloat))
                            : torch::empty_like(q);
  const bool* mask_ptr = nullptr;
  auto mask_c = mask.has_value() ? torch::optional<torch::Tensor>(mask->contiguous())
                                 : torch::optional<torch::Tensor>();
  if (mask_c.has_value()) mask_ptr = mask_c->data_ptr<bool>();
  auto stream = at::musa::getCurrentMUSAStream();
  int64_t rows = q.size(0) * q.size(1) * q.size(2);
  int64_t score_elements = rows * k.size(2);
  if (q.scalar_type() == at::kBFloat16) {
    attention_qk_kernel<c10::BFloat16>
        <<<blocks_for(score_elements), kThreads, 0, stream>>>(
            q_c.data_ptr<c10::BFloat16>(), k_c.data_ptr<c10::BFloat16>(),
            probs.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3),
            static_cast<float>(scale));
    if (k.size(2) <= 128) {
      attention_softmax_serial_kernel<<<rows, kThreads, 0, stream>>>(
          probs.data_ptr<float>(), lse.data_ptr<float>(), mask_ptr,
          q.size(1), q.size(2), k.size(2), causal);
    } else {
      attention_softmax_kernel<<<rows, kThreads, 0, stream>>>(
          probs.data_ptr<float>(), lse.data_ptr<float>(), mask_ptr,
          q.size(1), q.size(2), k.size(2), causal);
    }
    if (output_fp32) {
      attention_pv_kernel<c10::BFloat16, float>
          <<<blocks_for(q.numel()), kThreads, 0, stream>>>(
              probs.data_ptr<float>(), v_c.data_ptr<c10::BFloat16>(),
              output.data_ptr<float>(), q.size(0), q.size(1), k.size(1),
              q.size(2), k.size(2), q.size(3));
    } else {
      attention_pv_kernel<c10::BFloat16, c10::BFloat16>
          <<<blocks_for(q.numel()), kThreads, 0, stream>>>(
              probs.data_ptr<float>(), v_c.data_ptr<c10::BFloat16>(),
              output.data_ptr<c10::BFloat16>(), q.size(0), q.size(1), k.size(1),
              q.size(2), k.size(2), q.size(3));
    }
  } else {
    TORCH_CHECK(q.scalar_type() == at::kFloat, "attention supports BF16/FP32");
    attention_qk_kernel<float>
        <<<blocks_for(score_elements), kThreads, 0, stream>>>(
            q_c.data_ptr<float>(), k_c.data_ptr<float>(),
            probs.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3),
            static_cast<float>(scale));
    if (k.size(2) <= 128) {
      attention_softmax_serial_kernel<<<rows, kThreads, 0, stream>>>(
          probs.data_ptr<float>(), lse.data_ptr<float>(), mask_ptr,
          q.size(1), q.size(2), k.size(2), causal);
    } else {
      attention_softmax_kernel<<<rows, kThreads, 0, stream>>>(
          probs.data_ptr<float>(), lse.data_ptr<float>(), mask_ptr,
          q.size(1), q.size(2), k.size(2), causal);
    }
    attention_pv_kernel<float, float>
        <<<blocks_for(q.numel()), kThreads, 0, stream>>>(
            probs.data_ptr<float>(), v_c.data_ptr<float>(), output.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3));
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return {output, lse, probs};
}

std::vector<torch::Tensor> musa_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor probs,
    double scale) {
  auto grad = grad_output.contiguous();
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto probs_c = probs.contiguous();
  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  auto ds = torch::empty(
      {q.size(0), q.size(1), q.size(2), k.size(2)},
      q.options().dtype(torch::kFloat));
  auto stream = at::musa::getCurrentMUSAStream();
  int64_t rows = q.size(0) * q.size(1) * q.size(2);
  int64_t score_elements = rows * k.size(2);
  if (q.scalar_type() == at::kBFloat16) {
    attention_dp_kernel<c10::BFloat16>
        <<<blocks_for(score_elements), kThreads, 0, stream>>>(
            grad.data_ptr<c10::BFloat16>(), v_c.data_ptr<c10::BFloat16>(),
            ds.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3));
    attention_softmax_backward_kernel<<<rows, kThreads, 0, stream>>>(
        ds.data_ptr<float>(), probs_c.data_ptr<float>(), k.size(2));
    attention_dq_ds_kernel<c10::BFloat16>
        <<<blocks_for(q.numel()), kThreads, 0, stream>>>(
            ds.data_ptr<float>(), k_c.data_ptr<c10::BFloat16>(),
            dq.data_ptr<c10::BFloat16>(), q.size(0), q.size(1), k.size(1),
            q.size(2), k.size(2), q.size(3), static_cast<float>(scale));
    attention_dk_ds_kernel<c10::BFloat16>
        <<<blocks_for(k.numel()), kThreads, 0, stream>>>(
            ds.data_ptr<float>(), q_c.data_ptr<c10::BFloat16>(),
            dk.data_ptr<c10::BFloat16>(), q.size(0), q.size(1), k.size(1),
            q.size(2), k.size(2), q.size(3), static_cast<float>(scale));
    attention_dv_p_kernel<c10::BFloat16>
        <<<blocks_for(v.numel()), kThreads, 0, stream>>>(
            probs_c.data_ptr<float>(), grad.data_ptr<c10::BFloat16>(),
            dv.data_ptr<c10::BFloat16>(), q.size(0), q.size(1), k.size(1),
            q.size(2), k.size(2), q.size(3));
  } else {
    TORCH_CHECK(q.scalar_type() == at::kFloat, "attention supports BF16/FP32");
    attention_dp_kernel<float>
        <<<blocks_for(score_elements), kThreads, 0, stream>>>(
            grad.data_ptr<float>(), v_c.data_ptr<float>(), ds.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3));
    attention_softmax_backward_kernel<<<rows, kThreads, 0, stream>>>(
        ds.data_ptr<float>(), probs_c.data_ptr<float>(), k.size(2));
    attention_dq_ds_kernel<float>
        <<<blocks_for(q.numel()), kThreads, 0, stream>>>(
            ds.data_ptr<float>(), k_c.data_ptr<float>(), dq.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3),
            static_cast<float>(scale));
    attention_dk_ds_kernel<float>
        <<<blocks_for(k.numel()), kThreads, 0, stream>>>(
            ds.data_ptr<float>(), q_c.data_ptr<float>(), dk.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3),
            static_cast<float>(scale));
    attention_dv_p_kernel<float>
        <<<blocks_for(v.numel()), kThreads, 0, stream>>>(
            probs_c.data_ptr<float>(), grad.data_ptr<float>(), dv.data_ptr<float>(),
            q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3));
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  ds.reset();
  return {dq, dk, dv};
}
std::vector<torch::Tensor> musa_batch_invariant_logp_forward(
    torch::Tensor logits,
    torch::Tensor target,
    int64_t ignore_index) {
  check_same_device(logits, target, "batch invariant logp");
  TORCH_CHECK(logits.dim() >= 2, "batch invariant logp expects [..., V] logits");
  TORCH_CHECK(target.numel() == logits.numel() / logits.size(-1),
              "target must have one id per logits row");
  TORCH_CHECK(target.scalar_type() == at::kLong, "target must be int64");
  TORCH_CHECK(
      logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kFloat,
      "batch invariant logp supports BF16/FP32 logits");
  auto flat = logits.contiguous().reshape({-1, logits.size(-1)});
  auto ids = target.contiguous().reshape({-1});
  auto logp = torch::empty({flat.size(0)}, logits.options().dtype(torch::kFloat));
  auto lse = torch::empty_like(logp);
  if (flat.size(0) == 0) {
    auto shape = target.sizes().vec();
    return {logp.reshape(shape), lse.reshape(shape)};
  }
  auto stream = at::musa::getCurrentMUSAStream();
  if (logits.scalar_type() == at::kBFloat16) {
    batch_invariant_logp_kernel<c10::BFloat16>
        <<<flat.size(0), kBatchInvariantThreads, 0, stream>>>(
            flat.data_ptr<c10::BFloat16>(), ids.data_ptr<int64_t>(),
            logp.data_ptr<float>(), lse.data_ptr<float>(),
            flat.size(0), flat.size(1), ignore_index);
  } else {
    batch_invariant_logp_kernel<float>
        <<<flat.size(0), kBatchInvariantThreads, 0, stream>>>(
            flat.data_ptr<float>(), ids.data_ptr<int64_t>(),
            logp.data_ptr<float>(), lse.data_ptr<float>(),
            flat.size(0), flat.size(1), ignore_index);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  auto shape = target.sizes().vec();
  return {logp.reshape(shape), lse.reshape(shape)};
}

torch::Tensor musa_batch_invariant_logp_backward(
    torch::Tensor grad_output,
    torch::Tensor logits,
    torch::Tensor target,
    torch::Tensor lse,
    int64_t ignore_index) {
  check_same_device(grad_output, logits, "batch invariant logp backward");
  check_same_device(target, logits, "batch invariant logp backward");
  check_same_device(lse, logits, "batch invariant logp backward");
  auto flat = logits.contiguous().reshape({-1, logits.size(-1)});
  auto ids = target.contiguous().reshape({-1});
  auto lse_flat = lse.contiguous().reshape({-1});
  auto grad = grad_output.contiguous().reshape({-1}).to(torch::kFloat);
  TORCH_CHECK(ids.numel() == flat.size(0) && lse_flat.numel() == flat.size(0),
              "batch invariant logp backward row count mismatch");
  auto output = torch::empty_like(flat);
  if (flat.size(0) == 0) return output.reshape(logits.sizes().vec());
  auto stream = at::musa::getCurrentMUSAStream();
  const int64_t total = flat.size(0) * flat.size(1);
  if (logits.scalar_type() == at::kBFloat16) {
    batch_invariant_logp_backward_kernel<c10::BFloat16>
        <<<blocks_for(total), kThreads, 0, stream>>>(
            flat.data_ptr<c10::BFloat16>(), ids.data_ptr<int64_t>(),
            lse_flat.data_ptr<float>(), grad.data_ptr<float>(),
            output.data_ptr<c10::BFloat16>(), flat.size(0), flat.size(1), ignore_index);
  } else {
    batch_invariant_logp_backward_kernel<float>
        <<<blocks_for(total), kThreads, 0, stream>>>(
            flat.data_ptr<float>(), ids.data_ptr<int64_t>(),
            lse_flat.data_ptr<float>(), grad.data_ptr<float>(),
            output.data_ptr<float>(), flat.size(0), flat.size(1), ignore_index);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output.reshape(logits.sizes().vec());
}
