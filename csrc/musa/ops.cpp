// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

std::vector<torch::Tensor> ratio_kl_musa_forward_impl(torch::Tensor policy,
                                                      torch::Tensor reference,
                                                      torch::Tensor action,
                                                      torch::Tensor mask,
                                                      torch::Tensor old_logp);
torch::Tensor
ratio_kl_musa_backward_impl(torch::Tensor policy, torch::Tensor action,
                            torch::Tensor mask, torch::Tensor ratio,
                            torch::Tensor diff, torch::Tensor policy_logz,
                            torch::Tensor grad_ratio, torch::Tensor grad_kl);

void check_forward_inputs(const torch::Tensor &policy,
                          const torch::Tensor &reference,
                          const torch::Tensor &action,
                          const torch::Tensor &mask,
                          const torch::Tensor &old_logp) {
  TORCH_CHECK(policy.device().type() == c10::kPrivateUse1,
              "policy logits must be a MUSA tensor");
  TORCH_CHECK(reference.device() == policy.device() &&
                  action.device() == policy.device() &&
                  mask.device() == policy.device() &&
                  old_logp.device() == policy.device(),
              "all ratio_kl tensors must share the same MUSA device");
  TORCH_CHECK(
      policy.dim() == 2 && reference.sizes() == policy.sizes(),
      "policy and reference logits must have matching [rows, vocab] shapes");
  TORCH_CHECK(policy.scalar_type() == reference.scalar_type() &&
                  (policy.scalar_type() == at::ScalarType::Float ||
                   policy.scalar_type() == at::ScalarType::Half ||
                   policy.scalar_type() == at::ScalarType::BFloat16),
              "policy and reference logits must have matching FP32, FP16, or "
              "BF16 dtypes");
  TORCH_CHECK(action.dim() == 1 && mask.dim() == 1 && old_logp.dim() == 1 &&
                  action.size(0) == policy.size(0) &&
                  mask.size(0) == policy.size(0) &&
                  old_logp.size(0) == policy.size(0),
              "action, mask, and old_logp must have one value per logits row");
  TORCH_CHECK(action.scalar_type() == at::ScalarType::Long,
              "action must be int64");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Int, "mask must be int32");
  TORCH_CHECK(old_logp.scalar_type() == at::ScalarType::Float,
              "old_logp must be FP32");
  TORCH_CHECK(policy.size(1) > 0, "vocabulary dimension must be non-empty");
}

std::vector<torch::Tensor> ratio_kl_musa_forward(torch::Tensor policy,
                                                 torch::Tensor reference,
                                                 torch::Tensor action,
                                                 torch::Tensor mask,
                                                 torch::Tensor old_logp) {
  check_forward_inputs(policy, reference, action, mask, old_logp);
  return ratio_kl_musa_forward_impl(policy.contiguous(), reference.contiguous(),
                                    action.contiguous(), mask.contiguous(),
                                    old_logp.contiguous());
}

torch::Tensor ratio_kl_musa_backward(torch::Tensor policy, torch::Tensor action,
                                     torch::Tensor mask, torch::Tensor ratio,
                                     torch::Tensor diff,
                                     torch::Tensor policy_logz,
                                     torch::Tensor grad_ratio,
                                     torch::Tensor grad_kl) {
  TORCH_CHECK(policy.device().type() == c10::kPrivateUse1 && policy.dim() == 2,
              "policy logits must be a 2-D MUSA tensor");
  TORCH_CHECK(policy.scalar_type() == at::ScalarType::Float ||
                  policy.scalar_type() == at::ScalarType::Half ||
                  policy.scalar_type() == at::ScalarType::BFloat16,
              "policy logits must be FP32, FP16, or BF16");
  TORCH_CHECK(
      action.device() == policy.device() && mask.device() == policy.device() &&
          action.dim() == 1 && mask.dim() == 1 &&
          action.size(0) == policy.size(0) && mask.size(0) == policy.size(0),
      "action and mask must have one value per policy row on the same device");
  TORCH_CHECK(action.scalar_type() == at::ScalarType::Long,
              "action must be int64");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Int, "mask must be int32");
  for (const auto &tensor : {ratio, diff, policy_logz, grad_ratio, grad_kl}) {
    TORCH_CHECK(tensor.device() == policy.device() && tensor.dim() == 1 &&
                    tensor.size(0) == policy.size(0) &&
                    tensor.scalar_type() == at::ScalarType::Float,
                "saved states and upstream gradients must be per-row FP32 MUSA "
                "tensors");
  }
  return ratio_kl_musa_backward_impl(
      policy.contiguous(), action.contiguous(), mask.contiguous(),
      ratio.contiguous(), diff.contiguous(), policy_logz.contiguous(),
      grad_ratio.contiguous(), grad_kl.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ratio_kl_musa_forward", &ratio_kl_musa_forward);
  m.def("ratio_kl_musa_backward", &ratio_kl_musa_backward);
}
