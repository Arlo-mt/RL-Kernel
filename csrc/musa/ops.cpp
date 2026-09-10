// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

std::vector<torch::Tensor> fused_linear_ratio_kl_musa_forward(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor target,
    torch::Tensor mask, torch::Tensor old_logp, torch::Tensor reference_logp,
    torch::optional<torch::Tensor> bias);
std::vector<torch::Tensor> fused_linear_ratio_kl_musa_backward(
    torch::Tensor grad_ratio, torch::Tensor grad_kl, torch::Tensor hidden,
    torch::Tensor weight, torch::Tensor target, torch::Tensor mask,
    torch::Tensor ratio, torch::Tensor diff, torch::Tensor policy_lse,
    torch::optional<torch::Tensor> bias, bool compute_grad_hidden,
    bool compute_grad_weight, bool compute_grad_bias);

void check_common_inputs(const torch::Tensor &hidden,
                         const torch::Tensor &weight,
                         const torch::Tensor &target, const torch::Tensor &mask,
                         const torch::optional<torch::Tensor> &bias) {
  TORCH_CHECK(hidden.device().type() == c10::kPrivateUse1 &&
                  weight.device().type() == c10::kPrivateUse1 &&
                  target.device().type() == c10::kPrivateUse1 &&
                  mask.device().type() == c10::kPrivateUse1,
              "linear_ratio_kl requires MUSA tensors");
  TORCH_CHECK(hidden.device() == weight.device() &&
                  hidden.device() == target.device() &&
                  hidden.device() == mask.device(),
              "linear_ratio_kl tensors must share one MUSA device");
  TORCH_CHECK(hidden.dim() == 2 && weight.dim() == 2 && target.dim() == 1 &&
                  mask.dim() == 1,
              "expected hidden [N,H], weight [V,H], target [N], and mask [N]");
  TORCH_CHECK(hidden.scalar_type() == at::ScalarType::BFloat16 &&
                  weight.scalar_type() == at::ScalarType::BFloat16,
              "MUSA linear_ratio_kl currently supports BF16 hidden and weight");
  TORCH_CHECK(target.scalar_type() == at::ScalarType::Long,
              "target must be int64");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Int, "mask must be int32");
  TORCH_CHECK(hidden.size(1) == weight.size(1) &&
                  hidden.size(0) == target.size(0) &&
                  hidden.size(0) == mask.size(0),
              "linear_ratio_kl input shapes do not match");
  TORCH_CHECK(weight.size(0) > 0, "vocabulary dimension must be non-empty");
  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == hidden.device() &&
                    bias->scalar_type() == hidden.scalar_type() &&
                    bias->dim() == 1 && bias->size(0) == weight.size(0),
                "bias must be a MUSA BF16 tensor with shape [V]");
  }
  if (target.numel() > 0) {
    TORCH_CHECK(target.min().item<int64_t>() >= 0 &&
                    target.max().item<int64_t>() < weight.size(0),
                "target ids must be within the vocabulary range");
  }
}

void check_row_float(const torch::Tensor &tensor, const torch::Tensor &hidden,
                     const char *name) {
  TORCH_CHECK(tensor.device() == hidden.device() && tensor.dim() == 1 &&
                  tensor.size(0) == hidden.size(0) &&
                  tensor.scalar_type() == at::ScalarType::Float,
              name, " must be an FP32 MUSA tensor with one value per row");
}

std::vector<torch::Tensor>
checked_forward(torch::Tensor hidden, torch::Tensor weight,
                torch::Tensor target, torch::Tensor mask,
                torch::Tensor old_logp, torch::Tensor reference_logp,
                torch::optional<torch::Tensor> bias) {
  check_common_inputs(hidden, weight, target, mask, bias);
  check_row_float(old_logp, hidden, "old_logp");
  check_row_float(reference_logp, hidden, "reference_logp");
  return fused_linear_ratio_kl_musa_forward(
      hidden.contiguous(), weight.contiguous(), target.contiguous(),
      mask.contiguous(), old_logp.contiguous(), reference_logp.contiguous(),
      bias.has_value() ? torch::optional<torch::Tensor>(bias->contiguous())
                       : torch::optional<torch::Tensor>());
}

std::vector<torch::Tensor>
checked_backward(torch::Tensor grad_ratio, torch::Tensor grad_kl,
                 torch::Tensor hidden, torch::Tensor weight,
                 torch::Tensor target, torch::Tensor mask, torch::Tensor ratio,
                 torch::Tensor diff, torch::Tensor policy_lse,
                 torch::optional<torch::Tensor> bias, bool compute_grad_hidden,
                 bool compute_grad_weight, bool compute_grad_bias) {
  check_common_inputs(hidden, weight, target, mask, bias);
  check_row_float(grad_ratio, hidden, "grad_ratio");
  check_row_float(grad_kl, hidden, "grad_kl");
  check_row_float(ratio, hidden, "ratio");
  check_row_float(diff, hidden, "diff");
  check_row_float(policy_lse, hidden, "policy_lse");
  return fused_linear_ratio_kl_musa_backward(
      grad_ratio.contiguous(), grad_kl.contiguous(), hidden.contiguous(),
      weight.contiguous(), target.contiguous(), mask.contiguous(),
      ratio.contiguous(), diff.contiguous(), policy_lse.contiguous(),
      bias.has_value() ? torch::optional<torch::Tensor>(bias->contiguous())
                       : torch::optional<torch::Tensor>(),
      compute_grad_hidden, compute_grad_weight, compute_grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_linear_ratio_kl_musa_forward", &checked_forward,
        py::arg("hidden"), py::arg("weight"), py::arg("target"),
        py::arg("mask"), py::arg("old_logp"), py::arg("reference_logp"),
        py::arg("bias") = py::none());
  m.def("fused_linear_ratio_kl_musa_backward", &checked_backward,
        py::arg("grad_ratio"), py::arg("grad_kl"), py::arg("hidden"),
        py::arg("weight"), py::arg("target"), py::arg("mask"), py::arg("ratio"),
        py::arg("diff"), py::arg("policy_lse"), py::arg("bias") = py::none(),
        py::arg("compute_grad_hidden") = true,
        py::arg("compute_grad_weight") = true,
        py::arg("compute_grad_bias") = true);
}
