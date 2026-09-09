// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b);
torch::Tensor det_gemm_fwd_fp32(torch::Tensor a, torch::Tensor b);
torch::Tensor det_gemm_fwd_rhs_transposed(torch::Tensor a, torch::Tensor bt);
torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b);
torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc);
torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc);
std::vector<torch::Tensor> fused_linear_logp_musa_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias);
std::vector<torch::Tensor> fused_linear_logp_musa_backward(
    torch::Tensor grad_logp,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::Tensor lse,
    torch::optional<torch::Tensor> bias,
    bool compute_grad_hidden,
    bool compute_grad_weight,
    bool compute_grad_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("det_gemm_fwd", &det_gemm_fwd);
    m.def("det_gemm_fwd_fp32", &det_gemm_fwd_fp32);
    m.def("det_gemm_fwd_rhs_transposed", &det_gemm_fwd_rhs_transposed);
    m.def("det_gemm_da", &det_gemm_da);
    m.def("det_gemm_db", &det_gemm_db);
    m.def("det_gemm_db_transposed", &det_gemm_db_transposed);
    m.def("fused_linear_logp_musa_forward", &fused_linear_logp_musa_forward,
          py::arg("hidden"), py::arg("weight"), py::arg("target"),
          py::arg("bias") = py::none());
    m.def("fused_linear_logp_musa_backward", &fused_linear_logp_musa_backward,
          py::arg("grad_logp"), py::arg("hidden"), py::arg("weight"),
          py::arg("target"), py::arg("lse"), py::arg("bias") = py::none(),
          py::arg("compute_grad_hidden") = true,
          py::arg("compute_grad_weight") = true,
          py::arg("compute_grad_bias") = true);
}
