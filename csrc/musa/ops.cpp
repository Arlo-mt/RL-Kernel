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
std::vector<torch::Tensor> musa_embedding_forward(
    torch::Tensor token_ids, torch::Tensor weight);
torch::Tensor musa_rowwise_gemm_fp32(torch::Tensor a, torch::Tensor b);
torch::Tensor musa_lm_head_fp32(torch::Tensor hidden, torch::Tensor weight);
torch::Tensor musa_embedding_backward(
    torch::Tensor token_ids, torch::Tensor grad_output,
    int64_t vocab, int64_t hidden);
std::vector<torch::Tensor> musa_rmsnorm_forward(
    torch::Tensor x, torch::Tensor weight, double eps, bool output_fp32);
std::vector<torch::Tensor> musa_rmsnorm_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor weight, torch::Tensor rstd);
torch::Tensor musa_rope_forward(
    torch::Tensor x, torch::Tensor positions, double theta);
torch::Tensor musa_rope_apply(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin, double sin_sign);
torch::Tensor musa_rope_backward(
    torch::Tensor grad_output, torch::Tensor positions, double theta);
torch::Tensor musa_silu_forward(torch::Tensor x);
torch::Tensor musa_silu_backward(torch::Tensor grad_output, torch::Tensor x);
torch::Tensor musa_swiglu_forward(torch::Tensor gate, torch::Tensor up);
std::vector<torch::Tensor> musa_swiglu_backward(
    torch::Tensor grad_output, torch::Tensor gate, torch::Tensor up);
torch::Tensor musa_selected_logp_forward(
    torch::Tensor logits, torch::Tensor target, int64_t ignore_index);
torch::Tensor musa_selected_logp_backward(
    torch::Tensor grad_output, torch::Tensor logits,
    torch::Tensor target, int64_t ignore_index);
std::vector<torch::Tensor> musa_batch_invariant_logp_forward(
    torch::Tensor logits, torch::Tensor target, int64_t ignore_index);
torch::Tensor musa_batch_invariant_logp_backward(
    torch::Tensor grad_output, torch::Tensor logits,
    torch::Tensor target, torch::Tensor lse, int64_t ignore_index);
std::vector<torch::Tensor> musa_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool causal, double scale, torch::optional<torch::Tensor> mask,
    bool output_fp32);
std::vector<torch::Tensor> musa_attention_backward(
    torch::Tensor grad_output, torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor probs, double scale);

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
    m.def("musa_embedding_forward", &musa_embedding_forward);
    m.def("musa_rowwise_gemm_fp32", &musa_rowwise_gemm_fp32);
    m.def("musa_lm_head_fp32", &musa_lm_head_fp32);
    m.def("musa_embedding_backward", &musa_embedding_backward);
    m.def("musa_rmsnorm_forward", &musa_rmsnorm_forward,
          py::arg("x"), py::arg("weight"), py::arg("eps"),
          py::arg("output_fp32") = false);
    m.def("musa_rmsnorm_backward", &musa_rmsnorm_backward);
    m.def("musa_rope_forward", &musa_rope_forward);
    m.def("musa_rope_apply", &musa_rope_apply);
    m.def("musa_rope_backward", &musa_rope_backward);
    m.def("musa_silu_forward", &musa_silu_forward);
    m.def("musa_silu_backward", &musa_silu_backward);
    m.def("musa_swiglu_forward", &musa_swiglu_forward);
    m.def("musa_swiglu_backward", &musa_swiglu_backward);
    m.def("musa_selected_logp_forward", &musa_selected_logp_forward,
          py::arg("logits"), py::arg("target"), py::arg("ignore_index") = -100);
    m.def("musa_selected_logp_backward", &musa_selected_logp_backward,
          py::arg("grad_output"), py::arg("logits"),
          py::arg("target"), py::arg("ignore_index") = -100);
    m.def("musa_batch_invariant_logp_forward", &musa_batch_invariant_logp_forward,
          py::arg("logits"), py::arg("target"),
          py::arg("ignore_index") = -100);
    m.def("musa_batch_invariant_logp_backward", &musa_batch_invariant_logp_backward,
          py::arg("grad_output"), py::arg("logits"),
          py::arg("target"), py::arg("lse"),
          py::arg("ignore_index") = -100);
    m.def("musa_attention_forward", &musa_attention_forward,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("causal"), py::arg("scale"),
          py::arg("mask") = py::none(), py::arg("output_fp32") = false);
    m.def("musa_attention_backward", &musa_attention_backward);
}
