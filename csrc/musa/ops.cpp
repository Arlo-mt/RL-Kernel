// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

#include <limits>

torch::Tensor fused_logp_forward_musa(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor deterministic_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor silu_forward_cuda(torch::Tensor x);
torch::Tensor silu_backward_cuda(torch::Tensor dy, torch::Tensor x);
torch::Tensor swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up);
std::vector<torch::Tensor> swiglu_backward_cuda(
    torch::Tensor dy, torch::Tensor gate, torch::Tensor up);
torch::Tensor swiglu_packed_forward_cuda(torch::Tensor gate_up);
std::vector<torch::Tensor> swiglu_packed_backward_cuda(
    torch::Tensor dy, torch::Tensor gate_up);
void rmsnorm_forward_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor y, torch::Tensor rstd, double eps);
void rmsnorm_backward_dx_cuda(
    torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor rstd, torch::Tensor dx);
void rmsnorm_backward_partial_dw_cuda(
    torch::Tensor dy, torch::Tensor x, torch::Tensor rstd, torch::Tensor mask,
    torch::Tensor partial_dw);
void rmsnorm_backward_reduce_dw_cuda(torch::Tensor partial_dw, torch::Tensor dw);
void reduce_rows_fp32_left_fold_cuda(torch::Tensor rows, torch::Tensor output);
torch::Tensor embedding_sm90_forward_fp32(torch::Tensor token_ids, torch::Tensor weight);
torch::Tensor lm_head_sm90_forward_fp32(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias);
torch::Tensor det_gemm_rowwise_fwd_fp32(torch::Tensor a, torch::Tensor b);
torch::Tensor rope_apply_sm90(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    double sin_sign);
std::vector<torch::Tensor> deterministic_attention_forward_fp32(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double scale,
    torch::optional<torch::Tensor> key_padding_mask);
std::vector<torch::Tensor> deterministic_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor probs,
    bool causal,
    double scale,
    torch::optional<torch::Tensor> key_padding_mask);

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(
        logits.device().type() == c10::kPrivateUse1,
        "logits must be a MUSA tensor, got ",
        logits.device());
    TORCH_CHECK(
        token_ids.device().type() == c10::kPrivateUse1,
        "token_ids must be a MUSA tensor, got ",
        token_ids.device());
    TORCH_CHECK(logits.device() == token_ids.device(), "logits and token_ids must share a device");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be a 1D tensor");
    TORCH_CHECK(token_ids.scalar_type() == at::ScalarType::Long, "token_ids must be int64");
    TORCH_CHECK(token_ids.numel() == logits.size(0), "token_ids length must match logits rows");
    TORCH_CHECK(logits.size(0) <= std::numeric_limits<int>::max(), "too many logits rows");
    TORCH_CHECK(logits.size(1) > 0, "logits vocabulary dimension must be non-empty");
    if (token_ids.numel() > 0) {
        TORCH_CHECK(
            token_ids.min().item<int64_t>() >= 0 &&
                token_ids.max().item<int64_t>() < logits.size(1),
            "token_ids must be within the logits vocabulary dimension");
    }
    TORCH_CHECK(
        logits.scalar_type() == at::ScalarType::Float ||
            logits.scalar_type() == at::ScalarType::Half ||
            logits.scalar_type() == at::ScalarType::BFloat16,
        "MUSA fused_logp supports float32, float16, and bfloat16 logits");

    auto logits_contiguous = logits.contiguous();
    auto token_ids_contiguous = token_ids.contiguous();
    return fused_logp_forward_musa(logits_contiguous, token_ids_contiguous);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_logp", &fused_logp_forward, "MUSA fused selected-token log-probability");
    m.def("deterministic_logp_fp32", &deterministic_logp_forward_fp32);
    m.def("silu_forward", &silu_forward_cuda);
    m.def("silu_backward", &silu_backward_cuda);
    m.def("swiglu_forward", &swiglu_forward_cuda);
    m.def("swiglu_backward", &swiglu_backward_cuda);
    m.def("swiglu_packed_forward", &swiglu_packed_forward_cuda);
    m.def("swiglu_packed_backward", &swiglu_packed_backward_cuda);
    m.def("rmsnorm_forward", &rmsnorm_forward_cuda);
    m.def("rmsnorm_backward_dx", &rmsnorm_backward_dx_cuda);
    m.def("rmsnorm_backward_partial_dw", &rmsnorm_backward_partial_dw_cuda);
    m.def("rmsnorm_backward_reduce_dw", &rmsnorm_backward_reduce_dw_cuda);
    m.def("reduce_rows_fp32_left_fold", &reduce_rows_fp32_left_fold_cuda);
    m.def("embedding_fp32", &embedding_sm90_forward_fp32);
    m.def("lm_head_fp32", &lm_head_sm90_forward_fp32, py::arg("hidden"),
          py::arg("weight"), py::arg("bias") = py::none());
    m.def("det_gemm_rowwise_fwd_fp32", &det_gemm_rowwise_fwd_fp32);
    m.def("rope_apply", &rope_apply_sm90);
    m.def("deterministic_attention_fp32", &deterministic_attention_forward_fp32,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal"),
          py::arg("scale"), py::arg("key_padding_mask") = py::none());
    m.def("deterministic_attention_backward", &deterministic_attention_backward,
          py::arg("grad_output"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("probs"), py::arg("causal"), py::arg("scale"),
          py::arg("key_padding_mask") = py::none());
}
