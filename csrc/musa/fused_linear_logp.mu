// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

#include <ATen/Functions.h>

namespace {

torch::Tensor project_logits(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias) {
    auto logits = at::mm(hidden.to(torch::kFloat), weight.to(torch::kFloat).t());
    if (bias.has_value()) {
        logits.add_(bias->to(torch::kFloat));
    }
    return logits;
}

void check_inputs(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias) {
    TORCH_CHECK(
        hidden.device().type() == c10::kPrivateUse1 &&
            weight.device().type() == c10::kPrivateUse1 &&
            target.device().type() == c10::kPrivateUse1,
        "linear_logp requires MUSA tensors");
    TORCH_CHECK(hidden.device() == weight.device() && hidden.device() == target.device(),
                "linear_logp tensors must share a device");
    TORCH_CHECK(hidden.dim() == 2 && weight.dim() == 2 && target.dim() == 1,
                "expected hidden [N, H], weight [V, H], and target [N]");
    TORCH_CHECK(hidden.scalar_type() == at::ScalarType::BFloat16 &&
                    weight.scalar_type() == at::ScalarType::BFloat16,
                "MUSA linear_logp currently supports BF16 hidden and weight");
    TORCH_CHECK(target.scalar_type() == at::ScalarType::Long,
                "target must be int64");
    TORCH_CHECK(hidden.size(1) == weight.size(1) && target.size(0) == hidden.size(0),
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

}  // namespace

std::vector<torch::Tensor> fused_linear_logp_musa_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias) {
    check_inputs(hidden, weight, target, bias);
    auto logits = project_logits(hidden, weight, bias);
    auto log_probs = at::_log_softmax(logits, 1, false);
    auto selected = at::gather(log_probs, 1, target.unsqueeze(1)).squeeze(1);
    auto lse = at::logsumexp(logits, {1}, false);
    return {selected, lse};
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
    (void)lse;
    check_inputs(hidden, weight, target, bias);
    TORCH_CHECK(grad_logp.device() == hidden.device() && grad_logp.dim() == 1 &&
                    grad_logp.size(0) == hidden.size(0),
                "grad_logp must be a MUSA tensor with one value per token");
    auto logits = project_logits(hidden, weight, bias);
    auto probabilities = at::_softmax(logits, 1, false);
    auto one_hot = at::zeros_like(probabilities);
    one_hot.scatter_(1, target.unsqueeze(1), 1.0);
    auto dlogits = (one_hot - probabilities) * grad_logp.to(torch::kFloat).unsqueeze(1);

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
