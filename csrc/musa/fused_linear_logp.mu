// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

#include <ATen/Functions.h>

namespace {

constexpr int64_t kVocabTile = 8192;

void check_inputs(
    const torch::Tensor& hidden,
    const torch::Tensor& weight,
    const torch::Tensor& target,
    const torch::optional<torch::Tensor>& bias) {
    TORCH_CHECK(
        hidden.device().type() == c10::kPrivateUse1 &&
            weight.device().type() == c10::kPrivateUse1 &&
            target.device().type() == c10::kPrivateUse1,
        "linear_logp requires MUSA tensors");
    TORCH_CHECK(
        hidden.device() == weight.device() && hidden.device() == target.device(),
        "linear_logp tensors must share a device");
    TORCH_CHECK(
        hidden.dim() == 2 && weight.dim() == 2 && target.dim() == 1,
        "expected hidden [N, H], weight [V, H], and target [N]");
    TORCH_CHECK(
        hidden.scalar_type() == at::ScalarType::BFloat16 &&
            weight.scalar_type() == at::ScalarType::BFloat16,
        "MUSA linear_logp currently supports BF16 hidden and weight");
    TORCH_CHECK(target.scalar_type() == at::ScalarType::Long, "target must be int64");
    TORCH_CHECK(
        hidden.size(1) == weight.size(1) && target.size(0) == hidden.size(0),
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

torch::Tensor project_tile(
    const torch::Tensor& hidden_f,
    const torch::Tensor& weight,
    const torch::optional<torch::Tensor>& bias,
    int64_t vocab_start,
    int64_t vocab_count) {
    auto weight_tile = weight.narrow(0, vocab_start, vocab_count).to(torch::kFloat).t();
    auto logits = at::mm(hidden_f, weight_tile);
    if (bias.has_value()) {
        logits.add_(bias->narrow(0, vocab_start, vocab_count).to(torch::kFloat));
    }
    return logits;
}

}  // namespace

std::vector<torch::Tensor> fused_linear_logp_musa_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias) {
    check_inputs(hidden, weight, target, bias);
    const int64_t rows = hidden.size(0);
    const int64_t vocab = weight.size(0);
    auto hidden_f = hidden.to(torch::kFloat).contiguous();
    auto row_max = torch::full({rows}, -INFINITY, hidden.options().dtype(torch::kFloat));
    auto row_sum = torch::zeros({rows}, hidden.options().dtype(torch::kFloat));
    auto selected = torch::zeros({rows}, hidden.options().dtype(torch::kFloat));

    for (int64_t vocab_start = 0; vocab_start < vocab; vocab_start += kVocabTile) {
        const int64_t vocab_count = std::min(kVocabTile, vocab - vocab_start);
        auto logits = project_tile(hidden_f, weight, bias, vocab_start, vocab_count);
        auto tile_max = std::get<0>(logits.max(1));
        auto tile_sum = (logits - tile_max.unsqueeze(1)).exp().sum(1);
        auto new_max = torch::maximum(row_max, tile_max);
        row_sum = row_sum * (row_max - new_max).exp() +
            tile_sum * (tile_max - new_max).exp();
        row_max = new_max;

        auto local_target = (target - vocab_start).clamp(0, vocab_count - 1);
        auto target_value = logits.gather(1, local_target.unsqueeze(1)).squeeze(1);
        auto owns = (target >= vocab_start) & (target < vocab_start + vocab_count);
        selected = torch::where(owns, target_value, selected);
    }
    auto lse = row_max + row_sum.log();
    return {selected - lse, lse};
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
    check_inputs(hidden, weight, target, bias);
    TORCH_CHECK(
        grad_logp.device() == hidden.device() && grad_logp.dim() == 1 &&
            grad_logp.size(0) == hidden.size(0),
        "grad_logp must have one value per token");
    const int64_t rows = hidden.size(0);
    const int64_t vocab = weight.size(0);
    auto hidden_f = hidden.to(torch::kFloat).contiguous();
    auto weight_f = weight.to(torch::kFloat).contiguous();
    auto lse_f = lse.to(torch::kFloat).contiguous();
    auto grad_f = grad_logp.to(torch::kFloat).contiguous();
    auto grad_hidden = compute_grad_hidden
        ? torch::zeros({rows, hidden.size(1)}, hidden.options().dtype(torch::kFloat))
        : torch::Tensor();
    auto grad_weight = compute_grad_weight
        ? torch::empty({vocab, hidden.size(1)}, hidden.options().dtype(torch::kFloat))
        : torch::Tensor();
    auto grad_bias = compute_grad_bias && bias.has_value()
        ? torch::empty({vocab}, hidden.options().dtype(torch::kFloat))
        : torch::Tensor();

    for (int64_t vocab_start = 0; vocab_start < vocab; vocab_start += kVocabTile) {
        const int64_t vocab_count = std::min(kVocabTile, vocab - vocab_start);
        auto logits = project_tile(hidden_f, weight, bias, vocab_start, vocab_count);
        auto probabilities = (logits - lse_f.unsqueeze(1)).exp();
        auto local_target = (target - vocab_start).clamp(0, vocab_count - 1);
        auto one_hot = torch::zeros_like(probabilities);
        one_hot.scatter_(1, local_target.unsqueeze(1), 1.0);
        auto dlogits = (one_hot - probabilities) * grad_f.unsqueeze(1);

        if (compute_grad_hidden) {
            grad_hidden.add_(at::mm(dlogits, weight_f.narrow(0, vocab_start, vocab_count)));
        }
        if (compute_grad_weight) {
            grad_weight.narrow(0, vocab_start, vocab_count).copy_(
                at::mm(dlogits.t(), hidden_f));
        }
        if (compute_grad_bias && bias.has_value()) {
            grad_bias.narrow(0, vocab_start, vocab_count).copy_(dlogits.sum(0));
        }
    }

    return {
        compute_grad_hidden ? grad_hidden.to(hidden.scalar_type()) : torch::Tensor(),
        compute_grad_weight ? grad_weight.to(weight.scalar_type()) : torch::Tensor(),
        compute_grad_bias && bias.has_value() ? grad_bias.to(bias->scalar_type()) : torch::Tensor(),
    };
}
