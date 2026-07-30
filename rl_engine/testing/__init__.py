# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Testing helpers for RL-shaped kernel validation."""

from .attention_comparison import (
    AttentionComparisonInputs,
    AttentionComparisonReport,
    AttentionPathDrift,
    AttentionPathResult,
    DriftStats,
    TransformerEngineUnavailable,
    compare_single_gpu_attention,
    run_chunked_query_attention,
    run_full_attention,
    run_paged_kv_attention,
    transformer_engine_context_parallel_available,
)
from .reference_ops import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    masked_sum,
    selected_logprobs_reference,
    summarize_kernel_drift,
)
from .rl_batch import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch

__all__ = [
    "AttentionComparisonInputs",
    "AttentionComparisonReport",
    "AttentionPathDrift",
    "AttentionPathResult",
    "DriftStats",
    "SyntheticRLKernelBatch",
    "TransformerEngineUnavailable",
    "active_token_count",
    "compare_single_gpu_attention",
    "compute_policy_ratio",
    "compute_reference_kl",
    "make_synthetic_rl_kernel_batch",
    "masked_mean",
    "masked_sum",
    "run_chunked_query_attention",
    "run_full_attention",
    "run_paged_kv_attention",
    "selected_logprobs_reference",
    "summarize_kernel_drift",
    "transformer_engine_context_parallel_available",
]
