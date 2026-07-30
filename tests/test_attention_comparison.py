# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types

import torch

from rl_engine.kernels.gtest import run_operator_suite
from rl_engine.kernels.gtest.operator_specs import make_candidate, make_operator_case
from rl_engine.testing.attention_comparison import (
    AttentionComparisonInputs,
    compare_single_gpu_attention,
)

_TE_CONTEXT_PARALLEL_MODULE = (
    "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"
)


def _qkv(*, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 4, 6, 8, generator=gen)
    k = torch.randn(2, 2, 6, 8, generator=gen)
    v = torch.randn(2, 2, 6, 8, generator=gen)
    return q, k, v


def _comparison_inputs() -> AttentionComparisonInputs:
    q, k, v = _qkv()
    gen = torch.Generator().manual_seed(2)
    lm_head_weight = torch.randn(13, q.size(1) * q.size(3), generator=gen)
    target_ids = torch.randint(0, 13, (q.size(0), q.size(2)), generator=gen)
    active_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [False, True, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    return AttentionComparisonInputs(
        q=q,
        k=k,
        v=v,
        causal=True,
        lm_head_weight=lm_head_weight,
        target_ids=target_ids,
        active_token_mask=active_mask,
    )


def test_single_gpu_attention_harness_reports_out_lse_and_dlogp_drift():
    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=2,
        kv_page_size=3,
    )

    by_name = {drift.candidate_name: drift for drift in report.drifts}
    assert set(by_name) == {"chunked_prefill", "rl_kernel_paged_kv"}
    for drift in by_name.values():
        assert drift.out.max_abs <= 1.0e-6
        assert drift.lse.max_abs <= 1.0e-6
        assert drift.dlogp is not None
        assert drift.dlogp.active_count == 7
        assert drift.dlogp.p95_abs <= 1.0e-6

    payload = report.to_dict()
    assert payload["reference_name"] == "full_prefill"
    assert payload["drifts"][0]["out"]["p99_abs"] >= 0.0
    json.dumps(payload)


def test_single_gpu_attention_harness_preserves_key_padding_mask():
    q, k, v = _qkv(seed=3)
    key_padding_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, False, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    report = compare_single_gpu_attention(
        AttentionComparisonInputs(
            q=q,
            k=k,
            v=v,
            causal=True,
            key_padding_mask=key_padding_mask,
        ),
        query_chunk_size=4,
        kv_page_size=2,
    )

    assert report.unavailable == ()
    for drift in report.drifts:
        assert drift.out.max_abs <= 1.0e-6
        assert drift.lse.max_abs <= 1.0e-6


def test_transformer_engine_merge_oracle_can_be_reused_when_available(monkeypatch):
    calls = {"lse": 0, "out": 0}

    def lse_correction(softmax_lse, softmax_lse_per_step):
        calls["lse"] += 1
        softmax_lse.copy_(torch.logaddexp(softmax_lse, softmax_lse_per_step))

    def out_correction_init(out_init_step, softmax_lse, softmax_lse_init_step, seq_dim):
        scale = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
        return out_init_step * scale.unsqueeze(-1)

    def out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step, seq_dim):
        calls["out"] += 1
        scale = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
        out.add_(out_per_step * scale.unsqueeze(-1))

    monkeypatch.setitem(
        sys.modules,
        _TE_CONTEXT_PARALLEL_MODULE,
        types.SimpleNamespace(
            flash_attn_fwd_softmax_lse_correction=lse_correction,
            flash_attn_fwd_out_correction_init=out_correction_init,
            flash_attn_fwd_out_correction=out_correction,
        ),
    )

    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=3,
        kv_page_size=2,
        include_transformer_engine=True,
    )

    by_name = {drift.candidate_name: drift for drift in report.drifts}
    assert "transformer_engine_paged_kv" in by_name
    assert by_name["transformer_engine_paged_kv"].out.max_abs <= 1.0e-6
    assert by_name["transformer_engine_paged_kv"].lse.max_abs <= 1.0e-6
    assert calls["lse"] > 0
    assert calls["out"] > 0
    assert report.unavailable == ()


def test_transformer_engine_path_reports_unavailable_without_failing(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == _TE_CONTEXT_PARALLEL_MODULE:
            raise ImportError("test TE unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    report = compare_single_gpu_attention(
        _comparison_inputs(),
        query_chunk_size=3,
        kv_page_size=2,
        include_transformer_engine=True,
    )

    assert {drift.candidate_name for drift in report.drifts} == {
        "chunked_prefill",
        "rl_kernel_paged_kv",
    }
    assert report.unavailable == ("transformer_engine_paged_kv: test TE unavailable",)


def test_operator_comparison_specs_register_attention():
    args = argparse.Namespace(
        op="attention",
        candidate="pytorch",
        arch_key=None,
        batch=1,
        seq=3,
        vocab=17,
        seed=7,
        input_mode="random",
        constant_value=0.5,
        token_value=3,
        normalized_dim=128,
        k_dim=16,
        n_dim=32,
        theta=1.0e6,
        eps=1.0e-6,
    )

    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    candidate = make_candidate(args)
    report = run_operator_suite("attention", candidates=[candidate], cases=[case])

    assert report.passed
    assert report.candidates[0].cases[0].op_class == "attention"
