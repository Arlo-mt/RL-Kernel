# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU-safe validation for the strict WS2 Attention GPU acceptance runner."""

from __future__ import annotations

import json
import subprocess

from scripts.ws2_attention_gpu_acceptance import (
    build_acceptance_cases,
    parse_args,
    run_acceptance,
    validate_p2p_report,
    validate_pr5_report,
    validate_pr7_report,
)


def test_manifest_fails_closed_for_every_unexecuted_required_case(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = run_acceptance(args)

    assert report["status"] == "failed"
    assert report["passed"] is False
    assert "custom_cuda_ag_rs" in report["failed_required_cases"]
    assert all(not case["passed"] for case in report["cases"])


def test_matrix_contains_required_modes_splitk_and_communication(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    names = {case.name for case in build_acceptance_cases(args)}

    assert "pr5_cp_forward_backward_dlogp" in names
    assert "p2p_nccl_reference" in names
    assert "pr7_flashinfer_decode_disabled" in names
    assert "pr7_flashinfer_decode_fixed" in names
    assert "pr7_flashinfer_prefill_disabled" in names
    assert "pr7_flashinfer_prefill_fixed" in names
    assert "custom_cuda_ag_rs" in names


def test_run_mode_does_not_pass_when_reports_are_missing(tmp_path):
    args = parse_args(
        [
            "--mode",
            "run",
            "--output",
            str(tmp_path / "acceptance.json"),
        ]
    )

    def fake_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    report = run_acceptance(args, runner=fake_runner)

    assert report["passed"] is False
    assert "custom_cuda_ag_rs" in report["failed_required_cases"]
    assert any(case["status"] == "invalid_report" for case in report["cases"])


def test_pr7_strict_validation_rejects_requested_only_split_plan(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = {
        "status": "passed",
        "passed": True,
        "candidate_provenance": {
            "arithmetic_semantics_verified": True,
            "actual_split_kv_plans": [
                {
                    "actual_split_kv_policy": None,
                    "actual_split_boundaries": [],
                }
            ],
            "actual_split_kv_plan_set": None,
        },
        "drift": {
            "out": {"max_abs": 0.0},
            "lse": {"max_abs": 0.0},
            "dlogp": {"max_abs": 0.0},
        },
        "batch_invariant_sweep": {"passed": True},
        "page_layout_invariant_sweep": {"passed": True},
    }

    errors = validate_pr7_report(report, args, expected_policy="fixed")

    assert any("actual Split-K policy" in error for error in errors)
    assert any("boundaries" in error for error in errors)
    assert any("plan set" in error for error in errors)


def test_acceptance_report_is_json_serializable(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    json.dumps(run_acceptance(args))


def _valid_pr5_report():
    def case(mode, policy, split_size):
        stats = {"max_abs": 0.0}
        entries = []
        for tp_rank in range(2):
            for cp_rank in range(2):
                for owner_cp_rank, owner_range in enumerate(([0, 2], [2, 4])):
                    entries.append(
                        {
                            "batch_index": 0,
                            "tp_rank": tp_rank,
                            "cp_rank": cp_rank,
                            "owner_cp_rank": owner_cp_rank,
                            "expected_kv_range": owner_range,
                            "requested_split_kv_policy": policy,
                            "actual_split_kv_policy": policy,
                            "actual_split_kv_size": split_size,
                            "actual_split_kv_count": 1 if split_size is None else 2,
                            "actual_split_boundaries": (
                                [owner_range]
                                if split_size is None
                                else [[owner_range[0], owner_range[0] + 1], [owner_range[0] + 1, owner_range[1]]]
                            ),
                            "split_kv_merge_order": "global_block_index",
                            "split_kv_accum_dtype": "fp32",
                            "split_kv_downcast_at": "final_write",
                            "split_kv_plan_source": "test_runtime",
                            "split_kv_fallback": False,
                            "split_kv_fallback_reason": None,
                        }
                    )
        return {
            "case_name": f"{mode}-{policy}",
            "attention_mode": mode,
            "topology": {"tp_world_size": 2, "cp_world_size": 2},
            "provenance": {
                "requested_split_kv_policy": policy,
                "requested_split_kv_size": split_size,
                "rope": {"rope_state": "post_rope"},
                "actual_split_kv_plan_set": {
                    "batch_size": 1,
                    "tp_world_size": 2,
                    "cp_world_size": 2,
                    "total_kv_tokens": [4],
                    "entries": entries,
                    "coverage": "complete_batch_tp_cp_owner_cartesian_product",
                },
            },
            "drift": {"cp_merge_fp32": {"out": stats, "lse": stats}},
            "dlogp": {"status": "available", "drift": stats},
            "backward": {
                "status": "available",
                "report": {"drifts": [{"dq": stats, "dk": stats, "dv": stats}]},
            },
        }

    return {
        "schema_version": "ws2_cp_attention_drift/v2",
        "issue": 235,
        "pr": 5,
        "runtime": {"device": "cuda:0"},
        "target": {
            "model": "qwen3-8b",
            "dtype": "bf16",
            "global_num_query_heads": 32,
            "global_num_kv_heads": 8,
            "head_dim": 128,
            "batch": 1,
        },
        "cases": [case("prefill", "disabled", None), case("chunked_prefill", "fixed", 4)],
    }


def test_pr5_validation_binds_gpu_identity_and_nonempty_backward(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    assert validate_pr5_report(report, args) == []

    report["runtime"]["device"] = "cpu"
    report["cases"][0]["backward"]["report"]["drifts"] = []
    errors = validate_pr5_report(report, args)
    assert any("not produced on CUDA" in error for error in errors)
    assert any("backward drift rows" in error for error in errors)


def test_pr5_validation_rejects_nonfinite_or_negative_drift(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    report["cases"][0]["drift"]["cp_merge_fp32"]["out"] = {"max_abs": float("nan")}
    report["cases"][1]["dlogp"]["drift"] = {"max_abs": -1.0}

    errors = validate_pr5_report(report, args)
    assert sum("finite and non-negative" in error for error in errors) == 2


def test_p2p_validation_binds_nccl_rank_and_arithmetic_provenance():
    def row(rank, query_range):
        manifest = [
            {
                "global_block_index": block,
                "kv_block_start": block * 4,
                "kv_block_end": block * 4 + 4,
                "owner_cp_rank": 0 if block < 2 else 1,
                "owner_tp_rank": 0,
            }
            for block in range(4)
        ]
        return {
            "rank": rank,
            "world_size": 2,
            "passed": True,
            "global_failure_count": 0,
            "transport": "p2p_nccl_reference",
            "device": f"cuda:{rank}",
            "dtype": "bf16",
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "final_output_dtype": "bfloat16",
            "query_range": query_range,
            "expected_block_manifest": manifest,
            "gathered_block_indices": [0, 1, 2, 3],
            "out_max_abs": 0.0,
            "lse_max_abs": 0.0,
            "final_out_max_abs": 0.0,
            "atol": 2.0e-4,
            "final_write_atol": 2.0e-2,
        }

    report = {
        "schema_version": "ws2_p2p_nccl_attention_reference/v1",
        "backend": "nccl",
        "world_size": 2,
        "global_failure_count": 0,
        "ranks": [row(0, [0, 8]), row(1, [8, 16])],
    }
    assert validate_p2p_report(report) == []

    report["ranks"][1]["transport"] = "gloo"
    report["ranks"][1]["rank"] = 0
    errors = validate_p2p_report(report)
    assert any("NCCL reference transport" in error for error in errors)
    assert any("ranks 0 and 1" in error for error in errors)


def test_p2p_validation_rejects_claimed_downcast_without_final_output_evidence():
    report = {
        "schema_version": "ws2_p2p_nccl_attention_reference/v1",
        "backend": "nccl",
        "world_size": 2,
        "global_failure_count": 0,
        "ranks": [
            {
                "rank": rank,
                "world_size": 2,
                "passed": True,
                "global_failure_count": 0,
                "transport": "p2p_nccl_reference",
                "device": f"cuda:{rank}",
                "dtype": "bf16",
                "accum_dtype": "fp32",
                "downcast_at": "final_write",
                "query_range": [rank * 8, (rank + 1) * 8],
                "gathered_block_indices": [0, 1],
                "out_max_abs": 0.0,
                "lse_max_abs": 0.0,
                "atol": 2.0e-4,
            }
            for rank in range(2)
        ],
    }

    errors = validate_p2p_report(report)
    assert any("final output dtype" in error for error in errors)
    assert any("gathered block order/coverage" in error for error in errors)
    assert any("final_out_max_abs" in error for error in errors)


def test_p2p_validation_rejects_forged_manifest_and_rank_query_mapping():
    def row(rank, query_range):
        return {
            "rank": rank,
            "world_size": 2,
            "global_failure_count": 0,
            "passed": True,
            "transport": "p2p_nccl_reference",
            "device": f"cuda:{rank}",
            "dtype": "bf16",
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "final_output_dtype": "bfloat16",
            "query_range": query_range,
            "expected_block_manifest": [
                {
                    "global_block_index": 0,
                    "kv_block_start": 0,
                    "kv_block_end": 4,
                    "owner_cp_rank": 0,
                    "owner_tp_rank": 0,
                },
                {
                    "global_block_index": 1,
                    "kv_block_start": 5,
                    "kv_block_end": 8,
                    "owner_cp_rank": 3,
                    "owner_tp_rank": 0,
                },
            ],
            "gathered_block_indices": [0, 1],
            "out_max_abs": 0.0,
            "lse_max_abs": 0.0,
            "final_out_max_abs": 0.0,
            "atol": 2.0e-4,
            "final_write_atol": 2.0e-2,
        }

    report = {
        "schema_version": "ws2_p2p_nccl_attention_reference/v1",
        "backend": "nccl",
        "world_size": 2,
        "global_failure_count": 0,
        "ranks": [row(0, [8, 16]), row(1, [0, 8])],
    }

    errors = validate_p2p_report(report)
    assert any("gap-free KV coverage" in error for error in errors)
    assert any("outside the TP-local CP=2 group" in error for error in errors)
    assert any("both CP ranks" in error for error in errors)
    assert any("query ownership ranges" in error for error in errors)


def test_pr5_validation_rejects_forged_plan_set_coverage(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    plan_set = report["cases"][0]["provenance"]["actual_split_kv_plan_set"]
    plan_set["entries"] = plan_set["entries"][:-1]
    plan_set["entries"][0]["split_kv_accum_dtype"] = "bf16"

    errors = validate_pr5_report(report, args)
    assert any("coordinate coverage is incomplete" in error for error in errors)
    assert any("accumulation dtype is not fp32" in error for error in errors)


def test_pr5_validation_reports_malformed_coordinates_without_crashing(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    plan_set = report["cases"][0]["provenance"]["actual_split_kv_plan_set"]
    plan_set["entries"][0]["batch_index"] = []

    errors = validate_pr5_report(report, args)
    assert any("coordinate must contain integers" in error for error in errors)
    assert any("coordinate coverage is incomplete" in error for error in errors)
