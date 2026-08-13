# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU tests for the WS1 C8 four-judgment matrix schema."""

from __future__ import annotations

from rl_engine.kernels.gtest.four_judgment_matrix import (
    C8_REQUIRED_OPS,
    JUDGMENTS,
    PROFILES,
    TIERS,
    build_classified_matrix,
    hidden_required_na,
    undefined_cells,
)
from rl_engine.testing.ws1_workload import load_manifest


def test_matrix_covers_required_ops_profiles_judgments_and_tiers():
    report = build_classified_matrix()
    keys = {
        (cell.profile, cell.op_name, cell.judgment, cell.tier) for cell in report.cells
    }
    expected = {
        (profile, op_name, judgment, tier)
        for profile in PROFILES
        for op_name in C8_REQUIRED_OPS
        for judgment in JUDGMENTS
        for tier in TIERS
    }
    assert keys == expected
    assert undefined_cells(report) == ()


def test_triton_required_candidates_are_declared():
    report = build_classified_matrix()
    missing = [
        cell
        for cell in report.cells
        if cell.profile == "triton_cuda_bf16" and cell.op_name in {"embedding", "lm_head", "logp"}
    ]
    assert missing
    assert all(cell.candidate == "triton" for cell in missing)
    assert all(cell.status == "red" for cell in missing)
    assert hidden_required_na(report) == ()


def test_pack_is_explicit_na_with_c2_reason():
    report = build_classified_matrix()
    pack = [cell for cell in report.cells if cell.op_name == "pack"]
    assert pack
    assert all(cell.status == "N/A" for cell in pack)
    assert all("profile-independent" in cell.detail for cell in pack)


def test_sm90_declared_cells_are_pending_hopper():
    report = build_classified_matrix()
    hopper = [
        cell
        for cell in report.cells
        if cell.profile == "cuda_bf16"
        and cell.op_name in {"embedding", "lm_head", "rope"}
    ]
    assert hopper
    assert all(
        cell.status == "pending_hopper" for cell in hopper if cell.case_id is not None
    )
    assert all(cell.candidate == "cuda-sm90" for cell in hopper)


def test_declared_runnable_ops_have_short_and_primary_case_ids():
    manifest = load_manifest()
    report = build_classified_matrix(manifest)
    runnable = {
        "rms_norm",
        "qk_norm",
        "det_gemm",
        "attention",
        "silu",
        "swiglu",
    }
    for cell in report.cells:
        if cell.op_name not in runnable:
            continue
        if cell.status == "pending_hopper":
            continue
        assert cell.case_id, (cell.op_name, cell.profile, cell.tier)
        assert any(case["case_id"] == cell.case_id for case in manifest.representative_cases)


def test_triton_rope_has_case_ids_but_cuda_rope_is_hopper():
    report = build_classified_matrix()
    triton_rope = [
        cell
        for cell in report.cells
        if cell.op_name == "rope" and cell.profile == "triton_cuda_bf16"
    ]
    cuda_rope = [
        cell for cell in report.cells if cell.op_name == "rope" and cell.profile == "cuda_bf16"
    ]
    assert all(cell.case_id for cell in triton_rope)
    assert all(cell.status == "pending_hopper" for cell in cuda_rope)
