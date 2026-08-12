#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Execute WS1 C2 representative CUDA/Triton cases and emit runtime provenance."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest import run_operator_suite  # noqa: E402
from rl_engine.kernels.gtest.operator_specs import make_candidate, make_operator_case  # noqa: E402
from rl_engine.testing.ws1_workload import WorkloadError, load_manifest  # noqa: E402


def _object_path(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _case_args(case: dict[str, Any], seed: int) -> SimpleNamespace:
    shape = case["shape"]
    operator_spec = case["operator_spec"]
    common: dict[str, Any] = {
        "op": operator_spec,
        "candidate": case["expected_backend_id"],
        "arch_key": None,
        "input_mode": "random",
        "constant_value": 0.25,
        "token_value": 0,
        "normalized_dim": 4096,
        "k_dim": 4096,
        "n_dim": 4096,
        "theta": 1.0e6,
        "eps": 1.0e-6,
        "seed": seed,
    }
    if operator_spec == "det_gemm":
        common.update(batch=1, seq=shape["M"], k_dim=shape["K"], n_dim=shape["N"])
    elif operator_spec == "attention":
        common.update(
            batch=shape["B"],
            seq=shape["Sq"],
            skv=shape["Skv"],
            n_heads=shape["Hq"],
            n_kv_heads=shape["Hkv"],
            causal=1,
            use_padding=0,
            scale_mode="default",
        )
    elif operator_spec in {"logp", "batch_invariant_logp"}:
        common.update(batch=shape["B"], seq=shape["T"], vocab=shape["vocab"])
    else:
        raise WorkloadError(f"unsupported representative operator_spec {operator_spec!r}")
    return SimpleNamespace(**common)


def run_case(case: dict[str, Any], *, seed: int, device: torch.device) -> dict[str, Any]:
    args = _case_args(case, seed)
    candidate = make_candidate(args)
    actual_path = _object_path(candidate.fn)
    if candidate.backend != case["expected_backend_id"]:
        raise WorkloadError(
            f"case {case['case_id']} resolved backend {candidate.backend!r}, expected "
            f"{case['expected_backend_id']!r}"
        )
    if actual_path != case["expected_kernel_config_id"]:
        raise WorkloadError(
            f"case {case['case_id']} resolved kernel {actual_path!r}, expected "
            f"{case['expected_kernel_config_id']!r}"
        )

    operator_case = make_operator_case(args, torch.bfloat16, device)
    report = run_operator_suite(
        case["operator_spec"], candidates=[candidate], cases=[operator_case]
    )
    torch.cuda.synchronize(device)
    candidate_report = report.candidates[0]
    output_checks = [
        {
            "shape": list(output.shape),
            "dtype": output.candidate_dtype,
            "max_abs_error": output.max_abs_error,
            "passed": output.passed,
        }
        for checked_case in candidate_report.cases
        for output in checked_case.outputs
    ]
    return {
        "case_id": case["case_id"],
        "fixture_id": case["fixture_id"],
        "operator_spec": case["operator_spec"],
        "expected_backend_id": case["expected_backend_id"],
        "actual_backend_id": candidate.backend,
        "expected_kernel_config_id": case["expected_kernel_config_id"],
        "actual_kernel_config_id": actual_path,
        "algorithm_property": case["algorithm_property"],
        "shape": case["shape"],
        "runtime_status": "passed" if report.passed else "failed",
        "outputs": output_checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run manifest-pinned WS1 representative candidates on a real GPU."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--profile",
        action="append",
        choices=("cuda_bf16", "triton_cuda_bf16"),
        help="Profile to run; repeatable. Defaults to both required profiles.",
    )
    parser.add_argument("--case-id", action="append", help="Optional case_id filter.")
    parser.add_argument("--emit-json", default="-", help="Output path, or '-' for stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        print("error: CUDA is required for runtime candidate evidence", file=sys.stderr)
        return 2

    try:
        manifest = load_manifest(args.manifest)
        profiles = set(args.profile or ("cuda_bf16", "triton_cuda_bf16"))
        selected_ids = set(args.case_id or ())
        cases = [
            case
            for case in manifest.representative_cases
            if profiles.intersection(case["profile_ids"])
            and (not selected_ids or case["case_id"] in selected_ids)
        ]
        if selected_ids - {case["case_id"] for case in cases}:
            unknown = sorted(selected_ids - {case["case_id"] for case in cases})
            raise WorkloadError(f"unknown or profile-filtered case IDs: {unknown}")
        device = torch.device("cuda:0")
        log_stream = sys.stderr if args.emit_json == "-" else sys.stdout
        with contextlib.redirect_stdout(log_stream):
            results = [
                run_case(case, seed=manifest.seed + i, device=device)
                for i, case in enumerate(cases)
            ]
    except (RuntimeError, ValueError, WorkloadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    props = torch.cuda.get_device_properties(device)
    payload = {
        "schema_version": "ws1-c2-runtime-provenance-v1",
        "workload_id": manifest.workload_id,
        "fixture_identity_sha256": manifest.raw["fixture_identity_sha256"],
        "execution_dtype": "bfloat16",
        "device": {
            "index": device.index,
            "name": props.name,
            "compute_capability": f"sm{props.major}{props.minor}",
            "execution_world_size": 1,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "profiles": sorted(profiles),
        "passed": bool(results) and all(result["runtime_status"] == "passed" for result in results),
        "cases": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.emit_json == "-":
        sys.stdout.write(rendered)
    else:
        path = Path(args.emit_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote: {path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
