# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strict issue #235 WS2 Attention GPU acceptance orchestrator.

This runner combines reports produced by the existing PR branches.  A required
case that is missing, skipped, dry-run only, or lacks actual runtime provenance
fails closed.  It therefore separates a useful local report from a GPU/NCCL
acceptance artifact that is eligible to close issue #235.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ws2_attention_gpu_acceptance/v1"
DEFAULT_IMAGE = "ghcr.io/rl-align/rl-kernel/rl-kernel-ci:cuda"


@dataclass(frozen=True)
class AcceptanceCase:
    name: str
    command: tuple[str, ...] | None
    required: bool = True
    report_path: Path | None = None
    validator: Callable[[Mapping[str, Any]], list[str]] | None = None
    unavailable_reason: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["manifest", "run"],
        default="manifest",
        help="manifest records the matrix without executing GPU commands",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--head-sha", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument(
        "--megatron-te-script",
        type=Path,
        help="Megatron Bridge teacher script used for the native TE CP comparison",
    )
    parser.add_argument("--megatron-model", type=Path)
    parser.add_argument("--megatron-token-artifact", type=Path)
    parser.add_argument("--megatron-python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--collective-world-size",
        type=int,
        choices=(2, 4, 8),
        default=8,
        help="rank count for the self-owned AG/RS/AllReduce probe",
    )
    parser.add_argument("--out-atol", type=float, default=2.0e-4)
    parser.add_argument("--lse-atol", type=float, default=2.0e-4)
    # The synthetic dlogp leg consumes the final BF16 Attention write. Use
    # the shared WS1 logprob/BF16 tolerance instead of an FP32-only threshold.
    parser.add_argument("--dlogp-atol", type=float, default=5.0e-2)
    parser.add_argument("--grad-atol", type=float, default=5.0e-2)
    return parser.parse_args(argv)


def build_acceptance_cases(args: argparse.Namespace) -> tuple[AcceptanceCase, ...]:
    artifact_dir = args.output.resolve().parent
    pr5_report = artifact_dir / "ws2-pr5-forward-backward.json"
    pr7_reports = {
        name: artifact_dir / f"ws2-pr7-{name}.json"
        for name in (
            "decode-disabled",
            "decode-fixed",
            "prefill-disabled",
            "prefill-fixed",
        )
    }
    python = str(args.python)
    torchrun = str(args.torchrun)
    pr7_script = REPO_ROOT / "scripts" / "ws2_pr7_flashinfer_attention_check.py"
    pr7_available = pr7_script.is_file()
    pr7_unavailable = None if pr7_available else "PR7 validation script is absent; integrate #279"
    p2p_script = REPO_ROOT / "scripts" / "ws2_p2p_nccl_attention_reference_check.py"
    p2p_report = artifact_dir / "ws2-p2p-nccl-tp2-cp2.json"
    custom_ag_rs_report = artifact_dir / "ws2-custom-cuda-ag-rs-tp2-cp2.json"
    p2p_available = p2p_script.is_file()
    p2p_unavailable = (
        None
        if p2p_available
        else "three-stage Attention communication check is absent; integrate #279"
    )
    collective_script = REPO_ROOT / "scripts" / "ws2_deterministic_collective_attention_check.py"
    collective_report = artifact_dir / "ws2-self-owned-attention-collectives.json"
    collective_available = collective_script.is_file()
    collective_unavailable = (
        None if collective_available else "self-owned deterministic collective check is absent"
    )
    te_compare_script = REPO_ROOT / "scripts" / "ws2_megatron_te_cp_compare.py"
    te_inputs = (
        args.megatron_te_script,
        args.megatron_model,
        args.megatron_token_artifact,
    )
    te_available = te_compare_script.is_file() and all(
        path is not None and path.exists() for path in te_inputs
    )
    te_unavailable = (
        None
        if te_available
        else "native Megatron/TE comparison requires --megatron-te-script, "
        "--megatron-model, and --megatron-token-artifact"
    )
    te_report = artifact_dir / "ws2-megatron-te-cp-compare.json"

    cases: list[AcceptanceCase] = [
        AcceptanceCase(
            name="pr5_cp_forward_backward_dlogp",
            command=(
                python,
                str(REPO_ROOT / "benchmarks" / "benchmark_ws2_cp_attention_drift.py"),
                "--device",
                "cuda",
                "--tp-world-sizes",
                "2",
                "--cp-world-sizes",
                "2",
                "--kv-chunk-sizes",
                "none,4",
                "--include-backward",
                "--include-dlogp",
                "--output",
                str(pr5_report),
            ),
            report_path=pr5_report,
            validator=lambda report: validate_pr5_report(report, args),
        ),
        AcceptanceCase(
            name="p2p_nccl_reference",
            command=(
                (
                    torchrun,
                    "--standalone",
                    "--nproc-per-node=4",
                    str(p2p_script),
                    "--transport",
                    "p2p_nccl_reference",
                    "--repeats",
                    "3",
                    "--atol",
                    str(args.out_atol),
                    "--output",
                    str(p2p_report),
                )
                if p2p_available
                else None
            ),
            report_path=p2p_report,
            validator=lambda report: validate_p2p_report(
                report,
                expected_transport="p2p_nccl_reference",
                expected_world_size=4,
            ),
            unavailable_reason=p2p_unavailable,
        ),
        AcceptanceCase(
            name="native_te_kv_ring_cp_compare",
            # TE's native KV ring is a diagnostic/performance baseline.  It
            # currently exposes CP-dependent drift and must not gate the
            # self-owned AG/RS acceptance path.
            required=False,
            command=(
                (
                    python,
                    str(te_compare_script),
                    "--teacher-script",
                    str(args.megatron_te_script),
                    "--model",
                    str(args.megatron_model),
                    "--token-artifact",
                    str(args.megatron_token_artifact),
                    "--output-dir",
                    str(artifact_dir / "megatron-te-cp-runs"),
                    "--output",
                    str(te_report),
                    "--python",
                    str(args.megatron_python),
                    "--cp-comm-type",
                    "p2p",
                )
                if te_available
                else None
            ),
            report_path=te_report,
            validator=lambda report: validate_native_te_report(report, args),
            unavailable_reason=te_unavailable,
        ),
    ]
    for name, mode, query_len, policy, fixed_size in (
        ("decode-disabled", "decode", 1, "disabled", None),
        ("decode-fixed", "decode", 1, "fixed", 4),
        ("prefill-disabled", "prefill", 4, "disabled", None),
        ("prefill-fixed", "prefill", 4, "fixed", 4),
    ):
        command = [
            python,
            str(pr7_script),
            "--no-dry-run",
            "--device",
            "cuda",
            "--mode",
            mode,
            "--query-len",
            str(query_len),
            "--split-kv-policy",
            policy,
            "--output",
            str(pr7_reports[name]),
        ]
        if fixed_size is not None:
            command.extend(("--fixed-split-size", str(fixed_size)))

        def pr7_validator(report: Mapping[str, Any], expected_policy: str = policy) -> list[str]:
            return validate_pr7_report(
                report,
                args,
                expected_policy=expected_policy,
            )

        cases.append(
            AcceptanceCase(
                name=f"pr7_flashinfer_{name.replace('-', '_')}",
                command=tuple(command) if pr7_available else None,
                report_path=pr7_reports[name],
                validator=pr7_validator,
                unavailable_reason=pr7_unavailable,
            )
        )
    cases.append(
        AcceptanceCase(
            name="custom_cuda_ag_rs",
            command=(
                (
                    torchrun,
                    "--standalone",
                    "--nproc-per-node=4",
                    str(p2p_script),
                    "--transport",
                    "cuda_ag_rs",
                    "--repeats",
                    "3",
                    "--atol",
                    str(args.out_atol),
                    "--output",
                    str(custom_ag_rs_report),
                )
                if p2p_available
                else None
            ),
            report_path=custom_ag_rs_report,
            validator=lambda report: validate_p2p_report(
                report,
                expected_transport="cuda_ag_rs",
                expected_world_size=4,
            ),
            unavailable_reason=p2p_unavailable,
        )
    )
    cases.append(
        AcceptanceCase(
            name="custom_cuda_allreduce",
            required=False,
            command=(
                (
                    torchrun,
                    "--standalone",
                    f"--nproc-per-node={args.collective_world_size}",
                    str(collective_script),
                    "--output",
                    str(collective_report),
                )
                if collective_available
                else None
            ),
            report_path=collective_report,
            validator=lambda report: validate_collective_report(
                report, args, operation="allreduce"
            ),
            unavailable_reason=collective_unavailable,
        )
    )
    return tuple(cases)


def run_acceptance(
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if args.timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    rows: list[dict[str, Any]] = []
    for case in build_acceptance_cases(args):
        rows.append(_run_case(case, args, runner=runner))
    failed_required = [row["name"] for row in rows if row["required"] and not row["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 235,
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "mode": args.mode,
        "status": "passed" if not failed_required else "failed",
        "passed": not failed_required,
        "failed_required_cases": failed_required,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "image": args.image,
            "head_sha": args.head_sha,
            "command": " ".join(shlex.quote(item) for item in sys.argv),
        },
        "thresholds": {
            "out_max_abs": args.out_atol,
            "lse_max_abs": args.lse_atol,
            "dlogp_max_abs": args.dlogp_atol,
            "gradient_max_abs": args.grad_atol,
        },
        "required_matrix": {
            "topology": "Qwen3-8B TP=2 CP=2 BF16",
            "attention_modes": ["prefill", "chunked_prefill", "paged_prefill", "decode"],
            "split_kv": ["disabled", "fixed", "auto_diagnostic_only"],
            "outputs": ["out", "attention_lse", "active_token_dlogp", "dq", "dk", "dv"],
            "invariance": [
                "batch_composition",
                "query_position",
                "physical_page_order",
                "prefix_cache_identity",
                "global_block_merge_order",
            ],
            "communication": [
                "p2p_nccl_reference",
                "self_owned_cuda_ag_rs",
                "self_owned_cuda_allreduce",
            ],
        },
        "cases": rows,
    }


def _run_case(
    case: AcceptanceCase,
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": case.name,
        "required": case.required,
        "command": None if case.command is None else list(case.command),
        "report_path": None if case.report_path is None else str(case.report_path),
        "status": "pending",
        "passed": False,
        "errors": [],
    }
    if case.command is None:
        row.update(status="unavailable")
        row["errors"] = [case.unavailable_reason or "no executable implementation"]
        return row
    if args.mode == "manifest":
        row.update(status="not_run")
        row["errors"] = ["manifest mode does not execute GPU validation"]
        return row
    if case.report_path is not None:
        case.report_path.parent.mkdir(parents=True, exist_ok=True)
        case.report_path.unlink(missing_ok=True)
    try:
        completed = runner(
            list(case.command),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        row.update(status="execution_error")
        row["errors"] = [str(exc)]
        return row
    row["returncode"] = completed.returncode
    row["stdout_tail"] = completed.stdout[-4000:]
    row["stderr_tail"] = completed.stderr[-4000:]
    if completed.returncode != 0:
        if case.report_path is not None:
            try:
                unavailable_report = json.loads(case.report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                unavailable_report = None
            if (
                isinstance(unavailable_report, dict)
                and unavailable_report.get("status") == "not_available"
            ):
                row.update(status="not_available")
                row["errors"] = list(unavailable_report.get("errors") or []) or [
                    f"command exited with {completed.returncode}"
                ]
                row["report_summary"] = _report_summary(unavailable_report)
                return row
        row.update(status="failed")
        row["errors"] = [f"command exited with {completed.returncode}"]
        return row
    try:
        if case.report_path is not None:
            report = json.loads(case.report_path.read_text(encoding="utf-8"))
        else:
            report = _last_json_document(completed.stdout)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        row.update(status="invalid_report")
        row["errors"] = [str(exc)]
        return row
    errors = [] if case.validator is None else case.validator(report)
    row["errors"] = errors
    row["status"] = "passed" if not errors else "failed"
    row["passed"] = not errors
    row["report_summary"] = _report_summary(report)
    return row


def validate_native_te_report(report: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    """Validate the delegated native Megatron/TE CP=1 vs CP=2 run."""

    errors: list[str] = []
    if report.get("schema_version") != "ws2_megatron_te_cp_compare/v1":
        errors.append("native TE report schema is invalid")
    if report.get("transport") != "native_te_kv_ring":
        errors.append("native TE report did not use cp_comm_type=p2p")
    if report.get("status") != "passed" or report.get("passed") is not True:
        report_errors = report.get("errors")
        if isinstance(report_errors, list):
            errors.extend(str(error) for error in report_errors)
        else:
            errors.append("native TE report did not provide structured errors")
    requested = report.get("requested")
    if not isinstance(requested, Mapping):
        errors.append("native TE request metadata is missing")
        requested = {}
    if requested.get("cp_comm_type") != "p2p":
        errors.append("native TE request did not use cp_comm_type=p2p")
    if requested.get("context_parallel_sizes") != [1, 2]:
        errors.append("native TE comparison must cover CP=1 and CP=2")
    comparison = report.get("comparison")
    if not isinstance(comparison, Mapping):
        errors.append("native TE report is missing CP comparison")
        comparison_hash = None
    else:
        if comparison.get("pass") is not True:
            errors.append("native TE CP comparison did not pass")
        if comparison.get("left_cp_size") != 1 or comparison.get("right_cp_size") != 2:
            errors.append("native TE comparison order is not CP=1 then CP=2")
        errors.extend(
            _scalar_threshold_errors(
                comparison.get("max_abs"),
                args.dlogp_atol,
                "native TE CP logprob drift",
            )
        )
        comparison_hash = comparison.get("token_ids_sha256")
        if not (
            isinstance(comparison_hash, str)
            and len(comparison_hash) == 64
            and all(character in "0123456789abcdef" for character in comparison_hash)
        ):
            errors.append("native TE comparison token hash is invalid")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        errors.append("native TE report must contain exactly two runs")
        return errors
    seen_cp_sizes: set[int] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            errors.append(f"native TE run {index} is invalid")
            continue
        cp_size = run.get("cp_size")
        if isinstance(cp_size, int) and not isinstance(cp_size, bool):
            seen_cp_sizes.add(cp_size)
        if run.get("status") != "passed":
            errors.append(f"native TE CP={cp_size} run did not pass")
        active_token_count = run.get("active_token_count")
        if (
            isinstance(active_token_count, bool)
            or not isinstance(active_token_count, int)
            or active_token_count < 1
        ):
            errors.append(f"native TE CP={cp_size} active-token evidence is missing")
        if run.get("token_ids_sha256") != comparison_hash:
            errors.append(f"native TE CP={cp_size} token hash differs from the comparison")
        actual = run.get("actual")
        provider = actual.get("provider") if isinstance(actual, Mapping) else None
        if not isinstance(provider, Mapping):
            provider = {}
        if provider.get("transformer_impl") != "transformer_engine":
            errors.append("native TE run did not record transformer_engine")
        if provider.get("cp_comm_type") != "p2p":
            errors.append("native TE run did not record cp_comm_type=p2p")
    if seen_cp_sizes != {1, 2}:
        errors.append("native TE runs must cover CP=1 and CP=2 exactly")
    return errors


def validate_pr5_report(report: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != "ws2_cp_attention_drift/v2":
        errors.append("PR5 report schema is not ws2_cp_attention_drift/v2")
    if report.get("issue") != 235 or report.get("pr") != 5:
        errors.append("PR5 report identity is not issue #235 PR5")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or not str(runtime.get("device", "")).startswith("cuda"):
        errors.append("PR5 report was not produced on CUDA")
    target = report.get("target")
    if not isinstance(target, dict):
        errors.append("PR5 target metadata is missing")
    else:
        if target.get("model") != "qwen3-8b" or target.get("dtype") != "bf16":
            errors.append("PR5 target must be Qwen3-8B BF16")
        if target.get("global_num_query_heads") != 32:
            errors.append("PR5 target query-head count must be 32")
        if target.get("global_num_kv_heads") != 8 or target.get("head_dim") != 128:
            errors.append("PR5 target KV-head/head-dim metadata is invalid")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("PR5 report has no cases")
        return errors
    expected_modes = {"prefill", "chunked_prefill"}
    actual_modes = {case.get("attention_mode") for case in cases if isinstance(case, dict)}
    if not expected_modes.issubset(actual_modes):
        errors.append("PR5 report must contain prefill and chunked_prefill")
    actual_policies = {
        case.get("provenance", {}).get("requested_split_kv_policy")
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("provenance"), dict)
    }
    if not {"disabled", "fixed"}.issubset(actual_policies):
        errors.append("PR5 report must contain disabled and fixed Split-KV")
    for case in cases:
        if not isinstance(case, dict):
            errors.append("PR5 case must be an object")
            continue
        topology = case.get("topology", {})
        if topology.get("tp_world_size") != 2 or topology.get("cp_world_size") != 2:
            errors.append(f"{case.get('case_name')}: topology is not TP=2 CP=2")
        provenance = case.get("provenance", {})
        if provenance.get("rope", {}).get("rope_state") != "post_rope":
            errors.append(f"{case.get('case_name')}: RoPE was not composed before Attention")
        requested_policy = provenance.get("requested_split_kv_policy")
        requested_size = provenance.get("requested_split_kv_size")
        if requested_policy == "disabled" and requested_size is not None:
            errors.append(f"{case.get('case_name')}: disabled Split-KV has a split size")
        if requested_policy == "fixed" and not isinstance(requested_size, int):
            errors.append(f"{case.get('case_name')}: fixed Split-KV lacks an integer size")
        plan_set = provenance.get("actual_split_kv_plan_set")
        errors.extend(
            _validate_runtime_plan_set(
                plan_set,
                expected_batch=_report_positive_int(target, "batch"),
                expected_tp=2,
                expected_cp=2,
                expected_policy=requested_policy,
                label=f"{case.get('case_name')}.actual_split_kv_plan_set",
            )
        )
        drift = case.get("drift", {}).get("cp_merge_fp32", {})
        errors.extend(
            _threshold_errors(
                drift.get("out"),
                args.out_atol,
                f"{case.get('case_name')}.out",
            )
        )
        errors.extend(
            _threshold_errors(
                drift.get("lse"),
                args.lse_atol,
                f"{case.get('case_name')}.lse",
            )
        )
        dlogp = case.get("dlogp", {})
        if dlogp.get("status") != "available":
            errors.append(f"{case.get('case_name')}: active-token dlogp is unavailable")
        else:
            errors.extend(
                _threshold_errors(
                    dlogp.get("drift"),
                    args.dlogp_atol,
                    f"{case.get('case_name')}.dlogp",
                )
            )
        backward = case.get("backward", {})
        if backward.get("status") != "available":
            errors.append(f"{case.get('case_name')}: backward drift is unavailable")
        else:
            backward_drifts = backward.get("report", {}).get("drifts")
            if not isinstance(backward_drifts, list) or not backward_drifts:
                errors.append(f"{case.get('case_name')}: backward drift rows are missing")
                continue
            for item in backward_drifts:
                if not isinstance(item, dict):
                    errors.append(f"{case.get('case_name')}: backward drift row is invalid")
                    continue
                for name in ("dq", "dk", "dv"):
                    errors.extend(
                        _threshold_errors(
                            item.get(name),
                            args.grad_atol,
                            f"{case.get('case_name')}.{name}",
                        )
                    )
    return errors


def validate_pr7_report(
    report: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    expected_policy: str,
) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed" or report.get("passed") is not True:
        errors.append("PR7 report is not an executed pass")
    provenance = report.get("candidate_provenance")
    if not isinstance(provenance, dict):
        errors.append("PR7 report lacks candidate runtime provenance")
        return errors
    if provenance.get("arithmetic_semantics_verified") is not True:
        errors.append("PR7 arithmetic semantics are not runtime-verified")
    plans = provenance.get("actual_split_kv_plans")
    if not isinstance(plans, list) or not plans:
        errors.append("PR7 actual Split-K plans are missing")
    else:
        for plan in plans:
            if plan.get("actual_split_kv_policy") != expected_policy:
                errors.append("PR7 actual Split-K policy differs from the requested policy")
            if not plan.get("actual_split_boundaries"):
                errors.append("PR7 actual Split-K boundaries are missing")
    plan_set = provenance.get("actual_split_kv_plan_set")
    shape = report.get("shape", {})
    errors.extend(
        _validate_runtime_plan_set(
            plan_set,
            expected_batch=_report_positive_int(shape, "batch_size"),
            expected_tp=2,
            expected_cp=2,
            expected_policy=expected_policy,
            label="PR7 actual Split-KV plan set",
        )
    )
    drift = report.get("drift", {})
    errors.extend(_threshold_errors(drift.get("out"), args.out_atol, "PR7.out"))
    errors.extend(_threshold_errors(drift.get("lse"), args.lse_atol, "PR7.lse"))
    errors.extend(_threshold_errors(drift.get("dlogp"), args.dlogp_atol, "PR7.dlogp"))
    for key in ("batch_invariant_sweep", "page_layout_invariant_sweep"):
        sweep = report.get(key)
        if not isinstance(sweep, dict) or sweep.get("passed") is not True:
            errors.append(f"PR7 {key} did not pass")
    return errors


def validate_p2p_report(
    report: Mapping[str, Any],
    *,
    expected_transport: str = "p2p_nccl_reference",
    expected_world_size: int | None = None,
) -> list[str]:
    """Validate the real three-stage Attention communication report.

    World size 2 is retained for old artifacts.  The acceptance gate passes
    ``expected_world_size=4`` so a legacy two-rank report cannot accidentally
    satisfy the formal TP=2, CP=2 gate.
    """

    errors: list[str] = []
    if expected_transport not in {"p2p_nccl_reference", "cuda_ag_rs"}:
        return [f"unsupported P2P transport expectation: {expected_transport}"]
    expected_schema = (
        "ws2_p2p_nccl_attention_reference/v1"
        if expected_transport == "p2p_nccl_reference"
        else "ws2_cuda_ag_rs_attention/v1"
    )
    if report.get("schema_version") != expected_schema:
        errors.append("P2P report schema is invalid")
    if report.get("transport") != expected_transport:
        errors.append(f"P2P report did not use {expected_transport}")
    if "nccl" not in str(report.get("backend", "")).lower():
        errors.append("P2P report backend is not NCCL")
    world_size = report.get("world_size")
    if not isinstance(world_size, int) or world_size not in {2, 4}:
        errors.append("P2P report world size must be the legacy 2 or formal 4")
        return errors
    if expected_world_size is not None and world_size != expected_world_size:
        errors.append(f"P2P report world size is not {expected_world_size}")
    if report.get("tp_world_size") != (1 if world_size == 2 else 2):
        errors.append("P2P report TP world size is inconsistent with the rank topology")
    if report.get("cp_world_size") != 2:
        errors.append("P2P report CP world size is not 2")
    if report.get("sp_world_size") != 1:
        errors.append("P2P report SP world size must be 1 for the Attention gate")
    if report.get("global_failure_count") != 0:
        errors.append("P2P report has global rank failures")
    ranks = report.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        errors.append(f"P2P report must contain exactly {world_size} rank reports")
        return errors

    seen_ranks: set[int] = set()
    seen_coords: set[tuple[int, int]] = set()
    query_ranges_by_tp: dict[int, dict[int, list[int]]] = {}
    manifests_by_tp: dict[int, list[list[Any]]] = {}
    for index, row in enumerate(ranks):
        if not isinstance(row, dict):
            errors.append(f"P2P rank {index} report is invalid")
            continue
        rank = row.get("rank")
        tp_rank = row.get("tp_rank")
        cp_rank = row.get("cp_rank")
        if not isinstance(rank, int):
            errors.append(f"P2P row {index} lacks an integer rank")
            continue
        seen_ranks.add(rank)
        if rank < 0 or rank >= world_size:
            errors.append(f"P2P rank {index} is outside the world")
        expected_tp = 0 if world_size == 2 else rank // 2
        expected_cp = rank % 2
        if (tp_rank, cp_rank) != (expected_tp, expected_cp):
            errors.append(f"P2P rank {index} TP/CP coordinates are inconsistent with rank order")
        if isinstance(tp_rank, int) and isinstance(cp_rank, int):
            seen_coords.add((tp_rank, cp_rank))
        if row.get("global_world_size") != world_size:
            errors.append(f"P2P rank {index} global world size is inconsistent")
        if row.get("global_failure_count") != 0:
            errors.append(f"P2P rank {index} observed global rank failures")
        if row.get("passed") is not True:
            errors.append(f"P2P rank {index} did not pass")
        if row.get("transport") != expected_transport:
            errors.append(f"P2P rank {index} did not use {expected_transport}")
        if row.get("query_ag") != expected_transport:
            errors.append(f"P2P rank {index} did not execute the expected Q AllGather")
        if row.get("protocol") != "ag_query_local_kv_rs_out_lse":
            errors.append(f"P2P rank {index} did not execute the three-stage protocol")
        if row.get("query_ag_max_abs") != 0.0:
            errors.append(f"P2P rank {index} Q AllGather was not bitwise exact")
        if row.get("dtype") != "bf16" or row.get("accum_dtype") != "fp32":
            errors.append(f"P2P rank {index} arithmetic provenance is invalid")
        if row.get("downcast_at") != "final_write":
            errors.append(f"P2P rank {index} downcast provenance is invalid")
        if row.get("final_output_dtype") != "bfloat16":
            errors.append(f"P2P rank {index} final output dtype is not BF16")
        if not str(row.get("device", "")).startswith("cuda"):
            errors.append(f"P2P rank {index} was not executed on CUDA")
        if not isinstance(row.get("repeat_count"), int) or row["repeat_count"] < 2:
            errors.append(f"P2P rank {index} repeat count is insufficient")
        for repeat_name in (
            "repeat_query_bitwise",
            "repeat_out_bitwise",
            "repeat_lse_bitwise",
            "repeat_manifest_bitwise",
        ):
            if row.get(repeat_name) is not True:
                errors.append(f"P2P rank {index} {repeat_name} did not pass")

        query_range = row.get("query_range")
        if not (
            isinstance(query_range, list)
            and len(query_range) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in query_range)
            and query_range[0] < query_range[1]
        ):
            errors.append(f"P2P rank {index} query ownership is invalid")
        elif isinstance(tp_rank, int) and isinstance(cp_rank, int):
            query_ranges_by_tp.setdefault(tp_rank, {})[cp_rank] = query_range

        gathered_indices = row.get("gathered_block_indices")
        block_manifest = row.get("expected_block_manifest")
        manifest_errors, manifest_indices = _validate_p2p_block_manifest(
            block_manifest,
            expected_tp_rank=tp_rank if isinstance(tp_rank, int) else None,
        )
        errors.extend(f"P2P rank {index}: {error}" for error in manifest_errors)
        if not (
            isinstance(gathered_indices, list)
            and gathered_indices
            and gathered_indices == list(range(len(gathered_indices)))
            and manifest_indices == gathered_indices
        ):
            errors.append(f"P2P rank {index} gathered block order/coverage is invalid")
        if (
            isinstance(tp_rank, int)
            and isinstance(cp_rank, int)
            and isinstance(manifest_indices, list)
            and isinstance(block_manifest, list)
        ):
            local_indices = row.get("local_block_indices")
            expected_local = [
                block_index
                for block_index, block in enumerate(block_manifest or [])
                if isinstance(block, Mapping) and block.get("owner_cp_rank") == cp_rank
            ]
            if local_indices != expected_local:
                errors.append(f"P2P rank {index} local block ownership is invalid")
            manifests_by_tp.setdefault(tp_rank, []).append(block_manifest)
        for name in ("out_max_abs", "lse_max_abs"):
            errors.extend(
                _scalar_threshold_errors(row.get(name), row.get("atol"), f"P2P rank {index}.{name}")
            )
        errors.extend(
            _scalar_threshold_errors(
                row.get("final_out_max_abs"),
                row.get("final_write_atol"),
                f"P2P rank {index}.final_out_max_abs",
            )
        )

    expected_ranks = set(range(world_size))
    if seen_ranks != expected_ranks:
        errors.append(f"P2P report must cover ranks 0 through {world_size - 1} exactly")
    expected_coords = (
        {(0, cp) for cp in range(2)}
        if world_size == 2
        else {(tp, cp) for tp in range(2) for cp in range(2)}
    )
    if seen_coords != expected_coords:
        errors.append("P2P report must cover the canonical TP/CP coordinate grid")

    for tp_rank in range(1 if world_size == 2 else 2):
        ranges = query_ranges_by_tp.get(tp_rank, {})
        first_range = ranges.get(0)
        second_range = ranges.get(1)
        if (
            first_range is None
            or second_range is None
            or first_range[0] != 0
            or first_range[1] != second_range[0]
            or second_range[1] <= second_range[0]
        ):
            errors.append(f"P2P TP group {tp_rank} query ownership is not canonical and contiguous")
        if tp_rank > 0 and ranges != query_ranges_by_tp.get(0, {}):
            errors.append("P2P TP groups have different query ownership ranges")
        manifests = manifests_by_tp.get(tp_rank, [])
        if len(manifests) != 2:
            errors.append(f"P2P TP group {tp_rank} must contain both CP rank manifests")
        elif manifests[0] != manifests[1]:
            errors.append(f"P2P TP group {tp_rank} gathered different logical block manifests")
    return errors


def validate_collective_report(
    report: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    operation: str,
) -> list[str]:
    """Validate executed PR310/311/312 evidence, never configured-only claims."""

    errors: list[str] = []
    if report.get("schema_version") != "ws2_deterministic_attention_collectives/v1":
        errors.append("self-owned collective report schema is invalid")
    if report.get("world_size") != args.collective_world_size:
        errors.append("self-owned collective report world size differs from the requested size")
    if report.get("transport") != "self_owned_cuda_ag_rs":
        errors.append("self-owned report did not execute the CUDA AG/RS backend")
    if report.get("allreduce_transport") != "self_owned_cuda_allreduce":
        errors.append("self-owned report did not execute the CUDA AllReduce backend")
    if report.get("global_failure_count") != 0 or report.get("passed") is not True:
        errors.append("self-owned collective report contains rank failures")
    ranks = report.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != args.collective_world_size:
        errors.append("self-owned collective report must contain every rank")
        return errors
    required = {
        "ag_rs": ("all_gather_q", "reduce_scatter_out_lse"),
        "allreduce": ("all_reduce_o_proj",),
    }[operation]
    for index, row in enumerate(ranks):
        if not isinstance(row, Mapping):
            errors.append(f"self-owned rank {index} report is invalid")
            continue
        if row.get("passed") is not True:
            errors.append(f"self-owned rank {index} did not pass")
        operations = row.get("operations")
        if not isinstance(operations, Mapping):
            errors.append(f"self-owned rank {index} operation evidence is missing")
            continue
        for name in required:
            evidence = operations.get(name)
            if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
                errors.append(f"self-owned rank {index} {name} evidence did not pass")
    return errors


def _validate_p2p_block_manifest(
    manifest: Any,
    *,
    expected_tp_rank: int | None,
) -> tuple[list[str], list[int] | None]:
    if not isinstance(manifest, list) or not manifest:
        return ["expected block manifest is missing"], None
    required = {
        "global_block_index",
        "kv_block_start",
        "kv_block_end",
        "owner_cp_rank",
        "owner_tp_rank",
    }
    errors: list[str] = []
    indices: list[int] = []
    cursor = 0
    owners: set[int] = set()
    for index, block in enumerate(manifest):
        if not isinstance(block, dict) or not required.issubset(block):
            errors.append(f"manifest block {index} is missing required metadata")
            continue
        values = {name: block[name] for name in required}
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in values.values()
        ):
            errors.append(f"manifest block {index} metadata must contain integers")
            continue
        global_index = values["global_block_index"]
        start = values["kv_block_start"]
        end = values["kv_block_end"]
        owner_cp_rank = values["owner_cp_rank"]
        owner_tp_rank = values["owner_tp_rank"]
        indices.append(global_index)
        owners.add(owner_cp_rank)
        if global_index != index:
            errors.append(f"manifest block {index} has a non-canonical global index")
        if start != cursor or end <= start:
            errors.append(f"manifest block {index} does not preserve gap-free KV coverage")
        cursor = end
        if owner_cp_rank not in {0, 1} or (
            expected_tp_rank is not None and owner_tp_rank != expected_tp_rank
        ):
            errors.append(f"manifest block {index} owner is outside the TP-local CP=2 group")
    if owners != {0, 1}:
        errors.append("manifest does not assign KV blocks to both CP ranks")
    return errors, indices


def _threshold_errors(stats: Any, threshold: float, label: str) -> list[str]:
    if not isinstance(stats, dict) or "max_abs" not in stats:
        return [f"{label} drift is missing"]
    try:
        value = float(stats["max_abs"])
    except (TypeError, ValueError):
        return [f"{label} max_abs is not numeric"]
    if not math.isfinite(value) or value < 0:
        return [f"{label} max_abs must be finite and non-negative"]
    return [] if value <= threshold else [f"{label} max_abs={value} exceeds {threshold}"]


def _scalar_threshold_errors(value: Any, threshold: Any, label: str) -> list[str]:
    try:
        numeric_value = float(value)
        numeric_threshold = float(threshold)
    except (TypeError, ValueError):
        return [f"{label} or its threshold is not numeric"]
    if not math.isfinite(numeric_value) or numeric_value < 0:
        return [f"{label} must be finite and non-negative"]
    if not math.isfinite(numeric_threshold) or numeric_threshold < 0:
        return [f"{label} threshold must be finite and non-negative"]
    if numeric_value > numeric_threshold:
        return [f"{label}={numeric_value} exceeds {numeric_threshold}"]
    return []


def _validate_runtime_plan_set(
    plan_set: Any,
    *,
    expected_batch: int,
    expected_tp: int,
    expected_cp: int,
    expected_policy: Any,
    label: str,
) -> list[str]:
    if expected_batch < 1:
        return [f"{label} expected batch size is invalid"]
    if not isinstance(plan_set, dict):
        return [f"{label} is missing"]
    errors: list[str] = []
    if plan_set.get("coverage") != "complete_batch_tp_cp_owner_cartesian_product":
        errors.append(f"{label} coverage marker is invalid")
    topology = (
        plan_set.get("batch_size"),
        plan_set.get("tp_world_size"),
        plan_set.get("cp_world_size"),
    )
    expected_topology = (expected_batch, expected_tp, expected_cp)
    if topology != expected_topology:
        errors.append(f"{label} topology {topology} does not match {expected_topology}")
    totals = plan_set.get("total_kv_tokens")
    if not (
        isinstance(totals, list)
        and len(totals) == expected_batch
        and all(
            isinstance(total, int) and not isinstance(total, bool) and total > 0 for total in totals
        )
    ):
        errors.append(f"{label} total_kv_tokens is invalid")
        return errors
    entries = plan_set.get("entries")
    expected_coordinates = {
        (batch_index, tp_rank, cp_rank, owner_cp_rank)
        for batch_index in range(expected_batch)
        for tp_rank in range(expected_tp)
        for cp_rank in range(expected_cp)
        for owner_cp_rank in range(expected_cp)
    }
    if not isinstance(entries, list):
        errors.append(f"{label} entries are missing")
        return errors
    coordinates: list[tuple[Any, Any, Any, Any]] = []
    owner_ranges: dict[tuple[int, int, int], tuple[int, int]] = {}
    for index, entry in enumerate(entries):
        entry_label = f"{label}.entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label} is not an object")
            continue
        coordinate_values = tuple(
            entry.get(key) for key in ("batch_index", "tp_rank", "cp_rank", "owner_cp_rank")
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in coordinate_values
        ):
            errors.append(f"{entry_label} coordinate must contain integers")
            continue
        coordinate: tuple[int, int, int, int] = (
            cast(int, coordinate_values[0]),
            cast(int, coordinate_values[1]),
            cast(int, coordinate_values[2]),
            cast(int, coordinate_values[3]),
        )
        coordinates.append(coordinate)
        if coordinate not in expected_coordinates:
            errors.append(f"{entry_label} coordinate is out of range")
            continue
        batch_index, tp_rank, _, owner_cp_rank = coordinate
        expected_range = entry.get("expected_kv_range")
        if not (
            isinstance(expected_range, list)
            and len(expected_range) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) for value in expected_range
            )
            and 0 <= expected_range[0] < expected_range[1] <= totals[batch_index]
        ):
            errors.append(f"{entry_label} expected_kv_range is invalid")
            continue
        range_key = (batch_index, tp_rank, owner_cp_rank)
        range_tuple = (expected_range[0], expected_range[1])
        previous_range = owner_ranges.setdefault(range_key, range_tuple)
        if previous_range != range_tuple:
            errors.append(f"{entry_label} owner range differs across CP consumers")
        if entry.get("requested_split_kv_policy") != expected_policy:
            errors.append(f"{entry_label} requested Split-KV policy is wrong")
        if entry.get("actual_split_kv_policy") != expected_policy:
            errors.append(f"{entry_label} actual Split-KV policy is wrong")
        if entry.get("split_kv_merge_order") != "global_block_index":
            errors.append(f"{entry_label} merge order is not global_block_index")
        if entry.get("split_kv_accum_dtype") != "fp32":
            errors.append(f"{entry_label} accumulation dtype is not fp32")
        if entry.get("split_kv_downcast_at") != "final_write":
            errors.append(f"{entry_label} downcast point is not final_write")
        if entry.get("split_kv_fallback") is not False:
            errors.append(f"{entry_label} used a fallback")
        if not isinstance(entry.get("split_kv_plan_source"), str):
            errors.append(f"{entry_label} runtime plan source is missing")
        boundaries = entry.get("actual_split_boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            errors.append(f"{entry_label} actual split boundaries are missing")
            continue
        cursor = expected_range[0]
        valid_boundaries = True
        for boundary in boundaries:
            if not (
                isinstance(boundary, list)
                and len(boundary) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool) for value in boundary
                )
                and boundary[0] == cursor
                and boundary[0] < boundary[1] <= expected_range[1]
            ):
                valid_boundaries = False
                break
            cursor = boundary[1]
        if not valid_boundaries or cursor != expected_range[1]:
            errors.append(f"{entry_label} boundaries do not cover the owner range exactly")
        if entry.get("actual_split_kv_count") != len(boundaries):
            errors.append(f"{entry_label} actual split count is inconsistent")
    if len(coordinates) != len(set(coordinates)):
        errors.append(f"{label} contains duplicate coordinates")
    actual_coordinates = set(coordinates)
    if actual_coordinates != expected_coordinates:
        errors.append(f"{label} coordinate coverage is incomplete")
    for batch_index in range(expected_batch):
        for tp_rank in range(expected_tp):
            cursor = 0
            for owner_cp_rank in range(expected_cp):
                owner_range = owner_ranges.get((batch_index, tp_rank, owner_cp_rank))
                if owner_range is None or owner_range[0] != cursor:
                    errors.append(
                        f"{label} owner ranges are not contiguous for "
                        f"batch={batch_index}, tp={tp_rank}"
                    )
                    break
                cursor = owner_range[1]
            if cursor != totals[batch_index]:
                errors.append(
                    f"{label} owner ranges do not cover total KV for "
                    f"batch={batch_index}, tp={tp_rank}"
                )
    return errors


def _report_positive_int(container: Any, key: str) -> int:
    if not isinstance(container, dict):
        return 0
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 0
    return value


def _last_json_document(stdout: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if stdout[index + end :].strip() or not isinstance(value, dict):
            continue
        return value
    raise ValueError("command stdout does not end with a JSON object")


def _report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in ("schema_version", "status", "passed", "issue", "pr", "mode")
        if key in report
    }


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_acceptance(args)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
