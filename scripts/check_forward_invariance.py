#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run the WS1 C3 selected-logprob forward invariance gate on a real GPU."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest import (  # noqa: E402
    BackendProvenance,
    ConfigSpec,
    RuntimeObservation,
    assert_forward_batch_invariant,
    load_contract,
    normalize_dtype_name,
)
from rl_engine.kernels.gtest.operator_specs import OP_SPECS, _load_object  # noqa: E402
from rl_engine.kernels.gtest.tolerance import resolve_dtype_policy  # noqa: E402
from rl_engine.testing.ws1_workload import PaddedBatch, load_manifest  # noqa: E402


def _object_path(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _profile_node(manifest: Any, profile: str, op_name: str) -> dict[str, Any]:
    node_name = "logprob" if op_name == "logp" else op_name
    nodes = manifest.backend_profiles[profile]["required_nodes"]
    node = next((dict(item) for item in nodes if item["node"] == node_name), None)
    if node is None:
        raise RuntimeError(f"profile {profile!r} does not declare node {node_name!r}")
    if node["status"] != "declared":
        raise RuntimeError(
            f"profile {profile!r} node {node_name!r} is {node['status']!r}; "
            "missing required candidates are red, not fallback or N/A"
        )
    return node


def _candidate_family(candidate: str) -> str:
    if candidate.startswith("cuda"):
        return "cuda"
    if candidate == "triton":
        return "triton"
    return candidate


def _validate_candidate_selection(
    *, manifest: Any, profile: str, op_name: str, candidate: str
) -> dict[str, Any]:
    node = _profile_node(manifest, profile, op_name)
    expected_family = manifest.backend_profiles[profile]["backend_family"]
    actual_family = _candidate_family(candidate)
    if actual_family != expected_family:
        raise RuntimeError(
            f"candidate {candidate!r} belongs to {actual_family!r}, but profile "
            f"{profile!r} requires {expected_family!r}"
        )
    if candidate != node["expected_backend_id"]:
        raise RuntimeError(
            f"candidate {candidate!r} does not match the C2 declaration "
            f"{node['expected_backend_id']!r} for {profile}/{node['node']}"
        )
    return node


def _physical_rows(
    config: ConfigSpec,
) -> tuple[list[tuple[str, int] | None], tuple[int, ...]]:
    layout = config.physical_layout
    if isinstance(layout, PaddedBatch):
        keys = [key for row in layout.restore_map for key in row]
        return keys, (len(layout.restore_map), layout.padded_len)
    return list(layout.restore_map), (len(layout.restore_map),)


def _make_inputs(
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row-local deterministic logits from C2 logical identity."""

    keys, leading_shape = _physical_rows(config)
    vocab_axis = torch.arange(vocab_size, device=device, dtype=torch.int64)
    rows: list[torch.Tensor] = []
    targets: list[int] = []
    token_by_key = {
        (token.sample_id, token.token_position): token.token_id
        for sample in config.logical_batch.samples
        for token in sample.tokens()
    }
    for key in keys:
        if key is None:
            position, token_id = 0, 0
        else:
            position = key[1]
            token_id = token_by_key[key]
        # Integer construction makes each logical row independent of batching,
        # chunking, permutation, padding, and RNG consumption order.
        values = ((vocab_axis + token_id * 17 + position * 13) % 257) - 128
        rows.append((values.to(torch.float32) / 1024.0).to(dtype))
        targets.append(token_id % vocab_size)
    logits = torch.stack(rows).reshape(leading_shape + (vocab_size,))
    target_tensor = torch.tensor(targets, device=device, dtype=torch.long).reshape(leading_shape)
    return logits, target_tensor


def _make_runner(
    operator: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    reference: bool,
    backend_family: str | None = None,
    kernel_id: str | None = None,
):
    def run(config: ConfigSpec, **_: Any) -> torch.Tensor:
        logits, targets = _make_inputs(config, device=device, dtype=dtype, vocab_size=vocab_size)
        if reference:
            logits = logits.float()
        output = operator(logits, targets)
        if reference:
            return output
        if backend_family is None or kernel_id is None:
            raise RuntimeError("candidate telemetry must declare backend_family and kernel_id")
        return RuntimeObservation(
            output=output,
            actual_backend=backend_family,
            kernel_id=kernel_id,
            output_dtype=normalize_dtype_name(output.dtype),
            device=str(output.device),
        )

    return run


def _summarize(report: Any) -> None:
    print(
        f"op={report.op_name} profile={report.backend_profile} "
        f"candidate={report.candidate_id} passed={report.passed}"
    )
    print(
        f"  device={report.device} cc={report.compute_capability} seed={report.seed} "
        f"provenance_valid={report.provenance_valid}"
    )
    for acc in report.accuracy_reports:
        detail = acc.details[0]
        print(
            f"  accuracy config={acc.config_id} max_abs={detail.max_abs_error:.8e} "
            f"max_rel={detail.max_rel_error:.8e} passed={acc.passed}"
        )
    for inv in report.invariance_reports:
        detail = inv.details[0]
        print(
            f"  invariance pair={detail.config_pair} transform={inv.transform_kind} "
            f"max_abs={detail.max_abs_error:.8e} passed={inv.passed}"
        )
    if report.logprob_smoke is not None:
        print(f"  selected_logprob_smoke passed={report.logprob_smoke.passed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WS1 C3 forward invariance GPU gate")
    parser.add_argument("--op", choices=("logp", "batch_invariant_logp"), default="logp")
    parser.add_argument(
        "--candidate", required=True, help="Manifest-declared CUDA/Triton candidate"
    )
    parser.add_argument(
        "--backend-profile",
        choices=("cuda_bf16", "triton_cuda_bf16"),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("ERROR: C3 required-profile evidence requires an available CUDA device")
    if args.vocab <= 240:
        raise SystemExit("ERROR: --vocab must cover every fixed C2 workload token id")

    contract = load_contract()
    manifest = load_manifest()
    node = _validate_candidate_selection(
        manifest=manifest,
        profile=args.backend_profile,
        op_name=args.op,
        candidate=args.candidate,
    )
    spec = OP_SPECS[args.op]
    if args.candidate not in spec.candidate_paths:
        raise SystemExit(f"ERROR: operator {args.op!r} has no candidate {args.candidate!r}")

    candidate_op = _load_object(spec.candidate_paths[args.candidate])()
    gold_op = _load_object(spec.gold_path)()
    gold_method = getattr(gold_op, spec.gold_method)
    policy = resolve_dtype_policy(contract)
    family = _candidate_family(args.candidate)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cc_tuple = torch.cuda.get_device_capability(device)
    cc = f"sm{cc_tuple[0]}{cc_tuple[1]}"
    if args.candidate == "cuda-sm90" and cc_tuple[0] != 9:
        raise SystemExit(
            "ERROR: cuda-sm90 candidate requested on non-SM90 hardware; fallback forbidden"
        )

    provenance = BackendProvenance(
        backend_profile=args.backend_profile,
        requested_backend=manifest.backend_profiles[args.backend_profile]["backend_family"],
        actual_backend=family,
        execution_dtype=policy.execution_dtype,
        accumulation_dtype=policy.accumulation_dtype,
        output_dtype=policy.output_dtype_default,
        reference_dtype=policy.reference_dtype,
        candidate_tf32_enabled=torch.backends.cuda.matmul.allow_tf32,
        reference_tf32_enabled=torch.backends.cuda.matmul.allow_tf32,
    )
    report = assert_forward_batch_invariant(
        _make_runner(
            candidate_op,
            device=device,
            dtype=torch.bfloat16,
            vocab_size=args.vocab,
            reference=False,
            backend_family=family,
            kernel_id=_object_path(candidate_op),
        ),
        contract=contract,
        manifest=manifest,
        backend_profile=args.backend_profile,
        provenance=provenance,
        gold_fn=_make_runner(
            gold_method,
            device=device,
            dtype=torch.bfloat16,
            vocab_size=args.vocab,
            reference=True,
        ),
        op_class="logprob",
        dtype=torch.bfloat16,
        op_name=args.op,
        candidate_id=f"{_object_path(candidate_op)}::{node['expected_kernel_config_id']}",
        device=f"{device}:{torch.cuda.get_device_name(device)}",
        compute_capability=cc,
        observed_actual_backend=family,
        observed_kernel_id=_object_path(candidate_op),
        observed_output_dtype=policy.output_dtype_default,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        _summarize(report)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
