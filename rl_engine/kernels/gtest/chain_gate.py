# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C10 / #276: full Qwen3-8B Dense model-level train–inference gate.

Batch/Chunk invariance is bitwise after logical unpadding. Logprob pass/fail
uses only max_abs_dlogp / approx_kl0 / clipfrac0. Gradients use independent
C1 rows. C9 assembly alone is not EXIT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

import torch

from rl_engine.alignment.qwen3_dense import (
    Qwen3DenseBIModel,
    Qwen3DenseSpec,
    Qwen3DenseWeights,
    load_profile_ops,
)
from rl_engine.kernels.gtest.forward_invariance import (
    TensorComparisonDetail,
    _compare_logical_tensors,
)
from rl_engine.kernels.gtest.kv_consistency import make_profile_provenance
from rl_engine.kernels.gtest.tolerance import (
    BackendProvenance,
    LogprobAggregateVerdict,
    compute_logprob_aggregates,
    default_clip_interval,
    judge_logprob_aggregates,
    load_contract,
    resolve_comparison_roles,
    resolve_dtype_policy,
    resolve_tolerance,
)
from rl_engine.testing.ws1_workload import (
    LogicalBatch,
    WS1Manifest,
    apply_packing,
    apply_padding,
    build_logical_batch,
    chunk_plan_from_manifest,
    fixture_hash,
    load_manifest,
)

PRIMARY_CELLS = (
    "B1-singleton_aggregate/full",
    "BN/full",
    "B1-singleton_aggregate/chunked",
    "BN/chunked",
)
REQUIRED_GRAD_NAMES = (
    "norm.weight",
    "lm_head.weight",
    "layers.0.input_layernorm.weight",
)
GRADIENT_SCOPE = "representative_parameter_subset"
_TRAIN_INFER_KIND = "train_infer_logprob_parity"


@dataclass(frozen=True)
class CellOutput:
    cell_id: str
    selected_logp: dict[tuple[str, int], torch.Tensor]
    loss: torch.Tensor | None
    grads: dict[str, torch.Tensor]
    first_drift_node: str | None
    node_digests: dict[str, str]
    requested_backend: str
    actual_backend: str
    node_token_digests: dict[str, dict[str, str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "n_logp_tokens": len(self.selected_logp),
            "loss": (None if self.loss is None else float(self.loss.detach().float().cpu())),
            "grad_names": sorted(self.grads),
            "first_drift_node": self.first_drift_node,
            "node_digests": dict(self.node_digests),
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
        }


@dataclass(frozen=True)
class ChainGateReport:
    backend_profile: str
    workload_id: str
    fixture_hash: str
    config_fingerprint: dict[str, Any]
    weight_source: str
    weight_hash: str
    seed: int
    workload_seed: int
    device: str
    compute_capability: str | None
    backend_provenance: BackendProvenance
    runtime_backend_observations: dict[str, dict[str, Any]]
    cells: dict[str, CellOutput]
    invariance: tuple[TensorComparisonDetail, ...]
    gradient_invariance: tuple[TensorComparisonDetail, ...]
    train_infer: LogprobAggregateVerdict | None
    first_drift: str | None
    aggregates: LogprobAggregateVerdict | None
    gradient_scope: str
    required_grad_names: tuple[str, ...]
    all_parameter_gradients: bool
    passed: bool
    backward_executed: bool
    train_infer_executed: bool
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_profile": self.backend_profile,
            "workload_id": self.workload_id,
            "fixture_hash": self.fixture_hash,
            "config_fingerprint": dict(self.config_fingerprint),
            "weight_source": self.weight_source,
            "weight_hash": self.weight_hash,
            "seed": self.seed,
            "workload_seed": self.workload_seed,
            "device": self.device,
            "compute_capability": self.compute_capability,
            "backend_provenance": self.backend_provenance.to_dict(),
            "runtime_backend_observations": {
                key: dict(value) for key, value in self.runtime_backend_observations.items()
            },
            "cells": {key: value.to_dict() for key, value in self.cells.items()},
            "invariance": [item.to_dict() for item in self.invariance],
            "gradient_invariance": [item.to_dict() for item in self.gradient_invariance],
            "train_infer": (None if self.train_infer is None else self.train_infer.to_dict()),
            "first_drift": self.first_drift,
            "aggregates": (None if self.aggregates is None else self.aggregates.to_dict()),
            "gradient_scope": self.gradient_scope,
            "required_grad_names": sorted(self.required_grad_names),
            "all_parameter_gradients": self.all_parameter_gradients,
            "passed": self.passed,
            "backward_executed": self.backward_executed,
            "train_infer_executed": self.train_infer_executed,
            "disclaimer": self.disclaimer,
        }


def build_model(
    *,
    backend_profile: str,
    weights_mode: str,
    weights_path: str | None,
    device: torch.device,
    dtype: torch.dtype,
    manifest: WS1Manifest | None = None,
    allow_pytorch_gold: bool = False,
) -> Qwen3DenseBIModel:
    m = manifest if manifest is not None else load_manifest()
    spec = Qwen3DenseSpec.from_manifest(m)
    ops = load_profile_ops(backend_profile, m, allow_pytorch_gold=allow_pytorch_gold)
    if weights_mode == "synthetic":
        weights = Qwen3DenseWeights.synthetic(spec, device=device, dtype=dtype, seed=m.seed)
    elif weights_mode in {"hf", "required"}:
        if not weights_path:
            raise RuntimeError("C10/C11 require --weights-path to the pinned Qwen3-8B snapshot")
        weights = Qwen3DenseWeights.from_hf(spec, weights_path, device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown weights_mode {weights_mode!r}")
    return Qwen3DenseBIModel(spec, weights, ops, execution_dtype=dtype)


def run_chain_gate(
    *,
    backend_profile: str,
    model: Qwen3DenseBIModel,
    contract: Mapping[str, Any] | None = None,
    manifest: WS1Manifest | None = None,
    run_backward: bool = True,
    run_train_infer: bool = True,
    padding_sides: Sequence[str] = ("right", "left"),
    execution_seed: int | None = None,
) -> ChainGateReport:
    c = contract if contract is not None else load_contract()
    m = manifest if manifest is not None else load_manifest()
    policy = resolve_dtype_policy(c)
    batch = build_logical_batch(m)
    cells: dict[str, CellOutput] = {}
    resolved_seed = m.seed if execution_seed is None else int(execution_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)

    _configure_required_gradients(model, enabled=run_backward)

    global_active = batch.active_token_count()
    cells["BN/full"] = _run_padded_cell(
        model,
        batch,
        cell_id="BN/full",
        pad_side="right",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    cells["B1-singleton_aggregate/full"] = _run_singleton_cell(
        model,
        batch,
        cell_id="B1-singleton_aggregate/full",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    cells["BN/chunked"] = _run_chunked_cell(
        model,
        batch,
        cell_id="BN/chunked",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    cells["B1-singleton_aggregate/chunked"] = _run_singleton_chunked_cell(
        model,
        batch,
        cell_id="B1-singleton_aggregate/chunked",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    if run_backward:
        _validate_required_gradient_cells(cells)
    _configure_required_gradients(model, enabled=False)
    for side in padding_sides:
        cells[f"BN/padded_{side}"] = _run_padded_cell(
            model,
            batch,
            cell_id=f"BN/padded_{side}",
            pad_side=side,
            manifest=m,
            run_backward=False,
        )
    packing_status = str(m.fixtures.get("packing", {}).get("status", "unsupported"))
    if packing_status == "supported":
        cells["BN/packed"] = _run_packed_cell(
            model,
            batch,
            cell_id="BN/packed",
            run_backward=False,
            active_token_denominator=global_active,
        )

    inv_details: list[TensorComparisonDetail] = []
    grad_details: list[TensorComparisonDetail] = []
    canonical = cells["BN/full"]
    for cell_id in (
        "B1-singleton_aggregate/full",
        "BN/chunked",
        "B1-singleton_aggregate/chunked",
    ):
        inv_details.append(
            _compare_logp_maps(
                canonical.selected_logp,
                cells[cell_id].selected_logp,
                contract=c,
                judgment="forward_invariance",
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
                config_pair=("BN/full", cell_id),
            )
        )
        if run_backward:
            for name in sorted(set(canonical.grads) & set(cells[cell_id].grads)):
                grad_details.append(
                    _compare_logical_tensors(
                        canonical.grads[name],
                        cells[cell_id].grads[name],
                        judgment="gradient_invariance",
                        contract=c,
                        op_class="reduction",
                        dtype=policy.execution_dtype,
                        backend_profile=backend_profile,
                        tensor_name=name,
                        config_pair=("BN/full", cell_id),
                    )
                )

    for side in padding_sides:
        cell_id = f"BN/padded_{side}"
        inv_details.append(
            _compare_logp_maps(
                canonical.selected_logp,
                cells[cell_id].selected_logp,
                contract=c,
                judgment="forward_invariance",
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
                config_pair=("BN/full", cell_id),
            )
        )

    if packing_status == "supported":
        # Packing changes attention reduction width vs padded BN, so this
        # layout axis uses C1 forward_accuracy, not the #150 bitwise 2x2.
        inv_details.append(
            _compare_logp_maps(
                canonical.selected_logp,
                cells["BN/packed"].selected_logp,
                contract=c,
                judgment="forward_accuracy",
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
                config_pair=("BN/full", "BN/packed"),
            )
        )

    train_infer = None
    if run_train_infer:
        train_infer = _train_infer_parity(model, batch, contract=c, manifest=m)

    first_drift = _locate_first_drift(
        model,
        cells,
        inv_details,
        grad_details,
        train_infer,
    )

    family = str(m.backend_profiles[backend_profile]["backend_family"])
    runtime_observations = model.profile_ops.validated_runtime_observations()
    observed_families = {
        _backend_family(str(value["actual_backend"])) for value in runtime_observations.values()
    }
    if observed_families != {family}:
        raise RuntimeError(
            f"profile {backend_profile!r} observed backend families "
            f"{sorted(observed_families)}, expected only {family!r}"
        )
    provenance = make_profile_provenance(
        backend_profile=backend_profile,
        contract=c,
        requested_backend=family,
        actual_backend=family,
        output_dtype=policy.output_dtype_default,
    )
    device = next(iter(model.weights.tensors.values())).device
    cc = None
    if device.type == "cuda" and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device)
        cc = f"{major}.{minor}"

    # Cross-cell logprob aggregates (BN vs B1) as the named chain metrics.
    lhs, rhs, mask = _aligned_logp_vectors(
        canonical.selected_logp, cells["B1-singleton_aggregate/full"].selected_logp
    )
    roles = resolve_comparison_roles(c, "forward_invariance")
    # Invariance is bitwise; still report the three aggregates for the logprob outputs.
    train_roles = resolve_comparison_roles(c, _TRAIN_INFER_KIND)
    aggregates = judge_logprob_aggregates(
        compute_logprob_aggregates(
            lhs,
            rhs,
            mask,
            contract=c,
            report_kind=_TRAIN_INFER_KIND,
            clip_interval=default_clip_interval(c),
            comparison_lhs_role=train_roles.comparison_lhs_role,
            comparison_rhs_role=train_roles.comparison_rhs_role,
        ),
        c,
        execution_dtype=policy.execution_dtype,
    )

    passed = (
        run_backward
        and run_train_infer
        and all(item.passed for item in inv_details)
        and all(item.passed for item in grad_details)
        and train_infer is not None
        and train_infer.passed
    )
    # Bitwise invariance is the #150 matrix verdict; aggregates are reported
    # but BN-vs-B1 bitwise already covers output identity.
    del roles
    return ChainGateReport(
        backend_profile=backend_profile,
        workload_id=m.workload_id,
        fixture_hash=fixture_hash(m, batch=batch),
        config_fingerprint=dict(m.model_identity["config_fingerprint"]),
        weight_source=model.weights.source,
        weight_hash=model.weights.content_hash,
        seed=resolved_seed,
        workload_seed=m.seed,
        device=str(device),
        compute_capability=cc,
        backend_provenance=provenance,
        runtime_backend_observations=runtime_observations,
        cells=cells,
        invariance=tuple(inv_details),
        gradient_invariance=tuple(grad_details),
        train_infer=train_infer,
        first_drift=first_drift,
        aggregates=aggregates,
        gradient_scope=GRADIENT_SCOPE,
        required_grad_names=tuple(REQUIRED_GRAD_NAMES),
        all_parameter_gradients=False,
        passed=passed,
        backward_executed=run_backward,
        train_infer_executed=run_train_infer,
        disclaimer=(
            "C10 technical e2e proof with representative parameter gradients only; "
            "this is not an all-parameter backward claim. Public WS1 EXIT still "
            "requires C11 CI and parent #266 A/B/Final."
        ),
    )


def _run_packed_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    layout = apply_packing(batch)
    device = _device(model)
    input_ids = torch.tensor([layout.physical_token_ids], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids, dtype=torch.bool)
    positions: list[int] = []
    for length in layout.segment_lengths:
        positions.extend(range(int(length)))
    pos = torch.tensor([positions], device=device, dtype=torch.long)
    loss_mask = torch.tensor([layout.physical_loss_mask], device=device, dtype=torch.bool)
    out = model.forward(
        input_ids,
        attention_mask=attn,
        position_ids=pos,
        capture_nodes=True,
        segment_lengths=layout.segment_lengths,
    )
    restore = (tuple(layout.restore_map),)
    node_token_digests = _node_token_fingerprints(model, restore)
    return _finish_cell(
        model,
        out["logits"],
        input_ids,
        loss_mask,
        restore=restore,
        cell_id=cell_id,
        run_backward=run_backward,
        active_token_denominator=active_token_denominator,
        node_token_digests=node_token_digests,
    )


def _run_padded_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    pad_side: str,
    manifest: WS1Manifest,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    padded = apply_padding(batch, pad_side=pad_side, manifest=manifest)
    input_ids = torch.tensor(padded.physical_token_ids, device=_device(model), dtype=torch.long)
    attn = torch.tensor(padded.physical_attention_mask, device=_device(model), dtype=torch.bool)
    pos = torch.tensor(padded.physical_position_ids, device=_device(model), dtype=torch.long)
    loss_mask = torch.tensor(padded.physical_loss_mask, device=_device(model), dtype=torch.bool)
    return _forward_cell(
        model,
        input_ids,
        attn,
        pos,
        loss_mask,
        restore=padded.restore_map,
        cell_id=cell_id,
        run_backward=run_backward,
        active_token_denominator=active_token_denominator,
    )


def _run_singleton_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    manifest: WS1Manifest,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    merged: dict[tuple[str, int], torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    node_token_digests: dict[str, dict[str, str]] = {}
    loss_acc = None
    for sample in batch.samples:
        single = LogicalBatch(
            workload_id=batch.workload_id,
            seed=batch.seed,
            samples=(sample,),
            cell_id=cell_id,
        )
        cell = _run_padded_cell(
            model,
            single,
            cell_id=cell_id,
            pad_side="right",
            manifest=manifest,
            run_backward=run_backward,
            active_token_denominator=active_token_denominator,
        )
        merged.update(cell.selected_logp)
        if loss_acc is None:
            loss_acc = cell.loss
        elif cell.loss is not None and loss_acc is not None:
            loss_acc = loss_acc + cell.loss
        for name, value in cell.grads.items():
            grads[name] = value if name not in grads else grads[name] + value
        _merge_node_token_digests(node_token_digests, cell.node_token_digests)
    return CellOutput(
        cell_id=cell_id,
        selected_logp=merged,
        loss=loss_acc,
        grads=grads,
        first_drift_node=None,
        node_digests=_combined_node_digests(node_token_digests),
        requested_backend=model.profile_ops.provenance["attention"]["requested_backend"],
        actual_backend=model.profile_ops.provenance["attention"]["actual_backend"],
        node_token_digests=node_token_digests,
    )


def _run_chunked_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    manifest: WS1Manifest,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    plan = chunk_plan_from_manifest(manifest)
    # Materialize as right-padded full sequences, then consume chunk_size tokens
    # through the stateful cache so the chunk toggle changes the real path.
    padded = apply_padding(batch, pad_side="right", manifest=manifest)
    input_ids = torch.tensor(padded.physical_token_ids, device=_device(model), dtype=torch.long)
    attn = torch.tensor(padded.physical_attention_mask, device=_device(model), dtype=torch.bool)
    pos = torch.tensor(padded.physical_position_ids, device=_device(model), dtype=torch.long)
    loss_mask = torch.tensor(padded.physical_loss_mask, device=_device(model), dtype=torch.bool)
    cache = model.allocate_cache(input_ids.shape[0], input_ids.shape[1], _device(model))
    logits_parts: list[torch.Tensor] = []
    node_token_digests: dict[str, dict[str, str]] = {}
    seq = input_ids.shape[1]
    for start in range(0, seq, plan.chunk_size):
        end = min(start + plan.chunk_size, seq)
        step = model.forward(
            input_ids[:, start:end],
            attention_mask=attn[:, start:end],
            position_ids=pos[:, start:end],
            kv_cache=cache,
            capture_nodes=True,
        )
        logits_parts.append(step["logits"])
        chunk_restore = tuple(tuple(row[start:end]) for row in padded.restore_map)
        _merge_node_token_digests(
            node_token_digests,
            _node_token_fingerprints(model, chunk_restore),
        )
    logits = torch.cat(logits_parts, dim=1)
    return _finish_cell(
        model,
        logits,
        input_ids,
        loss_mask,
        restore=padded.restore_map,
        cell_id=cell_id,
        run_backward=run_backward,
        active_token_denominator=active_token_denominator,
        node_token_digests=node_token_digests,
    )


def _run_singleton_chunked_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    manifest: WS1Manifest,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    merged: dict[tuple[str, int], torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    node_token_digests: dict[str, dict[str, str]] = {}
    loss_acc = None
    for sample in batch.samples:
        single = LogicalBatch(
            workload_id=batch.workload_id,
            seed=batch.seed,
            samples=(sample,),
            cell_id=cell_id,
        )
        cell = _run_chunked_cell(
            model,
            single,
            cell_id=cell_id,
            manifest=manifest,
            run_backward=run_backward,
            active_token_denominator=active_token_denominator,
        )
        merged.update(cell.selected_logp)
        if loss_acc is None:
            loss_acc = cell.loss
        elif cell.loss is not None and loss_acc is not None:
            loss_acc = loss_acc + cell.loss
        for name, value in cell.grads.items():
            grads[name] = value if name not in grads else grads[name] + value
        _merge_node_token_digests(node_token_digests, cell.node_token_digests)
    return CellOutput(
        cell_id=cell_id,
        selected_logp=merged,
        loss=loss_acc,
        grads=grads,
        first_drift_node=None,
        node_digests=_combined_node_digests(node_token_digests),
        requested_backend=model.profile_ops.provenance["attention"]["requested_backend"],
        actual_backend=model.profile_ops.provenance["attention"]["actual_backend"],
        node_token_digests=node_token_digests,
    )


def _forward_cell(
    model: Qwen3DenseBIModel,
    input_ids: torch.Tensor,
    attn: torch.Tensor,
    pos: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    restore: Sequence[Sequence[tuple[str, int] | None]],
    cell_id: str,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    out = model.forward(
        input_ids,
        attention_mask=attn,
        position_ids=pos,
        capture_nodes=True,
    )
    node_token_digests = _node_token_fingerprints(model, restore)
    return _finish_cell(
        model,
        out["logits"],
        input_ids,
        loss_mask,
        restore=restore,
        cell_id=cell_id,
        run_backward=run_backward,
        active_token_denominator=active_token_denominator,
        node_token_digests=node_token_digests,
    )


def _finish_cell(
    model: Qwen3DenseBIModel,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    restore: Sequence[Sequence[tuple[str, int] | None]],
    cell_id: str,
    run_backward: bool,
    loss: torch.Tensor | None = None,
    active_token_denominator: int | None = None,
    node_token_digests: dict[str, dict[str, str]] | None = None,
) -> CellOutput:
    # Standard causal shift: logits[t] predicts token[t+1].
    pred = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = model._selected_logp(pred, targets)
    active = loss_mask[:, 1:].to(dtype=torch.bool)
    logp = logp.masked_fill(~active, 0.0)
    selected: dict[tuple[str, int], torch.Tensor] = {}
    for batch_idx, row in enumerate(restore):
        # restore is aligned to physical tokens; selected logp lives on the next token.
        for phys, key in enumerate(row[:-1]):
            nxt = row[phys + 1]
            if key is None or nxt is None:
                continue
            if not bool(active[batch_idx, phys].item()):
                continue
            selected[(nxt[0], nxt[1])] = logp[batch_idx, phys].detach()

    grads: dict[str, torch.Tensor] = {}
    cell_loss = loss
    if cell_loss is None:
        denom = float(
            active_token_denominator if active_token_denominator else int(active.sum().item()) or 1
        )
        cell_loss = -(logp.float() * active.float()).sum() / denom
    if run_backward:
        # The gate deliberately enables only REQUIRED_GRAD_NAMES before the
        # forward graph is built.  Clear those leaf gradients between cells;
        # do not turn on the remaining 8B weights after forward, since that
        # would not add them to an already-built autograd graph.
        for name in REQUIRED_GRAD_NAMES:
            model.weights.tensors[name].grad = None
        cell_loss.backward()
        for name in REQUIRED_GRAD_NAMES:
            tensor = model.weights.tensors[name]
            if tensor.grad is None:
                raise RuntimeError(f"required gradient {name!r} is missing in cell {cell_id!r}")
            grads[name] = tensor.grad.detach().float().cpu()

    return CellOutput(
        cell_id=cell_id,
        selected_logp=selected,
        loss=cell_loss.detach() if cell_loss is not None else None,
        grads=grads,
        first_drift_node=None,
        node_digests=_combined_node_digests(node_token_digests or {}),
        requested_backend=model.profile_ops.provenance["attention"]["requested_backend"],
        actual_backend=model.profile_ops.provenance["attention"]["actual_backend"],
        node_token_digests=dict(node_token_digests or {}),
    )


def _train_infer_parity(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    contract: Mapping[str, Any],
    manifest: WS1Manifest,
) -> LogprobAggregateVerdict:
    """Teacher-forcing prefill vs prompt-prefill + stateful decode of completions.

    Tokens are the C2 fixture (no sampling). Each sample is scored at B=1 so
    prompt_len is exact and padding cannot leak into the decode cursor.
    """

    device = _device(model)
    train_parts: list[torch.Tensor] = []
    infer_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    for sample in batch.samples:
        tokens = torch.tensor([sample.token_ids], device=device, dtype=torch.long)
        attn = torch.ones_like(tokens, dtype=torch.bool)
        pos = torch.arange(sample.seq_len, device=device).unsqueeze(0)
        loss_mask = torch.tensor(
            [[int(i >= sample.prompt_len) for i in range(sample.seq_len)]],
            device=device,
            dtype=torch.bool,
        )
        train = model.selected_logprobs(
            tokens, attention_mask=attn, loss_mask=loss_mask, position_ids=pos
        )
        infer = torch.zeros_like(train)
        prompt = sample.prompt_len
        cache = model.allocate_cache(1, sample.seq_len, device)
        prefill = model.forward(
            tokens[:, :prompt],
            attention_mask=attn[:, :prompt],
            position_ids=pos[:, :prompt],
            kv_cache=cache,
        )
        # logits[prompt-1] predicts token[prompt] (first completion).
        first = model._selected_logp(prefill["logits"][:, -1:, :], tokens[:, prompt : prompt + 1])
        infer[:, prompt - 1] = first[:, 0]
        for phys in range(prompt, sample.seq_len - 1):
            step = model.decode_step(
                tokens[:, phys : phys + 1],
                cache,
                position_ids=pos[:, phys : phys + 1],
                attention_mask=attn[:, phys : phys + 1],
            )
            logp = model._selected_logp(step["logits"], tokens[:, phys + 1 : phys + 2])
            infer[:, phys] = logp[:, 0]
        active = loss_mask[:, 1:]
        train_parts.append(train.reshape(-1))
        infer_parts.append(infer.reshape(-1))
        mask_parts.append(active.reshape(-1))

    train_cat = torch.cat(train_parts)
    infer_cat = torch.cat(infer_parts)
    mask_cat = torch.cat(mask_parts)
    roles = resolve_comparison_roles(contract, _TRAIN_INFER_KIND)
    aggregates = compute_logprob_aggregates(
        infer_cat,
        train_cat,
        mask_cat,
        contract=contract,
        report_kind=_TRAIN_INFER_KIND,
        clip_interval=default_clip_interval(contract),
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
    )
    policy = resolve_dtype_policy(contract)
    return judge_logprob_aggregates(aggregates, contract, execution_dtype=policy.execution_dtype)


def _compare_logp_maps(
    lhs: Mapping[tuple[str, int], torch.Tensor],
    rhs: Mapping[tuple[str, int], torch.Tensor],
    *,
    contract: Mapping[str, Any],
    judgment: str,
    dtype: str,
    backend_profile: str,
    config_pair: tuple[str, str],
) -> TensorComparisonDetail:
    keys = sorted(set(lhs) | set(rhs))
    if not keys or set(lhs) != set(rhs):
        spec = resolve_tolerance(
            contract,
            judgment=judgment,
            op_class="logprob",
            dtype=dtype,
            backend_profile=backend_profile,
        )
        return TensorComparisonDetail(
            tensor_name="selected_logp",
            config_pair=config_pair,
            shape=(len(keys),),
            dtype=dtype,
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            atol=spec.atol,
            rtol=spec.rtol,
            passed=False,
            judgment=judgment,
            comparison_lhs_role="transformed_config",
            comparison_rhs_role="canonical_config",
        )
    a = torch.stack([lhs[key].float().reshape(()) for key in keys])
    b = torch.stack([rhs[key].float().reshape(()) for key in keys])
    return _compare_logical_tensors(
        a,
        b,
        judgment=judgment,
        contract=contract,
        op_class="logprob",
        dtype=dtype,
        backend_profile=backend_profile,
        tensor_name="selected_logp",
        config_pair=config_pair,
    )


def _aligned_logp_vectors(
    lhs: Mapping[tuple[str, int], torch.Tensor],
    rhs: Mapping[tuple[str, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = sorted(set(lhs) & set(rhs))
    if not keys:
        raise RuntimeError("no overlapping logical tokens to compare")
    a = torch.stack([lhs[key].float().reshape(()) for key in keys])
    b = torch.stack([rhs[key].float().reshape(()) for key in keys])
    mask = torch.ones_like(a, dtype=torch.bool)
    return a, b, mask


def _device(model: Qwen3DenseBIModel) -> torch.device:
    return next(iter(model.weights.tensors.values())).device


def _configure_required_gradients(model: Qwen3DenseBIModel, *, enabled: bool) -> None:
    missing = sorted(set(REQUIRED_GRAD_NAMES) - set(model.weights.tensors))
    if missing:
        raise RuntimeError(f"model is missing required gradient tensors: {missing}")
    for name, tensor in model.weights.tensors.items():
        if tensor.is_floating_point():
            tensor.requires_grad_(enabled and name in REQUIRED_GRAD_NAMES)
            tensor.grad = None


def _validate_required_gradient_cells(cells: Mapping[str, CellOutput]) -> None:
    required = set(REQUIRED_GRAD_NAMES)
    for cell_id in PRIMARY_CELLS:
        observed = set(cells[cell_id].grads)
        if observed != required:
            raise RuntimeError(
                f"cell {cell_id!r} gradient set {sorted(observed)} != required {sorted(required)}"
            )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _node_token_fingerprints(
    model: Qwen3DenseBIModel,
    restore: Sequence[Sequence[tuple[str, int] | None]],
) -> dict[str, dict[str, str]]:
    """Hash every logical token at every captured node for exact first-drift."""

    result: dict[str, dict[str, str]] = {}
    batch = len(restore)
    seq = len(restore[0]) if restore else 0
    for node, value in model.captured_node_outputs().items():
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        if value.dim() < 2 or value.shape[0] != batch:
            continue
        if value.shape[1] == seq:
            token_axis = 1
        elif value.dim() >= 3 and value.shape[2] == seq:
            token_axis = 2
        else:
            continue
        cpu_value = value.detach().contiguous().cpu()
        node_values: dict[str, str] = {}
        for batch_index, row in enumerate(restore):
            for physical_index, logical_key in enumerate(row):
                if logical_key is None:
                    continue
                if token_axis == 1:
                    token_value = cpu_value[batch_index, physical_index]
                else:
                    token_value = cpu_value[batch_index, :, physical_index]
                key = f"{logical_key[0]}:{logical_key[1]}"
                node_values[key] = _tensor_digest(token_value)
        if node_values:
            result[node] = node_values
    return result


def _merge_node_token_digests(
    target: dict[str, dict[str, str]], source: Mapping[str, Mapping[str, str]]
) -> None:
    for node, values in source.items():
        destination = target.setdefault(node, {})
        overlap = set(destination) & set(values)
        for key in overlap:
            if destination[key] != values[key]:
                raise RuntimeError(f"node fingerprint collision for {node!r} logical token {key!r}")
        destination.update(values)


def _combined_node_digests(
    values: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    combined: dict[str, str] = {}
    for node, tokens in values.items():
        digest = hashlib.sha256()
        for key, value in sorted(tokens.items()):
            digest.update(f"{key}\t{value}\n".encode("utf-8"))
        combined[node] = digest.hexdigest()
    return combined


def _backend_family(value: str) -> str:
    if value.startswith("cuda"):
        return "cuda"
    return value


def _locate_first_drift(
    model: Qwen3DenseBIModel,
    cells: Mapping[str, CellOutput],
    inv_details: Sequence[TensorComparisonDetail],
    grad_details: Sequence[TensorComparisonDetail],
    train_infer: LogprobAggregateVerdict | None,
) -> str | None:
    """First failing layer/op, then tensor/config pair. Used by C10/C11 reports."""

    canonical = cells.get("BN/full")
    for detail in list(inv_details) + list(grad_details):
        if detail.passed:
            continue
        other_id = detail.config_pair[1]
        other = cells.get(other_id)
        if canonical is not None and other is not None:
            for node in model.node_names():
                left = canonical.node_token_digests.get(node)
                right = other.node_token_digests.get(node)
                if left is None or right is None:
                    continue
                if left != right:
                    return f"{node}:{detail.config_pair[0]}->{other_id}"
        return f"{detail.tensor_name}:{detail.config_pair[0]}->{other_id}"
    if train_infer is not None and not train_infer.passed:
        return "train_infer_selected_logp"
    return None


__all__ = [
    "PRIMARY_CELLS",
    "REQUIRED_GRAD_NAMES",
    "CellOutput",
    "ChainGateReport",
    "build_model",
    "run_chain_gate",
]
