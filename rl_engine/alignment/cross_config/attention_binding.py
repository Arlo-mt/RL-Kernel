# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Three-tier binding between rollout-side and training-side attention contracts.

Issue #235 PR4 requires that "rollout and training descriptors bind to the same
semantic attention contract". Under the frozen Megatron + vLLM deployment the two
sides can never produce *identical* :class:`AttentionContract` instances: training
runs full-sequence prefill over a CP-sharded sequence, while rollout runs vLLM
paged-KV chunked prefill and decode. Taking "same contract" literally would make
the target configuration permanently unbindable.

This module therefore splits binding into three tiers:

``IDENTICAL``
    Logical identity. Both sides must agree bit for bit, otherwise the pair is not
    comparable at all and no drift number from it means anything.

``SEMANTIC``
    The WS2 numerical claim: merge semantics, accumulation dtype, reduction order
    and downcast point are decided by the contract, not by the implementation.
    Both sides must carry the same values *and* those values must match the WS2
    mandate, otherwise the comparison fails closed.

``RECORDED``
    Materialization facts that the two sides are expected to differ on -- attention
    mode, RoPE fusion boundary, KV-cache paging, backend id, reduction engine. These
    differences are exactly what the experiment measures, so they are recorded into
    provenance rather than rejected.

Deliberately *not* in ``SEMANTIC``: ``engine``. Training may run the in-op
deterministic reference while rollout runs a Transformer Engine merge oracle; forcing
those equal would defeat the purpose of the oracle comparison in #235 PR2/3/5/6.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from rl_engine.kernels.attention_contract import (
    AttentionContract,
    AttentionDType,
    AttentionMerge,
    AttentionRole,
    DowncastPoint,
    ReductionOrder,
)

__all__ = [
    "ATTENTION_LSE_DOMAIN",
    "AttentionBindingError",
    "AttentionBindingResult",
    "BindingErrorCode",
    "BindingIssue",
    "BindingTier",
    "IDENTITY_FIELDS",
    "NULLABLE_IDENTITY_FIELDS",
    "RECORDED_FIELDS",
    "SEMANTIC_CONTRACT_FIELDS",
    "SEMANTIC_REDUCTION_FIELDS",
    "WS2_ATTENTION_REDUCTION_MANDATE",
    "bind_attention_contracts",
    "first_blocking_issue",
    "identity_fingerprint",
    "summarize_binding",
]


class AttentionBindingError(ValueError):
    """Raised when a caller supplies structurally unusable binding inputs."""


class BindingTier(str, Enum):
    """Which rule a field is governed by."""

    IDENTICAL = "identical"
    SEMANTIC = "semantic"
    RECORDED = "recorded"


class BindingErrorCode(str, Enum):
    """Stable, machine-readable reasons a binding is rejected.

    Callers branch on these; they are part of the artifact schema and must not be
    renamed without a schema version bump.
    """

    IDENTITY_MISSING = "IDENTITY_MISSING"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    REDUCTION_SEMANTIC_MISMATCH = "REDUCTION_SEMANTIC_MISMATCH"
    REDUCTION_MANDATE_VIOLATION = "REDUCTION_MANDATE_VIOLATION"
    LSE_NOT_EXPORTED = "LSE_NOT_EXPORTED"
    ROLE_COLLISION = "ROLE_COLLISION"
    DETERMINISM_INCOMPATIBLE = "DETERMINISM_INCOMPATIBLE"


#: Attention exports attention-domain LSE, never vocab-logprob LSE (#235).
#: Recorded explicitly so a future ``LogprobContract`` binding cannot be confused
#: with this one purely because both set ``export_lse=True``.
ATTENTION_LSE_DOMAIN = "attention"


#: Fields both sides must agree on bit for bit before any comparison is meaningful.
#: Sourced from #235 "Numerical Contract" preconditions plus the vime-owned rollout
#: provenance (weight version, sampling, padding) that the issue assumes but does
#: not enumerate.
IDENTITY_FIELDS: tuple[str, ...] = (
    "checkpoint_id",
    "model_version",
    "weight_version",
    "tokenizer_fingerprint",
    "token_ids_fingerprint",
    "active_mask_fingerprint",
    "position_ids_fingerprint",
    "padding_side",
    "pre_update_state",
    # model semantics that decide what attention *means*
    "q_heads",
    "kv_heads",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "rotary_dim",
    "qk_layernorm",
    # batch composition: batch-invariance is a claim about results not changing with
    # batch makeup, so two sides scoring different batches are not comparable at all
    "batch_size",
    # decode replay identity (#235 PR6)
    "global_token_positions_fingerprint",
    "kv_seq_lens_fingerprint",
)


#: Reduction fields that decide the numerical result. Both sides must carry the
#: same value, and that value must satisfy :data:`WS2_ATTENTION_REDUCTION_MANDATE`.
SEMANTIC_REDUCTION_FIELDS: tuple[str, ...] = (
    "merge",
    "acc_dtype",
    "order",
    "downcast_at",
)


#: Contract fields outside ``ReductionSpec`` that still decide the numerical result.
#: ``dtype`` is here rather than in :data:`RECORDED_FIELDS` because comparing a BF16
#: rollout against an FP16 training pass produces a real drift number attributable to
#: nothing. #235 PR5 does sweep BF16 against an FP32 reference; that sweep opts in via
#: ``allow_dtype_difference`` instead of loosening the default.
SEMANTIC_CONTRACT_FIELDS: tuple[str, ...] = ("dtype",)


#: The WS2 mandate itself. ``#236`` currently declares single-member enums for
#: ``merge`` / ``order`` / ``downcast_at``, so those checks are tautological today;
#: they are written out anyway so that widening any of those enums later fails here
#: instead of silently admitting a non-conforming backend.
WS2_ATTENTION_REDUCTION_MANDATE: Mapping[str, str] = {
    "merge": AttentionMerge.ONLINE_SOFTMAX_LSE.value,
    "acc_dtype": AttentionDType.FP32.value,
    "order": ReductionOrder.GLOBAL_BLOCK_INDEX.value,
    "downcast_at": DowncastPoint.FINAL_WRITE.value,
}


#: Materialization facts the two sides are expected to differ on. Recorded into
#: provenance; never a rejection reason.
RECORDED_FIELDS: tuple[str, ...] = (
    "mode",
    "backend_id",
    "reduction.engine",
    "rope.fusion_boundary",
    "rope.q_state",
    "rope.k_state",
    "rope.k_cache_state",
    "rope.cast_at",
    "rope.output_dtype",
    "kv_cache.page_size",
    "kv_cache.prefix_cache_enabled",
    "kv_cache.block_table_shape",
    "sharding.cp_world_size",
    "sharding.tp_world_size",
    "sharding.local_sequence_length",
    # Supplied by the caller, not by the contract: #236 has no split-KV field yet, so
    # the value comes from vLLM's flash_attn_max_num_splits_for_cuda_graph via the
    # adapter. Recorded so split-KV differences are at least visible in provenance
    # until #236 grows the field and it can move into the contract proper.
    "split_kv_policy",
)


@dataclass(frozen=True)
class BindingIssue:
    """One reason a binding is not comparable or not admissible."""

    code: BindingErrorCode
    tier: BindingTier
    field: str
    rollout: Any = None
    training: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "tier": self.tier.value,
            "field": self.field,
            "rollout": self.rollout,
            "training": self.training,
            "message": self.message,
        }


@dataclass(frozen=True)
class AttentionBindingResult:
    """Outcome of binding one rollout contract to one training contract.

    ``comparable`` and ``passed`` are deliberately separate. A pair whose identity
    does not match is *not comparable* -- reporting a drift number for it would be
    meaningless. A pair that is comparable but violates the reduction mandate *is*
    comparable yet must still fail closed, because the whole WS2 claim is that
    reduction order and accumulation precision come from the contract.
    """

    comparable: bool
    passed: bool
    issues: tuple[BindingIssue, ...] = ()
    identity_fingerprint: str = ""
    reduction_fingerprint: str = ""
    binding_fingerprint: str = ""
    recorded_differences: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "cross_config.attention_binding.v1"

    def issues_by_code(self, code: BindingErrorCode) -> tuple[BindingIssue, ...]:
        return tuple(issue for issue in self.issues if issue.code is code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparable": self.comparable,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "identity_fingerprint": self.identity_fingerprint,
            "reduction_fingerprint": self.reduction_fingerprint,
            "binding_fingerprint": self.binding_fingerprint,
            "recorded_differences": {
                key: dict(value) for key, value in self.recorded_differences.items()
            },
            "provenance": dict(self.provenance),
        }


def _canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def identity_fingerprint(identity: Mapping[str, Any]) -> str:
    """Fingerprint only the declared :data:`IDENTITY_FIELDS`, in a fixed order.

    Extra keys in ``identity`` are ignored on purpose: callers pass whole
    provenance bundles, and the fingerprint must not drift when an unrelated
    diagnostic field is added.
    """

    return _canonical_fingerprint({name: identity.get(name) for name in IDENTITY_FIELDS})


def _reduction_view(contract: AttentionContract) -> dict[str, Any]:
    reduction = contract.reduction
    return {
        "merge": reduction.merge.value,
        "acc_dtype": reduction.acc_dtype.value,
        "order": reduction.order.value,
        "downcast_at": reduction.downcast_at.value,
        "engine": reduction.engine.value,
    }


def _recorded_view(
    contract: AttentionContract,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    rope = contract.rope
    kv_cache = contract.kv_cache
    view: dict[str, Any] = {
        "mode": contract.mode.value,
        "backend_id": None,
        "reduction.engine": contract.reduction.engine.value,
        "sharding.cp_world_size": contract.sharding.cp_world_size,
        "sharding.tp_world_size": contract.sharding.tp_world_size,
        "sharding.local_sequence_length": contract.sharding.local_sequence_length,
    }
    if rope is not None:
        view.update(
            {
                "rope.fusion_boundary": rope.fusion_boundary.value,
                "rope.q_state": rope.q_state.value,
                "rope.k_state": rope.k_state.value,
                "rope.k_cache_state": rope.k_cache_state.value,
                "rope.cast_at": rope.cast_at.value,
                "rope.output_dtype": rope.output_dtype.value,
            }
        )
    if kv_cache is not None:
        view.update(
            {
                "kv_cache.page_size": kv_cache.page_size,
                "kv_cache.prefix_cache_enabled": kv_cache.prefix_cache_enabled,
                "kv_cache.block_table_shape": [
                    len(kv_cache.block_table),
                    max((len(row) for row in kv_cache.block_table), default=0),
                ],
            }
        )
    if extra:
        view.update(extra)
    return view


#: Identity fields where ``None`` is a real value rather than an omission. Qwen3-8B
#: applies no RoPE scaling, so ``rope_scaling=None`` must not read as "undeclared" --
#: both sides still have to agree on it, which the equality pass below handles.
NULLABLE_IDENTITY_FIELDS: frozenset[str] = frozenset({"rope_scaling"})


def _missing_identity_fields(identity: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name in IDENTITY_FIELDS
        if name not in NULLABLE_IDENTITY_FIELDS and identity.get(name) is None
    )


def bind_attention_contracts(
    *,
    rollout_contract: AttentionContract,
    training_contract: AttentionContract,
    rollout_identity: Mapping[str, Any],
    training_identity: Mapping[str, Any],
    rollout_backend_id: str,
    training_backend_id: str,
    determinism_issues: Sequence[BindingIssue] = (),
    require_full_identity: bool = True,
    allow_dtype_difference: bool = False,
    rollout_recorded_extra: Optional[Mapping[str, Any]] = None,
    training_recorded_extra: Optional[Mapping[str, Any]] = None,
) -> AttentionBindingResult:
    """Bind a rollout attention contract to a training attention contract.

    ``determinism_issues`` is threaded in from
    :mod:`rl_engine.alignment.cross_config.determinism` rather than computed here,
    so that this module stays free of framework probing and remains testable
    without Megatron or vLLM present.

    ``require_full_identity`` exists for the single-GPU harness in #235 PR2, which
    legitimately has no KV-cache or decode identity to declare. Distributed callers
    must leave it at ``True``.

    ``allow_dtype_difference`` exists for the #235 PR5 sweep that deliberately scores
    a BF16 path against an FP32 reference. It must stay ``False`` everywhere else.

    ``rollout_recorded_extra`` / ``training_recorded_extra`` carry materialization
    facts that #236 does not yet model -- today that is ``split_kv_policy``. They are
    merged into the recorded tier, never into identity or semantics.
    """

    if rollout_contract.role is not AttentionRole.INFER:
        raise AttentionBindingError(
            f"rollout_contract.role must be {AttentionRole.INFER.value!r}, "
            f"got {rollout_contract.role.value!r}"
        )
    if training_contract.role is not AttentionRole.TRAIN:
        raise AttentionBindingError(
            f"training_contract.role must be {AttentionRole.TRAIN.value!r}, "
            f"got {training_contract.role.value!r}"
        )

    issues: list[BindingIssue] = []

    # ---- tier 1: identity, bit for bit -------------------------------------
    if require_full_identity:
        for side, identity in (("rollout", rollout_identity), ("training", training_identity)):
            for name in _missing_identity_fields(identity):
                issues.append(
                    BindingIssue(
                        code=BindingErrorCode.IDENTITY_MISSING,
                        tier=BindingTier.IDENTICAL,
                        field=f"{side}.{name}",
                        message=f"{side} identity does not declare {name!r}",
                    )
                )

    for name in IDENTITY_FIELDS:
        rollout_value = rollout_identity.get(name)
        training_value = training_identity.get(name)
        if rollout_value != training_value:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.IDENTITY_MISMATCH,
                    tier=BindingTier.IDENTICAL,
                    field=name,
                    rollout=rollout_value,
                    training=training_value,
                    message=(
                        f"{name!r} differs between sides; the pair is not comparable "
                        "and any drift computed from it is meaningless"
                    ),
                )
            )

    comparable = not any(issue.tier is BindingTier.IDENTICAL for issue in issues)

    # ---- tier 2: reduction semantics, and the WS2 mandate -------------------
    rollout_reduction = _reduction_view(rollout_contract)
    training_reduction = _reduction_view(training_contract)

    for name in SEMANTIC_REDUCTION_FIELDS:
        if rollout_reduction[name] != training_reduction[name]:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.REDUCTION_SEMANTIC_MISMATCH,
                    tier=BindingTier.SEMANTIC,
                    field=f"reduction.{name}",
                    rollout=rollout_reduction[name],
                    training=training_reduction[name],
                    message=(
                        f"reduction.{name!r} must be decided by the contract, not by the "
                        "backend; the two sides disagree"
                    ),
                )
            )
        mandated = WS2_ATTENTION_REDUCTION_MANDATE[name]
        for side, view in (("rollout", rollout_reduction), ("training", training_reduction)):
            if view[name] != mandated:
                issues.append(
                    BindingIssue(
                        code=BindingErrorCode.REDUCTION_MANDATE_VIOLATION,
                        tier=BindingTier.SEMANTIC,
                        field=f"{side}.reduction.{name}",
                        rollout=rollout_reduction[name],
                        training=training_reduction[name],
                        message=(
                            f"WS2 requires reduction.{name} == {mandated!r}; "
                            f"{side} declares {view[name]!r}"
                        ),
                    )
                )

    if not allow_dtype_difference and rollout_contract.dtype is not training_contract.dtype:
        issues.append(
            BindingIssue(
                code=BindingErrorCode.REDUCTION_SEMANTIC_MISMATCH,
                tier=BindingTier.SEMANTIC,
                field="dtype",
                rollout=rollout_contract.dtype.value,
                training=training_contract.dtype.value,
                message=(
                    "the two sides compute in different dtypes; the resulting drift is "
                    "not attributable. Pass allow_dtype_difference=True only for a "
                    "deliberate precision sweep"
                ),
            )
        )

    for side, contract in (("rollout", rollout_contract), ("training", training_contract)):
        if not contract.export_lse:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.LSE_NOT_EXPORTED,
                    tier=BindingTier.SEMANTIC,
                    field=f"{side}.export_lse",
                    message=(
                        "attention-domain LSE must be exported; without it the deterministic "
                        "CP merge cannot be validated"
                    ),
                )
            )

    if rollout_backend_id == training_backend_id and rollout_backend_id:
        # Not an error, but worth surfacing: an identical backend on both sides means
        # the experiment is not actually measuring a cross-implementation difference.
        pass

    issues.extend(determinism_issues)

    # ---- tier 3: recorded differences --------------------------------------
    rollout_recorded = _recorded_view(rollout_contract, rollout_recorded_extra)
    rollout_recorded["backend_id"] = rollout_backend_id
    training_recorded = _recorded_view(training_contract, training_recorded_extra)
    training_recorded["backend_id"] = training_backend_id

    recorded_differences: dict[str, dict[str, Any]] = {}
    for name in RECORDED_FIELDS:
        rollout_value = rollout_recorded.get(name)
        training_value = training_recorded.get(name)
        if rollout_value != training_value:
            recorded_differences[name] = {
                "rollout": rollout_value,
                "training": training_value,
            }

    identity_fp = identity_fingerprint(training_identity if comparable else rollout_identity)
    reduction_fp = _canonical_fingerprint(
        {name: training_reduction[name] for name in SEMANTIC_REDUCTION_FIELDS}
    )
    passed = comparable and not any(issue.tier is BindingTier.SEMANTIC for issue in issues)

    provenance = {
        "lse_domain": ATTENTION_LSE_DOMAIN,
        "dtype": training_contract.dtype.value,
        "rollout": {
            "contract": rollout_contract.to_dict(),
            "backend_id": rollout_backend_id,
            "recorded": rollout_recorded,
        },
        "training": {
            "contract": training_contract.to_dict(),
            "backend_id": training_backend_id,
            "recorded": training_recorded,
        },
    }

    return AttentionBindingResult(
        comparable=comparable,
        passed=passed,
        issues=tuple(issues),
        identity_fingerprint=identity_fp,
        reduction_fingerprint=reduction_fp,
        binding_fingerprint=_canonical_fingerprint(
            {
                "identity": identity_fp,
                "reduction": reduction_fp,
                "lse_domain": ATTENTION_LSE_DOMAIN,
                "rollout_backend": rollout_backend_id,
                "training_backend": training_backend_id,
            }
        ),
        recorded_differences=recorded_differences,
        provenance=provenance,
    )


def summarize_binding(result: AttentionBindingResult) -> str:
    """One-line human summary for CLI output and failure messages."""

    if result.passed:
        return (
            f"attention binding OK "
            f"(identity={result.identity_fingerprint[:12]}, "
            f"{len(result.recorded_differences)} recorded difference(s))"
        )
    if not result.comparable:
        fields = ", ".join(
            sorted({issue.field for issue in result.issues if issue.tier is BindingTier.IDENTICAL})
        )
        return f"attention binding NOT COMPARABLE; identity problems: {fields}"
    fields = ", ".join(
        sorted({issue.field for issue in result.issues if issue.tier is BindingTier.SEMANTIC})
    )
    return f"attention binding FAILED CLOSED; semantic problems: {fields}"


def first_blocking_issue(
    result: AttentionBindingResult,
) -> Optional[BindingIssue]:
    """Return the issue a caller should report, preferring identity over semantics."""

    for tier in (BindingTier.IDENTICAL, BindingTier.SEMANTIC):
        for issue in result.issues:
            if issue.tier is tier:
                return issue
    return None
