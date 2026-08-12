# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from .forward_invariance import (
    AccuracyReport,
    ConfigSpec,
    ForwardInvarianceReport,
    InvarianceReport,
    LogprobSmokeResult,
    RuntimeObservation,
    TensorComparisonDetail,
    assert_forward_batch_invariant,
    build_config_matrix,
)
from .gradient_invariance import (
    GradientInvarianceReport,
    GradientObservation,
    GradientTensorSpec,
    MissingBackwardError,
    assert_gradient_batch_invariant,
)
from .op_checks import CandidateSpec, OperatorCase, run_operator_suite
from .tolerance import (
    BackendProvenance,
    ContractError,
    ContractResolveError,
    ContractSchemaError,
    load_contract,
    resolve_dtype_policy,
    resolve_tolerance,
    resolve_tolerance_support,
    validate_backend_provenance,
)

__all__ = [
    "AccuracyReport",
    "CandidateSpec",
    "ConfigSpec",
    "ForwardInvarianceReport",
    "GradientInvarianceReport",
    "GradientObservation",
    "GradientTensorSpec",
    "MissingBackwardError",
    "InvarianceReport",
    "LogprobSmokeResult",
    "RuntimeObservation",
    "OperatorCase",
    "TensorComparisonDetail",
    "assert_forward_batch_invariant",
    "assert_gradient_batch_invariant",
    "build_config_matrix",
    "run_operator_suite",
    "BackendProvenance",
    "ContractError",
    "ContractResolveError",
    "ContractSchemaError",
    "load_contract",
    "resolve_tolerance",
    "resolve_dtype_policy",
    "resolve_tolerance_support",
    "validate_backend_provenance",
]
