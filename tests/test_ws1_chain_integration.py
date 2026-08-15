# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-safe C10 report / schema / bitwise-rule tests. Full-model execute is H20."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_engine.alignment.qwen3_dense import Qwen3DenseSpec
from rl_engine.kernels.gtest.chain_gate import (
    GRADIENT_SCOPE,
    PRIMARY_CELLS,
    REQUIRED_GRAD_NAMES,
    ChainGateReport,
    _compare_logp_maps,
    _configure_required_gradients,
    _node_token_fingerprints,
)
from rl_engine.kernels.gtest.tolerance import load_contract
from rl_engine.testing.ws1_workload import load_manifest


def test_c10_primary_cells_match_c2_matrix():
    assert PRIMARY_CELLS == (
        "B1-singleton_aggregate/full",
        "BN/full",
        "B1-singleton_aggregate/chunked",
        "BN/chunked",
    )


def test_c10_packing_is_declared_supported_required_axis():
    manifest = load_manifest()
    assert manifest.fixtures["packing"]["status"] == "supported"


def test_c10_spec_fingerprint_is_full_qwen3():
    spec = Qwen3DenseSpec.from_manifest(load_manifest())
    assert spec.num_hidden_layers == 36
    assert spec.hidden_size == 4096
    assert spec.vocab_size == 151936


def test_c10_bitwise_invariance_rule_is_zero_tol():
    contract = load_contract()
    lhs = {("s0", 1): torch.tensor(0.25), ("s1", 2): torch.tensor(-0.5)}
    rhs = {("s0", 1): torch.tensor(0.25), ("s1", 2): torch.tensor(-0.5)}
    detail = _compare_logp_maps(
        lhs,
        rhs,
        contract=contract,
        judgment="forward_invariance",
        dtype="bfloat16",
        backend_profile="cuda_bf16",
        config_pair=("BN/full", "B1-singleton_aggregate/full"),
    )
    assert detail.atol == 0.0
    assert detail.rtol == 0.0
    assert detail.passed


def test_c10_bitwise_invariance_fails_on_drift():
    contract = load_contract()
    lhs = {("s0", 1): torch.tensor(0.25)}
    rhs = {("s0", 1): torch.tensor(0.26)}
    detail = _compare_logp_maps(
        lhs,
        rhs,
        contract=contract,
        judgment="forward_invariance",
        dtype="bfloat16",
        backend_profile="cuda_bf16",
        config_pair=("BN/full", "BN/chunked"),
    )
    assert detail.passed is False
    assert detail.max_abs_error > 0.0


def test_c10_report_schema_fields():
    fields = set(ChainGateReport.__dataclass_fields__)
    for name in (
        "backend_profile",
        "workload_id",
        "fixture_hash",
        "config_fingerprint",
        "weight_hash",
        "backend_provenance",
        "runtime_backend_observations",
        "invariance",
        "gradient_invariance",
        "train_infer",
        "first_drift",
        "aggregates",
        "passed",
        "backward_executed",
        "train_infer_executed",
        "gradient_scope",
        "required_grad_names",
        "all_parameter_gradients",
        "disclaimer",
    ):
        assert name in fields


def test_c10_required_gradients_are_enabled_before_forward():
    tensors = {name: torch.tensor(2.0) for name in REQUIRED_GRAD_NAMES}
    tensors["unused.weight"] = torch.tensor(3.0)
    model = SimpleNamespace(weights=SimpleNamespace(tensors=tensors))
    _configure_required_gradients(model, enabled=True)
    loss = tensors["norm.weight"] * tensors["lm_head.weight"]
    loss.backward()
    assert tensors["norm.weight"].grad is not None
    assert tensors["lm_head.weight"].grad is not None
    assert tensors["unused.weight"].requires_grad is False


def test_c10_gradient_contract_is_explicitly_representative():
    assert GRADIENT_SCOPE == "representative_parameter_subset"
    assert REQUIRED_GRAD_NAMES == (
        "norm.weight",
        "lm_head.weight",
        "layers.0.input_layernorm.weight",
    )


def test_c10_node_fingerprints_follow_logical_tokens():
    class FakeModel:
        def __init__(self, output):
            self.output = output

        def captured_node_outputs(self):
            return {"embedding": self.output}

    canonical = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    canonical_restore = (("s0", 0), ("s0", 1)), (("s1", 0), ("s1", 1))
    permuted = canonical.index_select(0, torch.tensor([1, 0]))
    permuted_restore = (canonical_restore[1], canonical_restore[0])
    lhs = _node_token_fingerprints(FakeModel(canonical), canonical_restore)
    rhs = _node_token_fingerprints(FakeModel(permuted), permuted_restore)
    assert lhs == rhs

    changed = canonical.clone()
    changed[1, 1, 0] += 1.0
    drifted = _node_token_fingerprints(FakeModel(changed), canonical_restore)
    assert lhs != drifted
