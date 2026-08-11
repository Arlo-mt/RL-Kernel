# WS1 Numerical Contract (C1 / #267)

> **Parent:** [#266](https://github.com/RL-Align/RL-Kernel/issues/266)
> **Issue:** [#267](https://github.com/RL-Align/RL-Kernel/issues/267)
> **SSOT files:** `rl_engine/kernels/gtest/tolerance_contract.json`, `rl_engine/kernels/gtest/tolerance.py`

This document freezes the **sole** numerical judgment source for WS1 ablations and
gates. New gates must obtain thresholds only through the shared resolver APIs; private
`atol` / `rtol` constants are forbidden.

## Scope boundary

**Allowed WS1 claim after full exit (#266):** single-GPU model-level train–inference
consistency for full Qwen3-8B Dense under required CUDA BF16 and Triton-on-CUDA BF16
profiles (in-repo BI stack).

**Not claimed here:** multi-GPU (WS2), real vLLM vs Megatron / vime product alignment
(WS3), or kernel bug fixes (open Blockers).

## Dtype policy

| Field | Lock |
| --- | --- |
| `execution_dtype` | **BF16** (mandatory) |
| `accumulation_dtype` | **FP32** |
| `reference_dtype` | **FP32** |
| `output_dtype.default` | follows execution |
| logprob aggregates compute dtype | **FP32** |
| FP8 | **out of scope** (request → hard fail) |
| FP16 | optional; rows complete when declared |
| TF32 (reference + candidate) | **disabled** (repo-wide single policy) |
| Backend profiles | `cuda_bf16`, `triton_cuda_bf16` (same thresholds) |
| Backend-private tolerance relaxation | **forbidden** |

Execution, accumulation, output, and reference dtypes resolve **independently** via
`resolve_dtype_policy()`.

## Four judgments

| Judgment | What it compares | Default mode |
| --- | --- | --- |
| `forward_accuracy` | BF16 candidate vs FP32 reference outputs | tolerance |
| `forward_invariance` | transformed vs canonical config, same backend/dtype/logical workload | **bitwise** (`atol=0`, `rtol=0`) |
| `gradient_accuracy` | candidate gradient/VJP vs FP32 reference gradient/VJP | tolerance (independent of forward) |
| `gradient_invariance` | transformed vs canonical gradients, same logical workload | **bitwise** (`atol=0`, `rtol=0`) |

Every declared-applicable `(judgment, op_class, dtype)` tuple must resolve. Missing
applicable cells hard-fail. Explicit `not_applicable` / `out_of_scope` is allowed only
when present in the schema. Use `resolve_tolerance_support()` to persist the explicit
support status; requesting thresholds for an N/A or out-of-scope cell still hard-fails.

**Batch/Chunk invariance** (issue #150 / C10 matrix) **must** use the invariance
judgments in bitwise mode. Nonzero tolerance cannot satisfy that gate.

Op classes: `elementwise`, `reduction`, `logprob`, `attention`.

## Comparison roles

Reports must record `comparison_lhs_role` / `comparison_rhs_role`. A bare `baseline`
field is forbidden. C2 `singleton_aggregate` is an **execution/aggregation mode**, not
a comparison role.

| Report kind | `comparison_lhs_role` | `comparison_rhs_role` |
| --- | --- | --- |
| `forward_accuracy` | `bf16_candidate` | `fp32_reference` |
| `forward_invariance` | `transformed_config` | `canonical_config` |
| `train_infer_logprob_parity` | `training_style_teacher_forcing` | `inference_style_rollout_decode` |
| `gradient_accuracy` | `bf16_candidate` | `fp32_reference` |
| `gradient_invariance` | `transformed_config` | `canonical_config` |

Direction is locked so train/infer preserves:

```text
dlogp = train_logp - rollout_logp
ratio0 = exp(dlogp)
```

Swapping lhs/rhs without a different declared contract row hard-fails.

API: `resolve_comparison_roles()`, `assert_comparison_roles()`.

Aggregate callers must also provide `contract`, `report_kind`, and both roles; the
compute API validates the direction before calculating any metric. gtest reports
persist these roles on every output verdict. Backend reports must include
`BackendProvenance` (requested/actual backend, all four dtypes, and TF32 state),
which `validate_backend_provenance()` checks against the selected profile.

## Chain-level logprob aggregates

These three metrics are the **only** chain-level logprob / ablation aggregates for WS1
pass/fail. Gradients use independent `gradient_*` tensor verdicts and **do not** use
these aggregates.

Computed in FP32 on **active selected tokens only**:

```text
dlogp         = comparison_lhs_logp - comparison_rhs_logp
max_abs_dlogp = max(abs(dlogp))
approx_kl0    = mean(exp(dlogp) - 1 - dlogp)
clipfrac0     = mean(1[exp(dlogp) outside clip_interval])
```

Rules:

- **All three** must pass (`require_all=true`).
- Empty active-token set → hard fail.
- NaN / Inf in `dlogp`, `ratio0`, or aggregates → hard fail.
- Clip interval is pinned by the workload manifest (C2); the contract stores a default
  and the field name `clip_interval`.

API: `compute_logprob_aggregates()`, `judge_logprob_aggregates()`,
`resolve_chain_aggregate_thresholds()`.

## Resolver usage

```python
from rl_engine.kernels.gtest.tolerance import (
    load_contract,
    resolve_tolerance,
    resolve_dtype_policy,
    compute_logprob_aggregates,
    judge_logprob_aggregates,
    default_clip_interval,
)

contract = load_contract()
policy = resolve_dtype_policy(contract)

fwd = resolve_tolerance(
    contract,
    judgment="forward_accuracy",
    op_class="logprob",
    dtype="bfloat16",
    backend_profile="cuda_bf16",
)
bwd = resolve_tolerance(
    contract,
    judgment="gradient_accuracy",
    op_class="logprob",
    dtype="bfloat16",
    backend_profile="triton_cuda_bf16",  # same thresholds as cuda_bf16
)
inv = resolve_tolerance(
    contract,
    judgment="forward_invariance",
    op_class="attention",
    dtype="bfloat16",
)
# inv.mode == "bitwise", inv.atol == 0.0, inv.rtol == 0.0
```

`op_checks.run_operator_suite` resolves **forward_accuracy** for outputs and
**gradient_accuracy** for gradients.

## Compatibility keys

For older tests that still dig into:

- `contract["accuracy"]["default"][op_class][dtype]` — mirror of `forward_accuracy`
- `contract["batch_invariance"]` — `{atol: 0, rtol: 0}`

New code should call the resolvers above. Schema validation fails if the compatibility
mirror drifts from `forward_accuracy` or if invariance rows leave bitwise mode.

## Related issues

| ID | Role |
| --- | --- |
| #266 | WS1 closeout parent |
| #267 | This contract (C1) |
| #268 | Full-model workload / clip interval pin in manifest |
| #269 / #270 | Forward / gradient invariance harnesses |
| #276 | Full-model train/infer gate consuming this contract |
| #154 / #108 | Historical contract owners (superseded remaining work → C1) |

## Migration of existing tests

Most operator tests still use **private** `atol` / `rtol` literals. That is expected
after C1: the SSOT exists, but call sites have not all moved.

See the full inventory, priority, and “when it must migrate” map:

- [WS1 gtest 阈值迁移清单](ws1-gtest-migration-checklist.md)

How to register ops and run the CLI (post-#267):

- [gtest usage guide](../contributing/gtest-usage.md)
