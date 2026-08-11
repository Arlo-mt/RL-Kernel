# #267 (C1) closeout evidence

**Issue:** [#267](https://github.com/RL-Align/RL-Kernel/issues/267)  
**Parent:** [#266](https://github.com/RL-Align/RL-Kernel/issues/266) (C1 only; does **not** close #266)  
**Branch:** `feat/ws1-c1-tolerance-contract-267`  
**Commit:** `af4d9c2` (and follow-ups on the same branch)

## Acceptance criteria map

| AC | Status | Where |
| --- | --- | --- |
| BF16 exec, FP32 ref/accum, FP8 out, TF32 policy documented + tested | Pass | `tolerance_contract.json` `policy`; `test_dtype_policy_*`; `docs/design/ws1-numerical-contract.md` |
| Independent execution/accum/output/reference + backend provenance | Pass | `resolve_dtype_policy`, `validate_backend_provenance`; tests |
| Missing applicable four-judgment cell → hard fail; explicit N/A only when declared | Pass | `resolve_tolerance` / `resolve_tolerance_support`; schema + unit tests |
| BF16+FP32 mandatory; FP16 optional complete; FP8 hard-fail | Pass | schema validation + resolve tests |
| Gradient tolerances independent of forward | Pass | separate `gradient_accuracy` rows; `op_checks` uses `gradient_accuracy`; independence test |
| Batch/Chunk inv rows bitwise `atol=0,rtol=0` | Pass | `forward_invariance` / `gradient_invariance`; schema rejects nonzero |
| Aggregate formulas, roles, active mask, clip, empty/NaN rules + boundary tests | Pass | `compute_logprob_aggregates` / `judge_logprob_aggregates`; tests |
| Reports persist `comparison_lhs_role` / `comparison_rhs_role`; reversed roles hard-fail | Pass | `OutputCheck` fields; `assert_comparison_roles` |
| No bare `baseline`; `singleton_aggregate` not a comparison role | Pass | `comparison_roles.forbidden` + schema tests |
| Named resolve for three aggregates; all three in logprob pass/fail | Pass | `resolve_chain_aggregate_thresholds`; `require_all` |
| Docs: three aggregates sole chain logprob metrics; grads independent | Pass | numerical contract + gtest usage guide |
| New gates obtain thresholds only via shared resolver | Pass for gtest path | `op_checks` / `check_operator`; residual private-atol inventory tracked in migration checklist (C3/C4/C8) |
| CUDA + Triton same contract rows; no backend-private relaxation | Pass | shared thresholds; `backend_private_tolerance_relaxation=false` |

## Docking paths

- `rl_engine/kernels/gtest/tolerance_contract.json`
- `rl_engine/kernels/gtest/tolerance.py`
- `rl_engine/kernels/gtest/op_checks.py`
- `tests/test_tolerance_contract.py`
- `tests/test_op_checks.py`
- `docs/design/ws1-numerical-contract.md`
- `docs/contributing/gtest-usage.md`
- `docs/design/ws1-gtest-migration-checklist.md`

## Local verification

```bash
python -m pytest tests/test_tolerance_contract.py tests/test_op_checks.py -q
# 41 passed
```

Sample resolve (logprob BF16):

```text
forward_accuracy:    mode=tolerance atol=0.05 rtol=0.0  lhs=bf16_candidate rhs=fp32_reference
forward_invariance:  mode=bitwise   atol=0.0  rtol=0.0  lhs=transformed_config rhs=canonical_config
gradient_accuracy:   mode=tolerance (independent keys; does not read forward)
gradient_invariance: mode=bitwise   atol=0.0  rtol=0.0
```

## Explicitly out of this issue (still open under #266)

- C2 full-model workload / manifest pin of clip interval (#268)
- C3/C4 shared invariance harnesses (#269/#270)
- Migrating every historical private-atol pytest (checklist; C8 evidence)
- Full-model train/infer gate and CI (#276/#277)
- Closing parent #266

## Suggested issue comment when PR is green

```text
C1 complete on <PR URL>.

- Contract + resolver + schema tests (41 passed locally)
- op_checks: forward_accuracy vs gradient_accuracy
- Docs: ws1-numerical-contract.md, gtest-usage.md, migration checklist
- Residual private-atol in legacy op tests tracked for C3/C4/C8; not a C1 dock gap

Closing #267. Parent #266 remains open (C2–C11).
```
