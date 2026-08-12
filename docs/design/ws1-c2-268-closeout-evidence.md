# WS1 C2 (#268) Closeout Evidence

**Issue:** #268 · **Parent:** #266 · **Workload:** `ws1-qwen3-8b-dense-primary-v3`  
**Branch:** `feat/ws1-c2-canonical-workload-268`

## Deliverables

| Path | Role |
| --- | --- |
| `rl_engine/testing/ws1_manifest.json` | SSOT workload identity / matrix / profiles / cases |
| `rl_engine/testing/ws1_workload.py` | Load, validate, logical identity, pad/pack/chunk restore |
| `scripts/ws1_reference.py` | One-command reference emission |
| `tests/test_ws1_workload.py` | CPU acceptance tests |
| `docs/design/ws1-c2-268-workload-plan.md` | Landing plan |
| `docs/design/ws1-c2-268-closeout-evidence.md` | This map |

## Acceptance criteria map

| #268 AC | Status | Evidence |
| --- | --- | --- |
| Manifest pins numerics-affecting fields | **Pass** | model, seed, tokens, prompt/completion lenses, masks, positions, dtypes, clip, aggregates, RNG, TF32 ref |
| Full Qwen3-8B Dense identity + weight hash | **Pass** | config fingerprint + shard SHA-256 `content_hash` |
| Same workload ID → same fixture/reference identity | **Pass** | `fixture_identity_sha256` + `fixture_hash` tests |
| pad/pack/chunk restore logical identity | **Pass** | `apply_padding` / `apply_packing` / `apply_chunking` + restore tests |
| B1 singleton_aggregate vs BN same multiset | **Pass** | `singleton_aggregate_plan` test |
| Naming: singleton_aggregate ≠ C1 roles; no bare baseline | **Pass** | `forbidden_comparison_roles` + `report_naming` |
| 2×2 + perm + multi-chunk non-divisible + pad/varlen | **Pass** | primary matrix + varlen samples `[11,16,13,19]` |
| clip_interval for clipfrac0 | **Pass** | `[0.8, 1.2]` aligned with C1 |
| Dropout/sampling/RNG policy; undeclared hard-fail | **Pass** | `stochastic_policy` + helper test |
| Short + representative fixtures hit declared candidates | **Pass\*** | fixture `candidate_case_ids` + registry resolution tests |
| Stable case_id for C8/C10/C11 reference | **Pass** | `representative_cases[].case_id` |
| expected + actual backend/kernel + algorithm property | **Pass\*** | registry-resolved actual; runtime observation owned by C8+ (declared in manifest) |
| One command emits reference (workload ID, seed, dtype) | **Pass** | `scripts/ws1_reference.py` |
| Packing / QK-Norm / required ops status | **Pass** | packing supported + packed fixture; qk_norm required |
| Both profiles enumerate required nodes; no untracked missing | **Pass** | Triton gaps are `missing_required` (red, tracked) |

\*C2 binds **registry-resolved** candidate paths. Live GPU dispatch observation is explicitly out of C2 (`provenance_boundary`) and owned by C3/C8/C10/C11.

## Verification commands

```bash
# From repo root with PYTHONPATH=repo root (or editable install)
python -m pytest tests/test_ws1_workload.py -q
python scripts/ws1_reference.py --dtype bf16 --cell-id BN/full --emit-json -
```

Expected: all C2 tests green; CLI prints `workload_id`, `seed`, `dtype`, `fixture_hash`, and `reference_outputs` digests.

## Residual (explicitly not #268)

| Item | Owner |
| --- | --- |
| Runtime observed actual backend on GPU | C3 / C8 / C10 / C11 |
| Triton `missing_required`: embedding, lm_head, logprob | later candidate work / Blocker; tracked red in C2 |
| #150 numerical asserts / full-model e2e | C9 / C10 |
| Full WS1 EXIT | #266 after C1–C11 |

## Close recommendation

Close **#268** once this branch is merged. Do **not** claim #266 WS1 EXIT from C2 alone.
