# WS1 C4 (#270) closeout evidence

**Parent:** #266 · **Depends on:** #267 / #268 · **Branch:** `feat/ws1-c4-gradient-invariance-270`
**Scope:** shared gradient harness + enumerable adapters only

## Acceptance map

| #270 / #266 criterion | Evidence |
| --- | --- |
| Cross-config API | `assert_gradient_batch_invariant(...) -> GradientInvarianceReport` |
| Accuracy vs invariance | `accuracy_reports` (`gradient_accuracy`) and `invariance_reports` / `singleton_aggregate_reports` (`gradient_invariance`) |
| Batch/Chunk bitwise after logical aggregation | C1 `gradient_invariance` resolver; adapters run on `config.physical_layout` and token grads restore through C2's map |
| Shared upstream / reduction / denom | Harness injects `active_token_denominator`, `loss_reduction`, `aggregation_order`; adapters seed `autograd.grad` with an upstream that is a pure function of logical identity |
| Stable grad names | `GRADIENT_ADAPTERS` (`dx`/`dweight`, `dX`/`dW`, `dQ/dK/dV`, …) |
| Every differentiable WS1 op enumerable | Registry + `test_required_ops_are_enumerable`; all 13 runnable adapters execute the full 13-cell matrix |
| Pack / KV rule | Pack registered (`layout_supported`), inactive tokens contribute 0; KV `absent_not_required` |
| Missing Triton required node is red | Status matrix marks C2 `missing_required` embedding / lm_head / logp as tracked red; CLI refuses them |
| No cross-profile borrow | Declared CUDA and Triton candidate paths must differ |
| No `atomicAdd` | Source-file audit on listed BI candidates (zero `atomicAdd` in `csrc/`) |
| No private thresholds | All compares go through the C1 resolver |
| Diagnostics | max abs/rel per named gradient, first failing op/tensor/config pair |
| C4 ≠ EXIT | This document does not claim C8/C10/C11 or full-model #150 |

## The matrix is falsifiable

Each C2 cell now hands the operator a genuinely different physical input. For
`rms_norm` with `hidden=8`:

| Config | Operator calls | Row shapes |
| --- | --- | --- |
| `BN/full` | 1 | `(59, 8)` |
| `BN/chunked` | 10 | `(7,8) (4,8) (7,8) (7,8) (2,8) (7,8) (6,8) (7,8) (7,8) (5,8)` |
| `B1-singleton_aggregate/full/s0` | 1 | `(11, 8)` |
| `BN/padded_right` / `BN/padded_left` | 1 | `(80, 8)` |
| `BN/permuted` | 1 | `(59, 8)` in permuted sample order |

`tests/test_gradient_invariance.py::TestPhysicalLayout` locks this in: a
layout-sensitive synthetic operator must be judged **red**, a
logical-identity-only operator must be judged **green**, `B=N` must be one
batched call, chunking must split it, and padding must reach the operator.
Without those guards a regression to layout-blind adapters makes every bitwise
verdict a tautology.

CPU-safe contract regression:

```bash
.venv/bin/python -m pytest -q \
  tests/test_tolerance_contract.py \
  tests/test_ws1_workload.py \
  tests/test_forward_invariance.py \
  tests/test_gradient_invariance.py \
  tests/test_op_checks.py
```

Result: `133 passed` (`test_gradient_invariance` 26 passed). `mypy` and
`flake8`/`black`/`isort` clean on the C4 files.

## Runtime verification

Every runnable adapter was swept against its C2-declared candidate on both
required profiles, on NVIDIA GeForce RTX 4070 Ti SUPER (`sm89`):

```bash
.venv/bin/python scripts/sweep_gradient_invariance.py
```

The sweep classifies every cell (`green` / `red_verdict` / `red_no_backward` /
`blocked_hardware` / `blocked_c2` / `skipped`) and exits non-zero unless all
non-skipped cells are green. Single cells still run through
`scripts/check_gradient_invariance.py --op … --candidate … --backend-profile …`.

Current tally: `green=8, red_verdict=6, red_no_backward=1, blocked_hardware=4,
blocked_c2=3, skipped=4`.

**Re-running on Hopper.** The four `blocked_hardware` cells are the only ones a
different GPU can resolve. On an H20 (`sm90`) the extension must first be built
with SM90 kernels, otherwise the candidates still fail to load:

```bash
KERNEL_ALIGN_FORCE_SM90=1 pip install -e .
python scripts/sweep_gradient_invariance.py
```

Hopper does **not** change the two open findings below or the three Triton
`missing_required` nodes — those are implementation gaps, not hardware gaps.

| Op | `cuda_bf16` | `triton_cuda_bf16` |
| --- | --- | --- |
| `attention` | **green** | **green** |
| `silu` | **green** | **green** |
| `swiglu` | **green** | **green** |
| `rope` | needs Hopper (`cuda-sm90`) | **green** |
| `batch_invariant_logp` | needs Hopper (`cuda-sm90`) | **green** |
| `rms_norm` / `qk_norm` | red — `dweight` | red — `dweight` |
| `det_gemm` | red — `dW` | red — `dW` |
| `embedding` / `lm_head` | needs Hopper (`cuda-sm90`) | C2 `missing_required` |
| `logp` | red — **no backward** | C2 `missing_required` |
| `linear_logp` | skip (`optional_fused`, no C2 node) | skip |
| `pack` | profile-independent (CPU contract test) | profile-independent |

7 of 26 cells green. Detail for the reds:

- **`dweight` / `dW` chunk + singleton aggregate** — `rms_norm` and `qk_norm`
  `dweight` max abs `1.77621841e-04` (chunk) and `3.07083130e-04` (N× B=1
  aggregate), identical on both profiles; `det_gemm` `dW` the same class. `dx`,
  `dX`, permutation and padding are all `0.0` bitwise. See the open finding
  below.
- **`logp` has no backward** — `FusedLogpGenericOp` is not a
  `torch.autograd.Function`, so its output has no `grad_fn`. Reported as
  `MissingBackwardError` → a categorised red, not an autograd stack trace.
- **Hopper-only cells** — the `cuda_bf16` profile declares `cuda-sm90`
  candidates for `embedding`, `lm_head`, `rope` and `batch_invariant_logp`.
  Complete CUDA-profile evidence requires a Hopper GPU with
  `KERNEL_ALIGN_FORCE_SM90=1`; this box cannot produce it, and the CLI refuses
  rather than falling back.
- **`pack`** — `layout_supported`, the same PyTorch op under both profiles and
  not a C2 backend node. C1 provenance requires
  `requested == actual == profile backend family`, so a per-profile gate could
  only pass by recording a backend that never ran. The CLI refuses; its
  gradient contract is covered on CPU instead.

The earlier "`0.0` everywhere, `rms_norm` only" evidence is **void**: it was
produced by adapters that ignored `config.physical_layout` and ran `B=N` as
N× `B=1`, so every cell compared one computation against itself.

The GPU gate also needs shapes the real kernels accept — the deterministic CUDA
attention requires `head_dim == 128`, so the CLI exposes `--n-heads`,
`--n-kv-heads` and `--head-dim` and defaults to a runnable shape.

## Open finding — CUDA `logprob` has no backward

`FusedLogpGenericOp` (`rl_engine/kernels/ops/cuda/loss/logp.py:94-133`) calls
`_C.fused_logp` directly and is not wired through `torch.autograd.Function`, so
`dlogits` cannot be produced at all. C2 declares `cuda_bf16 / logprob` as
`declared`, but #270 requires `dlogits` as a stable gradient name on the
training path. This is the same class as the three Triton `missing_required`
nodes, except C2 does not record it — so it is a **Blocker candidate**, not a
`missing_required` row that can simply be tracked.

## Open finding — RMSNorm `dweight` is not chunk/batch decomposable

`dx` is bitwise invariant across the whole matrix on both profiles. `dweight`
is not, and the cause is a row-count-dependent accumulation shape:

- CUDA: `csrc/cuda/rmsnorm.cu:71-75` fixes `RMSNORM_DW_ROWS_PER_CHUNK = 256` and
  derives `chunks = ceil(T / 256)`; `rmsnorm_partial_dw_kernel` left-folds rows
  inside a chunk (`csrc/cuda/rmsnorm.cu:181-196`).
- Triton: `_rmsnorm_bwd_dw_kernel` accumulates `acc += tl.sum(vals)` over
  `tl.range(0, T, BLOCK_T)` (`rl_engine/kernels/ops/triton/rmsnorm_triton.py:48-58`).

Both are deterministic for a fixed `T`, but splitting the same tokens across
launches re-associates the sum: a left fold over 59 rows is not bitwise equal to
the sum of left folds over 11 + 16 + 13 + 19 rows. That is precisely the
`shape_dependent_bwd_accum = forbidden` property the adapter registry declares —
previously asserted only as a string, never as behaviour. `det_gemm`'s `dW`
fails the same way on both profiles.

Per #266 this is a **Blocker candidate**, not a reason to reopen #145, and per
the C4 plan (§8) fixing the kernel is outside C4 (audit, not rewrite). Making it
green requires a `dweight` accumulation whose granularity composes across
launches — e.g. reducing in fixed row blocks aligned to logical sample
boundaries rather than to the per-launch row count.

Tracked red (unchanged, not N/A, not a silent pass): Triton `embedding`,
`lm_head`, and plain `logp` remain C2 `missing_required`. C4 surfaces them in
the status matrix and refuses to run them, so #270's "CUDA and Triton required
gradient adapters are complete and green" box stays unticked.

## Parent boundary

This closes only the C4 harness, adapter registry, and canonical aggregation
contract that C8/C10 must reuse. It does not claim the full-model, KV-cache, or
CI EXIT requirements of #266.
