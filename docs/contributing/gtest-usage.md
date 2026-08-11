# gtest usage guide (operator candidate vs gold)

> **Audience:** contributors implementing train–inference / batch-invariant operators
> **Entry point:** `scripts/check_operator.py` + `rl_engine/kernels/gtest/*`
> **Numerical SSOT:** [#267](https://github.com/RL-Align/RL-Kernel/issues/267) four-judgment contract
> **Related:** [WS1 numerical contract](../design/ws1-numerical-contract.md) · [migration checklist](../design/ws1-gtest-migration-checklist.md)

This is the official how-to for the gtest harness: register an op, build inputs, run the CLI for forward/backward checks, and obtain tolerances from the shared contract (not private `atol`/`rtol`).

---

## 1. What gtest is for

gtest validates a **single operator**:

| Capability | Meaning |
|------------|---------|
| Gold | Usually a PyTorch / `forward_fp32` reference path |
| Candidate | CUDA / Triton / arch-specific implementation |
| Forward check | Outputs within contract tolerance (`forward_accuracy`) |
| Backward check | Selected input gradients within contract tolerance (`gradient_accuracy`, **independent of forward**) |

It is **not**:

- The full Qwen3-8B model-level gate (#266 C9/C10)
- The final cross-config invariance harness (C3/C4 build on the same contract)
- Real vLLM vs Megatron engine alignment

The CLI primarily covers **accuracy** (candidate vs gold).
**Invariance** (bitwise across configs) and **train/infer aggregates** use the contract APIs / later harnesses—do not invent private gate thresholds in tests.

---

## 2. End-to-end flow

```text
1) (Optional) register the op in the runtime registry
        ↓
2) gtest/operator_specs.py  → OP_SPECS: gold + candidates
        ↓
3) gtest/operator_inputs.py → build input shapes / values
        ↓
4) scripts/check_operator.py → run suite, load tolerance_contract.json
        ↓
5) report max_abs / tol / passed
```

### 2.1 Key files

| Path | Role |
|------|------|
| `rl_engine/kernels/gtest/operator_specs.py` | `OP_SPECS`: name, `op_class`, gold, candidates, grad inputs |
| `rl_engine/kernels/gtest/operator_inputs.py` | Default Qwen3-8B dims + `make_operator_inputs` |
| `rl_engine/kernels/gtest/op_checks.py` | Suite execution and comparison |
| `rl_engine/kernels/gtest/tolerance_contract.json` | Numerical contract SSOT |
| `rl_engine/kernels/gtest/tolerance.py` | `load_contract` / `resolve_tolerance` / chain aggregates |
| `scripts/check_operator.py` | **CLI entry** |

---

## 3. Step 1: register the op in `OP_SPECS`

Edit `rl_engine/kernels/gtest/operator_specs.py` and add an entry to `OP_SPECS`. Example shape (logp / linear_logp):

```python
"logp": OperatorSpec(
    name="logp",
    op_class="logprob",          # selects the contract op_class row
    gold_path="rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp",
    gold_method="forward_fp32",  # method invoked on the gold instance
    candidate_paths={
        "pytorch": "rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp",
        "cuda": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp",
        "cuda-sm90": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpSM90Op",
    },
    grad_input_names=("logits",),  # inputs compared under --check-grad
),
```

### 3.1 `OperatorSpec` fields

| Field | Meaning |
|-------|---------|
| `name` | Value for CLI `--op` |
| `op_class` | Contract class: `elementwise` / `reduction` / `logprob` / `attention` |
| `gold_path` | Gold class path `module.Class` |
| `gold_method` | Method name, e.g. `forward_fp32`, `apply`, `__call__` |
| `candidate_paths` | Map `candidate name → implementation class`; CLI `--candidate cuda` looks up this map |
| `grad_input_names` | With `--check-grad`, enable grads and compare these inputs; missing config errors |

**Only ops registered in `OP_SPECS` can be invoked via `check_operator.py`.**

Currently registered (source of truth is the code):

```text
rms_norm, attention, logp, linear_logp, embedding, lm_head,
det_gemm, rope, silu, swiglu, batch_invariant_logp
```

---

## 4. Step 2: build inputs

File: `rl_engine/kernels/gtest/operator_inputs.py`.

### 4.1 Default model dims (Qwen3-8B Dense semantics)

Macros at the top of the file (local experiments may change them; WS1 full-model EXIT uses the official config fingerprint):

```text
DEFAULT_HIDDEN       = 4096
DEFAULT_N_HEADS      = 32
DEFAULT_N_KV_HEADS   = 8
DEFAULT_HEAD_DIM     = 128
DEFAULT_INTERMEDIATE = 12288
DEFAULT_VOCAB        = 151936
DEFAULT_ROPE_THETA   = 1.0e6
DEFAULT_RMS_EPS      = 1.0e-6
```

### 4.2 Shape names and input builders

- `operator_shape_name(op_name, args)` — human-readable case name (e.g. `2x16x257`)
- `_make_*_inputs` / `make_operator_inputs` — build the input dict from `--op` and CLI args
  - `random`: reproducible randomness (`--seed` plus per-tensor offsets)
  - `constant`: fixed values for debugging (`--constant-value` / `--token-value`)

When adding an op: extend the shape map and implement the matching `_make_xxx_inputs`.

### 4.3 Suggested GRPO-oriented shapes (local sweeps)

For GRPO, `B = P × G`. With `G=8`, batch is often a multiple of 8.
`B=1` is fine for smoke; fuller sweeps may use:

```text
B ∈ {1, 8, 16, 32, 64}
S ∈ {1, 31, 33, 127, 129, 255, 256, 257, 512, 1024, 4096, 8192}
```

Prefer short `S` when VRAM is tight; full-model gates are owned by #266 / C2.

---

## 5. Step 3: run the CLI

```bash
# From the repo root; prefer an editable install: pip install -e .
python scripts/check_operator.py --op logp --candidate pytorch --device cpu --dtype fp32 --batch 1 --seq 2 --vocab 17
```

### 5.1 Common examples

**Smoke (CPU / PyTorch self-check)**

```bash
python scripts/check_operator.py \
  --op logp --candidate pytorch --device cpu --dtype fp32 \
  --batch 1 --seq 2 --vocab 17
```

**Triton `linear_logp` + backward (BF16)**

```bash
python scripts/check_operator.py \
  --op linear_logp --candidate triton --device cuda --dtype bf16 \
  --batch 1 --seq 2 --vocab 1024 --normalized-dim 4096 \
  --check-grad
```

**CUDA deterministic attention + gradients**

```bash
python scripts/check_operator.py \
  --op attention --candidate cuda --device cuda --dtype bf16 \
  --batch 2 --seq 64 --check-grad --grad-mode random
```

**Full JSON report**

```bash
python scripts/check_operator.py --op rms_norm --candidate cuda --dtype bf16 --device cuda --json
```

### 5.2 CLI flags

| Flag | Meaning |
|------|---------|
| `--op` | Operator name from `OP_SPECS` |
| `--candidate` | Backend: `pytorch` / `cuda` / `cuda-generic` / `cuda-sm90` / `triton` / … (see that op’s `candidate_paths`) |
| `--dtype` | `fp32` / `bf16` / `fp16`; selects input dtype and contract row |
| `--device` | `auto` / `cpu` / `cuda` |
| `--batch` / `--seq` | Batch size and sequence length for inputs |
| `--vocab` | Vocab size; logp logits `[B,S,V]`; linear_logp weight `[V,H]` |
| `--input-mode` | `random` (default) or `constant` |
| `--constant-value` | Float fill in constant mode |
| `--token-value` | Token id in constant mode |
| `--normalized-dim` | Hidden dim for rms_norm / linear_logp, etc. |
| `--k-dim` / `--n-dim` | Matmul / det_gemm dims |
| `--theta` | RoPE theta |
| `--eps` | RMSNorm epsilon |
| `--seed` | Input RNG seed (per-tensor offsets still apply) |
| `--arch-key` | Arch override key, e.g. `sm90` (contract `arch_overrides`) |
| `--check-grad` | Also compare gradients (requires `grad_input_names`) |
| `--grad-mode` | `random` (default, stricter) / `ones` (≈ `output.sum().backward()`) |
| `--grad-seed` | Seed for random upstream gradients |
| `--json` | Print the full structured report |

---

## 6. Where tolerances come from (after #267)

### 6.1 Before vs after C1

| Before | After (C1 / #267) |
|--------|-------------------|
| Mostly `accuracy[op_class][dtype]` | **Four judgments**: forward/gradient × accuracy/invariance |
| Forward and grad often shared one tol | **Grad uses `gradient_accuracy` only** (no silent forward inheritance) |
| Flat threshold table | Plus dtype policy, comparison roles, chain logprob aggregates |

### 6.2 Which judgments the CLI / `op_checks` use

`run_operator_suite` / `check_operator.py`:

| Comparison | Judgment |
|------------|----------|
| Output vs gold | `forward_accuracy` |
| Gradient vs gold | `gradient_accuracy` |

Batch/chunk **bitwise invariance** and train/infer **three aggregates** are not separate CLI switches. Use:

```python
from rl_engine.kernels.gtest.tolerance import (
    load_contract,
    resolve_tolerance,
    compute_logprob_aggregates,
    judge_logprob_aggregates,
    default_clip_interval,
)

contract = load_contract()
# Cross-config invariance (gate path)
inv = resolve_tolerance(
    contract,
    judgment="forward_invariance",  # or gradient_invariance
    op_class="attention",
    dtype="bfloat16",
    backend_profile="cuda_bf16",
)
# inv.mode == "bitwise", inv.atol == inv.rtol == 0

# Train vs infer selected-logprob
agg = compute_logprob_aggregates(
    train_logp,
    rollout_logp,
    active_mask,
    contract=contract,
    report_kind="train_infer_logprob_parity",
    clip_interval=default_clip_interval(contract),
    comparison_lhs_role="training_style_teacher_forcing",
    comparison_rhs_role="inference_style_rollout_decode",
)
verdict = judge_logprob_aggregates(agg, contract, execution_dtype="bfloat16")
```

### 6.3 Policy locks (WS1)

| Item | Value |
|------|--------|
| Execution | BF16 mandatory for EXIT (CLI may still exercise fp32/fp16) |
| Reference / accumulation | FP32 |
| FP8 | Out of scope (resolve hard-fails) |
| TF32 | Disabled |

WS1 evidence must attach checked provenance to its candidate report:

```python
from rl_engine.kernels.gtest import BackendProvenance, CandidateSpec

provenance = BackendProvenance(
    backend_profile="cuda_bf16",  # use triton_cuda_bf16 + triton for Triton
    requested_backend="cuda",
    actual_backend="cuda",
    execution_dtype="bfloat16",
    accumulation_dtype="float32",
    output_dtype="bfloat16",
    reference_dtype="float32",
    candidate_tf32_enabled=False,
    reference_tf32_enabled=False,
)
candidate = CandidateSpec(
    name="cuda-candidate",
    backend="cuda",
    fn=op,
    provenance=provenance,
)
```

The suite rejects backend fallback, dtype drift, TF32 enablement, and observed output
dtypes that disagree with this provenance before producing a passing report.
| Profiles | `cuda_bf16` and `triton_cuda_bf16` share **the same** thresholds |

**Do not** use private `atol=1e-5` (etc.) as WS1 gate evidence. Inventory of legacy private thresholds: [migration checklist](../design/ws1-gtest-migration-checklist.md).

### 6.4 Report `tol=(atol=..., rtol=...)`

The CLI summary line:

```text
tol=(atol=..., rtol=...)
```

comes from the shared resolver—not hard-coded constants inside `check_operator.py`.

---

## 7. Recommended local test order

```text
1. --candidate pytorch --device cpu --dtype fp32
   → registration / inputs / plumbing smoke

2. Same shape with --dtype bf16 --device cuda --candidate triton|cuda
   → real candidate forward

3. Add --check-grad --grad-mode random
   → gradients (random upstream grads catch more bugs than ones)

4. --arch-key sm90 only when you need arch-specific contract overrides

5. Cross batch/layout: not CLI-only; use invariance judgments + dedicated tests
```

---

## 8. Common failures

| Symptom | Likely cause |
|---------|----------------|
| Unsupported / missing `--op` choice | Not registered in `OP_SPECS` |
| `--check-grad` missing grad inputs | Empty/wrong `grad_input_names` vs input keys |
| Candidate import error | Bad `candidate_paths` or extension not built |
| BF16 over tolerance | Confirm gold is `forward_fp32`; check contract row; do not loosen private atol |
| Missing SM90 symbols | Build without SM90 / non-sm90 GPU; pick another candidate or rebuild |
| Want FP8 | Hard-fail under WS1 contract; out of scope |

---

## 9. Relationship to pytest

| Path | Use |
|------|-----|
| `python scripts/check_operator.py ...` | Fast single-op shape/debug loops |
| `pytest tests/test_*.py` | Regression, invariance, integration |
| `pytest tests/test_tolerance_contract.py` | Contract schema / resolver |

Both paths should take thresholds from `tolerance_contract.json`.
New pytest code should call `resolve_tolerance` instead of copying magic numbers.

---

## 10. Minimal checklist for a new operator

- [ ] Implementation under `rl_engine/kernels/ops/{pytorch,cuda,triton}/...`
- [ ] (Optional) runtime `registry` registration
- [ ] `OP_SPECS` entry: gold + candidates + `op_class` + `grad_input_names`
- [ ] `operator_inputs` shape name + input builder
- [ ] `check_operator.py` smoke + bf16 + `--check-grad` green
- [ ] Contract already has the `op_class` row (extend schema + `test_tolerance_contract` if not)
- [ ] No new private `atol`/`rtol` as gate evidence
- [ ] Operator docs point at the contract for thresholds (do not restate ad-hoc numbers)

---

## 11. Further reading

| Doc | Content |
|-----|---------|
| [ws1-numerical-contract.md](../design/ws1-numerical-contract.md) | Four judgments, roles, aggregate formulas |
| [ws1-gtest-migration-checklist.md](../design/ws1-gtest-migration-checklist.md) | Which tests still use private thresholds and when to migrate |
| [testing.md](testing.md) | Short testing entry points |
| Issues [#266](https://github.com/RL-Align/RL-Kernel/issues/266) / [#267](https://github.com/RL-Align/RL-Kernel/issues/267) | WS1 closeout and C1 contract |

---

## 12. Changelog

| Date | Notes |
|------|--------|
| 2026-08-11 | Initial English guide aligned with C1; documents CLI, `OP_SPECS`, inputs, and contract usage |
