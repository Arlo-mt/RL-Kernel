# WS1 C8 (#274) four-judgment matrix

> The sm86 tally shown below is the pre-fix historical snapshot. The current
> sweep executes representative case accuracy/VJP and C3/C4 logical
> invariance as separate evidence. Use its output, not the historical tally,
> for closeout; SM90-only full-vocab cases remain pending until H-card runs.

**Parent:** #266 · **Depends on:** C3 / C4 · **Not a substitute for #150 / C10**

C8 collects `backend_profile × case_id × op × {forward_accuracy, forward_invariance, gradient_accuracy, gradient_invariance}` using the existing C3 and C4 CLIs. It does not invent a third comparator.

Classify-only (CPU):

```bash
python scripts/sweep_ws1_four_judgments.py
```

Execute on a GPU (sm86 or Hopper):

```bash
python scripts/sweep_ws1_four_judgments.py --execute
```

On Hopper, `cuda-sm90` cells become runnable automatically. Rebuild the extension with `KERNEL_ALIGN_FORCE_SM90=1` first.

## Cell status

| Status | Meaning |
| --- | --- |
| `green` | C3/C4 gate passed |
| `red` | judgment failed, or required cell not executed |
| `blocked_hardware` | declared `cuda-sm90` on a non-Hopper box |
| `blocked_c2` | C2 `missing_required` (Triton embedding / lm_head / logp) |
| `skipped` | pack (layout_supported) or optional fused path |

Required untested is **red**, never bare N/A.

## Known reds (sm86, before Hopper)

See `docs/design/ws1-blockers.md`:

- `rms_norm` / `qk_norm` `dweight`
- `det_gemm` `dW`
- CUDA `logp` no backward
- Triton attention `padded_left` 1 ULP

C2 version is `ws1-c2-v5` after adding short+primary `case_id`s for the remaining required ops.

sm86 execute tally (RTX 3060): `green=88, red=32, blocked_hardware=32, blocked_c2=24, skipped=16`.

Hopper re-run (after `KERNEL_ALIGN_FORCE_SM90=1 pip install -e .`):

```bash
python scripts/sweep_ws1_four_judgments.py --execute
```

This document does **not** claim C8 close (#274 requires zero red).
