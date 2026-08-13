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
| `pending_hopper` | declared `cuda-sm90` on a non-Hopper box |
| `N/A` | pack (layout_supported) with a C2/C4 reason |

Required untested is **red**, never bare N/A.

## Close status

H20 execute is checked in at `docs/design/ws1-c8-execute.json`: **green=176, N/A=16, red=0**. See `docs/design/ws1-c8-274-closeout-evidence.md`.

Classify-only still paints declared-but-unexecuted cells red. That is required-untested, not a close blocker once `--execute` is green.
