# WS1 local defect log (do not reopen #145–#151)

In-repo record only. Do **not** file GitHub issues from this list unless the
maintainer asks. Hopper re-runs go through the same repro commands; kernel
fixes land as PRs against this log.

## rmsnorm-dweight

- **Ops:** `rms_norm`, `qk_norm`
- **Profiles:** `cuda_bf16`, `triton_cuda_bf16`
- **Judgment:** `gradient_invariance`
- **Symptom:** `dx` is bitwise 0; `dweight` fails chunk / N×B=1 singleton aggregate (shape-dependent bwd accum).
- **Repro:**
  ```bash
  python scripts/check_gradient_invariance.py --op rms_norm --candidate cuda --backend-profile cuda_bf16
  python scripts/check_gradient_invariance.py --op rms_norm --candidate triton --backend-profile triton_cuda_bf16
  ```
- **Hopper:** will not clear this. Needs a kernel-side `dweight` reduction that composes across launches.

## det-gemm-dw

- **Op:** `det_gemm`
- **Profiles:** `cuda_bf16`, `triton_cuda_bf16`
- **Judgment:** `gradient_invariance`
- **Symptom:** `dX` bitwise 0; `dW` fails the same class as RMSNorm `dweight`.
- **Repro:**
  ```bash
  python scripts/check_gradient_invariance.py --op det_gemm --candidate cuda --backend-profile cuda_bf16
  python scripts/check_gradient_invariance.py --op det_gemm --candidate triton --backend-profile triton_cuda_bf16
  ```
- **Hopper:** will not clear this.

## cuda-logp-no-backward

**Resolved on 2026-08-13:** `FusedLogpGenericOp` now has a row-local FP32
softmax VJP bridge. RTX 3060 C4 reports all `dlogits` invariance errors as 0.

- **Op:** `logp`
- **Profile:** `cuda_bf16` (C2 status is `declared`, not `missing_required`)
- **Judgment:** `gradient_accuracy` / `gradient_invariance`
- **Symptom:** `FusedLogpGenericOp` calls `_C.fused_logp` with no `torch.autograd.Function`; no `dlogits`.
- **Repro:**
  ```bash
  python scripts/check_gradient_invariance.py --op logp --candidate cuda --backend-profile cuda_bf16
  ```
- **Hopper:** will not clear this.

## triton-attention-left-pad

**Resolved on 2026-08-13:** the Triton kernel rebases a contiguous valid KV
interval to logical columns before both softmax reduction passes. The former
strict xfail now passes bitwise on RTX 3060 at Qwen3 `head_dim=128`.

- **Op:** `attention`
- **Profile:** `triton_cuda_bf16`
- **Judgment:** `forward_invariance`
- **Symptom:** causal `BN/padded_left` vs `BN/full` differs by one bf16 ULP at `head_dim=128` (token `(s2, 9)`). CUDA/Native are bitwise 0. C4 is green because Triton backward uses `NativeAttentionOp`.
- **Repro:**
  ```bash
  pytest tests/test_triton_batch_invariant_attention.py::test_triton_attention_causal_left_pad_matches_right_pad_bitwise
  python scripts/check_forward_invariance.py --op attention --candidate triton --backend-profile triton_cuda_bf16
  ```
- **Hopper:** will not clear this.

## Tracked C2 gaps (not new defects)

Triton `embedding`, `lm_head`, and plain `logp` are now declared candidates and
must be re-run through the C8 case runner; no fallback is permitted.

## Hopper-only cells (not defects)

CUDA `embedding` / `lm_head` / `rope` / `batch_invariant_logp` are `cuda-sm90`. Re-run after `KERNEL_ALIGN_FORCE_SM90=1 pip install -e .`:

```bash
python scripts/sweep_ws1_four_judgments.py --execute
```
