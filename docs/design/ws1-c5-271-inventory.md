# WS1 C5 (#271) elementwise / RoPE inventory

**Parent:** #266 · **Depends on:** C2 / C3 / C4 · **Does not wait for C8 close**

C5 is a written inventory. Differentiable on-chain items reuse C3/C4. CUDA
RoPE remains a Hopper-only evidence item; no sm86-reproducible elementwise or
RoPE defect remains open.

## Inventory

| Item | CUDA | Triton | Evidence |
| --- | --- | --- | --- |
| `rope` | blocked_hardware (sm90) | pass | C3/C4 adapters; Triton green on sm86 |
| `silu` | pass | pass | C3 + C4 green both profiles |
| `swiglu` | pass | pass | C3 + C4 green both profiles |
| `residual_add` | pass | pass | `torch.add`; no cross-batch reduction |
| `scale` | pass | pass | `1/sqrt(head_dim)` broadcast |
| `bias` | pass | pass | official fingerprint `attention_bias=false` |
| `mask_fill` | pass | pass | Triton valid KV interval is rebased to logical reduction lanes |
| `dtype_cast` | pass | pass | C1 policy; provenance rejects drift |

Source of truth: `rl_engine/kernels/gtest/elementwise_inventory.py`.

## Hopper re-run

On sm90, re-check CUDA `rope` (and embedding / lm_head, which C8 owns) with:

```bash
python scripts/check_forward_invariance.py --op rope --candidate cuda-sm90 --backend-profile cuda_bf16
python scripts/check_gradient_invariance.py --op rope --candidate cuda-sm90 --backend-profile cuda_bf16
```
