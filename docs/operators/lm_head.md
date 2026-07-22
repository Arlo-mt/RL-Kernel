# LM Head

The lm_head operator projects hidden states back to vocabulary logits, the final
layer of the Qwen3/Llama stack. It is a WS1 ground-truth reference for issue #108:
a pure-PyTorch definition of the correct answer that downstream fused CUDA/Triton
kernels are validated against.

- **LM Head** (`NativeLMHeadOp`): mathematically `out = hidden @ weight.t() (+ bias)`.
  The native reference implements this as row-wise fixed-K GEMV projections so the
  reference path is batch-invariant.

For Qwen3-8B the weight is the output projection `[vocab=151936, hidden=4096]` in the
HF `nn.Linear` `[out, in]` convention. It is independent from the embedding table
(`tie_word_embeddings=false`), and Qwen3 has no bias (`bias=None`).

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

lm_head = kernel_registry.get_op("lm_head")

logits = lm_head(hidden, weight)          # [B, S, hidden], [vocab, hidden] -> [B, S, vocab]
logits = lm_head(hidden, weight, bias=b)  # optional [vocab] bias
```

The op exposes the WS1 dual-path contract:

- `forward(...)` projects in the input dtype and returns the input dtype.
- `forward_fp32(...)` upcasts to fp32, accumulates in fp32, and returns fp32. The
  fixed-K projection runs with autocast disabled and CUDA TF32 turned off, so it
  stays a true fp32 reference regardless of the caller's ambient precision context.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeLMHeadOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA / ROCm / Triton | N/A | N/A | Planned: downstream fused kernels validate against this reference. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `hidden` | `[B, S, hidden]` or any leading dims | fp16/bf16/fp32 | Hidden states. |
| `weight` | `[vocab, hidden]` | fp16/bf16/fp32 | Output projection in HF `[out, in]` layout. |
| `bias` | `[vocab]` or `None` | fp16/bf16/fp32 | Optional; Qwen3 uses `None`. |
| output | `hidden.shape[:-1] + (vocab,)` | `forward`: hidden dtype; `forward_fp32`: fp32 | Logits. |

Output dtype follows `hidden`. The op is pure: no randomness and no in-place mutation.

## Accuracy

Reference semantics (`forward_fp32`):

```python
flat_hidden = hidden.float().reshape(-1, hidden.size(-1))
rows = [torch.mv(weight.float(), row) for row in flat_hidden]
out = torch.stack(rows).reshape(*hidden.shape[:-1], weight.size(0))
if bias is not None:
    out = out + bias.float()
```

- **Ground truth**: `forward_fp32` accumulates in and returns fp32, with autocast and
  CUDA TF32 disabled.
- **Dtype path**: `forward` runs the projection in the input dtype. Because this is a
  reduction over `hidden`, low-precision accumulation drifts from the fp32 reference and
  is checked with tolerance.
- **Axis-A batch invariance**: a row's logits are bitwise-identical regardless of batch
  size or padding. The native reference enforces this by flattening leading dimensions
  and projecting each row through the same GEMV-shaped K reduction instead of relying on
  batched GEMM, whose reduction tree can change with `M = batch * seq`.

## Dispatch Behavior

`kernel_registry.get_op("lm_head")` resolves through the `OpBackend` priority map. On
`cuda`, `rocm`, and `cpu`, the only registered backend today is the PyTorch native op
(`PYTORCH_NATIVE_LM_HEAD`), so every device dispatches to this reference. When fused
kernels land, they should preserve the same batch-invariant contract.

## Tests

```bash
python -m pytest tests/test_lm_head.py -v
```

Covers fp32 correctness vs the fixed-K reference, precision-context safety, bf16/fp16
accuracy, output shape, bias semantics, Axis-A batch invariance, input purity, gradient
flow to `hidden` and `weight`, registry dispatch, and a GPU-only smoke test at the real
Qwen3-8B dimensions.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/linear/lm_head.py`
- `rl_engine/kernels/registry.py`
- `tests/test_lm_head.py`
