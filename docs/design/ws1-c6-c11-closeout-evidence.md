# WS1 C6–C11 closeout evidence

**Parent:** #266 · **Branch:** `feat/ws1-c6-c11-closeout-266`

C9 green is assembly only. Full WS1 EXIT still needs H20 execute of C10/C11
plus parent A/B/Final comments.

## What landed

| ID | Code | CPU evidence |
| --- | --- | --- |
| C6 | `rl_engine/kernels/gtest/kv_consistency.py`, `scripts/check_decode_prefill.py` | `tests/test_kv_consistency.py` |
| C7 | `StatefulKVCache` + same harness, `scripts/check_stateful_kv.py` | B1 writer/reader + generate-rescore; B2=`absent` |
| C9 | `rl_engine/alignment/qwen3_dense.py`, `scripts/ws1_chain_fwd_bwd.py` | topology / official fingerprint / profile resolution |
| C10 | `rl_engine/kernels/gtest/chain_gate.py`, `scripts/ws1_chain_gate.py` | report schema + bitwise `atol=0` rule |
| C11 | `ci/run_ws1_chain_gate.sh`, `.github/workflows/ws1-chain-gpu.yml` | CPU schema jobs in `ci.yml`; GPU job is required |

Private `_DECODE_ATOL` / `_PADDING_ATOL` were removed from
`tests/test_kv_cache_attention.py`. Those checks now resolve C1
`forward_accuracy` / attention.

## Local GPU evidence (not EXIT)

NVIDIA GeForce RTX 4070 Ti SUPER, CC 8.9, this branch:

| Gate | Profile | Result |
| --- | --- | --- |
| C6 `check_decode_prefill.py` | `cuda_bf16` | passed; 6/6 cells; attn max_abs = 0 |
| C6 `check_decode_prefill.py` | `triton_cuda_bf16` | passed; 6/6 cells; attn max_abs = 0 |
| C7 `check_stateful_kv.py` | `cuda_bf16` | B1 + generate-rescore passed; B2=`absent` |
| C7 `check_stateful_kv.py` | `triton_cuda_bf16` | B1 + generate-rescore passed; B2=`absent` |

C9/C10/C11 full 36-layer + pinned weights were **not** run here (16 GB card).
That execute is the remaining closeout step on H20.

Local pytest totals are command-scoped development evidence, not an EXIT
criterion. Always quote the exact test command and commit with a pass count;
do not cite a bare aggregate such as `220 passed` as closeout evidence.

## H20 commands (user-run)

Pinned Qwen3-8B snapshot (C2 revision + shard hashes):

```bash
export QWEN3_8B=/path/to/Qwen3-8B   # safetensors at revision b968826d9c46dd6066d109eabc6255188de91218
export KERNEL_ALIGN_FORCE_SM90=1

python scripts/prepare_ws1_weights.py --output "$QWEN3_8B" --verify-only

python scripts/check_decode_prefill.py --backend-profile cuda_bf16
python scripts/check_decode_prefill.py --backend-profile triton_cuda_bf16
python scripts/check_stateful_kv.py --backend-profile cuda_bf16
python scripts/check_stateful_kv.py --backend-profile triton_cuda_bf16

python scripts/ws1_chain_fwd_bwd.py --backend-profile cuda_bf16 --weights hf --weights-path "$QWEN3_8B"
python scripts/ws1_chain_fwd_bwd.py --backend-profile triton_cuda_bf16 --weights hf --weights-path "$QWEN3_8B"

python scripts/ws1_chain_gate.py --backend-profile cuda_bf16 --model qwen3-8b-dense --dtype bfloat16 --weights required --weights-path "$QWEN3_8B" --json
python scripts/ws1_chain_gate.py --backend-profile triton_cuda_bf16 --model qwen3-8b-dense --dtype bfloat16 --weights required --weights-path "$QWEN3_8B" --json
```

Bind the two C10 JSON files + `git rev-parse HEAD` + GPU/CC on the parent
issue before closing #266.

Each accepted JSON must report `backward_executed=true`,
`train_infer_executed=true`, a null `first_drift`, verified weight content hash,
complete runtime observations for all nine required node kinds, and
`git_dirty=false`. Generate the JSON only after committing the gate code; the
CI wrapper rejects dirty-worktree evidence.

The C10 backward contract is deliberately scoped to a representative subset,
not an all-parameter 8B training backward. The JSON must say
`gradient_scope=representative_parameter_subset`,
`all_parameter_gradients=false`, and list exactly these three required leaf
gradients: `norm.weight`, `lm_head.weight`, and
`layers.0.input_layernorm.weight`. This proves training-style backward and
cross-cell gradient invariance for the gate's contract; it must not be cited
as bitwise equality of every model parameter's gradient.

## Not claimed

- Multi-GPU / vime / real vLLM vs Megatron
- C9 skeleton alone as EXIT
- Production paged-KV (C7 B2 is explicitly absent)
