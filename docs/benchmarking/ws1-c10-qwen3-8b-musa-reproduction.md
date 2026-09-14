# WS1 C10 Qwen3-8B Dense MUSA Reproduction

固定版本：

```text
Repository: https://github.com/Arlo-mt/RL-Kernel
Branch: musa-native-qwen3-8b-dense
Commit: c2653f47850d4a86b59dd2318e632218a690905f
```

## 1. 获取代码

```bash
git clone --branch musa-native-qwen3-8b-dense \
  https://github.com/Arlo-mt/RL-Kernel.git
cd RL-Kernel
git switch --detach c2653f47850d4a86b59dd2318e632218a690905f
git rev-parse HEAD
```

必须输出：

```text
c2653f47850d4a86b59dd2318e632218a690905f
```

## 2. 权重

准备固定的 Qwen3-8B 权重目录：

```bash
export QWEN3_8B=/home/models/Qwen3-8B
```

验证权重：

```bash
python3 scripts/prepare_ws1_weights.py \
  --output "$QWEN3_8B" \
  --verify-only
```

固定校验值：

```text
model.safetensors.index.json
index_sha256:
f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc

content_hash:
fc664a19c52c82b6f5ddb33d4fe2723181daeb93a344b16fee6369963e5a13a5
```

服务器不能访问外网时，不要在服务器直接下载；在本地准备完成后通过 `scp` 传输整个权重目录。

## 3. 检查 MUSA

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.musa.is_available())
print(torch.musa.device_count())
print(torch.musa.get_device_name(0))
PY

mthreads-gmi
```

执行长测前，8 张卡应为空闲。至少确认目标卡显存没有被其它训练或 benchmark 占用。

## 4. 编译 Native Extension

```bash
export MUSA_HOME=/usr/local/musa
export TORCH_MUSA_ARCH_LIST=31
export RL_KERNEL_REQUIRE_EXT=1
export MAX_JOBS=1

python3 setup.py build_ext --inplace
```

确认 native symbols：

```bash
python3 - <<'PY'
import rl_engine._C as C
required = [
    "musa_embedding_forward",
    "musa_rmsnorm_forward",
    "musa_rope_apply",
    "musa_attention_forward",
    "musa_swiglu_forward",
    "musa_selected_logp_forward",
    "musa_batch_invariant_logp_forward",
    "musa_lm_head_fp32",
    "musa_rowwise_gemm_fp32",
]
missing = [name for name in required if not hasattr(C, name)]
if missing:
    raise SystemExit(f"missing native symbols: {missing}")
print("native extension OK")
PY
```

## 5. Native/Profile Regression

```bash
python3 -m pytest -q \
  tests/test_musa_ws1_native.py \
  tests/test_ws1_qwen3_dense.py \
  tests/test_ws1_workload.py \
  --disable-warnings
```

复现结果：

```text
47 passed
```

## 6. WS1 C10

只执行 MUSA 映射到 `cuda_bf16` profile 的 full-model gate：

```bash
export WS1_C10_JSON=/tmp/ws1-c10-musa.json

python3 scripts/ws1_chain_gate.py \
  --backend-profile cuda_bf16 \
  --model qwen3-8b-dense \
  --dtype bfloat16 \
  --weights required \
  --weights-path "$QWEN3_8B" \
  --json > "$WS1_C10_JSON"
```

检查结果：

```bash
python3 - "$WS1_C10_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
accuracy = payload["accuracy_aggregates"][0]["aggregates"]
train_infer = payload["train_infer"]["aggregates"]

print("passed:", payload["passed"])
print("forward max_abs_dlogp:", accuracy["max_abs_dlogp"])
print("train/infer max_abs_dlogp:", train_infer["max_abs_dlogp"])
print("gradient invariance failures:",
      sum(not item["passed"] for item in payload["gradient_invariance"]))
print("gradient accuracy failures:",
      sum(not item["passed"] for item in payload["gradient_accuracy"]))
print("decode/prefill failures:",
      sum(not item["passed"] for item in payload["decode_prefill"]))

assert payload["passed"] is True
PY
```

预期结果：

```text
passed: True
forward max_abs_dlogp: 0.05099058151245117
train/infer max_abs_dlogp: 0.0
gradient invariance failures: 0
gradient accuracy failures: 0
decode/prefill failures: 0
```

`max_abs_dlogp` 是 C10 正式 JSON 字段，表示 selected logp 相对 FP32 reference 的最大绝对差；它不是 train/infer 差异。性能文档中的 `max_abs_diff` 只是展示命名。

## 7. 环境边界

```text
Execution dtype: BF16
Reference dtype: FP32
Accumulation dtype: FP32
Hardware: MTT S5000
MUSA architecture: mp_31
```

C10 是单卡 full-model internal parity/accuracy gate，不等价于多卡生产训练、真实 vLLM/Megatron 联调或 C11 GitHub GPU workflow。
