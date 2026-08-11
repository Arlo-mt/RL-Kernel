# WS1 gtest 阈值迁移清单

> **关联：** [#266](https://github.com/RL-Align/RL-Kernel/issues/266) 父收尾 · [#267](https://github.com/RL-Align/RL-Kernel/issues/267) C1 契约 · [数值契约说明](ws1-numerical-contract.md)
> **目的：** 盘点「哪些测试仍用私有 `atol`/`rtol`、哪些已走 SSOT、何时必须迁到 `resolve_tolerance`」。
> **快照：** 基于 `feat/ws1-c1-tolerance-contract-267` 落地 C1 后的仓库状态；文件增减时请更新本表。

---

## 0. 迁移总原则

### 0.1 SSOT 入口（改后唯一推荐）

```python
from rl_engine.kernels.gtest.tolerance import (
    load_contract,
    resolve_tolerance,
    compute_logprob_aggregates,
    judge_logprob_aggregates,
    default_clip_interval,
)

contract = load_contract()
spec = resolve_tolerance(
    contract,
    judgment="forward_accuracy",   # 或 forward_invariance / gradient_*
    op_class="logprob",            # elementwise | reduction | logprob | attention
    dtype="bfloat16",
    backend_profile="cuda_bf16",   # 与 triton_cuda_bf16 同阈值
)
# assert_close(..., atol=spec.atol, rtol=spec.rtol)
# 不变性：spec.mode == "bitwise" 且 atol=rtol=0 → 优先 torch.equal
```

| Judgment | 用于 |
|----------|------|
| `forward_accuracy` | BF16 candidate vs FP32 reference |
| `forward_invariance` | 同逻辑 workload 跨 batch/chunk/layout（**bitwise**） |
| `gradient_accuracy` | 梯度 vs FP32 参考（**不得**读 forward 行） |
| `gradient_invariance` | 梯度跨 config（**bitwise**） |
| 三聚合 API | 链级 / 训推 selected-logprob（`max_abs_dlogp` / `approx_kl0` / `clipfrac0`） |

### 0.2 什么叫「私有阈值」（禁止作为 WS1 gate 证据）

- 测试文件内字面量：`atol=1e-5`、`atol=5e-2`、模块常量 `_DECODE_ATOL` 等
- 文档里写死但未从 `tolerance_contract.json` resolve 的数
- 从 `contract["accuracy"]...` 手抄数值后本地再改（应用 resolve，不要复制常量）
- 用非零 `atol` 充当 Batch/Chunk **invariance** 通过条件

### 0.3 什么可以保留（不必硬迁）

| 场景 | 处理 |
|------|------|
| **bitwise 身份断言**（`torch.equal`） | 合法；对应 invariance judgment 的 `mode=bitwise` |
| **非数值语义**（mask 形状、版本单调、manifest 字段） | 不迁 |
| **框架/集成单测**（bridge、vLLM mock、DeepSpeed worker 编排） | 非 WS1 op gate；可保留宽松 `allclose`，但**不能**当作 #266 EXIT 证据 |
| **生产 FA / SDPA 对齐**（`test_attention_correctness`） | 非 BI 候选路径；阈值可独立，**不得**写进 WS1 EXIT claim |
| **legacy `accuracy` 键** | 仅兼容；新代码禁止新增依赖，应改 `resolve_tolerance` |

### 0.4 建议迁移句式

```python
# BAD — 私有阈值
torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

# GOOD — accuracy
spec = resolve_tolerance(contract, judgment="forward_accuracy", op_class="reduction", dtype=dtype)
torch.testing.assert_close(out, ref, atol=spec.atol, rtol=spec.rtol)

# GOOD — invariance
ispec = resolve_tolerance(contract, judgment="forward_invariance", op_class="attention", dtype=dtype)
assert ispec.mode == "bitwise" and ispec.atol == 0.0
assert torch.equal(a, b)  # 或 assert_close(..., atol=0, rtol=0)

# GOOD — gradient accuracy（独立 judgment）
gspec = resolve_tolerance(contract, judgment="gradient_accuracy", op_class="logprob", dtype=dtype)
```

---

## 1. 状态总表（`tests/`）

图例：

| 标记 | 含义 |
|------|------|
| **A** | 已走 resolver / gtest suite（目标态） |
| **B** | 走 `load_contract` 旧键或 gtest 间接路径（过渡） |
| **C** | WS1 相关但 **私有 atol**（应迁） |
| **D** | 多为 `torch.equal` / 结构断言（ok 或仅需声明 judgment） |
| **E** | 非 WS1 门禁（框架/产品路径，低优先级） |

### 1.1 已对齐或接近 SSOT

| 文件 | 状态 | 说明 | 下一步 |
|------|------|------|--------|
| `test_tolerance_contract.py` | **A** | C1 schema + resolve + 聚合 | 保持；契约变更必跑 |
| `test_op_checks.py` | **A/B** | suite 已按 judgment 解析；部分用例注入最小 contract | 新 fixture 尽量带 `judgments` |
| `test_operator_inputs.py` | **B** | 输入/规格，无数值阈值主责 | 无需迁阈值 |
| `test_swiglu.py` | **B/D** | issue-108 harness + 大量 `torch.equal` | accuracy 路径确认走 suite；字面量 atol 清零 |
| `test_det_gemm.py` | **B** | `load_contract()["accuracy"]...` | **优先迁**：改为 `resolve_tolerance(..., forward/gradient_accuracy)` |
| `test_deterministic_attention_cuda.py` | **B/C** | 部分用 suite；仍见 `5e-2/2e-2` 字面量 | 字面量改为 resolve；invariance 保持 equal |

### 1.2 WS1 算子测试 — 私有阈值（应迁，按优先级）

| 优先级 | 文件 | op_class 建议 | 现状摘要 | 何时必须迁 |
|--------|------|---------------|----------|------------|
| **P0** | `test_batch_invariant_logp.py` | `logprob` | 大量 `1e-6`…`1e-2` 私有；含 bwd | 接 C3/C4/C8 证据前 |
| **P0** | `test_linear_logp.py` | `logprob` | `1e-5`…`1.5e-1` 混用；bf16 松阈值 | 同上；链级改用三聚合 API |
| **P0** | `test_logp.py` / `test_deterministic_logp.py` | `logprob` | 私有 atol | 关 #148 residual / C8 前 |
| **P0** | `test_rms_norm.py` | `reduction` | `1e-5`…`8e-2`；bwd 混用 | C8 RMSNorm 证据前 |
| **P0** | `test_triton_batch_invariant_attention.py` | `attention` | 混 `1e-5` 与 `5e-2/2e-2` | C8 Attention 证据前 |
| **P0** | `test_attention.py` | `attention` | native GT；`1e-4`/`2e-6` 等 | 与 contract `attention` 行对齐 |
| **P1** | `test_kv_cache_attention.py` | `attention` | 含 `2e-6` 等；#152 相关 | **C6/C7 前必须**消私有 decode 阈值 |
| **P1** | `test_issue151_embedding_lm_head_invariance.py` | emb + lm_head + logp | bf16 `5e-2` 手写 | C8 emb/lm_head 证据前 |
| **P1** | `test_lm_head.py` | `reduction` | 多 equal；grad `1e-5` 私有 | 迁 grad → `gradient_accuracy` |
| **P1** | `test_embedding.py` | `elementwise` | 多为 equal | 若有 tolerance 路径再 resolve |
| **P1** | `test_rope.py` | `elementwise` | `1e-3`…`2e-2` | C5 RoPE 证据前 |
| **P1** | `test_matmul.py` | `reduction` | 私有 `1e-4/1e-5` | 与 det_gemm 统一 |
| **P2** | `test_pack.py` | `elementwise` | 几乎 equal；gradcheck `1e-6` | packing 纳入 #150 时 |
| **P2** | `test_grpo_loss.py` / `test_ratio_kl.py` | （loss，契约暂无独立 class） | `1e-4` 等 | 若进 chain 则扩展 op_class 或显式 N/A |
| **P3** | `test_attention_correctness.py` | 非 BI EXIT | FA/SDPA 私有表 | **不迁入 WS1 SSOT**；文档标明 out of WS1 claim |
| **P3** | `test_op_accuracy.py` | 杂项 harness | `1e-3` | 废弃或改走 `check_operator` + contract |

### 1.3 非 WS1 门禁（低优先级 / 不阻塞 #267）

| 文件 | 状态 | 说明 |
|------|------|------|
| `test_deepspeed_training_worker.py` | **E** | 训练 worker；`atol=1e-5` 编排级 |
| `test_stateless_training_contract.py` | **E** | 契约字段/数值 smoke |
| `test_rl_kernel_loss_step.py` | **E** | 端到端 loss 步 |
| `test_sampler_temperature.py` | **E** | 采样 |
| `test_weight_sync_bridge.py` 等 | **D/E** | bridge / IPC |
| `test_vllm_rollout_sampler.py` | **D/E** | vLLM mock |
| `test_alignment_model_wrappers.py` | **D/E** | wrapper 行为 |
| `test_rl_batch_fixture.py` | **D** | fixture 身份 |
| `test_stateless_executor.py` / `*_hf_integration*` | **D/E** | 执行器集成 |

这些**不**作为 #266 Full WS1 EXIT 的数值证据来源；C10/C11 不得引用其私有阈值刷绿。

---

## 2. 按 #266 子 issue 的「何时必须迁」

| 子 issue | 阻塞迁移范围 | 完成信号 |
|----------|--------------|----------|
| **C1 #267** | 契约 + resolver + schema/报告测试 | **实现完成；待 CI / issue evidence** |
| **C3 #269** forward harness | 所有 **forward_accuracy / forward_invariance** 的 op 单测证据路径 | 无 private forward atol 作为 gate |
| **C4 #270** grad harness | 所有 **gradient_*** 证据路径 | 无「grad 抄 forward 字面量」 |
| **C5 #271** RoPE/elementwise | `test_rope.py`、activation/swiglu residual | audit 报告阈值均来自 resolve |
| **C6/C7 #272/#273** KV | `test_kv_cache_attention.py` 及后续 kv harness | **禁止** `_DECODE_ATOL` 类私有常量 |
| **C8 #274** closed-op 矩阵 | rmsnorm / gemm / attn / logp / emb / lm_head 测试 | 每格 `requested/actual backend` + resolve 阈值 |
| **C10 #276** 全模型 gate | 仅用 resolver + 三聚合；禁止任何测试内字面量阈值 | gate 报告无 private tol 字段 |
| **C11 #277** CI | CI 只跑 resolve 路径 | fail-closed |

**规则：** 某 op 的 PR 若声称「满足 #266/C8」，则该 PR 触达的 assert **必须**来自 `resolve_tolerance` / 聚合 API，而不是文件顶部的魔法数。

---

## 3. 文件级迁移清单（可勾选）

### 3.1 P0 — 直接挡 C3/C4/C8

- [ ] `tests/test_batch_invariant_logp.py` — fwd/bwd accuracy + invariance 拆 judgment
- [ ] `tests/test_linear_logp.py` — 同上；训推/链级改用三聚合
- [ ] `tests/test_logp.py`
- [ ] `tests/test_deterministic_logp.py`
- [ ] `tests/test_rms_norm.py`
- [ ] `tests/test_triton_batch_invariant_attention.py`
- [ ] `tests/test_attention.py`
- [ ] `tests/test_det_gemm.py` — 去掉 `contract["accuracy"]` 直读

### 3.2 P1 — C5/C6/C7/C8 residual

- [ ] `tests/test_kv_cache_attention.py`
- [ ] `tests/test_issue151_embedding_lm_head_invariance.py`
- [ ] `tests/test_lm_head.py`
- [ ] `tests/test_embedding.py`（若有 non-bitwise 路径）
- [ ] `tests/test_rope.py`
- [ ] `tests/test_matmul.py`
- [ ] `tests/test_deterministic_attention_cuda.py` 中剩余字面量
- [ ] `tests/test_swiglu.py` 中任何 residual 字面量

### 3.3 P2 — 进 chain 时

- [ ] `tests/test_pack.py`
- [ ] `tests/test_grpo_loss.py` / `tests/test_ratio_kl.py`（先扩 contract op_class 或标 N/A）
- [ ] `scripts/check_operator.py` 报告字段确认只回传 resolve 结果（已间接）

### 3.4 明确不迁入 WS1 SSOT

- [x] `tests/test_attention_correctness.py` — 生产 FA；文档标注非 EXIT
- [x] bridge / vLLM / DeepSpeed / sampler 类 **E** 组

---

## 4. 推荐落地动作（每个测试文件）

1. **分类每条 assert**
   - identity / batch-invariance → `forward_invariance` 或 `gradient_invariance` + `torch.equal`
   - vs fp32 gold → `*_accuracy`
   - train vs infer logp → 三聚合，不用单点 atol 冒充
2. **删除模块级 `_ATOL` / `_RTOL`**
3. **dtype 参数化** 时用 `resolve_tolerance(..., dtype=dtype)`，禁止 bf16 写死 `5e-2`
4. **报告**（若有）写上 `comparison_lhs_role` / `comparison_rhs_role`（从 spec 取）
5. **禁止** 为让 invariance 通过而调大 atol

### 4.1 与契约行不一致时怎么办

| 情况 | 动作 |
|------|------|
| 测试私有更松，契约更紧 → 测试红 | **修 kernel** 或开 Blocker；**禁止**在测试放宽 |
| 测试私有更紧，契约更松 | 迁到契约后可能变绿；可保留额外严格 assert 但须标注 *non-gate* |
| 需要新 op_class（如 `grpo_loss`） | 先改 `tolerance_contract.json` + schema 测试，再迁测试 |
| decode vs prefill 无法 bitwise | 用 contract 已声明的 semantic 行 / 三聚合；**不要**私设 `_DECODE_ATOL` |

---

## 5. 工具与 CI 建议（后续，非 C1 范围）

| 建议 | 作用 |
|------|------|
| 简单 lint：`tests/**/*.py` 禁止 `atol=\d`（allowlist 契约测试与 FA 测试） | 防回流 |
| `pytest` marker：`ws1_gate` 仅收集 resolve 路径 | C11 门禁清晰 |
| 在 `check_operator.py` 输出中强制打印 `judgment` + roles | 证据可检索 |

C1 **不**强制上 lint；C8/C10 前建议至少做 allowlist 扫描。

---

## 6. 现状一句话

| 层 | 状态 |
|----|------|
| **契约 + resolver（gtest 核心）** | 已就绪；待 CI / issue evidence（#267） |
| **op_checks 接入** | 已按 judgment 分叉，并持久化 roles / provenance |
| **存量 op 单测** | **多数仍私有 atol**（上表 P0/P1） |
| **#266 EXIT** | 依赖后续把 P0/P1 迁完，而不是只合 C1 |

**C1 的价值是「唯一入口已存在」；清单的价值是「知道还欠哪些文件」。**
未完成 P0/P1 迁移前，**不得**声称「全仓测试已统一走 WS1 数值契约」。

---

## 7. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-08-11 | 初版：C1 落地后基于 `tests/` 扫描的迁移清单与优先级 |
