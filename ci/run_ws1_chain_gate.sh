#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# WS1 C10/C11 full Qwen3-8B Dense model-level gate (CUDA BF16 and Triton-on-CUDA BF16).
# Intended for H20 / H100. Fails closed on skip, xfail, synthetic weights, or silent fallback.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
export RL_KERNEL_REQUIRE_EXT="${RL_KERNEL_REQUIRE_EXT:-1}"
WEIGHTS_PATH="${WS1_WEIGHTS_PATH:-${QWEN3_8B:-}}"

if [ -z "$WEIGHTS_PATH" ]; then
  echo "[ws1-chain] FATAL: set WS1_WEIGHTS_PATH or QWEN3_8B to the pinned Qwen3-8B snapshot"
  exit 2
fi

echo "[ws1-chain] interpreter=$PY weights=$WEIGHTS_PATH"

"$PY" -m pytest -q \
  tests/test_kv_consistency.py \
  tests/test_ws1_qwen3_dense.py \
  tests/test_ws1_chain_integration.py

for PROFILE in cuda_bf16 triton_cuda_bf16; do
  OUT="/tmp/ws1-c10-${PROFILE}.json"
  echo "[ws1-chain] C10/C11 $PROFILE"
  "$PY" scripts/ws1_chain_gate.py \
    --backend-profile "$PROFILE" \
    --model qwen3-8b-dense \
    --dtype bfloat16 \
    --seed 0 \
    --weights required \
    --weights-path "$WEIGHTS_PATH" \
    --json > "$OUT"
  "$PY" - "$OUT" "$PROFILE" <<'PY'
import json
import sys

path, profile = sys.argv[1], sys.argv[2]
payload = json.load(open(path, encoding="utf-8"))
manifest = json.load(
    open("rl_engine/testing/ws1_manifest.json", encoding="utf-8")
)
expected_weight_hash = manifest["model_identity"]["weight_snapshot"]["content_hash"]
if payload.get("schema_version") != "ws1-c10-c11-v2":
    raise SystemExit(f"{profile} artifact has an unsupported schema")
if payload.get("backend_profile") != profile:
    raise SystemExit(f"artifact profile mismatch for {profile}")
if payload.get("weight_hash") != expected_weight_hash:
    raise SystemExit(f"{profile} did not verify the pinned weight snapshot")
if payload.get("workload_seed") != manifest["seed"]:
    raise SystemExit(f"{profile} workload seed does not match the manifest")
if not payload.get("passed"):
    raise SystemExit(f"{profile} C10 gate failed first_drift={payload.get('first_drift')}")
if payload.get("weight_source", "").startswith("synthetic"):
    raise SystemExit(f"{profile} used synthetic weights; C11 forbids that")
if payload.get("git_sha") in {None, "", "unknown"}:
    raise SystemExit(f"{profile} has no commit SHA")
if payload.get("git_dirty"):
    raise SystemExit(f"{profile} was produced from a dirty worktree")
if not payload.get("backward_executed"):
    raise SystemExit(f"{profile} did not execute backward")
if not payload.get("train_infer_executed"):
    raise SystemExit(f"{profile} did not execute train/infer parity")
if payload.get("gradient_scope") != "representative_parameter_subset":
    raise SystemExit(f"{profile} has an unknown gradient scope")
if payload.get("all_parameter_gradients") is not False:
    raise SystemExit(f"{profile} must not claim all-parameter gradient evidence")
if payload.get("required_grad_names") != [
    "layers.0.input_layernorm.weight", "lm_head.weight", "norm.weight"
]:
    raise SystemExit(f"{profile} required gradient set is not the C10 contract")
if payload.get("first_drift") is not None:
    raise SystemExit(f"{profile} passed with a non-null first_drift")
observations = payload.get("runtime_backend_observations", {})
required_nodes = {
    "embedding", "rms_norm", "det_gemm", "qk_norm", "rope",
    "attention", "swiglu", "lm_head", "logprob",
}
if set(observations) != required_nodes:
    raise SystemExit(
        f"{profile} runtime observations mismatch: {sorted(observations)}"
    )
for node, observation in observations.items():
    if observation.get("execution_count", 0) <= 0:
        raise SystemExit(f"{profile} node {node} was not executed")
    if observation.get("expected_kernel_id") != observation.get("observed_kernel_id"):
        raise SystemExit(f"{profile} node {node} used an unexpected candidate")
    if observation.get("fallback_observed"):
        raise SystemExit(f"{profile} node {node} reported fallback")
print(f"[ws1-chain] {profile} passed first_drift={payload.get('first_drift')}")
PY
done

echo "[ws1-chain] both required profiles passed"
