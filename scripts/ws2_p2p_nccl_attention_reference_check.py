# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""NCCL correctness reference for issue #235 CP Attention communication.

Use two ranks for a CP-only diagnostic, four ranks for the formal TP=2/CP=2
target, or eight ranks for two independent TP=2/CP=2 replicas::

    torchrun --standalone --nproc-per-node=4 \
      scripts/ws2_p2p_nccl_attention_reference_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.ops.cuda.attention.cp_comm import (  # noqa: E402
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPMergedState,
    AttentionCPPartialState,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
    P2PNCCLAttentionCPCommunication,
)
from rl_engine.kernels.ops.pytorch.attention.cp_attention import (  # noqa: E402
    AttentionPartialState,
    DeterministicCPAttentionReferenceOp,
    merge_attention_partial_states,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    parser.add_argument("--final-write-atol", type=float, default=2.0e-2)
    parser.add_argument(
        "--transport",
        choices=("p2p_nccl_reference", "cuda_ag_rs"),
        default="p2p_nccl_reference",
        help="P2P is the correctness reference; cuda_ag_rs selects PR311/PR312",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this check requires at least two visible CUDA devices")
    dist.init_process_group("nccl", init_method="env://")
    try:
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if world_size not in {2, 4, 8}:
            raise RuntimeError("this check requires 2, 4, or 8 NCCL ranks")
        if torch.cuda.device_count() < world_size:
            raise RuntimeError("this single-node check requires one visible GPU per NCCL rank")
        local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        cp_groups = [
            dist.new_group(ranks=[pair_start, pair_start + 1])
            for pair_start in range(0, world_size, 2)
        ]
        replica_rank = global_rank if world_size < 8 else global_rank % 4
        sp_rank = 0 if world_size < 8 else global_rank // 4
        tp_rank = 0 if world_size == 2 else replica_rank // 2
        cp_rank = replica_rank % 2
        cp_group = cp_groups[global_rank // 2]
        result = run_check(
            args,
            global_rank=global_rank,
            tp_rank=tp_rank,
            cp_rank=cp_rank,
            sp_rank=sp_rank,
            cp_group=cp_group,
            device=device,
        )
        failures = torch.tensor(
            [0 if result["passed"] else 1],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(failures, op=dist.ReduceOp.SUM)
        result["global_failure_count"] = int(failures.item())
        reports: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(reports, result)
        if global_rank == 0:
            report = {
                "schema_version": (
                    "ws2_p2p_nccl_attention_reference/v1"
                    if args.transport == "p2p_nccl_reference"
                    else "ws2_cuda_ag_rs_attention/v1"
                ),
                "backend": str(dist.get_backend()),
                "transport": args.transport,
                "world_size": world_size,
                "tp_world_size": 1 if world_size == 2 else 2,
                "cp_world_size": 2,
                "sp_world_size": 2 if world_size == 8 else 1,
                "global_failure_count": int(failures.item()),
                "ranks": reports,
            }
            serialized = json.dumps(report, indent=2, sort_keys=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            print(serialized)
        return 0 if int(failures.item()) == 0 else 1
    finally:
        dist.destroy_process_group()


def run_check(
    args: argparse.Namespace,
    *,
    global_rank: int,
    tp_rank: int,
    cp_rank: int,
    sp_rank: int,
    cp_group: Any,
    device: torch.device,
) -> dict[str, object]:
    if args.batch < 1:
        raise ValueError("batch must be positive")
    if args.seq_len < 2 or args.seq_len % 2 != 0:
        raise ValueError("seq_len must be positive and divisible by CP=2")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2 for a bitwise stability check")
    if args.q_heads != 16 or args.kv_heads != 4 or args.head_dim != 128:
        raise ValueError("TP=2 Qwen3-8B local heads must be Hq=16, Hkv=4, D=128")
    for name in ("atol", "final_write_atol"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    generator = torch.Generator(device="cpu").manual_seed(args.seed + tp_rank + 100 * sp_rank)
    shape_q = (args.batch, args.q_heads, args.seq_len, args.head_dim)
    shape_kv = (args.batch, args.kv_heads, args.seq_len, args.head_dim)
    q = torch.randn(shape_q, generator=generator, dtype=torch.bfloat16).to(device)
    k = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    v = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    owner_ranges = ((0, args.seq_len // 2), (args.seq_len // 2, args.seq_len))
    blocks: list[AttentionCPBlockMetadata] = []
    for owner, (owner_start, owner_end) in enumerate(owner_ranges):
        for start in range(owner_start, owner_end, args.chunk_size):
            blocks.append(
                AttentionCPBlockMetadata(
                    global_block_index=len(blocks),
                    kv_block_start=start,
                    kv_block_end=min(start + args.chunk_size, owner_end),
                    owner_cp_rank=owner,
                    owner_tp_rank=tp_rank,
                )
            )
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=2,
            tp_rank=tp_rank,
            cp_world_size=2,
            cp_rank=cp_rank,
        ),
        backend=args.transport,
        status="implemented",
        expected_blocks=tuple(blocks),
        expected_kv_token_range=(0, args.seq_len),
        query_token_ranges=owner_ranges,
    )
    communication: P2PNCCLAttentionCPCommunication | CUDAAGRSAttentionCPCommunication
    if args.transport == "p2p_nccl_reference":
        communication = P2PNCCLAttentionCPCommunication(process_group=cp_group)
    else:
        communication = CUDAAGRSAttentionCPCommunication(process_group=cp_group)

    query_start, query_end = owner_ranges[cp_rank]
    q_local = q[:, :, query_start:query_end, :].contiguous()
    q_gathered = communication.all_gather_query(q_local, plan)
    query_ag_max_abs = float((q_gathered - q).abs().max().item())
    reference = DeterministicCPAttentionReferenceOp()
    local_states: list[AttentionCPPartialState] = []
    for block in reversed(blocks):
        if block.owner_cp_rank != cp_rank:
            continue
        state = reference.local_partial_state(
            q_gathered,
            k[:, :, block.kv_block_start : block.kv_block_end, :],
            v[:, :, block.kv_block_start : block.kv_block_end, :],
            q_start=0,
            k_start=block.kv_block_start,
            total_kv_len=args.seq_len,
            total_query_len=args.seq_len,
            causal=True,
        )
        local_states.append(AttentionCPPartialState(state.out, state.lse, block))

    def communicate() -> tuple[tuple[AttentionCPPartialState, ...], AttentionCPMergedState]:
        gathered_states = communication.all_gather_partial_states(tuple(local_states), plan)
        merged_state = merge_attention_partial_states(
            [
                AttentionPartialState(
                    state.out,
                    state.lse,
                    state.block.kv_block_start,
                    state.block.kv_block_end,
                )
                for state in gathered_states
            ]
        )
        local_state = communication.reduce_scatter_merged_state(
            AttentionCPMergedState(merged_state.out, merged_state.lse),
            plan,
        )
        return gathered_states, local_state

    gathered, local = communicate()
    gathered_indices = [state.block.global_block_index for state in gathered]
    repeat_query_bitwise = True
    repeat_out_bitwise = True
    repeat_lse_bitwise = True
    repeat_manifest_bitwise = True
    for _ in range(args.repeats - 1):
        repeated_q = communication.all_gather_query(q_local, plan)
        repeated_gathered, repeated_local = communicate()
        repeat_query_bitwise = repeat_query_bitwise and torch.equal(repeated_q, q_gathered)
        repeat_out_bitwise = repeat_out_bitwise and torch.equal(repeated_local.out, local.out)
        repeat_lse_bitwise = repeat_lse_bitwise and torch.equal(repeated_local.lse, local.lse)
        repeat_manifest_bitwise = (
            repeat_manifest_bitwise
            and [state.block.global_block_index for state in repeated_gathered] == gathered_indices
        )

    full_out, full_lse = reference.forward_fp32_with_lse(q, k, v, causal=True)
    start, end = owner_ranges[cp_rank]
    out_max_abs = float((local.out - full_out[:, :, start:end, :]).abs().max().item())
    lse_max_abs = float((local.lse - full_lse[:, :, start:end]).abs().max().item())
    final_out = local.out.to(q.dtype)
    expected_final_out = full_out[:, :, start:end, :].to(q.dtype)
    final_out_max_abs = float((final_out.float() - expected_final_out.float()).abs().max().item())
    passed = (
        gathered_indices == list(range(len(blocks)))
        and query_ag_max_abs == 0.0
        and repeat_query_bitwise
        and repeat_out_bitwise
        and repeat_lse_bitwise
        and repeat_manifest_bitwise
        and out_max_abs <= args.atol
        and lse_max_abs <= args.atol
        and final_out.dtype == q.dtype
        and final_out_max_abs <= args.final_write_atol
    )
    return {
        "rank": global_rank,
        "global_world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "tp_rank": tp_rank,
        "tp_world_size": 2,
        "cp_rank": cp_rank,
        "cp_world_size": 2,
        "sp_rank": sp_rank,
        "sp_world_size": 2 if dist.is_initialized() and dist.get_world_size() == 8 else 1,
        "device": str(device),
        "dtype": "bf16",
        "accum_dtype": "fp32",
        "downcast_at": "final_write",
        "final_output_dtype": str(final_out.dtype).removeprefix("torch."),
        "transport": args.transport,
        "protocol": "ag_query_local_kv_rs_out_lse",
        "query_ag": args.transport,
        "query_ag_max_abs": query_ag_max_abs,
        "query_range": [start, end],
        "expected_block_manifest": [block.provenance() for block in blocks],
        "local_block_indices": sorted(state.block.global_block_index for state in local_states),
        "gathered_block_indices": gathered_indices,
        "repeat_count": args.repeats,
        "repeat_query_bitwise": repeat_query_bitwise,
        "repeat_out_bitwise": repeat_out_bitwise,
        "repeat_lse_bitwise": repeat_lse_bitwise,
        "repeat_manifest_bitwise": repeat_manifest_bitwise,
        "out_max_abs": out_max_abs,
        "lse_max_abs": lse_max_abs,
        "final_out_max_abs": final_out_max_abs,
        "atol": args.atol,
        "final_write_atol": args.final_write_atol,
        "passed": passed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
