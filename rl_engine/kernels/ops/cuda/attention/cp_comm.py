# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CP/TP attention communication interface for WS2 PR7.

PR7 evaluates fused attention backends for the #235 target
``Qwen3-8B, TP=2, CP=2, BF16``.  The self-owned CUDA communication operators are
AG/RS and compute-communication decoupled.  They are not implemented in this
scaffold, but their interface is exposed here so backend adapters cannot
silently ignore the distributed contract.

The future communication path is expected to move attention partial states:

```text
local FlashInfer/TE attention over rank-owned KV blocks
  -> AttentionCPPartialState(out, lse, global_block_index, tp/cp rank metadata)
  -> custom CUDA AG communication operator
  -> sort by global_block_index
  -> PR3 FP32 online-softmax merge
  -> custom CUDA RS communication operator
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

CPCommunicationBackend = Literal["cuda_ag_rs", "local_debug"]
CPCommunicationStatus = Literal["interface_only", "implemented"]


class AttentionCPCommunicationUnavailable(RuntimeError):
    """Raised when a requested CP communication backend is not implemented."""


@dataclass(frozen=True)
class AttentionParallelSpec:
    """TP/CP identity carried by PR7 attention backend reports."""

    tp_world_size: int = 2
    tp_rank: int = 0
    cp_world_size: int = 2
    cp_rank: int = 0

    def validate(self) -> None:
        _positive_int(self.tp_world_size, "tp_world_size")
        _positive_int(self.cp_world_size, "cp_world_size")
        _rank_in_world(self.tp_rank, self.tp_world_size, "tp_rank")
        _rank_in_world(self.cp_rank, self.cp_world_size, "cp_rank")

    def provenance(self) -> dict[str, int]:
        self.validate()
        return {
            "tp_world_size": int(self.tp_world_size),
            "tp_rank": int(self.tp_rank),
            "cp_world_size": int(self.cp_world_size),
            "cp_rank": int(self.cp_rank),
        }


@dataclass(frozen=True)
class AttentionCPBlockMetadata:
    """Logical identity for one attention partial state."""

    global_block_index: int
    kv_block_start: int
    kv_block_end: int
    owner_cp_rank: int
    owner_tp_rank: int

    def validate(self, parallel: AttentionParallelSpec) -> None:
        parallel.validate()
        if self.global_block_index < 0:
            raise ValueError("global_block_index must be non-negative")
        if self.kv_block_start < 0 or self.kv_block_end <= self.kv_block_start:
            raise ValueError("KV block bounds must satisfy 0 <= start < end")
        _rank_in_world(self.owner_cp_rank, parallel.cp_world_size, "owner_cp_rank")
        _rank_in_world(self.owner_tp_rank, parallel.tp_world_size, "owner_tp_rank")

    def provenance(self) -> dict[str, int]:
        return {
            "global_block_index": int(self.global_block_index),
            "kv_block_start": int(self.kv_block_start),
            "kv_block_end": int(self.kv_block_end),
            "owner_cp_rank": int(self.owner_cp_rank),
            "owner_tp_rank": int(self.owner_tp_rank),
        }


@dataclass(frozen=True)
class AttentionCPPartialState:
    """One local or received ``(out, lse)`` state before CP merge."""

    out: torch.Tensor
    lse: torch.Tensor
    block: AttentionCPBlockMetadata

    def validate(self, parallel: AttentionParallelSpec) -> None:
        self.block.validate(parallel)
        if self.out.ndim != 4:
            raise ValueError("partial out must have shape [B, Hq, Sq, D]")
        if self.lse.ndim != 3:
            raise ValueError("partial lse must have shape [B, Hq, Sq]")
        if self.out.shape[:3] != self.lse.shape:
            raise ValueError("partial out and lse must share [B, Hq, Sq]")
        if self.lse.dtype != torch.float32:
            raise ValueError("partial lse must be attention-domain FP32")


@dataclass(frozen=True)
class AttentionCPMergedState:
    """Merged attention state before the CUDA RS communication operator."""

    out: torch.Tensor
    lse: torch.Tensor

    def validate(self) -> None:
        if self.out.ndim != 4:
            raise ValueError("merged out must have shape [B, Hq, Sq, D]")
        if self.lse.ndim != 3:
            raise ValueError("merged lse must have shape [B, Hq, Sq]")
        if self.out.shape[:3] != self.lse.shape:
            raise ValueError("merged out and lse must share [B, Hq, Sq]")
        if self.lse.dtype != torch.float32:
            raise ValueError("merged lse must be attention-domain FP32")


@dataclass(frozen=True)
class AttentionCPCommunicationPlan:
    """Requested AG/RS communication contract for CP attention partial states."""

    parallel: AttentionParallelSpec
    backend: CPCommunicationBackend = "cuda_ag_rs"
    status: CPCommunicationStatus = "interface_only"
    pattern: str = "ag_rs"
    compute_communication: str = "decoupled"
    merge_order: str = "global_block_index"
    accum_dtype: torch.dtype = torch.float32
    return_lse: bool = True

    def validate(self) -> None:
        self.parallel.validate()
        if self.backend not in {"cuda_ag_rs", "local_debug"}:
            raise ValueError(f"unsupported CP communication backend: {self.backend}")
        if self.status not in {"interface_only", "implemented"}:
            raise ValueError(f"unsupported CP communication status: {self.status}")
        if self.pattern != "ag_rs":
            raise ValueError("PR7 CP communication must use the custom CUDA AG/RS interface")
        if self.compute_communication != "decoupled":
            raise ValueError("PR7 CP communication must keep compute and communication decoupled")
        if self.merge_order != "global_block_index":
            raise ValueError("PR7 CP communication must preserve global_block_index merge order")
        if self.accum_dtype is not torch.float32:
            raise ValueError("PR7 CP merge accumulation must be FP32")
        if not self.return_lse:
            raise ValueError("PR7 CP communication requires LSE-carrying partial states")

    def provenance(self) -> dict[str, object]:
        self.validate()
        return {
            "cp_comm_backend": self.backend,
            "cp_comm_status": self.status,
            "cp_comm_pattern": self.pattern,
            "cp_comm_compute_communication": self.compute_communication,
            "cp_comm_merge_order": self.merge_order,
            "cp_comm_accum_dtype": "fp32",
            "cp_comm_return_lse": self.return_lse,
            "cp_comm_contract": "partial_out_lse_global_block_index",
            **self.parallel.provenance(),
        }


class AttentionCPCommunication(Protocol):
    """Protocol future custom CUDA AG/RS communication operators must implement."""

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        """Run the custom CUDA AG operator and return gathered partial states."""

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        """Run the custom CUDA RS operator and return this rank's output shard."""


class CUDAAGRSAttentionCPCommunication:
    """Fail-closed placeholder for future custom CUDA AG/RS communication operators."""

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        plan.validate()
        for state in local_states:
            state.validate(plan.parallel)
        raise AttentionCPCommunicationUnavailable(
            "custom CUDA AG attention communication is interface-only in this PR7 scaffold; "
            "future implementation must gather AttentionCPPartialState tensors before "
            "global_block_index sorting and PR3 FP32 merge"
        )

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        plan.validate()
        merged_state.validate()
        raise AttentionCPCommunicationUnavailable(
            "custom CUDA RS attention communication is interface-only in this PR7 scaffold; "
            "future implementation must scatter the PR3-merged attention state to CP ranks"
        )


def sort_attention_cp_partial_states(
    states: tuple[AttentionCPPartialState, ...],
    *,
    plan: AttentionCPCommunicationPlan,
) -> tuple[AttentionCPPartialState, ...]:
    """Validate and sort partial states by ``global_block_index``."""

    plan.validate()
    if not states:
        raise ValueError("at least one CP attention partial state is required")
    for state in states:
        state.validate(plan.parallel)
    ordered = tuple(sorted(states, key=lambda state: state.block.global_block_index))
    indices = [state.block.global_block_index for state in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate global_block_index values are not allowed")
    return ordered


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _rank_in_world(rank: int, world_size: int, name: str) -> None:
    if isinstance(rank, bool) or rank < 0 or rank >= world_size:
        raise ValueError(f"{name} must be in [0, world_size)")


__all__ = [
    "AttentionCPBlockMetadata",
    "AttentionCPCommunication",
    "AttentionCPCommunicationPlan",
    "AttentionCPCommunicationUnavailable",
    "AttentionCPMergedState",
    "AttentionCPPartialState",
    "AttentionParallelSpec",
    "CPCommunicationBackend",
    "CPCommunicationStatus",
    "CUDAAGRSAttentionCPCommunication",
    "sort_attention_cp_partial_states",
]
