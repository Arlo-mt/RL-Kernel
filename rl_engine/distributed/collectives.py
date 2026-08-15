# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import socket
import threading
from types import TracebackType
from typing import Any

import torch
import torch.distributed as dist

_TP8_WORLD_SIZE = 8
_DEFAULT_MAX_SIZE_BYTES = 64 * 1024 * 1024
_REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class DeterministicCollective:
    """Correctness-first deterministic CUDA collectives for one TP=8 node.

    The logical reduction tree is fixed to ``xor 1 -> xor 2 -> xor 4``:
    ``((rank0 + rank1) + (rank2 + rank3)) + ((rank4 + rank5) +
    (rank6 + rank7))``. Every node evaluates the lower logical rank before
    the higher logical rank.

    One instance owns a symmetric CUDA IPC staging buffer. All ranks must call
    its methods in the same order with matching shapes and dtypes. Calls are
    host-synchronizing by design; the first version prioritizes determinism and
    lifetime safety over overlap or throughput.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before collectives")
        if not torch.cuda.is_available():
            raise RuntimeError("deterministic collectives require CUDA")
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")

        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        if self.world_size != _TP8_WORLD_SIZE:
            raise ValueError(
                f"deterministic collectives require world_size=8, got {self.world_size}"
            )

        if device is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            normalized_device = torch.device("cuda", device)
        else:
            normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError(f"deterministic collectives require a CUDA device, got {device!r}")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        if normalized_device.index != torch.cuda.current_device():
            raise ValueError(
                "the collective device must be the current CUDA device; call "
                f"torch.cuda.set_device({normalized_device.index}) first"
            )

        try:
            from rl_engine import _C
        except ImportError as exc:
            raise RuntimeError(
                "the RL-Kernel CUDA extension is required; rebuild with "
                "`pip install --no-build-isolation -e .`"
            ) from exc
        required_symbols = (
            "deterministic_collective_ipc_meta",
            "deterministic_collective_create",
            "deterministic_collective_destroy",
            "deterministic_collective_stage",
            "deterministic_collective_all_reduce",
        )
        missing = [name for name in required_symbols if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                "the RL-Kernel CUDA extension lacks deterministic collectives: "
                + ", ".join(missing)
            )

        self.device = normalized_device
        self.max_size_bytes = int(max_size_bytes)
        self._extension = _C
        self._lock = threading.Lock()
        self._handle = 0
        self._staging = torch.empty(
            self.max_size_bytes,
            dtype=torch.uint8,
            device=self.device,
        )

        handle, offset = self._extension.deterministic_collective_ipc_meta(self._staging)
        local_meta = {
            "handle": handle,
            "offset": int(offset),
            "capacity": self.max_size_bytes,
            "hostname": socket.gethostname(),
        }
        gathered_meta: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_meta, local_meta, group=self.group)
        if any(meta is None for meta in gathered_meta):
            raise RuntimeError("failed to exchange CUDA IPC metadata")
        complete_meta = [meta for meta in gathered_meta if meta is not None]
        hostnames = {meta["hostname"] for meta in complete_meta}
        if len(hostnames) != 1:
            raise ValueError("deterministic collectives require all ranks on one host")
        capacities = {meta["capacity"] for meta in complete_meta}
        if capacities != {self.max_size_bytes}:
            raise ValueError("all ranks must use the same max_size_bytes")

        handles = [meta["handle"] for meta in complete_meta]
        offsets = [meta["offset"] for meta in complete_meta]
        self._handle = self._extension.deterministic_collective_create(
            self._staging,
            handles,
            offsets,
            self.rank,
        )
        self._synchronize_ranks()

    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the fixed-tree sum on every rank.

        Supported dtypes are float32, float16, and bfloat16. ``out`` may alias
        ``input``; the input is staged before the output kernel starts.
        """

        self._check_open()
        self._validate_reduction_input(input)
        if out is None:
            out = torch.empty_like(input)
        self._validate_output(out, input)

        with self._lock:
            self._validate_matching_signature("all_reduce", input)
            self._extension.deterministic_collective_stage(self._handle, input)
            self._synchronize_ranks()
            self._extension.deterministic_collective_all_reduce(self._handle, out)
            self._synchronize_ranks()
        return out

    def close(self) -> None:
        """Release imported CUDA IPC mappings after the last collective call."""

        handle = getattr(self, "_handle", 0)
        if not handle:
            return
        torch.cuda.synchronize(self.device)
        self._handle = 0
        self._extension.deterministic_collective_destroy(handle)

    def __enter__(self) -> DeterministicCollective:
        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if not getattr(self, "_handle", 0):
            raise RuntimeError("deterministic collective is closed")

    def _validate_reduction_input(self, input: torch.Tensor) -> None:
        if not input.is_cuda or input.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {input.device}")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        if input.dtype not in _REDUCTION_DTYPES:
            raise TypeError(
                "deterministic reductions support float32, float16, and bfloat16; "
                f"got {input.dtype}"
            )
        input_bytes = input.numel() * input.element_size()
        if input_bytes > self.max_size_bytes:
            raise ValueError(
                f"input requires {input_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _validate_output(self, output: torch.Tensor, input: torch.Tensor) -> None:
        if output.device != input.device:
            raise ValueError("out must be on the same device as input")
        if output.dtype != input.dtype:
            raise TypeError("out must have the same dtype as input")
        if output.shape != input.shape:
            raise ValueError("out must have the same shape as input")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")

    def _validate_matching_signature(self, op_name: str, input: torch.Tensor) -> None:
        signature = (op_name, tuple(input.shape), str(input.dtype), input.numel())
        signatures: list[tuple[Any, ...] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(peer_signature != signature for peer_signature in signatures):
            raise ValueError(
                f"all ranks must call {op_name} with matching shapes and dtypes; got {signatures}"
            )

    def _synchronize_ranks(self) -> None:
        torch.cuda.synchronize(self.device)
        backend = dist.get_backend(self.group)
        if backend == dist.Backend.NCCL or str(backend).lower() == "nccl":
            dist.barrier(group=self.group, device_ids=[self.device.index])
        else:
            dist.barrier(group=self.group)
