# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.distributed import DeterministicCollective

_WORLD_SIZE = 8
_EXTERNAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

pytestmark = [
    pytest.mark.skipif(
        _EXTERNAL_WORLD_SIZE != 1,
        reason="this TP=8 test owns its worker processes; run pytest directly",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < _WORLD_SIZE,
        reason="requires eight visible CUDA GPUs",
    ),
]


def _fixed_tree_reference(values: list[torch.Tensor]) -> torch.Tensor:
    sum01 = values[0] + values[1]
    sum23 = values[2] + values[3]
    sum45 = values[4] + values[5]
    sum67 = values[6] + values[7]
    sum03 = sum01 + sum23
    sum47 = sum45 + sum67
    return sum03 + sum47


def _all_gather_tensors(value: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(value) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, value)
    return gathered


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(minutes=3),
    )
    try:
        with DeterministicCollective(device=device, max_size_bytes=1024 * 1024) as collective:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                generator = torch.Generator(device=device).manual_seed(20260815 + rank)
                input = torch.randn(257, device=device, dtype=dtype, generator=generator)
                peer_inputs = _all_gather_tensors(input)
                expected = _fixed_tree_reference(peer_inputs)

                output = collective.all_reduce(input)
                assert torch.equal(output, expected)

                peer_outputs = _all_gather_tensors(output)
                assert all(torch.equal(peer_output, output) for peer_output in peer_outputs)

                baseline = output.clone()
                for _ in range(3):
                    repeated = collective.all_reduce(input)
                    assert torch.equal(repeated, baseline)

                inplace = input.clone()
                returned = collective.all_reduce(inplace, out=inplace)
                assert returned is inplace
                assert torch.equal(inplace, expected)
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_deterministic_all_reduce_tp8_cuda() -> None:
    mp.spawn(
        _worker,
        args=(_find_free_port(),),
        nprocs=_WORLD_SIZE,
        join=True,
    )
