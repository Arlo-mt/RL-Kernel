# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


def require_musa_symbol(symbol: str) -> None:
    if not _EXT_AVAILABLE or _C is None or not hasattr(_C, symbol):
        raise RuntimeError(f"MUSA native symbol {symbol!r} is unavailable")


def check_musa(*tensors: torch.Tensor) -> None:
    if any(t.device.type != "musa" for t in tensors):
        raise RuntimeError("MUSA native kernels require MUSA tensors")
