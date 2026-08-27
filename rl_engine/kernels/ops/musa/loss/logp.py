# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Correctness-first MUSA adapter for selected-token log probabilities."""

from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp


class MusaLogpOp(NativeLogpOp):
    """Expose the portable PyTorch LogP implementation as a MUSA backend."""
