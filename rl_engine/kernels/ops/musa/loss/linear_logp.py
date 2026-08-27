# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Correctness-first MUSA adapter for linear selected-token log probabilities."""

from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp


class MusaLinearLogpOp(NativeLinearLogpOp):
    """Expose the portable PyTorch linear LogP implementation as a MUSA backend."""
