# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol


class DetGemmOp:
    def __init__(self):
        require_musa_symbol("det_gemm_rowwise_fwd_fp32")

    def __call__(self, a, b):
        check_musa(a, b)
        return _C.det_gemm_rowwise_fwd_fp32(a, b)

    forward = __call__
    forward_fp32 = __call__
