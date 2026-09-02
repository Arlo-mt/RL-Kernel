# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.musa._common import check_musa, require_musa_symbol


class _MusaEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, weight):
        ids = token_ids.reshape(-1).to(dtype=torch.long).contiguous()
        output = _C.embedding_fp32(ids, weight.contiguous()).to(weight.dtype)
        ctx.save_for_backward(ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.output_shape = tuple(token_ids.shape) + (weight.size(1),)
        return output.reshape(ctx.output_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (ids,) = ctx.saved_tensors
        grad_weight = torch.zeros(
            ctx.weight_shape, device=grad_output.device, dtype=ctx.weight_dtype
        )
        grad_weight.index_add_(0, ids, grad_output.reshape(-1, ctx.weight_shape[1]))
        return None, grad_weight


class EmbeddingOp:
    def __init__(self):
        require_musa_symbol("embedding_fp32")

    def __call__(self, token_ids, weight):
        return self.forward(token_ids, weight)

    def forward(self, token_ids, weight):
        check_musa(token_ids, weight)
        return _MusaEmbeddingFunction.apply(token_ids, weight)

    def forward_fp32(self, token_ids, weight):
        check_musa(token_ids, weight)
        ids = token_ids.reshape(-1).to(device=weight.device, dtype=torch.long)
        return _C.embedding_fp32(ids, weight).reshape(*token_ids.shape, weight.size(1))
