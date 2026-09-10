# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.musa.loss.grpo_loss import MusaGRPOLossOp
from rl_engine.kernels.ops.pytorch.loss.grpo_loss import NativeGRPOLossOp
from rl_engine.kernels.registry import KernelRegistry
from rl_engine.testing import make_synthetic_rl_kernel_batch

_VOCAB = 257


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def _batch(seed=0, *, valid_density=0.85):
    return make_synthetic_rl_kernel_batch(
        num_prompts=3,
        samples_per_prompt=4,
        prompt_len=0,
        completion_len=6,
        vocab_size=_VOCAB,
        valid_density=valid_density,
        device="musa",
        seed=seed,
    )


def _logit_pair(batch, seed, dtype):
    gen = torch.Generator(device="musa").manual_seed(seed)
    policy = torch.randn(
        batch.batch_size,
        batch.completion_len,
        _VOCAB,
        generator=gen,
        device="musa",
        dtype=dtype,
    )
    reference = torch.randn_like(policy)
    return policy * 0.25, reference * 0.25


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_grpo_group_advantages_matches_reference():
    from rl_engine import _C

    assert hasattr(_C, "grpo_group_advantages_musa")
    native = NativeGRPOLossOp()
    musa = MusaGRPOLossOp()
    rewards = _batch(seed=11).rewards

    got = musa.group_advantages(rewards, samples_per_prompt=4)
    expected = native.group_advantages(rewards, samples_per_prompt=4)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)

    got_bounds = musa.group_advantages(rewards, group_boundaries=[0, 5, 12])
    exp_bounds = native.group_advantages(rewards, group_boundaries=[0, 5, 12])
    assert torch.allclose(got_bounds, exp_bounds, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        pytest.param(torch.float32, 3e-5, 3e-5, id="fp32"),
        pytest.param(torch.float16, 3e-3, 3e-3, id="fp16"),
        pytest.param(torch.bfloat16, 3e-2, 3e-2, id="bf16"),
    ],
)
def test_musa_grpo_forward_backward_matches_reference(dtype, atol, rtol):
    from rl_engine import _C

    assert hasattr(_C, "ratio_kl_musa_forward")
    assert hasattr(_C, "ratio_kl_musa_backward")
    assert hasattr(_C, "grpo_group_advantages_musa")

    batch = _batch(seed=23)
    policy_logits, ref_logits = _logit_pair(batch, seed=123, dtype=dtype)
    action_ids = batch.token_ids.clone()
    action_ids[batch.completion_mask == 0] = _VOCAB + 99
    old_logps = batch.old_logps.float() * 0.1
    rewards = batch.rewards.float()
    mask = batch.completion_mask.to(torch.int32)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=4)

    native = NativeGRPOLossOp()
    musa = MusaGRPOLossOp()
    pol_native = policy_logits.detach().clone().requires_grad_(True)
    pol_musa = policy_logits.detach().clone().requires_grad_(True)

    expected = native.forward(
        pol_native,
        ref_logits,
        action_ids,
        old_logps,
        rewards,
        mask,
        **kwargs,
    )
    got = musa.forward(
        pol_musa,
        ref_logits,
        action_ids,
        old_logps,
        rewards,
        mask,
        **kwargs,
    )

    for actual, expect in zip(got, expected):
        assert torch.allclose(actual, expect, atol=atol, rtol=rtol)

    upstream = torch.tensor(1.7, device="musa")
    (got[0] * upstream).backward()
    (expected[0] * upstream).backward()

    assert pol_musa.grad is not None
    assert pol_native.grad is not None
    assert torch.allclose(
        pol_musa.grad.float(),
        pol_native.grad.to(dtype).float(),
        atol=atol,
        rtol=rtol,
    )
    assert torch.equal(
        pol_musa.grad[mask == 0],
        torch.zeros_like(pol_musa.grad[mask == 0]),
    )


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_grpo_registry_dispatch():
    op = KernelRegistry().get_op("grpo_loss", device="musa")
    assert op.__class__.__name__ == "MusaGRPOLossOp"
