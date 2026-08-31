# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch


try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None


requires_musa_extension = pytest.mark.skipif(
    torch_musa is None or not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA runtime",
)


@requires_musa_extension
def test_musa_fused_logp_extension_matches_reference():
    from rl_engine import _C

    if not hasattr(_C, "fused_logp"):
        pytest.fail("MUSA extension is present but does not export fused_logp")

    torch.manual_seed(7)
    logits = torch.randn(5, 37, device="musa", dtype=torch.bfloat16)
    token_ids = torch.tensor([0, 3, 11, 29, 36], device="musa", dtype=torch.long)

    actual = _C.fused_logp(logits, token_ids)
    reference = torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids[:, None]).squeeze(1)
    torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=2e-2)
