import pytest
import torch

from rl_engine.kernels.ops.musa.loss.linear_logp import MusaLinearLogpOp
from rl_engine.kernels.registry import KernelRegistry


def _musa_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize("with_bias", [False, True])
def test_musa_linear_logp_matches_forward_and_backward(with_bias):
    torch.manual_seed(2030)
    hidden = torch.randn(4, 16, device="musa", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(23, 16, device="musa", dtype=torch.bfloat16, requires_grad=True)
    bias = (
        torch.randn(23, device="musa", dtype=torch.bfloat16, requires_grad=True)
        if with_bias
        else None
    )
    target = torch.tensor([0, 3, 12, 22], device="musa", dtype=torch.long)
    upstream = torch.tensor([0.25, -1.0, 1.5, 0.75], device="musa", dtype=torch.bfloat16)

    output = MusaLinearLogpOp()(hidden, weight, target, bias)
    logits = torch.nn.functional.linear(
        hidden.float(), weight.float(), None if bias is None else bias.float()
    )
    reference = torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    assert torch.allclose(output.float(), reference, atol=0.05, rtol=0.02)
    output.backward(upstream)

    rh = hidden.detach().float().requires_grad_(True)
    rw = weight.detach().float().requires_grad_(True)
    rb = None if bias is None else bias.detach().float().requires_grad_(True)
    rlogits = torch.nn.functional.linear(rh, rw, rb)
    routput = torch.log_softmax(rlogits, dim=-1).gather(1, target[:, None]).squeeze(1)
    routput.backward(upstream.float())
    assert torch.allclose(hidden.grad.float(), rh.grad, atol=0.2, rtol=0.05)
    assert torch.allclose(weight.grad.float(), rw.grad, atol=0.2, rtol=0.05)
    if bias is not None:
        assert torch.allclose(bias.grad.float(), rb.grad, atol=0.2, rtol=0.05)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_linear_logp_registry_dispatch():
    op = KernelRegistry().get_op("linear_logp", device="musa")
    assert op.__class__.__name__ == "MusaLinearLogpOp"
