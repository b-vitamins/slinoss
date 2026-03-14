from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from slinoss.ops import (
    cconv1d,
    cconv1d_cuda,
    cconv1d_is_available,
    cconv1d_reference,
)


def _reference_depthwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    seqlen = int(x.shape[-1])
    return F.conv1d(
        x,
        weight.unsqueeze(1),
        bias,
        padding=int(weight.shape[-1]) - 1,
        groups=int(x.shape[1]),
    )[..., :seqlen]


def test_cconv1d_reference_matches_torch_conv1d() -> None:
    torch.manual_seed(0)
    x = torch.randn((2, 24, 17), dtype=torch.float32)
    weight = torch.randn((24, 4), dtype=torch.float32)
    bias = torch.randn((24,), dtype=torch.float32)

    got = cconv1d_reference(x, weight, bias)
    assert isinstance(got, torch.Tensor)
    expect = _reference_depthwise(x, weight, bias)
    assert torch.allclose(got, expect, atol=1e-6, rtol=1e-6)


def test_cconv1d_dispatch_uses_reference_on_cpu() -> None:
    torch.manual_seed(1)
    x = torch.randn((2, 24, 9), dtype=torch.float32)
    weight = torch.randn((24, 3), dtype=torch.float32)
    bias = torch.randn((24,), dtype=torch.float32)

    got = cconv1d(x, weight, bias, prefer_cuda=True)
    assert isinstance(got, torch.Tensor)
    expect = cconv1d_reference(x, weight, bias)
    assert isinstance(expect, torch.Tensor)
    assert torch.equal(got, expect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not cconv1d_is_available(), reason="cconv1d extension not built")
def test_cconv1d_cuda_matches_reference_forward_backward() -> None:
    torch.manual_seed(2)
    x_ref = torch.randn(
        (2, 64, 65), device="cuda", dtype=torch.float16, requires_grad=True
    )
    weight_ref = torch.randn(
        (64, 4), device="cuda", dtype=torch.float16, requires_grad=True
    )
    bias_ref = torch.randn(
        (64,), device="cuda", dtype=torch.float16, requires_grad=True
    )
    x_cuda = x_ref.detach().clone().requires_grad_(True)
    weight_cuda = weight_ref.detach().clone().requires_grad_(True)
    bias_cuda = bias_ref.detach().clone().requires_grad_(True)

    y_ref = cconv1d_reference(x_ref, weight_ref, bias_ref, activation=None)
    assert isinstance(y_ref, torch.Tensor)
    y_cuda = cconv1d_cuda(x_cuda, weight_cuda, bias_cuda, activation=None)
    assert isinstance(y_cuda, torch.Tensor)
    assert torch.allclose(y_cuda, y_ref, atol=3e-3, rtol=3e-3)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_cuda.backward(grad)

    assert x_ref.grad is not None and x_cuda.grad is not None
    assert weight_ref.grad is not None and weight_cuda.grad is not None
    assert bias_ref.grad is not None and bias_cuda.grad is not None
    assert torch.allclose(x_cuda.grad, x_ref.grad, atol=3e-3, rtol=3e-3)
    assert torch.allclose(weight_cuda.grad, weight_ref.grad, atol=5e-3, rtol=5e-3)
    assert torch.allclose(bias_cuda.grad, bias_ref.grad, atol=5e-3, rtol=5e-3)
