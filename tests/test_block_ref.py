"""Block norm and activation reference.

Checked against properties that pin each map uniquely rather than against a
faster path: the reference is the authority. The fused add-and-norm is checked
against the unfused pair it replaces, which is the one comparison that can catch
a fusion that changed the answer.
"""

import math

import pytest
import torch
from torch.autograd import gradcheck
from torch.nn.functional import silu

from slinoss.ops.block import (
    NormResidual,
    rmsnorm_ref,
    rmsnorm_residual_ref,
    swiglu_ref,
)

EPS = 1e-5

SHAPES = [
    pytest.param((3, 8), id="rank-2"),
    pytest.param((2, 5, 16), id="rank-3"),
    pytest.param((1, 1, 64), id="one-token"),
    pytest.param((7,), id="rank-1"),
]


def _rnd(
    shape: tuple[int, ...], *, dtype: torch.dtype = torch.float64, seed: int = 0
) -> torch.Tensor:
    """One generator call, one dtype. Never the same seed at two widths."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_matches_the_written_form(shape: tuple[int, ...]) -> None:
    """The output is ``x * rsqrt(mean(x^2) + eps) * weight``."""
    x = _rnd(shape)
    weight = _rnd((shape[-1],), seed=1)
    got = rmsnorm_ref(x, weight, eps=EPS)
    want = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight
    assert torch.allclose(got, want, rtol=0.0, atol=1e-15)


def test_rmsnorm_is_scale_invariant_up_to_eps() -> None:
    """Scaling the input cancels, so ``eps`` is the only asymmetry."""
    x = _rnd((2, 4, 16))
    weight = _rnd((16,), seed=1)
    small = rmsnorm_ref(x, weight, eps=1e-30)
    large = rmsnorm_ref(1024.0 * x, weight, eps=1e-30)
    assert torch.allclose(small, large, rtol=1e-12, atol=1e-12)


def test_rmsnorm_unit_weight_gives_unit_mean_square() -> None:
    """With ``eps`` negligible the normed row has mean square exactly one."""
    x = _rnd((3, 16))
    weight = torch.ones(16, dtype=torch.float64)
    normed = rmsnorm_ref(x, weight, eps=1e-30)
    assert torch.allclose(
        normed.square().mean(-1),
        torch.ones(3, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_rmsnorm_zero_row_is_finite() -> None:
    """``eps`` is what keeps an all-zero row from dividing by zero."""
    got = rmsnorm_ref(torch.zeros(2, 8, dtype=torch.float64), torch.ones(8), eps=EPS)
    assert torch.isfinite(got).all()
    assert torch.equal(got, torch.zeros_like(got))


def test_rmsnorm_residual_equals_the_unfused_pair() -> None:
    """The fusion changes the traffic, not the answer."""
    x = _rnd((2, 4, 16))
    residual = _rnd((2, 4, 16), seed=1)
    weight = _rnd((16,), seed=2)
    out = rmsnorm_residual_ref(x, residual, weight, eps=EPS)
    assert isinstance(out, NormResidual)
    assert torch.equal(out.residual, x + residual)
    assert torch.allclose(
        out.normed, rmsnorm_ref(x + residual, weight, eps=EPS), rtol=0.0, atol=1e-15
    )


def test_rmsnorm_residual_none_is_the_first_block() -> None:
    """No incoming residual is the identity on the add, not a zero tensor add."""
    x = _rnd((2, 4, 16))
    weight = _rnd((16,), seed=1)
    out = rmsnorm_residual_ref(x, None, weight, eps=EPS)
    assert torch.equal(out.residual, x)
    assert torch.allclose(
        out.normed, rmsnorm_ref(x, weight, eps=EPS), rtol=0.0, atol=1e-15
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rmsnorm_residual_returns_a_wide_residual(dtype: torch.dtype) -> None:
    """The residual comes back wide even when the branch output is narrow.

    Narrowing the residual stream is what the fusion exists to avoid, so the
    dtype is part of the contract. The normed output still carries the input
    dtype.
    """
    x = _rnd((2, 4, 16), dtype=torch.float32).to(dtype)
    residual = _rnd((2, 4, 16), dtype=torch.float32, seed=1).to(dtype)
    weight = _rnd((16,), dtype=torch.float32, seed=2).to(dtype)
    out = rmsnorm_residual_ref(x, residual, weight, eps=EPS)
    assert out.residual.dtype is torch.float32
    assert out.normed.dtype is dtype


def test_rmsnorm_residual_accumulates_wider_than_its_operands() -> None:
    """A long chain of narrow adds drifts; the wide residual does not.

    Twenty rounds of add-and-norm in bfloat16 against the same twenty rounds
    carried in float32. The comparison needs no tolerance: the reference must be
    strictly closer to the float32 chain than a chain that narrows the residual
    at every step.
    """
    weight = _rnd((32,), dtype=torch.float32, seed=1)
    branches = [_rnd((4, 32), dtype=torch.float32, seed=10 + i) for i in range(20)]

    wide = None
    for branch in branches:
        wide = rmsnorm_residual_ref(branch, wide, weight, eps=EPS).residual

    kept = None
    narrowed = None
    for branch in branches:
        low = branch.to(torch.bfloat16)
        kept = rmsnorm_residual_ref(
            low, kept, weight.to(torch.bfloat16), eps=EPS
        ).residual
        step = rmsnorm_residual_ref(
            low,
            None if narrowed is None else narrowed,
            weight.to(torch.bfloat16),
            eps=EPS,
        )
        narrowed = step.residual.to(torch.bfloat16)

    assert wide is not None and kept is not None and narrowed is not None
    assert (kept - wide).abs().max() < (narrowed.float() - wide).abs().max()


@pytest.mark.parametrize("shape", SHAPES)
def test_swiglu_matches_the_written_form(shape: tuple[int, ...]) -> None:
    """``silu(gate) * up``."""
    gate = _rnd(shape)
    up = _rnd(shape, seed=1)
    assert torch.allclose(swiglu_ref(gate, up), silu(gate) * up, rtol=0.0, atol=1e-15)


def test_swiglu_closed_gate_kills_the_row() -> None:
    """Past the logistic underflow the product is an exact zero."""
    up = _rnd((2, 8))
    got = swiglu_ref(torch.full_like(up, -800.0), up)
    assert torch.equal(got, torch.zeros_like(got))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_swiglu_output_carries_the_dtype_of_up(dtype: torch.dtype) -> None:
    """The activation is an elementwise map, so it does not widen its output."""
    gate = _rnd((2, 8), dtype=torch.float32).to(dtype)
    up = _rnd((2, 8), dtype=torch.float32, seed=1).to(dtype)
    assert swiglu_ref(gate, up).dtype is dtype


def test_gradcheck_rmsnorm() -> None:
    """float64 autograd on the input and the weight."""
    x = _rnd((3, 8)).requires_grad_(True)
    weight = _rnd((8,), seed=1).requires_grad_(True)
    assert gradcheck(lambda a, w: rmsnorm_ref(a, w, eps=EPS), (x, weight))


def test_gradcheck_rmsnorm_residual() -> None:
    """float64 autograd on both outputs and all three inputs."""
    x = _rnd((3, 8)).requires_grad_(True)
    residual = _rnd((3, 8), seed=1).requires_grad_(True)
    weight = _rnd((8,), seed=2).requires_grad_(True)
    assert gradcheck(
        lambda a, r, w: rmsnorm_residual_ref(a, r, w, eps=EPS),
        (x, residual, weight),
    )


def test_gradcheck_rmsnorm_residual_none() -> None:
    """The no-residual path is differentiable too."""
    x = _rnd((3, 8)).requires_grad_(True)
    weight = _rnd((8,), seed=1).requires_grad_(True)
    assert gradcheck(
        lambda a, w: rmsnorm_residual_ref(a, None, w, eps=EPS), (x, weight)
    )


def test_gradcheck_swiglu() -> None:
    """float64 autograd on both operands."""
    gate = _rnd((3, 8)).requires_grad_(True)
    up = _rnd((3, 8), seed=1).requires_grad_(True)
    assert gradcheck(swiglu_ref, (gate, up))


def test_rejects_scalar_input() -> None:
    """A norm needs an axis to reduce over."""
    with pytest.raises(ValueError, match="at least one axis"):
        rmsnorm_ref(torch.tensor(1.0, dtype=torch.float64), torch.ones(1), eps=EPS)


def test_rejects_mismatched_weight() -> None:
    """The weight is one scalar per trailing element."""
    with pytest.raises(ValueError, match="weight must be"):
        rmsnorm_ref(_rnd((2, 8)), _rnd((7,), seed=1), eps=EPS)


@pytest.mark.parametrize("eps", [0.0, -1e-5, -math.inf])
def test_rejects_non_positive_eps(eps: float) -> None:
    """A non-positive epsilon reintroduces the division by zero it exists to stop."""
    with pytest.raises(ValueError, match="eps must be positive"):
        rmsnorm_ref(_rnd((2, 8)), _rnd((8,), seed=1), eps=eps)


@pytest.mark.parametrize("name", ["x", "weight"])
def test_rejects_unsupported_norm_dtype(name: str) -> None:
    """An integer operand has no path; the message names the offender."""
    tensors = {"x": _rnd((2, 8)), "weight": _rnd((8,), seed=1)}
    tensors[name] = tensors[name].to(torch.int32)
    with pytest.raises(TypeError, match=name):
        rmsnorm_ref(tensors["x"], tensors["weight"], eps=EPS)


def test_rejects_mismatched_residual() -> None:
    """The residual is elementwise against the branch output."""
    with pytest.raises(ValueError, match="residual must be"):
        rmsnorm_residual_ref(
            _rnd((2, 8)), _rnd((3, 8), seed=1), _rnd((8,), seed=2), eps=EPS
        )


def test_rejects_unsupported_residual_dtype() -> None:
    """The residual is checked like every other operand."""
    with pytest.raises(TypeError, match="residual"):
        rmsnorm_residual_ref(
            _rnd((2, 8)),
            torch.zeros(2, 8, dtype=torch.int32),
            _rnd((8,), seed=2),
            eps=EPS,
        )


def test_rejects_mismatched_swiglu() -> None:
    """The activation is elementwise, so no broadcasting."""
    with pytest.raises(ValueError, match="up must be"):
        swiglu_ref(_rnd((2, 8)), _rnd((2, 7), seed=1))


@pytest.mark.parametrize("name", ["gate", "up"])
def test_rejects_unsupported_swiglu_dtype(name: str) -> None:
    """Both activation operands are checked."""
    tensors = {"gate": _rnd((2, 8)), "up": _rnd((2, 8), seed=1)}
    tensors[name] = tensors[name].to(torch.int32)
    with pytest.raises(TypeError, match=name):
        swiglu_ref(tensors["gate"], tensors["up"])
