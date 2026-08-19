"""Block norm and activation reference.

Checked against properties that pin each map uniquely rather than against a
faster path: the reference is the authority. The fused add-and-norm is checked
against the unfused pair it replaces, which is the one comparison that can catch
a fusion that changed the answer.

Each pullback is checked by ``gradcheck`` against a finite difference of the
forward above it. That needs the pair presented as one op, so each one gets an
:class:`torch.autograd.Function` whose backward calls it; without the wrapper
autograd would differentiate the forward again and the pullback under test would
never run.
"""

import math
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from torch.autograd import gradcheck
from torch.nn.functional import silu

from slinoss.ops.block import (
    NormResidual,
    rmsnorm_bwd_ref,
    rmsnorm_ref,
    rmsnorm_residual_bwd_ref,
    rmsnorm_residual_ref,
    swiglu_bwd_ref,
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


class _RMSNorm(torch.autograd.Function):
    """:func:`rmsnorm_ref` with :func:`rmsnorm_bwd_ref` as its backward."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rmsnorm_ref(x, weight, eps=eps)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, dout: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        x, weight = ctx.saved_tensors
        grads = rmsnorm_bwd_ref(dout, x, weight, eps=ctx.eps)
        return grads.dx, grads.dweight, None


class _NormResidual(torch.autograd.Function):
    """:func:`rmsnorm_residual_ref` with :func:`rmsnorm_residual_bwd_ref`.

    Materialized gradients are off, so an output that carries no cotangent hands
    the pullback None rather than a zero tensor. That is the case the None policy
    exists for, and gradcheck differentiates one output at a time, so it runs on
    every call.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, residual, weight)
        ctx.eps = eps
        out = rmsnorm_residual_ref(x, residual, weight, eps=eps)
        return out.normed, out.residual

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, dnormed: torch.Tensor | None, dresidual: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        x, residual, weight = ctx.saved_tensors
        grads = rmsnorm_residual_bwd_ref(
            dnormed, dresidual, x, residual, weight, eps=ctx.eps
        )
        return grads.dx, grads.dresidual, grads.dweight, None


class _SwiGLU(torch.autograd.Function):
    """:func:`swiglu_ref` with :func:`swiglu_bwd_ref` as its backward."""

    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        return swiglu_ref(gate, up)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, dout: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        grads = swiglu_bwd_ref(dout, gate, up)
        return grads.dgate, grads.dup


# `Function.apply` is untyped, so each wrapper is cast at the one call site rather
# than at four, which is the pattern the interface modules use.
def _norm_pair(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return cast("torch.Tensor", _RMSNorm.apply(x, weight, EPS))


def _residual_pair(
    x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = _NormResidual.apply(x, residual, weight, EPS)
    return cast("tuple[torch.Tensor, torch.Tensor]", out)


def _swiglu_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return cast("torch.Tensor", _SwiGLU.apply(gate, up))


def test_gradcheck_rmsnorm_bwd_ref() -> None:
    """The pullback against a finite difference of the forward, in float64."""
    x = _rnd((3, 8)).requires_grad_(True)
    weight = _rnd((8,), seed=1).requires_grad_(True)
    assert gradcheck(_norm_pair, (x, weight))


def test_gradcheck_rmsnorm_residual_bwd_ref() -> None:
    """Both outputs and all three inputs, with the residual present."""
    x = _rnd((3, 8)).requires_grad_(True)
    residual = _rnd((3, 8), seed=1).requires_grad_(True)
    weight = _rnd((8,), seed=2).requires_grad_(True)
    assert gradcheck(_residual_pair, (x, residual, weight))


def test_gradcheck_rmsnorm_residual_bwd_ref_none() -> None:
    """The first block of a stack: no incoming stream, so no ``dresidual``."""
    x = _rnd((3, 8)).requires_grad_(True)
    weight = _rnd((8,), seed=1).requires_grad_(True)
    assert gradcheck(lambda a, w: _residual_pair(a, None, w), (x, weight))


def test_gradcheck_swiglu_bwd_ref() -> None:
    """Both operand gradients against a finite difference of the forward."""
    gate = _rnd((3, 8)).requires_grad_(True)
    up = _rnd((3, 8), seed=1).requires_grad_(True)
    assert gradcheck(_swiglu_pair, (gate, up))


def test_rmsnorm_residual_bwd_ref_with_no_cotangent_returns_nothing() -> None:
    """Neither output differentiated is no gradient, not a zero gradient.

    A zero tensor here would be indistinguishable from a real gradient that
    happened to vanish, and it would cost a full-size allocation per absent
    cotangent.
    """
    grads = rmsnorm_residual_bwd_ref(
        None, None, _rnd((3, 8)), _rnd((3, 8), seed=1), _rnd((8,), seed=2), eps=EPS
    )
    assert grads.dx is None
    assert grads.dresidual is None
    assert grads.dweight is None


def test_rmsnorm_residual_bwd_ref_residual_only_leaves_the_weight_out() -> None:
    """The weight does not reach the residual output, so its gradient is None.

    The residual output is the plain sum, so its pullback is the identity on both
    summands and the comparison needs no tolerance.
    """
    x = _rnd((3, 8))
    residual = _rnd((3, 8), seed=1)
    dresidual = _rnd((3, 8), seed=3)
    grads = rmsnorm_residual_bwd_ref(
        None, dresidual, x, residual, _rnd((8,), seed=2), eps=EPS
    )
    assert grads.dweight is None
    assert grads.dx is not None and torch.equal(grads.dx, dresidual)
    assert grads.dresidual is not None and torch.equal(grads.dresidual, dresidual)


def test_rmsnorm_residual_bwd_ref_normed_only_is_the_unfused_pullback() -> None:
    """With only the normed cotangent the fusion is the plain norm of the sum.

    Both input gradients are then the same ``ds``, which is what the fused kernel
    exploits, and the weight gradient is the unfused one.
    """
    x = _rnd((3, 8))
    residual = _rnd((3, 8), seed=1)
    weight = _rnd((8,), seed=2)
    dnormed = _rnd((3, 8), seed=3)

    grads = rmsnorm_residual_bwd_ref(dnormed, None, x, residual, weight, eps=EPS)
    want = rmsnorm_bwd_ref(dnormed, x + residual, weight, eps=EPS)

    assert grads.dx is not None and torch.allclose(
        grads.dx, want.dx, rtol=0.0, atol=1e-15
    )
    assert grads.dresidual is not None and torch.equal(grads.dresidual, grads.dx)
    assert grads.dweight is not None and torch.allclose(
        grads.dweight, want.dweight, rtol=0.0, atol=1e-15
    )


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


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(
            lambda: rmsnorm_bwd_ref(
                _rnd((2, 7), seed=3), _rnd((2, 8)), _rnd((8,), seed=1), eps=EPS
            ),
            r"dout must be \(2, 8\)",
            id="norm-dout",
        ),
        pytest.param(
            lambda: rmsnorm_residual_bwd_ref(
                _rnd((2, 7), seed=3),
                None,
                _rnd((2, 8)),
                _rnd((2, 8), seed=1),
                _rnd((8,), seed=2),
                eps=EPS,
            ),
            r"dnormed must be \(2, 8\)",
            id="residual-dnormed",
        ),
        pytest.param(
            lambda: rmsnorm_residual_bwd_ref(
                None,
                _rnd((2, 7), seed=3),
                _rnd((2, 8)),
                _rnd((2, 8), seed=1),
                _rnd((8,), seed=2),
                eps=EPS,
            ),
            r"dresidual must be \(2, 8\)",
            id="residual-dresidual",
        ),
        pytest.param(
            lambda: swiglu_bwd_ref(
                _rnd((2, 7), seed=3), _rnd((2, 8)), _rnd((2, 8), seed=1)
            ),
            r"dout must be \(2, 8\)",
            id="swiglu-dout",
        ),
        pytest.param(
            lambda: swiglu_bwd_ref(
                _rnd((2, 8), seed=3), _rnd((2, 8)), _rnd((2, 7), seed=1)
            ),
            r"up must be \(2, 8\)",
            id="swiglu-up",
        ),
    ],
)
def test_rejects_mismatched_cotangent(call: Callable[[], object], match: str) -> None:
    """A cotangent is elementwise against the output it belongs to.

    A pullback that broadcast instead would return a gradient of the wrong shape
    and the caller would only find out at the optimizer.
    """
    with pytest.raises(ValueError, match=match):
        call()
