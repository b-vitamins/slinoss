"""Mixer tail reference: identities, precision policy, gradients, rejections.

The reference is the authority, so it cannot be checked against a faster path.
It is checked against properties that pin it uniquely instead: scale invariance
of the norm, the two limits of the gate, exactness of the skip, and float64
autograd.
"""

import math

import pytest
import torch
from torch.autograd import gradcheck
from torch.nn.functional import silu

from slinoss.ops.mixer import mixer_tail_ref

EPS = 1e-5

SHAPES = [
    pytest.param(2, 3, 5, 8, id="small"),
    pytest.param(1, 1, 1, 8, id="one-of-everything"),
    pytest.param(2, 4, 7, 64, id="wide-rows"),
]


def _operands(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    *,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Five operands of the tail. One generator call per tensor, one dtype."""
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=dtype)

    return (
        rnd(bsz, heads, seqlen, rows),
        rnd(bsz, heads, seqlen, rows),
        rnd(bsz, heads, seqlen, rows),
        rnd(heads, rows),
        rnd(heads, rows),
    )


@pytest.mark.parametrize(("bsz", "heads", "seqlen", "rows"), SHAPES)
def test_matches_the_written_composition(
    bsz: int, heads: int, seqlen: int, rows: int
) -> None:
    """The output is the skip, the gate, and the per-head norm, in that order."""
    y, u, gate, d_skip, weight = _operands(bsz, heads, seqlen, rows)
    got = mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS)

    x = y + d_skip[:, None, :] * u
    x = x * silu(gate)
    want = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight[:, None, :]
    assert torch.allclose(got, want, rtol=0.0, atol=1e-15)


def test_reduction_does_not_cross_the_head_axis() -> None:
    """Perturbing one head leaves every other head bit-identical.

    This is the property the fused kernel's rowwise structure depends on. A
    reduction over ``d_inner`` would fail it.
    """
    y, u, gate, d_skip, weight = _operands(2, 3, 4, 8)
    base = mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS)
    y[:, 1] += 1.0
    bumped = mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS)
    assert torch.equal(base[:, 0], bumped[:, 0])
    assert torch.equal(base[:, 2], bumped[:, 2])
    assert not torch.equal(base[:, 1], bumped[:, 1])


def test_norm_is_scale_invariant_up_to_eps() -> None:
    """Scaling the pre-norm value cancels, so ``eps`` is the only asymmetry.

    Both the skip and the gate are homogeneous in nothing, so the scale is
    applied to ``y`` and ``u`` together with ``d_skip`` left alone and the gate
    frozen at a constant, which makes the pre-norm value exactly proportional.
    """
    y, u, _, d_skip, weight = _operands(2, 2, 3, 8)
    gate = torch.full_like(y, 4.0)
    small = mixer_tail_ref(y, u, gate, d_skip, weight, eps=1e-30)
    large = mixer_tail_ref(8.0 * y, 8.0 * u, gate, d_skip, weight, eps=1e-30)
    assert torch.allclose(small, large, rtol=1e-12, atol=1e-12)


def test_closed_gate_kills_the_row() -> None:
    """A gate far enough negative that ``silu`` underflows leaves exact zeros.

    ``-800`` is past the point where the logistic underflows in float64, so the
    gated value is an exact zero rather than a small one, and the norm cannot
    resurrect it.
    """
    y, u, _, d_skip, weight = _operands(2, 2, 3, 8)
    dead = mixer_tail_ref(y, u, torch.full_like(y, -800.0), d_skip, weight, eps=EPS)
    assert torch.equal(dead, torch.zeros_like(dead))


def test_skip_is_the_only_path_when_y_is_zero() -> None:
    """With ``y = 0`` the pre-norm value is exactly ``d_skip * u * silu(gate)``."""
    _, u, gate, d_skip, weight = _operands(2, 2, 3, 8)
    y = torch.zeros_like(u)
    got = mixer_tail_ref(y, u, gate, d_skip, weight, eps=1e-30)
    x = d_skip[:, None, :] * u * silu(gate)
    want = (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-30) * weight[:, None, :]
    )
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_zero_row_is_finite() -> None:
    """``eps`` is what keeps an all-zero row from dividing by zero."""
    y, u, gate, d_skip, weight = _operands(1, 1, 2, 8)
    got = mixer_tail_ref(
        torch.zeros_like(y),
        torch.zeros_like(u),
        gate,
        d_skip,
        weight,
        eps=EPS,
    )
    assert torch.isfinite(got).all()
    assert torch.equal(got, torch.zeros_like(got))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_output_carries_the_dtype_of_y(dtype: torch.dtype) -> None:
    """Low precision in, the same low precision out."""
    y, u, gate, d_skip, weight = _operands(2, 2, 4, 64, dtype=torch.float32)
    got = mixer_tail_ref(
        y.to(dtype),
        u.to(dtype),
        gate.to(dtype),
        d_skip.to(dtype),
        weight.to(dtype),
        eps=EPS,
    )
    assert got.dtype is dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reduction_is_wider_than_the_operands(dtype: torch.dtype) -> None:
    """The sum of squares accumulates wide, and that is observable.

    Compared against the same computation carried out entirely in the narrow
    dtype, over ``P = 64`` rows where a narrow accumulation loses several bits.
    The comparison needs no tolerance: the reference must be strictly closer to
    the float32 result than the narrow accumulation is.
    """
    y, u, gate, d_skip, weight = _operands(2, 2, 4, 64, dtype=torch.float32)
    oracle = mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS)
    got = mixer_tail_ref(
        y.to(dtype),
        u.to(dtype),
        gate.to(dtype),
        d_skip.to(dtype),
        weight.to(dtype),
        eps=EPS,
    )

    narrow = (y.to(dtype) + d_skip.to(dtype)[:, None, :] * u.to(dtype)) * silu(
        gate.to(dtype)
    )
    narrow = (
        narrow
        * torch.rsqrt(narrow.square().mean(-1, keepdim=True) + EPS)
        * weight.to(dtype)[:, None, :]
    )

    assert (got.float() - oracle).abs().max() < (narrow.float() - oracle).abs().max()


def test_float64_in_float64_out() -> None:
    """A float64 call stays float64 end to end, so it is an oracle."""
    y, u, gate, d_skip, weight = _operands(1, 2, 3, 8)
    assert mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS).dtype is torch.float64


def test_gradcheck() -> None:
    """float64 autograd on all five operands. No quantity is exempt."""
    operands = tuple(t.requires_grad_(True) for t in _operands(2, 2, 3, 8))
    assert gradcheck(
        lambda *args: mixer_tail_ref(*args, eps=EPS),
        operands,
        eps=1e-6,
        atol=1e-8,
        rtol=1e-6,
    )


def test_rejects_wrong_rank() -> None:
    """The tail is time-major and four-dimensional."""
    y, u, gate, d_skip, weight = _operands(2, 2, 3, 8)
    with pytest.raises(ValueError, match=r"y must be \(B,H,T,P\)"):
        mixer_tail_ref(y[0], u, gate, d_skip, weight, eps=EPS)


@pytest.mark.parametrize("name", ["u", "gate"])
def test_rejects_mismatched_activation(name: str) -> None:
    """``u`` and ``gate`` are elementwise against ``y``, so no broadcasting."""
    y, u, gate, d_skip, weight = _operands(2, 2, 3, 8)
    tensors = {"u": u, "gate": gate}
    tensors[name] = tensors[name][:, :, :-1]
    with pytest.raises(ValueError, match=f"{name} must be"):
        mixer_tail_ref(y, tensors["u"], tensors["gate"], d_skip, weight, eps=EPS)


@pytest.mark.parametrize("name", ["d_skip", "weight"])
def test_rejects_mismatched_parameter(name: str) -> None:
    """Both parameters are ``(H,P)``; a ``d_inner`` vector is a caller bug."""
    y, u, gate, d_skip, weight = _operands(2, 2, 3, 8)
    tensors = {"d_skip": d_skip, "weight": weight}
    tensors[name] = tensors[name].flatten()
    with pytest.raises(ValueError, match=f"{name} must be"):
        mixer_tail_ref(y, u, gate, tensors["d_skip"], tensors["weight"], eps=EPS)


@pytest.mark.parametrize("eps", [0.0, -1e-5, -math.inf])
def test_rejects_non_positive_eps(eps: float) -> None:
    """A non-positive epsilon reintroduces the division by zero it exists to stop."""
    y, u, gate, d_skip, weight = _operands(1, 1, 2, 8)
    with pytest.raises(ValueError, match="eps must be positive"):
        mixer_tail_ref(y, u, gate, d_skip, weight, eps=eps)


@pytest.mark.parametrize("name", ["y", "u", "gate", "d_skip", "weight"])
def test_rejects_unsupported_dtype(name: str) -> None:
    """An integer operand has no path; the message names the offender."""
    y, u, gate, d_skip, weight = _operands(1, 1, 2, 8)
    tensors = {"y": y, "u": u, "gate": gate, "d_skip": d_skip, "weight": weight}
    tensors[name] = tensors[name].to(torch.int32)
    with pytest.raises(TypeError, match=name):
        mixer_tail_ref(
            tensors["y"],
            tensors["u"],
            tensors["gate"],
            tensors["d_skip"],
            tensors["weight"],
            eps=EPS,
        )
