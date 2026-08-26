"""Mixer tail reference: identities, precision policy, gradients, rejections.

The reference is the authority, so it cannot be checked against a faster path.
It is checked against properties that pin it uniquely instead: scale invariance
of the norm, the two limits of the gate, exactness of the skip, and float64
autograd.

The tail is also the boundary between the head-major side and the token-major
side, so the two conversions are written out here rather than imported from the
reference. Importing them would compare the reference against itself.
"""

import math

import pytest
import torch
from torch import Tensor
from torch.autograd import gradcheck
from torch.nn.functional import silu

from slinoss.ops.mixer import (
    as_head_major,
    as_token_major,
    mixer_tail,
    mixer_tail_bwd_ref,
    mixer_tail_ref,
)

EPS = 1e-5

NAMES = ("y", "u", "gate", "d_skip", "weight")

SHAPES = [
    pytest.param(2, 3, 5, 8, id="small"),
    pytest.param(1, 1, 1, 8, id="one-of-everything"),
    pytest.param(2, 4, 7, 64, id="wide-rows"),
]


def _head_major(t: Tensor, heads: int) -> Tensor:
    """``(B,T,H*P) -> (B,H,T,P)``."""
    return t.unflatten(-1, (heads, -1)).permute(0, 2, 1, 3)


def _token_major(t: Tensor) -> Tensor:
    """``(B,H,T,P) -> (B,T,H*P)``."""
    return t.permute(0, 2, 1, 3).flatten(-2, -1)


def _operands(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    *,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Five operands of the tail. One generator call per tensor, one dtype.

    ``gate`` is token-major, ``(B,T,H*P)``, like the projection column band it is
    a slice of. Everything else is head-major.
    """
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=dtype)

    return (
        rnd(bsz, heads, seqlen, rows),
        rnd(bsz, heads, seqlen, rows),
        rnd(bsz, seqlen, heads * rows),
        rnd(heads),
        rnd(heads, rows),
    )


@pytest.mark.parametrize(("bsz", "heads", "seqlen", "rows"), SHAPES)
def test_matches_the_written_composition(
    bsz: int, heads: int, seqlen: int, rows: int
) -> None:
    """The output is the skip, the gate, and the per-head norm, in that order.

    The expectation is written head-major and converted at the end, which also
    pins the output order: head ``h`` lands at columns ``h*P`` through
    ``(h+1)*P``.
    """
    y, u, gate, d_skip, weight = _operands(bsz, heads, seqlen, rows)
    got = mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS)

    x = y + d_skip[:, None, None] * u
    x = x * silu(_head_major(gate, heads))
    want = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight[:, None, :]
    assert torch.allclose(got, _token_major(want), rtol=0.0, atol=1e-15)


def test_reduction_does_not_cross_the_head_axis() -> None:
    """Perturbing one head leaves every other head's column band bit-identical.

    This is the property the fused kernel's rowwise structure depends on. A
    reduction over ``d_inner`` would fail it.
    """
    y, u, gate, d_skip, weight = _operands(2, 3, 4, 8)
    base = _head_major(mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS), 3)
    y[:, 1] += 1.0
    bumped = _head_major(mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS), 3)
    assert torch.equal(base[:, 0], bumped[:, 0])
    assert torch.equal(base[:, 2], bumped[:, 2])
    assert not torch.equal(base[:, 1], bumped[:, 1])


def test_norm_is_scale_invariant_up_to_eps() -> None:
    """Scaling the pre-norm value cancels, so ``eps`` is the only asymmetry.

    Both the skip and the gate are homogeneous in nothing, so the scale is
    applied to ``y`` and ``u`` together with ``d_skip`` left alone and the gate
    frozen at a constant, which makes the pre-norm value exactly proportional.
    """
    y, u, shaped, d_skip, weight = _operands(2, 2, 3, 8)
    gate = torch.full_like(shaped, 4.0)
    small = mixer_tail_ref(y, u, gate, d_skip, weight, eps=1e-30)
    large = mixer_tail_ref(8.0 * y, 8.0 * u, gate, d_skip, weight, eps=1e-30)
    assert torch.allclose(small, large, rtol=1e-12, atol=1e-12)


def test_closed_gate_kills_the_row() -> None:
    """A gate far enough negative that ``silu`` underflows leaves exact zeros.

    ``-800`` is past the point where the logistic underflows in float64, so the
    gated value is an exact zero rather than a small one, and the norm cannot
    resurrect it.
    """
    y, u, shaped, d_skip, weight = _operands(2, 2, 3, 8)
    dead = mixer_tail_ref(
        y, u, torch.full_like(shaped, -800.0), d_skip, weight, eps=EPS
    )
    assert torch.equal(dead, torch.zeros_like(dead))


def test_skip_is_the_only_path_when_y_is_zero() -> None:
    """With ``y = 0`` the pre-norm value is exactly ``d_skip * u * silu(gate)``."""
    _, u, gate, d_skip, weight = _operands(2, 2, 3, 8)
    y = torch.zeros_like(u)
    got = mixer_tail_ref(y, u, gate, d_skip, weight, eps=1e-30)
    x = d_skip[:, None, None] * u * silu(_head_major(gate, 2))
    want = (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-30) * weight[:, None, :]
    )
    assert torch.allclose(got, _token_major(want), rtol=1e-12, atol=1e-12)


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

    narrow = (y.to(dtype) + d_skip.to(dtype)[:, None, None] * u.to(dtype)) * silu(
        _head_major(gate.to(dtype), 2)
    )
    narrow = (
        narrow
        * torch.rsqrt(narrow.square().mean(-1, keepdim=True) + EPS)
        * weight.to(dtype)[:, None, :]
    )

    assert (got.float() - oracle).abs().max() < (
        _token_major(narrow).float() - oracle
    ).abs().max()


def test_layout_helpers_are_mutual_inverses_and_the_forward_one_is_a_view() -> None:
    """``as_head_major`` and ``as_token_major`` agree with the written reshape.

    Both are public, because the mixer's projection has to hand out column bands
    under the same convention the tail reads them under. ``as_head_major`` must be
    a view: it is applied to a band of the projection output on the hot path, and
    a copy there is a second pass over the largest tensor in the block.
    """
    gate = _operands(2, 3, 5, 8)[2]
    head_major = as_head_major(gate, 3)
    assert torch.equal(head_major, _head_major(gate, 3))
    assert head_major.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()
    assert torch.equal(as_token_major(head_major), gate)


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
def test_rejects_the_other_parameter_width(name: str) -> None:
    """``d_skip`` is ``(H,)`` and ``weight`` is ``(H,P)``.

    Each is refused at the other's width, which is the confusion worth a guard:
    the two parameters sit side by side in every signature, and a per-row skip
    broadcasts against ``(B,H,T,P)`` as silently as a per-head one.
    """
    y, u, gate, d_skip, weight = _operands(2, 2, 3, 8)
    wide = d_skip[:, None].expand_as(weight).contiguous()
    args = (wide, weight) if name == "d_skip" else (d_skip, weight[:, 0])
    with pytest.raises(ValueError, match=f"{name} must be"):
        mixer_tail_ref(y, u, gate, *args, eps=EPS)


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


def test_reference_backward_matches_autograd_through_the_public_operator() -> None:
    """The dispatched reference path is the same gradient autograd produces.

    :func:`mixer_tail` routes the backward through the registry rather than
    through autograd's own graph, so the two are only equal if the backend and
    the interface agree about the saved set and the argument order. The forward
    is the reference in both arms, so any difference is dispatch.
    """
    operands = _operands(2, 3, 5, 8)
    # Index 2 is the gate, which carries the output's shape.
    dout = _operands(2, 3, 5, 8, seed=7)[2]

    leaves = tuple(t.detach().clone().requires_grad_(True) for t in operands)
    mixer_tail(*leaves, eps=EPS, backend="reference").mul(dout).sum().backward()

    direct = tuple(t.detach().clone().requires_grad_(True) for t in operands)
    mixer_tail_ref(*direct, eps=EPS).mul(dout).sum().backward()

    for got, want, name in zip(leaves, direct, NAMES, strict=True):
        assert got.grad is not None and want.grad is not None
        torch.testing.assert_close(got.grad, want.grad, msg=f"d{name}")


def test_reference_backward_rejects_a_mismatched_cotangent() -> None:
    """The cotangent shape is the output shape; a broadcastable one is a bug."""
    y, u, gate, d_skip, weight = _operands(2, 2, 3, 8)
    with pytest.raises(ValueError, match="dout must be"):
        mixer_tail_bwd_ref(gate[:, :, :-1], y, u, gate, d_skip, weight, eps=EPS)


def test_backward_writes_dgate_into_a_supplied_band() -> None:
    """A supplied destination receives ``dgate`` in full, and only it moves.

    The mixer allocates one ``dproj`` and hands each consumer the band its gradient
    belongs in, so a destination is a column band of a wider buffer rather than a
    buffer of its own. The wider buffer is NaN first: a write that spilled past the
    band, or one that left a column of the band alone, lands as a NaN that the
    comparison against the allocating call then catches.
    """
    y, u, gate, d_skip, weight = _operands(2, 3, 5, 8)
    dout = _operands(2, 3, 5, 8, seed=7)[2]
    pad, width = 16, 3 * 8
    wide = torch.full((2, 5, width + 2 * pad), float("nan"), dtype=torch.float64)
    band = wide[..., pad : pad + width]

    got = mixer_tail_bwd_ref(dout, y, u, gate, d_skip, weight, eps=EPS, dgate=band)
    want = mixer_tail_bwd_ref(dout, y, u, gate, d_skip, weight, eps=EPS)

    assert got.dgate is band
    assert bool(wide[..., :pad].isnan().all())
    assert bool(wide[..., pad + width :].isnan().all())
    for one, other, name in zip(got, want, NAMES, strict=True):
        assert torch.equal(one, other), f"d{name}"
