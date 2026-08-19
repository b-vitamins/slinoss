"""Bounded parameter maps. The numerical invariants live here, so they are
asserted here rather than guarded downstream."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from slinoss.ops.scanprep import (
    bounded_logscale,
    bounded_rotvec,
    scanprep,
    scanprep_bwd_ref,
    scanprep_ref,
)

Triple = tuple[Tensor, Tensor, Tensor]
Mutator = Callable[[Tensor, Tensor, Tensor], Triple]

EXTREME_RAWS: tuple[float, ...] = (
    -1e8,
    -1e4,
    -100.0,
    -20.0,
    -1.0,
    -1e-8,
    0.0,
    1e-8,
    1.0,
    20.0,
    100.0,
    1e4,
    1e8,
)
W_MAX = 3.0


def _raws(dtype: torch.dtype = torch.float64) -> Tensor:
    return torch.tensor(EXTREME_RAWS, dtype=dtype)


# ---------------------------------------------------------------------------
# I1: ls <= 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logscale_is_non_positive(dtype: torch.dtype) -> None:
    ls = bounded_logscale(_raws(dtype))
    assert bool((ls <= 0.0).all())
    assert bool(torch.isfinite(ls).all())


def test_logscale_decay_lies_in_the_unit_interval() -> None:
    # I1 admits underflow: at raw = 1e4 the decay is exp(-2e4), which is zero in
    # every float format. Zero decay is the correct limit, so the closed interval
    # is the invariant and the open one is asserted where it holds.
    decay = torch.exp(2.0 * bounded_logscale(_raws()))
    assert bool((decay >= 0.0).all())
    assert bool((decay <= 1.0).all())


def test_logscale_decay_is_strictly_positive_for_moderate_raws() -> None:
    decay = torch.exp(2.0 * bounded_logscale(torch.linspace(-40.0, 40.0, 201)))
    assert bool((decay > 0.0).all())
    assert bool((decay <= 1.0).all())


def test_logscale_is_strictly_negative_for_moderate_raws() -> None:
    ls = bounded_logscale(torch.linspace(-30.0, 30.0, 101, dtype=torch.float64))
    assert bool((ls < 0.0).all())


def test_logscale_matches_the_closed_form() -> None:
    raw = _raws()
    want = -torch.log1p(torch.exp(-raw.abs())) - raw.clamp_min(0.0)
    torch.testing.assert_close(bounded_logscale(raw), want, rtol=1e-14, atol=0.0)


def test_logscale_is_monotone_decreasing() -> None:
    ls = bounded_logscale(torch.linspace(-40.0, 40.0, 201, dtype=torch.float64))
    assert bool((ls.diff() <= 0.0).all())


def test_logscale_gradcheck() -> None:
    raw = torch.linspace(-8.0, 8.0, 17, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(bounded_logscale, (raw,))


# ---------------------------------------------------------------------------
# I2: |w| <= w_max < pi
# ---------------------------------------------------------------------------


# The exact map lands in the closed ball of radius w_max. The computed vector is
# that value rounded twice -- once in `w_max * rsqrt(1+s)` and once in the
# elementwise product -- so its norm can sit up to 2 ulp outside the radius.
# Measured worst case over 1.4e6 saturating samples: 2.00 ulp in float64, 1.81 in
# float32. The consumer is sized for the closed ball of radius pi, which absorbs
# that, so the answer is an honest bound here and not a clamp there.
ROUNDING_ULP = 3.0


def _ball_bound(dtype: torch.dtype) -> float:
    return W_MAX * (1.0 + ROUNDING_ULP * torch.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.0, 1e-8, 1.0, 1e4, 1e8, 1e16])
def test_rotvec_stays_inside_the_ball(dtype: torch.dtype, scale: float) -> None:
    gen = torch.Generator().manual_seed(2)
    raw = torch.randn(256, 3, generator=gen, dtype=torch.float64).to(dtype) * scale
    norm = bounded_rotvec(raw, W_MAX).double().norm(dim=-1)
    assert bool((norm <= _ball_bound(dtype)).all())
    assert bool((norm < math.pi).all())
    assert bool(torch.isfinite(norm).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotvec_saturates_to_the_bound(dtype: torch.dtype) -> None:
    """The bound is attained, not merely approached, which is why I2 is stated
    non-strictly."""
    gen = torch.Generator().manual_seed(3)
    raw = torch.randn(256, 3, generator=gen, dtype=torch.float64).to(dtype) * 1e8
    norm = bounded_rotvec(raw, W_MAX).double().norm(dim=-1)
    assert float(norm.min()) == pytest.approx(W_MAX, rel=8.0 * torch.finfo(dtype).eps)


def test_rotvec_survives_an_overflowing_raw_norm() -> None:
    """``|raw|^2`` overflows float32 near 1.8e19. ``rsqrt(inf)`` is zero, so the
    map collapses to the centre of the ball rather than producing a NaN. The
    result is still finite and still inside the ball, which is all I2 claims."""
    raw = torch.full((8, 3), 1e30, dtype=torch.float32)
    w = bounded_rotvec(raw, W_MAX)
    assert bool(torch.isfinite(w).all())
    assert float(w.double().norm(dim=-1).max()) <= _ball_bound(torch.float32)


def test_rotvec_matches_the_closed_form() -> None:
    gen = torch.Generator().manual_seed(4)
    raw = torch.randn(128, 3, generator=gen, dtype=torch.float64) * 7.0
    want = W_MAX * raw / torch.sqrt(1.0 + raw.pow(2).sum(-1, keepdim=True))
    assert float((bounded_rotvec(raw, W_MAX) - want).abs().max()) < 1e-15


def test_rotvec_at_zero_is_zero_exactly() -> None:
    raw = torch.zeros(4, 3, dtype=torch.float64)
    assert torch.equal(bounded_rotvec(raw, W_MAX), raw)


def test_rotvec_preserves_direction() -> None:
    gen = torch.Generator().manual_seed(6)
    raw = torch.randn(64, 3, generator=gen, dtype=torch.float64) * 3.0
    w = bounded_rotvec(raw, W_MAX)
    cross = torch.linalg.cross(raw, w)
    assert float(cross.abs().max()) < 1e-14
    assert bool(((raw * w).sum(-1) > 0.0).all())


def test_rotvec_is_monotone_in_the_raw_norm() -> None:
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    mags = torch.logspace(-6, 6, 121, dtype=torch.float64)[:, None]
    norms = bounded_rotvec(mags * axis, W_MAX).norm(dim=-1)
    assert bool((norms.diff() >= 0.0).all())
    assert float(norms[-1]) == pytest.approx(W_MAX, abs=1e-5)


def test_rotvec_gradcheck() -> None:
    gen = torch.Generator().manual_seed(8)
    raw = torch.randn(8, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: bounded_rotvec(x, W_MAX), (raw,))


def test_rotvec_gradcheck_at_zero() -> None:
    # 1 + |raw|^2 >= 1, so the rsqrt has no singularity and the map is smooth at
    # the origin. No clamp and no epsilon.
    raw = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: bounded_rotvec(x, W_MAX), (raw,))


@pytest.mark.parametrize("w_max", [0.0, -1.0, math.pi, 4.0, float("inf")])
def test_rotvec_rejects_illegal_bound(w_max: float) -> None:
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        bounded_rotvec(torch.zeros(1, 3), w_max)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def _raw_triple(
    bsz: int = 2,
    heads: int = 3,
    seqlen: int = 5,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)

    return (
        rnd(bsz, heads, seqlen, 3),
        rnd(bsz, heads, seqlen),
        rnd(bsz, heads, seqlen, 2, 3),
    )


def test_pack_shapes_and_contents() -> None:
    w_raw, ls_raw, tap_raw = _raw_triple()
    params = scanprep_ref(w_raw, ls_raw, tap_raw, w_max=W_MAX)
    assert params.trans.shape == (2, 3, 5, 4)
    assert params.K.shape == (2, 3, 5, 2, 4)
    assert params.trans.is_contiguous()
    assert params.K.is_contiguous()
    assert torch.equal(params.trans[..., :3], bounded_rotvec(w_raw, W_MAX))
    assert torch.equal(params.trans[..., 3], bounded_logscale(ls_raw))
    assert torch.equal(params.K[..., :3], tap_raw)


def test_pack_lane_three_is_a_hard_zero() -> None:
    params = scanprep_ref(*_raw_triple(), w_max=W_MAX)
    assert torch.equal(params.K[..., 3], torch.zeros_like(params.K[..., 3]))


def test_pack_invariants_hold_on_the_packed_tensors() -> None:
    w_raw, ls_raw, tap_raw = _raw_triple(seqlen=64, seed=1)
    params = scanprep_ref(w_raw * 1e6, ls_raw * 1e6, tap_raw, w_max=W_MAX)
    assert bool((params.trans[..., 3] <= 0.0).all())
    assert bool(
        (params.trans[..., :3].norm(dim=-1) <= _ball_bound(torch.float64)).all()
    )
    assert bool(torch.isfinite(params.trans).all())
    assert bool(torch.isfinite(params.K).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_pack_promotes_low_precision_to_float32(dtype: torch.dtype) -> None:
    params = scanprep_ref(*_raw_triple(dtype=dtype), w_max=W_MAX)
    assert params.trans.dtype is torch.float32
    assert params.K.dtype is torch.float32


def test_pack_keeps_float64() -> None:
    params = scanprep_ref(*_raw_triple(), w_max=W_MAX)
    assert params.trans.dtype is torch.float64
    assert params.K.dtype is torch.float64


def test_pack_gradcheck() -> None:
    w_raw, ls_raw, tap_raw = _raw_triple(bsz=1, heads=1, seqlen=3, seed=2)
    leaves = tuple(
        t.detach().clone().requires_grad_() for t in (w_raw, ls_raw, tap_raw)
    )

    def fn(w: Tensor, ls: Tensor, tap: Tensor) -> tuple[Tensor, Tensor]:
        params = scanprep_ref(w, ls, tap, w_max=W_MAX)
        return params.trans, params.K

    assert torch.autograd.gradcheck(fn, leaves)


def test_pack_runs_under_autocast() -> None:
    w_raw, ls_raw, tap_raw = _raw_triple(dtype=torch.float32)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        params = scanprep_ref(w_raw, ls_raw, tap_raw, w_max=W_MAX)
    assert params.trans.dtype is torch.float32
    assert params.K.dtype is torch.float32


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda w, ls, tap: (w[..., :2], ls, tap), r"w_raw must be \(B,H,T,3\)"),
        (lambda w, ls, tap: (w[0], ls, tap), r"w_raw must be \(B,H,T,3\)"),
        (lambda w, ls, tap: (w, ls[..., :-1], tap), "ls_raw must be"),
        (lambda w, ls, tap: (w, ls, tap[..., :2]), "tap_raw must be"),
        (lambda w, ls, tap: (w, ls, tap[..., :1, :]), "tap_raw must be"),
    ],
)
def test_pack_rejects_shape_mismatch(mutate: Mutator, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        scanprep_ref(*mutate(*_raw_triple()), w_max=W_MAX)


def test_pack_rejects_illegal_w_max() -> None:
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_ref(*_raw_triple(), w_max=math.pi)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_pack_rejects_unsupported_dtype(index: int) -> None:
    raws = list(_raw_triple())
    raws[index] = raws[index].to(torch.int64)
    with pytest.raises(TypeError, match="supported"):
        scanprep_ref(*raws, w_max=W_MAX)


def test_reference_backward_matches_autograd_through_the_public_operator() -> None:
    """The dispatched reference path is the same gradient autograd produces.

    :func:`scanprep` routes the backward through the registry rather than through
    autograd's own graph, and the registry's backward signature omits ``tap_raw``
    because the tap map is the identity. The two arms are only equal if that
    omission is sound and the saved set is right.
    """
    raws = _raw_triple(bsz=2, heads=3, seqlen=5, seed=3)
    dtrans = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    dK = torch.randn(2, 3, 5, 2, 4, dtype=torch.float64)

    leaves = tuple(t.detach().clone().requires_grad_(True) for t in raws)
    got = scanprep(*leaves, w_max=W_MAX, backend="reference")
    (got.trans * dtrans).sum().add((got.K * dK).sum()).backward()

    direct = tuple(t.detach().clone().requires_grad_(True) for t in raws)
    want = scanprep_ref(*direct, w_max=W_MAX)
    (want.trans * dtrans).sum().add((want.K * dK).sum()).backward()

    names = ("dw_raw", "dls_raw", "dtap_raw")
    for leaf, ref, name in zip(leaves, direct, names, strict=True):
        assert leaf.grad is not None and ref.grad is not None
        torch.testing.assert_close(leaf.grad, ref.grad, msg=name)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda dtrans, dK: (dtrans[..., :3], dK), "dtrans must be"),
        (lambda dtrans, dK: (dtrans, dK[..., :3]), "dK must be"),
    ],
)
def test_reference_backward_rejects_a_mismatched_cotangent(
    mutate: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    match: str,
) -> None:
    """Both cotangents are the packed layouts; a narrower one is a bug."""
    w_raw, ls_raw, _ = _raw_triple(bsz=1, heads=1, seqlen=2)
    dtrans = torch.zeros(1, 1, 2, 4, dtype=torch.float64)
    dK = torch.zeros(1, 1, 2, 2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match=match):
        scanprep_bwd_ref(*mutate(dtrans, dK), w_raw, ls_raw, w_max=W_MAX)
