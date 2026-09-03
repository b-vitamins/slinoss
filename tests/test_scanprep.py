"""Bounded parameter maps. The numerical invariants live here, so they are
asserted here rather than guarded downstream."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from slinoss.ops.scanprep import (
    LS_MAX_MAG,
    PARAM_COLS,
    ScanParams,
    anchored_rotvec,
    bounded_logscale,
    bounded_rotvec,
    foh_taps,
    pack_params,
    scanprep,
    scanprep_bwd_ref,
    scanprep_ref,
)
from slinoss.ops.scanprep.reference import FP32_FOH_TERMS
from slinoss.ops.so3ssd import skew, tap_matrix

Pair = tuple[Tensor, Tensor]
Mutator = Callable[[Tensor, Tensor], Pair]

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
# I1: -LS_MAX_MAG <= ls <= 0
# ---------------------------------------------------------------------------


# Both wide dtypes: float32 is what the pinned transition ships in and float64 is
# the oracle width. No kernel clamps ``ls``, so the bound is asserted in each
# format the map is evaluated in rather than in one and assumed in the other.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logscale_is_non_positive(dtype: torch.dtype) -> None:
    ls = bounded_logscale(_raws(dtype))
    assert bool((ls <= 0.0).all())
    assert bool(torch.isfinite(ls).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logscale_holds_the_lifetime_floor(dtype: torch.dtype) -> None:
    """The lower half of I1, and that it is attained rather than approached.

    The floor is what stops a token annihilating a row: at ``-LS_MAX_MAG`` the
    per-token amplitude factor is ``exp(-2*LS_MAX_MAG)``, a lifetime of
    ``0.5/LS_MAX_MAG`` tokens, and nothing downstream can drive it lower. No kernel
    reads the bound, so a map that lost it would pass every other test here.
    """
    ls = bounded_logscale(_raws(dtype))
    assert bool((ls >= -LS_MAX_MAG).all())
    assert float(ls.min()) == pytest.approx(-LS_MAX_MAG, rel=torch.finfo(dtype).eps)
    assert float(torch.exp(2.0 * ls).min()) == pytest.approx(
        math.exp(-2.0 * LS_MAX_MAG)
    )


def test_logscale_is_strictly_negative_with_a_positive_decay() -> None:
    """Both strict claims, on the moderate range, in both wide dtypes.

    I1 is stated closed at both ends because ``sigmoid`` saturates to an exact 0 and
    an exact 1 in float: past raw = -40 the map returns a signed zero and past +40 it
    returns the floor. Strictness holds where the sigmoid is not saturated, which is
    the whole range any initialized row occupies.
    """
    for dtype in (torch.float32, torch.float64):
        ls = bounded_logscale(torch.linspace(-16.0, 16.0, 201, dtype=dtype))
        decay = torch.exp(2.0 * ls)
        assert bool((ls < 0.0).all())
        assert bool((decay > 0.0).all())
        assert bool((decay < 1.0).all())


def test_logscale_matches_the_closed_form() -> None:
    """Against the definition, written out.

    The quotient is accurate in both tails without a fold: an overflowing
    ``exp(-raw)`` drives it to a signed zero, which is the map's own limit there, and
    a vanishing one leaves the floor exactly.
    """
    raw = _raws()
    want = -LS_MAX_MAG / (1.0 + torch.exp(-raw))
    torch.testing.assert_close(bounded_logscale(raw), want, rtol=1e-14, atol=0.0)


def test_logscale_is_monotone_decreasing() -> None:
    ls = bounded_logscale(torch.linspace(-40.0, 40.0, 201, dtype=torch.float64))
    assert bool((ls.diff() <= 0.0).all())


def test_logscale_gradcheck() -> None:
    raw = torch.linspace(-8.0, 8.0, 17, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(bounded_logscale, (raw,))


# ---------------------------------------------------------------------------
# The anchored drive
# ---------------------------------------------------------------------------


def _drive_pair(seed: int, radius: float) -> Pair:
    """``(band, bias)``, both ``(256,3)`` float64, the bias at a fixed radius."""
    gen = torch.Generator().manual_seed(seed)
    band = torch.randn(256, 3, generator=gen, dtype=torch.float64) * 4.0
    bias = torch.randn(256, 3, generator=gen, dtype=torch.float64)
    return band, radius * bias / bias.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("radius", [3.0, 1e-2, 4.8e-4])
def test_rotation_drive_is_an_unconstrained_displacement(radius: float) -> None:
    """Every head gets the same unit chart, independent of its initial period."""
    band, bias = _drive_pair(seed=21, radius=radius)
    drive = anchored_rotvec(band, bias) - bias
    torch.testing.assert_close(drive, band, rtol=2e-15, atol=2e-15)


def test_anchored_drive_is_the_bias_at_a_zero_band() -> None:
    """A zeroed parameter band leaves the initialized bank exactly, which is what the
    mixer's zeroed projection rows rely on."""
    _, bias = _drive_pair(seed=22, radius=0.5)
    assert torch.equal(anchored_rotvec(torch.zeros_like(bias), bias), bias)


def test_rotation_drive_at_a_zero_bias_is_still_the_band() -> None:
    """Training a head through the chart origin cannot delete its token dependence."""
    band, _ = _drive_pair(seed=23, radius=1.0)
    bias = torch.zeros_like(band)
    out = anchored_rotvec(band, bias)
    assert torch.equal(out, band)


def test_rotation_drive_has_unit_pullback_to_both_operands() -> None:
    band = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    anchored_rotvec(band, bias).sum().backward()
    assert torch.equal(band.grad, torch.ones_like(band))
    assert torch.equal(bias.grad, torch.ones_like(bias))


def test_anchored_drive_gradcheck() -> None:
    """Float64 gradcheck on both additive operands."""
    gen = torch.Generator().manual_seed(24)
    band = torch.randn(8, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(8, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(anchored_rotvec, (band, bias))


# ---------------------------------------------------------------------------
# I2: |w| <= 2*w_max < 2*pi
# ---------------------------------------------------------------------------


# The exact map lands in the closed ball of radius 2*w_max. The computed vector is
# that value rounded twice -- once in `w_max * rsqrt(1+s/4)` and once in the
# elementwise product -- so its norm can sit up to 2 ulp outside the radius.
# Measured worst case over 1.4e6 saturating samples: 2.00 ulp in float64, 1.81 in
# float32. The consumer is sized for the closed ball of radius 2*pi, which absorbs
# that, so the answer is an honest bound here and not a clamp there.
ROUNDING_ULP = 3.0


def _ball_bound(dtype: torch.dtype) -> float:
    return 2.0 * W_MAX * (1.0 + ROUNDING_ULP * torch.finfo(dtype).eps)


# (dtype, scale). The map is one branchless expression, so the raw magnitude
# selects a regime of the ratio ``|raw| / sqrt(1 + |raw|^2/4)`` rather than a path.
# One case per regime, crossed with dtype because the admissible excess over the
# radius is dtype-scaled and the two figures above were measured separately:
#
# - ``|raw| = 0``, the boundary where the ratio is 0/1 and a normalize-by-``|raw|``
#   map would divide by zero. float32 only; float64 at the origin is pinned to an
#   exact zero by ``test_rotvec_at_zero_is_zero_exactly``;
# - a generic ``|raw|``, where the ratio is strictly inside ``(0,1)`` in both
#   formats and the norm is strictly inside the ball;
# - ``|raw|`` past the reciprocal of the machine epsilon, where the ratio rounds to
#   one, the radius is attained, and the two roundings can put the norm outside it.
BALL_CASES = [
    pytest.param(torch.float32, 0.0, id="f32-zero"),
    pytest.param(torch.float32, 1.0, id="f32-generic"),
    pytest.param(torch.float64, 1.0, id="f64-generic"),
    pytest.param(torch.float32, 1e16, id="f32-saturated"),
    pytest.param(torch.float64, 1e16, id="f64-saturated"),
]


@pytest.mark.parametrize(("dtype", "scale"), BALL_CASES)
def test_rotvec_stays_inside_the_ball(dtype: torch.dtype, scale: float) -> None:
    gen = torch.Generator().manual_seed(2)
    raw = torch.randn(256, 3, generator=gen, dtype=torch.float64).to(dtype) * scale
    norm = bounded_rotvec(raw, W_MAX).double().norm(dim=-1)
    assert bool((norm <= _ball_bound(dtype)).all())
    assert bool((norm < 2.0 * math.pi).all())
    assert bool(torch.isfinite(norm).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotvec_saturates_to_the_bound(dtype: torch.dtype) -> None:
    """The bound is attained, not merely approached, which is why I2 is stated
    non-strictly."""
    gen = torch.Generator().manual_seed(3)
    raw = torch.randn(256, 3, generator=gen, dtype=torch.float64).to(dtype) * 1e8
    norm = bounded_rotvec(raw, W_MAX).double().norm(dim=-1)
    assert float(norm.min()) == pytest.approx(
        2.0 * W_MAX, rel=8.0 * torch.finfo(dtype).eps
    )


def test_rotvec_survives_an_overflowing_raw_norm() -> None:
    """``|raw|^2`` overflows float32 near 1.8e19. ``rsqrt(inf)`` is zero, so the
    map collapses to the centre of the ball rather than producing a NaN. The
    result is still finite and still inside the ball, which is all I2 claims."""
    raw = torch.full((8, 3), 1e30, dtype=torch.float32)
    w = bounded_rotvec(raw, W_MAX)
    assert bool(torch.isfinite(w).all())
    assert float(w.double().norm(dim=-1).max()) <= _ball_bound(torch.float32)


def test_rotvec_matches_the_closed_form() -> None:
    """Magnitude and direction both, against the closed form.

    The reference is a positive scalar times ``raw``, so agreement pins the
    direction too: a per-component ratio in place of the shared norm fails here.
    """
    gen = torch.Generator().manual_seed(4)
    raw = torch.randn(128, 3, generator=gen, dtype=torch.float64) * 7.0
    want = W_MAX * raw / torch.sqrt(1.0 + 0.25 * raw.pow(2).sum(-1, keepdim=True))
    assert float((bounded_rotvec(raw, W_MAX) - want).abs().max()) < 1e-15


def test_rotvec_at_zero_is_zero_exactly() -> None:
    raw = torch.zeros(4, 3, dtype=torch.float64)
    assert torch.equal(bounded_rotvec(raw, W_MAX), raw)


def test_rotvec_is_monotone_in_the_raw_norm() -> None:
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    mags = torch.logspace(-6, 6, 121, dtype=torch.float64)[:, None]
    norms = bounded_rotvec(mags * axis, W_MAX).norm(dim=-1)
    assert bool((norms.diff() >= 0.0).all())
    assert float(norms[-1]) == pytest.approx(2.0 * W_MAX, abs=1e-5)


def test_rotvec_gradcheck() -> None:
    gen = torch.Generator().manual_seed(8)
    raw = torch.randn(8, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: bounded_rotvec(x, W_MAX), (raw,))


def test_rotvec_gradcheck_at_zero() -> None:
    # 1 + |raw|^2/4 >= 1, so the rsqrt has no singularity and the map is smooth at
    # the origin. No clamp and no epsilon.
    raw = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: bounded_rotvec(x, W_MAX), (raw,))


@pytest.mark.parametrize("w_max", [0.0, -1.0, 3.14159265, math.pi, 4.0, float("inf")])
def test_rotvec_rejects_illegal_bound(w_max: float) -> None:
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        bounded_rotvec(torch.zeros(1, 3), w_max)


# ---------------------------------------------------------------------------
# The taps
# ---------------------------------------------------------------------------


def _moment_oracle(w: Tensor, ls: Tensor) -> Pair:
    """``(int_0^1 r exp(Lr) dr, int_0^1 (1-r) exp(Lr) dr)`` as explicit matrices.

    ``L = 2*ls*I + skew(w)`` and the two moments are ``phi_1(L) - phi_2(L)`` and
    ``phi_2(L)``, which sit in the first block row of

        exp([[L, I, 0], [0, 0, I], [0, 0, 0]])

    because that block matrix's powers shift the identity along the row. One
    ``matrix_exp`` therefore gives both, sharing no code with the chart under test:
    neither the recurrence nor the truncated series appears here.
    """
    lead = w.shape[:-1]
    eye = torch.eye(3, dtype=w.dtype).expand(*lead, 3, 3)
    block = torch.zeros(*lead, 9, 9, dtype=w.dtype)
    block[..., 0:3, 0:3] = 2.0 * ls[..., None, None] * eye + skew(w)
    block[..., 0:3, 3:6] = eye
    block[..., 3:6, 6:9] = eye
    out = torch.linalg.matrix_exp(block)
    phi1, phi2 = out[..., 0:3, 3:6], out[..., 0:3, 6:9]
    return phi1 - phi2, phi2


def test_taps_are_the_first_order_hold_moments() -> None:
    """The taps against the integrals they are defined as, as matrices.

    The comparison is at the matrix rather than at the chart, which is where the
    operator reads them: ``g`` carries ``1/|w|^2`` and is ill-conditioned as ``|w|``
    falls while ``g * w w^T`` is not, so a chart-level tolerance would be a claim
    about the corner and not about the operator.

    The grid crosses ``FOH_TAYLOR_RADIUS_SQ`` in both arguments and includes
    ``|w| = 0``, where the chart divides by the floor. Non-axis directions are in it
    because a transposed ``skew`` is invisible along a coordinate axis.
    """
    mags = torch.tensor([0.0, 1e-8, 1e-3, 0.5, 0.99, 2.0, 3.14], dtype=torch.float64)
    axes = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.4, -0.5, 0.7], [-0.9, 0.2, -0.1]],
        dtype=torch.float64,
    )
    axes = axes / axes.norm(dim=-1, keepdim=True)
    scales = torch.tensor([0.0, -1e-8, -1e-3, -0.5, -2.0, -9.0], dtype=torch.float64)
    shape = (len(mags), len(axes), len(scales))
    w = (mags[:, None, None, None] * axes[None, :, None, :]).expand(*shape, 3)
    ls = scales[None, None, :].expand(*shape)

    tap = foh_taps(w, ls)
    for slot, want in enumerate(_moment_oracle(w, ls)):
        got = tap_matrix(tap[..., slot, :].contiguous(), w)
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-13)


def test_taps_hold_the_float32_truncation_to_its_own_width() -> None:
    """The device path takes a shorter truncation of the same generator, so the two
    term counts are one contract and the shorter one is held to float32."""
    w, ls = _raw_pair(seed=12)
    w = bounded_rotvec(w, W_MAX)
    ls = bounded_logscale(ls)
    wide = foh_taps(w, ls)
    short = foh_taps(w, ls, terms=FP32_FOH_TERMS)
    torch.testing.assert_close(short, wide, rtol=0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Column packing
# ---------------------------------------------------------------------------


def _raw_pair(
    bsz: int = 2,
    heads: int = 3,
    seqlen: int = 5,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> Pair:
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)

    return rnd(bsz, heads, seqlen, 3), rnd(bsz, heads, seqlen)


def test_pack_params_lays_out_the_projection_column_order() -> None:
    """The column order is the operator's contract with the projection, and every
    kernel indexes it by hand, so it is asserted rather than assumed."""
    w_raw, ls_raw = _raw_pair()
    row = pack_params(w_raw, ls_raw)
    assert row.shape == (2, 5, 3 * PARAM_COLS)
    assert row.is_contiguous()
    head_major = row.unflatten(-1, (3, PARAM_COLS)).permute(0, 2, 1, 3)
    assert torch.equal(head_major[..., 0:3], w_raw)
    assert torch.equal(head_major[..., 3], ls_raw)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda w, ls: (w[..., :2], ls), r"w_raw must be \(B,H,T,3\)"),
        (lambda w, ls: (w[0], ls), r"w_raw must be \(B,H,T,3\)"),
        (lambda w, ls: (w, ls[..., :-1]), "ls_raw must be"),
    ],
)
def test_pack_params_rejects_shape_mismatch(mutate: Mutator, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pack_params(*mutate(*_raw_pair()))


# ---------------------------------------------------------------------------
# The frontier: maps and packing
# ---------------------------------------------------------------------------


def _operands(
    *,
    bsz: int = 2,
    heads: int = 3,
    seqlen: int = 5,
    dtype: torch.dtype = torch.float64,
    bias: float = 1.0,
    strided: bool = False,
    seed: int = 0,
) -> Pair:
    """``(params, param_bias)``.

    ``params`` is cut out of a wider row, which is the shipped layout: the mixer
    runs one projection GEMM and hands out views. ``strided`` keeps it as a view;
    otherwise it is compacted. The two hold the same values, so an output difference
    between them is a layout bug.

    ``bias`` scales the drawn bias rows and defaults to unit scale. Zero is also a
    regular additive operating point and is covered above.
    """
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)

    pwidth = heads * PARAM_COLS
    row = rnd(bsz, seqlen, 7 + pwidth + 5)
    params = row[..., 7 : 7 + pwidth]
    if not strided:
        params = params.contiguous()
    pinned = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    return params, rnd(heads, PARAM_COLS).to(pinned) * bias


def _apply(params: Tensor, param_bias: Tensor, *, heads: int = 3) -> ScanParams:
    """:func:`scanprep_ref` at the default bound."""
    return scanprep_ref(params, param_bias, heads=heads, w_max=W_MAX)


def _head_major(params: Tensor, param_bias: Tensor, heads: int) -> Tensor:
    """The map inputs the rows present, as ``(B,H,T,PARAM_COLS)``.

    The rotation columns go through :func:`anchored_rotvec` and the log-scale column
    is the same plain sum.
    """
    rows = params.unflatten(-1, (heads, PARAM_COLS)).permute(0, 2, 1, 3)
    bias = param_bias[:, None, :]
    drive = anchored_rotvec(rows[..., 0:3], bias[..., 0:3])
    return torch.cat((drive, rows[..., 3:] + bias[..., 3:]), dim=-1)


def test_frontier_shapes_dtypes_and_contiguity() -> None:
    params, pb = _operands()
    out = scanprep(params, pb, heads=3, w_max=W_MAX)
    assert isinstance(out, ScanParams)
    assert out.trans.shape == (2, 3, 5, 4)
    assert out.K.shape == (2, 3, 5, 2, 4)
    assert all(t.is_contiguous() for t in out)


def test_frontier_applies_the_bounded_maps_to_the_anchored_row() -> None:
    """The maps are the ones asserted above, applied after the row is anchored.

    A frontier that mapped before adding or dropped the bias would still have every
    shape and dtype right. ``K``
    is the taps of the transition the maps produce, not of the raw row, so it is
    asserted against the packed transition; the hard zero in lane 3 completes the
    packing contract.
    """
    params, pb = _operands(seed=1)
    rows = _head_major(params, pb, heads=3)
    out = _apply(params, pb)
    assert torch.equal(out.trans[..., :3], bounded_rotvec(rows[..., 0:3], W_MAX))
    # One ulp of LS_MAX_MAG, not bitwise. ``torch.sigmoid`` takes a vectorized path on
    # a contiguous input and a scalar one on a strided view, and the two disagree in
    # the last bit; the reference reaches it through a strided column of the row it
    # packs. The claim under test is which map is applied to which row, and a
    # dispatch difference inside the map does not bear on it.
    ls_ulp = LS_MAX_MAG * torch.finfo(torch.float64).eps
    assert float((out.trans[..., 3] - bounded_logscale(rows[..., 3])).abs().max()) <= (
        ls_ulp
    )
    want = foh_taps(out.trans[..., :3].contiguous(), out.trans[..., 3])
    torch.testing.assert_close(out.K[..., :3], want, rtol=2e-15, atol=2e-15)
    assert torch.equal(out.K[..., 3], torch.zeros_like(out.K[..., 3]))


def test_frontier_reads_a_projection_slice_without_repacking_it() -> None:
    """Bitwise equality against the compact operand. The shipped operand is a view
    of one projection output, so an implementation that only handled contiguous
    input would pass every other test in this file."""
    params, pb = _operands(strided=True, seed=3)
    assert params.stride(-1) == 1 and not params.is_contiguous()
    strided = _apply(params, pb)
    compact = _apply(params.contiguous(), pb)
    for got, want in zip(strided, compact, strict=True):
        assert torch.equal(got, want)


# ``pinned_dtype`` reads float64 or nothing, so the two 16-bit formats take one
# path. bfloat16 is the case where the pinned math is an upcast of the activation
# dtype; float32 is the case where it is the activation dtype itself and no cast
# happens. float64, the third branch, is the test below.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_frontier_pins_the_transition_to_float32(dtype: torch.dtype) -> None:
    params, pb = _operands(dtype=dtype)
    out = _apply(params, pb)
    assert out.trans.dtype is torch.float32
    assert out.K.dtype is torch.float32


def test_frontier_keeps_float64() -> None:
    params, pb = _operands()
    out = _apply(params, pb)
    assert out.trans.dtype is torch.float64
    assert out.K.dtype is torch.float64


def test_frontier_runs_under_autocast() -> None:
    """I4 against autocast: no ``custom_fwd``, so the transition stays float32 even
    when the surrounding region is bfloat16."""
    params, pb = _operands(dtype=torch.float32)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = _apply(params, pb)
    assert out.trans.dtype is torch.float32
    assert out.K.dtype is torch.float32


def test_frontier_invariants_hold_on_the_packed_tensors() -> None:
    params, pb = _operands(seqlen=64, bias=1e6, seed=4)
    out = _apply(params * 1e6, pb)
    assert bool((out.trans[..., 3] <= 0.0).all())
    assert bool((out.trans[..., 3] >= -LS_MAX_MAG).all())
    assert bool((out.trans[..., :3].norm(dim=-1) <= _ball_bound(torch.float64)).all())
    assert bool(torch.isfinite(out.trans).all())
    assert bool(torch.isfinite(out.K).all())


def test_frontier_gradcheck() -> None:
    """float64 gradcheck on both inputs, ``param_bias`` included."""
    params, pb = _operands(bsz=1, heads=1, seqlen=3, seed=5)
    leaves = tuple(t.detach().clone().requires_grad_() for t in (params, pb))

    def fn(p: Tensor, pbias: Tensor) -> tuple[Tensor, ...]:
        return tuple(_apply(p, pbias, heads=1))

    assert torch.autograd.gradcheck(fn, leaves)


def _stride_two(bsz: int, seqlen: int, width: int) -> Tensor:
    """A ``(B,T,width)`` view whose trailing stride is two."""
    row = torch.randn(bsz, seqlen, 2 * width, dtype=torch.float64)
    return row[..., ::2]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda p, pb: (p, pb, 0), ValueError, "heads must be positive"),
        (lambda p, pb: (p[..., :-1], pb, 3), ValueError, "params must be"),
        (lambda p, pb: (p, pb[:-1], 3), ValueError, "param_bias must be"),
        (
            lambda p, pb: (_stride_two(2, 5, 3 * PARAM_COLS), pb, 3),
            ValueError,
            "params must have unit stride",
        ),
        (lambda p, pb: (p.to(torch.int64), pb, 3), TypeError, "supported"),
        (
            lambda p, pb: (p, pb.to(torch.bfloat16), 3),
            TypeError,
            "float32-pinned",
        ),
    ],
)
def test_frontier_rejects_a_broken_operand_set(
    mutate: Callable[[Tensor, Tensor], tuple[Tensor, Tensor, int]],
    error: type[Exception],
    match: str,
) -> None:
    """Every raise in the operand contract. A frontier that trusted its operands
    would mis-index a projection whose columns moved, silently."""
    params, pb, heads = mutate(*_operands(seed=6))
    with pytest.raises(error, match=match):
        scanprep_ref(params, pb, heads=heads, w_max=W_MAX)


def test_frontier_rejects_illegal_w_max() -> None:
    params, pb = _operands()
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_ref(params, pb, heads=3, w_max=math.pi)


# ---------------------------------------------------------------------------
# The reference pullback
# ---------------------------------------------------------------------------


def _cotangents(
    bsz: int = 2,
    heads: int = 3,
    seqlen: int = 5,
    seed: int = 11,
) -> Pair:
    """``(dtrans, dK)`` at the packed layouts."""
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    return rnd(bsz, heads, seqlen, 4), rnd(bsz, heads, seqlen, 2, 4)


def test_reference_backward_matches_autograd_through_the_public_operator() -> None:
    """The dispatched reference path is the same gradient autograd produces.

    :func:`scanprep` routes the backward through the registry rather than through
    autograd's own graph. The two arms are equal only if the saved set is right.
    """
    params, pb = _operands(seed=8)
    cots = _cotangents()

    leaves = tuple(t.detach().clone().requires_grad_(True) for t in (params, pb))
    out = scanprep(*leaves, heads=3, w_max=W_MAX)
    total = out.trans.new_zeros(())
    for value, cot in zip(out, cots, strict=True):
        total = total + (value * cot).sum()
    total.backward()

    got = scanprep_bwd_ref(*cots, params, pb, heads=3, w_max=W_MAX)
    names = ("dparams", "dparam_bias")
    want = (got.dparams, got.dparam_bias)
    for leaf, ref, name in zip(leaves, want, names, strict=True):
        assert leaf.grad is not None
        torch.testing.assert_close(leaf.grad, ref, msg=name)


def test_reference_backward_ignores_the_cotangent_of_the_padding_lane() -> None:
    """Lane 3 of each tap is a constant zero, so its cotangent is the cotangent of
    nothing. A pullback that read it would leak into ``dparams``."""
    params, pb = _operands(seed=9)
    dtrans, dK = _cotangents()
    loud = dK.clone()
    loud[..., 3] = 1e6
    quiet = scanprep_bwd_ref(dtrans, dK, params, pb, heads=3, w_max=W_MAX)
    got = scanprep_bwd_ref(dtrans, loud, params, pb, heads=3, w_max=W_MAX)
    assert torch.equal(got.dparams, quiet.dparams)
    assert torch.equal(got.dparam_bias, quiet.dparam_bias)
    assert float(quiet.dparams.abs().max()) > 0.0


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d, k: (d[..., :3], k), "dtrans must be"),
        (lambda d, k: (d, k[..., :3]), "dK must be"),
    ],
)
def test_reference_backward_rejects_a_mismatched_cotangent(
    mutate: Callable[[Tensor, Tensor], Pair],
    match: str,
) -> None:
    """Each cotangent is one of the two packed layouts. A narrower one is a bug in
    the caller."""
    params, pb = _operands()
    cots = _cotangents()
    with pytest.raises(ValueError, match=match):
        scanprep_bwd_ref(*mutate(*cots), params, pb, heads=3, w_max=W_MAX)
