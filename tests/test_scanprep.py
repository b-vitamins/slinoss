"""Bounded parameter maps. The numerical invariants live here, so they are
asserted here rather than guarded downstream."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from slinoss.config import STATE_MULTIPLE
from slinoss.ops.scanprep import (
    PARAM_COLS,
    ScanParams,
    bounded_logscale,
    bounded_rotvec,
    pack_params,
    scanprep,
    scanprep_bwd_ref,
    scanprep_ref,
)

Triple = tuple[Tensor, Tensor, Tensor]
Mutator = Callable[[Tensor, Tensor, Tensor], Triple]
Quad = tuple[Tensor, Tensor, Tensor, Tensor]

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


# Both wide dtypes: float32 is what the pinned transition ships in and float64 is
# the oracle width. No kernel clamps ``ls``, so the bound is asserted in each
# format the map is evaluated in rather than in one and assumed in the other.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logscale_is_non_positive(dtype: torch.dtype) -> None:
    ls = bounded_logscale(_raws(dtype))
    assert bool((ls <= 0.0).all())
    assert bool(torch.isfinite(ls).all())


def test_logscale_is_strictly_negative_with_a_positive_decay() -> None:
    """Both strict claims, on the moderate range, in both wide dtypes.

    I1 admits underflow: at raw = 1e4 the decay is exp(-2e4), which is zero in
    every float format, and zero decay is the correct limit. So the closed interval
    is the invariant, asserted over the extremes above, and the open one holds only
    while softplus is nonzero.
    """
    for dtype in (torch.float32, torch.float64):
        ls = bounded_logscale(torch.linspace(-40.0, 40.0, 201, dtype=dtype))
        decay = torch.exp(2.0 * ls)
        assert bool((ls < 0.0).all())
        assert bool((decay > 0.0).all())
        assert bool((decay <= 1.0).all())


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


# (dtype, scale). The map is one branchless expression, so the raw magnitude
# selects a regime of the ratio ``|raw| / sqrt(1 + |raw|^2)`` rather than a path.
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
    """Magnitude and direction both, against the closed form.

    The reference is a positive scalar times ``raw``, so agreement pins the
    direction too: a per-component ratio in place of the shared norm fails here.
    """
    gen = torch.Generator().manual_seed(4)
    raw = torch.randn(128, 3, generator=gen, dtype=torch.float64) * 7.0
    want = W_MAX * raw / torch.sqrt(1.0 + raw.pow(2).sum(-1, keepdim=True))
    assert float((bounded_rotvec(raw, W_MAX) - want).abs().max()) < 1e-15


def test_rotvec_at_zero_is_zero_exactly() -> None:
    raw = torch.zeros(4, 3, dtype=torch.float64)
    assert torch.equal(bounded_rotvec(raw, W_MAX), raw)


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
# Column packing
# ---------------------------------------------------------------------------


def _raw_triple(
    bsz: int = 2,
    heads: int = 3,
    seqlen: int = 5,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> Triple:
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)

    return (
        rnd(bsz, heads, seqlen, 3),
        rnd(bsz, heads, seqlen),
        rnd(bsz, heads, seqlen, 2, 3),
    )


def test_pack_params_lays_out_the_projection_column_order() -> None:
    """The column order is the operator's contract with the projection, and every
    kernel indexes it by hand, so it is asserted rather than assumed."""
    w_raw, ls_raw, tap_raw = _raw_triple()
    row = pack_params(w_raw, ls_raw, tap_raw)
    assert row.shape == (2, 5, 3 * PARAM_COLS)
    assert row.is_contiguous()
    head_major = row.unflatten(-1, (3, PARAM_COLS)).permute(0, 2, 1, 3)
    assert torch.equal(head_major[..., 0:3], w_raw)
    assert torch.equal(head_major[..., 3], ls_raw)
    assert torch.equal(head_major[..., 4:].unflatten(-1, (2, 3)), tap_raw)


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
def test_pack_params_rejects_shape_mismatch(mutate: Mutator, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pack_params(*mutate(*_raw_triple()))


# ---------------------------------------------------------------------------
# The frontier: maps, packing, and the B/C permute
# ---------------------------------------------------------------------------


def _operands(
    *,
    bsz: int = 2,
    heads: int = 3,
    seqlen: int = 5,
    groups: int = 1,
    state_dim: int = STATE_MULTIPLE,
    dtype: torch.dtype = torch.float64,
    bias: float = 0.0,
    strided: bool = False,
    seed: int = 0,
) -> Triple:
    """``(params, bc, param_bias)``.

    Both operands are cut out of one wider row, which is the shipped layout: the
    mixer runs one projection GEMM and hands out views. ``strided`` keeps them as
    views; otherwise they are compacted. The two hold the same values, so an output
    difference between them is a layout bug.
    """
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)

    pwidth = heads * PARAM_COLS
    bwidth = 2 * groups * state_dim
    row = rnd(bsz, seqlen, 7 + pwidth + bwidth + 5)
    params = row[..., 7 : 7 + pwidth]
    bc = row[..., 7 + pwidth : 7 + pwidth + bwidth]
    if not strided:
        params, bc = params.contiguous(), bc.contiguous()
    pinned = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    return params, bc, rnd(heads, PARAM_COLS).to(pinned) * bias


def _apply(
    params: Tensor,
    bc: Tensor,
    param_bias: Tensor,
    *,
    heads: int = 3,
    state_dim: int = STATE_MULTIPLE,
) -> ScanParams:
    """:func:`scanprep_ref` at the default bound."""
    return scanprep_ref(
        params, bc, param_bias, heads=heads, state_dim=state_dim, w_max=W_MAX
    )


def _head_major(params: Tensor, param_bias: Tensor, heads: int) -> Tensor:
    """The biased parameter rows as ``(B,H,T,PARAM_COLS)``."""
    rows = params.unflatten(-1, (heads, PARAM_COLS)) + param_bias
    return rows.permute(0, 2, 1, 3)


def test_frontier_shapes_dtypes_and_contiguity() -> None:
    params, bc, pb = _operands(groups=3)
    out = scanprep(params, bc, pb, heads=3, state_dim=STATE_MULTIPLE, w_max=W_MAX)
    assert isinstance(out, ScanParams)
    assert out.trans.shape == (2, 3, 5, 4)
    assert out.K.shape == (2, 3, 5, 2, 4)
    assert out.B.shape == (2, 3, 5, STATE_MULTIPLE)
    assert out.C.shape == out.B.shape
    assert all(t.is_contiguous() for t in out)


def test_frontier_applies_the_bounded_maps_to_the_biased_row() -> None:
    """The maps are the ones asserted above, applied after the bias.

    A frontier that biased after the map, or dropped the bias, would still have
    every shape and dtype right. The tap columns and the hard zero in lane 3 are
    one packing contract, so they are asserted together.
    """
    params, bc, pb = _operands(bias=1.0, seed=1)
    rows = _head_major(params, pb, heads=3)
    out = _apply(params, bc, pb)
    assert torch.equal(out.trans[..., :3], bounded_rotvec(rows[..., 0:3], W_MAX))
    assert torch.equal(out.trans[..., 3], bounded_logscale(rows[..., 3]))
    assert torch.equal(out.K[..., :3], rows[..., 4:].unflatten(-1, (2, 3)))
    assert torch.equal(out.K[..., 3], torch.zeros_like(out.K[..., 3]))


@pytest.mark.parametrize(("groups", "state_dim"), [(1, STATE_MULTIPLE), (3, 96)])
def test_frontier_permutes_bc_group_major(groups: int, state_dim: int) -> None:
    """``bc`` is all of ``B`` then all of ``C``, each group-major. Swept at
    ``G = 1`` and ``G = H``; the state width does not interact with the grouping,
    so it moves once alongside it rather than crossing it."""
    params, bc, pb = _operands(groups=groups, state_dim=state_dim, seed=2)
    out = _apply(params, bc, pb, state_dim=state_dim)
    half = groups * state_dim
    for g in range(groups):
        lo = g * state_dim
        assert torch.equal(out.B[:, g], bc[..., lo : lo + state_dim])
        assert torch.equal(out.C[:, g], bc[..., half + lo : half + lo + state_dim])


def test_frontier_reads_a_projection_slice_without_repacking_it() -> None:
    """Bitwise equality against the compact operands. The shipped operands are
    views of one projection output, so an implementation that only handled
    contiguous input would pass every other test in this file."""
    params, bc, pb = _operands(groups=3, bias=1.0, strided=True, seed=3)
    assert params.stride(-1) == 1 and not params.is_contiguous()
    assert bc.stride(-1) == 1 and not bc.is_contiguous()
    strided = _apply(params, bc, pb)
    compact = _apply(params.contiguous(), bc.contiguous(), pb)
    for got, want in zip(strided, compact, strict=True):
        assert torch.equal(got, want)


# ``pinned_dtype`` reads float64 or nothing, so the two 16-bit formats take one
# path. bfloat16 is the case where the pinned math is an upcast of the activation
# dtype; float32 is the case where it is the activation dtype itself and no cast
# happens. float64, the third branch, is the test below.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_frontier_pins_the_transition_to_float32(dtype: torch.dtype) -> None:
    params, bc, pb = _operands(dtype=dtype)
    out = _apply(params, bc, pb)
    assert out.trans.dtype is torch.float32
    assert out.K.dtype is torch.float32
    assert out.B.dtype is dtype
    assert out.C.dtype is dtype


def test_frontier_keeps_float64() -> None:
    params, bc, pb = _operands()
    out = _apply(params, bc, pb)
    assert out.trans.dtype is torch.float64
    assert out.K.dtype is torch.float64


def test_frontier_runs_under_autocast() -> None:
    """I4 against autocast: no ``custom_fwd``, so the transition stays float32 even
    when the surrounding region is bfloat16."""
    params, bc, pb = _operands(dtype=torch.float32)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = _apply(params, bc, pb)
    assert out.trans.dtype is torch.float32
    assert out.K.dtype is torch.float32


def test_frontier_invariants_hold_on_the_packed_tensors() -> None:
    params, bc, pb = _operands(seqlen=64, bias=1e6, seed=4)
    out = _apply(params * 1e6, bc, pb)
    assert bool((out.trans[..., 3] <= 0.0).all())
    assert bool((out.trans[..., :3].norm(dim=-1) <= _ball_bound(torch.float64)).all())
    assert bool(torch.isfinite(out.trans).all())
    assert bool(torch.isfinite(out.K).all())


def test_frontier_gradcheck() -> None:
    """float64 gradcheck on all three inputs, ``param_bias`` included."""
    params, bc, pb = _operands(bsz=1, heads=1, seqlen=3, seed=5)
    leaves = tuple(t.detach().clone().requires_grad_() for t in (params, bc, pb))

    def fn(p: Tensor, b: Tensor, pbias: Tensor) -> tuple[Tensor, ...]:
        return tuple(_apply(p, b, pbias, heads=1))

    assert torch.autograd.gradcheck(fn, leaves)


def _stride_two(bsz: int, seqlen: int, width: int) -> Tensor:
    """A ``(B,T,width)`` view whose trailing stride is two."""
    row = torch.randn(bsz, seqlen, 2 * width, dtype=torch.float64)
    return row[..., ::2]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda p, b, pb: (p, b, pb, 0), ValueError, "heads must be positive"),
        (lambda p, b, pb: (p[..., :-1], b, pb, 3), ValueError, "params must be"),
        (lambda p, b, pb: (p[:, :-1], b, pb, 3), ValueError, "bc must be"),
        (lambda p, b, pb: (p, b[..., None], pb, 3), ValueError, "bc must be"),
        (lambda p, b, pb: (p, b, pb[:-1], 3), ValueError, "param_bias must be"),
        (lambda p, b, pb: (p, b[..., :-1], pb, 3), ValueError, r"2\*G\*"),
        (
            lambda p, b, pb: (_stride_two(2, 5, 3 * PARAM_COLS), b, pb, 3),
            ValueError,
            "params must have unit stride",
        ),
        (
            lambda p, b, pb: (p, _stride_two(2, 5, 2 * STATE_MULTIPLE), pb, 3),
            ValueError,
            "bc must have unit stride",
        ),
        (lambda p, b, pb: (p.to(torch.int64), b, pb, 3), TypeError, "supported"),
        (
            lambda p, b, pb: (p, b.to(torch.float32), pb, 3),
            TypeError,
            "one activation dtype per call",
        ),
        (
            lambda p, b, pb: (p, b, pb.to(torch.bfloat16), 3),
            TypeError,
            "float32-pinned",
        ),
    ],
)
def test_frontier_rejects_a_broken_operand_set(
    mutate: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, int]],
    error: type[Exception],
    match: str,
) -> None:
    """Every raise in the operand contract. A frontier that trusted its operands
    would mis-index a projection whose columns moved, silently."""
    params, bc, pb, heads = mutate(*_operands(seed=6))
    with pytest.raises(error, match=match):
        scanprep_ref(params, bc, pb, heads=heads, state_dim=STATE_MULTIPLE, w_max=W_MAX)


def test_frontier_rejects_a_grouping_the_head_count_cannot_hold() -> None:
    """``G`` comes off ``bc``, so a caller cannot claim one grouping and hand over
    another; a group must still hold a whole number of heads."""
    params, bc, pb = _operands(heads=3, groups=2, seed=7)
    with pytest.raises(ValueError, match="does not divide heads"):
        _apply(params, bc, pb)


@pytest.mark.parametrize("state_dim", [0, 1, STATE_MULTIPLE - 1, STATE_MULTIPLE + 1])
def test_frontier_rejects_an_illegal_state_width(state_dim: int) -> None:
    """``3N`` with ``N`` a multiple of 16 is what makes every downstream
    contraction MMA-k friendly with no padding."""
    params, bc, pb = _operands()
    with pytest.raises(ValueError, match="state_dim is 3N"):
        _apply(params, bc, pb, state_dim=state_dim)


def test_frontier_rejects_illegal_w_max() -> None:
    params, bc, pb = _operands()
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_ref(params, bc, pb, heads=3, state_dim=STATE_MULTIPLE, w_max=math.pi)


# ---------------------------------------------------------------------------
# The reference pullback
# ---------------------------------------------------------------------------


def _cotangents(
    bsz: int = 2,
    heads: int = 3,
    seqlen: int = 5,
    groups: int = 3,
    state_dim: int = STATE_MULTIPLE,
    seed: int = 11,
) -> Quad:
    """``(dtrans, dK, dB, dC)`` at the packed layouts."""
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    return (
        rnd(bsz, heads, seqlen, 4),
        rnd(bsz, heads, seqlen, 2, 4),
        rnd(bsz, groups, seqlen, state_dim),
        rnd(bsz, groups, seqlen, state_dim),
    )


def test_reference_backward_matches_autograd_through_the_public_operator() -> None:
    """The dispatched reference path is the same gradient autograd produces.

    :func:`scanprep` routes the backward through the registry rather than through
    autograd's own graph, and the registry's backward signature omits ``bc``
    because the permute is linear. The two arms are equal only if that omission is
    sound and the saved set is right.
    """
    params, bc, pb = _operands(groups=3, bias=1.0, seed=8)
    cots = _cotangents()

    leaves = tuple(t.detach().clone().requires_grad_(True) for t in (params, bc, pb))
    out = scanprep(*leaves, heads=3, state_dim=STATE_MULTIPLE, w_max=W_MAX)
    total = out.trans.new_zeros(())
    for value, cot in zip(out, cots, strict=True):
        total = total + (value * cot).sum()
    total.backward()

    got = scanprep_bwd_ref(
        *cots, params, pb, heads=3, state_dim=STATE_MULTIPLE, w_max=W_MAX
    )
    names = ("dparams", "dbc", "dparam_bias")
    want = (got.dparams, got.dbc, got.dparam_bias)
    for leaf, ref, name in zip(leaves, want, names, strict=True):
        assert leaf.grad is not None
        torch.testing.assert_close(leaf.grad, ref, msg=name)


def test_reference_backward_ignores_the_cotangent_of_the_padding_lane() -> None:
    """Lane 3 of each tap is a constant zero, so its cotangent is the cotangent of
    nothing. A pullback that read it would leak into ``dparams``."""
    params, _, pb = _operands(bias=1.0, seed=9)
    dtrans, dK, dB, dC = _cotangents(groups=1)
    loud = dK.clone()
    loud[..., 3] = 1e6
    quiet = scanprep_bwd_ref(
        dtrans, dK, dB, dC, params, pb, heads=3, state_dim=STATE_MULTIPLE, w_max=W_MAX
    )
    got = scanprep_bwd_ref(
        dtrans, loud, dB, dC, params, pb, heads=3, state_dim=STATE_MULTIPLE, w_max=W_MAX
    )
    assert torch.equal(got.dparams, quiet.dparams)
    assert torch.equal(got.dparam_bias, quiet.dparam_bias)
    assert float(quiet.dparams.abs().max()) > 0.0


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d, k, b, c: (d[..., :3], k, b, c), "dtrans must be"),
        (lambda d, k, b, c: (d, k[..., :3], b, c), "dK must be"),
        (lambda d, k, b, c: (d, k, b[..., :-1], c), "dB must be"),
        (lambda d, k, b, c: (d, k, b, c[..., :-1]), "dC must be"),
        (
            lambda d, k, b, c: (
                d,
                k,
                b.expand(2, 2, 5, STATE_MULTIPLE),
                c.expand(2, 2, 5, STATE_MULTIPLE),
            ),
            "with G dividing heads",
        ),
    ],
)
def test_reference_backward_rejects_a_mismatched_cotangent(
    mutate: Callable[[Tensor, Tensor, Tensor, Tensor], Quad],
    match: str,
) -> None:
    """Each cotangent is one of the four packed layouts. A narrower one, or a
    grouping the head count cannot hold, is a bug in the caller."""
    params, _, pb = _operands(groups=1)
    cots = _cotangents(groups=1)
    with pytest.raises(ValueError, match=match):
        scanprep_bwd_ref(
            *mutate(*cots),
            params,
            pb,
            heads=3,
            state_dim=STATE_MULTIPLE,
            w_max=W_MAX,
        )
