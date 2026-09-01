"""Adversarial suite.

Saturated decay, ``w = 0`` exactly, ``|w|`` at the bound, a bound just below pi,
extreme operand magnitudes, mixed dtypes, and the zero-padded ragged tail. No
output may contain a NaN or an infinity, and the chunked path must stay in
agreement with the sequential path everywhere.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from slinoss.config import MAX_CHUNK, MIN_CHUNK
from slinoss.ops.scanprep import LS_MAX_MAG
from slinoss.ops.so3ssd import SO3SSDResult, so3ssd_ref, so3ssm
from tests.conftest import ScanInputs, assert_max_rel, make_inputs, max_err

# Worst measured under the extremes below: 1.8e-15.
PARITY_REL = 1e-13
TINY: dict[str, Any] = {"bsz": 1, "heads": 1, "rows": 16, "lanes": 16}


def _finite(out: SO3SSDResult, label: str) -> None:
    for name, tensor in (
        ("y", out.y),
        ("state", out.state),
        ("b_last", out.b_last),
        ("u_last", out.u_last),
    ):
        assert bool(torch.isfinite(tensor.double()).all()), f"{label}: {name}"


def _both(inp: ScanInputs, chunk: int) -> tuple[SO3SSDResult, SO3SSDResult]:
    return so3ssm(*inp.args(), **inp.kw()), so3ssd_ref(*inp.args(), chunk, **inp.kw())


def _check_parity(inp: ScanInputs, chunk: int, label: str) -> None:
    want, got = _both(inp, chunk)
    _finite(want, f"{label}/sequential")
    _finite(got, f"{label}/chunked")
    assert_max_rel(got.y, want.y, PARITY_REL, f"{label} y")
    assert_max_rel(got.state, want.state, PARITY_REL, f"{label} state")


# ---------------------------------------------------------------------------
# The decay extremes
# ---------------------------------------------------------------------------


# The bias no longer sets the regime: ``ls`` saturates at ``-LS_MAX_MAG``, so the
# chunk-local decay ``exp(2*(lp_t - lp_s))`` bottoms out at ``exp(-2*LS_MAX_MAG*L)``
# and the chunk length is the only thing that moves it. One case per end of the legal
# range:
#
# - L = 16, where the floor puts the most distant pair at 3.4e-4 and every pair in
#   the chunk is significant, which is the regime every kernel test runs in;
# - L = 128, the longest chunk, where it reaches 1.6e-28 -- the closest the operator
#   can come to underflow, and still four orders inside a normal float32.
#
# So the arm the old bias ladder was reaching for does not exist any more, and the
# claim is the stronger one: underflow is unreachable rather than survived. The
# unbiased case is the file's control and runs as ``test_rotation_magnitude_generic``.
@pytest.mark.parametrize("chunk", [MIN_CHUNK, MAX_CHUNK])
def test_saturated_decay(chunk: int) -> None:
    """The floor bounds the chunk-local decay below, so underflow cannot meet
    overflow at all. Forming the difference before the exponential is still what
    keeps them apart: the factored form would reach ``exp(-2*LS_MAX_MAG*T)`` over the
    whole sequence, which no bound on ``ls`` can save."""
    inp = make_inputs(seqlen=2 * chunk, seed=67, ls_bias=400.0, **TINY)
    ls = inp.trans[..., 3]
    assert float(ls.max()) == -LS_MAX_MAG
    span = math.exp(-2.0 * LS_MAX_MAG * chunk)
    assert span > torch.finfo(torch.float32).tiny
    _check_parity(inp, chunk, f"floored decay, chunk {chunk}")


# -20.0 leaves ``ls`` nonzero, so the prefix still shrinks. -50.0 is the boundary of
# the predicate the body branches on, where the sigmoid has underflowed far enough
# that the decay is exactly one; a more negative bias is interior to it.
@pytest.mark.parametrize("ls_bias", [-20.0, -50.0])
def test_vanishing_decay(ls_bias: float) -> None:
    """``sigmoid`` underflows toward zero, so the transition approaches a pure
    rotation and the prefix stops shrinking. ``LS_MAX_MAG*sigmoid(x) <= exp(x)``
    bounds the residual; six standard deviations covers the raw sample."""
    inp = make_inputs(seqlen=48, seed=71, ls_bias=ls_bias, **TINY)
    ls = inp.trans[..., 3]
    assert float(ls.abs().max()) < math.exp(ls_bias + 6.0)
    if ls_bias <= -50.0:
        assert bool((torch.exp(2.0 * ls) == 1.0).all())
    _check_parity(inp, 16, f"decay bias {ls_bias}")


def test_decay_factors_stay_in_the_unit_interval() -> None:
    for ls_bias in (-400.0, -1.0, 0.0, 1.0, 400.0):
        inp = make_inputs(seqlen=64, seed=73, ls_bias=ls_bias, **TINY)
        prefix = torch.cumsum(inp.trans[..., 3].unflatten(-1, (-1, 16)), dim=-1)
        assert bool((prefix.diff(dim=-1) <= 0.0).all())
        decay = torch.exp(2.0 * (prefix[..., :, None] - prefix[..., None, :]))
        causal = torch.ones(16, 16, dtype=torch.bool).tril()
        assert bool((decay.masked_fill(~causal, 0.0) <= 1.0).all())


# ---------------------------------------------------------------------------
# The rotation extremes
# ---------------------------------------------------------------------------


def test_zero_rotation_exactly() -> None:
    """``w = 0`` is the boundary of the small-``|w|`` regime.

    The quaternion series sits on its constant term and every odd term vanishes. A
    nonzero but tiny ``w_scale`` is interior to this: it moves the series by less
    than the parity tolerance.
    """
    inp = make_inputs(seqlen=40, seed=79, w_scale=0.0, **TINY)
    assert float(inp.trans[..., :3].abs().max()) == 0.0
    _check_parity(inp, 16, "w = 0")


def test_rotation_magnitude_generic() -> None:
    """Unsaturated ``|w|`` at no log-scale bias, the control for every extreme."""
    inp = make_inputs(seqlen=40, seed=83, w_scale=1.0, **TINY)
    _check_parity(inp, 16, "w_scale 1.0")


# One case per distinct behaviour of the quaternion series at the bound, which
# ``w_scale`` saturates so that ``|w| == w_max`` at every token:
#
# - 1e-6, a near-identity rotation, where the vector part is the only departure
#   from the constant term;
# - 3.0, the shipped bound;
# - 3.1415926, five parts in 1e8 below pi, where the scalar part of the quaternion
#   cancels to zero.
#
# Every arm asserts the norm below pi, which is what sizes the consumer.
@pytest.mark.parametrize("w_max", [1e-6, 3.0, 3.1415926])
def test_rotation_at_the_bound(w_max: float) -> None:
    """``w_scale`` saturates the parameter map, so ``|w|`` sits at ``w_max`` to
    within the rounding of the final multiply. Near pi the scalar part of the
    quaternion cancels to zero; the series loses relative accuracy there but not
    absolute, which is what a unit quaternion needs."""
    inp = make_inputs(seqlen=40, seed=89, w_scale=1e12, w_max=w_max, **TINY)
    norm = inp.trans[..., :3].norm(dim=-1)
    slack = 3.0 * torch.finfo(torch.float64).eps
    assert float(norm.min()) == pytest.approx(w_max, rel=slack)
    assert float(norm.max()) <= w_max * (1.0 + slack)
    assert float(norm.max()) < math.pi
    _check_parity(inp, 16, f"w_max {w_max}")


# ---------------------------------------------------------------------------
# Zero padding of a ragged tail is an exact no-op
# ---------------------------------------------------------------------------


# (T, L). Both are ragged by construction, which is the premise: T = 33 leaves one
# real token in the last of three chunks, the largest pad the tail can carry, and
# T = 20 at L = 32 is a single chunk shorter than L.
@pytest.mark.parametrize(("seqlen", "chunk"), [(33, 16), (20, 32)])
def test_zero_padding_is_a_no_op(seqlen: int, chunk: int) -> None:
    """A padded token has ``w = 0`` and ``ls = 0``, so its transition is the
    identity, and zero taps kill its forcing. Padding to a whole number of
    chunks by hand must reproduce the ragged call bit for bit."""
    inp = make_inputs(seqlen=seqlen, seed=101, **TINY)
    tail = (-seqlen) % chunk

    def padded(t: Tensor) -> Tensor:
        shape = (*t.shape[:2], tail, *t.shape[3:])
        zeros = torch.zeros(shape, dtype=t.dtype, device=t.device)
        return torch.cat([t, zeros], dim=2).contiguous()

    u, trans, k, b, c = (padded(t) for t in inp.args())
    ragged = so3ssd_ref(*inp.args(), chunk, **inp.kw())
    whole = so3ssd_ref(u, trans, k, b, c, chunk, **inp.kw())
    assert max_err(whole.y[:, :, :seqlen], ragged.y) == 0.0
    assert max_err(whole.state, ragged.state) == 0.0


# ---------------------------------------------------------------------------
# Operand magnitudes
# ---------------------------------------------------------------------------


# The two ends of the range the cube of the scale still fits in float64: ``y``
# carries ``scale**3``, so 1e-30 puts it at 1e-90 and 1e30 at 1e90. Anything
# between is interior to both.
@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_operand_magnitude_sweep(scale: float) -> None:
    """The operator is linear in ``U``, ``B``, ``C``, and ``z0``, so a uniform
    rescale must come straight back out."""
    inp = make_inputs(seqlen=40, seed=103, **TINY)
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    scaled = inp._replace(
        U=(inp.U * scale).contiguous(),
        B=(inp.B * scale).contiguous(),
        C=(inp.C * scale).contiguous(),
        z0=(inp.z0 * scale * scale).contiguous(),
        b_prev=(inp.b_prev * scale).contiguous(),
        u_prev=(inp.u_prev * scale).contiguous(),
    )
    base = so3ssd_ref(*inp.args(), 16, **inp.kw())
    got = so3ssd_ref(*scaled.args(), 16, **scaled.kw())
    _finite(got, f"scale {scale}")
    assert_max_rel(got.y / scale**3, base.y, PARITY_REL, f"scale {scale} y")
    assert_max_rel(got.state / scale**2, base.state, PARITY_REL, f"scale {scale} state")


def test_alternating_extreme_magnitudes() -> None:
    inp = make_inputs(seqlen=64, seed=107, **TINY)
    ramp = torch.where(
        torch.arange(64, dtype=torch.float64) % 2 == 0,
        torch.full((64,), 1e12, dtype=torch.float64),
        torch.full((64,), 1e-12, dtype=torch.float64),
    )
    scaled = inp._replace(
        B=(inp.B * ramp[None, None, :, None]).contiguous(),
        C=(inp.C * ramp.flip(0)[None, None, :, None]).contiguous(),
    )
    _check_parity(scaled, 16, "alternating magnitudes")


# ---------------------------------------------------------------------------
# Low precision
# ---------------------------------------------------------------------------


def _downcast(inp: ScanInputs, pinned: torch.dtype, low: torch.dtype) -> ScanInputs:
    def cast(t: Tensor | None, dtype: torch.dtype) -> Tensor | None:
        return None if t is None else t.to(dtype).contiguous()

    return ScanInputs(
        U=inp.U.to(low).contiguous(),
        trans=inp.trans.to(pinned).contiguous(),
        K=inp.K.to(pinned).contiguous(),
        B=inp.B.to(low).contiguous(),
        C=inp.C.to(low).contiguous(),
        z0=cast(inp.z0, pinned),
        b_prev=cast(inp.b_prev, low),
        u_prev=cast(inp.u_prev, low),
    )


# U, B, and C carry the rounding; the accumulation is float32. A bfloat16 operand
# is exact to 2^-8, and y is a product of three of them, so a few percent of
# relative error is the arithmetic and not a defect. float16 keeps 11 bits.
# Measured: 5.1e-07, 4.5e-04, 4.1e-03.
LOW_PRECISION_ENVELOPE: tuple[tuple[torch.dtype, float], ...] = (
    (torch.float32, 2e-6),
    (torch.float16, 1e-3),
    (torch.bfloat16, 2e-2),
)


@pytest.mark.parametrize(("low", "bound"), LOW_PRECISION_ENVELOPE)
def test_low_precision_envelope(low: torch.dtype, bound: float) -> None:
    inp = make_inputs(seqlen=64, seed=109, rows=16, lanes=16, bsz=2, heads=2)
    oracle = so3ssd_ref(*inp.args(), 16, **inp.kw())
    cast = _downcast(inp, torch.float32, low)
    got = so3ssd_ref(*cast.args(), 16, **cast.kw())
    _finite(got, str(low))
    assert got.y.dtype is low
    assert got.state.dtype is torch.float32
    assert_max_rel(got.y.double(), oracle.y, bound, f"{low} y")


def test_mixed_operand_dtypes() -> None:
    inp = make_inputs(seqlen=40, seed=113, dtype=torch.float32, **TINY)
    mixed = inp._replace(
        U=inp.U.to(torch.bfloat16).contiguous(),
        B=inp.B.to(torch.float16).contiguous(),
        C=inp.C.contiguous(),
    )
    for out in _both(mixed, 16):
        _finite(out, "mixed dtypes")
        assert out.y.dtype is torch.bfloat16
        assert out.state.dtype is torch.float32


def test_low_precision_survives_saturated_decay() -> None:
    """float16 only: finiteness is a range claim and float16 has the narrow range.

    bfloat16 carries float32's exponent, so any magnitude that overflows it
    overflows float16 first, and an underflow is a zero either way. The bfloat16
    rounding is covered by the envelope above.
    """
    low = torch.float16
    inp = make_inputs(seqlen=64, seed=127, ls_bias=30.0, **TINY)
    cast = _downcast(inp, torch.float32, low)
    _finite(so3ssd_ref(*cast.args(), 16, **cast.kw()), f"{low} saturated")
    _finite(so3ssm(*cast.args(), **cast.kw()), f"{low} saturated sequential")


# ---------------------------------------------------------------------------
# Minimum geometry
# ---------------------------------------------------------------------------


# The ends of the legal chunk range, written against the constants so both stay
# covered if either is retuned. At T = 1 the chunk length is the pad length and
# nothing else, so the arms between the ends select nothing.
@pytest.mark.parametrize("chunk", [MIN_CHUNK, MAX_CHUNK])
def test_smallest_legal_shape(chunk: int) -> None:
    inp = make_inputs(bsz=1, heads=1, seqlen=1, rows=16, lanes=16, seed=131)
    _check_parity(inp, chunk, f"minimum at L={chunk}")


def test_chunk_larger_than_the_sequence() -> None:
    """A chunk past the legal maximum, over a sequence of more than one token."""
    inp = make_inputs(seqlen=5, seed=137, **TINY)
    _check_parity(inp, 256, "chunk beyond T")
