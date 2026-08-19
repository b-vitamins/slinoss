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


@pytest.mark.parametrize("ls_bias", [0.0, 5.0, 20.0, 50.0, 400.0])
def test_saturated_decay(ls_bias: float) -> None:
    """``exp(2*(lp_t - lp_s))`` underflows to zero and stays there. Forming the
    difference before the exponential is what keeps underflow from meeting
    overflow."""
    inp = make_inputs(seqlen=48, seed=67, ls_bias=ls_bias, **TINY)
    _check_parity(inp, 16, f"decay bias {ls_bias}")


@pytest.mark.parametrize("ls_bias", [-20.0, -50.0, -400.0])
def test_vanishing_decay(ls_bias: float) -> None:
    """``softplus`` underflows toward zero, so the transition approaches a pure
    rotation and the prefix stops shrinking. ``softplus(x) <= exp(x)`` bounds the
    residual; six standard deviations covers the raw sample."""
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
    inp = make_inputs(seqlen=40, seed=79, w_scale=0.0, **TINY)
    assert float(inp.trans[..., :3].abs().max()) == 0.0
    _check_parity(inp, 16, "w = 0")


@pytest.mark.parametrize("w_scale", [1e-14, 1e-7, 1.0, 1e7, 1e14])
def test_rotation_magnitude_sweep(w_scale: float) -> None:
    inp = make_inputs(seqlen=40, seed=83, w_scale=w_scale, **TINY)
    _check_parity(inp, 16, f"w_scale {w_scale}")


@pytest.mark.parametrize("w_max", [1e-6, 0.5, 3.0, 3.14, 3.1415926])
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
    _check_parity(inp, 16, f"w_max {w_max}")


def test_rotation_bound_is_below_pi() -> None:
    inp = make_inputs(seqlen=8, seed=97, w_scale=1e12, w_max=3.1415926, **TINY)
    assert float(inp.trans[..., :3].norm(dim=-1).max()) < torch.pi


# ---------------------------------------------------------------------------
# Zero padding of a ragged tail is an exact no-op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("seqlen", "chunk"), [(40, 16), (33, 16), (20, 32), (7, 16)])
def test_zero_padding_is_a_no_op(seqlen: int, chunk: int) -> None:
    """A padded token has ``w = 0`` and ``ls = 0``, so its transition is the
    identity, and zero taps kill its forcing. Padding to a whole number of
    chunks by hand must reproduce the ragged call bit for bit."""
    inp = make_inputs(seqlen=seqlen, seed=101, **TINY)
    tail = (-seqlen) % chunk
    assert tail > 0

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


@pytest.mark.parametrize("scale", [1e-30, 1e-8, 1e8, 1e30])
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


@pytest.mark.parametrize("low", [torch.bfloat16, torch.float16])
def test_low_precision_survives_saturated_decay(low: torch.dtype) -> None:
    inp = make_inputs(seqlen=64, seed=127, ls_bias=30.0, **TINY)
    cast = _downcast(inp, torch.float32, low)
    _finite(so3ssd_ref(*cast.args(), 16, **cast.kw()), f"{low} saturated")
    _finite(so3ssm(*cast.args(), **cast.kw()), f"{low} saturated sequential")


# ---------------------------------------------------------------------------
# Minimum geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [16, 32, 64, 128, 256])
def test_smallest_legal_shape(chunk: int) -> None:
    inp = make_inputs(bsz=1, heads=1, seqlen=1, rows=16, lanes=16, seed=131)
    _check_parity(inp, chunk, f"minimum at L={chunk}")


def test_chunk_larger_than_the_sequence() -> None:
    inp = make_inputs(seqlen=5, seed=137, **TINY)
    _check_parity(inp, 256, "chunk beyond T")
