"""Device-side adjoints of the transition math against float64 autograd.

One probe kernel reads a token's rotation vector, tap, quaternion, and two
cotangents, calls every adjoint helper in
:mod:`slinoss.ops.so3ssd.cute.common`, and writes each result out. The probe
exists because none of these quantities reaches global memory on the shipped
path; the alternative is checking them through the output of the backward
kernels that consume them, which localizes nothing.

Ground truth is float64 autograd through the reference primals. A hand-derived
oracle shares its derivation with the device form, so a derivation error would
pass silently in both.

Inputs are built in float32, then upcast for the oracle, so both paths see the
same bits and every difference is float32 arithmetic. The quaternion is a probe
input rather than a device exponential of the same ``w``: feeding it keeps the
exponential's own rounding out of the rotation adjoint's error.

``|w|`` is the swept axis. It is the series argument through ``s = |w|^2`` and
the magnitude of every term carrying a factor of ``w``, and it includes the
origin, where the axis normal form has no derivative and this chart is analytic.
Nothing else interacts with it: a cotangent enters every adjoint linearly, and
the tap enters ``dw`` linearly. So the sweep is one-dimensional, and the two
helpers that read neither a series nor ``w`` are checked once.
"""

import math
from typing import NamedTuple

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute

from slinoss._cute import dev_tensor
from slinoss.ops.so3ssd import (
    deriv_coeffs,
    quat_exp,
    rot_matrix,
    series_coeffs,
    skew,
    tap_matrix,
)
from slinoss.ops.so3ssd.cute.common import (
    COS_HALF_D,
    FP32_SERIES_TERMS,
    SINC_HALF_D,
    THREADS,
    mat3_add,
    mat3_outer,
    quat_exp_vjp,
    rot_hom_vjp,
    sym_asym,
    tap_matrix_vjp,
)
from tests.conftest import assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# float32 device arithmetic against a float64 oracle on identical input bits.
# Every adjoint is a handful of products and a sum of at most three of them, so
# the error is that sum's reassociation plus, for the exponential, the
# cancellation between the two Horner evaluations at the top of the domain. Worst
# measured over every label of the whole sweep: 1.24e-07, at w_scale 8 on the
# exponential, which is where that cancellation is worst.
VJP_REL = 5e-7

# Truncating the derivative series at nine terms against the reference's
# thirteen, over s <= pi^2. Both sides are float64 at fixed arguments, so this is
# the truncation alone. Worst measured: 2.78e-14, on the scalar series.
SERIES_REL = 1e-13


class _Probe(NamedTuple):
    """One probe launch: its inputs, then one field per helper output.

    Attributes:
        w: Rotation vectors, ``(M,3)`` float32.
        tap: ``(kr, g, h)`` of the previous tap, ``(M,3)`` float32.
        q: Unit quaternions of ``w``, ``(M,4)`` float32.
        dm: Cotangent matrices, ``(M,9)`` float32, row-major.
        dq: Cotangent quaternions, ``(M,4)`` float32.
        dq_rot: :func:`rot_hom_vjp` of ``(dm, q)``, ``(M,4)``.
        dtap: Tap half of :func:`tap_matrix_vjp`, ``(M,3)``.
        dw_tap: ``w`` half of :func:`tap_matrix_vjp`, ``(M,3)``.
        dw_exp: :func:`quat_exp_vjp` of ``(dq, w)``, ``(M,3)``.
        sym: :func:`sym_asym` of ``dm``, ``(M,9)``: three symmetric pairs, the
            axial vector, then the diagonal.
        add: :func:`mat3_add` of ``dm`` and ``outer``, ``(M,9)``.
        outer: :func:`mat3_outer` of ``(w, tap)``, ``(M,9)``.
    """

    w: Tensor
    tap: Tensor
    q: Tensor
    dm: Tensor
    dq: Tensor
    dq_rot: Tensor
    dtap: Tensor
    dw_tap: Tensor
    dw_exp: Tensor
    sym: Tensor
    add: Tensor
    outer: Tensor


@cute.kernel
def _vjp_kernel(
    gw: cute.Tensor,
    gtap: cute.Tensor,
    gq: cute.Tensor,
    gdm: cute.Tensor,
    gdq: cute.Tensor,
    odq: cute.Tensor,
    odtap: cute.Tensor,
    odwtap: cute.Tensor,
    odwexp: cute.Tensor,
    osym: cute.Tensor,
    oadd: cute.Tensor,
    oouter: cute.Tensor,
    count: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * threads + tid
    # One point per thread, so the last block overhangs. The index is clamped
    # unconditionally and only the stores are predicated: a global load inside a
    # divergent branch cannot be hoisted, and every helper below is branchless.
    at = cutlass.min(idx, cutlass.Int32(count - 1))

    w = (gw[at, 0], gw[at, 1], gw[at, 2])
    tap = (gtap[at, 0], gtap[at, 1], gtap[at, 2])
    q = (gq[at, 0], gq[at, 1], gq[at, 2], gq[at, 3])
    dq = (gdq[at, 0], gdq[at, 1], gdq[at, 2], gdq[at, 3])
    dm = (
        gdm[at, 0],
        gdm[at, 1],
        gdm[at, 2],
        gdm[at, 3],
        gdm[at, 4],
        gdm[at, 5],
        gdm[at, 6],
        gdm[at, 7],
        gdm[at, 8],
    )

    dq_rot = rot_hom_vjp(dm, q)
    dtap, dw_tap = tap_matrix_vjp(dm, tap, w)
    dw_exp = quat_exp_vjp(dq, w)
    sym, axial, diag = sym_asym(dm)
    outer = mat3_outer(w, tap)
    added = mat3_add(dm, outer)

    if idx < count:
        for j in cutlass.range_constexpr(4):
            odq[idx, j] = dq_rot[j]
        for j in cutlass.range_constexpr(3):
            odtap[idx, j] = dtap[j]
            odwtap[idx, j] = dw_tap[j]
            odwexp[idx, j] = dw_exp[j]
            osym[idx, j] = sym[j]
            osym[idx, 3 + j] = axial[j]
            osym[idx, 6 + j] = diag[j]
        for j in cutlass.range_constexpr(9):
            oadd[idx, j] = added[j]
            oouter[idx, j] = outer[j]


@cute.jit
def _vjp_launch(
    gw: cute.Tensor,
    gtap: cute.Tensor,
    gq: cute.Tensor,
    gdm: cute.Tensor,
    gdq: cute.Tensor,
    odq: cute.Tensor,
    odtap: cute.Tensor,
    odwtap: cute.Tensor,
    odwexp: cute.Tensor,
    osym: cute.Tensor,
    oadd: cute.Tensor,
    oouter: cute.Tensor,
    count: cutlass.Constexpr,
    blocks: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    _vjp_kernel(
        gw,
        gtap,
        gq,
        gdm,
        gdq,
        odq,
        odtap,
        odwtap,
        odwexp,
        osym,
        oadd,
        oouter,
        count,
        threads,
    ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1))


def _run(w_scale: float) -> _Probe:
    """Build one point per token and run every adjoint on it.

    ``w`` and the tap come from the parameter map, so ``|w| <= w_max`` holds by
    construction (I2) and the tap is the unconstrained triple the chart takes.
    The two cotangents are drawn: an adjoint's cotangent is arbitrary, and the
    pipeline that produces one is the kernel under test's caller.

    Args:
        w_scale: Multiplies the raw rotation vector. Zero gives ``w = 0``.

    Returns:
        A :class:`_Probe`.
    """
    inp = make_inputs(
        bsz=2,
        heads=3,
        seqlen=40,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=w_scale,
    )
    w = inp.trans[..., :3].reshape(-1, 3).contiguous()
    tap = inp.K[..., 0, :3].reshape(-1, 3).contiguous()
    q = quat_exp(w.double()).float().contiguous()
    count = int(w.shape[0])

    gen = torch.Generator(device=w.device).manual_seed(2)
    opts = {"device": w.device, "dtype": torch.float32}
    dm = torch.randn(count, 9, generator=gen, **opts)
    dq = torch.randn(count, 4, generator=gen, **opts)
    odq = torch.empty(count, 4, **opts)
    odtap = torch.empty(count, 3, **opts)
    odwtap = torch.empty(count, 3, **opts)
    odwexp = torch.empty(count, 3, **opts)
    osym = torch.empty(count, 9, **opts)
    oadd = torch.empty(count, 9, **opts)
    oouter = torch.empty(count, 9, **opts)

    _vjp_launch(
        dev_tensor(w),
        dev_tensor(tap),
        dev_tensor(q),
        dev_tensor(dm),
        dev_tensor(dq),
        dev_tensor(odq),
        dev_tensor(odtap),
        dev_tensor(odwtap),
        dev_tensor(odwexp),
        dev_tensor(osym),
        dev_tensor(oadd),
        dev_tensor(oouter),
        count,
        (count + THREADS - 1) // THREADS,
        THREADS,
    )
    torch.cuda.synchronize()
    return _Probe(
        w=w,
        tap=tap,
        q=q,
        dm=dm,
        dq=dq,
        dq_rot=odq,
        dtap=odtap,
        dw_tap=odwtap,
        dw_exp=odwexp,
        sym=osym,
        add=oadd,
        outer=oouter,
    )


def _poly(s: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    """Horner evaluation of a series in ``s``, on the host."""
    out = torch.full_like(s, coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        out = out * s + coeff
    return out


# The parameter map sends |raw| to w_max * |raw| / sqrt(1 + |raw|^2), so the
# scale reaches the top of the domain rather than passing through it: 8.0 puts
# |w| within a percent of w_max, where the series argument is largest and its
# cancellation worst. Zero is w = 0 exactly and 1e-8 is the neighbourhood of it.
W_SCALES = [0.0, 1e-8, 1.0, 8.0]


@pytest.mark.parametrize("w_scale", W_SCALES)
def test_adjoints_match_float64_autograd(w_scale: float) -> None:
    """Every adjoint against autograd through its own primal.

    The rotation cotangent is split into its scalar and vector halves and the tap
    cotangent into its three components, because the relative error is measured
    against the largest entry of what it is compared with: at small ``|w|`` the
    scalar half and the ``g`` and ``h`` components fall off with ``|w|`` while the
    rest stay of order one, so a single bound over the packed tensor would
    normalize them by a magnitude that is not theirs.

    The primal of :func:`rot_hom_vjp` is the unit-norm rotation matrix, which is
    what the reference differentiates and what :func:`rot_hom` equals at unit
    norm. The two adjoints differ by a radial multiple of ``q``, so this catches
    the trace term that the homogeneous form's own adjoint would carry.

    At ``w = 0`` the scalar rotation cotangent and the ``g`` and ``h`` tap
    cotangents vanish, and the exponential's is exactly ``0.5 dq_v``. Those
    labels measure zero error there rather than measuring nothing: a term that
    failed to vanish is divided by an oracle that did.
    """
    probe = _run(w_scale)
    w = probe.w.double().requires_grad_(True)
    tap = probe.tap.double().requires_grad_(True)
    q = probe.q.double().requires_grad_(True)
    dm = probe.dm.double().unflatten(-1, (3, 3))

    (want_q,) = torch.autograd.grad(rot_matrix(q), q, dm)
    want_tap, want_w = torch.autograd.grad(tap_matrix(tap, w), (tap, w), dm)
    (want_exp,) = torch.autograd.grad(quat_exp(w), w, probe.dq.double())

    tag = f"cute-vjp[w_scale {w_scale:g}]"
    assert_max_rel(probe.dq_rot[:, 0], want_q[:, 0], VJP_REL, f"{tag}.rot.scalar")
    assert_max_rel(probe.dq_rot[:, 1:], want_q[:, 1:], VJP_REL, f"{tag}.rot.vector")
    for j, name in enumerate(("kr", "g", "h")):
        assert_max_rel(probe.dtap[:, j], want_tap[:, j], VJP_REL, f"{tag}.tap.d{name}")
    assert_max_rel(probe.dw_tap, want_w, VJP_REL, f"{tag}.tap.dw")
    assert_max_rel(probe.dw_exp, want_exp, VJP_REL, f"{tag}.exp.dw")


def test_sym_asym_splits_a_cotangent_by_symmetry() -> None:
    """The three parts of the symmetry split, at one ``|w|``.

    :func:`sym_asym` is a rearrangement of ``dm`` and reads no ``w``, so it is not
    swept; ``w`` appears here only as the test vector of the axial contraction.

    The diagonal is a copy and each symmetric pair is one float32 add of the two
    entries torch adds, so both are asserted bitwise. The axial vector is
    asserted through the identity that defines it, against the reference's own
    ``skew``: a flipped sign there leaves both consumers plausible and wrong,
    because each contracts the axial vector against a vector it also scales.
    """
    probe = _run(1.0)
    dm = probe.dm.unflatten(-1, (3, 3))
    diag = torch.stack([dm[:, 0, 0], dm[:, 1, 1], dm[:, 2, 2]], dim=-1)
    pairs = torch.stack(
        [
            dm[:, 0, 1] + dm[:, 1, 0],
            dm[:, 0, 2] + dm[:, 2, 0],
            dm[:, 1, 2] + dm[:, 2, 1],
        ],
        dim=-1,
    )
    assert torch.equal(probe.sym[:, 6:], diag)
    assert torch.equal(probe.sym[:, :3], pairs)

    got = (probe.sym[:, 3:6].double() * probe.w.double()).sum(-1)
    want = (dm.double() * skew(probe.w.double())).sum((-2, -1))
    assert_max_rel(got, want, VJP_REL, "cute-vjp.sym_asym.axial")


def test_mat3_add_and_outer_are_entrywise() -> None:
    """The two parameter-epilogue primitives, at one ``|w|``.

    Neither reads a series or a quaternion, so neither is swept. ``w`` and the tap
    are two unequal vectors, which is what makes a transposed outer product fail,
    and the outer product is one float32 multiply per entry, so it is asserted
    bitwise. The sum is compared against float64 instead: the device is free to
    contract it with the product it follows into a single FMA.
    """
    probe = _run(1.0)
    outer = probe.w[:, :, None] * probe.tap[:, None, :]
    assert torch.equal(probe.outer.unflatten(-1, (3, 3)), outer)

    want = probe.dm.double() + (
        probe.w.double()[:, :, None] * probe.tap.double()[:, None, :]
    ).flatten(-2, -1)
    assert_max_rel(probe.add, want, VJP_REL, "cute-vjp.mat3_add")


def test_fp32_derivative_series_holds_over_the_reachable_domain() -> None:
    """Nine derivative terms are enough over ``s = |w|^2 <= pi^2``.

    ``FP32_SERIES_TERMS`` is sized for the primal series. Differentiating in ``s``
    scales term ``k`` by ``k`` and drops the constant term, so the truncation
    error grows by up to that factor and is re-justified here against the
    reference's own truncation, which is the one the float64 oracle
    differentiates. Sized for the closed ball rather than for ``w_max``, as the
    primal is.
    """
    assert len(COS_HALF_D) == FP32_SERIES_TERMS - 1
    assert len(SINC_HALF_D) == FP32_SERIES_TERMS - 1
    s = torch.linspace(0.0, math.pi**2, 1025, dtype=torch.float64)
    for offset, short in ((0, COS_HALF_D), (1, SINC_HALF_D)):
        assert_max_rel(
            _poly(s, short),
            _poly(s, deriv_coeffs(series_coeffs(offset))),
            SERIES_REL,
            f"cute-vjp.series_d[{offset}]",
        )
