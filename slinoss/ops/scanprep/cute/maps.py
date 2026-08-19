"""Bounded parameter maps. CuTe DSL forward and backward.

    w  = w_max * raw * rsqrt(1 + |raw|^2)
    ls = -softplus(raw)
    K  = tap, unchanged

One kernel and one launch per direction. The forward writes both packed layouts
directly, ``(B,H,T,4)`` and ``(B,H,T,2,4)``, and writes lane 3 of each tap as a
hard zero, so the packing costs no concatenation, no ``zeros_like``, and no
``aten::fill_``. Output buffers are ``torch.empty``: every element is written.

``trans`` and ``K`` are float32 at every input width, including under autocast
(I4). Low-precision inputs are widened on load, so both maps are evaluated in
float32; the gradients are narrowed back to the input width on store.

Parallel decomposition. One thread per token over the flattened ``B*H*T`` axis,
which is the only axis in the problem. Nothing is shared between tokens, so
neither kernel holds shared memory or a barrier. ``T`` is arbitrary, so the grid
is ``ceil(B*H*T / THREADS)`` blocks and the token index carries a bounds
predicate; the predicate is warp-uniform except in the last warp of the last
block.

Invariants. I1 and I2 are produced here rather than asserted here. ``rsqrt`` acts
on ``1 + |raw|^2 >= 1`` and ``softplus`` is evaluated through an identity whose
exponential argument is never positive, so no clamp, epsilon, or validity pass
exists on this path.

DRAM-bound. Per token the forward moves ten input elements in and twelve float32
out: 88 B at float32, 68 B at bfloat16. The backward moves twelve float32
cotangents and four input elements in, ten input elements out: 104 B at float32,
76 B at bfloat16.
"""

import math
from typing import Any, NamedTuple, cast

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    LOG2_E,
    Scalar,
    dev_tensor,
    f32,
    narrow,
    select,
    widen,
)
from slinoss._guard import Named, check_layout
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.scanprep.reference import ScanParams

__all__ = [
    "THREADS",
    "ScanGrads",
    "ScanPrepFunction",
    "scanprep",
    "scanprep_backward",
    "scanprep_bwd",
    "scanprep_bwd_kernel",
    "scanprep_forward",
    "scanprep_fwd",
    "scanprep_fwd_kernel",
]

# One token per thread, eight warps. Live state is under a dozen float32 and
# there is no shared memory, so occupancy is capped by nothing.
THREADS = 256

# ---------------------------------------------------------------------------
# Device math
# ---------------------------------------------------------------------------


def _softplus_parts(raw: Scalar) -> tuple[Any, Scalar]:
    """``(raw > 0, exp(-|raw|))``.

    The exponent is non-positive at every input, so the value lies in ``(0, 1]``
    and no input magnitude overflows. The absolute value is a select, not a
    branch, so the predicate costs one predicated move and no divergence.
    """
    positive = raw > cutlass.Float32(0.0)
    return positive, f32(cute.exp2(select(positive, -raw, raw) * LOG2_E))


def _log_scale(raw: Scalar) -> Scalar:
    """``-softplus(raw) <= 0`` (I1).

    Evaluated as ``min(-raw, 0) - log1p(exp(-|raw|))``. Both terms are bounded by
    ``|raw|`` for every finite input, so neither the exponential nor the sum can
    overflow, and both halves are selects on one predicate.

    ``log1p`` is formed as ``log(1 + e)``. That addition drops the part of ``e``
    below float32 epsilon, which is an absolute error of at most ``2^-24`` on a
    quantity whose magnitude is ``|raw|``, and it drops nothing at all wherever
    ``e`` is normal against one.
    """
    positive, small = _softplus_parts(raw)
    return select(positive, -raw, cutlass.Float32(0.0)) - f32(cute.log(small + 1.0))


def _log_scale_grad(raw: Scalar) -> Scalar:
    """``d(-softplus)/draw = -sigmoid(raw)``.

    ``sigmoid(raw)`` is ``1 / (1 + e)`` where ``raw > 0`` and ``e / (1 + e)``
    elsewhere, with ``e = exp(-|raw|)`` in ``(0, 1]``: one select, and no
    intermediate exceeds one, so no input magnitude overflows.
    """
    positive, small = _softplus_parts(raw)
    return -select(positive, cutlass.Float32(1.0), small) / (small + 1.0)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@cute.kernel
def scanprep_fwd_kernel(
    gw: cute.Tensor,
    gls: cute.Tensor,
    gtap: cute.Tensor,
    gtrans: cute.Tensor,
    gpack: cute.Tensor,
    tokens: cutlass.Int32,
    w_max: cutlass.Float32,
    threads: cutlass.Constexpr,
) -> None:
    """Apply both maps and write the packed layouts.

    Args:
        gw: ``(B*H*T, 3)`` unconstrained rotation vectors, input dtype.
        gls: ``(B*H*T,)`` unconstrained log-scales, input dtype.
        gtap: ``(B*H*T, 6)`` unconstrained taps, ``3*tap + j``, input dtype.
        gtrans: ``(B*H*T, 4)`` float32, written with ``(w_x, w_y, w_z, ls)``.
        gpack: ``(B*H*T, 8)`` float32, written with ``(kr, g, h, 0)`` per tap at
            component ``4*tap + j``.
        tokens: ``B*H*T``. Dynamic.
        w_max: Rotation-vector bound. Dynamic, so one compiled variant covers
            every bound.
        threads: Block width. Compile-time.

    Invariants:
        Every input dtype is one dtype, so one widening type serves all three
        reads. ``tokens`` is arbitrary, so the last block is predicated.
    """
    tile, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    token = tile * threads + tid

    if token < tokens:
        src = gw.element_type
        rx = widen(gw[token, 0], src)
        ry = widen(gw[token, 1], src)
        rz = widen(gw[token, 2], src)
        # 1 + |raw|^2 >= 1, so the rsqrt is regular over the whole domain and the
        # product lands in the closed ball of radius w_max (I2).
        scale = w_max * f32(cute.rsqrt(rx * rx + ry * ry + rz * rz + 1.0))
        gtrans[token, 0] = rx * scale
        gtrans[token, 1] = ry * scale
        gtrans[token, 2] = rz * scale
        gtrans[token, 3] = _log_scale(widen(gls[token], src))

        zero = cutlass.Float32(0.0)
        for tap in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(3):
                gpack[token, 4 * tap + j] = widen(gtap[token, 3 * tap + j], src)
            gpack[token, 4 * tap + 3] = zero


@cute.jit
def scanprep_fwd(
    gw: cute.Tensor,
    gls: cute.Tensor,
    gtap: cute.Tensor,
    gtrans: cute.Tensor,
    gpack: cute.Tensor,
    tokens: cutlass.Int32,
    tiles: cutlass.Int32,
    w_max: cutlass.Float32,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`scanprep_fwd_kernel`.

    Only ``threads`` is compile-time, so one compiled variant per input dtype
    covers every batch, head, token count, and bound.
    """
    scanprep_fwd_kernel(gw, gls, gtap, gtrans, gpack, tokens, w_max, threads).launch(
        grid=(tiles, 1, 1), block=(threads, 1, 1)
    )


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@cute.kernel
def scanprep_bwd_kernel(
    gdtrans: cute.Tensor,
    gdpack: cute.Tensor,
    gw: cute.Tensor,
    gls: cute.Tensor,
    gdw: cute.Tensor,
    gdls: cute.Tensor,
    gdtap: cute.Tensor,
    tokens: cutlass.Int32,
    w_max: cutlass.Float32,
    threads: cutlass.Constexpr,
) -> None:
    """Pull the cotangents of both maps back to the unconstrained parameters.

    Args:
        gdtrans: ``(B*H*T, 4)`` float32 cotangent of ``trans``.
        gdpack: ``(B*H*T, 8)`` float32 cotangent of ``K``. Component
            ``4*tap + 3`` is the cotangent of a constant and is not read.
        gw: ``(B*H*T, 3)`` unconstrained rotation vectors, input dtype.
        gls: ``(B*H*T,)`` unconstrained log-scales, input dtype.
        gdw: ``(B*H*T, 3)`` written, input dtype.
        gdls: ``(B*H*T,)`` written, input dtype.
        gdtap: ``(B*H*T, 6)`` written, ``3*tap + j``, input dtype.
        tokens: ``B*H*T``. Dynamic.
        w_max: Rotation-vector bound. Dynamic.
        threads: Block width. Compile-time.

    Invariants:
        The rotation-vector map is recomputed from ``gw`` rather than read back
        from ``trans``: the pullback needs ``raw``, not ``w``.
    """
    tile, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    token = tile * threads + tid

    if token < tokens:
        src = gw.element_type
        dst = gdw.element_type
        rx = widen(gw[token, 0], src)
        ry = widen(gw[token, 1], src)
        rz = widen(gw[token, 2], src)
        gx = gdtrans[token, 0]
        gy = gdtrans[token, 1]
        gz = gdtrans[token, 2]

        inv = f32(cute.rsqrt(rx * rx + ry * ry + rz * rz + 1.0))
        scale = w_max * inv
        # The map is a radial rescaling, so its Jacobian is the scale times a
        # rank-one correction along raw; inv*inv is 1/(1 + |raw|^2).
        pull = inv * inv * (gx * rx + gy * ry + gz * rz)
        gdw[token, 0] = narrow(scale * (gx - pull * rx), dst)
        gdw[token, 1] = narrow(scale * (gy - pull * ry), dst)
        gdw[token, 2] = narrow(scale * (gz - pull * rz), dst)

        raw_ls = widen(gls[token], src)
        gdls[token] = narrow(_log_scale_grad(raw_ls) * gdtrans[token, 3], dst)

        for tap in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(3):
                gdtap[token, 3 * tap + j] = narrow(gdpack[token, 4 * tap + j], dst)


@cute.jit
def scanprep_bwd(
    gdtrans: cute.Tensor,
    gdpack: cute.Tensor,
    gw: cute.Tensor,
    gls: cute.Tensor,
    gdw: cute.Tensor,
    gdls: cute.Tensor,
    gdtap: cute.Tensor,
    tokens: cutlass.Int32,
    tiles: cutlass.Int32,
    w_max: cutlass.Float32,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`scanprep_bwd_kernel`."""
    scanprep_bwd_kernel(
        gdtrans, gdpack, gw, gls, gdw, gdls, gdtap, tokens, w_max, threads
    ).launch(grid=(tiles, 1, 1), block=(threads, 1, 1))


# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------


def _check_w_max(w_max: float) -> None:
    """Raises:
    ValueError: If ``w_max`` is outside ``(0, pi)``, which I2 requires.
    """
    if not 0.0 < w_max < math.pi:
        raise ValueError(f"w_max must lie in (0, pi), got {w_max}")


def _check_dtypes(named: Named) -> None:
    """Raises:
    TypeError: If an operand dtype has no kernel path, or if the operands do
        not share one dtype. One dtype per call keeps a single widening type
        inside the kernel.
    """
    for tensor, name in named:
        if tensor.dtype not in KERNEL_DTYPES:
            raise TypeError(
                f"{name} has dtype {tensor.dtype}; kernel dtypes: {KERNEL_DTYPES}"
            )
    head, head_name = named[0]
    for tensor, name in named[1:]:
        if tensor.dtype is not head.dtype:
            raise TypeError(
                f"{name} is {tensor.dtype} and {head_name} is {head.dtype}; "
                "one dtype per call"
            )


def _check_pinned(named: Named) -> None:
    """Raises:
    ValueError: If a cotangent of ``trans`` or ``K`` is not float32. Both are
        float32-pinned (I4), so their cotangents are too.
    """
    for tensor, name in named:
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must be float32 (I4), got {tensor.dtype}")


def _lead(w_raw: Tensor) -> tuple[int, int, int]:
    """The ``(B, H, T)`` prefix of a rotation-vector operand.

    Raises:
        ValueError: If ``w_raw`` is not ``(B,H,T,3)``, or if it holds no token.
            A zero-token call has no launchable grid, so it is refused rather
            than special-cased.
    """
    if w_raw.ndim != 4 or w_raw.shape[-1] != 3:
        raise ValueError(f"w_raw must be (B,H,T,3), got {tuple(w_raw.shape)}")
    lead = (int(w_raw.shape[0]), int(w_raw.shape[1]), int(w_raw.shape[2]))
    if lead[0] * lead[1] * lead[2] == 0:
        raise ValueError(f"w_raw must hold at least one token, got {lead}")
    return lead


def _check_raw_shapes(w_raw: Tensor, ls_raw: Tensor) -> tuple[int, int, int]:
    """The ``(B, H, T)`` prefix shared by the two unconstrained operands.

    Raises:
        ValueError: On a rank or shape mismatch.
    """
    lead = _lead(w_raw)
    if tuple(ls_raw.shape) != lead:
        raise ValueError(f"ls_raw must be {lead}, got {tuple(ls_raw.shape)}")
    return lead


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def _flat(tensor: Tensor, *shape: int) -> cute.Tensor:
    """View an operand in its flattened token form and wrap it for a launch.

    The view aliases the same storage, so nothing is copied.
    :func:`slinoss._cute.dev_tensor` handles the detach the DLPack export needs.
    """
    return dev_tensor(tensor.view(*shape))


class ScanGrads(NamedTuple):
    """Gradients of the bounded maps.

    Attributes:
        dw_raw: ``(B,H,T,3)``, input dtype.
        dls_raw: ``(B,H,T)``, input dtype.
        dtap_raw: ``(B,H,T,2,3)``, input dtype.
    """

    dw_raw: Tensor
    dls_raw: Tensor
    dtap_raw: Tensor


def scanprep_forward(
    w_raw: Tensor,
    ls_raw: Tensor,
    tap_raw: Tensor,
    *,
    w_max: float,
) -> ScanParams:
    """Apply the bounded maps and pack, in one launch.

    Args:
        w_raw: Unconstrained rotation vectors, ``(B,H,T,3)``, contiguous CUDA,
            one of :data:`slinoss._precision.KERNEL_DTYPES`.
        ls_raw: Unconstrained log-scales, ``(B,H,T)``, same dtype.
        tap_raw: Unconstrained taps ``(kr, g, h)``, ``(B,H,T,2,3)``, same dtype.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanParams`. ``trans`` and ``K`` are
        float32 whatever the input dtype (I4), and lane 3 of each tap of ``K``
        is zero.

    Raises:
        ValueError: On a shape mismatch, a zero-token operand, a non-CUDA or
            non-contiguous operand, or a ``w_max`` outside ``(0, pi)``.
        TypeError: On a dtype with no kernel path, or on operands that do not
            share one dtype.
    """
    _check_w_max(w_max)
    lead = _check_raw_shapes(w_raw, ls_raw)
    if tuple(tap_raw.shape) != (*lead, 2, 3):
        raise ValueError(f"tap_raw must be {(*lead, 2, 3)}, got {tuple(tap_raw.shape)}")
    named = ((w_raw, "w_raw"), (ls_raw, "ls_raw"), (tap_raw, "tap_raw"))
    _check_dtypes(named)
    check_layout(named)

    tokens = lead[0] * lead[1] * lead[2]
    trans = torch.empty(*lead, 4, dtype=torch.float32, device=w_raw.device)
    packed = torch.empty(*lead, 2, 4, dtype=torch.float32, device=w_raw.device)
    scanprep_fwd(
        _flat(w_raw, tokens, 3),
        _flat(ls_raw, tokens),
        _flat(tap_raw, tokens, 6),
        _flat(trans, tokens, 4),
        _flat(packed, tokens, 8),
        tokens,
        (tokens + THREADS - 1) // THREADS,
        float(w_max),
        THREADS,
    )
    return ScanParams(trans=trans, K=packed)


def scanprep_backward(
    dtrans: Tensor,
    dK: Tensor,
    w_raw: Tensor,
    ls_raw: Tensor,
    *,
    w_max: float,
) -> ScanGrads:
    """Pull the cotangents of ``trans`` and ``K`` back to the raw parameters.

    ``tap_raw`` is not read: the tap map is the identity, so its pullback is a
    narrowing of ``dK``. The cotangent of lane 3 is the cotangent of a constant
    and is discarded.

    Args:
        dtrans: Cotangent of ``trans``, ``(B,H,T,4)`` float32, contiguous CUDA.
        dK: Cotangent of ``K``, ``(B,H,T,2,4)`` float32, contiguous CUDA.
        w_raw: The forward's rotation-vector operand, ``(B,H,T,3)``.
        ls_raw: The forward's log-scale operand, ``(B,H,T)``, same dtype.
        w_max: The bound the forward was called with, in ``(0, pi)``.

    Returns:
        A :class:`ScanGrads` in the dtype of ``w_raw``.

    Raises:
        ValueError: On a shape mismatch, a zero-token operand, a non-float32
            cotangent, a non-CUDA or non-contiguous operand, or a ``w_max``
            outside ``(0, pi)``.
        TypeError: On a dtype with no kernel path, or on raw operands that do
            not share one dtype.
    """
    _check_w_max(w_max)
    lead = _check_raw_shapes(w_raw, ls_raw)
    if tuple(dtrans.shape) != (*lead, 4):
        raise ValueError(f"dtrans must be {(*lead, 4)}, got {tuple(dtrans.shape)}")
    if tuple(dK.shape) != (*lead, 2, 4):
        raise ValueError(f"dK must be {(*lead, 2, 4)}, got {tuple(dK.shape)}")
    raws = ((w_raw, "w_raw"), (ls_raw, "ls_raw"))
    _check_dtypes(raws)
    _check_pinned(((dtrans, "dtrans"), (dK, "dK")))
    check_layout(((dtrans, "dtrans"), (dK, "dK"), *raws))

    tokens = lead[0] * lead[1] * lead[2]
    dw_raw = torch.empty_like(w_raw)
    dls_raw = torch.empty_like(ls_raw)
    dtap_raw = torch.empty(*lead, 2, 3, dtype=w_raw.dtype, device=w_raw.device)
    scanprep_bwd(
        _flat(dtrans, tokens, 4),
        _flat(dK, tokens, 8),
        _flat(w_raw, tokens, 3),
        _flat(ls_raw, tokens),
        _flat(dw_raw, tokens, 3),
        _flat(dls_raw, tokens),
        _flat(dtap_raw, tokens, 6),
        tokens,
        (tokens + THREADS - 1) // THREADS,
        float(w_max),
        THREADS,
    )
    return ScanGrads(dw_raw=dw_raw, dls_raw=dls_raw, dtap_raw=dtap_raw)


_Packed = tuple[Tensor, Tensor]


class ScanPrepFunction(torch.autograd.Function):
    """Differentiable bounded maps.

    Returns a positional tuple because :class:`torch.autograd.Function` requires
    one. :func:`scanprep` names the fields.

    Saves ``w_raw`` and ``ls_raw`` only. The tap map is the identity, so
    ``tap_raw`` carries nothing the backward needs, and neither packed output is
    read back: the backward sees the two cotangents and the two raw operands.
    """

    @staticmethod
    def forward(
        ctx: Any,
        w_raw: Tensor,
        ls_raw: Tensor,
        tap_raw: Tensor,
        w_max: float,
    ) -> _Packed:
        out = scanprep_forward(w_raw, ls_raw, tap_raw, w_max=w_max)
        ctx.save_for_backward(w_raw, ls_raw)
        ctx.w_max = w_max
        return out.trans, out.K

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dtrans: Tensor,
        dK: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        w_raw, ls_raw = ctx.saved_tensors
        grads = scanprep_backward(dtrans, dK, w_raw, ls_raw, w_max=ctx.w_max)
        return grads.dw_raw, grads.dls_raw, grads.dtap_raw, None


def scanprep(
    w_raw: Tensor,
    ls_raw: Tensor,
    tap_raw: Tensor,
    *,
    w_max: float,
) -> ScanParams:
    """Bounded maps with an analytic backward. The public fast path.

    Args:
        w_raw: Unconstrained rotation vectors, ``(B,H,T,3)``.
        ls_raw: Unconstrained log-scales, ``(B,H,T)``.
        tap_raw: Unconstrained taps ``(kr, g, h)``, ``(B,H,T,2,3)``.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanParams`, float32 (I4).

    Raises:
        ValueError: On a shape, layout, device, or bound violation.
        TypeError: On an unsupported or mixed dtype.
    """
    trans, packed = cast(
        "_Packed", ScanPrepFunction.apply(w_raw, ls_raw, tap_raw, w_max)
    )
    return ScanParams(trans=trans, K=packed)
