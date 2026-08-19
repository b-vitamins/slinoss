"""Fused mixer tail. CuTe DSL forward and backward.

    x   = y + d_skip * u
    x   = x * silu(gate)
    out = x * rsqrt(mean(x^2) + eps) * weight

One kernel and one launch per direction. The skip, the gate, and the norm are one
pass over the operands: every element is read once and every intermediate stays in
registers.

Parallel decomposition. The reduction runs over ``P`` and never crosses the head
axis, so one ``(b, h, t)`` triple is one independent problem of length ``P``. One
warp owns one triple: lane ``l`` holds columns ``l``, ``l + 32``, ... and the sum
over ``P`` is a full-warp butterfly, so the reduction never touches shared memory
and each segment is 32 consecutive columns across the lanes of the warp, which is
one coalesced run. A warp runs :data:`ROWS_PER_WARP` triples in sequence and a
block is :data:`WARPS` warps, so the grid is ``(ceil(B*T / ROWS), H)``, which at
any trained shape exceeds twice the SM count. ``B*T`` is arbitrary, so the row
index carries a bounds predicate;
the predicate depends on the warp and not on the lane, so it is warp-uniform and
the butterfly inside it is entered by all 32 lanes or by none.

Precision. The sum of squares, the dot product the backward reduces, and the
reciprocal square root are float32 at every operand width. ``y``, ``u``, ``gate``
and the output are the only low-precision tensors; they are widened on load and
narrowed on store. Operand width and parameter width are independent, so float32
parameters against bfloat16 activations is one call and not a cast.

Parameter gradients. ``d_skip`` and ``weight`` are ``(H,P)``, so their gradients
reduce over ``(B,T)``. Each lane accumulates its own columns across the rows its
warp runs, the block sums the warps through one shared tile, and each block stores
one ``(H,P)`` row of tile partials. That is the kernel's epilogue; there is no
second pass over the operands. Closing the reduction inside the launch would need
an accumulator zeroed before it, and a zero fill on the hot path is not available,
so the partial buffer is ``torch.empty``, every element of it is written by the
kernel, and the cross-tile sum is a reduction over that buffer alone.

Shared memory. One tile, :func:`param_tile`, holding the per-warp parameter
partials. Consecutive lanes touch consecutive words both when the warps write it
and when warp 0 reads it back, so neither access can conflict. Its budget is
computed from the layout and checked against the queried capacity, which is what
bounds ``P``.

DRAM-bound, both directions. Analytic byte counts, no measurement is committed.
Per ``(b,h,t)`` row the forward reads ``y``, ``u`` and ``gate`` and writes the
output: ``4*P`` operand elements, ``16*P`` B at float32 and ``8*P`` B at bfloat16.
The backward reads ``dout``, ``y``, ``u`` and ``gate`` and writes ``dy``, ``du``
and ``dgate``: ``7*P`` operand elements, ``28*P`` B at float32 and ``14*P`` B at
bfloat16. On top of that both directions read the two ``(H,P)`` parameter rows,
``2*H*P`` elements, and the backward writes ``2*ceil(B*T/ROWS)*H*P`` float32 of
tile partials and reads them back once.
"""

import functools
from typing import Any, NamedTuple, cast

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Scalar,
    Tile,
    assert_smem_fits,
    dev_tensor,
    f32,
    narrow,
    select,
    sigmoid,
    silu,
    silu_grad,
    smem_bytes,
    widen,
)
from slinoss._guard import Named, check_layout
from slinoss._precision import KERNEL_DTYPES

__all__ = [
    "ROWS",
    "ROWS_PER_WARP",
    "SLOTS",
    "SLOT_DSKIP",
    "SLOT_WEIGHT",
    "THREADS",
    "WARPS",
    "MixerTailFunction",
    "MixerTailGrads",
    "mixer_tail",
    "mixer_tail_backward",
    "mixer_tail_bwd",
    "mixer_tail_bwd_kernel",
    "mixer_tail_forward",
    "mixer_tail_fwd",
    "mixer_tail_fwd_kernel",
    "param_tile",
]

WARPS = 8
"""Warps per block. One warp owns one ``(b, h, t)`` row, so this is also the rows
a block holds in flight."""

THREADS = WARPS * cute.arch.WARP_SIZE
"""Threads per block."""

ROWS_PER_WARP = 4
"""Rows one warp runs in sequence.

The epilogue writes one ``(H,P)`` row of partials per block whatever this is, so
raising it divides the partial traffic by itself. Live state is per row and is
reused across the sequence, so the register cost is the accumulators alone."""

ROWS = WARPS * ROWS_PER_WARP
"""Rows one block covers on the flattened ``B*T`` axis."""

SLOT_DSKIP = 0
"""Partial slot holding the ``d_skip`` gradient."""

SLOT_WEIGHT = 1
"""Partial slot holding the ``weight`` gradient."""

SLOTS = 2
"""Parameter gradients the epilogue reduces."""


def param_tile(segments: int) -> Tile:
    """Per-warp parameter-gradient partials: ``(SLOTS, WARPS, 32*segments)``.

    The trailing extent is the warp-strided column span rather than ``P``, so
    every lane writes a slot and every lane reads one back. The ``P`` predicate
    then lives on the global store alone, and the block reduction needs none.

    Args:
        segments: Columns per lane, ``ceil(P/32)``.

    Returns:
        The tile.
    """
    span = cute.arch.WARP_SIZE * segments
    return Tile((SLOTS, WARPS, span), (WARPS * span, span, 1))


# ---------------------------------------------------------------------------
# Device math
# ---------------------------------------------------------------------------

# The logistic family is in slinoss._cute: the block's activation kernels round
# the same function, and two roundings of one activation is a divergence.


def _columns(
    lane: Any, rows: Any, segments: int, exact: bool
) -> list[tuple[Any, Any, bool]]:
    """Per-segment ``(column, read position, masked)`` for one lane.

    ``P`` is arbitrary, so the last segment can run past it. That segment reads a
    clamped position and drops the value with a select, which keeps the load in
    bounds without a branch. No earlier segment can overrun: ``P`` exceeds
    ``32*(segments-1)``.

    Args:
        lane: Lane index within the warp.
        rows: ``P``. Dynamic.
        segments: ``ceil(P/32)``. Compile-time.
        exact: Whether ``P`` is a multiple of the warp width. Compile-time.

    Returns:
        One entry per segment, outermost first.
    """
    out: list[tuple[Any, Any, bool]] = []
    for j in range(segments):
        col = j * cute.arch.WARP_SIZE + lane
        masked = not exact and j == segments - 1
        out.append((col, cutlass.min(col, rows - 1) if masked else col, masked))
    return out


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@cute.kernel
def mixer_tail_fwd_kernel(
    gy: cute.Tensor,
    gu: cute.Tensor,
    ggate: cute.Tensor,
    gdskip: cute.Tensor,
    gweight: cute.Tensor,
    gout: cute.Tensor,
    tokens: cutlass.Int32,
    seqlen: cutlass.Int32,
    rows: cutlass.Int32,
    denom: cutlass.Float32,
    eps: cutlass.Float32,
    segments: cutlass.Constexpr,
    exact: cutlass.Constexpr,
) -> None:
    """Apply the skip, the gate, and the per-head RMS norm.

    Args:
        gy: ``(B,H,T,P)`` scan output, operand dtype.
        gu: ``(B,H,T,P)`` scan input, operand dtype.
        ggate: ``(B,H,T,P)`` gate, operand dtype.
        gdskip: ``(H,P)`` skip scale, parameter dtype.
        gweight: ``(H,P)`` norm scale, parameter dtype.
        gout: ``(B,H,T,P)`` written, operand dtype.
        tokens: ``B*T``. Dynamic.
        seqlen: ``T``, the divisor that splits a flat row index. Dynamic.
        rows: ``P``. Dynamic.
        denom: ``P`` as float32, the mean divisor. Passed rather than converted
            on the device.
        eps: Added to the mean square. Dynamic, so one variant covers every
            epsilon.
        segments: ``ceil(P/32)``. Compile-time.
        exact: Whether ``P`` is a multiple of the warp width. Compile-time.

    Invariants:
        The sum of squares is float32 whatever the operand width. The row
        predicate is warp-uniform, so every lane of a live warp reaches the
        butterfly.
    """
    tile, head, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    src = gy.element_type
    par = gweight.element_type
    dst = gout.element_type
    zero = cutlass.Float32(0.0)
    cols = _columns(lane, rows, segments, exact)

    for step in cutlass.range_constexpr(ROWS_PER_WARP):
        token = tile * ROWS + step * WARPS + warp
        if token < tokens:
            bidx = token // seqlen
            tidx = token - bidx * seqlen

            held: list[Scalar] = []
            sumsq = zero
            for j in cutlass.range_constexpr(segments):
                col, pos, masked = cols[j]
                gate = widen(ggate[bidx, head, tidx, pos], src)
                value = (
                    widen(gy[bidx, head, tidx, pos], src)
                    + widen(gdskip[head, pos], par)
                    * widen(gu[bidx, head, tidx, pos], src)
                ) * silu(gate, sigmoid(gate))
                if cutlass.const_expr(masked):
                    value = select(col < rows, value, zero)
                sumsq = sumsq + value * value
                held.append(value)

            total = f32(cute.arch.warp_reduction_sum(sumsq))
            scale = f32(cute.rsqrt(total / denom + eps))

            for j in cutlass.range_constexpr(segments):
                col, pos, masked = cols[j]
                out = narrow(held[j] * scale * widen(gweight[head, pos], par), dst)
                if cutlass.const_expr(masked):
                    if col < rows:
                        gout[bidx, head, tidx, col] = out
                else:
                    gout[bidx, head, tidx, col] = out


@cute.jit
def mixer_tail_fwd(
    gy: cute.Tensor,
    gu: cute.Tensor,
    ggate: cute.Tensor,
    gdskip: cute.Tensor,
    gweight: cute.Tensor,
    gout: cute.Tensor,
    tokens: cutlass.Int32,
    seqlen: cutlass.Int32,
    rows: cutlass.Int32,
    denom: cutlass.Float32,
    eps: cutlass.Float32,
    tiles: cutlass.Int32,
    heads: cutlass.Int32,
    segments: cutlass.Constexpr,
    exact: cutlass.Constexpr,
) -> None:
    """Launch :func:`mixer_tail_fwd_kernel`.

    Only ``segments`` and ``exact`` are compile-time, so one compiled variant per
    dtype pair covers every batch, head, and token count at a given ``P`` bucket.
    """
    mixer_tail_fwd_kernel(
        gy,
        gu,
        ggate,
        gdskip,
        gweight,
        gout,
        tokens,
        seqlen,
        rows,
        denom,
        eps,
        segments,
        exact,
    ).launch(grid=(tiles, heads, 1), block=(THREADS, 1, 1))


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@cute.kernel
def mixer_tail_bwd_kernel(
    gdout: cute.Tensor,
    gy: cute.Tensor,
    gu: cute.Tensor,
    ggate: cute.Tensor,
    gdskip: cute.Tensor,
    gweight: cute.Tensor,
    gdy: cute.Tensor,
    gdu: cute.Tensor,
    gdgate: cute.Tensor,
    gpartial: cute.Tensor,
    tokens: cutlass.Int32,
    seqlen: cutlass.Int32,
    rows: cutlass.Int32,
    denom: cutlass.Float32,
    eps: cutlass.Float32,
    segments: cutlass.Constexpr,
    exact: cutlass.Constexpr,
) -> None:
    """Pull the cotangent of the tail back to the three operands and both
    parameters.

    Args:
        gdout: ``(B,H,T,P)`` cotangent of the output, operand dtype.
        gy: ``(B,H,T,P)`` scan output, operand dtype.
        gu: ``(B,H,T,P)`` scan input, operand dtype.
        ggate: ``(B,H,T,P)`` gate, operand dtype.
        gdskip: ``(H,P)`` skip scale, parameter dtype.
        gweight: ``(H,P)`` norm scale, parameter dtype.
        gdy: ``(B,H,T,P)`` written, operand dtype.
        gdu: ``(B,H,T,P)`` written, operand dtype.
        gdgate: ``(B,H,T,P)`` written, operand dtype.
        gpartial: ``(SLOTS,tiles,H,P)`` float32, written with this block's
            contribution to both parameter gradients. Every element is written:
            a block whose rows are all out of range stores its zero
            accumulators.
        tokens: ``B*T``. Dynamic.
        seqlen: ``T``. Dynamic.
        rows: ``P``. Dynamic.
        denom: ``P`` as float32.
        eps: The forward's epsilon. Dynamic.
        segments: ``ceil(P/32)``. Compile-time.
        exact: Whether ``P`` is a multiple of the warp width. Compile-time.

    Invariants:
        Both reductions over ``P`` are float32. The norm scale is recomputed from
        the operands rather than read back, so the backward reads no forward
        intermediate. Masked columns carry a zero value and a zero cotangent, so
        they contribute nothing to either accumulator.
    """
    tile, head, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    smem = cutlass.utils.SmemAllocator()
    spartial = smem.allocate_tensor(cutlass.Float32, param_tile(segments).layout(), 16)
    src = gy.element_type
    par = gweight.element_type
    dst = gdy.element_type
    zero = cutlass.Float32(0.0)
    cols = _columns(lane, rows, segments, exact)

    acc_skip: list[Scalar] = [zero] * segments
    acc_weight: list[Scalar] = [zero] * segments

    for step in cutlass.range_constexpr(ROWS_PER_WARP):
        token = tile * ROWS + step * WARPS + warp
        if token < tokens:
            bidx = token // seqlen
            tidx = token - bidx * seqlen

            held: list[tuple[Scalar, ...]] = []
            sumsq = zero
            dot = zero
            for j in cutlass.range_constexpr(segments):
                col, pos, masked = cols[j]
                gate = widen(ggate[bidx, head, tidx, pos], src)
                sig = sigmoid(gate)
                act = silu(gate, sig)
                skip = widen(gdskip[head, pos], par)
                uval = widen(gu[bidx, head, tidx, pos], src)
                pre = widen(gy[bidx, head, tidx, pos], src) + skip * uval
                dout = widen(gdout[bidx, head, tidx, pos], src)
                value = pre * act
                cot = dout * widen(gweight[head, pos], par)
                if cutlass.const_expr(masked):
                    inside = col < rows
                    value = select(inside, value, zero)
                    cot = select(inside, cot, zero)
                sumsq = sumsq + value * value
                dot = dot + cot * value
                held.append(
                    (value, pre, act, silu_grad(gate, sig), uval, skip, dout, cot)
                )

            total = f32(cute.arch.warp_reduction_sum(sumsq))
            paired = f32(cute.arch.warp_reduction_sum(dot))
            scale = f32(cute.rsqrt(total / denom + eps))
            # d(mean square) enters through the norm scale alone, so the whole
            # coupling across the row is this one scalar.
            coupling = scale * scale * scale * paired / denom

            for j in cutlass.range_constexpr(segments):
                col, _, masked = cols[j]
                value, pre, act, dact, uval, skip, dout, cot = held[j]
                dvalue = scale * cot - coupling * value
                dpre = dvalue * act
                acc_skip[j] = acc_skip[j] + dpre * uval
                acc_weight[j] = acc_weight[j] + dout * value * scale
                dy = narrow(dpre, dst)
                du = narrow(dpre * skip, dst)
                dgate = narrow(dvalue * pre * dact, dst)
                if cutlass.const_expr(masked):
                    if col < rows:
                        gdy[bidx, head, tidx, col] = dy
                        gdu[bidx, head, tidx, col] = du
                        gdgate[bidx, head, tidx, col] = dgate
                else:
                    gdy[bidx, head, tidx, col] = dy
                    gdu[bidx, head, tidx, col] = du
                    gdgate[bidx, head, tidx, col] = dgate

    for j in cutlass.range_constexpr(segments):
        col, _, _ = cols[j]
        spartial[SLOT_DSKIP, warp, col] = acc_skip[j]
        spartial[SLOT_WEIGHT, warp, col] = acc_weight[j]
    cute.arch.sync_threads()

    if warp == 0:
        for j in cutlass.range_constexpr(segments):
            col, _, masked = cols[j]
            total_skip = zero
            total_weight = zero
            for other in cutlass.range_constexpr(WARPS):
                total_skip = total_skip + spartial[SLOT_DSKIP, other, col]
                total_weight = total_weight + spartial[SLOT_WEIGHT, other, col]
            if cutlass.const_expr(masked):
                if col < rows:
                    gpartial[SLOT_DSKIP, tile, head, col] = total_skip
                    gpartial[SLOT_WEIGHT, tile, head, col] = total_weight
            else:
                gpartial[SLOT_DSKIP, tile, head, col] = total_skip
                gpartial[SLOT_WEIGHT, tile, head, col] = total_weight


@cute.jit
def mixer_tail_bwd(
    gdout: cute.Tensor,
    gy: cute.Tensor,
    gu: cute.Tensor,
    ggate: cute.Tensor,
    gdskip: cute.Tensor,
    gweight: cute.Tensor,
    gdy: cute.Tensor,
    gdu: cute.Tensor,
    gdgate: cute.Tensor,
    gpartial: cute.Tensor,
    tokens: cutlass.Int32,
    seqlen: cutlass.Int32,
    rows: cutlass.Int32,
    denom: cutlass.Float32,
    eps: cutlass.Float32,
    tiles: cutlass.Int32,
    heads: cutlass.Int32,
    segments: cutlass.Constexpr,
    exact: cutlass.Constexpr,
) -> None:
    """Launch :func:`mixer_tail_bwd_kernel`."""
    mixer_tail_bwd_kernel(
        gdout,
        gy,
        gu,
        ggate,
        gdskip,
        gweight,
        gdy,
        gdu,
        gdgate,
        gpartial,
        tokens,
        seqlen,
        rows,
        denom,
        eps,
        segments,
        exact,
    ).launch(grid=(tiles, heads, 1), block=(THREADS, 1, 1))


# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------


def _segments(rows: int) -> int:
    """Columns one lane owns, ``ceil(P/32)``."""
    return (rows + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE


@functools.cache
def _budget(segments: int) -> int:
    """Bytes the backward's partial tile occupies, against the queried capacity.

    Cached on ``segments`` because the capacity query walks the DSL's
    architecture table and the answer depends on nothing else.

    Args:
        segments: ``ceil(P/32)``.

    Returns:
        The tile's byte count.

    Raises:
        ValueError: If the tile exceeds capacity, which is what bounds ``P``.
            Both directions check it, so a shape the backward cannot
            differentiate is refused by the forward too.
    """
    return assert_smem_fits("mixer_tail_bwd", smem_bytes([(param_tile(segments), 4)]))


def _check_eps(eps: float) -> None:
    """Raises:
    ValueError: If ``eps`` is not positive. The reduction is over ``P`` and the
        summand is non-negative, so ``eps`` is the only thing standing between a
        row of exact zeros and a division by zero.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")


def _check_dtypes(named: Named) -> None:
    """Raises:
    TypeError: If an operand dtype has no kernel path, or if the group does not
        share one dtype. One dtype per group keeps a single widening type
        inside the kernel; the operand group and the parameter group are
        independent, so float32 parameters against low-precision activations is
        one call.
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
                "one dtype per group"
            )


def _check_shapes(
    y: Tensor, u: Tensor, gate: Tensor, d_skip: Tensor, weight: Tensor
) -> tuple[int, int, int, int]:
    """The ``(B, H, T, P)`` shape the five operands agree on.

    Raises:
        ValueError: On a rank or shape mismatch, or on an empty operand. An
            empty operand has no launchable grid, so it is refused rather than
            special-cased.
    """
    if y.ndim != 4:
        raise ValueError(f"y must be (B,H,T,P), got {tuple(y.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in y.shape)
    shape = (bsz, heads, seqlen, rows)
    for name, tensor in (("u", u), ("gate", gate)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(tensor.shape)}")
    for name, tensor in (("d_skip", d_skip), ("weight", weight)):
        if tuple(tensor.shape) != (heads, rows):
            raise ValueError(
                f"{name} must be {(heads, rows)}, got {tuple(tensor.shape)}"
            )
    if bsz * heads * seqlen * rows == 0:
        raise ValueError(f"y must hold at least one element, got {shape}")
    return shape


def _check_operands(
    y: Tensor,
    u: Tensor,
    gate: Tensor,
    d_skip: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[int, int, int, int]:
    """Every host-side guard the two directions share.

    Returns:
        ``(B, H, T, P)``.

    Raises:
        ValueError: On a shape mismatch, an empty operand, a non-positive
            ``eps``, a non-CUDA or non-contiguous operand, or a ``P`` whose
            partial tile exceeds the shared-memory capacity.
        TypeError: On a dtype with no kernel path, or on a group that does not
            share one dtype.
    """
    shape = _check_shapes(y, u, gate, d_skip, weight)
    _check_eps(eps)
    _check_dtypes(((y, "y"), (u, "u"), (gate, "gate")))
    _check_dtypes(((d_skip, "d_skip"), (weight, "weight")))
    check_layout(
        ((y, "y"), (u, "u"), (gate, "gate"), (d_skip, "d_skip"), (weight, "weight"))
    )
    _budget(_segments(shape[3]))
    return shape


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


class MixerTailGrads(NamedTuple):
    """Gradients of the fused tail.

    Attributes:
        dy: ``(B,H,T,P)``, operand dtype.
        du: ``(B,H,T,P)``, operand dtype.
        dgate: ``(B,H,T,P)``, operand dtype.
        dd_skip: ``(H,P)``, parameter dtype.
        dweight: ``(H,P)``, parameter dtype.
    """

    dy: Tensor
    du: Tensor
    dgate: Tensor
    dd_skip: Tensor
    dweight: Tensor


def mixer_tail_forward(
    y: Tensor,
    u: Tensor,
    gate: Tensor,
    d_skip: Tensor,
    weight: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Apply the skip, the gate, and the per-head RMS norm, in one launch.

    Args:
        y: Scan output, ``(B,H,T,P)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`.
        u: Scan input, ``(B,H,T,P)``, same dtype.
        gate: Gate, ``(B,H,T,P)``, same dtype.
        d_skip: Per-row skip scale, ``(H,P)``, one of :data:`slinoss._precision.KERNEL_DTYPES`.
        weight: Per-row norm scale, ``(H,P)``, same dtype as ``d_skip``.
        eps: Added to the mean square before the reciprocal square root.

    Returns:
        ``(B,H,T,P)`` in the dtype of ``y``.

    Raises:
        ValueError: On a shape mismatch, an empty operand, a non-positive
            ``eps``, a non-CUDA or non-contiguous operand, or a ``P`` whose
            partial tile exceeds the shared-memory capacity.
        TypeError: On a dtype with no kernel path, or on a group that does not
            share one dtype.
    """
    bsz, heads, seqlen, rows = _check_operands(y, u, gate, d_skip, weight, eps)
    out = torch.empty_like(y)
    tokens = bsz * seqlen
    mixer_tail_fwd(
        dev_tensor(y),
        dev_tensor(u),
        dev_tensor(gate),
        dev_tensor(d_skip),
        dev_tensor(weight),
        dev_tensor(out),
        tokens,
        seqlen,
        rows,
        float(rows),
        float(eps),
        (tokens + ROWS - 1) // ROWS,
        heads,
        _segments(rows),
        rows % cute.arch.WARP_SIZE == 0,
    )
    return out


def mixer_tail_backward(
    dout: Tensor,
    y: Tensor,
    u: Tensor,
    gate: Tensor,
    d_skip: Tensor,
    weight: Tensor,
    *,
    eps: float,
) -> MixerTailGrads:
    """Pull the cotangent of the tail back to all five operands, in one launch.

    The norm scale is recomputed from the operands, so the forward saves no
    intermediate. Both parameter gradients are accumulated in the kernel's
    epilogue as one ``(H,P)`` row of tile partials; the cross-tile sum is a
    reduction over that buffer and reads no operand a second time.

    Args:
        dout: Cotangent of the output, ``(B,H,T,P)``, contiguous CUDA, dtype of
            ``y``.
        y: The forward's scan output, ``(B,H,T,P)``.
        u: The forward's scan input, ``(B,H,T,P)``.
        gate: The forward's gate, ``(B,H,T,P)``.
        d_skip: The forward's skip scale, ``(H,P)``.
        weight: The forward's norm scale, ``(H,P)``.
        eps: The epsilon the forward was called with.

    Returns:
        A :class:`MixerTailGrads`.

    Raises:
        ValueError: On a shape mismatch, an empty operand, a non-positive
            ``eps``, a non-CUDA or non-contiguous operand, or a ``P`` whose
            partial tile exceeds the shared-memory capacity.
        TypeError: On a dtype with no kernel path, or on a group that does not
            share one dtype.
    """
    bsz, heads, seqlen, rows = _check_operands(y, u, gate, d_skip, weight, eps)
    if tuple(dout.shape) != (bsz, heads, seqlen, rows):
        raise ValueError(
            f"dout must be {(bsz, heads, seqlen, rows)}, got {tuple(dout.shape)}"
        )
    _check_dtypes(((y, "y"), (dout, "dout")))
    check_layout(((dout, "dout"),))

    dy = torch.empty_like(y)
    du = torch.empty_like(y)
    dgate = torch.empty_like(y)
    tokens = bsz * seqlen
    tiles = (tokens + ROWS - 1) // ROWS
    # Every element is written by the kernel, so the accumulator is initialized
    # inside the launch and nothing is filled here.
    partial = torch.empty(
        SLOTS, tiles, heads, rows, dtype=torch.float32, device=y.device
    )
    mixer_tail_bwd(
        dev_tensor(dout),
        dev_tensor(y),
        dev_tensor(u),
        dev_tensor(gate),
        dev_tensor(d_skip),
        dev_tensor(weight),
        dev_tensor(dy),
        dev_tensor(du),
        dev_tensor(dgate),
        dev_tensor(partial),
        tokens,
        seqlen,
        rows,
        float(rows),
        float(eps),
        tiles,
        heads,
        _segments(rows),
        rows % cute.arch.WARP_SIZE == 0,
    )
    totals = partial.sum(1)
    return MixerTailGrads(
        dy=dy,
        du=du,
        dgate=dgate,
        dd_skip=totals[SLOT_DSKIP].to(d_skip.dtype),
        dweight=totals[SLOT_WEIGHT].to(weight.dtype),
    )


class MixerTailFunction(torch.autograd.Function):
    """Differentiable fused tail.

    Saves the five operands. The backward recomputes the gate, the pre-norm
    value, and the norm scale from them, so no forward intermediate crosses the
    boundary. All five gradients are produced whatever the leaves require: a
    variant per subset of ``needs_input_grad`` would be a second entry point
    into the same kernel.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        y: Tensor,
        u: Tensor,
        gate: Tensor,
        d_skip: Tensor,
        weight: Tensor,
        eps: float,
    ) -> Tensor:
        out = mixer_tail_forward(y, u, gate, d_skip, weight, eps=eps)
        ctx.save_for_backward(y, u, gate, d_skip, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dout: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None]:
        y, u, gate, d_skip, weight = ctx.saved_tensors
        grads = mixer_tail_backward(dout, y, u, gate, d_skip, weight, eps=ctx.eps)
        return (
            grads.dy,
            grads.du,
            grads.dgate,
            grads.dd_skip,
            grads.dweight,
            None,
        )


def mixer_tail(
    y: Tensor,
    u: Tensor,
    gate: Tensor,
    d_skip: Tensor,
    weight: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Fused tail with an analytic backward. The public fast path.

    Args:
        y: Scan output, ``(B,H,T,P)``.
        u: Scan input, ``(B,H,T,P)``.
        gate: Gate, ``(B,H,T,P)``.
        d_skip: Per-row skip scale, ``(H,P)``.
        weight: Per-row norm scale, ``(H,P)``.
        eps: Added to the mean square before the reciprocal square root.

    Returns:
        ``(B,H,T,P)`` in the dtype of ``y``.

    Raises:
        ValueError: On a shape, layout, device, or epsilon violation.
        TypeError: On an unsupported or mixed dtype.
    """
    return cast("Tensor", MixerTailFunction.apply(y, u, gate, d_skip, weight, eps))
