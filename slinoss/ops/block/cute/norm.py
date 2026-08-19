"""RMS norm, plain and fused with the residual add. CuTe DSL both directions.

    normed = x * rsqrt(mean(x^2) + eps) * weight

The fused form adds the incoming residual stream first and hands the sum back:

    s      = x + residual
    normed = s * rsqrt(mean(s^2) + eps) * weight

``s`` is returned float32 at every operand width. That is what the reference
means by wide: its accumulation dtype is float32 unless an operand is float64,
and float64 reaches no kernel. A stack therefore carries its residual at float32
instead of narrowing once per block.

Parallel decomposition, forward. One block per row over the flattened ``B*T``
axis, 256 threads. The reduction is over ``D`` only, so no row shares anything
with another. Each thread strides over ``D`` accumulating in float32, the lanes
of a warp are combined by a shuffle add-scan, and thread 0 sums the eight warp
totals into one float32 shared slot that every thread then reads. ``D`` is
compile-time, so the strided loop needs no bounds predicate: a thread past the
end runs zero iterations and contributes the scan identity.

The second pass re-reads its input rather than holding the row in registers.
``D`` reaches 4096, so staging it would cost occupancy, and the re-read hits L1
or L2. In the fused form the second pass reads the wide sum it has just written,
so the add is evaluated once and ``normed`` is a function of the residual that is
returned rather than of a second summation.

Parallel decomposition, backward. Both parameter gradients reduce over rows, so
the grid is fixed at twice the SM count and each block strides over the row axis,
which bounds the partial buffer at one float32 row per block instead of one per
row. Every block covers the same columns on every row it runs, so a thread's
``ceil(D/256)`` weight-gradient accumulators stay in registers across the whole
stride loop and the epilogue is one store per column. There are no atomics and no
second pass over the operands. The trip count of the stride loop is
block-uniform, which is what makes the barriers inside the row reduction safe.
Under twice the SM count the grid is the row count: splitting a row across blocks
would put the row reduction across a grid barrier, so the row count is the whole
available parallelism at this decomposition.

The backward's row loop needs the sum of squares and the dot product of the
cotangent with the row, so the block reduction carries two accumulators through
one pair of barriers rather than running twice.

Nothing is saved from the forward. The row scale is recomputed from ``x`` (and
``residual``) with the expression the forward uses, which is why the two
directions share :func:`_block_totals` and :func:`_scale_of`.

Shared memory: one float32 partial per warp per accumulator plus one float32
broadcast slot per accumulator, so 36 B forward and 72 B backward, asserted
against the queried capacity by a test. One lane per warp writes one partial, so
the writes land in distinct banks, and every broadcast read is one address across
the block; neither needs a swizzle.

DRAM-bound, both directions. Analytic traffic per row, at operand itemsize ``i``
and cotangent itemsize ``i_c``, with no measured bandwidth claimed here:

- ``rmsnorm_fwd``: ``2*D*i``, plus the ``4*D`` float32 weight.
- ``rmsnorm_residual_fwd``: ``D*(i_x + i_residual + 4 + 4 + i_normed)`` -- one
  read of ``x``, one of ``residual``, one write and one read of the wide sum, one
  write of ``normed``.
- ``rmsnorm_bwd``: ``D*(i_c + i_x + i_x)`` -- one read of the cotangent, one of
  ``x``, one write of ``dx``.
- ``rmsnorm_residual_bwd``: ``D*(i_c + 4 + i_x + i_residual + i_x + i_residual)``
  -- the two cotangents, ``x`` and ``residual``, then ``dx`` and ``dresidual``.
  An absent cotangent or an absent residual drops its own terms.

Both backwards add the ``4*D`` float32 weight and ``4*D`` float32 of partials per
block, written once and read once by the host-side sum.
"""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Scalar,
    Tile,
    cute_dtype,
    dev_tensor,
    f32,
    jit_launch,
    narrow,
    select,
    shuffle_up,
    smem_bytes,
    widen,
)
from slinoss._guard import check_dtypes, check_layout
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.block.reference import NormResidual, NormResidualGrads, RMSNormGrads

__all__ = [
    "BWD_SLOTS",
    "FWD_SLOTS",
    "NORM_THREADS",
    "SLOT_DOT",
    "SLOT_SUMSQ",
    "WARPS",
    "check_operand",
    "norm_smem_bytes",
    "reduce_tile",
    "rmsnorm_backward",
    "rmsnorm_bwd",
    "rmsnorm_bwd_kernel",
    "rmsnorm_forward",
    "rmsnorm_fwd",
    "rmsnorm_fwd_kernel",
    "rmsnorm_residual_backward",
    "rmsnorm_residual_bwd",
    "rmsnorm_residual_bwd_kernel",
    "rmsnorm_residual_forward",
    "rmsnorm_residual_fwd",
    "rmsnorm_residual_fwd_kernel",
    "row_blocks",
    "sm_count",
    "total_tile",
]

NORM_THREADS = 256
"""Block width of every norm kernel. Eight warps, one row reduced at a time."""

WARPS = NORM_THREADS // cute.arch.WARP_SIZE
"""Warps per block, and therefore float32 partials per accumulator."""

SLOT_SUMSQ = 0
"""Reduction slot holding the row's sum of squares. Both directions."""

SLOT_DOT = 1
"""Reduction slot holding the backward's cotangent dot product."""

FWD_SLOTS = 1
"""Accumulators the forward reduces: the sum of squares alone."""

BWD_SLOTS = 2
"""Accumulators the backward reduces: the sum of squares and the dot product."""


def reduce_tile(slots: int) -> Tile:
    """Per-warp partials for ``slots`` independent block reductions.

    Flat rather than ``(slots, WARPS)``: slot ``s`` occupies ``s*WARPS`` onward,
    and one lane per warp writes one element, so consecutive warps of one slot
    land in consecutive banks.

    Args:
        slots: Accumulators reduced together.

    Returns:
        The tile.
    """
    return Tile((slots * WARPS,), (1,))


def total_tile(slots: int) -> Tile:
    """Broadcast slot per block reduction, holding the summed accumulator.

    Args:
        slots: Accumulators reduced together.

    Returns:
        The tile.
    """
    return Tile((slots,), (1,))


def norm_smem_bytes(slots: int) -> int:
    """Shared memory a norm kernel holds, in bytes, from the tile layouts.

    Args:
        slots: :data:`FWD_SLOTS` or :data:`BWD_SLOTS`.

    Returns:
        Total bytes.
    """
    return smem_bytes([(reduce_tile(slots), 4), (total_tile(slots), 4)])


@cache
def sm_count(index: int) -> int:
    """Multiprocessors on one CUDA device. Cached: the grid is sized per launch.

    Args:
        index: CUDA device ordinal.

    Returns:
        The multiprocessor count.
    """
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def row_blocks(rows: int, index: int) -> int:
    """Blocks a backward launch uses over ``rows`` rows.

    Twice the SM count is the block-count floor, and the row stride loop covers
    any row count from that one grid. Fewer rows than that floor caps the grid at
    the row count: a row reduction cannot cross a grid barrier, so the row count
    is the whole available parallelism at this decomposition.

    Args:
        rows: Rows on the flattened axis. At least one.
        index: CUDA device ordinal.

    Returns:
        The block count, which is also the row stride.
    """
    return min(rows, 2 * sm_count(index))


def _warp_offsets() -> tuple[int, ...]:
    """Shuffle distances of an inclusive add-scan over a full warp."""
    offsets: list[int] = []
    reach = 1
    while reach < cute.arch.WARP_SIZE:
        offsets.append(reach)
        reach *= 2
    return tuple(offsets)


WARP_OFFSETS = _warp_offsets()


# ---------------------------------------------------------------------------
# Device math
# ---------------------------------------------------------------------------


def _warp_total(value: Scalar, lane: cutlass.Int32) -> Scalar:
    """Sum one float32 across a full warp. The last lane holds the total.

    An inclusive add-scan by up-shuffles, guarded by a select so that a lane
    below the shuffle distance keeps its own partial instead of doubling it. The
    up direction is used because it is the one full-warp shuffle whose clamp
    field is already pinned in :mod:`slinoss._cute`.

    Args:
        value: The lane's partial sum.
        lane: Lane index within the warp.

    Returns:
        The warp total in lane ``WARP_SIZE - 1``, a partial prefix elsewhere.
    """
    for offset in WARP_OFFSETS:
        shifted = shuffle_up(value, offset)
        value = select(lane >= offset, value + shifted, value)
    return value


@cute.jit
def _block_totals(
    spart: cute.Tensor,
    stotal: cute.Tensor,
    first: Scalar,
    second: Scalar,
    tid: cutlass.Int32,
    slots: cutlass.Constexpr,
) -> None:
    """Sum one or two float32 accumulators across the block.

    Entered by the whole block. Both barriers are here rather than in the caller
    because both tiles are private to this reduction, and the trailing barrier is
    what makes every slot readable by every thread on return.

    Two accumulators share the barriers instead of reducing in sequence: the
    backward needs the sum of squares and the cotangent dot product of the same
    row, and a second call would double the barrier count. A caller with one
    accumulator passes it twice and asks for one slot, which leaves slot 1
    unwritten and unread.

    Args:
        spart: :func:`reduce_tile` of ``slots``, float32.
        stotal: :func:`total_tile` of ``slots``, float32. Written with the block
            totals.
        first: The thread's contribution to slot :data:`SLOT_SUMSQ`.
        second: Its contribution to slot :data:`SLOT_DOT`. Read only when
            ``slots`` exceeds one.
        tid: Thread index within the block.
        slots: :data:`FWD_SLOTS` or :data:`BWD_SLOTS`. Compile-time.
    """
    warp = tid // cute.arch.WARP_SIZE
    lane = tid - warp * cute.arch.WARP_SIZE
    total = _warp_total(first, lane)
    # Bound before the trace-time branch, not inside it: a name that exists on one
    # side of a `const_expr` and not the other is unbound to every reader of the
    # source, whatever the trace does with it.
    paired = second
    if cutlass.const_expr(slots > 1):
        paired = _warp_total(second, lane)
    if lane == cute.arch.WARP_SIZE - 1:
        spart[warp] = total
        if cutlass.const_expr(slots > 1):
            spart[WARPS + warp] = paired
    cute.arch.sync_threads()

    if tid == 0:
        for slot in cutlass.range_constexpr(slots):
            block = cutlass.Float32(0.0)
            for index in cutlass.range_constexpr(WARPS):
                block = block + spart[slot * WARPS + index]
            stotal[slot] = block
    cute.arch.sync_threads()


def _scale_of(total: Scalar, eps: cutlass.Float32, width: int) -> Scalar:
    """``rsqrt(mean + eps)`` from a row's sum of squares.

    Evaluated by every thread from the one broadcast slot rather than once per
    block: the input is block-uniform, so the answers agree bit for bit, and a
    second shared slot to carry the scale would need a third barrier.

    Args:
        total: The row's sum of squares.
        eps: Added to the mean square. Positive, so the argument is positive even
            on an all-zero row.
        width: ``D``. Compile-time.

    Returns:
        The row scale.
    """
    return f32(cute.rsqrt(total / cutlass.Float32(float(width)) + eps))


def _slice(tid: Any, width: int, threads: int) -> list[tuple[Any, Any, bool]]:
    """Per-segment ``(column, read position, masked)`` for one thread.

    The backward holds one weight-gradient accumulator per segment in registers,
    which needs a compile-time segment count; the forward's dynamic strided loop
    has none. ``D`` is compile-time, so the count is too. Only the last segment
    can run past ``D``, and it reads a clamped position and drops the value with a
    select, which keeps the load in bounds without a branch.

    Args:
        tid: Thread index within the block.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.

    Returns:
        One entry per segment, outermost first.
    """
    segments = -(-width // threads)
    ragged = width % threads != 0
    out: list[tuple[Any, Any, bool]] = []
    for j in range(segments):
        col = j * threads + tid
        masked = ragged and j == segments - 1
        out.append((col, cutlass.min(col, width - 1) if masked else col, masked))
    return out


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@cute.kernel
def rmsnorm_fwd_kernel(
    gx: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Normalize one row per block.

    Args:
        gx: ``(rows, D)`` input, activation dtype.
        gw: ``(D,)`` float32 weight (I4).
        gy: ``(rows, D)`` output, dtype of ``gx``.
        eps: Added to the mean square. Dynamic, so one variant covers every
            epsilon.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.

    Invariants:
        The mean square is accumulated in float32 whatever the operand width, and
        the weight is float32, so only the store narrows.
    """
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(cutlass.Float32, reduce_tile(FWD_SLOTS).layout(), 16)
    stotal = smem.allocate_tensor(cutlass.Float32, total_tile(FWD_SLOTS).layout(), 16)

    src = gx.element_type
    dst = gy.element_type
    acc = cutlass.Float32(0.0)
    for d in cutlass.range(tid, width, threads):
        value = widen(gx[row, d], src)
        acc = acc + value * value

    _block_totals(spart, stotal, acc, acc, tid, FWD_SLOTS)
    scale = _scale_of(stotal[SLOT_SUMSQ], eps, width)
    for d in cutlass.range(tid, width, threads):
        gy[row, d] = narrow(widen(gx[row, d], src) * scale * gw[d], dst)


@cute.kernel
def rmsnorm_residual_fwd_kernel(
    gx: cute.Tensor,
    gres: cute.Tensor,
    gsum: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    has_residual: cutlass.Constexpr,
) -> None:
    """Add the residual, write the wide sum, then normalize it.

    Args:
        gx: ``(rows, D)`` branch output, activation dtype.
        gres: ``(rows, D)`` incoming residual stream. Read only when
            ``has_residual``; the first-block variant is handed ``gx`` here so
            the signature has one form.
        gsum: ``(rows, D)`` float32, written with ``x + residual`` and read back
            by the second pass.
        gw: ``(D,)`` float32 weight (I4).
        gy: ``(rows, D)`` normed output, dtype of ``gx``.
        eps: Added to the mean square. Dynamic.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.
        has_residual: Whether a residual stream is supplied. Compile-time.

    Invariants:
        The sum is formed once, in float32, and both outputs derive from that one
        value. Each thread reads back only addresses it wrote itself.
    """
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(cutlass.Float32, reduce_tile(FWD_SLOTS).layout(), 16)
    stotal = smem.allocate_tensor(cutlass.Float32, total_tile(FWD_SLOTS).layout(), 16)

    src = gx.element_type
    rsrc = gres.element_type
    dst = gy.element_type
    acc = cutlass.Float32(0.0)
    for d in cutlass.range(tid, width, threads):
        value = widen(gx[row, d], src)
        if cutlass.const_expr(has_residual):
            value = value + widen(gres[row, d], rsrc)
        gsum[row, d] = value
        acc = acc + value * value

    _block_totals(spart, stotal, acc, acc, tid, FWD_SLOTS)
    scale = _scale_of(stotal[SLOT_SUMSQ], eps, width)
    for d in cutlass.range(tid, width, threads):
        gy[row, d] = narrow(gsum[row, d] * scale * gw[d], dst)


@cute.kernel
def rmsnorm_bwd_kernel(
    gdout: cute.Tensor,
    gx: cute.Tensor,
    gw: cute.Tensor,
    gdx: cute.Tensor,
    gpartial: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    span: cutlass.Int32,
    dtype: cutlass.Constexpr,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Pull the cotangent of the norm back to the input and the weight.

    The pullback of ``n_i = s_i * r * w_i`` with ``r = rsqrt(mean(s^2) + eps)`` is

        dx_i     = r * c_i - r^3 * <c, s> * s_i / D,   c_i = dn_i * w_i
        dweight_j = sum over rows of dn_j * s_j * r

    so the whole coupling across a row is the one scalar ``r^3 <c,s> / D``.

    Args:
        gdout: ``(rows, D)`` cotangent of the output, ``dtype``.
        gx: ``(rows, D)`` the forward's input, ``dtype``.
        gw: ``(D,)`` float32 weight (I4).
        gdx: ``(rows, D)`` written, ``dtype``.
        gpartial: ``(blocks, D)`` float32, written with this block's contribution
            to the weight gradient. Every element is written, so nothing is filled
            on the host.
        eps: The forward's epsilon. Dynamic.
        rows: Rows on the flattened axis. Dynamic.
        span: Rows one grid step advances, the block count. Dynamic.
        dtype: Element type of the cotangent, the input, and ``dx``, which are one
            width by contract. Compile-time and passed rather than read off the
            tensor because it keys the executor cache in
            :func:`slinoss._cute.jit_launch`; a property that shapes the generated
            code and is not in that key lets two widths share one executor.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.

    Invariants:
        Both reductions are float32 whatever the operand width. The row scale is
        recomputed here, so the forward saves nothing. The stride loop's trip
        count is block-uniform, which is what makes the barriers inside the
        reduction safe. Masked columns carry a zero value and a zero cotangent, so
        they reach neither accumulator.
    """
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(cutlass.Float32, reduce_tile(BWD_SLOTS).layout(), 16)
    stotal = smem.allocate_tensor(cutlass.Float32, total_tile(BWD_SLOTS).layout(), 16)

    zero = cutlass.Float32(0.0)
    denom = cutlass.Float32(float(width))
    cols = _slice(tid, width, threads)
    segments = len(cols)
    # A fragment rather than a list of scalars: the accumulator is carried across
    # a dynamic loop, so it has to be addressable storage and not a rebound value.
    # The fill is the initialization, inside the kernel.
    dweight = cute.make_fragment((segments,), cutlass.Float32)
    dweight.fill(0.0)

    for row in cutlass.range(block, rows, span):
        sumsq = zero
        dot = zero
        for j in cutlass.range_constexpr(segments):
            col, pos, masked = cols[j]
            value = widen(gx[row, pos], dtype)
            cot = widen(gdout[row, pos], dtype) * gw[pos]
            if cutlass.const_expr(masked):
                inside = col < width
                value = select(inside, value, zero)
                cot = select(inside, cot, zero)
            sumsq = sumsq + value * value
            dot = dot + cot * value

        _block_totals(spart, stotal, sumsq, dot, tid, BWD_SLOTS)
        scale = _scale_of(stotal[SLOT_SUMSQ], eps, width)
        coupling = scale * scale * scale * stotal[SLOT_DOT] / denom

        for j in cutlass.range_constexpr(segments):
            col, pos, masked = cols[j]
            value = widen(gx[row, pos], dtype)
            dnorm = widen(gdout[row, pos], dtype)
            if cutlass.const_expr(masked):
                inside = col < width
                value = select(inside, value, zero)
                dnorm = select(inside, dnorm, zero)
            dweight[j] = dweight[j] + dnorm * value * scale
            dvalue = narrow(scale * dnorm * gw[pos] - coupling * value, dtype)
            if cutlass.const_expr(masked):
                if col < width:
                    gdx[row, col] = dvalue
            else:
                gdx[row, col] = dvalue

    for j in cutlass.range_constexpr(segments):
        col, _, masked = cols[j]
        if cutlass.const_expr(masked):
            if col < width:
                gpartial[block, col] = dweight[j]
        else:
            gpartial[block, col] = dweight[j]


@cute.kernel
def rmsnorm_residual_bwd_kernel(
    gdnormed: cute.Tensor,
    gdres: cute.Tensor,
    gx: cute.Tensor,
    gres: cute.Tensor,
    gw: cute.Tensor,
    gdx: cute.Tensor,
    gdresout: cute.Tensor,
    gpartial: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    span: cutlass.Int32,
    dtype: cutlass.Constexpr,
    rdtype: cutlass.Constexpr,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    has_residual: cutlass.Constexpr,
    has_normed: cutlass.Constexpr,
    has_dres: cutlass.Constexpr,
) -> None:
    """Pull both cotangents of the fused form back to the three inputs.

    The forward's two outputs share the sum ``s = x + residual``, so both
    cotangents meet in one ``ds``: the normed half is the pullback in
    :func:`rmsnorm_bwd_kernel` and the residual half enters unchanged. ``x`` and
    ``residual`` then take the same ``ds``, differing only in width.

    Args:
        gdnormed: ``(rows, D)`` cotangent of ``normed``, ``dtype``. Read only when
            ``has_normed``; the variant that does not read it is handed ``gx``, so
            the signature has one form.
        gdres: ``(rows, D)`` float32 cotangent of the wide residual, which the
            forward returns at float32 whatever its operands. Read only when
            ``has_dres``.
        gx: ``(rows, D)`` the forward's branch output, ``dtype``.
        gres: ``(rows, D)`` the forward's incoming stream, ``rdtype``. Read only
            when ``has_residual``.
        gw: ``(D,)`` float32 weight (I4).
        gdx: ``(rows, D)`` written, ``dtype``.
        gdresout: ``(rows, D)`` written when ``has_residual``, ``rdtype``.
        gpartial: ``(blocks, D)`` float32, written when ``has_normed``. Every
            element is written, so nothing is filled on the host.
        eps: The forward's epsilon. Dynamic.
        rows: Rows on the flattened axis. Dynamic.
        span: Rows one grid step advances, the block count. Dynamic.
        dtype: Element type of ``x``, of the cotangent of ``normed``, and of
            ``dx``, which are one width by contract. Compile-time for the reason
            given in :func:`rmsnorm_bwd_kernel`.
        rdtype: Element type of the incoming stream and of its gradient.
            Independent of ``dtype``: the stream arrives float32 while the branch
            output is low precision. Compile-time.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.
        has_residual: Whether the forward was given a residual stream, and
            therefore whether ``dresidual`` is an output. Compile-time.
        has_normed: Whether ``normed`` carries a cotangent. Compile-time.
        has_dres: Whether the wide residual carries a cotangent. Compile-time.

    Invariants:
        At least one of ``has_normed`` and ``has_dres`` holds; with neither there
        is no gradient and the host returns without launching. The sum is
        recomputed from ``x`` and ``residual``, so the forward's wide output is
        not read back. An absent cotangent contributes nothing rather than a zero
        tensor: its half of the expression is closed at compile time.
    """
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(cutlass.Float32, reduce_tile(BWD_SLOTS).layout(), 16)
    stotal = smem.allocate_tensor(cutlass.Float32, total_tile(BWD_SLOTS).layout(), 16)

    zero = cutlass.Float32(0.0)
    denom = cutlass.Float32(float(width))
    cols = _slice(tid, width, threads)
    segments = len(cols)
    dweight = cute.make_fragment((segments,), cutlass.Float32)
    dweight.fill(0.0)

    for row in cutlass.range(block, rows, span):
        # Bound before the trace-time branch for the reason given in
        # `_block_totals`. Neither reaches a store unless `has_normed` holds.
        scale = zero
        coupling = zero
        if cutlass.const_expr(has_normed):
            sumsq = zero
            dot = zero
            for j in cutlass.range_constexpr(segments):
                col, pos, masked = cols[j]
                value = widen(gx[row, pos], dtype)
                if cutlass.const_expr(has_residual):
                    value = value + widen(gres[row, pos], rdtype)
                cot = widen(gdnormed[row, pos], dtype) * gw[pos]
                if cutlass.const_expr(masked):
                    inside = col < width
                    value = select(inside, value, zero)
                    cot = select(inside, cot, zero)
                sumsq = sumsq + value * value
                dot = dot + cot * value

            _block_totals(spart, stotal, sumsq, dot, tid, BWD_SLOTS)
            scale = _scale_of(stotal[SLOT_SUMSQ], eps, width)
            coupling = scale * scale * scale * stotal[SLOT_DOT] / denom

        for j in cutlass.range_constexpr(segments):
            col, pos, masked = cols[j]
            total = zero
            if cutlass.const_expr(has_normed):
                value = widen(gx[row, pos], dtype)
                if cutlass.const_expr(has_residual):
                    value = value + widen(gres[row, pos], rdtype)
                dnorm = widen(gdnormed[row, pos], dtype)
                if cutlass.const_expr(masked):
                    inside = col < width
                    value = select(inside, value, zero)
                    dnorm = select(inside, dnorm, zero)
                dweight[j] = dweight[j] + dnorm * value * scale
                total = scale * dnorm * gw[pos] - coupling * value
            if cutlass.const_expr(has_dres):
                total = total + gdres[row, pos]
            if cutlass.const_expr(masked):
                if col < width:
                    gdx[row, col] = narrow(total, dtype)
                    if cutlass.const_expr(has_residual):
                        gdresout[row, col] = narrow(total, rdtype)
            else:
                gdx[row, col] = narrow(total, dtype)
                if cutlass.const_expr(has_residual):
                    gdresout[row, col] = narrow(total, rdtype)

    if cutlass.const_expr(has_normed):
        for j in cutlass.range_constexpr(segments):
            col, _, masked = cols[j]
            if cutlass.const_expr(masked):
                if col < width:
                    gpartial[block, col] = dweight[j]
            else:
                gpartial[block, col] = dweight[j]


@cute.jit
def rmsnorm_fwd(
    gx: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_fwd_kernel`, one block per row."""
    rmsnorm_fwd_kernel(gx, gw, gy, eps, width, threads).launch(
        grid=(rows, 1, 1), block=(threads, 1, 1)
    )


@cute.jit
def rmsnorm_residual_fwd(
    gx: cute.Tensor,
    gres: cute.Tensor,
    gsum: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    has_residual: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_residual_fwd_kernel`, one block per row."""
    rmsnorm_residual_fwd_kernel(
        gx, gres, gsum, gw, gy, eps, width, threads, has_residual
    ).launch(grid=(rows, 1, 1), block=(threads, 1, 1))


@cute.jit
def rmsnorm_bwd(
    gdout: cute.Tensor,
    gx: cute.Tensor,
    gw: cute.Tensor,
    gdx: cute.Tensor,
    gpartial: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    blocks: cutlass.Int32,
    dtype: cutlass.Constexpr,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_bwd_kernel` over a fixed grid.

    ``blocks`` is the grid extent and the row stride both, so the kernel takes one
    launch-geometry argument rather than two that must agree.
    """
    rmsnorm_bwd_kernel(
        gdout, gx, gw, gdx, gpartial, eps, rows, blocks, dtype, width, threads
    ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1))


@cute.jit
def rmsnorm_residual_bwd(
    gdnormed: cute.Tensor,
    gdres: cute.Tensor,
    gx: cute.Tensor,
    gres: cute.Tensor,
    gw: cute.Tensor,
    gdx: cute.Tensor,
    gdresout: cute.Tensor,
    gpartial: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    blocks: cutlass.Int32,
    dtype: cutlass.Constexpr,
    rdtype: cutlass.Constexpr,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    has_residual: cutlass.Constexpr,
    has_normed: cutlass.Constexpr,
    has_dres: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_residual_bwd_kernel` over a fixed grid."""
    rmsnorm_residual_bwd_kernel(
        gdnormed,
        gdres,
        gx,
        gres,
        gw,
        gdx,
        gdresout,
        gpartial,
        eps,
        rows,
        blocks,
        dtype,
        rdtype,
        width,
        threads,
        has_residual,
        has_normed,
        has_dres,
    ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1))


# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------


def check_operand(tensor: Tensor, name: str) -> None:
    """Reject an operand no block kernel can read. Shared by both modules.

    The layout half comes from :func:`slinoss._guard.check_layout`; the dtype
    policy is the block's own, and is wider than the scan's because these kernels
    are rowwise and read float32 natively.

    One operand at a time, so the shared checker's second half is inert here: the
    block kernels widen each operand on load independently, so a bfloat16 input
    against a float32 cotangent is a supported call rather than a mixed group.

    Args:
        tensor: The operand.
        name: Name used in the message.

    Raises:
        ValueError: If the tensor is off CUDA or not contiguous.
        TypeError: If the dtype has no kernel path.
    """
    named = ((tensor, name),)
    check_layout(named)
    check_dtypes(named, KERNEL_DTYPES, "kernel dtypes")


def _check_norm(x: Tensor, weight: Tensor, eps: float) -> tuple[int, int]:
    """Validate the operands shared by both norm entry points.

    Args:
        x: Input, ``(..., D)``.
        weight: ``(D,)`` float32.
        eps: Added to the mean square.

    Returns:
        ``(rows, D)``, the flattened extents the launch uses.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, or a non-CUDA or non-contiguous
            operand. An empty operand has no launchable grid, so it is refused
            rather than special-cased.
        TypeError: On a dtype with no kernel path.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one axis")
    if x.numel() == 0:
        raise ValueError(f"x must hold at least one row, got {tuple(x.shape)}")
    width = int(x.shape[-1])
    if tuple(weight.shape) != (width,):
        raise ValueError(f"weight must be ({width},), got {tuple(weight.shape)}")
    if weight.dtype is not torch.float32:
        raise ValueError(f"weight must be float32 (I4), got {weight.dtype}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    check_operand(x, "x")
    check_operand(weight, "weight")
    return x.numel() // width, width


def _check_shape(tensor: Tensor, x: Tensor, name: str) -> None:
    """Reject an operand whose shape is not the shape of ``x``.

    Args:
        tensor: The operand.
        x: The tensor whose shape it must carry.
        name: Name used in the message.

    Raises:
        ValueError: On a shape mismatch.
    """
    if tuple(tensor.shape) != tuple(x.shape):
        raise ValueError(f"{name} must be {tuple(x.shape)}, got {tuple(tensor.shape)}")


def _check_stream(x: Tensor, residual: Tensor | None) -> None:
    """Validate the optional incoming residual stream against the branch output.

    Its dtype is deliberately unconstrained beyond the kernel set: the stream
    arrives float32 while the branch output is low precision.

    Args:
        x: Branch output, ``(..., D)``.
        residual: Incoming residual stream, or None for the first block.

    Raises:
        ValueError: On a shape mismatch, or a non-CUDA or non-contiguous stream.
        TypeError: On a dtype with no kernel path.
    """
    if residual is not None:
        _check_shape(residual, x, "residual")
        check_operand(residual, "residual")


def _check_cotangent(cot: Tensor, x: Tensor, name: str) -> None:
    """Validate a cotangent that carries the dtype of ``x``.

    The output whose cotangent this is carries the dtype of ``x``, so a cotangent
    at another width is a caller error rather than a case to convert: converting
    it would hide the mismatch and cost a whole-tensor pass.

    Args:
        cot: The cotangent.
        x: The forward's input.
        name: Name used in the message.

    Raises:
        ValueError: On a shape mismatch, or a non-CUDA or non-contiguous
            cotangent.
        TypeError: On a dtype with no kernel path, or on a dtype other than that
            of ``x``.
    """
    _check_shape(cot, x, name)
    if cot.dtype is not x.dtype:
        raise TypeError(f"{name} is {cot.dtype} and x is {x.dtype}; one dtype per call")
    check_operand(cot, name)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def rmsnorm_forward(x: Tensor, weight: Tensor, *, eps: float) -> Tensor:
    """RMS norm over the trailing axis, in one launch.

    Args:
        x: Shape ``(..., D)``, contiguous CUDA, one of :data:`slinoss._precision.KERNEL_DTYPES`.
        weight: Shape ``(D,)`` float32, contiguous CUDA.
        eps: Added to the mean square. Positive.

    Returns:
        Shape ``(..., D)``, dtype and layout of ``x``.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, or a non-CUDA or non-contiguous
            operand.
        TypeError: On a dtype with no kernel path.
    """
    rows, width = _check_norm(x, weight, eps)
    out = torch.empty_like(x)
    jit_launch(
        rmsnorm_fwd,
        (x.view(rows, width), weight, out.view(rows, width), float(eps), rows),
        (width, NORM_THREADS),
    )
    return out


def rmsnorm_residual_forward(
    x: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    *,
    eps: float,
) -> NormResidual:
    """Add the residual and normalize the sum, in one launch.

    Args:
        x: Branch output, ``(..., D)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`.
        residual: Incoming residual stream, same shape, one of
            :data:`slinoss._precision.KERNEL_DTYPES`, or None for the first block of a stack. Its
            dtype is independent of ``x``: the stream arrives float32 while the
            branch output is low precision.
        weight: Shape ``(D,)`` float32, contiguous CUDA.
        eps: Added to the mean square. Positive.

    Returns:
        A :class:`slinoss.ops.block.NormResidual`. ``normed`` carries the dtype of
        ``x``; ``residual`` is float32.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, or a non-CUDA or non-contiguous
            operand.
        TypeError: On a dtype with no kernel path.
    """
    rows, width = _check_norm(x, weight, eps)
    _check_stream(x, residual)

    normed = torch.empty_like(x)
    total = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    stream = x if residual is None else residual
    jit_launch(
        rmsnorm_residual_fwd,
        (
            x.view(rows, width),
            stream.view(rows, width),
            total.view(rows, width),
            weight,
            normed.view(rows, width),
            float(eps),
            rows,
        ),
        (width, NORM_THREADS, residual is not None),
    )
    return NormResidual(normed=normed, residual=total)


def rmsnorm_backward(
    dout: Tensor,
    x: Tensor,
    weight: Tensor,
    /,
    *,
    eps: float,
) -> RMSNormGrads:
    """Pullback of :func:`rmsnorm_forward`, in one launch.

    The row scale is recomputed from ``x``, so the forward saves nothing for this
    call. ``dweight`` is one float32 partial row per block, summed on the host: a
    reduction over rows inside the launch would need an accumulator zeroed before
    it, and a zero fill on the hot path is not available.

    Args:
        dout: Cotangent of the output, shape and dtype of ``x``, contiguous CUDA.
        x: The forward's input, ``(..., D)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`.
        weight: The forward's scale, ``(D,)`` float32, contiguous CUDA.
        eps: The forward's epsilon. Positive.

    Returns:
        A :class:`slinoss.ops.block.RMSNormGrads`. ``dx`` carries the dtype of
        ``x``; ``dweight`` is float32.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, or a non-CUDA or non-contiguous
            operand.
        TypeError: On a dtype with no kernel path, or on a cotangent whose dtype
            is not that of ``x``.
    """
    rows, width = _check_norm(x, weight, eps)
    _check_cotangent(dout, x, "dout")

    dx = torch.empty_like(x)
    blocks = row_blocks(rows, x.device.index)
    partial = torch.empty((blocks, width), dtype=torch.float32, device=x.device)
    jit_launch(
        rmsnorm_bwd,
        (
            dout.view(rows, width),
            x.view(rows, width),
            weight,
            dx.view(rows, width),
            partial,
            float(eps),
            rows,
            blocks,
        ),
        (cute_dtype(x.dtype), width, NORM_THREADS),
    )
    return RMSNormGrads(dx=dx, dweight=partial.sum(0))


def rmsnorm_residual_backward(
    dnormed: Tensor | None,
    dresidual: Tensor | None,
    x: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    /,
    *,
    eps: float,
) -> NormResidualGrads:
    """Pullback of :func:`rmsnorm_residual_forward`, in one launch.

    The two outputs share the sum ``x + residual``, so both cotangents meet in one
    kernel and the sum is traversed once. An absent cotangent closes its half of
    the expression at compile time rather than contracting a zero tensor, and the
    tensor slot it would have filled is handed ``x``, which the kernel never reads,
    so the launch has one signature. With both absent nothing was differentiated
    and no launch happens.

    Args:
        dnormed: Cotangent of ``normed``, shape and dtype of ``x``, or None when
            the caller consumed only the residual.
        dresidual: Cotangent of the wide residual, shape of ``x`` and float32,
            which is the dtype the forward returns it at, or None when the caller
            consumed only the normed output.
        x: The forward's branch output, ``(..., D)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`.
        residual: The forward's incoming stream, same shape, or None.
        weight: The forward's scale, ``(D,)`` float32, contiguous CUDA.
        eps: The forward's epsilon. Positive.

    Returns:
        A :class:`slinoss.ops.block.NormResidualGrads`. ``dx`` carries the dtype of
        ``x`` and ``dresidual`` that of the incoming stream. ``dresidual`` is None
        when the forward took no stream, ``dweight`` is None when ``normed``
        carries no cotangent, and every field is None when both cotangents are
        absent.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, a non-float32 ``dresidual``, or a
            non-CUDA or non-contiguous operand.
        TypeError: On a dtype with no kernel path, or on a ``dnormed`` whose dtype
            is not that of ``x``.
    """
    rows, width = _check_norm(x, weight, eps)
    _check_stream(x, residual)
    if dnormed is not None:
        _check_cotangent(dnormed, x, "dnormed")
    if dresidual is not None:
        _check_shape(dresidual, x, "dresidual")
        if dresidual.dtype is not torch.float32:
            raise ValueError(
                f"dresidual must be float32, the width the forward returns the "
                f"residual at, got {dresidual.dtype}"
            )
        check_operand(dresidual, "dresidual")
    if dnormed is None and dresidual is None:
        return NormResidualGrads(dx=None, dresidual=None, dweight=None)

    stream = x if residual is None else residual
    dx = torch.empty_like(x)
    dres_out = None if residual is None else torch.empty_like(stream)
    blocks = row_blocks(rows, x.device.index)
    partial = (
        None
        if dnormed is None
        else torch.empty((blocks, width), dtype=torch.float32, device=x.device)
    )

    # One descriptor fills every slot the compiled variant does not read. `x` is
    # the placeholder because it is the one operand always present, and the
    # kernel's `const_expr` guards mean an unread slot is never addressed. Built
    # here rather than handed to `jit_launch` as a tensor: one build shared by up
    # to five slots is cheaper than five pooled borrows of one layout, which
    # would run past the pool's depth.
    gx = dev_tensor(x.view(rows, width))
    absent = gx
    jit_launch(
        rmsnorm_residual_bwd,
        (
            absent if dnormed is None else dnormed.view(rows, width),
            absent if dresidual is None else dresidual.view(rows, width),
            gx,
            absent if residual is None else residual.view(rows, width),
            weight,
            dx.view(rows, width),
            absent if dres_out is None else dres_out.view(rows, width),
            absent if partial is None else partial,
            float(eps),
            rows,
            blocks,
        ),
        (
            cute_dtype(x.dtype),
            cute_dtype(stream.dtype),
            width,
            NORM_THREADS,
            residual is not None,
            dnormed is not None,
            dresidual is not None,
        ),
    )
    return NormResidualGrads(
        dx=dx,
        dresidual=dres_out,
        dweight=None if partial is None else partial.sum(0),
    )
