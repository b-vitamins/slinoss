"""RMS norm, plain and fused with the residual add. CuTe DSL both directions.

    normed = x * rsqrt(mean(x^2) + eps) * weight

The fused form adds the incoming residual stream first and hands the sum back:

    s      = x + residual
    normed = s * rsqrt(mean(s^2) + eps) * weight

``s`` is returned float32 at every operand width. That is what the reference
means by wide: its accumulation dtype is float32 unless an operand is float64,
and float64 reaches no kernel. A stack therefore carries its residual at float32
instead of narrowing once per block.

Parallel decomposition, forward. 256 threads, and the reduction is over ``D``
only, so no row shares anything with another. Each thread holds its
``ceil(D/256)`` columns of the row in float32 registers, the lanes of a warp are
combined by a shuffle add-scan, and every thread sums the eight warp totals out of
shared memory. The second pass reads no operand: the row is already in registers,
and in the fused form the wide sum is written from them, so the add is evaluated
once and ``normed`` is a function of the residual that is returned rather than of
a second summation.

The plain forward's grid is :func:`fill_blocks` and each block strides over rows,
as the backwards do, and it reads the next row a trip ahead. One block per row put
the row's load, its reduction and its store in one dependent chain with nothing to
overlap it, and made the ``4*D`` weight read a per-row cost. The fused forward
keeps one block per row: it moves three times the bytes per row, which covers the
same chain, and it already saturates the bus.

Parallel decomposition, backward. Both parameter gradients reduce over rows, so
the grid is :func:`fill_blocks` and each block strides over the row axis, which
bounds the partial buffer at one float32 row per block instead of one per row.
Every block covers the same columns on every row it runs, so a thread's
``ceil(D/256)`` weight-gradient accumulators stay in registers across the whole
stride loop and the epilogue is one store per column. The weight itself is
row-invariant, so a thread's slice of it is loaded once by :func:`_weight_of` and
read from registers thereafter. There are no atomics and no second pass over the
operands. The trip count of the stride loop is block-uniform, which is what makes
the barriers inside the row reduction safe. With fewer rows than that grid the
grid is the row count: splitting a row across blocks would put the row reduction
across a grid barrier, so the row count is the whole available parallelism at this
decomposition.

The backward's row loop needs the sum of squares and the dot product of the
cotangent with the row, so the block reduction carries two accumulators through
one pair of barriers rather than running twice.

Its second pass reads no operand. The row value and its cotangent are held in two
``ceil(D/256)`` float32 fragments across the reduction, which the first pass has
already loaded and the second pass needs unchanged. Re-reading them instead put
three loads immediately after the barrier with nothing to overlap their latency:
measured on sm_86 at the standard shape, 40 percent of the kernel's L1TEX load
sectors and a ``long_scoreboard`` share of 68 percent at 47 percent L1TEX and 22
percent L2 utilization, which is a latency profile and not a throughput one.

Holding costs registers linear in ``ceil(D/256)``: six float32 fragments of it in
either backward -- the weight gradient, the weight, the row, its cotangent, and the
two lookaheads -- and three in the plain forward. Every standard shape is two
segments, so the register cost of a width the workload does not reach is measured
separately. On sm_86 at a fixed 8.3 M elements, registers per thread and the
fraction of that run's measured ceiling, plain forward, plain backward, fused
forward, fused backward:

- ``D = 2048``, 8 segments: 40 regs at 94.9 percent, 63 at 97.0, 36 at 98.0, 80 at
  98.9.
- ``D = 4096``, 16 segments: 71 at 95.8, 126 at 98.8, 48 at 97.5, 142 at 85.8.

The fused backward at ``D = 4096`` runs one block per multiprocessor at 16.5
percent achieved occupancy and still clears the class floor: at 16 segments the
loads in flight come from the segment count rather than from the resident warp
count. Holding wins at both widths. Past them the register file is the bound and
not the traffic -- at ``D = 8192``, 32 segments, both backwards pin at the
255-register cap and spill, and the fused backward falls to 83.4 percent with
``lg_throttle`` at 34 percent of its stalls -- so 4096 is the widest ``D`` this
decomposition is measured at.

Nothing is saved from the forward. The row scale is recomputed from ``x`` (and
``residual``) with the expression the forward uses, which is why the two
directions share :func:`_block_totals` and :func:`_scale_of`.

The partial buffer is reduced to the weight gradient by a second launch,
:func:`rmsnorm_dweight_kernel`, whose grid is over ``D`` and whose block splits the
partial rows across slots. Reducing it with ``partial.sum(0)`` instead gave torch a
grid sized by the output alone: measured on sm_86 at the standard shape, 3 blocks
at 8 percent achieved occupancy, 580 kB in 15.9 us, 37 GB/s against a measured 680
GB/s ceiling, and 17 percent of the residual backward.

Shared memory: one float32 partial per warp per accumulator per round, so 32 B in
the fused forward, 64 B in the plain one, 128 B backward, and one float32 per
thread in the weight-gradient
reduction, so 2 kB there; all asserted against the queried capacity by a test. One
lane per warp writes one partial, so the writes land in distinct banks, and every
read of a partial is one address across the block; neither needs a swizzle.

DRAM-bound, both directions. Analytic traffic per row, at operand itemsize ``i``
and cotangent itemsize ``i_c``, with no measured bandwidth claimed here:

- ``rmsnorm_fwd``: ``2*D*i``, plus the ``4*D`` float32 weight.
- ``rmsnorm_residual_fwd``: ``D*(i_x + i_residual + 4 + i_normed)`` -- one read of
  ``x``, one of ``residual``, one write of the wide sum, one write of ``normed``.
  The sum is not read back: it stays in registers across the reduction.
- ``rmsnorm_bwd``: ``D*(i_c + i_x + i_x)`` -- one read of the cotangent, one of
  ``x``, one write of ``dx``.
- ``rmsnorm_residual_bwd``: ``D*(i_c + 4 + i_x + i_residual + i_x + i_residual)``
  -- the two cotangents, ``x`` and ``residual``, then ``dx`` and ``dresidual``.
  An absent cotangent or an absent residual drops its own terms.

Both backwards add the ``4*D`` float32 weight and ``4*D`` float32 of partials per
block, written once and read once by the reduction launch.

``rmsnorm_dweight_kernel`` is the one kernel here whose traffic is not a sequence
extent: ``4*D`` per block of the backward grid, so it grows with ``D`` alone. At
every standard width it is SERIAL-tiny and held to a share of the step, 2.84 us at
``d_model = 288`` and under one percent of it. It leaves that class as ``D`` grows:
17.0 us at ``D = 4096`` and 30.8 at ``D = 8192``, 76 and 90 percent of the measured
ceiling, where bandwidth is the bound instead.
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
    "DWEIGHT_COLS",
    "DWEIGHT_THREADS",
    "FWD_SLOTS",
    "LOOP_ROUNDS",
    "NORM_THREADS",
    "ONE_ROUND",
    "SLOT_DOT",
    "SLOT_SUMSQ",
    "WARPS",
    "check_operand",
    "dweight_smem_bytes",
    "dweight_tile",
    "fill_blocks",
    "norm_smem_bytes",
    "reduce_tile",
    "rmsnorm_backward",
    "rmsnorm_bwd",
    "rmsnorm_bwd_kernel",
    "rmsnorm_dweight",
    "rmsnorm_dweight_kernel",
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
    "threads_per_sm",
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

ONE_ROUND = 1
"""Reductions in flight for a kernel whose block reduces once: the residual
forward, which owns one row."""

LOOP_ROUNDS = 2
"""Reductions in flight for a kernel that reduces per trip of a row loop. Two, so
consecutive trips write different partials and one barrier per trip is enough."""

DWEIGHT_COLS = 8
"""Columns one block of :func:`rmsnorm_dweight_kernel` owns. Eight float32 is one
32-byte sector, so the tile is the narrowest one whose row segment is a whole
sector, and therefore the widest grid over ``D`` that wastes none of one."""

DWEIGHT_THREADS = 512
"""Block width of :func:`rmsnorm_dweight_kernel`. Divisible by
:data:`DWEIGHT_COLS`, so the block is a rectangle of row slots by columns. Wide,
because the grid is ``D / DWEIGHT_COLS`` blocks and cannot fill the device: the
slots are the only other axis, and each one costs a dependent load."""


def reduce_tile(slots: int, rounds: int) -> Tile:
    """Per-warp partials for ``slots`` block reductions over ``rounds`` buffers.

    Flat rather than ``(rounds, slots, WARPS)``: round ``r`` slot ``s`` occupies
    ``(r*slots + s)*WARPS`` onward, and one lane per warp writes one element, so
    consecutive warps of one slot land in consecutive banks.

    Args:
        slots: Accumulators reduced together.
        rounds: Reductions in flight. A caller that reduces once needs one; a
            caller that reduces per loop trip needs two, because with one the
            barrier that ends a reduction is also the only thing stopping the next
            trip's write from landing on a partial another warp has not read.

    Returns:
        The tile.
    """
    return Tile((rounds * slots * WARPS,), (1,))


def dweight_tile(threads: int) -> Tile:
    """Per-thread accumulators of the weight-gradient reduction.

    One float32 per thread, the block laid out as ``threads // DWEIGHT_COLS`` row
    slots by :data:`DWEIGHT_COLS` columns with the column index innermost, so a
    slot's partials are contiguous and the combine over slots reads one bank per
    column.

    Args:
        threads: Block width. Compile-time.

    Returns:
        The tile.
    """
    return Tile((threads,), (1,))


def norm_smem_bytes(slots: int, rounds: int) -> int:
    """Shared memory a norm kernel holds, in bytes, from the tile layout.

    Args:
        slots: :data:`FWD_SLOTS` or :data:`BWD_SLOTS`.
        rounds: :data:`ONE_ROUND` or :data:`LOOP_ROUNDS`.

    Returns:
        Total bytes.
    """
    return smem_bytes([(reduce_tile(slots, rounds), 4)])


def dweight_smem_bytes(threads: int) -> int:
    """Shared memory :func:`rmsnorm_dweight_kernel` holds, in bytes.

    Args:
        threads: Block width.

    Returns:
        Total bytes.
    """
    return smem_bytes([(dweight_tile(threads), 4)])


@cache
def sm_count(index: int) -> int:
    """Multiprocessors on one CUDA device. Cached: the grid is sized per launch.

    Args:
        index: CUDA device ordinal.

    Returns:
        The multiprocessor count.
    """
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


@cache
def threads_per_sm(index: int) -> int:
    """Resident threads one multiprocessor holds. Cached with :func:`sm_count`.

    Args:
        index: CUDA device ordinal.

    Returns:
        The resident-thread limit.
    """
    return int(torch.cuda.get_device_properties(index).max_threads_per_multi_processor)


def fill_blocks(threads: int, index: int) -> int:
    """Blocks a grid-strided launch needs to fill the device.

    Twice the SM count is the block-count floor, not the grid: at 256 threads it
    leaves two blocks per SM, which caps achieved occupancy at 33% whatever the
    register count allows. Measured on sm_86 at the standard shape, every
    grid-strided kernel here sat at exactly that cap with ``long_scoreboard``
    above 60%, which is too few memory requests in flight rather than a traffic
    problem. The grid is therefore the resident-thread limit divided by the block
    width, which is the largest grid that adds resident warps.

    Args:
        threads: Block width. Divides the resident-thread limit on every
            architecture this runs on.
        index: CUDA device ordinal.

    Returns:
        The block count.
    """
    return sm_count(index) * (threads_per_sm(index) // threads)


def row_blocks(rows: int, index: int) -> int:
    """Blocks a backward launch uses over ``rows`` rows.

    :func:`fill_blocks` is the grid, and the row stride loop covers any row count
    from it. Fewer rows than that caps the grid at the row count: a row reduction
    cannot cross a grid barrier, so the row count is the whole available
    parallelism at this decomposition.

    Args:
        rows: Rows on the flattened axis. At least one.
        index: CUDA device ordinal.

    Returns:
        The block count, which is also the row stride.
    """
    return min(rows, fill_blocks(NORM_THREADS, index))


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
    first: Scalar,
    second: Scalar,
    tid: cutlass.Int32,
    slots: cutlass.Constexpr,
    base: Any,
) -> tuple[Scalar, Scalar]:
    """Sum one or two float32 accumulators across the block. Entered by the block.

    One barrier. Every thread sums the ``WARPS`` partials itself out of shared
    memory, which is eight broadcast loads and seven adds, rather than thread 0
    summing them into a slot the block then has to reach a second barrier to read.
    Every thread sums the same partials in the same order, so the totals agree bit
    for bit and :func:`_scale_of` stays a function of the row. Measured on sm_86 at
    the standard shape the second barrier was 18 percent of the residual
    backward's stalls once its loads were pipelined.

    Two accumulators share the barrier instead of reducing in sequence: the
    backward needs the sum of squares and the cotangent dot product of the same
    row, and a second call would double the barrier count. A caller with one
    accumulator passes it twice and asks for one slot, which leaves slot 1
    unwritten and unread.

    Args:
        spart: :func:`reduce_tile` of ``slots`` and the caller's round count,
            float32.
        first: The thread's contribution to slot :data:`SLOT_SUMSQ`.
        second: Its contribution to slot :data:`SLOT_DOT`. Read only when
            ``slots`` exceeds one.
        tid: Thread index within the block.
        slots: :data:`FWD_SLOTS` or :data:`BWD_SLOTS`. Compile-time.
        base: First element of this round's buffer, ``round * slots * WARPS``. A
            caller that reduces per loop trip alternates it; see
            :func:`reduce_tile`.

    Returns:
        The block total of ``first``, and of ``second`` when ``slots`` exceeds one
        or ``first``'s total again when it does not.
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
        spart[base + warp] = total
        if cutlass.const_expr(slots > 1):
            spart[base + WARPS + warp] = paired
    cute.arch.sync_threads()

    summed = cutlass.Float32(0.0)
    for index in cutlass.range_constexpr(WARPS):
        summed = summed + spart[base + index]
    pair = summed
    if cutlass.const_expr(slots > 1):
        pair = cutlass.Float32(0.0)
        for index in cutlass.range_constexpr(WARPS):
            pair = pair + spart[base + WARPS + index]
    return summed, pair


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

    Both directions hold one register per segment -- the row itself in the forward,
    the row and the weight-gradient accumulator in the backward -- which needs a
    compile-time segment count. ``D`` is compile-time, so the count is too. Only the
    last segment can run past ``D``. Its clamped read position keeps
    :func:`_weight_of`, which reads once per block, in bounds without a branch; the
    row loads carry a predicate instead, for the reason in :func:`_read_row`.

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


def _offsets(width: int, threads: int) -> tuple[tuple[int, bool], ...]:
    """Per-segment ``(column offset, masked)``, with no thread index in it.

    :func:`_slice` bakes the thread index into a column, which makes its result a
    device value and a `@cute.jit` helper unable to take it as anything it can
    branch on. This is the same chart with the index left out, so a helper can take
    it as a `Constexpr` and add the index itself.

    Args:
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.

    Returns:
        One entry per segment, outermost first.
    """
    segments = -(-width // threads)
    ragged = width % threads != 0
    return tuple((j * threads, ragged and j == segments - 1) for j in range(segments))


def _weight_of(
    gw: cute.Tensor, cols: list[tuple[Any, Any, bool]], segments: int
) -> cute.Tensor:
    """One thread's slice of the weight, loaded once into registers.

    The weight does not depend on the row, so a read of it inside a backward's
    row-stride loop is reissued once per row per pass. Measured on sm_86 at the
    standard shape it was 86 of 315 L1TEX load sectors per row, 27 percent of the
    kernel's load traffic, for ``4 * D`` bytes the block reads once. The forward
    holds one row per block, so the same read is reissued once per block instead:
    36 of 72 load sectors per row on a bfloat16 operand, half the kernel's load
    traffic.

    Args:
        gw: ``(D,)`` float32 weight (I4).
        cols: :func:`_slice` of this thread. A masked segment's read position is
            clamped, so every load is in bounds.
        segments: ``len(cols)``. Compile-time.

    Returns:
        A ``(segments,)`` float32 fragment, entry ``j`` the weight at segment
        ``j``'s read position.
    """
    weight = cute.make_fragment((segments,), cutlass.Float32)
    # A plain `range`, not `cutlass.range_constexpr`: the DSL preprocessor rewrites
    # only the bodies it decorates, and this is a helper called from one.
    for j in range(segments):
        weight[j] = gw[cols[j][1]]
    return weight


@cute.jit
def _read_values(
    values: cute.Tensor,
    gx: cute.Tensor,
    tid: cutlass.Int32,
    row: Any,
    offsets: cutlass.Constexpr,
    width: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
) -> None:
    """Put one row into a register fragment. The forward's read.

    Separate from :func:`_read_row` rather than a case of it: the forward has no
    cotangent, and a shared helper would either take a tensor it does not read or
    split the backward's two reads into two loops.

    Args:
        values: ``(segments,)`` float32, written with the row.
        gx: ``(rows, D)`` input, ``dtype``.
        tid: Thread index within the block.
        row: Row to read. In range at every call site.
        offsets: :func:`_offsets` for this block width. Compile-time.
        width: ``D``. Compile-time.
        dtype: Element type of ``gx``. Compile-time.
    """
    for j in cutlass.range_constexpr(len(offsets)):
        col = offsets[j][0] + tid
        if cutlass.const_expr(offsets[j][1]):
            values[j] = cutlass.Float32(0.0)
            if col < width:
                values[j] = widen(gx[row, col], dtype)
        else:
            values[j] = widen(gx[row, col], dtype)


@cute.jit
def _read_row(
    values: cute.Tensor,
    dnorms: cute.Tensor,
    gx: cute.Tensor,
    gres: cute.Tensor,
    gdnormed: cute.Tensor,
    tid: cutlass.Int32,
    row: Any,
    offsets: cutlass.Constexpr,
    width: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    rdtype: cutlass.Constexpr,
    has_residual: cutlass.Constexpr,
) -> None:
    """Put one row's value and cotangent into two register fragments.

    A backward reads each of them once and uses each twice, once in the row
    reduction and once in the pullback on the far side of the reduction's
    barriers. Issued a whole row iteration ahead of the use, so the load latency
    overlaps the reduction rather than preceding it: the barriers couple the
    block's eight warps into one instruction stream, so without the lead the
    resident streams per multiprocessor are the blocks and not the warps.

    A masked segment is predicated and not clamped, which is why this is a
    `@cute.jit` helper and takes the offset chart rather than a column. ``D % 256``
    is a multiple of the warp size at every shape here, so the predicate is
    warp-uniform and the branch costs no divergence, while a clamped read by every
    thread past ``D`` cost 7 of every 25 L1TEX sectors on a bfloat16 operand and 7
    of 43 on a float32 one, and 28 of 64 load instructions per row.

    Args:
        values: ``(segments,)`` float32, written with ``x`` plus the residual.
        dnorms: ``(segments,)`` float32, written with the cotangent of ``normed``.
        gx: ``(rows, D)`` input, ``dtype``.
        gres: ``(rows, D)`` residual, ``rdtype``. Read only when ``has_residual``;
            the caller without one passes ``gx``.
        gdnormed: ``(rows, D)`` cotangent of ``normed``, ``dtype``.
        tid: Thread index within the block.
        row: Row to read. In range at every call site, which is what lets the
            caller's lookahead clamp rather than predicate.
        offsets: :func:`_offsets` for this block width. Compile-time.
        width: ``D``. Compile-time.
        dtype: Element type of ``gx`` and ``gdnormed``. Compile-time.
        rdtype: Element type of ``gres``. Compile-time.
        has_residual: Whether to add ``gres``. Compile-time.
    """
    for j in cutlass.range_constexpr(len(offsets)):
        col = offsets[j][0] + tid
        if cutlass.const_expr(offsets[j][1]):
            values[j] = cutlass.Float32(0.0)
            dnorms[j] = cutlass.Float32(0.0)
            if col < width:
                value = widen(gx[row, col], dtype)
                if cutlass.const_expr(has_residual):
                    value = value + widen(gres[row, col], rdtype)
                values[j] = value
                dnorms[j] = widen(gdnormed[row, col], dtype)
        else:
            value = widen(gx[row, col], dtype)
            if cutlass.const_expr(has_residual):
                value = value + widen(gres[row, col], rdtype)
            values[j] = value
            dnorms[j] = widen(gdnormed[row, col], dtype)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@cute.kernel
def rmsnorm_fwd_kernel(
    gx: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    span: cutlass.Int32,
    dtype: cutlass.Constexpr,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Normalize a strided run of rows per block.

    A row-stride loop and not one block per row, for the reason the backwards have
    one: the row's load, its reduction and its store are a dependent chain, and one
    row per block leaves nothing to overlap it with. Measured on sm_86 at the
    standard shape, one block per row ran at 72 percent of the ceiling with
    ``long_scoreboard`` at 41 percent, and the weight read cost 36 of every 72 load
    sectors per row because a block that owns one row reads ``4*D`` for it.

    Args:
        gx: ``(rows, D)`` input, ``dtype``.
        gw: ``(D,)`` float32 weight (I4).
        gy: ``(rows, D)`` output, ``dtype``.
        eps: Added to the mean square. Dynamic, so one variant covers every
            epsilon.
        rows: Rows on the flattened axis. Dynamic.
        span: Rows one grid step advances, the block count. Dynamic.
        dtype: Element type of ``gx`` and ``gy``, which are one width by contract.
            Compile-time, for the reason in :func:`rmsnorm_bwd_kernel`.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.

    Invariants:
        The mean square is accumulated in float32 whatever the operand width, and
        the weight is float32, so only the store narrows. Each row is read once,
        one trip ahead of its use, and held in registers across the reduction. The
        stride loop's trip count is block-uniform, which is what makes the barrier
        inside the reduction safe. Masked columns carry a zero, so they do not
        reach the accumulator.
    """
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(
        cutlass.Float32, reduce_tile(FWD_SLOTS, LOOP_ROUNDS).layout(), 16
    )

    dst = gy.element_type
    cols = _slice(tid, width, threads)
    segments = len(cols)
    weight = _weight_of(gw, cols, segments)
    values = cute.make_fragment((segments,), cutlass.Float32)
    ahead = cute.make_fragment((segments,), cutlass.Float32)
    # Loop-carried, so storage and not a rebound value; see :func:`rmsnorm_bwd_kernel`.
    round_of = cute.make_fragment((1,), cutlass.Int32)
    round_of[0] = 0
    last = rows - 1
    offsets = _offsets(width, threads)
    _read_values(values, gx, tid, block, offsets, width, dtype)

    for row in cutlass.range(block, rows, span):
        acc = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(segments):
            acc = acc + values[j] * values[j]

        _read_values(
            ahead, gx, tid, cutlass.min(row + span, last), offsets, width, dtype
        )
        sumsq, _ = _block_totals(
            spart, acc, acc, tid, FWD_SLOTS, round_of[0] * FWD_SLOTS * WARPS
        )
        round_of[0] = 1 - round_of[0]
        scale = _scale_of(sumsq, eps, width)

        for j in cutlass.range_constexpr(segments):
            col, _, masked = cols[j]
            value = narrow(values[j] * scale * weight[j], dst)
            values[j] = ahead[j]
            if cutlass.const_expr(masked):
                if col < width:
                    gy[row, col] = value
            else:
                gy[row, col] = value


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
        The sum is formed once, in float32, held in registers across the reduction,
        and both outputs derive from that one value. ``gsum`` is written and never
        read back, so the wide stream costs one pass and not two.
    """
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(
        cutlass.Float32, reduce_tile(FWD_SLOTS, ONE_ROUND).layout(), 16
    )

    src = gx.element_type
    rsrc = gres.element_type
    dst = gy.element_type
    cols = _slice(tid, width, threads)
    segments = len(cols)
    weight = _weight_of(gw, cols, segments)
    values = cute.make_fragment((segments,), cutlass.Float32)
    acc = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(segments):
        col, _, masked = cols[j]
        if cutlass.const_expr(masked):
            values[j] = cutlass.Float32(0.0)
            if col < width:
                value = widen(gx[row, col], src)
                if cutlass.const_expr(has_residual):
                    value = value + widen(gres[row, col], rsrc)
                gsum[row, col] = value
                values[j] = value
        else:
            value = widen(gx[row, col], src)
            if cutlass.const_expr(has_residual):
                value = value + widen(gres[row, col], rsrc)
            gsum[row, col] = value
            values[j] = value
        acc = acc + values[j] * values[j]

    sumsq, _ = _block_totals(spart, acc, acc, tid, FWD_SLOTS, 0)
    scale = _scale_of(sumsq, eps, width)
    for j in cutlass.range_constexpr(segments):
        col, _, masked = cols[j]
        value = narrow(values[j] * scale * weight[j], dst)
        if cutlass.const_expr(masked):
            if col < width:
                gy[row, col] = value
        else:
            gy[row, col] = value


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
    spart = smem.allocate_tensor(
        cutlass.Float32, reduce_tile(BWD_SLOTS, LOOP_ROUNDS).layout(), 16
    )

    zero = cutlass.Float32(0.0)
    denom = cutlass.Float32(float(width))
    cols = _slice(tid, width, threads)
    segments = len(cols)
    # A fragment rather than a list of scalars: the accumulator is carried across
    # a dynamic loop, so it has to be addressable storage and not a rebound value.
    # The fill is the initialization, inside the kernel.
    dweight = cute.make_fragment((segments,), cutlass.Float32)
    dweight.fill(0.0)
    weight = _weight_of(gw, cols, segments)
    values = cute.make_fragment((segments,), cutlass.Float32)
    dnorms = cute.make_fragment((segments,), cutlass.Float32)
    ahead = cute.make_fragment((segments,), cutlass.Float32)
    dahead = cute.make_fragment((segments,), cutlass.Float32)
    # The buffer index is loop-carried, so it is storage and not a rebound value for
    # the reason above. It cannot be derived from the row: consecutive trips differ
    # by the block count, whose parity is fixed.
    round_of = cute.make_fragment((1,), cutlass.Int32)
    round_of[0] = 0
    last = rows - 1
    offsets = _offsets(width, threads)
    _read_row(
        values, dnorms, gx, gx, gdout, tid, block, offsets, width, dtype, dtype, False
    )

    for row in cutlass.range(block, rows, span):
        sumsq = zero
        dot = zero
        for j in cutlass.range_constexpr(segments):
            sumsq = sumsq + values[j] * values[j]
            dot = dot + dnorms[j] * weight[j] * values[j]

        # The lookahead clamps rather than predicates: on the last trip it reads a
        # row that is in range and discards it, which costs one row of L1 hits and
        # keeps the loop body free of a branch.
        _read_row(
            ahead,
            dahead,
            gx,
            gx,
            gdout,
            tid,
            cutlass.min(row + span, last),
            offsets,
            width,
            dtype,
            dtype,
            False,
        )
        total, dotted = _block_totals(
            spart, sumsq, dot, tid, BWD_SLOTS, round_of[0] * BWD_SLOTS * WARPS
        )
        round_of[0] = 1 - round_of[0]
        scale = _scale_of(total, eps, width)
        coupling = scale * scale * scale * dotted / denom

        for j in cutlass.range_constexpr(segments):
            col, _, masked = cols[j]
            value = values[j]
            dnorm = dnorms[j]
            dweight[j] = dweight[j] + dnorm * value * scale
            dvalue = narrow(scale * dnorm * weight[j] - coupling * value, dtype)
            values[j] = ahead[j]
            dnorms[j] = dahead[j]
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
    spart = smem.allocate_tensor(
        cutlass.Float32, reduce_tile(BWD_SLOTS, LOOP_ROUNDS).layout(), 16
    )

    zero = cutlass.Float32(0.0)
    denom = cutlass.Float32(float(width))
    cols = _slice(tid, width, threads)
    segments = len(cols)
    dweight = cute.make_fragment((segments,), cutlass.Float32)
    dweight.fill(0.0)
    weight = _weight_of(gw, cols, segments)
    values = cute.make_fragment((segments,), cutlass.Float32)
    dnorms = cute.make_fragment((segments,), cutlass.Float32)
    ahead = cute.make_fragment((segments,), cutlass.Float32)
    dahead = cute.make_fragment((segments,), cutlass.Float32)
    round_of = cute.make_fragment((1,), cutlass.Int32)
    round_of[0] = 0
    last = rows - 1
    offsets = _offsets(width, threads)
    if cutlass.const_expr(has_normed):
        _read_row(
            values,
            dnorms,
            gx,
            gres,
            gdnormed,
            tid,
            block,
            offsets,
            width,
            dtype,
            rdtype,
            has_residual,
        )

    for row in cutlass.range(block, rows, span):
        # Bound before the trace-time branch for the reason given in
        # `_block_totals`. Neither reaches a store unless `has_normed` holds.
        scale = zero
        coupling = zero
        if cutlass.const_expr(has_normed):
            sumsq = zero
            dot = zero
            for j in cutlass.range_constexpr(segments):
                sumsq = sumsq + values[j] * values[j]
                dot = dot + dnorms[j] * weight[j] * values[j]

            # The lookahead clamps rather than predicates: on the last trip it reads
            # a row that is in range and discards it, which costs one row of L1 hits
            # and keeps the loop body free of a branch.
            _read_row(
                ahead,
                dahead,
                gx,
                gres,
                gdnormed,
                tid,
                cutlass.min(row + span, last),
                offsets,
                width,
                dtype,
                rdtype,
                has_residual,
            )
            rowsq, rowdot = _block_totals(
                spart, sumsq, dot, tid, BWD_SLOTS, round_of[0] * BWD_SLOTS * WARPS
            )
            round_of[0] = 1 - round_of[0]
            scale = _scale_of(rowsq, eps, width)
            coupling = scale * scale * scale * rowdot / denom

        for j in cutlass.range_constexpr(segments):
            col, pos, masked = cols[j]
            total = zero
            if cutlass.const_expr(has_normed):
                value = values[j]
                dnorm = dnorms[j]
                values[j] = ahead[j]
                dnorms[j] = dahead[j]
                dweight[j] = dweight[j] + dnorm * value * scale
                total = scale * dnorm * weight[j] - coupling * value
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


@cute.kernel
def rmsnorm_dweight_kernel(
    gpartial: cute.Tensor,
    gdweight: cute.Tensor,
    blocks: cutlass.Int32,
    width: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Sum the backward's per-block partials into the weight gradient.

    A block owns ``cols`` columns and splits the partial rows across
    ``threads // cols`` slots, so the grid is ``ceil(D / cols)`` and the row axis
    supplies the rest of the parallelism. The partial buffer is one row per block
    of the backward launch, which is a device-sized extent and not a
    sequence-sized one, so no shape makes this kernel large enough to hold to a
    bandwidth and its block count is bounded by ``D`` rather than by the device.

    Args:
        gpartial: ``(blocks, D)`` float32 partials, every element written by the
            backward.
        gdweight: ``(D,)`` float32, written once per column.
        blocks: Rows of ``gpartial``. Dynamic, so one variant covers every grid the
            backward chose.
        width: ``D``. Compile-time.
        cols: Columns per block, :data:`DWEIGHT_COLS`. Compile-time, and divides
            ``threads``.
        threads: Block width. Compile-time.

    Invariants:
        The reduction order is fixed by the launch geometry alone: ascending row
        within a slot, then ascending slot. It has no atomics, so a rerun at one
        shape reproduces the result bit for bit. A column past ``D`` reads a
        clamped position and stores nothing.
    """
    tile, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sacc = smem.allocate_tensor(cutlass.Float32, dweight_tile(threads).layout(), 16)

    slots = threads // cols
    slot = tid // cols
    lane = tid - slot * cols
    col = tile * cols + lane
    # Clamped rather than predicated: only the last tile can run past `D`, and the
    # read is discarded by the store's guard below.
    acc = cutlass.Float32(0.0)
    for row in cutlass.range(slot, blocks, slots):
        acc = acc + gpartial[row, cutlass.min(col, width - 1)]

    sacc[slot * cols + lane] = acc
    cute.arch.sync_threads()

    if slot == 0:
        total = cutlass.Float32(0.0)
        # Rolled, not unrolled: the chain of adds is serial either way, and at 64
        # slots the unrolled form is the slower thing to compile.
        for index in cutlass.range(slots):
            total = total + sacc[index * cols + lane]
        if col < width:
            gdweight[col] = total


@cute.jit
def rmsnorm_fwd(
    gx: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    blocks: cutlass.Int32,
    dtype: cutlass.Constexpr,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_fwd_kernel` over a fixed grid.

    ``blocks`` is the grid extent and the row stride both, as in
    :func:`rmsnorm_bwd`.
    """
    rmsnorm_fwd_kernel(gx, gw, gy, eps, rows, blocks, dtype, width, threads).launch(
        grid=(blocks, 1, 1), block=(threads, 1, 1)
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


@cute.jit
def rmsnorm_dweight(
    gpartial: cute.Tensor,
    gdweight: cute.Tensor,
    blocks: cutlass.Int32,
    width: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_dweight_kernel`, one block per column tile."""
    rmsnorm_dweight_kernel(gpartial, gdweight, blocks, width, cols, threads).launch(
        grid=(-(-width // cols), 1, 1), block=(threads, 1, 1)
    )


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
        (
            x.view(rows, width),
            weight,
            out.view(rows, width),
            float(eps),
            rows,
            row_blocks(rows, x.device.index),
        ),
        (cute_dtype(x.dtype), width, NORM_THREADS),
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


def _dweight_of(partial: Tensor, width: int) -> Tensor:
    """Reduce the backward's partial buffer to the weight gradient.

    A second launch rather than ``partial.sum(0)``: torch reduces a
    ``(blocks, D)`` buffer over its outer axis with a grid sized by the output, so
    at ``D = 288`` it ran 3 blocks at 8 percent achieved occupancy and moved 580 kB
    in 15.9 us, 37 GB/s against a measured 680 GB/s ceiling. The cost is launch
    geometry over a small buffer and not bandwidth, and it was 17 percent of the
    residual backward. :func:`rmsnorm_dweight_kernel` splits the row axis too.

    Args:
        partial: ``(blocks, D)`` float32, every element written by the backward.
        width: ``D``.

    Returns:
        ``(D,)`` float32.
    """
    dweight = torch.empty((width,), dtype=torch.float32, device=partial.device)
    jit_launch(
        rmsnorm_dweight,
        (partial, dweight, partial.shape[0]),
        (width, DWEIGHT_COLS, DWEIGHT_THREADS),
    )
    return dweight


def rmsnorm_backward(
    dout: Tensor,
    x: Tensor,
    weight: Tensor,
    /,
    *,
    eps: float,
) -> RMSNormGrads:
    """Pullback of :func:`rmsnorm_forward`, in two launches.

    The row scale is recomputed from ``x``, so the forward saves nothing for this
    call. ``dweight`` is one float32 partial row per block, reduced by
    :func:`_dweight_of`: a reduction over rows inside the first launch would need
    an accumulator zeroed before it, and a zero fill on the hot path is not
    available.

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
    return RMSNormGrads(dx=dx, dweight=_dweight_of(partial, width))


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
    """Pullback of :func:`rmsnorm_residual_forward`, in one launch over the rows.

    The two outputs share the sum ``x + residual``, so both cotangents meet in one
    kernel and the sum is traversed once. The weight gradient adds the second
    launch of :func:`_dweight_of` when ``normed`` carries a cotangent. An absent cotangent closes its half of
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
        dweight=None if partial is None else _dweight_of(partial, width),
    )
