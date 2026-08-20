"""The row reduction of a per-block partial buffer. No operator owns it.

A kernel that closed a parameter gradient inside its own launch would need the
accumulator zeroed before it, and a zero fill on a hot path is not available. So a
fused backward writes one partial row per block into ``torch.empty``, every element
of it, and the sum over those rows is a second launch. Two backwards reach this
one: the scan's parameter frontier and the fused mixer tail. The buffer is
``(S, R, W)``, the reduction is over ``R``, and a caller reaches that form with
views, which are free on the contiguous buffer its own kernel wrote.

The row extent here is a sequence extent: ``R`` is ``B*ceil(T/TILE_TOKENS)`` at the
frontier and ``ceil(B*T/ROWS)`` at the tail, so it grows with the sequence while
``W`` stays a parameter width.
:func:`slinoss.ops.block.cute.norm.rmsnorm_dweight_kernel` is the same geometry
over a row extent bounded by its own grid, and it carries its own measured record
at four widths.

Parallel decomposition. A block owns :data:`REDUCE_COLS` columns of one slab and
splits the rows across ``REDUCE_THREADS // REDUCE_COLS`` slots, so the grid is
``(ceil(W / REDUCE_COLS), S)`` and the row axis supplies the parallelism a grid
over ``W`` alone cannot. A slot walks its rows at a stride of the slot count and
reads a column clamped to ``W - 1`` rather than predicated; the store is what
guards a ragged column tile. One float32 per thread in shared memory,
1,024 B, one barrier, then slot 0 sums the slots.

The reduction order is fixed by the launch geometry: ascending row within a slot,
then ascending slot. There are no atomics, so one shape reproduces bit for bit.

``W`` and the two block constants are compile-time, ``R`` is dynamic and ``S`` is a
grid extent, so one compiled variant per width and output dtype covers every
sequence length and every batch. The output dtype is a compile-time property of
the destination and the narrowing happens on the store, so a caller whose
parameters are not float32 needs no cast after the launch.

SERIAL-tiny. Traffic is ``4*S*R*W`` B in and ``S*W`` elements out, which follows
the reduced width and the producer's tile count rather than a pass over the
sequence: 962 KB at the frontier's ragged shape and 1.18 MB at the tail's
standard shape, both inside L2, so the counters there describe the cache and no
bandwidth verdict applies. Measured on sm_86, clocks unlocked, three captured
launches each: 5.952 us per launch at ``S=1 R=2004 W=120``, 3.8% spread, 0.999%
of the operator's step; 4.128 us at ``S=2 R=256 W=576`` into a bfloat16
destination, 5.4% spread, 0.555% of it. The bound is 2%.
"""

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import Stream, Tile, jit_launch, narrow, smem_bytes
from slinoss._guard import check_dtypes, check_layout
from slinoss._precision import KERNEL_DTYPES

__all__ = [
    "REDUCE_COLS",
    "REDUCE_THREADS",
    "reduce_partials",
    "reduce_rows",
    "reduce_rows_kernel",
    "slot_smem_bytes",
    "slot_tile",
]

REDUCE_COLS = 8
"""Columns one block owns. Eight float32 is one 32-byte sector, so a block's row
segment is a whole sector, and therefore the widest grid over ``W`` that wastes
none of one."""

REDUCE_THREADS = 256
"""Block width. Divisible by :data:`REDUCE_COLS`, so the block is a rectangle of
row slots by columns, 32 slots deep. The grid is ``ceil(W / REDUCE_COLS)`` blocks
per slab and cannot fill the device: the slots are the only other axis, and each
one costs a dependent load."""


def slot_tile(threads: int) -> Tile:
    """Per-thread accumulators of the row reduction.

    One float32 per thread, the block laid out as ``threads // REDUCE_COLS`` row
    slots by :data:`REDUCE_COLS` columns with the column index innermost, so a
    slot's partials are contiguous and the combine over slots reads one bank per
    column.

    Args:
        threads: Block width. Compile-time.

    Returns:
        The tile.
    """
    return Tile((threads,), (1,))


def slot_smem_bytes(threads: int) -> int:
    """Shared memory :func:`reduce_rows_kernel` holds, in bytes.

    Args:
        threads: Block width.

    Returns:
        Total bytes.
    """
    return smem_bytes([(slot_tile(threads), 4)])


@cute.kernel
def reduce_rows_kernel(
    gpartial: cute.Tensor,
    gout: cute.Tensor,
    rows: cutlass.Int32,
    width: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Sum one slab's rows into one row of the destination.

    Args:
        gpartial: ``(S, R, W)`` float32 partials, every element written by the
            producer.
        gout: ``(S, W)``, one of :data:`slinoss._precision.KERNEL_DTYPES`. The
            element type is read off the operand, so the narrowing store takes no
            second compile-time argument.
        rows: ``R``. Dynamic, so one variant covers every sequence length.
        width: ``W``. Compile-time.
        cols: Columns per block, :data:`REDUCE_COLS`. Compile-time, and divides
            ``threads``.
        threads: Block width. Compile-time.

    Invariants:
        The reduction order is fixed by the launch geometry alone: ascending row
        within a slot, then ascending slot. It has no atomics, so a rerun at one
        shape reproduces the result bit for bit. A column past ``W`` reads a
        clamped position and stores nothing. The accumulator is float32 whatever
        the destination width, and the narrowing is the one rounding on the path.
    """
    tile, slab, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sacc = smem.allocate_tensor(cutlass.Float32, slot_tile(threads).layout(), 16)
    dst = gout.element_type

    slots = threads // cols
    slot = tid // cols
    lane = tid - slot * cols
    col = tile * cols + lane
    # Clamped rather than predicated: only the last tile can run past `W`, and the
    # read is discarded by the store's guard below.
    acc = cutlass.Float32(0.0)
    for row in cutlass.range(slot, rows, slots):
        acc = acc + gpartial[slab, row, cutlass.min(col, width - 1)]

    sacc[slot * cols + lane] = acc
    cute.arch.sync_threads()

    if slot == 0:
        total = cutlass.Float32(0.0)
        # Rolled, not unrolled: the chain of adds is serial either way, and the
        # unrolled form is the slower thing to compile.
        for index in cutlass.range(slots):
            total = total + sacc[index * cols + lane]
        if col < width:
            gout[slab, col] = narrow(total, dst)


@cute.jit
def reduce_rows(
    gpartial: cute.Tensor,
    gout: cute.Tensor,
    rows: cutlass.Int32,
    slabs: cutlass.Int32,
    stream: Stream,
    width: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`reduce_rows_kernel`, one block per column tile per slab."""
    reduce_rows_kernel(gpartial, gout, rows, width, cols, threads).launch(
        grid=(-(-width // cols), slabs, 1), block=(threads, 1, 1), stream=stream
    )


def reduce_partials(
    partial: Tensor,
    *,
    out: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """Sum a partial buffer over its row axis, in one launch.

    Args:
        partial: ``(S, R, W)`` float32, contiguous CUDA. The reduction is over
            ``R``. float32 is I4 and not a preference: the producer accumulates at
            that width and the sum closes it.
        out: Destination, ``(S, W)`` contiguous CUDA in one of
            :data:`slinoss._precision.KERNEL_DTYPES`, written in full. ``None``
            allocates one.
        out_dtype: Dtype of the allocated result, float32 by default. Refused
            beside ``out``, which carries its own.

    Returns:
        ``(S, W)``: the supplied destination when there is one, a contiguous
        allocation otherwise.

    Raises:
        ValueError: On a rank other than three, an empty operand, a destination of
            the wrong shape, a non-CUDA or non-contiguous operand, or an
            ``out_dtype`` beside an ``out``.
        TypeError: If ``partial`` is not float32, or the destination dtype has no
            kernel path.
    """
    if partial.ndim != 3:
        raise ValueError(f"partial must be (S, R, W), got {tuple(partial.shape)}")
    slabs, rows, width = (int(extent) for extent in partial.shape)
    if slabs * rows * width == 0:
        raise ValueError(
            f"partial must hold at least one element, got {tuple(partial.shape)}"
        )
    check_dtypes(((partial, "partial"),), (torch.float32,), "float32 (I4)")
    if out is not None:
        if out_dtype is not None:
            raise ValueError("out carries its own dtype; pass one or the other")
        if tuple(out.shape) != (slabs, width):
            raise ValueError(f"out must be {(slabs, width)}, got {tuple(out.shape)}")
    dest = (
        torch.empty(
            slabs,
            width,
            dtype=torch.float32 if out_dtype is None else out_dtype,
            device=partial.device,
        )
        if out is None
        else out
    )
    check_dtypes(((dest, "out"),), KERNEL_DTYPES, "kernel dtypes")
    check_layout(((partial, "partial"), (dest, "out")))

    jit_launch(
        reduce_rows,
        (partial, dest, rows, slabs),
        (width, REDUCE_COLS, REDUCE_THREADS),
    )
    return dest
