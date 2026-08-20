"""Chunk-boundary and streaming epilogue of the backward.

Four scatter-adds and two copies. No math beyond the adds:

    dU[.., (c+1)L-1, :] += carry_u[.., c+1, :]     c in [0, C-1)
    dB[.., (c+1)L-1, :] += carry_b[.., c+1, :]     c in [0, C-1)
    dU[.., T-1, :]      += du_last                 when supplied
    dB[.., T-1, :]      += db_last                 when supplied
    du_prev              = carry_u[.., 0, :]       when requested
    db_prev              = carry_b[.., 0, :]       when requested

``b_{t-1}`` and ``u_{t-1}`` are the same tensors read one token earlier, so their
cotangents shift forward by one token. Inside a chunk the shift belongs to the
producing kernel. Across a boundary it lands on the previous chunk's last token,
which is another block's tile, and at ``c = 0`` it leaves the sequence as the
streaming feedback gradient. Those are the terms here.

Block ``c`` owns the last token of chunk ``c``, so it reads carry slot ``c+1``.
Indexing the other way -- block ``c`` writing token ``cL-1`` -- would put the
write in chunk ``c-1``'s tile, which the split reduction below also writes, and
two blocks would race for one row.

Split reduction. A ``dB`` producer split ``S`` ways writes no ``dB``; it writes
``(B,G,S,T,3N)`` float32 partials, and the sum over ``S`` is taken here. So at
``S > 1`` this kernel writes every ``dB`` row of its chunk and the boundary term
is added on top of the narrowed sum, which leaves a boundary row carrying one
rounding more than an interior one. At ``S == 1`` the producer writes ``dB``
itself, there is no partial buffer, and every row this kernel touches is
read-modified. A one-split partial buffer is a staging copy, so it is refused
rather than reduced.

Grouping. ``carry_b``, the partials, ``db_last`` and ``db_prev`` are already
reduced over the heads of a group, so the whole ``b`` side runs on one head per
group: the group's first, ``h == g * (H // G)``. Any other head would add the
same group's carry a second time. At ``G == H`` the predicate reads
``hidx == hidx``.

Launch: ``(C, B, H)`` blocks of ``THREADS``. A block updates ``P`` elements of one
``dU`` row and, on the owning head, ``3N`` elements of one ``dB`` row, each from
its own thread index in a stride loop, so at ``standard``, where ``P`` and ``3N``
are both 48, 48 of 128 threads iterate in either pass. The width is
:data:`slinoss.ops.so3ssd.cute.common.THREADS` because every launch in the tree
is, and one launch geometry for the tree is worth more here than the idle threads
cost. They are not tuned away.

No shared memory, no barrier, no chunk-local prefix, no GEMM. Nothing is shared
between threads and every value read is already a cotangent.

SERIAL-tiny at ``S == 1``, by the budget clause of the kernel-class rule rather
than by a bandwidth fraction. Traffic is ``(2e + 4)`` bytes per updated element --
a read and a write of the gradient row at the activation itemsize ``e``, plus one
float32 carry row -- over ``C - 1`` rows of ``P`` per ``(B,H)`` and ``C - 1`` rows
of ``3N`` per ``(B,G)``, and one float32 row each for the carry-out. At ``B=4 H=18
T=2048 P=64 3N=240 L=64`` with ``G=1`` and bfloat16 activations that is 1.38 MB, and
at ``standard`` 1.14 MB.

None of it is DRAM traffic. The rows touched are ``C - 1`` of every ``C``, so the
whole working set is 1.4 MB against a 6 MB L2 and the device sees write-back alone:
0.10 MB and 0.07 MB measured per launch at the first shape, 0.12 MB at ``standard``,
all of it writes, 2.4% to 4.8% of DRAM speed-of-light. There is no bandwidth verdict
to take below L2, and the percentage-of-floor the class rule computes is meaningless
at this byte count.

Measured on one RTX A6000, clocks unlocked, nothing but the MPS daemon resident
before and after: 4.4 us and 4.7 us per launch at the first shape over two runs, and
3.7 us at ``standard``. A launch this short is at the limit of the profiler's
resolution -- the spread across NCU replay passes is 27% and 50% respectively -- so
the figure is a range and not a median. 16 registers per thread at the first shape,
19 at ``standard``, no shared memory, no bank conflicts, and 2304 and 1536 blocks.

The budget clause is taken against the step, not against the forward. The backward
launches this kernel once per layer, 13 times per step at the acceptance geometry,
so 57 to 61 us of a step whose device time measures 461 to 465 ms on the same host:
0.013%. It does not appear in the printed rows of
``scripts/perf/attribute_step.py``, which cut off two orders of magnitude above it,
which is why ``scripts/perf/profile_boundary_bwd.py`` exists.

Neither the grid nor any allocation follows ``L``: the block count is ``C = T / L``
and the traffic is ``C - 1`` rows, so both fall as the chunk length rises. At
``L = 128`` the launch is 1152 blocks, still above twice the SM count.

The ``S > 1`` path is not covered by that estimate and is not SERIAL-tiny. It
streams ``S`` passes over a ``(B,G,S,T,3N)`` float32 tensor and one over ``dB``:
at ``standard`` with ``S = 2``, 37.7 MB read and 9.4 MB written. That is a
DRAM-bound reduction and has to be declared and measured as one when a producer
that emits partials lands.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import Stream, cute_dtype, jit_launch, narrow, widen
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.so3ssd.cute.common import THREADS
from slinoss.ops.so3ssd.cute.guard import (
    Named,
    check_dtypes,
    check_layout,
    check_pinned,
    check_pitched,
)

__all__ = [
    "BoundaryStream",
    "boundary_backward",
    "boundary_bwd",
    "boundary_bwd_kernel",
]


# ---------------------------------------------------------------------------
# Row updates
# ---------------------------------------------------------------------------
#
# Every one of these walks a row from the calling thread's index in steps of the
# block width. The bound is compile-time and the start is not, so the trip count
# is dynamic: threads at or past the row width iterate zero times, which is how
# a row narrower than the block is covered without a predicate.


@cute.jit
def _add_carry(
    gdst: cute.Tensor,
    gcarry: cute.Tensor,
    bidx: cutlass.Int32,
    xidx: cutlass.Int32,
    token: cutlass.Int32,
    slot: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
) -> None:
    """Add one chunk slot of a carry into one token row.

    Args:
        gdst: ``(B,X,T,W)`` gradient at the activation width. Read-modified.
        gcarry: ``(B,X,C,W)`` float32 carry.
        bidx: Batch index.
        xidx: Head index for ``dU``, group index for ``dB``.
        token: Destination token.
        slot: Chunk slot read from the carry.
        tid: Thread index in the block.
        threads: Block width. Compile-time.
        width: ``P`` or ``3N``. Compile-time.
    """
    dst = gdst.element_type
    for col in cutlass.range(tid, width, threads):
        total = widen(gdst[bidx, xidx, token, col], dst) + gcarry[bidx, xidx, slot, col]
        gdst[bidx, xidx, token, col] = narrow(total, dst)


@cute.jit
def _add_tap(
    gdst: cute.Tensor,
    gsrc: cute.Tensor,
    bidx: cutlass.Int32,
    xidx: cutlass.Int32,
    token: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
    enabled: cutlass.Constexpr,
) -> None:
    """Add a per-``(B,X)`` cotangent into one token row.

    Args:
        gdst: ``(B,X,T,W)`` gradient at the activation width. Read-modified.
        gsrc: ``(B,X,W)`` cotangent at the activation width.
        bidx: Batch index.
        xidx: Head index for ``dU``, group index for ``dB``.
        token: Destination token.
        tid: Thread index in the block.
        threads: Block width. Compile-time.
        width: ``P`` or ``3N``. Compile-time.
        enabled: Whether the cotangent was supplied. Compile-time: a disabled tap
            emits nothing, so the placeholder standing in for an absent one is
            never loaded and costs no instruction.
    """
    if cutlass.const_expr(enabled):
        dst = gdst.element_type
        src = gsrc.element_type
        for col in cutlass.range(tid, width, threads):
            total = widen(gdst[bidx, xidx, token, col], dst) + widen(
                gsrc[bidx, xidx, col], src
            )
            gdst[bidx, xidx, token, col] = narrow(total, dst)


@cute.jit
def _copy_first_slot(
    gdst: cute.Tensor,
    gcarry: cute.Tensor,
    bidx: cutlass.Int32,
    xidx: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
    enabled: cutlass.Constexpr,
) -> None:
    """Write chunk slot 0 of a carry into a ``(B,X,W)`` float32 output.

    Args:
        gdst: ``(B,X,W)`` float32, written.
        gcarry: ``(B,X,C,W)`` float32 carry.
        bidx: Batch index.
        xidx: Head index for ``du_prev``, group index for ``db_prev``.
        tid: Thread index in the block.
        threads: Block width. Compile-time.
        width: ``P`` or ``3N``. Compile-time.
        enabled: Whether the carry-out was requested. Compile-time: when it was
            not, the destination is the carry itself and this must emit no store.
    """
    if cutlass.const_expr(enabled):
        for col in cutlass.range(tid, width, threads):
            gdst[bidx, xidx, col] = gcarry[bidx, xidx, 0, col]


@cute.jit
def _reduce_splits(
    gdB: cute.Tensor,
    gpartial: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    splits: cutlass.Constexpr,
) -> None:
    """Sum one chunk's split partials into ``dB``.

    The token loop runs to the chunk's valid length, so a ragged tail needs no
    predicate and no row past ``T`` is touched. The split loop is compile-time,
    so a token's ``S`` loads issue together.

    Args:
        gdB: ``(B,G,T,3N)`` gradient at the activation width. Written.
        gpartial: ``(B,G,S,T,3N)`` float32 partials.
        bidx: Batch index.
        gidx: Group index.
        t0: First token of the chunk.
        valid: Tokens of the chunk inside the sequence.
        tid: Thread index in the block.
        threads: Block width. Compile-time.
        dim: ``3N``. Compile-time.
        splits: ``S``. Compile-time. At 1 there is no partial buffer, the producer
            wrote ``dB`` itself, and this emits nothing.
    """
    if cutlass.const_expr(splits > 1):
        dst = gdB.element_type
        for step in cutlass.range(valid):
            token = t0 + step
            for col in cutlass.range(tid, dim, threads):
                total = cutlass.Float32(0.0)
                for s in cutlass.range_constexpr(splits):
                    total = total + gpartial[bidx, gidx, s, token, col]
                gdB[bidx, gidx, token, col] = narrow(total, dst)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.kernel
def boundary_bwd_kernel(
    gcarry_u: cute.Tensor,
    gcarry_b: cute.Tensor,
    gpartial: cute.Tensor,
    gdu_last: cute.Tensor,
    gdb_last: cute.Tensor,
    gdU: cute.Tensor,
    gdB: cute.Tensor,
    gdu_prev: cute.Tensor,
    gdb_prev: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    has_du_last: cutlass.Constexpr,
    has_db_last: cutlass.Constexpr,
    want_prev: cutlass.Constexpr,
) -> None:
    """Apply one chunk's boundary row and, on the first block, the carry-out.

    One block per ``(chunk, batch, head)``.

    Args:
        gcarry_u: ``(B,H,C,P)`` float32 per-chunk ``dU`` carry.
        gcarry_b: ``(B,G,C,3N)`` float32 per-chunk ``dB`` carry.
        gpartial: ``(B,G,S,T,3N)`` float32 ``dB`` partials. Read only when
            ``splits > 1``; otherwise a placeholder of the same rank and dtype.
        gdu_last: ``(B,H,P)`` cotangent of ``u_last``, or a placeholder. Read only
            when ``has_du_last``.
        gdb_last: ``(B,G,3N)`` cotangent of ``b_last``, or a placeholder. Read only
            when ``has_db_last``.
        gdU: ``(B,H,T,P)`` activation-dtype gradient. Read-modified at the chunk
            boundary and at ``T-1``.
        gdB: ``(B,G,T,3N)`` activation-dtype gradient. Written over the chunk when
            ``splits > 1``, then read-modified at the same two tokens.
        gdu_prev: ``(B,H,P)`` float32, written when ``want_prev``, else a
            placeholder.
        gdb_prev: ``(B,G,3N)`` float32, written when ``want_prev``, else a
            placeholder.
        seqlen: ``T``. Dynamic.
        chunks: ``C``. Dynamic.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        per_group: ``H // G``, heads sharing one group. Compile-time.
        splits: ``S``, 1 when there is no partial buffer. Compile-time.
        has_du_last: Whether ``gdu_last`` was supplied. Compile-time.
        has_db_last: Whether ``gdb_last`` was supplied. Compile-time.
        want_prev: Whether the streaming carry-out is wanted. Compile-time.

    Invariants:
        ``C == ceil(T / L)`` and ``per_group`` divides ``H``, both checked on the
        host. Every row of ``gdU`` and ``gdB`` this kernel touches is touched by
        exactly one block, and within a block by exactly one thread, so the
        read-modify pairs need no atomic and the write-then-add on a boundary row
        is ordered by one thread's program order.
    """
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    # Only the b side is grouped. The divide is trace-time optional so the
    # ungrouped shape emits none rather than an identity one.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group
    # The group's first head carries the b side. Block-uniform, so the loads
    # under it are not in a divergent branch; at per_group == 1 it is hidx == hidx.
    owner = hidx == gidx * per_group

    # First, because it writes the boundary row the next block of statements then
    # read-modifies. Same block and same thread for a given column, so program
    # order is the whole ordering argument.
    t0 = cidx * chunk
    if owner:
        _reduce_splits(
            gdB,
            gpartial,
            bidx,
            gidx,
            t0,
            cutlass.min(cutlass.Int32(chunk), seqlen - t0),
            tid,
            threads,
            dim,
            splits,
        )

    # The last token of this chunk takes the next chunk's carry. The last chunk
    # has no successor, so its block adds nothing here.
    if cidx + 1 < chunks:
        slot = cidx + 1
        token = slot * chunk - 1
        _add_carry(gdU, gcarry_u, bidx, hidx, token, slot, tid, threads, rows)
        if owner:
            _add_carry(gdB, gcarry_b, bidx, gidx, token, slot, tid, threads, dim)

    if cidx + 1 == chunks:
        # T-1 is never a chunk boundary: C == ceil(T/L) makes T > (C-1)*L.
        last = seqlen - 1
        _add_tap(gdU, gdu_last, bidx, hidx, last, tid, threads, rows, has_du_last)
        if owner:
            _add_tap(gdB, gdb_last, bidx, gidx, last, tid, threads, dim, has_db_last)

    if cidx == 0:
        _copy_first_slot(gdu_prev, gcarry_u, bidx, hidx, tid, threads, rows, want_prev)
        if owner:
            _copy_first_slot(
                gdb_prev, gcarry_b, bidx, gidx, tid, threads, dim, want_prev
            )


@cute.jit
def boundary_bwd(
    gcarry_u: cute.Tensor,
    gcarry_b: cute.Tensor,
    gpartial: cute.Tensor,
    gdu_last: cute.Tensor,
    gdb_last: cute.Tensor,
    gdU: cute.Tensor,
    gdB: cute.Tensor,
    gdu_prev: cute.Tensor,
    gdb_prev: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    stream: Stream,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    has_du_last: cutlass.Constexpr,
    has_db_last: cutlass.Constexpr,
    want_prev: cutlass.Constexpr,
) -> None:
    """Launch :func:`boundary_bwd_kernel`.

    ``dtype`` is the activation dtype. The kernel reads element types off its
    tensors, so nothing here consumes it; it is in the compile-time run because
    that run is the executor cache key and the element type shapes the generated
    code. Batch, head, chunk count, and sequence length are dynamic.
    """
    boundary_bwd_kernel(
        gcarry_u,
        gcarry_b,
        gpartial,
        gdu_last,
        gdb_last,
        gdU,
        gdB,
        gdu_prev,
        gdb_prev,
        seqlen,
        chunks,
        threads,
        chunk,
        rows,
        dim,
        per_group,
        splits,
        has_du_last,
        has_db_last,
        want_prev,
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------


def _check_shapes(
    carry_u: Tensor, carry_b: Tensor, dU: Tensor, dB: Tensor, chunk_size: int
) -> tuple[int, int, int, int, int, int, int]:
    """Check the four required operands against each other.

    ``dU`` sets ``(B, H, T, P)``; ``carry_b`` sets ``G`` and ``3N``, so a caller
    cannot claim one grouping and hand over another. ``C`` is derived from ``T``
    and ``chunk_size`` rather than read off a carry: a ``chunk_size`` that does
    not match the one the producing kernels ran at would otherwise place every
    boundary row at the wrong token and raise nothing.

    Args:
        carry_u: ``(B,H,C,P)``.
        carry_b: ``(B,G,C,3N)``.
        dU: ``(B,H,T,P)``.
        dB: ``(B,G,T,3N)``.
        chunk_size: ``L``.

    Returns:
        ``(B, H, G, C, T, P, 3N)``.

    Raises:
        ValueError: On a rank or shape mismatch, an empty sequence, a
            non-positive ``chunk_size``, or a ``G`` that does not divide ``H``.
    """
    if dU.ndim != 4:
        raise ValueError(f"dU must be (B,H,T,P), got {tuple(dU.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in dU.shape)
    if seqlen < 1:
        raise ValueError(f"dU must hold at least one token, got {tuple(dU.shape)}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    chunks = -(-seqlen // chunk_size)
    if tuple(carry_u.shape) != (bsz, heads, chunks, rows):
        raise ValueError(
            f"carry_u must be {(bsz, heads, chunks, rows)} at L={chunk_size}, "
            f"got {tuple(carry_u.shape)}"
        )
    if (
        carry_b.ndim != 4
        or int(carry_b.shape[0]) != bsz
        or int(carry_b.shape[2]) != chunks
    ):
        raise ValueError(
            f"carry_b must be (B,G,C,3N) with B={bsz} and C={chunks}, "
            f"got {tuple(carry_b.shape)}"
        )
    groups, dim = int(carry_b.shape[1]), int(carry_b.shape[3])
    if groups < 1 or heads % groups != 0:
        raise ValueError(f"carry_b carries G={groups}, which does not divide H={heads}")
    if tuple(dB.shape) != (bsz, groups, seqlen, dim):
        raise ValueError(
            f"dB must be {(bsz, groups, seqlen, dim)}, got {tuple(dB.shape)}"
        )
    return bsz, heads, groups, chunks, seqlen, rows, dim


def _check_splits(partial_bc: Tensor | None, shape: tuple[int, int, int, int]) -> int:
    """Check the split partial buffer and return ``S``.

    Args:
        partial_bc: ``(B,G,S,T,3N)`` or ``None``.
        shape: ``(B, G, T, 3N)``.

    Returns:
        ``S``, or 1 when there is no buffer.

    Raises:
        ValueError: On a rank or shape mismatch, or on a buffer carrying one
            split. A single split writes ``dB`` directly; routing it through a
            partial buffer would be a staging copy.
    """
    if partial_bc is None:
        return 1
    bsz, groups, seqlen, dim = shape
    if partial_bc.ndim != 5:
        raise ValueError(
            f"partial_bc must be (B,G,S,T,3N), got {tuple(partial_bc.shape)}"
        )
    splits = int(partial_bc.shape[2])
    if splits < 2:
        raise ValueError(
            f"partial_bc must carry at least two splits, got S={splits}; "
            "one split writes dB directly"
        )
    if tuple(partial_bc.shape) != (bsz, groups, splits, seqlen, dim):
        raise ValueError(
            f"partial_bc must be {(bsz, groups, splits, seqlen, dim)}, "
            f"got {tuple(partial_bc.shape)}"
        )
    return splits


def _check_taps(
    du_last: Tensor | None,
    db_last: Tensor | None,
    urow: tuple[int, int, int],
    brow: tuple[int, int, int],
) -> None:
    """Check the two end-of-sequence cotangents.

    Independent of each other: ``u_last`` is per head and ``b_last`` is a time
    slice of the grouped ``B``, and the forward emits both whether or not a
    caller differentiates both.

    Args:
        du_last: ``(B,H,P)`` or ``None``.
        db_last: ``(B,G,3N)`` or ``None``.
        urow: ``(B, H, P)``.
        brow: ``(B, G, 3N)``.

    Raises:
        ValueError: On a shape mismatch.
    """
    if du_last is not None and tuple(du_last.shape) != urow:
        raise ValueError(f"du_last must be {urow}, got {tuple(du_last.shape)}")
    if db_last is not None and tuple(db_last.shape) != brow:
        raise ValueError(f"db_last must be {brow}, got {tuple(db_last.shape)}")


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


class BoundaryStream(NamedTuple):
    """Streaming carry-out of the backward.

    ``dU`` and ``dB`` are updated in place, so they are not repeated here.

    Attributes:
        du_prev: ``(B,H,P)`` float32 cotangent of ``u_{-1}``, or ``None`` when it
            was not requested.
        db_prev: ``(B,G,3N)`` float32 cotangent of ``b_{-1}``, or ``None``.
    """

    du_prev: Tensor | None
    db_prev: Tensor | None


def boundary_backward(
    carry_u: Tensor,
    carry_b: Tensor,
    dU: Tensor,
    dB: Tensor,
    chunk_size: int,
    *,
    partial_bc: Tensor | None = None,
    du_last: Tensor | None = None,
    db_last: Tensor | None = None,
    want_prev: bool = False,
) -> BoundaryStream:
    """Add every chunk boundary and every streaming term into ``dU`` and ``dB``.

    Args:
        carry_u: ``(B,H,C,P)`` float32, contiguous. Slot ``c`` is the cotangent
            chunk ``c``'s first token sends to the token before it.
        carry_b: ``(B,G,C,3N)`` float32, contiguous. Slot ``c`` likewise, already
            reduced over the heads of the group.
        dU: ``(B,H,T,P)``, one of :data:`slinoss._precision.KERNEL_DTYPES`,
            contiguous. Read-modified at ``(c+1)L-1`` and ``T-1``.
        dB: ``(B,G,T,3N)``, the dtype of ``dU``, pitched. The gradient of one
            column band of the mixer's fused projection, so its token stride is the
            projection width and a contiguous buffer is the case where the two
            agree. Written over every token when ``partial_bc`` is given, then
            read-modified at the same two tokens as ``dU``.
        chunk_size: ``L``. Must match the chunk length the carries were produced
            at; ``C`` is derived from it.
        partial_bc: ``(B,G,S,T,3N)`` float32, contiguous, ``S >= 2``. The split
            ``dB`` partials. Omitted when the producer wrote ``dB`` itself.
        du_last: ``(B,H,P)`` cotangent of ``u_last``, the dtype of ``dU``.
        db_last: ``(B,G,3N)`` cotangent of ``b_last``, the dtype of ``dU``.
        want_prev: Return the streaming carry-out from chunk slot 0.

    Returns:
        A :class:`BoundaryStream`. Both fields are ``None`` unless ``want_prev``.

    Raises:
        ValueError: On a layout, rank, shape, or split-count violation, an empty
            sequence, a non-positive ``chunk_size``, a ``G`` that does not divide
            ``H``, or a float32-pinned operand that is not float32.
        TypeError: On a gradient dtype with no kernel path, or on gradients that
            do not share one dtype.
    """
    taps: Named = ()
    if du_last is not None:
        taps = (*taps, (du_last, "du_last"))
    if db_last is not None:
        taps = (*taps, (db_last, "db_last"))
    activations: Named = ((dU, "dU"), (dB, "dB"), *taps)
    pinned: Named = ((carry_u, "carry_u"), (carry_b, "carry_b"))
    if partial_bc is not None:
        pinned = (*pinned, (partial_bc, "partial_bc"))

    # dB is the one pitched operand. Every other tensor here is either a whole
    # buffer or a reduction the kernels allocate, so none of them is a band.
    check_layout(((dU, "dU"), *taps, *pinned))
    check_pitched(((dB, "dB"),))
    dtype = check_dtypes(activations, KERNEL_DTYPES, "kernel dtypes")
    check_pinned(pinned)
    bsz, heads, groups, chunks, seqlen, rows, dim = _check_shapes(
        carry_u, carry_b, dU, dB, chunk_size
    )
    splits = _check_splits(partial_bc, (bsz, groups, seqlen, dim))
    _check_taps(du_last, db_last, (bsz, heads, rows), (bsz, groups, dim))

    # Every element of both is written by the kernel: one block per batch and head
    # covers dU's rows, and the group's first head covers dB's. Nothing is filled
    # on the host.
    opts = {"dtype": torch.float32, "device": dU.device}
    du_prev = torch.empty(bsz, heads, rows, **opts) if want_prev else None
    db_prev = torch.empty(bsz, groups, dim, **opts) if want_prev else None

    # Placeholders keep one launch signature. None is read: every branch that
    # would read one is closed at compile time. Each carries the shape and the
    # dtype of the operand it stands in for, so the absent case adds no compiled
    # variant of its own.
    jit_launch(
        boundary_bwd,
        (
            carry_u,
            carry_b,
            carry_b.unsqueeze(2) if partial_bc is None else partial_bc,
            dU[:, :, 0] if du_last is None else du_last,
            dB[:, :, 0] if db_last is None else db_last,
            dU,
            dB,
            carry_u[:, :, 0] if du_prev is None else du_prev,
            carry_b[:, :, 0] if db_prev is None else db_prev,
            seqlen,
            chunks,
            bsz,
            heads,
        ),
        (
            cute_dtype(dtype),
            THREADS,
            chunk_size,
            rows,
            dim,
            heads // groups,
            splits,
            du_last is not None,
            db_last is not None,
            want_prev,
        ),
    )
    return BoundaryStream(du_prev=du_prev, db_prev=db_prev)
