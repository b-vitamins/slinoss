"""Chunk-start state cotangent, the ``y_off`` half of the output cotangent.

The forward emits ``y_off_t = exp(2*lp_t) * <crot_t, zstart>`` with
``crot_t = R(Q_t)^T c_t``. Its cotangent in the chunk-start state is one GEMM:

    dgram(L,P)    = dy * exp(2*lp)
    dzstart(P,3N) = dgram^T crot

which is the increment form with ``dgram`` in place of ``u`` and ``crot`` in place
of the rotated forcing, so both operands are staged token-major and both views
transpose. ``dcrot``, the other half of the same product, belongs to the ``dC``
kernel and is not formed here.

The weight rides ``dgram``, which is ``P`` wide, rather than ``crot``, which is
``3N`` wide. Neither placement adds a rounding, because both operands already pass
through float32 on their way into shared memory, so the choice is the narrower
operand. ``exp(2*lp) <= 1`` by I1 and comes from :func:`slinoss._cute.decay` on the
prefix itself, never from a ratio of two exponentials (I3).

Staging. ``crot`` is transformed on the way in by :func:`stage_rotated` and never
reaches global memory. The transform table is built at ``mats == 1``, which writes
``Ac`` alone: no tap matrix, so no ``K`` is read and no tap tile is allocated.

A ragged tail needs no separate path. Both staging passes zero every row at or past
``valid``, so the K extent stays the whole chunk and the padded rows contribute
nothing.

The K extent is the whole chunk, one GEMM and no slice loop, so every tile in
:func:`start_smem_bytes` is proportional to ``L``. Against the 101,376 B carveout:
``L=64`` at ``P=64, 3N=240`` is 46,336 B and two blocks per SM, ``L=128`` is
92,672 B and one, ``L=192`` is 139,008 B and refused. At ``3N=48`` the same
progression is four, two, and one. This is the only kernel on the chunk-start path
whose footprint follows ``L``. The table's padded pitch costs ``12 * L`` bytes of
that and changes no residency at any shape the bench covers.

Dispatch order. The grid is head-fastest. ``C`` is per group, so at ``G < H`` the
``H // G`` blocks sharing one group's readout tile have to be co-resident for L2 to
hold it. Chunk-fastest dispatch instead passes a whole head's blocks, and their
``dzstart`` writes, between two reads of the same tile, and refetches it every time.

Measured on one A6000, clocks unlocked, one launch per NCU run, nothing but the MPS
daemon resident before and after. At ``B=4 H=18 T=2048 P=64 3N=240 L=64`` with
``G=1``, per launch:

    dy       4*18*2048*64*2    =  18.87 MB read
    trans    4*18*2048*4*4     =   2.36 MB read
    C        4*2048*240*2      =   3.93 MB read, one band for all 18 heads
    dzstart  4*18*32*64*240*4  = 141.56 MB written
                                 166.72 MB

Charging ``C`` once per head instead of once gives 233.57 MB. Measured 167.14 MB and
167.03 MB over two runs, 25.49 MB of it read against the 25.16 MB the table reads:
L2 holds the band across the heads that share it. Chunk-fastest dispatch measured
217.88 MB and 217.79 MB, 76.14 MB and 76.11 MB read, which is 50.7 MB of the same
band fetched again.

On the fitted copy law ``t = c + bytes/B``, ``c = 4.01 us`` and ``B = 682.8 GB/s``
with a worst residual of 0.23%, 166.72 MB floors at 248.6 us. Measured 316.8 us and
315.6 us, 78.8% of that floor, against 338.3 us and 338.8 us chunk-fastest. So the
reorder removes 23% of the bytes and 6% of the time, and the bus stops being what
bounds the launch: DRAM speed-of-light falls from 88.1% to 72.1% and
``mio_throttle`` rises from 8.7% to 15.8% of stalls. What bounds it then is
occupancy, 16.7% theoretical and 16.0% achieved at two blocks per SM. Below the 85%
bar, and the byte count is no longer the reason.

At ``standard`` the traffic is ``dy 9.44 + trans 1.57 + C 9.44 + dzstart 14.16 =
34.60 MB``, and ``G == H`` there so no band is shared and the two byte counts
coincide. Measured 34.36 MB, 54.7 us against a 54.9 us floor, 100.3% of it, at four
blocks per SM. Dispatch order is neutral where nothing is shared: 34.50 MB and
54.3 us chunk-fastest, inside the 4.4% pass spread.

The same 34.60 MB carries ``1536 * 2*64*48*64 = 604 MFLOP``, 17.5 flop/byte against
a ridge point of 164, which is what makes the padded M mode and the recomputed
prefixes affordable. Neither was timed against a variant without it.
"""

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Stream,
    Tile,
    assert_smem_fits,
    cute_dtype,
    jit_launch,
    smem_bytes,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC_SOLE,
    THREADS,
    scalar_tile,
    table_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.guard import (
    Named,
    check_extents,
    check_layout,
    check_operands,
    check_pinned,
    check_pitched,
    check_shapes,
)
from slinoss.ops.so3ssd.cute.mma import (
    SMEM_SEGMENT,
    make_mma,
    mma_acc,
    mma_gemm,
    mma_rows,
    mma_store,
    operand_tile,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes
from slinoss.ops.so3ssd.cute.table import (
    TABLE_PITCH,
    TABLE_QUAD,
    build_table,
    stage_pad,
    stage_raw,
    stage_rotated,
    stage_trans,
    weight_rows,
)

__all__ = [
    "chunk_start_backward",
    "chunk_start_bwd",
    "chunk_start_bwd_kernel",
    "gram_tile",
    "rotated_tile",
    "start_chunk",
    "start_loaded",
    "start_scanned",
    "start_smem_bytes",
]


def gram_tile(chunk: int, rows: int) -> Tile:
    """Weighted output cotangent tile, ``(L, pitch)``.

    ``P`` is the M mode of this kernel's output, so the row width is the rounded
    extent rather than ``P`` itself.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return operand_tile(chunk, mma_rows(rows))


def rotated_tile(chunk: int, dim: int) -> Tile:
    """Rotated readout tile, ``(L, pitch)``.

    Args:
        chunk: ``L``.
        dim: ``3N``.
    """
    return operand_tile(chunk, dim)


def start_smem_bytes(chunk: int, rows: int, dim: int, itemsize: int = 2) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_start_bwd_kernel` allocates, in the same order.
    Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 1, TABLE_PITCH), 4),
            (rotated_tile(chunk, dim), itemsize),
            (gram_tile(chunk, rows), itemsize),
        ]
    )


@cute.jit
def _read_run(
    src: cute.Tensor,
    dst: cute.Tensor,
    total: cutlass.Constexpr,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
) -> None:
    """Copy a dense float32 run from global to shared, one segment a thread a step.

    Args:
        src: Dense float32 global row of at least ``total`` elements.
        dst: Dense float32 shared tile of at least ``total`` elements.
        total: Elements to copy. Compile-time, and a multiple of
            :data:`slinoss.ops.so3ssd.cute.table.TABLE_QUAD` because
            :func:`slinoss.ops.so3ssd.cute.guard.check_extents` holds ``L`` to a
            multiple of 16.
        tid: Thread index within the block.
        threads: Block width. Compile-time.

    Invariants:
        The alignment claim is restated on both iterators, because a tile arriving
        as a parameter reports one element whatever its allocation asked for and
        ``autovec_copy`` caps the access at the claim. The global run's origin is a
        whole number of ``L``-element or ``4L``-element records and ``L`` is a
        multiple of 16, so a record starts on a segment.
    """
    quads = total // TABLE_QUAD
    unit = cute.make_layout((quads, TABLE_QUAD), stride=(TABLE_QUAD, 1))
    from_words = cute.make_tensor(src.iterator.align(SMEM_SEGMENT), unit)
    to_words = cute.make_tensor(dst.iterator.align(SMEM_SEGMENT), unit)
    for step in cutlass.range_constexpr(-(-quads // threads)):
        q = tid + step * threads
        if cutlass.const_expr(quads % threads == 0):
            cute.autovec_copy(from_words[(q, None)], to_words[(q, None)])
        else:
            if q < quads:
                cute.autovec_copy(from_words[(q, None)], to_words[(q, None)])


@cute.jit
def start_scanned(
    gtrans: cute.Tensor,
    strans: cute.Tensor,
    slp: cute.Tensor,
    squat: cute.Tensor,
    seqlen: cutlass.Int32,
    cidx: cutlass.Int32,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    """Stage one chunk's transitions and rescan the two prefixes from them.

    Two barriers: the scan reads what the staging wrote, and
    :func:`start_chunk` reads what the scan wrote.

    Args:
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        strans: ``(4,L)`` float32 transition tile, written.
        slp: ``(L,)`` float32 log-decay prefix tile, written.
        squat: ``(4,L)`` float32 quaternion prefix tile, written.
        seqlen: ``T``. Dynamic.
        cidx: Chunk. Dynamic.
        bidx: Batch index. Dynamic.
        hidx: Head index. Dynamic.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
    """
    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)
    stage_trans(gtrans[bidx, hidx, None, None], strans, t0, valid, tid, threads, chunk)
    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()


@cute.jit
def start_loaded(
    gslp: cute.Tensor,
    gsquat: cute.Tensor,
    slp: cute.Tensor,
    squat: cute.Tensor,
    cidx: cutlass.Int32,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    """Read one chunk's two prefixes, already scanned, into shared memory.

    The scan :func:`start_scanned` runs is warp 0's alone, so the rest of the block
    waits at its barrier. Reading the result of
    :func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_prefix_bwd_kernel`
    instead makes the prefixes ``5 * L`` words of the staging pass, which leaves one
    barrier where the scan needed two.

    The transitions do not come with them. Only the scan reads ``strans``:
    :func:`slinoss.ops.so3ssd.cute.table.build_table` at ``mats == 1`` composes
    ``Ac`` from the quaternion prefix alone, and nothing else in :func:`start_chunk`
    or in a chunk-serial caller's loop touches that tile. So the read arm stages
    ``4 * L`` fewer words a chunk than the rescan rather than ``5 * L`` more, and the
    tile it leaves unwritten is the slot ``build_table`` is documented to ignore.

    Args:
        gslp: ``(B,H,C,L)`` float32 inclusive log-scale scan.
        gsquat: ``(B,H,C,4,L)`` float32 inclusive quaternion prefix product,
            component-major and renormalized once.
        slp: ``(L,)`` float32 log-decay prefix tile, written.
        squat: ``(4,L)`` float32 quaternion prefix tile, written.
        cidx: Chunk. Dynamic.
        bidx: Batch index. Dynamic.
        hidx: Head index. Dynamic.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.

    Invariants:
        ``gslp`` and ``gsquat`` hold the output of the same :func:`chunk_prefixes`
        code, renormalization included, so the tiles this fills are bitwise what
        the rescan would have written.
    """
    _read_run(gslp[bidx, hidx, cidx, None], slp, chunk, tid, threads)
    _read_run(gsquat[bidx, hidx, cidx, None, None], squat, 4 * chunk, tid, threads)
    cute.arch.sync_threads()


@cute.jit
def start_chunk(
    gdy: cute.Tensor,
    gc: cute.Tensor,
    dst: cute.Tensor,
    strans: cute.Tensor,
    slp: cute.Tensor,
    squat: cute.Tensor,
    stable: cute.Tensor,
    scrot: cute.Tensor,
    sdy: cute.Tensor,
    acc: cute.Tensor,
    tiled_mma: cute.TiledMma,
    seqlen: cutlass.Int32,
    cidx: cutlass.Int32,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    gidx: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    fenced: cutlass.Constexpr,
) -> None:
    """One chunk's contraction, from the table build to the store.

    Split out of :func:`chunk_start_bwd_kernel` so the block-per-chunk launch and
    the chunk-serial launch run the same body rather than two copies of it. The
    prefixes are the caller's, because the two forms of getting them --
    :func:`start_scanned` and :func:`start_loaded` -- cost different barriers and
    the choice is the launch's.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gc: ``(B,G,T,3N)`` operand-dtype output vectors. A caller contracting one
            lane band hands over a view whose origin is that band's first column.
        dst: Float32 ``(P,dim)`` destination, row-major and contiguous, already
            sliced to this chunk. Global for the launch that writes ``dzstart``,
            shared for the launch that consumes it on chip.
        strans: ``(4,L)`` float32 transition tile, filled.
        slp: ``(L,)`` float32 log-decay prefix tile, filled.
        squat: ``(4,L)`` float32 quaternion prefix tile, filled.
        stable: ``(1,L,TABLE_PITCH)`` float32 transform table, ``Ac`` alone. Allocated
            at :data:`slinoss.ops.so3ssd.cute.table.TABLE_PITCH`, so the caller's
            allocation and the readers below must state that pitch together.
        scrot: ``(L,pitch)`` operand-dtype rotated readout tile. Padded columns
            are zero on entry and are not restaged here.
        sdy: ``(mpad,pitch)`` operand-dtype weighted cotangent tile, same.
        acc: Float32 accumulator fragment from :func:`mma_acc`. Zeroed here, so
            one fragment serves every chunk a block walks.
        tiled_mma: From :func:`make_mma`.
        seqlen: ``T``. Dynamic.
        cidx: Chunk this call contracts. Dynamic.
        bidx: Batch index. Dynamic.
        hidx: Head index. Dynamic.
        gidx: Group index, ``hidx // (H // G)``. Dynamic.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: Columns contracted: ``3N``, or the width of one lane band of it.
            Compile-time.
        fenced: Whether to separate the contraction from the store with a barrier.
            Compile-time. Required of a caller whose ``dst`` overlaps ``sdy`` or
            ``scrot``, since without it one warp stores its accumulator over an
            operand another warp is still reading.

    Invariants:
        Every barrier here is reached by the whole block, so the body is safe to
        call inside a loop whose trip count is uniform. On entry ``strans``,
        ``slp`` and ``squat`` hold this chunk's transitions and prefixes and a
        barrier has published them, and no thread may still be reading ``sdy`` or
        ``scrot`` from a previous call.
    """
    lanes = dim // 3
    mpad = mma_rows(rows)
    lda = smem_pitch(mpad)
    ldb = smem_pitch(dim)

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)

    # mats == 1 writes Ac alone and reads neither the tap tile nor strans, so the
    # transition tile stands in for the tap tile that is never allocated.
    build_table(
        strans, strans, squat, stable, tid, threads, chunk, 1, pitch=TABLE_PITCH
    )
    cute.arch.sync_threads()

    # Both passes issue their global loads before either consumes one, so the two
    # reads overlap rather than serializing on one latency each. dy goes in
    # unweighted and is scaled in place, one segment an access on each side, where
    # the fused pass carried one element an access with the weight inside it.
    stage_raw(gdy, sdy, bidx, hidx, t0, valid, tid, threads, chunk, rows)
    weight_rows(sdy, slp, valid, tid, threads, chunk, rows)
    stage_rotated(
        gc,
        gc,
        scrot,
        stable,
        slp,
        bidx,
        gidx,
        t0,
        0,
        valid,
        tid,
        TABLE_AC_SOLE,
        0,
        threads,
        chunk,
        lanes,
        False,
        False,
        False,
        TABLE_PITCH,
    )
    cute.arch.sync_threads()

    acc.fill(0.0)
    va = cute.make_tensor(
        sdy.iterator, cute.make_layout((mpad, chunk), stride=(1, lda))
    )
    vb = cute.make_tensor(
        scrot.iterator, cute.make_layout((dim, chunk), stride=(1, ldb))
    )
    mma_gemm(tiled_mma, tid, acc, va, vb, False, False)
    if cutlass.const_expr(fenced):
        cute.arch.sync_threads()
    mma_store(tiled_mma, tid, acc, dst, (mpad, dim), rows)


@cute.kernel
def chunk_start_bwd_kernel(
    gdy: cute.Tensor,
    gtrans: cute.Tensor,
    gc: cute.Tensor,
    gdz: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    serial: cutlass.Constexpr,
) -> None:
    """Contract the weighted output cotangent against the rotated readout.

    One block per ``(chunk, batch, head)``, or one block per ``(batch, head)``
    walking the chunks in reverse under ``serial``.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gc: ``(B,G,T,3N)`` operand-dtype output vectors.
        gdz: ``(B,H,C,P,3N)`` float32, written with the chunk-start cotangent.
        seqlen: ``T``. Dynamic.
        chunks: ``C``. Dynamic. Read only under ``serial``.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        per_group: ``H // G``, heads sharing one ``c``. Compile-time.
        serial: Whether one block walks every chunk instead of one block taking
            one chunk. Compile-time. The chunk-serial order is the one a fusion
            with the reverse state recurrence would force, so it is the arm that
            prices that fusion's parallelism.

    Invariants:
        ``chunk`` is a multiple of :data:`MMA_TILE_K` and ``dim`` of
        :data:`MMA_TILE_N`. ``rows`` is free: M is rounded up in shared memory,
        zero-filled, and the store is predicated. ``per_group`` divides ``H``.
        The prefixes and the table are float32 (I4) and the quaternion prefix is
        renormalized once, inside :func:`chunk_prefixes` (I5).
    """
    tid, _, _ = cute.arch.thread_idx()
    # Head is the fastest grid mode. Blocks are dispatched in that order, so the
    # ``H // G`` blocks that read one group's readout tile are co-resident and the
    # tile is fetched from DRAM once instead of once per head.
    hidx, cidx, bidx = cute.arch.block_idx()

    # Only gc is grouped; everything else this block reads is per head. The branch
    # is trace-time, so the ungrouped shape emits no divide at all.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    mpad = mma_rows(rows)
    ldb = smem_pitch(dim)
    lda = smem_pitch(mpad)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(
        cutlass.Float32, table_tile(chunk, 1, TABLE_PITCH).layout(), SMEM_SEGMENT
    )
    scrot = smem.allocate_tensor(
        gc.element_type, rotated_tile(chunk, dim).layout(), SMEM_SEGMENT
    )
    sdy = smem.allocate_tensor(
        gdy.element_type, gram_tile(chunk, rows).layout(), SMEM_SEGMENT
    )

    # Columns at or past the data width are read as operands but never restaged, so
    # they are zeroed once here. ``sdy`` runs to its full pitch because its M mode
    # is the rounded extent: columns P..mpad-1 are read as zero rows.
    stage_pad(scrot, tid, threads, chunk, dim, ldb)
    stage_pad(sdy, tid, threads, chunk, rows, lda)

    acc = mma_acc(tiled_mma, tid, (mpad, dim))
    if cutlass.const_expr(serial):
        # Reverse order, the order the state recurrence consumes the chunks in, so
        # the arm prices the launch a fused kernel would have to make.
        for step in cutlass.range(chunks):
            cur = chunks - 1 - step
            start_scanned(
                gtrans,
                strans,
                slp,
                squat,
                seqlen,
                cur,
                bidx,
                hidx,
                tid,
                threads,
                chunk,
            )
            start_chunk(
                gdy,
                gc,
                gdz[bidx, hidx, cur, None, None],
                strans,
                slp,
                squat,
                stable,
                scrot,
                sdy,
                acc,
                tiled_mma,
                seqlen,
                cur,
                bidx,
                hidx,
                gidx,
                tid,
                threads,
                chunk,
                rows,
                dim,
                False,
            )
    else:
        start_scanned(
            gtrans,
            strans,
            slp,
            squat,
            seqlen,
            cidx,
            bidx,
            hidx,
            tid,
            threads,
            chunk,
        )
        start_chunk(
            gdy,
            gc,
            gdz[bidx, hidx, cidx, None, None],
            strans,
            slp,
            squat,
            stable,
            scrot,
            sdy,
            acc,
            tiled_mma,
            seqlen,
            cidx,
            bidx,
            hidx,
            gidx,
            tid,
            threads,
            chunk,
            rows,
            dim,
            False,
        )


@cute.jit
def chunk_start_bwd(
    gdy: cute.Tensor,
    gtrans: cute.Tensor,
    gc: cute.Tensor,
    gdz: cute.Tensor,
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
    serial: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_start_bwd_kernel`.

    ``P``, ``3N``, and ``H // G`` are compile-time because the accumulator's
    partition shape is and because the group index folds away at ``G == H``. Batch,
    head, chunk count, and sequence length are dynamic.

    The grid is head-fastest, against the chunk-fastest order the other chunked
    kernels use, so that the ``H // G`` blocks sharing a readout tile run together.
    The unpack in :func:`chunk_start_bwd_kernel` matches it. Under ``serial`` the
    chunk mode leaves the grid and becomes a loop inside the block.
    """
    tiles = cutlass.Int32(1) if cutlass.const_expr(serial) else chunks
    chunk_start_bwd_kernel(
        gdy,
        gtrans,
        gc,
        gdz,
        seqlen,
        chunks,
        make_mma(dtype),
        threads,
        chunk,
        rows,
        dim,
        per_group,
        serial,
    ).launch(grid=(heads, tiles, bsz), block=(threads, 1, 1), stream=stream)


def chunk_start_backward(
    dy: Tensor,
    trans: Tensor,
    C: Tensor,
    chunk_size: int,
    *,
    serial: bool = False,
) -> Tensor:
    """Accumulate every chunk's chunk-start state cotangent.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous. A
            caller with no ``dy`` skips this kernel rather than passing zeros.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        C: ``(B,G,T,3N)``, the dtype of ``dy``, pitched. One column band of the
            mixer's fused projection, so the token stride is the projection width
            rather than ``3N``; a contiguous buffer is the case where the two
            agree. ``G`` divides ``H``; head ``h`` reads group ``h // (H // G)``.
        chunk_size: ``L``. A multiple of 16.
        serial: Whether one block walks every chunk in reverse instead of one
            block per chunk. Same result, ``C`` times fewer blocks. It exists to
            price a fusion with the reverse state recurrence, which can only keep
            the accumulator on chip if the chunk mode leaves the grid, and the
            module docstring carries what it measured. Not a tuning knob.

    Returns:
        ``(B,H,C,P,3N)`` float32 cotangent of the chunk-start state (I4).

    Raises:
        ValueError: On a layout, rank, shape, or extent violation.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (C, "C"))
    pinned: Named = ((trans, "trans"),)
    check_layout(((dy, "dy"), *pinned))
    check_pitched(((C, "C"),))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(
        dy, trans, None, (C, "C"), label="dy"
    )
    check_extents(chunk_size, dim, chunk_size)

    assert_smem_fits(
        f"chunk_start_bwd[L{chunk_size}/P{rows}/3N{dim}]",
        start_smem_bytes(chunk_size, rows, dim, dy.element_size()),
    )

    chunks = -(-seqlen // chunk_size)
    dzstart = torch.empty(
        bsz, heads, chunks, rows, dim, dtype=torch.float32, device=dy.device
    )
    jit_launch(
        chunk_start_bwd,
        (
            dy,
            trans,
            C,
            dzstart,
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
            serial,
        ),
    )
    return dzstart
