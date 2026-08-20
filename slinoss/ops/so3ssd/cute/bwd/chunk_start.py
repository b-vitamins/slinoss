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

The K extent is the whole chunk, one GEMM and no slice loop, so the two operand
tiles scale with ``L``. From the layouts: ``standard`` is resident four blocks per
SM, ``MAX_CHUNK`` at ``3N = 48`` two, and ``MAX_CHUNK`` at ``3N = 96`` one.

DRAM-bound. Analytic traffic at ``standard`` is ``dy 9.44 + trans 1.57 + C 9.44 +
dzstart 14.16 = 34.61 MB`` against ``1536 * 2*64*48*64 = 604 MFLOP``, so 17.5
flop/byte against a ridge point of 164: memory bound by a factor of nine, which is
what makes the padded M mode and the recomputed prefixes affordable. Both add
arithmetic and no traffic: measured at ``standard`` on one A6000, clocks unlocked,
device otherwise idle, DRAM traffic is 34.62 MB against that analytic 34.61. Neither
was timed against a variant without it.
"""

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
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
    build_table,
    stage_pad,
    stage_rotated,
    stage_trans,
    stage_weighted,
)

__all__ = [
    "chunk_start_backward",
    "chunk_start_bwd",
    "chunk_start_bwd_kernel",
    "gram_tile",
    "rotated_tile",
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
            (table_tile(chunk, 1), 4),
            (rotated_tile(chunk, dim), itemsize),
            (gram_tile(chunk, rows), itemsize),
        ]
    )


@cute.kernel
def chunk_start_bwd_kernel(
    gdy: cute.Tensor,
    gtrans: cute.Tensor,
    gc: cute.Tensor,
    gdz: cute.Tensor,
    seqlen: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
) -> None:
    """Contract the weighted output cotangent against the rotated readout.

    One block per ``(chunk, batch, head)``.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gc: ``(B,G,T,3N)`` operand-dtype output vectors.
        gdz: ``(B,H,C,P,3N)`` float32, written with the chunk-start cotangent.
        seqlen: ``T``. Dynamic.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        per_group: ``H // G``, heads sharing one ``c``. Compile-time.

    Invariants:
        ``chunk`` is a multiple of :data:`MMA_TILE_K` and ``dim`` of
        :data:`MMA_TILE_N`. ``rows`` is free: M is rounded up in shared memory,
        zero-filled, and the store is predicated. ``per_group`` divides ``H``.
        The prefixes and the table are float32 (I4) and the quaternion prefix is
        renormalized once, inside :func:`chunk_prefixes` (I5).
    """
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    # Only gc is grouped; everything else this block reads is per head. The branch
    # is trace-time, so the ungrouped shape emits no divide at all.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    lanes = dim // 3
    mpad = mma_rows(rows)
    lda = smem_pitch(mpad)
    ldb = smem_pitch(dim)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 1).layout(), 16)
    scrot = smem.allocate_tensor(
        gc.element_type, rotated_tile(chunk, dim).layout(), SMEM_SEGMENT
    )
    sdy = smem.allocate_tensor(
        gdy.element_type, gram_tile(chunk, rows).layout(), SMEM_SEGMENT
    )

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)

    stage_trans(gtrans[bidx, hidx, None, None], strans, t0, valid, tid, threads, chunk)
    # Columns at or past the data width are read as operands but never restaged, so
    # they are zeroed once here. ``sdy`` runs to its full pitch because its M mode
    # is the rounded extent: columns P..mpad-1 are read as zero rows.
    stage_pad(scrot, tid, threads, chunk, dim, ldb)
    stage_pad(sdy, tid, threads, chunk, rows, lda)

    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()

    # mats == 1 writes Ac alone and reads neither the tap tile nor strans, so the
    # transition tile stands in for the tap tile that is never allocated.
    build_table(strans, strans, squat, stable, tid, threads, chunk, 1)
    cute.arch.sync_threads()

    # Both passes issue their global loads before either consumes one, so the two
    # reads overlap rather than serializing on one latency each.
    stage_weighted(gdy, sdy, slp, bidx, hidx, t0, valid, tid, threads, chunk, rows)
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
    )
    cute.arch.sync_threads()

    acc = mma_acc(tiled_mma, tid, (mpad, dim))
    va = cute.make_tensor(
        sdy.iterator, cute.make_layout((mpad, chunk), stride=(1, lda))
    )
    vb = cute.make_tensor(
        scrot.iterator, cute.make_layout((dim, chunk), stride=(1, ldb))
    )
    mma_gemm(tiled_mma, tid, acc, va, vb, False, False)
    mma_store(tiled_mma, tid, acc, gdz[bidx, hidx, cidx, None, None], (mpad, dim), rows)


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
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_start_bwd_kernel`.

    ``P``, ``3N``, and ``H // G`` are compile-time because the accumulator's
    partition shape is and because the group index folds away at ``G == H``. Batch,
    head, chunk count, and sequence length are dynamic.
    """
    chunk_start_bwd_kernel(
        gdy,
        gtrans,
        gc,
        gdz,
        seqlen,
        make_mma(dtype),
        threads,
        chunk,
        rows,
        dim,
        per_group,
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1))


def chunk_start_backward(
    dy: Tensor,
    trans: Tensor,
    C: Tensor,
    chunk_size: int,
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
        ),
    )
    return dzstart
