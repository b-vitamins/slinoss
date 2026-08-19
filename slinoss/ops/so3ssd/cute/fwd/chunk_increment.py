"""Chunk increment: the intra-chunk contribution to the next chunk's state.

One GEMM form, run twice per K slice to sum the two forcing taps:

    inc_local(P,3N) = (u * wgt)^T Bn + (ushift * wgt)^T Bp

with ``Bn_r = An_r b_r``, ``Bp_r = Ap_r b_{r-1}``, and
``wgt_r = exp(2*(lp_{L-1} - lp_r))``. Both taps share the same weight and the
same output tile, so they accumulate into one float32 fragment rather than
concatenating along K.

The weight is folded into the ``b`` side of the product, not the ``u`` side.
``b`` already passes through a 3x3 matvec with float32 scalars in registers, so
the scale costs no extra rounding there; folding it into ``u`` would add a
narrow-widen round trip to a tensor that is otherwise copied verbatim. The two
are equal term by term:

    sum_r u[r,p] * (wgt[r] * bn[r,d]) == sum_r (wgt[r] * u[r,p]) * bn[r,d]

``wgt <= 1`` by I1 and the exponent is formed from a log difference, never as a
ratio of two exponentials (I3). Low precision carries float32's exponent range in
both supported operand dtypes, so a weight small enough to flush ``wgt * b`` to
zero is a contribution that is already zero, which is the graceful underflow I1
guarantees.

Output frame. The increment is emitted in the chunk-local frame, without the
``R(Q_{L-1})`` that carries it into the global frame. The state recurrence applies
that rotation to the sum of the decayed state and the increment, which is the same
thing by linearity:

    a*(R z) + R inc == R (a*z + inc)

Applying it here instead would need three neighbouring N columns per thread, and
the ``m16n8k16`` C fragment gives each thread two, so it would cost either a
cross-thread shuffle or a float32 round trip through shared memory. The chunk
transition is emitted alongside the increment, as a unit quaternion and a separate
decay, because this kernel already has both chunk-local prefixes in shared memory
and the recurrence would otherwise recompute them.

Staging. ``u`` is staged once per K slice by :func:`stage_shifted` and read by both
taps through two views one row of pitch apart. Re-reading it from global for the
second tap would add its whole extent to a forward pass that moves about 131 MB.

``b`` is transformed on the way in by :func:`stage_rotated` and restaged between
the two taps, so the rotated forcing never reaches global memory.

DRAM-bound. Analytic traffic at ``standard`` is about 42.4 MB against 906 MFLOP,
so 24 flop/byte against a ridge point of 165: memory bound by a factor of seven,
which is why the padded M mode costs nothing measurable.

A ragged tail needs no separate path. ``stage_chunk`` stages the pad as a zero tap
and the identity transition, so both tap matrices are zero past ``valid`` and every
padded row of the ``b`` tile is zero regardless of what ``u`` holds.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Tile,
    assert_smem_fits,
    cute_dtype,
    decay,
    jit_launch,
    smem_bytes,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AN,
    TABLE_AP,
    THREADS,
    scalar_tile,
    table_tile,
    tap_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.guard import (
    Named,
    check_extents,
    check_layout,
    check_operands,
    check_pinned,
    check_shapes,
    check_stream,
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
from slinoss.ops.so3ssd.cute.prefix import chunk_endpoint, chunk_prefixes
from slinoss.ops.so3ssd.cute.table import (
    build_table,
    stage_chunk,
    stage_pad,
    stage_rotated,
    stage_shifted,
)

__all__ = [
    "KBLOCK_MAX",
    "ChunkIncrement",
    "chunk_increment_forward",
    "chunk_increment_fwd",
    "chunk_increment_fwd_kernel",
    "forced_tile",
    "increment_smem_bytes",
    "input_tile",
    "kblock",
]

KBLOCK_MAX: int = 64
"""Longest K slice. The two operand tiles are the only per-slice allocations, and
capping their K extent keeps ``MAX_CHUNK`` resident two blocks per SM instead of
one. Every legal chunk length is a power of two, so this divides it exactly and
the slice count is one or two."""


def kblock(chunk: int) -> int:
    """K extent of one slice.

    Args:
        chunk: ``L``.

    Returns:
        ``min(L, KBLOCK_MAX)``.
    """
    return min(chunk, KBLOCK_MAX)


def input_tile(kblk: int, rows: int) -> Tile:
    """``u`` staging tile, ``(kblk + 1, pitch)``.

    The extra row is the token before the slice, which the previous tap reads.
    ``P`` is the M mode of this kernel's output, so the row width is the rounded
    extent rather than ``P`` itself.

    Args:
        kblk: K extent of the slice.
        rows: ``P``.
    """
    return operand_tile(kblk + 1, mma_rows(rows))


def forced_tile(kblk: int, dim: int) -> Tile:
    """Rotated forcing tile, ``(kblk, pitch)``.

    Args:
        kblk: K extent of the slice.
        dim: ``3N``.
    """
    return operand_tile(kblk, dim)


def increment_smem_bytes(chunk: int, rows: int, dim: int, itemsize: int = 2) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_increment_fwd_kernel` allocates, in the same
    order. Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    kblk = kblock(chunk)
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 2), 4),
            (input_tile(kblk, rows), itemsize),
            (forced_tile(kblk, dim), itemsize),
        ]
    )


@cute.kernel
def chunk_increment_fwd_kernel(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    ginc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    seqlen: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Accumulate one chunk's local increment and emit its transition.

    One block per ``(chunk, batch, head)``.

    Args:
        gu: ``(B,H,T,P)`` operand-dtype input weights.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 per-tap ``(kr, g, h, 0)``.
        gb: ``(B,G,T,3N)`` operand-dtype input vectors.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``, or a placeholder.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``, or a placeholder.
        ginc: ``(B,H,C,P,3N)`` float32, written with the chunk-local increment.
        gcquat: ``(B,H,C,4)`` float32, written with the unit chunk rotation.
        gcscale: ``(B,H,C)`` float32, written with ``exp(2*lp_{L-1})``.
        seqlen: ``T``. Dynamic.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        per_group: ``H // G``, heads sharing one ``b``. Compile-time.
        has_prev: Whether the streaming carry-in was supplied. Compile-time.

    Invariants:
        ``chunk`` is a multiple of :data:`MMA_TILE_K` and of ``kblock(chunk)``,
        and ``dim`` is a multiple of :data:`MMA_TILE_N`. ``rows`` is free: M is
        rounded up in shared memory, zero-filled, and the store is predicated.
        ``per_group`` divides ``H``.
    """
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    # Only gb and gbprev are grouped; everything else this block reads is per head.
    # The branch is trace-time, so the ungrouped shape emits no divide at all rather
    # than an identity one for ptxas to fold.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    kblk = kblock(chunk)
    slices = chunk // kblk
    lanes = dim // 3
    mpad = mma_rows(rows)
    lda = smem_pitch(mpad)
    ldb = smem_pitch(dim)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    swgt = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 2).layout(), 16)
    su = smem.allocate_tensor(
        gu.element_type, input_tile(kblk, rows).layout(), SMEM_SEGMENT
    )
    sb = smem.allocate_tensor(
        gb.element_type, forced_tile(kblk, dim).layout(), SMEM_SEGMENT
    )

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)

    stage_chunk(
        gtrans[bidx, hidx, None, None],
        gtap[bidx, hidx, None, None, None],
        strans,
        stap,
        t0,
        valid,
        tid,
        threads,
        chunk,
    )
    # Columns at or past the data width are read as operands but never restaged,
    # so they are zeroed once here. ``su`` runs to its full pitch because its M
    # mode is the rounded extent: columns P..mpad-1 are read as zero rows.
    stage_pad(sb, tid, threads, kblk, dim, ldb)
    stage_pad(su, tid, threads, kblk + 1, rows, lda)

    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()

    build_table(strans, stap, squat, stable, tid, threads, chunk, 2)
    last = chunk - 1
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            # I3: one exponential of a log difference, never a ratio of two.
            swgt[token] = decay(slp[last] - slp[token])
    if tid == 0:
        cquat, cscale = chunk_endpoint(squat, slp, chunk)
        for j in cutlass.range_constexpr(4):
            gcquat[bidx, hidx, cidx, j] = cquat[j]
        gcscale[bidx, hidx, cidx] = cscale

    acc = mma_acc(tiled_mma, tid, (mpad, dim))
    # Two views of one staging tile, one row of pitch apart. The current tap
    # reads token t0+lbase+k, the previous one reads t0+lbase+k-1.
    va_now = cute.make_tensor(
        su.iterator + lda, cute.make_layout((mpad, kblk), stride=(1, lda))
    )
    va_prv = cute.make_tensor(
        su.iterator, cute.make_layout((mpad, kblk), stride=(1, lda))
    )
    vb = cute.make_tensor(sb.iterator, cute.make_layout((dim, kblk), stride=(1, ldb)))

    for s in cutlass.range_constexpr(slices):
        lbase = s * kblk
        cute.arch.sync_threads()
        stage_shifted(
            gu,
            guprev,
            su,
            bidx,
            hidx,
            t0,
            lbase,
            valid,
            tid,
            threads,
            kblk,
            rows,
            has_prev,
        )
        stage_rotated(
            gb,
            gbprev,
            sb,
            stable,
            swgt,
            bidx,
            gidx,
            t0,
            lbase,
            valid,
            tid,
            TABLE_AN,
            0,
            threads,
            kblk,
            lanes,
            has_prev,
            True,
        )
        cute.arch.sync_threads()
        mma_gemm(tiled_mma, tid, acc, va_now, vb, False, False)
        cute.arch.sync_threads()
        stage_rotated(
            gb,
            gbprev,
            sb,
            stable,
            swgt,
            bidx,
            gidx,
            t0,
            lbase,
            valid,
            tid,
            TABLE_AP,
            1,
            threads,
            kblk,
            lanes,
            has_prev,
            True,
        )
        cute.arch.sync_threads()
        mma_gemm(tiled_mma, tid, acc, va_prv, vb, False, False)

    mma_store(
        tiled_mma, tid, acc, ginc[bidx, hidx, cidx, None, None], (mpad, dim), rows
    )


@cute.jit
def chunk_increment_fwd(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    ginc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
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
    has_prev: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_increment_fwd_kernel`.

    ``P``, ``3N``, and ``H // G`` are compile-time because the accumulator's
    partition shape is and because the group index folds away at ``G == H``. Batch,
    head, chunk count, and sequence length are dynamic.
    """
    chunk_increment_fwd_kernel(
        gu,
        gtrans,
        gtap,
        gb,
        guprev,
        gbprev,
        ginc,
        gcquat,
        gcscale,
        seqlen,
        make_mma(dtype),
        threads,
        chunk,
        rows,
        dim,
        per_group,
        has_prev,
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1))


class ChunkIncrement(NamedTuple):
    """Result of the chunk increment.

    Attributes:
        inc: ``(B,H,C,P,3N)`` float32 chunk-local increment. Feeds
            :func:`slinoss.ops.so3ssd.cute.fwd.state_passing.state_passing_forward`,
            which consumes it in place.
        cquat: ``(B,H,C,4)`` float32 unit chunk rotation, scalar-first.
        cscale: ``(B,H,C)`` float32 chunk decay ``exp(2*lp_{L-1})``.
    """

    inc: Tensor
    cquat: Tensor
    cscale: Tensor


def chunk_increment_forward(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    chunk_size: int,
    *,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
) -> ChunkIncrement:
    """Accumulate every chunk's local increment and its transition.

    Args:
        U: ``(B,H,T,P)``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)`` float32, contiguous. Per-tap ``(kr, g, h, 0)``.
        B: ``(B,G,T,3N)``, the dtype of ``U``, contiguous. ``G`` divides ``H``;
            head ``h`` reads group ``h // (H // G)``.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, the dtype of ``U``. Paired with
            ``b_prev``.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, the dtype of ``U``.

    Returns:
        A :class:`ChunkIncrement`.

    Raises:
        ValueError: On a layout, rank, shape, extent, or pairing violation.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((U, "U"), (B, "B"))
    if u_prev is not None and b_prev is not None:
        activations = (*activations, (u_prev, "u_prev"), (b_prev, "b_prev"))

    pinned: Named = ((trans, "trans"), (K, "K"))
    check_layout((*activations, *pinned))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(U, trans, K, (B, "B"))
    check_extents(chunk_size, dim, kblock(chunk_size))
    has_prev = check_stream(u_prev, b_prev, (bsz, heads, groups, rows, dim))

    assert_smem_fits(
        f"chunk_increment[L{chunk_size}/P{rows}/3N{dim}]",
        increment_smem_bytes(chunk_size, rows, dim, U.element_size()),
    )

    chunks = -(-seqlen // chunk_size)
    opts = {"dtype": torch.float32, "device": U.device}
    inc = torch.empty(bsz, heads, chunks, rows, dim, **opts)
    cquat = torch.empty(bsz, heads, chunks, 4, **opts)
    cscale = torch.empty(bsz, heads, chunks, **opts)

    # A placeholder keeps one launch signature. It is never read: the branch that
    # would read it is closed at compile time.
    ustream = U[:, :, 0] if u_prev is None else u_prev
    bstream = B[:, :, 0] if b_prev is None else b_prev
    jit_launch(
        chunk_increment_fwd,
        (
            U,
            trans,
            K,
            B,
            ustream,
            bstream,
            inc,
            cquat,
            cscale,
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
            has_prev,
        ),
    )
    return ChunkIncrement(inc=inc, cquat=cquat, cscale=cscale)
