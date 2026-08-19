"""Chunk scan: every token's output, from its chunk-start state and its own chunk.

Three of the four GEMM forms, over one rowwise change of basis into the
chunk-local frame:

    crot_t = Ac_t c_t
    bn_r   = An_r b_r
    bp_r   = Ap_r b_{r-1}

    y_off(t,p)  = exp(2*lp_t) * <crot_t, zstart_p>
    score(t,r)  = <crot_t, b*_r>
    dmask(t,r)  = exp(2*(lp_t - lp_r)) * [r <= t]
    y_diag(t,p) = sum_r score_now(t,r) dmask(t,r) u(r,p)
                + sum_r score_prv(t,r) dmask(t,r) u(r-1,p)

    y = y_diag + y_off

One float32 accumulator per output tile carries all of it. The offset term runs
first, alone, so the per-row factor ``exp(2*lp_t)`` applies to it and not to the
diagonal terms; the diagonal GEMMs then accumulate on top. Both taps land in the
same accumulator rather than concatenating along K, and both reuse one pair of
shared tiles.

I6. The decay mask multiplies the float32 score accumulator in registers and is
narrowed once on the way into the score tile. Folding it into either bfloat16
operand would round the mask itself, and the mask spans the whole dynamic range of
the chunk decay. The causal half is a select against exact zero rather than a
masked exponential, so no infinity is formed (I3).

Score slicing. The score is computed in column slices of :data:`NBLOCK_MAX` source
tokens. Its accumulator lives alongside the output accumulator, so an unsliced
score at ``MAX_CHUNK`` would be four times the float32 register footprint of the
output it feeds. The slice count is one at ``L`` up to 32 and ``L/32`` above it.

The readout basis is staged once per chunk and stays resident: it is the A operand
of every GEMM in the kernel. The forcing tile is not, because the two taps need
different table slots and different source tokens; it is restaged per tap, and it
doubles as the chunk-start state tile for the offset GEMM, which is why it is
allocated at the wider of ``P`` and the slice width. Neither the rotated forcing
nor the rotated readout reaches global memory.

DRAM-bound. Analytic traffic at ``standard`` is about 61 MB against 2.87 GFLOP, so
47 flop/byte against a ridge point of 165.

A ragged tail needs no separate path. ``stage_chunk`` stages the pad as a zero tap
and the identity transition, so the rows past the sequence are zero in every
operand tile, and the store is predicated on the token existing. The rows the M
mode was rounded up by are zeroed by the same predicate.
"""

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Tile,
    assert_smem_fits,
    cute_dtype,
    decay,
    dev_tensor,
    jit_launch,
    narrow,
    select,
    smem_bytes,
    smem_capacity,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
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
    check_rows,
    check_shapes,
    check_stream,
)
from slinoss.ops.so3ssd.cute.mma import (
    MMA_TILE_M,
    SMEM_SEGMENT,
    make_mma,
    mma_acc,
    mma_coords,
    mma_gemm,
    mma_rows,
    operand_tile,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes
from slinoss.ops.so3ssd.cute.table import (
    build_table,
    stage_chunk,
    stage_rotated,
    stage_shifted,
    stage_state,
)

__all__ = [
    "NBLOCK_MAX",
    "RESIDENT_MAX",
    "chunk_scan_forward",
    "chunk_scan_fwd",
    "chunk_scan_fwd_kernel",
    "nblock",
    "readout_tile",
    "scan_smem_bytes",
    "score_tile",
]

NBLOCK_MAX: int = 32
"""Widest score column slice. The score accumulator is ``mma_rows(L) * nblk /
THREADS`` float32 per thread and is live alongside the output accumulator, so the
cap bounds the pair independently of ``L``. Every legal chunk length is a power of
two at or above 16, so this divides it exactly."""

RESIDENT_MAX: int = 3
"""Ceiling on the blocks per SM the launch bound asks for.

The launch asks for the residency the shared-memory budget already allows, which
is what makes the register allocator target that residency instead of spending
whatever the schedule prefers. The ceiling exists because the register file is the
scarcer resource of the two: ``N`` blocks of :data:`THREADS` threads cap each
thread at ``65536 / (N * THREADS)``, so a short chunk whose tiles leave room for
five blocks would ask for a cap of 102, and this body's live set is two float32
accumulators, both GEMMs' fragments, and one staging group. Three is the largest
value measured; at the cap it implies the body spills a few words and still
measures faster than the two-block schedule."""


def nblock(chunk: int) -> int:
    """Column extent of one score slice.

    Args:
        chunk: ``L``.

    Returns:
        ``min(L, NBLOCK_MAX)``.
    """
    return min(chunk, NBLOCK_MAX)


def readout_tile(chunk: int, dim: int) -> Tile:
    """Rotated readout tile, ``(mma_rows(L), pitch)``.

    ``L`` is the M mode of the output, so the rows are the rounded extent.

    Args:
        chunk: ``L``.
        dim: ``3N``.
    """
    return operand_tile(mma_rows(chunk), dim)


def score_tile(chunk: int, nblk: int) -> Tile:
    """Masked score tile, ``(mma_rows(L), pitch)``.

    Args:
        chunk: ``L``.
        nblk: Column extent of one slice.
    """
    return operand_tile(mma_rows(chunk), nblk)


def scan_smem_bytes(chunk: int, rows: int, dim: int, itemsize: int = 2) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_scan_fwd_kernel` allocates, in the same order.
    Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    nblk = nblock(chunk)
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 3), 4),
            (readout_tile(chunk, dim), itemsize),
            (operand_tile(max(rows, nblk), dim), itemsize),
            (operand_tile(nblk + 1, rows), itemsize),
            (score_tile(chunk, nblk), itemsize),
        ]
    )


@cute.kernel
def chunk_scan_fwd_kernel(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    gc: cute.Tensor,
    gz: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    gy: cute.Tensor,
    seqlen: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Write one chunk of the output.

    One block per ``(chunk, batch, head)``.

    Args:
        gu: ``(B,H,T,P)`` operand-dtype input weights.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 per-tap ``(kr, g, h, 0)``.
        gb: ``(B,G,T,3N)`` operand-dtype input vectors.
        gc: ``(B,G,T,3N)`` operand-dtype readout vectors.
        gz: ``(B,H,C,P,3N)`` float32 chunk-start states.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``, or a placeholder.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``, or a placeholder.
        gy: ``(B,H,T,P)`` operand-dtype output, written.
        seqlen: ``T``. Dynamic.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        per_group: ``H // G``, heads sharing one ``b`` and ``c``. Compile-time.
        has_prev: Whether the streaming carry-in was supplied. Compile-time.

    Invariants:
        ``chunk`` is a multiple of :data:`MMA_TILE_K` and of ``nblock(chunk)``, and
        ``dim`` and ``rows`` are multiples of :data:`MMA_TILE_N`. ``L`` is the one
        padded mode: M is rounded up in shared memory, the rounded rows are zeroed
        by the staging predicate, and the store drops them. ``per_group`` divides
        ``H``.
    """
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    # gb and gc are grouped; the state, the weights, the table, and the output are
    # per head. The branch is trace-time, so the ungrouped shape emits no divide.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    nblk = nblock(chunk)
    slices = chunk // nblk
    lanes = dim // 3
    mpad = mma_rows(chunk)
    ldv = smem_pitch(dim)
    ldu = smem_pitch(rows)
    lds = smem_pitch(nblk)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 3).layout(), 16)
    scrot = smem.allocate_tensor(
        gc.element_type, readout_tile(chunk, dim).layout(), SMEM_SEGMENT
    )
    sbz = smem.allocate_tensor(
        gb.element_type, operand_tile(max(rows, nblk), dim).layout(), SMEM_SEGMENT
    )
    su = smem.allocate_tensor(
        gu.element_type, operand_tile(nblk + 1, rows).layout(), SMEM_SEGMENT
    )
    sscore = smem.allocate_tensor(
        gc.element_type, score_tile(chunk, nblk).layout(), SMEM_SEGMENT
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
    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()
    build_table(strans, stap, squat, stable, tid, threads, chunk, 3)
    cute.arch.sync_threads()

    # The readout basis is the A operand of every GEMM below, so it is staged once
    # and never restaged. ``slp`` is passed as the scale tile and left unread: the
    # per-row factor belongs to the offset term alone, and applying it here would
    # scale the score too.
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
        TABLE_AC,
        0,
        threads,
        mpad,
        lanes,
        False,
        False,
    )
    stage_state(gz[bidx, hidx, cidx, None, None], sbz, tid, threads, rows, dim)
    cute.arch.sync_threads()

    va_crot = cute.make_tensor(
        scrot.iterator, cute.make_layout((mpad, dim), stride=(ldv, 1))
    )
    vb_z = cute.make_tensor(
        sbz.iterator, cute.make_layout((rows, dim), stride=(ldv, 1))
    )
    vb_f = cute.make_tensor(
        sbz.iterator, cute.make_layout((nblk, dim), stride=(ldv, 1))
    )
    va_score = cute.make_tensor(
        sscore.iterator, cute.make_layout((mpad, nblk), stride=(lds, 1))
    )
    # Two views of one staging tile, one row of pitch apart: the current tap reads
    # token t0+nbase+k, the previous one reads t0+nbase+k-1.
    vb_unow = cute.make_tensor(
        su.iterator + ldu, cute.make_layout((rows, nblk), stride=(1, ldu))
    )
    vb_uprv = cute.make_tensor(
        su.iterator, cute.make_layout((rows, nblk), stride=(1, ldu))
    )

    ycrd = mma_coords(tiled_mma, tid, (mpad, rows))
    acc = mma_acc(tiled_mma, tid, (mpad, rows))
    mma_gemm(tiled_mma, tid, acc, va_crot, vb_z, True, True)
    last = chunk - 1
    for i in cutlass.range_constexpr(cute.size(acc)):
        m, _ = ycrd[i]
        # The clamp only feeds rows the M mode was rounded up by, whose accumulator
        # is zero because the readout tile zeroes them.
        acc[i] = acc[i] * decay(slp[cutlass.min(m, last)])

    zero = cutlass.Float32(0.0)
    elem = sscore.element_type
    scrd = mma_coords(tiled_mma, tid, (mpad, nblk))
    sacc = mma_acc(tiled_mma, tid, (mpad, nblk))
    # The slice loop's form follows the chunk length, because the two costs it
    # trades scale differently. Unrolled, the slice base is a trace-time constant and
    # every score-epilogue index folds into an immediate, which is worth
    # ``cute.size(sacc)`` addresses a slice; dynamic, the body is emitted once and
    # ptxas schedules it without carrying every slice's live set at once. At one M
    # tile the schedule wins and at two the folding does.
    for s in cutlass.range(slices, unroll_full=chunk > MMA_TILE_M):
        nbase = s * nblk
        for tap in cutlass.range_constexpr(2):
            cute.arch.sync_threads()
            stage_rotated(
                gb,
                gbprev,
                sbz,
                stable,
                slp,
                bidx,
                gidx,
                t0,
                nbase,
                valid,
                tid,
                TABLE_AN if tap == 0 else TABLE_AP,
                tap,
                threads,
                nblk,
                lanes,
                has_prev,
                False,
            )
            if cutlass.const_expr(tap == 0):
                stage_shifted(
                    gu,
                    guprev,
                    su,
                    bidx,
                    hidx,
                    t0,
                    nbase,
                    valid,
                    tid,
                    threads,
                    nblk,
                    rows,
                    has_prev,
                )
            cute.arch.sync_threads()
            sacc.fill(0.0)
            mma_gemm(tiled_mma, tid, sacc, va_crot, vb_f, True, True)
            for i in cutlass.range_constexpr(cute.size(sacc)):
                m, r = scrd[i]
                src = nbase + r
                # I6: the mask lands on the float32 accumulator, then one narrowing
                # into the operand. I3: one exponential of a log difference.
                masked = sacc[i] * decay(slp[cutlass.min(m, last)] - slp[src])
                sscore[m, r] = narrow(select(m >= src, masked, zero), elem)
            cute.arch.sync_threads()
            mma_gemm(
                tiled_mma,
                tid,
                acc,
                va_score,
                vb_unow if tap == 0 else vb_uprv,
                True,
                False,
            )

    # mma_store takes the logical row count at compile time; here it is
    # min(L, T - t0), so the store is predicated by hand on the same coordinates.
    out = gy.element_type
    for i in cutlass.range_constexpr(cute.size(acc)):
        m, n = ycrd[i]
        if m < valid:
            gy[bidx, hidx, t0 + m, n] = narrow(acc[i], out)


@cute.jit
def chunk_scan_fwd(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    gc: cute.Tensor,
    gz: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    gy: cute.Tensor,
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
    """Launch :func:`chunk_scan_fwd_kernel`.

    ``L``, ``P``, ``3N``, and ``H // G`` are compile-time because the accumulator
    partition shapes are and because the group index folds away at ``G == H``.
    Batch, head, chunk count, and sequence length are dynamic.

    The launch carries a residency bound. Without one the register allocator spends
    218 per thread on this body and two blocks per SM is all that fits, which
    measures slower than the same schedule at the residency the shared-memory budget
    already allows. The bound is that residency, computed rather than chosen, so it
    asks for no register cut that occupancy cannot spend.
    """
    resident = min(RESIDENT_MAX, smem_capacity() // scan_smem_bytes(chunk, rows, dim))
    chunk_scan_fwd_kernel(
        gu,
        gtrans,
        gtap,
        gb,
        gc,
        gz,
        guprev,
        gbprev,
        gy,
        seqlen,
        make_mma(dtype),
        threads,
        chunk,
        rows,
        dim,
        per_group,
        has_prev,
    ).launch(
        grid=(chunks, bsz, heads),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
    )


def chunk_scan_forward(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    zstart: Tensor,
    chunk_size: int,
    *,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
) -> Tensor:
    """Write the output of every token.

    Args:
        U: ``(B,H,T,P)``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)`` float32, contiguous. Per-tap ``(kr, g, h, 0)``.
        B: ``(B,G,T,3N)``, the dtype of ``U``, contiguous. ``G`` divides ``H``;
            head ``h`` reads group ``h // (H // G)``.
        C: ``(B,G,T,3N)``, the dtype of ``U``, contiguous. Grouped like ``B``.
        zstart: ``(B,H,C,P,3N)`` float32, contiguous. Every chunk's start state, as
            :func:`slinoss.ops.so3ssd.cute.fwd.state_passing.state_passing_forward`
            writes it; chunk 0 holds ``z0`` or zero, so no chunk is a special case.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, the dtype of ``U``. Paired with
            ``b_prev``.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, the dtype of ``U``.

    Returns:
        ``(B,H,T,P)`` output in the dtype of ``U``, contiguous.

    Raises:
        ValueError: On a layout, rank, shape, extent, or pairing violation.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((U, "U"), (B, "B"), (C, "C"))
    if u_prev is not None and b_prev is not None:
        activations = (*activations, (u_prev, "u_prev"), (b_prev, "b_prev"))

    pinned: Named = ((trans, "trans"), (K, "K"), (zstart, "zstart"))
    check_layout((*activations, *pinned))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(
        U, trans, K, (B, "B"), (C, "C")
    )
    check_extents(chunk_size, dim, nblock(chunk_size))
    check_rows(rows)
    has_prev = check_stream(u_prev, b_prev, (bsz, heads, groups, rows, dim))

    chunks = -(-seqlen // chunk_size)
    want = (bsz, heads, chunks, rows, dim)
    if tuple(zstart.shape) != want:
        raise ValueError(f"zstart must be {want}, got {tuple(zstart.shape)}")

    assert_smem_fits(
        f"chunk_scan[L{chunk_size}/P{rows}/3N{dim}]",
        scan_smem_bytes(chunk_size, rows, dim, U.element_size()),
    )

    Y = torch.empty(bsz, heads, seqlen, rows, dtype=dtype, device=U.device)

    # A placeholder keeps one launch signature. It is never read: the branch that
    # would read it is closed at compile time.
    ustream = U[:, :, 0] if u_prev is None else u_prev
    bstream = B[:, :, 0] if b_prev is None else b_prev
    jit_launch(
        chunk_scan_fwd,
        (
            dev_tensor(U),
            dev_tensor(trans),
            dev_tensor(K),
            dev_tensor(B),
            dev_tensor(C),
            dev_tensor(zstart),
            dev_tensor(ustream),
            dev_tensor(bstream),
            dev_tensor(Y),
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
    return Y
