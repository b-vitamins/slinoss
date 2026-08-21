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
narrowed once into the operand dtype. Folding it into either bfloat16 operand would
round the mask itself, and the mask spans the whole dynamic range of the chunk
decay. The causal half is a select against exact zero rather than a masked
exponential, so no infinity is formed (I3).

The narrowed score never reaches shared memory. The score GEMM's C fragment is the
diagonal GEMM's A fragment thread for thread, two N atoms of the atom's C tile
being one K atom of its A tile, so the score is retiled in registers by
:func:`slinoss.ops.so3ssd.cute.mma.mma_areg`. That removes a score tile from the
shared budget and, per tap, one scalar store per accumulator element, one
``ldmatrix``, and one barrier.

Score slicing. The score is computed in column slices of :func:`nblock` source
tokens. Its accumulator lives alongside the output accumulator, so an unsliced
score at ``MAX_CHUNK`` would be four times the float32 register footprint of the
output it feeds. The slice is 32 wide up to ``L`` 64 and 16 above it, where the
output spans more than one M tile and the wider slice spills.

Both contractions over ``3N`` are blocked over K in :data:`KBLOCK_MAX` passes for
the same reason: an operand is copied whole into registers before it issues, so the
unblocked form holds a live set proportional to ``3N``.

The mask stays a mask and is not turned into a skip. Splitting both accumulators
along M into :data:`MMA_TILE_M` row tiles makes a tile whose rows all precede a
slice's first source token dead, which is a quarter of the diagonal work at ``L``
128, and the split is what lets it be dropped. Measured on sm_86 that is slower:
209.6 us against 201.4, with 48.3 M instructions against 41.6 M. The work dropped
is tensor-pipe work, and the tensor pipe is 35% utilized at ``L`` 128 while the
body issues at 36%, so an instruction is worth more than a multiply-accumulate
here; the per-tile operand loads and the branch cost more than the tiles save.
Hoisting the shared operand load out of the row-tile loop recovers 1 point of the
16. Nothing in this body is short of arithmetic.

The readout basis is staged once per chunk and stays resident: it is the A operand
of the offset and the score GEMM. The forcing tile is not, because the two taps need
different table slots and different source tokens; it is restaged per tap, and it
doubles as the chunk-start state tile for the offset GEMM, which is why it is
allocated at the wider of ``P`` and the slice width. Neither the rotated forcing
nor the rotated readout reaches global memory.

DRAM-bound. Analytic traffic at ``standard`` is about 61 MB against 2.87 GFLOP, so
47 flop/byte against a ridge point of 165.

Grid mode order. ``b`` and ``c`` are per group and everything else is per head, so
the ``H // G`` heads of one ``(batch, chunk)`` read one pair of ``(L, 3N)`` slabs.
Head is the fastest-varying grid mode because CUDA dispatches x first: the heads
sharing a slab are then consecutive blocks and the second one reads it from L2. With
chunk in x they sit ``chunks * B`` blocks apart, are never co-resident, and each head
re-reads both slabs from DRAM. Measured on sm_86 at ``B`` 4, ``H`` 18, ``T`` 2048,
``P`` 64, ``3N`` 240, ``L`` 64 and ``G`` 1, where ``H // G`` is 18, one launch at
1.7934 GHz:

    grid            cycles      DRAM read   DRAM write   L2 read hit
    (chunks,B,H)    1,173,300   309.98 MB   63.31 MB     20.62%
    (H,B,chunks)      988,061   176.17 MB   50.43 MB     54.93%

The base read 141.94 MB of ``b`` and ``c`` against a 7.86 MB compulsory footprint, an
18.05x re-read equal to ``H // G``, and 133.81 MB of it is gone. The write falls too,
by 12.87 MB at an unchanged local sector count: the read pressure was evicting dirty
spill lines to DRAM before they were overwritten. The map from block to data does not
change, so the output is bit-identical. At ``H == G`` there is nothing to share and
the order is neutral: cycles move -1.85%, +0.89%, +0.19% and +0.55% at ``standard``,
``ragged``, ``wide`` and ``long``, and DRAM bytes move under 0.1% at all four.

Residency 2 is out of reach at ``3N`` 240 and the arena is why. 79,504 B against the
50,176 B a second block needs, of which the two ``(L, 3N)`` operand tiles are
63,488 B; deleting every float32 tile in the arena still lands at 68,240 B. Both
operand tiles are live from the offset GEMM to the last diagonal GEMM, and the
forcing tile already shares its rows with the state. Slicing K to shrink them makes
``3N`` the outer loop, and because the score's C fragment is the diagonal GEMM's A
fragment in registers, every ``(slice, tap)`` score accumulator is then live at once:
64 float32 against 16, on a body already at the register ceiling.

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
    Stream,
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
    check_pitched,
    check_rows,
    check_shapes,
    check_stream,
)
from slinoss.ops.so3ssd.cute.mma import (
    MMA_PAIR,
    MMA_TILE_M,
    SMEM_SEGMENT,
    make_mma,
    mma_acc,
    mma_areg,
    mma_coords,
    mma_gemm,
    mma_gemm_areg,
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
    "KBLOCK_MAX",
    "NBLOCK_LONG",
    "NBLOCK_MAX",
    "RESIDENT_MAX",
    "chunk_scan_forward",
    "chunk_scan_fwd",
    "chunk_scan_fwd_kernel",
    "gemm_kblocks",
    "nblock",
    "readout_tile",
    "scan_smem_bytes",
]

NBLOCK_MAX: int = 32
"""Widest score column slice. The score accumulator is ``mma_rows(L) * nblk /
THREADS`` float32 per thread and is live alongside the output accumulator, so the
cap bounds the pair independently of ``L``. Every legal chunk length is a power of
two at or above 16, so this divides it exactly."""

NBLOCK_LONG: int = 16
"""Score column slice once ``L`` passes :data:`MMA_TILE_M`.

Above that the output tile is more than one M tile, which doubles both
accumulators, the score's narrowed operand, and every operand fragment at once. The
live set then passes the architectural 255 and the allocator spills: measured on
sm_86 at ``P`` 48, ``3N`` 48 and ``L`` 128, a 32-wide slice takes 255 registers and
983,040 local load and 245,760 local store sectors a launch, 40 bytes a thread
stored and each reloaded four times. Halving the slice takes the same geometry to
228 registers and no local traffic at all, for 3.2% more time. The spill is not what
costs the time here -- the body is latency-bound with bandwidth to spare, so the
spilling arm is the faster one -- but a spill fails the class outright, and every
other way to bound the live set at this ``L`` measured worse: splitting the
accumulators along M costs 4%.

Only ``L`` selects it. At ``L`` at or below :data:`MMA_TILE_M` the narrower slice is
worse at every width measured, by 6.6% at ``3N`` 96 and 10.9% at the smallest
shape."""

KBLOCK_MAX: int = 16
"""K extent of one pass over a ``3N`` contraction.

:func:`slinoss.ops.so3ssd.cute.mma.mma_gemm` copies a whole operand into registers
before it issues, so an unblocked contraction holds ``mma_rows(L) * 3N`` and
``N_extent * 3N`` operand elements live at once. That is the term in this body's
live set that grows with ``3N``: at ``P`` 64, ``3N`` 240 and ``L`` 64 the pair is
240 bytes a thread each, 120 registers of the architectural 255, and the allocator
spills. Blocking K costs no arithmetic and no traffic -- the same ``ldmatrix`` and
the same ``mma`` issue, against the same accumulator, in the same K order -- and it
bounds the pair at ``KBLOCK_MAX / 3N`` of its unblocked size.

Every legal ``3N`` is a multiple of 48 and every K extent must be a multiple of
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_K`, so 16 and 48 are the only values
that divide every geometry."""

RESIDENT_MAX: int = 2
"""Ceiling on the blocks per SM the launch bound asks for.

The launch asks for the residency the shared-memory budget already allows, which
is what makes the register allocator target that residency instead of spending
whatever the schedule prefers. The ceiling exists because the register file is the
scarcer resource of the two: ``N`` blocks of :data:`THREADS` threads cap each
thread at ``65536 / (N * THREADS)``, so a short chunk whose tiles leave room for
five blocks would ask for a cap of 102, and this body's live set is two float32
accumulators, both GEMMs' fragments, and one staging group.

Two, because the score fragment is one of those groups. Measured on sm_86 at the
standard shape, asking for three caps the thread at 168 registers and the body
spills 491,520 load and 245,760 store sectors per launch, for 114.8 us; asking for
two leaves 255 registers, no spill at all, and 104.4 us. The bound's function here
is to stop the allocator targeting a residency this body does not fit in, not to
make two blocks resident: at 255 registers one is. Before the score moved into
registers the ordering was the other way round, 107.0 us at three against 118.7 at
two, so this constant follows the live set and is not a fixed property of the SM.

Whether it binds is a property of the tile widths, not of ``L``: at ``P`` 48 and
``3N`` 48 the budget allows three and this is what refuses the third, while at the
wider readout and the longer chunk the budget allows two on its own."""


def nblock(chunk: int) -> int:
    """Column extent of one score slice.

    Args:
        chunk: ``L``.

    Returns:
        ``min(L, NBLOCK_MAX)``, or ``min(L, NBLOCK_LONG)`` once ``L`` passes
        :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_M`.
    """
    return min(chunk, NBLOCK_MAX if chunk <= MMA_TILE_M else NBLOCK_LONG)


def readout_tile(chunk: int, dim: int) -> Tile:
    """Rotated readout tile, ``(mma_rows(L), pitch)``.

    ``L`` is the M mode of the output, so the rows are the rounded extent.

    Args:
        chunk: ``L``.
        dim: ``3N``.
    """
    return operand_tile(mma_rows(chunk), dim)


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
        ]
    )


@cute.jit
def gemm_kblocks(
    tiled_mma: cute.TiledMma,
    tid: cutlass.Int32,
    acc: cute.Tensor,
    va: cute.Tensor,
    vb: cute.Tensor,
    arows: cutlass.Constexpr,
    brows: cutlass.Constexpr,
    kdim: cutlass.Constexpr,
    ldv: cutlass.Constexpr,
) -> None:
    """Accumulate ``va @ vb^T`` over K in :data:`KBLOCK_MAX` passes.

    Args:
        tiled_mma: From :func:`slinoss.ops.so3ssd.cute.mma.make_mma`.
        tid: Thread index within the block.
        acc: From :func:`slinoss.ops.so3ssd.cute.mma.mma_acc`. Updated in place.
        va: Shared-memory view of shape ``(arows, kdim)``.
        vb: Shared-memory view of shape ``(brows, kdim)``.
        arows: M extent of ``va``.
        brows: N extent of ``vb``.
        kdim: K extent of both, the stride-1 mode of both.
        ldv: Row pitch both views carry, in elements.

    Invariants:
        Operands are not swizzled and K is their stride-1 mode, so a K block is the
        same view at an element offset. Blocks are visited in increasing K, which is
        the order one unblocked call accumulates in, so the sum is unchanged.
    """
    kblk = min(KBLOCK_MAX, kdim)
    assert kdim % kblk == 0
    for k in cutlass.range_constexpr(kdim // kblk):
        off = k * kblk
        mma_gemm(
            tiled_mma,
            tid,
            acc,
            cute.make_tensor(
                va.iterator + off, cute.make_layout((arows, kblk), stride=(ldv, 1))
            ),
            cute.make_tensor(
                vb.iterator + off, cute.make_layout((brows, kblk), stride=(ldv, 1))
            ),
            True,
            True,
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

    One block per ``(chunk, batch, head)``, dispatched head first.

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
    hidx, bidx, cidx = cute.arch.block_idx()

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

    # The readout basis is the A operand of the offset and the score GEMM, so it is
    # staged once and never restaged. ``slp`` is passed as the scale tile and left
    # unread: the per-row factor belongs to the offset term alone, and applying it
    # here would scale the score too.
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
    gemm_kblocks(tiled_mma, tid, acc, va_crot, vb_z, mpad, rows, dim, ldv)
    last = chunk - 1
    for i in cutlass.range_constexpr(cute.size(acc)):
        m, _ = ycrd[i]
        # The clamp only feeds rows the M mode was rounded up by, whose accumulator
        # is zero because the readout tile zeroes them.
        acc[i] = acc[i] * decay(slp[cutlass.min(m, last)])

    zero = cutlass.Float32(0.0)
    elem = gc.element_type
    scrd = mma_coords(tiled_mma, tid, (mpad, nblk))
    sacc = mma_acc(tiled_mma, tid, (mpad, nblk))
    # The narrowed score is the A operand of the diagonal GEMM. Fragment and view
    # are built once: the retile is a layout, so nothing here is per-slice work.
    sfrag = cute.make_fragment_like(sacc, elem)
    fa_score = mma_areg(sfrag)
    # The slice body is emitted once, never unrolled. Unrolling folds every
    # score-epilogue index into an immediate, but ptxas then schedules all
    # ``2 * slices`` copies against one register file: at ``L`` 128 that is 257
    # registers of demand against the architectural 255, and the two integer
    # addresses it evicts cost 73,728 local load and 49,152 local store sectors a
    # launch. Measured on sm_86, dropping the unroll at ``L`` 128 runs 212.1 us
    # against 242.0 us unrolled. It does not by itself make the body spill-free:
    # the slice width does, and :data:`NBLOCK_LONG` records that measurement.
    for s in cutlass.range(slices):
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
            gemm_kblocks(tiled_mma, tid, sacc, va_crot, vb_f, mpad, nblk, dim, ldv)
            for i in cutlass.range_constexpr(cute.size(sacc)):
                m, r = scrd[i]
                src = nbase + r
                # I6: the mask lands on the float32 accumulator, then one narrowing
                # into the operand. I3: one exponential of a log difference.
                masked = sacc[i] * decay(slp[cutlass.min(m, last)] - slp[src])
                sfrag[i] = narrow(select(m >= src, masked, zero), elem)
            mma_gemm_areg(
                tiled_mma,
                tid,
                acc,
                fa_score,
                vb_unow if tap == 0 else vb_uprv,
                False,
            )

    # mma_store's predicated path, inlined because both of its bounds are dynamic
    # here: the row count is min(L, T - t0) and the destination row is offset by t0.
    # Its invariants and its alignment restatement carry over unchanged. Without the
    # pair the store moved four sectors per payload sector, since a scalar subscript
    # never reaches cute.autovec_copy and at two bytes an element the quad of lanes
    # covering four columns touched 8 useful bytes of a 32-byte sector.
    out = gy.element_type
    dst = gy[bidx, hidx, None, None]
    flat = cute.make_tensor(
        dst.iterator.align(MMA_PAIR * (out.width // 8)),
        cute.make_layout((seqlen * rows,), stride=(1,)),
    )
    vy = cute.zipped_divide(flat, (MMA_PAIR,))
    fy = cute.make_fragment((MMA_PAIR,), out)
    # Row-band order rather than accumulator order. M is the accumulator's second
    # mode, so its flat order alternates between bands :data:`MMA_TILE_M` apart and
    # holds twice as many rows' sectors open at once; walking M outermost closes each
    # band before opening the next. A store of a lane pair covers 4 of a sector's 32
    # bytes, so a sector is finished by eight lanes across two column groups and is
    # exposed to eviction until then. Measured on sm_86 at ``L`` 128 over two runs an
    # arm, the same stores move 23.39 MB in this order against 23.65 MB flat, against
    # 18.87 MB of payload, at an identical instruction count. At ``L`` at or below
    # :data:`MMA_TILE_M` the M mode has one value and the order is the flat one.
    band = 2 * MMA_PAIR
    mits = mpad // MMA_TILE_M
    nits = cute.size(acc) // (band * mits)
    for m_it in cutlass.range_constexpr(mits):
        for n_it in cutlass.range_constexpr(nits):
            for q in cutlass.range_constexpr(band // MMA_PAIR):
                i = q * MMA_PAIR + band * (m_it + mits * n_it)
                m, n = ycrd[i]
                # Filled before the predicate: a value produced inside a dynamic
                # branch is not readable after it. The fill is free on the rows the
                # predicate drops. ``rows`` and ``n`` are both even, so the flat index
                # of the pair is exact.
                for j in cutlass.range_constexpr(MMA_PAIR):
                    fy[j] = narrow(acc[i + j], out)
                if m < valid:
                    cute.autovec_copy(fy, vy[(None, ((t0 + m) * rows + n) // MMA_PAIR)])


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
    stream: Stream,
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
    218 per thread on this body and the residency is whatever that leaves; with one
    the thread cap follows the residency the shared-memory budget allows, so the
    schedule is chosen rather than inherited. The A/B the ceiling rests on is in
    :data:`RESIDENT_MAX`. The bound is computed rather than chosen, so it asks for no
    register cut that occupancy cannot spend.
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
        grid=(heads, bsz, chunks),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
        stream=stream,
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
        B: ``(B,G,T,3N)``, the dtype of ``U``, pitched. One column band of the
            mixer's fused projection, so the token stride is the projection width
            rather than ``3N``; a contiguous buffer is the case where the two
            agree. ``G`` divides ``H``; head ``h`` reads group ``h // (H // G)``.
        C: ``(B,G,T,3N)``, the dtype of ``U``, pitched. A second band of the same
            projection, grouped like ``B``.
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
    # ``B`` and ``C`` are bands and the rest is not, so the layout rule splits while
    # the dtype group stays whole: one call is what makes a mixed-dtype pair
    # reachable.
    activations: Named = ((U, "U"), (B, "B"), (C, "C"))
    dense: Named = ((U, "U"),)
    if u_prev is not None and b_prev is not None:
        activations = (*activations, (u_prev, "u_prev"), (b_prev, "b_prev"))
        dense = (*dense, (u_prev, "u_prev"), (b_prev, "b_prev"))

    pinned: Named = ((trans, "trans"), (K, "K"), (zstart, "zstart"))
    check_layout((*dense, *pinned))
    check_pitched(((B, "B"), (C, "C")))
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
