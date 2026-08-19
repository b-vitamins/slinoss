"""``dU``, the input carry, and the log-scale and chunk-transition cotangents.

Everything the backward owes the forcing input and the closing transition of a
chunk, in one block per ``(chunk, batch, head)``. The reference terms are ``du``,
``dushift``, ``dexpo``, ``duw``, ``dupw``, ``dexpw``, ``dchunk_rot`` and
``dchunk_scale``.

Five contractions, all of them dense real GEMMs off the one atom, all indexed by
the source token first:

    D_tap(r,d)   = sum_p  u_tap(r,p) dinc_local(p,d)
    duw_tap(r,p) = sum_d  b_tap(r,d) dinc_local(p,d)
    score(r,t)   = sum_d  b_tap(r,d) crot(t,d)
    dm(r,t)      = sum_p  u_tap(r,p) dy(t,p)
    ddiag(r,p)   = sum_t  Smasked(r,t) dy(t,p)

with ``u_tap`` and ``b_tap`` the current tap at ``t`` and the previous tap at
``t-1``, ``Smasked = score * dmask``, and ``dinc_local = Ac_{L-1} dinc`` the
increment cotangent carried back into the chunk-local frame. ``ddiag`` and ``duw``
accumulate into one pair of fragments, so the current tap's pair is ``du`` and the
previous tap's is ``dushift``.

The score is built transposed relative to the forward's. That is what makes the
masked score the left operand of the diagonal GEMM with no shared-memory round
trip: its N mode is the target token and the diagonal GEMM contracts over the
target token, so :func:`slinoss.ops.so3ssd.cute.mma.mma_areg` rereads the fragment
in place. It removes a score tile, one ``ldmatrix`` and one barrier per slice, and
it is what lets the two ``B`` passes below fuse. ``dm`` is transposed with it,
because the two are multiplied elementwise.

The cost of the transpose is the direction of the ``dexpo`` reduction. Summed over
the source token it is now a reduction over the accumulator's M mode, whose rows
are disjoint across warps, so it goes through one scratch row per warp and is
summed over warps at the store. Over the N mode it would have been two shuffles
and no scratch.

One pass over ``B`` per tap. Every contraction that reads a tap's forcing vector
runs while that tap is staged, so ``B`` is read twice rather than four times: 19.0
MB against 38.0 MB at ``standard``, on a kernel whose floor is its traffic. The
price is that the increment cotangent, the readout and the output cotangent are
live at once, 42,832 B against 37,456 B, which the 100 KB carveout holds two blocks
deep either way. Splitting the phases would fit the score tile back in at 52,048 B
and one block per SM, which is why the two changes are one change.

``dU`` is ``du(t) + dushift(t+1)``, the second term one row behind, which is why
``dushift`` goes through a float32 tile before the store. The chunk's last valid
token gets ``du`` alone: its ``dushift`` partner lives in the next chunk and
:func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds it there.
``carry_u`` is row 0 of ``dushift``, the cotangent that crosses into the previous
chunk.

The log-scale term is
``dlogp(t) = 2 * (sum_r dexpo(r,t) - <u(t),du(t)> - <ushift(t),dushift(t)>)``
plus ``2 * sum_r dexpw(r)`` at the chunk's last slot, which is what the reference's
``_scatter_last`` writes and holds whether or not that slot carries a token. The
sum of ``dexpo`` over the target token is never formed: it equals the two inner
products against the finished fragments, so it costs one lane reduction rather than
a second pass over the score.

The chunk-transition pair factors through the same frame change. With
``X = chunk_scale * zstart + inc_local``,

    dchunk_rot = R(Q_{L-1}) sum_{p,n} outer(dinc_local(p,n), X(p,n))

because ``dinc = R dinc_local`` and ``R`` comes out of the sum, so the increment
never has to be rematerialized: its half of the outer product is
``sum_r wgt(r) sum_n D_tap(r,3n+i) b_tap(r,3n+j)``, read off the ``D`` fragment in
its epilogue. ``dchunk_scale`` is ``sum <dinc_local, zstart>`` for the same reason.
Both are eleven float32 block reductions and ten stores per block.

Shared memory is one resident set and one phase arena. Resident: the log-scale
prefix, the increment weight, the per-warp log-scale scratch, the reduction
scratch, the three-slot transform table, and the shifted ``U`` tile that serves
both taps. The arena holds the forcing tile, restaged per tap, then the increment
cotangent, the readout and the output cotangent, staged once; the ``trans``, ``K``
and quaternion tiles of the prologue and the float32 shift tile of the epilogue
alias the last three, neither being live when the other is.

DRAM-bound. Analytic traffic at ``standard`` is ``dy 9.44 + U 9.58 + trans 1.57 +
K 3.15 + B 19.02 + C 9.44 + dinc 14.16 + zstart 14.16 + dU 9.44 + carry_u 0.29 +
dlogp 0.39 + dchunk_rot 0.06 + dchunk_scale 0.01 = 90.70 MB`` against ``1536 *
3.54 MFLOP = 5.44 GFLOP``, so 60.0 flop/byte against a ridge point of 164: memory
bound by a factor of nearly three.

The class is not yet met. Measured on an A6000 at ``standard``: 350.8 us per
launch, 160.3 MB of device traffic against the 90.70 MB above, 457.0 GB/s, which is
67.0% of a measured 681.3 GB/s copy and 68.0% of the copy's time law at the same
traffic, against a bar of 85%. The kernel is at the 255-register architectural cap
and spills 73.1 MB per launch each way. The model and the measurement differ by
69.6 MB, which the store side alone covers; the load side does not reach device
memory in full. The spill holds occupancy at two blocks of four warps, 16.7%, where
``long_scoreboard`` takes 37.4% of warp-active cycles at a 21.8% issue rate: memory
latency with too few warps, not bandwidth. The live
fragment set does not fit 255 registers at four warps and one 64-row M tile, so the
fix is a wider block, which is
:data:`slinoss.ops.so3ssd.cute.common.WARPS` and
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_M`, not anything in this module.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Tile,
    assert_smem_fits,
    block_reduce_add,
    cute_dtype,
    decay,
    jit_launch,
    narrow,
    select,
    shuffle_xor,
    smem_bytes,
    smem_capacity,
    widen,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AN,
    TABLE_AP,
    THREADS,
    WARPS,
    mat3_matvec,
    mat3_mul,
    mat3_transpose,
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
    SMEM_SEGMENT,
    fp32_tile,
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
    PREFETCH,
    build_table,
    stage_chunk,
    stage_rotated,
    stage_shifted,
)

__all__ = [
    "REDUCTIONS",
    "RESIDENT_MAX",
    "TBLOCK_MAX",
    "Arena",
    "ChunkInputBwd",
    "arena",
    "chunk_input_backward",
    "chunk_input_bwd",
    "chunk_input_bwd_kernel",
    "cotangent_tile",
    "forced_tile",
    "input_smem_bytes",
    "input_tile",
    "local_tile",
    "reduce_tile",
    "shift_tile",
    "tblock",
    "warp_tile",
]

TBLOCK_MAX: int = 32
"""Target-token columns of the score and of ``dm`` computed at once.

Two float32 fragments of ``(mma_rows(L), TBLOCK_MAX)`` are live at once on top of
both output fragments: at ``standard`` 16 and 16 against 24 and 24, inside the 170
registers per thread that three resident blocks of 128 threads allow. A multiple of
16, which :func:`slinoss.ops.so3ssd.cute.mma.mma_areg` requires of the N extent it
rereads as K."""

REDUCTIONS: int = 11
"""Float32 block reductions the epilogue pays: nine for the chunk-rotation
cotangent, one for the chunk-scale cotangent, one for the increment weight's
log-scale term. One scratch row each, so no ordering between them and no barrier of
the caller's own."""

RESIDENT_MAX: int = 3
"""Blocks per SM the launch asks for, before the shared-memory budget lowers it.

The budget lowers it to two at every standard size, and two is also what the
register file allows: the kernel sits at the 255-register architectural cap and
spills, so a third block is unreachable whatever this asks for. The cap is not
derived from shared memory alone because a shape whose arena is small enough for
three blocks still would not get them."""


def tblock(chunk: int) -> int:
    """Target-token slice width, ``min(L, TBLOCK_MAX)``.

    Args:
        chunk: ``L``.
    """
    return min(chunk, TBLOCK_MAX)


def input_tile(chunk: int, rows: int) -> Tile:
    """Shifted ``U`` tile, ``(mma_rows(L) + 1, pitch)``.

    Row ``j`` holds token ``t0 + j - 1``, so the previous tap reads rows
    ``0..mma_rows(L)-1`` and the current tap the same rows one further on. Both are
    M modes of a GEMM, hence the rounded row count, and the extra row is what the
    current tap's rounded extent needs.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return operand_tile(mma_rows(chunk) + 1, rows)


def forced_tile(chunk: int, dim: int) -> Tile:
    """Rotated forcing or readout tile, ``(mma_rows(L), pitch)``.

    Args:
        chunk: ``L``.
        dim: ``3N``.
    """
    return operand_tile(mma_rows(chunk), dim)


def local_tile(rows: int, dim: int) -> Tile:
    """Increment cotangent tile in the chunk-local frame, ``(P, pitch)``.

    ``P`` is an N mode of one GEMM and a K mode of the other, never an M mode, so
    the row count is ``P`` itself. Both uses need ``P`` to be a multiple of
    :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N`, which
    :func:`slinoss.ops.so3ssd.cute.guard.check_rows` enforces.

    Args:
        rows: ``P``.
        dim: ``3N``.
    """
    return operand_tile(rows, dim)


def cotangent_tile(chunk: int, rows: int) -> Tile:
    """Output cotangent tile, ``(mma_rows(L), pitch)``.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return operand_tile(mma_rows(chunk), rows)


def shift_tile(chunk: int, rows: int) -> Tile:
    """Float32 tile the shifted ``dushift`` passes through, ``(mma_rows(L), pitch)``.

    Float32 because it is not an operand. It is a gradient on its way to ``dU``, and
    the reference rounds that once, at the store; a second rounding here would
    double the error on the shifted half of every row.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return fp32_tile(mma_rows(chunk), rows)


def warp_tile(chunk: int) -> Tile:
    """Per-warp log-scale scratch, ``(WARPS, pitch)``.

    One row per warp because the ``dexpo`` reduction is over the accumulator's M
    mode, whose rows are split across warps: a single row would be four warps
    reading and writing one address.

    Args:
        chunk: ``L``.
    """
    return Tile((WARPS, chunk), (smem_pitch(chunk, 4), 1))


def reduce_tile() -> Tile:
    """Block-reduction scratch, ``(REDUCTIONS, WARPS)``.

    One word per warp per reduced value, which is what lets the epilogue reduce
    eleven float32 with no barrier of its own between them.
    """
    return Tile((REDUCTIONS, WARPS), (WARPS, 1))


class Arena(NamedTuple):
    """Float32-word offsets of the phase-shared tiles inside the one arena.

    The tiles below overlap in address and not in time. The forcing tile, the
    increment cotangent, the readout and the output cotangent are live together
    through the tap loop and are laid out end to end; the prologue's staging tiles
    and the epilogue's shift tile alias the last three.

    Attributes:
        forced: The rotated forcing tile, restaged once per tap.
        local: The increment cotangent in the chunk-local frame.
        readout: The rotated readout.
        cotangent: The output cotangent.
        shift: The float32 shift tile. Epilogue only.
        trans: ``trans`` staging. Prologue only.
        tap: ``K`` staging. Prologue only.
        quat: Quaternion prefix staging. Prologue only.
        words: Float32 words the arena spans.
    """

    forced: int
    local: int
    readout: int
    cotangent: int
    shift: int
    trans: int
    tap: int
    quat: int
    words: int


def _words(tile: Tile, itemsize: int) -> int:
    """Float32 words a tile of ``itemsize``-byte elements spans.

    Exact at every legal shape: an operand pitch is an odd multiple of eight
    elements and a float32 pitch an odd multiple of four, so both spans are a whole
    number of float32 words and every offset below is 16-byte aligned.

    Args:
        tile: The tile.
        itemsize: Bytes per element.
    """
    return itemsize * tile.words // 4


def arena(chunk: int, rows: int, dim: int, itemsize: int = 2) -> Arena:
    """Lay the phase-shared tiles out in one allocation.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    forced = _words(forced_tile(chunk, dim), itemsize)
    local = _words(local_tile(rows, dim), itemsize)
    readout = forced
    cotangent = _words(cotangent_tile(chunk, rows), itemsize)
    return Arena(
        forced=0,
        local=forced,
        readout=forced + local,
        cotangent=forced + local + readout,
        shift=forced,
        trans=forced,
        tap=forced + 4 * chunk,
        quat=forced + 12 * chunk,
        words=forced
        + max(
            local + readout + cotangent,
            _words(shift_tile(chunk, rows), 4),
            16 * chunk,
        ),
    )


def input_smem_bytes(chunk: int, rows: int, dim: int, itemsize: int = 2) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_input_bwd_kernel` allocates, in the same order.
    Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    return smem_bytes(
        [
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (warp_tile(chunk), 4),
            (reduce_tile(), 4),
            (table_tile(chunk, 3), 4),
            (input_tile(chunk, rows), itemsize),
            (Tile((arena(chunk, rows, dim, itemsize).words,), (1,)), 4),
        ]
    )


def _tile_at(base: cute.Tensor, words: int, tile: Tile, elem: object) -> cute.Tensor:
    """One arena tile, at a float32-word offset and possibly a narrower element.

    Undecorated, so the branch is taken during the trace and no recast reaches the
    IR for a float32 view.

    Args:
        base: The float32 arena tensor.
        words: Float32-word offset, from :func:`arena`.
        tile: Layout to build at that offset.
        elem: Element type of the view.
    """
    ptr = base.iterator + words
    if elem is not cutlass.Float32:
        ptr = cute.recast_ptr(ptr, dtype=elem)
    return cute.make_tensor(ptr, tile.layout())


def _sum_over_n(value: cutlass.Float32) -> cutlass.Float32:
    """Sum one accumulator element over the four lanes that share its row.

    The atom gives the four lanes of an aligned quad the same accumulator row and
    disjoint columns, so two butterfly rounds leave that row's partial column sum in
    all four. Rows are disjoint across quads and across warps, so the read-modify-
    write that follows is by one thread per row and needs no barrier.

    Args:
        value: The lane's contribution.
    """
    value = value + shuffle_xor(value, 1)
    return value + shuffle_xor(value, 2)


def _sum_over_m(value: cutlass.Float32) -> cutlass.Float32:
    """Sum one accumulator element over the eight lanes that share its column.

    The atom gives lanes ``l``, ``l^4``, ``l^8`` and ``l^16`` the same accumulator
    column and disjoint rows, so three butterfly rounds leave that column's
    within-warp row sum in all eight. Columns are shared across warps, so the caller
    reduces over warps as well, through one scratch row each.

    Args:
        value: The lane's contribution.
    """
    value = value + shuffle_xor(value, 4)
    value = value + shuffle_xor(value, 8)
    return value + shuffle_xor(value, 16)


@cute.kernel
def chunk_input_bwd_kernel(
    gdy: cute.Tensor,
    gu: cute.Tensor,
    guprev: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    gc: cute.Tensor,
    gdinc: cute.Tensor,
    gz: cute.Tensor,
    gdu: cute.Tensor,
    gcarry: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    seqlen: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    tblk: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Differentiate one chunk's forcing input and closing transition.

    One block per ``(chunk, batch, head)``.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gu: ``(B,H,T,P)`` operand-dtype forcing input.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 ``(kr, g, h, 0)`` per tap.
        gb: ``(B,G,T,3N)`` operand-dtype forcing vectors.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``. Read only when ``has_prev``.
        gc: ``(B,G,T,3N)`` operand-dtype output vectors.
        gdinc: ``(B,H,C,P,3N)`` float32 increment cotangent, global frame.
        gz: ``(B,H,C,P,3N)`` float32 chunk-start state.
        gdu: ``(B,H,T,P)`` operand-dtype, written with ``dU`` except at the chunk's
            last valid token, which gets the diagonal term alone.
        gcarry: ``(B,H,C,P)`` float32, written with row 0 of ``dushift``.
        gdlp: ``(B,H,C,L)`` float32, written with the diagonal and increment half of
            the log-scale-prefix cotangent, every slot including the padded ones.
        gdrot: ``(B,H,C,3,3)`` float32, written with the closing rotation cotangent.
        gdscale: ``(B,H,C)`` float32, written with the closing scale cotangent.
        seqlen: ``T``. Dynamic.
        tiled_mma: From :func:`slinoss.ops.so3ssd.cute.mma.make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        tblk: Target-token slice, from :func:`tblock`. Compile-time.
        per_group: ``H // G``, heads sharing one ``b`` and ``c``. Compile-time.
        has_prev: Whether the streaming carry-in pair was supplied. Compile-time.

    Invariants:
        ``chunk``, ``dim``, ``rows`` and ``tblk`` are multiples of the atom's
        extents, so no contraction mode is padded; only ``M``, a token count, is
        rounded, and its rows are zero-filled by the stagers. The prefixes, the
        table, the increment weight and every reduction are float32 (I4). Both
        decays come from one exponential of a log difference (I3), the mask lands on
        the float32 accumulator before the one narrowing (I6), and the quaternion
        prefix is renormalized once inside :func:`chunk_prefixes` (I5).
    """
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    # Only gb and gc are grouped; everything else this block reads is per head. The
    # branch is trace-time, so the ungrouped shape emits no divide at all.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    lanes = dim // 3
    mpad = mma_rows(chunk)
    last = chunk - 1
    slices = chunk // tblk
    elem = gdy.element_type
    out = gdu.element_type
    zero = cutlass.Float32(0.0)

    ldu = smem_pitch(rows)
    ldv = smem_pitch(dim)
    where = arena(chunk, rows, dim)

    smem = cutlass.utils.SmemAllocator()
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    swgt = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdlp = smem.allocate_tensor(cutlass.Float32, warp_tile(chunk).layout(), 16)
    sred = smem.allocate_tensor(cutlass.Float32, reduce_tile().layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 3).layout(), 16)
    su = smem.allocate_tensor(elem, input_tile(chunk, rows).layout(), SMEM_SEGMENT)
    pool = smem.allocate_tensor(
        cutlass.Float32, Tile((where.words,), (1,)).layout(), SMEM_SEGMENT
    )

    sbu = _tile_at(pool, where.forced, forced_tile(chunk, dim), elem)
    sdinc = _tile_at(pool, where.local, local_tile(rows, dim), elem)
    sc = _tile_at(pool, where.readout, forced_tile(chunk, dim), elem)
    sdy = _tile_at(pool, where.cotangent, cotangent_tile(chunk, rows), elem)
    sshift = _tile_at(pool, where.shift, shift_tile(chunk, rows), cutlass.Float32)
    strans = _tile_at(pool, where.trans, trans_tile(chunk), cutlass.Float32)
    stap = _tile_at(pool, where.tap, tap_tile(chunk), cutlass.Float32)
    squat = _tile_at(pool, where.quat, trans_tile(chunk), cutlass.Float32)

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)
    warp = tid // 32

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
    for step in cutlass.range_constexpr(-(-(WARPS * chunk) // threads)):
        i = tid + step * threads
        if i < WARPS * chunk:
            sdlp[i // chunk, i - (i // chunk) * chunk] = zero

    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()

    for step in cutlass.range_constexpr(-(-chunk // threads)):
        token = tid + step * threads
        if token < chunk:
            # I3: one exponential of a log difference. The padded tokens carry the
            # identity transition, so the prefix flattens past valid and the last
            # slot's weight is the last token's weight.
            swgt[token] = decay(slp[last] - slp[token])
    build_table(strans, stap, squat, stable, tid, threads, chunk, 3)
    cute.arch.sync_threads()

    # The closing transition, read once per block. Ac is R(Q)^T, so its transpose is
    # the rotation the chunk-transition cotangent is expressed in.
    # A plain range: the DSL preprocessor rewrites `range_constexpr` in a `for`
    # statement only, so inside a comprehension it reaches the runtime stub and
    # raises. Both unroll at trace time; only the statement form needs the marker.
    aclast = tuple(stable[TABLE_AC, last, i] for i in range(9))
    cscale = decay(slp[last])

    # Everything staged once. The three passes issue their global loads before any of
    # them consumes one, so the reads overlap rather than serializing.
    stage_shifted(
        gu, guprev, su, bidx, hidx, t0, 0, valid, tid, threads, mpad, rows, has_prev
    )
    stage_shifted(
        gdy, gdy, sdy, bidx, hidx, t0, 1, valid, tid, threads, mpad - 1, rows, False
    )
    stage_rotated(
        gc,
        gc,
        sc,
        stable,
        swgt,
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

    # The increment cotangent, into the chunk-local frame and into the two products
    # it feeds. One matrix for the whole chunk, so its nine entries are a broadcast
    # read and the pass is one 3-vector per thread per step: six coalesced float32
    # reads, nine FMA for the frame change, twelve for the products. Only the operand
    # copy narrows (I4).
    mrot = [zero for _ in range(9)]
    dscale = zero
    total = rows * lanes
    steps = -(-total // threads)
    exact = total % threads == 0
    for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
        count = min(PREFETCH, steps - group * PREFETCH)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * PREFETCH + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            p = i // lanes
            n = i - p * lanes
            held.append(
                (
                    p,
                    n,
                    (
                        gdinc[bidx, hidx, cidx, p, 3 * n],
                        gdinc[bidx, hidx, cidx, p, 3 * n + 1],
                        gdinc[bidx, hidx, cidx, p, 3 * n + 2],
                    ),
                    (
                        gz[bidx, hidx, cidx, p, 3 * n],
                        gz[bidx, hidx, cidx, p, 3 * n + 1],
                        gz[bidx, hidx, cidx, p, 3 * n + 2],
                    ),
                )
            )

        for step in cutlass.range_constexpr(count):
            p, n, got, state = held[step]
            local = mat3_matvec(aclast, got)
            if cutlass.const_expr(not exact):
                # A clamped step repeats the last element, so its store repeats a
                # correct value and only the reductions need the zero. Zeroing the
                # state zeroes both of them.
                live = tid + (group * PREFETCH + step) * threads < total
                state = tuple(select(live, state[j], zero) for j in range(3))
            for j in cutlass.range_constexpr(3):
                sdinc[p, 3 * n + j] = narrow(local[j], elem)
                dscale = dscale + local[j] * state[j]
                for i in cutlass.range_constexpr(3):
                    mrot[3 * i + j] = mrot[3 * i + j] + local[i] * state[j]
    for i in cutlass.range_constexpr(9):
        mrot[i] = cscale * mrot[i]

    du = mma_acc(tiled_mma, tid, (mpad, rows))
    dushift = mma_acc(tiled_mma, tid, (mpad, rows))
    wcrd = mma_coords(tiled_mma, tid, (mpad, rows))
    sacc = mma_acc(tiled_mma, tid, (mpad, tblk))
    dmacc = mma_acc(tiled_mma, tid, (mpad, tblk))
    scrd = mma_coords(tiled_mma, tid, (mpad, tblk))
    # The narrowed score is the A operand of the diagonal GEMM. Fragment and view are
    # built once: the retile is a layout, so nothing here is per-slice work.
    sfrag = cute.make_fragment_like(sacc, elem)
    fa_score = mma_areg(sfrag)
    dexpw = zero

    vlocal_k = cute.make_tensor(
        sdinc.iterator, cute.make_layout((dim, rows), stride=(1, ldv))
    )
    vlocal_n = cute.make_tensor(
        sdinc.iterator, cute.make_layout((rows, dim), stride=(ldv, 1))
    )
    vforced = cute.make_tensor(
        sbu.iterator, cute.make_layout((mpad, dim), stride=(ldv, 1))
    )

    # The two taps differ by the table slot, by which token the forcing vector comes
    # from, and by which row of the shifted tile pairs with an output row. Every
    # contraction that reads the tap runs while it is staged, which is what holds the
    # forcing tensor to one pass per tap.
    for tap in cutlass.range_constexpr(2):
        cute.arch.sync_threads()
        stage_rotated(
            gb,
            gbprev,
            sbu,
            stable,
            swgt,
            bidx,
            gidx,
            t0,
            0,
            valid,
            tid,
            TABLE_AP if tap == 0 else TABLE_AN,
            1 - tap,
            threads,
            mpad,
            lanes,
            has_prev,
            False,
        )
        cute.arch.sync_threads()

        vu = cute.make_tensor(
            su.iterator + tap * ldu, cute.make_layout((mpad, rows), stride=(ldu, 1))
        )
        target = dushift if tap == 0 else du

        # sum_p u_tap(r,p) dinc_local(p,d), the other half of the increment's outer
        # product. The lane triple of one element is not held by one thread, so the
        # three matrix rows are selected rather than indexed: the component index is
        # the accumulator's column modulo three, and dynamic.
        dloc = mma_acc(tiled_mma, tid, (mpad, dim))
        mma_gemm(tiled_mma, tid, dloc, vu, vlocal_k, True, False)
        dcrd = mma_coords(tiled_mma, tid, (mpad, dim))
        for i in cutlass.range_constexpr(cute.size(dloc)):
            m, d = dcrd[i]
            comp = d % 3
            base = d - comp
            weighted = dloc[i] * swgt[cutlass.min(m, last)]
            picked = [
                select(comp == cutlass.Int32(k), weighted, zero) for k in range(3)
            ]
            for j in cutlass.range_constexpr(3):
                forced = widen(sbu[m, base + j], elem)
                for k in cutlass.range_constexpr(3):
                    mrot[3 * k + j] = mrot[3 * k + j] + picked[k] * forced

        # sum_d b_tap(r,d) dinc_local(p,d), the increment's contribution to the
        # forcing cotangent. The weight rides the accumulator, where it is one
        # multiply per output element rather than one per operand element.
        dw = mma_acc(tiled_mma, tid, (mpad, rows))
        mma_gemm(tiled_mma, tid, dw, vforced, vlocal_n, True, True)
        for i in cutlass.range_constexpr(cute.size(dw)):
            m, p = wcrd[i]
            weight = swgt[cutlass.min(m, last)]
            dexpw = dexpw + dw[i] * widen(su[m + tap, p], elem) * weight
            target[i] = target[i] + dw[i] * weight

        for s in cutlass.range_constexpr(slices):
            tbase = s * tblk
            vb_c = cute.make_tensor(
                sc.iterator + tbase * ldv,
                cute.make_layout((tblk, dim), stride=(ldv, 1)),
            )
            vb_dy = cute.make_tensor(
                sdy.iterator + tbase * ldu,
                cute.make_layout((tblk, rows), stride=(ldu, 1)),
            )
            vdiag = cute.make_tensor(
                sdy.iterator + tbase * ldu,
                cute.make_layout((rows, tblk), stride=(1, ldu)),
            )
            sacc.fill(0.0)
            dmacc.fill(0.0)
            mma_gemm(tiled_mma, tid, sacc, vforced, vb_c, True, True)
            mma_gemm(tiled_mma, tid, dmacc, vu, vb_dy, True, True)
            for i in cutlass.range_constexpr(cute.size(sacc)):
                m, n = scrd[i]
                token = tbase + n
                # I6: the mask lands on the float32 accumulator, then one narrowing
                # into the operand. I3: one exponential of a log difference. The clamp
                # only feeds rows the M mode was rounded up by, whose operands the
                # stagers zeroed.
                masked = sacc[i] * decay(slp[token] - slp[cutlass.min(m, last)])
                masked = select(token >= m, masked, zero)
                sfrag[i] = narrow(masked, elem)
                # The exponent's cotangent, summed over the source token, which is
                # this accumulator's M mode. Its other sum, over the target token, is
                # the pair of inner products the epilogue takes against the finished
                # fragments, so the score is never revisited.
                column = _sum_over_m(masked * dmacc[i])
                if tid % 32 < 4:
                    sdlp[warp, token] = sdlp[warp, token] + column
            mma_gemm_areg(tiled_mma, tid, target, fa_score, vdiag, False)

    # Both fragments are final, so the log-scale sum over the target token, the carry
    # and the shift all read them in place.
    for i in cutlass.range_constexpr(cute.size(du)):
        m, p = wcrd[i]
        held = _sum_over_n(
            du[i] * widen(su[m + 1, p], elem) + dushift[i] * widen(su[m, p], elem)
        )
        if tid % 4 == 0 and m < chunk:
            sdlp[warp, m] = sdlp[warp, m] - held
        if m == 0:
            gcarry[bidx, hidx, cidx, p] = dushift[i]

    total_expw = block_reduce_add(dexpw, sred[0, None], tid, threads)
    total_scale = block_reduce_add(dscale, sred[1, None], tid, threads)
    mfull = tuple(
        block_reduce_add(mrot[i], sred[2 + i, None], tid, threads) for i in range(9)
    )
    drot = mat3_mul(mat3_transpose(aclast), mfull)
    if tid == 0:
        gdscale[bidx, hidx, cidx] = total_scale
        for i in cutlass.range_constexpr(3):
            for j in cutlass.range_constexpr(3):
                gdrot[bidx, hidx, cidx, i, j] = drot[3 * i + j]

    for step in cutlass.range_constexpr(-(-chunk // threads)):
        token = tid + step * threads
        if token < chunk:
            summed = zero
            for w in cutlass.range_constexpr(WARPS):
                summed = summed + sdlp[w, token]
            # The scatter lands on the chunk's last slot whether or not it carries a
            # token, because the increment weight differentiates the padded prefix
            # too.
            gdlp[bidx, hidx, cidx, token] = 2.0 * (
                summed + select(token == last, total_expw, zero)
            )

    for i in cutlass.range_constexpr(cute.size(dushift)):
        m, p = wcrd[i]
        sshift[m, p] = dushift[i]
    cute.arch.sync_threads()

    for i in cutlass.range_constexpr(cute.size(du)):
        m, p = wcrd[i]
        # The row above, clamped and then corrected, never predicated. The last valid
        # token's partner is the next chunk's first row and belongs to the boundary
        # kernel, so it reads zero here.
        above = sshift[cutlass.min(m + 1, mpad - 1), p]
        held = du[i] + select(m + 1 < valid, above, zero)
        if m < valid:
            gdu[bidx, hidx, t0 + m, p] = narrow(held, out)


@cute.jit
def chunk_input_bwd(
    gdy: cute.Tensor,
    gu: cute.Tensor,
    guprev: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    gc: cute.Tensor,
    gdinc: cute.Tensor,
    gz: cute.Tensor,
    gdu: cute.Tensor,
    gcarry: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    tblk: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    resident: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_input_bwd_kernel`.

    ``P``, ``3N``, the slice width and ``H // G`` are compile-time because the
    accumulator partitions and the arena offsets are. Batch, head, chunk count and
    sequence length are dynamic.
    """
    chunk_input_bwd_kernel(
        gdy,
        gu,
        guprev,
        gtrans,
        gtap,
        gb,
        gbprev,
        gc,
        gdinc,
        gz,
        gdu,
        gcarry,
        gdlp,
        gdrot,
        gdscale,
        seqlen,
        make_mma(dtype),
        threads,
        chunk,
        rows,
        dim,
        tblk,
        per_group,
        has_prev,
    ).launch(
        grid=(chunks, bsz, heads),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
    )


class ChunkInputBwd(NamedTuple):
    """What one launch of the chunk-input backward produces.

    Attributes:
        dU: ``(B,H,T,P)`` cotangent of the forcing input, in the activation dtype.
            The chunk-boundary rows carry the diagonal term alone;
            :func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds the
            shifted term there.
        carry_u: ``(B,H,C,P)`` float32 cotangent that each chunk's first token sends
            to the token before it. Index 0 is the streaming feedback.
        dlogp: ``(B,H,C,L)`` float32 diagonal and increment half of the
            log-scale-prefix cotangent, the reference's ``dlogp_scan``.
        dchunk_rot: ``(B,H,C,3,3)`` float32 cotangent of each chunk's closing
            rotation, row-major.
        dchunk_scale: ``(B,H,C)`` float32 cotangent of each chunk's closing scale.
    """

    dU: Tensor
    carry_u: Tensor
    dlogp: Tensor
    dchunk_rot: Tensor
    dchunk_scale: Tensor


def chunk_input_backward(
    dy: Tensor,
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    dinc: Tensor,
    zstart: Tensor,
    chunk_size: int,
    *,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
) -> ChunkInputBwd:
    """Differentiate the forcing input and the closing transition of every chunk.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous. A
            caller with no ``dy`` passes zeros: unlike the chunk-start cotangent, the
            increment terms survive and this kernel still has work.
        U: ``(B,H,T,P)`` forcing input, the dtype of ``dy``, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)`` float32, contiguous. ``(kr, g, h, 0)`` per tap.
        B: ``(B,G,T,3N)``, the dtype of ``dy``, pitched. ``G`` divides ``H``; head
            ``h`` reads group ``h // (H // G)``.
        C: ``(B,G,T,3N)``, the dtype of ``dy``, pitched.
        dinc: ``(B,H,C,P,3N)`` float32 increment cotangent in the global frame,
            contiguous, from
            :func:`slinoss.ops.so3ssd.cute.bwd.state_passing.state_passing_backward`.
        zstart: ``(B,H,C,P,3N)`` float32 chunk-start state, contiguous, from the
            rematerialized forward.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, or None.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, or None.

    Returns:
        :class:`ChunkInputBwd`.

    Raises:
        ValueError: On a layout, rank, shape or extent violation, on a shared-memory
            budget the device cannot hold, or on half a streaming pair.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (U, "U"), (B, "B"), (C, "C"))
    pinned: Named = ((trans, "trans"), (K, "K"), (dinc, "dinc"), (zstart, "zstart"))
    check_layout(((dy, "dy"), (U, "U"), *pinned))
    check_pitched(((B, "B"), (C, "C")))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(
        U, trans, K, (B, "B"), (C, "C")
    )
    if tuple(dy.shape) != tuple(U.shape):
        raise ValueError(f"dy must be {tuple(U.shape)}, got {tuple(dy.shape)}")
    check_rows(rows)
    check_extents(chunk_size, dim, tblock(chunk_size))
    has_prev = check_stream(u_prev, b_prev, (bsz, heads, groups, rows, dim))

    chunks = -(-seqlen // chunk_size)
    state = (bsz, heads, chunks, rows, dim)
    for tensor, name in ((dinc, "dinc"), (zstart, "zstart")):
        if tuple(tensor.shape) != state:
            raise ValueError(f"{name} must be {state}, got {tuple(tensor.shape)}")

    budget = assert_smem_fits(
        f"chunk_input_bwd[L{chunk_size}/P{rows}/3N{dim}]",
        input_smem_bytes(chunk_size, rows, dim, dy.element_size()),
    )

    device = dy.device
    dU = torch.empty(bsz, heads, seqlen, rows, dtype=dtype, device=device)
    carry_u = torch.empty(bsz, heads, chunks, rows, dtype=torch.float32, device=device)
    dlogp = torch.empty(
        bsz, heads, chunks, chunk_size, dtype=torch.float32, device=device
    )
    dchunk_rot = torch.empty(
        bsz, heads, chunks, 3, 3, dtype=torch.float32, device=device
    )
    dchunk_scale = torch.empty(bsz, heads, chunks, dtype=torch.float32, device=device)
    jit_launch(
        chunk_input_bwd,
        (
            dy,
            U,
            U if u_prev is None else u_prev,
            trans,
            K,
            B,
            B if b_prev is None else b_prev,
            C,
            dinc,
            zstart,
            dU,
            carry_u,
            dlogp,
            dchunk_rot,
            dchunk_scale,
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
            tblock(chunk_size),
            heads // groups,
            has_prev,
            min(RESIDENT_MAX, max(1, smem_capacity() // budget)),
        ),
    )
    return ChunkInputBwd(
        dU=dU,
        carry_u=carry_u,
        dlogp=dlogp,
        dchunk_rot=dchunk_rot,
        dchunk_scale=dchunk_scale,
    )
