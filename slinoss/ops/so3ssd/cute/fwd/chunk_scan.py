"""Chunk scan: every token's output, from its chunk-start state and its own chunk.

Three of the four GEMM forms, over one rowwise change of basis into the
chunk-local frame:

    crot_t  = Ac_t c_t
    bfuse_s = Afuse_s b_{s-1}
    bnow_t  = An_t b_t

    y_off(t,p)  = exp(2*lp_t) * <crot_t, zstart_p>
    score(t,s)  = <crot_t, bfuse_s>
    dmask(t,s)  = exp(2*(lp_t - lp_s)) * [s <= t]
    dnow_t      = <crot_t, bnow_t>
    y_diag(t,p) = sum_s score(t,s) dmask(t,s) u(s-1,p) + dnow_t * u(t,p)

    y = y_diag + y_off

One score column, not two. ``Afuse_s = Ap_s + exp(2*ls_s) An_{s-1}`` carries both
taps of every source token in one column, because the mask satisfies
``dmask(t,s-1) = dmask(t,s) exp(2*ls_s)`` and that factor is a per-column constant
(I3: the raw per-step decay, never a ratio of prefix exponentials). The identity
has no later column to fold ``s == t`` into, so the now-tap of a token's own step
stays behind as the diagonal residue ``dnow_t * u(t,p)``, one scalar per token
against a GEMM column. Score and diagonal both halve: ``2pd + 4pL + 4dL``
multiply-accumulates a chunk become ``2pd + 2pL + 2dL``.

One float32 accumulator per output tile carries all of it. The offset term runs
first, alone, so the per-row factor ``exp(2*lp_t)`` applies to it and not to the
diagonal terms; the diagonal GEMM then accumulates on top, and the residue lands
in the store epilogue, after both.

I6. The decay mask multiplies the float32 score accumulator in registers and is
narrowed once into the operand dtype. Folding it into either bfloat16 operand would
round the mask itself, and the mask spans the whole dynamic range of the chunk
decay. The causal half is a select against exact zero rather than a masked
exponential, so no infinity is formed (I3). ``Afuse``'s own factor is folded in
float32, inside the table, before the operand narrowing, so the fused column
carries one rounding where the two taps carried one each.

The narrowed score reaches shared memory at one block width only. At four warps the
score GEMM's C fragment is the diagonal GEMM's A fragment thread for thread, two N
atoms of the atom's C tile being one K atom of its A tile, so the score is retiled in
registers by :func:`slinoss.ops.so3ssd.cute.mma.mma_areg`. That removes a score tile
from the shared budget and, per slice, one scalar store per accumulator element, one
``ldmatrix``, and one barrier. At eight warps the second warp group takes half the N
mode of every tile, the fragment stops being a K-contiguous A operand, and the score
goes through a staged tile instead. :func:`scan_threads` prices that trade.

The residue is not a GEMM operand. Staging ``bnow`` as a second ``(L, 3N)`` tile
costs a rotate-and-stage pass -- three paired global reads, nine broadcast table
words and three paired shared stores per pair, ``7.5N`` memory instructions a
token -- and then a second diagonal GEMM against it, which is the whole win.
Reduced in registers instead: ``THREADS // L`` threads share a token, each folds
``An_t`` into its own run of ``b_t`` pairs against the resident ``crot_t``, and one
butterfly over those lanes finishes it. ``9 * tpt + 3N`` memory instructions a
token, 258 against 600 at ``3N`` 240. The residue is also the more accurate half of
the diagonal: it never narrows to the operand dtype, where the score does.

The residue's ``u`` factor is read in the store epilogue rather than staged. The
row and column it needs are the ones the store already computes, so one clamped
pair index serves both, and the rows it reads are the rows this block staged a pass
earlier, 6 KB at ``standard``, served by L1 rather than DRAM.

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
128, and the split is what lets it be dropped. Measured on sm_86 over the two-tap
body, so the absolutes are that body's, that is slower: 209.6 us against 201.4,
with 48.3 M instructions against 41.6 M. The work dropped is tensor-pipe work, and
the tensor pipe is 35% utilized at ``L`` 128 while the body issues at 36%, so an
instruction is worth more than a multiply-accumulate here; the per-tile operand
loads and the branch cost more than the tiles save. Hoisting the shared operand load
out of the row-tile loop recovers 1 point of the 16. Fusion halves the work the
split would drop and leaves the instructions it would add, so it only widens the
gap. Nothing in this body is short of arithmetic.

The readout basis is staged once per chunk and stays resident: it is the A operand
of the offset and the score GEMM, and the residue reduces against it. The forcing
tile is restaged per score slice, and it doubles as the chunk-start state tile for
the offset GEMM, which is why it is allocated at the wider of ``P`` and the slice
width. Neither the rotated forcing nor the rotated readout reaches global memory.

DRAM-bound. Analytic traffic at ``standard`` is about 61 MB, which fusion leaves
alone: the fused column reads ``b`` once for the score and once for the residue
where the two taps read it once each, and the residue's second read of ``u`` is L1
served. The arithmetic it runs against falls 2.87 GFLOP to 1.71, so this body sits
at 28 flop/byte against a ridge point of 165 where the two-tap body sat at 47.

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

Residency 2 is out of reach at ``3N`` 240 and the arena is why. 81,920 B against the
50,176 B a second block needs, of which the two ``(L, 3N)`` operand tiles are
63,488 B; deleting every float32 tile in the arena still lands at 68,096 B. Both
operand tiles are live from the offset GEMM to the last diagonal GEMM, and the
forcing tile already shares its rows with the state. Slicing K to shrink them makes
``3N`` the outer loop, and because the score's C fragment is the diagonal GEMM's A
fragment in registers, every slice's score accumulator is then live at once: 32
float32 against 16, on a body already at the register ceiling. Fusion moves the
arena by +112 B here and +144 B at ``standard``, the shifted forcing tile losing the
row the second tap needed and the residue tile taking a float32 word per token, so
no residency bar moves in either direction.

Block width. That arena is what pays for the second warp group. One block per SM
fills 8.1% of the warp slots, and where the shared budget has already conceded the
second block a score tile costs no residency: 81,920 B and 87,040 B are both one
block. Measured on sm_86 at ``B`` 4, ``H`` 18, ``T`` 2048, ``P`` 64, ``3N`` 240 and
``L`` 64, one launch an arm in one profile: warps per scheduler 3.90 to 7.76, issue
23.9% to 30.6%, ``not_selected`` 0 to 20.8 M warp-cycles, 255 registers with 9,216
local load and 9,216 local store instructions to 222 with none, LSU 9.32 M to
11.05 M warp-inst as ``ldmatrix`` doubles against an A operand now broadcast to two
N groups, tensor unchanged at 2,506,752. Paired in one process over 200 alternations
with the launch order swapped, 466.9 us against 402.9 us: -64.8 us, interval
[-65.3, -64.3], 1.159x. Where the narrow arena still holds two or three blocks the
tile takes one back and the trade inverts, by 36.3% at ``L`` 128 and 2.9% at
``standard``. :func:`scan_threads` is that gate.

A ragged tail needs no separate path. ``stage_chunk`` stages the pad as a zero tap
and the identity transition, so the rows past the sequence are zero in every
operand tile, and the store is predicated on the token existing. The rows the M
mode was rounded up by are zeroed by the same predicate.

One fused column is not zero past the sequence, and it is invisible here rather than
absent. At ``T mod L == n > 0`` token ``n`` stages ``ls`` as zero, so its factor is
one and ``Afuse_n`` is the last real token's ``An_{n-1}``. That column enters rows
``t >= n`` alone, which the store predicate drops, so ``y`` cannot see it; a kernel
that also wrote a state quantity would, and this one writes ``y`` alone. The staging
predicate zeroes the column's vector in any case, so both readings of the pad row
give the same output.
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
    shuffle_xor,
    smem_bytes,
    smem_capacity,
    smem_residency,
    widen,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AFUSE,
    TABLE_AN,
    THREADS,
    WARPS,
    mat3_matvec,
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
    check_stored,
    check_stream,
)
from slinoss.ops.so3ssd.cute.mma import (
    MMA_PAIR,
    MMA_TILE_M,
    SMEM_SEGMENT,
    THREADS_WIDE,
    WARPS_WIDE,
    make_mma,
    mma_acc,
    mma_areg,
    mma_coords,
    mma_gemm,
    mma_gemm_areg,
    mma_groups,
    mma_offsets,
    mma_rows,
    operand_tile,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes
from slinoss.ops.so3ssd.cute.table import (
    LANE_PAIR,
    PREFETCH,
    TABLE_PITCH,
    build_table,
    mat_at,
    paired,
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
    "scan_dnow",
    "scan_smem_bytes",
    "scan_threads",
    "score_tile",
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
wider readout and the longer chunk the budget allows two on its own.

A four-warp measurement. The wide block of :func:`scan_threads` is taken only where
the arena allows one block, so this ceiling never binds there; that form lands at 222
registers with no spill on its own."""


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


def score_tile(chunk: int) -> Tile:
    """Staged score tile, ``(mma_rows(L), pitch)`` over one column slice.

    Allocated at :data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE` alone, where the
    register retile is illegal and the diagonal GEMM reads its A operand from
    shared memory. One slice, not the whole ``L x L``: the score is consumed
    slice by slice.

    Args:
        chunk: ``L``.
    """
    return operand_tile(mma_rows(chunk), nblock(chunk))


def scan_smem_bytes(
    chunk: int, rows: int, dim: int, itemsize: int = 2, warps: int = WARPS
) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_scan_fwd_kernel` allocates, in the same order.
    Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
        warps: Warps per block. Past :data:`slinoss.ops.so3ssd.cute.common.WARPS`
            the score is staged, which is the one term the block width moves.
    """
    nblk = nblock(chunk)
    tiles = [
        (trans_tile(chunk), 4),
        (tap_tile(chunk), 4),
        (scalar_tile(chunk), 4),
        (trans_tile(chunk), 4),
        (table_tile(chunk, 3, TABLE_PITCH), 4),
        (scalar_tile(chunk), 4),
        (readout_tile(chunk, dim), itemsize),
        (operand_tile(max(rows, nblk), dim), itemsize),
        (operand_tile(nblk, rows), itemsize),
    ]
    if warps != WARPS:
        tiles.append((score_tile(chunk), itemsize))
    return smem_bytes(tiles)


def scan_threads(chunk: int, rows: int, dim: int, itemsize: int = 2) -> int:
    """Block width for a geometry.

    Eight warps where the narrow arena has already spent the whole carveout on one
    block, four warps otherwise. The choice is the arena's, not ``L``'s.

    A second warp group subdivides the N mode of every tile, which is what makes
    the score's C fragment stop being the diagonal GEMM's A fragment: the wide form
    pays a score tile, one barrier and one shared round trip a slice. Where the
    narrow form is already pinned to one resident block that buys the second warp
    per scheduler for those bytes; where it holds two blocks it would give one back,
    and the same warps arrive with none of the round trip.

    Narrow arenas on sm_86, at the shipped 101,376 B carveout, against the paired
    delta the wide form measures at that geometry over 200 alternations:

        L    P    3N    narrow   blocks   wide      delta
        64   16   48    26,112   3        31,232    +0.6%
        64   48   48    29,952   3        35,072    +2.9%
        64   64   96    45,056   2        50,176    -2.7%
        128  48   48    49,152   2        55,296    +36.3%
        64   64   240   81,920   1        87,040    -13.9%

    Only the last takes the wide form. ``L`` 128 is the near case and is refused by
    5,120 B: its score tile is 6,144 B against the 1,024 B of headroom the two-block
    bar leaves it, and losing that block costs a third of the launch.

    Banked, not taken: ``P`` 64 ``3N`` 96 keeps both blocks and still measures 4.6 us
    faster wide, resolving. That is not the lever this gate prices -- there the
    second warp group arrives at an unchanged residency -- so the win has no account
    here and taking it on the residency-preserving rule would also take ``P`` 16
    ``3N`` 48, which measures 1.3 us slower.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element.

    Returns:
        :data:`slinoss.ops.so3ssd.cute.mma.THREADS_WIDE` or
        :data:`slinoss.ops.so3ssd.cute.common.THREADS`.
    """
    narrow_bytes = scan_smem_bytes(chunk, rows, dim, itemsize)
    if smem_residency(narrow_bytes) > 1:
        return THREADS
    wide_bytes = scan_smem_bytes(chunk, rows, dim, itemsize, warps=WARPS_WIDE)
    if wide_bytes > smem_capacity():
        return THREADS
    return THREADS_WIDE


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

    Refused:
        Rolling the loop into IR. The unrolled form emits 15 bodies at ``3N`` 240
        and spills 3,133,440 local sectors a launch, so code size reads as the
        lever. Two toolchain facts and one measurement close it.

        A dynamic offset added to a shared iterator loses the alignment proof
        ``ldmatrix`` requires. Under ``cutlass.range`` the copy in
        :func:`slinoss.ops.so3ssd.cute.mma.mma_gemm` fails IR verification with
        ``'cute.copy' op src ptr alignment (16 bits) does not meet requirement (128
        bits) of atom '!cute_nvgpu.atom.ldsm<val_type = bf16, num_matrices = 4,
        n>'``. An offset added to an iterator carries no stride to reduce the
        alignment against, so the pointer type falls to one element; a tile index
        into a divided layout keeps the guarantee, because the divided mode's
        stride is static. ``.align(SMEM_SEGMENT)`` on the offset iterator
        re-asserts it and compiles, and the assertion holds at every legal
        geometry: the pitch is a multiple of 16 bytes and a K block is 32.

        ``cutlass.range`` over a trace-time-constant trip count is unrolled back by
        the backend. Every counter comes out bit-identical to the unrolled form,
        registers and local sectors included. Code size needs the explicit
        ``cutlass.range(n, unroll=k)``.

        The response to unroll depth is not monotone. Measured on sm_86 at ``P``
        64, ``3N`` 240 and ``L`` 64 against this form, interleaved in one process:
        depth 1 costs 9.5 us, depth 3 saves 22.2 us, depth 15 saves 1.8 us. Depth 1
        cuts local traffic 76.5% and pays it back in ``short_scoreboard``, 2.5% to
        6.1%; depth 3 cuts the same traffic and holds the stall at 2.5%. At ``3N``
        96 this form does not spill at all and depth 3 costs 13.7% of the cycles, so
        the win is a function of the spill and needs a geometry gate. The spill is
        4% of this body's traffic once the state buffers narrow, and the exposed
        memory latency it trades against is 40%.
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


def _row_pairs(row: cute.Tensor) -> cute.Tensor:
    """View one contiguous global row in units of :data:`LANE_PAIR` elements.

    Args:
        row: ``(3N,)`` unit-stride view of one token's vectors.

    Returns:
        The retiled view. Element ``(None, k)`` is elements ``LANE_PAIR * k``
        through ``LANE_PAIR * k + LANE_PAIR - 1``.

    Invariants:
        ``3N`` is a multiple of 48 and ``LANE_PAIR`` divides 48, so a row offset is a
        whole number of pairs and every access is aligned to ``LANE_PAIR``.

        The row index is applied before the claim, not after. A claim survives only
        the offsets that are a compile-time multiple of it, and on a ``(T,3N)`` view
        the row stride ``3N`` is dynamic, so ``row * 3N`` is not provably a whole
        number of pairs however the extents are constrained. ``autovec_copy`` then
        issues ``LANE_PAIR`` accesses where one would do. This is why
        :func:`slinoss.ops.so3ssd.cute.table.paired` takes a shared tile, whose pitch
        is static, and never a global view.
    """
    base = row.iterator.align(LANE_PAIR * (row.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, row.layout), (LANE_PAIR,))


@cute.jit
def scan_dnow(
    gb: cute.Tensor,
    scrot: cute.Tensor,
    stable: cute.Tensor,
    sdnow: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Reduce the diagonal residue ``dnow_t = <crot_t, An_t b_t>``, one per token.

    The term tap fusion leaves behind. It is one scalar per token, so it is
    contracted in registers against the resident rotated readout rather than through
    a staged operand tile and a fifth GEMM: ``threads // L`` threads share a token,
    each takes its own run of that token's pairs, and a butterfly over those lanes
    finishes the row.

    Args:
        gb: ``(B,G,T,3N)`` operand-dtype input vectors.
        scrot: Rotated readout tile, ``(mma_rows(L), pitch)``, already staged.
        stable: ``(mats, L, 9)`` float32 transform table, already built.
        sdnow: ``(L,)`` float32, written. One value per chunk-local token.
        bidx: Batch index.
        gidx: Group index, ``h // (H // G)``. The table is per head and the vector
            per group, as in :func:`slinoss.ops.so3ssd.cute.table.stage_rotated`.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Threads the reduction is partitioned over. Compile-time. The scan
            passes :data:`slinoss.ops.so3ssd.cute.common.THREADS` at every block
            width, so the float32 summation order does not move with the width.
        chunk: ``L``. Compile-time.
        lanes: ``N``. Compile-time.

    Invariants:
        ``lanes`` is a multiple of 16 and ``L`` a power of two from 16 up, so the
        threads per token divide both the block width and a row's pair count, and a
        token's lanes never cross a warp. Nothing is loaded under a predicate: the
        indices are clamped, for the reason given in ``stage_rotated``.

        The readout row is read at the unclamped token and rows past ``valid`` are
        zero there, so a pad token reduces to exactly zero whatever the clamped
        source row and table row hold.

        Every lane of a token holds the same sum after the butterfly, so the store is
        unpredicated: which lane the ISA lets win cannot change the value.
    """
    src = gb.element_type
    elem = scrot.element_type
    wide = 3 * LANE_PAIR
    pairs = lanes // LANE_PAIR
    tpt = min(pairs, max(1, threads // chunk))
    assert pairs % tpt == 0 and threads % tpt == 0
    ppt = pairs // tpt
    span = threads // tpt
    passes = -(-chunk // span)
    exact = chunk % span == 0
    depth = max(1, PREFETCH // LANE_PAIR)

    cwords = paired(scrot)
    loads = cute.make_fragment((3 * depth, LANE_PAIR), src)
    reads = cute.make_fragment((3, LANE_PAIR), elem)
    sub = tid % tpt

    for base in cutlass.range_constexpr(passes):
        raw = base * span + tid // tpt
        token = cutlass.min(raw, chunk - 1) if cutlass.const_expr(not exact) else raw
        # One clamp bounds the table read and the global read together, as in
        # ``stage_rotated``: ``valid`` is at most the chunk.
        tsafe = cutlass.min(token, valid - 1)
        # Row first, then the pair claim: see ``_row_pairs``. One row view a pass
        # against three accesses a pair, so the address arithmetic is not the cost.
        bwords = _row_pairs(gb[bidx, gidx, t0 + tsafe, None])
        # One entry per token, not per pair: both of a pair's 3-vectors take the same
        # matrix, and every pair of the token takes it too.
        mat = mat_at(stable, TABLE_AN, tsafe, TABLE_PITCH)
        total = cutlass.Float32(0.0)
        for group in cutlass.range_constexpr(-(-ppt // depth)):
            width = min(depth, ppt - group * depth)
            held = []
            for step in cutlass.range_constexpr(width):
                pair = (group * depth + step) * tpt + sub
                for k in cutlass.range_constexpr(3):
                    cute.autovec_copy(
                        bwords[(None, 3 * pair + k)],
                        loads[(3 * step + k, None)],
                    )
                held.append(pair)

            for step in cutlass.range_constexpr(width):
                pair = held[step]
                for k in cutlass.range_constexpr(3):
                    cute.autovec_copy(
                        cwords[(None, (token, 3 * pair + k))], reads[(k, None)]
                    )
                got = tuple(
                    widen(loads[3 * step + j // LANE_PAIR, j % LANE_PAIR], src)
                    for j in range(wide)
                )
                out = tuple(
                    widen(reads[j // LANE_PAIR, j % LANE_PAIR], elem)
                    for j in range(wide)
                )
                for half in cutlass.range_constexpr(LANE_PAIR):
                    o = 3 * half
                    rot = mat3_matvec(mat, (got[o], got[o + 1], got[o + 2]))
                    total = (
                        total
                        + out[o] * rot[0]
                        + out[o + 1] * rot[1]
                        + out[o + 2] * rot[2]
                    )

        reach = 1
        while reach < tpt:
            total = total + shuffle_xor(total, reach)
            reach *= 2
        if cutlass.const_expr(exact):
            sdnow[token] = total
        else:
            if raw < chunk:
                sdnow[token] = total


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
        gz: ``(B,H,C,P,3N)`` chunk-start states at the operand dtype.
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
    one_group = mma_groups(tiled_mma) == 1
    # The arena and the MMA are sized from the same width, so a launch that widened
    # one and not the other would run off the tile it never allocated.
    assert one_group == (threads == THREADS), "the arena must match the block width"

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(
        cutlass.Float32, table_tile(chunk, 3, TABLE_PITCH).layout(), SMEM_SEGMENT
    )
    sdnow = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    scrot = smem.allocate_tensor(
        gc.element_type, readout_tile(chunk, dim).layout(), SMEM_SEGMENT
    )
    sbz = smem.allocate_tensor(
        gb.element_type, operand_tile(max(rows, nblk), dim).layout(), SMEM_SEGMENT
    )
    su = smem.allocate_tensor(
        gu.element_type, operand_tile(nblk, rows).layout(), SMEM_SEGMENT
    )
    # Last, so the narrow arena is byte for byte the arena it was before the wide
    # form existed. Dedicated rather than aliased onto a staging tile: both operand
    # tiles are live across the diagonal GEMM that reads this one. Unbound at one N
    # group, where the score stays in registers and no tile exists to view.
    vscore = None
    if cutlass.const_expr(not one_group):
        sscore = smem.allocate_tensor(
            gc.element_type, score_tile(chunk).layout(), SMEM_SEGMENT
        )
        vscore = cute.make_tensor(
            sscore.iterator,
            cute.make_layout((mpad, nblk), stride=(smem_pitch(nblk), 1)),
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
    # The fused column, so the first slot is TABLE_AFUSE rather than TABLE_AP. The
    # second slot still holds ``An``, which the diagonal residue reduces against.
    build_table(strans, stap, squat, stable, tid, threads, chunk, 3, True, TABLE_PITCH)
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
        False,
        TABLE_PITCH,
    )
    stage_state(gz[bidx, hidx, cidx, None, None], sbz, tid, threads, rows, dim)
    cute.arch.sync_threads()

    # Before the accumulators exist, so the peak live set is still the slice loop's.
    # It reads the readout tile and the table, both published by the barrier above,
    # and the barrier below publishes ``sdnow`` to the store epilogue.
    #
    # The reduction partition is :data:`THREADS`, whatever the block width. It is
    # ``threads // L`` threads to a token and a butterfly over them, so a wider block
    # would regroup the partial sums and the float32 total would change value. The
    # extra warp group sits this pass out: the block width is a scheduling arm and
    # its output is bit-identical either way.
    if cutlass.const_expr(one_group):
        scan_dnow(
            gb, scrot, stable, sdnow, bidx, gidx, t0, valid, tid, THREADS, chunk, lanes
        )
    else:
        if tid < THREADS:
            scan_dnow(
                gb,
                scrot,
                stable,
                sdnow,
                bidx,
                gidx,
                t0,
                valid,
                tid,
                THREADS,
                chunk,
                lanes,
            )
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
    # The shifted forcing: row k of the tile is token t0+nbase+k-1, which is the
    # source the fused column's tap acts on. The tile is one row narrower than the
    # slice, because the token's own forcing is the residue's, not this operand's.
    vb_ushift = cute.make_tensor(
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
    # The narrowed score is the A operand of the diagonal GEMM: in registers at one
    # N group, through ``sscore`` at two. Fragment and retile are built once, so
    # nothing here is per-slice work either way.
    sfrag = cute.make_fragment_like(sacc, elem)
    fa_score = mma_areg(sfrag) if one_group else sfrag
    # The slice body is emitted once, never unrolled. Unrolling folds every
    # score-epilogue index into an immediate, but ptxas then schedules every slice's
    # copies against one register file: measured on the two-tap body at ``L`` 128
    # that was 257 registers of demand against the architectural 255, and the two
    # integer addresses it evicted cost 73,728 local load and 49,152 local store
    # sectors a launch, 242.0 us unrolled against 212.1 us here. Fusion halves the
    # copies an unrolled form would hold live; the refusal is not re-measured, so it
    # stands on the two-tap figures. It does not by itself make the body spill-free:
    # the slice width does, and :data:`NBLOCK_LONG` records that measurement.
    for s in cutlass.range(slices):
        nbase = s * nblk
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
            TABLE_AFUSE,
            1,
            threads,
            nblk,
            lanes,
            has_prev,
            False,
            False,
            TABLE_PITCH,
        )
        # ``span`` one below the slice width: the pass fills ``span + 1`` rows and
        # only the shifted view survives fusion, so it stops at the slice's last
        # source token rather than one past it.
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
            nblk - 1,
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
        if cutlass.const_expr(one_group):
            mma_gemm_areg(tiled_mma, tid, acc, fa_score, vb_ushift, False)
        else:
            # At two N groups a thread's consecutive N steps are two atoms apart, so
            # the fragment cannot be reread as a K-contiguous A operand. The score
            # goes through shared memory instead. One barrier, not two: the previous
            # slice's ``ldmatrix`` of this tile is already two barriers back, at the
            # top of this iteration.
            cute.autovec_copy(sfrag, tiled_mma.get_slice(tid).partition_C(vscore))
            cute.arch.sync_threads()
            mma_gemm(tiled_mma, tid, acc, vscore, vb_ushift, True, False)

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
    # The residue's forcing factor, over the same flat pair index as the store. Both
    # tensors are ``(B,H,T,P)`` contiguous, so one clamped index serves both: the
    # clamp only moves rows the store drops.
    usrc = gu.element_type
    ufrom = cute.make_tensor(
        gu[bidx, hidx, None, None].iterator.align(MMA_PAIR * (usrc.width // 8)),
        cute.make_layout((seqlen * rows,), stride=(1,)),
    )
    vu = cute.zipped_divide(ufrom, (MMA_PAIR,))
    fu = cute.make_fragment((MMA_PAIR,), usrc)
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
    # A thread's row is a property of the M tile and the column pair alone, so the
    # residue's per-row factor is one read per ``(m_it, q)`` rather than one per pair.
    # ``mma_offsets`` is what says so: it is the atom's own C map at trace time, where
    # the coordinate itself is dynamic.
    offs = mma_offsets(tiled_mma, (mpad, rows))
    assert all(
        offs[q * MMA_PAIR + band * (m_it + mits * n_it)][0]
        == offs[q * MMA_PAIR + band * m_it][0]
        for m_it in range(mits)
        for q in range(band // MMA_PAIR)
        for n_it in range(nits)
    ), "the atom's C map ties a thread's row to the N tile"
    dnow = tuple(
        tuple(
            sdnow[cutlass.min(ycrd[q * MMA_PAIR + band * m_it][0], last)]
            for q in range(band // MMA_PAIR)
        )
        for m_it in range(mits)
    )
    for m_it in cutlass.range_constexpr(mits):
        for n_it in cutlass.range_constexpr(nits):
            for q in cutlass.range_constexpr(band // MMA_PAIR):
                i = q * MMA_PAIR + band * (m_it + mits * n_it)
                m, n = ycrd[i]
                # One index for the load and the store. The clamp is what bounds the
                # load: an M mode rounded up past the sequence would otherwise read
                # off the end of ``u``, and every row it moves is a row the predicate
                # drops. ``rows`` and ``n`` are both even, so the pair index is exact.
                pair = (cutlass.min(t0 + m, seqlen - 1) * rows + n) // MMA_PAIR
                cute.autovec_copy(vu[(None, pair)], fu)
                # Filled before the predicate: a value produced inside a dynamic
                # branch is not readable after it. The fill is free on the rows the
                # predicate drops.
                for j in cutlass.range_constexpr(MMA_PAIR):
                    resid = dnow[m_it][q] * widen(fu[j], usrc)
                    fy[j] = narrow(acc[i + j] + resid, out)
                if m < valid:
                    cute.autovec_copy(fy, vy[(None, pair)])


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

    ``threads`` sets the tiled MMA's warp count and the arena together, so both read
    the width from one place. :func:`scan_threads` is what chooses it.

    The launch carries a residency bound. Without one the register allocator spends
    218 per thread on this body and the residency is whatever that leaves; with one
    the thread cap follows the residency the shared-memory budget allows, so the
    schedule is chosen rather than inherited. The A/B the ceiling rests on is in
    :data:`RESIDENT_MAX`. The bound is computed rather than chosen, so it asks for no
    register cut that occupancy cannot spend.
    """
    warps = threads // 32
    resident = min(
        RESIDENT_MAX, smem_residency(scan_smem_bytes(chunk, rows, dim, warps=warps))
    )
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
        make_mma(dtype, warps),
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
    threads: int | None = None,
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
        zstart: ``(B,H,C,P,3N)``, the dtype of ``U``, contiguous. Every chunk's start
            state, as
            :func:`slinoss.ops.so3ssd.cute.fwd.increment_passing.increment_passing_forward`
            writes it; chunk 0 holds ``z0`` or zero, so no chunk is a special case.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, the dtype of ``U``. Paired with
            ``b_prev``.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, the dtype of ``U``.
        threads: Block width. ``None`` takes :func:`scan_threads`, the measured
            choice for the geometry; an explicit width overrides it and is what an
            A/B against that choice passes.

    Returns:
        ``(B,H,T,P)`` output in the dtype of ``U``, contiguous.

    Raises:
        ValueError: On a layout, rank, shape, extent, or pairing violation, on a
            float32-pinned operand that is not float32, or on a stored state that is
            not at the activation dtype.
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

    pinned: Named = ((trans, "trans"), (K, "K"))
    stored: Named = ((zstart, "zstart"),)
    check_layout((*dense, *pinned, *stored))
    check_pitched(((B, "B"), (C, "C")))
    dtype = check_operands(activations)
    check_pinned(pinned)
    check_stored(stored, dtype)
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

    itemsize = U.element_size()
    width = (
        scan_threads(chunk_size, rows, dim, itemsize) if threads is None else threads
    )
    assert_smem_fits(
        f"chunk_scan[L{chunk_size}/P{rows}/3N{dim}/T{width}]",
        scan_smem_bytes(chunk_size, rows, dim, itemsize, warps=width // 32),
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
            width,
            chunk_size,
            rows,
            dim,
            heads // groups,
            has_prev,
        ),
    )
    return Y
