"""Chunk increment and the inter-chunk recurrence, in one launch.

    Afuse_s       = Ap_s + exp(2*ls_s) An_{s-1},   Afuse_0 = Ap_0
    inc_c(P,span) = (ushift*wgt)^T Bfuse + u_{L-1} (x) bnow_{L-1}
    zstart_c      = s_c,    s_{c+1} = R(Q_c) (a_c s_c + inc_c)

The two kernels this replaces passed ``inc`` through DRAM: the increment wrote a
``(B,H,C,P,3N)`` float32 buffer and the recurrence read it back and overwrote it
with ``zstart``. Here the increment reaches shared memory and stops there. The
buffer is written once, with ``zstart``, and the round trip in between is gone.

Structure. One block per ``(band, batch, head)``, walking the chunks forward, with
the state for its own columns carried in registers across the whole sequence. The
band is :data:`SPLIT` columns of ``3N``, so the accumulator and the state fragment
are both a fifth of the model geometry's full width and both fit the register file
at once. Every band recomputes the chunk-local prefixes and the transform table
from ``trans``, exactly as the unfused pair's kernels each did.

One forcing tap, not two. Reindexing the now-tap of token ``s-1`` onto slot ``s``
gives both taps the operand ``b_{s-1}``, the weight ``u_{s-1}`` and the scale
``wgt_s``, so one table column carries both and one GEMM replaces two: the
contraction cost is ``2pd`` where it was ``4pd``.
:func:`slinoss.ops.so3ssd.cute.table.build_table` writes that column at ``fused``.
What the reindex leaves over is the now-tap of the chunk's last slot, which has no
slot to move to. It is a rank-one ``u_{L-1} (x) bnow_{L-1}``, two vectors at one
token rather than a contraction, so it is produced as one row and folded into the
accumulator elementwise in the recurrence below.

The second forcing tile was not free at the slice the tiling sweep chose. The arena
is the larger of the operands and the float32 result, and at ``L 64 P 64 span 48``
two operand tiles are 23,696 B against the result's 12,288, so the second tile cost
7,168 B and deleting it returns them. The operands stop sizing the arena only at
``368*kblk + 144 <= 12,288``, which is ``kblk <= 32``; the shipped slice is 64.

Barriers are unchanged, two a K slice and seven a chunk at one slice. The pair is
the write-after-read before the stage and the read-after-write after it, and it
guards the tile the GEMM reads rather than how many tiles there are.

Fusion measured on sm_86 at the shape below, bf16, both arms in one process with the
order swapped every iteration: -57.9 us a launch and -13.9%, the median of 40 pairs a
trial over three trials, the interval [-60.4, -55.2] clear of zero at 96.2% coverage.
The instruction stream is the part of that no other tenant can move: 70,942,844 to
59,424,284 instructions, the LSU port carrying 13,225,328 to 10,103,408, tensor
warp-instructions 2,211,840 to 1,105,920, and the excess shared wavefronts 5,162,400 to
1,658,880. Registers go 141 to 146 with no spill either side. The arena goes 32,912 to
25,952 B at residency 3 both ways: the next bar is 24,576 B, and four blocks of 128
threads cap a thread at 128 registers, so the 6,960 returned bytes buy no block.

The rotation stays factored out of the increment and out of the state. The GEMM
accumulates in the chunk-local frame, the recurrence adds the scaled state to it
there, and one ``R(Q_c)`` carries the sum into the global frame:
``a (R z) + R inc == R(a z + inc)``. The accumulator is never rotated, so no lane
needs its neighbours' columns and nothing crosses a thread.

Arena. The two GEMM operand tiles and the float32 tile the recurrence reads are
never live at once: the contraction reads the operands, the store writes the
result, and the next chunk restages the operands only after the recurrence has read
that result. One region holds all three, which is what leaves room for
:data:`RESIDENT_MAX` blocks per SM. The residue row is outside it, because the store
writes the whole region and the recurrence reads the residue after that store.

Measured on sm_86 at ``B 4 H 18 T 2048 P 64 3N 240 L 64 G 1``, bf16, clocks unlocked,
both arms in one process: 401.3 us and 187.98 MB a launch against the pair's 787.0 us
and 522.83 MB, at 145 registers, no spills, and 3.00 blocks per SM as NCU reports it.
The pair timed before the tiling sweep and again after it read 800.8 and 788.0 us, so
the factor of two resolves by two orders of magnitude. At 64.3% of peak bandwidth the
launch is off the bus, where both kernels it replaces sat at 89.5% and 92.2%.

Those figures were taken with a float32 ``zstart``. At the activation dtype the same
launch reads 46.41 MB, writes 74.34 MB, and takes 368.2 us: -67.33 MB and -31.2 us,
both arms in one process at 1.78 GHz. Off the bus the deleted bytes buy 0.32 of the
time they would take at peak bandwidth, against 0.61 in ``chunk_scan_fwd``, which
reads the same plane and is still on it.

The curve the sweep priced, medians of thirty launches, four warps then eight: at
:data:`SPLIT` columns, 517.1, 408.1, 395.3 and 583.2, 442.4, 421.9 us at slices 16,
32, 64; at the whole 240 columns, 749.1, 746.0, 675.8 and 547.8, 442.4, 430.6. A wider
slice wins at every band width. The whole-width band loses because its arena holds one
block where the band's holds three, which costs 35 to 71 us at a matched slice. Four
warps win at the band and lose at the whole width, so the shipped configuration is the
band on four warps at the widest slice :func:`fused_kblock` admits.

Splitting ``P`` instead of ``3N`` is not a tiling this kernel has.
:func:`slinoss.ops.so3ssd.cute.mma.mma_rows` rounds the GEMM's M mode to 64, so a row
tile below that buys no arithmetic and no operand bytes, and it would cut the state
each block carries without cutting what the block stages.

The one-deep software pipeline of the unfused recurrence does not carry over. It
existed because a load of ``inc`` aliased the store overwriting it, so the fetch had
to be lifted above the store or serialize behind it. Here the increment is a shared
read and ``zstart`` is store-only: no load in the chunk loop can alias a store in
it, and the loop's global reads are the operand stages, which the staging helpers
already issue before either is consumed.

``zstart`` is stored at the activation dtype and written in one place, the scalar
store into ``gzstart`` inside the recurrence of
:func:`increment_passing_fwd_kernel`. The carried state is float32 either way, so the
store takes a :func:`slinoss._cute.narrow`; I4 pins float32 to the recurrence, not to
the copy a later GEMM reads, and that GEMM narrows the copy on the way into shared
memory whatever width it arrives at.

The store's sector ratio was three sectors per payload sector at either dtype while a
thread owned one triple. Each of the three subscript stores was one warp request over
32 elements at stride 3, 384 B touched for 128 B of payload at float32 and 192 for 64
at the operand dtype, and the narrowing halved the sectors while leaving the request
count and the ratio alone. A triple at two bytes bases the lane at ``6*lane`` bytes,
4-byte aligned on even lanes only, so no vector store was legal there.

Two adjacent triples base the lane at ``12*pair``, which is 4-byte aligned always, and
the pair-fragment form
:func:`slinoss.ops.so3ssd.cute.fwd.chunk_scan.chunk_scan_fwd_kernel` stores ``y``
through then applies. One cell is 3 STG.32 for 6 STG.16, and the float32 read of the
increment tile is 3 LDS.64 for 6 LDS.32. Measured on sm_86 at the shape above, a
deterministic counter pass for the instructions and three lock-held runs a side for the
time: -1,105,920 LSU warp-instructions, -4,127,040 of all instructions once the
addresses a cell no longer needs go with them, 369.1 to 356.1 us, 595,024 to 584,193
active cycles, at 141 registers. Wavefronts per instruction double where the width does,
so the wavefront count is unchanged and the deletion is in the instruction stream.

Duration falls 3.52% against 1.82% of the active cycles, at a grid and a residency that
did not move. The rest is the tail: 360 blocks over 252 slots leaves the second wave
43% full, so ``sm__cycles_active.avg`` is dragged toward the SMs that take one block
while the wall follows the SMs that take two. Cycles are the invariant for the
instruction stream and not for the wall, even at fixed geometry.

DRAM falls 3.57 MB and none of it is the store. Read bytes fall 3.94 MB while write
bytes rise 0.65 MB, the L2 read lookup count is flat to 0.24% on a read stream this
change does not touch, its hit rate rises 0.99 points, and the store's own sector count
at L1 rises 5.9% on 48% fewer requests. L2 write lookups miss zero times at either
width, so no partial sector is ever filled from DRAM and the wider store buys no
coverage. The byte delta is a read hit rate downstream of the schedule.

Four adjacent triples are legal by the same arithmetic, 8-byte aligned, conflict-free
on the shared read, and slower: 396.2 us against 356.1 at 644,410 active cycles, with
the byte gain lost. A warp's cells then span eight rows of the plane rather than four,
the store spread
:func:`slinoss.ops.so3ssd.cute.fwd.chunk_scan.chunk_scan_fwd_kernel` walks M outermost
to avoid; the sector account above does not establish that as the cause, and the
duration is enough to refuse the width. Two is the width that pays. One sector per
payload sector still needs the
``(3, P*N)`` planar layout the unfused recurrence priced and rejected, which moves the
cost onto the consumer that wants ``3N`` contiguous.

The chunk transition is emitted by the ``sidx == 0`` band alone: every band computes
the same prefix from the same ``trans``, so one writer is enough and the others read
their copy out of registers. The segment carry-out splits the other way, each band
copying its own columns of ``b`` at token ``T-1`` and the first band copying ``u``.
"""

from typing import NamedTuple

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
    jit_launch,
    narrow,
    smem_bytes,
    smem_capacity,
    smem_residency,
    widen,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AFUSE,
    TABLE_AN,
    WARPS,
    mat3_matvec,
    rot_hom,
    scalar_tile,
    table_tile,
    tap_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.fwd.chunk_increment import (
    KBLOCK_MAX,
    forced_tile,
    input_tile,
)
from slinoss.ops.so3ssd.cute.guard import (
    Named,
    check_extents,
    check_layout,
    check_operands,
    check_pinned,
    check_pitched,
    check_shapes,
    check_stream,
)
from slinoss.ops.so3ssd.cute.mma import (
    MMA_TILE_K,
    MMA_TILE_N,
    SMEM_SEGMENT,
    fp32_tile,
    make_mma,
    mma_acc,
    mma_atoms,
    mma_gemm,
    mma_rows,
    mma_store,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_endpoint, chunk_prefixes
from slinoss.ops.so3ssd.cute.table import (
    LANE_PAIR,
    TABLE_PITCH,
    build_table,
    stage_chunk,
    stage_pad,
    stage_rotated,
    stage_shifted,
)

__all__ = [
    "RESIDENT_MAX",
    "SLICE_PITCH",
    "SPLIT",
    "IncrementPassing",
    "arena_words",
    "fused_kblock",
    "fused_smem_bytes",
    "increment_passing_forward",
    "increment_passing_fwd",
    "increment_passing_fwd_kernel",
    "residue_tile",
    "state_tile",
]

RESIDENT_MAX: int = 3
"""Blocks per SM the launch bound asks for, before the shared-memory budget cuts it.

The bound caps a thread at ``65536 / (blocks * threads)`` registers, and this kernel
holds a GEMM accumulator and the recurrence's state at once, so the residency and
the register count trade against each other rather than both being free.
"""

SLICE_PITCH: int = 9
"""Table pitch :func:`fused_kblock` prices the budget at: the unpadded one.

The pad :data:`slinoss.ops.so3ssd.cute.table.TABLE_PITCH` carries is
``2 * L * 3 * 4`` B, 3,072 at ``L=128``, enough to push the widest resident slice off
:data:`RESIDENT_MAX` and leave the fallback the narrowest slice. The slice is worth
more than the pad. Measured on sm_86 at ``L=128 P=48 span=48``, where slices 16 and
32 allocate the same 34,000 B and hold the same two blocks: slice 32 runs 336.9 us
against slice 16's 422.9 us, and the pad at a matched slice and residency is worth
-9.9 us. So the search prices the slice as if the table were unpadded, and the pad
comes out of the residency.

At ``L=64`` the pad is 1,536 B, the budget goes 25,952 to 27,488 B at ``P=64``, and
the residency stays at three, so ``L=128`` is the only shape the bench covers where
the pad costs a block.
"""

SPLIT: int = 48
"""Columns of ``3N`` one block contracts and carries.

A multiple of 3, so a band holds whole 3-vectors, and of
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N`, so it is an N extent the atom
covers. 48 is the smallest width satisfying both, and every legal ``3N`` is a
multiple of it, so one width divides every shape.
"""


def _alignment(nbytes: int) -> int:
    """Largest power of two dividing ``nbytes``.

    An access width claim has to be a power of two, and a cell of three or six
    elements is neither. Six bytes claims two, twelve claims four, twenty-four claims
    eight, which is what each cell's base is actually aligned to.

    Args:
        nbytes: Bytes in one cell. Positive.
    """
    return nbytes & -nbytes


def state_tile(rows: int, span: int) -> Tile:
    """Increment tile the recurrence reads, ``(mpad, span)`` float32.

    Contiguous and row-major, which is what :func:`mma_store` requires of any
    destination, and pitched to the band width so the flat index of a 3-vector is
    the same arithmetic in shared memory and in ``zstart``.

    The pitch carries no padding and needs none. A phase of 32 threads covers
    ``32 / (span/3)`` whole rows of 3-vectors, and the addresses ``row * span +
    3 * lane + j`` land on 32 distinct banks for every ``j`` when ``span`` is
    :data:`SPLIT`: within a row the stride is 3, which is coprime to 32, and
    consecutive rows are offset by ``span mod 32 == 16``, which is half the bank
    count times an odd multiple. A pitch rounded away from ``span`` would break
    :func:`mma_store`, whose destination layout is ``(M, N)`` at stride ``(N, 1)``.

    The rounded M extent is allocated even where the store predicates the added rows
    away, because the vectorized path writes all of them.

    Args:
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
    """
    return Tile((mma_rows(rows), span), (span, 1))


def arena_words(kblk: int, rows: int, span: int, itemsize: int = 2) -> tuple[int, int]:
    """Float32-word extent of the overlaid region, and the offset inside it.

    One forcing tile, not two: the fused column contracts the shifted operand both
    taps share.

    Args:
        kblk: K extent of one slice.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.

    Returns:
        ``(words, forced_words)``: the region's float32-word extent, and the offset
        at which the forcing tile starts. The offset is a whole word and a whole
        :data:`slinoss.ops.so3ssd.cute.mma.SMEM_SEGMENT`, because every tile's row
        pitch is a multiple of the segment.
    """
    weights = smem_bytes([(input_tile(kblk, rows), itemsize)])
    forced = smem_bytes([(forced_tile(kblk, span), itemsize)])
    state = smem_bytes([(state_tile(rows, span), 4)])
    words = max(weights + forced, state) // 4
    return words, weights // 4


def residue_tile(span: int) -> Tile:
    """Rank-one residue vector the recurrence folds in, ``(1, pitch)`` float32.

    One row of the band, holding ``An_{L-1} b_{L-1}`` over this block's columns. Not
    an operand, so I4's float32 applies and the pitch is the float32 one; at one row
    the pitch carries only padding, and it is taken from
    :func:`slinoss.ops.so3ssd.cute.mma.fp32_tile` anyway because
    :func:`slinoss.ops.so3ssd.cute.table.paired` states the pitch rule as an
    invariant of every tile it views.

    Args:
        span: Band width, :data:`SPLIT`.
    """
    return fp32_tile(1, span)


def fused_smem_bytes(
    chunk: int,
    rows: int,
    span: int,
    itemsize: int = 2,
    *,
    kblk: int = KBLOCK_MAX,
    pitch: int = TABLE_PITCH,
) -> int:
    """Shared memory the kernel allocates, in bytes.

    The six chunk-sized float32 tiles of the increment, the residue row, then the one
    region :func:`arena_words` describes. Computed from the layouts, so there is one
    description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
        kblk: K extent of one slice.
        pitch: Float32 words a table token occupies. The launch allocates
            :data:`slinoss.ops.so3ssd.cute.table.TABLE_PITCH`; :data:`SLICE_PITCH` is
            what the slice search prices.
    """
    words, _ = arena_words(kblk, rows, span, itemsize)
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 2, pitch), 4),
            (residue_tile(span), 4),
            (Tile((words,), (1,)), 4),
        ]
    )


def fused_kblock(chunk: int, rows: int, span: int, itemsize: int = 2) -> int:
    """Widest K slice that still holds :data:`RESIDENT_MAX` blocks on one SM.

    A wider slice is one barrier pair per chunk fewer, and the residency is worth more
    than the slice. Measured on sm_86 at ``L=64 P=64 span=48``, both directions of that
    trade: slice 64 at residency 3 runs 395.3 us against slice 16's 517.1 us, and the
    whole-width band, which reaches residency 1, costs 430.6 us at its own widest
    slice. So the search returns the widest slice inside the residency, never the
    widest slice.

    Width is not monotone inside a residency. Measured at ``L=128 P=48 span=48`` at
    residency 2, slice 32 runs 336.9 us against slice 64's 363.5 us, so the widest
    resident slice is the rule and not the optimum.

    The budget is priced at :data:`SLICE_PITCH`, which is not the pitch the launch
    allocates.

    Dividing the chunk is necessary and not sufficient. A slice that divides ``L``
    but not the atom's K extent splits the K loop across an MMA step, and the atom
    reads its fragment whole, so the tail step reads operand columns the slice never
    staged: at ``L=144`` an unconstrained search returns 18 and the launch produces
    55,349 NaN in ``zstart`` with no error raised. Every power of two at or above 16
    steps down to 16, so the config's power-of-two rule hid this; the guard admits
    80, 112 and 144, which do not.

    Args:
        chunk: ``L``.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.

    Returns:
        A divisor of ``chunk`` and a multiple of
        :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_K`. Nothing wider than one MMA
        step reaches the residency once the chunk-sized tiles are large enough, so
        the fallback is
        :data:`slinoss.ops.so3ssd.cute.fwd.chunk_increment.KBLOCK_MAX`.

    Raises:
        ValueError: If no legal slice fits the carveout at this shape. The floor is
            one MMA step, so there is nothing narrower to fall back to.
    """
    kblk = chunk - chunk % MMA_TILE_K
    while kblk > KBLOCK_MAX:
        if chunk % kblk == 0:
            budget = fused_smem_bytes(
                chunk, rows, span, itemsize, kblk=kblk, pitch=SLICE_PITCH
            )
            if smem_residency(budget) >= RESIDENT_MAX:
                return kblk
        kblk -= MMA_TILE_K
    kblk = min(chunk - chunk % MMA_TILE_K, KBLOCK_MAX)
    if fused_smem_bytes(chunk, rows, span, itemsize, kblk=kblk) > smem_capacity():
        raise ValueError(
            f"no legal K slice fits at chunk={chunk} rows={rows} span={span}: "
            f"one MMA step of {kblk} needs "
            f"{fused_smem_bytes(chunk, rows, span, itemsize, kblk=kblk)} B of "
            f"shared memory, capacity is {smem_capacity()} B"
        )
    return kblk


@cute.kernel
def increment_passing_fwd_kernel(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    gz0: cute.Tensor,
    gzstart: cute.Tensor,
    gstate: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gblast: cute.Tensor,
    gulast: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    kblk: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    has_z0: cutlass.Constexpr,
) -> None:
    """Accumulate each chunk's local increment and run the chunk recurrence over it.

    One block per ``(band, batch, head)``, walking the chunks forward.

    Args:
        gu: ``(B,H,T,P)`` operand-dtype input weights.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 per-tap ``(kr, g, h, 0)``.
        gb: ``(B,G,T,3N)`` operand-dtype input vectors.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``, or a placeholder.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``, or a placeholder.
        gz0: ``(B,H,3*P*N)`` float32 initial state. Read only when ``has_z0``; the
            zero-start variant is handed ``gstate`` here so the signature has one
            form.
        gzstart: ``(B,H,C,3*P*N)`` at the activation dtype, written with the state
            entering each chunk. Write only, and narrowed on the store: its one
            consumer is a GEMM that would narrow it on the way into shared memory.
        gstate: ``(B,H,3*P*N)`` float32, written with the state after the last
            chunk.
        gcquat: ``(B,H,C,4)`` float32, written with the unit chunk rotation.
        gcscale: ``(B,H,C)`` float32, written with ``exp(2*lp_{L-1})``.
        gblast: ``(B,G,3N)`` operand-dtype, written with ``b`` at token ``T-1``.
        gulast: ``(B,H,P)`` operand-dtype, written with ``u`` at token ``T-1``.
        seqlen: ``T``. Dynamic.
        chunks: ``C``. Dynamic.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        span: Band width, :data:`SPLIT`. Compile-time.
        kblk: K extent of one slice. Compile-time.
        per_group: ``H // G``, heads sharing one ``b``. Compile-time.
        has_prev: Whether the streaming carry-in was supplied. Compile-time.
        has_z0: Whether an initial state was supplied. Compile-time.

    Invariants:
        ``span`` divides ``dim``, 3 and
        :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N` divide ``span``, ``kblk``
        divides ``chunk``, and ``threads`` divides ``rows * span / 3``, so the band
        is a whole number of 3-vectors and every thread owns the same count of them.
        ``rows`` is free: M is rounded up in shared memory, zero-filled, and the
        store predicated. ``|R(Q_c)| == 1`` and ``a_c`` lies in ``(0, 1]`` by I1, so
        the recurrence cannot grow.

        ``kblk`` dividing ``chunk`` is what makes the fused stage's one-past-``valid``
        row legal: a slice never reaches past the chunk, so the table index the widened
        keep admits is at most ``chunk - 1``.
    """
    tid, _, _ = cute.arch.thread_idx()
    # Head is the fastest grid mode, as in the unfused GEMM: the ``H // G`` blocks
    # reading one group's forcing band are co-resident. The band mode is next, so
    # the bands sharing a head's ``u`` are close enough behind it to hit L2.
    hidx, sidx, bidx = cute.arch.block_idx()

    # Only gb and gbprev are grouped; everything else this block reads is per head.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    slices = chunk // kblk
    mpad = mma_rows(rows)
    lanes = span // 3
    # Adjacent 3-vectors per cell, where the block width admits the pairing. One triple
    # at two bytes bases the lane at ``6*lane`` bytes, which is 4-byte aligned on even
    # lanes only; a pair of adjacent triples bases it at ``12*pair``, always 4-byte
    # aligned, so the ``zstart`` store and the shared read of the increment each take
    # one width up. Four triples are legal and measured slower; the module docstring
    # holds the numbers and the reason. The recurrence below is written over a cell and
    # is the same code at either width; at ``wide == 1`` every access is the scalar one
    # again, which is what keeps a shape the pairing does not divide legal without a
    # second body.
    wide = 2 if (rows * lanes) % (2 * threads) == 0 else 1
    unit = 3 * wide
    cells = rows * lanes // (wide * threads)
    # The residue row is ``lanes // LANE_PAIR`` threads' work, so the warps past it
    # have no task. Branched around rather than run empty: the guard is warp-uniform,
    # so it costs no divergence, and the pass it skips is three global reads, ten
    # table reads and eighteen FMA a thread.
    reswarps = min(threads, 32 * -(-(lanes // LANE_PAIR) // 32))
    lda = smem_pitch(mpad)
    ldb = smem_pitch(span)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    swgt = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(
        cutlass.Float32, table_tile(chunk, 2, TABLE_PITCH).layout(), SMEM_SEGMENT
    )
    # Outside the arena: the store below writes the whole region, and the recurrence
    # reads this row after that store.
    sres = smem.allocate_tensor(cutlass.Float32, residue_tile(span).layout(), 16)

    # One region, three views: the two GEMM operands, and the contraction's result
    # over the same bytes.
    words, fwords = arena_words(kblk, rows, span, gu.element_type.width // 8)
    arena = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((words,), stride=(1,)), SMEM_SEGMENT
    )
    su = cute.make_tensor(
        cute.recast_ptr(arena.iterator, dtype=gu.element_type),
        input_tile(kblk, rows).layout(),
    )
    sbfuse = cute.make_tensor(
        cute.recast_ptr(arena.iterator + fwords, dtype=gb.element_type),
        forced_tile(kblk, span).layout(),
    )
    sinc = cute.make_tensor(arena.iterator, state_tile(rows, span).layout())
    # The recurrence reads a whole cell, so it addresses the tile by cell rather than
    # by row and column. The layout is compact row-major and ``unit`` divides both the
    # pitch and the extent, so the two views agree by construction.
    scell = cute.zipped_divide(
        cute.make_tensor(
            arena.iterator.align(_alignment(4 * unit)),
            cute.make_layout((mpad * span,), stride=(1,)),
        ),
        (unit,),
    )
    # The residue in the same units, so a cell's column index serves both views.
    rescell = cute.zipped_divide(
        cute.make_tensor(
            sres.iterator.align(_alignment(4 * unit)),
            cute.make_layout((span,), stride=(1,)),
        ),
        (unit,),
    )

    # The band's origin is a pointer offset: ``3N`` is the last mode at unit stride,
    # so the staging pass indexes the band's own columns and never learns that the
    # state is wider. The streaming carry-in is banded with it.
    band = cute.make_tensor(gb.iterator + sidx * span, gb.layout)
    bandprev = cute.make_tensor(gbprev.iterator + sidx * span, gbprev.layout)

    acc = mma_acc(tiled_mma, tid, (mpad, span))
    state = cute.make_fragment((unit * cells,), cutlass.Float32)
    elem = gzstart.element_type
    finc = cute.make_fragment((unit,), cutlass.Float32)
    fres = cute.make_fragment((unit,), cutlass.Float32)
    fz = cute.make_fragment((unit,), elem)
    # ``u`` at the chunk's last token, one value per cell: the residue's left factor is
    # a row index and the recurrence's cells are addressed by row.
    fu = cute.make_fragment((cells,), cutlass.Float32)

    # One cell's index in the tile and in the ``(P,3N)`` plane, counted in cells and not
    # in elements, so neither consumer divides by ``unit`` at run time. The band index
    # is dynamic and the cell index is not, so these are hoisted out of the chunk loop
    # rather than recomputed inside it.
    scols = span // unit
    zcols = dim // unit
    slot = []
    plane = []
    col = []
    rowof = []
    for k in cutlass.range_constexpr(cells):
        owned = tid + k * threads
        row = owned // (lanes // wide)
        cell = owned - row * (lanes // wide)
        slot.append(row * scols + cell)
        plane.append(row * zcols + sidx * scols + cell)
        # ``scols == lanes // wide``, so the cell index within a row is already the
        # residue row's cell index and needs no second divide.
        col.append(cell)
        rowof.append(row)

    # The segment carry-out. Every block walks the last chunk, so the writer is
    # chosen by band rather than found: the first band copies ``u`` and each band
    # copies its own columns of ``b``. The index is ``seqlen - 1`` and not the last
    # chunk slot, because a ragged tail pads the chunk and a padded token is a no-op
    # whose b and u are zero. Ahead of the staging, so the loads issue while shared
    # memory fills.
    tlast = seqlen - 1
    if sidx == 0:
        for step in cutlass.range_constexpr((rows + threads - 1) // threads):
            p = tid + step * threads
            if p < rows:
                gulast[bidx, hidx, p] = gu[bidx, hidx, tlast, p]
    # One writer per group, or every head in a group would write the same row. The
    # compare folds away at ``G == H``, where ``gidx`` is ``hidx``.
    if hidx == gidx * per_group:
        for step in cutlass.range_constexpr((span + threads - 1) // threads):
            d = tid + step * threads
            if d < span:
                gblast[bidx, gidx, sidx * span + d] = band[bidx, gidx, tlast, d]

    for k in cutlass.range_constexpr(cells):
        for j in cutlass.range_constexpr(unit):
            if cutlass.const_expr(has_z0):
                state[unit * k + j] = gz0[bidx, hidx, unit * plane[k] + j]
            else:
                state[unit * k + j] = cutlass.Float32(0.0)

    # The fused tap's weight is ``u`` at the token before the slot, which is rows
    # ``0..kblk-1`` of the staging tile. Row ``kblk`` is the slice's own last token and
    # is read only by the residue, at the last slice.
    va_prv = cute.make_tensor(
        su.iterator, cute.make_layout((mpad, kblk), stride=(1, lda))
    )
    vbfuse = cute.make_tensor(
        sbfuse.iterator, cute.make_layout((span, kblk), stride=(1, ldb))
    )

    last = chunk - 1
    for cidx in cutlass.range(chunks):
        t0 = cidx * chunk
        valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)

        acc.fill(0.0)
        # Restaged every chunk, not once: the columns at or past each tile's data
        # width are read as operands and never restaged by the stagers, and the
        # previous chunk's result was written over them. ``su`` runs to its full
        # pitch because its M mode is the rounded extent: columns P..mpad-1 are read
        # as zero rows.
        stage_pad(sbfuse, tid, threads, kblk, span, ldb)
        stage_pad(su, tid, threads, kblk + 1, rows, lda)
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

        build_table(
            strans,
            stap,
            squat,
            stable,
            tid,
            threads,
            chunk,
            2,
            fused=True,
            pitch=TABLE_PITCH,
        )
        for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
            token = tid + step * threads
            if token < chunk:
                # I3: one exponential of a log difference, never a ratio of two.
                swgt[token] = decay(slp[last] - slp[token])
        # Read by every thread for the recurrence below and stored by one band, so
        # the transition the backward reads is the one this block applied.
        quat, decayed = chunk_endpoint(squat, slp, chunk)
        if sidx == 0 and tid == 0:
            for j in cutlass.range_constexpr(4):
                gcquat[bidx, hidx, cidx, j] = quat[j]
            gcscale[bidx, hidx, cidx] = decayed

        # The slice loop stays unrolled: a dynamic loop over the same body costs the
        # unfused increment 168 registers against 114.
        for s in cutlass.range_constexpr(slices):
            lbase = s * kblk
            cute.arch.sync_threads()
            # The rank-one residue, one row and one column vector at slot ``L-1``. The
            # reindex that fused the taps has no slot to move that token's now-tap to,
            # and ``wgt_{L-1}`` is ``exp(0)``, so the row carries no weight and
            # ``scaled`` is false rather than a multiply by one.
            #
            # A ragged tail cancels it instead of needing a case. Slot ``L-1`` of a tail
            # chunk is a pad token, so ``chunk - 1 < valid`` is false and the helper's
            # own keep predicate zeroes the row, while ``u`` there is zero for the same
            # reason; the last real token's residue arrives through slot ``valid`` of
            # the fused operand below. Confirmed against
            # :func:`slinoss.ops.so3ssd.reference.chunked_forward_fused`, which forms
            # the same two terms over a padded chunk and whose explicit residue is
            # likewise zero there.
            #
            # Staged ahead of the operands rather than after the loop. ``sres`` is
            # outside the arena and the barrier above already published ``stable``, so
            # the position is legal, and the residue's global load then covers under the
            # operand staging and the contraction instead of standing in front of the
            # arena's write-after-read barrier, whose stall is 32.28% of this kernel's
            # ``barrier`` class. Measured on sm_86, paired, 320 pairs a shape, bitwise
            # identical at every one: -9.73 us at ``B=4 H=18 T=2048 P=64 N=80 L=64``,
            # -2.56 at ``B=4 H=12 T=2048 P=48 N=16``, -1.02 at ``P=64 N=32``, -3.07 at
            # the ragged shape, and +0.00 unresolved at the one-head shape. ``L=128``,
            # where the guard below sends the residue back, reads -0.01 in the same loop
            # and is the null control on the instrument.
            #
            # Only at one slice. At more than one the same move costs time and the
            # counters do not say why: measured at ``L=128``, four slices, it is +5.12 us
            # on the last slice and +4.61 on the first, while registers fall 236 to 216
            # with no spill either side and both ``barrier`` and ``long_scoreboard`` fall
            # -- neither a register nor a stall effect, so the position stays where it
            # was measured to belong.
            #
            # ``TABLE_AN`` is read whatever ``fused`` selects: the flag chooses the
            # first column and ``An`` is a second one. Converting the last unfused
            # caller in the tree retires the flag, not the ``An`` column, which this
            # read needs.
            if cutlass.const_expr(slices == 1) and tid < reswarps:
                stage_rotated(
                    band,
                    bandprev,
                    sres,
                    stable,
                    swgt,
                    bidx,
                    gidx,
                    t0,
                    last,
                    valid,
                    tid,
                    TABLE_AN,
                    0,
                    threads,
                    1,
                    lanes,
                    has_prev,
                    False,
                    pitch=TABLE_PITCH,
                )
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
            # ``valid + 1``, not ``valid``. The staging helper zeroes rows at or past
            # the token count it is handed, and slot ``valid`` of a ragged tail chunk
            # is load-bearing: ``ls`` stages as zero there so ``Afuse_valid`` is the
            # last real token's ``An``, the operand is that token's ``b``, and the
            # weight is one because ``lp`` is flat past the sequence. Zeroing it as a
            # pad row leaves ``y`` at roundoff and moves ``state`` by O(1). The
            # shifted-token read stays in bounds: the clamp becomes
            # ``min(lbase + r, valid)``, which is at most ``chunk - 1`` because
            # ``kblk`` divides ``chunk``, so the table index and the global token both
            # hold.
            stage_rotated(
                band,
                bandprev,
                sbfuse,
                stable,
                swgt,
                bidx,
                gidx,
                t0,
                lbase,
                valid + 1,
                tid,
                TABLE_AFUSE,
                1,
                threads,
                kblk,
                lanes,
                has_prev,
                True,
                pitch=TABLE_PITCH,
            )
            cute.arch.sync_threads()
            mma_gemm(tiled_mma, tid, acc, va_prv, vbfuse, False, False)

        # The same residue at more than one slice, in the position the measurement kept.
        # Written before the barrier below and read after the next one, so the store's
        # own pair publishes it and the residue adds no barrier of its own.
        if cutlass.const_expr(slices > 1) and tid < reswarps:
            stage_rotated(
                band,
                bandprev,
                sres,
                stable,
                swgt,
                bidx,
                gidx,
                t0,
                last,
                valid,
                tid,
                TABLE_AN,
                0,
                threads,
                1,
                lanes,
                has_prev,
                False,
                pitch=TABLE_PITCH,
            )
        # ``u`` at the chunk's last token, from the row the shifted stage fills past the
        # slice. Read here because the store below writes over the whole arena, and
        # zero on a ragged tail because the stage zeroes rows past the sequence.
        for k in cutlass.range_constexpr(cells):
            fu[k] = widen(su[kblk, rowof[k]], gu.element_type)

        # The store writes over the operands the contraction just read, the two
        # being one region.
        cute.arch.sync_threads()
        mma_store(tiled_mma, tid, acc, sinc, (mpad, span), rows)
        cute.arch.sync_threads()

        # The matrix is one per chunk for every 3-vector the thread carries, and the
        # transpose the backward's mirror needs is not this direction's.
        mat = rot_hom(quat)
        # A sub-tensor taken at a dynamic index reports its iterator as aligned to one
        # element whatever the parent claimed, and cute.autovec_copy caps the access at
        # the iterator's claim, so the cell's own alignment is restated here.
        zcell = cute.zipped_divide(
            cute.make_tensor(
                gzstart[bidx, hidx, cidx, None].iterator.align(
                    _alignment(unit * (elem.width // 8))
                ),
                cute.make_layout((rows * dim,), stride=(1,)),
            ),
            (unit,),
        )
        for k in cutlass.range_constexpr(cells):
            cute.autovec_copy(scell[(None, slot[k])], finc)
            cute.autovec_copy(rescell[(None, col[k])], fres)
            # The state entering the chunk is the accumulator before this chunk's
            # increment enters it, so the narrowed copy is filled and stored before the
            # update overwrites what it holds.
            for j in cutlass.range_constexpr(unit):
                fz[j] = narrow(state[unit * k + j], elem)
            cute.autovec_copy(fz, zcell[(None, plane[k])])
            # The scale multiplies the state alone, then the sum is rotated once.
            # The increment is in the chunk-local frame, so it shares the rotation
            # rather than needing its own. One matrix serves every triple in the cell.
            # The residue is in that frame too, and it is one FMA a component here
            # against a second ``L x span`` operand tile and a second contraction.
            for h in cutlass.range_constexpr(wide):
                base = unit * k + 3 * h
                turned = mat3_matvec(
                    mat,
                    (
                        decayed * state[base] + finc[3 * h] + fu[k] * fres[3 * h],
                        decayed * state[base + 1]
                        + finc[3 * h + 1]
                        + fu[k] * fres[3 * h + 1],
                        decayed * state[base + 2]
                        + finc[3 * h + 2]
                        + fu[k] * fres[3 * h + 2],
                    ),
                )
                for j in cutlass.range_constexpr(3):
                    state[base + j] = turned[j]
        # The next chunk's staging writes over the tile just read, the two being one
        # region, so the reads have to finish first. The barriers inside the staging
        # come after its writes and cannot stand in for this one.
        cute.arch.sync_threads()

    for k in cutlass.range_constexpr(cells):
        for j in cutlass.range_constexpr(unit):
            gstate[bidx, hidx, unit * plane[k] + j] = state[unit * k + j]


@cute.jit
def increment_passing_fwd(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    gz0: cute.Tensor,
    gzstart: cute.Tensor,
    gstate: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gblast: cute.Tensor,
    gulast: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bands: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    stream: Stream,
    dtype: cutlass.Constexpr,
    warps: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    kblk: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    has_z0: cutlass.Constexpr,
    resident: cutlass.Constexpr,
) -> None:
    """Launch :func:`increment_passing_fwd_kernel`.

    The grid is ``(H, 3N/span, B)``, head-fastest, so the ordering argument the
    unfused GEMM makes about ``b`` survives the fusion.

    ``resident`` is the launch bound, computed from the tiles by the host entry
    rather than chosen here; see :data:`RESIDENT_MAX`.

    The block width is the tiling's warp count and nothing else, so it arrives as
    ``warps`` and the thread count is derived from it: two parameters would let the
    launch geometry and the accumulator partition disagree.
    """
    threads = warps * 32
    increment_passing_fwd_kernel(
        gu,
        gtrans,
        gtap,
        gb,
        guprev,
        gbprev,
        gz0,
        gzstart,
        gstate,
        gcquat,
        gcscale,
        gblast,
        gulast,
        seqlen,
        chunks,
        make_mma(dtype, warps),
        threads,
        chunk,
        rows,
        dim,
        span,
        kblk,
        per_group,
        has_prev,
        has_z0,
    ).launch(
        grid=(heads, bands, bsz),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
        stream=stream,
    )


class IncrementPassing(NamedTuple):
    """Result of the fused increment and recurrence.

    Attributes:
        zstart: ``(B,H,C,P,3N)`` state entering each chunk, at the activation dtype.
            A fresh buffer: the increment it was formed from never exists in memory.
        state: ``(B,H,P,3N)`` float32 state after the last chunk.
        cquat: ``(B,H,C,4)`` float32 unit chunk rotation, scalar-first.
        cscale: ``(B,H,C)`` float32 chunk decay ``exp(2*lp_{L-1})``.
        b_last: ``(B,G,3N)`` ``b`` at token ``T-1``, the dtype of ``B``, contiguous.
        u_last: ``(B,H,P)`` ``u`` at token ``T-1``, the dtype of ``U``, contiguous.
    """

    zstart: Tensor
    state: Tensor
    cquat: Tensor
    cscale: Tensor
    b_last: Tensor
    u_last: Tensor


def increment_passing_forward(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    chunk_size: int,
    *,
    z0: Tensor | None = None,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
    span: int = SPLIT,
    warps: int = WARPS,
    kblk: int | None = None,
    resident: int | None = None,
) -> IncrementPassing:
    """Form every chunk's start state, its transition, and the segment carry-out.

    Args:
        U: ``(B,H,T,P)``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)`` float32, contiguous. Per-tap ``(kr, g, h, 0)``.
        B: ``(B,G,T,3N)``, the dtype of ``U``, pitched. One column band of the
            mixer's fused projection. ``G`` divides ``H``; head ``h`` reads group
            ``h // (H // G)``.
        chunk_size: ``L``. A multiple of ``kblk`` and of
            :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_K`.
        z0: ``(B,H,P,3N)`` float32, contiguous. Zero state if omitted.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, the dtype of ``U``. Paired with
            ``b_prev``.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, the dtype of ``U``.
        span: Band width. :data:`SPLIT` is the narrowest legal one; it is an
            argument so a driver can price a wider one.
        warps: Warps per block, a multiple of
            :data:`slinoss.ops.so3ssd.cute.common.WARPS` at most
            :data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE`. Warps past the first
            four go to the tile's N mode, which halves both accumulators at
            unchanged shared bytes and unchanged traffic. Measured slower at
            :data:`SPLIT` and faster at the whole width, so the default is four and
            a driver pricing a wide band raises it.
        kblk: K extent of one slice, a divisor of ``chunk_size``. None takes
            :func:`fused_kblock`. A wider slice widens both operand tiles against one
            fewer barrier pair per chunk.
        resident: Blocks per SM the launch bound asks for. Defaults to
            :data:`RESIDENT_MAX` capped by the shared-memory budget.

    Returns:
        An :class:`IncrementPassing`.

    Raises:
        ValueError: On a layout, rank, shape, extent, or pairing violation, on a
            band width the launch cannot cover exactly, or on a ``warps`` that is
            not a legal block width.
        TypeError: On an activation dtype with no tensor-core path.
    """
    # ``B`` is a band and the rest is not, so the layout rule splits while the dtype
    # group stays whole: one call is what makes a mixed-dtype pair reachable.
    activations: Named = ((U, "U"), (B, "B"))
    dense: Named = ((U, "U"),)
    if u_prev is not None and b_prev is not None:
        activations = (*activations, (u_prev, "u_prev"), (b_prev, "b_prev"))
        dense = (*dense, (u_prev, "u_prev"), (b_prev, "b_prev"))

    pinned: Named = ((trans, "trans"), (K, "K"))
    if z0 is not None:
        pinned = (*pinned, (z0, "z0"))
    check_layout((*dense, *pinned))
    check_pitched(((B, "B"),))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(U, trans, K, (B, "B"))
    if kblk is None:
        kblk = fused_kblock(chunk_size, rows, span, U.element_size())
    check_extents(chunk_size, dim, kblk)
    has_prev = check_stream(u_prev, b_prev, (bsz, heads, groups, rows, dim))

    # Raises on an illegal width, so the block geometry is checked here rather than
    # inside the trace.
    mma_atoms(warps)
    threads = warps * 32
    if (
        span % 3 != 0
        or span % MMA_TILE_N != 0
        or dim % span != 0
        or (rows * span // 3) % threads != 0
    ):
        raise ValueError(
            f"span must divide 3N={dim}, be a multiple of 3 and of {MMA_TILE_N}, "
            f"and give a whole number of {threads}-thread tiles of P*span/3, got "
            f"span={span} P={rows}"
        )
    budget = fused_smem_bytes(chunk_size, rows, span, U.element_size(), kblk=kblk)
    assert_smem_fits(
        f"increment_passing_fwd[L{chunk_size}/P{rows}/span{span}/K{kblk}/W{warps}]",
        budget,
    )
    asked = RESIDENT_MAX if resident is None else resident
    blocks = min(asked, smem_residency(budget))

    chunks = -(-seqlen // chunk_size)
    opts = {"dtype": torch.float32, "device": U.device}
    zstart = torch.empty(bsz, heads, chunks, rows, dim, dtype=dtype, device=U.device)
    state = torch.empty(bsz, heads, rows, dim, **opts)
    cquat = torch.empty(bsz, heads, chunks, 4, **opts)
    cscale = torch.empty(bsz, heads, chunks, **opts)
    b_last = torch.empty(bsz, groups, dim, dtype=dtype, device=U.device)
    u_last = torch.empty(bsz, heads, rows, dtype=dtype, device=U.device)

    if z0 is None:
        start = state
    elif tuple(z0.shape) != (bsz, heads, rows, dim):
        raise ValueError(f"z0 must be {(bsz, heads, rows, dim)}, got {tuple(z0.shape)}")
    else:
        start = z0

    # A placeholder keeps one launch signature. It is never read: the branch that
    # would read it is closed at compile time.
    ustream = U[:, :, 0] if u_prev is None else u_prev
    bstream = B[:, :, 0] if b_prev is None else b_prev
    jit_launch(
        increment_passing_fwd,
        (
            U,
            trans,
            K,
            B,
            ustream,
            bstream,
            start.view(bsz, heads, rows * dim),
            zstart.view(bsz, heads, chunks, rows * dim),
            state.view(bsz, heads, rows * dim),
            cquat,
            cscale,
            b_last,
            u_last,
            seqlen,
            chunks,
            dim // span,
            bsz,
            heads,
        ),
        (
            cute_dtype(dtype),
            warps,
            chunk_size,
            rows,
            dim,
            span,
            kblk,
            heads // groups,
            has_prev,
            z0 is not None,
            blocks,
        ),
    )
    return IncrementPassing(
        zstart=zstart,
        state=state,
        cquat=cquat,
        cscale=cscale,
        b_last=b_last,
        u_last=u_last,
    )
