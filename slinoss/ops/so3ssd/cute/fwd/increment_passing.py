"""Chunk increment and the inter-chunk recurrence, in one launch.

    inc_c(P,span) = (u*wgt)^T Bn + (ushift*wgt)^T Bp
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

Both forcing taps are staged before either contracts. The unfused increment holds
one forcing tile and restages it between the two taps, which costs four barriers a
K slice; at a band width the second tile is free, because the arena is sized by the
float32 tile the recurrence reads and not by the operands, so the fused form stages
both and pays two.

The rotation stays factored out of the increment and out of the state. The GEMM
accumulates in the chunk-local frame, the recurrence adds the scaled state to it
there, and one ``R(Q_c)`` carries the sum into the global frame:
``a (R z) + R inc == R(a z + inc)``. The accumulator is never rotated, so no lane
needs its neighbours' columns and nothing crosses a thread.

Arena. The two GEMM operand tiles and the float32 tile the recurrence reads are
never live at once: the contraction reads the operands, the store writes the
result, and the next chunk restages the operands only after the recurrence has read
that result. One region holds all three, which is what leaves room for
:data:`RESIDENT_MAX` blocks per SM.

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
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AN,
    TABLE_AP,
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
    build_table,
    stage_chunk,
    stage_pad,
    stage_rotated,
    stage_shifted,
)

__all__ = [
    "RESIDENT_MAX",
    "SPLIT",
    "IncrementPassing",
    "arena_words",
    "fused_kblock",
    "fused_smem_bytes",
    "increment_passing_forward",
    "increment_passing_fwd",
    "increment_passing_fwd_kernel",
    "state_tile",
]

RESIDENT_MAX: int = 3
"""Blocks per SM the launch bound asks for, before the shared-memory budget cuts it.

The bound caps a thread at ``65536 / (blocks * threads)`` registers, and this kernel
holds a GEMM accumulator and the recurrence's state at once, so the residency and
the register count trade against each other rather than both being free.
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


def arena_words(
    kblk: int, rows: int, span: int, itemsize: int = 2
) -> tuple[int, int, int]:
    """Float32-word extent of the overlaid region, and the offsets inside it.

    Args:
        kblk: K extent of one slice.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.

    Returns:
        ``(words, bn_words, bp_words)``: the region's float32-word extent, and the
        offsets at which the two forcing tiles start. Every offset is a whole word
        and a whole :data:`slinoss.ops.so3ssd.cute.mma.SMEM_SEGMENT`, because every
        tile's row pitch is a multiple of the segment.
    """
    weights = smem_bytes([(input_tile(kblk, rows), itemsize)])
    forced = smem_bytes([(forced_tile(kblk, span), itemsize)])
    state = smem_bytes([(state_tile(rows, span), 4)])
    words = max(weights + 2 * forced, state) // 4
    return words, weights // 4, (weights + forced) // 4


def fused_smem_bytes(
    chunk: int, rows: int, span: int, itemsize: int = 2, *, kblk: int = KBLOCK_MAX
) -> int:
    """Shared memory the kernel allocates, in bytes.

    The six chunk-sized float32 tiles of the increment, then the one region
    :func:`arena_words` describes. Computed from the layouts, so there is one
    description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
        kblk: K extent of one slice.
    """
    words, _, _ = arena_words(kblk, rows, span, itemsize)
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 2), 4),
            (Tile((words,), (1,)), 4),
        ]
    )


def fused_kblock(chunk: int, rows: int, span: int, itemsize: int = 2) -> int:
    """Widest K slice that still holds :data:`RESIDENT_MAX` blocks on one SM.

    A wider slice is one barrier pair per chunk fewer and is monotonically faster
    while the residency holds, and the residency is worth more than the slice.
    Measured on sm_86 at ``L=64 P=64 span=48``, both directions of that trade:
    slice 64 at residency 3 runs 395.3 us against slice 16's 517.1 us, and the
    whole-width band, which reaches residency 1, costs 430.6 us at its own widest
    slice. So the search returns the widest slice inside the residency, never the
    widest slice.

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
            budget = fused_smem_bytes(chunk, rows, span, itemsize, kblk=kblk)
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
    lda = smem_pitch(mpad)
    ldb = smem_pitch(span)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    swgt = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 2).layout(), 16)

    # One region, four views: the three GEMM operands, and the contraction's result
    # over the same bytes.
    words, bnwords, bpwords = arena_words(kblk, rows, span, gu.element_type.width // 8)
    arena = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((words,), stride=(1,)), SMEM_SEGMENT
    )
    su = cute.make_tensor(
        cute.recast_ptr(arena.iterator, dtype=gu.element_type),
        input_tile(kblk, rows).layout(),
    )
    sbn = cute.make_tensor(
        cute.recast_ptr(arena.iterator + bnwords, dtype=gb.element_type),
        forced_tile(kblk, span).layout(),
    )
    sbp = cute.make_tensor(
        cute.recast_ptr(arena.iterator + bpwords, dtype=gb.element_type),
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

    # The band's origin is a pointer offset: ``3N`` is the last mode at unit stride,
    # so the staging pass indexes the band's own columns and never learns that the
    # state is wider. The streaming carry-in is banded with it.
    band = cute.make_tensor(gb.iterator + sidx * span, gb.layout)
    bandprev = cute.make_tensor(gbprev.iterator + sidx * span, gbprev.layout)

    acc = mma_acc(tiled_mma, tid, (mpad, span))
    state = cute.make_fragment((unit * cells,), cutlass.Float32)
    elem = gzstart.element_type
    finc = cute.make_fragment((unit,), cutlass.Float32)
    fz = cute.make_fragment((unit,), elem)

    # One cell's index in the tile and in the ``(P,3N)`` plane, counted in cells and not
    # in elements, so neither consumer divides by ``unit`` at run time. The band index
    # is dynamic and the cell index is not, so these are hoisted out of the chunk loop
    # rather than recomputed inside it.
    scols = span // unit
    zcols = dim // unit
    slot = []
    plane = []
    for k in cutlass.range_constexpr(cells):
        owned = tid + k * threads
        row = owned // (lanes // wide)
        cell = owned - row * (lanes // wide)
        slot.append(row * scols + cell)
        plane.append(row * zcols + sidx * scols + cell)

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

    # Two views of one staging tile, one row of pitch apart. The current tap reads
    # token t0+lbase+k, the previous one reads t0+lbase+k-1.
    va_now = cute.make_tensor(
        su.iterator + lda, cute.make_layout((mpad, kblk), stride=(1, lda))
    )
    va_prv = cute.make_tensor(
        su.iterator, cute.make_layout((mpad, kblk), stride=(1, lda))
    )
    vbn = cute.make_tensor(
        sbn.iterator, cute.make_layout((span, kblk), stride=(1, ldb))
    )
    vbp = cute.make_tensor(
        sbp.iterator, cute.make_layout((span, kblk), stride=(1, ldb))
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
        stage_pad(sbn, tid, threads, kblk, span, ldb)
        stage_pad(sbp, tid, threads, kblk, span, ldb)
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

        build_table(strans, stap, squat, stable, tid, threads, chunk, 2)
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
                band,
                bandprev,
                sbn,
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
            stage_rotated(
                band,
                bandprev,
                sbp,
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
            mma_gemm(tiled_mma, tid, acc, va_now, vbn, False, False)
            mma_gemm(tiled_mma, tid, acc, va_prv, vbp, False, False)

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
            # The state entering the chunk is the accumulator before this chunk's
            # increment enters it, so the narrowed copy is filled and stored before the
            # update overwrites what it holds.
            for j in cutlass.range_constexpr(unit):
                fz[j] = narrow(state[unit * k + j], elem)
            cute.autovec_copy(fz, zcell[(None, plane[k])])
            # The scale multiplies the state alone, then the sum is rotated once.
            # The increment is in the chunk-local frame, so it shares the rotation
            # rather than needing its own. One matrix serves every triple in the cell.
            for h in cutlass.range_constexpr(wide):
                base = unit * k + 3 * h
                turned = mat3_matvec(
                    mat,
                    (
                        decayed * state[base] + finc[3 * h],
                        decayed * state[base + 1] + finc[3 * h + 1],
                        decayed * state[base + 2] + finc[3 * h + 2],
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
            :func:`fused_kblock`. A wider slice is two more operand tiles against
            one fewer barrier pair per chunk.
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
