"""Chunk-start cotangent and reverse state recurrence in one launch.

Unfused, the two halves talk through a whole ``(B,H,C,P,3N)`` float32 buffer:
:func:`slinoss.ops.so3ssd.cute.bwd.chunk_start.chunk_start_backward` writes
``dzstart`` and :func:`slinoss.ops.so3ssd.cute.bwd.state_passing.state_passing_backward`
reads it back and overwrites it with ``dinc``. That round trip is 283 MB of the
pair's 455 MB at the model geometry, and it is the only thing either kernel has
left to give: the recurrence measured 99.6% of its own DRAM floor.

Fused, ``dzstart`` never reaches DRAM. One block holds a chunk's tile in shared
memory, the recurrence consumes it there, and only ``dinc`` and ``dz0`` are
written:

    dgram(L,P)      = dy * exp(2*lp)
    dzstart_c(P,3N) = dgram_c^T crot_c
    acc_C = dstate,  dinc_c = acc_{c+1},
    acc_c = a_c R(Q_c)^T acc_{c+1} + dzstart_c,  dz0 = acc_0

Parallelism. The recurrence is serial in the chunk, so the chunk mode leaves the
grid and becomes a reverse loop inside the block, exactly the arm
``chunk_start_backward(serial=True)`` prices. What replaces it is the lane band:
the state's ``3N`` columns split into ``SPLIT``-wide bands, one block per band,
because each 3-vector's recurrence is independent of every other. ``dy`` and
``trans`` are then read once per band rather than once, and ``C`` is still read
once because each band reads its own columns of it.

Traffic at ``B=4 H=18 T=2048 P=64 3N=240 L=64``, ``G=1``, five bands, against the
455 MB the unfused pair moves:

    dy       5*18.87           =  94.35 MB read
    trans    5*2.36            =  11.79 MB read
    C        4*2048*240*2      =   3.93 MB read
    cquat    5*4*18*32*4*4     =   0.23 MB read
    cscale   5*4*18*32*4       =   0.06 MB read
    dinc     4*18*32*64*240*4  = 141.56 MB written
    dz0      4*18*64*240*4     =   4.42 MB written
                                 256.34 MB

The five bands of one head are dispatched 18 blocks apart and the whole grid is
360 blocks against 168 resident, so the bands sharing a head's ``dy`` are
co-resident and L2 can serve four of their five reads. The counted figure above
charges none of that.

Registers are what sets ``SPLIT``. The GEMM accumulator is ``mpad*SPLIT/threads``
float32 and the recurrence carries ``rows*SPLIT/threads`` more, so the pair grows
linearly in the band width: at ``P=64`` and 128 threads, 48 columns is 24 and 24.
The whole state at once is 120 and 120, which cannot fit 255 registers, and a
spilled recurrence accumulator touched once per chunk would move exactly the bytes
this kernel exists to delete. The band is not a tuning knob for anything else.

Shared memory. The two GEMM operand tiles and the tile the recurrence reads share
one region, because no chunk has both live at once: the contraction reads the
operands and then writes its result, and the next chunk restages the operands only
after the recurrence has read that result. At the model geometry that makes the
recurrence's tile free and the block's footprint the unfused GEMM's 20,992 B, which
is four resident blocks of the 101,376 B carveout against the two that 33,280 B
allowed. Occupancy is what this kernel is short of, so the overlay is the point of
it and not a saving.

Barriers. Three per chunk, and the overlay is why two of them exist. One inside
:func:`start_chunk` between the contraction and the store, or a warp writes its
accumulator over an operand another warp is still reading. One after the store,
before the recurrence reads the tile. One after the recurrence, before the next
chunk's staging writes over what it just read -- the barriers inside the staging
come after its writes and cannot stand in for this one.

Measured on sm_86, bfloat16, no final-state cotangent, at the geometry above, both
arms in one process under one floor fit, every figure twice:

    arm    us/call       DRAM MB/call
    pair   741.0  741.7  453.9  454.0
    fused  474.9  474.9  176.5  176.8

267 us and 277 MB a call, one call a layer. The fused kernel reaches 55.2% of the
262 us floor its own DRAM traffic implies, where the recurrence it absorbed reached
99.3% of its; ranking the arms by that percentage inverts them, and
:data:`slinoss.perf.declared.DECLARED` carries the reading. The request stream is
the 256 MB counted above, not the 176 MB crossing the bus, and the gap is the band
re-reads L2 serves. What is left is 51.1% of the bus at 26.3% issue, 20.0% achieved
occupancy of 25.0% theoretical, 152 registers, and no local memory.

Block width. Occupancy is what is left, and the width buys it without buying bytes:
``mma_atoms`` pins the M mode, so warps past the first four go to the tile's N mode
at atom granularity and both accumulators halve at unchanged shared bytes. The GEMM
accumulator goes from ``mpad*span/128`` to ``mpad*span/256`` and the recurrence's
from ``rows*lanes/128`` to half that, which takes the kernel from 152 registers to
80 and lets the same 20,992 B block hold three resident blocks of eight warps rather
than three of four. Measured on sm_86 at the geometry above, one call, medians of
three event runs and one NCU capture of three launches each:

    warps  us/launch  MB/launch  GB/s  of 85%  regs  occ theo/ach  issue  barrier
    4          473.7     176.51  372.6   55.4%   152  25.0%/19.2%  26.3%    12.8%
    8          408.7     176.52  431.9   64.3%    80  50.0%/39.0%  32.5%    22.3%

65 us a call for a parameter, traffic identical to two decimal places and nothing
spilled at either width. The percentage rises with the time here, unlike the
fusion's, because the width moves no bytes. What it does move is the barrier stall,
12.8% to 22.3%: three barriers a chunk in this kernel and four more inside
:func:`start_chunk`, and the chunk prefix behind one of them is warp 0's work, so a
wider block waits wider. That is the next lever and it is not the width's.

Row band. 360 blocks against 168 resident is 1.43 waves at three blocks an SM, and
the 0.715 quantization efficiency of that wave count accounts for the whole of the
39.0%-against-50.0% occupancy gap. Splitting ``P`` into bands as well as ``3N``
doubles the grid and closes it, and the kernel is slower anyway:
:func:`slinoss.ops.so3ssd.cute.mma.mma_rows` rounds the M extent back up to
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_M`, so half of every contraction is
padding, and ``trans``, ``C``, ``cquat`` and ``cscale`` are requested once per row
band. Measured on sm_86 at the geometry above, eight warps, one NCU capture of three
launches, event medians of three runs:

    rows  blocks  us/launch  MB/launch  occ theo/ach  issue  tensor  l1tex  smem st
    64       360      410.1     176.52  50.0%/39.0%  32.4%    7.5%  44.8%  5.21 M
    32       720      454.5     199.03  66.7%/58.4%  48.3%   13.8%  66.4%  7.40 M
    16      1440      738.2          -            -      -       -      -       -

The 64-row line recaptures the width table's eight-warp line and lands 0.3% off it.
Occupancy and issue both rose; the shared pipe and the bus took more than they gave.
The class percentage rose with them, 64.0% to 64.9%, on a kernel 11% slower -- an arm
:data:`slinoss.perf.declared.DECLARED` cannot be ranked by. The M mode has no atom
narrower than 64 rows, so the band has no cheaper form and is not a lever.

``warps`` is a parameter and the default is four; nothing in the operator's backward
passes it.
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
    smem_capacity,
)
from slinoss.ops.so3ssd.cute.bwd.chunk_start import (
    gram_tile,
    rotated_tile,
    start_chunk,
)
from slinoss.ops.so3ssd.cute.bwd.state_passing import StatePassingBwd
from slinoss.ops.so3ssd.cute.common import (
    WARPS,
    mat3_matvec,
    mat3_transpose,
    rot_hom,
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
    mma_atoms,
    mma_rows,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.table import stage_pad

__all__ = [
    "RESIDENT_MAX",
    "SPLIT",
    "fold_smem_bytes",
    "start_passing_backward",
    "start_passing_bwd",
    "start_passing_bwd_kernel",
    "state_tile",
]

RESIDENT_MAX: int = 3
"""Blocks per SM the launch bound asks for, before the shared-memory budget cuts it.

The chunk loop is serial and every iteration barriers three times, so a block spends
most of its time waiting: measured on sm_86 at the model geometry with the tiles
allocated separately, 42.1% of the issue slots stalled on ``long_scoreboard`` at
36.4% of the bus and 20.8% issue, which is a kernel bounded by having eight warps per
SM and not by the pipe. Residency is the only thing that covers it, and residency is
what overlaying the recurrence's tile on the operands' bytes buys: 20,992 B a block
at the model geometry, four blocks of it against the 101,376 B carveout.

Four is what the tiles admit; three is what the register file prefers, because the
bound caps a thread at ``65536 / (blocks * threads)`` registers and this kernel holds
a GEMM accumulator and a recurrence accumulator at once. Measured on sm_86 at the
model geometry, one call of the fused kernel:

    blocks/SM   us    regs   occupancy theo/achieved
    2          607.6   196   16.7% / 15.8%
    3          473.6   152   25.0% / 19.5%
    4          514.2   128   33.3% / 29.6%

Nothing spills at any of the three, so the cost of asking for four is the scheduling
the 24 registers buy and not local memory.

Three is also what the wide block prefers, and there the bound is the whole of it:
at eight warps the cap is 85 registers at three blocks and 128 at two, and the
kernel wants 80. Measured on sm_86 at the model geometry, medians of three event
runs, one call:

    blocks/SM   us
    1          492.5
    2          492.5
    3          411.6
    4          439.3

One and two tie because both admit the register count the kernel wants and both fit
two blocks of eight warps per SM; three is the first bound that fits a third.
"""

SPLIT: int = 48
"""Columns of ``3N`` one block contracts and carries.

A multiple of 3, so a band holds whole 3-vectors, and of
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N`, so it is an N extent the atom
covers. 48 is the smallest width satisfying both, and every legal ``3N`` is a
multiple of it, so one width divides every shape. The module docstring carries why
a wider band does not fit the register file.
"""


def state_tile(rows: int, span: int) -> Tile:
    """Chunk-start cotangent tile, ``(mpad, span)`` float32.

    Contiguous and row-major, which is what :func:`mma_store` requires of any
    destination, and pitched to the band width so the flat index of a 3-vector is
    the same arithmetic in shared memory and in ``dinc``.

    The rounded M extent is allocated even where the store predicates the added
    rows away, because the vectorized path writes all of them.

    Args:
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
    """
    return Tile((mma_rows(rows), span), (span, 1))


def arena_words(chunk: int, rows: int, span: int, itemsize: int = 2) -> tuple[int, int]:
    """Float32-word extent of the overlaid region, and the offset inside it.

    The two GEMM operand tiles and the tile the recurrence reads are never live at
    the same time: the contraction reads the operands and then writes its result,
    and the next chunk restages the operands only after the recurrence has read
    that result. So one region holds both, and the recurrence's tile costs nothing
    at the shape the kernel is declared against -- which is what leaves room for the
    residency :data:`RESIDENT_MAX` asks for.

    Args:
        chunk: ``L``.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.

    Returns:
        ``(words, gram_words)``: the region's float32-word extent, and the offset at
        which the rotated readout tile starts. Both are whole words because every
        tile's row pitch is a multiple of :data:`SMEM_SEGMENT`.
    """
    gram = smem_bytes([(gram_tile(chunk, rows), itemsize)])
    rotated = smem_bytes([(rotated_tile(chunk, span), itemsize)])
    state = smem_bytes([(state_tile(rows, span), 4)])
    return max(gram + rotated, state) // 4, gram // 4


def fold_smem_bytes(chunk: int, rows: int, span: int, itemsize: int = 2) -> int:
    """Shared memory the kernel allocates, in bytes.

    The four small float32 tiles of the chunk-start GEMM, then the one region
    :func:`arena_words` describes.

    Args:
        chunk: ``L``.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    words, _ = arena_words(chunk, rows, span, itemsize)
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 1), 4),
            (Tile((words,), (1,)), 4),
        ]
    )


@cute.kernel
def start_passing_bwd_kernel(
    gdy: cute.Tensor,
    gtrans: cute.Tensor,
    gc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gdstate: cute.Tensor,
    gdinc: cute.Tensor,
    gdz0: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_dstate: cutlass.Constexpr,
) -> None:
    """Contract each chunk's readout cotangent and run the reverse recurrence.

    One block per ``(band, batch, head)``, walking the chunks in reverse.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gc: ``(B,G,T,3N)`` operand-dtype output vectors.
        gcquat: ``(B,H,C,4)`` float32 unit chunk rotations.
        gcscale: ``(B,H,C)`` float32 chunk decays, ``exp(2*lp_{L-1})``.
        gdstate: ``(B,H,3*P*N)`` float32 cotangent of the final state. Read only
            when ``has_dstate``; the zero-seed variant is handed ``gdz0`` here so
            the signature has one form.
        gdinc: ``(B,H,C,3*P*N)`` float32, written with the cotangent of each
            chunk's increment in the global frame.
        gdz0: ``(B,H,3*P*N)`` float32, written with the cotangent of the initial
            state.
        seqlen: ``T``. Dynamic.
        chunks: ``C``. Dynamic.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        span: Band width, :data:`SPLIT`. Compile-time.
        per_group: ``H // G``, heads sharing one ``c``. Compile-time.
        has_dstate: Whether a final-state cotangent is supplied. Compile-time.

    Invariants:
        ``span`` divides ``dim``, 3 divides ``span``, and ``threads`` divides
        ``rows * span / 3``, so the band is a whole number of 3-vectors and every
        thread owns the same count of them. ``|R(Q_c)| == 1`` and ``a_c`` lies in
        ``(0, 1]`` by I1, so the reverse recurrence cannot grow.
    """
    tid, _, _ = cute.arch.thread_idx()
    # Head is the fastest grid mode, as in the unfused GEMM: the ``H // G`` blocks
    # reading one group's readout band are co-resident. The band mode is next, so
    # the bands sharing a head's ``dy`` are close enough behind it to hit L2.
    hidx, sidx, bidx = cute.arch.block_idx()

    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    mpad = mma_rows(rows)
    lanes = span // 3
    vecs = rows * lanes // threads
    lda = smem_pitch(mpad)
    ldb = smem_pitch(span)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 1).layout(), 16)

    # One region, three views: the two GEMM operands, and the contraction's result
    # over the same bytes. The pitches make every offset a whole float32 word and a
    # whole 16-byte segment, which is the alignment both stagers and ``mma_store``
    # restate on the pointer they are handed.
    words, gwords = arena_words(chunk, rows, span, gdy.element_type.width // 8)
    arena = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((words,), stride=(1,)), SMEM_SEGMENT
    )
    sdy = cute.make_tensor(
        cute.recast_ptr(arena.iterator, dtype=gdy.element_type),
        gram_tile(chunk, rows).layout(),
    )
    scrot = cute.make_tensor(
        cute.recast_ptr(arena.iterator + gwords, dtype=gc.element_type),
        rotated_tile(chunk, span).layout(),
    )
    sdz = cute.make_tensor(arena.iterator, state_tile(rows, span).layout())

    # The band's origin is a pointer offset: ``3N`` is the last mode at unit stride,
    # so the staging pass indexes the band's own columns and never learns that the
    # state is wider.
    band = cute.make_tensor(gc.iterator + sidx * span, gc.layout)

    acc = mma_acc(tiled_mma, tid, (mpad, span))
    state = cute.make_fragment((3 * vecs,), cutlass.Float32)

    # One 3-vector's coordinates in the tile and in the ``(P,3N)`` plane. The band
    # index is dynamic and the vector index is not, so these are hoisted out of the
    # chunk loop rather than recomputed inside it.
    tile_row = []
    tile_col = []
    plane = []
    for k in cutlass.range_constexpr(vecs):
        # A name reused inside the chunk loop below at another structure is read as
        # a loop-carried variable whose type changes, which the DSL refuses.
        owned = tid + k * threads
        row = owned // lanes
        lane = owned - row * lanes
        tile_row.append(row)
        tile_col.append(3 * lane)
        plane.append(row * dim + sidx * span + 3 * lane)

    for k in cutlass.range_constexpr(vecs):
        for j in cutlass.range_constexpr(3):
            if cutlass.const_expr(has_dstate):
                state[3 * k + j] = gdstate[bidx, hidx, plane[k] + j]
            else:
                state[3 * k + j] = cutlass.Float32(0.0)

    for step in cutlass.range(chunks):
        cidx = chunks - 1 - step
        # Restaged every chunk, not once: the columns at or past each tile's data
        # width are read as operands and never restaged by the stagers, and the
        # previous chunk's result was written over them.
        stage_pad(scrot, tid, threads, chunk, span, ldb)
        stage_pad(sdy, tid, threads, chunk, rows, lda)
        start_chunk(
            gdy,
            gtrans,
            band,
            sdz,
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
            span,
            True,
        )
        cute.arch.sync_threads()

        quat = (
            gcquat[bidx, hidx, cidx, 0],
            gcquat[bidx, hidx, cidx, 1],
            gcquat[bidx, hidx, cidx, 2],
            gcquat[bidx, hidx, cidx, 3],
        )
        decayed = gcscale[bidx, hidx, cidx]
        # The transpose is a reindexing of a tuple at trace time, and the matrix is
        # one per chunk for every 3-vector the thread carries.
        mat = mat3_transpose(rot_hom(quat))
        for k in cutlass.range_constexpr(vecs):
            carried = (state[3 * k], state[3 * k + 1], state[3 * k + 2])
            # The increment cotangent is the accumulator before this chunk's
            # readout term enters it, so the store precedes the update.
            for j in cutlass.range_constexpr(3):
                gdinc[bidx, hidx, cidx, plane[k] + j] = carried[j]
            # One rotation, then the scale, then the readout cotangent: the forward
            # scales the state alone and rotates the sum, and transposing that
            # order puts the scale outside the rotation.
            turned = mat3_matvec(mat, carried)
            for j in cutlass.range_constexpr(3):
                state[3 * k + j] = (
                    decayed * turned[j] + sdz[tile_row[k], tile_col[k] + j]
                )
        # The next chunk's staging writes over the tile just read, the two being one
        # region, so the reads have to finish first. The barriers inside the staging
        # come after its writes and cannot stand in for this one.
        cute.arch.sync_threads()

    for k in cutlass.range_constexpr(vecs):
        for j in cutlass.range_constexpr(3):
            gdz0[bidx, hidx, plane[k] + j] = state[3 * k + j]


@cute.jit
def start_passing_bwd(
    gdy: cute.Tensor,
    gtrans: cute.Tensor,
    gc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gdstate: cute.Tensor,
    gdinc: cute.Tensor,
    gdz0: cute.Tensor,
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
    per_group: cutlass.Constexpr,
    has_dstate: cutlass.Constexpr,
    resident: cutlass.Constexpr,
) -> None:
    """Launch :func:`start_passing_bwd_kernel`.

    The grid is ``(H, 3N/span, B)``, head-fastest, so the ordering argument the
    unfused GEMM makes about ``C`` survives the fusion.

    ``resident`` is the launch bound, computed from the tiles by the host entry
    rather than chosen here; see :data:`RESIDENT_MAX`.

    The block width is the tiling's warp count and nothing else, so it arrives as
    ``warps`` and the thread count is derived from it: two parameters would let the
    launch geometry and the accumulator partition disagree.
    """
    threads = warps * 32
    start_passing_bwd_kernel(
        gdy,
        gtrans,
        gc,
        gcquat,
        gcscale,
        gdstate,
        gdinc,
        gdz0,
        seqlen,
        chunks,
        make_mma(dtype, warps),
        threads,
        chunk,
        rows,
        dim,
        span,
        per_group,
        has_dstate,
    ).launch(
        grid=(heads, bands, bsz),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
        stream=stream,
    )


def start_passing_backward(
    dy: Tensor,
    trans: Tensor,
    C: Tensor,
    cquat: Tensor,
    cscale: Tensor,
    chunk_size: int,
    dstate: Tensor | None = None,
    *,
    span: int = SPLIT,
    warps: int = WARPS,
    resident: int | None = None,
) -> StatePassingBwd:
    """Form every chunk's start-state cotangent and pass it back through the scan.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous. A
            caller with no ``dy`` has no chunk-start cotangent to form and runs the
            recurrence alone rather than this kernel against zeros.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        C: ``(B,G,T,3N)``, the dtype of ``dy``, pitched. ``G`` divides ``H``; head
            ``h`` reads group ``h // (H // G)``.
        cquat: ``(B,H,C,4)`` float32, contiguous. Unit chunk rotations
            ``Q_{L-1}``, scalar-first.
        cscale: ``(B,H,C)`` float32, contiguous. Chunk decays ``exp(2*lp_{L-1})``.
        chunk_size: ``L``. A multiple of 16.
        dstate: ``(B,H,P,3N)`` float32, contiguous. Zero seed if omitted.
        span: Band width. :data:`SPLIT` is the only value the register budget
            admits at every legal ``P``; it is an argument so a driver can price a
            wider one.
        warps: Warps per block, a multiple of
            :data:`slinoss.ops.so3ssd.cute.common.WARPS` at most
            :data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE`. Warps past the first
            four go to the tile's N mode, which halves both accumulators at
            unchanged shared bytes and unchanged traffic; see the module docstring
            for what that is worth.
        resident: Blocks per SM the launch bound asks for. Defaults to
            :data:`RESIDENT_MAX` capped by the shared-memory budget; it is an
            argument for the same reason ``span`` is, since the cap it puts on the
            register file is what decides whether the residency is reached.

    Returns:
        A :class:`slinoss.ops.so3ssd.cute.bwd.state_passing.StatePassingBwd`. Both
        fields are fresh buffers: nothing is written in place, because the
        chunk-start cotangent this fuses away never exists in memory.

    Raises:
        ValueError: On a layout, rank, shape, or extent violation, on a band width
            the launch cannot cover exactly, or on a ``warps`` that is not a legal
            block width.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (C, "C"))
    pinned: Named = ((trans, "trans"), (cquat, "cquat"), (cscale, "cscale"))
    if dstate is not None:
        pinned = (*pinned, (dstate, "dstate"))
    check_layout(((dy, "dy"), *pinned))
    check_pitched(((C, "C"),))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(
        dy, trans, None, (C, "C"), label="dy"
    )
    check_extents(chunk_size, dim, chunk_size)

    chunks = -(-seqlen // chunk_size)
    if tuple(cquat.shape) != (bsz, heads, chunks, 4):
        raise ValueError(
            f"cquat must be {(bsz, heads, chunks, 4)}, got {tuple(cquat.shape)}"
        )
    if tuple(cscale.shape) != (bsz, heads, chunks):
        raise ValueError(
            f"cscale must be {(bsz, heads, chunks)}, got {tuple(cscale.shape)}"
        )
    # Raises on an illegal width, so the block geometry is checked here rather than
    # inside the trace.
    mma_atoms(warps)
    threads = warps * 32
    if span % 3 != 0 or dim % span != 0 or (rows * span // 3) % threads != 0:
        raise ValueError(
            f"span must divide 3N={dim}, be a multiple of 3, and give a whole "
            f"number of {threads}-thread tiles of P*span/3, got span={span} P={rows}"
        )
    budget = fold_smem_bytes(chunk_size, rows, span, dy.element_size())
    assert_smem_fits(
        f"start_passing_bwd[L{chunk_size}/P{rows}/span{span}/W{warps}]", budget
    )
    asked = RESIDENT_MAX if resident is None else resident
    blocks = min(asked, max(1, smem_capacity() // budget))

    dinc = torch.empty(
        bsz, heads, chunks, rows, dim, dtype=torch.float32, device=dy.device
    )
    dz0 = torch.empty(bsz, heads, rows, dim, dtype=torch.float32, device=dy.device)
    if dstate is None:
        seed = dz0
    elif tuple(dstate.shape) != (bsz, heads, rows, dim):
        raise ValueError(
            f"dstate must be {(bsz, heads, rows, dim)}, got {tuple(dstate.shape)}"
        )
    else:
        seed = dstate

    jit_launch(
        start_passing_bwd,
        (
            dy,
            trans,
            C,
            cquat,
            cscale,
            seed.view(bsz, heads, rows * dim),
            dinc.view(bsz, heads, chunks, rows * dim),
            dz0.view(bsz, heads, rows * dim),
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
            heads // groups,
            dstate is not None,
            blocks,
        ),
    )
    return StatePassingBwd(dinc=dinc, dz0=dz0)
