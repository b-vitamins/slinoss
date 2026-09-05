"""Per-chunk staging and the 3x3 transform table.

The chunked factorization needs three matrices per token:

    Ap_t = R(Q_t)^T Kprev_t         applied to b_{t-1}
    An_t = R(Q_t)^T Kcurr_t         applied to b_t
    Ac_t = R(Q_t)^T                 applied to c_t

Each is built once per token, in shared memory, by one thread. Every vector
transform afterwards is a nine-FMA 3x3 matvec whose matrix operand is a broadcast
shared-memory read. Applying the rotation and the tap as two separate per-lane
passes would cost more arithmetic and two more passes over the ``3N`` data, so
they are composed here instead.

``Ac`` is an intermediate of both tap matrices, so a kernel that needs only the
taps still builds it and merely does not store it. ``mats`` selects that: two
slots for the chunk increment, three for the chunk scan. Slot order puts the taps
first so the increment's table is a prefix of the scan's and the slot indices are
the same constants in both.

One slot is the other direction. A kernel that reads out but does not force needs
``Ac`` and nothing else, so at ``mats == 1`` the build computes strictly less: no
tap matrix, no 3x3 product, and no read of ``stap`` or ``strans`` at all, which
also frees the caller of staging ``K``. Its sole slot is ``TABLE_AC_SOLE``, a
distinct constant, because at one slot the order is no longer a prefix of the
three-slot order.

Cost. Building one token is one quaternion exponential, one rotation matrix, two
tap matrices, and two 3x3 products: order 120 FMA. Applying it is ``9*N`` FMA per
tap. The build amortizes from ``N = 16``, which is the smallest legal lane count.

Table storage is ``(mats, L, pitch)`` float32, nine entries innermost, at a pitch
the kernel chooses. At the natural pitch of nine the build's stores are a bank
permutation, nine being coprime with the 32 banks, and the nine entries are read one
at a time. At :data:`TABLE_PITCH` the entry is three whole segments, so a read is
three loads rather than nine and the build's stores take a four-way conflict
instead. Every read during application is a broadcast, and neither pitch needs a
swizzle.

Staging is transposed on the way in: global ``(L, 4)`` and ``(L, 8)`` become
shared ``(4, L)`` and ``(8, L)``. One thread owns one token here and in the build,
so a component access is unit stride across the warp. The prefix scan reads the
same tiles at a block stride instead, conflict-free by the block-width argument in
``common.py``, which is the constraint ``MAX_CHUNK`` is set by.

A ragged tail is staged as the identity transition and a zero tap, which is what
:func:`slinoss.ops.so3ssd.reference.chunk_pad` does: ``quat_exp(0)`` is the
identity and a zero tap kills the forcing, so the padded tokens contribute
nothing and need no separate code path.

Applying the table is also here, as :func:`stage_rotated`. Every kernel in the
tree transforms a ``(L, 3N)`` tensor by one table slot on its way into a
shared-memory operand tile, and the transform is the same nine FMAs whichever slot
and whichever tensor: the current tap on ``b``, the previous tap on ``b`` shifted
one token, and the readout matrix on ``c``. One implementation covers all three,
because two copies of a rowwise transform diverge and the divergence is a
correctness bug. Arithmetic intensity is near 1.5 flop/byte, so the transform is
memory bound and never gets a kernel of its own; the rotated tensors do not reach
global memory.

The backward transforms by ``A^T`` rather than ``A``. That is the same nine FMAs
over a permuted index triple, so it is the ``transposed`` flag of
:func:`stage_rotated` rather than a second helper: the flag selects which table
entry feeds each output component during the trace, and ptxas sees one
straight-line matvec either way.

The stagings that need no table are here for the same reason: more than one kernel
needs each. :func:`stage_shifted` lays a ``(T, P)`` tensor into a tile one row
behind, so the two taps read one staging pass through two views;
:func:`stage_state` narrows a chunk-start state into an operand tile; and
:func:`stage_matrix` transforms a ``(P, 3N)`` tensor by one matrix that holds for
the whole chunk, keeping the float32 result beside the narrowed operand when the
consumer needs both widths; and :func:`stage_weighted` scales a ``(T, P)`` tensor
by the chunk-local decay on its way in, which every backward operand that carries
the weight rather than its partner needs. :func:`stage_score` has one caller and
is here for the run helpers rather than for a second one: the segment width, the
retiling and the alignment claim are this module's and are not exported.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

from slinoss._cute import Scalar, decay, narrow, select, widen
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AC_SOLE,
    TABLE_AN,
    TABLE_AP,
    Mat3,
    mat3_matvec,
    mat3_mul,
    mat3_transpose,
    rot_hom,
    tap_matrix,
)
from slinoss.ops.so3ssd.cute.mma import SMEM_SEGMENT

__all__ = [
    "LANE_PAIR",
    "PREFETCH",
    "ROT_RUN",
    "TABLE_PITCH",
    "TABLE_QUAD",
    "apply_matrix",
    "apply_rotated",
    "build_table",
    "mat_at",
    "matrix_frag",
    "paired",
    "read_matrix",
    "read_rotated",
    "rotated_frags",
    "stage_chunk",
    "stage_matrix",
    "stage_pad",
    "stage_raw",
    "stage_rotated",
    "stage_score",
    "stage_shifted",
    "stage_state",
    "stage_trans",
    "stage_weighted",
    "store_pair",
    "weight_rows",
]

PREFETCH: int = 4
"""Staging steps whose global loads are issued before any of them is consumed.

Bounds the load phase at ``3 * PREFETCH`` live source elements while keeping
``3 * PREFETCH`` loads outstanding per thread, which is what covers one global
latency. One is the serial form: load, transform, store, wait, repeat.

Eight does not convert, and the reason bounds what depth can ever buy here. Every
group loop below is a ``range_constexpr``, so a staging pass is already one
straight-line block and the depth sets the emission order alone: doubling it
leaves ``sm__inst_executed`` at 69,306,624 with every per-pipe count identical.
Measured on sm_86, one process, bit-exact at all six shapes, 4 -> 8:

    shape       registers    local sectors  cycles              long_scoreboard
    standard    220 -> 220   0 -> 0         172,595 -> 172,105  41.24 -> 40.93
    wide        255 -> 255   0 -> 0         298,842 -> 298,741  33.63 -> 32.80
    acceptance  255 -> 255   3,133,440 both 987,700 -> 992,270  31.66 -> 29.90

The memory stall falls between 0.3 and 1.8 points and the duration does not
follow: acceptance moves 4.1 us of 577 and resolves in no rep of four. The
residual is service time for a working set that does not fit, not an issue order
the scheduler had left uncovered. Register pressure is not the refusal either --
``standard`` runs at 220 registers with 35 spare and no spill, and converts least
of the three.

Divided by the run, not a fixed count of loads. A flat pass at :func:`_flat_run` width
reaches one step in flight, and restoring two there is a regression at equal registers:
what covers a latency is the elements the group holds, and the wide access already
holds them. A stall share does not price it either way, since deleting instructions
shrinks the denominator every share is taken over."""

TABLE_QUAD: int = SMEM_SEGMENT // 4
"""Float32 words in one 16-byte shared-memory segment."""

TABLE_PITCH: int = 3 * TABLE_QUAD
"""Padded float32 pitch of one transform-table entry: nine words in three segments.

At the natural pitch of nine the token stride is 36 bytes, so only every fourth
entry starts on a segment boundary and no entry can be read at vector width. At
twelve the stride is 48 bytes, every entry is aligned, and three 16-byte loads cover
a row exactly: :func:`mat_at` reads nine words in three instructions rather than
nine.

The cost is a third more table bytes, ``mats * L * 12`` against ``mats * L * 9``,
and a four-way conflict on the build's stores in place of none: nine is coprime with
the 32 banks so a nine-word stride permutes them, while ``12 * token`` modulo 32
takes eight values. The build stores each entry once a block and every application
reads it once an element, so the trade is decided by the reads.

Adopt it per kernel, by passing this pitch to
:func:`slinoss.ops.so3ssd.cute.common.table_tile` and to the readers. A kernel whose
table is at the natural pitch is unaffected."""

LANE_PAIR: int = 2
"""Adjacent 3-vectors one thread transforms and stores per step.

Two, not one, because of the shared-memory store. A pair's six elements are one
contiguous 12-byte run at four-byte alignment, so they go out as three accesses of
:data:`LANE_PAIR` elements rather than six scalars. The source elements of a pair
are contiguous too, so the read is three paired accesses over the same sectors.

What that buys is instruction count, not a conflict-free pattern. Measured at
``standard``, forward, sm_86, median of four launches: ``chunk_scan_fwd`` shared
stores fall 840,192 to 619,008 and store conflicts 442,368 to 221,184,
``chunk_increment_fwd`` 608,256 to 460,800 and 363,332 to 184,320. The conflicts
track the halved store count, so a paired store is conflicted where the two scalar
stores it replaces were. Global load requests fall 743,424 to 522,240 and 485,568
to 338,112; duration falls 120.4 to 107.7 and 69.3 to 64.0 us; neither kernel
spills. The residue is not attributed to a site, because the counters are per
kernel and the staging stores share them with the operand stores; separating the
two needs source-level counters.

The pattern the pairing leaves is two-way wherever a warp's 32 linear indices straddle
a row. At a 48-column bf16 tile the pair stride is 28 four-byte words, so 24 pairs of
one row and eight of the next take banks ``0..23`` and ``28..31, 0..3``. A map of eight
rows by four pairs a warp clears it: ``28 * r`` modulo 32 over eight rows is a
stride-four permutation and four consecutive pairs tile it exactly. It converts to zero,
by an identity and not by a close call. The remap costs the global read its contiguity
-- a warp goes from 128 adjacent bytes and four sectors to eight rows of 16 bytes and
eight sectors -- so it trades one extra shared wavefront on the store for four extra
sectors on the load, and both pipes run at 128 bytes a cycle, so it is one cycle for one
cycle at every geometry. At ``chunk_vector_bwd``, ``standard``, source-level counters put
the store side at 245,760 excess wavefronts a launch over 20 program counters; the same
census puts the load side at 983,040 added sectors. 1.6 us each of 215.

A step carries twice the elements, so both paired passes take ``PREFETCH //
LANE_PAIR`` steps per group and the live element count and the elements in flight
are the ones :data:`PREFETCH` states.

``lanes`` is even at every legal shape:
:func:`slinoss.ops.so3ssd.cute.guard.check_extents` requires ``3N`` to be a multiple
of 3 and of :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N`, so ``3N`` is a multiple
of 48 and ``lanes`` a multiple of 16.

The ``pairs = extent // LANE_PAIR`` divisor below sits on the address chain feeding
a global load, one division per step, and it cannot become an induction variable.
The row index advances by a constant per step only when ``threads`` is a multiple of
``pairs``, and that condition holds only where ``pairs`` is a power of two and the
division is already a shift. At 128 threads:

    helper           pairs         geometries
    stage_rotated    lanes // 4    4 is a shift; 20 leaves 128 % 20 = 8
    stage_state      dim // 2      24, 48, 120 divide 128 in no case
    stage_shifted    width // 2    32 is a shift; 24 leaves 128 % 24 = 8

:func:`stage_rotated` divides by :data:`ROT_RUN`, so its row above is the wide one.

So the rewrite is available at exactly the geometries where it removes nothing, and
absent at every geometry that pays a magic multiply. Enabling it needs a different
thread-to-element map, which gives up the contiguity the pairing exists for."""

_MATRIX_DEPTH: int = max(1, PREFETCH // LANE_PAIR)
"""Steps of a matrix pass whose loads are in flight at once."""

ROT_RUN: int = 4
"""Adjacent 3-vectors :func:`stage_rotated` transforms and stores per step.

Four rather than :data:`LANE_PAIR`, on the same argument the pairing rests on and
one step further. Twelve components are one contiguous run, so the three source
reads and the three destination accesses each carry four elements instead of two:
per twelve components the pass emits three loads, one table read and three stores
where the pairing emits six, two and six. The address arithmetic, the table read and
the scale read halve with them; the widen, the select, the FMA and the narrow are
per component and do not move.

Alignment holds at both destination widths and is not a claim the compiler has to
take on trust. ``3N`` is a multiple of 48, so a global row offset is a multiple of
96 bytes at the operand width; :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`
returns an odd multiple of ``SMEM_SEGMENT // itemsize`` elements, so a shared row
offset is a multiple of 16 bytes; and a run's column offset is a multiple of four
elements. Four float32 is the widest run that stays inside a 16-byte pitch multiple,
which is why the constant stops here.

Conflict-free within a row at both widths. An eight-byte access is serviced in
phases of sixteen threads over sixteen bank pairs and a sixteen-byte access in
phases of eight over eight segments; a run's index inside a row is ``3 * p + k``,
and 3 is invertible modulo both, so consecutive runs of one row are a bijection onto
the phase. Where a phase straddles a row the pattern is two-way, as it is at
:data:`LANE_PAIR`, and there are half as many accesses to be conflicted.

Not every geometry takes it: :func:`_rot_run` falls back to :data:`LANE_PAIR` where
the wider run costs no step, since halving the work items below ``threads`` idles
warps instead of shortening the pass. Seven of the nine call sites take it at every
standard shape; the two that do not are ``increment_passing_fwd``'s residue passes,
which stage one row.

Measured at ``acceptance``, one step, against the same file at :data:`LANE_PAIR`:
tree warp-instructions 629,040,144 to 615,466,128, ``chunk_scan_fwd`` 222 registers
to 216 with no spill either side, ``chunk_input_bwd`` 4,644,864 local sectors to
3,322,368 at 255 registers both sides. Shared load wavefronts fall 3.81% tree-wide;
store wavefronts move 0.06% and load bank conflicts 0.73%, so the store side is an
instruction win and not a wavefront one. Every output is bitwise unchanged."""


def paired(tile: cute.Tensor, run: cutlass.Constexpr = LANE_PAIR) -> cute.Tensor:
    """View a row-major tile in units of ``run`` adjacent elements.

    Args:
        tile: ``(rows, pitch)`` shared tile, unit stride on the columns and a pitch
            from :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`.
        run: Elements per access. Compile-time, :data:`LANE_PAIR` by default.

    Returns:
        The retiled view. Element ``(None, (r, k))`` is elements ``run * k`` through
        ``run * k + run - 1`` of row ``r``, statically shaped, which is what lets
        :func:`cutlass.cute.autovec_copy` pick one access rather than ``run``.

    Invariants:
        Every access is aligned to ``run`` elements. The pitch is a whole number of
        16-byte segments, so a row starts on a 16-byte boundary, and the column
        offset is a multiple of ``run``. The claim is restated on the iterator
        because a tile arriving as a parameter reports one element whatever its
        allocation asked for, and ``autovec_copy`` caps the access at the claim.
    """
    base = tile.iterator.align(run * (tile.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, tile.layout), (1, run))


def _paired_row(row: cute.Tensor, run: cutlass.Constexpr = LANE_PAIR) -> cute.Tensor:
    """View one contiguous row in units of ``run`` adjacent elements.

    Args:
        row: ``(3N,)`` unit-stride view of one row of a global tensor.
        run: Elements per access. Compile-time, :data:`LANE_PAIR` by default.

    Returns:
        The retiled view. Element ``(None, k)`` is elements ``run * k`` through
        ``run * k + run - 1``.

    Invariants:
        ``3N`` is a multiple of 48, so a row offset is a multiple of 48 elements and
        every access is aligned to ``run`` for every ``run`` dividing 48. The claim
        is restated for the reason given in :func:`paired`.
    """
    base = row.iterator.align(run * (row.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, row.layout), (run,))


@cute.jit
def _load_pair(
    row: cute.Tensor, frag: cute.Tensor, slot: cutlass.Constexpr, pair: cutlass.Int32
) -> None:
    """Read one run's ``3 * run`` elements as three accesses.

    Args:
        row: A row from :func:`_paired_row`, retiled at the same ``run`` as ``frag``.
        frag: ``(slots, run)`` fragment of the row's element type. Its second extent
            is the run width; the retiled row supplies it.
        pair: Which run of 3-vectors. Its accesses are ``3 * pair`` to
            ``3 * pair + 2``.
        slot: First fragment row to fill, three from there. Compile-time: one slot
            per outstanding load, since a reused slot would order the loads.
    """
    for k in cutlass.range_constexpr(3):
        cute.autovec_copy(row[(None, 3 * pair + k)], frag[(slot + k, None)])


@cute.jit
def _store_run(
    words: cute.Tensor,
    frag: cute.Tensor,
    row: cutlass.Int32,
    col: cutlass.Int32,
    vals: tuple[Scalar, ...],
) -> None:
    """Store one run of adjacent values as one access.

    The run width is ``len(vals)``, which the trace knows, so one body serves every
    width the callers use.

    The convert here reaches no instruction of its own, and the byte permute keyed to
    this line is not it. Measured on sm_86 at the acceptance shape, the bf16 traffic
    of ``chunk_vector_bwd`` is four disjoint SASS classes over 314,818,560
    warp-instructions:

    ================== ========== ====== =========================================
    class              inst       share  role
    ================== ========== ====== =========================================
    SHF.L.U32 by 0x10  31,334,400 9.95%  widen, one instruction per element
    PRMT 0x7632        14,376,960 4.57%  unpack the high half of a packed pair
    F2FP.BF16.PACK_AB   4,239,360 1.35%  narrow, two elements at once
    PRMT 0x5410         2,949,120 0.94%  pack two halves for a shared store
    ================== ========== ====== =========================================

    2,211,840 of the ``PRMT 0x7632`` and 1,105,920 of the ``F2FP`` key to this line.
    Carrying :func:`stage_shifted` and :func:`stage_state` at the destination tile's
    own element type, so that this line converts nothing in either, leaves all four
    counts unchanged to the instruction -- not one of the 55 convert-class
    ``(line, opcode)`` pairs moves -- and costs 92,160 ``IADD3``. Adding a round trip
    to :func:`stage_state`, which had none, likewise moves no per-pipe count on any
    kernel of the tree, so ``bfloat16 -> float32 -> bfloat16`` around a pass with no
    arithmetic in it is folded before ptxas. What is left is forced by the packed
    tile: an element crossing between a packed 16-bit tile and a float32 register
    costs one instruction per direction, which is the floor. Half the ``PRMT`` is
    ``chunk_vector_bwd``'s own operand widening on two scalar dot products over
    shared bfloat16 tiles. Reproduce with ``sm__inst_executed.sum`` and the five
    per-pipe counters over ``scripts/perf/profile_target.py --op so3ssd --shape
    acceptance``; every output is bitwise unchanged at all five shapes either way.

    Args:
        words: A tile from :func:`paired`, retiled at ``len(vals)``.
        frag: ``(1, len(vals))`` fragment of the tile's element type, built once by
            the caller: a fragment per step is an allocation per step.
        row: Row of the tile.
        col: Column in units of the run.
        vals: The run's values in element order, float32. Narrowed here, once.
    """
    elem = words.element_type
    for j in cutlass.range_constexpr(len(vals)):
        frag[0, j] = narrow(vals[j], elem)
    cute.autovec_copy(frag, words[(None, (row, col))])


@cute.jit
def store_pair(
    words: cute.Tensor,
    frag: cute.Tensor,
    row: cutlass.Int32,
    col: cutlass.Int32,
    vals: tuple[Scalar, ...],
) -> None:
    """Store one run's ``3 * run`` values as three accesses.

    The run width is ``len(vals) // 3``, which the trace knows, so one body serves
    every width the callers use.

    Args:
        words: A tile from :func:`paired`, retiled at ``len(vals) // 3``.
        frag: ``(1, len(vals) // 3)`` fragment of the tile's element type, built once
            by the caller: a fragment per step is an allocation per step.
        row: Row of the tile.
        col: First column of the run, ``3 * m`` for run ``m``.
        vals: The values in element order, float32. Narrowed here, once.
    """
    run = len(vals) // 3
    for k in cutlass.range_constexpr(3):
        _store_run(words, frag, row, col + k, vals[run * k : run * (k + 1)])


@cute.jit
def stage_trans(
    gtrans: cute.Tensor,
    strans: cute.Tensor,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    """Stage one chunk of ``trans`` into shared memory, transposed.

    The transition half of :func:`stage_chunk`, for a kernel that reads no tap.
    Staging ``K`` to reach the same tile would add its whole extent to a pass that
    has no tap matrix to build.

    Args:
        gtrans: ``(T, 4)`` float32 view of ``trans`` for one ``(b, h)``.
        strans: ``(4, L)`` float32 shared tile, written.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist, at least one. Tokens at or past
            this index are staged as zeros, which is the identity transition. The
            clamp below reads ``valid - 1``, so zero would read the token before
            the chunk; a chunk with no valid token is not launched.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.

    Invariants:
        The row is read at :func:`_segment_run` width, not one component at a time.
        A lane's four components span 16 bytes and 32 lanes stride 16 bytes, so a
        scalar read touches every sector of the warp's 512-byte span and uses four
        of every sixteen bytes: 64 sector-touches for 512 bytes against 16 for one
        wide access. The claim :func:`_wide_row` makes holds because ``trans``
        reaches every launcher through :func:`slinoss._guard.check_layout`, so the
        row sits at a static multiple of its own 16 bytes off a contiguous base.
        The shared side stays scalar: it is transposed and coalesces either way.

        Measured on sm_86 at the acceptance shape,
        ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld`` falls 96 per chunk per block
        here and 480 at a :func:`stage_chunk` site, 14,596,395 over the step, with
        every output bitwise unchanged at all five shapes. The step is 23.3 us
        shorter for it, which is 1.6 us per million sectors: a sixteenth of the rate
        at which deleting one whole staging pass converted, so a sector count on its
        own does not price a staging arm. What that deletion also took was a barrier
        wait, and this takes none.
    """
    zero = cutlass.Float32(0.0)
    elem = gtrans.element_type
    run = _segment_run(elem.width // 8, 4)
    runs = 4 // run
    steps = (chunk + threads - 1) // threads
    held = cute.make_fragment((steps, runs, run), elem)
    for step in cutlass.range_constexpr(steps):
        token = tid + step * threads
        if token < chunk:
            inside = token < valid
            # The clamp keeps the read in bounds when the chunk overhangs the
            # sequence; the select then replaces it with the pad value.
            pos = t0 + cutlass.min(token, valid - 1)
            wide = _wide_row(gtrans[pos, None], run)
            for k in cutlass.range_constexpr(runs):
                cute.autovec_copy(wide[(None, k)], held[(step, k, None)])
            for j in cutlass.range_constexpr(4):
                got = held[(step, j // run, j % run)]
                strans[j, token] = select(inside, got, zero)


@cute.jit
def stage_chunk(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    strans: cute.Tensor,
    stap: cute.Tensor,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    """Stage one chunk of ``trans`` and ``K`` into shared memory, transposed.

    The transition half is :func:`stage_trans`; only the tap half is here. Both
    loops are unrolled straight-line code over the same token index, so the two
    groups of loads issue together.

    Args:
        gtrans: ``(T, 4)`` float32 view of ``trans`` for one ``(b, h)``.
        gtap: ``(T, 2, 4)`` float32 view of ``K`` for one ``(b, h)``.
        strans: ``(4, L)`` float32 shared tile, written.
        stap: ``(8, L)`` float32 shared tile, written. Component ``4*tap + j``.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist, at least one. Tokens at or past
            this index are staged as zeros.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.

    Invariants:
        One access per tap, for the reason in :func:`stage_trans`. A tap row is 16
        bytes of a 32-byte token, so the scalar form was the worse of the two: eight
        reads over the warp's 1,024-byte span, each touching all 32 of its sectors,
        256 sector-touches for 1,024 bytes against 64 for two wide accesses.
    """
    stage_trans(gtrans, strans, t0, valid, tid, threads, chunk)
    zero = cutlass.Float32(0.0)
    elem = gtap.element_type
    run = _segment_run(elem.width // 8, 4)
    runs = 4 // run
    steps = (chunk + threads - 1) // threads
    held = cute.make_fragment((steps, 2, runs, run), elem)
    for step in cutlass.range_constexpr(steps):
        token = tid + step * threads
        if token < chunk:
            inside = token < valid
            pos = t0 + cutlass.min(token, valid - 1)
            for tap in cutlass.range_constexpr(2):
                wide = _wide_row(gtap[pos, tap, None], run)
                for k in cutlass.range_constexpr(runs):
                    cute.autovec_copy(wide[(None, k)], held[(step, tap, k, None)])
            for tap in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(4):
                    got = held[(step, tap, j // run, j % run)]
                    stap[4 * tap + j, token] = select(inside, got, zero)


@cute.jit
def stage_weighted(
    gsrc: cute.Tensor,
    sdst: cute.Tensor,
    slp: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    width: cutlass.Constexpr,
) -> None:
    """Stage one chunk of a ``(T, P)`` tensor scaled by the chunk-local decay.

    Row ``r`` holds ``exp(2*lp_r) * gsrc[t0+r]``. Rows at or past ``valid`` are
    zeroed; a zero row contributes nothing to a contraction, so no consumer needs a
    predicate. Columns at or past ``width`` are the caller's business, through
    :func:`stage_pad`.

    The pass runs in groups of :data:`PREFETCH` steps, loads first, on a clamped
    index with a select afterwards, for the reason given in :func:`stage_rotated`.

    Args:
        gsrc: ``(B,H,T,P)`` operand-dtype source.
        sdst: Operand-dtype tile of at least ``chunk`` rows, written over ``width``
            columns.
        slp: ``(L,)`` float32 chunk-local log-scale prefix.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        width: Columns that carry data, ``P``. Compile-time.
    """
    src = gsrc.element_type
    zero = cutlass.Float32(0.0)
    total = chunk * width
    steps = -(-total // threads)
    exact = total % threads == 0

    for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
        count = min(PREFETCH, steps - group * PREFETCH)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * PREFETCH + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // width
            p = i - r * width
            # One clamp bounds the read; the select below drops the pad rows. r is
            # below the chunk by construction, so slp needs no clamp of its own.
            token = cutlass.min(r, valid - 1)
            got = widen(gsrc[bidx, hidx, t0 + token, p], src)
            held.append((r, p, r < valid, got))

        for step in cutlass.range_constexpr(count):
            r, p, keep, got = held[step]
            # I3: the weight is one exponential of the prefix, never a ratio of two.
            # Zeroing the float32 input rather than the product keeps the pad rows
            # exactly zero whatever the prefix holds there.
            out = narrow(select(keep, got, zero) * decay(slp[r]), src)
            if cutlass.const_expr(exact):
                sdst[r, p] = out
            else:
                if tid + (group * PREFETCH + step) * threads < total:
                    sdst[r, p] = out


def _segment_run(itemsize: int, width: int) -> int:
    """Elements per access: the widest whole run of ``width`` inside one segment.

    Args:
        itemsize: Bytes per element, the same at both ends of the copy.
        width: Elements a row carries.

    Returns:
        A power of two at most ``SMEM_SEGMENT // itemsize``, dividing ``width``. One
        where ``width`` is odd, which is the scalar access the run generalizes.
    """
    run = SMEM_SEGMENT // itemsize
    while run > 1 and width % run != 0:
        run //= 2
    return run


def _flat_run(itemsize: int, threads: int, rows: int, width: int) -> int:
    """The run width a flat ``(rows, width)`` copy uses at one geometry.

    :func:`_segment_run` bounds a run to one segment; this narrows it to the point
    where narrowing starts costing steps, which is :func:`_rot_run`'s second condition
    without its first. A flat copy applies nothing across a run, so unlike
    :func:`stage_rotated` there is no run-wide transform for a ragged tail to repeat,
    and the only reason to prefer the narrow width at an equal step count is that its
    thread-to-element map leaves no idle lanes.

    Args:
        itemsize: Bytes per element at the wider of the two ends. A run must sit inside
            one segment at both, and only the wider end binds.
        threads: Block width.
        rows: Rows the pass fills.
        width: Elements a row carries.

    Returns:
        The narrowest run attaining the fewest steps, a divisor of ``width``, at least
        :data:`LANE_PAIR` wherever ``width`` is even.
    """
    run = _segment_run(itemsize, width)
    steps = -(-(rows * (width // run)) // threads)
    while run > LANE_PAIR:
        half = run // 2
        if -(-(rows * (width // half)) // threads) > steps:
            break
        run = half
    return run


def _wide_row(row: cute.Tensor, run: cutlass.Constexpr) -> cute.Tensor:
    """View one contiguous row in units of ``run`` adjacent elements.

    :func:`_paired_row` at a run the caller sizes, and over a shared row as well as
    a global one.

    Args:
        row: Unit-stride view of one row, global or shared.
        run: Elements per access, from :func:`_segment_run`.

    Returns:
        The retiled view. Element ``(None, k)`` is elements ``run * k`` through
        ``run * k + run - 1``.

    Invariants:
        Both ends are aligned to ``run`` elements, so the claim on the iterator
        holds; it is restated for the reason given in :func:`paired`. A shared row
        starts at a whole :data:`slinoss.ops.so3ssd.cute.mma.SMEM_SEGMENT` because
        :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch` returns an odd multiple of
        one, and a global row starts at a multiple of ``run`` because ``run`` divides
        the row extent.
    """
    base = row.iterator.align(run * (row.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, row.layout), (run,))


def _async_atom(elem, run: cutlass.Constexpr) -> cute.CopyAtom:
    """The ``cp.async`` atom that carries one ``run`` of ``elem`` global to shared.

    ``cp.async`` moves a run global to shared without staging it in registers and
    without an intervening dependence, so a flat pass costs one instruction per run
    where the register form costs a load, a dependent store, the registers that hold
    the run between them, and the round trip through float32 that the store's select
    and narrow impose on every element.

    ``LoadCacheMode.ALWAYS`` is ``cp.async.ca``, the only mode legal at every run
    width: ``LoadCacheMode.GLOBAL``, ``cp.async.cg``, is refused below 16 bytes by both
    the ``static_assert`` in CUTLASS's ``copy_sm80.hpp`` and the DSL's own check. It is
    also the faster of the two at the widths where both are legal, and the choice is a
    measurement rather than a derivation: a staged row is read once by the block that
    stages it, so L1 residency ought to be worth nothing to it, and the mode that
    bypasses L1 is nonetheless the slower one.

    Args:
        elem: Element type, the same at both ends. ``cp.async`` cannot convert, so a
            pass whose ends differ in width has no async form.
        run: Elements per copy. Compile-time. ``run * elem.width`` must be 32, 64 or
            128 bits, which is what :func:`_segment_run` and :func:`_flat_run` give.

    Returns:
        The atom, for :func:`cutlass.cute.copy` with a rank-1 global source run and a
        rank-1 shared destination run.

    Invariants:
        The caller owns the completion. An issued copy is visible to the issuing thread
        only after :func:`cutlass.cute.arch.cp_async_wait_group` retires its group, and
        to the block only after a barrier past that wait. A pass that issues here and
        is read without both is a stale tile, not a race.
    """
    return cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
        elem,
        num_bits_per_copy=run * elem.width,
    )


@cute.jit
def stage_raw(
    gsrc: cute.Tensor,
    sdst: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    width: cutlass.Constexpr,
) -> None:
    """Stage one chunk of a ``(T, P)`` tensor unweighted, one segment an access.

    The load half of :func:`stage_weighted`, with :func:`weight_rows` as the other:
    row ``r`` holds ``gsrc[t0+r]`` at the source dtype, and the decay and the
    pad-row zeros arrive afterwards. Nothing here reads the chunk-local prefix, so
    the pass can be issued ahead of the scan that fills it and its latency covered
    by that scan; the fused pass orders the whole read behind the scan for no
    dependence.

    A step carries one whole :data:`slinoss.ops.so3ssd.cute.mma.SMEM_SEGMENT` of
    adjacent elements, one access on each side, where the fused pass carries one
    element. At ``L=64 P=64`` and 256 threads that is 2 global loads and 2 shared
    stores a thread a chunk against 16 of each, and the source elements a thread
    holds live are ``PREFETCH``-bounded as everywhere else in this file.

    A 2-byte store puts two threads in one 4-byte bank word and is conflicted
    whatever the address; the segment store cannot be, and a phase of eight threads
    covers eight distinct segments modulo eight wherever a row is a whole multiple
    of eight segments.

    Args:
        gsrc: ``(B,H,T,P)`` operand-dtype source, contiguous. ``width`` is its last
            extent, so a row is one run of ``width`` elements.
        sdst: Tile of at least ``chunk`` rows at ``gsrc``'s element type, written
            over ``width`` columns. Columns at or past ``width`` are the caller's
            business, through :func:`stage_pad`.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist. Rows at or past it hold row
            ``valid - 1``'s data until :func:`weight_rows` zeroes them.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        width: Columns that carry data, ``P``. Compile-time.
    """
    src = gsrc.element_type
    run = _segment_run(src.width // 8, width)
    runs = width // run
    total = chunk * runs
    steps = -(-total // threads)
    exact = total % threads == 0

    loads = cute.make_fragment((min(PREFETCH, steps), run), src)

    for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
        count = min(PREFETCH, steps - group * PREFETCH)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * PREFETCH + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // runs
            col = i - r * runs
            # One clamp bounds the read; weight_rows drops the pad rows, so no
            # select is needed at the source dtype here.
            token = cutlass.min(r, valid - 1)
            cute.autovec_copy(
                _wide_row(gsrc[bidx, hidx, t0 + token, None], run)[(None, col)],
                loads[(step, None)],
            )
            held.append((r, col))

        for step in cutlass.range_constexpr(count):
            r, col = held[step]
            dst = _wide_row(sdst[r, None], run)[(None, col)]
            if cutlass.const_expr(exact):
                cute.autovec_copy(loads[(step, None)], dst)
            else:
                if tid + (group * PREFETCH + step) * threads < total:
                    cute.autovec_copy(loads[(step, None)], dst)


@cute.jit
def weight_rows(
    sdst: cute.Tensor,
    slp: cute.Tensor,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    width: cutlass.Constexpr,
) -> None:
    """Scale a staged chunk in place by the chunk-local decay, one segment an access.

    The scale half of :func:`stage_weighted`, over what :func:`stage_raw` left in
    shared memory. The arithmetic is that pass's term for term: one widen of the
    element, one select against the pad row, one multiply by
    :func:`slinoss._cute.decay` of the row's prefix, one narrow. A round trip
    through shared memory at the tile's own element type is exact, so the result is
    bit-identical to the fused pass.

    The decay is one evaluation an access rather than one an element, because a run
    lies within one row: at ``L=64 P=64`` and 256 threads, 2 exponentials a thread a
    chunk against 16.

    The thread-to-element map is :func:`stage_raw`'s, so a thread reads back the
    elements it wrote and the two passes are coherent with or without a barrier
    between them.

    Args:
        sdst: The tile :func:`stage_raw` wrote, read and written over ``width``
            columns.
        slp: ``(L,)`` float32 chunk-local log-scale prefix.
        valid: Tokens of the chunk that exist. Rows at or past it are zeroed.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        width: Columns that carry data, ``P``. Compile-time.

    Invariants:
        ``exp(2*lp) <= 1`` by I1 and comes from the prefix itself, never from a
        ratio of two exponentials (I3). Zeroing the float32 input rather than the
        product keeps the pad rows exactly zero whatever the prefix holds there.
    """
    elem = sdst.element_type
    zero = cutlass.Float32(0.0)
    run = _segment_run(elem.width // 8, width)
    runs = width // run
    total = chunk * runs
    steps = -(-total // threads)
    exact = total % threads == 0

    vals = cute.make_fragment((min(PREFETCH, steps), run), elem)

    for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
        count = min(PREFETCH, steps - group * PREFETCH)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * PREFETCH + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // runs
            col = i - r * runs
            cute.autovec_copy(
                _wide_row(sdst[r, None], run)[(None, col)], vals[(step, None)]
            )
            # The prefix read joins the load phase: its latency is the run's.
            held.append((r, col, r < valid, decay(slp[r])))

        for step in cutlass.range_constexpr(count):
            r, col, keep, factor = held[step]
            for j in cutlass.range_constexpr(run):
                got = select(keep, widen(vals[step, j], elem), zero)
                vals[step, j] = narrow(got * factor, elem)
            dst = _wide_row(sdst[r, None], run)[(None, col)]
            if cutlass.const_expr(exact):
                cute.autovec_copy(vals[(step, None)], dst)
            else:
                if tid + (group * PREFETCH + step) * threads < total:
                    cute.autovec_copy(vals[(step, None)], dst)


@cute.jit
def stage_score(
    gsrc: cute.Tensor,
    sdst: cute.Tensor,
    nbase: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    span: cutlass.Constexpr,
    pad: cutlass.Constexpr,
    asynch: cutlass.Constexpr = False,
) -> None:
    """Stage a column block of an ``(L, L)`` record into a tile, one segment an access.

    The record is dense in its second mode and the tile is dense in its columns, so
    a run of adjacent source tokens is one access at either end and the pass is one
    load and one store a thread a step. No transform: the producer wrote the value
    the consumer reads, so a round trip is bits.

    Args:
        gsrc: ``(L, L)`` slice of the record at this block's ``(batch, head,
            chunk)``, the tile's element type. Row is the target token, column the
            source token.
        sdst: Tile of at least ``rows + pad`` rows at ``gsrc``'s element type,
            written over ``span`` columns. Columns at or past ``span`` are the
            caller's business.
        nbase: First source token of the block. A multiple of ``span``.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        rows: Target tokens the record carries, ``L``. Compile-time.
        span: Source tokens of the block. Compile-time.
        pad: Rows past ``rows`` to zero, the M mode's rounding. Compile-time, and
            zero at every shipped shape, where the trace drops the fill.
        asynch: Issue the pass as ``cp.async`` and commit one group, leaving the wait
            and the publishing barrier to the caller. Compile-time. The pad fill is
            not part of the group: it writes shared from a register constant, has no
            global source, and runs ahead of the issues so the group closes last.

    Invariants:
        ``L`` is a multiple of 16 and ``span`` divides it, so a record row starts on
        a 16-byte segment and so does the column offset: the alignment claim
        :func:`_wide_row` restates holds at both ends for every run dividing 8. The
        pad rows reach the K extent of the consumer's transposed view, so a shape
        with ``mma_rows(L) > L`` needs them zeroed and not left stale.
    """
    elem = sdst.element_type
    zero = cutlass.Float32(0.0)
    run = _segment_run(elem.width // 8, span)
    runs = span // run
    total = rows * runs
    steps = -(-total // threads)
    exact = total % threads == 0
    # The column offset in units of the run. Both terms are exact: ``span`` is a
    # multiple of the run and ``nbase`` a multiple of ``span``.
    coff = nbase // run

    # Ahead of the copy, so an asynchronous pass commits its group last and nothing
    # after the commit touches the destination. The fill's rows are past the copy's,
    # so the order between them is free.
    if cutlass.const_expr(pad != 0):
        zeros = cute.make_fragment((1, run), elem)
        for j in cutlass.range_constexpr(run):
            zeros[0, j] = narrow(zero, elem)
        for step in cutlass.range_constexpr(-(-(pad * runs) // threads)):
            i = tid + step * threads
            if i < pad * runs:
                r = i // runs
                cute.autovec_copy(
                    zeros[(0, None)],
                    _wide_row(sdst[rows + r, None], run)[(None, i - r * runs)],
                )

    if cutlass.const_expr(asynch and gsrc.element_type.width == elem.width):
        atom = _async_atom(elem, run)
        for step in cutlass.range_constexpr(steps):
            i = tid + step * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // runs
            col = i - r * runs
            got = _wide_row(gsrc[r, None], run)[(None, coff + col)]
            put = _wide_row(sdst[r, None], run)[(None, col)]
            if cutlass.const_expr(exact):
                cute.copy(atom, got, put)
            else:
                # Past the extent the clamped destination is a run another step owns,
                # so the issue goes rather than a predicate zeroing that run.
                if tid + step * threads < total:
                    cute.copy(atom, got, put)
        cute.arch.cp_async_commit_group()
        return

    loads = cute.make_fragment((min(PREFETCH, steps), run), elem)

    for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
        count = min(PREFETCH, steps - group * PREFETCH)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * PREFETCH + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // runs
            col = i - r * runs
            cute.autovec_copy(
                _wide_row(gsrc[r, None], run)[(None, coff + col)],
                loads[(step, None)],
            )
            held.append((r, col))

        for step in cutlass.range_constexpr(count):
            r, col = held[step]
            dst = _wide_row(sdst[r, None], run)[(None, col)]
            if cutlass.const_expr(exact):
                cute.autovec_copy(loads[(step, None)], dst)
            else:
                if tid + (group * PREFETCH + step) * threads < total:
                    cute.autovec_copy(loads[(step, None)], dst)


def _table_duty(
    mats: cutlass.Constexpr, parts: cutlass.Constexpr
) -> tuple[tuple[int, ...], ...]:
    """Assign the table's slots to ``parts`` groups, the heavy slot alone.

    The groups are deliberately unbalanced, and two is the most a split can use.
    Slot costs are 155, 70 and 28 float instructions: the fused first slot is two
    tap matrices and two 3x3 products over two rotations, the second slot one of
    each, the third slot nothing beyond the rotation every group computes anyway.
    A third group therefore leaves the critical group at 155 where two groups
    already put it, and pays a rotation and a copy of the body for nothing.

    Args:
        mats: Slot count, 1, 2 or 3. Compile-time.
        parts: Groups to form, 1 or 2. Compile-time.

    Returns:
        One tuple of slot ids per group.
    """
    if mats == 1:
        return ((TABLE_AC_SOLE,),)
    third = (TABLE_AC,) if mats == 3 else ()
    if parts == 1:
        return ((TABLE_AP, TABLE_AN, *third),)
    # The assignment is the measured one. ``Ac`` with ``An`` instead would put 98
    # float instructions against 155 rather than 183 against 70, and is a separate
    # arm: which multiply-adds ptxas contracts moves with the group's store list.
    return ((TABLE_AP, *third), (TABLE_AN,))


def _table_slots(
    strans: cute.Tensor,
    stap: cute.Tensor,
    squat: cute.Tensor,
    stable: cute.Tensor,
    token: cutlass.Int32,
    jobs: cutlass.Constexpr,
    mats: cutlass.Constexpr,
    fused: cutlass.Constexpr,
    pitch: cutlass.Constexpr,
) -> None:
    """Compute and store one group's slots for one token.

    Every group recomputes ``Ac`` rather than reading a shared copy: the rotation
    is 28 float instructions, a shared round trip is at least two LSU operations,
    and an LSU operation on this device costs like a dozen FMAs. Undecorated, so
    the caller's dynamic group branch holds the whole body.

    Both products are computed before either is stored, and in slot order. That is
    not style: ``Ac`` must come out bitwise identical to the sole-slot build, and
    reordering the arithmetic against the stores is enough to move which
    multiply-add ptxas contracts.

    Args:
        strans: ``(4, L)`` staged transition parameters.
        stap: ``(8, L)`` staged tap parameters.
        squat: ``(4, L)`` quaternion prefix.
        stable: ``(mats, L, pitch)`` table, written at this group's slots.
        token: Chunk-local token, already bounded by ``L``.
        jobs: Slot ids this group writes, from :func:`_table_duty`. Compile-time.
            Unread at ``mats == 1``, whose sole slot admits no split.
        mats: Slot count. Compile-time.
        fused: Fuse the one-tap column into the first slot. Compile-time.
        pitch: The table's float32 pitch. Compile-time.
    """
    ac = mat3_transpose(
        rot_hom((squat[0, token], squat[1, token], squat[2, token], squat[3, token]))
    )
    if cutlass.const_expr(mats == 1):
        _store_mat(stable, TABLE_AC_SOLE, token, ac, pitch)
        return
    wvec = (strans[0, token], strans[1, token], strans[2, token])
    first = ac
    an = ac
    if cutlass.const_expr(TABLE_AP in jobs):
        first = mat3_mul(
            ac, tap_matrix((stap[0, token], stap[1, token], stap[2, token]), wvec)
        )
    if cutlass.const_expr(TABLE_AN in jobs):
        an = mat3_mul(
            ac, tap_matrix((stap[4, token], stap[5, token], stap[6, token]), wvec)
        )
    if cutlass.const_expr(TABLE_AP in jobs):
        if cutlass.const_expr(fused):
            # ``An_{t-1}`` is recomputed from the staged parameters of token
            # ``t-1`` rather than read back out of the slot this pass is still
            # writing. A read-back needs a barrier between the two halves of the
            # table, and barrier is the top stall of every kernel that builds one.
            # The recompute is one rotation and one tap matrix per token, O(L)
            # against an O(L*N) launch.
            #
            # Clamped rather than branched: token 0 reads its own row and the
            # factor below discards it.
            prev = cutlass.max(token - 1, 0)
            acp = mat3_transpose(
                rot_hom(
                    (squat[0, prev], squat[1, prev], squat[2, prev], squat[3, prev])
                )
            )
            anp = mat3_mul(
                acp,
                tap_matrix(
                    (stap[4, prev], stap[5, prev], stap[6, prev]),
                    (strans[0, prev], strans[1, prev], strans[2, prev]),
                ),
            )
            # I3: the raw per-step decay, never a ratio of prefixes, so
            # ``ls <= 0`` puts it in ``(0, 1]``.
            scale = select(token > 0, decay(strans[3, token]), cutlass.Float32(0.0))
            first = tuple(first[e] + scale * anp[e] for e in range(9))
        _store_mat(stable, TABLE_AP, token, first, pitch)
    if cutlass.const_expr(TABLE_AN in jobs):
        _store_mat(stable, TABLE_AN, token, an, pitch)
    if cutlass.const_expr(TABLE_AC in jobs):
        _store_mat(stable, TABLE_AC, token, ac, pitch)


@cute.jit
def build_table(
    strans: cute.Tensor,
    stap: cute.Tensor,
    squat: cute.Tensor,
    stable: cute.Tensor,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    mats: cutlass.Constexpr = 3,
    fused: cutlass.Constexpr = False,
    pitch: cutlass.Constexpr = 9,
) -> None:
    """Compose the ``(mats, L, pitch)`` transform table in shared memory.

    Args:
        strans: ``(4, L)`` float32 staged transition parameters, ``(w, ls)``
            component-major.
        stap: ``(8, L)`` float32 staged tap parameters. Unread at ``mats == 1``,
            which is the point of that slot count, so a caller there may pass any
            tile of the right dtype.
        squat: ``(4, L)`` float32 quaternion prefix, already renormalized.
        stable: ``(mats, L, pitch)`` float32, written. Slots are
            :data:`slinoss.ops.so3ssd.cute.common.TABLE_AP`, ``TABLE_AN`` and
            ``TABLE_AC``. At ``mats == 1`` the sole slot is ``TABLE_AC_SOLE``. At
            ``fused`` the first slot is ``TABLE_AFUSE``.
        tid: Thread index within the block.
        threads: Block width. Compile-time. At ``threads >= 2 * chunk`` the build
            widens: one thread group per slot rather than one thread per token.
        chunk: ``L``. Compile-time. A multiple of 32 where the build widens, which
            is what keeps the group index warp-uniform.
        mats: Slots to write, 1, 2 or 3. Compile-time, and must match the ``mats``
            the tile was allocated at. One writes ``Ac`` alone and computes neither
            tap matrix, so it also reads neither ``stap`` nor ``strans``.
        fused: Write the one-tap column ``Afuse_t = Ap_t + exp(2*ls_t) An_{t-1}``
            into the first slot instead of ``Ap_t``. Compile-time. Ignored at
            ``mats == 1``, which builds no tap matrix. The second slot still holds
            ``An_t``, which the diagonal residue needs.
        pitch: The table's float32 pitch, as passed to
            :func:`slinoss.ops.so3ssd.cute.common.table_tile`. Compile-time.
            :data:`TABLE_PITCH` buys the vector-width store.

    Invariants:
        ``Afuse_0 == Ap_0``: the previous chunk's ``An_{L-1}`` lives in the previous
        chunk's frame and its contribution arrives through the carried state, so
        injecting it here is wrong rather than redundant.

        A pad token's row is the one an identity rotation and zero taps produce, so
        ``Afuse`` past ``valid`` is zero. Slot ``valid`` itself is not: ``ls`` stages
        as zero there, making the factor one, and the term is the last real token's
        ``An``. That row is load-bearing under fusion and must not be predicated
        away.
    """
    parts = 2 if mats >= 2 and threads >= 2 * chunk else 1
    duty = _table_duty(mats, parts)
    if cutlass.const_expr(parts == 1):
        for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
            token = tid + step * threads
            if token < chunk:
                _table_slots(
                    strans, stap, squat, stable, token, duty[0], mats, fused, pitch
                )
    else:
        # One thread group per slot rather than one thread per token. The phase is
        # bounded by the shared load of the quaternion prefix, not by the arithmetic
        # over it, and ``chunk`` threads out of ``threads`` leave the scheduler
        # nothing to switch to when that load misses. Splitting by slot puts
        # ``parts`` loads in flight per token. ``group`` is warp-uniform because
        # ``chunk`` is a multiple of the warp, so the branch below is a jump rather
        # than a divergence, and ``token`` stays consecutive within a warp, so no
        # store pattern moves.
        total = chunk * parts
        for step in cutlass.range_constexpr((total + threads - 1) // threads):
            idx = tid + step * threads
            # One guard, not two: past ``total`` the group index runs past ``parts``
            # and matches no group below.
            group = idx // chunk
            token = idx % chunk
            for one in cutlass.range_constexpr(parts):
                if group == one:
                    _table_slots(
                        strans,
                        stap,
                        squat,
                        stable,
                        token,
                        duty[one],
                        mats,
                        fused,
                        pitch,
                    )


@cute.jit
def stage_shifted(
    gu: cute.Tensor,
    guprev: cute.Tensor,
    su: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    lbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    width: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    asynch: cutlass.Constexpr = False,
) -> None:
    """Stage a run of tokens of a ``(T, P)`` tensor into a tile, one row behind.

    Row ``r`` holds token ``t0 + lbase + r - 1``, so one staging pass serves both
    taps: the previous tap reads rows ``0..span-1`` and the current tap reads rows
    ``1..span`` off two views that differ by one row of pitch. The shift is global,
    not per chunk, so row 0 of the first run of a chunk crosses the chunk boundary
    and, at the first chunk, reaches the streaming ``u_prev``.

    Rows past the sequence are zeroed. A zero there multiplies a forcing row that
    is itself zero, so no store is skipped. Columns at or past ``width`` are not
    touched: they are the caller's business, through :func:`stage_pad`, because
    they never change and a per-slice restage would rewrite the same zeros.

    The pass runs in groups of ``PREFETCH // run`` steps, loads first, on clamped
    indices with a select afterwards, for the reason given in :func:`stage_rotated`.
    This is the longest staging pass in the operator: ``(span + 1) * width`` elements
    at :func:`_flat_run` elements per thread per step, which is one read and one write
    per step at either end.

    Args:
        gu: ``(B,H,T,P)`` operand-dtype source.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        su: Operand-dtype tile of ``span + 1`` rows and ``lda`` pitch. Columns
            below ``width`` are written.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        lbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        span: Tokens of the run. Compile-time.
        width: Columns that carry data, ``P``. Compile-time.
        has_prev: Whether ``guprev`` was supplied. Compile-time.
        asynch: Issue the pass as ``cp.async`` and commit one group, leaving the wait
            and the publishing barrier to the caller. Compile-time, and a request
            rather than an instruction: the register form stands wherever the async
            one cannot express the pass, which is a carry select, having two sources
            against the copy's one, and a width change, ``cp.async`` having no
            conversion. The zero-fill past the sequence is the copy's own predicate
            there, performed by the hardware in place of a select on a loaded value.

    Invariants:
        Nothing is applied across a run: the shift is a row index and the zero-fill a
        per-element select, so the run width is free and :func:`_flat_run` takes it,
        as it is in :func:`stage_raw` over the same geometry. The 3-vector store
        :data:`LANE_PAIR` argues from and the twelve-component run :data:`ROT_RUN`
        argues from are both absent here.

        ``width`` is ``P`` or a lane tile of ``3N``, so it is a multiple of 8:
        :func:`slinoss.ops.so3ssd.cute.guard.check_rows` holds ``P`` to a multiple of
        16 and a lane tile is a multiple of 48. A global row is therefore a whole
        number of 16-byte segments, and ``su`` is pitched by
        :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`, an odd multiple of one, so
        both ends of a run access are aligned to it at every width the helper returns.

        The run predicate is reachable where the element predicate was not: widening
        divides the extent, so an extent that was a multiple of the block width need
        not stay one.
    """
    src = gu.element_type
    zero = cutlass.Float32(0.0)
    wide = max(src.width, su.element_type.width) // 8
    run = _flat_run(wide, threads, span + 1, width)
    runs = width // run
    total = (span + 1) * runs
    steps = -(-total // threads)
    exact = total % threads == 0
    depth = max(1, PREFETCH // run)

    if cutlass.const_expr(
        asynch and not has_prev and src.width == su.element_type.width
    ):
        atom = _async_atom(su.element_type, run)
        # One predicate element per copy element: the DSL sizes it by the atom's value
        # layout, not by the access. The run is uniform in it, the shift being a row.
        keeps = cute.make_fragment((run,), cutlass.Boolean)
        for step in cutlass.range_constexpr(steps):
            i = tid + step * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // runs
            p = i - r * runs
            token = lbase + r - 1
            gbase = t0 + cutlass.min(token, valid - 1)
            keep = (token < valid) & (gbase >= 0)
            for j in cutlass.range_constexpr(run):
                keeps[j] = keep
            got = _wide_row(gu[bidx, hidx, cutlass.max(gbase, 0), None], run)
            put = _wide_row(su[r, None], run)
            if cutlass.const_expr(exact):
                cute.copy(atom, got[(None, p)], put[(None, p)], pred=keeps)
            else:
                # Not the copy's predicate. A false predicate zeroes the destination,
                # and past the extent the clamped destination is a run another step
                # owns, so the issue itself has to go.
                if tid + step * threads < total:
                    cute.copy(atom, got[(None, p)], put[(None, p)], pred=keeps)
        cute.arch.cp_async_commit_group()
        return

    words = paired(su, run)
    frag = cute.make_fragment((1, run), su.element_type)
    loads = cute.make_fragment((depth, run), src)
    # The false arm aliases the current fragment, which is never read under it: every
    # use of the carry fragment sits under the same compile-time flag.
    prior = (
        cute.make_fragment((depth, run), src) if cutlass.const_expr(has_prev) else loads
    )

    for group in cutlass.range_constexpr(-(-steps // depth)):
        count = min(depth, steps - group * depth)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * depth + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // runs
            p = i - r * runs
            token = lbase + r - 1
            # token < valid implies t0 + token < seqlen, so clamping the token
            # bounds the read above; the row before the sequence bounds it below.
            gbase = t0 + cutlass.min(token, valid - 1)
            row = _paired_row(gu[bidx, hidx, cutlass.max(gbase, 0), None], run)
            cute.autovec_copy(row[(None, p)], loads[(step, None)])
            if cutlass.const_expr(has_prev):
                back = _paired_row(guprev[bidx, hidx, None], run)
                cute.autovec_copy(back[(None, p)], prior[(step, None)])
            keep = token < valid
            if cutlass.const_expr(not has_prev):
                keep = keep & (gbase >= 0)
            held.append((r, p, keep, gbase < 0))

        for step in cutlass.range_constexpr(count):
            r, p, keep, at_start = held[step]
            # The select is float32 because there is one select helper; the
            # operand round trip through float32 is exact at every operand width.
            got = tuple(widen(loads[step, j], src) for j in range(run))
            if cutlass.const_expr(has_prev):
                got = tuple(
                    select(at_start, widen(prior[step, j], src), got[j])
                    for j in range(run)
                )
            out = tuple(select(keep, value, zero) for value in got)
            if cutlass.const_expr(exact):
                _store_run(words, frag, r, p, out)
            else:
                if tid + (group * depth + step) * threads < total:
                    _store_run(words, frag, r, p, out)


@cute.jit
def stage_pad(
    dst: cute.Tensor,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    width: cutlass.Constexpr,
    pitch: cutlass.Constexpr,
) -> None:
    """Zero the columns of a tile between its data width and its row pitch.

    An MMA operand view whose N mode is the rounded extent reads columns past the
    data, so garbage there is read as an operand. Those columns never change, so
    they are zeroed once per block rather than on every restage.

    No-op at compile time when the pitch carries no pad.

    Args:
        dst: Operand-dtype tile, written at columns ``width .. pitch - 1``.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        span: Rows to cover. Compile-time.
        width: First column to zero. Compile-time.
        pitch: Row pitch. Compile-time.
    """
    if cutlass.const_expr(pitch > width):
        pad = pitch - width
        zero = dst.element_type(0.0)
        for i in cutlass.range(tid, span * pad, threads):
            r = i // pad
            dst[r, width + i - r * pad] = zero


@cute.jit
def stage_state(
    gz: cute.Tensor,
    sz: cute.Tensor,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    asynch: cutlass.Constexpr = False,
) -> None:
    """Stage a chunk-start state into an operand tile.

    Both callers hand this a state the producing recurrence already stored at the
    operand dtype, so the conversion below is width-preserving there. It is kept
    because it is the only thing that would have to change to stage a float32 state,
    and because the chunk-start state is read by one contraction per chunk and never
    written, so there is no accumulation for a narrowing to compound through.

    ``(P, 3N)`` is one contiguous run and the loop walks it at the block stride, so a
    warp covers 64 contiguous elements per step and no index arithmetic survives. The
    steps run in groups of :data:`PREFETCH`, loads first, so the group's loads
    overlap: this is the largest single read in the operator and a serial
    step-by-step form pays one global latency per element per thread.

    A step carries :func:`_flat_run` adjacent columns, which is one read and one write
    of the run's width in place of one of each per element. The port issues one access
    per two cycles whatever its width, so the run divides the cost of the pass by its
    own length.

    Args:
        gz: ``(P, 3N)`` view of the chunk-start state for one
            ``(chunk, batch, head)``, at the operand dtype or float32.
        sz: Operand-dtype tile of at least ``width`` rows, written over ``dim``
            columns. The rest of the pitch is outside every view.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        width: Rows to fill, ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        asynch: Issue the pass as ``cp.async`` and commit one group, leaving the wait
            and the publishing barrier to the caller. Compile-time, and honoured only
            where the two element widths agree, ``cp.async`` having no conversion.
            There is no predicate: the pass has no clamp and no zero row.

    Invariants:
        ``dim`` is a multiple of 48 or a lane tile of one, so it is a multiple of 8,
        and ``sz`` is pitched by
        :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`: both ends of a run access are
        aligned to it at every width :func:`_flat_run` returns, the source bound by
        the wider of the two element types. Each element is narrowed on its own, so
        the run does not associate anything.
    """
    src = gz.element_type
    wide = max(src.width, sz.element_type.width) // 8
    run = _flat_run(wide, threads, width, dim)
    runs = dim // run
    total = width * runs
    steps = -(-total // threads)
    exact = total % threads == 0
    depth = max(1, PREFETCH // run)

    if cutlass.const_expr(asynch and src.width == sz.element_type.width):
        atom = _async_atom(sz.element_type, run)
        for step in cutlass.range_constexpr(steps):
            i = tid + step * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            p = i // runs
            col = i - p * runs
            got = _wide_row(gz[p, None], run)
            put = _wide_row(sz[p, None], run)
            if cutlass.const_expr(exact):
                cute.copy(atom, got[(None, col)], put[(None, col)])
            else:
                if tid + step * threads < total:
                    cute.copy(atom, got[(None, col)], put[(None, col)])
        cute.arch.cp_async_commit_group()
        return

    words = paired(sz, run)
    frag = cute.make_fragment((1, run), sz.element_type)
    loads = cute.make_fragment((depth, run), src)

    for group in cutlass.range_constexpr(-(-steps // depth)):
        count = min(depth, steps - group * depth)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * depth + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            p = i // runs
            col = i - p * runs
            cute.autovec_copy(
                _paired_row(gz[p, None], run)[(None, col)], loads[(step, None)]
            )
            held.append((p, col))

        for step in cutlass.range_constexpr(count):
            p, col = held[step]
            out = tuple(loads[step, j] for j in range(run))
            if cutlass.const_expr(exact):
                _store_run(words, frag, p, col, out)
            else:
                if tid + (group * depth + step) * threads < total:
                    _store_run(words, frag, p, col, out)


def _matrix_pass(
    threads: cutlass.Constexpr, rows: cutlass.Constexpr, lanes: cutlass.Constexpr
) -> tuple[int, int, int, bool]:
    """The stride-loop geometry of one :func:`stage_matrix` pass.

    Args:
        threads: Block width.
        rows: Rows to fill, ``P``.
        lanes: ``N``.

    Returns:
        The runs a row holds, the runs the pass covers, the steps a thread takes, and
        whether the block width divides the run count.
    """
    pairs = lanes // LANE_PAIR
    total = rows * pairs
    return pairs, total, -(-total // threads), total % threads == 0


def matrix_frag(
    gv: cute.Tensor,
    threads: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> cute.Tensor:
    """The register file one whole :func:`stage_matrix` pass reads into.

    Allocated by the caller so the read and the transform can sit on either side of a
    barrier. Three accesses a step, one per component of the run's 3-vectors.

    Args:
        gv: The pass's source, for its element type.
        threads: Block width.
        rows: Rows to fill, ``P``.
        lanes: ``N``.

    Returns:
        The ``(3 * steps, LANE_PAIR)`` fragment, indexed ``[3 * step + k, j]`` by the
        step, the component and the element within the run.
    """
    _, _, steps, _ = _matrix_pass(threads, rows, lanes)
    return cute.make_fragment((3 * steps, LANE_PAIR), gv.element_type)


@cute.jit
def read_matrix(
    gv: cute.Tensor,
    loads: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    cidx: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Issue every global read of one :func:`stage_matrix` pass.

    The pass's own step is the only cover its runs get where it stands, and three
    accesses do not cover a round trip. Split out, the reads go as far above the
    transform as the caller has work to put between them; the transform reads nothing
    but the fragment. A barrier between the two is legal because the reads are global
    and target registers.

    Args:
        gv: ``(B,H,C,P,3N)`` source, as :func:`stage_matrix` takes it.
        loads: The fragment from :func:`matrix_frag`, at the same geometry.
        bidx: Batch index.
        hidx: Head index.
        cidx: Chunk index.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        rows: Rows to fill, ``P``. Compile-time.
        lanes: ``N``. Compile-time.

    Invariants:
        The whole pass is in flight at once, so no step's slot is reused and the
        fragment is ``3 * steps`` rows rather than the ``3 * PREFETCH // LANE_PAIR``
        :func:`stage_matrix` bounds itself to. A caller that hoists is choosing to pay
        the deeper prefetch in registers.
    """
    pairs, total, steps, exact = _matrix_pass(threads, rows, lanes)
    for step in cutlass.range_constexpr(steps):
        p, m = _matrix_step(tid, step, threads, pairs, total, exact)
        _load_pair(_paired_row(gv[bidx, hidx, cidx, p, None]), loads, 3 * step, m)


@cute.jit
def apply_matrix(
    loads: cute.Tensor,
    dst: cute.Tensor,
    sfp32: cute.Tensor,
    mat: Mat3,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    keep_fp32: cutlass.Constexpr,
) -> None:
    """Transform one :func:`read_matrix` fragment into an operand tile.

    The step's coordinate is recomputed rather than carried: it is a shift and a
    multiply-subtract off ``tid``, against two live registers a step held across
    whatever the caller put between the halves.

    Args:
        loads: The fragment :func:`read_matrix` filled.
        dst: Operand-dtype tile, as :func:`stage_matrix` takes it.
        sfp32: Float32 tile, written only when ``keep_fp32``.
        mat: The 3x3, row-major, entry ``3*r + c``.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        rows: Rows to fill, ``P``. Compile-time.
        lanes: ``N``. Compile-time.
        keep_fp32: Whether to write ``sfp32``. Compile-time.
    """
    pairs, total, steps, exact = _matrix_pass(threads, rows, lanes)
    words, frag, words32, frag32 = _matrix_dst(dst, sfp32, keep_fp32)
    for step in cutlass.range_constexpr(steps):
        p, m = _matrix_step(tid, step, threads, pairs, total, exact)
        _matrix_store(
            loads,
            3 * step,
            words,
            frag,
            words32,
            frag32,
            mat,
            p,
            m,
            exact or tid + step * threads < total,
            keep_fp32,
        )


def _matrix_step(
    tid: cutlass.Int32,
    step: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    pairs: cutlass.Constexpr,
    total: cutlass.Constexpr,
    exact: cutlass.Constexpr,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """The row and the run one step of a matrix pass covers.

    An out-of-range step is clamped to the last run rather than predicated: it reads
    a run that exists and its store is dropped instead.

    Args:
        tid: Thread index within the block.
        step: Which step. Compile-time.
        threads: Block width. Compile-time.
        pairs: Runs a row holds. Compile-time.
        total: Runs the pass covers. Compile-time.
        exact: Whether the block width divides ``total``. Compile-time.

    Returns:
        The row ``P`` index and the run index within it.
    """
    i = tid + step * threads
    if cutlass.const_expr(not exact):
        i = cutlass.min(i, total - 1)
    p = i // pairs
    return p, i - p * pairs


def _matrix_dst(
    dst: cute.Tensor, sfp32: cute.Tensor, keep_fp32: cutlass.Constexpr
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
    """The paired views and the staging fragments a matrix pass stores through.

    Args:
        dst: Operand-dtype tile.
        sfp32: Float32 tile, used only when ``keep_fp32``.
        keep_fp32: Whether the float32 copy is written. Compile-time.

    Returns:
        The operand view and its fragment, then the float32 view and its fragment.
        The float32 pair aliases the operand pair when ``keep_fp32`` is false: every
        use is under the same compile-time flag, so the alias is never reached and
        never allocated, and the name is bound on both paths.
    """
    words = paired(dst)
    frag = cute.make_fragment((1, LANE_PAIR), dst.element_type)
    if cutlass.const_expr(keep_fp32):
        words32 = paired(sfp32)
        frag32 = cute.make_fragment((1, LANE_PAIR), cutlass.Float32)
    else:
        words32 = words
        frag32 = frag
    return words, frag, words32, frag32


@cute.jit
def _matrix_store(
    loads: cute.Tensor,
    slot: cutlass.Constexpr,
    words: cute.Tensor,
    frag: cute.Tensor,
    words32: cute.Tensor,
    frag32: cute.Tensor,
    mat: Mat3,
    p: cutlass.Int32,
    m: cutlass.Int32,
    live: cutlass.Int32 | bool,
    keep_fp32: cutlass.Constexpr,
) -> None:
    """Apply the matrix to one step's two 3-vectors and store the pair.

    Args:
        loads: The read fragment.
        slot: First fragment row of this step, three from there. Compile-time.
        words: The operand view from :func:`_matrix_dst`.
        frag: Its staging fragment.
        words32: The float32 view, read only when ``keep_fp32``.
        frag32: Its staging fragment.
        mat: The 3x3, row-major.
        p: Row of the tile.
        m: Run within the row.
        live: Whether this step's store is in range. ``True`` when the block width
            divides the run count, and then the guard is traced away.
        keep_fp32: Whether to write ``words32``. Compile-time.
    """
    src = loads.element_type
    got = tuple(
        widen(loads[slot + j // LANE_PAIR, j % LANE_PAIR], src)
        for j in range(3 * LANE_PAIR)
    )
    out = mat3_matvec(mat, (got[0], got[1], got[2])) + mat3_matvec(
        mat, (got[3], got[4], got[5])
    )
    col = 3 * m
    if cutlass.const_expr(live is True):
        store_pair(words, frag, p, col, out)
        if cutlass.const_expr(keep_fp32):
            store_pair(words32, frag32, p, col, out)
    else:
        if live:
            store_pair(words, frag, p, col, out)
            if cutlass.const_expr(keep_fp32):
                store_pair(words32, frag32, p, col, out)


@cute.jit
def stage_matrix(
    gv: cute.Tensor,
    dst: cute.Tensor,
    sfp32: cute.Tensor,
    mat: Mat3,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    cidx: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    keep_fp32: cutlass.Constexpr,
) -> None:
    """Transform a ``(P, 3N)`` tensor by one 3x3 matrix into an operand tile.

    One matrix for the whole chunk, not a per-token table, so the caller reads the
    nine entries once as a broadcast and hands them over in registers. Every one of
    the ``N`` lanes takes the same matrix, so the stride loop owns :data:`LANE_PAIR`
    3-vectors per thread per step: three paired global reads, eighteen FMA, three
    paired shared-memory accesses per tile written.

    ``keep_fp32`` adds the float32 copy for a consumer that needs the untruncated
    result as well as the operand. That is one extra shared-memory access per pair
    against a second pass over the ``(P, 3N)`` global read.

    The pass runs in groups of ``PREFETCH // LANE_PAIR`` steps, loads first, for the
    reason given in :func:`stage_rotated`. A pass that fits one group is
    :func:`read_matrix` followed by :func:`apply_matrix`; a caller with work to put
    between the two halves calls them itself.

    Args:
        gv: ``(B,H,C,P,3N)`` source at the operand dtype or float32, widened on the
            read. Read at ``[bidx, hidx, cidx]``.
        dst: Operand-dtype tile of at least ``rows`` rows, written over ``3N``
            columns. The rest of the pitch is outside every view.
        sfp32: Float32 tile of the same extents, written only when ``keep_fp32``.
            Untouched otherwise, so a caller that needs one width passes any tile.
        mat: The 3x3, row-major, entry ``3*r + c``. Float32 by I4.
        bidx: Batch index.
        hidx: Head index.
        cidx: Chunk index.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        rows: Rows to fill, ``P``. Compile-time.
        lanes: ``N``. Compile-time.
        keep_fp32: Whether to write ``sfp32``. Compile-time.

    Invariants:
        ``lanes`` is even and both tiles are pitched by
        :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`, which is what :data:`LANE_PAIR`
        rests on.
    """
    src = gv.element_type
    pairs, total, steps, exact = _matrix_pass(threads, rows, lanes)

    # One group is the whole pass, so the split form emits the same block in the same
    # order and there is one body rather than two.
    if cutlass.const_expr(steps <= _MATRIX_DEPTH):
        loads = matrix_frag(gv, threads, rows, lanes)
        read_matrix(gv, loads, bidx, hidx, cidx, tid, threads, rows, lanes)
        apply_matrix(loads, dst, sfp32, mat, tid, threads, rows, lanes, keep_fp32)
        return

    words, frag, words32, frag32 = _matrix_dst(dst, sfp32, keep_fp32)
    loads = cute.make_fragment((3 * _MATRIX_DEPTH, LANE_PAIR), src)

    for group in cutlass.range_constexpr(-(-steps // _MATRIX_DEPTH)):
        count = min(_MATRIX_DEPTH, steps - group * _MATRIX_DEPTH)
        held = []
        for step in cutlass.range_constexpr(count):
            first = group * _MATRIX_DEPTH + step
            p, m = _matrix_step(tid, first, threads, pairs, total, exact)
            _load_pair(_paired_row(gv[bidx, hidx, cidx, p, None]), loads, 3 * step, m)
            held.append((p, m, first))

        for step in cutlass.range_constexpr(count):
            p, m, first = held[step]
            _matrix_store(
                loads,
                3 * step,
                words,
                frag,
                words32,
                frag32,
                mat,
                p,
                m,
                exact or tid + first * threads < total,
                keep_fp32,
            )


def _table_quads(stable: cute.Tensor, slot: cutlass.Constexpr, token: cutlass.Int32):
    """One table entry retiled into 16-byte segments, and the segment count.

    The alignment claim is restated on the sliced row rather than taken from the
    allocation, for the reason :func:`paired` gives: a tile arriving as a parameter
    reports one element whatever its allocation asked for, so the claim has to be
    made where the row is known.

    Args:
        stable: ``(mats, L, pitch)`` float32 table, ``pitch`` a multiple of
            :data:`TABLE_QUAD`.
        slot: Table slot. Compile-time.
        token: Chunk-local token, already bounded by ``L``.

    Returns:
        The retiled row, and the segments nine words span.
    """
    row = stable[slot, token, None]
    quads = cute.zipped_divide(
        cute.make_tensor(row.iterator.align(SMEM_SEGMENT), row.layout),
        (TABLE_QUAD,),
    )
    # Three segments cover nine words. A pitch wider than that pads further and is
    # neither written nor read for it.
    return quads, -(-9 // TABLE_QUAD)


def _store_mat(
    stable: cute.Tensor,
    slot: cutlass.Constexpr,
    token: cutlass.Int32,
    mat: tuple[Scalar, ...],
    pitch: cutlass.Constexpr = 9,
) -> None:
    """Write one 3x3 into a table slot.

    Three 16-byte shared stores at a pitch that is a whole number of segments, nine
    scalar ones otherwise. Conflict-free either way, by the argument
    :func:`mat_at` gives for the read: nine is coprime with the 32 banks, and a
    segment store is serviced in phases of eight threads whose segment index is a
    bijection on consecutive tokens.

    The padding words are written zero rather than left alone. A vector store covers
    the whole segment, so the alternative is to put an uninitialized register into
    shared memory, and the build is ``O(L)`` against an ``O(L*N)`` launch.

    Args:
        stable: ``(mats, L, pitch)`` float32 table, written at ``[slot, token]``.
        slot: Table slot. Compile-time.
        token: Chunk-local token, already bounded by ``L``.
        mat: The nine entries in row-major order.
        pitch: The table's float32 pitch. Compile-time.
    """
    # Undecorated, and a plain branch and ``range`` for the reason :func:`mat_at`
    # gives.
    if pitch % TABLE_QUAD != 0:
        for entry in range(9):
            stable[slot, token, entry] = mat[entry]
        return
    quads, span = _table_quads(stable, slot, token)
    frag = cute.make_fragment((span, TABLE_QUAD), cutlass.Float32)
    zero = cutlass.Float32(0.0)
    for i in range(span * TABLE_QUAD):
        frag[i // TABLE_QUAD, i % TABLE_QUAD] = mat[i] if i < 9 else zero
    for quad in range(span):
        cute.autovec_copy(frag[(quad, None)], quads[(None, quad)])


def mat_at(
    stable: cute.Tensor,
    slot: cutlass.Constexpr,
    token: cutlass.Int32,
    pitch: cutlass.Constexpr = 9,
) -> Mat3:
    """One transform-table entry as a 3x3, row-major.

    Three 16-byte shared loads at :data:`TABLE_PITCH`, nine scalar ones at the
    natural pitch of nine. The alignment claim is restated on the sliced iterator
    rather than taken from the allocation, for the reason :func:`paired` gives: a
    tile arriving as a parameter reports one element whatever its allocation asked
    for, so the claim has to be made where the row is known.

    Undecorated, so the slice and the retile are trace-time algebra and every
    fragment index is compile-time, which is what keeps the fragment in registers.
    A plain branch and a plain ``range``: the preprocessor rewrites ``const_expr``
    and ``range_constexpr`` only inside a decorated body, so either would reach the
    runtime stub here and raise. The pitch is compile-time, so the branch is taken
    during the trace regardless.

    Conflict-free at every map the callers use. A load at vector width is serviced in
    four phases of eight threads, so the unit is the segment and the modulus is 8
    rather than 32: eight threads on consecutive tokens take segment ``3 * token + q``
    modulo 8, a bijection, and threads sharing a token share an address and broadcast.

    Args:
        stable: ``(mats, L, pitch)`` float32 table from
            :func:`slinoss.ops.so3ssd.cute.common.table_tile`.
        slot: Table slot. Compile-time.
        token: Chunk-local token, already bounded by ``L``.
        pitch: The table's float32 pitch. Compile-time.

    Returns:
        Entries 0 through 8. At the padded pitch the three padding words ride the
        third load and are dropped.
    """
    if pitch % TABLE_QUAD != 0:
        held = [stable[slot, token, entry] for entry in range(9)]
    else:
        quads, span = _table_quads(stable, slot, token)
        frag = cute.make_fragment((span, TABLE_QUAD), cutlass.Float32)
        for quad in range(span):
            cute.autovec_copy(quads[(None, quad)], frag[(quad, None)])
        held = [frag[i // TABLE_QUAD, i % TABLE_QUAD] for i in range(9)]
    return (
        held[0],
        held[1],
        held[2],
        held[3],
        held[4],
        held[5],
        held[6],
        held[7],
        held[8],
    )


@cute.jit
def _store_rotated(
    words: cute.Tensor,
    frag: cute.Tensor,
    stable: cute.Tensor,
    sscale: cute.Tensor,
    row: cutlass.Int32,
    token: cutlass.Int32,
    pair: cutlass.Int32,
    vecs: tuple[Scalar, ...],
    slot: cutlass.Constexpr,
    scaled: cutlass.Constexpr,
    transposed: cutlass.Constexpr = False,
    pitch: cutlass.Constexpr = 9,
    live: cutlass.Int32 | bool = True,
) -> None:
    """Transform one run of 3-vectors by one table slot and store the run.

    Every vector of the run sits in one row, so they take the same matrix and the
    same scale: the table entry and the one scale word are read once and applied
    ``len(vecs) // 3`` times, which is what divides the table reads per element.

    The matvecs are independent and each sums three products in a fixed order, so
    the run width changes no float sum.

    Args:
        words: The destination tile through :func:`paired`, written at row ``row``,
            columns ``3 * pair`` through ``3 * pair + 2`` in units of the run.
        frag: ``(1, len(vecs) // 3)`` fragment of the tile's element type.
        stable: ``(mats, L, 9)`` float32 transform table.
        sscale: ``(L,)`` float32 per-token scale. Read only when ``scaled``.
        row: Row of the destination tile.
        token: Chunk-local token index, indexing ``stable`` and ``sscale``. Already
            clamped below ``valid`` by the caller: an M extent rounded up past the
            chunk would otherwise read both tiles out of bounds.
        pair: Which run of 3-vectors, below ``lanes // run``.
        vecs: The run's ``3 * run`` components in element order, already widened to
            float32 and already zeroed if the row carries no token.
        slot: Table slot. Compile-time.
        scaled: Whether to multiply by ``sscale[token]``. Compile-time.
        transposed: Apply the slot's transpose. Compile-time: the reads are the same
            reads and the permutation happens during the trace, so the emitted
            matvec is unchanged.
        pitch: The table's float32 pitch. Compile-time.
        live: Whether this run's store is in range. ``True`` where the block width
            divides the run count, and then the guard is traced away. The transform
            runs either way: the read is clamped rather than predicated, so an
            out-of-range run computes a value nobody stores.
    """
    mat = mat_at(stable, slot, token, pitch)
    if cutlass.const_expr(transposed):
        mat = mat3_transpose(mat)
    out: tuple[Scalar, ...] = ()
    for q in cutlass.range_constexpr(len(vecs) // 3):
        out = out + mat3_matvec(mat, (vecs[3 * q], vecs[3 * q + 1], vecs[3 * q + 2]))
    if cutlass.const_expr(scaled):
        weight = sscale[token]
        out = tuple(weight * value for value in out)
    if cutlass.const_expr(live is True):
        store_pair(words, frag, row, 3 * pair, out)
    else:
        if live:
            store_pair(words, frag, row, 3 * pair, out)


def _rot_run(
    threads: cutlass.Constexpr, span: cutlass.Constexpr, lanes: cutlass.Constexpr
) -> int:
    """The run width :func:`stage_rotated` uses at one geometry.

    :data:`ROT_RUN` where it costs the pass fewer steps, :data:`LANE_PAIR`
    otherwise. Two conditions, and neither is about alignment:

    ``lanes % ROT_RUN`` decides whether a run fits a row. ``lanes`` is a multiple of
    16 at every legal shape, so this holds wherever the pairing holds, and is checked
    rather than assumed because the helper takes ``lanes`` from the caller.

    The step counts decide the rest. Halving the work items halves the steps only
    while there are more items than threads; at or below the block width both widths
    take one step, and the wide one takes it with half the lanes doing anything. A
    ragged wide form is admitted where it still costs fewer steps: the loads clamp
    and the stores are predicated either way, so the tail costs redundant transforms,
    not correctness, and it is bought with a whole step.

    Args:
        threads: Block width.
        span: Rows to fill.
        lanes: 3-vectors per row.

    Returns:
        The run width, a divisor of ``lanes``.
    """
    if lanes % ROT_RUN != 0:
        return LANE_PAIR
    wide = -(-(span * (lanes // ROT_RUN)) // threads)
    narrow = -(-(span * (lanes // LANE_PAIR)) // threads)
    if wide >= narrow:
        return LANE_PAIR
    return ROT_RUN


def _rot_carry(has_prev: cutlass.Constexpr, back: cutlass.Constexpr) -> bool:
    """Whether a rotated pass reads the streaming carry-in.

    ``g < 0`` is reachable only for the previous tap at the first token of the first
    chunk, which is exactly the carry-in.

    Args:
        has_prev: Whether the caller supplied ``v_{-1}``.
        back: Token offset of the vector, 0 or 1.

    Returns:
        Whether the second fragment is read.
    """
    return has_prev and back == 1


def _rot_pass(
    threads: cutlass.Constexpr, span: cutlass.Constexpr, lanes: cutlass.Constexpr
) -> tuple[int, int, int, int, bool]:
    """The stride-loop geometry of one :func:`stage_rotated` pass.

    Args:
        threads: Block width.
        span: Rows to fill.
        lanes: ``N``.

    Returns:
        The run width, the runs a row holds, the runs the pass covers, the steps a
        thread takes, and whether the block width divides the run count.
    """
    run = _rot_run(threads, span, lanes)
    pairs = lanes // run
    total = span * pairs
    return run, pairs, total, -(-total // threads), total % threads == 0


def rotated_frags(
    gv: cute.Tensor,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    back: cutlass.Constexpr,
) -> tuple[cute.Tensor, cute.Tensor]:
    """The register file one whole :func:`stage_rotated` pass reads into.

    Allocated by the caller so the read and the transform can sit on either side of a
    barrier: the transform reads the transform table, so it cannot precede the table
    build, while the reads it consumes have no such order to keep.

    Args:
        gv: The pass's source, for its element type.
        threads: Block width.
        span: Rows to fill.
        lanes: ``N``.
        has_prev: Whether the caller supplied ``v_{-1}``.
        back: Token offset of the vector, 0 or 1.

    Returns:
        The current-tap fragment and the carry-in fragment, each ``(3 * steps, run)``,
        indexed ``[3 * step + k, j]`` by the step, the component and the element
        within the run. The second aliases the first where there is no carry-in, for
        the reason given in :func:`stage_rotated`.
    """
    run, _, _, steps, _ = _rot_pass(threads, span, lanes)
    loads = cute.make_fragment((3 * steps, run), gv.element_type)
    if cutlass.const_expr(_rot_carry(has_prev, back)):
        return loads, cute.make_fragment((3 * steps, run), gv.element_type)
    return loads, loads


def _rotated_step(
    tid: cutlass.Int32,
    step: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    pairs: cutlass.Constexpr,
    total: cutlass.Constexpr,
    exact: cutlass.Constexpr,
    t0: cutlass.Int32,
    lbase: cutlass.Int32,
    valid: cutlass.Int32,
    back: cutlass.Constexpr,
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """The row, the run and the two token indices one step of a rotated pass covers.

    One clamp serves both reads: ``valid`` is at most the chunk, so the clamped token
    indexes the table and the scale in bounds even when the M extent was rounded up
    past the chunk, and ``t0`` plus it is inside the sequence.

    Args:
        tid: Thread index within the block.
        step: Which step. Compile-time.
        threads: Block width. Compile-time.
        pairs: Runs a row holds. Compile-time.
        total: Runs the pass covers. Compile-time.
        exact: Whether the block width divides ``total``. Compile-time.
        t0: First token of the chunk.
        lbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        back: Token offset of the vector, 0 or 1. Compile-time.

    Returns:
        The destination row, the run within it, the clamped chunk-local token, and the
        global token the vector comes from, which is negative only at the carry-in.
    """
    i = tid + step * threads
    if cutlass.const_expr(not exact):
        i = cutlass.min(i, total - 1)
    r = i // pairs
    tsafe = cutlass.min(lbase + r, valid - 1)
    return r, i - r * pairs, tsafe, t0 + tsafe - back


@cute.jit
def _rotated_vec(
    loads: cute.Tensor,
    prior: cute.Tensor,
    slot: cutlass.Constexpr,
    run: cutlass.Constexpr,
    r: cutlass.Int32,
    lbase: cutlass.Int32,
    valid: cutlass.Int32,
    gbase: cutlass.Int32,
    has_prev: cutlass.Constexpr,
    back: cutlass.Constexpr,
) -> tuple[Scalar, ...]:
    """One step's run, widened, with the carry-in and the dead rows resolved.

    Args:
        loads: The current-tap fragment.
        prior: The carry-in fragment, read only where there is one.
        slot: First fragment row of this step, three from there. Compile-time.
        run: Elements per access. Compile-time.
        r: Destination row.
        lbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        gbase: The global token the vector comes from.
        has_prev: Whether the caller supplied ``v_{-1}``. Compile-time.
        back: Token offset of the vector, 0 or 1. Compile-time.

    Returns:
        The run's ``3 * run`` float32 components, zeroed where the row carries no
        token.
    """
    src = loads.element_type
    # A plain range, not range_constexpr: a comprehension reaches the runtime stub.
    # Both unroll at trace time.
    got = tuple(widen(loads[slot + j // run, j % run], src) for j in range(3 * run))
    if cutlass.const_expr(_rot_carry(has_prev, back)):
        at_start = gbase < 0
        got = tuple(
            select(at_start, widen(prior[slot + j // run, j % run], src), got[j])
            for j in range(3 * run)
        )
    keep = lbase + r < valid
    if cutlass.const_expr(back == 1 and not has_prev):
        keep = keep & (gbase >= 0)
    return tuple(select(keep, value, cutlass.Float32(0.0)) for value in got)


@cute.jit
def read_rotated(
    gv: cute.Tensor,
    gvprev: cute.Tensor,
    loads: cute.Tensor,
    prior: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    t0: cutlass.Int32,
    lbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    back: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Issue every global read of one :func:`stage_rotated` pass.

    Split out so the reads can precede the transform table's build, which the
    transform half depends on and they do not. Nothing is loaded under a predicate:
    the index is clamped into range and the out-of-range value is replaced afterwards
    by a select.

    Args:
        gv: ``(B,G,T,3N)`` operand-dtype source.
        gvprev: ``(B,G,3N)`` streaming ``v_{-1}``. Read only where there is a
            carry-in.
        loads: The current-tap fragment from :func:`rotated_frags`.
        prior: The carry-in fragment from the same call.
        bidx: Batch index.
        gidx: Group index.
        t0: First token of the chunk.
        lbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        back: Token offset of the vector, 0 or 1. Compile-time.
        threads: Block width. Compile-time.
        span: Rows to fill. Compile-time.
        lanes: ``N``. Compile-time.
        has_prev: Whether ``gvprev`` was supplied. Compile-time.

    Invariants:
        The whole pass is in flight at once, so no step's slot is reused and the
        fragment is ``3 * steps`` rows rather than the ``3 * PREFETCH // run``
        :func:`stage_rotated` bounds itself to.
    """
    run, pairs, total, steps, exact = _rot_pass(threads, span, lanes)
    for step in cutlass.range_constexpr(steps):
        _, m, _, gbase = _rotated_step(
            tid, step, threads, pairs, total, exact, t0, lbase, valid, back
        )
        _load_pair(
            _paired_row(gv[bidx, gidx, cutlass.max(gbase, 0), None], run),
            loads,
            3 * step,
            m,
        )
        if cutlass.const_expr(_rot_carry(has_prev, back)):
            _load_pair(_paired_row(gvprev[bidx, gidx, None], run), prior, 3 * step, m)


@cute.jit
def apply_rotated(
    loads: cute.Tensor,
    prior: cute.Tensor,
    dst: cute.Tensor,
    stable: cute.Tensor,
    sscale: cute.Tensor,
    t0: cutlass.Int32,
    lbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    slot: cutlass.Constexpr,
    back: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    scaled: cutlass.Constexpr,
    transposed: cutlass.Constexpr = False,
    pitch: cutlass.Constexpr = 9,
) -> None:
    """Transform one :func:`read_rotated` fragment into a shared operand tile.

    The step's coordinates are recomputed rather than carried, for the reason given in
    :func:`apply_matrix`.

    Args:
        loads: The current-tap fragment :func:`read_rotated` filled.
        prior: The carry-in fragment from the same call.
        dst: Operand-dtype tile of at least ``span`` rows, written.
        stable: ``(mats, L, pitch)`` float32 transform table.
        sscale: ``(L,)`` float32 per-token scale. Read only when ``scaled``.
        t0: First token of the chunk.
        lbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        slot: Table slot. Compile-time.
        back: Token offset of the vector, 0 or 1. Compile-time.
        threads: Block width. Compile-time.
        span: Rows to fill. Compile-time.
        lanes: ``N``. Compile-time.
        has_prev: Whether a carry-in was supplied. Compile-time.
        scaled: Whether to apply ``sscale``. Compile-time.
        transposed: Apply the slot's transpose. Compile-time.
        pitch: The table's float32 pitch. Compile-time.
    """
    run, pairs, total, steps, exact = _rot_pass(threads, span, lanes)
    words = paired(dst, run)
    frag = cute.make_fragment((1, run), dst.element_type)
    for step in cutlass.range_constexpr(steps):
        r, m, tsafe, gbase = _rotated_step(
            tid, step, threads, pairs, total, exact, t0, lbase, valid, back
        )
        _store_rotated(
            words,
            frag,
            stable,
            sscale,
            r,
            tsafe,
            m,
            _rotated_vec(
                loads, prior, 3 * step, run, r, lbase, valid, gbase, has_prev, back
            ),
            slot,
            scaled,
            transposed,
            pitch,
            exact or tid + step * threads < total,
        )


@cute.jit
def stage_rotated(
    gv: cute.Tensor,
    gvprev: cute.Tensor,
    dst: cute.Tensor,
    stable: cute.Tensor,
    sscale: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    t0: cutlass.Int32,
    lbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    slot: cutlass.Constexpr,
    back: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    scaled: cutlass.Constexpr,
    transposed: cutlass.Constexpr = False,
    pitch: cutlass.Constexpr = 9,
) -> None:
    """Transform a run of tokens by one table slot into a shared operand tile.

    Row ``r`` holds ``A_slot[lbase + r] v[t0 + lbase + r - back]``, optionally
    scaled, with ``A_slot`` transposed when ``transposed``, which is what every
    rowwise transform on the backward path applies. ``back`` is 0 for the current
    tap and the readout, 1 for the previous tap: the matrix is indexed at the token
    it acts on while the previous tap's vector comes from the token before it.

    One thread owns a run of adjacent 3-vectors of one row -- :func:`_rot_run` picks
    the width -- as three global reads, nine FMA a vector over one matrix read, and
    three shared-memory accesses.

    The pass runs in groups of ``PREFETCH // run`` steps, loads first and transforms
    second, so ``3 * PREFETCH`` elements are in flight when the first of them is
    consumed at either width. Nothing is loaded under a predicate: the index is
    clamped into range and the out-of-range value is replaced afterwards by a select.
    A load inside a divergent branch cannot be hoisted above the branch, and a value
    produced inside one has no phi node to leave through, so the predicated form
    serializes on one global latency per step.

    Rows whose token is at or past ``valid`` are zeroed, which also zeroes the rows
    an M extent was rounded up by, since ``lbase`` is zero whenever ``span``
    exceeds the chunk. A zero row contributes nothing to any contraction, so no
    consumer needs a predicate. Zeroing the float32 input rather than the stored
    output is one select per component and makes the FMAs exact.

    Args:
        gv: ``(B,G,T,3N)`` operand-dtype source.
        gvprev: ``(B,G,3N)`` streaming ``v_{-1}``. Read only when ``has_prev`` and
            ``back`` is 1.
        dst: Operand-dtype tile of at least ``span`` rows, written.
        stable: ``(mats, L, pitch)`` float32 transform table.
        sscale: ``(L,)`` float32 per-token scale. Read only when ``scaled``.
        bidx: Batch index.
        gidx: Group index, ``h // (H // G)``. The transform table is per head and
            the vector is per group, so the two indices are distinct and only the
            caller knows the head.
        t0: First token of the chunk.
        lbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        slot: Table slot. Compile-time.
        back: Token offset of the vector, 0 or 1. Compile-time.
        threads: Block width. Compile-time.
        span: Rows of ``dst`` to fill. Compile-time.
        lanes: ``N``. Compile-time.
        has_prev: Whether ``gvprev`` was supplied. Compile-time.
        scaled: Whether to apply ``sscale``. Compile-time.
        transposed: Apply the slot's transpose rather than the slot. Compile-time,
            and free: it permutes an index triple during the trace.
        pitch: The table's float32 pitch, as passed to
            :func:`slinoss.ops.so3ssd.cute.common.table_tile`. Compile-time.
            :data:`TABLE_PITCH` buys the vector-width table read.

    Invariants:
        ``lanes`` is a multiple of the run width and ``dst`` is pitched by
        :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`, which is what
        :data:`LANE_PAIR` and :data:`ROT_RUN` rest on.
    """
    # The staging extents are all multiples of the block width at every legal
    # shape, so the store predicate below is elided. Both extents are multiples of
    # 16 and the block is four warps, so pairing keeps that. The general form is
    # kept because it costs nothing when it is not needed.
    run, pairs, total, steps, exact = _rot_pass(threads, span, lanes)
    depth = max(1, PREFETCH // run)

    # One group is the whole pass, so the split form emits the same block in the same
    # order and there is one body rather than two.
    if cutlass.const_expr(steps <= depth):
        loads, prior = rotated_frags(gv, threads, span, lanes, has_prev, back)
        read_rotated(
            gv,
            gvprev,
            loads,
            prior,
            bidx,
            gidx,
            t0,
            lbase,
            valid,
            tid,
            back,
            threads,
            span,
            lanes,
            has_prev,
        )
        apply_rotated(
            loads,
            prior,
            dst,
            stable,
            sscale,
            t0,
            lbase,
            valid,
            tid,
            slot,
            back,
            threads,
            span,
            lanes,
            has_prev,
            scaled,
            transposed,
            pitch,
        )
        return

    src = gv.element_type
    carry = _rot_carry(has_prev, back)
    words = paired(dst, run)
    frag = cute.make_fragment((1, run), dst.element_type)
    loads = cute.make_fragment((3 * depth, run), src)
    # The false arm aliases the current-tap fragment, which is never read under it:
    # every use of the carry fragment sits under the same compile-time flag.
    prior = (
        cute.make_fragment((3 * depth, run), src)
        if cutlass.const_expr(carry)
        else loads
    )

    for group in cutlass.range_constexpr(-(-steps // depth)):
        width = min(depth, steps - group * depth)
        held = []
        for step in cutlass.range_constexpr(width):
            first = group * depth + step
            r, m, tsafe, gbase = _rotated_step(
                tid, first, threads, pairs, total, exact, t0, lbase, valid, back
            )
            _load_pair(
                _paired_row(gv[bidx, gidx, cutlass.max(gbase, 0), None], run),
                loads,
                3 * step,
                m,
            )
            if cutlass.const_expr(carry):
                _load_pair(
                    _paired_row(gvprev[bidx, gidx, None], run), prior, 3 * step, m
                )
            held.append((r, m, tsafe, gbase, first))

        for step in cutlass.range_constexpr(width):
            r, m, tsafe, gbase, first = held[step]
            _store_rotated(
                words,
                frag,
                stable,
                sscale,
                r,
                tsafe,
                m,
                _rotated_vec(
                    loads, prior, 3 * step, run, r, lbase, valid, gbase, has_prev, back
                ),
                slot,
                scaled,
                transposed,
                pitch,
                exact or tid + first * threads < total,
            )
