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

Table storage is ``(mats, L, 9)`` float32, nine entries innermost. The build
stores nine words at a nine-word stride, and nine is coprime with the 32 banks,
so the store pattern is a bank permutation. Every read during application is a
broadcast. Neither needs a swizzle.

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
the weight rather than its partner needs.
"""

import cutlass
import cutlass.cute as cute

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
    "build_table",
    "paired",
    "stage_chunk",
    "stage_matrix",
    "stage_pad",
    "stage_raw",
    "stage_rotated",
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
of the three."""

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
    stage_rotated    lanes // 2    8, 16 are shifts; 40 leaves 128 % 40 = 8
    stage_state      dim // 2      24, 48, 120 divide 128 in no case
    stage_shifted    width // 2    32 is a shift; 24 leaves 128 % 24 = 8

So the rewrite is available at exactly the geometries where it removes nothing, and
absent at every geometry that pays a magic multiply. Enabling it needs a different
thread-to-element map, which gives up the contiguity the pairing exists for."""


def paired(tile: cute.Tensor) -> cute.Tensor:
    """View a row-major tile in units of :data:`LANE_PAIR` adjacent elements.

    Args:
        tile: ``(rows, pitch)`` shared tile, unit stride on the columns and a pitch
            from :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`.

    Returns:
        The retiled view. Element ``(None, (r, k))`` is elements
        ``LANE_PAIR * k`` through ``LANE_PAIR * k + LANE_PAIR - 1`` of row ``r``,
        statically shaped, which is what lets :func:`cutlass.cute.autovec_copy` pick
        one access rather than :data:`LANE_PAIR`.

    Invariants:
        Every access is aligned to ``LANE_PAIR`` elements. The pitch is a whole
        number of 16-byte segments, so a row starts on a 16-byte boundary, and the
        column offset is a multiple of ``LANE_PAIR``. The claim is restated on the
        iterator because a tile arriving as a parameter reports one element whatever
        its allocation asked for, and ``autovec_copy`` caps the access at the claim.
    """
    base = tile.iterator.align(LANE_PAIR * (tile.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, tile.layout), (1, LANE_PAIR))


def _paired_row(row: cute.Tensor) -> cute.Tensor:
    """View one contiguous row in units of :data:`LANE_PAIR` adjacent elements.

    Args:
        row: ``(3N,)`` unit-stride view of one row of a global tensor.

    Returns:
        The retiled view. Element ``(None, k)`` is elements ``LANE_PAIR * k``
        through ``LANE_PAIR * k + LANE_PAIR - 1``.

    Invariants:
        ``3N`` is a multiple of 48, so a row offset is a multiple of 48 elements and
        every access is aligned to ``LANE_PAIR``. The claim is restated for the
        reason given in :func:`paired`.
    """
    base = row.iterator.align(LANE_PAIR * (row.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, row.layout), (LANE_PAIR,))


@cute.jit
def _load_pair(
    row: cute.Tensor, frag: cute.Tensor, slot: cutlass.Constexpr, pair: cutlass.Int32
) -> None:
    """Read one pair's ``3 * LANE_PAIR`` elements as three paired accesses.

    Args:
        row: A row from :func:`_paired_row`.
        frag: ``(slots, LANE_PAIR)`` fragment of the row's element type.
        pair: Which pair of 3-vectors. Its accesses are ``3 * pair`` to
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
    """Store ``LANE_PAIR`` adjacent values as one access.

    Args:
        words: A tile from :func:`paired`.
        frag: ``(1, LANE_PAIR)`` fragment of the tile's element type, built once by
            the caller: a fragment per step is an allocation per step.
        row: Row of the tile.
        col: Paired column.
        vals: ``LANE_PAIR`` values in element order, float32. Narrowed here, once.
    """
    elem = words.element_type
    for j in cutlass.range_constexpr(LANE_PAIR):
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
    """Store one pair's ``3 * LANE_PAIR`` values as three paired accesses.

    Args:
        words: A tile from :func:`paired`.
        frag: ``(1, LANE_PAIR)`` fragment of the tile's element type, built once by
            the caller: a fragment per step is an allocation per step.
        row: Row of the tile.
        col: First paired column of the pair, ``3 * m`` for pair ``m``.
        vals: The values in element order, float32. Narrowed here, once.
    """
    for k in cutlass.range_constexpr(3):
        _store_run(words, frag, row, col + k, vals[LANE_PAIR * k : LANE_PAIR * (k + 1)])


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
    """
    zero = cutlass.Float32(0.0)
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            inside = token < valid
            # The clamp keeps the read in bounds when the chunk overhangs the
            # sequence; the select then replaces it with the pad value.
            pos = t0 + cutlass.min(token, valid - 1)
            for j in cutlass.range_constexpr(4):
                strans[j, token] = select(inside, gtrans[pos, j], zero)


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
    """
    stage_trans(gtrans, strans, t0, valid, tid, threads, chunk)
    zero = cutlass.Float32(0.0)
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            inside = token < valid
            pos = t0 + cutlass.min(token, valid - 1)
            for tap in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(4):
                    stap[4 * tap + j, token] = select(inside, gtap[pos, tap, j], zero)


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
) -> None:
    """Compose the ``(mats, L, 9)`` transform table in shared memory.

    Args:
        strans: ``(4, L)`` float32 staged transition parameters, ``(w, ls)``
            component-major.
        stap: ``(8, L)`` float32 staged tap parameters. Unread at ``mats == 1``,
            which is the point of that slot count, so a caller there may pass any
            tile of the right dtype.
        squat: ``(4, L)`` float32 quaternion prefix, already renormalized.
        stable: ``(mats, L, 9)`` float32, written. Slots are
            :data:`slinoss.ops.so3ssd.cute.common.TABLE_AP`, ``TABLE_AN``, and,
            when ``mats`` is three, ``TABLE_AC``. At ``mats == 1`` the sole slot is
            ``TABLE_AC_SOLE``. At ``fused`` the first slot is ``TABLE_AFUSE``.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        mats: Slots to write, 1, 2 or 3. Compile-time, and must match the ``mats``
            the tile was allocated at. One writes ``Ac`` alone and computes neither
            tap matrix, so it also reads neither ``stap`` nor ``strans``.
        fused: Write the one-tap column ``Afuse_t = Ap_t + exp(2*ls_t) An_{t-1}``
            into the first slot instead of ``Ap_t``. Compile-time. Ignored at
            ``mats == 1``, which builds no tap matrix. The second slot still holds
            ``An_t``, which the diagonal residue needs.

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
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            ac = mat3_transpose(
                rot_hom(
                    (
                        squat[0, token],
                        squat[1, token],
                        squat[2, token],
                        squat[3, token],
                    )
                )
            )
            if cutlass.const_expr(mats == 1):
                for entry in cutlass.range_constexpr(9):
                    stable[TABLE_AC_SOLE, token, entry] = ac[entry]
            else:
                wvec = (strans[0, token], strans[1, token], strans[2, token])
                ap = mat3_mul(
                    ac,
                    tap_matrix((stap[0, token], stap[1, token], stap[2, token]), wvec),
                )
                an = mat3_mul(
                    ac,
                    tap_matrix((stap[4, token], stap[5, token], stap[6, token]), wvec),
                )
                first = ap
                if cutlass.const_expr(fused):
                    # ``An_{t-1}`` is recomputed from the staged parameters of token
                    # ``t-1`` rather than read back out of the slot this pass is
                    # still writing. A read-back needs a barrier between the two
                    # halves of the table, and barrier is the top stall of every
                    # kernel that builds one. The recompute is one rotation and one
                    # tap matrix per token, O(L) against an O(L*N) launch.
                    #
                    # Clamped rather than branched: token 0 reads its own row and the
                    # factor below discards it.
                    prev = cutlass.max(token - 1, 0)
                    acp = mat3_transpose(
                        rot_hom(
                            (
                                squat[0, prev],
                                squat[1, prev],
                                squat[2, prev],
                                squat[3, prev],
                            )
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
                    scale = select(
                        token > 0, decay(strans[3, token]), cutlass.Float32(0.0)
                    )
                    first = tuple(ap[e] + scale * anp[e] for e in range(9))
                for entry in cutlass.range_constexpr(9):
                    stable[TABLE_AP, token, entry] = first[entry]
                    stable[TABLE_AN, token, entry] = an[entry]
                if cutlass.const_expr(mats == 3):
                    for entry in cutlass.range_constexpr(9):
                        stable[TABLE_AC, token, entry] = ac[entry]


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

    The pass runs in groups of ``PREFETCH // LANE_PAIR`` steps, loads first, on
    clamped indices with a select afterwards, for the reason given in
    :func:`stage_rotated`. This is the longest staging pass in the operator:
    ``(span + 1) * width`` elements at :data:`LANE_PAIR` elements per thread per
    step, which is one read and one write per step at either end.

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

    Invariants:
        ``width`` is ``P`` or a lane tile of ``3N``, so it is even:
        :func:`slinoss.ops.so3ssd.cute.guard.check_rows` holds ``P`` to a multiple of
        16 and a lane tile is a multiple of 48. ``su`` is pitched by
        :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`, which is what
        :data:`LANE_PAIR` rests on. The pair predicate is reachable where the element
        predicate was not: pairing halves the extent, so an extent that was a
        multiple of the block width need not stay one.
    """
    src = gu.element_type
    zero = cutlass.Float32(0.0)
    pairs = width // LANE_PAIR
    total = (span + 1) * pairs
    steps = -(-total // threads)
    exact = total % threads == 0
    depth = max(1, PREFETCH // LANE_PAIR)

    words = paired(su)
    frag = cute.make_fragment((1, LANE_PAIR), su.element_type)
    loads = cute.make_fragment((depth, LANE_PAIR), src)
    # The false arm aliases the current fragment, which is never read under it: every
    # use of the carry fragment sits under the same compile-time flag.
    prior = (
        cute.make_fragment((depth, LANE_PAIR), src)
        if cutlass.const_expr(has_prev)
        else loads
    )

    for group in cutlass.range_constexpr(-(-steps // depth)):
        count = min(depth, steps - group * depth)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * depth + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // pairs
            p = i - r * pairs
            token = lbase + r - 1
            # token < valid implies t0 + token < seqlen, so clamping the token
            # bounds the read above; the row before the sequence bounds it below.
            gbase = t0 + cutlass.min(token, valid - 1)
            row = _paired_row(gu[bidx, hidx, cutlass.max(gbase, 0), None])
            cute.autovec_copy(row[(None, p)], loads[(step, None)])
            if cutlass.const_expr(has_prev):
                back = _paired_row(guprev[bidx, hidx, None])
                cute.autovec_copy(back[(None, p)], prior[(step, None)])
            keep = token < valid
            if cutlass.const_expr(not has_prev):
                keep = keep & (gbase >= 0)
            held.append((r, p, keep, gbase < 0))

        for step in cutlass.range_constexpr(count):
            r, p, keep, at_start = held[step]
            # The select is float32 because there is one select helper; the
            # operand round trip through float32 is exact at every operand width.
            got = tuple(widen(loads[step, j], src) for j in range(LANE_PAIR))
            if cutlass.const_expr(has_prev):
                got = tuple(
                    select(at_start, widen(prior[step, j], src), got[j])
                    for j in range(LANE_PAIR)
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

    A step carries :data:`LANE_PAIR` adjacent columns, which is one read and one write
    of the pair's width in place of two of each. The port issues one access per two
    cycles whatever its width, so the pair halves the cost of the pass.

    Args:
        gz: ``(P, 3N)`` view of the chunk-start state for one
            ``(chunk, batch, head)``, at the operand dtype or float32.
        sz: Operand-dtype tile of at least ``width`` rows, written over ``dim``
            columns. The rest of the pitch is outside every view.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        width: Rows to fill, ``P``. Compile-time.
        dim: ``3N``. Compile-time.

    Invariants:
        ``dim`` is a multiple of 48 or a lane tile of one, so it is even, and ``sz`` is
        pitched by :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`: both ends of a pair
        access are aligned to :data:`LANE_PAIR` elements. Each element is narrowed on
        its own, so the pair does not associate anything.
    """
    src = gz.element_type
    pairs = dim // LANE_PAIR
    total = width * pairs
    steps = -(-total // threads)
    exact = total % threads == 0
    depth = max(1, PREFETCH // LANE_PAIR)

    words = paired(sz)
    frag = cute.make_fragment((1, LANE_PAIR), sz.element_type)
    loads = cute.make_fragment((depth, LANE_PAIR), src)

    for group in cutlass.range_constexpr(-(-steps // depth)):
        count = min(depth, steps - group * depth)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * depth + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            p = i // pairs
            col = i - p * pairs
            cute.autovec_copy(
                _paired_row(gz[p, None])[(None, col)], loads[(step, None)]
            )
            held.append((p, col))

        for step in cutlass.range_constexpr(count):
            p, col = held[step]
            out = tuple(loads[step, j] for j in range(LANE_PAIR))
            if cutlass.const_expr(exact):
                _store_run(words, frag, p, col, out)
            else:
                if tid + (group * depth + step) * threads < total:
                    _store_run(words, frag, p, col, out)


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
    reason given in :func:`stage_rotated`.

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
    elem = dst.element_type
    span = 3 * LANE_PAIR
    pairs = lanes // LANE_PAIR
    total = rows * pairs
    steps = -(-total // threads)
    exact = total % threads == 0
    depth = max(1, PREFETCH // LANE_PAIR)

    words = paired(dst)
    frag = cute.make_fragment((1, LANE_PAIR), elem)
    loads = cute.make_fragment((3 * depth, LANE_PAIR), src)
    # The false arm aliases the operand pair. Every use is under the same
    # compile-time flag, so the alias is never reached and never allocated, and the
    # name is bound on both paths.
    if cutlass.const_expr(keep_fp32):
        words32 = paired(sfp32)
        frag32 = cute.make_fragment((1, LANE_PAIR), cutlass.Float32)
    else:
        words32 = words
        frag32 = frag

    for group in cutlass.range_constexpr(-(-steps // depth)):
        count = min(depth, steps - group * depth)
        held = []
        for step in cutlass.range_constexpr(count):
            i = tid + (group * depth + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            p = i // pairs
            m = i - p * pairs
            _load_pair(_paired_row(gv[bidx, hidx, cidx, p, None]), loads, 3 * step, m)
            held.append((p, m))

        for step in cutlass.range_constexpr(count):
            p, m = held[step]
            got = tuple(
                widen(loads[3 * step + j // LANE_PAIR, j % LANE_PAIR], src)
                for j in range(span)
            )
            out = mat3_matvec(mat, (got[0], got[1], got[2])) + mat3_matvec(
                mat, (got[3], got[4], got[5])
            )
            col = 3 * m
            if cutlass.const_expr(exact):
                store_pair(words, frag, p, col, out)
                if cutlass.const_expr(keep_fp32):
                    store_pair(words32, frag32, p, col, out)
            else:
                if tid + (group * depth + step) * threads < total:
                    store_pair(words, frag, p, col, out)
                    if cutlass.const_expr(keep_fp32):
                        store_pair(words32, frag32, p, col, out)


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
) -> None:
    """Transform one pair of 3-vectors by one table slot and store both.

    Both vectors of the pair sit in one row, so they take the same matrix and the
    same scale: the nine table words and the one scale word are read once and
    applied twice, which halves the table reads per element.

    Args:
        words: The destination tile through :func:`paired`, written at row ``row``,
            paired columns ``3 * pair`` through ``3 * pair + 2``.
        frag: ``(1, LANE_PAIR)`` fragment of the tile's element type.
        stable: ``(mats, L, 9)`` float32 transform table.
        sscale: ``(L,)`` float32 per-token scale. Read only when ``scaled``.
        row: Row of the destination tile.
        token: Chunk-local token index, indexing ``stable`` and ``sscale``. Already
            clamped below ``valid`` by the caller: an M extent rounded up past the
            chunk would otherwise read both tiles out of bounds.
        pair: Which pair of 3-vectors, below ``lanes // LANE_PAIR``.
        vecs: The pair's ``3 * LANE_PAIR`` components in element order, already
            widened to float32 and already zeroed if the row carries no token.
        slot: Table slot. Compile-time.
        scaled: Whether to multiply by ``sscale[token]``. Compile-time.
        transposed: Apply the slot's transpose. Compile-time: the nine reads are
            the same nine and the permutation happens during the trace, so the
            emitted matvec is unchanged.
    """
    mat = (
        stable[slot, token, 0],
        stable[slot, token, 1],
        stable[slot, token, 2],
        stable[slot, token, 3],
        stable[slot, token, 4],
        stable[slot, token, 5],
        stable[slot, token, 6],
        stable[slot, token, 7],
        stable[slot, token, 8],
    )
    if cutlass.const_expr(transposed):
        mat = mat3_transpose(mat)
    out = mat3_matvec(mat, (vecs[0], vecs[1], vecs[2])) + mat3_matvec(
        mat, (vecs[3], vecs[4], vecs[5])
    )
    if cutlass.const_expr(scaled):
        weight = sscale[token]
        out = tuple(weight * value for value in out)
    store_pair(words, frag, row, 3 * pair, out)


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
) -> None:
    """Transform a run of tokens by one table slot into a shared operand tile.

    Row ``r`` holds ``A_slot[lbase + r] v[t0 + lbase + r - back]``, optionally
    scaled, with ``A_slot`` transposed when ``transposed``, which is what every
    rowwise transform on the backward path applies. ``back`` is 0 for the current
    tap and the readout, 1 for the previous tap: the matrix is indexed at the token
    it acts on while the previous tap's vector comes from the token before it.

    One thread owns :data:`LANE_PAIR` adjacent 3-vectors of one row: three paired
    global reads, eighteen FMA over one matrix read, three paired shared-memory
    accesses.

    The pass runs in groups of ``PREFETCH // LANE_PAIR`` steps, loads first and
    transforms second, so ``3 * PREFETCH`` elements are in flight when the first of
    them is consumed. Nothing is loaded under a predicate: the index is clamped into
    range and the out-of-range value is replaced afterwards by a select. A load inside a
    divergent branch cannot be hoisted above the branch, and a value produced
    inside one has no phi node to leave through, so the predicated form serializes
    on one global latency per step.

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
        stable: ``(mats, L, 9)`` float32 transform table.
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

    Invariants:
        ``lanes`` is even and ``dst`` is pitched by
        :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`, which is what
        :data:`LANE_PAIR` rests on.
    """
    src = gv.element_type
    zero = cutlass.Float32(0.0)
    wide = 3 * LANE_PAIR
    pairs = lanes // LANE_PAIR
    total = span * pairs
    steps = -(-total // threads)
    # The staging extents are all multiples of the block width at every legal
    # shape, so the store predicate below is elided. Both extents are multiples of
    # 16 and the block is four warps, so pairing keeps that. The general form is
    # kept because it costs nothing when it is not needed.
    exact = total % threads == 0
    depth = max(1, PREFETCH // LANE_PAIR)
    # g < 0 is reachable only for the previous tap at the first token of the first
    # chunk, which is exactly the streaming carry-in.
    carry = has_prev and back == 1

    words = paired(dst)
    frag = cute.make_fragment((1, LANE_PAIR), dst.element_type)
    loads = cute.make_fragment((3 * depth, LANE_PAIR), src)
    # The false arm aliases the current-tap fragment, which is never read under it:
    # every use of the carry fragment sits under the same compile-time flag.
    prior = (
        cute.make_fragment((3 * depth, LANE_PAIR), src)
        if cutlass.const_expr(carry)
        else loads
    )

    for group in cutlass.range_constexpr(-(-steps // depth)):
        width = min(depth, steps - group * depth)
        held = []
        for step in cutlass.range_constexpr(width):
            i = tid + (group * depth + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // pairs
            m = i - r * pairs
            # One clamp serves both reads: valid is at most the chunk, so the
            # clamped token indexes stable and sscale in bounds even when the M
            # extent was rounded up past the chunk, and t0 + it is inside the
            # sequence.
            tsafe = cutlass.min(lbase + r, valid - 1)
            gbase = t0 + tsafe - back
            gsafe = cutlass.max(gbase, 0)
            _load_pair(_paired_row(gv[bidx, gidx, gsafe, None]), loads, 3 * step, m)
            if cutlass.const_expr(carry):
                _load_pair(_paired_row(gvprev[bidx, gidx, None]), prior, 3 * step, m)
            held.append((r, m, tsafe, gbase))

        for step in cutlass.range_constexpr(width):
            r, m, tsafe, gbase = held[step]
            # A plain range, not range_constexpr: a comprehension reaches the
            # runtime stub. Both unroll at trace time.
            got = tuple(
                widen(loads[3 * step + j // LANE_PAIR, j % LANE_PAIR], src)
                for j in range(wide)
            )
            if cutlass.const_expr(carry):
                at_start = gbase < 0
                got = tuple(
                    select(
                        at_start,
                        widen(prior[3 * step + j // LANE_PAIR, j % LANE_PAIR], src),
                        got[j],
                    )
                    for j in range(wide)
                )
            keep = lbase + r < valid
            if cutlass.const_expr(back == 1 and not has_prev):
                keep = keep & (gbase >= 0)
            vec = tuple(select(keep, value, zero) for value in got)
            if cutlass.const_expr(exact):
                _store_rotated(
                    words,
                    frag,
                    stable,
                    sscale,
                    r,
                    tsafe,
                    m,
                    vec,
                    slot,
                    scaled,
                    transposed,
                )
            else:
                if tid + (group * depth + step) * threads < total:
                    _store_rotated(
                        words,
                        frag,
                        stable,
                        sscale,
                        r,
                        tsafe,
                        m,
                        vec,
                        slot,
                        scaled,
                        transposed,
                    )
