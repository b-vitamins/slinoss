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
same tiles at a block stride instead; both shared-memory bank-conflict counters
are measured zero at every legal chunk size, which is the constraint
``MAX_CHUNK`` is set by.

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

The two stagings that need no table are here for the same reason: more than one
kernel needs each. :func:`stage_shifted` lays a ``(T, P)`` tensor into a tile one
row behind, so the two taps read one staging pass through two views; and
:func:`stage_state` narrows a chunk-start state into an operand tile.
"""

import cutlass
import cutlass.cute as cute

from slinoss._cute import narrow, select, widen
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AN,
    TABLE_AP,
    Vec3,
    mat3_matvec,
    mat3_mul,
    mat3_transpose,
    rot_hom,
    tap_matrix,
)

__all__ = [
    "PREFETCH",
    "build_table",
    "stage_chunk",
    "stage_pad",
    "stage_rotated",
    "stage_shifted",
    "stage_state",
]

PREFETCH: int = 4
"""Staging steps whose global loads are issued before any of them is consumed.

Bounds the load phase at ``3 * PREFETCH`` live float32 registers while keeping
``3 * PREFETCH`` loads outstanding per thread, which is what covers one global
latency. One is the serial form: load, transform, store, wait, repeat."""


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

    Args:
        gtrans: ``(T, 4)`` float32 view of ``trans`` for one ``(b, h)``.
        gtap: ``(T, 2, 4)`` float32 view of ``K`` for one ``(b, h)``.
        strans: ``(4, L)`` float32 shared tile, written.
        stap: ``(8, L)`` float32 shared tile, written. Component ``4*tap + j``.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist, at least one. Tokens at or past
            this index are staged as zeros. The clamp below reads ``valid - 1``,
            so zero would read the token before the chunk; a chunk with no valid
            token is not launched.
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
            for tap in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(4):
                    stap[4 * tap + j, token] = select(inside, gtap[pos, tap, j], zero)


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
) -> None:
    """Compose the ``(mats, L, 9)`` transform table in shared memory.

    Args:
        strans: ``(4, L)`` float32 staged transition parameters.
        stap: ``(8, L)`` float32 staged tap parameters.
        squat: ``(4, L)`` float32 quaternion prefix, already renormalized.
        stable: ``(mats, L, 9)`` float32, written. Slots are
            :data:`slinoss.ops.so3ssd.cute.common.TABLE_AP`, ``TABLE_AN``, and,
            when ``mats`` is three, ``TABLE_AC``.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        mats: Slots to write, 2 or 3. Compile-time, and must match the ``mats``
            the tile was allocated at.
    """
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            wvec = (strans[0, token], strans[1, token], strans[2, token])
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
            ap = mat3_mul(
                ac, tap_matrix((stap[0, token], stap[1, token], stap[2, token]), wvec)
            )
            an = mat3_mul(
                ac, tap_matrix((stap[4, token], stap[5, token], stap[6, token]), wvec)
            )
            for entry in cutlass.range_constexpr(9):
                stable[TABLE_AP, token, entry] = ap[entry]
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

    The pass runs in groups of :data:`PREFETCH` steps, loads first, on clamped
    indices with a select afterwards, for the reason given in
    :func:`stage_rotated`. This is the longest staging pass in the operator:
    ``(span + 1) * width`` elements at one element per thread per step.

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
    """
    src = gu.element_type
    zero = cutlass.Float32(0.0)
    total = (span + 1) * width
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
            token = lbase + r - 1
            # token < valid implies t0 + token < seqlen, so clamping the token
            # bounds the read above; the row before the sequence bounds it below.
            gbase = t0 + cutlass.min(token, valid - 1)
            got = widen(gu[bidx, hidx, cutlass.max(gbase, 0), p], src)
            if cutlass.const_expr(has_prev):
                got = select(gbase < 0, widen(guprev[bidx, hidx, p], src), got)
            keep = token < valid
            if cutlass.const_expr(not has_prev):
                keep = keep & (gbase >= 0)
            held.append((r, p, keep, got))

        for step in cutlass.range_constexpr(count):
            r, p, keep, got = held[step]
            # The select is float32 because there is one select helper; the
            # operand round trip through float32 is exact at every operand width.
            out = narrow(select(keep, got, zero), src)
            if cutlass.const_expr(exact):
                su[r, p] = out
            else:
                if tid + (group * PREFETCH + step) * threads < total:
                    su[r, p] = out


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
    """Narrow a chunk-start state into an operand tile.

    The state is float32 by I4 and every GEMM operand is low precision, so the
    narrowing happens here, once, on the way into shared memory. The chunk-start
    state is read by one contraction per chunk and never written, so there is no
    accumulation for the narrowing to compound through.

    ``(P, 3N)`` is one contiguous float32 run and the loop walks it at the block
    stride, so a warp covers 512 contiguous bytes per step and no index arithmetic
    survives. The steps run in groups of :data:`PREFETCH`, loads first, so the
    group's loads overlap: this is the largest single read in the operator and a
    serial step-by-step form pays one global latency per element per thread.

    Args:
        gz: ``(P, 3N)`` float32 view of the chunk-start state for one
            ``(chunk, batch, head)``.
        sz: Operand-dtype tile of at least ``width`` rows, written over ``dim``
            columns. The rest of the pitch is outside every view.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        width: Rows to fill, ``P``. Compile-time.
        dim: ``3N``. Compile-time.
    """
    elem = sz.element_type
    total = width * dim
    steps = -(-total // threads)
    exact = total % threads == 0

    for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
        span = min(PREFETCH, steps - group * PREFETCH)
        held = []
        for step in cutlass.range_constexpr(span):
            i = tid + (group * PREFETCH + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            p = i // dim
            held.append((p, i - p * dim))
        got = [gz[p, col] for p, col in held]
        for step in cutlass.range_constexpr(span):
            p, col = held[step]
            if cutlass.const_expr(exact):
                sz[p, col] = narrow(got[step], elem)
            else:
                if tid + (group * PREFETCH + step) * threads < total:
                    sz[p, col] = narrow(got[step], elem)


@cute.jit
def _store_rotated(
    dst: cute.Tensor,
    stable: cute.Tensor,
    sscale: cute.Tensor,
    row: cutlass.Int32,
    token: cutlass.Int32,
    lane: cutlass.Int32,
    vec: Vec3,
    slot: cutlass.Constexpr,
    scaled: cutlass.Constexpr,
) -> None:
    """Transform one 3-vector by one table slot and store it.

    Args:
        dst: Operand-dtype tile, written at ``[row, 3*lane .. 3*lane+2]``.
        stable: ``(mats, L, 9)`` float32 transform table.
        sscale: ``(L,)`` float32 per-token scale. Read only when ``scaled``.
        row: Row of the destination tile.
        token: Chunk-local token index, indexing ``stable`` and ``sscale``. Already
            clamped below ``valid`` by the caller: an M extent rounded up past the
            chunk would otherwise read both tiles out of bounds.
        lane: Which of the ``N`` 3-vectors.
        vec: The 3-vector, already widened to float32 and already zeroed if the
            row carries no token.
        slot: Table slot. Compile-time.
        scaled: Whether to multiply by ``sscale[token]``. Compile-time.
    """
    out = mat3_matvec(
        (
            stable[slot, token, 0],
            stable[slot, token, 1],
            stable[slot, token, 2],
            stable[slot, token, 3],
            stable[slot, token, 4],
            stable[slot, token, 5],
            stable[slot, token, 6],
            stable[slot, token, 7],
            stable[slot, token, 8],
        ),
        vec,
    )
    elem = dst.element_type
    if cutlass.const_expr(scaled):
        weight = sscale[token]
        out = (weight * out[0], weight * out[1], weight * out[2])
    dst[row, 3 * lane] = narrow(out[0], elem)
    dst[row, 3 * lane + 1] = narrow(out[1], elem)
    dst[row, 3 * lane + 2] = narrow(out[2], elem)


@cute.jit
def stage_rotated(
    gv: cute.Tensor,
    gvprev: cute.Tensor,
    dst: cute.Tensor,
    stable: cute.Tensor,
    sscale: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
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
) -> None:
    """Transform a run of tokens by one table slot into a shared operand tile.

    Row ``r`` holds ``A_slot[lbase + r] v[t0 + lbase + r - back]``, optionally
    scaled. ``back`` is 0 for the current tap and the readout, 1 for the previous
    tap: the matrix is indexed at the token it acts on while the previous tap's
    vector comes from the token before it.

    One thread owns one 3-vector: three coalesced global reads, nine FMA, three
    shared-memory stores.

    The pass runs in groups of :data:`PREFETCH` steps, loads first and transforms
    second, so ``3 * PREFETCH`` loads are outstanding when the first of them is
    consumed. Nothing is loaded under a predicate: the index is clamped into range
    and the out-of-range value is replaced afterwards by a select. A load inside a
    divergent branch cannot be hoisted above the branch, and a value produced
    inside one has no phi node to leave through, so the predicated form serializes
    on one global latency per step.

    Rows whose token is at or past ``valid`` are zeroed, which also zeroes the rows
    an M extent was rounded up by, since ``lbase`` is zero whenever ``span``
    exceeds the chunk. A zero row contributes nothing to any contraction, so no
    consumer needs a predicate. Zeroing the float32 input rather than the stored
    output is the same three selects and makes the nine FMA exact.

    Args:
        gv: ``(B,H,T,3N)`` operand-dtype source.
        gvprev: ``(B,H,3N)`` streaming ``v_{-1}``. Read only when ``has_prev`` and
            ``back`` is 1.
        dst: Operand-dtype tile of at least ``span`` rows, written.
        stable: ``(mats, L, 9)`` float32 transform table.
        sscale: ``(L,)`` float32 per-token scale. Read only when ``scaled``.
        bidx: Batch index.
        hidx: Head index.
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
    """
    src = gv.element_type
    zero = cutlass.Float32(0.0)
    total = span * lanes
    steps = -(-total // threads)
    # The staging extents are all multiples of the block width at every legal
    # shape, so the store predicate below is elided. The general form is kept
    # because it costs nothing when it is not needed.
    exact = total % threads == 0
    # g < 0 is reachable only for the previous tap at the first token of the first
    # chunk, which is exactly the streaming carry-in.
    carry = has_prev and back == 1

    for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
        width = min(PREFETCH, steps - group * PREFETCH)
        held = []
        for step in cutlass.range_constexpr(width):
            i = tid + (group * PREFETCH + step) * threads
            if cutlass.const_expr(not exact):
                i = cutlass.min(i, total - 1)
            r = i // lanes
            n = i - r * lanes
            # One clamp serves both reads: valid is at most the chunk, so the
            # clamped token indexes stable and sscale in bounds even when the M
            # extent was rounded up past the chunk, and t0 + it is inside the
            # sequence.
            tsafe = cutlass.min(lbase + r, valid - 1)
            gbase = t0 + tsafe - back
            gsafe = cutlass.max(gbase, 0)
            got = (
                widen(gv[bidx, hidx, gsafe, 3 * n], src),
                widen(gv[bidx, hidx, gsafe, 3 * n + 1], src),
                widen(gv[bidx, hidx, gsafe, 3 * n + 2], src),
            )
            if cutlass.const_expr(carry):
                at_start = gbase < 0
                got = (
                    select(at_start, widen(gvprev[bidx, hidx, 3 * n], src), got[0]),
                    select(at_start, widen(gvprev[bidx, hidx, 3 * n + 1], src), got[1]),
                    select(at_start, widen(gvprev[bidx, hidx, 3 * n + 2], src), got[2]),
                )
            keep = lbase + r < valid
            if cutlass.const_expr(back == 1 and not has_prev):
                keep = keep & (gbase >= 0)
            held.append((r, n, tsafe, keep, got))

        for step in cutlass.range_constexpr(width):
            r, n, tsafe, keep, got = held[step]
            vec = (
                select(keep, got[0], zero),
                select(keep, got[1], zero),
                select(keep, got[2], zero),
            )
            if cutlass.const_expr(exact):
                _store_rotated(dst, stable, sscale, r, tsafe, n, vec, slot, scaled)
            else:
                if tid + (group * PREFETCH + step) * threads < total:
                    _store_rotated(dst, stable, sscale, r, tsafe, n, vec, slot, scaled)
