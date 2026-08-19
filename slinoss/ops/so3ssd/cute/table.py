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
    "build_table",
    "stage_chunk",
    "stage_rotated",
    "stage_shifted",
    "stage_state",
]


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
    lda: cutlass.Constexpr,
    width: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Stage a run of tokens of a ``(T, P)`` tensor into a tile, one row behind.

    Row ``r`` holds token ``t0 + lbase + r - 1``, so one staging pass serves both
    taps: the previous tap reads rows ``0..span-1`` and the current tap reads rows
    ``1..span`` off two views that differ by one row of pitch. The shift is global,
    not per chunk, so row 0 of the first run of a chunk crosses the chunk boundary
    and, at the first chunk, reaches the streaming ``u_prev``.

    Columns past ``width`` and rows past the sequence are zeroed. A zero there
    multiplies a forcing row that is itself zero, so no store is skipped.

    Every store is inside a divergent branch and every ``(r, p)`` pair is owned by
    exactly one thread on exactly one branch. Two guarded stores to the same address
    would race, because the thread that owns the address in the flat loop is not the
    one that owns it in an overlay pass.

    Args:
        gu: ``(B,H,T,P)`` operand-dtype source.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        su: Operand-dtype tile of ``span + 1`` rows and ``lda`` pitch, written.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        lbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        span: Tokens of the run. Compile-time.
        lda: Row pitch of ``su``. Compile-time.
        width: Columns that carry data, ``P``. Compile-time.
        has_prev: Whether ``guprev`` was supplied. Compile-time.
    """
    zero = su.element_type(0.0)
    for i in cutlass.range(tid, (span + 1) * lda, threads):
        r = i // lda
        p = i - r * lda
        # The predicate keeps the read in bounds on its own: token < valid implies
        # t0 + token < seqlen, so no clamp is needed under a divergent branch,
        # where an inactive lane issues no load.
        token = lbase + r - 1
        g = t0 + token
        if (p < width) & (g >= 0) & (token < valid):
            su[r, p] = gu[bidx, hidx, g, p]
        else:
            if cutlass.const_expr(has_prev):
                if (p < width) & (g < 0):
                    su[r, p] = guprev[bidx, hidx, p]
                else:
                    su[r, p] = zero
            else:
                su[r, p] = zero


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
    for i in cutlass.range(tid, width * dim, threads):
        p = i // dim
        sz[p, i - p * dim] = narrow(gz[p, i - p * dim], elem)


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
        token: Chunk-local token index, indexing ``stable`` and ``sscale``.
        lane: Which of the ``N`` 3-vectors.
        vec: The 3-vector, already widened to float32.
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

    Rows whose token is at or past ``valid`` are zeroed, which also zeroes the rows
    an M extent was rounded up by, since ``lbase`` is zero whenever ``span``
    exceeds the chunk. A zero row contributes nothing to any contraction, so no
    consumer needs a predicate.

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
    zero = dst.element_type(0.0)
    for i in cutlass.range(tid, span * lanes, threads):
        r = i // lanes
        n = i - r * lanes
        token = lbase + r
        g = t0 + token - back
        if (token < valid) & (g >= 0):
            _store_rotated(
                dst,
                stable,
                sscale,
                r,
                token,
                n,
                (
                    widen(gv[bidx, hidx, g, 3 * n], src),
                    widen(gv[bidx, hidx, g, 3 * n + 1], src),
                    widen(gv[bidx, hidx, g, 3 * n + 2], src),
                ),
                slot,
                scaled,
            )
        else:
            # g < 0 is reachable only for the previous tap at the first token of
            # the first chunk, which is exactly the streaming carry-in.
            if cutlass.const_expr(has_prev and back == 1):
                if token < valid:
                    _store_rotated(
                        dst,
                        stable,
                        sscale,
                        r,
                        token,
                        n,
                        (
                            widen(gvprev[bidx, hidx, 3 * n], src),
                            widen(gvprev[bidx, hidx, 3 * n + 1], src),
                            widen(gvprev[bidx, hidx, 3 * n + 2], src),
                        ),
                        slot,
                        scaled,
                    )
                else:
                    dst[r, 3 * n] = zero
                    dst[r, 3 * n + 1] = zero
                    dst[r, 3 * n + 2] = zero
            else:
                dst[r, 3 * n] = zero
                dst[r, 3 * n + 1] = zero
                dst[r, 3 * n + 2] = zero
