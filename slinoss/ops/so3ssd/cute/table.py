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
"""

import cutlass
import cutlass.cute as cute

from slinoss._cute import select
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AN,
    TABLE_AP,
    mat3_mul,
    mat3_transpose,
    rot_hom,
    tap_matrix,
)

__all__ = ["build_table", "stage_chunk"]


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
