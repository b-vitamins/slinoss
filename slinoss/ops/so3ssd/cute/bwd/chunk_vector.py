"""``dB``, ``dC``, ``dtrans``, ``dK`` and the forcing-vector carry.

Everything the backward owes the two rowwise vectors and the per-token transition
parameters, in one block per ``(chunk, batch, group)``. The reference terms are
``dcrot``, ``dbnow``, ``dbprv``, ``dac``, ``dan``, ``dap``, ``dtap``, ``dw`` and
``dls``.

Six contractions, all dense real GEMMs off the one atom:

    dm_tap(t,r)  = sum_p  dy(t,p) u_tap(r,p)
    dmT_tap(r,t) = sum_p  u_tap(r,p) dy(t,p)
    dcrot(t,d)   = sum_p  dy(t,p) zstart(p,d)  + sum_r Smasked(t,r) brot_tap(r,d)
    dbnow(r,d)   = wgt(r) sum_p u_tap(r,p) dlocal(p,d)
                                              + sum_t SmaskedT(r,t) crot(t,d)

with ``u_tap`` the current tap at ``r`` and the previous tap at ``r-1``,
``Smasked = dm * dmask``, ``dlocal = R(Q_{L-1})^T dinc`` the increment cotangent
carried back into the chunk-local frame, and ``wgt(r) = exp(2*(lp_{L-1} - lp_r))``.
The offset term opens the readout accumulator and the increment term opens the
forcing accumulator, because both carry a factor that depends on the accumulator's
M mode alone: ``exp(2*lp_t)`` on the first and ``wgt(r)`` on the second, and a
factor applied to a finished sum of two terms would reach the wrong one.

``dm`` is built once and transposed through shared memory. The readout consumer
contracts over the source token and the forcing consumer over the target token, so
each wants the other's N mode as its K mode, and
:func:`slinoss.ops.so3ssd.cute.mma.mma_areg` rereads a fragment in place only along
its own N. The readout consumer takes the fragment; the forcing consumer reads the
same values back through a transposed ``ldmatrix`` from an operand tile that aliases
the tap transfer region. One ``L*L*P`` GEMM per tap instead of two, one float32
accumulator and one operand fragment fewer in the register frame, two barriers more,
and no bytes either way.

``B`` is read once. The two rotated forcing tiles are built from the raw tile in
shared memory by :func:`_rotate_rows`, which is the transform
:func:`slinoss.ops.so3ssd.cute.table.stage_rotated` applies, sourced from shared
memory instead of from global: the raw tile is needed anyway for the tap
cotangent, and the rotation is nine FMA against a second and third pass over
``B``. Bit-identical to the staged form, which widens the same stored value.

``C`` is read once, rotated on the way in, and the raw readout vector is never
read. ``ac`` is a rotation, so ``c = ac^T crot``, and every term that would read
``c`` collapses onto the rotated tile::

    dac = sum_n outer(dcrot_n, c_n) + dap Kprev^T + dan Kcurr^T
        = [sum_n outer(dcrot_n, crot_n) + outer(dbprv_n, bprv_n)
                                       + outer(dbnow_n, bnow_n)] ac

because ``Kcurr b = ac^T bnow`` and ``Kprev bshift = ac^T bprv`` by the same
identity. One 3x3 product per token replaces a second pass over ``C`` and two
tiles. The raw forcing vector survives the collapse only in the tap cotangent,
``ac^T dan = sum_n outer(ac^T dbnow_n, b_n)``, whose second factor no rotation of
``b`` can supply.

The log-scale offset term is never a second pass. With
``gram(t,p) = <crot_t, zstart_p>``,

    dlp_off(t) = 2 exp(2 lp_t) sum_p dy(t,p) gram(t,p)
               = 2 exp(2 lp_t) <crot_t, dcrot_unscaled_t>

so it is one lane reduction against the offset GEMM's own accumulator, taken
before the exponential scales it.

The tap and rotation cotangents are lane reductions over ``N`` of an outer
product, so they are taken in the epilogue that walks the finished vector tile
three columns per thread. ``tap_matrix_vjp`` runs there too: its input is complete
the moment that tap's reduction is, which keeps the scratch at nine floats per
token rather than twenty-seven.

One block walks every head of its group. ``dB``, ``dC`` and the carry are sums
over the heads sharing a group, and neither way of splitting that sum across
blocks exists here: there are no atomics in the tree, and
:mod:`slinoss.ops.so3ssd.cute.bwd.boundary` reduces partials of ``dB`` alone. A
read-modify-write of a low-precision output would also round once per head where
the reference rounds once. So the sum is float32 in shared memory over the fold
``H // G``, the store happens after the last head, and no partial buffer exists.
The fold is one at ``standard`` and twelve at the default configuration.

The state width is tiled. Every tile of either set that spans ``3N`` spans
:data:`LANE_BLOCK` lanes of it, so the live set is bounded by a lane tile and one
launch path serves every ``d_state``. The tile is a grid axis, not a loop: ``3N`` is
the one extent the budget does not bound, so a loop over it multiplies the code by a
count that grows with the model while a grid axis multiplies the blocks. At the
default configuration that is 640 blocks where the loop form launched 128, on an
84-multiprocessor part.

What crosses a lane tile is the transition chart: ``dtrans`` and ``dK`` are sums over
lanes. A loop accumulated them in the row one block owned; separate blocks cannot,
so each tile writes its own float32 slot row and :func:`close_slots` sums the slots
in a second launch. At one tile there are no slots and the store is the output
itself, which is the whole of the difference between the two modes.

Shared memory is one resident set and one phase arena. Resident: ``trans``, ``K``,
the two chunk-local prefixes, the three-slot transform table, the nine-float
per-token scratch, the log-scale and quaternion cotangents, and the rotated
readout. The arena holds the two float32 sums that outlive a head, one region per
tap holding either the float32 forcing gradient or the narrowed score, and five
operand tiles: the output cotangent, one tile that carries the chunk-start state and
then the increment cotangent, the raw and rotated forcing tiles and the ``U`` tile.
The float32 readout gradient of the epilogue aliases those five, none being live
when it is.

The source-token block is :func:`vblock`, one M tile of the atom where the budget
allows it and half of one where it does not. Below one M tile all four warps still
carry rows of every GEMM, because the transposed contractions round their M mode
up to the tile.

The budget bounds ``L``, ``P`` and the fold. It does not bound ``3N``, and this is
still the widest live set in the tree. ``L 16`` and ``L 32`` fit at every ``P``,
every fold and every ``3N``.
``L 64`` fits to ``P 64`` at every fold, in 91,344 B at fold one and 93,392 B above
it, whether ``3N`` is 48 or 240. ``L 64`` at ``P 128`` and ``L 128`` at every ``P``
are refused: the smallest live set at ``L 128`` is 120,528 B, above the capacity of
every device the DSL reports. :func:`slinoss._cute.assert_smem_fits` refuses the
rest rather than any path here degrading.

The largest ``L`` this layout admits is 64, at one resident block, at ``P 64`` and at
``P 48`` alike. Two resident blocks of 128 threads need 50,688 B of the 101,376 B
carveout, and no legal shape at ``P 64`` reaches it: the smallest arena there is
53,648 B, at ``L 16`` and fold one. Splitting the fold across blocks removes the one
accumulator that exists only above fold one, 13,312 B of the 93,392, and leaves
80,080 B, so it buys parallelism and not occupancy. Four extents scale with ``L`` --
the two float32 fold sums, the output tile and the aliased float32 readout -- which
is why 128 is refused and why grid-izing the lane tile does not unpin it.

DRAM-bound. Analytic traffic at ``standard``, operand by operand, with ``U`` and
``B`` at the ``L + 1`` rows per chunk their shifted span reads::

    reads   dy 9.44 + U 9.58 + B 9.58 + C 9.44 + trans 1.57 + K 3.15
          + dinc 14.16 + zstart 14.16 + dlogp 0.39 + dchunk_rot 0.06
          + dchunk_scale 0.01                                        = 71.53 MB
    writes  dB 9.44 + dC 9.44 + dtrans 1.57 + dK 3.15 + carry_b 0.29 = 23.89 MB

95.42 MB against ``1536 * 4.03 MFLOP = 6.19 GFLOP``, so 64.9 flop/byte against a
ridge point of 164: memory bound by a factor of 2.5. That table is the ``span 64``
form. A shape whose budget forces ``span 32`` doubles the ``U`` term, since the
``U`` tile is one atom M tile whatever the block, and raises the intensity to 108
flop/byte at the default configuration, still under the ridge.

It is also the one-lane-tile form. Every operand carrying ``3N`` is read once
whatever the tile count, since each tile reads its own columns; the operands with no
state extent are read once per tile. At the scale above that is ``dy``, ``U``,
``trans``, ``K``, ``dlogp`` and the two chunk cotangents, 24.20 MB. ``standard`` is
one tile, so the table stands there as written.

At the default configuration, ``B 4 H 18 T 2048 P 64 3N 240 L 64`` chunk 64 with one
group, there are five tiles and the split is what the tile count costs::

    per tile x5   dy 18.87 + U 38.34 + trans 2.36 + K 4.72 + dlogp 0.59
                + dchunk_rot 0.08 + dchunk_scale 0.01     = 64.97 -> 324.86 MB
    once          B 3.99 + C 3.93 + dinc 141.56 + zstart 141.56 r,
                  dB 3.93 + dC 3.93 + carry_b 0.12 w      =           298.99 MB
    slot rows     dtrans 2.36 + dK 4.72 written and read back per tile,
                  then 7.08 written out once               =           77.85 MB

701.75 MB, and the slot rows are 14.16 MB more than the loop form's
read-modify-write of the same two outputs, 2.1%. ``U`` dominates the per-tile term
because its tile is one atom M tile whatever the ``span``, so a ``span 32`` shape
reads it twice. ``dinc`` and ``zstart`` are float32 ``(B, H, C, P, 3N)`` and together
are 40% of the total.

Measured, the bar is missed, and the distance is latency and not traffic. Every
counter below is from one profile of this kernel on an RTX A6000, ``sm_86``, 84
multiprocessors, six profiled launches per counter pass, clocks unlocked because
locking is denied on this fleet. A floor is ``c + bytes / B`` on a fit taken in the
same process, ``c`` about 4.3 us and ``B`` about 685 GB/s at a worst residual of
0.40%, and ``bytes`` is the analytic traffic above unless stated otherwise. A
duration is stamped with the compute-apps query taken before and after it; where
that query named another process the duration is a bound, not a rate.

At the default configuration, 640 blocks of 128 threads, five lane tiles, fold 18:
985.73 MB of DRAM per launch against the 701.75 MB of the table, 40% over, and
11,982.6 us. That is 8.6% of the 1,028.5 us floor of the table and 12.0% of the
1,442.9 us floor of the bytes the counters move. Both are far under the 85% the
class asks.

Neither percentage is a traffic problem. 93,392 B admits one 128-thread block per
multiprocessor: 8.330% theoretical occupancy, 8.330% achieved, one warp per
scheduler and nothing to cover an instruction fetch or a barrier. ``no_instruction``
is 64.1% of the warp cycles, issue-active 8.80%, memory speed-of-light 21.9%, tensor
3.5%. Shared conflicts are 0.1556 per wavefront, load and store together, and the
padding rule is held centrally.

The allocator stops at the 255-register cap in every compile and spills 479.07 MB of
local load and 443.35 MB of local store per launch, which fails the spill rule
whatever the percentage says. The spill is a live set, not a loop form: recompiling
with every device loop at trip count one raises the local traffic to 1,290.4 MB and
the DRAM to 1,654.10 MB, and the register count and the footprint do not move.
Neither does occupancy move with the tile count -- 255 registers and 8.330% at
``3N 48`` and at ``3N 240`` alike -- and at ``3N 48``, one lane tile, the local
traffic is 276.96 MB and the launch is 2,999.2 us against a 196.8 us floor.

Closing the gap needs two resident blocks, which needs 50,688 B. The smallest arena
at ``P 64`` is 53,648 B, at ``L 16`` and fold one, so no legal shape reaches it.
Splitting the fold across blocks leaves 80,080 B. Splitting the kernel by output
group would: the transition chart needs the five float32 ``L`` tiles, the table, the
scratch and the score tile, and the state outputs need the two float32 fold sums and
the four operand tiles, and neither half holds the other's. It costs a second read
of ``dy``, ``U``, ``trans`` and ``K``, 324.86 MB at this configuration, for a floor
of about 1,027 MB against today's 701.75. The other candidate is 256 threads per
block, which is the atom tiling's shape and not this file's: at eight warps the M
tile doubles and the same live set is 128,432 B, off the device. The declared class
follows the traffic; the figures above are what the kernel reaches against it.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Scalar,
    Stream,
    Tile,
    assert_smem_fits,
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
from slinoss._reduce import reduce_partials
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AN,
    TABLE_AP,
    THREADS,
    Mat3,
    Vec3,
    mat3_add,
    mat3_matvec,
    mat3_mul,
    mat3_outer,
    mat3_transpose,
    quat_exp_vjp,
    rot_hom_vjp,
    scalar_tile,
    table_tile,
    tap_matrix_vjp,
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
    MMA_TILE_M,
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
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes, chunk_suffix, quat_suffix_vjp
from slinoss.ops.so3ssd.cute.table import (
    build_table,
    stage_chunk,
    stage_matrix,
    stage_rotated,
    stage_shifted,
    stage_state,
)
from slinoss.ops.so3ssd.reference import check_grad_band

__all__ = [
    "LANE_BLOCK",
    "LANE_GROUP",
    "RESIDENT_MAX",
    "ROW_WORDS",
    "Arena",
    "ChunkVectorBwd",
    "Slots",
    "arena",
    "chunk_vector_backward",
    "chunk_vector_bwd",
    "chunk_vector_bwd_kernel",
    "close_slots",
    "forced_tile",
    "gradient_tile",
    "lane_block",
    "open_slots",
    "out_tile",
    "readout_tile",
    "row_tile",
    "score_tile",
    "shifted_tile",
    "state_tile",
    "vblock",
    "vector_smem_bytes",
]

LANE_BLOCK: int = 16
"""Lanes of one lane tile: the block the resident set is bounded by, not ``N``.

Every tile that spans ``3N`` is cut to ``3 * LANE_BLOCK`` columns and the kernel
loops over the cut, so the footprint is flat in ``3N`` and one launch path serves
every state width. 16 lanes is the smallest legal tile: 48 columns is a multiple of
the atom's N mode and of the 3-vector, and no smaller multiple of 3 is a multiple
of 16. It is also the largest tile that divides every legal ``3N``, since ``3N`` is
a multiple of 48 and 96 does not divide 240.

The cut is along a mode the six contractions carry as N or M, never as K, so no
partial sum crosses a tile. What crosses is the transition chart: ``dtrans`` and
``dK`` are sums over lanes, so a tile past the first accumulates into them."""

LANE_GROUP: int = 16
"""Threads that cooperate on one token in a rowwise epilogue.

``N`` is a multiple of 16 at every legal ``3N``, so this divides the lane count
whatever the shape, and a 16-lane butterfly stays inside a warp. One thread holds
one 3-vector, which is what the rowwise transforms and the outer products need and
what an accumulator fragment cannot give: the atom hands a thread two adjacent
columns, and a 3-vector straddles that pair."""

ROW_WORDS: int = 9
"""Float32 scratch per token: the 3x3 rotation cotangent, summed over ``N``.

The tap cotangents do not appear because ``tap_matrix_vjp`` runs inside the
epilogue that reduces them, so only the rotation's own sum outlives a phase.

The pitch is this count itself. One thread owns one token's nine words, so the
warp reading word ``k`` touches banks ``(9*tid + k) mod 32``; nine and 32 are
coprime, so 32 consecutive tokens land in 32 distinct banks. That is a property of
the pitch rather than a counter reading, and it is why the count is not rounded up
to a power of two. It holds at every ``L``: the chunk length sets how many lanes
the ``token < chunk`` guard leaves active, and masking lanes off removes accesses
rather than colliding them."""

RESIDENT_MAX: int = 2
"""Blocks per SM the launch asks for, before the shared-memory budget lowers it.

The budget lowers it to one at every standard size. Asking for two costs nothing
where it cannot be had and takes it at the small shapes where the arena is half as
wide."""


def lane_block(dim: int) -> int:
    """Columns of one lane tile, ``min(3N, 3 * LANE_BLOCK)``.

    Divides ``3N`` at every legal shape:
    :func:`slinoss.ops.so3ssd.cute.guard.check_extents` holds ``3N`` to a multiple
    of 3 and of 16, which are coprime, so ``3N`` is a multiple of 48.

    Args:
        dim: ``3N``.
    """
    return min(dim, 3 * LANE_BLOCK)


def row_tile(chunk: int) -> Tile:
    """Per-token float32 scratch, ``(L, ROW_WORDS)``."""
    return Tile((chunk, ROW_WORDS), (ROW_WORDS, 1))


def readout_tile(chunk: int, dim: int) -> Tile:
    """Rotated readout tile, ``(mma_rows(L), pitch)``.

    An M mode of the offset and the readout GEMM and a K mode of the forcing GEMM,
    hence the rounded row count.

    Args:
        chunk: ``L``.
        dim: ``3N``.
    """
    return operand_tile(mma_rows(chunk), dim)


def forced_tile(span: int, dim: int) -> Tile:
    """Rotated forcing tile, ``(span, pitch)``.

    Only ever an N mode, of the readout GEMM, so the row count is the block
    itself.

    Args:
        span: Tokens of the source-token block.
        dim: ``3N``.
    """
    return operand_tile(span, dim)


def shifted_tile(span: int, width: int) -> Tile:
    """Shifted staging tile, ``(span + 1, pitch)``.

    Row ``j`` holds token ``nbase + j - 1``, so the previous tap reads rows
    ``0..span-1`` and the current tap the same rows one further on.

    Args:
        span: Tokens of the run.
        width: ``P`` or ``3N``.
    """
    return operand_tile(span + 1, width)


def score_tile(chunk: int, span: int) -> Tile:
    """Narrowed score tile, ``(mma_rows(L), pitch of mma_rows(span))``.

    The masked score, target token by source token, on its way from the readout
    orientation to the forcing one. Both modes are rounded up: the target token is
    the M mode of the GEMM that writes it and the K mode of the GEMM that reads it,
    and the source token is the read's M mode, which the transposed
    ``ldmatrix`` needs a whole atom tile of whatever the source-token block is.

    Args:
        chunk: ``L``.
        span: Tokens of the source-token block.
    """
    return operand_tile(mma_rows(chunk), mma_rows(span))


def state_tile(rows: int, dim: int) -> Tile:
    """Chunk-start state or increment cotangent, ``(P, pitch)``.

    ``P`` is a K mode of both GEMMs that read these, never an M mode.

    Args:
        rows: ``P``.
        dim: ``3N``.
    """
    return operand_tile(rows, dim)


def out_tile(chunk: int, rows: int) -> Tile:
    """Output cotangent tile, ``(mma_rows(L), pitch)``.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return operand_tile(mma_rows(chunk), rows)


def gradient_tile(rows: int, dim: int) -> Tile:
    """Float32 tile a vector gradient passes through, ``(rows, pitch)``.

    Float32 because it is not an operand. It is a gradient on its way to ``dB``,
    ``dC`` and the transition parameters, and the reference rounds that once, at
    the store; a second rounding here would double the error on every term it
    feeds, including the two float32 outputs.

    Args:
        rows: Rows to allocate.
        dim: ``3N``.
    """
    return fp32_tile(rows, dim)


class Arena(NamedTuple):
    """Float32-word offsets of the phase-shared tiles inside the one arena.

    The tiles below overlap in address and not in time. The three float32 tiles
    come first and alias nothing: two of them are live across the whole fold and
    the third carries one tap. The five operand tiles follow, and the readout
    gradient of the epilogue aliases all five, none being live when it is.

    ``state`` holds the chunk-start state through the offset contraction and the
    increment cotangent for the rest of a head's pass. One tile rather than two:
    the two have the same extents, neither is read while the other is being
    written, and the barrier that separates them is the one the source-token loop
    needs anyway.

    ``summed`` spans no words at ``fold == 1``, where the readout gradient goes
    straight to global and nothing reads the tile; its offset then aliases
    ``out``.

    Attributes:
        forcing: The float32 forcing gradient, summed over taps, blocks and the
            fold. Row ``t + 1`` is token ``t`` and row 0 is the row that crosses
            the chunk boundary.
        tapped: The float32 forcing gradient of one tap, the GEMM's own output, and
            the narrowed score of that same tap, which the forcing GEMM has finished
            reading before the gradient it writes exists. The region is the wider of
            the two.
        summed: The float32 readout gradient summed over the fold.
        out: The output cotangent, ``dy``.
        state: The chunk-start state, then the increment cotangent in the
            chunk-local frame.
        raw: The raw forcing tile, restaged once per source-token block.
        forced: The rotated forcing tile, rebuilt once per tap.
        input: The shifted ``U`` tile.
        readout: The float32 readout gradient of one head. Epilogue only.
        words: Float32 words the arena spans.
    """

    forcing: int
    tapped: int
    summed: int
    out: int
    state: int
    raw: int
    forced: int
    input: int
    readout: int
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


def arena(
    chunk: int, rows: int, dim: int, fold: int, span: int, itemsize: int = 2
) -> Arena:
    """Lay the phase-shared tiles out in one allocation.

    Every tile that spans the state width spans :func:`lane_block` of it, so this is
    flat in ``3N`` above the first lane tile.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        fold: Heads one block walks, ``H // G``.
        span: Source-token block, from :func:`vblock`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    tile = lane_block(dim)
    forcing = _words(gradient_tile(chunk + 1, tile), 4)
    tapped = max(
        _words(gradient_tile(span, tile), 4),
        _words(score_tile(chunk, span), itemsize),
    )
    summed = _words(gradient_tile(chunk, tile), 4) if fold > 1 else 0
    out = _words(out_tile(chunk, rows), itemsize)
    state = _words(state_tile(rows, tile), itemsize)
    raw = _words(shifted_tile(span, tile), itemsize)
    forced = _words(forced_tile(span, tile), itemsize)
    inp = _words(shifted_tile(mma_rows(span), rows), itemsize)
    base = forcing + tapped + summed
    return Arena(
        forcing=0,
        tapped=forcing,
        summed=forcing + tapped,
        out=base,
        state=base + out,
        raw=base + out + state,
        forced=base + out + state + raw,
        input=base + out + state + raw + forced,
        readout=base,
        words=base
        + max(
            out + state + raw + forced + inp,
            _words(gradient_tile(mma_rows(chunk), tile), 4),
        ),
    )


def vector_smem_bytes(
    chunk: int, rows: int, dim: int, fold: int, span: int, itemsize: int = 2
) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_vector_bwd_kernel` allocates, in the same order.
    Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        fold: Heads one block walks, ``H // G``.
        span: Source-token block, from :func:`vblock`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (table_tile(chunk, 3), 4),
            (row_tile(chunk), 4),
            (readout_tile(chunk, lane_block(dim)), itemsize),
            (
                Tile(
                    (arena(chunk, rows, dim, fold, span, itemsize).words,),
                    (1,),
                ),
                4,
            ),
        ]
    )


def vblock(chunk: int, rows: int, dim: int, fold: int, itemsize: int = 2) -> int:
    """Source-token block: one M tile of the atom, or half of one to fit.

    ``min(L, MMA_TILE_M)`` is the block every mode of every GEMM is happiest at,
    and it is taken wherever the budget holds it. Where it does not, the block
    halves once, which is the only other candidate: four of the nine regions the
    arena holds scale with the block, and only two of those four keep scaling below
    half an M tile, since the other two round the block back up to one. At
    ``L 64/P 48/3N 48`` and fold one the first halving buys 11,264 B and a second
    would buy 3,584 B for a fourth pass over ``U``. Both candidates divide ``L`` and
    are multiples of the atom's K and N modes, which is what
    :func:`slinoss.ops.so3ssd.cute.guard.check_extents` and
    :func:`slinoss.ops.so3ssd.cute.mma.mma_areg` require.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        fold: Heads one block walks, ``H // G``.
        itemsize: Bytes per operand element.

    Returns:
        The block. A shape that fits at neither candidate is refused by
        :func:`slinoss._cute.assert_smem_fits`, not run at a third.
    """
    span = min(chunk, MMA_TILE_M)
    floor = min(chunk, MMA_TILE_M // 2)
    if span > floor:
        budget = vector_smem_bytes(chunk, rows, dim, fold, span, itemsize)
        if budget > smem_capacity():
            span = floor
    return span


class Slots(NamedTuple):
    """One output and the slot rows the grid writes it through.

    The lane tile is a grid axis, so the blocks holding the terms of a sum over
    lanes are concurrent and none can see the others' partials. Each such output
    gains a slot axis immediately outside its row axis: a block writes row
    ``slot * rows + local``, and :func:`close_slots` sums the ``slots`` copies of a
    row into the output's own.

    At ``slots == 1`` there is no buffer and ``dest`` is the output. The row index is
    the same expression either way, ``slot`` being zero, so the kernel body does not
    know which mode it is in and neither mode carries a branch.

    Outside the row axis rather than inside it, because the reduction's slab count is
    what its grid puts on the y axis and that axis stops at 65,535. Inside, one slab
    is one row: at ``B 4 H 18 T 2048`` that is 147,456 slabs of four columns and the
    launch is refused. Outside, a slab is a head: 72 slabs of ``T*4`` columns, the
    grid over columns carries the parallelism, and the block count rises rather than
    the slab count.

    Only ``dtrans`` and ``dK`` reach here. Every other output of this kernel spans the
    state width, so a lane tile owns its own columns and no sum crosses a block.

    Attributes:
        dest: What the kernel writes. The output at one slot, a float32 partial
            buffer above one.
        slots: Copies of each row, one per lane tile.
        out: The output the sum closes onto, or None at one slot.
        slabs: ``S`` of the reduction, the modes before the slot axis.
        width: ``W`` of the reduction, the row axis and everything after it.
    """

    dest: Tensor
    slots: int
    out: Tensor | None
    slabs: int
    width: int


def open_slots(out: Tensor, slots: int, axis: int = -2) -> Slots:
    """Allocate the slot rows one output needs, or none.

    Args:
        out: The output, contiguous.
        slots: Copies of each row, the lane tile count.
        axis: Row axis of ``out``, the mode the slot count multiplies. ``-2`` where
            one trailing mode follows it, ``-3`` where two do.

    Returns:
        :class:`Slots`. ``torch.empty``, never zeroed: the kernel writes every
        element of every slot row it allocates, one block per row.
    """
    shape = [int(extent) for extent in out.shape]
    axis %= len(shape)
    slabs, width = 1, 1
    for extent in shape[:axis]:
        slabs *= extent
    for extent in shape[axis:]:
        width *= extent
    if slots == 1:
        return Slots(dest=out, slots=1, out=None, slabs=slabs, width=width)
    held = torch.empty(
        *shape[:axis],
        shape[axis] * slots,
        *shape[axis + 1 :],
        dtype=torch.float32,
        device=out.device,
    )
    return Slots(dest=held, slots=slots, out=out, slabs=slabs, width=width)


def close_slots(held: Slots) -> Tensor:
    """Sum a slot buffer onto its output, in one launch, or return the output.

    Args:
        held: From :func:`open_slots`.

    Returns:
        The output.
    """
    if held.out is None:
        return held.dest
    reduce_partials(
        held.dest.view(held.slabs, held.slots, held.width),
        out=held.out.view(held.slabs, held.width),
    )
    return held.out


def _lane_view(src: cute.Tensor, joff: cutlass.Int32) -> cute.Tensor:
    """The same global tensor with its lane origin moved to a lane tile.

    Every tensor this is applied to carries ``3N`` as its last mode at unit stride,
    so a column offset is a pointer offset and the layout is the one the tensor
    arrived with. The staging helpers and the stores then index the lane tile's own
    columns, and none of them learns that the state is wider.

    Undecorated, so the offset folds into the addresses the trace already builds.

    Args:
        src: Global tensor whose last mode is ``3N``.
        joff: First column of the lane tile.
    """
    return cute.make_tensor(src.iterator + joff, src.layout)


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


def _sum_over_n(value: Scalar) -> Scalar:
    """Sum one accumulator element over the four lanes that share its row.

    The atom gives the four lanes of an aligned quad the same accumulator row and
    disjoint columns, so two butterfly rounds leave that row's partial column sum
    in all four. Rows are disjoint across quads and across warps, so the
    read-modify-write that follows is by one thread per row and needs no barrier.

    Args:
        value: The lane's contribution.
    """
    value = value + shuffle_xor(value, 1)
    return value + shuffle_xor(value, 2)


def _sum_over_lanes(vals: tuple[Scalar, ...]) -> tuple[Scalar, ...]:
    """Sum a tuple of floats over the :data:`LANE_GROUP` lanes of one token.

    Undecorated: the round count is compile-time and the loop is unrolled during
    the trace.

    Args:
        vals: One value per component, this lane's partial.

    Returns:
        The group's sum, in every lane of the group.
    """
    out = vals
    reach = 1
    while reach < LANE_GROUP:
        out = tuple(v + shuffle_xor(v, reach) for v in out)
        reach *= 2
    return out


def _mat_at(stable: cute.Tensor, slot: int, token: cutlass.Int32) -> Mat3:
    """One transform-table entry as a 3x3, row-major.

    Args:
        stable: ``(mats, L, 9)`` float32 table.
        slot: Table slot. Compile-time.
        token: Chunk-local token, already bounded by ``L``.
    """
    return (
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


def _vec_at(src: cute.Tensor, row: cutlass.Int32, col: cutlass.Int32) -> Vec3:
    """One lane's 3-vector of a shared tile, widened to float32.

    Args:
        src: Shared tile.
        row: Row.
        col: First of the three columns.
    """
    elem = src.element_type
    return (
        widen(src[row, col], elem),
        widen(src[row, col + 1], elem),
        widen(src[row, col + 2], elem),
    )


@cute.jit
def _fill_zero(
    dst: cute.Tensor, total: cutlass.Constexpr, tid: cutlass.Int32, threads: int
) -> None:
    """Zero a dense shared tile, padding included.

    Args:
        dst: Shared float32 tile whose storage is dense.
        total: Elements the tile spans, padding included. Compile-time.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
    """
    flat = cute.make_tensor(dst.iterator, cute.make_layout((total,), stride=(1,)))
    for step in cutlass.range_constexpr(-(-total // threads)):
        i = tid + step * threads
        if cutlass.const_expr(total % threads == 0):
            flat[i] = 0.0
        else:
            if i < total:
                flat[i] = 0.0


@cute.jit
def _rotate_rows(
    src: cute.Tensor,
    dst: cute.Tensor,
    stable: cute.Tensor,
    tid: cutlass.Int32,
    nbase: cutlass.Int32,
    slot: cutlass.Constexpr,
    shift: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Transform a shifted shared tile by one table slot into another.

    ``dst[r] = A_slot[nbase + r] src[r + shift]``, which is what
    :func:`slinoss.ops.so3ssd.cute.table.stage_rotated` writes from global. Reading
    the raw tile the tap cotangent needs anyway costs nine FMA a lane and saves a
    pass over ``B`` per tap.

    Rows of ``src`` past the chunk's valid tokens are already zero, so the rows an
    M extent was rounded up by stay zero and no consumer needs a predicate.

    Args:
        src: Operand-dtype shifted tile, row ``j`` holding token ``nbase + j - 1``.
        dst: Operand-dtype tile of at least ``span`` rows, written.
        stable: ``(mats, L, 9)`` float32 transform table.
        tid: Thread index within the block.
        nbase: First chunk-local token of the run.
        slot: Table slot. Compile-time.
        shift: Row offset into ``src``, which is the tap index: the previous tap
            takes the row before the token and the current tap the token's own.
            Compile-time.
        threads: Block width. Compile-time.
        span: Rows of ``dst`` to fill. Compile-time.
        lanes: ``N``. Compile-time.
    """
    elem = dst.element_type
    total = span * lanes
    exact = total % threads == 0

    for step in cutlass.range_constexpr(-(-total // threads)):
        i = tid + step * threads
        if cutlass.const_expr(not exact):
            i = cutlass.min(i, total - 1)
        r = i // lanes
        n = i - r * lanes
        col = 3 * n
        out = mat3_matvec(
            _mat_at(stable, slot, nbase + r), _vec_at(src, r + shift, col)
        )
        if cutlass.const_expr(exact):
            for j in cutlass.range_constexpr(3):
                dst[r, col + j] = narrow(out[j], elem)
        else:
            if tid + step * threads < total:
                for j in cutlass.range_constexpr(3):
                    dst[r, col + j] = narrow(out[j], elem)


@cute.jit
def _tap_epilogue(
    gdtap: cute.Tensor,
    sdb: cute.Tensor,
    sbrot: cute.Tensor,
    sb: cute.Tensor,
    ssum: cute.Tensor,
    stable: cute.Tensor,
    stap: cute.Tensor,
    strans: cute.Tensor,
    srow: cute.Tensor,
    sdw: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    nbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    jbase: cutlass.Int32,
    tap: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Turn one tap's finished forcing gradient into ``dB``, ``dK`` and two sums.

    Per token and lane, with ``atap`` the tap's table slot and ``ac`` the readout
    slot::

        dbs      = atap^T dbnow                 into the forcing sum
        rotation += outer(dbnow, brot)          into the nine-float scratch
        tap, w    = tap_matrix_vjp(sum_n outer(ac^T dbnow, b), tap, w)

    The rotation term is the collapsed form: ``outer(dbnow, ac^T bnow) ac`` with
    the trailing ``ac`` deferred to the one place that applies it, so no raw
    readout vector is read. The tap term does not collapse, which is the only
    reason the raw forcing tile is staged.

    The forcing sum is indexed by token rather than by row of the run: row ``t + 1``
    is token ``t``, so the current tap lands one row above the previous tap's and
    the previous tap of the chunk's first token lands on row 0, which is the carry.

    Args:
        gdtap: ``(B,H,tiles*T,2,4)`` float32 ``dK`` or its slot buffer, written at
            this tap.
        sdb: ``(span, pitch)`` float32 forcing gradient, the GEMM's output.
        sbrot: ``(span, pitch)`` operand-dtype rotated forcing tile.
        sb: ``(span + 1, pitch)`` operand-dtype raw forcing tile.
        ssum: ``(L + 1, pitch)`` float32 forcing sum, accumulated.
        stable: ``(mats, L, 9)`` float32 transform table.
        stap: ``(8, L)`` float32 tap parameters, component-major.
        strans: ``(4, L)`` float32 ``(w, ls)``, component-major.
        srow: ``(L, ROW_WORDS)`` float32 rotation scratch, accumulated.
        sdw: ``(4, L)`` float32 rotation-vector cotangent, accumulated.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        nbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        jbase: First row of this block's slot, ``jstep * T``. Zero at one lane tile,
            where the destination is ``dK`` itself.
        tap: 0 for the previous tap and 1 for the current, the order ``K`` packs
            them in. Compile-time.
        threads: Block width. Compile-time.
        span: Tokens of the run. Compile-time.
        lanes: Lanes of the lane tile. Compile-time.
    """
    slot = TABLE_AP if cutlass.const_expr(tap == 0) else TABLE_AN
    # The previous tap's gradient belongs to the token before its own, which is one
    # row of the shifted tiles back, so the row offset is the tap index itself.
    shift = tap
    per_pass = threads // LANE_GROUP
    exact = span % per_pass == 0
    lane = tid % LANE_GROUP
    zero = cutlass.Float32(0.0)

    for step in cutlass.range_constexpr(-(-span // per_pass)):
        r = tid // LANE_GROUP + step * per_pass
        # Clamped rather than branched: a row past the run reads real data whose
        # every use below is predicated away.
        rs = cutlass.min(r, span - 1)
        token = nbase + rs
        inside = r < span
        gsum = tuple(zero for _ in range(9))
        msum = tuple(zero for _ in range(9))
        act = mat3_transpose(_mat_at(stable, TABLE_AC, token))
        atapt = mat3_transpose(_mat_at(stable, slot, token))
        for rep in cutlass.range_constexpr(lanes // LANE_GROUP):
            col = 3 * (lane + rep * LANE_GROUP)
            dvec = (sdb[rs, col], sdb[rs, col + 1], sdb[rs, col + 2])
            gsum = mat3_add(gsum, mat3_outer(dvec, _vec_at(sbrot, rs, col)))
            msum = mat3_add(
                msum,
                mat3_outer(mat3_matvec(act, dvec), _vec_at(sb, rs + shift, col)),
            )
            out = mat3_matvec(atapt, dvec)
            if cutlass.const_expr(exact):
                for j in cutlass.range_constexpr(3):
                    ssum[token + shift, col + j] += out[j]
            else:
                if inside:
                    for j in cutlass.range_constexpr(3):
                        ssum[token + shift, col + j] += out[j]
        gsum = _sum_over_lanes(gsum)
        msum = _sum_over_lanes(msum)
        keep = lane == 0
        if cutlass.const_expr(not exact):
            keep = keep & inside
        if keep:
            for k in cutlass.range_constexpr(ROW_WORDS):
                srow[token, k] += gsum[k]
            dtap, dw = tap_matrix_vjp(
                msum,
                (
                    stap[4 * tap, token],
                    stap[4 * tap + 1, token],
                    stap[4 * tap + 2, token],
                ),
                (strans[0, token], strans[1, token], strans[2, token]),
            )
            for j in cutlass.range_constexpr(3):
                sdw[j, token] += dw[j]
            if token < valid:
                krow = jbase + t0 + token
                for j in cutlass.range_constexpr(3):
                    gdtap[bidx, hidx, krow, tap, j] = dtap[j]
                # Lane 3 of K is a hard zero in the forward, so it is one here.
                gdtap[bidx, hidx, krow, tap, 3] = zero


@cute.jit
def _readout_epilogue(
    gdc: cute.Tensor,
    sdc: cute.Tensor,
    ssum: cute.Tensor,
    scrot: cute.Tensor,
    stable: cute.Tensor,
    srow: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    fold: cutlass.Constexpr,
) -> None:
    """Turn one head's finished readout gradient into ``dC`` and a rotation sum.

    Per token and lane, ``dc = ac^T dcrot`` and ``rotation += outer(dcrot,
    crot)``, the same collapsed form the tap epilogue accumulates into.

    A single head writes ``dC`` in the output dtype directly. A fold above one
    accumulates in float32 instead, because the group's ``dC`` is a sum over its
    heads and the reference rounds that sum once.

    Args:
        gdc: ``(B,G,T,3N)`` ``dC``, written when ``fold`` is one.
        sdc: ``(mma_rows(L), pitch)`` float32 readout gradient.
        ssum: ``(L, pitch)`` float32 readout sum over the fold, accumulated when
            ``fold`` is above one and untouched otherwise.
        scrot: ``(mma_rows(L), pitch)`` operand-dtype rotated readout.
        stable: ``(mats, L, 9)`` float32 transform table.
        srow: ``(L, ROW_WORDS)`` float32 rotation scratch, accumulated.
        bidx: Batch index.
        gidx: Group index.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        lanes: ``N``. Compile-time.
        fold: Heads one block walks. Compile-time.
    """
    out = gdc.element_type
    per_pass = threads // LANE_GROUP
    exact = chunk % per_pass == 0
    lane = tid % LANE_GROUP
    zero = cutlass.Float32(0.0)

    for step in cutlass.range_constexpr(-(-chunk // per_pass)):
        token = tid // LANE_GROUP + step * per_pass
        ts = cutlass.min(token, chunk - 1)
        inside = token < chunk
        gsum = tuple(zero for _ in range(9))
        act = mat3_transpose(_mat_at(stable, TABLE_AC, ts))
        for rep in cutlass.range_constexpr(lanes // LANE_GROUP):
            col = 3 * (lane + rep * LANE_GROUP)
            dvec = (sdc[ts, col], sdc[ts, col + 1], sdc[ts, col + 2])
            gsum = mat3_add(gsum, mat3_outer(dvec, _vec_at(scrot, ts, col)))
            dc = mat3_matvec(act, dvec)
            keep = ts < valid
            if cutlass.const_expr(not exact):
                keep = keep & inside
            if keep:
                for j in cutlass.range_constexpr(3):
                    if cutlass.const_expr(fold == 1):
                        gdc[bidx, gidx, t0 + ts, col + j] = narrow(dc[j], out)
                    else:
                        ssum[ts, col + j] += dc[j]
        gsum = _sum_over_lanes(gsum)
        keep = lane == 0
        if cutlass.const_expr(not exact):
            keep = keep & inside
        if keep:
            for k in cutlass.range_constexpr(ROW_WORDS):
                srow[ts, k] += gsum[k]


@cute.kernel
def chunk_vector_bwd_kernel(
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
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    gdtrans: cute.Tensor,
    gdtap: cute.Tensor,
    seqlen: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    fold: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Differentiate one chunk's rowwise vectors and transition parameters.

    One block per ``(chunk, lane tile, batch, group)``, walking the ``fold`` heads of
    the group. Everything a head owns alone is rebuilt per head; the two vector sums
    and the carry are the group's, outlive the fold and belong to the block's own lane
    tile. ``dtrans`` and ``dK`` are the only outputs a lane tile does not own: both are
    sums over lanes, so each tile writes its own slot row and
    :func:`close_slots` sums them.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gu: ``(B,H,T,P)`` operand-dtype forcing input.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 ``(kr, g, h, 0)`` per tap.
        gb: ``(B,G,T,3N)`` operand-dtype forcing vectors.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``. Read only when ``has_prev``.
        gc: ``(B,G,T,3N)`` operand-dtype readout vectors.
        gdinc: ``(B,H,C,P,3N)`` float32 increment cotangent, global frame.
        gz: ``(B,H,C,P,3N)`` float32 chunk-start state.
        gdlp: ``(B,H,C,L)`` float32 diagonal and increment half of the log-scale
            cotangent, from the chunk-input stage.
        gdrot: ``(B,H,C,3,3)`` float32 closing-rotation cotangent, row-major, from
            the chunk-input stage.
        gdscale: ``(B,H,C)`` float32 closing-scale cotangent, from the chunk-input
            stage.
        gdb: ``(B,G,T,3N)`` ``dB``, written.
        gdc: ``(B,G,T,3N)`` ``dC``, written.
        gcarry: ``(B,G,C,3N)`` float32, written with the forcing gradient of the
            token before the chunk's first.
        gdtrans: ``(B,H,tiles*T,4)`` float32 ``dtrans`` or its slot buffer, written.
        gdtap: ``(B,H,tiles*T,2,4)`` float32 ``dK`` or its slot buffer, written.
        seqlen: ``T``. Dynamic.
        tiled_mma: From :func:`slinoss.ops.so3ssd.cute.mma.make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        span: Source-token block, :func:`vblock`. Compile-time.
        fold: Heads of one group, ``H // G``. Compile-time.
        has_prev: Whether the streaming carry-in was supplied. Compile-time.

    Invariants:
        ``chunk`` is a multiple of ``span`` and of :data:`MMA_TILE_K`, ``dim`` and
        ``rows`` are multiples of :data:`MMA_TILE_N`, ``N`` is a multiple of
        :data:`LANE_GROUP`, and ``dim`` is a multiple of ``3 * LANE_BLOCK``, which
        follows from being a multiple of 3 and of 16. ``L`` and the source-token
        block are the padded modes:
        M is rounded up in shared memory, the rounded rows are zeroed by the
        staging predicate or masked by the score, and every store is predicated.
        ``fold`` divides ``H``.
    """
    tid, _, _ = cute.arch.thread_idx()
    xidx, bidx, gidx = cute.arch.block_idx()

    tile = lane_block(dim)
    tiles = dim // tile
    lanes = tile // 3
    mpad = mma_rows(chunk)
    spad = mma_rows(span)
    blocks = chunk // span
    last = chunk - 1
    ldv = smem_pitch(tile)
    ldu = smem_pitch(rows)
    lds = smem_pitch(spad)
    ldf = smem_pitch(tile, 4)
    zero = cutlass.Float32(0.0)
    elem = gb.element_type
    out = gdb.element_type

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdrot = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdquat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdw = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdlp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdls = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 3).layout(), 16)
    srow = smem.allocate_tensor(cutlass.Float32, row_tile(chunk).layout(), 16)
    scrot = smem.allocate_tensor(elem, readout_tile(chunk, tile).layout(), SMEM_SEGMENT)
    space = arena(chunk, rows, dim, fold, span, elem.width // 8)
    base = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((space.words,), stride=(1,)), 16
    )
    sdy = _tile_at(base, space.out, out_tile(chunk, rows), elem)
    sstate = _tile_at(base, space.state, state_tile(rows, tile), elem)
    sb = _tile_at(base, space.raw, shifted_tile(span, tile), elem)
    sbrot = _tile_at(base, space.forced, forced_tile(span, tile), elem)
    su = _tile_at(base, space.input, shifted_tile(spad, rows), elem)
    sdb = _tile_at(base, space.tapped, gradient_tile(span, tile), cutlass.Float32)
    sscore = _tile_at(base, space.tapped, score_tile(chunk, span), elem)
    sdc = _tile_at(base, space.readout, gradient_tile(mpad, tile), cutlass.Float32)
    sumb = _tile_at(
        base, space.forcing, gradient_tile(chunk + 1, tile), cutlass.Float32
    )
    sumc = _tile_at(base, space.summed, gradient_tile(chunk, tile), cutlass.Float32)

    # Every view a GEMM reads is a layout over a tile that never moves, so all of
    # them are built once and none is per head, per block, per tap or per lane tile.
    vdy = cute.make_tensor(
        sdy.iterator, cute.make_layout((mpad, rows), stride=(ldu, 1))
    )
    vstate = cute.make_tensor(
        sstate.iterator, cute.make_layout((tile, rows), stride=(1, ldv))
    )
    vbrot = cute.make_tensor(
        sbrot.iterator, cute.make_layout((tile, span), stride=(1, ldv))
    )
    vcrot = cute.make_tensor(
        scrot.iterator, cute.make_layout((tile, mpad), stride=(1, ldv))
    )
    # The score in the orientation the forcing GEMM contracts: source token as M,
    # target token as K, so the stride-1 mode is M and the load transposes.
    vscore = cute.make_tensor(
        sscore.iterator, cute.make_layout((spad, mpad), stride=(1, lds))
    )

    dcacc = mma_acc(tiled_mma, tid, (mpad, tile))
    ccrd = mma_coords(tiled_mma, tid, (mpad, tile))
    dmacc = mma_acc(tiled_mma, tid, (mpad, span))
    mcrd = mma_coords(tiled_mma, tid, (mpad, span))
    dbacc = mma_acc(tiled_mma, tid, (spad, tile))
    bcrd = mma_coords(tiled_mma, tid, (spad, tile))
    # The narrowed score is the A operand of the GEMM that rereads it in place.
    # Fragment and view are built once: the retile is a layout, so nothing here is
    # per-tap work.
    #
    # These accumulators reach registers only when both loops below have a trip count
    # of one. Every accumulator allocation is hoisted to the kernel entry, and a
    # rolled loop between the allocation and its uses defeats register promotion, so
    # each fragment access becomes a local load and a local store. Measured at
    # ``P = 64``, ``L = 64``, span 64, fold one, at 255 registers and 91,344 B of
    # shared memory in both runs: one lane tile moves no local traffic at all, and
    # five lane tiles move 1,892.16 MB per call. The declaration site is not the
    # lever, and moving all four inside both loops leaves the counters unchanged to
    # the sector.
    mfrag = cute.make_fragment_like(dmacc, elem)
    fa_m = mma_areg(mfrag)

    # The lane tile is a grid axis, not a loop. Every accumulator above then sits
    # between its hoisted allocation and its uses with no rolled loop in between,
    # which is the condition register promotion needs; the divisor is compile-time,
    # so the decode is a multiply and a subtract rather than an integer division.
    cidx = xidx // tiles
    jstep = xidx - cidx * tiles

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)

    joff = jstep * tile
    # Slot row base of the two outputs that are sums over lanes. The slot axis sits
    # immediately outside the token axis, so this is the whole of the row offset and
    # it is zero at one tile, where the destination is the output itself.
    jbase = jstep * seqlen
    first = jstep == 0
    gbj = _lane_view(gb, joff)
    gbprevj = _lane_view(gbprev, joff)
    gcj = _lane_view(gc, joff)
    gdincj = _lane_view(gdinc, joff)
    gzj = _lane_view(gz, joff)
    gdbj = _lane_view(gdb, joff)
    gdcj = _lane_view(gdc, joff)
    gcarryj = _lane_view(gcarry, joff)
    # The chunk-input stage's half of the log-scale cotangent is the head's, not
    # the lane tile's, so a later tile must not add it a second time.
    wlp = cutlass.Float32(1.0)
    if cutlass.const_expr(tiles > 1):
        wlp = select(first, cutlass.Float32(1.0), zero)

    cute.arch.sync_threads()
    _fill_zero(sumb, (chunk + 1) * ldf, tid, threads)
    if cutlass.const_expr(fold > 1):
        _fill_zero(sumc, chunk * ldf, tid, threads)

    # Rolled. Unrolling it at trace time was measured and does not promote the
    # accumulators: local traffic rose from 1,135.3 MB to 1,290.4 MB per launch at
    # ``3N 240`` fold 18 with every device loop at trip count one, so the spill is a
    # live set against the 255-register cap and not a loop form. The two vector sums
    # and the carry stay the group's and outlive the fold, so nothing here becomes a
    # partial buffer.
    for hstep in cutlass.range(fold, unroll=1):
        hidx = gidx * fold + hstep
        cute.arch.sync_threads()
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
        _fill_zero(srow, chunk * ROW_WORDS, tid, threads)
        _fill_zero(sdlp, chunk, tid, threads)
        _fill_zero(sdw, 4 * chunk, tid, threads)
        cute.arch.sync_threads()
        chunk_prefixes(strans, slp, squat, tid, chunk)
        cute.arch.sync_threads()
        build_table(strans, stap, squat, stable, tid, threads, chunk, 3)
        cute.arch.sync_threads()

        # The closing transition, read once per head. Ac is R(Q)^T, so it is the
        # frame change the increment cotangent needs. Its own two cotangents are
        # read where the chart closes instead of here: nine live floats across the
        # source-token loop is nine the accumulators do not get.
        # A plain range: the DSL preprocessor rewrites `range_constexpr` in a `for`
        # statement only, so inside a comprehension it reaches the runtime stub and
        # raises. Both unroll at trace time; only the statement form needs the
        # marker.
        aclast = tuple(stable[TABLE_AC, last, i] for i in range(9))
        lplast = slp[last]

        # Three staging passes back to back, so their global loads overlap rather
        # than serializing. The readout basis is the M mode of two GEMMs and the K
        # mode of a third, so it is staged once; ``slp`` is passed as its scale tile
        # and left unread, the per-token exponential belonging to the offset term
        # alone.
        stage_rotated(
            gcj,
            gcj,
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
        )
        stage_shifted(
            gdy,
            gdy,
            sdy,
            bidx,
            hidx,
            t0,
            1,
            valid,
            tid,
            threads,
            mpad - 1,
            rows,
            False,
        )
        stage_state(gzj[bidx, hidx, cidx, None, None], sstate, tid, threads, rows, tile)
        cute.arch.sync_threads()

        # The offset term, and the log-scale cotangent it carries. The scale is
        # per target token, so it rides the accumulator's M mode and is applied
        # after the reduction that needs the unscaled value.
        dcacc.fill(0.0)
        mma_gemm(tiled_mma, tid, dcacc, vdy, vstate, True, False)
        for i in cutlass.range_constexpr(cute.size(dcacc)):
            m, d = ccrd[i]
            expl = decay(slp[cutlass.min(m, last)])
            held = _sum_over_n(dcacc[i] * widen(scrot[m, d], elem))
            if tid % 4 == 0 and m < chunk:
                sdlp[m] = sdlp[m] + 2.0 * expl * held
            dcacc[i] = dcacc[i] * expl

        cute.arch.sync_threads()
        stage_matrix(
            gdincj,
            sstate,
            sstate,
            aclast,
            bidx,
            hidx,
            cidx,
            tid,
            threads,
            rows,
            lanes,
            False,
        )
        cute.arch.sync_threads()

        for nstep in cutlass.range_constexpr(blocks):
            nbase = nstep * span
            stage_shifted(
                gbj,
                gbprevj,
                sb,
                bidx,
                gidx,
                t0,
                nbase,
                valid,
                tid,
                threads,
                span,
                tile,
                has_prev,
            )
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
                spad,
                rows,
                has_prev,
            )
            for tap in cutlass.range_constexpr(2):
                cute.arch.sync_threads()
                _rotate_rows(
                    sb,
                    sbrot,
                    stable,
                    tid,
                    nbase,
                    TABLE_AP if tap == 0 else TABLE_AN,
                    tap,
                    threads,
                    span,
                    lanes,
                )
                cute.arch.sync_threads()

                # Two views of the one staged run, one row of pitch apart: the
                # current tap reads token nbase+r, the previous one nbase+r-1.
                vun = cute.make_tensor(
                    su.iterator + tap * ldu,
                    cute.make_layout((span, rows), stride=(ldu, 1)),
                )
                vum = cute.make_tensor(
                    su.iterator + tap * ldu,
                    cute.make_layout((spad, rows), stride=(ldu, 1)),
                )

                # The score, target token by source token, into the readout
                # accumulator. I6: the mask lands on the float32 accumulator, then
                # one narrowing into the operand. I3: one exponential of a log
                # difference.
                dmacc.fill(0.0)
                mma_gemm(tiled_mma, tid, dmacc, vdy, vun, True, True)
                for i in cutlass.range_constexpr(cute.size(dmacc)):
                    m, n = mcrd[i]
                    src = nbase + n
                    masked = dmacc[i] * decay(slp[cutlass.min(m, last)] - slp[src])
                    mfrag[i] = narrow(select(src <= m, masked, zero), elem)
                mma_gemm_areg(tiled_mma, tid, dcacc, fa_m, vbrot, False)

                # The same score, transposed, for the forcing accumulator, which
                # contracts over the target token the readout consumer holds as N.
                # A fragment cannot be reread across its M mode, so the transpose
                # is a store and a transposed ``ldmatrix``: the pair of adjacent
                # columns a thread holds lands in one bank word, and the eight rows
                # of a warp's quads stride four banks apart, so the store is
                # conflict-free by the pitch. The pad rows the source-token M mode
                # was rounded up by are never written and reach only the
                # accumulator rows the store below drops.
                for i in cutlass.range_constexpr(cute.size(dmacc)):
                    m, n = mcrd[i]
                    sscore[m, n] = mfrag[i]

                # The increment term opens the forcing accumulator, because its
                # weight is per source token and the score term's is not. It runs
                # between the score's store and the barrier that publishes it.
                dbacc.fill(0.0)
                mma_gemm(tiled_mma, tid, dbacc, vum, vstate, True, False)
                for i in cutlass.range_constexpr(cute.size(dbacc)):
                    r, _ = bcrd[i]
                    src = nbase + cutlass.min(r, span - 1)
                    dbacc[i] = dbacc[i] * decay(lplast - slp[src])
                cute.arch.sync_threads()
                mma_gemm(tiled_mma, tid, dbacc, vscore, vcrot, False, False)
                # The gradient overwrites the score it was built from, so the read
                # has to be complete in every thread before the store begins.
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(cute.size(dbacc)):
                    r, d = bcrd[i]
                    if r < span:
                        sdb[r, d] = dbacc[i]

                cute.arch.sync_threads()
                _tap_epilogue(
                    gdtap,
                    sdb,
                    sbrot,
                    sb,
                    sumb,
                    stable,
                    stap,
                    strans,
                    srow,
                    sdw,
                    bidx,
                    hidx,
                    t0,
                    nbase,
                    valid,
                    tid,
                    jbase,
                    tap,
                    threads,
                    span,
                    lanes,
                )
            cute.arch.sync_threads()

        # The readout accumulator is final. It goes to shared memory because its
        # three columns per token are held by two threads, and the transform and
        # the outer product below need all three in one.
        for i in cutlass.range_constexpr(cute.size(dcacc)):
            m, d = ccrd[i]
            sdc[m, d] = dcacc[i]
        cute.arch.sync_threads()
        _readout_epilogue(
            gdcj,
            sdc,
            sumc,
            scrot,
            stable,
            srow,
            bidx,
            gidx,
            t0,
            valid,
            tid,
            threads,
            chunk,
            lanes,
            fold,
        )
        cute.arch.sync_threads()

        # The rotation cotangent is complete, so the transition chart closes: one
        # 3x3 product per token, the chunk-transition cotangent on the last token,
        # then the two reverse scans the chunk-local prefixes owe.
        #
        # The transition's own two cotangents are read here rather than before the
        # source-token loop, where they would sit live in eleven registers across
        # the widest phase of the kernel. Both are the head's, so a tiled state
        # width takes them on its first lane tile and adds zero on the rest.
        dclose = tuple(gdrot[bidx, hidx, cidx, i // 3, i % 3] for i in range(9))
        dclast = gdscale[bidx, hidx, cidx]
        cscale = decay(slp[last])
        if cutlass.const_expr(tiles > 1):
            dclose = tuple(select(first, v, zero) for v in dclose)
            dclast = select(first, dclast, zero)
        for step in cutlass.range_constexpr(-(-chunk // threads)):
            token = tid + step * threads
            if token < chunk:
                gsum = tuple(srow[token, k] for k in range(ROW_WORDS))
                dac = mat3_mul(gsum, _mat_at(stable, TABLE_AC, token))
                closing = token == last
                drot = tuple(
                    select(
                        closing,
                        dac[3 * (k % 3) + k // 3] + dclose[k],
                        dac[3 * (k % 3) + k // 3],
                    )
                    for k in range(9)
                )
                dquat = rot_hom_vjp(
                    drot,
                    (
                        squat[0, token],
                        squat[1, token],
                        squat[2, token],
                        squat[3, token],
                    ),
                )
                for j in cutlass.range_constexpr(4):
                    sdrot[j, token] = dquat[j]
                sdlp[token] = (
                    sdlp[token]
                    + wlp * gdlp[bidx, hidx, cidx, token]
                    + select(closing, 2.0 * cscale * dclast, zero)
                )
        cute.arch.sync_threads()
        chunk_suffix(sdlp, sdls, tid, chunk)
        quat_suffix_vjp(squat, sdrot, sdquat, tid, chunk)
        cute.arch.sync_threads()

        for step in cutlass.range_constexpr(-(-chunk // threads)):
            token = tid + step * threads
            if token < chunk:
                dexp = quat_exp_vjp(
                    (
                        sdquat[0, token],
                        sdquat[1, token],
                        sdquat[2, token],
                        sdquat[3, token],
                    ),
                    (strans[0, token], strans[1, token], strans[2, token]),
                )
                if token < valid:
                    # Both halves are sums over lanes, so a tiled state width writes
                    # one slot row per tile and :func:`close_slots` sums them. The
                    # expression is the token's own row at one tile.
                    trow = jbase + t0 + token
                    for j in cutlass.range_constexpr(3):
                        gdtrans[bidx, hidx, trow, j] = sdw[j, token] + dexp[j]
                    gdtrans[bidx, hidx, trow, 3] = sdls[token]

    cute.arch.sync_threads()

    # The group's two sums for this lane tile, rounded once. Row t+1 of the
    # forcing sum is token t and row 0 is the row the boundary kernel owns.
    total = chunk * tile
    for step in cutlass.range_constexpr(-(-total // threads)):
        i = tid + step * threads
        if i < total:
            t = i // tile
            c = i - t * tile
            if t < valid:
                gdbj[bidx, gidx, t0 + t, c] = narrow(sumb[t + 1, c], out)
                if cutlass.const_expr(fold > 1):
                    gdcj[bidx, gidx, t0 + t, c] = narrow(sumc[t, c], out)
    for step in cutlass.range_constexpr(-(-tile // threads)):
        c = tid + step * threads
        if c < tile:
            gcarryj[bidx, gidx, cidx, c] = sumb[0, c]


@cute.jit
def chunk_vector_bwd(
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
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    gdtrans: cute.Tensor,
    gdtap: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    groups: cutlass.Int32,
    stream: Stream,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    fold: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    resident: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_vector_bwd_kernel`.

    ``P``, ``3N``, the source-token block and the fold are compile-time because the
    accumulator partitions and the arena offsets are. Batch, group, chunk count and
    sequence length are dynamic.

    The x extent carries the chunk and the lane tile, chunk outermost. The lane tile
    is a grid axis rather than a loop so that no rolled loop sits between an
    accumulator's allocation and its uses, and the block count rises by the tile
    count with it.
    """
    chunk_vector_bwd_kernel(
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
        gdlp,
        gdrot,
        gdscale,
        gdb,
        gdc,
        gcarry,
        gdtrans,
        gdtap,
        seqlen,
        make_mma(dtype),
        threads,
        chunk,
        rows,
        dim,
        span,
        fold,
        has_prev,
    ).launch(
        grid=(chunks * (dim // lane_block(dim)), bsz, groups),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
        stream=stream,
    )


class ChunkVectorBwd(NamedTuple):
    """What one launch of the chunk-vector backward produces.

    Attributes:
        dB: ``(B,G,T,3N)`` cotangent of the forcing vectors, in the activation
            dtype, summed over the heads of each group. The chunk-boundary rows
            carry the current tap alone;
            :func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds the
            previous tap's row there.
        dC: ``(B,G,T,3N)`` cotangent of the readout vectors, in the activation
            dtype, summed over the heads of each group.
        carry_b: ``(B,G,C,3N)`` float32 cotangent that each chunk's first token
            sends to the token before it. Index 0 is the streaming feedback.
        dtrans: ``(B,H,T,4)`` float32 cotangent of ``(w_x, w_y, w_z, ls)``.
        dK: ``(B,H,T,2,4)`` float32 cotangent of the two taps. Lane 3 is zero.
    """

    dB: Tensor
    dC: Tensor
    carry_b: Tensor
    dtrans: Tensor
    dK: Tensor


def chunk_vector_backward(
    dy: Tensor,
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    dinc: Tensor,
    zstart: Tensor,
    dlogp: Tensor,
    dchunk_rot: Tensor,
    dchunk_scale: Tensor,
    chunk_size: int,
    *,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
    dB: Tensor | None = None,
    dC: Tensor | None = None,
) -> ChunkVectorBwd:
    """Differentiate the rowwise vectors and the transition parameters.

    The three cotangents this takes from the chunk-input stage are consumed, never
    recomputed: ``dlogp`` is that stage's half of the log-scale cotangent, and the
    closing rotation and scale are one contraction over the chunk-start state that
    stage already ran.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous. A
            caller with no ``dy`` passes zeros: the increment terms survive.
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
        dlogp: ``(B,H,C,L)`` float32, contiguous, from
            :func:`slinoss.ops.so3ssd.cute.bwd.chunk_input.chunk_input_backward`.
        dchunk_rot: ``(B,H,C,3,3)`` float32, contiguous, from the same.
        dchunk_scale: ``(B,H,C)`` float32, contiguous, from the same.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, or None.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, or None.
        dB: Destination for the ``B`` cotangent, with the shape, dtype and device of
            ``B`` and possibly pitched. Every row is written by the kernel's own
            indexed stores, so it is never accumulated into and never zeroed first,
            and it is returned as this same object. ``None`` allocates. See
            :func:`slinoss.ops.so3ssd.reference.check_grad_band`.
        dC: Destination for the ``C`` cotangent, under the contract of ``dB``.

    Returns:
        :class:`ChunkVectorBwd`.

    Raises:
        ValueError: On a layout, rank, shape or extent violation, on a destination
            that is not the band of its operand, on a shared-memory budget the
            device cannot hold, or on half a streaming pair.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (U, "U"), (B, "B"), (C, "C"))
    pinned: Named = (
        (trans, "trans"),
        (K, "K"),
        (dinc, "dinc"),
        (zstart, "zstart"),
        (dlogp, "dlogp"),
        (dchunk_rot, "dchunk_rot"),
        (dchunk_scale, "dchunk_scale"),
    )
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
    fold = heads // groups
    span = vblock(chunk_size, rows, dim, fold, dy.element_size())
    check_extents(chunk_size, dim, span)
    has_prev = check_stream(u_prev, b_prev, (bsz, heads, groups, rows, dim))

    chunks = -(-seqlen // chunk_size)
    state = (bsz, heads, chunks, rows, dim)
    for tensor, name in ((dinc, "dinc"), (zstart, "zstart")):
        if tuple(tensor.shape) != state:
            raise ValueError(f"{name} must be {state}, got {tuple(tensor.shape)}")
    closing = (
        (dlogp, "dlogp", (bsz, heads, chunks, chunk_size)),
        (dchunk_rot, "dchunk_rot", (bsz, heads, chunks, 3, 3)),
        (dchunk_scale, "dchunk_scale", (bsz, heads, chunks)),
    )
    for tensor, name, shape in closing:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(tensor.shape)}")

    budget = assert_smem_fits(
        f"chunk_vector_bwd[L{chunk_size}/P{rows}/3N{dim}/fold{fold}]",
        vector_smem_bytes(chunk_size, rows, dim, fold, span, dy.element_size()),
    )

    # After the operand guards, so a destination is measured against an operand that
    # has already been held to its own shape and layout.
    if dB is not None:
        check_grad_band(dB, B, "dB")
    if dC is not None:
        check_grad_band(dC, C, "dC")

    device = dy.device
    if dB is None:
        dB = torch.empty(bsz, groups, seqlen, dim, dtype=dtype, device=device)
    if dC is None:
        dC = torch.empty(bsz, groups, seqlen, dim, dtype=dtype, device=device)
    carry_b = torch.empty(bsz, groups, chunks, dim, dtype=torch.float32, device=device)
    dtrans = torch.empty(bsz, heads, seqlen, 4, dtype=torch.float32, device=device)
    dK = torch.empty(bsz, heads, seqlen, 2, 4, dtype=torch.float32, device=device)
    # The two outputs that are sums over lanes. Every other output spans the state
    # width, so the lane tile that owns a column owns it alone.
    tiles = dim // lane_block(dim)
    held = (open_slots(dtrans, tiles), open_slots(dK, tiles, axis=-3))
    jit_launch(
        chunk_vector_bwd,
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
            dlogp,
            dchunk_rot,
            dchunk_scale,
            dB,
            dC,
            carry_b,
            held[0].dest,
            held[1].dest,
            seqlen,
            chunks,
            bsz,
            groups,
        ),
        (
            cute_dtype(dtype),
            THREADS,
            chunk_size,
            rows,
            dim,
            span,
            fold,
            has_prev,
            min(RESIDENT_MAX, max(1, smem_capacity() // budget)),
        ),
    )
    for slots in held:
        close_slots(slots)
    return ChunkVectorBwd(dB=dB, dC=dC, carry_b=carry_b, dtrans=dtrans, dK=dK)
