"""Chunk increment: the intra-chunk contribution to the next chunk's state.

One GEMM form, run twice per K slice to sum the two forcing taps:

    inc_local(P,3N) = (u * wgt)^T Bn + (ushift * wgt)^T Bp

with ``Bn_r = An_r b_r``, ``Bp_r = Ap_r b_{r-1}``, and
``wgt_r = exp(2*(lp_{L-1} - lp_r))``. Both taps share the same weight and the
same output tile, so they accumulate into one float32 fragment rather than
concatenating along K.

The weight is folded into the ``b`` side of the product, not the ``u`` side.
``b`` already passes through a 3x3 matvec with float32 scalars in registers, so
the scale costs no extra rounding there; folding it into ``u`` would add a
narrow-widen round trip to a tensor that is otherwise copied verbatim. The two
are equal term by term:

    sum_r u[r,p] * (wgt[r] * bn[r,d]) == sum_r (wgt[r] * u[r,p]) * bn[r,d]

``wgt <= 1`` by I1 and the exponent is formed from a log difference, never as a
ratio of two exponentials (I3). Low precision carries float32's exponent range in
both supported operand dtypes, so a weight small enough to flush ``wgt * b`` to
zero is a contribution that is already zero, which is the graceful underflow I1
guarantees.

Output frame. The increment is emitted in the chunk-local frame, without the
``R(Q_{L-1})`` that carries it into the global frame. The state recurrence applies
that rotation to the sum of the decayed state and the increment, which is the same
thing by linearity:

    a*(R z) + R inc == R (a*z + inc)

Applying it here instead would need three neighbouring N columns per thread, and
the ``m16n8k16`` C fragment gives each thread two, so it would cost either a
cross-thread shuffle or a float32 round trip through shared memory. The chunk
transition is emitted alongside the increment, as a unit quaternion and a separate
decay, because this kernel already has both chunk-local prefixes in shared memory
and the recurrence would otherwise recompute them.

Staging. ``u`` is staged once per K slice into a tile of ``kblk + 1`` rows: row
``r`` holds token ``t0 + s*kblk + r - 1``, so the current tap reads rows ``1..``
and the previous tap reads rows ``0..`` off the same tile through two views that
differ by one row of pitch. The shift is global, not per chunk, so row 0 of the
first slice of a chunk crosses the chunk boundary and, at the first chunk, reaches
the streaming ``u_prev``. Re-reading ``u`` from global for the second tap would add
its whole extent to a forward pass that moves about 131 MB.

``b`` is transformed on the way in and restaged between the two taps, so the
rotated forcing never reaches global memory. One thread owns one 3-vector: three
coalesced global reads, nine FMA, three shared-memory stores.

DRAM-bound. Analytic traffic at ``standard`` is about 42.4 MB against 906 MFLOP,
so 24 flop/byte against a ridge point of 165: memory bound by a factor of seven,
which is why the padded M mode costs nothing measurable.

A ragged tail needs no separate path. ``stage_chunk`` stages the pad as a zero tap
and the identity transition, so both tap matrices are zero past ``valid`` and every
padded row of the ``b`` tile is zero regardless of what ``u`` holds.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Tile,
    assert_smem_fits,
    cute_dtype,
    decay,
    dev_tensor,
    narrow,
    smem_bytes,
    widen,
)
from slinoss._precision import LOW_PRECISION_DTYPES
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AN,
    TABLE_AP,
    THREADS,
    mat3_matvec,
    table_tile,
    tap_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.mma import (
    MMA_TILE_K,
    MMA_TILE_N,
    SMEM_SEGMENT,
    make_mma,
    mma_acc,
    mma_gemm,
    mma_rows,
    mma_store,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_endpoint, chunk_prefixes
from slinoss.ops.so3ssd.cute.table import build_table, stage_chunk

__all__ = [
    "KBLOCK_MAX",
    "OPERAND_DTYPES",
    "ChunkIncrement",
    "chunk_increment_forward",
    "chunk_increment_fwd",
    "chunk_increment_fwd_kernel",
    "forced_tile",
    "increment_smem_bytes",
    "input_tile",
    "kblock",
    "scalar_tile",
]

KBLOCK_MAX: int = 64
"""Longest K slice. The two operand tiles are the only per-slice allocations, and
capping their K extent keeps ``MAX_CHUNK`` resident two blocks per SM instead of
one. Every legal chunk length is a power of two, so this divides it exactly and
the slice count is one or two."""

OPERAND_DTYPES: tuple[torch.dtype, ...] = LOW_PRECISION_DTYPES
"""Activation dtypes with a tensor-core path. The atom is 16-bit times 16-bit into
float32, so a float32 activation resolves to the reference backend rather than
being downcast behind the caller."""


def kblock(chunk: int) -> int:
    """K extent of one slice.

    Args:
        chunk: ``L``.

    Returns:
        ``min(L, KBLOCK_MAX)``.
    """
    return min(chunk, KBLOCK_MAX)


def scalar_tile(chunk: int) -> Tile:
    """One float32 per token. Dense."""
    return Tile((chunk,), (1,))


def input_tile(kblk: int, rows: int) -> Tile:
    """``u`` staging tile, ``(kblk + 1, pitch)``.

    The extra row is the token before the slice, which the previous tap reads.

    Args:
        kblk: K extent of the slice.
        rows: ``P``.
    """
    lda = smem_pitch(mma_rows(rows))
    return Tile((kblk + 1, lda), (lda, 1))


def forced_tile(kblk: int, dim: int) -> Tile:
    """Rotated forcing tile, ``(kblk, pitch)``.

    Args:
        kblk: K extent of the slice.
        dim: ``3N``.
    """
    ldb = smem_pitch(dim)
    return Tile((kblk, ldb), (ldb, 1))


def increment_smem_bytes(chunk: int, rows: int, dim: int, itemsize: int = 2) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_increment_fwd_kernel` allocates, in the same
    order. Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    kblk = kblock(chunk)
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 2), 4),
            (input_tile(kblk, rows), itemsize),
            (forced_tile(kblk, dim), itemsize),
        ]
    )


@cute.jit
def _store_forced(
    sb: cute.Tensor,
    stable: cute.Tensor,
    swgt: cute.Tensor,
    row: cutlass.Int32,
    token: cutlass.Int32,
    lane: cutlass.Int32,
    vec: tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32],
    slot: cutlass.Constexpr,
) -> None:
    """Transform one 3-vector by a tap matrix, scale it, and store it.

    Args:
        sb: ``(kblk, pitch)`` operand-dtype tile, written at
            ``[row, 3*lane .. 3*lane+2]``.
        stable: ``(2, L, 9)`` float32 transform table.
        swgt: ``(L,)`` float32 chunk weights.
        row: Row of the operand tile.
        token: Chunk-local token index, indexing ``stable`` and ``swgt``.
        lane: Which of the ``N`` 3-vectors.
        vec: The 3-vector, already widened to float32.
        slot: :data:`TABLE_AP` or :data:`TABLE_AN`. Compile-time.
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
    dst = sb.element_type
    weight = swgt[token]
    sb[row, 3 * lane] = narrow(weight * out[0], dst)
    sb[row, 3 * lane + 1] = narrow(weight * out[1], dst)
    sb[row, 3 * lane + 2] = narrow(weight * out[2], dst)


@cute.jit
def _stage_input(
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
    kblk: cutlass.Constexpr,
    lda: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Stage ``u`` for one K slice, one row behind.

    Row ``r`` holds token ``t0 + lbase + r - 1``, so one tile serves both taps.
    Columns past ``P`` and rows past the sequence are zeroed; a zero there
    multiplies a forcing row that is itself zero, so no store is skipped.

    Every store is inside a divergent branch and every ``(r, p)`` pair is owned by
    exactly one thread on exactly one branch. Two guarded stores to the same
    address would race, because the thread that owns the address in the flat loop
    is not the one that owns it in an overlay pass.

    Args:
        gu: ``(B,H,T,P)`` operand-dtype input weights.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        su: From :func:`input_tile`, written.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        lbase: First chunk-local token of the slice.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        kblk: K extent of the slice. Compile-time.
        lda: Row pitch of ``su``. Compile-time.
        rows: ``P``. Compile-time.
        has_prev: Whether ``guprev`` was supplied. Compile-time.
    """
    zero = su.element_type(0.0)
    for i in cutlass.range(tid, (kblk + 1) * lda, threads):
        r = i // lda
        p = i - r * lda
        # The predicate keeps the read in bounds on its own: token < valid
        # implies t0 + token < seqlen, so no clamp is needed under a divergent
        # branch, where an inactive lane issues no load.
        token = lbase + r - 1
        g = t0 + token
        if (p < rows) & (g >= 0) & (token < valid):
            su[r, p] = gu[bidx, hidx, g, p]
        else:
            if cutlass.const_expr(has_prev):
                if (p < rows) & (g < 0):
                    su[r, p] = guprev[bidx, hidx, p]
                else:
                    su[r, p] = zero
            else:
                su[r, p] = zero


@cute.jit
def _stage_forced(
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    sb: cute.Tensor,
    stable: cute.Tensor,
    swgt: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    lbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    slot: cutlass.Constexpr,
    back: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    kblk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Transform and stage one tap's forcing for one K slice.

    Row ``r`` holds ``A_slot[lbase + r] b[t0 + lbase + r - back]``, scaled by the
    chunk weight. ``back`` is 0 for the current tap and 1 for the previous one:
    both matrices are indexed at the token they act on, while the previous tap's
    vector comes from the token before it.

    Args:
        gb: ``(B,H,T,3N)`` operand-dtype input vectors.
        gbprev: ``(B,H,3N)`` streaming ``b_{-1}``. Read only when ``has_prev``
            and ``back`` is 1.
        sb: From :func:`forced_tile`, written.
        stable: ``(2, L, 9)`` float32 transform table.
        swgt: ``(L,)`` float32 chunk weights.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        lbase: First chunk-local token of the slice.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        slot: :data:`TABLE_AP` or :data:`TABLE_AN`. Compile-time.
        back: Token offset of the vector, 0 or 1. Compile-time.
        threads: Block width. Compile-time.
        kblk: K extent of the slice. Compile-time.
        lanes: ``N``. Compile-time.
        has_prev: Whether ``gbprev`` was supplied. Compile-time.
    """
    src = gb.element_type
    zero = sb.element_type(0.0)
    for i in cutlass.range(tid, kblk * lanes, threads):
        r = i // lanes
        n = i - r * lanes
        token = lbase + r
        g = t0 + token - back
        if (token < valid) & (g >= 0):
            _store_forced(
                sb,
                stable,
                swgt,
                r,
                token,
                n,
                (
                    widen(gb[bidx, hidx, g, 3 * n], src),
                    widen(gb[bidx, hidx, g, 3 * n + 1], src),
                    widen(gb[bidx, hidx, g, 3 * n + 2], src),
                ),
                slot,
            )
        else:
            # g < 0 is reachable only for the previous tap at the first token of
            # the first chunk, which is exactly the streaming carry-in.
            if cutlass.const_expr(has_prev and back == 1):
                if token < valid:
                    _store_forced(
                        sb,
                        stable,
                        swgt,
                        r,
                        token,
                        n,
                        (
                            widen(gbprev[bidx, hidx, 3 * n], src),
                            widen(gbprev[bidx, hidx, 3 * n + 1], src),
                            widen(gbprev[bidx, hidx, 3 * n + 2], src),
                        ),
                        slot,
                    )
                else:
                    sb[r, 3 * n] = zero
                    sb[r, 3 * n + 1] = zero
                    sb[r, 3 * n + 2] = zero
            else:
                sb[r, 3 * n] = zero
                sb[r, 3 * n + 1] = zero
                sb[r, 3 * n + 2] = zero


@cute.kernel
def chunk_increment_fwd_kernel(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    ginc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    seqlen: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Accumulate one chunk's local increment and emit its transition.

    One block per ``(chunk, batch, head)``.

    Args:
        gu: ``(B,H,T,P)`` operand-dtype input weights.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 per-tap ``(kr, g, h, 0)``.
        gb: ``(B,H,T,3N)`` operand-dtype input vectors.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``, or a placeholder.
        gbprev: ``(B,H,3N)`` streaming ``b_{-1}``, or a placeholder.
        ginc: ``(B,H,C,P,3N)`` float32, written with the chunk-local increment.
        gcquat: ``(B,H,C,4)`` float32, written with the unit chunk rotation.
        gcscale: ``(B,H,C)`` float32, written with ``exp(2*lp_{L-1})``.
        seqlen: ``T``. Dynamic.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        has_prev: Whether the streaming carry-in was supplied. Compile-time.

    Invariants:
        ``chunk`` is a multiple of :data:`MMA_TILE_K` and of ``kblock(chunk)``,
        and ``dim`` is a multiple of :data:`MMA_TILE_N`. ``rows`` is free: M is
        rounded up in shared memory, zero-filled, and the store is predicated.
    """
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    kblk = kblock(chunk)
    slices = chunk // kblk
    lanes = dim // 3
    mpad = mma_rows(rows)
    lda = smem_pitch(mpad)
    ldb = smem_pitch(dim)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    swgt = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 2).layout(), 16)
    su = smem.allocate_tensor(
        gu.element_type, input_tile(kblk, rows).layout(), SMEM_SEGMENT
    )
    sb = smem.allocate_tensor(
        gb.element_type, forced_tile(kblk, dim).layout(), SMEM_SEGMENT
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
    # Columns past 3N take part in no MMA read, but garbage there would be read
    # as an operand. The per-slice restage covers only the first 3N columns, so
    # the pad is zeroed once here.
    if cutlass.const_expr(ldb > dim):
        pad = ldb - dim
        zero = sb.element_type(0.0)
        for i in cutlass.range(tid, kblk * pad, threads):
            r = i // pad
            sb[r, dim + i - r * pad] = zero

    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()

    build_table(strans, stap, squat, stable, tid, threads, chunk, 2)
    last = chunk - 1
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            # I3: one exponential of a log difference, never a ratio of two.
            swgt[token] = decay(slp[last] - slp[token])
    if tid == 0:
        cquat, cscale = chunk_endpoint(squat, slp, chunk)
        for j in cutlass.range_constexpr(4):
            gcquat[bidx, hidx, cidx, j] = cquat[j]
        gcscale[bidx, hidx, cidx] = cscale

    acc = mma_acc(tiled_mma, tid, (mpad, dim))
    # Two views of one staging tile, one row of pitch apart. The current tap
    # reads token t0+lbase+k, the previous one reads t0+lbase+k-1.
    va_now = cute.make_tensor(
        su.iterator + lda, cute.make_layout((mpad, kblk), stride=(1, lda))
    )
    va_prv = cute.make_tensor(
        su.iterator, cute.make_layout((mpad, kblk), stride=(1, lda))
    )
    vb = cute.make_tensor(sb.iterator, cute.make_layout((dim, kblk), stride=(1, ldb)))

    for s in cutlass.range_constexpr(slices):
        lbase = s * kblk
        cute.arch.sync_threads()
        _stage_input(
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
            lda,
            rows,
            has_prev,
        )
        _stage_forced(
            gb,
            gbprev,
            sb,
            stable,
            swgt,
            bidx,
            hidx,
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
        )
        cute.arch.sync_threads()
        mma_gemm(tiled_mma, tid, acc, va_now, vb, False, False)
        cute.arch.sync_threads()
        _stage_forced(
            gb,
            gbprev,
            sb,
            stable,
            swgt,
            bidx,
            hidx,
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
        )
        cute.arch.sync_threads()
        mma_gemm(tiled_mma, tid, acc, va_prv, vb, False, False)

    mma_store(
        tiled_mma, tid, acc, ginc[bidx, hidx, cidx, None, None], (mpad, dim), rows
    )


@cute.jit
def chunk_increment_fwd(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    guprev: cute.Tensor,
    gbprev: cute.Tensor,
    ginc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_increment_fwd_kernel`.

    ``P`` and ``3N`` are compile-time because the accumulator's partition shape
    is. Batch, head, chunk count, and sequence length are dynamic.
    """
    chunk_increment_fwd_kernel(
        gu,
        gtrans,
        gtap,
        gb,
        guprev,
        gbprev,
        ginc,
        gcquat,
        gcscale,
        seqlen,
        make_mma(dtype),
        threads,
        chunk,
        rows,
        dim,
        has_prev,
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1))


class ChunkIncrement(NamedTuple):
    """Result of the chunk increment.

    Attributes:
        inc: ``(B,H,C,P,3N)`` float32 chunk-local increment. Feeds
            :func:`slinoss.ops.so3ssd.cute.fwd.state_passing.state_passing_forward`,
            which consumes it in place.
        cquat: ``(B,H,C,4)`` float32 unit chunk rotation, scalar-first.
        cscale: ``(B,H,C)`` float32 chunk decay ``exp(2*lp_{L-1})``.
    """

    inc: Tensor
    cquat: Tensor
    cscale: Tensor


def _check_layout(named: tuple[tuple[Tensor, str], ...]) -> None:
    """Raises:
    ValueError: If any operand is off CUDA or not contiguous. The tensor
        contract is time-major and contiguous, so a repack here would be the
        staging copy the kernel exists to avoid.
    """
    for tensor, name in named:
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA device, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _check_operands(named: tuple[tuple[Tensor, str], ...]) -> torch.dtype:
    """Raises:
    TypeError: If an activation dtype has no tensor-core path, or if the
        activations do not share one dtype.

    Returns:
        The shared activation dtype.
    """
    for tensor, name in named:
        if tensor.dtype not in OPERAND_DTYPES:
            raise TypeError(
                f"{name} has dtype {tensor.dtype}; "
                f"tensor-core operand dtypes: {OPERAND_DTYPES}"
            )
    head, head_name = named[0]
    for tensor, name in named[1:]:
        if tensor.dtype is not head.dtype:
            raise TypeError(
                f"{name} is {tensor.dtype} and {head_name} is {head.dtype}; "
                "one activation dtype per call"
            )
    return head.dtype


def _check_pinned(named: tuple[tuple[Tensor, str], ...]) -> None:
    """Raises:
    ValueError: If a float32-pinned operand is not float32 (I4).
    """
    for tensor, name in named:
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must be float32 (I4), got {tensor.dtype}")


def _check_shapes(
    U: Tensor, trans: Tensor, K: Tensor, B: Tensor
) -> tuple[int, int, int, int, int]:
    """Raises:
    ValueError: On a rank or shape mismatch.

    Returns:
        ``(B, H, T, P, 3N)``.
    """
    if U.ndim != 4:
        raise ValueError(f"U must be (B,H,T,P), got {tuple(U.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in U.shape)
    lead = (bsz, heads, seqlen)
    if tuple(trans.shape) != (*lead, 4):
        raise ValueError(f"trans must be {(*lead, 4)}, got {tuple(trans.shape)}")
    if tuple(K.shape) != (*lead, 2, 4):
        raise ValueError(f"K must be {(*lead, 2, 4)}, got {tuple(K.shape)}")
    if B.ndim != 4 or tuple(B.shape[:3]) != lead:
        raise ValueError(f"B must be (B,H,T,3N) with {lead}, got {tuple(B.shape)}")
    return bsz, heads, seqlen, rows, int(B.shape[3])


def _check_extents(chunk_size: int, dim: int) -> None:
    """Raises:
    ValueError: If ``L`` or ``3N`` is an extent the atom cannot cover. The fix
        for any of these is the shape, never a padding path.
    """
    if chunk_size < MMA_TILE_K or chunk_size % MMA_TILE_K != 0:
        raise ValueError(
            f"chunk_size must be a positive multiple of {MMA_TILE_K}, got {chunk_size}"
        )
    if chunk_size % kblock(chunk_size) != 0:
        raise ValueError(
            f"chunk_size {chunk_size} is not a multiple of its K slice "
            f"{kblock(chunk_size)}"
        )
    if dim % 3 != 0 or dim % MMA_TILE_N != 0:
        raise ValueError(f"3N must be a multiple of 3 and of {MMA_TILE_N}, got {dim}")


def chunk_increment_forward(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    chunk_size: int,
    *,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
) -> ChunkIncrement:
    """Accumulate every chunk's local increment and its transition.

    Args:
        U: ``(B,H,T,P)``, one of :data:`OPERAND_DTYPES`, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)`` float32, contiguous. Per-tap ``(kr, g, h, 0)``.
        B: ``(B,H,T,3N)``, the dtype of ``U``, contiguous.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, the dtype of ``U``. Paired with
            ``b_prev``.
        b_prev: ``(B,H,3N)`` streaming ``b_{-1}``, the dtype of ``U``.

    Returns:
        A :class:`ChunkIncrement`.

    Raises:
        ValueError: On a layout, rank, shape, extent, or pairing violation.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: tuple[tuple[Tensor, str], ...] = ((U, "U"), (B, "B"))
    if (u_prev is None) != (b_prev is None):
        raise ValueError("u_prev and b_prev are supplied together or not at all")
    has_prev = u_prev is not None and b_prev is not None
    if has_prev:
        assert u_prev is not None and b_prev is not None
        activations = (*activations, (u_prev, "u_prev"), (b_prev, "b_prev"))

    pinned: tuple[tuple[Tensor, str], ...] = ((trans, "trans"), (K, "K"))
    _check_layout((*activations, *pinned))
    dtype = _check_operands(activations)
    _check_pinned(pinned)
    bsz, heads, seqlen, rows, dim = _check_shapes(U, trans, K, B)
    _check_extents(chunk_size, dim)
    if has_prev:
        assert u_prev is not None and b_prev is not None
        if tuple(u_prev.shape) != (bsz, heads, rows):
            raise ValueError(
                f"u_prev must be {(bsz, heads, rows)}, got {tuple(u_prev.shape)}"
            )
        if tuple(b_prev.shape) != (bsz, heads, dim):
            raise ValueError(
                f"b_prev must be {(bsz, heads, dim)}, got {tuple(b_prev.shape)}"
            )

    assert_smem_fits(
        f"chunk_increment[L{chunk_size}/P{rows}/3N{dim}]",
        increment_smem_bytes(chunk_size, rows, dim, U.element_size()),
    )

    chunks = -(-seqlen // chunk_size)
    opts = {"dtype": torch.float32, "device": U.device}
    inc = torch.empty(bsz, heads, chunks, rows, dim, **opts)
    cquat = torch.empty(bsz, heads, chunks, 4, **opts)
    cscale = torch.empty(bsz, heads, chunks, **opts)

    # A placeholder keeps one launch signature. It is never read: the branch that
    # would read it is closed at compile time.
    ustream = U[:, :, 0] if u_prev is None else u_prev
    bstream = B[:, :, 0] if b_prev is None else b_prev
    chunk_increment_fwd(
        dev_tensor(U),
        dev_tensor(trans),
        dev_tensor(K),
        dev_tensor(B),
        dev_tensor(ustream),
        dev_tensor(bstream),
        dev_tensor(inc),
        dev_tensor(cquat),
        dev_tensor(cscale),
        seqlen,
        chunks,
        bsz,
        heads,
        cute_dtype(dtype),
        THREADS,
        chunk_size,
        rows,
        dim,
        has_prev,
    )
    return ChunkIncrement(inc=inc, cquat=cquat, cscale=cscale)
