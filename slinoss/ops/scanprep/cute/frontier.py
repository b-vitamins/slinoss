"""The scan's parameter frontier. CuTe DSL forward and backward.

One kernel and one launch per direction. The forward reads the two token-major
projection slices and writes every per-token operand the scan reads except ``U``:
``trans``, ``K``, ``B``, ``C``. The maps themselves are in
:mod:`slinoss.ops.scanprep.cute.maps`, which is their only device-side
implementation.

Parallel decomposition. One block per ``(batch, token tile)``. One thread owns one
``(head, token)`` pair with the token innermost, reads that head's ten parameter
columns itself, adds the per-head bias, and writes the head-major ``(B,H,T,4)``
and ``(B,H,T,2,4)`` rows for its token. No shared memory and no barrier: the
parameter slice's ten columns per head are the thread's own working set, so the
transpose that a coalesced row load would need never arises, and neither does the
tile it would be staged in. Bank conflicts are unreachable rather than avoided.

That read is a ten-column gather per thread, not a coalesced row. It costs L1
requests, not DRAM traffic: a block reads every head of its own token rows, so
every sector it touches is fully consumed, and the ten loads of one column re-read
the sectors the other nine touch. At one group and the minimum state width the
parameter slice is a fifth of the bytes the forward moves and, with ``dparams``, a
third of the backward's; at four groups and twice the state, a sixteenth and a
ninth.

``bc`` is a separate phase over its own flat item space. A run of consecutive
source columns inside one ``(half, group)`` segment maps to a run of consecutive
destination columns, so that permute is coalesced on both sides. The destination
row is ``3N * itemsize`` bytes, at least 96, so every store covers whole 32-byte
sectors.

Operand layout. ``params`` and ``bc`` are slices of one projection output: the
trailing axis has unit stride and the row stride is the full projection width,
taken from the operand at runtime. Nothing here repacks either one. The
precondition beyond unit trailing stride is that the base address and the row
pitch both land on a multiple of ``ALIGN_BYTES // itemsize`` elements, which the
producer gets by padding its column offsets and its projection width; it is
checked on the host so no alignment branch reaches the kernel.

The backward is the same decomposition. One thread recovers its own biased
parameter row, applies both Jacobians, stores its ten gradient columns, and then
reduces them over the tile's tokens by warp shuffle. ``TILE_TOKENS`` divides the
warp width, so a token run lies inside one warp and the reduction needs neither
shared memory nor a barrier; the run's last lane writes the block's partial
``dparam_bias`` row.

Invariants. I1 and I2 are produced here rather than asserted here, and I4 holds:
``trans`` and ``K`` are float32 at every input width, including under autocast,
and the parameter rows are widened on load so both maps and the bias reduction
run in float32. Gradients are narrowed back to the input width once, on the store
to ``dparams``.

DRAM-bound. Per token the forward moves ``i*(10*H + 4*G*3N) + 48*H`` bytes at
activation itemsize ``i``: the parameter row in, ``bc`` in, ``B`` and ``C`` out,
and 48 bytes of packed float32 per head. The backward moves
``i*(20*H + 4*G*3N) + 48*H + 40*H/TILE_TOKENS`` bytes: both float32 cotangents and
``dB``, ``dC``, and the parameter row in, ``dparams`` and ``dbc`` out, and the
partial bias row. The per-head bias is ``40*H`` bytes for the whole launch and is
read once per block out of cache. No measured bandwidth is claimed here.
"""

import math

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Scalar,
    cute_dtype,
    jit_launch,
    narrow,
    select,
    shuffle_up,
    widen,
)
from slinoss._guard import Named, check_layout, check_pitched
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.scanprep.cute.maps import (
    log_scale,
    log_scale_grad,
    rotvec,
    rotvec_grad,
)
from slinoss.ops.scanprep.reference import (
    PARAM_COLS,
    ScanGrads,
    ScanParams,
    check_cotangents,
    check_operands,
)

__all__ = [
    "PREFETCH",
    "THREADS",
    "TILE_TOKENS",
    "scanprep_backward",
    "scanprep_bwd",
    "scanprep_bwd_kernel",
    "scanprep_forward",
    "scanprep_fwd",
    "scanprep_fwd_kernel",
]

THREADS = 128
"""Block width. Four warps.

Every phase's lane map repeats with the warp, so any warp multiple keeps the
arguments below. Set by occupancy alone.
"""

TILE_TOKENS = 4
"""Tokens one block covers.

A divisor of the warp width, which is what lets the backward's bias reduction be a
warp shuffle: a token run then lies inside one warp, so the reduction needs no
shared memory and no barrier. Small, because the tile is not a unit of reuse --
nothing here is read twice -- and a small tile is what keeps the grid long enough
to hide the launch tail: at ``T`` tokens per batch the grid is ``B * T /
TILE_TOKENS`` blocks, and a tile of 32 leaves too few to fill the device evenly.

``T`` is arbitrary against it. The last tile's reads are clamped and its stores
predicated.
"""

PREFETCH = 4
"""Global loads issued before the group's stores.

The two ``bc`` phases carry most of the traffic and their outer loop is not
unrolled, so the group is what puts several loads in flight per iteration instead
of one. Every other loop has a trace-time trip count and unrolls.
"""


# ---------------------------------------------------------------------------
# Device phases
# ---------------------------------------------------------------------------
#
# Every index is clamped unconditionally and never predicated, so no global load
# sits behind a branch, and a clamped thread reads the slot its owner reads. Every
# global store carries a predicate, because a duplicate global store would be
# traffic.


def _biased_row(
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    bidx: cutlass.Int32,
    token: cutlass.Int32,
    base: cutlass.Int32,
    dtype: cutlass.Constexpr,
) -> list[Scalar]:
    """Read one head's parameter columns for one token and add the bias.

    The one place either direction reads ``params``, so both evaluate the maps at
    the same point. All ``PARAM_COLS`` loads are issued before the first use, which
    is what puts them in flight together; the group is the head's whole row, so no
    prefetch depth is chosen here.

    Args:
        gparams: ``(B,T,H*PARAM_COLS)`` projection slice, activation dtype.
        gbias: ``(H*PARAM_COLS,)`` float32 per-head bias, flattened.
        bidx: Batch index.
        token: Token index. Clamped by the caller.
        base: ``head * PARAM_COLS``, the row's first column.
        dtype: Activation element type. Compile-time.

    Returns:
        ``PARAM_COLS`` float32 values in column order, widened on load (I4).
    """
    raw = [gparams[bidx, token, base + slot] for slot in range(PARAM_COLS)]
    return [widen(raw[slot], dtype) + gbias[base + slot] for slot in range(PARAM_COLS)]


def _bc_slots(
    flat: cutlass.Int32,
    t0: cutlass.Int32,
    last: cutlass.Int32,
    tokens: cutlass.Constexpr,
    groups: cutlass.Constexpr,
    state_dim: cutlass.Constexpr,
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Resolve a flat ``bc`` work item into its token, column, group, and lane.

    Args:
        flat: Flat work index over ``tokens * G * 3N``. Clamped here.
        t0: First token of the tile.
        last: ``T - 1``, the read clamp.
        tokens: Tile tokens. Compile-time.
        groups: ``G``. Compile-time.
        state_dim: ``3N``. Compile-time.

    Returns:
        ``(row, column within the half, group, lane, clamped token)``.

    Invariants:
        At ``G == 1`` the group divide folds away at trace time rather than being
        emitted as an identity for ptxas to remove.
    """
    span = groups * state_dim
    idx = cutlass.min(flat, cutlass.Int32(tokens * span - 1))
    row = idx // span
    rest = idx - row * span
    grp = cutlass.Int32(0)
    lane = rest
    if cutlass.const_expr(groups != 1):
        grp = rest // state_dim
        lane = rest - grp * state_dim
    return row, rest, grp, lane, cutlass.min(t0 + row, last)


@cute.jit
def _split_half(
    gbc: cute.Tensor,
    gout: cute.Tensor,
    col0: cutlass.Constexpr,
    bidx: cutlass.Int32,
    t0: cutlass.Int32,
    last: cutlass.Int32,
    seqlen: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    groups: cutlass.Constexpr,
    state_dim: cutlass.Constexpr,
) -> None:
    """Permute one half of ``bc`` into its head-major output.

    Args:
        gbc: ``(B,T,2*G*3N)`` projection slice, activation dtype.
        gout: ``(B,G,T,3N)`` destination, same dtype.
        col0: First ``bc`` column of this half, ``0`` or ``G*3N``. Compile-time.
        bidx: Batch index.
        t0: First token of the tile.
        last: ``T - 1``, the read clamp.
        seqlen: ``T``, the store predicate.
        tid: Thread index.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        groups: ``G``. Compile-time.
        state_dim: ``3N``. Compile-time.

    Invariants:
        Source and destination dtypes are equal, so the copy converts nothing.

        The outer loop is not unrolled. This phase holds the tile's largest item
        count, and unrolling it lets the scheduler hoist every group's loads above
        every group's stores, which costs one live register per load in flight
        across the whole phase and spills.
    """
    total = tokens * groups * state_dim
    span = PREFETCH * threads
    exact = total % span == 0
    for base in cutlass.range(0, total, span, unroll=1):
        held = []
        for slot in cutlass.range_constexpr(PREFETCH):
            flat = base + slot * threads + tid
            row, rest, grp, lane, token = _bc_slots(
                flat, t0, last, tokens, groups, state_dim
            )
            held.append((flat, row, grp, lane, gbc[bidx, token, col0 + rest]))
        for slot in cutlass.range_constexpr(PREFETCH):
            flat, row, grp, lane, value = held[slot]
            token = t0 + row
            inside = token < seqlen
            if cutlass.const_expr(not exact):
                inside = inside & (flat < cutlass.Int32(total))
            if inside:
                gout[bidx, grp, token, lane] = value


@cute.jit
def _join_half(
    gin: cute.Tensor,
    gdbc: cute.Tensor,
    col0: cutlass.Constexpr,
    bidx: cutlass.Int32,
    t0: cutlass.Int32,
    last: cutlass.Int32,
    seqlen: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    groups: cutlass.Constexpr,
    state_dim: cutlass.Constexpr,
) -> None:
    """Gather one head-major cotangent back into its half of ``dbc``.

    Args:
        gin: ``(B,G,T,3N)`` cotangent of ``B`` or of ``C``, activation dtype.
        gdbc: ``(B,T,2*G*3N)`` destination, contiguous, same dtype.
        col0: First ``dbc`` column of this half. Compile-time.
        bidx: Batch index.
        t0: First token of the tile.
        last: ``T - 1``, the read clamp.
        seqlen: ``T``, the store predicate.
        tid: Thread index.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        groups: ``G``. Compile-time.
        state_dim: ``3N``. Compile-time.

    Invariants:
        The outer loop is not unrolled, for the reason :func:`_split_half` states.
    """
    total = tokens * groups * state_dim
    span = PREFETCH * threads
    exact = total % span == 0
    for base in cutlass.range(0, total, span, unroll=1):
        held = []
        for slot in cutlass.range_constexpr(PREFETCH):
            flat = base + slot * threads + tid
            row, rest, grp, lane, token = _bc_slots(
                flat, t0, last, tokens, groups, state_dim
            )
            held.append((flat, row, rest, gin[bidx, grp, token, lane]))
        for slot in cutlass.range_constexpr(PREFETCH):
            flat, row, rest, value = held[slot]
            token = t0 + row
            inside = token < seqlen
            if cutlass.const_expr(not exact):
                inside = inside & (flat < cutlass.Int32(total))
            if inside:
                gdbc[bidx, token, col0 + rest] = value


@cute.jit
def _emit_maps(
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    gtrans: cute.Tensor,
    gpack: cute.Tensor,
    bidx: cutlass.Int32,
    t0: cutlass.Int32,
    last: cutlass.Int32,
    seqlen: cutlass.Int32,
    tid: cutlass.Int32,
    w_max: cutlass.Float32,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
) -> None:
    """Apply both maps to one ``(head, token)`` pair per thread and pack the result.

    Args:
        gparams: ``(B,T,H*PARAM_COLS)`` projection slice, activation dtype.
        gbias: ``(H*PARAM_COLS,)`` float32 per-head bias, flattened.
        gtrans: ``(B,H,T,4)`` float32, written ``(w_x, w_y, w_z, ls)``.
        gpack: ``(B,H,T,2,4)`` float32, written ``(kr, g, h, 0)`` per tap.
        bidx: Batch index.
        t0: First token of the tile.
        last: ``T - 1``, the read clamp.
        seqlen: ``T``, the store predicate.
        tid: Thread index.
        w_max: Rotation-vector bound.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        heads: ``H``. Compile-time.
        dtype: Activation element type. Compile-time.

    Invariants:
        Lane 3 of each tap is written as a hard zero, so the packing costs no
        concatenation and no fill. Consecutive threads take consecutive tokens of
        one head, so both stores advance with the thread index inside that head's
        token run and one store instruction covers a contiguous span of it.
    """
    items = tokens * heads
    steps = -(-items // threads)
    exact = items % threads == 0
    zero = cutlass.Float32(0.0)
    for step in cutlass.range_constexpr(steps):
        flat = step * threads + tid
        idx = cutlass.min(flat, cutlass.Int32(items - 1))
        head = idx // tokens
        row = idx - head * tokens
        token = cutlass.min(t0 + row, last)
        base = head * PARAM_COLS
        value = _biased_row(gparams, gbias, bidx, token, base, dtype)
        wx, wy, wz = rotvec(value[0], value[1], value[2], w_max)
        ls = log_scale(value[3])
        inside = t0 + row < seqlen
        if cutlass.const_expr(not exact):
            inside = inside & (flat < cutlass.Int32(items))
        if inside:
            gtrans[bidx, head, token, 0] = wx
            gtrans[bidx, head, token, 1] = wy
            gtrans[bidx, head, token, 2] = wz
            gtrans[bidx, head, token, 3] = ls
            for tap in cutlass.range_constexpr(2):
                for comp in cutlass.range_constexpr(3):
                    gpack[bidx, head, token, tap, comp] = value[4 + 3 * tap + comp]
                gpack[bidx, head, token, tap, 3] = zero


def _run_offsets(tokens: int) -> tuple[int, ...]:
    """Shuffle distances of an add-scan over a run of ``tokens`` lanes.

    Args:
        tokens: Run length.

    Returns:
        The doubling distances, shortest first.
    """
    offsets = []
    reach = 1
    while reach < tokens:
        offsets.append(reach)
        reach *= 2
    return tuple(offsets)


def _run_sum(value: Scalar, row: cutlass.Int32, tokens: cutlass.Constexpr) -> Scalar:
    """Sum one value over a token run by warp shuffle.

    An inclusive add-scan by up-shuffles, guarded by a select so that a lane below
    the shuffle distance keeps its own partial instead of doubling it. This is the
    run-length form of the full-warp reduction in
    :mod:`slinoss.ops.block.cute.norm`, whose distances are pinned to the warp
    width; the reduction wanted here is over one head's token run, which is
    shorter.

    ``TILE_TOKENS`` divides the warp width and the lane map is token-innermost, so
    lanes ``[k*tokens, (k+1)*tokens)`` are one head's run, the scan never crosses a
    run boundary, and the guard is the lane's offset inside the run, which is its
    token offset.

    Args:
        value: This lane's contribution.
        row: The lane's token offset inside the run, in ``[0, tokens)``.
        tokens: Run length, a divisor of the warp width. Compile-time.

    Returns:
        The run total, in the run's last lane. Other lanes hold a partial.
    """
    for offset in _run_offsets(tokens):
        shifted = shuffle_up(value, offset)
        value = select(row >= offset, value + shifted, value)
    return value


@cute.jit
def _pull_tokens(
    gdtrans: cute.Tensor,
    gdpack: cute.Tensor,
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    gdparams: cute.Tensor,
    gdbias: cute.Tensor,
    bidx: cutlass.Int32,
    tidx: cutlass.Int32,
    t0: cutlass.Int32,
    last: cutlass.Int32,
    seqlen: cutlass.Int32,
    tid: cutlass.Int32,
    w_max: cutlass.Float32,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
) -> None:
    """Pull both maps back for one ``(head, token)`` pair per thread.

    Args:
        gdtrans: ``(B,H,T,4)`` float32 cotangent of ``trans``.
        gdpack: ``(B,H,T,2,4)`` float32 cotangent of ``K``. Lane 3 is the
            cotangent of a constant and is not read.
        gparams: ``(B,T,H*PARAM_COLS)`` projection slice the forward read.
        gbias: ``(H*PARAM_COLS,)`` float32 per-head bias, flattened.
        gdparams: ``(B,T,H*PARAM_COLS)`` contiguous destination, activation dtype.
        gdbias: ``(B, tiles, H*PARAM_COLS)`` float32 partial, one row per block.
        bidx: Batch index.
        tidx: Token-tile index.
        t0: First token of the tile.
        last: ``T - 1``, the read clamp.
        seqlen: ``T``, the store predicate.
        tid: Thread index.
        w_max: The bound the forward used.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        heads: ``H``. Compile-time.
        dtype: Activation element type. Compile-time.

    Invariants:
        A lane whose token lies past ``T`` takes a zero cotangent, so it contributes
        zero to the run total and the partial bias row of a ragged tile is the sum
        over that tile's real tokens alone.

        The bias reduction is float32 over the gradients before they are narrowed,
        which is what the reference sums; narrowing first would lose the accumulator
        width. It also runs on every lane, outside the store predicate, because a
        shuffle needs the whole warp.
    """
    items = tokens * heads
    steps = -(-items // threads)
    exact = items % threads == 0
    zero = cutlass.Float32(0.0)
    for step in cutlass.range_constexpr(steps):
        flat = step * threads + tid
        idx = cutlass.min(flat, cutlass.Int32(items - 1))
        head = idx // tokens
        row = idx - head * tokens
        token = cutlass.min(t0 + row, last)
        keep = t0 + row < seqlen
        base = head * PARAM_COLS
        value = _biased_row(gparams, gbias, bidx, token, base, dtype)
        cot = [gdtrans[bidx, head, token, comp] for comp in range(4)]
        for tap in cutlass.range_constexpr(2):
            for comp in cutlass.range_constexpr(3):
                cot.append(gdpack[bidx, head, token, tap, comp])
        dx, dy, dz = rotvec_grad(
            value[0],
            value[1],
            value[2],
            select(keep, cot[0], zero),
            select(keep, cot[1], zero),
            select(keep, cot[2], zero),
            w_max,
        )
        # Column order of one head's row, so both stores below are one loop.
        grad = [dx, dy, dz, log_scale_grad(value[3]) * select(keep, cot[3], zero)]
        for slot in cutlass.range_constexpr(2 * 3):
            grad.append(select(keep, cot[4 + slot], zero))
        inside = keep
        if cutlass.const_expr(not exact):
            inside = inside & (flat < cutlass.Int32(items))
        if inside:
            for slot in cutlass.range_constexpr(PARAM_COLS):
                gdparams[bidx, token, base + slot] = narrow(grad[slot], dtype)
        run = [_run_sum(grad[slot], row, tokens) for slot in range(PARAM_COLS)]
        emit = row == cutlass.Int32(tokens - 1)
        if cutlass.const_expr(not exact):
            emit = emit & (flat < cutlass.Int32(items))
        if emit:
            for slot in cutlass.range_constexpr(PARAM_COLS):
                gdbias[bidx, tidx, base + slot] = run[slot]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@cute.kernel
def scanprep_fwd_kernel(
    gparams: cute.Tensor,
    gbc: cute.Tensor,
    gbias: cute.Tensor,
    gtrans: cute.Tensor,
    gpack: cute.Tensor,
    gb: cute.Tensor,
    gc: cute.Tensor,
    seqlen: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    state_dim: cutlass.Constexpr,
    groups: cutlass.Constexpr,
) -> None:
    """Apply the maps, pack, and permute ``bc``. One block per ``(B, token tile)``.

    Args:
        gparams: ``(B,T,H*PARAM_COLS)`` projection slice, activation dtype.
        gbc: ``(B,T,2*G*3N)`` projection slice, same dtype.
        gbias: ``(H*PARAM_COLS,)`` float32 per-head bias, flattened.
        gtrans: ``(B,H,T,4)`` float32, written.
        gpack: ``(B,H,T,2,4)`` float32, written.
        gb: ``(B,G,T,3N)`` activation dtype, written.
        gc: ``(B,G,T,3N)`` activation dtype, written.
        seqlen: ``T``. Dynamic.
        w_max: Rotation-vector bound. Dynamic, so one variant covers every bound.
        dtype: Activation element type. Compile-time.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        heads: ``H``. Compile-time.
        state_dim: ``3N``. Compile-time.
        groups: ``G``. Compile-time.

    Invariants:
        No shared memory and no barrier, so the two phases are ordered only by the
        instruction stream: the map phase issues its ten loads first and the ``bc``
        stream follows. ``T`` is arbitrary: the last tile's stores are predicated
        and its reads clamped.
    """
    tid, _, _ = cute.arch.thread_idx()
    tidx, bidx, _ = cute.arch.block_idx()
    t0 = tidx * tokens
    last = seqlen - 1

    _emit_maps(
        gparams,
        gbias,
        gtrans,
        gpack,
        bidx,
        t0,
        last,
        seqlen,
        tid,
        w_max,
        threads,
        tokens,
        heads,
        dtype,
    )
    for half in cutlass.range_constexpr(2):
        _split_half(
            gbc,
            gb if half == 0 else gc,
            half * groups * state_dim,
            bidx,
            t0,
            last,
            seqlen,
            tid,
            threads,
            tokens,
            groups,
            state_dim,
        )


@cute.jit
def scanprep_fwd(
    gparams: cute.Tensor,
    gbc: cute.Tensor,
    gbias: cute.Tensor,
    gtrans: cute.Tensor,
    gpack: cute.Tensor,
    gb: cute.Tensor,
    gc: cute.Tensor,
    seqlen: cutlass.Int32,
    tiles: cutlass.Int32,
    bsz: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    state_dim: cutlass.Constexpr,
    groups: cutlass.Constexpr,
) -> None:
    """Launch :func:`scanprep_fwd_kernel`.

    ``H``, ``3N``, and ``G`` are compile-time because every phase's lane map and the
    ``bc`` index arithmetic are. Batch, token count, and bound are dynamic.
    """
    scanprep_fwd_kernel(
        gparams,
        gbc,
        gbias,
        gtrans,
        gpack,
        gb,
        gc,
        seqlen,
        w_max,
        dtype,
        threads,
        tokens,
        heads,
        state_dim,
        groups,
    ).launch(grid=(tiles, bsz, 1), block=(threads, 1, 1))


@cute.kernel
def scanprep_bwd_kernel(
    gdtrans: cute.Tensor,
    gdpack: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    gdparams: cute.Tensor,
    gdbc: cute.Tensor,
    gdbias: cute.Tensor,
    seqlen: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    state_dim: cutlass.Constexpr,
    groups: cutlass.Constexpr,
) -> None:
    """Pull all four cotangents back. One block per ``(B, token tile)``.

    Args:
        gdtrans: ``(B,H,T,4)`` float32 cotangent of ``trans``.
        gdpack: ``(B,H,T,2,4)`` float32 cotangent of ``K``.
        gdb: ``(B,G,T,3N)`` cotangent of ``B``, activation dtype.
        gdc: ``(B,G,T,3N)`` cotangent of ``C``, same dtype.
        gparams: ``(B,T,H*PARAM_COLS)`` projection slice the forward read.
        gbias: ``(H*PARAM_COLS,)`` float32 per-head bias, flattened.
        gdparams: ``(B,T,H*PARAM_COLS)`` contiguous, written.
        gdbc: ``(B,T,2*G*3N)`` contiguous, written.
        gdbias: ``(B, tiles, H*PARAM_COLS)`` float32 partial, written.
        seqlen: ``T``. Dynamic.
        w_max: The bound the forward used. Dynamic.
        dtype: Activation element type. Compile-time.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        heads: ``H``. Compile-time.
        state_dim: ``3N``. Compile-time.
        groups: ``G``. Compile-time.

    Invariants:
        No shared memory and no barrier. Every gradient a thread produces stays in
        its registers, and the only cross-thread step is the shuffle reduction
        inside :func:`_pull_tokens`, which is warp-synchronous.
    """
    tid, _, _ = cute.arch.thread_idx()
    tidx, bidx, _ = cute.arch.block_idx()
    t0 = tidx * tokens
    last = seqlen - 1

    _pull_tokens(
        gdtrans,
        gdpack,
        gparams,
        gbias,
        gdparams,
        gdbias,
        bidx,
        tidx,
        t0,
        last,
        seqlen,
        tid,
        w_max,
        threads,
        tokens,
        heads,
        dtype,
    )
    for half in cutlass.range_constexpr(2):
        _join_half(
            gdb if half == 0 else gdc,
            gdbc,
            half * groups * state_dim,
            bidx,
            t0,
            last,
            seqlen,
            tid,
            threads,
            tokens,
            groups,
            state_dim,
        )


@cute.jit
def scanprep_bwd(
    gdtrans: cute.Tensor,
    gdpack: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    gdparams: cute.Tensor,
    gdbc: cute.Tensor,
    gdbias: cute.Tensor,
    seqlen: cutlass.Int32,
    tiles: cutlass.Int32,
    bsz: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    state_dim: cutlass.Constexpr,
    groups: cutlass.Constexpr,
) -> None:
    """Launch :func:`scanprep_bwd_kernel`."""
    scanprep_bwd_kernel(
        gdtrans,
        gdpack,
        gdb,
        gdc,
        gparams,
        gbias,
        gdparams,
        gdbc,
        gdbias,
        seqlen,
        w_max,
        dtype,
        threads,
        tokens,
        heads,
        state_dim,
        groups,
    ).launch(grid=(tiles, bsz, 1), block=(threads, 1, 1))


# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------


def _check_w_max(w_max: float) -> None:
    """Raises:
    ValueError: If ``w_max`` is outside ``(0, pi)``, which I2 requires.
    """
    if not 0.0 < w_max < math.pi:
        raise ValueError(f"w_max must lie in (0, pi), got {w_max}")


def _check_kernel_dtype(named: Named) -> torch.dtype:
    """Narrow the activation dtypes to the ones with a kernel path.

    Set membership only. ``slinoss._guard.check_dtypes`` also rejects a group that
    mixes dtypes, which here belongs to :func:`check_operands`: both operands are
    slices of one projection, so its rejection names that and the reference path
    raises the same words.

    Args:
        named: ``(tensor, name)`` pairs. Order is the reporting order.

    Returns:
        The shared activation dtype.

    Raises:
        TypeError: If an operand dtype has no kernel path. float64 is the reference
            oracle's width and reaches no kernel.
    """
    for tensor, name in named:
        if tensor.dtype not in KERNEL_DTYPES:
            raise TypeError(
                f"{name} has dtype {tensor.dtype}; kernel dtypes: {KERNEL_DTYPES}"
            )
    return named[0][0].dtype


def _check_tokens(bsz: int, seqlen: int) -> None:
    """Raises:
    ValueError: If the call holds no token. A zero-token call has no launchable
        grid, so it is refused rather than special-cased.
    """
    if bsz * seqlen == 0:
        raise ValueError(f"params must hold at least one token, got B={bsz} T={seqlen}")


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def scanprep_forward(
    params: Tensor,
    bc: Tensor,
    param_bias: Tensor,
    *,
    heads: int,
    state_dim: int,
    w_max: float,
) -> ScanParams:
    """Apply the bounded maps, pack, and permute ``bc``, in one launch.

    Args:
        params: Projection slice, ``(B,T,H*PARAM_COLS)``, CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`. Trailing stride one; base
            address and row pitch on a multiple of ``ALIGN_BYTES // itemsize``
            elements. The row pitch itself is read at runtime.
        bc: Projection slice, ``(B,T,2*G*3N)``, same dtype and same layout rules.
        param_bias: ``(H,PARAM_COLS)`` float32, contiguous CUDA.
        heads: ``H``.
        state_dim: ``3N``. ``G`` is read off ``bc``.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanParams`. ``trans`` and ``K`` are
        float32 whatever the input dtype (I4) and lane 3 of each tap is zero;
        ``B`` and ``C`` keep the activation dtype. All four are contiguous.

    Raises:
        ValueError: On a shape mismatch, a trailing stride other than one, a
            zero-token call, an off-CUDA or unaligned operand, a ``G`` that does
            not divide ``heads``, or a ``w_max`` outside ``(0, pi)``.
        TypeError: On a dtype with no kernel path, or on two activation dtypes.
    """
    _check_w_max(w_max)
    named: Named = ((params, "params"), (bc, "bc"))
    dtype = _check_kernel_dtype(named)
    groups = check_operands(params, bc, param_bias, heads, state_dim)
    check_pitched(named)
    check_layout(((param_bias, "param_bias"),))
    bsz, seqlen = int(params.shape[0]), int(params.shape[1])
    _check_tokens(bsz, seqlen)

    wide = {"dtype": torch.float32, "device": params.device}
    trans = torch.empty(bsz, heads, seqlen, 4, **wide)
    packed = torch.empty(bsz, heads, seqlen, 2, 4, **wide)
    narrowed = {"dtype": dtype, "device": params.device}
    bout = torch.empty(bsz, groups, seqlen, state_dim, **narrowed)
    cout = torch.empty(bsz, groups, seqlen, state_dim, **narrowed)

    jit_launch(
        scanprep_fwd,
        (
            params,
            bc,
            param_bias.reshape(-1),
            trans,
            packed,
            bout,
            cout,
            seqlen,
            -(-seqlen // TILE_TOKENS),
            bsz,
            float(w_max),
        ),
        (
            cute_dtype(dtype),
            THREADS,
            TILE_TOKENS,
            heads,
            state_dim,
            groups,
        ),
    )
    return ScanParams(trans=trans, K=packed, B=bout, C=cout)


def scanprep_backward(
    dtrans: Tensor,
    dK: Tensor,
    dB: Tensor,
    dC: Tensor,
    params: Tensor,
    param_bias: Tensor,
    *,
    heads: int,
    state_dim: int,
    w_max: float,
) -> ScanGrads:
    """Pull all four cotangents back to ``params``, ``bc``, and ``param_bias``.

    ``bc`` is not read: the permute is linear, so its pullback is the inverse
    permute of ``dB`` and ``dC``. The cotangent of lane 3 of each tap is the
    cotangent of a constant and is discarded.

    Args:
        dtrans: Cotangent of ``trans``, ``(B,H,T,4)`` float32, contiguous CUDA.
        dK: Cotangent of ``K``, ``(B,H,T,2,4)`` float32, contiguous CUDA.
        dB: Cotangent of ``B``, ``(B,G,T,3N)``, activation dtype, contiguous CUDA.
        dC: Cotangent of ``C``, same shape, dtype, and layout as ``dB``.
        params: The forward's projection slice, ``(B,T,H*PARAM_COLS)``.
        param_bias: The forward's bias, ``(H,PARAM_COLS)`` float32. The maps'
            Jacobians are evaluated at ``params + param_bias``.
        heads: ``H``.
        state_dim: ``3N``.
        w_max: The bound the forward used, in ``(0, pi)``.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanGrads`. ``dparams`` and ``dbc`` are
        contiguous at the activation dtype; ``dparam_bias`` is ``(H,PARAM_COLS)``
        float32.

    Raises:
        ValueError: On a shape mismatch, a trailing stride other than one, a
            zero-token call, a non-float32 pinned cotangent, an off-CUDA or
            unaligned operand, or a ``w_max`` outside ``(0, pi)``.
        TypeError: On a dtype with no kernel path, or on two activation dtypes.
    """
    _check_w_max(w_max)
    named: Named = ((params, "params"), (dB, "dB"), (dC, "dC"))
    dtype = _check_kernel_dtype(named)
    bsz, seqlen, groups = check_cotangents(
        dtrans, dK, dB, dC, params, param_bias, heads, state_dim
    )
    for tensor, name in ((dtrans, "dtrans"), (dK, "dK")):
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must be float32 (I4), got {tensor.dtype}")
    # check_cotangents takes params only to shape-check against it, so the pitched
    # contract on it is this path's to state.
    check_pitched(((params, "params"),))
    check_layout(
        (
            (dtrans, "dtrans"),
            (dK, "dK"),
            (dB, "dB"),
            (dC, "dC"),
            (param_bias, "param_bias"),
        )
    )
    _check_tokens(bsz, seqlen)

    tiles = -(-seqlen // TILE_TOKENS)
    width = heads * PARAM_COLS
    dparams = torch.empty(bsz, seqlen, width, dtype=dtype, device=params.device)
    dbc = torch.empty(
        bsz, seqlen, 2 * groups * state_dim, dtype=dtype, device=params.device
    )
    # Its own tensor, not a gradient doubling as scratch: every element is written
    # by the block that owns the row, and the launch-wide sum is the gradient.
    partial = torch.empty(bsz, tiles, width, dtype=torch.float32, device=params.device)

    jit_launch(
        scanprep_bwd,
        (
            dtrans,
            dK,
            dB,
            dC,
            params,
            param_bias.reshape(-1),
            dparams,
            dbc,
            partial,
            seqlen,
            tiles,
            bsz,
            float(w_max),
        ),
        (
            cute_dtype(dtype),
            THREADS,
            TILE_TOKENS,
            heads,
            state_dim,
            groups,
        ),
    )
    return ScanGrads(
        dparams=dparams,
        dbc=dbc,
        dparam_bias=partial.sum(dim=(0, 1)).view(heads, PARAM_COLS),
    )
