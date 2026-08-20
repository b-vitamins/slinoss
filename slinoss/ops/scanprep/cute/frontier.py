"""The scan's parameter frontier. CuTe DSL forward and backward.

One kernel per direction, and one launch for the forward. The forward reads the
token-major parameter slice of the projection and writes ``trans`` and ``K``.
``B`` and ``C`` reach the scan's kernels as pitched bands of the same projection,
so no phase here touches them. The maps themselves are in
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
the sectors the other nine touch.

Operand layout. ``params`` is a slice of one projection output: the trailing axis
has unit stride and the row stride is the full projection width, taken from the
operand at runtime. Nothing here repacks it. The precondition beyond unit trailing
stride is that the base address and the row pitch both land on a sector, which the
producer gets by padding its column offsets and its projection width; it is
checked on the host so no alignment branch reaches the kernel.

The backward's ``dparams`` destination is the same geometry on the store side. A
caller may hand it one band of a wider gradient buffer, and the band's row pitch is
read from the operand exactly as ``params``'s is, so the store needs no repack and
the kernel needs no second addressing form.

The backward is the same decomposition. One thread recovers its own biased
parameter row, applies both Jacobians, stores its ten gradient columns, and then
reduces them over the tile's tokens by warp shuffle. ``TILE_TOKENS`` divides the
warp width, so a token run lies inside one warp and the reduction needs neither
shared memory nor a barrier; the run's last lane writes the block's partial
``dparam_bias`` row. Summing those rows is the backward's second launch,
:func:`slinoss._reduce.reduce_partials` over the ``(batch, tile)`` axes flattened
into one.

Invariants. I1 and I2 are produced here rather than asserted here, and I4 holds:
``trans`` and ``K`` are float32 at every input width, including under autocast,
and the parameter rows are widened on load so both maps and the bias reduction
run in float32. Gradients are narrowed back to the input width once, on the store
to ``dparams``.

DRAM-bound. Per token the forward moves ``i*10*H + 48*H`` bytes at activation
itemsize ``i``: the parameter row in, and 48 bytes of packed float32 per head out.
The backward moves ``i*20*H + 48*H + 40*H/TILE_TOKENS`` bytes: both float32
cotangents and the parameter row in, ``dparams`` out, and the partial bias row. The
per-head bias is ``40*H`` bytes for the whole launch and is read once per block out
of cache. No measured bandwidth is claimed here.
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
from slinoss._guard import check_layout, check_pitched
from slinoss._precision import KERNEL_DTYPES
from slinoss._reduce import reduce_partials
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
    check_dparams_out,
    check_operands,
)

__all__ = [
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
        gdparams: ``(B,T,H*PARAM_COLS)`` destination, activation dtype. Contiguous or
            one band of a wider buffer; the row pitch comes from the operand.
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
    gbias: cute.Tensor,
    gtrans: cute.Tensor,
    gpack: cute.Tensor,
    seqlen: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
) -> None:
    """Apply the maps and pack. One block per ``(B, token tile)``.

    Args:
        gparams: ``(B,T,H*PARAM_COLS)`` projection slice, activation dtype.
        gbias: ``(H*PARAM_COLS,)`` float32 per-head bias, flattened.
        gtrans: ``(B,H,T,4)`` float32, written.
        gpack: ``(B,H,T,2,4)`` float32, written.
        seqlen: ``T``. Dynamic.
        w_max: Rotation-vector bound. Dynamic, so one variant covers every bound.
        dtype: Activation element type. Compile-time.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        heads: ``H``. Compile-time.

    Invariants:
        No shared memory and no barrier. ``T`` is arbitrary: the last tile's stores
        are predicated and its reads clamped.
    """
    tid, _, _ = cute.arch.thread_idx()
    tidx, bidx, _ = cute.arch.block_idx()

    _emit_maps(
        gparams,
        gbias,
        gtrans,
        gpack,
        bidx,
        tidx * tokens,
        seqlen - 1,
        seqlen,
        tid,
        w_max,
        threads,
        tokens,
        heads,
        dtype,
    )


@cute.jit
def scanprep_fwd(
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    gtrans: cute.Tensor,
    gpack: cute.Tensor,
    seqlen: cutlass.Int32,
    tiles: cutlass.Int32,
    bsz: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
) -> None:
    """Launch :func:`scanprep_fwd_kernel`.

    ``H`` is compile-time because the lane map is. Batch, token count, and bound
    are dynamic.
    """
    scanprep_fwd_kernel(
        gparams,
        gbias,
        gtrans,
        gpack,
        seqlen,
        w_max,
        dtype,
        threads,
        tokens,
        heads,
    ).launch(grid=(tiles, bsz, 1), block=(threads, 1, 1))


@cute.kernel
def scanprep_bwd_kernel(
    gdtrans: cute.Tensor,
    gdpack: cute.Tensor,
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    gdparams: cute.Tensor,
    gdbias: cute.Tensor,
    seqlen: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
) -> None:
    """Pull both cotangents back. One block per ``(B, token tile)``.

    Args:
        gdtrans: ``(B,H,T,4)`` float32 cotangent of ``trans``.
        gdpack: ``(B,H,T,2,4)`` float32 cotangent of ``K``.
        gparams: ``(B,T,H*PARAM_COLS)`` projection slice the forward read.
        gbias: ``(H*PARAM_COLS,)`` float32 per-head bias, flattened.
        gdparams: ``(B,T,H*PARAM_COLS)`` at the operand's own row pitch, written.
        gdbias: ``(B, tiles, H*PARAM_COLS)`` float32 partial, written.
        seqlen: ``T``. Dynamic.
        w_max: The bound the forward used. Dynamic.
        dtype: Activation element type. Compile-time.
        threads: Block width. Compile-time.
        tokens: Tile tokens. Compile-time.
        heads: ``H``. Compile-time.

    Invariants:
        No shared memory and no barrier. Every gradient a thread produces stays in
        its registers, and the only cross-thread step is the shuffle reduction
        inside :func:`_pull_tokens`, which is warp-synchronous.
    """
    tid, _, _ = cute.arch.thread_idx()
    tidx, bidx, _ = cute.arch.block_idx()

    _pull_tokens(
        gdtrans,
        gdpack,
        gparams,
        gbias,
        gdparams,
        gdbias,
        bidx,
        tidx,
        tidx * tokens,
        seqlen - 1,
        seqlen,
        tid,
        w_max,
        threads,
        tokens,
        heads,
        dtype,
    )


@cute.jit
def scanprep_bwd(
    gdtrans: cute.Tensor,
    gdpack: cute.Tensor,
    gparams: cute.Tensor,
    gbias: cute.Tensor,
    gdparams: cute.Tensor,
    gdbias: cute.Tensor,
    seqlen: cutlass.Int32,
    tiles: cutlass.Int32,
    bsz: cutlass.Int32,
    w_max: cutlass.Float32,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
) -> None:
    """Launch :func:`scanprep_bwd_kernel`."""
    scanprep_bwd_kernel(
        gdtrans,
        gdpack,
        gparams,
        gbias,
        gdparams,
        gdbias,
        seqlen,
        w_max,
        dtype,
        threads,
        tokens,
        heads,
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


def _check_kernel_dtype(params: Tensor) -> torch.dtype:
    """Narrow the activation dtype to the ones with a kernel path.

    Set membership only. ``slinoss._guard.check_dtypes`` also rejects a group that
    mixes dtypes; either direction here reads one activation operand, so there is no
    group to mix.

    Args:
        params: The projection slice.

    Returns:
        The activation dtype.

    Raises:
        TypeError: If the dtype has no kernel path. float64 is the reference
            oracle's width and reaches no kernel.
    """
    if params.dtype not in KERNEL_DTYPES:
        raise TypeError(
            f"params has dtype {params.dtype}; kernel dtypes: {KERNEL_DTYPES}"
        )
    return params.dtype


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
    param_bias: Tensor,
    *,
    heads: int,
    w_max: float,
) -> ScanParams:
    """Apply the bounded maps and pack, in one launch.

    Args:
        params: Projection slice, ``(B,T,H*PARAM_COLS)``, CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`. Trailing stride one; base
            address and row pitch on a multiple of ``ALIGN_BYTES // itemsize``
            elements. The row pitch itself is read at runtime.
        param_bias: ``(H,PARAM_COLS)`` float32, contiguous CUDA.
        heads: ``H``.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanParams`. Both fields are float32
        whatever the input dtype (I4) and contiguous, and lane 3 of each tap is
        zero.

    Raises:
        ValueError: On a shape mismatch, a trailing stride other than one, a
            zero-token call, an off-CUDA or unaligned operand, or a ``w_max``
            outside ``(0, pi)``.
        TypeError: On a dtype with no kernel path.
    """
    _check_w_max(w_max)
    dtype = _check_kernel_dtype(params)
    check_operands(params, param_bias, heads)
    check_pitched(((params, "params"),))
    check_layout(((param_bias, "param_bias"),))
    bsz, seqlen = int(params.shape[0]), int(params.shape[1])
    _check_tokens(bsz, seqlen)

    wide = {"dtype": torch.float32, "device": params.device}
    trans = torch.empty(bsz, heads, seqlen, 4, **wide)
    packed = torch.empty(bsz, heads, seqlen, 2, 4, **wide)

    jit_launch(
        scanprep_fwd,
        (
            params,
            param_bias.reshape(-1),
            trans,
            packed,
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
        ),
    )
    return ScanParams(trans=trans, K=packed)


def scanprep_backward(
    dtrans: Tensor,
    dK: Tensor,
    params: Tensor,
    param_bias: Tensor,
    *,
    heads: int,
    w_max: float,
    dparams: Tensor | None = None,
) -> ScanGrads:
    """Pull both cotangents back to ``params`` and ``param_bias``.

    The cotangent of lane 3 of each tap is the cotangent of a constant and is
    discarded.

    Args:
        dtrans: Cotangent of ``trans``, ``(B,H,T,4)`` float32, contiguous CUDA.
        dK: Cotangent of ``K``, ``(B,H,T,2,4)`` float32, contiguous CUDA.
        params: The forward's projection slice, ``(B,T,H*PARAM_COLS)``.
        param_bias: The forward's bias, ``(H,PARAM_COLS)`` float32. The maps'
            Jacobians are evaluated at ``params + param_bias``.
        heads: ``H``.
        w_max: The bound the forward used, in ``(0, pi)``.
        dparams: Destination for the parameter gradient, or ``None`` to allocate one.
            One band of the mixer's fused gradient buffer is what it is for; see
            :func:`slinoss.ops.scanprep.reference.check_dparams_out`.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanGrads`. ``dparams`` is the supplied
        destination when there is one and a contiguous allocation at the activation
        dtype otherwise; ``dparam_bias`` is ``(H,PARAM_COLS)`` float32.

    Raises:
        ValueError: On a shape mismatch, a trailing stride other than one, a
            zero-token call, a non-float32 pinned cotangent, an off-CUDA or
            unaligned operand, or a ``w_max`` outside ``(0, pi)``.
        TypeError: On a dtype with no kernel path, or a destination whose dtype is
            not that of ``params``.
    """
    _check_w_max(w_max)
    dtype = _check_kernel_dtype(params)
    bsz, seqlen = check_cotangents(dtrans, dK, params, param_bias, heads)
    for tensor, name in ((dtrans, "dtrans"), (dK, "dK")):
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must be float32 (I4), got {tensor.dtype}")
    if dparams is not None:
        check_dparams_out(dparams, params, heads)
    # check_cotangents takes params only to shape-check against it, so the pitched
    # contract on it is this path's to state.
    check_pitched(((params, "params"),))
    check_layout(
        (
            (dtrans, "dtrans"),
            (dK, "dK"),
            (param_bias, "param_bias"),
        )
    )
    _check_tokens(bsz, seqlen)

    tiles = -(-seqlen // TILE_TOKENS)
    width = heads * PARAM_COLS
    if dparams is None:
        dparams = torch.empty(bsz, seqlen, width, dtype=dtype, device=params.device)
    # Its own tensor, not a gradient doubling as scratch: every element is written
    # by the block that owns the row, and the launch-wide sum is the gradient.
    partial = torch.empty(bsz, tiles, width, dtype=torch.float32, device=params.device)

    jit_launch(
        scanprep_bwd,
        (
            dtrans,
            dK,
            params,
            param_bias.reshape(-1),
            dparams,
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
        ),
    )
    # One slab of `bsz * tiles` rows: batch and tile are both reduced away, and the
    # buffer is contiguous, so the flattening is a view.
    return ScanGrads(
        dparams=dparams,
        dparam_bias=reduce_partials(partial.view(1, bsz * tiles, width)).view(
            heads, PARAM_COLS
        ),
    )
