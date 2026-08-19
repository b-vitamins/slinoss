"""RMS norm, plain and fused with the residual add. CuTe DSL forward.

    normed = x * rsqrt(mean(x^2) + eps) * weight

The fused form adds the incoming residual stream first and hands the sum back:

    s      = x + residual
    normed = s * rsqrt(mean(s^2) + eps) * weight

``s`` is returned float32 at every operand width. That is what the reference
means by wide: its accumulation dtype is float32 unless an operand is float64,
and float64 reaches no kernel. A stack therefore carries its residual at float32
instead of narrowing once per block.

Parallel decomposition. One block per row over the flattened ``B*T`` axis, 256
threads. The reduction is over ``D`` only, so no row shares anything with
another. Each thread strides over ``D`` accumulating in float32, the lanes of a
warp are combined by a shuffle add-scan, and thread 0 sums the eight warp totals
and broadcasts ``rsqrt(mean + eps)`` through one float32 shared slot. ``D`` is
compile-time, so the strided loop needs no bounds predicate: a thread past the
end runs zero iterations and contributes the scan identity.

The second pass re-reads its input rather than holding the row in registers.
``D`` reaches 4096, so staging it would cost occupancy, and the re-read hits L1
or L2. In the fused form the second pass reads the wide sum it has just written,
so the add is evaluated once and ``normed`` is a function of the residual that is
returned rather than of a second summation.

Shared memory: eight float32 warp partials and one float32 broadcast slot, a
tile budget of 36 B, asserted against the queried capacity by a test. One lane
per warp writes one partial, so the eight writes land in eight banks, and the
broadcast is one address across the block; neither needs a swizzle.

DRAM-bound. Analytic traffic per row at operand itemsize ``i``: the plain norm
moves ``2*D*i`` bytes plus the ``4*D`` float32 weight, and the fused form moves
``D*(i_x + i_residual + 4 + 4 + i_normed)`` -- one read of ``x``, one of
``residual``, one write and one read of the wide sum, one write of ``normed``. No
measured bandwidth is claimed here.
"""

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Scalar,
    Tile,
    dev_tensor,
    f32,
    narrow,
    select,
    shuffle_up,
    smem_bytes,
    widen,
)
from slinoss._guard import check_layout
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.block.reference import NormResidual

__all__ = [
    "NORM_THREADS",
    "PARTIAL_TILE",
    "SCALE_TILE",
    "WARPS",
    "check_operand",
    "norm_smem_bytes",
    "rmsnorm_forward",
    "rmsnorm_fwd",
    "rmsnorm_fwd_kernel",
    "rmsnorm_residual_forward",
    "rmsnorm_residual_fwd",
    "rmsnorm_residual_fwd_kernel",
]

NORM_THREADS = 256
"""Block width of both norm kernels. Eight warps, one block per row."""

WARPS = NORM_THREADS // cute.arch.WARP_SIZE
"""Warps per block, and therefore float32 partials in the cross-warp tile."""

PARTIAL_TILE = Tile((WARPS,), (1,))
"""One float32 per warp. Written by one lane per warp, so eight distinct banks."""

SCALE_TILE = Tile((1,), (1,))
"""The broadcast slot holding ``rsqrt(mean + eps)`` for the row."""


def norm_smem_bytes() -> int:
    """Shared memory both norm kernels hold, in bytes, from the tile layouts."""
    return smem_bytes([(PARTIAL_TILE, 4), (SCALE_TILE, 4)])


def _warp_offsets() -> tuple[int, ...]:
    """Shuffle distances of an inclusive add-scan over a full warp."""
    offsets: list[int] = []
    reach = 1
    while reach < cute.arch.WARP_SIZE:
        offsets.append(reach)
        reach *= 2
    return tuple(offsets)


WARP_OFFSETS = _warp_offsets()


# ---------------------------------------------------------------------------
# Device math
# ---------------------------------------------------------------------------


def _warp_total(value: Scalar, lane: cutlass.Int32) -> Scalar:
    """Sum one float32 across a full warp. The last lane holds the total.

    An inclusive add-scan by up-shuffles, guarded by a select so that a lane
    below the shuffle distance keeps its own partial instead of doubling it. The
    up direction is used because it is the one full-warp shuffle whose clamp
    field is already pinned in :mod:`slinoss._cute`.

    Args:
        value: The lane's partial sum.
        lane: Lane index within the warp.

    Returns:
        The warp total in lane ``WARP_SIZE - 1``, a partial prefix elsewhere.
    """
    for offset in WARP_OFFSETS:
        shifted = shuffle_up(value, offset)
        value = select(lane >= offset, value + shifted, value)
    return value


@cute.jit
def _row_scale(
    spart: cute.Tensor,
    sscale: cute.Tensor,
    acc: Scalar,
    tid: cutlass.Int32,
    eps: cutlass.Float32,
    width: cutlass.Constexpr,
) -> None:
    """Reduce the row and leave ``rsqrt(mean + eps)`` in the broadcast slot.

    Entered by the whole block. Both barriers are here rather than in the caller
    because both tiles are private to this reduction, and the trailing barrier is
    what makes the slot readable by every thread on return.

    Args:
        spart: :data:`PARTIAL_TILE`, float32.
        sscale: :data:`SCALE_TILE`, float32. Written with the row scale.
        acc: The thread's sum of squares over its slice of the row.
        tid: Thread index within the block.
        eps: Added to the mean square. Positive, so the ``rsqrt`` argument is
            positive even on an all-zero row.
        width: ``D``. Compile-time.
    """
    warp = tid // cute.arch.WARP_SIZE
    lane = tid - warp * cute.arch.WARP_SIZE
    total = _warp_total(acc, lane)
    if lane == cute.arch.WARP_SIZE - 1:
        spart[warp] = total
    cute.arch.sync_threads()

    if tid == 0:
        block = cutlass.Float32(0.0)
        for index in cutlass.range_constexpr(WARPS):
            block = block + spart[index]
        sscale[0] = f32(cute.rsqrt(block / cutlass.Float32(float(width)) + eps))
    cute.arch.sync_threads()


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@cute.kernel
def rmsnorm_fwd_kernel(
    gx: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Normalize one row per block.

    Args:
        gx: ``(rows, D)`` input, activation dtype.
        gw: ``(D,)`` float32 weight (I4).
        gy: ``(rows, D)`` output, dtype of ``gx``.
        eps: Added to the mean square. Dynamic, so one variant covers every
            epsilon.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.

    Invariants:
        The mean square is accumulated in float32 whatever the operand width, and
        the weight is float32, so only the store narrows.
    """
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(cutlass.Float32, PARTIAL_TILE.layout(), 16)
    sscale = smem.allocate_tensor(cutlass.Float32, SCALE_TILE.layout(), 16)

    src = gx.element_type
    dst = gy.element_type
    acc = cutlass.Float32(0.0)
    for d in cutlass.range(tid, width, threads):
        value = widen(gx[row, d], src)
        acc = acc + value * value

    _row_scale(spart, sscale, acc, tid, eps, width)
    scale = sscale[0]
    for d in cutlass.range(tid, width, threads):
        gy[row, d] = narrow(widen(gx[row, d], src) * scale * gw[d], dst)


@cute.kernel
def rmsnorm_residual_fwd_kernel(
    gx: cute.Tensor,
    gres: cute.Tensor,
    gsum: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    has_residual: cutlass.Constexpr,
) -> None:
    """Add the residual, write the wide sum, then normalize it.

    Args:
        gx: ``(rows, D)`` branch output, activation dtype.
        gres: ``(rows, D)`` incoming residual stream. Read only when
            ``has_residual``; the first-block variant is handed ``gx`` here so
            the signature has one form.
        gsum: ``(rows, D)`` float32, written with ``x + residual`` and read back
            by the second pass.
        gw: ``(D,)`` float32 weight (I4).
        gy: ``(rows, D)`` normed output, dtype of ``gx``.
        eps: Added to the mean square. Dynamic.
        width: ``D``. Compile-time.
        threads: Block width. Compile-time.
        has_residual: Whether a residual stream is supplied. Compile-time.

    Invariants:
        The sum is formed once, in float32, and both outputs derive from that one
        value. Each thread reads back only addresses it wrote itself.
    """
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    spart = smem.allocate_tensor(cutlass.Float32, PARTIAL_TILE.layout(), 16)
    sscale = smem.allocate_tensor(cutlass.Float32, SCALE_TILE.layout(), 16)

    src = gx.element_type
    rsrc = gres.element_type
    dst = gy.element_type
    acc = cutlass.Float32(0.0)
    for d in cutlass.range(tid, width, threads):
        value = widen(gx[row, d], src)
        if cutlass.const_expr(has_residual):
            value = value + widen(gres[row, d], rsrc)
        gsum[row, d] = value
        acc = acc + value * value

    _row_scale(spart, sscale, acc, tid, eps, width)
    scale = sscale[0]
    for d in cutlass.range(tid, width, threads):
        gy[row, d] = narrow(gsum[row, d] * scale * gw[d], dst)


@cute.jit
def rmsnorm_fwd(
    gx: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_fwd_kernel`, one block per row."""
    rmsnorm_fwd_kernel(gx, gw, gy, eps, width, threads).launch(
        grid=(rows, 1, 1), block=(threads, 1, 1)
    )


@cute.jit
def rmsnorm_residual_fwd(
    gx: cute.Tensor,
    gres: cute.Tensor,
    gsum: cute.Tensor,
    gw: cute.Tensor,
    gy: cute.Tensor,
    eps: cutlass.Float32,
    rows: cutlass.Int32,
    width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    has_residual: cutlass.Constexpr,
) -> None:
    """Launch :func:`rmsnorm_residual_fwd_kernel`, one block per row."""
    rmsnorm_residual_fwd_kernel(
        gx, gres, gsum, gw, gy, eps, width, threads, has_residual
    ).launch(grid=(rows, 1, 1), block=(threads, 1, 1))


# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------


def check_operand(tensor: Tensor, name: str) -> None:
    """Reject an operand no block kernel can read. Shared by both modules.

    The layout half comes from :func:`slinoss._guard.check_layout`; the dtype
    policy is the block's own, and is wider than the scan's because these kernels
    are rowwise and read float32 natively.

    Args:
        tensor: The operand.
        name: Name used in the message.

    Raises:
        ValueError: If the tensor is off CUDA or not contiguous.
        TypeError: If the dtype has no kernel path.
    """
    check_layout(((tensor, name),))
    if tensor.dtype not in KERNEL_DTYPES:
        raise TypeError(
            f"{name} has dtype {tensor.dtype}; kernel dtypes: {KERNEL_DTYPES}"
        )


def _check_norm(x: Tensor, weight: Tensor, eps: float) -> tuple[int, int]:
    """Validate the operands shared by both norm entry points.

    Args:
        x: Input, ``(..., D)``.
        weight: ``(D,)`` float32.
        eps: Added to the mean square.

    Returns:
        ``(rows, D)``, the flattened extents the launch uses.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, or a non-CUDA or non-contiguous
            operand. An empty operand has no launchable grid, so it is refused
            rather than special-cased.
        TypeError: On a dtype with no kernel path.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one axis")
    if x.numel() == 0:
        raise ValueError(f"x must hold at least one row, got {tuple(x.shape)}")
    width = int(x.shape[-1])
    if tuple(weight.shape) != (width,):
        raise ValueError(f"weight must be ({width},), got {tuple(weight.shape)}")
    if weight.dtype is not torch.float32:
        raise ValueError(f"weight must be float32 (I4), got {weight.dtype}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    check_operand(x, "x")
    check_operand(weight, "weight")
    return x.numel() // width, width


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def rmsnorm_forward(x: Tensor, weight: Tensor, *, eps: float) -> Tensor:
    """RMS norm over the trailing axis, in one launch.

    Args:
        x: Shape ``(..., D)``, contiguous CUDA, one of :data:`slinoss._precision.KERNEL_DTYPES`.
        weight: Shape ``(D,)`` float32, contiguous CUDA.
        eps: Added to the mean square. Positive.

    Returns:
        Shape ``(..., D)``, dtype and layout of ``x``.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, or a non-CUDA or non-contiguous
            operand.
        TypeError: On a dtype with no kernel path.
    """
    rows, width = _check_norm(x, weight, eps)
    out = torch.empty_like(x)
    rmsnorm_fwd(
        dev_tensor(x.view(rows, width)),
        dev_tensor(weight),
        dev_tensor(out.view(rows, width)),
        float(eps),
        rows,
        width,
        NORM_THREADS,
    )
    return out


def rmsnorm_residual_forward(
    x: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    *,
    eps: float,
) -> NormResidual:
    """Add the residual and normalize the sum, in one launch.

    Args:
        x: Branch output, ``(..., D)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`.
        residual: Incoming residual stream, same shape, one of
            :data:`slinoss._precision.KERNEL_DTYPES`, or None for the first block of a stack. Its
            dtype is independent of ``x``: the stream arrives float32 while the
            branch output is low precision.
        weight: Shape ``(D,)`` float32, contiguous CUDA.
        eps: Added to the mean square. Positive.

    Returns:
        A :class:`slinoss.ops.block.NormResidual`. ``normed`` carries the dtype of
        ``x``; ``residual`` is float32.

    Raises:
        ValueError: On a rank or shape mismatch, an empty operand, a non-positive
            ``eps``, a non-float32 weight, or a non-CUDA or non-contiguous
            operand.
        TypeError: On a dtype with no kernel path.
    """
    rows, width = _check_norm(x, weight, eps)
    if residual is not None:
        if tuple(residual.shape) != tuple(x.shape):
            raise ValueError(
                f"residual must be {tuple(x.shape)}, got {tuple(residual.shape)}"
            )
        check_operand(residual, "residual")

    normed = torch.empty_like(x)
    total = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    stream = x if residual is None else residual
    rmsnorm_residual_fwd(
        dev_tensor(x.view(rows, width)),
        dev_tensor(stream.view(rows, width)),
        dev_tensor(total.view(rows, width)),
        dev_tensor(weight),
        dev_tensor(normed.view(rows, width)),
        float(eps),
        rows,
        width,
        NORM_THREADS,
        residual is not None,
    )
    return NormResidual(normed=normed, residual=total)
