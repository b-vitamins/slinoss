"""Inter-chunk state recurrence.

The only serial step in the operator. One chunk transition is four float32:
``exp(lp_{L-1}) * Q_{L-1}``, packed by
:func:`slinoss.ops.so3ssd.cute.prefix.quat_prefix_endpoint`. Because
:func:`slinoss.ops.so3ssd.cute.common.rot_hom` is homogeneous of degree two, the
matrix it builds from those four floats is exactly ``exp(2*lp_{L-1}) R(Q_{L-1})``,
so the scale needs no separate tensor and no separate multiply.

    zstart_c = s_c,    s_{c+1} = M_c s_c + inc_c,    M_c = rot_hom(ctrans_c)

Parallel over every independent 3-vector: ``B*H*P*N`` of them, one per thread,
serial over chunks. Nothing is shared between threads, so the kernel holds no
shared memory and no barrier.

``P`` is a multiple of ``HEAD_MULTIPLE`` and ``N`` of ``LANE_MULTIPLE``, so
``P*N`` is a multiple of their product and therefore of the block width. The
launch is exact: no tail tile, no bounds predicate, no padding path.

``zstart`` is written in place over ``inc``. Each thread reads ``inc_c`` into
registers before storing ``zstart_c``, so the alias is safe, and the operator
carries one ``(B,H,C,P,3N)`` buffer instead of two.

``M_c`` is rebuilt by every thread on every chunk rather than staged once. That
buys a dynamic chunk count, so no recompile per sequence length. Measured cost:
``sm__throughput`` is 2.10% of peak against ``dram__throughput`` at 10.68%, so the
redundant arithmetic is not what bounds the kernel.

SERIAL-tiny, not DRAM-bound. The chunk recurrence is the one provably serial step
in the operator: chunk ``c+1`` cannot start before ``c`` finishes, so the kernel is
latency bound on that chain and cannot reach a bandwidth fraction. Measured at
10.68% of peak DRAM and 2.10% of peak SM, which is neither of the other two
classes. Held instead to the serial budget: under 2% of step time, asserted from
the committed step-time artifact rather than from this docstring.

The grid is ``(P*N/threads, B, H)``. That is under twice the SM count only when
``B*H`` is small; the kernel is exempt from the block-count rule as the documented
serial step, and widening it is not available, because the parallelism is exactly
``B*H*P*N`` independent 3-vectors and no more.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch

from slinoss._cute import dev_tensor
from slinoss.ops.so3ssd.cute.common import THREADS, mat3_matvec, rot_hom

__all__ = [
    "StatePassing",
    "state_passing_forward",
    "state_passing_fwd",
    "state_passing_fwd_kernel",
]


@cute.kernel
def state_passing_fwd_kernel(
    ginc: cute.Tensor,
    gctrans: cute.Tensor,
    gz0: cute.Tensor,
    gstate: cute.Tensor,
    chunks: cutlass.Int32,
    threads: cutlass.Constexpr,
    has_z0: cutlass.Constexpr,
) -> None:
    """Run the chunk recurrence and overwrite ``inc`` with ``zstart``.

    Args:
        ginc: ``(B,H,C,3*P*N)`` float32. Read as the chunk increments, written as
            the chunk-start states.
        gctrans: ``(B,H,C,4)`` float32 packed chunk transitions.
        gz0: ``(B,H,3*P*N)`` float32 initial state. Read only when ``has_z0``;
            the zero-start variant is handed ``gstate`` here so the signature has
            one form.
        gstate: ``(B,H,3*P*N)`` float32, written with the state after the last
            chunk.
        chunks: ``C``. Dynamic.
        threads: Block width. Compile-time.
        has_z0: Whether an initial state is supplied. Compile-time.

    Invariants:
        Every ``|M_c|`` is at most one (I1), so the recurrence cannot grow.
        ``grid.x * threads == P*N`` exactly, so no thread is out of range.
    """
    tid, _, _ = cute.arch.thread_idx()
    tile, bidx, hidx = cute.arch.block_idx()
    base = 3 * (tile * threads + tid)

    state = (cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0))
    if cutlass.const_expr(has_z0):
        state = (
            gz0[bidx, hidx, base],
            gz0[bidx, hidx, base + 1],
            gz0[bidx, hidx, base + 2],
        )
    sx, sy, sz = state

    for c in cutlass.range(chunks):
        incx = ginc[bidx, hidx, c, base]
        incy = ginc[bidx, hidx, c, base + 1]
        incz = ginc[bidx, hidx, c, base + 2]
        ginc[bidx, hidx, c, base] = sx
        ginc[bidx, hidx, c, base + 1] = sy
        ginc[bidx, hidx, c, base + 2] = sz
        moved = mat3_matvec(
            rot_hom(
                (
                    gctrans[bidx, hidx, c, 0],
                    gctrans[bidx, hidx, c, 1],
                    gctrans[bidx, hidx, c, 2],
                    gctrans[bidx, hidx, c, 3],
                )
            ),
            (sx, sy, sz),
        )
        sx = moved[0] + incx
        sy = moved[1] + incy
        sz = moved[2] + incz

    gstate[bidx, hidx, base] = sx
    gstate[bidx, hidx, base + 1] = sy
    gstate[bidx, hidx, base + 2] = sz


@cute.jit
def state_passing_fwd(
    ginc: cute.Tensor,
    gctrans: cute.Tensor,
    gz0: cute.Tensor,
    gstate: cute.Tensor,
    chunks: cutlass.Int32,
    tiles: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    threads: cutlass.Constexpr,
    has_z0: cutlass.Constexpr,
) -> None:
    """Launch :func:`state_passing_fwd_kernel`.

    Only ``threads`` and ``has_z0`` are compile-time, so one compiled variant
    covers every batch, head, row, lane, and chunk count.
    """
    state_passing_fwd_kernel(
        ginc, gctrans, gz0, gstate, chunks, threads, has_z0
    ).launch(grid=(tiles, bsz, heads), block=(threads, 1, 1))


class StatePassing(NamedTuple):
    """Result of the chunk recurrence.

    Attributes:
        zstart: ``(B,H,C,P,3N)`` float32 state entering each chunk. Aliases the
            ``inc`` buffer that was passed in.
        state: ``(B,H,P,3N)`` float32 state after the last chunk.
    """

    zstart: torch.Tensor
    state: torch.Tensor


def _check_operand(name: str, tensor: torch.Tensor) -> None:
    """Reject a device or layout the kernel cannot read.

    Both checks are on the host side of the launch. A host pointer handed to a
    kernel raises inside CUDA and leaves the context unusable for the rest of the
    process, and a strided operand is either silently misread or fails later in an
    internal reshape.

    Args:
        name: Operand name, for the message.
        tensor: The operand.

    Raises:
        ValueError: If the tensor is not on a cuda device or is not contiguous.
    """
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on a cuda device, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous; no repacking is done")


def state_passing_forward(
    inc: torch.Tensor,
    ctrans: torch.Tensor,
    z0: torch.Tensor | None = None,
) -> StatePassing:
    """Run the inter-chunk recurrence in place over ``inc``.

    Args:
        inc: ``(B,H,C,P,3N)`` float32, contiguous. Consumed: on return this
            buffer holds ``zstart``.
        ctrans: ``(B,H,C,4)`` float32, contiguous. Packed chunk transitions
            ``exp(lp_{L-1}) * Q_{L-1}``.
        z0: ``(B,H,P,3N)`` float32, contiguous. Zero state if omitted.

    Returns:
        A :class:`StatePassing`. ``zstart`` is a view of ``inc``.

    Raises:
        ValueError: On a dtype, device, layout, rank, or shape mismatch, or on a
            ``(P, 3N)`` pair the exact launch cannot cover.
    """
    _check_operand("inc", inc)
    _check_operand("ctrans", ctrans)
    if inc.dtype is not torch.float32 or ctrans.dtype is not torch.float32:
        raise ValueError("state_passing needs float32 inc and ctrans (I4)")
    if inc.ndim != 5 or ctrans.ndim != 4:
        raise ValueError(f"expected (B,H,C,P,3N) and (B,H,C,4), got {tuple(inc.shape)}")
    bsz, heads, chunks, rows, dim = inc.shape
    if tuple(ctrans.shape) != (bsz, heads, chunks, 4):
        raise ValueError(f"ctrans shape {tuple(ctrans.shape)} does not match inc")
    if dim % 3 != 0 or (rows * dim // 3) % THREADS != 0:
        raise ValueError(
            f"P*N must be a multiple of {THREADS} and 3N of 3, got P={rows} 3N={dim}"
        )

    state = torch.empty(bsz, heads, rows, dim, dtype=torch.float32, device=inc.device)
    if z0 is None:
        start = state
    else:
        _check_operand("z0", z0)
        if z0.dtype is not torch.float32:
            raise ValueError("state_passing needs a float32 z0 (I4)")
        if tuple(z0.shape) != (bsz, heads, rows, dim):
            raise ValueError(f"z0 shape {tuple(z0.shape)} does not match inc")
        start = z0

    # The trailing (P,3N) pair is one flat run of 3*P*N floats, so the thread
    # that owns 3-vector v owns exactly elements 3v..3v+2 of it. No index
    # arithmetic, and a warp covers 384 contiguous bytes.
    tiles = rows * dim // 3 // THREADS
    state_passing_fwd(
        dev_tensor(inc.view(bsz, heads, chunks, rows * dim)),
        dev_tensor(ctrans),
        dev_tensor(start.view(bsz, heads, rows * dim)),
        dev_tensor(state.view(bsz, heads, rows * dim)),
        chunks,
        tiles,
        bsz,
        heads,
        THREADS,
        z0 is not None,
    )
    return StatePassing(zstart=inc, state=state)
