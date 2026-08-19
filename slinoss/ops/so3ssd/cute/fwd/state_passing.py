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
``P*N`` is a multiple of their product, which is the block width. The launch is
therefore exact: no tail tile, no bounds predicate, no padding path.

``zstart`` is written in place over ``inc``. Each thread reads ``inc_c`` into
registers before storing ``zstart_c``, so the alias is safe, and the operator
carries one ``(B,H,C,P,3N)`` buffer instead of two.

``M_c`` is rebuilt by every thread on every chunk rather than staged once. That is
twenty redundant FMA per thread-step against a kernel whose arithmetic intensity
is under 0.1 flop/byte, so it buys a dynamic chunk count -- no recompile per
sequence length -- for under 3% of the kernel's own DRAM floor.

DRAM-bound. Traffic is one read and one write of ``inc`` plus four floats per
chunk transition.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch

from slinoss.config import HEAD_MULTIPLE, LANE_MULTIPLE
from slinoss.ops.so3ssd.cute.common import dev_tensor, mat3_matvec, rot_hom

__all__ = [
    "THREADS",
    "StatePassing",
    "state_passing_forward",
    "state_passing_fwd",
    "state_passing_fwd_kernel",
]

# One 3-vector per thread. The width is the product of the two shape multiples,
# which is what makes every launch exact; changing it reintroduces a tail tile.
# Four warps, three float32 of live state, no shared memory: occupancy is capped
# by nothing.
THREADS = HEAD_MULTIPLE * LANE_MULTIPLE


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
        ValueError: On a dtype, rank, or shape mismatch, or on a ``(P, 3N)`` pair
            the exact launch cannot cover.
    """
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
