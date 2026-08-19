"""Inter-chunk state recurrence.

The only serial step in the operator.

    zstart_c = s_c,    s_{c+1} = R(Q_c) (a_c s_c + inc_c)

with ``a_c = exp(2*lp_{L-1})`` and ``Q_c`` the unit quaternion prefix at the end
of chunk ``c``, both from
:func:`slinoss.ops.so3ssd.cute.prefix.chunk_endpoint`. ``inc_c`` arrives in the
chunk-local frame, so the rotation that carries it into the global frame is the
same one the recurrence applies to the state and is factored out of both. That is
why the producing kernel emits a raw local increment: rotating a ``(P,3N)``
accumulator in place needs three neighbouring N columns per lane, which the MMA's
C fragment does not give one thread, so it would cost either a cross-thread
shuffle or a float32 round trip through shared memory.

Parallel over every independent 3-vector: ``B*H*P*N`` of them, one per thread,
serial over chunks. Nothing is shared between threads, so the kernel holds no
shared memory and no barrier.

``P`` is a multiple of ``HEAD_MULTIPLE`` and ``N`` of ``LANE_MULTIPLE``, so
``P*N`` is a multiple of their product and therefore of the block width. The
launch is exact: no tail tile, no bounds predicate, no padding path.

``zstart`` is written in place over ``inc``. Each thread reads ``inc_c`` into
registers before storing ``zstart_c``, so the alias is safe, and the operator
carries one ``(B,H,C,P,3N)`` buffer instead of two.

``R(Q_c)`` is rebuilt by every thread on every chunk rather than staged once. That
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
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gz0: cute.Tensor,
    gstate: cute.Tensor,
    chunks: cutlass.Int32,
    threads: cutlass.Constexpr,
    has_z0: cutlass.Constexpr,
) -> None:
    """Run the chunk recurrence and overwrite ``inc`` with ``zstart``.

    Args:
        ginc: ``(B,H,C,3*P*N)`` float32. Read as the chunk-local increments,
            written as the chunk-start states.
        gcquat: ``(B,H,C,4)`` float32 unit chunk rotations.
        gcscale: ``(B,H,C)`` float32 chunk decays, ``exp(2*lp_{L-1})``.
        gz0: ``(B,H,3*P*N)`` float32 initial state. Read only when ``has_z0``;
            the zero-start variant is handed ``gstate`` here so the signature has
            one form.
        gstate: ``(B,H,3*P*N)`` float32, written with the state after the last
            chunk.
        chunks: ``C``. Dynamic.
        threads: Block width. Compile-time.
        has_z0: Whether an initial state is supplied. Compile-time.

    Invariants:
        ``|R(Q_c)| == 1`` and ``a_c`` lies in ``(0, 1]`` by I1, so the recurrence
        cannot grow. ``grid.x * threads == P*N`` exactly, so no thread is out of
        range.
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
        # The scale multiplies the state alone, then the sum is rotated once. The
        # local increment is already in the chunk-local frame, so it shares the
        # rotation rather than needing its own.
        decayed = gcscale[bidx, hidx, c]
        moved = mat3_matvec(
            rot_hom(
                (
                    gcquat[bidx, hidx, c, 0],
                    gcquat[bidx, hidx, c, 1],
                    gcquat[bidx, hidx, c, 2],
                    gcquat[bidx, hidx, c, 3],
                )
            ),
            (decayed * sx + incx, decayed * sy + incy, decayed * sz + incz),
        )
        sx, sy, sz = moved

    gstate[bidx, hidx, base] = sx
    gstate[bidx, hidx, base + 1] = sy
    gstate[bidx, hidx, base + 2] = sz


@cute.jit
def state_passing_fwd(
    ginc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
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
        ginc, gcquat, gcscale, gz0, gstate, chunks, threads, has_z0
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
    """Reject a device, dtype, or layout the kernel cannot read.

    Every check is on the host side of the launch. A host pointer handed to a
    kernel raises inside CUDA and leaves the context unusable for the rest of the
    process, and a strided operand is either silently misread or fails later in an
    internal reshape.

    Args:
        name: Operand name, for the message.
        tensor: The operand.

    Raises:
        ValueError: If the tensor is not on a cuda device, is not float32, or is
            not contiguous.
    """
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on a cuda device, got {tensor.device}")
    if tensor.dtype is not torch.float32:
        raise ValueError(f"{name} must be float32 (I4), got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous; no repacking is done")


def state_passing_forward(
    inc: torch.Tensor,
    cquat: torch.Tensor,
    cscale: torch.Tensor,
    z0: torch.Tensor | None = None,
) -> StatePassing:
    """Run the inter-chunk recurrence in place over ``inc``.

    Args:
        inc: ``(B,H,C,P,3N)`` float32, contiguous. The chunk-local increments.
            Consumed: on return this buffer holds ``zstart``.
        cquat: ``(B,H,C,4)`` float32, contiguous. Unit chunk rotations
            ``Q_{L-1}``, scalar-first.
        cscale: ``(B,H,C)`` float32, contiguous. Chunk decays
            ``exp(2*lp_{L-1})``.
        z0: ``(B,H,P,3N)`` float32, contiguous. Zero state if omitted.

    Returns:
        A :class:`StatePassing`. ``zstart`` is a view of ``inc``.

    Raises:
        ValueError: On a dtype, device, layout, rank, or shape mismatch, or on a
            ``(P, 3N)`` pair the exact launch cannot cover.
    """
    _check_operand("inc", inc)
    _check_operand("cquat", cquat)
    _check_operand("cscale", cscale)
    if inc.ndim != 5 or cquat.ndim != 4 or cscale.ndim != 3:
        raise ValueError(
            "expected (B,H,C,P,3N), (B,H,C,4) and (B,H,C), got "
            f"{tuple(inc.shape)}, {tuple(cquat.shape)}, {tuple(cscale.shape)}"
        )
    bsz, heads, chunks, rows, dim = inc.shape
    if tuple(cquat.shape) != (bsz, heads, chunks, 4):
        raise ValueError(f"cquat shape {tuple(cquat.shape)} does not match inc")
    if tuple(cscale.shape) != (bsz, heads, chunks):
        raise ValueError(f"cscale shape {tuple(cscale.shape)} does not match inc")
    if dim % 3 != 0 or (rows * dim // 3) % THREADS != 0:
        raise ValueError(
            f"P*N must be a multiple of {THREADS} and 3N of 3, got P={rows} 3N={dim}"
        )

    state = torch.empty(bsz, heads, rows, dim, dtype=torch.float32, device=inc.device)
    if z0 is None:
        start = state
    else:
        _check_operand("z0", z0)
        if tuple(z0.shape) != (bsz, heads, rows, dim):
            raise ValueError(f"z0 shape {tuple(z0.shape)} does not match inc")
        start = z0

    # The trailing (P,3N) pair is one flat run of 3*P*N floats, so the thread
    # that owns 3-vector v owns exactly elements 3v..3v+2 of it. No index
    # arithmetic, and a warp covers 384 contiguous bytes.
    tiles = rows * dim // 3 // THREADS
    state_passing_fwd(
        dev_tensor(inc.view(bsz, heads, chunks, rows * dim)),
        dev_tensor(cquat),
        dev_tensor(cscale),
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
