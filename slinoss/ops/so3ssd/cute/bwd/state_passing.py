"""Reverse inter-chunk state recurrence. The adjoint of the forward's.

The forward carries

    zstart_c = s_c,    s_{c+1} = R(Q_c) (a_c s_c + inc_c)

so ``s_c`` reaches the loss through the readout of chunk ``c`` and through the
transition into chunk ``c+1``, while ``inc_c`` reaches it through that transition
alone. Both adjoints run backwards along the same chain:

    acc_C = dstate,   dinc_c = acc_{c+1},
    acc_c = a_c R(Q_c)^T acc_{c+1} + dzstart_c,   dz0 = acc_0

``dinc_c`` is the cotangent of the increment in the global frame, which is where
the forward's factored rotation leaves it: the change of basis back into the
chunk-local frame belongs to the kernel that consumes ``dinc``, exactly as the
rotation belongs to the kernel that produced ``inc``. That is the quantity
:attr:`slinoss.ops.so3ssd.backward.ChunkedBackward.dinc` names, and that
reference is the authority.

Parallel over every independent 3-vector: ``B*H*P*N`` of them, one per thread,
serial over chunks. Nothing is shared between threads, so the kernel holds no
shared memory and no barrier.

``P`` is a multiple of ``HEAD_MULTIPLE`` and ``N`` of ``LANE_MULTIPLE``, so
``P*N`` is a multiple of their product and therefore of the block width. The
launch is exact: no tail tile, no bounds predicate, no padding path.

``dinc`` is written in place over ``dzstart``. Chunk ``c`` is read before chunk
``c`` is written and the launch is a bijection from threads onto 3-vectors, so
only the thread that owns a 3-vector ever touches its three elements and the
alias is entirely intra-thread. The backward carries one ``(B,H,C,P,3N)`` buffer
instead of two.

The chunk loop is a one-deep software pipeline, the mirror of the forward's:
chunk ``c-1`` is fetched, then chunk ``c`` is stored, then chunk ``c`` is
transformed. That order is required rather than incidental. A load of
``gdzstart`` may alias the store that overwrites it, so no compiler will lift the
fetch above a preceding store to the same tensor, and a fetch emitted after the
store is pinned behind an aliasing write: one full round trip per chunk with
nothing to overlap it. Fetching first hides that round trip behind the previous
chunk's rotation. The prefetch index is clamped to chunk zero rather than
branched, so the tail re-reads one chunk whose value is never consumed; a value
carried out of a dynamic branch has no phi node here.

Depth one is the whole gain. The forward measured depth two and depth three at
the standard shape and moved by 0.08%, inside the run-to-run spread, for 14 more
registers; the two kernels have the same chain and the same operand count.

``R(Q_c)^T`` is rebuilt by every thread on every chunk rather than staged once.
That buys a dynamic chunk count, so no recompile per sequence length. The
transpose is a reindexing of a Python tuple at trace time and costs nothing on
the device. Measured cost of the rebuild: ``sm__throughput`` is 13.63% of peak
against ``dram__throughput`` at 84.64%, so the redundant arithmetic is not what
bounds the kernel.

Both cotangent seeds are compile-time. An absent ``dstate`` drops its load and
starts the accumulator at zero; an absent ``dzstart`` -- which is what an absent
``dy`` leaves, the whole buffer identically zero and the kernel that would have
filled it never launched -- drops both its load and its add. Neither is
multiplied by zero at runtime.

DRAM-bound. Analytic traffic at the standard shape, ``B=4 H=12 T=2048 P=48 N=16
L=64`` and so ``C=32``, every tensor float32 and touched once: ``dzstart``
14.156 MB read, ``dinc`` 14.156 MB written over it, ``dstate`` 0.442 MB,
``dz0`` 0.442 MB, ``cquat`` 0.025 MB, ``cscale`` 0.006 MB, total 29.227 MB. NCU
puts the traffic at 29.125 MB and 29.193 MB per launch over two runs, within
0.35% of that count, so nothing is read twice and nothing spills.

Measured at 46.992 us and 47.013 us per launch over two independent NCU runs,
three launches each, clocks unlocked and pass spread 3.4%. Achieved 619.794 GB/s
and 620.947 GB/s against a measured achievable 680.010 GB/s on the verification
host, which is 91.0% and 91.3% of ceiling; the same ceiling puts the floor for
this byte count at 43.0 us. ``long_scoreboard`` is the dominant stall reason at
88.79% and ``issue_active`` is 15.02%: at 84.64% of peak DRAM throughput that is
warps waiting on a saturated bus, not on an idle one. 48 registers per thread, no
shared memory, no bank conflicts, 32.0 bytes per sector, and 32.00 active threads
per warp.

The grid is ``(P*N/threads, B, H)``. That is under twice the SM count only when
``B*H`` is small, and widening it is not available, because the parallelism is
exactly ``B*H*P*N`` independent 3-vectors and no more. At the standard shape it is
288 blocks of 128 threads, 0.34 waves per multiprocessor, and achieved occupancy
is 26.5% against a theoretical 83.33%: capped by the launch, not by the register
count. The class is asserted for shapes whose grid covers the device. A shape with
too few blocks to fill it is a statement about the shape rather than about the
kernel, and no second class is claimed for it here.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch

from slinoss._cute import Stream, jit_launch
from slinoss.ops.so3ssd.cute.common import (
    THREADS,
    mat3_matvec,
    mat3_transpose,
    rot_hom,
)
from slinoss.ops.so3ssd.cute.guard import Named, check_layout, check_pinned

__all__ = [
    "StatePassingBwd",
    "state_passing_backward",
    "state_passing_bwd",
    "state_passing_bwd_kernel",
]


@cute.kernel
def state_passing_bwd_kernel(
    gdzstart: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gdstate: cute.Tensor,
    gdz0: cute.Tensor,
    chunks: cutlass.Int32,
    threads: cutlass.Constexpr,
    has_dstate: cutlass.Constexpr,
    has_dzstart: cutlass.Constexpr,
) -> None:
    """Run the reverse chunk recurrence and overwrite ``dzstart`` with ``dinc``.

    Args:
        gdzstart: ``(B,H,C,3*P*N)`` float32. Read as the cotangent of each
            chunk's start state, written as the cotangent of each chunk's
            increment in the global frame. Read only when ``has_dzstart``; the
            store is unconditional.
        gcquat: ``(B,H,C,4)`` float32 unit chunk rotations.
        gcscale: ``(B,H,C)`` float32 chunk decays, ``exp(2*lp_{L-1})``.
        gdstate: ``(B,H,3*P*N)`` float32 cotangent of the final state. Read only
            when ``has_dstate``; the zero-seed variant is handed ``gdz0`` here so
            the signature has one form.
        gdz0: ``(B,H,3*P*N)`` float32, written with the cotangent of the initial
            state.
        chunks: ``C``. Dynamic.
        threads: Block width. Compile-time.
        has_dstate: Whether a final-state cotangent is supplied. Compile-time.
        has_dzstart: Whether ``gdzstart`` carries a value on entry.
            Compile-time.

    Invariants:
        ``|R(Q_c)| == 1`` and ``a_c`` lies in ``(0, 1]`` by I1, so the reverse
        recurrence cannot grow any more than the forward one can.
        ``grid.x * threads == P*N`` exactly, so no thread is out of range.
    """
    tid, _, _ = cute.arch.thread_idx()
    tile, bidx, hidx = cute.arch.block_idx()
    base = 3 * (tile * threads + tid)

    seed = (cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0))
    if cutlass.const_expr(has_dstate):
        seed = (
            gdstate[bidx, hidx, base],
            gdstate[bidx, hidx, base + 1],
            gdstate[bidx, hidx, base + 2],
        )
    ax, ay, az = seed

    last = chunks - 1
    readout = (cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0))
    if cutlass.const_expr(has_dzstart):
        readout = (
            gdzstart[bidx, hidx, last, base],
            gdzstart[bidx, hidx, last, base + 1],
            gdzstart[bidx, hidx, last, base + 2],
        )
    dzx, dzy, dzz = readout
    qw = gcquat[bidx, hidx, last, 0]
    qx = gcquat[bidx, hidx, last, 1]
    qy = gcquat[bidx, hidx, last, 2]
    qz = gcquat[bidx, hidx, last, 3]
    decayed = gcscale[bidx, hidx, last]

    for step in cutlass.range(chunks):
        chunk = last - step
        # The fetch precedes the store because the store aliases it. Reversed,
        # the load is pinned behind an aliasing write and the pipeline collapses
        # to one round trip per chunk. The index is clamped, not branched: no phi
        # node out of a dynamic branch, and the tail re-read is never consumed.
        prv = cutlass.max(chunk - 1, 0)
        ahead = (cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0))
        if cutlass.const_expr(has_dzstart):
            ahead = (
                gdzstart[bidx, hidx, prv, base],
                gdzstart[bidx, hidx, prv, base + 1],
                gdzstart[bidx, hidx, prv, base + 2],
            )
        pqw = gcquat[bidx, hidx, prv, 0]
        pqx = gcquat[bidx, hidx, prv, 1]
        pqy = gcquat[bidx, hidx, prv, 2]
        pqz = gcquat[bidx, hidx, prv, 3]
        pdecayed = gcscale[bidx, hidx, prv]

        gdzstart[bidx, hidx, chunk, base] = ax
        gdzstart[bidx, hidx, chunk, base + 1] = ay
        gdzstart[bidx, hidx, chunk, base + 2] = az
        # One rotation, then the scale, then the readout cotangent. The forward
        # scales the state alone and rotates the sum; transposing that order puts
        # the scale outside the rotation here.
        rx, ry, rz = mat3_matvec(
            mat3_transpose(rot_hom((qw, qx, qy, qz))), (ax, ay, az)
        )
        if cutlass.const_expr(has_dzstart):
            ax = decayed * rx + dzx
            ay = decayed * ry + dzy
            az = decayed * rz + dzz
        else:
            ax = decayed * rx
            ay = decayed * ry
            az = decayed * rz

        if cutlass.const_expr(has_dzstart):
            dzx, dzy, dzz = ahead
        qw, qx, qy, qz = pqw, pqx, pqy, pqz
        decayed = pdecayed

    gdz0[bidx, hidx, base] = ax
    gdz0[bidx, hidx, base + 1] = ay
    gdz0[bidx, hidx, base + 2] = az


@cute.jit
def state_passing_bwd(
    gdzstart: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gdstate: cute.Tensor,
    gdz0: cute.Tensor,
    chunks: cutlass.Int32,
    tiles: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    stream: Stream,
    threads: cutlass.Constexpr,
    has_dstate: cutlass.Constexpr,
    has_dzstart: cutlass.Constexpr,
) -> None:
    """Launch :func:`state_passing_bwd_kernel`.

    Only ``threads``, ``has_dstate`` and ``has_dzstart`` are compile-time, so
    four compiled variants cover every batch, head, row, lane, and chunk count.
    """
    state_passing_bwd_kernel(
        gdzstart,
        gcquat,
        gcscale,
        gdstate,
        gdz0,
        chunks,
        threads,
        has_dstate,
        has_dzstart,
    ).launch(grid=(tiles, bsz, heads), block=(threads, 1, 1), stream=stream)


class StatePassingBwd(NamedTuple):
    """Result of the reverse chunk recurrence.

    Attributes:
        dinc: ``(B,H,C,P,3N)`` float32 cotangent of each chunk's increment, in
            the global frame. Aliases the ``dzstart`` buffer that was passed in.
        dz0: ``(B,H,P,3N)`` float32 cotangent of the initial state. Present
            whether or not the forward carried one, because the chunk-increment
            adjoint has no use for it and the operator's own gradient contract
            drops it.
    """

    dinc: torch.Tensor
    dz0: torch.Tensor


def state_passing_backward(
    dzstart: torch.Tensor,
    cquat: torch.Tensor,
    cscale: torch.Tensor,
    dstate: torch.Tensor | None = None,
    *,
    has_dzstart: bool = True,
) -> StatePassingBwd:
    """Run the reverse inter-chunk recurrence in place over ``dzstart``.

    Args:
        dzstart: ``(B,H,C,P,3N)`` float32, contiguous. The cotangent of each
            chunk's start state. Consumed: on return this buffer holds ``dinc``.
            With ``has_dzstart`` false it is output-only and its contents on
            entry are never read.
        cquat: ``(B,H,C,4)`` float32, contiguous. Unit chunk rotations
            ``Q_{L-1}``, scalar-first. The same tensor the forward consumed.
        cscale: ``(B,H,C)`` float32, contiguous. Chunk decays
            ``exp(2*lp_{L-1})``.
        dstate: ``(B,H,P,3N)`` float32, contiguous. Zero seed if omitted.
        has_dzstart: Whether ``dzstart`` carries a value. False when the
            operator's ``dy`` was absent: the readout cotangent is then
            identically zero, the kernel that would have produced it is not
            launched, and the load and the add are dropped at compile time
            rather than run against a zero buffer.

    Returns:
        A :class:`StatePassingBwd`. ``dinc`` is a view of ``dzstart``.

    Raises:
        ValueError: On a dtype, device, layout, rank, or shape mismatch, or on a
            ``(P, 3N)`` pair the exact launch cannot cover.
    """
    pinned: Named = ((dzstart, "dzstart"), (cquat, "cquat"), (cscale, "cscale"))
    if dstate is not None:
        pinned = (*pinned, (dstate, "dstate"))
    check_layout(pinned)
    check_pinned(pinned)
    if dzstart.ndim != 5 or cquat.ndim != 4 or cscale.ndim != 3:
        raise ValueError(
            "expected (B,H,C,P,3N), (B,H,C,4) and (B,H,C), got "
            f"{tuple(dzstart.shape)}, {tuple(cquat.shape)}, {tuple(cscale.shape)}"
        )
    bsz, heads, chunks, rows, dim = dzstart.shape
    if tuple(cquat.shape) != (bsz, heads, chunks, 4):
        raise ValueError(f"cquat shape {tuple(cquat.shape)} does not match dzstart")
    if tuple(cscale.shape) != (bsz, heads, chunks):
        raise ValueError(f"cscale shape {tuple(cscale.shape)} does not match dzstart")
    if dim % 3 != 0 or (rows * dim // 3) % THREADS != 0:
        raise ValueError(
            f"P*N must be a multiple of {THREADS} and 3N of 3, got P={rows} 3N={dim}"
        )

    dz0 = torch.empty(bsz, heads, rows, dim, dtype=torch.float32, device=dzstart.device)
    if dstate is None:
        seed = dz0
    else:
        if tuple(dstate.shape) != (bsz, heads, rows, dim):
            raise ValueError(
                f"dstate shape {tuple(dstate.shape)} does not match dzstart"
            )
        seed = dstate

    # The trailing (P,3N) pair is one flat run of 3*P*N floats, so the thread
    # that owns 3-vector v owns exactly elements 3v..3v+2 of it. No index
    # arithmetic, and a warp covers 384 contiguous bytes.
    tiles = rows * dim // 3 // THREADS
    jit_launch(
        state_passing_bwd,
        (
            dzstart.view(bsz, heads, chunks, rows * dim),
            cquat,
            cscale,
            seed.view(bsz, heads, rows * dim),
            dz0.view(bsz, heads, rows * dim),
            chunks,
            tiles,
            bsz,
            heads,
        ),
        (THREADS, dstate is not None, has_dzstart),
    )
    return StatePassingBwd(dinc=dzstart, dz0=dz0)
