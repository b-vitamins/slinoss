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

``zstart`` is written in place over ``inc``. The launch is a bijection from threads
onto 3-vectors, so only the thread that owns a 3-vector ever touches its three
elements and the alias is entirely intra-thread. The operator carries one
``(B,H,C,P,3N)`` buffer instead of two.

Store ratio. A thread owns 3-vector ``v``, so a warp's 32 stores cover 384
contiguous bytes as 12 sectors per request rather than the 4 a 32-bit coalesced
store would take. A planar ``(3, P*N)`` state makes each store one plane: measured,
that is 12.000 sectors per store request down to 4.001, loads 5.125 down to 2.125,
and 48 registers down to 42. It does not pay. ``long`` 49.520 us to 50.592 and
91.70% to 86.44% of the DRAM time floor, ``standard`` 45.984 to 46.256 and 98.53%
to 95.49%, ``wide`` 118.336 to 115.888 and 98.41% to 99.38%. The ratio is L1 tag
work, not traffic: at ``long`` the kernel reads 13.85 MB and writes 14.23 MB per
launch against a 14.16 MB ``inc`` buffer, so every element crosses DRAM once each
way, L2 merges the sectors before eviction, and there is no amplification to
remove. ``l2_pct`` stays at 33.7% to 37.8% and ``l1tex_pct`` at 28.2% to 32.6%,
neither near binding. The interleaved layout is
also the one that keeps ``3N`` contiguous for the consumer, so the planar variant
would move a cost onto ``chunk_scan_fwd`` in exchange for none of a gain here.

The chunk loop is a one-deep software pipeline: chunk ``c+1`` is fetched, then
chunk ``c`` is stored, then chunk ``c`` is transformed. That order is required
rather than incidental. A load of ``ginc`` may alias the store that overwrites it,
so no compiler will lift the fetch above a preceding store to the same tensor, and
a fetch emitted after the store is pinned behind an aliasing write: one full round
trip per chunk with nothing to overlap it. Fetching first hides that round trip
behind the previous chunk's rotation. The prefetch index is clamped to the last
chunk rather than branched, so the tail re-reads one chunk whose value is never
consumed; a value carried out of a dynamic branch has no phi node here.

Depth one is the whole gain. Depth two and depth three were measured at the
standard shape and moved the kernel by 0.08%, inside the run-to-run spread, while
raising the register count from 48 to 62. Depth two was remeasured at ``long``,
the shape with the fewest blocks and so the most to gain from more bytes in flight:
49.520 us to 48.960, 91.70% to 91.97% of the DRAM time floor, for the same 48 to 62
registers, theoretical occupancy 83.33% to 66.67%, and achieved occupancy at
``wide`` 62.79% to 59.17%. ``long_scoreboard`` did not move, 82.63% to 83.47%, so
the extra depth bought no overlap rather than trading overlap for something
unmeasured. Little's law predicts the near miss: 18,432 threads carrying 12 B each
is 221 KB in flight against the roughly 476 KB the latency-bandwidth product needs,
and doubling the depth does not close that.

``R(Q_c)`` is rebuilt by every thread on every chunk rather than staged once. That
buys a dynamic chunk count, so no recompile per sequence length. Measured cost, in
this kernel and not its backward twin: ``sm__throughput`` is 13.28% of peak against
``dram__throughput`` 84.03% at ``standard``, and 12.07% against 78.42% at ``long``.
The redundant arithmetic bounds neither. Arithmetic intensity is 2.3 flop/byte, 55
flops per thread per chunk against 24 bytes moved, against a ridge point of 164.4:
DRAM rather than tensor by a factor of seventy, and the kernel issues no tensor
instruction at all.

DRAM-bound at the standard shape: 605.233 GB/s against a fitted achievable 684.708
GB/s, and 97.65% of the DRAM time floor the gate holds the class to. The recurrence
is still the one serial step in the operator, but the serial chain is the rotation,
not the fetch, so the kernel reaches a bandwidth fraction rather than a latency
floor. ``long_scoreboard`` remains the dominant stall reason at 89.06%; at 84.03%
of peak DRAM throughput that is warps waiting on a saturated bus, not on an idle
one. The class is asserted for shapes whose grid covers the device. A shape with too
few blocks to fill it is a statement about the shape rather than about the kernel,
and no second class is claimed for it here.

The grid is ``(P*N/threads, B, H)``. That is under twice the SM count only when
``B*H`` is small, and widening it is not available, because the parallelism is
exactly ``B*H*P*N`` independent 3-vectors and no more. Achieved occupancy is
therefore capped by the launch, not by the register count: 62.63% at ``wide`` with
768 blocks, 26.35% at ``standard`` with 288, and 13.55% at ``long`` with 144 over 84
SMs, all against a register-permitted 83.33%. ``long`` still reaches 91.69% of the
floor at 1.7 blocks per SM, so the launch bound costs the class nothing.

``long`` and ``tiny`` are the two bench shapes whose grid is under twice the SM
count, 144 blocks and 2. ``docs/kernels.md`` allows that only for a serial case
measured under 2% of the step. At ``long`` the kernel is 13.75% of the forward's
device time and 8.14% of its wall, so the exemption is unavailable and the class is
carried by the floor instead, 89.29% on the repo's runner. At ``tiny`` it is 0.45%
of the forward wall and inside the exemption, with no floor verdict because the
traffic fits in L2.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch

from slinoss._cute import Stream, jit_launch
from slinoss.ops.so3ssd.cute.common import THREADS, mat3_matvec, rot_hom
from slinoss.ops.so3ssd.cute.guard import Named, check_layout, check_pinned

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

    last = chunks - 1
    incx = ginc[bidx, hidx, 0, base]
    incy = ginc[bidx, hidx, 0, base + 1]
    incz = ginc[bidx, hidx, 0, base + 2]
    qw = gcquat[bidx, hidx, 0, 0]
    qx = gcquat[bidx, hidx, 0, 1]
    qy = gcquat[bidx, hidx, 0, 2]
    qz = gcquat[bidx, hidx, 0, 3]
    decayed = gcscale[bidx, hidx, 0]

    for c in cutlass.range(chunks):
        # The fetch precedes the store because the store aliases it. Reversed,
        # the load is pinned behind an aliasing write and the pipeline collapses
        # to one round trip per chunk. The index is clamped, not branched: no phi
        # node out of a dynamic branch, and the tail re-read is never consumed.
        nxt = cutlass.min(c + 1, last)
        pincx = ginc[bidx, hidx, nxt, base]
        pincy = ginc[bidx, hidx, nxt, base + 1]
        pincz = ginc[bidx, hidx, nxt, base + 2]
        pqw = gcquat[bidx, hidx, nxt, 0]
        pqx = gcquat[bidx, hidx, nxt, 1]
        pqy = gcquat[bidx, hidx, nxt, 2]
        pqz = gcquat[bidx, hidx, nxt, 3]
        pdecayed = gcscale[bidx, hidx, nxt]

        ginc[bidx, hidx, c, base] = sx
        ginc[bidx, hidx, c, base + 1] = sy
        ginc[bidx, hidx, c, base + 2] = sz
        # The scale multiplies the state alone, then the sum is rotated once. The
        # local increment is already in the chunk-local frame, so it shares the
        # rotation rather than needing its own.
        sx, sy, sz = mat3_matvec(
            rot_hom((qw, qx, qy, qz)),
            (decayed * sx + incx, decayed * sy + incy, decayed * sz + incz),
        )

        incx, incy, incz = pincx, pincy, pincz
        qw, qx, qy, qz = pqw, pqx, pqy, pqz
        decayed = pdecayed

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
    stream: Stream,
    threads: cutlass.Constexpr,
    has_z0: cutlass.Constexpr,
) -> None:
    """Launch :func:`state_passing_fwd_kernel`.

    Only ``threads`` and ``has_z0`` are compile-time, so one compiled variant
    covers every batch, head, row, lane, and chunk count.
    """
    state_passing_fwd_kernel(
        ginc, gcquat, gcscale, gz0, gstate, chunks, threads, has_z0
    ).launch(grid=(tiles, bsz, heads), block=(threads, 1, 1), stream=stream)


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
    pinned: Named = ((inc, "inc"), (cquat, "cquat"), (cscale, "cscale"))
    if z0 is not None:
        pinned = (*pinned, (z0, "z0"))
    check_layout(pinned)
    check_pinned(pinned)
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
        if tuple(z0.shape) != (bsz, heads, rows, dim):
            raise ValueError(f"z0 shape {tuple(z0.shape)} does not match inc")
        start = z0

    # The trailing (P,3N) pair is one flat run of 3*P*N floats, so the thread
    # that owns 3-vector v owns exactly elements 3v..3v+2 of it. No index
    # arithmetic, and a warp covers 384 contiguous bytes.
    tiles = rows * dim // 3 // THREADS
    jit_launch(
        state_passing_fwd,
        (
            inc.view(bsz, heads, chunks, rows * dim),
            cquat,
            cscale,
            start.view(bsz, heads, rows * dim),
            state.view(bsz, heads, rows * dim),
            chunks,
            tiles,
            bsz,
            heads,
        ),
        (THREADS, z0 is not None),
    )
    return StatePassing(zstart=inc, state=state)
