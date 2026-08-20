"""Host orchestrator for the CuTe forward.

Three launches, in order:

1. ``chunk_increment_fwd`` -- every chunk's local increment, its unit chunk
   rotation, and its chunk decay.
2. ``state_passing_fwd`` -- the inter-chunk recurrence, in place over the increment
   buffer, leaving each chunk's start state where its increment was.
3. ``chunk_scan_fwd`` -- every token's output, from the chunk-local score matrix and
   the chunk start state.

Nothing between the launches and nothing after them. No reshape, no cast, no zero
fill, no staging copy: each kernel writes the layout the next one reads, the
increment buffer is reused as the start-state buffer rather than allocated twice,
and the segment carry-out is written by the increment kernel rather than sliced out
of the inputs here. A step that appeared here would be glue on the hot path, and
glue on the hot path is the defect this file exists to make visible.

The chunk-local prefixes do not appear here either. Each kernel recomputes them from
``trans``, so they never reach global memory and the three kernels cannot disagree
about them.
"""

from __future__ import annotations

from torch import Tensor

from slinoss.ops.so3ssd.cute.fwd.chunk_increment import chunk_increment_forward
from slinoss.ops.so3ssd.cute.fwd.chunk_scan import chunk_scan_forward
from slinoss.ops.so3ssd.cute.fwd.state_passing import state_passing_forward
from slinoss.ops.so3ssd.reference import SO3SSDResult

__all__ = ["so3ssd_fwd_cute"]


def so3ssd_fwd_cute(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    /,
    *,
    z0: Tensor | None = None,
    b_prev: Tensor | None = None,
    u_prev: Tensor | None = None,
) -> SO3SSDResult:
    """Chunked SO(3) scan forward on the CuTe kernels.

    Args:
        U: Input weights, ``(B,H,T,P)``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous.
        trans: ``(w_x, w_y, w_z, ls)`` per token, ``(B,H,T,4)`` float32, contiguous.
        K: Per-tap ``(kr, g, h, 0)``, ``(B,H,T,2,4)`` float32, contiguous. Tap index
            0 is previous and 1 is current; lane 3 is a hard zero.
        B: Input vectors, ``(B,G,T,3N)``, the dtype of ``U``, pitched: one column
            band of the fused projection, unit stride on the lane axis. ``G``
            divides ``H``; head ``h`` reads group ``h // (H // G)``.
        C: Output vectors, ``(B,G,T,3N)``, the dtype of ``U``, pitched like ``B``.
            Grouped like ``B``.
        chunk_size: Chunk length ``L``. A multiple of 16.
        z0: Initial state, ``(B,H,P,3N)`` float32, contiguous. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, ``(B,G,3N)``, the dtype of ``U``.
            Supplied with ``u_prev`` or not at all.
        u_prev: ``u_{-1}`` for a streaming split, ``(B,H,P)``, the dtype of ``U``.

    Returns:
        A :class:`slinoss.ops.so3ssd.reference.SO3SSDResult`. ``state`` is float32;
        ``y`` carries the dtype of ``U``.

    Raises:
        ValueError: On a layout, rank, shape, extent, or pairing violation, or a
            shared-memory budget the device cannot carve out.
        TypeError: On an activation dtype with no tensor-core path, or a
            low-precision float32-pinned operand.
    """
    increment = chunk_increment_forward(
        U, trans, K, B, chunk_size, u_prev=u_prev, b_prev=b_prev
    )
    passing = state_passing_forward(
        increment.inc, increment.cquat, increment.cscale, z0
    )
    y = chunk_scan_forward(
        U,
        trans,
        K,
        B,
        C,
        passing.zstart,
        chunk_size,
        u_prev=u_prev,
        b_prev=b_prev,
    )
    return SO3SSDResult(
        y=y,
        state=passing.state,
        b_last=increment.b_last,
        u_last=increment.u_last,
    )
