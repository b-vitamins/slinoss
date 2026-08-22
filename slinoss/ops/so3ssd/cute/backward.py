"""Host orchestrator for the CuTe backward.

Five backward launches, in order:

1. ``chunk_prefix_bwd`` -- the two chunk-local transition prefixes, scanned once
   per ``(batch, head, chunk)`` into a ``5 * L`` float32 workspace a chunk.
2. ``start_passing_bwd`` -- the readout half of every chunk-start state cotangent
   and the reverse inter-chunk recurrence over it, in one launch that keeps that
   cotangent in shared memory. With ``dy`` absent the readout half is identically
   zero, and ``state_passing_bwd`` runs the recurrence alone in place over an
   allocated buffer.
3. ``chunk_input_bwd`` -- ``dU``, the streaming input carry, and the log-scale and
   closing-transition cotangents.
4. ``chunk_vector_bwd`` -- ``dB``, ``dC``, ``dtrans``, ``dK``, and the streaming
   vector carry.
5. ``boundary_bwd`` -- the chunk-boundary rows of ``dU`` and ``dB``, and the
   streaming terms.

Nothing between the launches. No reshape, no cast, no staging copy: each kernel
writes the layout the next one reads, and every cotangent leaves in the layout the
operator contract states.

Three quantities cross the chunk boundary rather than a token boundary and none of
the four kernels can rebuild one: the chunk-start state and the two chunk
transitions. A caller holding the forward's ``prologue`` passes it and no forward
kernel runs here. A caller without one -- a direct call, or a forward whose
intermediates were freed -- gets ``increment_passing_fwd`` run again to rebuild all
three, which is one launch and no intermediate buffer. Both paths read the same
three tensors from that point on, and the rebuild is the forward's own code, so they
cannot disagree.

The chunk-local prefixes cross no boundary, so nothing about them depends on which
path supplied the three that do. ``chunk_prefix_bwd`` scans them once and the two
launches that would otherwise rescan them per lane band read them instead, from one
pair of tensors, so no two kernels can disagree about a prefix. ``chunk_input_bwd``
rescans, and the workspace is not handed to it.

The three caller-owned buffers cross straight to the kernel that fills them.
``dB`` and ``dC`` reach ``chunk_vector_bwd``'s store and ``dU_init`` reaches
``chunk_input_bwd``'s epilogue, so a supplied destination costs no pass of its
own and a supplied seed costs one extra read rather than a read, a read, and a
write over ``(B,H,T,P)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from slinoss.ops.so3ssd.backward import SO3SSDGrads
from slinoss.ops.so3ssd.cute.bwd.boundary import boundary_backward
from slinoss.ops.so3ssd.cute.bwd.chunk_input import chunk_input_backward
from slinoss.ops.so3ssd.cute.bwd.chunk_vector import chunk_vector_backward
from slinoss.ops.so3ssd.cute.bwd.start_passing import (
    chunk_prefix_backward,
    start_passing_backward,
)
from slinoss.ops.so3ssd.cute.bwd.state_passing import state_passing_backward
from slinoss.ops.so3ssd.cute.fwd.increment_passing import increment_passing_forward
from slinoss.ops.so3ssd.cute.guard import check_cotangents, check_shapes
from slinoss.ops.so3ssd.reference import ScanPrologue, check_grad_band

__all__ = ["so3ssd_bwd_cute"]


def so3ssd_bwd_cute(
    dy: Tensor | None,
    dstate: Tensor | None,
    db_last: Tensor | None,
    du_last: Tensor | None,
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
    dB: Tensor | None = None,
    dC: Tensor | None = None,
    dU_init: Tensor | None = None,
    prologue: ScanPrologue | None = None,
) -> SO3SSDGrads:
    """Chunked SO(3) scan backward on the CuTe kernels.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, the dtype of ``U``, contiguous, or
            None.
        dstate: ``(B,H,P,3N)`` cotangent of the final state, float32, contiguous,
            or None.
        db_last: ``(B,G,3N)`` cotangent of ``b_last``, or None.
        du_last: ``(B,H,P)`` cotangent of ``u_last``, or None.
        U: ``(B,H,T,P)`` forcing input, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous.
        K: ``(B,H,T,2,4)`` float32, contiguous.
        B: ``(B,G,T,3N)``, the dtype of ``U``, pitched.
        C: ``(B,G,T,3N)``, the dtype of ``U``, pitched.
        chunk_size: Chunk length ``L``. A multiple of 16.
        z0: ``(B,H,P,3N)`` initial state, float32, contiguous, or None.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, or None.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, or None.
        dB: Destination for ``dB``, shaped and typed like ``B``, pitched, or None
            to allocate. Written in full and returned by identity.
        dC: Destination for ``dC``, shaped and typed like ``C``. Like ``dB``.
        dU_init: Addend for ``dU``, shaped and typed like ``U``, pitched, or None.
            Read only. The returned ``dU`` is it plus the cotangent of ``U``.
        prologue: The matching forward's
            :class:`slinoss.ops.so3ssd.reference.ScanPrologue`, read and never
            written, or None to rebuild it with one forward launch. Supplying one
            the forward did not produce at this ``chunk_size`` and these inputs
            gives wrong gradients and raises nothing: the shapes agree.

    Returns:
        A :class:`slinoss.ops.so3ssd.backward.SO3SSDGrads`. ``dz0`` is present
        exactly when ``z0`` was; ``db_prev`` and ``du_prev`` exactly when the
        streaming pair was.

    Raises:
        ValueError: On a layout, rank, shape, extent, or pairing violation, on no
            cotangent at all, or on a shared-memory budget the device cannot carve
            out.
        TypeError: On an activation dtype with no tensor-core path, a
            low-precision float32-pinned operand, or a caller buffer whose dtype
            is not its operand's.
    """
    shape = check_shapes(U, trans, K, (B, "B"), (C, "C"))
    check_cotangents(dy, dstate, db_last, du_last, shape)
    for buffer, operand, name in (
        (dB, B, "dB"),
        (dC, C, "dC"),
        (dU_init, U, "dU_init"),
    ):
        if buffer is not None:
            check_grad_band(buffer, operand, name)
    bsz, heads, groups, seqlen, rows, dim = shape
    chunks = -(-seqlen // chunk_size)

    # Rebuild the chunk boundary only when the caller kept none of it. The launch
    # costs more than every other host-side term here put together, and both paths
    # leave the same three read-only tensors.
    if prologue is None:
        rebuilt = increment_passing_forward(
            U, trans, K, B, chunk_size, z0=z0, u_prev=u_prev, b_prev=b_prev
        )
        prologue = ScanPrologue(
            zstart=rebuilt.zstart,
            cquat=rebuilt.cquat,
            cscale=rebuilt.cscale,
        )

    # One scan of the two chunk-local prefixes for the two launches that read them.
    # The workspace outlives both, so it is allocated here rather than inside either.
    prefixes = chunk_prefix_backward(trans, chunk_size, groups)

    # The chunk-start cotangent never reaches memory: the fused launch contracts
    # each chunk's readout cotangent into shared memory and the reverse recurrence
    # consumes it there, which deletes a (B,H,C,P,3N) float32 round trip.
    #
    # An absent dy leaves the readout half of that cotangent identically zero, and
    # with it the reason to fuse: the fused kernel would run its GEMM against
    # zeros. That path keeps the recurrence alone over an allocated buffer, where
    # has_dzstart drops the load and the add at compile time. Nothing reads that
    # buffer before the recurrence writes it, so it carries the dtype its consumers
    # want rather than the recurrence's own.
    if dy is not None:
        reverse = start_passing_backward(
            dy,
            trans,
            C,
            prologue.cquat,
            prologue.cscale,
            chunk_size,
            dstate,
            prefixes=prefixes,
        )
    else:
        reverse = state_passing_backward(
            torch.empty(bsz, heads, chunks, rows, dim, dtype=U.dtype, device=U.device),
            prologue.cquat,
            prologue.cscale,
            dstate,
            has_dzstart=False,
        )

    # The chunk-input stage has work whatever dy is: the increment terms survive
    # an absent readout cotangent, so this is the one place a zero fill is
    # cheaper than a second compiled variant of a 400-line kernel.
    forcing = (
        dy
        if dy is not None
        else torch.zeros(bsz, heads, seqlen, rows, dtype=U.dtype, device=U.device)
    )
    inputs = chunk_input_backward(
        forcing,
        U,
        trans,
        K,
        B,
        C,
        reverse.dinc,
        prologue.zstart,
        chunk_size,
        u_prev=u_prev,
        b_prev=b_prev,
        du_init=dU_init,
    )
    vectors = chunk_vector_backward(
        forcing,
        U,
        trans,
        K,
        B,
        C,
        reverse.dinc,
        prologue.zstart,
        inputs.dlogp,
        inputs.dchunk_rot,
        inputs.dchunk_scale,
        chunk_size,
        u_prev=u_prev,
        b_prev=b_prev,
        dB=dB,
        dC=dC,
        prefix_lp=prefixes.lp,
        prefix_q=prefixes.q,
        arrived=inputs.arrived,
    )
    stream = boundary_backward(
        inputs.carry_u,
        vectors.carry_b,
        inputs.dU,
        vectors.dB,
        chunk_size,
        du_last=du_last,
        db_last=db_last,
        want_prev=b_prev is not None,
    )
    return SO3SSDGrads(
        dU=inputs.dU,
        dtrans=vectors.dtrans,
        dK=vectors.dK,
        dB=vectors.dB,
        dC=vectors.dC,
        dz0=reverse.dz0 if z0 is not None else None,
        # The boundary kernel writes both streaming carries float32, because it
        # writes them with the float32 carry buffers they come out of. The
        # operator owes them in the dtype of the inputs they belong to, and they
        # are the two smallest tensors it has, so the narrowing is here rather
        # than a second store dtype in that kernel.
        # B and U carry the dtypes of b_prev and u_prev: the streaming pair joins
        # the activation operand group, which is held to one dtype.
        db_prev=None if stream.db_prev is None else stream.db_prev.to(B.dtype),
        du_prev=None if stream.du_prev is None else stream.du_prev.to(U.dtype),
    )
