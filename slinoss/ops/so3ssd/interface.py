"""Autograd entry point for the SO(3) scan.

Saves the inputs and the chunk boundary. Every chunk-local intermediate -- the
log-scale prefix, the quaternion prefix, the 3x3 table, the rotated ``B`` and
``C``, the two score matrices, the decay mask, and the chunk increments -- is
recomputed in the backward by the kernel that reads it, so none of it reaches
global memory and no two kernels can disagree about it.

The chunk boundary is the exception. The chunk-start state and the two chunk
transitions cross a chunk boundary rather than a token boundary, so no backward
kernel can rebuild one from what it already reads: rebuilding them means running
the forward's first two launches again, 13% of the operator's device time at the
acceptance shape. Holding them instead costs one ``(B,H,C,P,3N)`` float32 buffer
per layer, which is the same buffer the rebuilding backward allocated for itself,
so the trade is live activation across the step against launches inside it. They
are held. A backend that returns no boundary saves three ``None`` and its own
backward rebuilds; the reference does.

In the training path, with no streaming carry, the saved set is eight tensors per
layer.

No ``torch.amp.custom_fwd``. It casts every input to the autocast dtype, which
would demote ``trans``, ``K``, and ``z0`` and break the float32 pinning of
:mod:`slinoss._precision`.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from slinoss.ops.so3ssd.backends import get, resolve
from slinoss.ops.so3ssd.reference import ScanPrologue, SO3SSDResult

__all__ = ["SO3SSDFunction", "so3ssd"]

_Outputs = tuple[Tensor, Tensor, Tensor, Tensor]
_Grads = tuple[
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    None,
    None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
]


class SO3SSDFunction(torch.autograd.Function):
    """Differentiable chunked SO(3) scan.

    Returns a positional tuple because :class:`torch.autograd.Function` requires
    one. :func:`so3ssd` names the fields.
    """

    @staticmethod
    def forward(
        ctx: Any,
        U: Tensor,
        trans: Tensor,
        K: Tensor,
        B: Tensor,
        C: Tensor,
        chunk_size: int,
        backend_name: str,
        z0: Tensor | None,
        b_prev: Tensor | None,
        u_prev: Tensor | None,
    ) -> _Outputs:
        out = get(backend_name).forward(
            U, trans, K, B, C, chunk_size, z0=z0, b_prev=b_prev, u_prev=u_prev
        )
        # Held, not rebuilt: nothing writes the chunk boundary after the forward's
        # inter-chunk recurrence leaves it, and autograd's version counter fails
        # loudly if that ever stops being true.
        prologue = (None, None, None) if out.prologue is None else out.prologue
        ctx.save_for_backward(U, trans, K, B, C, z0, b_prev, u_prev, *prologue)
        ctx.chunk_size = chunk_size
        ctx.backend_name = backend_name
        return out.y, out.state, out.b_last, out.u_last

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dy: Tensor | None,
        dstate: Tensor | None,
        db_last: Tensor | None,
        du_last: Tensor | None,
    ) -> _Grads:
        U, trans, K, B, C, z0, b_prev, u_prev, zstart, cquat, cscale = ctx.saved_tensors
        grads = get(ctx.backend_name).backward(
            dy,
            dstate,
            db_last,
            du_last,
            U,
            trans,
            K,
            B,
            C,
            ctx.chunk_size,
            z0=z0,
            b_prev=b_prev,
            u_prev=u_prev,
            prologue=(None if zstart is None else ScanPrologue(zstart, cquat, cscale)),
        )
        return (
            grads.dU,
            grads.dtrans,
            grads.dK,
            grads.dB,
            grads.dC,
            None,
            None,
            grads.dz0,
            grads.db_prev,
            grads.du_prev,
        )


def so3ssd(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    *,
    z0: Tensor | None = None,
    b_prev: Tensor | None = None,
    u_prev: Tensor | None = None,
    backend: str | None = None,
) -> SO3SSDResult:
    """Chunked SO(3) scan with an analytic backward. The public operator.

    Args:
        U: Input weights, shape ``(B,H,T,P)``.
        trans: ``(w_x, w_y, w_z, ls)`` per token, shape ``(B,H,T,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned. Tap index 0
            is previous and 1 is current; lane 3 is ignored.
        B: Input vectors, shape ``(B,G,T,3N)``. Grouped: ``G`` divides ``H`` and
            head ``h`` reads group ``h // (H // G)``.
        C: Output vectors, shape ``(B,G,T,3N)``. Grouped like ``B``.
        chunk_size: Chunk length ``L``.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, shape ``(B,G,3N)``.
        u_prev: ``u_{-1}`` for a streaming split, shape ``(B,H,P)``.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device.

    Returns:
        A :class:`SO3SSDResult`. ``prologue`` is ``None``: the chunk boundary is
        the backward's, and the graph holds it.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation, a
            non-positive ``chunk_size``, or an unusable backend.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    impl = resolve(backend, U.device.type, U.dtype)
    y, state, b_last, u_last = cast(
        "_Outputs",
        SO3SSDFunction.apply(
            U, trans, K, B, C, chunk_size, impl.name, z0, b_prev, u_prev
        ),
    )
    return SO3SSDResult(y=y, state=state, b_last=b_last, u_last=u_last)
