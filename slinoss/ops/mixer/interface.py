"""Autograd entry point for the fused mixer tail.

Saves the five operands. The backward recomputes the gate, the pre-norm value, and
the norm scale from them, so no forward intermediate crosses the boundary: five
saved tensors, and the two that are parameters were live anyway.

All five gradients are produced whatever the leaves require. A variant per subset of
``needs_input_grad`` would be a second entry point into the same kernel.

No ``torch.amp.custom_fwd``. It casts every input to the autocast dtype, which would
silently promote or demote the two parameters and put the reference and the kernel
on different footings.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from slinoss.ops.mixer.backends import get, resolve

__all__ = ["MixerTailFunction", "mixer_tail"]

_Grads = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None, None]


class MixerTailFunction(torch.autograd.Function):
    """Differentiable fused tail."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        y: Tensor,
        u: Tensor,
        gate: Tensor,
        d_skip: Tensor,
        weight: Tensor,
        eps: float,
        backend_name: str,
    ) -> Tensor:
        out = get(backend_name).forward(y, u, gate, d_skip, weight, eps=eps)
        ctx.save_for_backward(y, u, gate, d_skip, weight)
        ctx.eps = eps
        ctx.backend_name = backend_name
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dout: Tensor,
    ) -> _Grads:
        y, u, gate, d_skip, weight = ctx.saved_tensors
        grads = get(ctx.backend_name).backward(
            dout, y, u, gate, d_skip, weight, eps=ctx.eps
        )
        return (
            grads.dy,
            grads.du,
            grads.dgate,
            grads.dd_skip,
            grads.dweight,
            None,
            None,
        )


def mixer_tail(
    y: Tensor,
    u: Tensor,
    gate: Tensor,
    d_skip: Tensor,
    weight: Tensor,
    *,
    eps: float,
    backend: str | None = None,
) -> Tensor:
    """Skip, gate, and per-head RMS norm. The public operator.

    Args:
        y: Scan output, ``(B,H,T,P)``, head-major.
        u: Scan input, ``(B,H,T,P)``, head-major.
        gate: Gate, ``(B,T,H*P)``, token-major.
        d_skip: Per-head skip scale, ``(H,)``.
        weight: Per-row norm scale, ``(H,P)``.
        eps: Added to the mean square before the reciprocal square root.
        backend: Backend name, or ``None`` to select the fastest registered backend
            for the device and dtype.

    Returns:
        ``(B,T,H*P)``, token-major, in the dtype of ``y``.

    Raises:
        ValueError: On a shape, layout, device, or epsilon violation, or an unusable
            backend.
        TypeError: On an unsupported or mixed dtype.
    """
    impl = resolve(backend, y.device.type, y.dtype)
    return cast(
        "Tensor",
        MixerTailFunction.apply(y, u, gate, d_skip, weight, eps, impl.name),
    )
