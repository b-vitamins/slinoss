"""Autograd entry point for the scan's parameter frontier.

Saves ``params`` and ``transition_bias``. The maps' Jacobians are evaluated at their
sum, so both are needed; nothing else is, because neither packed output is read
back. That is two saved tensors, one of them ``(H,4)``.

No ``torch.amp.custom_fwd``. It casts every input to the autocast dtype, which is
the opposite of I4: the maps produce float32 whatever the input width, and the
backend decides the promotion.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from slinoss.ops.scanprep.backends import get, resolve
from slinoss.ops.scanprep.reference import ScanParams

__all__ = ["ScanPrepFunction", "scanprep"]

_Packed = tuple[Tensor, Tensor]
_Grads = tuple[Tensor, Tensor, None, None, None]


class ScanPrepFunction(torch.autograd.Function):
    """Differentiable parameter frontier.

    Returns a positional tuple because :class:`torch.autograd.Function` requires
    one. :func:`scanprep` names the fields.
    """

    @staticmethod
    def forward(
        ctx: Any,
        params: Tensor,
        transition_bias: Tensor,
        heads: int,
        w_max: float,
        backend_name: str,
    ) -> _Packed:
        out = get(backend_name).forward(
            params, transition_bias, heads=heads, w_max=w_max
        )
        ctx.save_for_backward(params, transition_bias)
        ctx.heads = heads
        ctx.w_max = w_max
        ctx.backend_name = backend_name
        return out.trans, out.K

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dtrans: Tensor,
        dK: Tensor,
    ) -> _Grads:
        params, transition_bias = ctx.saved_tensors
        grads = get(ctx.backend_name).backward(
            dtrans,
            dK,
            params,
            transition_bias,
            heads=ctx.heads,
            w_max=ctx.w_max,
        )
        return (grads.dparams, grads.dtransition_bias, None, None, None)


def scanprep(
    params: Tensor,
    transition_bias: Tensor,
    *,
    heads: int,
    w_max: float,
    backend: str | None = None,
) -> ScanParams:
    """The scan's parameter frontier. The public operator.

    Args:
        params: Projection slice, ``(B,T,H*4)``, activation dtype, trailing
            stride one. Per head, in order ``(w_x, w_y, w_z, ls)``.
        transition_bias: ``(H,4)`` float32, added to every token row.
        heads: ``H``.
        w_max: Rotation-vector chart scale, in ``(0, pi)``.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device and dtype.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanParams`, both fields float32 (I4).

    Raises:
        ValueError: On a shape, layout, device, or bound violation, or an unusable
            backend.
        TypeError: On an unsupported dtype.
    """
    impl = resolve(backend, params.device.type, params.dtype)
    trans, packed = cast(
        "_Packed",
        ScanPrepFunction.apply(params, transition_bias, heads, w_max, impl.name),
    )
    return ScanParams(trans=trans, K=packed)
