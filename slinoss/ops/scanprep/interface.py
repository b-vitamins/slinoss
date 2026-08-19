"""Autograd entry point for the scan's parameter frontier.

Saves ``params`` and ``param_bias``. The maps' Jacobians are evaluated at
``params + param_bias``, so both are needed; nothing else is. ``bc`` carries no
information the backward wants, because a permute's pullback is a permute, and
neither packed output is read back. That is two saved tensors, one of them
``(H,10)``, against three operands and four outputs.

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

_Packed = tuple[Tensor, Tensor, Tensor, Tensor]
_Grads = tuple[Tensor, Tensor, Tensor, None, None, None, None]


class ScanPrepFunction(torch.autograd.Function):
    """Differentiable parameter frontier.

    Returns a positional tuple because :class:`torch.autograd.Function` requires
    one. :func:`scanprep` names the fields.
    """

    @staticmethod
    def forward(
        ctx: Any,
        params: Tensor,
        bc: Tensor,
        param_bias: Tensor,
        heads: int,
        state_dim: int,
        w_max: float,
        backend_name: str,
    ) -> _Packed:
        out = get(backend_name).forward(
            params, bc, param_bias, heads=heads, state_dim=state_dim, w_max=w_max
        )
        ctx.save_for_backward(params, param_bias)
        ctx.heads = heads
        ctx.state_dim = state_dim
        ctx.w_max = w_max
        ctx.backend_name = backend_name
        return out.trans, out.K, out.B, out.C

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dtrans: Tensor,
        dK: Tensor,
        dB: Tensor,
        dC: Tensor,
    ) -> _Grads:
        params, param_bias = ctx.saved_tensors
        grads = get(ctx.backend_name).backward(
            dtrans,
            dK,
            dB,
            dC,
            params,
            param_bias,
            heads=ctx.heads,
            state_dim=ctx.state_dim,
            w_max=ctx.w_max,
        )
        return (
            grads.dparams,
            grads.dbc,
            grads.dparam_bias,
            None,
            None,
            None,
            None,
        )


def scanprep(
    params: Tensor,
    bc: Tensor,
    param_bias: Tensor,
    *,
    heads: int,
    state_dim: int,
    w_max: float,
    backend: str | None = None,
) -> ScanParams:
    """The scan's parameter frontier. The public operator.

    Args:
        params: Projection slice, ``(B,T,H*10)``, activation dtype, trailing
            stride one. Per head, in order
            ``(w_x, w_y, w_z, ls, kr0, g0, h0, kr1, g1, h1)``.
        bc: Projection slice, ``(B,T,2*G*3N)``, same dtype, trailing stride one.
            First ``G*3N`` columns are ``B``, second ``G*3N`` are ``C``.
        param_bias: ``(H,10)`` float32, added to every token's row before the
            maps.
        heads: ``H``.
        state_dim: ``3N``. ``G`` is read off ``bc``.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device and dtype.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanParams`. ``trans`` and ``K`` are
        float32 (I4); ``B`` and ``C`` keep the activation dtype.

    Raises:
        ValueError: On a shape, layout, device, or bound violation, or an unusable
            backend.
        TypeError: On an unsupported or mixed dtype.
    """
    impl = resolve(backend, params.device.type, params.dtype)
    trans, packed, bout, cout = cast(
        "_Packed",
        ScanPrepFunction.apply(
            params, bc, param_bias, heads, state_dim, w_max, impl.name
        ),
    )
    return ScanParams(trans=trans, K=packed, B=bout, C=cout)
