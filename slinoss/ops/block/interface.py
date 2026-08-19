"""Autograd entry points for the block norm and activation.

One :class:`torch.autograd.Function` per family, each saving inputs only.

- :class:`RMSNormFunction` saves ``x`` and ``weight``. The row scale is
  recomputed, and the normed output is not read back: that is two saved tensors
  against two operands and one output.
- :class:`NormResidualFunction` saves ``x``, ``residual``, and ``weight``. The
  wide sum the forward returns is re-formed from the two summands rather than
  saved, so that is three saved tensors against three operands and two outputs,
  and two when the stream is absent.
- :class:`SwiGLUFunction` saves ``gate`` and ``up``. The logistic is recomputed:
  two saved tensors against two operands and one output.

``NormResidualFunction`` sets ``ctx.set_materialize_grads(False)``. The forward
has two outputs and a caller that consumes one leaves the other's cotangent
undefined; without that call torch hands the backward a zero tensor it has just
allocated and the backend contracts over it. The backends already take
``Tensor | None`` in both cotangent slots, so what autograd reports absent is
passed through absent. An absent incoming ``residual`` reaches the backward the
same way, as None rather than as a zero tensor.

No ``torch.amp.custom_fwd``. It casts every input to the autocast dtype, which is
the opposite of I4: the norm weight is float32 and the fused residual comes back
in the accumulation dtype, so the backend decides the promotion.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from slinoss.ops.block.backends import (
    rmsnorm_get,
    rmsnorm_residual_get,
    rmsnorm_residual_resolve,
    rmsnorm_resolve,
    swiglu_get,
    swiglu_resolve,
)
from slinoss.ops.block.reference import NormResidual

__all__ = [
    "NormResidualFunction",
    "RMSNormFunction",
    "SwiGLUFunction",
    "rmsnorm",
    "rmsnorm_residual",
    "swiglu",
]

_NormGrads = tuple[Tensor, Tensor, None, None]
_Outputs = tuple[Tensor, Tensor]
_ResidualGrads = tuple[Tensor | None, Tensor | None, Tensor | None, None, None]
_ActGrads = tuple[Tensor, Tensor, None]


class RMSNormFunction(torch.autograd.Function):
    """Differentiable RMS norm over the trailing axis."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: Tensor,
        weight: Tensor,
        eps: float,
        backend_name: str,
    ) -> Tensor:
        out = rmsnorm_get(backend_name).forward(x, weight, eps=eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.backend_name = backend_name
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dout: Tensor,
    ) -> _NormGrads:
        x, weight = ctx.saved_tensors
        grads = rmsnorm_get(ctx.backend_name).backward(dout, x, weight, eps=ctx.eps)
        return grads.dx, grads.dweight, None, None


class NormResidualFunction(torch.autograd.Function):
    """Differentiable fused residual add and norm.

    Returns a positional tuple because :class:`torch.autograd.Function` requires
    one. :func:`rmsnorm_residual` names the fields.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        residual: Tensor | None,
        weight: Tensor,
        eps: float,
        backend_name: str,
    ) -> _Outputs:
        ctx.set_materialize_grads(False)
        out = rmsnorm_residual_get(backend_name).forward(x, residual, weight, eps=eps)
        ctx.save_for_backward(x, residual, weight)
        ctx.eps = eps
        ctx.backend_name = backend_name
        return out.normed, out.residual

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dnormed: Tensor | None,
        dresidual: Tensor | None,
    ) -> _ResidualGrads:
        x, residual, weight = ctx.saved_tensors
        grads = rmsnorm_residual_get(ctx.backend_name).backward(
            dnormed, dresidual, x, residual, weight, eps=ctx.eps
        )
        return grads.dx, grads.dresidual, grads.dweight, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Differentiable ``silu(gate) * up``."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        gate: Tensor,
        up: Tensor,
        backend_name: str,
    ) -> Tensor:
        out = swiglu_get(backend_name).forward(gate, up)
        ctx.save_for_backward(gate, up)
        ctx.backend_name = backend_name
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dout: Tensor,
    ) -> _ActGrads:
        gate, up = ctx.saved_tensors
        grads = swiglu_get(ctx.backend_name).backward(dout, gate, up)
        return grads.dgate, grads.dup, None


def rmsnorm(
    x: Tensor,
    weight: Tensor,
    *,
    eps: float,
    backend: str | None = None,
) -> Tensor:
    """RMS norm over the trailing axis. The public operator.

    Args:
        x: Shape ``(..., D)``.
        weight: Shape ``(D,)``. Float32 on the kernel backends (I4).
        eps: Added to the mean square. Positive, and the only thing standing
            between a zero row and a division by zero.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device and dtype.

    Returns:
        Shape ``(..., D)``, dtype of ``x``.

    Raises:
        ValueError: On a rank, shape, layout, device, or epsilon violation, or an
            unusable backend.
        TypeError: On a dtype the selected backend does not implement.
    """
    impl = rmsnorm_resolve(backend, x.device.type, x.dtype)
    return cast("Tensor", RMSNormFunction.apply(x, weight, eps, impl.name))


def rmsnorm_residual(
    x: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    *,
    eps: float,
    backend: str | None = None,
) -> NormResidual:
    """Add the residual, then RMS norm over the trailing axis. The public operator.

    Args:
        x: Branch output, shape ``(..., D)``.
        residual: Incoming residual stream, same shape, or None for the first
            block of a stack.
        weight: Shape ``(D,)``. Float32 on the kernel backends (I4).
        eps: Added to the mean square. Positive.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device and dtype.

    Returns:
        A :class:`slinoss.ops.block.NormResidual`. ``normed`` carries the dtype of
        ``x``; ``residual`` comes back in the accumulation dtype, which is what
        keeps a long stack from narrowing its residual stream once per block.

    Raises:
        ValueError: On a rank, shape, layout, device, or epsilon violation, or an
            unusable backend.
        TypeError: On a dtype the selected backend does not implement.
    """
    impl = rmsnorm_residual_resolve(backend, x.device.type, x.dtype)
    normed, total = cast(
        "_Outputs",
        NormResidualFunction.apply(x, residual, weight, eps, impl.name),
    )
    return NormResidual(normed=normed, residual=total)


def swiglu(gate: Tensor, up: Tensor, *, backend: str | None = None) -> Tensor:
    """``silu(gate) * up``. The public operator.

    Args:
        gate: Shape ``(..., D)``.
        up: Same shape.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device and dtype.

    Returns:
        Same shape, dtype of ``up``.

    Raises:
        ValueError: On a shape, layout, or device violation, or an unusable
            backend.
        TypeError: On a dtype the selected backend does not implement.
    """
    impl = swiglu_resolve(backend, gate.device.type, gate.dtype)
    return cast("Tensor", SwiGLUFunction.apply(gate, up, impl.name))
