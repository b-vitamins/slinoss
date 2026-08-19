"""Autograd entry point for the causal depthwise conv1d.

Saves inputs only. The tap sum and the sigmoid the activation needs are
recomputed in the backward by the same expression the forward uses, so the two
passes cannot disagree about what they share. The saved set is four tensors, or
two in the training path with no bias and no streaming carry.

``ctx.set_materialize_grads(False)``: a caller that reads ``y`` and drops the
returned window leaves that window's cotangent absent rather than zero, and the
backend skips the whole state pullback instead of contracting a zero tensor.

No ``torch.amp.custom_fwd``. It casts every input to the autocast dtype, which
would silently promote or demote the taps and put the reference and the kernel
on different footings.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from slinoss.ops.conv.backends import get, resolve
from slinoss.ops.conv.reference import ConvStep

__all__ = ["CausalConv1dFunction", "causal_conv1d"]

_Outputs = tuple[Tensor, Tensor]
_Grads = tuple[
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    None,
    None,
]


class CausalConv1dFunction(torch.autograd.Function):
    """Differentiable causal depthwise conv1d.

    Returns a positional tuple because :class:`torch.autograd.Function` requires
    one. :func:`causal_conv1d` names the fields.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        initial_state: Tensor | None,
        activation: bool,
        backend_name: str,
    ) -> _Outputs:
        out = get(backend_name).forward(
            x, weight, bias, activation=activation, initial_state=initial_state
        )
        ctx.save_for_backward(x, weight, bias, initial_state)
        ctx.activation = activation
        ctx.backend_name = backend_name
        ctx.set_materialize_grads(False)
        return out.y, out.state

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dy: Tensor | None,
        dfinal_state: Tensor | None,
    ) -> _Grads:
        x, weight, bias, initial_state = ctx.saved_tensors
        grads = get(ctx.backend_name).backward(
            dy,
            dfinal_state,
            x,
            weight,
            bias,
            activation=ctx.activation,
            initial_state=initial_state,
        )
        return (
            grads.dx,
            grads.dweight,
            grads.dbias,
            grads.dinitial_state,
            None,
            None,
        )


def causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    activation: bool = True,
    initial_state: Tensor | None = None,
    backend: str | None = None,
) -> ConvStep:
    """Causal depthwise conv1d. The public operator.

    Threading the returned state into the next call's ``initial_state``
    reproduces the whole-sequence result exactly, so ``T = 1`` is the decode
    step and no separate operator is needed for it.

    Args:
        x: Activations, shape ``(B,T,D)``, contiguous.
        weight: Taps, shape ``(D,W)``. Tap ``k`` multiplies lag ``W-1-k``, so tap
            ``W-1`` is the current token.
        bias: Per-channel bias, shape ``(D,)``, or None.
        activation: Apply SiLU to the tap sum. Fused into the kernel epilogue.
        initial_state: The ``W-1`` timesteps before ``x``, shape ``(B,W-1,D)``.
            Zero if omitted.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device.

    Returns:
        A :class:`ConvStep`.

    Raises:
        ValueError: On a shape, contiguity, device, or width violation, or an
            unusable backend.
        TypeError: On an unsupported dtype.
        RuntimeError: If the selected backend needs the extension and it is not
            built.
    """
    impl = resolve(backend, x.device.type, x.dtype)
    y, state = cast(
        "_Outputs",
        CausalConv1dFunction.apply(
            x, weight, bias, initial_state, activation, impl.name
        ),
    )
    return ConvStep(y=y, state=state)
