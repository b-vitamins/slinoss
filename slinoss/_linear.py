"""Pullback of ``linear`` over a ``(B,T,*)`` batch.

Stated once, because three autograd nodes own a projection GEMM: the mixer's input
and output projections and the stack's head. Each of them exists to keep a
gradient buffer or a masked column band out of the graph, and none of them can
reach ``torch.autograd`` for the GEMM's pullback.
"""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor

from slinoss._precision import cast_to

__all__ = ["LinearGrads", "linear_backward"]


class LinearGrads(NamedTuple):
    """Gradients of one ``linear``.

    Attributes:
        dinput: ``(B,T,in)``, dtype of ``dout``.
        dweight: ``(out,in)``, dtype of ``dout``.
        dbias: ``(out,)``, dtype of ``dout``, or None when the layer has no bias.
    """

    dinput: Tensor
    dweight: Tensor
    dbias: Tensor | None


def linear_backward(
    dout: Tensor, inp: Tensor, weight: Tensor, *, has_bias: bool
) -> LinearGrads:
    """Pullback of ``linear`` over a ``(B,T,*)`` batch.

    ``has_bias`` is the layer's signature, not a gradient-need test: a gradient
    returned for an absent input is an autograd error rather than an optimization.

    Args:
        dout: Cotangent, ``(B,T,out)``.
        inp: Forward input, ``(B,T,in)``.
        weight: ``(out,in)``, any dtype.
        has_bias: Whether the layer carries a bias.

    Returns:
        A :class:`LinearGrads`.
    """
    # flatten over batch and token is a view when the operand is contiguous. dout
    # is always a GEMM or kernel output, so it is; a caller's non-contiguous x
    # makes the other one copy.
    flat = dout.flatten(0, 1)
    dinput = dout @ cast_to(weight, dout.dtype)
    dweight = flat.t() @ cast_to(inp.flatten(0, 1), dout.dtype)
    return LinearGrads(dinput, dweight, flat.sum(0) if has_bias else None)
