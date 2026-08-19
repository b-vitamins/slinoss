"""Block norm and activation. Pure-PyTorch reference.

Three maps, each the authority for one fused kernel:

- :func:`rmsnorm_ref`, an RMS norm over the trailing axis.
- :func:`rmsnorm_residual_ref`, the same norm fused with the residual add. A
  pre-norm stack adds the previous branch output and then normalizes, so the two
  operations are always adjacent and touch the same bytes. Fusing them halves the
  traffic and gives the residual back in the accumulation dtype, so a long stack
  does not accumulate the residual in bfloat16.
- :func:`swiglu_ref`, the FFN activation ``silu(gate) * up``.

Precision. Reductions and the residual accumulate in float32, or float64 when any
operand is float64. Normed output carries the input dtype; the residual is
returned wide, because narrowing it is what the fusion exists to avoid.

Each map has a pullback here, taken by autograd through the forward above it.
Those are the gradient authority the kernels are measured against, and in float64
they are the oracle.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import silu

from slinoss._precision import autocast_disabled, check_supported, pinned_dtype

__all__ = [
    "NormResidual",
    "NormResidualGrads",
    "RMSNormGrads",
    "SwiGLUGrads",
    "rmsnorm_bwd_ref",
    "rmsnorm_ref",
    "rmsnorm_residual_bwd_ref",
    "rmsnorm_residual_ref",
    "swiglu_bwd_ref",
    "swiglu_ref",
]


def _check_norm(x: Tensor, weight: Tensor, eps: float) -> None:
    """Validate the operands shared by both norm entry points.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one axis")
    if tuple(weight.shape) != (int(x.shape[-1]),):
        raise ValueError(
            f"weight must be ({int(x.shape[-1])},), got {tuple(weight.shape)}"
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    check_supported(x, "x")
    check_supported(weight, "weight")


def rmsnorm_ref(x: Tensor, weight: Tensor, *, eps: float) -> Tensor:
    """RMS norm over the trailing axis.

    Args:
        x: Shape ``(..., D)``.
        weight: Shape ``(D,)``.
        eps: Added to the mean square. The summand is non-negative, so this is
            the only thing standing between a zero row and a division by zero.

    Returns:
        Shape ``(..., D)``, dtype of ``x``.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    _check_norm(x, weight, eps)
    dtype = pinned_dtype(x, weight)
    with autocast_disabled(x.device.type):
        wide = x.to(dtype)
        scale = torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps)
        return (wide * scale * weight.to(dtype)).to(x.dtype)


def _check_shape_like(tensor: Tensor, like: Tensor, name: str) -> None:
    """Reject an operand whose shape does not match the tensor it pairs with.

    Args:
        tensor: The operand.
        like: The tensor whose shape it must carry.
        name: Name used in the message.

    Raises:
        ValueError: On a shape mismatch.
    """
    if tuple(tensor.shape) != tuple(like.shape):
        raise ValueError(
            f"{name} must be {tuple(like.shape)}, got {tuple(tensor.shape)}"
        )


class RMSNormGrads(NamedTuple):
    """Gradients of :func:`rmsnorm_ref`.

    Attributes:
        dx: Shape ``(..., D)``, dtype of ``x``.
        dweight: Shape ``(D,)``, dtype of ``weight``.
    """

    dx: Tensor
    dweight: Tensor


def rmsnorm_bwd_ref(
    dout: Tensor,
    x: Tensor,
    weight: Tensor,
    /,
    *,
    eps: float,
) -> RMSNormGrads:
    """Pullback of :func:`rmsnorm_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an
    algebra error passes silently. In float64 this is the gradient authority the
    kernel is measured against.

    Args:
        dout: Cotangent of the output, shape ``(..., D)``.
        x: The forward's input, shape ``(..., D)``.
        weight: The forward's scale, shape ``(D,)``.
        eps: The forward's epsilon.

    Returns:
        A :class:`RMSNormGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    _check_norm(x, weight, eps)
    _check_shape_like(dout, x, "dout")
    leaves = tuple(tensor.detach().requires_grad_(True) for tensor in (x, weight))
    with torch.enable_grad():
        out = rmsnorm_ref(*leaves, eps=eps)
    return RMSNormGrads(*torch.autograd.grad(out, leaves, dout))


def _check_stream(x: Tensor, residual: Tensor | None) -> None:
    """Validate the optional incoming residual stream against the branch output.

    Args:
        x: Branch output, ``(..., D)``.
        residual: Incoming residual stream, or None for the first block.

    Raises:
        ValueError: On a shape mismatch.
        TypeError: On an unsupported dtype.
    """
    if residual is not None:
        _check_shape_like(residual, x, "residual")
        check_supported(residual, "residual")


class NormResidual(NamedTuple):
    """Result of the fused residual add and norm.

    Attributes:
        normed: Normed output, shape ``(..., D)``, dtype of ``x``.
        residual: ``x + residual``, shape ``(..., D)``, in the accumulation
            dtype. Feeds the next block's fused add so the residual stream is
            never narrowed.
    """

    normed: Tensor
    residual: Tensor


def rmsnorm_residual_ref(
    x: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    *,
    eps: float,
) -> NormResidual:
    """Add the residual, then RMS norm over the trailing axis.

    Args:
        x: Branch output, shape ``(..., D)``.
        residual: Incoming residual stream, same shape, or None for the first
            block of a stack.
        weight: Shape ``(D,)``.
        eps: Added to the mean square.

    Returns:
        A :class:`NormResidual`.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    _check_norm(x, weight, eps)
    _check_stream(x, residual)

    operands = (x, weight) if residual is None else (x, residual, weight)
    dtype = pinned_dtype(*operands)
    with autocast_disabled(x.device.type):
        total = x.to(dtype) if residual is None else x.to(dtype) + residual.to(dtype)
        scale = torch.rsqrt(total.square().mean(-1, keepdim=True) + eps)
        normed = (total * scale * weight.to(dtype)).to(x.dtype)
        return NormResidual(normed=normed, residual=total)


class NormResidualGrads(NamedTuple):
    """Gradients of :func:`rmsnorm_residual_ref`.

    Every field is optional because the forward has two outputs and a caller
    need not consume both. A field is None exactly when nothing it depends on was
    given a cotangent, which is what autograd reports for an input outside the
    differentiated graph.

    Attributes:
        dx: ``(..., D)``, dtype of ``x``. None only when neither output carries a
            cotangent.
        dresidual: ``(..., D)``, dtype of the incoming stream. None when the
            forward was called without one, and when neither output carries a
            cotangent.
        dweight: ``(D,)``, dtype of ``weight``. None whenever ``normed`` carries
            no cotangent: the weight does not reach the residual output.
    """

    dx: Tensor | None
    dresidual: Tensor | None
    dweight: Tensor | None


def rmsnorm_residual_bwd_ref(
    dnormed: Tensor | None,
    dresidual: Tensor | None,
    x: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    /,
    *,
    eps: float,
) -> NormResidualGrads:
    """Pullback of :func:`rmsnorm_residual_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP, for the reason
    given in :func:`rmsnorm_bwd_ref`. The two cotangents are pushed through one
    ``grad`` call rather than summed afterwards, so the shared sum is traversed
    once.

    Args:
        dnormed: Cotangent of ``normed``, shape ``(..., D)``, or None when the
            caller consumed only the residual.
        dresidual: Cotangent of ``residual``, same shape, or None when the caller
            consumed only the normed output.
        x: The forward's branch output, shape ``(..., D)``.
        residual: The forward's incoming residual stream, or None.
        weight: The forward's scale, shape ``(D,)``.
        eps: The forward's epsilon.

    Returns:
        A :class:`NormResidualGrads`. Every field is None when both cotangents
        are absent: nothing was differentiated, so there is no gradient to
        report.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    _check_norm(x, weight, eps)
    _check_stream(x, residual)
    if dnormed is not None:
        _check_shape_like(dnormed, x, "dnormed")
    if dresidual is not None:
        _check_shape_like(dresidual, x, "dresidual")
    if dnormed is None and dresidual is None:
        return NormResidualGrads(dx=None, dresidual=None, dweight=None)

    operands = (x, weight) if residual is None else (x, residual, weight)
    leaves = tuple(tensor.detach().requires_grad_(True) for tensor in operands)
    with torch.enable_grad():
        out = (
            rmsnorm_residual_ref(leaves[0], None, leaves[1], eps=eps)
            if residual is None
            else rmsnorm_residual_ref(leaves[0], leaves[1], leaves[2], eps=eps)
        )
    pairs = tuple(
        (tensor, cot)
        for tensor, cot in ((out.normed, dnormed), (out.residual, dresidual))
        if cot is not None
    )
    # allow_unused: the weight does not reach `residual`, so a residual-only
    # cotangent leaves it outside the graph and autograd reports None for it.
    # That None is the answer, not a case to fill with zeros.
    grads = torch.autograd.grad(
        [tensor for tensor, _ in pairs],
        leaves,
        [cot for _, cot in pairs],
        allow_unused=True,
    )
    if residual is None:
        return NormResidualGrads(dx=grads[0], dresidual=None, dweight=grads[1])
    return NormResidualGrads(dx=grads[0], dresidual=grads[1], dweight=grads[2])


def swiglu_ref(gate: Tensor, up: Tensor) -> Tensor:
    """``silu(gate) * up``.

    Args:
        gate: Shape ``(..., D)``.
        up: Same shape.

    Returns:
        Same shape, dtype of ``up``.

    Raises:
        ValueError: On a shape mismatch.
        TypeError: On an unsupported dtype.
    """
    _check_shape_like(up, gate, "up")
    check_supported(gate, "gate")
    check_supported(up, "up")
    dtype = pinned_dtype(gate, up)
    with autocast_disabled(gate.device.type):
        return (silu(gate.to(dtype)) * up.to(dtype)).to(up.dtype)


class SwiGLUGrads(NamedTuple):
    """Gradients of :func:`swiglu_ref`.

    Attributes:
        dgate: Shape ``(..., D)``, dtype of ``gate``.
        dup: Shape ``(..., D)``, dtype of ``up``.
    """

    dgate: Tensor
    dup: Tensor


def swiglu_bwd_ref(dout: Tensor, gate: Tensor, up: Tensor, /) -> SwiGLUGrads:
    """Pullback of :func:`swiglu_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP, for the reason
    given in :func:`rmsnorm_bwd_ref`. The logistic appears in both gradients and
    in the forward, so a written-out form would restate it three times.

    Args:
        dout: Cotangent of the output, shape ``(..., D)``.
        gate: The forward's gate operand, shape ``(..., D)``.
        up: The forward's up operand, same shape.

    Returns:
        A :class:`SwiGLUGrads`.

    Raises:
        ValueError: On a shape mismatch.
        TypeError: On an unsupported dtype.
    """
    _check_shape_like(up, gate, "up")
    _check_shape_like(dout, gate, "dout")
    check_supported(gate, "gate")
    check_supported(up, "up")
    leaves = tuple(tensor.detach().requires_grad_(True) for tensor in (gate, up))
    with torch.enable_grad():
        out = swiglu_ref(*leaves)
    return SwiGLUGrads(*torch.autograd.grad(out, leaves, dout))
