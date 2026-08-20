"""Autograd entry point for the fused cross entropy.

Saves the logits, the labels, and the row normalizer the forward already reduced.
The logits were live anyway -- the head produced them -- so the boundary carries one
tensor of its own, ``(rows,)`` float32.

Only the logits take a gradient. The labels are integers and the class count is not
a tensor.

No ``torch.amp.custom_fwd``. It casts every input to the autocast dtype, which would
demote a float32 logits tensor and put the reference and the kernel on different
footings.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from slinoss.ops.xent.backends import get, resolve

__all__ = ["CrossEntropyFunction", "cross_entropy"]

_Grads = tuple[Tensor, None, None, None]


class CrossEntropyFunction(torch.autograd.Function):
    """Differentiable fused cross entropy."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        logits: Tensor,
        labels: Tensor,
        classes: int,
        backend_name: str,
    ) -> Tensor:
        state = get(backend_name).forward(logits, labels, classes=classes)
        ctx.save_for_backward(logits, labels, state.lse)
        ctx.classes = classes
        ctx.backend_name = backend_name
        return state.loss

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dloss: Tensor,
    ) -> _Grads:
        logits, labels, lse = ctx.saved_tensors
        grads = get(ctx.backend_name).backward(
            dloss, logits, labels, lse, classes=ctx.classes
        )
        return grads.dlogits, None, None, None


def cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    classes: int,
    backend: str | None = None,
) -> Tensor:
    """Mean cross entropy over a class axis the operand width may exceed.

    Args:
        logits: ``(rows, width)``, contiguous, finite. A sequence reaches this shape
            by flattening the batch and token axes, which is a view.
        labels: ``(rows,)`` int32 or int64, every entry in ``[0, classes)``.
        classes: Classes the labels index, at most ``width``. Never
            ``logits.shape[-1]``: a head padded to a tensor-core multiple emits
            columns past the vocabulary, and a pad column is not a class a label
            indexes.
        backend: Backend name, or ``None`` to select the fastest registered backend
            for the device and dtype.

    Returns:
        The mean loss, 0-d, float32 at any operand width below it.

    Raises:
        ValueError: On a shape, layout or device violation, or an unusable backend.
        TypeError: On an unsupported operand or label dtype.
    """
    impl = resolve(backend, logits.device.type, logits.dtype)
    return cast(
        "Tensor",
        CrossEntropyFunction.apply(logits, labels, classes, impl.name),
    )
