"""Cross entropy over a padded class axis. Pure PyTorch, the authority.

    lse  = logsumexp(logits[:, :classes])
    loss = mean(lse - logits[row, labels[row]])

The class count is an argument and never ``logits.shape[-1]``. A head padded to a
tensor-core multiple emits columns past the vocabulary, and a pad column is not a
class a label indexes: it holds whatever the GEMM left there. Reading the width
would put those columns in the partition function and change the loss.

``lse`` crosses the forward-backward boundary. The backward needs the row's
normalizer to form a probability, and it is either saved at 4 B per row or
recomputed by a second pass over the logits, which at any trained vocabulary is
another read of the largest tensor in the step. The save is the smaller cost by
five orders of magnitude.

Accumulation is float32 or wider: the partition function sums the whole class axis,
and a bfloat16 sum over 50,257 terms loses the small ones. ``promote_types`` rather
than a fixed width, so a float64 call stays float64 and this module is the fp64
oracle its own kernel is checked against.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = [
    "LABEL_DTYPES",
    "XentGrads",
    "XentState",
    "xent_bwd_ref",
    "xent_ref",
    "xent_shape",
]

LABEL_DTYPES: tuple[torch.dtype, ...] = (torch.int32, torch.int64)
"""Label dtypes the operator reads.

int64 is what an embedding index is in torch, so accepting it is what keeps a cast
kernel off the path; int32 is accepted because the kernel widens the index anyway.
"""


class XentState(NamedTuple):
    """What the forward produces.

    Attributes:
        loss: 0-d float32, the mean over rows.
        lse: ``(rows,)`` float32, the per-row log partition function over the first
            ``classes`` columns. Saved for the backward.
    """

    loss: Tensor
    lse: Tensor


class XentGrads(NamedTuple):
    """What the backward produces.

    Attributes:
        dlogits: Same shape, dtype and layout as the logits. Columns at or past
            ``classes`` are exactly zero: the loss does not read them, so they
            cannot carry a gradient.
    """

    dlogits: Tensor


def xent_shape(logits: Tensor, labels: Tensor, classes: int) -> tuple[int, int]:
    """Validate the operands and return the row count and the operand width.

    Args:
        logits: ``(rows, width)``.
        labels: ``(rows,)`` integer, every entry in ``[0, classes)``.
        classes: Classes the labels index, at most ``width``.

    Returns:
        ``(rows, width)``.

    Raises:
        ValueError: On a rank other than two, a label count that is not the row
            count, an empty operand, or a class count outside ``[1, width]``.
        TypeError: If the labels are not an integer dtype.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (rows, width), got {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be (rows,), got {tuple(labels.shape)}")
    rows, width = (int(extent) for extent in logits.shape)
    if int(labels.shape[0]) != rows:
        raise ValueError(
            f"labels must hold one entry per row, got {int(labels.shape[0])} for "
            f"{rows} rows"
        )
    if rows == 0 or width == 0:
        raise ValueError(f"logits must be non-empty, got {tuple(logits.shape)}")
    if not 1 <= classes <= width:
        raise ValueError(f"classes must lie in [1, {width}], got {classes}")
    if labels.dtype not in LABEL_DTYPES:
        raise TypeError(f"labels have dtype {labels.dtype}; expected {LABEL_DTYPES}")
    return rows, width


def _accumulator(logits: Tensor, classes: int) -> Tensor:
    """The class band at accumulation width."""
    dtype = torch.promote_types(logits.dtype, torch.float32)
    return logits[:, :classes].to(dtype)


def xent_ref(logits: Tensor, labels: Tensor, /, *, classes: int) -> XentState:
    """Mean cross entropy of ``logits`` against ``labels``. The reference.

    Args:
        logits: ``(rows, width)``, any supported dtype. Finite.
        labels: ``(rows,)`` integer, every entry in ``[0, classes)``.
        classes: Classes the labels index, at most ``width``.

    Returns:
        The loss and the per-row normalizer, both float32 at any operand width
        below it and float64 at float64.

    Raises:
        ValueError: On a shape violation.
        TypeError: On a label dtype that is not an integer.
    """
    xent_shape(logits, labels, classes)
    acc = _accumulator(logits, classes)
    lse = torch.logsumexp(acc, dim=-1)
    target = acc.gather(-1, labels.long()[:, None]).squeeze(-1)
    return XentState(loss=(lse - target).mean(), lse=lse)


def xent_bwd_ref(
    dloss: Tensor,
    logits: Tensor,
    labels: Tensor,
    lse: Tensor,
    /,
    *,
    classes: int,
) -> XentGrads:
    """Gradient of :func:`xent_ref` with respect to the logits. The reference.

    Args:
        dloss: 0-d, the cotangent of the mean.
        logits: ``(rows, width)``, as the forward read them.
        labels: ``(rows,)`` integer.
        lse: ``(rows,)``, the forward's normalizer.
        classes: Classes the labels index.

    Returns:
        ``dlogits``, the operand's shape and dtype, zero at every column at or
        past ``classes``.

    Raises:
        ValueError: On a shape violation.
        TypeError: On a label dtype that is not an integer.
    """
    rows, _ = xent_shape(logits, labels, classes)
    acc = _accumulator(logits, classes)
    prob = torch.exp(acc - lse.to(acc.dtype)[:, None])
    index = labels.long()[:, None]
    prob.scatter_add_(
        -1, index, torch.full_like(index, -1, dtype=prob.dtype, device=prob.device)
    )
    dlogits = torch.zeros_like(logits)
    dlogits[:, :classes] = (prob * (dloss.to(prob.dtype) / rows)).to(logits.dtype)
    return XentGrads(dlogits=dlogits)
