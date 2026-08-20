"""Cross-entropy reference: the formula, the VJP, and the operand rules.

The authority for the formula is ``torch.nn.functional.cross_entropy`` over the class
band alone. The authority for the gradient is float64 autograd through the reference
forward, never the hand-derived expression itself: a derivation error shared between
the reference backward and the kernel would pass a parity test silently.

Every case here runs on the CPU in float64. What the class band excludes is a shape
property, not a precision one, so no device is needed to pin it.
"""

from __future__ import annotations

import pytest
import torch

from slinoss.ops.xent import xent_bwd_ref, xent_ref, xent_shape
from tests.conftest import assert_max_rel

ROWS = 7
CLASSES = 11
WIDTH = 16
"""One padded shape, exercised by every case. ``WIDTH`` exceeds ``CLASSES``, so a
formula that read the operand width instead of the class count fails here."""


def operands() -> tuple[torch.Tensor, torch.Tensor]:
    """A padded logits tensor whose pad columns dominate its class band.

    The pad columns hold values far above every class logit, so including one in the
    partition function moves the loss by orders of magnitude rather than by a
    tolerance.

    Returns:
        ``(logits, labels)``, float64 and int64.
    """
    torch.manual_seed(0)
    logits = torch.randn(ROWS, WIDTH, dtype=torch.float64) * 4.0
    logits[:, CLASSES:] = 30.0
    return logits, torch.randint(0, CLASSES, (ROWS,))


def test_matches_torch_over_the_class_band_alone() -> None:
    """The formula, and that the class count and not the width bounds it.

    Failure mode: a partition function over the operand width. A padded head emits
    columns no label indexes, and the pad columns here are large enough that
    including them would swamp the loss.
    """
    logits, labels = operands()
    want = torch.nn.functional.cross_entropy(logits[:, :CLASSES], labels)
    got = xent_ref(logits, labels, classes=CLASSES)
    assert_max_rel(got.loss, want, 1e-14, "xent/ref-loss")
    assert_max_rel(
        got.lse, torch.logsumexp(logits[:, :CLASSES], dim=-1), 1e-14, "xent/ref-lse"
    )


def test_backward_matches_float64_autograd() -> None:
    """The hand-derived VJP against autograd through the forward.

    Failure mode: a wrong probability, a wrong one-hot column, a wrong mean scale,
    or a pad column that carries a gradient. Autograd is the authority for all four,
    and the cotangent is not one so that the scale is checked rather than cancelled.
    """
    logits, labels = operands()
    cotangent = torch.tensor(0.25, dtype=torch.float64)
    leaf = logits.clone().requires_grad_(True)
    xent_ref(leaf, labels, classes=CLASSES).loss.backward(cotangent)
    assert leaf.grad is not None

    state = xent_ref(logits, labels, classes=CLASSES)
    got = xent_bwd_ref(cotangent, logits, labels, state.lse, classes=CLASSES)
    assert_max_rel(got.dlogits, leaf.grad, 1e-14, "xent/ref-dlogits")
    assert bool((got.dlogits[:, CLASSES:] == 0.0).all())


@pytest.mark.parametrize(
    ("shape", "labels", "classes", "error"),
    [
        pytest.param((4,), (4,), 4, ValueError, id="logits-rank"),
        pytest.param((4, 8), (4, 1), 8, ValueError, id="labels-rank"),
        pytest.param((4, 8), (3,), 8, ValueError, id="row-count"),
        pytest.param((0, 8), (0,), 8, ValueError, id="empty"),
        pytest.param((4, 8), (4,), 9, ValueError, id="classes-past-width"),
        pytest.param((4, 8), (4,), 0, ValueError, id="no-classes"),
    ],
)
def test_rejects_an_operand_it_cannot_index(
    shape: tuple[int, ...],
    labels: tuple[int, ...],
    classes: int,
    error: type[Exception],
) -> None:
    """Shape rules, one case per rule.

    Failure mode: an operand the kernel would index out of its own bounds, or a
    class count that admits a pad column. Both are silent on the device.
    """
    with pytest.raises(error):
        xent_shape(torch.zeros(shape), torch.zeros(labels, dtype=torch.int64), classes)


def test_rejects_a_float_label() -> None:
    """A label that is not an integer.

    Failure mode: a float label tensor, which the kernel would read at the wrong
    element width and index with whatever that produced.
    """
    with pytest.raises(TypeError):
        xent_shape(torch.zeros(4, 8), torch.zeros(4), 8)
