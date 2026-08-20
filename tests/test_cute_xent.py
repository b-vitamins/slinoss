"""Fused cross entropy: the CuTe kernels against the float64 reference.

The authority is :func:`slinoss.ops.xent.xent_ref` in float64 over the same values
the kernel loads, and the gradient authority is float64 autograd through it. The cast
these kernels replace is measured against the same oracle, so the swap is stated as a
direction rather than assumed.

Operands are built once in float32 and cast down, never built twice at two dtypes:
the generator consumes a different number of raw words per element at each width, so
the same seed at two dtypes is two different problems. Every cast to float64 on the
way back up is exact, so the kernel and the oracle see identical values.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss.ops.xent import cross_entropy, xent_ref
from slinoss.ops.xent.cute import xent_backward, xent_forward
from tests.conftest import assert_max_rel

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

DTYPES = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float16, id="f16"),
    pytest.param(torch.float32, id="f32"),
]

# (rows, classes, width). One block of 256 threads owns one row and walks the class
# axis at the block stride, so the cases are the ways that walk ends: fewer classes
# than threads, so most threads reach no column at all; exactly the block width;
# a stride that leaves a ragged last round, under a width the class count does not
# reach; and a vocabulary of hundreds of rounds. One row and many rows appear
# across the four.
SHAPES = [
    pytest.param(1, 7, 7, id="one-row-idle-threads"),
    pytest.param(3, 256, 256, id="exact-block-width"),
    pytest.param(5, 1000, 1024, id="ragged-walk-padded-width"),
    pytest.param(64, 50257, 50257, id="vocabulary"),
]

LOSS_REL = 1e-5
"""Bound on the loss and the normalizer.

The kernel accumulates the partition function in float32 whatever the operand width,
so the error against float64 over the same values is the accumulation's alone: a
serial run per thread, then a butterfly, then eight words.
"""

GRAD_REL = 1e-2
"""Bound on the logit gradient.

The gradient is stored at the operand width, and bfloat16 carries eight mantissa
bits, so the store is the dominant error at the low precisions and the bound is the
store's.
"""


def operands(
    rows: int, classes: int, width: int, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """A logits tensor and labels that index its class band alone.

    Args:
        rows: Rows.
        classes: Classes the labels index.
        width: Operand width, at least ``classes``.
        dtype: Operand dtype.

    Returns:
        ``(logits, labels)`` on the current CUDA device, the labels int64.
    """
    torch.manual_seed(0)
    logits = torch.randn(rows, width, device="cuda", dtype=torch.float32) * 4.0
    labels = torch.randint(0, classes, (rows,), device="cuda")
    return logits.to(dtype), labels


@pytest.mark.parametrize(("rows", "classes", "width"), SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_float64(
    rows: int, classes: int, width: int, dtype: torch.dtype
) -> None:
    """The loss and the saved normalizer against the float64 oracle.

    Failure mode: the online rescale. A thread that reaches no column holds a
    sentinel peak, a thread whose round is ragged holds fewer terms than its
    neighbours, and the block combine scales every partial sum to one maximum; an
    error in any of the three is a wrong partition function, which the loss carries
    logarithmically and would hide at a loose bound.
    """
    logits, labels = operands(rows, classes, width, dtype)
    got = xent_forward(logits, labels, classes=classes)
    want = xent_ref(logits.double(), labels, classes=classes)
    assert_max_rel(got.loss, want.loss, LOSS_REL, "xent/loss")
    assert_max_rel(got.lse, want.lse, LOSS_REL, "xent/lse")
    assert got.loss.dtype is torch.float32
    assert got.loss.shape == ()


@pytest.mark.parametrize(("rows", "classes", "width"), SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_matches_float64_autograd(
    rows: int, classes: int, width: int, dtype: torch.dtype
) -> None:
    """The logit gradient against float64 autograd through the reference.

    Failure mode: the one-hot column, the mean scale, or a pad column left
    uninitialized. The destination is ``torch.empty`` of the operand's full width,
    so a pad column the kernel does not write holds whatever the allocator last put
    there and the test would fail nondeterministically rather than not at all.

    The cotangent is not one, so the scale is checked rather than cancelled.
    """
    logits, labels = operands(rows, classes, width, dtype)
    cotangent = torch.tensor(0.25, dtype=torch.float32, device="cuda")

    leaf = logits.double().requires_grad_(True)
    xent_ref(leaf, labels, classes=classes).loss.backward(cotangent.double())
    assert leaf.grad is not None

    state = xent_forward(logits, labels, classes=classes)
    got = xent_backward(cotangent, logits, labels, state.lse, classes=classes)
    assert_max_rel(got.dlogits, leaf.grad, GRAD_REL, f"xent/dlogits-{dtype}")
    assert got.dlogits.dtype is logits.dtype
    assert bool((got.dlogits[:, classes:] == 0.0).all())


def test_pad_columns_enter_neither_direction() -> None:
    """A padded width whose pad columns dominate the class band.

    Failure mode: a partition function over ``logits.shape[-1]``. An aligned head
    pads its output past the vocabulary and the pad columns hold whatever the GEMM
    left there, so the only safe statement is that the loss does not move when they
    change and their gradient stays zero. The values here are far enough above the
    class logits that including one would swamp the loss rather than perturb it.
    """
    classes, width = 1000, 1024
    logits, labels = operands(8, classes, width, torch.bfloat16)
    plain = xent_forward(logits, labels, classes=classes)

    padded = logits.clone()
    padded[:, classes:] = 30.0
    other = xent_forward(padded, labels, classes=classes)
    assert bool(torch.equal(plain.loss, other.loss))
    assert bool(torch.equal(plain.lse, other.lse))

    cotangent = torch.ones((), dtype=torch.float32, device="cuda")
    grads = xent_backward(cotangent, padded, labels, other.lse, classes=classes)
    assert bool((grads.dlogits[:, classes:] == 0.0).all())


def test_matches_the_cast_expression_it_replaces() -> None:
    """The kernel against ``cross_entropy`` over a float32 copy of the logits.

    Failure mode: a swap that moves the loss. The expression this replaces widens the
    logits with a cast kernel and reduces at float32; the kernel reads them at their
    own width and accumulates at float32, which is the same arithmetic in a different
    order. Both are measured against the float64 oracle so the direction is stated
    rather than assumed.
    """
    rows, classes = 2048, 50257
    logits, labels = operands(rows, classes, classes, torch.bfloat16)
    oracle = xent_ref(logits.double(), labels, classes=classes).loss

    fused = xent_forward(logits, labels, classes=classes).loss
    cast = torch.nn.functional.cross_entropy(logits.float(), labels)
    assert_max_rel(fused, oracle, LOSS_REL, "xent/fused-against-f64")
    assert_max_rel(cast, oracle, LOSS_REL, "xent/cast-against-f64")
    assert_max_rel(fused, cast, LOSS_REL, "xent/fused-against-cast")


def test_autograd_matches_torch_end_to_end() -> None:
    """The public callable through autograd, against torch's own expression.

    Failure mode: the boundary. The forward saves the normalizer and the class count
    and returns one of two outputs, so a wrong save, a wrong grad position, or a loss
    that carries no graph all show up here and nowhere in the two parity tests.

    An int32 label tensor runs the same call, which is what pins the second label
    dtype: the kernel widens the index, and an element width read wrongly would index
    a different row.
    """
    rows, classes, width = 512, 1000, 1024
    logits, labels = operands(rows, classes, width, torch.bfloat16)

    leaf = logits.clone().requires_grad_(True)
    loss = cross_entropy(leaf, labels, classes=classes)
    loss.backward()
    assert leaf.grad is not None

    want = logits.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(want[:, :classes].float(), labels).backward()
    assert want.grad is not None

    band = logits[:, :classes].float()
    assert_max_rel(
        loss,
        torch.nn.functional.cross_entropy(band, labels),
        LOSS_REL,
        "xent/autograd-loss",
    )
    assert_max_rel(leaf.grad, want.grad, GRAD_REL, "xent/autograd-dlogits")

    as_int32 = cross_entropy(logits, labels.int(), classes=classes)
    assert bool(torch.equal(as_int32, cross_entropy(logits, labels, classes=classes)))


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        pytest.param(lambda t, i: (t[:, :8], i), ValueError, id="non-contiguous"),
        pytest.param(lambda t, i: (t, i.float()), TypeError, id="float-labels"),
        pytest.param(lambda t, i: (t.double(), i), TypeError, id="float64-operand"),
        pytest.param(lambda t, i: (t.cpu(), i), ValueError, id="host-operand"),
    ],
)
def test_rejects_an_operand_with_no_kernel_path(
    mutate: object, error: type[Exception]
) -> None:
    """Host guards, one case per rule.

    Failure mode: an operand the kernel would read at the wrong element width, the
    wrong stride, or off the device. All three are silent on the device, and the
    first two produce a plausible number.
    """
    logits, labels = operands(4, 8, 16, torch.bfloat16)
    bad_logits, bad_labels = mutate(logits, labels)  # type: ignore[operator]
    with pytest.raises(error):
        xent_forward(bad_logits, bad_labels, classes=8)
