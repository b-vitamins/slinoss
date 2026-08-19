"""Block dispatch: backend resolution and the autograd wiring.

What is tested here is what the dispatch layer adds. Which backends each of the
three families registers, what ``resolve`` returns for it, that the public
callable neither perturbs the forward nor misnames the fields it rebuilds, and
that autograd reaches the registry backward. Kernel parity is
tests/test_cute_block.py and the maps themselves are tests/test_block_ref.py;
neither is repeated.

Ground truth is ``gradcheck`` in float64, which resolves to the reference by
dtype because no kernel has a float64 instantiation. That is the intended
resolution rather than a hole: it is what makes the float64 oracle reachable
through the public path, and the forward-to-backward connection at kernel widths
is what the kernel file already covers.

Shape is not swept. It is not a resolution axis and no part of this layer reads
it, so one shape per family runs here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import pytest
import torch
from torch import Tensor
from torch.autograd import gradcheck

from slinoss._precision import KERNEL_DTYPES, SUPPORTED_DTYPES
from slinoss._registry import Backend
from slinoss.ops.block import (
    NormResidual,
    NormResidualGrads,
    rmsnorm,
    rmsnorm_get,
    rmsnorm_names,
    rmsnorm_ref,
    rmsnorm_residual,
    rmsnorm_residual_get,
    rmsnorm_residual_names,
    rmsnorm_residual_ref,
    rmsnorm_residual_resolve,
    rmsnorm_resolve,
    swiglu,
    swiglu_get,
    swiglu_names,
    swiglu_ref,
    swiglu_resolve,
)
from slinoss.ops.block import interface as block_interface

EPS = 1e-5

SHAPE = (3, 8)


class Family(NamedTuple):
    """One family's lookup surface, so a rule can be asserted once per family.

    Attributes:
        names: The family's registered names.
        get: Lookup by name.
        resolve: Selection by name, device type, and dtype.
    """

    names: Callable[[], tuple[str, ...]]
    get: Callable[[str], Backend[Any, Any]]
    resolve: Callable[[str | None, str, torch.dtype], Backend[Any, Any]]


FAMILIES = [
    pytest.param(Family(rmsnorm_names, rmsnorm_get, rmsnorm_resolve), id="rmsnorm"),
    pytest.param(
        Family(rmsnorm_residual_names, rmsnorm_residual_get, rmsnorm_residual_resolve),
        id="rmsnorm_residual",
    ),
    pytest.param(Family(swiglu_names, swiglu_get, swiglu_resolve), id="swiglu"),
]

FAMILY_NAMES = ["rmsnorm", "rmsnorm_residual", "swiglu"]

CUTE_REGISTERED = "cute" in rmsnorm_names()

needs_cute = pytest.mark.skipif(
    not CUTE_REGISTERED, reason="no CuTe backend registered on this host"
)


def _rnd(
    shape: tuple[int, ...], *, dtype: torch.dtype = torch.float64, seed: int = 0
) -> Tensor:
    """One generator call, one dtype. Never the same seed at two widths."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


def _call(
    family: str,
    x: Tensor,
    weight: Tensor,
    *,
    backend: str | None,
) -> object:
    """Invoke one public callable with operands that pass every operand check.

    Args:
        family: ``"rmsnorm"``, ``"rmsnorm_residual"``, or ``"swiglu"``.
        x: First operand, and the residual and the up operand as well.
        weight: Norm scale. Unused by the activation.
        backend: Backend name, or None to autoselect.

    Returns:
        Whatever the callable returns.
    """
    if family == "rmsnorm":
        return rmsnorm(x, weight, eps=EPS, backend=backend)
    if family == "rmsnorm_residual":
        return rmsnorm_residual(x, x, weight, eps=EPS, backend=backend)
    return swiglu(x, x, backend=backend)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


# The resolution rule itself is tested in tests/test_registry.py against a
# registry that test owns. What is the block's own is which backends each of its
# three registries holds and what that makes resolve return.
@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_registers_the_reference(family: Family) -> None:
    """A family whose registry is empty resolves to nothing at all.

    Three registries in one module is three chances to bind a lookup to the wrong
    one, and the symptom is a family that cannot run on any device.
    """
    assert "reference" in family.names()
    backend = family.get("reference")
    assert backend.priority == 0
    assert backend.device_types == ("cpu", "cuda")
    # The reference is the float64 oracle, so it declares the operator's whole
    # supported set rather than the kernel set.
    assert backend.dtypes == SUPPORTED_DTYPES


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("family", FAMILIES)
def test_float64_autoselects_the_reference(family: Family, device_type: str) -> None:
    """float64 has no kernel path, so a kernel backend must not be selected for it.

    Resolution is a lookup over declared device types and dtypes and touches no
    device, so both device types are checked on any host. A kernel backend that
    declared float64 would take the oracle path and gradcheck would run against a
    kernel that cannot compute it.
    """
    assert family.resolve(None, device_type, torch.float64).name == "reference"


@needs_cute
@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("family", FAMILIES)
def test_the_kernel_backend_declares_only_what_it_can_run(family: Family) -> None:
    """A CuTe backend registered over cpu or over float64 would be selected for a
    call it cannot take, and the failure would surface from inside the launch
    instead of from resolution."""
    backend = family.get("cute")
    assert backend.priority == 10
    assert backend.device_types == ("cuda",)
    assert backend.dtypes == KERNEL_DTYPES


@needs_cute
@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_a_named_kernel_backend_refuses_a_cpu_operand(family: str) -> None:
    """Naming a cuda-only backend for a cpu operand is refused before the launch."""
    x = _rnd(SHAPE, dtype=torch.float32)
    weight = _rnd((SHAPE[-1],), dtype=torch.float32, seed=1)
    with pytest.raises(
        ValueError, match=r"backend 'cute' supports \('cuda',\), not 'cpu'"
    ):
        _call(family, x, weight, backend="cute")


@needs_cute
@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_a_named_kernel_backend_refuses_float64(family: str) -> None:
    """The dtype rule is reported too, not only the device rule.

    The operand is on cuda so that resolution reaches the dtype check: a named
    backend reports the device first, so a cpu float64 operand would be refused
    under the device rule and this branch would never run.
    """
    x = torch.zeros(SHAPE, dtype=torch.float64, device="cuda")
    weight = torch.ones(SHAPE[-1], dtype=torch.float64, device="cuda")
    with pytest.raises(
        ValueError, match=r"backend 'cute' supports .*not torch\.float64"
    ):
        _call(family, x, weight, backend="cute")


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_an_unknown_backend_name_is_refused(family: str) -> None:
    """The public callable resolves before it applies, so a typo raises from the
    call and not from inside the graph."""
    x = _rnd(SHAPE)
    weight = _rnd((SHAPE[-1],), seed=1)
    with pytest.raises(ValueError, match=r"unknown backend 'nope'"):
        _call(family, x, weight, backend="nope")


# ---------------------------------------------------------------------------
# The Function does not perturb the forward
# ---------------------------------------------------------------------------


def test_rmsnorm_forward_is_bitwise_the_reference() -> None:
    """The Function wraps the forward; it does not recompute it."""
    x = _rnd(SHAPE)
    weight = _rnd((SHAPE[-1],), seed=1)
    assert torch.equal(rmsnorm(x, weight, eps=EPS), rmsnorm_ref(x, weight, eps=EPS))


def test_rmsnorm_residual_forward_is_bitwise_the_reference() -> None:
    """Both fields, in the right order.

    The Function returns a positional pair and the public callable renames it, so
    a swap here would hand a stack the normed output as its residual stream and
    the shapes and dtypes would not catch it.
    """
    x = _rnd(SHAPE)
    residual = _rnd(SHAPE, seed=1)
    weight = _rnd((SHAPE[-1],), seed=2)
    got = rmsnorm_residual(x, residual, weight, eps=EPS)
    want = rmsnorm_residual_ref(x, residual, weight, eps=EPS)
    assert isinstance(got, NormResidual)
    assert torch.equal(got.normed, want.normed)
    assert torch.equal(got.residual, want.residual)


def test_swiglu_forward_is_bitwise_the_reference() -> None:
    """The activation takes no epsilon and no weight, so its Function carries one
    fewer non-tensor argument than the norms; a misplaced one would land on the
    backend name."""
    gate = _rnd(SHAPE)
    up = _rnd(SHAPE, seed=1)
    assert torch.equal(swiglu(gate, up), swiglu_ref(gate, up))


# ---------------------------------------------------------------------------
# Autograd through the public callable
# ---------------------------------------------------------------------------


def test_gradcheck_rmsnorm() -> None:
    """float64 through the public callable, against a finite difference of the
    forward it just ran. Without this the backward could return the gradients in
    the wrong slots and every forward test would still pass."""
    x = _rnd(SHAPE).requires_grad_(True)
    weight = _rnd((SHAPE[-1],), seed=1).requires_grad_(True)
    assert gradcheck(lambda a, w: rmsnorm(a, w, eps=EPS), (x, weight))


def test_gradcheck_rmsnorm_residual() -> None:
    """Both outputs and all three inputs, with the stream present.

    gradcheck differentiates one output at a time and also checks the undefined
    cotangent, so the absent-cotangent branches run here as well; the tests below
    pin down which operand each branch sees.
    """
    x = _rnd(SHAPE).requires_grad_(True)
    residual = _rnd(SHAPE, seed=1).requires_grad_(True)
    weight = _rnd((SHAPE[-1],), seed=2).requires_grad_(True)

    def call(a: Tensor, r: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        out = rmsnorm_residual(a, r, w, eps=EPS)
        return out.normed, out.residual

    assert gradcheck(call, (x, residual, weight))


def test_gradcheck_rmsnorm_residual_without_a_stream() -> None:
    """The first block of a stack. The residual slot takes no gradient at all, so
    the backward returns one fewer than in the case above and a fixed-width
    gradient tuple would misalign."""
    x = _rnd(SHAPE).requires_grad_(True)
    weight = _rnd((SHAPE[-1],), seed=1).requires_grad_(True)

    def call(a: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        out = rmsnorm_residual(a, None, w, eps=EPS)
        return out.normed, out.residual

    assert gradcheck(call, (x, weight))


def test_gradcheck_swiglu() -> None:
    """Both operand gradients through the public callable."""
    gate = _rnd(SHAPE).requires_grad_(True)
    up = _rnd(SHAPE, seed=1).requires_grad_(True)
    assert gradcheck(swiglu, (gate, up))


# ---------------------------------------------------------------------------
# What the backward sees when an operand or a cotangent is absent
# ---------------------------------------------------------------------------


class Seen(NamedTuple):
    """Which of the fused backward's optional arguments arrived.

    Attributes:
        dnormed: The cotangent of ``normed`` was supplied.
        dresidual: The cotangent of the wide residual was supplied.
        residual: The forward's incoming stream was supplied.
    """

    dnormed: bool
    dresidual: bool
    residual: bool


def _observe(
    monkeypatch: pytest.MonkeyPatch, *, stream: bool, consume: str
) -> tuple[Seen, ...]:
    """Run one fused forward and backward and report what the backend was handed.

    The reference backward is wrapped rather than replaced, so the gradients the
    call produces are still the reference's and only the arguments are observed.

    Args:
        monkeypatch: Undoes the wrap at the end of the test.
        stream: Pass an incoming residual to the forward.
        consume: ``"normed"`` or ``"residual"``, the one output differentiated.

    Returns:
        One entry per backward invocation.
    """
    seen: list[Seen] = []
    reference = rmsnorm_residual_get("reference")

    def spy(
        dnormed: Tensor | None,
        dresidual: Tensor | None,
        x: Tensor,
        residual: Tensor | None,
        weight: Tensor,
        /,
        *,
        eps: float,
    ) -> NormResidualGrads:
        seen.append(
            Seen(
                dnormed=dnormed is not None,
                dresidual=dresidual is not None,
                residual=residual is not None,
            )
        )
        return reference.backward(dnormed, dresidual, x, residual, weight, eps=eps)

    monkeypatch.setattr(
        block_interface,
        "rmsnorm_residual_get",
        lambda _name: reference._replace(backward=spy),
    )

    x = _rnd(SHAPE).requires_grad_(True)
    weight = _rnd((SHAPE[-1],), seed=1).requires_grad_(True)
    residual = _rnd(SHAPE, seed=2).requires_grad_(True) if stream else None
    out = rmsnorm_residual(x, residual, weight, eps=EPS)
    (out.normed if consume == "normed" else out.residual).square().sum().backward()
    return tuple(seen)


@pytest.mark.parametrize(
    ("stream", "consume", "want"),
    [
        pytest.param(
            True,
            "normed",
            Seen(dnormed=True, dresidual=False, residual=True),
            id="normed-only",
        ),
        pytest.param(
            True,
            "residual",
            Seen(dnormed=False, dresidual=True, residual=True),
            id="residual-only",
        ),
        pytest.param(
            False,
            "normed",
            Seen(dnormed=True, dresidual=False, residual=False),
            id="no-stream",
        ),
    ],
)
def test_the_backward_receives_an_absent_operand_as_none(
    monkeypatch: pytest.MonkeyPatch, stream: bool, consume: str, want: Seen
) -> None:
    """An absent cotangent and an absent stream arrive as None, not as zeros.

    Without ``ctx.set_materialize_grads(False)`` torch allocates and fills a
    zero tensor for the output nobody consumed, and the backend then contracts
    over it: a full-size allocation and a full-size pass per absent cotangent, on
    the training path, producing a gradient the caller cannot distinguish from a
    real one that vanished. The backends already take None in both cotangent
    slots and in the stream slot, so nothing is materialized here either.
    """
    assert _observe(monkeypatch, stream=stream, consume=consume) == (want,)


def test_a_residual_only_cotangent_leaves_the_weight_without_a_gradient() -> None:
    """The weight does not reach the residual output, so its gradient is absent.

    This is the observable half of the test above: a materialized ``dnormed``
    would put a zero tensor on ``weight.grad`` where nothing was differentiated
    with respect to it, and an optimizer cannot tell that apart from a gradient
    that happened to vanish.
    """
    x = _rnd(SHAPE).requires_grad_(True)
    residual = _rnd(SHAPE, seed=1).requires_grad_(True)
    weight = _rnd((SHAPE[-1],), seed=2).requires_grad_(True)

    out = rmsnorm_residual(x, residual, weight, eps=EPS)
    out.residual.square().sum().backward()

    assert weight.grad is None
    # The residual output is the plain sum, so its pullback is the identity on
    # both summands and the comparison needs no tolerance.
    assert x.grad is not None and torch.equal(x.grad, 2.0 * (x + residual))
    assert residual.grad is not None and torch.equal(residual.grad, x.grad)


# ---------------------------------------------------------------------------
# Operand rejection through the Function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(
            lambda: rmsnorm(_rnd(SHAPE), _rnd((7,), seed=1), eps=EPS),
            r"weight must be \(8,\)",
            id="rmsnorm-weight",
        ),
        pytest.param(
            lambda: rmsnorm_residual(
                _rnd(SHAPE), _rnd((2, 8), seed=1), _rnd((8,), seed=2), eps=EPS
            ),
            r"residual must be \(3, 8\)",
            id="rmsnorm_residual-residual",
        ),
        pytest.param(
            lambda: swiglu(_rnd(SHAPE), _rnd((3, 7), seed=1)),
            r"up must be \(3, 8\)",
            id="swiglu-up",
        ),
    ],
)
def test_a_rejected_operand_raises_through_the_function(
    call: Callable[[], object], match: str
) -> None:
    """The backend's operand check is the one check, and it reaches the caller.

    A Function that swallowed it, or a public callable that validated on its own,
    would either hide the message or state the rule a second time.
    """
    with pytest.raises(ValueError, match=match):
        call()
