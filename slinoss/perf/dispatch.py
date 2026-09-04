"""What each operator's registry resolves to, asked before a profiler runs.

An operator whose kernel backend did not register resolves to its reference and runs.
It produces the right numbers, at torch's speed, in torch's kernels, and every rule
the audit applies passes because no declared kernel was profiled. The measured case
was a conv audit on a tree where the compiled extension had not been built:
:mod:`slinoss.ops.conv.backends` registers the native backend only under
``_C.is_available()``, so the guard held and the audit judged nothing.

The signal generalises because the fallback does. Every operator selects through
:class:`slinoss._registry.Registry`, the reference is the priority-0 entry named
``reference``, and every kernel backend registers behind a guard: the extension for
the conv, a CuTe DSL import and a visible CUDA device for the other five. Asking each
registry what it resolves for the profiled device and dtype therefore names the
fallback before any kernel runs, whatever made it happen.

Asked before the profilers rather than after: nine NCU passes over a reference path
cost the same as nine over a kernel path and answer nothing.

This module is deliberately absent from ``slinoss.perf``'s exports. It imports every
operator's backend module, which depends on the timing primitives, so exporting it
from the package initializer would build a cycle. The verdict types it returns live
in :mod:`slinoss.perf.coverage`, which imports nothing from ``slinoss.ops``, so a
report can carry a dispatch verdict without that cycle.
"""

from __future__ import annotations

from typing import Final, NamedTuple, Protocol

import torch

from slinoss.ops.block.backends import (
    rmsnorm_residual_resolve,
    rmsnorm_resolve,
    swiglu_resolve,
)
from slinoss.ops.conv.backends import resolve as conv_resolve
from slinoss.ops.decode.backends import resolve as decode_resolve
from slinoss.ops.mixer.backends import resolve as mixer_resolve
from slinoss.ops.scanprep.backends import resolve as prep_resolve
from slinoss.ops.so3ssd.backends import resolve as scan_resolve
from slinoss.ops.xent.backends import resolve as xent_resolve
from slinoss.perf.coverage import DispatchVerdict, RegistryChoice
from slinoss.perf.units import Count
from slinoss.perf.workload import BLOCK, CONV, DECODE, MIXER, SCANPREP, SO3SSD, XENT

__all__ = [
    "OP_REGISTRIES",
    "REFERENCE",
    "Chosen",
    "OpRegistry",
    "Resolver",
    "dispatch_verdict",
]

REFERENCE: Final = "reference"
"""The name every operator's reference backend registers under.

One string, checked against what a registry returned rather than against a priority:
priority is comparative and a tree with no kernel backend has one entry, so the
maximum priority is the reference's own."""


class Chosen(Protocol):
    """What this module reads off a resolved backend."""

    @property
    def name(self) -> str:
        """Registry key of the selected implementation."""
        ...


class Resolver(Protocol):
    """One registry's ``resolve``, narrowed to what this module calls it with.

    A protocol rather than a ``Callable`` alias: the six registries return six
    parameterizations of :class:`slinoss._registry.Backend`, whose type parameters are
    invariant, so one alias cannot hold them all. Only the selected name is read.
    """

    def __call__(
        self, name: str | None, device_type: str, dtype: torch.dtype, /
    ) -> Chosen: ...


class OpRegistry(NamedTuple):
    """One registry an operator selects through.

    Attributes:
        label: Registry name, as :class:`slinoss._registry.Registry` was constructed
            with. Three of them for the block, whose three families have three.
        resolve: The registry's ``resolve``, by explicit name, device type and dtype.
    """

    label: str
    resolve: Resolver


OP_REGISTRIES: Final[dict[str, tuple[OpRegistry, ...]]] = {
    SO3SSD: (OpRegistry("so3ssd", scan_resolve),),
    CONV: (OpRegistry("conv", conv_resolve),),
    SCANPREP: (OpRegistry("scanprep", prep_resolve),),
    BLOCK: (
        OpRegistry("rmsnorm", rmsnorm_resolve),
        OpRegistry("rmsnorm_residual", rmsnorm_residual_resolve),
        OpRegistry("swiglu", swiglu_resolve),
    ),
    MIXER: (OpRegistry("mixer_tail", mixer_resolve),),
    XENT: (OpRegistry("cross_entropy", xent_resolve),),
    DECODE: (
        OpRegistry("conv", conv_resolve),
        OpRegistry("scanprep", prep_resolve),
        OpRegistry("decode", decode_resolve),
        OpRegistry("mixer_tail", mixer_resolve),
        OpRegistry("rmsnorm_residual", rmsnorm_residual_resolve),
        OpRegistry("swiglu", swiglu_resolve),
    ),
}
"""Every benchmarked operator, and the registries its arm dispatches through.

The block's arm runs all three of its families, so all three are asked: a tree where
only the swiglu backend failed to register would otherwise read as dispatching to
kernels.

The decode arm is a whole block, so it asks six. Not ``rmsnorm``: every norm on the
step's path is the fused one, including the stack's final one, so asking the plain
registry would fail a verdict on a call the arm never makes. Not ``so3ssd`` either:
:meth:`slinoss.mixer.SLinOSSMixer.step` selects the one-token recurrence through
:mod:`slinoss.ops.decode` at ``T == 1``, which is the extent this arm runs, so
``decode`` is the registry the step's scan stage dispatches through and the chunked
one is never asked. That entry and the ``("decode", "forward")`` entry of
:data:`slinoss.perf.coverage.COVERAGE` move together: a registry listed here that the
step does not select through, or one it does select through and that is not listed,
both read as a dispatch verdict on the wrong program."""


def dispatch_verdict(
    op: str,
    *,
    device_type: str,
    dtype: torch.dtype,
    backend: str | None = None,
) -> DispatchVerdict:
    """Ask every registry one operator uses what it would run.

    Args:
        op: Operator name, one of :data:`slinoss.perf.workload.OPS`.
        device_type: Torch device type the profile runs on.
        dtype: Activation dtype the profile runs at. A kernel backend declares the
            dtypes it has an instantiation for, so a dtype with no fast path resolves
            to the reference and this reports it.
        backend: Explicit backend name, or None for automatic selection. Forwarded
            unchanged, so the verdict describes the call the profile makes.

    Returns:
        The verdict. ``passed`` is False when any registry resolved to the reference.

    Raises:
        KeyError: If ``op`` has no registry list.
        ValueError: From the registry, if a named backend does not support the device
            or the dtype, or if nothing registered supports the pair.
    """
    registries = OP_REGISTRIES.get(op)
    if registries is None:
        raise KeyError(f"no registry list for op {op!r}; have {sorted(OP_REGISTRIES)}")
    choices = tuple(
        RegistryChoice(
            registry=one.label,
            backend=chosen.name,
            is_reference=chosen.name == REFERENCE,
        )
        for one, chosen in (
            (one, one.resolve(backend, device_type, dtype)) for one in registries
        )
    )
    fell_back = tuple(one.registry for one in choices if one.is_reference)
    if fell_back:
        detail = (
            f"{op} resolves to the reference for {list(fell_back)} on "
            f"{device_type} {dtype}; no declared kernel runs, so an audit of this "
            f"process would judge nothing"
        )
    else:
        detail = ", ".join(f"{one.registry}={one.backend}" for one in choices)
    return DispatchVerdict(
        op=op,
        device_type=device_type,
        dtype=str(dtype),
        choices=choices,
        reference_count=Count(len(fell_back)),
        passed=not fell_back,
        detail=detail,
    )
