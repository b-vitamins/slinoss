"""One operator, allocated at one shape, with the runner over it.

Every driver needs the same three things from an operator: the shape record the
report names, the inputs, and a region-labelled callable. The seven families in
:mod:`slinoss.perf.workload` do not share a signature, so the name is resolved to
one of them here. A driver reaches every operator through one call, and a family
lands in one place rather than in every driver.

The inputs are allocated once per arm. Two sets differ in address and in cache
residency, and a paired comparison would attribute that difference to the backend,
so both arms of a comparison run the runner this returns over one set.

Six of the seven arms carry a forward and a step. ``decode`` carries a forward only,
and refuses ``grads`` rather than returning a forward under the step's name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import torch
from torch import Tensor

from slinoss.perf.units import Count
from slinoss.perf.workload import (
    BLOCK,
    CONV,
    DECODE,
    MIXER,
    OPS,
    SCANPREP,
    SO3SSD,
    XENT,
    block_forward_only,
    block_shape_by_name,
    block_step,
    conv_forward_only,
    conv_shape_by_name,
    conv_step,
    decode_forward_only,
    decode_shape_by_name,
    forward_only,
    make_block_inputs,
    make_conv_inputs,
    make_decode_inputs,
    make_inputs,
    make_mixer_inputs,
    make_prep_inputs,
    make_xent_inputs,
    mixer_forward_only,
    mixer_shape_by_name,
    mixer_step,
    prep_forward_only,
    prep_shape_by_name,
    prep_step,
    shape_by_name,
    step,
    xent_forward_only,
    xent_shape_by_name,
    xent_step,
)

__all__ = ["BenchShape", "OpArm", "op_arm"]


class BenchShape(Protocol):
    """What a driver reads off any operator's shape record."""

    @property
    def name(self) -> str:
        """Shape name, as it appears on the command line and in the report."""
        ...

    @property
    def token_count(self) -> Count:
        """Tokens one call processes, the throughput denominator."""
        ...

    def describe(self) -> str:
        """One line naming every extent."""
        ...


class OpArm(NamedTuple):
    """One operator at one shape, allocated, with grads on or off.

    Attributes:
        shape: The shape record, for the token count and the report's notes.
        prefix: Region label prefix the family uses by default. Not the operator
            name: the scan labels its regions ``op.*``.
        differentiable: The inputs a backward differentiates with respect to,
            for the saved-tensor probe.
        run: Runner factory, by backend name and region prefix. Two arms measured
            in one loop pass two prefixes over the one allocated input set.
    """

    shape: BenchShape
    prefix: str
    differentiable: tuple[Tensor, ...]
    run: Callable[[str | None, str], Callable[[], None]]


def op_arm(
    op: str,
    shape_name: str,
    device: torch.device,
    *,
    dtype: torch.dtype,
    grads: bool,
    d_head: int | None = None,
) -> OpArm:
    """Allocate one operator at one shape and return the arm over it.

    Args:
        op: Operator name, one of :data:`slinoss.perf.workload.OPS`.
        shape_name: Shape name, resolved against the family's own table.
        device: Where to allocate.
        dtype: Input dtype.
        grads: Whether the differentiable inputs carry gradients. The runner
            measures a forward and a backward when they do.
        d_head: Rows per head for the conv output layout, or None for token-major.
            Every other operator ignores it.

    Returns:
        The arm.

    Raises:
        ValueError: If ``op`` names no family. Reachable only from a caller that is
            not argparse, which is where the choices list already refuses one.
    """
    if op == SO3SSD:
        scan_shape = shape_by_name(shape_name)
        scan = make_inputs(scan_shape, device, dtype=dtype, requires_grad=grads)

        def scan_run(backend: str | None, prefix: str) -> Callable[[], None]:
            if grads:
                return step(scan, scan_shape.chunk, backend=backend, prefix=prefix)
            return forward_only(scan, scan_shape.chunk, backend=backend, prefix=prefix)

        return OpArm(scan_shape, "op", scan.differentiable, scan_run)
    if op == CONV:
        conv_shape = conv_shape_by_name(shape_name)
        conv = make_conv_inputs(
            conv_shape, device, dtype=dtype, requires_grad=grads, d_head=d_head
        )

        def conv_run(backend: str | None, prefix: str) -> Callable[[], None]:
            if grads:
                return conv_step(conv, backend=backend, prefix=prefix)
            return conv_forward_only(conv, backend=backend, prefix=prefix)

        return OpArm(conv_shape, "conv", conv.differentiable, conv_run)
    if op == SCANPREP:
        prep_shape = prep_shape_by_name(shape_name)
        prep = make_prep_inputs(prep_shape, device, dtype=dtype, requires_grad=grads)

        def prep_run(backend: str | None, prefix: str) -> Callable[[], None]:
            if grads:
                return prep_step(prep, prep_shape, backend=backend, prefix=prefix)
            return prep_forward_only(prep, prep_shape, backend=backend, prefix=prefix)

        return OpArm(prep_shape, "prep", prep.differentiable, prep_run)
    if op == BLOCK:
        block_shape = block_shape_by_name(shape_name)
        block = make_block_inputs(block_shape, device, dtype=dtype, requires_grad=grads)

        def block_run(backend: str | None, prefix: str) -> Callable[[], None]:
            if grads:
                return block_step(block, block_shape, backend=backend, prefix=prefix)
            return block_forward_only(
                block, block_shape, backend=backend, prefix=prefix
            )

        return OpArm(block_shape, "block", block.differentiable, block_run)
    if op == MIXER:
        mixer_shape = mixer_shape_by_name(shape_name)
        mixer = make_mixer_inputs(mixer_shape, device, dtype=dtype, requires_grad=grads)

        def mixer_run(backend: str | None, prefix: str) -> Callable[[], None]:
            if grads:
                return mixer_step(mixer, mixer_shape, backend=backend, prefix=prefix)
            return mixer_forward_only(
                mixer, mixer_shape, backend=backend, prefix=prefix
            )

        return OpArm(mixer_shape, "mixer", mixer.differentiable, mixer_run)
    if op == XENT:
        xent_shape = xent_shape_by_name(shape_name)
        xent = make_xent_inputs(xent_shape, device, dtype=dtype, requires_grad=grads)

        def xent_run(backend: str | None, prefix: str) -> Callable[[], None]:
            if grads:
                return xent_step(xent, xent_shape, backend=backend, prefix=prefix)
            return xent_forward_only(xent, xent_shape, backend=backend, prefix=prefix)

        return OpArm(xent_shape, "xent", xent.differentiable, xent_run)
    if op == DECODE:
        # Refused before the shape is resolved and before anything is allocated, so
        # the refusal costs nothing and reaches a caller with no device.
        if grads:
            raise ValueError(
                "decode has no step arm: SLinOSSMixer.step is a no_grad node, so "
                "there is no backward at T=1 to measure; use --mode forward"
            )
        decode_shape = decode_shape_by_name(shape_name)
        decode = make_decode_inputs(decode_shape, device, dtype=dtype)

        def decode_run(backend: str | None, prefix: str) -> Callable[[], None]:
            return decode_forward_only(decode, backend=backend, prefix=prefix)

        return OpArm(decode_shape, "decode", decode.differentiable, decode_run)
    raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
