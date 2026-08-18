"""The benchmarked workload. One definition, shared by every driver.

The bench, the NCU target, and the NSYS target all call the same functions here,
so the benchmarked path is the shipped path: :func:`slinoss.ops.so3ssd.so3ssd`
with no variant reachable from a script and not from the public API.

``trans`` and ``K`` are produced by the real parameter maps and then detached, so
the numerical invariants hold on the benchmarked tensors -- ``ls <= 0`` and
``|w| <= w_max`` -- while the measurement covers the scan alone. Fabricating them
from ``randn`` would put ``ls > 0`` into a decay prefix and measure a kernel that
cannot run in training.

The backward is measured with :func:`torch.autograd.grad` rather than
``loss.backward()``: no reduction is added to the graph, and no gradient is
accumulated into a ``.grad`` buffer, so an ``aten::fill_`` cannot contaminate the
backward bucket.

This module is deliberately absent from ``slinoss.perf``'s exports. It depends on
:mod:`slinoss.ops.so3ssd`, which depends on the timing primitives, so importing
it from the package initializer would build a cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

import torch
from torch import Tensor

from slinoss.ops.scanprep.reference import scanprep_ref
from slinoss.ops.so3ssd import so3ssd
from slinoss.perf.timing import region
from slinoss.perf.units import Count

__all__ = [
    "SHAPES",
    "W_MAX",
    "OpInputs",
    "OpShape",
    "forward_only",
    "make_inputs",
    "shape_by_name",
    "step",
]

W_MAX: Final = 3.0
"""Rotation-vector bound used by every benchmark. Below pi, as I2 requires."""


@dataclass(frozen=True)
class OpShape:
    """One benchmarked problem size.

    Attributes:
        name: Shape name, used on the command line and in the report.
        bsz: Batch, ``B``.
        heads: Heads, ``H``.
        seq: Sequence length, ``T``. Not required to be a multiple of ``chunk``.
        rows: Rows per head, ``P``. Multiple of 8.
        lanes: Independent 3-vectors, ``N``. Multiple of 16.
        chunk: Chunk length ``L``.
    """

    name: str
    bsz: int
    heads: int
    seq: int
    rows: int
    lanes: int
    chunk: int

    @property
    def d_state(self) -> int:
        """Trailing state width, ``3N``."""
        return 3 * self.lanes

    @property
    def token_count(self) -> Count:
        """Tokens per call, ``B*T``."""
        return Count(self.bsz * self.seq)

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.name}: B={self.bsz} H={self.heads} T={self.seq} "
            f"P={self.rows} N={self.lanes} 3N={self.d_state} L={self.chunk}"
        )


SHAPES: Final[tuple[OpShape, ...]] = (
    OpShape("tiny", bsz=1, heads=1, seq=256, rows=8, lanes=16, chunk=64),
    OpShape("standard", bsz=4, heads=12, seq=2048, rows=48, lanes=16, chunk=64),
    OpShape("wide", bsz=4, heads=12, seq=2048, rows=64, lanes=32, chunk=64),
    OpShape("long", bsz=2, heads=12, seq=8192, rows=48, lanes=16, chunk=128),
    OpShape("ragged", bsz=4, heads=12, seq=2000, rows=48, lanes=16, chunk=64),
)
"""The standard sizes. Every optimization is measured at all of them, before and
after, with the same commands. ``ragged`` has a sequence length that is not a
multiple of the chunk, so a tail-handling regression shows up in the bench and
not only in the tests."""


def shape_by_name(name: str) -> OpShape:
    """Look up a standard shape.

    Args:
        name: Shape name.

    Returns:
        The shape.

    Raises:
        KeyError: If the name is not one of :data:`SHAPES`.
    """
    for shape in SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"no shape {name!r}; have {[s.name for s in SHAPES]}")


class OpInputs(NamedTuple):
    """Operator inputs at one shape.

    Attributes:
        U: ``(B,H,T,P)``, low precision.
        trans: ``(B,H,T,4)``, float32, packing ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)``, float32, packing ``(kr, g, h, 0)`` per tap.
        B: ``(B,H,T,3N)``, low precision.
        C: ``(B,H,T,3N)``, low precision.
        dy: ``(B,H,T,P)`` output-gradient seed, preallocated so the backward
            measurement contains no allocation of its own.
    """

    U: Tensor
    trans: Tensor
    K: Tensor
    B: Tensor
    C: Tensor
    dy: Tensor

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """The five tensors gradients are taken with respect to."""
        return (self.U, self.trans, self.K, self.B, self.C)


def make_inputs(
    shape: OpShape,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = True,
    seed: int = 0,
) -> OpInputs:
    """Build operator inputs at one shape.

    Args:
        shape: The problem size.
        device: Where to allocate.
        dtype: Dtype for ``U``, ``B``, ``C``, and ``dy``. ``trans`` and ``K`` are
            float32 regardless, as I4 requires.
        requires_grad: Whether the five differentiable inputs carry gradients.
        seed: Generator seed, so two runs benchmark the same numbers.

    Returns:
        The inputs.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    lead = (shape.bsz, shape.heads, shape.seq)

    def randn(*size: int, dt: torch.dtype = dtype) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    with torch.no_grad():
        params = scanprep_ref(
            randn(*lead, 3, dt=torch.float32),
            randn(*lead, dt=torch.float32),
            randn(*lead, 2, 3, dt=torch.float32),
            w_max=W_MAX,
        )
    trans = params.trans.detach().requires_grad_(requires_grad)
    K = params.K.detach().requires_grad_(requires_grad)
    return OpInputs(
        U=randn(*lead, shape.rows).requires_grad_(requires_grad),
        trans=trans,
        K=K,
        B=randn(*lead, shape.d_state).requires_grad_(requires_grad),
        C=randn(*lead, shape.d_state).requires_grad_(requires_grad),
        dy=randn(*lead, shape.rows),
    )


def forward_only(
    inputs: OpInputs, chunk: int, *, backend: str | None = None, prefix: str = "op"
) -> Callable[[], None]:
    """A callable that runs the forward under ``no_grad``.

    Args:
        inputs: Operator inputs.
        chunk: Chunk length.
        backend: Backend name, or None for the fastest registered one.
        prefix: Region label prefix. Two arms measured in one loop need two
            prefixes, or the recorder sums their regions into one and the inner
            tree describes neither.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.
    """

    def run() -> None:
        with torch.no_grad(), region(f"{prefix}.forward"):
            so3ssd(
                inputs.U,
                inputs.trans,
                inputs.K,
                inputs.B,
                inputs.C,
                chunk,
                backend=backend,
            )

    return run


def step(
    inputs: OpInputs,
    chunk: int,
    *,
    backend: str | None = None,
    wrt: Sequence[Tensor] | None = None,
    prefix: str = "op",
) -> Callable[[], None]:
    """A callable that runs the forward and the backward.

    Args:
        inputs: Operator inputs. The five differentiable ones must require grad.
        chunk: Chunk length.
        backend: Backend name, or None for the fastest registered one.
        wrt: Tensors to differentiate with respect to. Defaults to all five.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.

    Raises:
        ValueError: If no input requires grad, which would time a forward and
            call it a step.
    """
    targets = tuple(inputs.differentiable if wrt is None else wrt)
    if not any(t.requires_grad for t in targets):
        raise ValueError("step needs at least one input requiring grad")

    def run() -> None:
        with region(f"{prefix}.forward"):
            y = so3ssd(
                inputs.U,
                inputs.trans,
                inputs.K,
                inputs.B,
                inputs.C,
                chunk,
                backend=backend,
            ).y
        with region(f"{prefix}.backward"):
            torch.autograd.grad(y, targets, inputs.dy)

    return run
