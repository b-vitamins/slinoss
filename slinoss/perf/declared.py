"""Which class each kernel declares, as data a driver can read.

Every kernel is declared DRAM-bound, TENSOR-bound, or SERIAL-tiny and held to
that class. The declaration itself is prose in the kernel's module docstring,
which no driver can parse, so it is restated here as a table and the profile
driver judges against it. One table in the perf package, rather than an export
from each op package: the perf package is the only consumer, and importing every
op package to read a constant would drag the CuTe DSL and the compiled extension
into a harness that runs without either.

Keys are function names, matched as substrings of the symbol NCU reports. Neither
symbol equals its source name: the CuTe DSL emits
``kernel_cutlass_<function>_<traced signature>_0`` and the extension emits the
namespace-qualified name with its template arguments. Substring matching is what
survives both, and a symbol matching two keys raises rather than picking one.

An undeclared kernel is not silently unverdicted. A profiled symbol carrying one
of :data:`OWNED_MARKERS` and matching no key raises; a symbol carrying neither
marker came from torch, cuBLAS, or the driver, so it is reported as unjudged
rather than judged against a class this repo did not declare.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from slinoss.perf.ceiling import (
    DRAM_BOUND,
    SERIAL_TINY,
    TENSOR_BOUND,
    Ceilings,
    ClassVerdict,
    dram_verdict,
    serial_verdict,
)
from slinoss.perf.ncu import KernelCounters
from slinoss.perf.units import Microseconds, pct_of

__all__ = [
    "DECLARED",
    "OWNED_MARKERS",
    "ClassAudit",
    "class_audit",
    "declared_class",
]

OWNED_MARKERS: Final[tuple[str, ...]] = ("kernel_cutlass_", "slinoss::")
"""Symbol markers for a kernel this repo compiles: the CuTe DSL prefix and the
extension namespace."""

DECLARED: Final[dict[str, str]] = {
    "boundary_bwd_kernel": SERIAL_TINY,
    "chunk_increment_fwd_kernel": DRAM_BOUND,
    "chunk_scan_fwd_kernel": DRAM_BOUND,
    "chunk_start_bwd_kernel": DRAM_BOUND,
    "conv1d_bwd_kernel": DRAM_BOUND,
    "conv1d_fwd_kernel": DRAM_BOUND,
    "mixer_tail_bwd_kernel": DRAM_BOUND,
    "mixer_tail_fwd_kernel": DRAM_BOUND,
    "rmsnorm_fwd_kernel": DRAM_BOUND,
    "rmsnorm_residual_fwd_kernel": DRAM_BOUND,
    "scanprep_bwd_kernel": DRAM_BOUND,
    "scanprep_fwd_kernel": DRAM_BOUND,
    "state_passing_bwd_kernel": DRAM_BOUND,
    "state_passing_fwd_kernel": DRAM_BOUND,
    "swiglu_fwd_kernel": DRAM_BOUND,
}
"""Every kernel this repo compiles, and the class its module docstring declares.

The chunk recurrence in the two ``state_passing`` kernels is the one provably serial
step in the operator, but serial is not the same as latency bound: once the chunk
fetch is pipelined ahead of the rotation, the serial chain is arithmetic and the
kernel saturates the bus. Both are held to a bandwidth like the rest.

``boundary_bwd_kernel`` is the one SERIAL-tiny entry. Its traffic is a fixed few
rows per chunk rather than a pass over the sequence, so no shape makes it large
enough to hold to a bandwidth. That is a property of the kernel and belongs here. A
shape with too few blocks to fill the device is a different thing: a statement about
a shape rather than about a kernel, and it does not live here."""


def declared_class(kernel: str) -> str | None:
    """Look up the class a profiled kernel declares.

    Args:
        kernel: Kernel symbol, as NCU reports it.

    Returns:
        The declared class, or None if the symbol is not one this repo compiles.

    Raises:
        ValueError: If the symbol is one this repo compiles and matches no entry
            of :data:`DECLARED`, or if it matches more than one.
    """
    if not any(marker in kernel for marker in OWNED_MARKERS):
        return None
    hits = sorted(name for name in DECLARED if name in kernel)
    if not hits:
        raise ValueError(
            f"kernel {kernel!r} is compiled here and declares no class; "
            f"add it to DECLARED"
        )
    if len(hits) > 1:
        raise ValueError(f"kernel {kernel!r} matches {hits}; one symbol, one class")
    return DECLARED[hits[0]]


@dataclass(frozen=True)
class ClassAudit:
    """The class check over one profiled capture window.

    Attributes:
        verdicts: One verdict per profiled kernel this repo compiles, in the
            order profiled.
        unjudged: Symbols of profiled kernels this repo does not compile.
    """

    verdicts: tuple[ClassVerdict, ...]
    unjudged: tuple[str, ...]


def class_audit(
    kernels: Sequence[KernelCounters],
    *,
    limits: Ceilings,
    step_duration_us: Microseconds,
    capture_iters: int,
) -> ClassAudit:
    """Judge every profiled kernel against the class it declares.

    Args:
        kernels: Merged NCU counters for one capture window.
        limits: Ceilings measured on the same device at the same clocks, so
            numerator and denominator drift together.
        step_duration_us: Measured per-iteration wall. The SERIAL-tiny divisor.
        capture_iters: Iterations the capture window contained. Divides a counter
            sum onto the same per-iteration footing as ``step_duration_us``.

    Returns:
        The audit.

    Raises:
        ValueError: If ``capture_iters`` is not positive, or if a declared class
            cannot be judged from the collected counters.
    """
    if capture_iters <= 0:
        raise ValueError(f"capture_iters must be positive, got {capture_iters}")
    verdicts: list[ClassVerdict] = []
    unjudged: list[str] = []
    for one in kernels:
        declared = declared_class(one.kernel)
        if declared is None:
            unjudged.append(one.kernel)
        elif declared == DRAM_BOUND:
            verdicts.append(dram_verdict(one.kernel, one.achieved_gbs, limits.dram))
        elif declared == SERIAL_TINY:
            per_iter = Microseconds(one.duration_us / capture_iters)
            verdicts.append(
                serial_verdict(one.kernel, pct_of(per_iter, step_duration_us))
            )
        else:
            raise ValueError(
                f"kernel {one.kernel!r} declares {declared}, which the collected "
                f"counters cannot judge: {TENSOR_BOUND} needs a flop count and no "
                f"table in NCU_TABLES collects one"
            )
    return ClassAudit(verdicts=tuple(verdicts), unjudged=tuple(unjudged))
