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

:func:`floor_audit` judges that table: a DRAM-bound kernel against the copy's time
law at the kernel's own traffic, and a register spill as a failure outright.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from slinoss.perf.ceiling import (
    DRAM_BOUND,
    SERIAL_TINY,
    TENSOR_BOUND,
    ClassVerdict,
    DramTimeFloor,
    dram_floor_verdict,
    serial_verdict,
)
from slinoss.perf.ncu import KernelCounters, SpillCounters
from slinoss.perf.units import Bytes, Microseconds, pct_of

__all__ = [
    "DECLARED",
    "OWNED_MARKERS",
    "SPILL_FREE_CLASSES",
    "FloorAudit",
    "declared_class",
    "floor_audit",
]

OWNED_MARKERS: Final[tuple[str, ...]] = ("kernel_cutlass_", "slinoss::")
"""Symbol markers for a kernel this repo compiles: the CuTe DSL prefix and the
extension namespace."""

DECLARED: Final[dict[str, str]] = {
    "boundary_bwd_kernel": SERIAL_TINY,
    "chunk_increment_fwd_kernel": DRAM_BOUND,
    "chunk_input_bwd_kernel": DRAM_BOUND,
    "chunk_scan_fwd_kernel": DRAM_BOUND,
    "chunk_start_bwd_kernel": DRAM_BOUND,
    "conv1d_bwd_kernel": DRAM_BOUND,
    "conv1d_fwd_kernel": DRAM_BOUND,
    "conv1d_reduce_parts_kernel": SERIAL_TINY,
    "mixer_tail_bwd_kernel": DRAM_BOUND,
    "mixer_tail_fwd_kernel": DRAM_BOUND,
    "rmsnorm_bwd_kernel": DRAM_BOUND,
    "rmsnorm_dweight_kernel": SERIAL_TINY,
    "rmsnorm_fwd_kernel": DRAM_BOUND,
    "rmsnorm_residual_bwd_kernel": DRAM_BOUND,
    "rmsnorm_residual_fwd_kernel": DRAM_BOUND,
    "scanprep_bwd_kernel": DRAM_BOUND,
    "scanprep_fwd_kernel": DRAM_BOUND,
    "state_passing_bwd_kernel": DRAM_BOUND,
    "state_passing_fwd_kernel": DRAM_BOUND,
    "swiglu_bwd_kernel": DRAM_BOUND,
    "swiglu_fwd_kernel": DRAM_BOUND,
}
"""Every kernel this repo compiles, and the class its module docstring declares.

The chunk recurrence in the two ``state_passing`` kernels is the one provably serial
step in the operator, but serial is not the same as latency bound: once the chunk
fetch is pipelined ahead of the rotation, the serial chain is arithmetic and the
kernel saturates the bus. Both are held to a bandwidth like the rest.

Three entries are SERIAL-tiny, and only the first is unconditionally so.
``boundary_bwd_kernel`` reads a fixed few rows per chunk rather than a pass over the
sequence, so no shape makes it large enough to hold to a bandwidth. That is a
property of the kernel and belongs here. A shape with too few blocks to fill the
device is a different thing: a statement about a shape rather than about a kernel,
and it does not live here.

``conv1d_reduce_parts_kernel`` and ``rmsnorm_dweight_kernel`` are reduction tails
over a per-block partial, so their traffic follows the reduced width rather than the
sequence. That width is bounded in the configuration for the first and is ``D`` for
the second, which means the second's declaration has a range. Measured on sm_86:
2.84 us at ``d_model`` 288, under 1% of the backward step; 17.0 us at ``D`` 4096,
76-80% of the copy ceiling and 11% of the step; 30.8 us at ``D`` 8192, 88.5-89.5%.
SERIAL-tiny is right for every shape the driver measures and stops being right
somewhere above ``D`` 2048. Widening the workload past that needs the class
revisited, not the floor lowered."""


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


SPILL_FREE_CLASSES: Final[frozenset[str]] = frozenset((DRAM_BOUND, TENSOR_BOUND))
"""Classes a register spill fails outright.

Both hold a kernel to a rate, and a rate is a counted quantity over a duration. A
spill adds traffic and instructions that the class's own byte or flop model does
not contain, so it moves the counted quantity as well as the duration and the
percentage stops ordering two configurations of the same kernel by speed:
measured on an A6000, ``chunk_scan_fwd_kernel`` at two blocks per SM spilled
nothing and ran 5.8% faster than the same body at three blocks per SM, and scored
2.7 points lower for it under the floor this module judges against.

SERIAL-tiny is absent because its bar is a share of the step wall, which a spill
can only worsen. A spilling SERIAL-tiny kernel is still worth fixing; it is not a
corrupted verdict.
"""


@dataclass(frozen=True)
class FloorAudit:
    """The class check against the measured time floor, with the spill rule.

    Attributes:
        verdicts: One verdict per profiled kernel this repo compiles, in the order
            profiled. A verdict failed by the spill rule carries the percentage it
            achieved and ``passed`` False.
        unjudged: Symbols of profiled kernels this repo does not compile.
        spilled: Kernels the spill rule failed, whatever their percentage.
    """

    verdicts: tuple[ClassVerdict, ...]
    unjudged: tuple[str, ...]
    spilled: tuple[str, ...]


def floor_audit(
    kernels: Sequence[KernelCounters],
    *,
    floor: DramTimeFloor,
    spills: Sequence[SpillCounters],
    step_duration_us: Microseconds,
    capture_iters: int,
) -> FloorAudit:
    """Judge every profiled kernel against its class, at the floor and for spills.

    A DRAM-bound kernel is scored against the time floor at its own measured
    traffic rather than against the rate of the largest copy the device can run;
    see :func:`slinoss.perf.ceiling.dram_floor_verdict`. The bar in
    ``CLASS_FLOOR_PCT`` is the same 85%.

    A kernel in :data:`SPILL_FREE_CLASSES` that touched local memory fails,
    whatever its percentage. The percentage is still reported, because the
    achieved figure is what says how far the spill moved the kernel.

    Args:
        kernels: Merged NCU counters for one capture window.
        floor: Time floor measured on the same device in the same process, so
            numerator and denominator drift together under an unlocked clock.
        spills: One :func:`slinoss.perf.ncu.spill_counters` record per kernel, from
            a :data:`slinoss.perf.ncu.SPILL_TABLE` pass over the same window.
        step_duration_us: Measured per-iteration wall. The SERIAL-tiny divisor.
        capture_iters: Iterations the capture window contained. Divides a counter
            sum onto the same per-iteration footing as ``step_duration_us``.

    Returns:
        The audit.

    Raises:
        ValueError: If ``capture_iters`` is not positive, if a kernel in a
            spill-free class carries no spill record, or if a declared class cannot
            be judged from the collected counters. A missing spill record is a pass
            that was never run, and treating it as no spill would report every
            spilling kernel as clean.
    """
    if capture_iters <= 0:
        raise ValueError(f"capture_iters must be positive, got {capture_iters}")
    by_kernel = {one.kernel: one for one in spills}
    verdicts: list[ClassVerdict] = []
    unjudged: list[str] = []
    spilled: list[str] = []
    for one in kernels:
        declared = declared_class(one.kernel)
        if declared is None:
            unjudged.append(one.kernel)
            continue
        if declared in SPILL_FREE_CLASSES and one.kernel not in by_kernel:
            raise ValueError(
                f"kernel {one.kernel!r} declares {declared} and carries no spill "
                f"record; run SPILL_TABLE over the same window"
            )
        if declared == DRAM_BOUND:
            verdict = dram_floor_verdict(
                one.kernel,
                moved_bytes=Bytes(one.dram_read_bytes + one.dram_write_bytes),
                launch_count=one.launch_count,
                duration_us=one.duration_us,
                floor=floor,
            )
        elif declared == SERIAL_TINY:
            per_iter = Microseconds(one.duration_us / capture_iters)
            verdict = serial_verdict(one.kernel, pct_of(per_iter, step_duration_us))
        else:
            raise ValueError(
                f"kernel {one.kernel!r} declares {declared}, which the collected "
                f"counters cannot judge: {TENSOR_BOUND} needs a flop count and no "
                f"table in NCU_TABLES collects one"
            )
        if declared in SPILL_FREE_CLASSES and by_kernel[one.kernel].spilled:
            spilled.append(one.kernel)
            verdict = replace(verdict, passed=False)
        verdicts.append(verdict)
    return FloorAudit(
        verdicts=tuple(verdicts), unjudged=tuple(unjudged), spilled=tuple(spilled)
    )
