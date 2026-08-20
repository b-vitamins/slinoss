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
law at the kernel's own traffic, and a register spill as a failure outright. A
kernel whose traffic stays inside L2 gets no bandwidth verdict at all, because
there the counters describe the cache.

The same pass judges the two launch-geometry rules, which are not a class floor and
do not go quiet with one. Every declared kernel gets a
:class:`slinoss.perf.ceiling.GeometryVerdict` whatever its class and whatever its
traffic did, so a kernel left without a bandwidth verdict for being inside L2 is
still held to its occupancy and its grid.
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
    GeometryVerdict,
    dram_floor_verdict,
    geometry_verdict,
    serial_verdict,
)
from slinoss.perf.device import DeviceInfo
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
    "chunk_vector_bwd_kernel": DRAM_BOUND,
    "conv1d_bwd_kernel": DRAM_BOUND,
    "conv1d_fwd_kernel": DRAM_BOUND,
    "conv1d_reduce_parts_kernel": SERIAL_TINY,
    "mixer_tail_bwd_kernel": DRAM_BOUND,
    "mixer_tail_fwd_kernel": DRAM_BOUND,
    "reduce_rows_kernel": SERIAL_TINY,
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
    "xent_bwd_kernel": DRAM_BOUND,
    "xent_fwd_kernel": DRAM_BOUND,
}
"""Every kernel this repo compiles, and the class its module docstring declares.

The chunk recurrence in the two ``state_passing`` kernels is the one provably serial
step in the operator, but serial is not the same as latency bound: once the chunk
fetch is pipelined ahead of the rotation, the serial chain is arithmetic and the
kernel saturates the bus. Both are held to a bandwidth like the rest.

Four entries are SERIAL-tiny, and only the first is unconditionally so.
``boundary_bwd_kernel`` on the single-partial path reads a fixed few rows per chunk
rather than a pass over the sequence, so no shape makes that path large enough to
hold to a bandwidth. That is a property of the kernel and belongs here. Its
multi-partial path is a pass over ``S`` partials, is not SERIAL-tiny, and takes its
own class when a producer that emits partials lands. A shape with too few blocks to
fill the device is a different thing again: a statement about a shape rather than
about a kernel, and it does not live here.

``conv1d_reduce_parts_kernel`` and ``rmsnorm_dweight_kernel`` are reduction tails
over a per-block partial, so their traffic follows the reduced width rather than the
sequence. That width is bounded in the configuration for the first and is ``D`` for
the second, which means the second's declaration has a range. Measured on sm_86:
2.84 us at ``d_model`` 288, under 1% of the backward step; 17.0 us at ``D`` 4096,
76-80% of the copy ceiling and 11% of the step; 30.8 us at ``D`` 8192, 88.5-89.5%.
SERIAL-tiny is right for every shape the driver measures and stops being right
somewhere above ``D`` 2048. Widening the workload past that needs the class
revisited, not the floor lowered.

A key is a source kernel and a verdict is an instantiated symbol, so a template
parameter that changes a kernel's resource profile still gets its own line in the
report without its own key here. ``conv1d_bwd_kernel`` is the case that shows it:
the symbol NCU reports carries the dtype, the filter width, and the staging flag,
so the audit judged ``<c10::BFloat16, 8, true>`` and ``<c10::BFloat16, 4, true>``
separately and the width-8 failure was never averaged into the width-4 pass. A
per-width key could not be added beside the generic one in any case:
:func:`declared_class` raises on a symbol matching two keys, and every width
matches the shorter name.

What the single key does assert is that the class is a property of the kernel and
not of an instantiation. Both widths are held to the same 85%, so a width that
cannot reach it is a defect in that width rather than grounds for a second entry.

``reduce_rows_kernel`` is the same kind of tail, shared by the scan's parameter
frontier and the fused mixer tail. Its row extent follows the sequence where the
other two are bounded by their own grid, but so does the pass that produced the
rows: one partial row per block of that pass, so the reduction's traffic is a fixed
fraction of it and the share holds as ``B*T`` grows rather than climbing with it.
Measured on sm_86, clocks unlocked: 5.952 us at the frontier's ragged shape,
0.999% of the scan-prep step; 4.128 us at the tail's standard shape into a
bfloat16 destination, 0.555% of the mixer-tail step."""


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
measured on an A6000 at the standard shape, ``chunk_scan_fwd_kernel`` at two blocks
per SM spilled nothing and ran 9% faster than the same body at three blocks per SM,
104.4 us against 114.8, and scored 2.7 points lower for it under the floor this
module judges against.

SERIAL-tiny is absent because its bar is a share of the step wall, which a spill
can only worsen. A spilling SERIAL-tiny kernel is still worth fixing; it is not a
corrupted verdict.
"""


@dataclass(frozen=True)
class FloorAudit:
    """The class check against the measured time floor, with the spill rule.

    Attributes:
        verdicts: One verdict per profiled kernel this repo compiles and can judge,
            in the order profiled. A verdict failed by the spill rule carries the
            percentage it achieved and ``passed`` False.
        geometry: One launch-geometry verdict per profiled kernel this repo
            compiles, in the order profiled. Every declared kernel appears, unlike
            ``verdicts``: a geometry rule needs no traffic and no class the counters
            can judge.
        unjudged: Symbols of profiled kernels this repo does not compile.
        spilled: Kernels the spill rule failed, whatever their percentage. A cached
            kernel appears here too: it has no verdict to fail, and a spill is a
            defect at any footprint.
        cached: DRAM-bound kernels left without a verdict because their per-launch
            traffic did not exceed L2.
    """

    verdicts: tuple[ClassVerdict, ...]
    geometry: tuple[GeometryVerdict, ...]
    unjudged: tuple[str, ...]
    spilled: tuple[str, ...]
    cached: tuple[str, ...]

    @property
    def failures(self) -> tuple[str, ...]:
        """One line per rule a kernel failed, naming the rule and the margin.

        The four rules are separate lines, so a kernel failing two produces two:
        the spill rule and the class floor both bear on one percentage, and a
        geometry rule bears on none of it.
        """
        out = [
            f"{kernel}: touched local memory, which fails its class outright"
            for kernel in self.spilled
        ]
        out += [
            f"{one.kernel}: {one.declared} reached {one.achieved_pct:.2f}% "
            f"against the {one.required_pct:.1f}% bar"
            for one in self.verdicts
            if not one.passed
        ]
        out += [
            f"{one.kernel}: {one.detail}" for one in self.geometry if not one.passed
        ]
        return tuple(out)

    @property
    def passed(self) -> bool:
        """Whether every judged kernel cleared every rule this audit applied.

        An audit that judged nothing passes vacuously. What makes a class no driver
        reaches a defect is the coverage rule in ``docs/measurement.md``, not this
        property, which can only report on what a capture contained.
        """
        return not self.failures


def floor_audit(
    kernels: Sequence[KernelCounters],
    *,
    floor: DramTimeFloor,
    spills: Sequence[SpillCounters],
    step_duration_us: Microseconds,
    capture_iters: int,
    device: DeviceInfo,
) -> FloorAudit:
    """Judge every profiled kernel against its class, its geometry, and for spills.

    A DRAM-bound kernel is scored against the time floor at its own measured
    traffic rather than against the rate of the largest copy the device can run;
    see :func:`slinoss.perf.ceiling.dram_floor_verdict`. The bar in
    ``CLASS_FLOOR_PCT`` is the same 85%.

    The occupancy and block-count rules are judged for every declared kernel, first
    and unconditionally; see :func:`slinoss.perf.ceiling.geometry_verdict`. They
    rest on the launch configuration and the warp census, so neither the traffic
    nor the class can withhold them, and only SERIAL-tiny waives the block floor.

    A kernel in :data:`SPILL_FREE_CLASSES` that touched local memory fails,
    whatever its percentage. The percentage is still reported, because the
    achieved figure is what says how far the spill moved the kernel.

    A DRAM-bound kernel whose per-launch traffic did not exceed L2 gets no verdict.
    Measured traffic under the cache size is not a lower bound on the work: it says
    the launch could have been served without reaching DRAM, so the same kernel
    reads high or low with the cache state rather than with its own quality. The
    fitted law is also extrapolated there, every swept footprint being above L2.
    This is the case at the smallest shape, where DRAM reads are literally zero.

    Args:
        kernels: Merged NCU counters for one capture window.
        floor: Time floor measured on the same device in the same process, so
            numerator and denominator drift together under an unlocked clock.
        spills: One :func:`slinoss.perf.ncu.spill_counters` record per kernel, from
            a :data:`slinoss.perf.ncu.SPILL_TABLE` pass over the same window.
        step_duration_us: Measured per-iteration wall. The SERIAL-tiny divisor.
        capture_iters: Iterations the capture window contained. Divides a counter
            sum onto the same per-iteration footing as ``step_duration_us``.
        device: The part the kernels ran on. Sets the block-count floor at twice
            its multiprocessor count, so the bar is queried rather than written
            down.

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
    geometry: list[GeometryVerdict] = []
    unjudged: list[str] = []
    spilled: list[str] = []
    cached: list[str] = []
    for one in kernels:
        declared = declared_class(one.kernel)
        if declared is None:
            unjudged.append(one.kernel)
            continue
        # Recorded before the cached exit, so a kernel with no bandwidth verdict
        # still reports the geometry it ran at.
        geometry.append(
            geometry_verdict(
                one.kernel,
                declared=declared,
                block_count=one.block_count,
                thread_per_block_count=one.thread_per_block_count,
                achieved_occupancy_pct=one.achieved_occupancy_pct,
                theoretical_occupancy_pct=one.theoretical_occupancy_pct,
                device=device,
            )
        )
        if declared in SPILL_FREE_CLASSES and one.kernel not in by_kernel:
            raise ValueError(
                f"kernel {one.kernel!r} declares {declared} and carries no spill "
                f"record; run SPILL_TABLE over the same window"
            )
        # Recorded before the cached exit, so a kernel with no verdict still reports
        # the spill.
        failed_by_spill = (
            declared in SPILL_FREE_CLASSES and by_kernel[one.kernel].spilled
        )
        if failed_by_spill:
            spilled.append(one.kernel)
        if declared == DRAM_BOUND:
            traffic = Bytes(one.dram_read_bytes + one.dram_write_bytes)
            if one.launch_count > 0 and traffic // one.launch_count <= floor.l2_bytes:
                cached.append(one.kernel)
                continue
            verdict = dram_floor_verdict(
                one.kernel,
                moved_bytes=traffic,
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
        if failed_by_spill:
            verdict = replace(verdict, passed=False)
        verdicts.append(verdict)
    return FloorAudit(
        verdicts=tuple(verdicts),
        geometry=tuple(geometry),
        unjudged=tuple(unjudged),
        spilled=tuple(spilled),
        cached=tuple(cached),
    )
