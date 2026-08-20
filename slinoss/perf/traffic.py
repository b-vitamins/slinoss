"""What a kernel asked for against what DRAM delivered.

:func:`slinoss.perf.ceiling.dram_floor_verdict` scores a DRAM-bound kernel against
the time the measured DRAM traffic implies. That denominator is the kernel's own
bytes, so eliminating a round trip takes bytes out of the numerator and out of the
floor together, and a kernel that moves less traffic is measured against a smaller
floor. Two arms computing the same thing therefore do not compare: measured on sm_86
at the model geometry, the fused chunk-start-and-reverse-recurrence kernel moved
176.5 MB in 474.9 us and scored 55.2% of its floor, while the two kernels it replaces
moved 453.9 MB in 741.0 us and scored 78.7% and 99.3% of theirs. The fused arm is
36.0% faster, moves 61.1% less, and reads red.

This module reports the gap the floor does not see. The LSU's request stream is
counted at L1TEX, so the difference between it and the DRAM stream is what the caches
served: a re-read of a band that stayed resident is real work the kernel issued, real
latency it hid, and no byte at the DRAM boundary. The same fused kernel requests
256 MB and reads 176 MB from DRAM, so 31% of what it asked for never left the chip.

Report only. Nothing here feeds a verdict, and the floor is unchanged: which
denominator is right for a kernel whose request stream exceeds its DRAM stream is
settled in ``docs/measurement.md``, and changing a floor is the one move that can
launder a failure into a pass, so it does not happen as a side effect of adding a
column.

Derived from counters :data:`slinoss.perf.ncu.NCU_TABLES` already collects, so the
table costs no pass and no metric that could be absent on another part.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Final

from slinoss.perf.ncu import KernelCounters
from slinoss.perf.units import MEDIAN, SUM, Bytes, Count, Percent, PerfRecord, Ratio

__all__ = ["SECTOR_BYTES", "TrafficMix", "traffic_mix"]

SECTOR_BYTES: Final = 32
"""Bytes in one cache sector.

Fixed at 32 on every part this repo targets, from Pascal on. The global counters in
the ``global`` table are sector counts, and a sector is the unit L1TEX moves, so the
byte figure is a count times this and not a second measurement."""


@dataclass(frozen=True)
class TrafficMix(PerfRecord):
    """One kernel's request stream beside its DRAM stream.

    Attributes:
        kernel: Demangled kernel name.
        launch_count: Launches profiled, so the two byte figures are read on the
            same footing as the counter row they come from.
        requested_bytes: Bytes the LSU asked L1TEX for, loads plus stores, from the
            global sector counters. The kernel's own demand, before any cache
            answered it.
        dram_bytes: Bytes that crossed the DRAM boundary, reads plus writes. The
            numerator of the DRAM-bound floor.
        cached_bytes: The difference, which L1 or L2 served. Negative is possible in
            principle -- a write-allocate or an eviction can move a byte no request
            asked for -- and is reported rather than clamped, because a kernel whose
            DRAM stream exceeds its requests is doing something the byte model does
            not describe.
        dram_per_request_ratio: ``dram_bytes`` over ``requested_bytes``. One means
            every request reached DRAM and the floor's denominator is the kernel's
            whole demand. Below one, the floor prices a fraction of the work.
        cache_served_pct: ``cached_bytes`` as a percentage of ``requested_bytes``.
        l2_pct: L2 throughput against peak sustained, carried from the counter row.
            The corroboration: a large served fraction beside a busy L2 says the
            gap is L2 hits rather than a mis-parsed counter.
    """

    kernel: str
    launch_count: Annotated[Count, SUM]
    requested_bytes: Annotated[Bytes, SUM]
    dram_bytes: Annotated[Bytes, SUM]
    cached_bytes: Annotated[Bytes, SUM]
    dram_per_request_ratio: Annotated[Ratio, MEDIAN]
    cache_served_pct: Annotated[Percent, MEDIAN]
    l2_pct: Annotated[Percent, MEDIAN]


def traffic_mix(kernels: Sequence[KernelCounters]) -> tuple[TrafficMix, ...]:
    """Split every kernel's traffic into what it asked for and what DRAM delivered.

    Args:
        kernels: Merged counters, as :func:`slinoss.perf.ncu.kernel_counters`
            returns them. Order is preserved, so the table reads in the same order
            as the counter table beside it.

    Returns:
        One record per kernel. A kernel that issued no global access has zero
        requested bytes and both derived figures zero, rather than a division by
        zero: a kernel touching only shared memory has no request stream for the
        floor to mis-price.
    """
    out: list[TrafficMix] = []
    for one in kernels:
        requested = Bytes(
            SECTOR_BYTES
            * (one.global_load_sector_count + one.global_store_sector_count)
        )
        dram = Bytes(one.dram_read_bytes + one.dram_write_bytes)
        cached = Bytes(requested - dram)
        out.append(
            TrafficMix(
                kernel=one.kernel,
                launch_count=one.launch_count,
                requested_bytes=requested,
                dram_bytes=dram,
                cached_bytes=cached,
                dram_per_request_ratio=Ratio(
                    0.0 if requested == 0 else dram / requested
                ),
                cache_served_pct=Percent(
                    0.0 if requested == 0 else 100.0 * cached / requested
                ),
                l2_pct=one.l2_pct,
            )
        )
    return tuple(out)
