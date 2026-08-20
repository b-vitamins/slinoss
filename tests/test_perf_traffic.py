"""The request stream beside the DRAM stream, and the ratio between them.

The DRAM-bound floor prices the bytes that crossed the DRAM boundary, so a kernel
whose requests exceed that is scored against a fraction of the work it issued. This
table reports the fraction. It feeds no verdict, which is the reason it can be added
without moving a bar, so what is pinned is the arithmetic and the degenerate cases.

The counter records are fabricated field by field: the subject is a subtraction over
counters :mod:`slinoss.perf.ncu` already collects, not the collection.
"""

from __future__ import annotations

import pytest

from slinoss.perf.ncu import KernelCounters
from slinoss.perf.traffic import SECTOR_BYTES, traffic_mix
from tests.test_script_profile_op import counter_record


def counters(
    *, load_sectors: int, store_sectors: int, dram_bytes: int
) -> KernelCounters:
    """One counter row with the four fields the split reads."""
    return counter_record(
        read_bytes=dram_bytes,
        write_bytes=0,
        load_sectors=load_sectors,
        store_sectors=store_sectors,
    )


def test_the_split_is_what_dram_served_and_what_the_caches_did() -> None:
    """The measured case, in the proportions it was measured in.

    The fused reverse-recurrence kernel requests about 256 MB and reads about 176 MB
    from DRAM, so near a third of its demand never left the chip and the floor prices
    the rest. A sector is 32 B on every part this repo targets, so the request figure
    is a count times that constant and not a second measurement.
    """
    requested = 256 << 20
    dram = 176 << 20
    mix = traffic_mix(
        [
            counters(
                load_sectors=requested // SECTOR_BYTES - 1024,
                store_sectors=1024,
                dram_bytes=dram,
            )
        ]
    )
    assert len(mix) == 1
    one = mix[0]
    assert one.requested_bytes == requested
    assert one.dram_bytes == dram
    assert one.cached_bytes == requested - dram
    assert one.dram_per_request_ratio == pytest.approx(dram / requested)
    assert one.cache_served_pct == pytest.approx(100.0 * (requested - dram) / requested)
    # Carried, not derived: a large served fraction beside a busy L2 is what says the
    # gap is cache hits rather than a mis-parsed counter.
    assert one.l2_pct == 27.0


def test_a_kernel_with_no_global_access_reports_zero_rather_than_dividing() -> None:
    # A kernel touching only shared memory has no request stream for the floor to
    # mis-price, and the table is emitted for every profiled kernel.
    one = traffic_mix([counters(load_sectors=0, store_sectors=0, dram_bytes=0)])[0]
    assert one.requested_bytes == 0
    assert one.dram_per_request_ratio == 0.0
    assert one.cache_served_pct == 0.0


def test_more_dram_than_was_requested_is_reported_rather_than_clamped() -> None:
    # A write-allocate or an eviction moves a byte no request asked for. Clamping
    # would hide a kernel whose traffic the byte model does not describe, and that is
    # exactly the kernel whose floor is wrong.
    one = traffic_mix([counters(load_sectors=32, store_sectors=0, dram_bytes=1 << 20)])[
        0
    ]
    assert one.cached_bytes < 0
    assert one.dram_per_request_ratio > 1.0
    assert one.cache_served_pct < 0.0
