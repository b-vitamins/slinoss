"""The measured time floor, the spill rule, and the arcs the audit chooses between.

The fit is pure arithmetic and is driven from synthetic samples carrying an exact
timing law, so the test states what the estimator must recover rather than what
one device happened to read. Only the agreement between the fit and the existing
single-point ceiling needs hardware, and that one is skipped off CUDA.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import pytest
import torch

from slinoss.perf.ceiling import (
    CLASS_FLOOR_PCT,
    DRAM_BOUND,
    MIN_OCCUPANCY_PCT,
    SERIAL_TINY,
    TENSOR_BOUND,
    CopySample,
    DramTimeFloor,
    dram_ceiling,
    dram_floor_verdict,
    dram_time_floor,
    geometry_verdict,
)
from slinoss.perf.declared import DECLARED, FloorAudit, floor_audit
from slinoss.perf.device import ClockPolicy, Contention, DeviceInfo
from slinoss.perf.ncu import KernelCounters, SpillCounters
from slinoss.perf.units import (
    Bytes,
    Count,
    Mebibytes,
    Megahertz,
    Microseconds,
    Percent,
    Ratio,
    Spread,
    gbs_from_bytes_us,
)

FIXED_US: Final = 3.75
"""Synthetic fixed cost of one copy, in microseconds."""

RATE_GBS: Final = 700.0
"""Synthetic asymptotic rate, in gigabytes per second."""

L2_BYTES: Final = Bytes(6 << 20)
"""L2 size the synthetic samples were swept against."""

OWNED: Final = "kernel_cutlass_chunk_scan_fwd_kernel_bf16_Ampere_0"
"""A DRAM-bound kernel under the symbol NCU reports."""

SM_COUNT: Final = Count(84)
"""Multiprocessors the fabricated part reports. Twice this is the block floor."""


def device_record() -> DeviceInfo:
    """A part to judge a launch geometry against.

    Only ``sm_count`` reaches a verdict: the block floor is twice it. Every other
    field is filled because the record demands them, and none is asserted.
    """
    return DeviceInfo(
        name="Test Part",
        capability="8.6",
        sm_count=SM_COUNT,
        warp_thread_count=Count(32),
        max_threads_per_sm_count=Count(1536),
        regs_per_sm_count=Count(65536),
        smem_per_block_bytes=Bytes(49152),
        smem_optin_per_block_bytes=Bytes(101376),
        smem_per_sm_bytes=Bytes(102400),
        l2_bytes=L2_BYTES,
        total_memory_bytes=Bytes(51041271808),
        clocks=ClockPolicy(
            locked=False,
            sm_clock_mhz=Megahertz(1740.0),
            max_sm_clock_mhz=Megahertz(1800.0),
            detail="fabricated",
        ),
        sharing=Contention(
            probed=True,
            foreign_process_count=Count(0),
            foreign_memory_mib=Mebibytes(0.0),
            utilization_pct=Percent(0.0),
            detail="fabricated",
        ),
    )


def law_us(moved_bytes: int) -> Microseconds:
    """Duration the synthetic law assigns to a copy of ``moved_bytes``."""
    return Microseconds(FIXED_US + moved_bytes / (1e3 * RATE_GBS))


def sample(moved_bytes: int, *, duration_us: float | None = None) -> CopySample:
    """One copy sample on the synthetic law, or at an imposed duration."""
    taken = law_us(moved_bytes) if duration_us is None else Microseconds(duration_us)
    return CopySample(
        moved_bytes=Bytes(moved_bytes),
        duration=Spread.of((taken,)),
        achieved_gbs=gbs_from_bytes_us(Bytes(moved_bytes), taken),
        l2_multiple_ratio=Ratio(moved_bytes / (2 * L2_BYTES)),
    )


def synthetic_floor(*moved_bytes: int) -> DramTimeFloor:
    """The fit over samples of the synthetic law at the given footprints."""
    return DramTimeFloor.of(
        "synthetic", [sample(size) for size in moved_bytes], l2_bytes=L2_BYTES
    )


SWEEP: Final[tuple[int, ...]] = (24 << 20, 48 << 20, 96 << 20, 192 << 20, 384 << 20)
"""Synthetic sweep footprints, read plus write, spanning sixteenfold."""


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_the_fit_recovers_the_timing_law_it_was_given() -> None:
    """Both terms, and the floor the pair extrapolates to a small footprint.

    Recovering the slope alone is not enough. The intercept is the whole of the
    correction this record exists for: it is what makes the floor at a 10 MB
    footprint different from the rate of a 512 MiB copy.
    """
    floor = synthetic_floor(*SWEEP)
    assert floor.fixed_duration_us == pytest.approx(FIXED_US)
    assert floor.asymptotic_gbs == pytest.approx(RATE_GBS)
    assert floor.max_residual_pct == pytest.approx(0.0)
    assert floor.copies[0].moved_bytes == SWEEP[0]
    small = Bytes(10_030_000)
    assert floor.floor_us(small) == pytest.approx(law_us(small))
    # The size-matched rate is the same law read as a bandwidth, and it is well
    # under the asymptote at a footprint this small. That gap is defect A.
    assert floor.floor_gbs(small) == pytest.approx(
        gbs_from_bytes_us(small, law_us(small))
    )
    assert floor.floor_gbs(small) < 0.85 * floor.asymptotic_gbs


def test_the_fit_reports_a_sample_that_does_not_lie_on_it() -> None:
    """A residual is the only thing that says whether the law holds.

    Fitting cannot fail, so a sweep that is not a line still returns two terms. The
    residual is what a reader checks them against, and it is reported rather than
    gated: a threshold would hide the one fact that says whether the denominator
    can be trusted at the footprint it is used at.
    """
    off = list(SWEEP)
    exact = synthetic_floor(*off)
    perturbed = DramTimeFloor.of(
        "perturbed",
        [
            sample(off[0], duration_us=law_us(off[0]) * 1.2),
            *[sample(s) for s in off[1:]],
        ],
        l2_bytes=L2_BYTES,
    )
    assert exact.max_residual_pct < 1e-9
    assert perturbed.max_residual_pct > 1.0


def test_the_fit_refuses_a_sweep_that_pins_no_law() -> None:
    """Both rejections. Either one would otherwise return a fitted-looking pair."""
    with pytest.raises(ValueError, match="two distinct footprints"):
        synthetic_floor(1 << 24, 1 << 24)
    # A copy that does not take longer as it grows measured something other than
    # bandwidth. The reciprocal slope would come back negative and read as a rate.
    with pytest.raises(ValueError, match="is not positive"):
        DramTimeFloor.of(
            "flat",
            [sample(1 << 24, duration_us=50.0), sample(1 << 25, duration_us=40.0)],
            l2_bytes=L2_BYTES,
        )


def test_the_floor_charges_the_fixed_cost_once_per_launch() -> None:
    """A capture window of n launches pays the fixed cost n times.

    Both NCU sums cover every launch in the window, so folding the fixed term in
    once understates the floor by ``(n - 1) * c`` and scores a multi-launch kernel
    low for a cost it really paid.
    """
    floor = synthetic_floor(*SWEEP)
    per_launch = Bytes(10_030_000)
    launches = Count(8)
    at_floor = Microseconds(launches * floor.floor_us(per_launch))
    verdict = dram_floor_verdict(
        OWNED,
        moved_bytes=Bytes(per_launch * launches),
        launch_count=launches,
        duration_us=at_floor,
        floor=floor,
    )
    assert verdict.achieved_pct == pytest.approx(100.0)
    assert verdict.required_pct == CLASS_FLOOR_PCT[DRAM_BOUND]
    assert verdict.passed
    # Charging the whole window's traffic once is the same law with seven fixed
    # costs missing, and it scores a kernel sitting exactly on its floor as under
    # it by that much.
    folded = 100.0 * floor.floor_us(Bytes(per_launch * launches)) / at_floor
    assert folded == pytest.approx(100.0 * (1.0 - 7.0 * FIXED_US / at_floor), rel=1e-12)
    assert folded < 100.0
    with pytest.raises(ValueError, match="launch_count must be positive"):
        dram_floor_verdict(
            OWNED,
            moved_bytes=Bytes(1 << 20),
            launch_count=Count(0),
            duration_us=Microseconds(1.0),
            floor=floor,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_the_floor_agrees_with_the_single_point_ceiling_at_its_footprint() -> None:
    """The two probes must read the same device at the same footprint.

    The floor is the ceiling generalized, so at the largest swept footprint the
    fitted law and the single-point copy have to agree. Disagreement there means
    the sweep is measuring a different copy than ``dram_ceiling`` does, and every
    small-footprint floor extrapolated from it would be wrong by that much with
    nothing to show it.

    The tolerance is nine times the disagreement measured on this fleet: ten
    back-to-back takes on an unlocked A6000 disagreed by at most 0.11%, at both
    iteration counts. The largest footprint dominates the fit, so this is the one
    place the two probes are pinned together; the small end is not, and
    ``max_residual_pct`` is where that shows.
    """
    device = torch.device("cuda")
    floor = dram_time_floor(device, iters=5, warmup=2)
    largest = floor.copies[-1]
    ceiling = dram_ceiling(
        device,
        requested_bytes=largest.moved_bytes // 2,
        iters=5,
        warmup=2,
    )
    # Every point above L2, which is what makes the samples DRAM rates. The
    # nominal multiple is not asserted: buffers round down to a whole mebibyte and
    # an L2 that is not mebibyte-aligned puts a point just under its multiple.
    assert all(one.l2_multiple_ratio > 1.0 for one in floor.copies)
    # Compared at the footprint the ceiling actually measured, which free memory
    # can clamp below the request. Inside the sweep, so this interpolates.
    assert ceiling.moved_bytes >= floor.copies[0].moved_bytes
    assert floor.floor_gbs(ceiling.moved_bytes) == pytest.approx(
        ceiling.achieved_gbs, rel=0.01
    )
    # A copy of the kernel's own size is slower than the largest copy the device
    # can run. That is the whole of defect A, and it is a property of the part.
    assert floor.floor_gbs(Bytes(10_030_000)) < ceiling.achieved_gbs


# ---------------------------------------------------------------------------
# The spill rule
# ---------------------------------------------------------------------------


WINDOW_US: Final = Microseconds(250.0)
"""Summed duration of the fixture's three launches.

Set so the fixture clears the bar against the synthetic floor. The spill test needs
a record that passes on its percentage, because a record that failed anyway would
not show that the spill rule is what failed it.
"""

TRAFFIC_BYTES: Final = Bytes(157_440_000)
"""Summed DRAM read plus write over the same three launches."""


def spill_record(sectors: int, *, kernel: str = OWNED) -> SpillCounters:
    """Local-memory sectors for one kernel, split over load and store."""
    return SpillCounters(
        kernel=kernel,
        launch_count=Count(3),
        duration_us=WINDOW_US,
        local_load_sector_count=Count(2 * sectors),
        local_store_sector_count=Count(sectors),
    )


def counters(kernel: str = OWNED) -> KernelCounters:
    """Merged counters for one DRAM-bound kernel over three launches."""
    return KernelCounters(
        kernel=kernel,
        launch_count=Count(3),
        duration_us=WINDOW_US,
        pass_duration_spread_pct=Percent(0.4),
        dram_read_bytes=Bytes(120_000_000),
        dram_write_bytes=Bytes(TRAFFIC_BYTES - 120_000_000),
        dram_pct=Percent(92.0),
        achieved_gbs=gbs_from_bytes_us(TRAFFIC_BYTES, WINDOW_US),
        global_load_request_count=Count(1 << 18),
        global_store_request_count=Count(1 << 17),
        global_load_sector_count=Count(1 << 20),
        global_store_sector_count=Count(1 << 19),
        sector_per_load_request_ratio=Ratio(4.0),
        sector_per_store_request_ratio=Ratio(4.0),
        wavefront_count=Count(4096),
        shared_load_conflict_count=Count(0),
        shared_store_conflict_count=Count(0),
        conflict_per_wavefront_ratio=Ratio(0.0),
        register_per_thread_count=Count(168),
        static_smem_bytes=Bytes(0),
        dynamic_smem_bytes=Bytes(65536),
        # Over the occupancy bar, and 252 blocks is over the 168-block floor, so
        # the fixture is clean on every rule and each test fails the one it names.
        theoretical_occupancy_pct=Percent(75.0),
        achieved_occupancy_pct=Percent(62.5),
        tensor_pipe_pct=Percent(0.0),
        inst_count=Count(1 << 22),
        active_thread_per_warp_ratio=Ratio(32.0),
        block_count=Count(252),
        thread_per_block_count=Count(256),
        wave_per_sm_ratio=Ratio(3.0),
        issue_active_pct=Percent(11.0),
        dominant_stall="long_scoreboard",
        dominant_stall_pct=Percent(62.0),
        stall_barrier_pct=Percent(0.5),
        stall_branch_resolving_pct=Percent(0.25),
        stall_dispatch_stall_pct=Percent(0.1),
        stall_drain_pct=Percent(0.05),
        stall_imc_miss_pct=Percent(0.2),
        stall_lg_throttle_pct=Percent(0.3),
        stall_long_scoreboard_pct=Percent(62.0),
        stall_math_pipe_throttle_pct=Percent(1.0),
        stall_membar_pct=Percent(0.05),
        stall_mio_throttle_pct=Percent(2.5),
        stall_misc_pct=Percent(0.4),
        stall_no_instruction_pct=Percent(1.25),
        stall_not_selected_pct=Percent(3.0),
        stall_short_scoreboard_pct=Percent(4.5),
        stall_sleeping_pct=Percent(0.0),
        stall_tex_throttle_pct=Percent(0.0),
        stall_wait_pct=Percent(5.5),
        sm_pct=Percent(15.0),
        memory_pct=Percent(41.0),
        l1tex_pct=Percent(28.0),
        l2_pct=Percent(24.0),
    )


def audit_one(
    spill: SpillCounters | None,
    *,
    step_duration_us: float = 5000.0,
    capture_iters: int = 3,
) -> FloorAudit:
    """Audit one kernel against the synthetic floor.

    Args:
        spill: The kernel's spill record, or None to run the audit without one.
        step_duration_us: Measured per-iteration wall, the SERIAL-tiny divisor.
        capture_iters: Iterations the capture window contained.

    Returns:
        The audit.
    """
    return floor_audit(
        (counters(),),
        floor=synthetic_floor(*SWEEP),
        spills=() if spill is None else (spill,),
        step_duration_us=Microseconds(step_duration_us),
        capture_iters=capture_iters,
        device=device_record(),
    )


def test_a_spill_fails_the_class_whatever_the_percentage_says() -> None:
    """One record, one counter changed, two verdicts.

    The percentage is identical in both, which is the point: a spill is a defect of
    the same kind as a bank conflict, and it is failed on its own rather than left
    to a bar the spilled traffic itself moves.
    """
    clean = audit_one(spill_record(0))
    dirty = audit_one(spill_record(245_760))
    assert clean.spilled == ()
    assert dirty.spilled == (OWNED,)
    assert clean.verdicts[0].achieved_pct == dirty.verdicts[0].achieved_pct
    assert clean.verdicts[0].achieved_pct > CLASS_FLOOR_PCT[DRAM_BOUND]
    assert clean.verdicts[0].passed
    assert not dirty.verdicts[0].passed
    assert not spill_record(0).spilled
    assert spill_record(245_760).spill_sector_count == 3 * 245_760


def audit_traffic(traffic: int, *, sectors: int = 0) -> FloorAudit:
    """Audit one DRAM-bound kernel whose three launches moved ``traffic`` bytes."""
    one = replace(
        counters(),
        dram_read_bytes=Bytes(0),
        dram_write_bytes=Bytes(traffic),
        achieved_gbs=gbs_from_bytes_us(Bytes(traffic), WINDOW_US),
    )
    return floor_audit(
        (one,),
        floor=synthetic_floor(*SWEEP),
        spills=(spill_record(sectors),),
        step_duration_us=Microseconds(5000.0),
        capture_iters=3,
        device=device_record(),
    )


def test_a_kernel_inside_l2_gets_no_bandwidth_verdict() -> None:
    """What the smallest shape reads, where DRAM reads are literally zero.

    Traffic under the cache size is not a lower bound on the work, so the same
    kernel scores anywhere depending on what L2 already held. Withholding the
    verdict is the only honest reading; the previous one divided by a floor
    extrapolated below every swept footprint, and at zero bytes it had none at all.

    The boundary is asserted in both directions, because at-or-below is a choice: a
    launch whose whole traffic equals the cache size could have been served without
    reaching DRAM.
    """
    inside = audit_traffic(3 * L2_BYTES)
    assert inside.cached == (OWNED,)
    assert inside.verdicts == ()
    assert inside.spilled == ()
    assert audit_traffic(3 * (L2_BYTES + 1)).cached == ()
    assert len(audit_traffic(3 * (L2_BYTES + 1)).verdicts) == 1
    # The spill rule does not go quiet with the verdict. There is nothing left to
    # fail, so the record is what carries the defect.
    spilling = audit_traffic(3 * L2_BYTES, sectors=245_760)
    assert (spilling.cached, spilling.spilled) == ((OWNED,), (OWNED,))
    assert spilling.verdicts == ()


def test_the_audit_refuses_a_kernel_with_no_spill_record() -> None:
    """A pass that was never run must not read as a clean kernel.

    This is the failure mode that makes the spill table safe to keep out of
    ``NCU_TABLES``: forgetting it fails every judged kernel loudly instead of
    passing every spilling one silently.
    """
    with pytest.raises(ValueError, match="carries no spill record"):
        floor_audit(
            (counters(),),
            floor=synthetic_floor(*SWEEP),
            spills=(spill_record(0, kernel="kernel_cutlass_swiglu_fwd_kernel_0"),),
            step_duration_us=Microseconds(5000.0),
            capture_iters=3,
            device=device_record(),
        )


def test_the_audit_leaves_a_foreign_kernel_unjudged() -> None:
    """A kernel this repo does not compile needs no spill record and gets no bar."""
    foreign = "void at::native::vectorized_elementwise_kernel<4, ...>(int, ...)"
    audit = floor_audit(
        (replace(counters(), kernel=foreign),),
        floor=synthetic_floor(*SWEEP),
        spills=(),
        step_duration_us=Microseconds(5000.0),
        capture_iters=3,
        device=device_record(),
    )
    assert audit.unjudged == (foreign,)
    assert audit.verdicts == ()
    assert audit.spilled == ()


def test_a_serial_tiny_declaration_is_judged_by_its_share_of_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit's other arc: an upper bound on step share, and no spill record.

    The declaration is patched onto a DRAM-bound kernel rather than read off the one
    real SERIAL-tiny entry, so the arc under test does not move when the table does.
    Both directions come off one record: 250 us over three capture iterations is
    83.3 us, under the 2% bar of a 5,000 us step and over it at 2,000 us.

    No spill record is supplied. SERIAL-tiny is absent from SPILL_FREE_CLASSES,
    because its bar is a share of the step and a spill can only worsen it; a
    spilling SERIAL-tiny kernel is worth fixing, not a corrupted verdict.
    """
    monkeypatch.setitem(DECLARED, "chunk_scan_fwd_kernel", SERIAL_TINY)
    under = audit_one(None)
    one = under.verdicts[0]
    assert (one.kernel, one.declared) == (OWNED, SERIAL_TINY)
    assert one.achieved_pct == pytest.approx(100.0 * (WINDOW_US / 3) / 5000.0)
    assert one.required_pct == CLASS_FLOOR_PCT[SERIAL_TINY]
    assert one.passed
    assert under.spilled == ()
    over = audit_one(None, step_duration_us=2000.0)
    assert over.verdicts[0].achieved_pct == pytest.approx(
        100.0 * (WINDOW_US / 3) / 2000.0
    )
    assert not over.verdicts[0].passed


def test_the_audit_refuses_a_judgement_the_counters_cannot_make(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two refusals, each louder than a verdict resting on nothing."""
    with pytest.raises(ValueError, match="capture_iters must be positive"):
        audit_one(spill_record(0), capture_iters=0)
    # No table collects a flop count, so a TENSOR-bound declaration says so rather
    # than judging a tensor kernel by a bandwidth it was never held to.
    monkeypatch.setitem(DECLARED, "chunk_scan_fwd_kernel", TENSOR_BOUND)
    with pytest.raises(ValueError, match="needs a flop count"):
        audit_one(spill_record(0))


# ---------------------------------------------------------------------------
# The launch-geometry rules
# ---------------------------------------------------------------------------


def audit_launch(one: KernelCounters, *, spill: SpillCounters | None) -> FloorAudit:
    """Audit one kernel at an imposed launch geometry.

    Args:
        one: The counters, geometry included.
        spill: The kernel's spill record, or None for a class that needs none.

    Returns:
        The audit.
    """
    return floor_audit(
        (one,),
        floor=synthetic_floor(*SWEEP),
        spills=() if spill is None else (spill,),
        step_duration_us=Microseconds(5000.0),
        capture_iters=3,
        device=device_record(),
    )


def test_the_occupancy_rule_fails_a_kernel_whose_class_floor_passes() -> None:
    """One record, one counter changed, and the class verdict identical in both.

    The two rules are independent and this is the direction that matters: a kernel
    reaches a share of its own DRAM floor with whatever occupancy it happens to
    have, so no bandwidth figure can fail a launch that leaves the multiprocessor
    nothing to hide with. That is why the bar is not in ``CLASS_FLOOR_PCT``.
    """
    over = audit_one(spill_record(0))
    under = audit_launch(
        replace(counters(), achieved_occupancy_pct=Percent(8.33)),
        spill=spill_record(0),
    )
    assert over.geometry[0].passed
    assert over.passed
    assert under.verdicts[0].passed
    assert under.verdicts[0].achieved_pct == over.verdicts[0].achieved_pct
    assert not under.passed
    one = under.geometry[0]
    assert (one.kernel, one.declared) == (OWNED, DRAM_BOUND)
    assert one.required_occupancy_pct == MIN_OCCUPANCY_PCT
    assert not one.occupancy_passed
    assert one.block_floor_passed
    # Which rule, by how much, at what geometry. All three in the verdict, and the
    # audit's failure line carries it verbatim.
    assert "achieved occupancy 8.33% is 41.67 points under the 50.0% bar" in one.detail
    assert "252 blocks x 256 threads, theoretical occupancy 75.00%" in one.detail
    assert under.failures == (f"{OWNED}: {one.detail}",)


def test_the_block_floor_is_waived_by_serial_tiny_and_by_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One 128-block launch, judged under two declarations.

    The floor is twice the multiprocessor count the part reports, so it comes off
    the device record rather than out of a constant, and a part with a different SM
    count moves it. The exemption is on this rule only: a kernel provably serial in
    its own extent has no blocks to add, and a waived floor says so rather than
    reading as a count that cleared.
    """
    thin = replace(counters(), block_count=Count(128))
    failed = audit_launch(thin, spill=spill_record(0)).geometry[0]
    assert failed.block_floor_count == 2 * SM_COUNT
    assert not failed.block_floor_exempt
    assert not failed.block_floor_passed
    assert failed.occupancy_passed
    assert not failed.passed
    assert "128 blocks is 40 under the 168-block floor" in failed.detail
    assert "twice the 84 multiprocessors" in failed.detail
    monkeypatch.setitem(DECLARED, "chunk_scan_fwd_kernel", SERIAL_TINY)
    waived = audit_launch(thin, spill=None)
    one = waived.geometry[0]
    assert one.block_floor_exempt
    assert one.block_floor_passed
    assert one.passed
    assert "the 168-block floor is waived by SERIAL-tiny" in one.detail
    assert waived.passed


def test_a_kernel_with_no_bandwidth_verdict_still_carries_its_geometry() -> None:
    """The geometry rules do not go quiet with the class floor.

    Traffic inside L2 withholds the bandwidth verdict, and the grid and the warp
    census are readable off the launch whatever the traffic did. A geometry verdict
    that disappeared with the bandwidth one would leave a launch unjudged at exactly
    the shape where it is cheapest to profile.
    """
    inside = audit_traffic(3 * L2_BYTES)
    assert inside.cached == (OWNED,)
    assert inside.verdicts == ()
    assert len(inside.geometry) == 1
    assert inside.geometry[0].kernel == OWNED
    assert inside.geometry[0].passed


def test_a_geometry_verdict_refuses_a_launch_extent_that_did_not_happen() -> None:
    """A zero extent is a broken profile, not a geometry failure.

    Reported as one, it would read as a kernel launching too few blocks, which is a
    defect a lane would then go looking for in a kernel that ran fine.
    """
    for blocks, threads in ((Count(0), Count(256)), (Count(168), Count(0))):
        with pytest.raises(ValueError, match="launch extent must be positive"):
            geometry_verdict(
                OWNED,
                declared=DRAM_BOUND,
                block_count=blocks,
                thread_per_block_count=threads,
                achieved_occupancy_pct=Percent(60.0),
                theoretical_occupancy_pct=Percent(75.0),
                device=device_record(),
            )


def test_every_rule_a_kernel_broke_reaches_a_failure_line_of_its_own() -> None:
    """Three rules broken at once, three lines, in rule order.

    The exit status a driver returns is this tuple being empty, so a rule whose
    failure does not reach it is a rule with no gate behind it. The two geometry
    rules share one line because one launch configuration decides both and the line
    names each of them.
    """
    broken = replace(
        counters(),
        duration_us=Microseconds(4 * WINDOW_US),
        block_count=Count(128),
        achieved_occupancy_pct=Percent(8.33),
    )
    audit = audit_launch(broken, spill=spill_record(245_760))
    assert not audit.passed
    assert len(audit.failures) == 3
    spill, floor, geometry = audit.failures
    assert "touched local memory" in spill
    assert f"{DRAM_BOUND} reached 23.62% against the 85.0% bar" in floor
    assert "achieved occupancy 8.33%" in geometry
    assert "128 blocks is 40 under" in geometry
