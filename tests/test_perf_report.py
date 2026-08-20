"""Report emission, serialization, and the cross-check that gates both.

Every record is built from literal samples, so nothing here needs a profiler, a
trace file, or a GPU. The point of the file is the refusal path: a report whose
clocks disagree must not reach a file, because a stale report that survives a
failed run is indistinguishable from a fresh pass.

:mod:`slinoss.perf.declared` is covered here as well: it is the only producer of
the verdict record this module renders, and both take the same fabricated
counters and ceilings.

``_row`` and ``_table`` are imported directly. ``_row``'s non-dataclass early
return is unreachable from :func:`markdown`, which only ever hands it dataclasses,
and ``_table``'s one-shape rule is enforced at every call site so no report can
reach it with a mixture.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated

import pytest

from slinoss.perf.budget import BucketDelta, BucketTiming, BudgetReport
from slinoss.perf.ceiling import (
    DRAM_BOUND,
    SERIAL_TINY,
    Ceilings,
    ClassVerdict,
    DramCeiling,
    TensorCeiling,
)
from slinoss.perf.declared import DECLARED, declared_class
from slinoss.perf.device import ClockPolicy, Contention, DeviceInfo
from slinoss.perf.dispersion import (
    GrowthRow,
    PairedRow,
    RepeatRow,
    growth,
    paired,
    repeats,
)
from slinoss.perf.memory import (
    MemoryPeaks,
    PoolRetention,
    RegionSaved,
    SavedStorages,
)
from slinoss.perf.ncu import KernelCounters
from slinoss.perf.nsys import NsysKernel, NsysTrace
from slinoss.perf.report import (
    TOLERANCE_PCT,
    Agreement,
    AgreementError,
    Report,
    _row,
    _table,
    agreement,
    json_text,
    markdown,
    payload,
    rate_table,
    write_report,
)
from slinoss.perf.timing import Throughput
from slinoss.perf.units import (
    MODELLED,
    SUM,
    Bytes,
    Count,
    GBPerSecond,
    Mebibytes,
    Megahertz,
    Microseconds,
    Percent,
    PerfRecord,
    Ratio,
    Spread,
    TFlopsPerSecond,
)

CAPTURE_ITERS = 3
"""Iterations the fabricated capture window contains. Both profiler sums cover
all three, so a per-iteration figure is a third of them."""

CUTE_SCAN = "kernel_cutlass_chunk_scan_fwd_kernel_bf16_0"
CUTE_STATE = "kernel_cutlass_state_passing_fwd_kernel_bf16_0"
CONV_FWD = "void slinoss::(anonymous namespace)::conv1d_fwd_kernel<c10::BFloat16>"
FOREIGN = "void at::native::vectorized_elementwise_kernel<4>"
"""Kernel symbols in the mangled form NCU reports them, one per declaration arc:
two CuTe DSL kernels, the compiled extension, and a kernel from torch that this
repo does not compile.

The test that exercises the SERIAL-tiny arc patches the class onto one of these
symbols rather than reaching for the entry that declares it. Binding the arc to
whichever kernel currently declares the class would make an unrelated
reclassification break a test about the audit."""


# ---------------------------------------------------------------------------
# Record builders. Literal samples only.
# ---------------------------------------------------------------------------


def _us(*values: float) -> list[Microseconds]:
    """Microsecond samples from raw floats."""
    return [Microseconds(v) for v in values]


def _header_under(text: str, title: str) -> str:
    """The header row of the table under one markdown section."""
    return text.split(f"## {title}\n\n")[1].splitlines()[0]


def _spread(median_us: float) -> Spread:
    """Three samples one percent either side of ``median_us``.

    The middle sample is the median exactly, so every figure derived from it is
    exact: the range is 2 percent and the floor is 1 percent of the median.
    """
    half = median_us / 100.0
    return Spread.of(
        [
            Microseconds(median_us - half),
            Microseconds(median_us),
            Microseconds(median_us + half),
        ]
    )


def _growth() -> tuple[GrowthRow, ...]:
    """Two prefixes of one run, the second long enough to resolve anything."""
    return growth(_us(100.0, 104.0, 96.0, 300.0, 101.0, 99.0), 3)


def _scatter() -> RepeatRow:
    """Two independent runs of identical work, one microsecond apart."""
    return repeats("step", (Spread.of(_us(99.0, 100.0, 101.0)), _spread(101.0)))


def _comparison() -> PairedRow:
    """Six pairs, the arm under test faster by 10 us in every one of them."""
    base = _us(100.0, 104.0, 96.0, 300.0, 101.0, 99.0)
    return paired("scan", "reference", base, "cute", _us(*(v - 10.0 for v in base)))


def _device() -> DeviceInfo:
    return DeviceInfo(
        name="Test Part",
        capability="8.6",
        sm_count=Count(84),
        warp_thread_count=Count(32),
        max_threads_per_sm_count=Count(1536),
        regs_per_sm_count=Count(65536),
        smem_per_block_bytes=Bytes(49152),
        smem_optin_per_block_bytes=Bytes(101376),
        smem_per_sm_bytes=Bytes(102400),
        l2_bytes=Bytes(6291456),
        total_memory_bytes=Bytes(51041271808),
        clocks=ClockPolicy(
            locked=False,
            sm_clock_mhz=Megahertz(1740.0),
            max_sm_clock_mhz=Megahertz(1800.0),
            detail="fabricated",
        ),
        sharing=Contention(
            probed=True,
            foreign_process_count=Count(1),
            foreign_memory_mib=Mebibytes(36918.0),
            utilization_pct=Percent(100.0),
            detail="fabricated",
        ),
    )


def _counters(kernel: str = "scan", duration_us: float = 3030.0) -> KernelCounters:
    return KernelCounters(
        kernel=kernel,
        launch_count=Count(CAPTURE_ITERS),
        duration_us=Microseconds(duration_us),
        pass_duration_spread_pct=Percent(0.4),
        dram_read_bytes=Bytes(1 << 24),
        dram_write_bytes=Bytes(1 << 23),
        dram_pct=Percent(88.0),
        achieved_gbs=GBPerSecond(760.0),
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
        register_per_thread_count=Count(96),
        static_smem_bytes=Bytes(0),
        dynamic_smem_bytes=Bytes(65536),
        theoretical_occupancy_pct=Percent(50.0),
        achieved_occupancy_pct=Percent(46.0),
        tensor_pipe_pct=Percent(12.0),
        inst_count=Count(1 << 22),
        active_thread_per_warp_ratio=Ratio(32.0),
        block_count=Count(168),
        thread_per_block_count=Count(256),
        wave_per_sm_ratio=Ratio(2.0),
        issue_active_pct=Percent(8.5),
        dominant_stall="long_scoreboard",
        dominant_stall_pct=Percent(74.0),
        stall_barrier_pct=Percent(0.5),
        stall_branch_resolving_pct=Percent(0.25),
        stall_dispatch_stall_pct=Percent(0.1),
        stall_drain_pct=Percent(0.05),
        stall_imc_miss_pct=Percent(0.2),
        stall_lg_throttle_pct=Percent(0.3),
        stall_long_scoreboard_pct=Percent(74.0),
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


def _trace(
    *,
    kernel_sum_us: float = 3000.0,
    memcpy_us: float = 0.0,
    memset_us: float = 0.0,
    with_kernels: bool = True,
) -> NsysTrace:
    kernels = (
        NsysKernel(
            kernel="scan",
            launch_count=Count(CAPTURE_ITERS),
            duration_us=Microseconds(kernel_sum_us),
            duration=_spread(kernel_sum_us / CAPTURE_ITERS),
            share_pct=Percent(100.0),
        ),
    )
    return NsysTrace(
        label="step",
        report_path="fabricated.nsys-rep",
        kernel_sum_duration_us=Microseconds(kernel_sum_us),
        memcpy_sum_duration_us=Microseconds(memcpy_us),
        memset_sum_duration_us=Microseconds(memset_us),
        memcpy_count=Count(1 if memcpy_us else 0),
        memset_count=Count(1 if memset_us else 0),
        kernels=kernels if with_kernels else (),
    )


def _agreement(
    *,
    event_us: float = 1200.0,
    kernel_sum_us: float = 3000.0,
    ncu_us: float = 3030.0,
    memcpy_us: float = 0.0,
    memset_us: float = 0.0,
    tolerance_pct: Percent = TOLERANCE_PCT,
) -> Agreement:
    return agreement(
        "step",
        event=_spread(event_us),
        trace=_trace(
            kernel_sum_us=kernel_sum_us, memcpy_us=memcpy_us, memset_us=memset_us
        ),
        kernels=(_counters(duration_us=ncu_us),),
        capture_iters=CAPTURE_ITERS,
        tolerance_pct=tolerance_pct,
    )


def _budget() -> BudgetReport:
    return BudgetReport(
        label="step",
        clocks="unlocked",
        total=_spread(1200.0),
        buckets=(
            BucketTiming(
                label="forward",
                parent="",
                derived=False,
                median_duration_us=Microseconds(800.0),
                spread_pct=Percent(1.5),
                resolution_pct=Percent(0.75),
                coverage_pct=Percent(95.7),
                sample_count=Count(30),
                share_of_parent_pct=Percent(100.0),
                share_of_total_pct=Percent(66.667),
            ),
        ),
    )


def _ceilings() -> Ceilings:
    return Ceilings(
        device=_device(),
        dram=DramCeiling(
            label="device-to-device copy, 512 MiB per buffer",
            moved_bytes=Bytes(1073741824),
            duration=_spread(1400.0),
            achieved_gbs=GBPerSecond(767.0),
        ),
        tensor=TensorCeiling(
            label="8192x8192x8192 torch.bfloat16 gemm",
            flop_count=Count(1099511627776),
            duration=_spread(4200.0),
            achieved_tflops=TFlopsPerSecond(261.8),
        ),
    )


def _saved(*, with_regions: bool = True) -> SavedStorages:
    regions = (
        RegionSaved(
            label="forward.scan",
            storage_count=Count(2),
            save_event_count=Count(3),
            saved_bytes=Bytes(2097152),
        ),
    )
    return SavedStorages(
        label="step",
        storage_count=Count(2),
        save_event_count=Count(3),
        saved_bytes=Bytes(2097152),
        input_bytes=Bytes(1048576),
        derived_bytes=Bytes(1048576),
        regions=regions if with_regions else (),
    )


def _report(
    *,
    title: str = "full",
    check: Agreement | None = None,
    everything: bool = True,
    with_regions: bool = True,
) -> Report:
    if not everything:
        return Report(title=title, device=_device(), agreement=check)
    return Report(
        title=title,
        device=_device(),
        agreement=check,
        budget=_budget(),
        throughput=(Throughput.of("prefill", Count(4096), _spread(1200.0)),),
        ceilings=_ceilings(),
        kernels=(_counters(),),
        trace=_trace(),
        saved=_saved(with_regions=with_regions),
        peaks=MemoryPeaks(
            label="step",
            peak_allocated_bytes=Bytes(268435456),
            peak_reserved_bytes=Bytes(335544320),
        ),
        pool=PoolRetention(
            label="step",
            layout_count=Count(12),
            descriptor_count=Count(19),
            retained_bytes=Bytes(14712832),
        ),
        verdicts=(
            ClassVerdict(
                kernel="scan",
                declared=DRAM_BOUND,
                achieved_pct=Percent(90.0),
                required_pct=Percent(85.0),
                passed=True,
            ),
            ClassVerdict(
                kernel="norm",
                declared=SERIAL_TINY,
                achieved_pct=Percent(1.2),
                required_pct=Percent(2.0),
                passed=True,
            ),
        ),
        deltas=(
            BucketDelta(
                label="forward.scan",
                before_duration_us=Microseconds(900.0),
                after_duration_us=Microseconds(800.0),
                delta_pct=Percent(-11.111),
                speedup_ratio=Ratio(1.125),
                floor_pct=Percent(3.0),
                resolved=True,
            ),
        ),
        growth=_growth(),
        scatter=_scatter(),
        comparisons=(_comparison(),),
        notes=("clocks unlocked; the resolution floor bounds every claim",),
    )


@dataclass(frozen=True)
class _Modelled(PerfRecord):
    """A modelled field, since no shipped record carries one.

    Attributes:
        label: What is modelled.
        est_traffic_bytes: Analytic byte count.
    """

    label: str
    est_traffic_bytes: Annotated[Bytes, MODELLED, SUM]


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------


def test_agreement_passes_when_both_checks_hold() -> None:
    check = _agreement()
    assert check.agrees
    assert check.detail == "ncu and nsys agree; event wall covers the device sum"
    assert check.tolerance_pct == 5.0
    assert check.capture_iter_count == CAPTURE_ITERS
    # The gap is the event wall over the device sum, not an absorbed residue.
    assert check.kernel_delta_pct == pytest.approx(1.0)
    assert check.gap_pct == pytest.approx(100.0 * 200.0 / 1200.0)


def test_agreement_divides_both_profiler_sums_by_the_capture_iters() -> None:
    # Both sums cover three iterations; all three reported figures are per
    # iteration, so they are directly comparable. The device sum counts copies
    # and fills, the kernel sum does not.
    check = _agreement(
        kernel_sum_us=3000.0, ncu_us=3030.0, memcpy_us=300.0, memset_us=150.0
    )
    assert check.nsys_kernel_sum_duration_us == 1000.0
    assert check.nsys_device_sum_duration_us == 1150.0
    assert check.ncu_kernel_sum_duration_us == 1010.0
    assert check.event_duration_us == 1200.0


def test_agreement_fails_on_a_kernel_sum_disagreement() -> None:
    check = _agreement(kernel_sum_us=3000.0, ncu_us=3300.0)
    assert not check.agrees
    assert check.kernel_delta_pct == pytest.approx(10.0)
    # The detail names both sums, so the failure is diagnosable from the message.
    assert "1100.000" in check.detail
    assert "1000.000" in check.detail
    assert "10.00%" in check.detail
    # The bar is a parameter, and the same disagreement passes a wider one.
    wide = _agreement(ncu_us=3300.0, tolerance_pct=Percent(20.0))
    assert wide.agrees
    assert wide.tolerance_pct == 20.0


def test_agreement_fails_when_the_wall_is_below_the_device_sum() -> None:
    check = _agreement(event_us=900.0, kernel_sum_us=3000.0, ncu_us=3000.0)
    assert not check.agrees
    assert check.gap_pct < 0.0
    assert "event wall 900.000 us is below the nsys device sum 1000.000 us" in (
        check.detail
    )
    # 2% short of the device sum, inside the 5% bar: a timeline this close is
    # clock skew, not an impossible ordering.
    skewed = _agreement(event_us=1000.0, kernel_sum_us=3060.0, ncu_us=3060.0)
    assert skewed.gap_pct == pytest.approx(-2.0)
    assert skewed.agrees


def test_agreement_rejects_a_check_it_cannot_take() -> None:
    with pytest.raises(ValueError, match="capture_iters must be positive"):
        agreement(
            "step",
            event=_spread(1200.0),
            trace=_trace(),
            kernels=(_counters(),),
            capture_iters=0,
        )
    with pytest.raises(ValueError, match="at least one NCU kernel"):
        agreement(
            "step",
            event=_spread(1200.0),
            trace=_trace(),
            kernels=(),
            capture_iters=CAPTURE_ITERS,
        )


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def test_markdown_refuses_to_render_without_agreeing_clocks() -> None:
    report = _report(check=_agreement(ncu_us=3300.0))
    with pytest.raises(AgreementError, match=r"clocks disagree beyond 5\.0%"):
        markdown(report)
    with pytest.raises(AgreementError, match="no CUDA-event / NSYS / NCU cross-check"):
        markdown(_report(check=None))


def test_markdown_header_carries_the_device_evidence() -> None:
    text = markdown(_report(check=_agreement()))
    assert text.startswith("# full\n")
    assert "- device: Test Part, capability 8.6, 84 SM" in text
    assert "- clocks: unlocked" in text
    # A competitor on the device moves a median further than any change under
    # test, so the header states it beside the clock stamp rather than leaving it
    # in the JSON for nobody to read.
    assert (
        "- sharing: shared with 1 process holding 36,918 MiB at 100% utilization"
        in text
    )
    assert "- smem opt-in per block: 101,376 bytes" in text
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_markdown_renders_every_present_section() -> None:
    text = markdown(_report(check=_agreement()))
    for title in (
        "## cross-check",
        "## budget",
        "## throughput",
        "## measured dram ceiling",
        "## measured tensor ceiling",
        "## class verdicts",
        "## kernel counters",
        "## warp stalls",
        "## speed of light",
        "## gpu trace",
        "## saved tensors",
        "## memory peaks",
        "## descriptor pool",
        "## bucket deltas",
        "## dispersion against sample count",
        "## run-to-run median scatter",
        "## paired comparisons",
        "## notes",
    ):
        assert title in text
    assert "- total_duration_us: 1,200.000" in text
    assert "- kernel_sum_duration_us: 3,000.000" in text
    assert "over 0 copies" in text
    assert "over 0 fills" in text
    assert "- clocks unlocked; the resolution floor bounds every claim" in text
    # Both ceilings print in full. One shared table would take its headers from
    # the DRAM row and drop every tensor field.
    for field in ("moved_bytes", "achieved_gbs", "flop_count", "achieved_tflops"):
        assert field in text


def test_markdown_budget_header_carries_the_floor_beside_the_range() -> None:
    text = markdown(_report(check=_agreement()))
    header = text.split("## budget\n\n")[1].split("\n\n")[0].splitlines()
    # The floor sits between the range and the count it was derived from, so a
    # reader cannot take the range for the bound. Its coverage follows it: three
    # samples put the median's interval at 75 percent, well under nominal, and the
    # floor is worth nothing without that figure beside it.
    assert header == [
        "- total_duration_us: 1,200.000",
        "- spread_pct: 2.000",
        "- resolution_pct: 1.000",
        "- coverage_pct: 75.000",
        "- sample_count: 3",
    ]


def test_markdown_renders_the_dispersion_sections() -> None:
    text = markdown(_report(check=_agreement()))
    rows = _header_under(text, "dispersion against sample count")
    assert "sample_count" in rows
    assert "spread_pct" in rows
    assert "resolution_pct" in rows
    # Every floor prints beside the coverage of the interval it came from, in both
    # sections. A floor alone reads as though its interval reached nominal.
    assert "coverage_pct" in rows
    assert "resolves" in rows
    scatter = _header_under(text, "run-to-run median scatter")
    assert "scatter_pct" in scatter
    assert "floor_pct" in scatter
    assert "coverage_pct" in scatter
    assert "floor_holds" in scatter
    # The paired section is the only one whose verdict survives a comparison
    # between two runs, so it carries the interval it was judged on.
    pair = _header_under(text, "paired comparisons")
    assert "delta_median_duration_us" in pair
    assert "delta_low_duration_us" in pair
    assert "delta_high_duration_us" in pair
    assert "coverage_pct" in pair
    assert "resolves" in pair


def test_markdown_omits_absent_sections_and_never_prints_them_as_zero() -> None:
    text = markdown(
        _report(title="minimal", check=None, everything=False), require_agreement=False
    )
    # A single-clock measurement says so in the header instead of omitting the
    # line, so a report cannot read as cross-checked by silence.
    assert "- cross-check: not run" in text
    for title in (
        "## cross-check",
        "## budget",
        "## throughput",
        "## measured dram ceiling",
        "## measured tensor ceiling",
        "## class verdicts",
        "## kernel counters",
        "## warp stalls",
        "## speed of light",
        "## gpu trace",
        "## saved tensors",
        "## memory peaks",
        "## descriptor pool",
        "## bucket deltas",
        "## dispersion against sample count",
        "## run-to-run median scatter",
        "## paired comparisons",
        "## notes",
    ):
        assert title not in text
    for field in (
        "total_duration_us",
        "kernel_sum_duration_us",
        "peak_allocated_bytes",
        "retained_bytes",
        "achieved_gbs",
        "saved_bytes",
        "resolution_pct",
        "coverage_pct",
        "scatter_pct",
        "resolves",
    ):
        assert field not in text
    assert "0.000" not in text


def test_markdown_formats_a_cell_by_type_and_dots_into_a_nested_record() -> None:
    text = markdown(_report(check=_agreement()))
    assert "| yes |" in text  # bool
    assert "1,073,741,824" in text  # int, grouped
    assert "1,200.000" in text  # float, three places
    assert "| scan |" in text  # str, verbatim
    # NsysKernel.duration is a nested Spread, so the leaf keeps its own suffix.
    assert "duration.median_duration_us" in text
    # SavedStorages.regions is a table of its own, never a column.
    header = _header_under(text, "saved tensors")
    assert "derived_bytes" in header
    assert "regions" not in header
    # Spread.samples_duration_us is a tuple, so it reaches the JSON and no row.
    assert "samples_duration_us" not in text


def test_the_counters_print_as_three_tables_and_no_field_is_dropped() -> None:
    """Every counter field lands in exactly one per-kernel table.

    A stall percentage, a bandwidth, and a unit utilization answer three
    questions, and one table of every field is fifty columns wide. The split is by
    column, so the union of the three headers is the whole record and only the two
    that identify a row repeat.
    """
    text = markdown(_report(check=_agreement()))
    counters = _header_under(text, "kernel counters").strip("| ").split(" | ")
    stalls = _header_under(text, "warp stalls").strip("| ").split(" | ")
    sol = _header_under(text, "speed of light").strip("| ").split(" | ")
    assert set(counters) | set(stalls) | set(sol) == {
        f.name for f in fields(KernelCounters)
    }
    identity = {"kernel", "duration_us"}
    assert set(counters) & set(stalls) == identity
    assert set(counters) & set(sol) == identity
    assert set(stalls) & set(sol) == identity
    # The distinction the split exists for: the issue rate and the dominant stall
    # sit together, away from the DRAM figures they contradict.
    assert "issue_active_pct" in stalls
    assert "stall_long_scoreboard_pct" in stalls
    assert "dram_pct" in counters
    assert "memory_pct" in sol


def test_markdown_prints_none_for_an_empty_table() -> None:
    text = markdown(_report(check=_agreement(), with_regions=False))
    assert "(none)" in text


# ---------------------------------------------------------------------------
# _row
# ---------------------------------------------------------------------------


def test_row_of_a_non_record_is_empty() -> None:
    assert _row("not a record") == {}
    assert _row(Agreement) == {}


# ---------------------------------------------------------------------------
# _table
# ---------------------------------------------------------------------------


def test_table_rejects_two_record_shapes() -> None:
    with pytest.raises(ValueError, match="one table takes one record shape"):
        _table([_ceilings().dram, _ceilings().tensor])


def test_table_rejects_a_column_the_records_do_not_have() -> None:
    # A projection that silently dropped an unknown column would print a table
    # missing whichever field was renamed under it.
    with pytest.raises(KeyError, match="dram_gbs"):
        _table([_counters()], ("kernel", "dram_gbs"))


# ---------------------------------------------------------------------------
# payload and json_text
# ---------------------------------------------------------------------------


def test_payload_keeps_every_field_name_verbatim() -> None:
    # A unit suffix and an `est_` prefix are the only thing separating a measured
    # figure from a modelled one, so serialization may not rewrite either.
    data = payload(_report(check=_agreement()))
    assert "nsys_kernel_sum_duration_us" in data["agreement"]
    assert "peak_allocated_bytes" in data["peaks"]
    assert "retained_bytes" in data["pool"]
    assert payload(_Modelled(label="dram traffic", est_traffic_bytes=Bytes(1024))) == {
        "label": "dram traffic",
        "est_traffic_bytes": 1024,
    }


def test_payload_nests_records_and_lists_tuples() -> None:
    data = payload(_report(check=_agreement()))
    assert data["device"]["clocks"]["sm_clock_mhz"] == 1740.0
    assert data["ceilings"]["dram"]["duration"]["median_duration_us"] == 1400.0
    assert isinstance(data["kernels"], list)
    assert isinstance(data["notes"], list)
    assert data["notes"] == ["clocks unlocked; the resolution floor bounds every claim"]


def test_payload_passes_scalars_and_types_through() -> None:
    assert payload(3.5) == 3.5
    assert payload(None) is None
    assert payload("text") == "text"
    assert payload(Agreement) is Agreement
    assert payload({1: (2, 3)}) == {"1": [2, 3]}


def test_json_text_round_trips_in_field_order() -> None:
    report = _report(check=_agreement())
    text = json_text(report)
    assert json.loads(text) == payload(report)
    assert list(json.loads(text))[:3] == ["title", "device", "agreement"]


# ---------------------------------------------------------------------------
# rate_table
# ---------------------------------------------------------------------------


def test_rate_table_prints_every_rate_beside_its_dispersion() -> None:
    rows = [
        (
            "standard/step/mamba-g12",
            Throughput.of("mamba-g12", Count(8192), _spread(1200.0)),
        ),
        (
            "standard/step/so3ssd-auto",
            Throughput.of("so3ssd-auto", Count(8192), _spread(1000.0)),
        ),
    ]
    header, *body = rate_table(rows, width=30).splitlines()
    assert header.split() == [
        "config",
        "duration_us",
        "spread_pct",
        "resolution_pct",
        "coverage_pct",
        "tps",
    ]
    assert [line.split()[0] for line in body] == [name for name, _ in rows]
    # Every rate carries the four dispersion columns, so no driver can print a
    # throughput figure without what says whether a difference in it is real.
    assert body[0].split()[1:] == [
        "1,200.000",
        "2.000",
        "1.000",
        "75.000",
        "6,826,667",
    ]
    # No rates is the header alone, never a blank line that reads as a run.
    empty = rate_table([], width=8).splitlines()
    assert len(empty) == 1
    assert empty[0].startswith("config  ")


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------


def test_write_report_writes_markdown_and_json(tmp_path: Path) -> None:
    md, js = write_report(_report(check=_agreement()), tmp_path / "run")
    assert (md, js) == (tmp_path / "run.md", tmp_path / "run.json")
    assert md.read_text().startswith("# full\n")
    assert json.loads(js.read_text())["title"] == "full"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json", "run.md"]
    nested_md, nested_js = write_report(
        _report(check=_agreement()), tmp_path / "a" / "b" / "run"
    )
    assert nested_md.exists()
    assert nested_js.exists()


def test_two_bases_differing_after_a_dot_do_not_collide(tmp_path: Path) -> None:
    # The suffix is appended to the whole name. Substituting the last one would
    # send both of these to run.md and lose the first report to the second.
    first = write_report(_report(title="first", check=_agreement()), tmp_path / "run.1")
    assert first == (tmp_path / "run.1.md", tmp_path / "run.1.json")
    write_report(_report(title="second", check=_agreement()), tmp_path / "run.2")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "run.1.json",
        "run.1.md",
        "run.2.json",
        "run.2.md",
    ]
    assert (tmp_path / "run.1.md").read_text().startswith("# first\n")
    assert (tmp_path / "run.2.md").read_text().startswith("# second\n")


def test_write_report_accepts_a_single_clock_measurement(tmp_path: Path) -> None:
    md, _js = write_report(
        _report(check=None), tmp_path / "run", require_agreement=False
    )
    assert "- cross-check: not run" in md.read_text()


def test_a_refused_report_leaves_no_file(tmp_path: Path) -> None:
    # The load-bearing rule: a stale report must never be mistaken for a fresh
    # pass, so the refusal happens before anything is written or created, whether
    # the check failed or was never taken.
    with pytest.raises(AgreementError):
        write_report(_report(check=_agreement(ncu_us=3300.0)), tmp_path / "a" / "run")
    with pytest.raises(AgreementError):
        write_report(_report(check=None), tmp_path / "b" / "run")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# declared
#
# The class audit, beside the verdict record it is the only producer of.
# ---------------------------------------------------------------------------


def test_declared_class_reads_the_table_through_a_mangled_symbol() -> None:
    # Neither toolchain emits the source name: the DSL appends a traced signature
    # and the extension wraps the name in a namespace and template arguments.
    assert declared_class(CUTE_SCAN) == DRAM_BOUND
    assert declared_class(CONV_FWD) == DRAM_BOUND
    assert declared_class(CUTE_STATE) == DRAM_BOUND
    # A kernel from torch, cuBLAS, or the driver is not this repo's to declare.
    assert declared_class(FOREIGN) is None


def test_declared_class_refuses_a_symbol_it_cannot_place() -> None:
    with pytest.raises(ValueError, match="declares no class"):
        declared_class("kernel_cutlass_brand_new_fwd_kernel_0")
    with pytest.raises(ValueError, match="one symbol, one class"):
        declared_class("kernel_cutlass_conv1d_fwd_kernel_conv1d_bwd_kernel_0")


def test_the_table_names_every_kernel_in_the_tree_and_nothing_else() -> None:
    """The table against the source, in both directions.

    An undeclared kernel raises only once something profiles it, and a
    declaration that outlived its kernel never raises at all: it reads as
    coverage of a symbol no run can produce. Both are decidable from the source,
    which is where the declaration itself lives.

    Read rather than imported. Importing the kernel modules drags in the CuTe DSL
    and the compiled extension, which is what keeps this module runnable without
    either.
    """
    root = Path(__file__).resolve().parents[1]
    compiled = {
        found.group(1)
        for path in (root / "slinoss").rglob("*.py")
        for found in re.finditer(r"@cute\.kernel\s+def\s+(\w+)", path.read_text())
    } | {
        found.group(1)
        for path in (root / "csrc").rglob("*.c*")
        # A launch-bounds attribute sits between ``void`` and the name, so the
        # first identifier after the return type is not always the kernel's.
        for found in re.finditer(
            r"__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+)",
            path.read_text(),
        )
    }
    assert compiled
    assert set(DECLARED) == compiled
