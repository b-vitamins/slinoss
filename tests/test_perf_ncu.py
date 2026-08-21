"""Nsight Compute driver: units, command, parsing, and the cross-pass merge.

The profiler binary is absent on this host and on the verification fleet, so no
test launches ``ncu``. Every pure function is driven with fixture text held in
this module, and :func:`run_ncu` is exercised through a fake
:func:`subprocess.run` that records its argv.

Every call names the binary by path. A bare name is resolved against PATH and the
CUDA bin directories, so a default here would pass on a host that has the profiler
and raise on one that does not, and the subject is the driver rather than the host.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Final

import pytest

from slinoss.perf.ncu import (
    NCU_TABLES,
    REQUIRED_METRICS,
    SOL_FIELDS,
    SOURCE_TABLE,
    SOURCE_VIEW,
    STALL_FIELDS,
    STALL_REASONS,
    NcuPass,
    NcuTable,
    export_flags,
    import_command,
    kernel_counters,
    lsu_floor_us,
    metric_scale,
    ncu_command,
    parse_ncu_csv,
    parse_rule_csv,
    parse_source_csv,
    pcsamp_metric,
    report_file,
    rules_command,
    run_ncu,
    run_rules,
    run_source,
    stall_field,
    stall_metric,
)
from slinoss.perf.units import Count, Megahertz

DURATION: Final = "gpu__time_duration.sum"
ISSUE: Final = "smsp__issue_active.avg.pct_of_peak_sustained_active"

SOL_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("sm_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("memory_pct", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l1tex_pct", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("l2_pct", "lts__throughput.avg.pct_of_peak_sustained_active"),
)
"""Speed-of-light field paired with the metric it reads."""

TARGET: Final[tuple[str, ...]] = (
    "python3",
    "scripts/perf/profile_target.py",
    "--iters",
    "8",
)

DRAM: Final = next(t for t in NCU_TABLES if t.name == "dram")

CHUNK: Final = "so3ssd_chunk_fwd"
CPASYNC: Final = "so3ssd_state_cpasync_fwd"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


SCALES: Final[tuple[tuple[str, float], ...]] = (
    # Dimensionless, whitespace and NCU's own "nan" cell included.
    ("", 1.0),
    ("%", 1.0),
    ("ratio", 1.0),
    ("nan", 1.0),
    (" % ", 1.0),
    ("  ", 1.0),
    # Durations reach nanoseconds.
    ("nsecond", 1.0),
    ("usecond", 1e3),
    ("msecond", 1e6),
    ("second", 1e9),
    ("ns", 1.0),
    ("us", 1e3),
    ("ms", 1e6),
    # Every counter base, singular and plural.
    ("byte", 1.0),
    ("cycle", 1.0),
    ("inst", 1.0),
    ("sector", 1.0),
    ("request", 1.0),
    ("register", 1.0),
    ("wavefront", 1.0),
    ("warp", 1.0),
    ("block", 1.0),
    ("thread", 1.0),
    ("bytes", 1.0),
    ("cycles", 1.0),
    ("insts", 1.0),
    ("sectors", 1.0),
    ("requests", 1.0),
    ("registers", 1.0),
    ("wavefronts", 1.0),
    ("warps", 1.0),
    ("blocks", 1.0),
    ("threads", 1.0),
    # SI prefixes, on any base and on a plural.
    ("Kbyte", 1e3),
    ("Mbyte", 1e6),
    ("Gbyte", 1e9),
    ("Tbyte", 1e12),
    ("Kbytes", 1e3),
    ("Ksector", 1e3),
    ("Minst", 1e6),
    ("Gcycle", 1e9),
    # A rate scales by its numerator. NCU reports bandwidth as Gbyte/s, and
    # scaling by the numerator prefix takes the cell to byte/s; the denominator is
    # part of the metric's meaning, not of its scale.
    ("byte/s", 1.0),
    ("Gbyte/s", 1e9),
    ("Kbyte/block", 1e3),
    ("register/thread", 1.0),
    ("sector/request", 1.0),
)
"""Every unit NCU has been seen to print, and the factor taking it to base."""


def test_metric_scale_converts_every_unit_ncu_prints() -> None:
    for unit, want in SCALES:
        assert metric_scale(unit) == want, unit


def test_metric_scale_rejects_unknown() -> None:
    # A raise is the whole point. Reading an Mbyte as a byte is a 10^6 error in
    # every bandwidth figure derived from it, and a default of 1.0 hides it.
    for unit in (
        "furlong",
        "mbyte",  # a prefix outside K/M/G/T is not a prefix
        "Xbyte",
        "s",
        "seconds",  # the plural fallback covers counter bases, not time units
    ):
        with pytest.raises(ValueError, match="unknown ncu metric unit"):
            metric_scale(unit)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_every_table_rereads_the_duration_and_their_union_is_required() -> None:
    # Counters from two passes describe two executions. The duration is the only
    # metric they share, so it is the only cross-check on replay stability.
    want: list[str] = []
    for table in NCU_TABLES:
        assert DURATION in table.metrics
        assert len(table.metrics) > 1
        assert len(set(table.metrics)) == len(table.metrics)
        for metric in table.metrics:
            if metric not in want:
                want.append(metric)
    assert list(REQUIRED_METRICS) == want
    assert REQUIRED_METRICS[0] == DURATION


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def test_ncu_command_flags_in_order() -> None:
    # Neither control flag is optional. Clock locking is denied on the
    # verification fleet, so the profiled clock must be the benchmark's clock;
    # and the default cache control profiles every kernel cold while the
    # benchmark runs warm, which inflates both the duration and the DRAM traffic.
    # The target follows --metrics directly; ncu parses a bare "--" as an empty
    # long option and exits on it.
    assert ncu_command(DRAM, TARGET) == [
        "ncu",
        "--csv",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--metrics",
        "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,"
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "python3",
        "scripts/perf/profile_target.py",
        "--iters",
        "8",
    ]
    got = ncu_command(
        DRAM, TARGET, ncu="/opt/nsight/ncu", extra=("--kernel-name", "so3")
    )
    assert got[0] == "/opt/nsight/ncu"
    # The target is the tail, so extra flags sit immediately in front of it.
    assert got[-len(TARGET) :] == list(TARGET)
    assert got[-len(TARGET) - 2 : -len(TARGET)] == ["--kernel-name", "so3"]


def test_ncu_command_needs_metrics_and_a_target() -> None:
    with pytest.raises(ValueError, match="table 'empty' requests no metrics"):
        ncu_command(NcuTable("empty", ()), TARGET)
    with pytest.raises(ValueError, match="needs a target command"):
        ncu_command(DRAM, ())


# ---------------------------------------------------------------------------
# Fixture text
#
# The column set, the quoting, and the ==PROF== framing are NCU's own. Values in
# the Metric Value column are grouped by thousands and scaled by the unit in the
# Metric Unit column, both of which the parser has to undo.
# ---------------------------------------------------------------------------

HEADER: Final = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
    '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"'
)

DRAM_CSV: Final = f"""==PROF== Connected to process 4711 (/gnu/store/py3-3.11.11/bin/python3)
==PROF== Profiling "so3ssd_chunk_fwd" - 0: 0%....50%....100% - 1 pass
==PROF== Profiling "so3ssd_chunk_fwd" - 1: 0%....50%....100% - 1 pass
==PROF== Profiling "so3ssd_state_cpasync_fwd" - 2: 0%....50%....100% - 1 pass
{HEADER}
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","gpu__time_duration.sum","usecond","41.28"
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__bytes_read.sum","Mbyte","2.5"
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__bytes_write.sum","Kbyte","640"
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","62.5"
"1","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","gpu__time_duration.sum","usecond","43.52"
"1","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__bytes_read.sum","Mbyte","2.5"
"1","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__bytes_write.sum","Kbyte","640"
"1","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","70.0"
"2","4711","python3","host","so3ssd_state_cpasync_fwd","2026-08-19 09:14:04","1","7","","gpu__time_duration.sum","usecond","3.2"
"2","4711","python3","host","so3ssd_state_cpasync_fwd","2026-08-19 09:14:04","1","7","","dram__bytes_read.sum","Kbyte","96"
"2","4711","python3","host","so3ssd_state_cpasync_fwd","2026-08-19 09:14:04","1","7","","dram__bytes_write.sum","Kbyte","32"
"2","4711","python3","host","so3ssd_state_cpasync_fwd","2026-08-19 09:14:04","1","7","","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","12.5"
==PROF== Disconnected from process 4711
"""

SPARSE_CSV: Final = f"""==PROF== Connected to process 4711 (/gnu/store/py3-3.11.11/bin/python3)
{HEADER}
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","gpu__time_duration.sum","usecond","41.28"
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__bytes_read.sum","Mbyte",""
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__bytes_write.sum","Kbyte","n/a"
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","nan"
"","4711","python3","host","","2026-08-19 09:14:03","1","7","","gpu__time_duration.sum","usecond","1.0"
==PROF== Disconnected from process 4711
"""

BAD_UNIT_CSV: Final = f"""{HEADER}
"0","4711","python3","host","so3ssd_chunk_fwd","2026-08-19 09:14:03","1","7","","gpu__time_duration.sum","furlong","41.28"
"""

NO_UNIT_COLUMN_CSV: Final = """"ID","Kernel Name","Metric Name","Metric Value"
"0","so3ssd_chunk_fwd","gpu__time_duration.sum","41.28"
"""

NO_HEADER_CSV: Final = """==PROF== Connected to process 4711 (/gnu/store/py3-3.11.11/bin/python3)
==ERROR== The application returned an error code (11).
==PROF== Disconnected from process 4711
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_ncu_csv_keeps_launch_order_and_scales_by_the_declared_unit() -> None:
    got = parse_ncu_csv(DRAM_CSV, DRAM.metrics, table="dram", command=("ncu", "--csv"))
    assert got.table == "dram"
    assert got.command == ("ncu", "--csv")
    assert [i.launch_id for i in got.invocations] == ["0", "1", "2"]
    assert [i.kernel for i in got.invocations] == [CHUNK, CHUNK, CPASYNC]
    assert got.missing_metrics == ()
    first = got.invocations[0].values
    assert first[DURATION] == pytest.approx(41280.0)  # usecond to nanosecond
    assert first["dram__bytes_read.sum"] == pytest.approx(2.5e6)  # Mbyte to byte
    assert first["dram__bytes_write.sum"] == pytest.approx(640e3)  # Kbyte to byte
    assert first["dram__throughput.avg.pct_of_peak_sustained_elapsed"] == 62.5
    assert got.invocations[2].values["dram__bytes_read.sum"] == pytest.approx(96e3)


def test_parse_ncu_csv_lists_every_metric_no_row_carried() -> None:
    # An empty, non-numeric, or nan value is not a zero, and neither is a metric
    # absent from the output. Both land in missing_metrics, which is the loud
    # failure the module documents.
    absent = parse_ncu_csv(DRAM_CSV, (*DRAM.metrics, "dram__sectors_read.sum"))
    assert absent.missing_metrics == ("dram__sectors_read.sum",)
    got = parse_ncu_csv(SPARSE_CSV, DRAM.metrics)
    assert list(got.invocations[0].values) == [DURATION]
    assert got.missing_metrics == (
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    )
    # A record that names no launch cannot be attributed to one, so the sparse
    # fixture's trailing id-less row is not a launch.
    assert [i.launch_id for i in got.invocations] == ["0"]


def test_parse_ncu_csv_rejects_output_it_cannot_read() -> None:
    with pytest.raises(
        ValueError, match="no CSV header in ncu output for table 'dram'"
    ):
        parse_ncu_csv(NO_HEADER_CSV, DRAM.metrics, table="dram")
    with pytest.raises(ValueError, match="no 'Metric Unit' column"):
        parse_ncu_csv(NO_UNIT_COLUMN_CSV, DRAM.metrics)
    with pytest.raises(ValueError, match="unknown ncu metric unit 'furlong'"):
        parse_ncu_csv(BAD_UNIT_CSV, DRAM.metrics)


# ---------------------------------------------------------------------------
# Counter fixture
#
# Metric name to unit and per-launch display values. A one-element tuple is a
# value the launch configuration fixes; a longer one varies per launch, which is
# what separates a summed counter from a median. so3ssd_state_cpasync_fwd moves
# its operands with cp.async, so the LSU sector and shared-wavefront counters
# read zero for it: that is the guarded-ratio case, not a broken label.
# ---------------------------------------------------------------------------

Fixture = dict[str, tuple[str, tuple[str, ...]]]


def stall_fixture(**by_reason: tuple[str, ...]) -> Fixture:
    """Warp-stall entries for one kernel, one per member of :data:`STALL_REASONS`.

    Args:
        by_reason: Per-launch display values, keyed by stall reason. A reason not
            named reads zero, which is what NCU prints for a stall a kernel never
            hits.

    Returns:
        The entries, keyed by metric name.
    """
    return {
        stall_metric(reason): ("%", by_reason.get(reason, ("0",)))
        for reason in STALL_REASONS
    }


def sol_fixture(*values: tuple[str, ...]) -> Fixture:
    """Speed-of-light entries, one per pair of :data:`SOL_METRICS` in order."""
    return {metric: ("%", v) for (_field, metric), v in zip(SOL_METRICS, values)}


COUNTERS: Final[dict[str, Fixture]] = {
    CHUNK: {
        DURATION: ("usecond", ("41.28", "82.56", "412.8")),
        "launch__grid_size": ("", ("512",)),
        "launch__block_size": ("", ("256",)),
        "launch__waves_per_multiprocessor": ("", ("6.4",)),
        "dram__bytes_read.sum": ("Mbyte", ("2.5",)),
        "dram__bytes_write.sum": ("Kbyte", ("640",)),
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": (
            "%",
            ("62.5", "70", "81.25"),
        ),
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": ("request", ("10,240",)),
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": ("request", ("2,048",)),
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": ("sector", ("40,960",)),
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": ("sector", ("16,384",)),
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": ("", ("8,192",)),
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": ("", ("4,096",)),
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": ("", ("0",)),
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": ("", ("128",)),
        "launch__registers_per_thread": ("register", ("104",)),
        "launch__shared_mem_per_block_static": ("Kbyte", ("48",)),
        "launch__shared_mem_per_block_dynamic": ("byte", ("32768",)),
        "sm__maximum_warps_per_active_cycle_pct": ("%", ("50",)),
        "sm__warps_active.avg.pct_of_peak_sustained_active": ("%", ("41.5",)),
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": (
            "%",
            ("72.5",),
        ),
        "sm__inst_executed.sum": ("inst", ("1,048,576",)),
        "smsp__thread_inst_executed_per_inst_executed.ratio": ("", ("32",)),
        ISSUE: ("%", ("3.5",)),
        # The dominant reason varies per launch, so its field is a median and not
        # a sum of the three.
        **stall_fixture(
            long_scoreboard=("48", "52", "60"),
            wait=("9",),
            short_scoreboard=("6",),
            mio_throttle=("4",),
            not_selected=("2",),
            no_instruction=("1.5",),
        ),
        **sol_fixture(("18.5",), ("44.25",), ("31",), ("27.5",)),
    },
    CPASYNC: {
        DURATION: ("usecond", ("3.2", "4.8")),
        "launch__grid_size": ("", ("108",)),
        "launch__block_size": ("", ("128",)),
        "launch__waves_per_multiprocessor": ("", ("1",)),
        "dram__bytes_read.sum": ("Kbyte", ("96",)),
        "dram__bytes_write.sum": ("Kbyte", ("32",)),
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": ("%", ("12.5", "25")),
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": ("request", ("0",)),
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": ("request", ("0",)),
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": ("sector", ("0",)),
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": ("sector", ("0",)),
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": ("", ("0",)),
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": ("", ("0",)),
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": ("", ("0",)),
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": ("", ("0",)),
        "launch__registers_per_thread": ("register", ("56",)),
        "launch__shared_mem_per_block_static": ("Kbyte", ("0",)),
        "launch__shared_mem_per_block_dynamic": ("Kbyte", ("64",)),
        "sm__maximum_warps_per_active_cycle_pct": ("%", ("75",)),
        "sm__warps_active.avg.pct_of_peak_sustained_active": ("%", ("58.25",)),
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": (
            "%",
            ("0",),
        ),
        "sm__inst_executed.sum": ("inst", ("65,536",)),
        "smsp__thread_inst_executed_per_inst_executed.ratio": ("", ("31.75",)),
        ISSUE: ("%", ("21.5",)),
        # A different dominant reason, so the derivation is a per-kernel maximum
        # and not the family's first entry or the other kernel's answer.
        **stall_fixture(
            wait=("70", "74"),
            long_scoreboard=("5",),
            barrier=("8",),
            drain=("3",),
        ),
        **sol_fixture(("6.75",), ("11.5",), ("4",), ("9.25",)),
    },
}

ALL_METRICS: Final = NcuTable("all", REQUIRED_METRICS)


def record(launch: str, kernel: str, metric: str, unit: str, value: str) -> str:
    """One CSV record, in NCU's column order."""
    return (
        f'"{launch}","4711","python3","host","{kernel}",'
        f'"2026-08-19 09:14:03","1","7","","{metric}","{unit}","{value}"\n'
    )


def ncu_pass(
    table: NcuTable, *, durations: Mapping[str, tuple[str, ...]] | None = None
) -> NcuPass:
    """Parse a synthetic pass for one table, built from :data:`COUNTERS`.

    Args:
        table: Metrics to emit, in table order.
        durations: Per-kernel duration display values replacing the fixture's.
            One pass disagreeing on the duration is what
            ``pass_duration_spread_pct`` measures.

    Returns:
        The parsed pass.
    """
    override = durations or {}
    depth = max(len(m[DURATION][1]) for m in COUNTERS.values())
    text = f"{HEADER}\n"
    launch = 0
    for index in range(depth):
        for kernel, metrics in COUNTERS.items():
            if index >= len(metrics[DURATION][1]):
                continue
            for metric in table.metrics:
                unit, values = metrics[metric]
                if metric == DURATION and kernel in override:
                    values = override[kernel]
                text += record(
                    str(launch), kernel, metric, unit, values[index % len(values)]
                )
            launch += 1
    return parse_ncu_csv(
        text,
        table.metrics,
        table=table.name,
        command=ncu_command(table, TARGET),
    )


def full_passes() -> tuple[NcuPass, ...]:
    """One pass per table, in :data:`NCU_TABLES` order."""
    return tuple(ncu_pass(table) for table in NCU_TABLES)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_counters_merge_every_table() -> None:
    got = kernel_counters(full_passes())
    assert [k.kernel for k in got] == [CHUNK, CPASYNC]
    chunk = got[0]
    assert chunk.launch_count == 3
    # 41.28 + 82.56 + 412.8 usecond, read as nanoseconds and reported as
    # microseconds: a summed counter adds over the launches in a pass.
    assert chunk.duration_us == pytest.approx(536.64)
    assert chunk.dram_read_bytes == 7_500_000
    assert chunk.dram_write_bytes == 1_920_000
    assert chunk.global_load_request_count == 30_720
    assert chunk.global_store_request_count == 6_144
    assert chunk.global_load_sector_count == 122_880
    assert chunk.global_store_sector_count == 49_152
    assert chunk.wavefront_count == 36_864
    assert chunk.shared_load_conflict_count == 0
    assert chunk.shared_store_conflict_count == 384
    assert chunk.inst_count == 3_145_728
    # A metric with no .sum suffix is a rate or a launch property and takes the
    # median of the launches, not their sum: median(62.5, 70, 81.25) is 70.
    assert chunk.dram_pct == 70.0
    assert chunk.theoretical_occupancy_pct == 50.0
    assert chunk.achieved_occupancy_pct == 41.5
    assert chunk.tensor_pipe_pct == 72.5
    assert chunk.active_thread_per_warp_ratio == 32.0
    assert chunk.register_per_thread_count == 104
    assert chunk.static_smem_bytes == 48_000
    assert chunk.dynamic_smem_bytes == 32_768
    assert chunk.block_count == 512
    assert chunk.thread_per_block_count == 256
    assert chunk.wave_per_sm_ratio == 6.4
    assert chunk.achieved_gbs == pytest.approx(9_420_000 / 536_640)
    # Distinct on purpose: the load side asks four sectors per request, which is
    # one full request of four bytes a lane, and the store side asks eight, which
    # is the scattered access this ratio exists to name.
    assert chunk.sector_per_load_request_ratio == pytest.approx(4.0)
    assert chunk.sector_per_store_request_ratio == pytest.approx(8.0)
    assert chunk.conflict_per_wavefront_ratio == pytest.approx(384 / 36_864)
    assert chunk.smem_bytes == 48_000 + 32_768
    # Longest first, so the row that owns the step heads the table.
    assert [k.duration_us for k in got] == sorted(
        (k.duration_us for k in got), reverse=True
    )


def test_counters_take_an_even_median_and_guard_a_zero_denominator() -> None:
    cpasync = kernel_counters(full_passes())[1]
    assert cpasync.launch_count == 2
    assert cpasync.duration_us == pytest.approx(8.0)
    assert cpasync.dram_read_bytes == 192_000
    assert cpasync.dram_write_bytes == 64_000
    assert cpasync.dram_pct == 18.75  # median(12.5, 25) over two launches
    assert cpasync.achieved_gbs == pytest.approx(32.0)
    assert cpasync.inst_count == 131_072
    assert cpasync.active_thread_per_warp_ratio == 31.75
    assert cpasync.smem_bytes == 64_000
    # cp.async moves the operands, so the LSU and shared counters read zero. Their
    # ratios are guarded rather than dividing by that zero.
    assert cpasync.global_load_sector_count == 0
    assert cpasync.global_store_sector_count == 0
    assert cpasync.sector_per_load_request_ratio == 0.0
    assert cpasync.sector_per_store_request_ratio == 0.0
    assert cpasync.wavefront_count == 0
    assert cpasync.conflict_per_wavefront_ratio == 0.0


def fixture_median(kernel: str, metric: str) -> float:
    """The value a per-launch fixture entry must merge to.

    A fixture entry shorter than the launch count repeats, so the median is taken
    over the launches rather than over the entry.
    """
    launches = len(COUNTERS[kernel][DURATION][1])
    _unit, values = COUNTERS[kernel][metric]
    return median(float(values[i % len(values)]) for i in range(launches))


def test_the_stall_and_sol_tables_reach_fields_and_merge_by_median() -> None:
    """Every stall and speed-of-light metric lands in a field, reduced by median.

    Both families are percentages NCU has already normalized, per warp-active
    cycle or per unit peak, so a sum over three launches would report 300% of a
    whole that is per launch. One test over the whole family rather than one per
    metric: the reduction is a property of the normalization, not of the reason.
    """
    chunk, cpasync = kernel_counters(full_passes())
    # Field totality: the reasons plus the issue rate and the derived pair.
    assert len(STALL_FIELDS) == len(STALL_REASONS) + 3
    assert len(SOL_FIELDS) == len(SOL_METRICS)
    for reason in STALL_REASONS:
        metric = stall_metric(reason)
        want = fixture_median(CHUNK, metric)
        assert getattr(chunk, stall_field(reason)) == pytest.approx(want), reason
    # median(48, 52, 60), not their sum and not the largest launch.
    assert chunk.stall_long_scoreboard_pct == 52.0
    assert chunk.issue_active_pct == 3.5
    assert chunk.dominant_stall == "long_scoreboard"
    assert chunk.dominant_stall_pct == 52.0
    for field, metric in SOL_METRICS:
        want = fixture_median(CHUNK, metric)
        assert getattr(chunk, field) == pytest.approx(want), field
    assert chunk.memory_pct == 44.25
    # A second kernel, so the maximum is taken per kernel: median(70, 74) beats
    # the first kernel's dominant reason, which reads 5 here.
    assert cpasync.dominant_stall == "wait"
    assert cpasync.dominant_stall_pct == 72.0
    assert cpasync.stall_long_scoreboard_pct == 5.0
    assert cpasync.issue_active_pct == 21.5
    assert cpasync.l2_pct == 9.25


def test_spread_reports_a_pass_disagreement() -> None:
    for one in kernel_counters(full_passes()):
        assert one.pass_duration_spread_pct == 0.0
    passes = [
        ncu_pass(table, durations={CHUNK: ("50", "100", "500")} if index == 1 else None)
        for index, table in enumerate(NCU_TABLES)
    ]
    chunk = kernel_counters(passes)[0]
    # Seven passes summed 536640 ns; one summed 650000 ns. The median is the base.
    assert chunk.pass_duration_spread_pct == pytest.approx(
        100.0 * (650_000.0 - 536_640.0) / 536_640.0
    )


def test_duration_does_not_depend_on_pass_order() -> None:
    # Eight passes, one of them disagreeing. The duration is the median over the
    # passes, which is the statistic pass_duration_spread_pct is a spread of, so
    # the same eight passes give the same answer in either order.
    disagree = ncu_pass(NCU_TABLES[1], durations={CHUNK: ("50", "100", "500")})
    rest = [ncu_pass(t) for t in NCU_TABLES if t.name != NCU_TABLES[1].name]
    first = kernel_counters([*rest, disagree])[0]
    second = kernel_counters([disagree, *rest])[0]
    assert first.duration_us == pytest.approx(536.64)
    assert second.duration_us == first.duration_us
    assert second.pass_duration_spread_pct == first.pass_duration_spread_pct


def test_counters_reject_a_zero_consensus_duration() -> None:
    # A kernel that launched and took no time is a broken profile. Keeping the
    # zero would make it the fastest kernel in the report and divide by it to get
    # a bandwidth.
    zero = ncu_pass(NCU_TABLES[0], durations={CHUNK: ("0",), CPASYNC: ("0",)})
    with pytest.raises(ValueError, match=r"zero duration for kernel .* over 3 passes"):
        kernel_counters([ncu_pass(ALL_METRICS), zero, zero])


def test_counters_need_every_required_metric() -> None:
    # Filling an absent counter with a zero would report a broken label as a
    # free operation, so one table on its own is an error.
    with pytest.raises(
        ValueError, match=r"missing 41 metrics, first 'dram__bytes_read\.sum'"
    ):
        kernel_counters([ncu_pass(NCU_TABLES[0])])


def test_counters_reject_passes_that_profiled_nothing() -> None:
    # No pass at all and passes that carried no launch are the same failure: there
    # is nothing to report, and an empty table would read as a step with no work.
    empty = NcuPass(table="dram", command=(), invocations=(), missing_metrics=())
    for passes in ((), [empty, empty]):
        with pytest.raises(ValueError, match="no kernel launches in any ncu pass"):
            kernel_counters(passes)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class FakeRun:
    """Stands in for :func:`subprocess.run`, recording argv and replaying results.

    The last result repeats, so one entry covers any number of calls.
    """

    def __init__(self, *results: tuple[int, str, str]) -> None:
        self.results = results
        self.commands: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(
        self, command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        self.kwargs.append(dict(kwargs))
        code, out, err = self.results[
            min(len(self.commands) - 1, len(self.results) - 1)
        ]
        return subprocess.CompletedProcess(list(command), code, out, err)


DIAGNOSTIC: Final = "\n".join(f"==ERROR== diagnostic {i:02d}" for i in range(1, 16))


def test_run_ncu_parses_its_own_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun((0, DRAM_CSV, ""))
    monkeypatch.setattr(subprocess, "run", fake)
    got = run_ncu(DRAM, TARGET, ncu="/opt/nsight/ncu", cwd="/tmp", timeout_s=30.0)
    assert fake.commands == [ncu_command(DRAM, TARGET, ncu="/opt/nsight/ncu")]
    assert fake.kwargs[0]["capture_output"] is True
    assert fake.kwargs[0]["text"] is True
    assert fake.kwargs[0]["check"] is False
    assert fake.kwargs[0]["cwd"] == "/tmp"
    assert fake.kwargs[0]["timeout"] == 30.0
    assert got.table == "dram"
    assert got.command == tuple(fake.commands[0])
    assert len(got.invocations) == 3
    assert got.missing_metrics == ()


def test_run_ncu_raises_with_the_diagnostic_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", FakeRun((2, "", DIAGNOSTIC)))
    with pytest.raises(RuntimeError) as caught:
        run_ncu(DRAM, TARGET, ncu="/opt/nsight/ncu")
    message = str(caught.value)
    assert "ncu table 'dram' exited 2" in message
    assert "diagnostic 15" in message
    assert "diagnostic 04" in message
    assert "diagnostic 03" not in message  # the tail is the last twelve lines
    assert " | " in message
    # NCU prints ERR_NVGPUCTRPERM on stdout, so an empty stderr falls back to it
    # rather than raising with a bare exit code.
    monkeypatch.setattr(
        subprocess, "run", FakeRun((1, "==ERROR== ERR_NVGPUCTRPERM: permission", ""))
    )
    with pytest.raises(RuntimeError, match="ERR_NVGPUCTRPERM"):
        run_ncu(DRAM, TARGET, ncu="/opt/nsight/ncu")


# ---------------------------------------------------------------------------
# Rules
#
# The details page carries a rule on the row that fired it, in NCU's own columns
# rather than a rendered table. A counter table collects no section, so its
# details page has no such column at all, which is the state this project profiled
# in until now.
# ---------------------------------------------------------------------------

DETAILS_HEADER: Final = (
    f'{HEADER},"Rule Name","Rule Type","Rule Description",'
    '"Estimated Speedup Type","Estimated Speedup"'
)

_DETAIL_ROW: Final = (
    '"0","4711","python3","host","chunk_vector_bwd_kernel","2026-08-18 11:02:07",'
    '"1","7"'
)

DETAILS_CSV: Final = f"""==PROF== Connected to process 4711 (/gnu/store/py3-3.11.11/bin/python3)
{DETAILS_HEADER}
{_DETAIL_ROW},"Occupancy","Achieved Occupancy","%","16.61","","","","",""
{_DETAIL_ROW},"SpeedOfLight","","","","SOLBottleneck","INF","Compute and Memory are well-balanced","",""
{_DETAIL_ROW},"ComputeWorkloadAnalysis","","","","HighPipeUtilization","OPT","All compute pipelines are under-utilized","local","85.2"
{_DETAIL_ROW},"SourceCounters","","","","UncoalescedGlobalAccess","OPT","28,016,640 excessive sectors","global","41.81"
{_DETAIL_ROW},"MemoryWorkloadAnalysis_Tables","","","","SharedMemoryConflicts","OPT","1.5-way bank conflict","global","20.33"
==PROF== Disconnected from process 4711
"""


def test_a_new_pass_profiles_under_the_conditions_the_counter_tables_do() -> None:
    counters = ncu_command(DRAM, TARGET, ncu="/opt/nsight/ncu")
    rules = rules_command(
        TARGET, report="/tmp/cvb", ncu="/opt/nsight/ncu", sections=("Occupancy",)
    )
    # A pass taken with the clocks or the caches under a different policy is not
    # comparable with the ten counter tables, so both share one fixed prefix.
    fixed = counters[1 : counters.index("--metrics")]
    assert fixed
    assert rules[1 : 1 + len(fixed)] == fixed
    assert rules[len(rules) - len(TARGET) :] == list(TARGET)
    assert rules[rules.index("--section") + 1] == "Occupancy"
    assert rules[rules.index("--apply-rules") + 1] == "yes"
    # Without --force-overwrite NCU exits nonzero on an existing report, losing
    # the measurement just taken in order to keep a stale one.
    assert "--force-overwrite" in rules
    assert rules[rules.index("--export") + 1] == "/tmp/cvb"
    # NCU appends the suffix, so the import has to read the written name.
    assert report_file("/tmp/cvb") == "/tmp/cvb.ncu-rep"
    assert report_file("/tmp/cvb.ncu-rep") == "/tmp/cvb.ncu-rep"


def test_the_new_passes_reject_a_request_that_would_collect_nothing() -> None:
    # Each of these produces a command NCU accepts and a report with nothing in
    # it, so the raise has to come before the target runs.
    with pytest.raises(ValueError, match="no sections"):
        rules_command(TARGET, report="/tmp/cvb", sections=())
    with pytest.raises(ValueError, match="target command"):
        rules_command((), report="/tmp/cvb")
    with pytest.raises(ValueError, match="needs a report path"):
        rules_command(TARGET, report="")
    with pytest.raises(ValueError, match="needs a report path"):
        export_flags("")
    with pytest.raises(ValueError, match="needs a report path"):
        import_command("")
    with pytest.raises(ValueError, match="needs a page"):
        import_command("/tmp/cvb.ncu-rep", page="")


def test_a_details_page_with_no_rule_column_is_not_a_clean_kernel() -> None:
    # The output of a counter table. Reading it as a kernel no rule objected to
    # is the reading that kept every rule silent here, so it raises instead.
    with pytest.raises(ValueError, match="no 'Rule Name' column"):
        parse_rule_csv(DRAM_CSV)
    with pytest.raises(ValueError, match="no CSV header"):
        parse_rule_csv(NO_HEADER_CSV)


def test_rules_keep_a_local_estimate_apart_from_a_kernel_estimate() -> None:
    got = parse_rule_csv(DETAILS_CSV, report="/tmp/cvb.ncu-rep")
    assert got.report == "/tmp/cvb.ncu-rep"
    # The metric row carries no rule and is not a message.
    assert [one.rule for one in got.messages] == [
        "SOLBottleneck",
        "HighPipeUtilization",
        "UncoalescedGlobalAccess",
        "SharedMemoryConflicts",
    ]
    informational = got.messages[0]
    assert informational.severity == "INF"
    assert informational.speedup_scope == ""
    assert informational.speedup_pct is None
    assert informational.section == "SpeedOfLight"
    # The rule text carries the counters the verdict came from, which no other
    # output holds, so it is kept verbatim.
    assert got.messages[3].message == "1.5-way bank conflict"
    # 85.2% of one under-utilized pipeline is not 85.2% of the kernel. Ranking
    # the two scopes together puts the largest local estimate first and aims the
    # next change at nothing.
    assert [one.rule for one in got.ranked()] == [
        "UncoalescedGlobalAccess",
        "SharedMemoryConflicts",
    ]
    assert [one.rule for one in got.ranked(scope="local")] == ["HighPipeUtilization"]


# ---------------------------------------------------------------------------
# Per-line attribution
#
# The source page is a sequence of blocks, each opened by a File Path row and a
# Function Name row and then its own header. A row carrying a line number is
# NCU's aggregate for that line and the rows after it are the instructions
# correlated to it. Two columns are named Source: the first is the high-level
# line, the second the SASS, whose text carries commas of its own.
# ---------------------------------------------------------------------------

SOURCE_METRICS: Final[tuple[str, ...]] = (
    # Not the order SOURCE_TABLE requests them in. NCU orders this header itself.
    "smsp__pcsamp_sample_count",
    "inst_executed",
    "memory_access_size_type",
    "memory_l1_wavefronts_shared",
    "memory_l1_wavefronts_shared_ideal",
    *(pcsamp_metric(reason) for reason in STALL_REASONS),
)

_METRIC_ALIAS: Final[Mapping[str, str]] = {
    "samples": "smsp__pcsamp_sample_count",
    "inst": "inst_executed",
    "size": "memory_access_size_type",
    "wavefronts": "memory_l1_wavefronts_shared",
    "ideal": "memory_l1_wavefronts_shared_ideal",
}


def _source_header(*, metrics: Sequence[str] = SOURCE_METRICS) -> str:
    """A source-page header row, ``Source`` named twice as NCU names it."""
    names = ("Line No", "Source", "Address", "Source", *metrics)
    return ",".join(f'"{name}"' for name in names)


def _source_row(
    line: str,
    text: str,
    address: str,
    sass: str,
    *,
    metrics: Sequence[str] = SOURCE_METRICS,
    **values: int,
) -> str:
    """One source-page row. A metric not named is blank, as NCU leaves it.

    Args:
        line: The ``Line No`` cell, blank on an instruction row.
        text: High-level source, blank on an instruction row.
        address: Instruction address, blank on a line row.
        sass: Disassembly, blank on a line row.
        metrics: The header this row is printed under, when it is not the full one.
        **values: Metric values, by :data:`_METRIC_ALIAS` key or stall reason.
    """
    filled = {
        _METRIC_ALIAS.get(key, pcsamp_metric(key)): value
        for key, value in values.items()
    }
    cells = [line, text, address, sass]
    cells += [str(filled.get(metric, "")) for metric in metrics]
    return ",".join(f'"{cell}"' for cell in cells)


CVB_PATH: Final = "/lane/slinoss/ops/so3ssd/cute/bwd/chunk_vector.py"

SOURCE_CSV: Final = "\n".join(
    (
        f'"File Path","{CVB_PATH}"',
        '"Function Name","chunk_vector_bwd_kernel"',
        _source_header(),
        # Instruction printed before any line row: NCU correlated it to nothing.
        _source_row("", "", "0x0000000000007f00", "LDS.U.32 R4, [R8]", inst=8),
        _source_row(
            "1163",
            "    return sview[row, col]",
            "",
            "",
            samples=34,
            inst=300,
            wavefronts=300,
            ideal=200,
            mio_throttle=18,
            wait=5,
        ),
        _source_row(
            "", "", "0x0000000000008000", "LDS.U.32 R4, [R8]", inst=100, size=32
        ),
        # Predicated, and one opcode with two modifiers.
        _source_row(
            "",
            "",
            "0x0000000000008010",
            "@!P0 LDS.U.32 R6, [R8+0x4]",
            inst=100,
            size=32,
        ),
        _source_row("", "", "0x0000000000008020", "IADD3 R9, R9, 0x4, RZ", inst=100),
        # NCU elides a run of unattributed source with this row.
        _source_row("...", "", "", ""),
        _source_row(
            "1184", "    return vview[j]", "", "", samples=12, inst=60, mio_throttle=7
        ),
        _source_row(
            "", "", "0x0000000000008030", "LDS.U.16 R4, [R8]", inst=60, size=16
        ),
        _source_row(
            "922",
            "    return shuffle_xor(val, mask)",
            "",
            "",
            samples=90,
            inst=300,
            no_instruction=6,
            mio_throttle=64,
        ),
        _source_row(
            "",
            "",
            "0x0000000000008040",
            "SHFL.BFLY.IDX PT, R5, R4, 0x10, 0x1f",
            inst=300,
        ),
        '"Function Name","vector_reduce_kernel"',
        _source_header(),
        _source_row("1163", "    return sview[row, col]", "", "", samples=4, inst=50),
        _source_row(
            "", "", "0x0000000000009000", "STS.U.16 [R2], R4", inst=50, size=16
        ),
    )
)

NO_LINE_SOURCE_CSV: Final = "\n".join(
    (
        f'"File Path","{CVB_PATH}"',
        '"Function Name","chunk_vector_bwd_kernel"',
        _source_header(),
        _source_row("", "", "0x0000000000008000", "LDS.U.32 R4, [R8]", inst=100),
        _source_row("", "", "0x0000000000008010", "IADD3 R9, R9, 0x4, RZ", inst=100),
    )
)

RAGGED_SOURCE_CSV: Final = "\n".join(
    (
        f'"File Path","{CVB_PATH}"',
        '"Function Name","chunk_vector_bwd_kernel"',
        _source_header(),
        # A line row one cell short of the header, as NCU prints it when the last
        # requested metric has no value for the line.
        _source_row(
            "1163", "    return sview[row, col]", "", "", samples=34, inst=100
        ).rsplit(",", 1)[0],
        _source_row(
            "", "", "0x0000000000008000", "LDS.U.32 R4, [R8]", inst=100, size=32
        ),
    )
)

_NARROW_METRICS: Final[tuple[str, ...]] = tuple(
    m for m in SOURCE_METRICS if not m.startswith("memory_l1_wavefronts_shared")
)

NARROWED_SOURCE_CSV: Final = "\n".join(
    (
        f'"File Path","{CVB_PATH}"',
        '"Function Name","chunk_vector_bwd_kernel"',
        _source_header(),
        _source_row(
            "1163", "    return sview[row, col]", "", "", inst=100, wavefronts=7
        ),
        _source_row(
            "", "", "0x0000000000008000", "LDS.U.32 R4, [R8]", inst=100, size=32
        ),
        # A kernel that touches no shared memory: NCU drops both wavefront
        # columns from its header, so this block's rows decode under a narrower
        # map than the block above.
        '"Function Name","vector_reduce_kernel"',
        _source_header(metrics=_NARROW_METRICS),
        _source_row(
            "1163",
            "    return sview[row, col]",
            "",
            "",
            metrics=_NARROW_METRICS,
            samples=4,
            inst=50,
            mio_throttle=3,
        ),
        _source_row(
            "",
            "",
            "0x0000000000009000",
            "LDG.E R4, [R2]",
            metrics=_NARROW_METRICS,
            inst=50,
            size=32,
        ),
    )
)

NO_INST_SOURCE_CSV: Final = "\n".join(
    (
        f'"File Path","{CVB_PATH}"',
        '"Function Name","chunk_vector_bwd_kernel"',
        _source_header(
            metrics=tuple(m for m in SOURCE_METRICS if m != "inst_executed")
        ),
    )
)


def test_the_pc_sampling_family_respells_one_stall_reason() -> None:
    names = {reason: pcsamp_metric(reason) for reason in STALL_REASONS}
    assert len(set(names.values())) == len(STALL_REASONS)
    assert set(names.values()) <= set(SOURCE_TABLE.metrics)
    # The two families name the same reasons and disagree on this one. Requesting
    # the per-cycle spelling gets no such metric, and one reason of the seventeen
    # is silently absent from the attribution.
    assert "_no_instruction_per" in stall_metric("no_instruction")
    assert names["no_instruction"].endswith("_no_instructions_not_issued")
    for reason in STALL_REASONS:
        if reason != "no_instruction":
            assert (
                names[reason] == f"smsp__pcsamp_warps_issue_stalled_{reason}_not_issued"
            )
    # The duration, so a source pass can be placed in the same window as the
    # counter tables that bound it.
    assert DURATION in SOURCE_TABLE.metrics


def test_the_source_page_attributes_each_instruction_to_a_line_or_to_none() -> None:
    got = parse_source_csv(SOURCE_CSV, report="/tmp/cvb.ncu-rep")
    assert [(one.kernel, one.line) for one in got.lines] == [
        ("chunk_vector_bwd_kernel", 922),
        ("chunk_vector_bwd_kernel", 1163),
        ("chunk_vector_bwd_kernel", 1184),
        # Same file and line as the first kernel's, and a separate record: the
        # block's Function Name is part of the key.
        ("vector_reduce_kernel", 1163),
    ]
    assert all(one.file == CVB_PATH for one in got.lines)
    mat = got.lines[1]
    # Opcode class without its modifiers, predicate prefix stripped. The histogram
    # covers every pipe and the LSU count covers one, because the integer work is
    # what an LSU-only census hides: it is a third of the instruction stream in the
    # two kernels that dominate the backward, and it issues at half the FMA rate.
    assert mat.opcode_inst == {"IADD3": 100, "LDS": 200}
    assert mat.inst_count == 300
    assert mat.lsu_inst_count == 200
    assert mat.access_bit_inst == {32: 200}
    assert mat.sample_count == 34
    assert mat.stall_samples["mio_throttle"] == 18
    assert mat.stall_samples["wait"] == 5
    assert mat.not_issued_count == 23
    # A conflicted wavefront is a replayed LSU instruction, so the excess over
    # ideal is the part of this line a better layout deletes.
    assert mat.shared_wavefront_excess_count == 100
    # A shuffle moves no memory and issues on the same port, so it is LSU work
    # with no access width to it.
    shuffle = got.lines[0]
    assert shuffle.opcode_inst == {"SHFL": 300}
    assert shuffle.access_bit_inst == {}
    assert shuffle.stall_samples["no_instruction"] == 6
    # 16-bit accesses are the deleteable ones: two of them pack into one 32-bit
    # access and the second instruction buys no byte.
    assert got.lines[2].access_bit_inst == {16: 60}
    assert got.lines[3].opcode_inst == {"STS": 50}
    assert got.lsu_inst_count == 610
    # An instruction NCU printed under no line is the shortfall in the table, not
    # a row to drop.
    assert got.unattributed_inst_count == 8
    # The window aggregate crosses kernels: LDS is 200 from one and 60 from the
    # other. Descending, because the head of the mapping is the instruction budget
    # in the order it has to be spent, and a pipe's share is only readable against
    # the whole stream.
    assert got.opcode_inst == {"SHFL": 300, "LDS": 260, "IADD3": 100, "STS": 50}
    assert list(got.opcode_inst) == ["SHFL", "LDS", "IADD3", "STS"]


def test_a_source_page_with_no_correlated_line_names_the_missing_lineinfo() -> None:
    # The profile this project has always taken: SASS with no line against it.
    # A CuTe DSL kernel gets line information from the environment rather than
    # from a build flag, so the message has to name the variable.
    with pytest.raises(ValueError, match="CUTE_DSL_LINEINFO=1"):
        parse_source_csv(NO_LINE_SOURCE_CSV)


def test_a_line_row_short_of_the_header_keeps_its_pass() -> None:
    # NCU can print a line row missing its trailing cells. Dropping the row would
    # cost the whole pass, which is an hour of collection, so an absent cell reads
    # as zero.
    got = parse_source_csv(RAGGED_SOURCE_CSV)
    (one,) = got.lines
    assert one.line == 1163
    assert one.lsu_inst_count == 100
    assert one.stall_samples[STALL_REASONS[-1]] == 0


def test_a_kernel_with_no_shared_traffic_decodes_under_its_own_header() -> None:
    # NCU drops a metric column for a kernel that has no traffic of that kind, so
    # one window can hold two column maps. Decoding every row under the last
    # header seen loses the pass, and every window that names this kernel holds a
    # shared-memory-free one beside it.
    got = parse_source_csv(NARROWED_SOURCE_CSV)
    wide, narrow = got.lines
    assert (wide.kernel, wide.shared_wavefront_count) == (
        "chunk_vector_bwd_kernel",
        7,
    )
    assert narrow.kernel == "vector_reduce_kernel"
    assert narrow.shared_wavefront_count == 0
    assert narrow.sample_count == 4
    assert narrow.stall_samples["mio_throttle"] == 3


def test_parse_source_csv_rejects_output_it_cannot_read() -> None:
    # The details page of a counter table opens no block.
    with pytest.raises(ValueError, match="no source block"):
        parse_source_csv(DRAM_CSV)
    with pytest.raises(ValueError, match="no 'inst_executed' column"):
        parse_source_csv(NO_INST_SOURCE_CSV)


def test_the_lsu_floor_moves_with_the_clock() -> None:
    inst = Count(166_152_960)
    # Both clocks occur on one uncontrolled part, and 84 multiprocessors at
    # 1.88 GHz issue 79.1 thousand LSU warp-instructions per microsecond against
    # 74.4 thousand at 1.77 GHz. A floor written down as a constant holds on one
    # run of the two.
    fast = lsu_floor_us(inst, sm_count=Count(84), clock_mhz=Megahertz(1882.848))
    slow = lsu_floor_us(inst, sm_count=Count(84), clock_mhz=Megahertz(1771.0))
    assert fast == pytest.approx(2101.1, abs=0.1)
    assert slow == pytest.approx(2233.8, abs=0.1)
    assert lsu_floor_us(
        inst, sm_count=Count(168), clock_mhz=Megahertz(1882.848)
    ) == pytest.approx(fast / 2)
    with pytest.raises(ValueError, match="sm_count must be positive"):
        lsu_floor_us(inst, sm_count=Count(0), clock_mhz=Megahertz(1882.848))
    with pytest.raises(ValueError, match="clock_mhz must be positive"):
        lsu_floor_us(inst, sm_count=Count(84), clock_mhz=Megahertz(0.0))


def test_both_new_passes_export_a_report_and_read_it_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An --export run prints no counter table at all, so parsing the collection's
    # own stdout reads every pass as empty. Each pass is two invocations.
    fake = FakeRun((0, "", ""), (0, DETAILS_CSV, ""))
    monkeypatch.setattr(subprocess, "run", fake)
    rules = run_rules(
        TARGET, report="/tmp/cvb", ncu="/opt/nsight/ncu", sections=("Occupancy",)
    )
    assert fake.commands == [
        rules_command(
            TARGET, report="/tmp/cvb", ncu="/opt/nsight/ncu", sections=("Occupancy",)
        ),
        # The written name, not the name NCU was asked for.
        import_command("/tmp/cvb.ncu-rep", ncu="/opt/nsight/ncu", page="details"),
    ]
    assert rules.report == "/tmp/cvb.ncu-rep"
    assert rules.command == tuple(fake.commands[0])
    assert len(rules.messages) == 4

    fake = FakeRun((0, "", ""), (0, SOURCE_CSV, ""))
    monkeypatch.setattr(subprocess, "run", fake)
    source = run_source(TARGET, report="/tmp/cvb.ncu-rep", ncu="/opt/nsight/ncu")
    assert fake.commands == [
        ncu_command(
            SOURCE_TABLE,
            TARGET,
            ncu="/opt/nsight/ncu",
            extra=export_flags("/tmp/cvb.ncu-rep"),
        ),
        # cuda,sass is the only view that carries a line number and a counter at
        # once: cuda alone has no counters and sass alone has no line.
        import_command(
            "/tmp/cvb.ncu-rep",
            ncu="/opt/nsight/ncu",
            page="source",
            print_source=SOURCE_VIEW,
        ),
    ]
    assert source.lines
    assert source.report == "/tmp/cvb.ncu-rep"
