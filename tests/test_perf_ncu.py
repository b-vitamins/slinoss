"""Nsight Compute driver: units, command, parsing, and the cross-pass merge.

The profiler binary is absent on this host and on the verification fleet, so no
test launches ``ncu``. Every pure function is driven with fixture text held in
this module, and :func:`run_ncu` is exercised through a fake
:func:`subprocess.run` that records its argv.
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
    STALL_FIELDS,
    STALL_REASONS,
    NcuPass,
    NcuTable,
    kernel_counters,
    metric_scale,
    ncu_command,
    parse_ncu_csv,
    run_ncu,
    stall_field,
    stall_metric,
)

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
        run_ncu(DRAM, TARGET)
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
        run_ncu(DRAM, TARGET)
