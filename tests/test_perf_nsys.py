"""Nsight Systems: command construction, the duration unit, the trace, the timeline.

The profiler binary is absent on this host and on the verification fleet, so no
test launches ``nsys``. Every pure function is driven with fixture text held in
this module, and :func:`run_nsys` is exercised through a fake
:func:`subprocess.run` that records its argv.

Every call names the binary by path. A bare name is resolved against PATH and the
CUDA bin directories, so a default here would pass on a host that has the profiler
and raise on one that does not, and the subject is the driver rather than the host.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

from slinoss.perf.nsys import (
    GpuEvent,
    duration_scale_ns,
    events_within,
    nsys_profile_command,
    nsys_report_texts,
    nsys_stats_command,
    occupancy,
    parse_gpu_events,
    parse_gpu_trace,
    parse_nvtx_projection,
    repeat_windows,
    run_nsys,
)
from slinoss.perf.units import Microseconds

TARGET: Final[tuple[str, ...]] = (
    "python3",
    "scripts/perf/profile_target.py",
    "--iters",
    "8",
)

CHUNK: Final = "so3ssd_chunk_fwd"
CPASYNC: Final = "so3ssd_state_cpasync_fwd"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_profile_command() -> None:
    # The capture range is the CUDA profiler API, so warmup and compilation stay
    # outside the totals; --force-overwrite keeps a stale report from being read
    # as a fresh one. The target follows -o directly; nsys parses a bare "--" as
    # an empty long option and exits on it.
    assert nsys_profile_command(TARGET, Path("out/profile-op")) == [
        "nsys",
        "profile",
        "--trace=cuda",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        "out/profile-op",
        "python3",
        "scripts/perf/profile_target.py",
        "--iters",
        "8",
    ]
    got = nsys_profile_command(
        TARGET, Path("/tmp/run"), nsys="/opt/nsight/nsys", trace="cuda,nvtx"
    )
    assert got[0] == "/opt/nsight/nsys"
    assert got[2] == "--trace=cuda,nvtx"
    # The target is the tail, immediately after the output path.
    assert got[-len(TARGET) :] == list(TARGET)
    assert got[-len(TARGET) - 2 : -len(TARGET)] == ["-o", "/tmp/run"]
    with pytest.raises(ValueError, match="nsys needs a target command"):
        nsys_profile_command((), Path("out/profile-op"))


def test_stats_command() -> None:
    assert nsys_stats_command(Path("out/profile-op.nsys-rep")) == [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_trace",
        "--format",
        "csv",
        "--force-export=true",
        "out/profile-op.nsys-rep",
    ]
    got = nsys_stats_command(
        Path("out/run.nsys-rep"), nsys="/opt/nsight/nsys", name="cuda_gpu_kern_sum"
    )
    assert got[0] == "/opt/nsight/nsys"
    assert got[3] == "cuda_gpu_kern_sum"


# ---------------------------------------------------------------------------
# Duration unit
#
# NSYS has emitted this column in nanoseconds and in microseconds across
# versions, so the unit is read from the header parenthetical rather than
# assumed. Assuming nanoseconds against a microsecond column is a 1000x error in
# every duration in the report.
# ---------------------------------------------------------------------------


def test_duration_scale_ns_reads_the_header_parenthetical() -> None:
    for column, want in (
        ("Duration (ns)", 1.0),
        ("Duration (nsec)", 1.0),
        ("Duration (NS)", 1.0),
        ("Duration (us)", 1e3),
        ("Duration (usec)", 1e3),
        # The micro sign, spelled as an escape so this file stays ASCII.
        ("Duration (\u00b5s)", 1e3),
        ("Duration ( us )", 1e3),
        ("Duration (ms)", 1e6),
        ("Duration (msec)", 1e6),
        ("Duration (s)", 1e9),
        ("Duration (sec)", 1e9),
    ):
        assert duration_scale_ns(column) == want
    for column in ("Duration", "Duration ()", "Duration (fortnight)"):
        with pytest.raises(ValueError, match="carries no recognized duration unit"):
            duration_scale_ns(column)


# ---------------------------------------------------------------------------
# Fixture text
#
# The column set and the preamble are NSYS's own. The Device column is quoted
# because it contains a comma, which is why the parser reads CSV rather than
# splitting on commas.
# ---------------------------------------------------------------------------

PREAMBLE: Final = """Generating '/tmp/nsys-report-9f31.sqlite'
[1/1] Executing 'cuda_gpu_trace' stats report

 ** CUDA GPU Trace (cuda_gpu_trace):
"""

TRACE_HEADER: Final = (
    "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,"
    "StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MBps),SrcMemKd,DstMemKd,"
    "Device,Ctx,GreenCtx,Strm,Name"
)

GPU_TRACE_CSV: Final = f"""{PREAMBLE}{TRACE_HEADER}
1719000000,41280,1201,512,1,1,256,1,1,104,0.048,0.033,,,,,"NVIDIA RTX A6000 (0)",1,,7,so3ssd_chunk_fwd
1719050000,43520,1204,512,1,1,256,1,1,104,0.048,0.033,,,,,"NVIDIA RTX A6000 (0)",1,,7,so3ssd_chunk_fwd
1719100000,3200,1209,108,1,1,128,1,1,56,0.000,0.064,,,,,"NVIDIA RTX A6000 (0)",1,,7,so3ssd_state_cpasync_fwd
1719150000,2560,1212,,,,,,,,,,0.004,1562.500,Device,Pinned,"NVIDIA RTX A6000 (0)",1,,7,[CUDA memcpy DtoH]
1719160000,1920,1213,,,,,,,,,,0.004,2083.333,Pinned,Device,"NVIDIA RTX A6000 (0)",1,,7,[CUDA memcpy HtoD]
1719170000,1280,1215,,,,,,,,,,0.262,,,,"NVIDIA RTX A6000 (0)",1,,7,[CUDA memset]
"""

MICROSECOND_CSV: Final = """Start (us),Duration (us),Name
1719000.000,41.28,so3ssd_chunk_fwd
"""

GROUPED_CSV: Final = """Start (ns),Duration (ns),Name
1719000000,"1,048,576",so3ssd_chunk_fwd
"""

UNUSABLE_CSV: Final = """Start (ns),Duration (ns),Name
1719000000,,so3ssd_chunk_fwd
1719050000,N/A,so3ssd_chunk_fwd
1719100000,3200,
1719150000,3200,so3ssd_state_cpasync_fwd
"""

COPY_ONLY_CSV: Final = """Start (ns),Duration (ns),Name
1719150000,2560,[CUDA memcpy DtoH]
"""

EMPTY_WINDOW_CSV: Final = f"""{PREAMBLE}{TRACE_HEADER}
"""

NO_HEADER_CSV: Final = """Generating '/tmp/nsys-report-9f31.sqlite'
SKIPPED: /tmp/nsys-report-9f31.sqlite does not contain CUDA trace data.
"""

NO_DURATION_COLUMN_CSV: Final = """Start (ns),Kernel Duration (ns),Name
1719000000,41280,so3ssd_chunk_fwd
"""

NO_NAME_COLUMN_CSV: Final = """Start (ns),Duration (ns),Kernel Name
1719000000,41280,so3ssd_chunk_fwd
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_gpu_trace_splits_the_work_kinds() -> None:
    # A staging copy or a fill summed into the kernel total is invisible, and
    # both are defects under the kernel rules, so they are parsed out separately.
    trace = parse_gpu_trace(GPU_TRACE_CSV, label="so3ssd fwd", report_path="out/x.rep")
    assert trace.label == "so3ssd fwd"
    assert trace.report_path == "out/x.rep"
    assert trace.kernel_sum_duration_us == pytest.approx(88.0)
    assert trace.memcpy_sum_duration_us == pytest.approx(4.48)
    assert trace.memset_sum_duration_us == pytest.approx(1.28)
    assert trace.memcpy_count == 2
    assert trace.memset_count == 1
    assert trace.device_sum_duration_us == pytest.approx(93.76)
    # Longest first, and no copy or fill among them.
    assert [k.kernel for k in trace.kernels] == [CHUNK, CPASYNC]
    assert [k.duration_us for k in trace.kernels] == sorted(
        (k.duration_us for k in trace.kernels), reverse=True
    )
    assert all(not k.kernel.startswith("[CUDA") for k in trace.kernels)


def test_parse_gpu_trace_sums_the_launches_of_one_kernel() -> None:
    trace = parse_gpu_trace(GPU_TRACE_CSV)
    chunk = trace.kernel(CHUNK)
    assert chunk.launch_count == 2
    assert chunk.duration_us == pytest.approx(84.8)
    assert chunk.duration.sample_count == 2
    assert chunk.duration.median_duration_us == pytest.approx(42.4)
    assert chunk.duration.min_duration_us == pytest.approx(41.28)
    assert chunk.duration.max_duration_us == pytest.approx(43.52)
    assert chunk.duration.spread_pct == pytest.approx(100.0 * 2.24 / 42.4)
    # Two launches bound the median by their own range, so the floor is half of it.
    assert chunk.duration.resolution_pct == pytest.approx(50.0 * 2.24 / 42.4)
    # The per-launch durations are kept, so a reader can see the mixture rather
    # than infer it from the summary.
    assert list(chunk.duration.samples_duration_us) == pytest.approx([41.28, 43.52])
    assert chunk.share_pct == pytest.approx(100.0 * 84.8 / 93.76)
    assert trace.kernel(CPASYNC).share_pct == pytest.approx(100.0 * 3.2 / 93.76)


def test_parse_gpu_trace_reads_the_duration_cell_as_nsys_wrote_it() -> None:
    # A microsecond column read as nanoseconds is a 1000x error in every duration,
    # and a grouped cell parsed with its separators is not a number at all.
    assert parse_gpu_trace(MICROSECOND_CSV).kernel(CHUNK).duration_us == pytest.approx(
        41.28
    )
    assert parse_gpu_trace(GROUPED_CSV).kernel(CHUNK).duration_us == pytest.approx(
        1048.576
    )


def test_parse_gpu_trace_drops_unusable_rows() -> None:
    trace = parse_gpu_trace(UNUSABLE_CSV)
    assert [k.kernel for k in trace.kernels] == [CPASYNC]
    assert trace.kernel(CPASYNC).launch_count == 1


def test_parse_gpu_trace_without_a_kernel() -> None:
    trace = parse_gpu_trace(COPY_ONLY_CSV)
    assert trace.kernels == ()
    assert trace.kernel_sum_duration_us == 0.0
    assert trace.memcpy_count == 1
    assert trace.device_sum_duration_us == pytest.approx(2.56)


def test_parse_gpu_trace_rejects_output_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="no cuda_gpu_trace header in nsys output"):
        parse_gpu_trace(NO_HEADER_CSV, label="so3ssd fwd")
    with pytest.raises(ValueError, match="no 'duration' column"):
        parse_gpu_trace(NO_DURATION_COLUMN_CSV)
    with pytest.raises(ValueError, match="no 'name' column"):
        parse_gpu_trace(NO_NAME_COLUMN_CSV)
    # No device work means the capture range never opened. That is a broken run,
    # not a fast one.
    with pytest.raises(ValueError, match="the capture range never opened"):
        parse_gpu_trace(EMPTY_WINDOW_CSV, label="so3ssd fwd")


def test_kernel_lookup_rejects_an_absent_name() -> None:
    trace = parse_gpu_trace(GPU_TRACE_CSV, label="so3ssd fwd")
    with pytest.raises(KeyError, match="no kernel 'so3ssd_chunk_bwd' in 'so3ssd fwd'"):
        trace.kernel("so3ssd_chunk_bwd")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

PROFILE_LOG: Final = """Collection started.
Capture range started.
Capture range ended.
Generating '/tmp/nsys-report-9f31.nsys-rep'
"""

DIAGNOSTIC: Final = "\n".join(f"** ERROR: diagnostic {i:02d}" for i in range(1, 16))


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


def test_run_nsys_profiles_then_reads_the_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeRun((0, PROFILE_LOG, ""), (0, GPU_TRACE_CSV, ""))
    monkeypatch.setattr(subprocess, "run", fake)
    base = tmp_path / "profile-op"
    report = tmp_path / "profile-op.nsys-rep"
    trace = run_nsys(
        TARGET,
        base,
        label="so3ssd fwd",
        nsys="/opt/nsight/nsys",
        trace="cuda,nvtx",
        cwd="/tmp",
        timeout_s=600.0,
    )
    # The profile runs first: the stats command reads the report it wrote.
    assert [command[1] for command in fake.commands] == ["profile", "stats"]
    assert fake.commands[0] == nsys_profile_command(
        TARGET, base, nsys="/opt/nsight/nsys", trace="cuda,nvtx"
    )
    assert fake.commands[1] == nsys_stats_command(report, nsys="/opt/nsight/nsys")
    assert fake.commands[1][-1].endswith(".nsys-rep")
    assert all(kwargs["cwd"] == "/tmp" for kwargs in fake.kwargs)
    assert all(kwargs["timeout"] == 600.0 for kwargs in fake.kwargs)
    assert all(kwargs["check"] is False for kwargs in fake.kwargs)
    assert all(kwargs["capture_output"] is True for kwargs in fake.kwargs)
    assert all(kwargs["text"] is True for kwargs in fake.kwargs)
    # Parsed from the stats stdout, not the profile log.
    assert trace.label == "so3ssd fwd"
    assert trace.report_path == str(report)
    assert trace.kernel_sum_duration_us == pytest.approx(88.0)


def test_run_nsys_appends_the_suffix_to_a_dotted_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # nsys appends `.nsys-rep` to the whole name it was given. Substituting for the
    # last suffix instead would read `run.nsys-rep` here and fail to find the
    # report after the full profile had already been paid for.
    fake = FakeRun((0, PROFILE_LOG, ""), (0, GPU_TRACE_CSV, ""))
    monkeypatch.setattr(subprocess, "run", fake)
    base = tmp_path / "run.standard.fwd"
    trace = run_nsys(TARGET, base, nsys="/opt/nsight/nsys")
    assert trace.report_path == str(tmp_path / "run.standard.fwd.nsys-rep")
    assert fake.commands[1][-1] == trace.report_path


def test_run_nsys_raises_on_either_failed_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeRun((3, "", DIAGNOSTIC))
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as caught:
        run_nsys(TARGET, tmp_path / "profile-op", nsys="/opt/nsight/nsys")
    message = str(caught.value)
    assert "profile exited 3" in message
    assert "diagnostic 15" in message
    assert "diagnostic 04" in message
    assert "diagnostic 03" not in message  # the tail is the last twelve lines
    assert len(fake.commands) == 1  # the stats command never ran
    # A profile that ran and a stats export that failed is the other failure, and
    # it names the command that failed rather than the pair.
    stats = FakeRun((0, PROFILE_LOG, ""), (4, "", "** ERROR: report file not found"))
    monkeypatch.setattr(subprocess, "run", stats)
    with pytest.raises(RuntimeError) as failed:
        run_nsys(TARGET, tmp_path / "profile-op", nsys="/opt/nsight/nsys")
    assert "stats exited 4" in str(failed.value)
    assert "report file not found" in str(failed.value)
    assert len(stats.commands) == 2
    # nsys prints some diagnostics on stdout, so an empty stderr falls back to it.
    monkeypatch.setattr(
        subprocess, "run", FakeRun((1, "** ERROR: no CUDA device present", ""))
    )
    with pytest.raises(RuntimeError, match="no CUDA device present"):
        run_nsys(TARGET, tmp_path / "profile-op", nsys="/opt/nsight/nsys")


# ---------------------------------------------------------------------------
# The timeline
#
# parse_gpu_trace answers what a kernel costs. These answer where the step went,
# which needs the start times: a sum of durations cannot say whether the device
# was executing between two launches.
# ---------------------------------------------------------------------------

TIMELINE_CSV: Final = f"""{PREAMBLE}{TRACE_HEADER}
1719050000,20000,1204,512,1,1,256,1,1,104,0.048,0.033,,,,,"NVIDIA RTX A6000 (0)",1,,7,{CHUNK}
1719000000,10000,1201,512,1,1,256,1,1,104,0.048,0.033,,,,,"NVIDIA RTX A6000 (0)",1,,7,{CHUNK}
1719005000,10000,1209,108,1,1,128,1,1,56,0.000,0.064,,,,,"NVIDIA RTX A6000 (0)",1,,20,{CPASYNC}
"""

MICROSECOND_TIMELINE_CSV: Final = """Start (us),Duration (us),Strm,Name
1719000.000,41.28,7,so3ssd_chunk_fwd
"""

NO_START_COLUMN_CSV: Final = """First (ns),Duration (ns),Name
1719000000,41280,so3ssd_chunk_fwd
"""


def test_parse_gpu_events_reads_the_start_column_in_its_own_unit() -> None:
    # A microsecond start read as nanoseconds puts every event 1000x too early and
    # every gap 1000x too small, which reads as a device that never waited.
    events = parse_gpu_events(MICROSECOND_TIMELINE_CSV)
    assert [e.start_us for e in events] == pytest.approx([1719000.0])
    assert events[0].end_us == pytest.approx(1719041.28)
    assert events[0].stream == "7"


def test_parse_gpu_events_orders_the_timeline_by_start() -> None:
    # NSYS emits rows in correlation-id order, not start order, and every window
    # and gap below depends on the order being the timeline's.
    events = parse_gpu_events(TIMELINE_CSV)
    assert [e.start_us for e in events] == pytest.approx(
        [1719000.0, 1719005.0, 1719050.0]
    )
    assert [e.name for e in events] == [CHUNK, CPASYNC, CHUNK]
    assert [e.stream for e in events] == ["7", "20", "7"]


def test_parse_gpu_events_rejects_output_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="no cuda_gpu_trace header in nsys output"):
        parse_gpu_events(NO_HEADER_CSV)
    with pytest.raises(ValueError, match="no 'start' column"):
        parse_gpu_events(NO_START_COLUMN_CSV)
    # An empty window is a capture range that never opened, not a device that idled.
    with pytest.raises(ValueError, match="the capture range never opened"):
        parse_gpu_events(EMPTY_WINDOW_CSV)


def test_occupancy_unions_the_streams_rather_than_summing_them() -> None:
    # Two streams overlap here: 1719000-1719010 on stream 7 and 1719005-1719015 on
    # stream 20, then 1719050-1719070 on stream 7. Busy is 15 + 20 = 35 us over a
    # 70 us span. A sum of durations would report 40 us busy, which exceeds the
    # window's own occupancy and can exceed the window itself.
    got = occupancy("step", parse_gpu_events(TIMELINE_CSV))
    assert got.label == "step"
    assert got.span_us == pytest.approx(70.0)
    assert got.busy_us == pytest.approx(35.0)
    assert got.sum_duration_us == pytest.approx(40.0)
    assert got.idle_us == pytest.approx(35.0)
    assert got.idle_pct == pytest.approx(50.0)
    assert got.busy_us + got.idle_us == pytest.approx(got.span_us)
    assert got.event_count == 3
    # One gap, 1719015 to 1719050. The overlap is not a second gap.
    assert got.gap_count == 1
    assert got.max_gap_us == pytest.approx(35.0)


def test_occupancy_refuses_an_empty_window() -> None:
    # Zero idle over zero span reads as a device that never waited.
    with pytest.raises(ValueError, match=r"occupancy\('step'\) needs at least one"):
        occupancy("step", ())


def test_events_within_selects_by_start_so_windows_tile() -> None:
    # By overlap instead, an event straddling a boundary would land in both windows
    # and its duration would be counted twice.
    events = parse_gpu_events(TIMELINE_CSV)
    first = events_within(events, 1719000.0, 1719020.0)
    second = events_within(events, 1719020.0001, 1719100.0)
    assert [e.name for e in first] == [CHUNK, CPASYNC]
    assert [e.name for e in second] == [CHUNK]
    assert len(first) + len(second) == len(events)
    assert events_within(events, 1719100.0, 1719200.0) == ()


# ---------------------------------------------------------------------------
# NVTX projection
#
# The column set is NSYS 2023.3's own. Eight of its columns end in "Start" or
# "Duration", which is why the parser matches on two needles rather than a prefix.
# The first row is NSYS's synthetic root: no name, no orig columns, and every
# device operation of the run counted into it.
# ---------------------------------------------------------------------------

NVTX_PREAMBLE: Final = """Generating '/tmp/nsys-report-9f31.sqlite'
[1/1] Executing 'nvtx_gpu_proj_trace' stats report

 ** NVTX GPU Projection Trace (nvtx_gpu_proj_trace):
"""

NVTX_HEADER: Final = (
    "Name,Projected Start (ns),Projected Duration (ns),Orig Start (ns),"
    "Orig Duration (ns),Style,PID,TID,NumGPUOps,Lvl,NumChild,RangeId,ParentId,"
    "RangeStack"
)

NVTX_CSV: Final = f"""{NVTX_PREAMBLE}{NVTX_HEADER}
,10758298,33981358,,,,295239,295820,164,,0,,,
so3ssd-auto,14739612,1031280,14108904,3554743,PushPop,295239,295239,2,0,0,2,,:2
mamba-g1,8895767,1177902,8450530,5597874,PushPop,295239,295239,5,0,0,1,,:1
"""

NVTX_NOTHING_PROJECTED_CSV: Final = f"""{NVTX_PREAMBLE}{NVTX_HEADER}
mamba-g1,,,8450530,5597874,PushPop,295239,295239,0,0,0,1,,:1
"""


def test_parse_nvtx_projection_reads_the_projected_columns_not_the_orig_ones() -> None:
    # Both intervals are kept and neither is read for the other. The projected span
    # is device time, the orig span is the pushing thread's push to its pop, and
    # which is larger says which side the range is bound by.
    spans = parse_nvtx_projection(NVTX_CSV)
    assert [s.name for s in spans] == ["mamba-g1", "so3ssd-auto"]  # by projected start
    mamba = spans[0]
    assert mamba.start_us == pytest.approx(8895.767)
    assert mamba.duration_us == pytest.approx(1177.902)
    assert mamba.end_us == pytest.approx(10073.669)
    assert mamba.host_start_us == pytest.approx(8450.530)
    assert mamba.host_duration_us == pytest.approx(5597.874)
    assert mamba.host_end_us == pytest.approx(14048.404)
    # The projection is per thread, so this count is the launches the pushing thread
    # made and not the launches the range enclosed. Five here against the 164 the
    # run made: a PyTorch backward runs on the autograd engine's own thread.
    assert mamba.gpu_op_count == 5
    # NSYS's synthetic root carries no name and every operation of the run. Kept, it
    # would be a third arm holding both of the real ones.
    assert all(s.name for s in spans)
    assert len(spans) == 2


def test_parse_nvtx_projection_rejects_output_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="no nvtx_gpu_proj_trace header"):
        parse_nvtx_projection(NO_HEADER_CSV)
    with pytest.raises(ValueError, match="no 'projected\\+start' column"):
        parse_nvtx_projection(GPU_TRACE_CSV)
    # A range that projected nothing means NVTX went untraced or the pushes fell
    # outside the capture window. Both are broken runs, not empty ones.
    with pytest.raises(ValueError, match="projected no NVTX range onto the device"):
        parse_nvtx_projection(NVTX_NOTHING_PROJECTED_CSV)


# ---------------------------------------------------------------------------
# Repetition windows
# ---------------------------------------------------------------------------


def event(name: str, start_us: float, duration_us: float) -> GpuEvent:
    """One timeline event, for the segmentation tests."""
    return GpuEvent(
        name=name,
        start_us=Microseconds(start_us),
        duration_us=Microseconds(duration_us),
    )


def loop_timeline(count: int, *, stride: float = 100.0) -> tuple[GpuEvent, ...]:
    """``count`` repetitions of a two-kernel step, one per ``stride``."""
    return tuple(
        event(name, i * stride + offset, 10.0)
        for i in range(count)
        for name, offset in ((CHUNK, 0.0), (CPASYNC, 20.0))
    )


def test_repeat_windows_cuts_the_timeline_by_repetition_not_by_gap() -> None:
    # The gap inside a step (10 to 20) is smaller than the gap between two steps
    # (30 to 100) here, but a step whose largest gap fell at its own boundary would
    # be mis-cut by a gap rule, and the idle it holds would be charged to the loop.
    windows = repeat_windows(loop_timeline(3), 3)
    assert [[e.name for e in w] for w in windows] == [[CHUNK, CPASYNC]] * 3
    assert [w[0].start_us for w in windows] == pytest.approx([0.0, 100.0, 200.0])
    assert occupancy("step", windows[1]).span_us == pytest.approx(30.0)


def test_repeat_windows_rejects_a_timeline_that_is_not_the_loop_it_was_told() -> None:
    with pytest.raises(ValueError, match="needs a positive count"):
        repeat_windows(loop_timeline(2), 0)
    # An extra operation is compilation, an allocator fill, or another arm sharing
    # the window. Divided anyway, it would shift every later window by one launch.
    with pytest.raises(ValueError, match="5 device operations do not divide into 2"):
        repeat_windows((*loop_timeline(2), event(CHUNK, 500.0, 10.0)), 2)
    # Equal counts and unequal sequences: the segmentation would line up two
    # different steps and the per-step median would be of a mixture.
    odd = (*loop_timeline(1), event(CHUNK, 100.0, 10.0), event(CHUNK, 120.0, 10.0))
    with pytest.raises(
        ValueError, match=r"repetition 1 launched .* where repetition 0"
    ):
        repeat_windows(odd, 2)


def test_nsys_report_texts_profiles_once_for_every_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One profile, then one export per report. Profiling per report would give the
    # GPU trace and the NVTX projection two different runs, and every cross-check
    # between them would then be between two timelines.
    fake = FakeRun((0, PROFILE_LOG, ""), (0, GPU_TRACE_CSV, ""), (0, NVTX_CSV, ""))
    monkeypatch.setattr(subprocess, "run", fake)
    base = tmp_path / "run.acceptance"
    texts = nsys_report_texts(
        TARGET,
        base,
        ("cuda_gpu_trace", "nvtx_gpu_proj_trace"),
        nsys="/opt/nsight/nsys",
        trace="cuda,nvtx",
    )
    assert [command[1] for command in fake.commands] == ["profile", "stats", "stats"]
    assert [command[3] for command in fake.commands[1:]] == [
        "cuda_gpu_trace",
        "nvtx_gpu_proj_trace",
    ]
    report = str(tmp_path / "run.acceptance.nsys-rep")
    assert all(command[-1] == report for command in fake.commands[1:])
    assert set(texts) == {"cuda_gpu_trace", "nvtx_gpu_proj_trace"}
    assert parse_gpu_events(texts["cuda_gpu_trace"])[0].name == CHUNK
    assert parse_nvtx_projection(texts["nvtx_gpu_proj_trace"])[0].name == "mamba-g1"
    monkeypatch.setattr(subprocess, "run", FakeRun((3, "", DIAGNOSTIC)))
    with pytest.raises(RuntimeError, match="profile exited 3"):
        nsys_report_texts(TARGET, base, ("cuda_gpu_trace",), nsys="/opt/nsight/nsys")
