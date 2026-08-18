"""Nsight Systems driver: command construction, the duration unit, and the trace.

The profiler binary is absent on this host and on the verification fleet, so no
test launches ``nsys``. Every pure function is driven with fixture text held in
this module, and :func:`run_nsys` is exercised through a fake
:func:`subprocess.run` that records its argv.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

from slinoss.perf.nsys import (
    duration_scale_ns,
    nsys_profile_command,
    nsys_stats_command,
    parse_gpu_trace,
    run_nsys,
)

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


def test_profile_command_overrides() -> None:
    got = nsys_profile_command(
        TARGET, Path("/tmp/run"), nsys="/opt/nsight/nsys", trace="cuda,nvtx"
    )
    assert got[0] == "/opt/nsight/nsys"
    assert got[2] == "--trace=cuda,nvtx"
    # The target is the tail, immediately after the output path.
    assert got[-len(TARGET) :] == list(TARGET)
    assert got[-len(TARGET) - 2 : -len(TARGET)] == ["-o", "/tmp/run"]


def test_profile_command_needs_a_target() -> None:
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


def test_stats_command_overrides() -> None:
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


@pytest.mark.parametrize(
    ("column", "want"),
    [
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
    ],
)
def test_duration_scale_ns(column: str, want: float) -> None:
    assert duration_scale_ns(column) == want


@pytest.mark.parametrize("column", ["Duration", "Duration ()", "Duration (fortnight)"])
def test_duration_scale_ns_rejects(column: str) -> None:
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
    assert [k.kernel for k in trace.kernels] == [CHUNK, CPASYNC]


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


def test_parse_gpu_trace_orders_kernels_by_descending_duration() -> None:
    got = parse_gpu_trace(GPU_TRACE_CSV).kernels
    assert [k.duration_us for k in got] == sorted(
        (k.duration_us for k in got), reverse=True
    )
    assert all(not k.kernel.startswith("[CUDA") for k in got)


def test_parse_gpu_trace_reads_a_microsecond_column() -> None:
    trace = parse_gpu_trace(MICROSECOND_CSV)
    assert trace.kernel(CHUNK).duration_us == pytest.approx(41.28)


def test_parse_gpu_trace_strips_thousands_separators() -> None:
    trace = parse_gpu_trace(GROUPED_CSV)
    assert trace.kernel(CHUNK).duration_us == pytest.approx(1048.576)


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


def test_parse_gpu_trace_needs_a_header() -> None:
    with pytest.raises(ValueError, match="no cuda_gpu_trace header in nsys output"):
        parse_gpu_trace(NO_HEADER_CSV, label="so3ssd fwd")


def test_parse_gpu_trace_needs_a_duration_column() -> None:
    with pytest.raises(ValueError, match="no 'duration' column"):
        parse_gpu_trace(NO_DURATION_COLUMN_CSV)


def test_parse_gpu_trace_needs_a_name_column() -> None:
    with pytest.raises(ValueError, match="no 'name' column"):
        parse_gpu_trace(NO_NAME_COLUMN_CSV)


def test_parse_gpu_trace_rejects_an_empty_window() -> None:
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
    trace = run_nsys(TARGET, base)
    assert trace.report_path == str(tmp_path / "run.standard.fwd.nsys-rep")
    assert fake.commands[1][-1] == trace.report_path


def test_run_nsys_raises_on_a_failed_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeRun((3, "", DIAGNOSTIC))
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as caught:
        run_nsys(TARGET, tmp_path / "profile-op")
    message = str(caught.value)
    assert "profile exited 3" in message
    assert "diagnostic 15" in message
    assert "diagnostic 04" in message
    assert "diagnostic 03" not in message  # the tail is the last twelve lines
    assert len(fake.commands) == 1  # the stats command never ran


def test_run_nsys_raises_on_failed_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeRun((0, PROFILE_LOG, ""), (4, "", "** ERROR: report file not found"))
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as caught:
        run_nsys(TARGET, tmp_path / "profile-op")
    assert "stats exited 4" in str(caught.value)
    assert "report file not found" in str(caught.value)
    assert len(fake.commands) == 2


def test_run_nsys_falls_back_to_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess, "run", FakeRun((1, "** ERROR: no CUDA device present", ""))
    )
    with pytest.raises(RuntimeError, match="no CUDA device present"):
        run_nsys(TARGET, tmp_path / "profile-op")
