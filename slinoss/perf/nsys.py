"""Nsight Systems driver: the GPU trace, split by work kind.

NCU replays a kernel in isolation, so it answers what a kernel costs but not what
the step spends. NSYS traces the real launch stream, so it answers where the step
went. The two are cross-checked against each other and against the CUDA-event
wall in :mod:`slinoss.perf.report`; a report that cannot make them agree refuses
to emit.

Copies and fills are parsed out separately rather than folded into the kernel
total. A staging copy or an ``aten::fill_`` on a hot path is a defect under the
kernel rules, and a defect that has been summed into a kernel total is invisible.

The duration unit is read from the CSV header's own parenthetical, not assumed.
NSYS has emitted this column in nanoseconds and in microseconds across versions.

:func:`parse_gpu_trace` keeps durations and drops start times, which is all a
per-kernel table needs. Device idle needs the start times: a sum of durations
cannot say whether the device was executing between two launches, and the
difference between a step's wall and its kernel sum is idle plus event placement
plus host time in unknown proportion. :func:`parse_gpu_events` keeps the whole
timeline and :func:`occupancy` splits one window of it into busy and idle, by
union rather than by sum, because two streams can overlap and a sum of overlapping
intervals exceeds the window it sits in.
"""

from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

from slinoss.perf.tools import resolve_tool
from slinoss.perf.units import (
    MEDIAN,
    SUM,
    Count,
    Microseconds,
    Nanoseconds,
    Percent,
    PerfRecord,
    Spread,
    pct_of,
    us_from_ns,
)

__all__ = [
    "GpuEvent",
    "NsysKernel",
    "NsysTrace",
    "NvtxSpan",
    "Occupancy",
    "duration_scale_ns",
    "events_within",
    "nsys_profile_command",
    "nsys_report_texts",
    "nsys_stats_command",
    "occupancy",
    "parse_gpu_events",
    "parse_gpu_trace",
    "parse_nvtx_projection",
    "repeat_windows",
    "run_nsys",
]

_MEMCPY: Final = "[CUDA memcpy"
_MEMSET: Final = "[CUDA memset"


def nsys_profile_command(
    argv: Sequence[str],
    base: Path,
    *,
    nsys: str = "nsys",
    trace: str = "cuda",
) -> list[str]:
    """Build the profile command.

    The capture range is the CUDA profiler API, so only
    :func:`slinoss.perf.capture.profiler_window` is traced and warmup stays out
    of the totals.

    The target follows the flags directly. ``nsys`` takes no ``--`` separator: it
    parses one as an empty long option and exits with an ambiguous-option error.

    Args:
        argv: The target command, already split.
        base: Output path without the ``.nsys-rep`` suffix.
        nsys: Path to the ``nsys`` binary.
        trace: Value for ``--trace``.

    Returns:
        The full argv.

    Raises:
        ValueError: If the target command is empty.
    """
    if not argv:
        raise ValueError("nsys needs a target command")
    return [
        nsys,
        "profile",
        f"--trace={trace}",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        str(base),
        *argv,
    ]


def nsys_stats_command(
    report: Path,
    *,
    nsys: str = "nsys",
    name: str = "cuda_gpu_trace",
) -> list[str]:
    """Build the stats command that exports one report as CSV.

    Args:
        report: Path to the ``.nsys-rep`` file.
        nsys: Path to the ``nsys`` binary.
        name: Report name.

    Returns:
        The full argv.
    """
    return [
        nsys,
        "stats",
        "--report",
        name,
        "--format",
        "csv",
        "--force-export=true",
        str(report),
    ]


_NS_PER: Final[dict[str, float]] = {
    "ns": 1.0,
    "nsec": 1.0,
    "us": 1e3,
    "usec": 1e3,
    "µs": 1e3,
    "ms": 1e6,
    "msec": 1e6,
    "s": 1e9,
    "sec": 1e9,
}


def duration_scale_ns(column: str) -> float:
    """Factor taking a duration column's values to nanoseconds.

    Args:
        column: The header cell, for instance ``Duration (ns)``.

    Returns:
        The multiplier.

    Raises:
        ValueError: If the header carries no unit or an unrecognized one.
    """
    _, _, tail = column.partition("(")
    unit = tail.partition(")")[0].strip().lower()
    if unit not in _NS_PER:
        raise ValueError(f"column {column!r} carries no recognized duration unit")
    return _NS_PER[unit]


@dataclass(frozen=True)
class NsysKernel(PerfRecord):
    """One kernel's contribution to the traced window.

    Attributes:
        kernel: Kernel name as NSYS reports it.
        launch_count: Launches in the window.
        duration_us: Summed duration over those launches.
        duration: Per-launch dispersion. A wide spread on one kernel name means
            the launches are not the same work and the sum is a mixture.
        share_pct: Share of the traced device time.
    """

    kernel: str
    launch_count: Annotated[Count, SUM]
    duration_us: Annotated[Microseconds, SUM]
    duration: Spread
    share_pct: Annotated[Percent, MEDIAN]


@dataclass(frozen=True)
class NsysTrace(PerfRecord):
    """The traced GPU timeline, split by work kind.

    Attributes:
        label: What was traced.
        report_path: The ``.nsys-rep`` the numbers came from.
        kernel_sum_duration_us: Summed kernel time.
        memcpy_sum_duration_us: Summed copy time. Nonzero on a hot path is a
            defect, not a datum.
        memset_sum_duration_us: Summed fill time. Same.
        memcpy_count: Copies traced.
        memset_count: Fills traced.
        kernels: One record per kernel name, ordered by descending duration.
    """

    label: str
    report_path: str
    kernel_sum_duration_us: Annotated[Microseconds, SUM]
    memcpy_sum_duration_us: Annotated[Microseconds, SUM]
    memset_sum_duration_us: Annotated[Microseconds, SUM]
    memcpy_count: Annotated[Count, SUM]
    memset_count: Annotated[Count, SUM]
    kernels: tuple[NsysKernel, ...]

    @property
    def device_sum_duration_us(self) -> Microseconds:
        """Kernels, copies, and fills together."""
        return Microseconds(
            self.kernel_sum_duration_us
            + self.memcpy_sum_duration_us
            + self.memset_sum_duration_us
        )

    def kernel(self, name: str) -> NsysKernel:
        """Look up one kernel by exact name.

        Raises:
            KeyError: If the trace has no such kernel.
        """
        for entry in self.kernels:
            if entry.kernel == name:
                return entry
        raise KeyError(f"no kernel {name!r} in {self.label!r}")


def _column(columns: Sequence[str], wanted: str) -> str:
    for column in columns:
        if column.strip().lower().startswith(wanted):
            return column
    raise ValueError(f"nsys CSV has no {wanted!r} column, got {list(columns)}")


def _column_with(columns: Sequence[str], *needles: str) -> str:
    """The one column whose header contains every needle, spaces ignored.

    ``nvtx_gpu_proj_trace`` carries four duration columns and four start columns,
    two of them the projected pair this module wants, so a prefix match on
    ``duration`` is ambiguous there in a way it is not on ``cuda_gpu_trace``.

    Args:
        columns: The header cells.
        *needles: Lowercase fragments, matched against the header with its spaces
            removed.

    Returns:
        The header cell.

    Raises:
        ValueError: If no column contains every needle.
    """
    for column in columns:
        flat = column.strip().lower().replace(" ", "")
        if all(needle in flat for needle in needles):
            return column
    raise ValueError(
        f"nsys CSV has no {'+'.join(needles)!r} column, got {list(columns)}"
    )


def _csv_body(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "," in line and "duration" in lowered and "name" in lowered:
            return "".join(lines[index:])
    return ""


def parse_gpu_trace(text: str, *, label: str = "", report_path: str = "") -> NsysTrace:
    """Parse ``cuda_gpu_trace`` CSV into per-kernel and per-kind totals.

    Args:
        text: ``nsys stats`` stdout, preamble included.
        label: What was traced.
        report_path: The report the CSV came from.

    Returns:
        The trace.

    Raises:
        ValueError: If the output carries no trace header, if the duration column
            has no unit, or if the window traced no device work. An empty window
            means the capture range never opened, which is a broken run rather
            than a fast one.
    """
    body = _csv_body(text)
    if not body:
        raise ValueError(f"no cuda_gpu_trace header in nsys output for {label!r}")
    reader = csv.DictReader(io.StringIO(body))
    columns = reader.fieldnames or []
    duration_column = _column(columns, "duration")
    name_column = _column(columns, "name")
    scale = duration_scale_ns(duration_column)
    kernels: dict[str, list[Microseconds]] = {}
    copies: list[Microseconds] = []
    fills: list[Microseconds] = []
    for row in reader:
        raw = (row.get(duration_column) or "").strip().replace(",", "")
        name = (row.get(name_column) or "").strip()
        if not raw or not name:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        duration = us_from_ns(Nanoseconds(value * scale))
        if name.startswith(_MEMCPY):
            copies.append(duration)
        elif name.startswith(_MEMSET):
            fills.append(duration)
        else:
            kernels.setdefault(name, []).append(duration)
    if not kernels and not copies and not fills:
        raise ValueError(
            f"nsys traced no device work for {label!r}; the capture range never opened"
        )
    kernel_sum = Microseconds(sum(sum(v) for v in kernels.values()))
    copy_sum = Microseconds(sum(copies))
    fill_sum = Microseconds(sum(fills))
    device_sum = kernel_sum + copy_sum + fill_sum
    entries = [
        NsysKernel(
            kernel=name,
            launch_count=Count(len(samples)),
            duration_us=Microseconds(sum(samples)),
            duration=Spread.of(samples),
            share_pct=pct_of(sum(samples), device_sum),
        )
        for name, samples in kernels.items()
    ]
    return NsysTrace(
        label=label,
        report_path=report_path,
        kernel_sum_duration_us=kernel_sum,
        memcpy_sum_duration_us=copy_sum,
        memset_sum_duration_us=fill_sum,
        memcpy_count=Count(len(copies)),
        memset_count=Count(len(fills)),
        kernels=tuple(sorted(entries, key=lambda k: k.duration_us, reverse=True)),
    )


# ---------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuEvent:
    """One device operation, with its place on the traced timeline.

    Attributes:
        name: Kernel, copy, or fill name as NSYS reports it.
        start_us: Start, on the trace's own clock. Comparable only against other
            events of the same trace.
        duration_us: Duration.
        stream: Stream identifier as a string, empty when the column is absent.
            Two events on two streams can overlap, which is why a busy total over
            these is a union and never a sum.
    """

    name: str
    start_us: Microseconds
    duration_us: Microseconds
    stream: str = ""

    @property
    def end_us(self) -> Microseconds:
        """One past the last microsecond the device was on this event."""
        return Microseconds(self.start_us + self.duration_us)


@dataclass(frozen=True)
class NvtxSpan:
    """One NVTX range projected onto the GPU timeline.

    The projection is NSYS's: the range starts at the first device operation it
    encloses and ends at the last. So the span is what the device spent on the
    range and excludes the host time before the range's first launch reached the
    device, which is the term a subtraction against a CUDA-event interval isolates.

    The push-to-pop interval is kept beside it. Those two answer different
    questions: the host interval is what the thread spent enqueueing, the projected
    interval is what the device spent executing, and which is the larger says which
    side the range is bound by. A subtraction of a kernel sum from a wall says
    neither.

    The projection is per thread. A range pushed on one thread projects only the
    launches that thread made, so a PyTorch backward -- which the autograd engine
    runs on its own worker thread -- projects into no range at all, and
    ``gpu_op_count`` then counts the forward's launches alone. Use the projected
    span to attribute device work to a range only after checking that count against
    the launches the range should hold. ``host_start_us`` and ``host_duration_us``
    have no such limit: the push and the pop are the pushing thread's own, and a
    blocking call between them is inside the interval whichever thread served it.

    Attributes:
        name: Range name, as pushed.
        start_us: Projected start.
        duration_us: Projected duration.
        host_start_us: Push, on the same clock.
        host_duration_us: Push to pop.
        gpu_op_count: Device operations NSYS projected into the range.
    """

    name: str
    start_us: Microseconds
    duration_us: Microseconds
    host_start_us: Microseconds
    host_duration_us: Microseconds
    gpu_op_count: Count

    @property
    def end_us(self) -> Microseconds:
        """Projected end."""
        return Microseconds(self.start_us + self.duration_us)

    @property
    def host_end_us(self) -> Microseconds:
        """Pop."""
        return Microseconds(self.host_start_us + self.host_duration_us)


@dataclass(frozen=True)
class Occupancy(PerfRecord):
    """How much of one window of the timeline the device was executing.

    ``busy_us + idle_us == span_us`` by construction, so the record is a partition
    of the window and not two independent measurements of it.

    Attributes:
        label: What window this covers.
        span_us: First start to last end.
        busy_us: Union of the event intervals. A union, so two overlapping streams
            contribute the wall they cover and not the sum of their durations.
        sum_duration_us: Sum of the event durations. Above ``busy_us`` exactly when
            events overlapped, so the difference is the concurrency in the window.
        idle_us: ``span_us - busy_us``. Device idle strictly inside the window,
            attributable to whatever did not have the next launch ready.
        idle_pct: ``idle_us`` over ``span_us``.
        event_count: Events in the window.
        gap_count: Idle intervals. Never more than ``event_count - 1``.
        max_gap_us: Longest single idle interval.
    """

    label: str
    span_us: Annotated[Microseconds, SUM]
    busy_us: Annotated[Microseconds, SUM]
    sum_duration_us: Annotated[Microseconds, SUM]
    idle_us: Annotated[Microseconds, SUM]
    idle_pct: Annotated[Percent, MEDIAN]
    event_count: Annotated[Count, SUM]
    gap_count: Annotated[Count, SUM]
    max_gap_us: Annotated[Microseconds, MEDIAN]


def parse_gpu_events(text: str) -> tuple[GpuEvent, ...]:
    """Parse ``cuda_gpu_trace`` CSV into a timeline, ordered by start.

    Kernels, copies, and fills are all kept: device idle is idle whatever the
    device would have been doing, so a window's busy total counts every kind. The
    split by kind stays in :func:`parse_gpu_trace`.

    Args:
        text: ``nsys stats`` stdout, preamble included.

    Returns:
        Every usable row, ordered by start time.

    Raises:
        ValueError: If the output carries no trace header, if a required column is
            absent or carries no recognized unit, or if no row was usable. An empty
            timeline means the capture range never opened.
    """
    body = _csv_body(text)
    if not body:
        raise ValueError("no cuda_gpu_trace header in nsys output")
    reader = csv.DictReader(io.StringIO(body))
    columns = reader.fieldnames or []
    start_column = _column(columns, "start")
    duration_column = _column(columns, "duration")
    name_column = _column(columns, "name")
    start_scale = duration_scale_ns(start_column)
    duration_scale = duration_scale_ns(duration_column)
    stream_column = next(
        (c for c in columns if c.strip().lower().startswith("strm")), None
    )
    events: list[GpuEvent] = []
    for row in reader:
        name = (row.get(name_column) or "").strip()
        start = _cell(row.get(start_column))
        duration = _cell(row.get(duration_column))
        if not name or start is None or duration is None:
            continue
        events.append(
            GpuEvent(
                name=name,
                start_us=us_from_ns(Nanoseconds(start * start_scale)),
                duration_us=us_from_ns(Nanoseconds(duration * duration_scale)),
                stream=(row.get(stream_column) or "").strip() if stream_column else "",
            )
        )
    if not events:
        raise ValueError("nsys traced no device work; the capture range never opened")
    return tuple(sorted(events, key=lambda e: e.start_us))


def parse_nvtx_projection(text: str) -> tuple[NvtxSpan, ...]:
    """Parse ``nvtx_gpu_proj_trace`` CSV, ordered by projected start.

    Args:
        text: ``nsys stats`` stdout, preamble included.

    Returns:
        One span per range instance. A range NSYS projected no device operation
        into is dropped: it has no interval on the GPU timeline to partition.

    Raises:
        ValueError: If the output carries no projection header, if a required
            column is absent or carries no recognized unit, or if no range
            projected onto the device. The last means the ranges were pushed
            outside the capture window, or NVTX was not traced.
    """
    body = _csv_body(text)
    if not body:
        raise ValueError("no nvtx_gpu_proj_trace header in nsys output")
    reader = csv.DictReader(io.StringIO(body))
    columns = reader.fieldnames or []
    start_column = _column_with(columns, "projected", "start")
    duration_column = _column_with(columns, "projected", "duration")
    host_start_column = _column_with(columns, "orig", "start")
    host_duration_column = _column_with(columns, "orig", "duration")
    name_column = _column(columns, "name")
    ops_column = next(
        (c for c in columns if "gpuops" in c.strip().lower().replace(" ", "")), None
    )
    start_scale = duration_scale_ns(start_column)
    duration_scale = duration_scale_ns(duration_column)
    host_start_scale = duration_scale_ns(host_start_column)
    host_duration_scale = duration_scale_ns(host_duration_column)
    spans: list[NvtxSpan] = []
    for row in reader:
        name = (row.get(name_column) or "").strip()
        start = _cell(row.get(start_column))
        duration = _cell(row.get(duration_column))
        host_start = _cell(row.get(host_start_column))
        host_duration = _cell(row.get(host_duration_column))
        if not name or start is None or duration is None:
            continue
        ops = _cell(row.get(ops_column)) if ops_column else None
        spans.append(
            NvtxSpan(
                name=name,
                start_us=us_from_ns(Nanoseconds(start * start_scale)),
                duration_us=us_from_ns(Nanoseconds(duration * duration_scale)),
                host_start_us=us_from_ns(
                    Nanoseconds((host_start or 0.0) * host_start_scale)
                ),
                host_duration_us=us_from_ns(
                    Nanoseconds((host_duration or 0.0) * host_duration_scale)
                ),
                gpu_op_count=Count(int(ops) if ops is not None else 0),
            )
        )
    if not spans:
        raise ValueError(
            "nsys projected no NVTX range onto the device; either the ranges were "
            "pushed outside the capture window or --trace did not include nvtx"
        )
    return tuple(sorted(spans, key=lambda s: s.start_us))


def events_within(
    events: Sequence[GpuEvent], start_us: float, end_us: float
) -> tuple[GpuEvent, ...]:
    """Events whose start falls in ``[start_us, end_us]``.

    Membership is by start and not by overlap, so every event belongs to exactly
    one of a set of disjoint windows and no duration is counted twice.

    Args:
        events: The timeline.
        start_us: Window start, inclusive.
        end_us: Window end, inclusive.

    Returns:
        The events, in the order given.
    """
    return tuple(e for e in events if start_us <= e.start_us <= end_us)


def repeat_windows(
    events: Sequence[GpuEvent], count: int
) -> tuple[tuple[GpuEvent, ...], ...]:
    """Split a timeline of ``count`` identical repetitions into one window each.

    A loop that runs one callable ``count`` times launches the same sequence of
    device operations ``count`` times, so the timeline partitions into equal blocks
    of consecutive events. Segmenting by block index rather than by the largest gaps
    keeps a gap inside one repetition from being read as a boundary between two.

    The name sequence of every block must match the first block's. That check is
    what makes the segmentation a measurement rather than an assumption: a loop that
    launched a different kernel on some iteration, or a profiler that dropped a row,
    fails it.

    Args:
        events: The whole timeline, from :func:`parse_gpu_events`.
        count: Repetitions the timeline holds.

    Returns:
        One tuple of events per repetition, in timeline order.

    Raises:
        ValueError: If ``count`` is not positive, if the event count is not a
            multiple of ``count``, or if the blocks do not carry one name sequence.
    """
    if count <= 0:
        raise ValueError(f"repeat_windows needs a positive count, got {count}")
    ordered = sorted(events, key=lambda e: e.start_us)
    if len(ordered) % count != 0:
        raise ValueError(
            f"{len(ordered)} device operations do not divide into {count} "
            "repetitions; the traced window holds work the loop did not launch"
        )
    width = len(ordered) // count
    if width == 0:
        raise ValueError(f"{len(ordered)} device operations over {count} repetitions")
    blocks = tuple(tuple(ordered[i * width : (i + 1) * width]) for i in range(count))
    wanted = tuple(e.name for e in blocks[0])
    for index, block in enumerate(blocks[1:], start=1):
        got = tuple(e.name for e in block)
        if got != wanted:
            differs = next(
                i for i, (x, y) in enumerate(zip(wanted, got, strict=True)) if x != y
            )
            raise ValueError(
                f"repetition {index} launched {got[differs]!r} where repetition 0 "
                f"launched {wanted[differs]!r}, at position {differs}"
            )
    return blocks


def occupancy(label: str, events: Sequence[GpuEvent]) -> Occupancy:
    """Split the window these events span into busy and idle.

    Args:
        label: What the window is.
        events: The events. Order does not matter.

    Returns:
        The partition.

    Raises:
        ValueError: If no events were given. An empty window has no span to
            partition, and reporting zero idle over zero span reads as a device
            that never waited.
    """
    if not events:
        raise ValueError(f"occupancy({label!r}) needs at least one event")
    ordered = sorted(events, key=lambda e: e.start_us)
    first = ordered[0].start_us
    cursor = first
    busy = 0.0
    gaps: list[float] = []
    for event in ordered:
        if event.start_us > cursor:
            gaps.append(event.start_us - cursor)
            cursor = event.start_us
        if event.end_us > cursor:
            busy += event.end_us - cursor
            cursor = event.end_us
    span = Microseconds(cursor - first)
    idle = Microseconds(span - busy)
    return Occupancy(
        label=label,
        span_us=span,
        busy_us=Microseconds(busy),
        sum_duration_us=Microseconds(sum(e.duration_us for e in ordered)),
        idle_us=idle,
        idle_pct=pct_of(idle, span),
        event_count=Count(len(ordered)),
        gap_count=Count(len(gaps)),
        max_gap_us=Microseconds(max(gaps, default=0.0)),
    )


def _cell(raw: str | None) -> float | None:
    """One numeric CSV cell, or None when it is blank or not a number."""
    text = (raw or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def nsys_report_texts(
    argv: Sequence[str],
    base: Path,
    reports: Sequence[str],
    *,
    nsys: str = "nsys",
    trace: str = "cuda",
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, str]:
    """Profile a command once and export several stats reports from that run.

    One profile, several exports: two profiles of one workload are two timelines,
    and a kernel table taken from one against a range projection taken from the
    other cannot be joined.

    Args:
        argv: The target command.
        base: Output path. ``.nsys-rep`` is appended to the whole name.
        reports: Stats report names, exported in order.
        nsys: Path to the ``nsys`` binary, or a bare name to resolve.
        trace: Value for ``--trace``. NVTX projection needs ``nvtx`` in it.
        cwd: Working directory for the target.
        timeout_s: Wall clock limit for each command.

    Returns:
        Report name to that report's stdout, verbatim.

    Raises:
        ValueError: If no report was named.
        ToolNotFoundError: If ``nsys`` resolves to nothing. Raised before the
            profile runs, so a missing profiler costs no measurement.
        RuntimeError: If any command exits nonzero. The message carries the tail of
            the tool's own diagnostic.
    """
    if not reports:
        raise ValueError("nsys_report_texts needs at least one report name")
    # Resolved once: every stats pass must read the report the profile pass wrote,
    # and two searches could pick two toolkits whose report formats differ.
    binary = resolve_tool(nsys)
    # nsys appends the suffix to the whole name it was given, so the reader must
    # append too. `with_suffix` replaces everything after the last dot, which for a
    # base like `out/run.1-standard-step` reads `out/run.nsys-rep` and fails after
    # the full profile has already been paid for.
    report = base.with_name(base.name + ".nsys-rep")
    commands = [nsys_profile_command(argv, base, nsys=binary, trace=trace)]
    commands += [nsys_stats_command(report, nsys=binary, name=name) for name in reports]
    out: dict[str, str] = {}
    for index, command in enumerate(commands):
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout_s,
            check=False,
        )
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError(
                f"{command[1]} exited {done.returncode}: " + " | ".join(tail)
            )
        if index:
            out[reports[index - 1]] = done.stdout
    return out


def run_nsys(
    argv: Sequence[str],
    base: Path,
    *,
    label: str = "",
    nsys: str = "nsys",
    trace: str = "cuda",
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> NsysTrace:
    """Profile a command and parse its GPU trace.

    Args:
        argv: The target command.
        base: Output path. ``.nsys-rep`` is appended to the whole name, not
            substituted for its last suffix.
        label: What is being traced.
        nsys: Path to the ``nsys`` binary, or a bare name to resolve through
            :func:`slinoss.perf.tools.resolve_tool`.
        trace: Value for ``--trace``.
        cwd: Working directory for the target.
        timeout_s: Wall clock limit for each of the two commands.

    Returns:
        The trace.

    Raises:
        ToolNotFoundError: If ``nsys`` is a bare name on neither PATH nor any CUDA
            bin directory. Raised before the profile runs, so a missing profiler
            costs no measurement.
        RuntimeError: If either command exits nonzero. The message carries the
            tail of the tool's own diagnostic.
    """
    texts = nsys_report_texts(
        argv,
        base,
        ("cuda_gpu_trace",),
        nsys=nsys,
        trace=trace,
        cwd=cwd,
        timeout_s=timeout_s,
    )
    report = base.with_name(base.name + ".nsys-rep")
    return parse_gpu_trace(
        texts["cuda_gpu_trace"], label=label, report_path=str(report)
    )
