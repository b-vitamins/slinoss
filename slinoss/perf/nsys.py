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
"""

from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

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
    "NsysKernel",
    "NsysTrace",
    "duration_scale_ns",
    "nsys_profile_command",
    "nsys_stats_command",
    "parse_gpu_trace",
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
        nsys: Path to the ``nsys`` binary.
        trace: Value for ``--trace``.
        cwd: Working directory for the target.
        timeout_s: Wall clock limit for each of the two commands.

    Returns:
        The trace.

    Raises:
        RuntimeError: If either command exits nonzero. The message carries the
            tail of the tool's own diagnostic.
    """
    # nsys appends the suffix to the whole name it was given, so the reader must
    # append too. `with_suffix` replaces everything after the last dot, which for a
    # base like `out/run.1-standard-step` reads `out/run.nsys-rep` and fails after
    # the full profile has already been paid for.
    report = base.with_name(base.name + ".nsys-rep")
    stdout = ""
    for command in (
        nsys_profile_command(argv, base, nsys=nsys, trace=trace),
        nsys_stats_command(report, nsys=nsys),
    ):
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
        stdout = done.stdout
    return parse_gpu_trace(stdout, label=label, report_path=str(report))
