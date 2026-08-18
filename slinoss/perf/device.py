"""Device identity, clock state, and sharing, queried at runtime.

Nothing here is hardcoded per architecture. Shared-memory capacity, SM count,
and register file size are read from the device, so a report is valid on the
part it was taken on and carries the evidence to prove which part that was.

Two conditions the numbers depend on and neither the code nor the shape controls
are probed and stamped: whether the clock is pinned, and whether anything else
was on the device. Both fail towards the pessimistic reading, so a probe that
does not run never produces a report claiming a locked clock or an idle part.

No spec-sheet peak appears in this module. A ceiling divides a measurement, so
it is itself measured; see :mod:`slinoss.perf.ceiling`. A modelled peak beside a
measured rate is the exact adjacency the schema in :mod:`slinoss.perf.units`
forbids.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, cast

import torch

from slinoss.perf.units import (
    INVARIANT,
    Bytes,
    Count,
    Mebibytes,
    Megahertz,
    Percent,
    PerfRecord,
)

__all__ = [
    "ClockPolicy",
    "ComputeAppsQuery",
    "Contention",
    "DeviceInfo",
    "SmiQuery",
    "clock_policy",
    "compute_apps_query",
    "contention",
    "device_info",
    "device_ordinal",
    "smi_query",
]

SmiQuery = Callable[[Sequence[str], int], "str | None"]
"""Reads comma-separated ``nvidia-smi`` query fields, or None if unavailable."""

ComputeAppsQuery = Callable[[int], "str | None"]
"""Reads one device's compute processes as ``pid, used_memory`` lines.

Empty means the device has none. None means the probe did not run, which is a
different fact and is reported as one.
"""


def device_ordinal(device: torch.device) -> int:
    """Ordinal a device resolves to, or ``-1`` when it has none.

    ``torch.device("cuda")`` carries ``index`` ``None`` at runtime and means the
    current device, though the type stub declares the field non-optional. A CPU
    device has no ordinal at all.

    Args:
        device: Any device.

    Returns:
        The ordinal, the current CUDA device's ordinal for an index-less CUDA
        device, or ``-1``.
    """
    index = cast("int | None", device.index)
    if index is not None:
        return index
    if device.type == "cuda":
        return torch.cuda.current_device()
    return -1


def smi_query(fields: Sequence[str], index: int) -> str | None:
    """Run one ``nvidia-smi`` query.

    Args:
        fields: Query field names, passed to ``--query-gpu``.
        index: Device index.

    Returns:
        The single output line, or None if ``nvidia-smi`` is missing, times out,
        or exits nonzero.
    """
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    line = done.stdout.strip().splitlines()
    return line[0] if line else None


def compute_apps_query(index: int) -> str | None:
    """Read one device's compute processes.

    Args:
        index: Device index.

    Returns:
        One ``pid, used_memory`` line per process with the memory in mebibytes,
        the empty string when the device carries none, or None if ``nvidia-smi``
        is missing, times out, or exits nonzero.
    """
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


@dataclass(frozen=True)
class ClockPolicy(PerfRecord):
    """Whether the SM clock is pinned, and to what.

    Locking is a privileged operation and is denied on the verification fleet, so
    the honest value of ``locked`` is False and every report carries the stamp.
    An unlocked clock is not a reason to skip a measurement; it is a reason to
    report the spread and to refuse a delta smaller than it.

    Attributes:
        locked: True only if an applications clock is set and active.
        sm_clock_mhz: SM clock at the moment of the query.
        max_sm_clock_mhz: Maximum SM clock the device reports.
        detail: What was probed, verbatim, including a probe failure.
    """

    locked: bool
    sm_clock_mhz: Annotated[Megahertz, INVARIANT]
    max_sm_clock_mhz: Annotated[Megahertz, INVARIANT]
    detail: str

    @property
    def stamp(self) -> str:
        """One-word clock state for a report header."""
        return f"locked at {self.sm_clock_mhz:.0f} MHz" if self.locked else "unlocked"


def _as_float(text: str) -> float:
    """A queried field as a number, or zero when the device answered ``[N/A]``."""
    try:
        return float(text.strip())
    except ValueError:
        return 0.0


def _as_mhz(text: str) -> Megahertz:
    return Megahertz(_as_float(text))


def clock_policy(index: int = 0, query: SmiQuery = smi_query) -> ClockPolicy:
    """Probe the clock state of one device.

    A failed probe reports unlocked. That is the conservative direction: it keeps
    the spread discipline on rather than silently claiming a pinned clock.

    Args:
        index: Device index.
        query: Injected ``nvidia-smi`` reader. Tests supply their own.

    Returns:
        The clock policy, with the raw probe text in ``detail``.
    """
    fields = (
        "clocks.sm",
        "clocks.max.sm",
        "clocks.applications.graphics",
        "clocks_throttle_reasons.applications_clocks_setting",
    )
    line = query(fields, index)
    if line is None:
        return ClockPolicy(
            locked=False,
            sm_clock_mhz=Megahertz(0.0),
            max_sm_clock_mhz=Megahertz(0.0),
            detail="nvidia-smi unavailable",
        )
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(fields):
        return ClockPolicy(
            locked=False,
            sm_clock_mhz=Megahertz(0.0),
            max_sm_clock_mhz=Megahertz(0.0),
            detail=f"unparsed nvidia-smi output: {line!r}",
        )
    current, maximum, applications, throttle = parts
    locked = throttle.lower() == "active" and _as_mhz(applications) > 0.0
    return ClockPolicy(
        locked=locked,
        sm_clock_mhz=_as_mhz(current),
        max_sm_clock_mhz=_as_mhz(maximum),
        detail=(
            f"clocks.sm={current} clocks.max.sm={maximum} "
            f"clocks.applications.graphics={applications} "
            f"applications_clocks_setting={throttle}"
        ),
    )


@dataclass(frozen=True)
class Contention(PerfRecord):
    """Whether anything else was running on the device.

    A shared device is a measurement condition on the order of the effect being
    measured: one foreign process on this fleet moved the same median from 53,092
    us to 123,890 us, a factor of 2.33, while run-to-run scatter stayed at 0.1%
    either way. Reproducibility therefore says nothing about contention, and the
    only defence is to report it.

    The probe is a point sample taken outside the timed region with the device
    drained, so it catches a process that outlives the measurement and misses one
    that arrives and leaves inside it. The second case is what ``spread_pct`` is
    for.

    ``exclusive`` is False when the probe did not run. Unknown is not exclusive,
    and claiming exclusivity off a probe that failed is the one direction that
    turns a contended median into a clean-looking one.

    Attributes:
        probed: True if both queries returned something parseable.
        foreign_process_count: Compute processes on the device other than this
            one. Zero when the probe failed, which is why it is not the field to
            read.
        foreign_memory_mib: Device memory those processes hold, as reported.
        utilization_pct: Device utilization at probe time. The probe runs with
            this process idle, so a nonzero reading is somebody else.
        detail: What was probed, verbatim, including a probe failure.
    """

    probed: bool
    foreign_process_count: Annotated[Count, INVARIANT]
    foreign_memory_mib: Annotated[Mebibytes, INVARIANT]
    utilization_pct: Annotated[Percent, INVARIANT]
    detail: str

    @property
    def exclusive(self) -> bool:
        """True only if the probe ran and found nothing else on the device."""
        return self.probed and self.foreign_process_count == 0

    @property
    def stamp(self) -> str:
        """One-line sharing state for a report header."""
        if not self.probed:
            return "sharing unknown"
        if self.exclusive:
            return "exclusive"
        plural = "" if self.foreign_process_count == 1 else "es"
        return (
            f"shared with {self.foreign_process_count} process{plural} holding "
            f"{self.foreign_memory_mib:,.0f} MiB at "
            f"{self.utilization_pct:.0f}% utilization"
        )


def _unknown(detail: str) -> Contention:
    return Contention(
        probed=False,
        foreign_process_count=Count(0),
        foreign_memory_mib=Mebibytes(0.0),
        utilization_pct=Percent(0.0),
        detail=detail,
    )


def contention(
    index: int = 0,
    *,
    apps: ComputeAppsQuery = compute_apps_query,
    query: SmiQuery = smi_query,
    own_pid: int | None = None,
) -> Contention:
    """Probe what else is running on one device.

    Args:
        index: Device index.
        apps: Injected compute-process reader. Tests supply their own.
        query: Injected ``nvidia-smi`` reader for utilization and memory.
        own_pid: Process to exclude. Defaults to this one, which holds a context
            on the device it measures and would otherwise read as a competitor.

    Returns:
        The contention record, with the raw probe text in ``detail``.
    """
    mine = os.getpid() if own_pid is None else own_pid
    text = apps(index)
    fields = ("utilization.gpu", "memory.used")
    line = query(fields, index)
    if text is None or line is None:
        return _unknown("nvidia-smi unavailable")
    gpu = [p.strip() for p in line.split(",")]
    if len(gpu) != len(fields):
        return _unknown(f"unparsed nvidia-smi output: {line!r}")
    rows = [row for row in text.splitlines() if row.strip()]
    foreign: list[tuple[int, float]] = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 2:
            return _unknown(f"unparsed nvidia-smi output: {row!r}")
        pid, used = parts
        if int(_as_float(pid)) != mine:
            foreign.append((int(_as_float(pid)), _as_float(used)))
    utilization, memory = gpu
    return Contention(
        probed=True,
        foreign_process_count=Count(len(foreign)),
        foreign_memory_mib=Mebibytes(sum(used for _, used in foreign)),
        utilization_pct=Percent(_as_float(utilization)),
        detail=(
            f"compute apps {len(rows)} (own pid {mine} excluded) "
            f"foreign pids {[pid for pid, _ in foreign]} "
            f"utilization.gpu={utilization} memory.used={memory}"
        ),
    )


@dataclass(frozen=True)
class DeviceInfo(PerfRecord):
    """What the device reports about itself.

    ``smem_optin_per_block_bytes`` is the number a kernel may opt into with a
    carveout, and is the only shared-memory budget worth asserting against.
    ``smem_per_block_bytes`` is the default limit, which is smaller and is not
    the budget.

    Attributes:
        name: Marketing name, as reported.
        capability: Compute capability, ``major.minor``.
        sm_count: Streaming multiprocessors. Twice this is the block-count floor.
        warp_thread_count: Threads per warp.
        max_threads_per_sm_count: Resident thread ceiling per SM.
        regs_per_sm_count: Register file size per SM, in registers.
        smem_per_block_bytes: Default per-block shared-memory limit.
        smem_optin_per_block_bytes: Per-block limit reachable with a carveout.
        smem_per_sm_bytes: Shared memory per SM.
        l2_bytes: L2 cache size.
        total_memory_bytes: Device memory.
        clocks: Clock state at probe time.
        sharing: What else was on the device at probe time.
    """

    name: str
    capability: str
    sm_count: Annotated[Count, INVARIANT]
    warp_thread_count: Annotated[Count, INVARIANT]
    max_threads_per_sm_count: Annotated[Count, INVARIANT]
    regs_per_sm_count: Annotated[Count, INVARIANT]
    smem_per_block_bytes: Annotated[Bytes, INVARIANT]
    smem_optin_per_block_bytes: Annotated[Bytes, INVARIANT]
    smem_per_sm_bytes: Annotated[Bytes, INVARIANT]
    l2_bytes: Annotated[Bytes, INVARIANT]
    total_memory_bytes: Annotated[Bytes, INVARIANT]
    clocks: ClockPolicy
    sharing: Contention

    @property
    def block_floor_count(self) -> Count:
        """Fewest blocks a non-serial kernel may launch: twice the SM count."""
        return Count(2 * self.sm_count)


def device_info(
    index: int = 0,
    query: SmiQuery = smi_query,
    apps: ComputeAppsQuery = compute_apps_query,
) -> DeviceInfo:
    """Read the identity, limits, clock state, and sharing of one CUDA device.

    Args:
        index: Device index.
        query: Injected ``nvidia-smi`` reader for the clock and sharing probes.
        apps: Injected compute-process reader for the sharing probe.

    Returns:
        The device record.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("device_info needs CUDA")
    props = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        name=props.name,
        capability=f"{props.major}.{props.minor}",
        sm_count=Count(props.multi_processor_count),
        warp_thread_count=Count(props.warp_size),
        max_threads_per_sm_count=Count(props.max_threads_per_multi_processor),
        regs_per_sm_count=Count(props.regs_per_multiprocessor),
        smem_per_block_bytes=Bytes(props.shared_memory_per_block),
        smem_optin_per_block_bytes=Bytes(props.shared_memory_per_block_optin),
        smem_per_sm_bytes=Bytes(props.shared_memory_per_multiprocessor),
        l2_bytes=Bytes(props.L2_cache_size),
        total_memory_bytes=Bytes(props.total_memory),
        clocks=clock_policy(index, query),
        sharing=contention(index, apps=apps, query=query),
    )
