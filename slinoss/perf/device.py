"""Device identity, clock state, and sharing, queried at runtime.

Nothing here is hardcoded per architecture. Shared-memory capacity, SM count,
and register file size are read from the device, so a report is valid on the
part it was taken on and carries the evidence to prove which part that was.

Two conditions the numbers depend on and neither the code nor the shape controls
are probed and stamped: whether the clock is pinned, and whether anything else
was on the device. Both fail towards the pessimistic reading, so a probe that
does not run never produces a report claiming a locked clock or an idle part.

Two index spaces meet here and only one crosses the module boundary. Every entry
point takes a torch ordinal; the ``nvidia-smi`` readers take a driver selector,
which :func:`smi_selector` resolves from the ordinal by UUID. Handing an ordinal
to the driver under ``CUDA_VISIBLE_DEVICES`` stamps one part while measuring
another, and the stamp reads clean because the probe succeeded.

No spec-sheet peak appears in this module. A ceiling divides a measurement, so
it is itself measured; see :mod:`slinoss.perf.ceiling`. A modelled peak beside a
measured rate is the exact adjacency the schema in :mod:`slinoss.perf.units`
forbids.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, cast

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
    "FOREIGN_MIB_FLOOR",
    "ClockPolicy",
    "ComputeAppsQuery",
    "ContendedDevice",
    "Contention",
    "ContentionProbe",
    "DeviceInfo",
    "SmiQuery",
    "await_exclusive",
    "clock_policy",
    "compute_apps_query",
    "contention",
    "device_info",
    "device_ordinal",
    "require_cuda",
    "smi_query",
    "smi_selector",
]

SmiQuery = Callable[[Sequence[str], str], "str | None"]
"""Reads comma-separated ``nvidia-smi`` query fields, or None if unavailable.

The second argument is a driver selector, never a torch ordinal; see
:func:`smi_selector`.
"""

ComputeAppsQuery = Callable[[str], "str | None"]
"""Reads one device's compute processes as ``pid, used_memory`` lines.

Empty means the device has none. None means the probe did not run, which is a
different fact and is reported as one.

The argument is a driver selector, never a torch ordinal; see
:func:`smi_selector`.
"""


def require_cuda(spec: str) -> torch.device:
    """Resolve a device string for a measurement driver, or refuse.

    One implementation for every driver. Each of them writes a report that names
    the part the numbers came from, divides by a measured ceiling, or reads a
    hardware counter, and none of those exists off CUDA. Refusing here costs a
    string comparison; refusing later costs the allocation and the warmup, and
    reports whichever probe reached for the ordinal first rather than the reason.

    Args:
        spec: Device string as given on the command line, ``cuda`` or ``cuda:N``.

    Returns:
        The device.

    Raises:
        RuntimeError: If the string is not a CUDA device, or if CUDA is
            unavailable.
    """
    device = torch.device(spec)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"{spec!r} is not a usable cuda device; the measurement drivers report "
            f"per-kernel counters against measured ceilings and have no host path"
        )
    return device


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


def smi_selector(ordinal: int) -> str:
    """Driver selector for a torch ordinal, for ``nvidia-smi --id=``.

    ``nvidia-smi`` numbers devices the way the driver does, and
    ``CUDA_VISIBLE_DEVICES`` renumbers torch's ordinals without renumbering the
    driver's. The ordinal a report names and the index a probe needs are
    therefore two different integers, and passing one for the other stamps a
    part the measurement did not run on. A UUID is in neither space and is
    correct whatever the variable holds, in whatever order, including when it
    names devices by UUID rather than by index.

    Args:
        ordinal: Torch device ordinal, as :func:`device_ordinal` returns.

    Returns:
        ``GPU-<uuid>`` when torch reports the part's UUID, else the entry
        ``CUDA_VISIBLE_DEVICES`` maps the ordinal to, verbatim, else the ordinal
        as a string. A selector the driver rejects fails the probe, and a failed
        probe reports unknown rather than a clean device.
    """
    try:
        uuid = torch.cuda.get_device_properties(ordinal).uuid
    except (AssertionError, AttributeError, RuntimeError):
        # No CUDA, no such ordinal, or a torch with no UUID on the properties.
        pass
    else:
        return f"GPU-{uuid}"
    visible = [
        item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    ]
    if 0 <= ordinal < len(visible) and visible[ordinal]:
        return visible[ordinal]
    return str(ordinal)


def smi_query(fields: Sequence[str], selector: str) -> str | None:
    """Run one ``nvidia-smi`` query.

    Args:
        fields: Query field names, passed to ``--query-gpu``.
        selector: Driver selector, as :func:`smi_selector` returns. A torch
            ordinal is not one.

    Returns:
        The single output line, or None if ``nvidia-smi`` is missing, times out,
        or exits nonzero.
    """
    cmd = [
        "nvidia-smi",
        f"--id={selector}",
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


def compute_apps_query(selector: str) -> str | None:
    """Read one device's compute processes.

    Args:
        selector: Driver selector, as :func:`smi_selector` returns. A torch
            ordinal is not one.

    Returns:
        One ``pid, used_memory`` line per process with the memory in mebibytes,
        the empty string when the device carries none, or None if ``nvidia-smi``
        is missing, times out, or exits nonzero.
    """
    cmd = [
        "nvidia-smi",
        f"--id={selector}",
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


def clock_policy(ordinal: int = 0, query: SmiQuery = smi_query) -> ClockPolicy:
    """Probe the clock state of one device.

    A failed probe reports unlocked. That is the conservative direction: it keeps
    the spread discipline on rather than silently claiming a pinned clock.

    Args:
        ordinal: Torch device ordinal. Resolved to a driver selector here, so the
            probe reads the part torch would run on.
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
    line = query(fields, smi_selector(ordinal))
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

    def quiet(self, *, ceiling_pct: float, mib_floor: float) -> bool:
        """True if nothing on the device can move an absolute duration.

        Exclusive is stricter than the measurement needs and stricter than the
        fleet allows. An idle interpreter with a CUDA context open is one foreign
        process forever, and requiring the count to be zero makes the gate
        unopenable rather than safe. What moves a duration is somebody else's
        kernels, which utilization reads, and somebody else's residency, which a
        context too small to hold a workload's tensors does not have.

        Args:
            ceiling_pct: Highest utilization this may report.
            mib_floor: Most foreign device memory this may report. Above it the
                probe is dirty however idle the device reads, because a resident
                workload evicts and can start at any time.

        Returns:
            Whether the probe is clean. A probe that did not run never is.
        """
        return (
            self.probed
            and self.utilization_pct <= ceiling_pct
            and (
                self.foreign_process_count == 0 or self.foreign_memory_mib <= mib_floor
            )
        )

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


def _unknown(selector: str, reason: str) -> Contention:
    return Contention(
        probed=False,
        foreign_process_count=Count(0),
        foreign_memory_mib=Mebibytes(0.0),
        utilization_pct=Percent(0.0),
        detail=f"device {selector}: {reason}",
    )


def contention(
    ordinal: int = 0,
    *,
    apps: ComputeAppsQuery = compute_apps_query,
    query: SmiQuery = smi_query,
    own_pid: int | None = None,
) -> Contention:
    """Probe what else is running on one device.

    Both probes read one driver selector resolved from the ordinal, so the record
    names the part the ordinal resolves to and not whichever device the driver
    happens to number the same.

    Args:
        ordinal: Torch device ordinal. Resolved to a driver selector here.
        apps: Injected compute-process reader. Tests supply their own.
        query: Injected ``nvidia-smi`` reader for utilization and memory.
        own_pid: Process to exclude. Defaults to this one, which holds a context
            on the device it measures and would otherwise read as a competitor.

    Returns:
        The contention record, naming the probed part and carrying the raw probe
        text in ``detail``.
    """
    mine = os.getpid() if own_pid is None else own_pid
    selector = smi_selector(ordinal)
    text = apps(selector)
    fields = ("utilization.gpu", "memory.used")
    line = query(fields, selector)
    if text is None or line is None:
        return _unknown(selector, "nvidia-smi unavailable")
    gpu = [p.strip() for p in line.split(",")]
    if len(gpu) != len(fields):
        return _unknown(selector, f"unparsed nvidia-smi output: {line!r}")
    rows = [row for row in text.splitlines() if row.strip()]
    foreign: list[tuple[int, float]] = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 2:
            return _unknown(selector, f"unparsed nvidia-smi output: {row!r}")
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
            f"device {selector}: compute apps {len(rows)} "
            f"(own pid {mine} excluded) "
            f"foreign pids {[pid for pid, _ in foreign]} "
            f"utilization.gpu={utilization} memory.used={memory}"
        ),
    )


ContentionProbe = Callable[[int], Contention]


class ContendedDevice(RuntimeError):
    """A run required an idle device and the device did not become idle."""


FOREIGN_MIB_FLOOR: Final[float] = 512.0
"""Foreign device memory the gate treats as holding no workload.

A bare CUDA context plus driver overhead is tens of MiB on this fleet; the jobs
that actually move a duration hold thousands. The floor sits between the two,
nearer the small side, so an idle context does not hold the gate shut and a
resident workload does.
"""


def await_exclusive(
    ordinal: int = 0,
    *,
    samples: int = 5,
    ceiling_pct: float = 5.0,
    mib_floor: float = FOREIGN_MIB_FLOOR,
    interval_s: float = 2.0,
    timeout_s: float = 600.0,
    probe: ContentionProbe = contention,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Contention:
    """Block until one device reads idle on consecutive probes.

    A point sample is not a gate. A foreign job between two of its own kernels
    reads exclusive once, so one clean probe admits exactly the contention the
    probe was added to exclude. ``samples`` consecutive clean probes are required
    and the count resets on any dirty one. Utilization is checked separately from
    the process count because a process may hold memory without running kernels,
    and because the count cannot attribute residency. The count is not checked at
    all: an interpreter with a context open is one foreign process for as long as
    it lives, so what it holds decides whether it counts. See
    :meth:`Contention.quiet`.

    Contention is waited out rather than stamped whenever the number being taken
    is an absolute duration. One foreign process on this fleet moved the same
    median by 2.33x while run-to-run scatter stayed at 0.1% either way, so the
    spread that rejects every other bad sample reports the contended run as
    reproducible. A paired delta inside one process cancels the common-mode part
    of that; a standalone baseline and a cross-implementation ratio do not.

    Args:
        ordinal: Torch device ordinal, resolved to a driver selector by the probe.
        samples: Consecutive clean probes required.
        ceiling_pct: Highest utilization a clean probe may report.
        mib_floor: Foreign device memory a clean probe may report. See
            :data:`FOREIGN_MIB_FLOOR` and :meth:`Contention.quiet`.
        interval_s: Delay between probes.
        timeout_s: Longest total wait before raising.
        probe: Injected contention probe. Tests supply their own.
        sleep: Injected delay.
        clock: Injected monotonic clock, in seconds.

    Returns:
        The last probe, clean by construction.

    Raises:
        ValueError: If ``samples`` is not positive.
        ContendedDevice: If the device did not read clean ``samples`` times in a
            row inside ``timeout_s``, naming the last state seen. Refusing to run
            is the point: the alternative is an absolute duration that carries
            somebody else's kernels and reports as reproducible.
    """
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    deadline = clock() + timeout_s
    clean = 0
    while True:
        last = probe(ordinal)
        idle = last.quiet(ceiling_pct=ceiling_pct, mib_floor=mib_floor)
        clean = clean + 1 if idle else 0
        if clean >= samples:
            return last
        if clock() >= deadline:
            raise ContendedDevice(
                f"device did not idle for {samples} consecutive probes inside "
                f"{timeout_s:.0f}s, holding the gate shut at foreign memory above "
                f"{mib_floor:,.0f} MiB or utilization above {ceiling_pct:.0f}%: "
                f"{last.stamp} ({last.detail})"
            )
        sleep(interval_s)


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
    ordinal: int = 0,
    query: SmiQuery = smi_query,
    apps: ComputeAppsQuery = compute_apps_query,
) -> DeviceInfo:
    """Read the identity, limits, clock state, and sharing of one CUDA device.

    Args:
        ordinal: Torch device ordinal. The properties come from torch and the
            clock and sharing probes from the driver, and the two number devices
            differently, so the probes resolve a selector rather than reuse this.
        query: Injected ``nvidia-smi`` reader for the clock and sharing probes.
        apps: Injected compute-process reader for the sharing probe.

    Returns:
        The device record.

    Raises:
        ValueError: If the ordinal is negative. :func:`device_ordinal` yields
            ``-1`` for a device that has no ordinal, so this is a CPU device
            arriving where a part is being named. Checked first, or the failure
            reports the absence of CUDA on a host that has it.
        RuntimeError: If CUDA is unavailable.
    """
    if ordinal < 0:
        raise ValueError(f"device_info needs a cuda ordinal, got {ordinal}")
    if not torch.cuda.is_available():
        raise RuntimeError("device_info needs CUDA")
    props = torch.cuda.get_device_properties(ordinal)
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
        clocks=clock_policy(ordinal, query),
        sharing=contention(ordinal, apps=apps, query=query),
    )
