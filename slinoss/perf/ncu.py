"""Nsight Compute driver: explicit metrics, one pass per table, units from NCU.

Three departures from the usual way of doing this, each closing a specific way a
report has lied before.

Metrics are requested by name with ``--metrics`` and parsed out of NCU's own CSV
by metric name. Section output is English labels in a human-oriented table;
matching those labels with a regular expression breaks silently when a row is
renamed, and a silently absent row reads as a zero. Every requested metric that
does not come back is listed in ``missing_metrics``, so a wrong metric name is a
loud failure on the first run.

Units come from NCU's ``Metric Unit`` column and are converted here, once. NCU
scales counters for display, so a byte counter can arrive as ``Mbyte`` and a
duration as ``usecond``. An unrecognized unit raises rather than defaulting to a
scale of one.

``--clock-control none`` is not optional. The default pins the clock to base for
the profiled kernel, which is not the clock the benchmark ran at, so the
resulting per-kernel durations do not compose into the measured step time.

``--cache-control none`` is not optional either, for two reasons. The default
flushes L1 and L2 before every replay, so every kernel is profiled cold while the
benchmark runs warm: measured on this fleet at 2,096,864 ns against 1,764,928 ns
over the same 816 launches, 18.8% of the total. And a cold cache inflates DRAM
traffic, which inflates achieved bandwidth, which lets a cache-resident kernel
pass a DRAM-bound bar it has no business passing. Warm is both the execution the
step time came from and the conservative side of the class verdict.

One NCU pass per table. Counters from different passes describe different
executions, so they do not share a row without a stated disagreement:
``pass_duration_spread_pct`` carries the duration disagreement between passes,
which is the replay-stability signal for everything else in the record.

:data:`SPILL_TABLE` is a ninth pass and is deliberately not in
:data:`NCU_TABLES`. Its two counters are a verdict input rather than a row in a
counter table: a register spill invalidates the byte model a DRAM-bound verdict
rests on, so it is judged, not printed. Keeping it out of the merge also keeps it
out of :class:`KernelCounters`, which is constructed outside this package; a field
added there would need a default, and a defaulted spill count reads as no spill
wherever a caller omits it. :func:`slinoss.perf.declared.floor_audit` requires a
:class:`SpillCounters` record for every kernel it judges instead, so a pass that
was never run fails loudly rather than passing everything.

The ``stall`` and ``sol`` tables answer two different questions and are two
tables for that reason. ``stall`` is the scheduler's view: how often it issued,
and every reason it did not, as percentages of warp-active cycles. ``sol`` is the
unit view: each engine against its own peak. A kernel far below the DRAM ceiling
whose dominant stall is ``long_scoreboard`` while ``issue_active_pct`` sits in
single digits is memory-latency bound with too few loads in flight, which no
counter in the other six tables distinguishes from a bandwidth bound.
"""

from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Final

from slinoss.perf.units import (
    INVARIANT,
    MEDIAN,
    SUM,
    Bytes,
    Count,
    GBPerSecond,
    Microseconds,
    Nanoseconds,
    Percent,
    PerfRecord,
    Ratio,
    gbs_from_bytes_us,
    pct_of,
    us_from_ns,
)

__all__ = [
    "NCU_TABLES",
    "REQUIRED_METRICS",
    "SOL_FIELDS",
    "SPILL_TABLE",
    "STALL_FIELDS",
    "STALL_REASONS",
    "KernelCounters",
    "NcuInvocation",
    "NcuPass",
    "NcuTable",
    "SpillCounters",
    "kernel_counters",
    "metric_scale",
    "ncu_command",
    "parse_ncu_csv",
    "run_ncu",
    "spill_counters",
    "stall_field",
    "stall_metric",
]


@dataclass(frozen=True)
class NcuTable:
    """One NCU pass: a name and the metrics it requests.

    Attributes:
        name: Table name, used as a report heading.
        metrics: Raw NCU metric names, requested verbatim.
    """

    name: str
    metrics: tuple[str, ...]


_DURATION: Final = "gpu__time_duration.sum"

_STALL_PREFIX: Final = "smsp__warp_issue_stalled_"
_STALL_SUFFIX: Final = "_per_warp_active.pct"

_ISSUE_ACTIVE: Final = "smsp__issue_active.avg.pct_of_peak_sustained_active"

_SM_SOL: Final = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_MEMORY_SOL: Final = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
_L1TEX_SOL: Final = "l1tex__throughput.avg.pct_of_peak_sustained_active"
_L2_SOL: Final = "lts__throughput.avg.pct_of_peak_sustained_active"

STALL_REASONS: Final[tuple[str, ...]] = (
    "barrier",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "imc_miss",
    "lg_throttle",
    "long_scoreboard",
    "math_pipe_throttle",
    "membar",
    "mio_throttle",
    "misc",
    "no_instruction",
    "not_selected",
    "short_scoreboard",
    "sleeping",
    "tex_throttle",
    "wait",
)
"""Every warp-issue stall reason, in NCU's own spelling.

The family also carries ``selected``, which is not a stall; the issue rate it
describes is ``issue_active_pct``. The two sub-breakdowns ``long_scoreboard`` and
``mio_throttle`` expose per pipe are absent, because they double-count their
parent. The reasons do not partition warp-active cycles -- measured on this fleet
they sum to 87 to 93 percent -- so no total is derived from them.
"""


def stall_metric(reason: str) -> str:
    """NCU metric name carrying one stall reason.

    Args:
        reason: A member of :data:`STALL_REASONS`.

    Returns:
        The metric name, requested verbatim.
    """
    return f"{_STALL_PREFIX}{reason}{_STALL_SUFFIX}"


def stall_field(reason: str) -> str:
    """:class:`KernelCounters` field carrying one stall reason.

    Args:
        reason: A member of :data:`STALL_REASONS`.

    Returns:
        The field name.
    """
    return f"stall_{reason}_pct"


NCU_TABLES: Final[tuple[NcuTable, ...]] = (
    NcuTable(
        "duration",
        (
            _DURATION,
            "launch__grid_size",
            "launch__block_size",
            "launch__waves_per_multiprocessor",
        ),
    ),
    NcuTable(
        "dram",
        (
            _DURATION,
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
    ),
    NcuTable(
        "global",
        (
            _DURATION,
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        ),
    ),
    NcuTable(
        "shared",
        (
            _DURATION,
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        ),
    ),
    NcuTable(
        "occupancy",
        (
            _DURATION,
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
            "launch__shared_mem_per_block_dynamic",
            "sm__maximum_warps_per_active_cycle_pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ),
    ),
    NcuTable(
        "pipe",
        (
            _DURATION,
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed.sum",
            "smsp__thread_inst_executed_per_inst_executed.ratio",
        ),
    ),
    NcuTable(
        "stall",
        (
            _DURATION,
            _ISSUE_ACTIVE,
            *(stall_metric(reason) for reason in STALL_REASONS),
        ),
    ),
    NcuTable("sol", (_DURATION, _SM_SOL, _MEMORY_SOL, _L1TEX_SOL, _L2_SOL)),
)
"""The tables. Each is one NCU pass; every pass re-reads the duration so the
passes can be checked against each other. ``dram__throughput`` is the DRAM row of
the speed-of-light breakdown and stays in the ``dram`` table beside the byte
counters it explains, so ``sol`` does not request it twice."""

REQUIRED_METRICS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(metric for table in NCU_TABLES for metric in table.metrics)
)
"""Every metric :func:`kernel_counters` needs, in table order."""

_LOCAL_LD: Final = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"
_LOCAL_ST: Final = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"

SPILL_TABLE: Final[NcuTable] = NcuTable("spill", (_DURATION, _LOCAL_LD, _LOCAL_ST))
"""Local-memory sectors, the counter a register spill lands in.

Local memory carries spilled registers and any per-thread array the compiler could
not keep in registers. Either is the register budget running out, and neither is
something a kernel held to a bandwidth is entitled to.

Sectors rather than bytes, because the question is whether the spill happened at
all: a spilled store and its reload are two events and both are counted, and a
store with no reload still says the register budget ran out.
"""

STALL_FIELDS: Final[tuple[str, ...]] = (
    "issue_active_pct",
    "dominant_stall",
    "dominant_stall_pct",
    *(stall_field(reason) for reason in STALL_REASONS),
)
""":class:`KernelCounters` fields the ``stall`` table fills, in print order."""

SOL_FIELDS: Final[tuple[str, ...]] = (
    "sm_pct",
    "memory_pct",
    "l1tex_pct",
    "l2_pct",
)
""":class:`KernelCounters` fields the ``sol`` table fills, in print order."""


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

_TIME_NS: Final[dict[str, float]] = {
    "nsecond": 1.0,
    "usecond": 1e3,
    "msecond": 1e6,
    "second": 1e9,
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
}

_PREFIX: Final[dict[str, float]] = {
    "": 1.0,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
}

_BASES: Final[tuple[str, ...]] = (
    "byte",
    "cycle",
    "inst",
    "sector",
    "request",
    "register",
    "wavefront",
    "warp",
    "block",
    "thread",
)

_DIMENSIONLESS: Final[frozenset[str]] = frozenset(("", "%", "ratio", "nan"))


def metric_scale(unit: str) -> float:
    """Factor taking an NCU display value to its base unit.

    Base units are bytes, cycles, instructions, sectors, requests, registers,
    wavefronts, warps, blocks, threads, percent, and nanoseconds for durations.

    A per-something unit scales by its numerator: ``register/thread`` and
    ``byte/block`` are already per-thread and per-block, and the denominator is
    part of the metric's meaning rather than of its scale.

    Args:
        unit: The ``Metric Unit`` cell, verbatim.

    Returns:
        The multiplier.

    Raises:
        ValueError: If the unit is not recognized. Assuming a scale of one for an
            unknown unit is how a Mbyte counter becomes a millionfold error.
    """
    text = unit.strip().partition("/")[0].strip()
    if text in _DIMENSIONLESS:
        return 1.0
    if text in _TIME_NS:
        return _TIME_NS[text]
    singular = text[:-1] if text.endswith("s") else text
    for candidate in (text, singular):
        for base in _BASES:
            if candidate.endswith(base):
                prefix = candidate[: -len(base)]
                if prefix in _PREFIX:
                    return _PREFIX[prefix]
    raise ValueError(f"unknown ncu metric unit {unit!r}")


def _sums(metric: str) -> bool:
    """Whether a metric adds across invocations.

    NCU's own suffix decides: ``.sum`` is a counter over the whole launch and
    adds, everything else is a rate, a ratio, or a launch configuration and does
    not. This reads NCU's naming convention, not an English label.
    """
    return metric.endswith(".sum")


# ---------------------------------------------------------------------------
# Command and parsing
# ---------------------------------------------------------------------------


def ncu_command(
    table: NcuTable,
    argv: Sequence[str],
    *,
    ncu: str = "ncu",
    extra: Sequence[str] = (),
) -> list[str]:
    """Build the command line for one table.

    The target follows the flags directly. ``ncu`` takes no ``--`` separator: it
    parses one as an empty long option and exits with an ambiguous-option error.

    Args:
        table: The table to request.
        argv: The target command, already split.
        ncu: Path to the ``ncu`` binary.
        extra: Additional NCU flags, inserted before the target.

    Returns:
        The full argv.

    Raises:
        ValueError: If the table requests no metrics or the target is empty.
    """
    if not table.metrics:
        raise ValueError(f"table {table.name!r} requests no metrics")
    if not argv:
        raise ValueError("ncu needs a target command")
    return [
        ncu,
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
        ",".join(table.metrics),
        *extra,
        *argv,
    ]


@dataclass(frozen=True)
class NcuInvocation:
    """One kernel launch, as one NCU pass saw it.

    Attributes:
        launch_id: NCU's launch identifier, kept so two passes can be lined up.
        kernel: Demangled kernel name.
        values: Metric name to base-unit value. Durations are nanoseconds.
    """

    launch_id: str
    kernel: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class NcuPass:
    """The result of one NCU invocation.

    Attributes:
        table: Table name.
        command: The argv that produced this, for the report to quote.
        invocations: Every profiled launch, in NCU's order.
        missing_metrics: Requested metrics that no row carried.
    """

    table: str
    command: tuple[str, ...]
    invocations: tuple[NcuInvocation, ...]
    missing_metrics: tuple[str, ...]


_HEADER_KEY: Final = "Metric Name"


def _csv_body(text: str) -> str:
    """Drop NCU's ``==PROF==`` preamble and return the CSV from its header."""
    for index, line in enumerate(text.splitlines(keepends=True)):
        if _HEADER_KEY in line:
            return "".join(text.splitlines(keepends=True)[index:])
    return ""


def _number(cell: str) -> float | None:
    text = cell.strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if value != value else value


def parse_ncu_csv(
    text: str, metrics: Sequence[str], *, table: str = "", command: Sequence[str] = ()
) -> NcuPass:
    """Parse NCU's long-form CSV into per-launch metric values.

    Args:
        text: NCU stdout, preamble included.
        metrics: The metrics that were requested. Anything absent from the output
            lands in ``missing_metrics``.
        table: Table name to record.
        command: Command to record.

    Returns:
        The pass.

    Raises:
        ValueError: If the output carries no CSV header, if a required column is
            absent, or if a unit is unrecognized.
    """
    body = _csv_body(text)
    if not body:
        raise ValueError(f"no CSV header in ncu output for table {table!r}")
    reader = csv.DictReader(io.StringIO(body))
    columns = reader.fieldnames or []
    for column in ("ID", "Kernel Name", _HEADER_KEY, "Metric Unit", "Metric Value"):
        if column not in columns:
            raise ValueError(f"ncu CSV has no {column!r} column, got {columns}")
    order: list[str] = []
    kernels: dict[str, str] = {}
    values: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for row in reader:
        launch = (row["ID"] or "").strip()
        if not launch:
            continue
        metric = (row[_HEADER_KEY] or "").strip()
        number = _number(row["Metric Value"] or "")
        if not metric or number is None:
            continue
        if launch not in values:
            order.append(launch)
            values[launch] = {}
            kernels[launch] = (row["Kernel Name"] or "").strip()
        values[launch][metric] = number * metric_scale(row["Metric Unit"] or "")
        seen.add(metric)
    return NcuPass(
        table=table,
        command=tuple(command),
        invocations=tuple(
            NcuInvocation(
                launch_id=launch, kernel=kernels[launch], values=values[launch]
            )
            for launch in order
        ),
        missing_metrics=tuple(m for m in metrics if m not in seen),
    )


def run_ncu(
    table: NcuTable,
    argv: Sequence[str],
    *,
    ncu: str = "ncu",
    extra: Sequence[str] = (),
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> NcuPass:
    """Run one NCU pass and parse it.

    Args:
        table: The table to request.
        argv: The target command.
        ncu: Path to the ``ncu`` binary.
        extra: Additional NCU flags.
        cwd: Working directory for the target.
        timeout_s: Wall clock limit, or None.

    Returns:
        The parsed pass.

    Raises:
        RuntimeError: If NCU exits nonzero. The message carries the tail of
            stderr, because NCU's own diagnostic names the cause and a bare exit
            code does not.
    """
    command = ncu_command(table, argv, ncu=ncu, extra=extra)
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
            f"ncu table {table.name!r} exited {done.returncode}: " + " | ".join(tail)
        )
    return parse_ncu_csv(done.stdout, table.metrics, table=table.name, command=command)


# ---------------------------------------------------------------------------
# Per-kernel record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelCounters(PerfRecord):
    """Every counter for one kernel, merged across passes.

    Durations are summed over the kernel's launches, because the question a
    kernel row answers is what share of the step it owns. Rates are formed from
    those sums, so the numerator and the denominator cover the same launches.

    Every field from the ``stall`` and the ``sol`` table is a percentage NCU has
    already normalized -- per warp-active cycle for a stall reason, per unit peak
    for a throughput -- so each takes the median across launches. Adding one over
    three launches would report 300 percent of a whole that is per launch.

    Attributes:
        kernel: Demangled kernel name.
        launch_count: Launches profiled.
        duration_us: Summed kernel duration.
        pass_duration_spread_pct: Disagreement between passes on that duration.
            Above a few percent, kernel replay is not reproducing the same work
            and nothing else in this record is comparable across passes.
        dram_read_bytes: Device memory read.
        dram_write_bytes: Device memory written.
        dram_pct: NCU's own DRAM throughput as a percentage of peak sustained.
            Trusted over the reconstructed rate beside it.
        achieved_gbs: ``(read + write)`` over ``duration_us``.
        global_load_request_count: Requests the LSU sent to L1TEX for loads. One
            per warp instruction that is not replayed.
        global_store_request_count: The same for stores.
        global_load_sector_count: L1 sectors touched by loads.
        global_store_sector_count: L1 sectors touched by stores.
        sector_per_load_request_ratio: Sectors over requests on the load side.
            Read against the instruction's own bytes per lane, not a constant:
            a warp asking four bytes each wants four sectors, and sixteen bytes
            each wants sixteen. Above that, the access is scattered and pays for
            sectors it does not use; below it, the request is under-filled and
            the kernel is leaving in-flight bytes on the table.
        sector_per_store_request_ratio: The same for stores.
        wavefront_count: Shared-memory wavefronts, load plus store. The
            denominator without which a conflict count means nothing.
        shared_load_conflict_count: Bank conflicts on shared loads.
        shared_store_conflict_count: Bank conflicts on shared stores.
        conflict_per_wavefront_ratio: Conflicts over wavefronts, zero when the
            kernel makes no shared access.
        register_per_thread_count: Registers per thread, from the launch.
        static_smem_bytes: Static shared memory per block.
        dynamic_smem_bytes: Dynamic shared memory per block.
        theoretical_occupancy_pct: Occupancy the launch configuration allows.
        achieved_occupancy_pct: Occupancy realized.
        tensor_pipe_pct: Tensor pipe active cycles as a percentage of peak
            sustained.
        inst_count: Instructions executed.
        active_thread_per_warp_ratio: Threads per executed instruction, out of
            32. Divergence shows here.
        block_count: Blocks in the grid.
        thread_per_block_count: Threads per block.
        wave_per_sm_ratio: Waves per multiprocessor. Below one, the grid does not
            fill the device.
        issue_active_pct: Issue slots used, as a percentage of peak sustained. Low
            beside a high stall reason is a latency bound; a bandwidth bound
            reaches the DRAM ceiling instead.
        dominant_stall: The reason in :data:`STALL_REASONS` holding the largest
            share. Ties break in declaration order.
        dominant_stall_pct: That reason's share of warp-active cycles.
        stall_barrier_pct: Waiting at a CTA barrier.
        stall_branch_resolving_pct: Waiting for a branch target to resolve.
        stall_dispatch_stall_pct: Waiting for a busy dispatch port.
        stall_drain_pct: Waiting for outstanding memory to drain after an exit.
        stall_imc_miss_pct: Waiting on an immediate-constant cache miss.
        stall_lg_throttle_pct: Local and global instruction queue full.
        stall_long_scoreboard_pct: Waiting on an L1TEX scoreboard dependency,
            chiefly a global load. The memory-latency signal.
        stall_math_pipe_throttle_pct: Math pipe oversubscribed.
        stall_membar_pct: Waiting at a memory barrier.
        stall_mio_throttle_pct: MIO instruction queue full, chiefly shared memory.
        stall_misc_pct: Everything not otherwise attributed.
        stall_no_instruction_pct: Waiting on an instruction fetch.
        stall_not_selected_pct: Eligible, and another warp was issued instead.
        stall_short_scoreboard_pct: Waiting on an MIO scoreboard dependency,
            chiefly shared memory.
        stall_sleeping_pct: Warp asleep.
        stall_tex_throttle_pct: Texture and L1 request queue full.
        stall_wait_pct: Waiting on a fixed-latency execution dependency.
        sm_pct: SM throughput against peak sustained. NCU's Compute SOL row.
        memory_pct: Memory-pipeline throughput against peak sustained, the
            maximum over the memory subsystem. NCU's Memory SOL row. At or above
            ``dram_pct`` by construction.
        l1tex_pct: L1TEX throughput against peak sustained.
        l2_pct: L2 throughput against peak sustained.
    """

    kernel: str
    launch_count: Annotated[Count, SUM]
    duration_us: Annotated[Microseconds, SUM]
    pass_duration_spread_pct: Annotated[Percent, MEDIAN]
    dram_read_bytes: Annotated[Bytes, SUM]
    dram_write_bytes: Annotated[Bytes, SUM]
    dram_pct: Annotated[Percent, MEDIAN]
    achieved_gbs: Annotated[GBPerSecond, MEDIAN]
    global_load_request_count: Annotated[Count, SUM]
    global_store_request_count: Annotated[Count, SUM]
    global_load_sector_count: Annotated[Count, SUM]
    global_store_sector_count: Annotated[Count, SUM]
    sector_per_load_request_ratio: Annotated[Ratio, MEDIAN]
    sector_per_store_request_ratio: Annotated[Ratio, MEDIAN]
    wavefront_count: Annotated[Count, SUM]
    shared_load_conflict_count: Annotated[Count, SUM]
    shared_store_conflict_count: Annotated[Count, SUM]
    conflict_per_wavefront_ratio: Annotated[Ratio, MEDIAN]
    register_per_thread_count: Annotated[Count, INVARIANT]
    static_smem_bytes: Annotated[Bytes, INVARIANT]
    dynamic_smem_bytes: Annotated[Bytes, INVARIANT]
    theoretical_occupancy_pct: Annotated[Percent, MEDIAN]
    achieved_occupancy_pct: Annotated[Percent, MEDIAN]
    tensor_pipe_pct: Annotated[Percent, MEDIAN]
    inst_count: Annotated[Count, SUM]
    active_thread_per_warp_ratio: Annotated[Ratio, MEDIAN]
    block_count: Annotated[Count, INVARIANT]
    thread_per_block_count: Annotated[Count, INVARIANT]
    wave_per_sm_ratio: Annotated[Ratio, MEDIAN]
    issue_active_pct: Annotated[Percent, MEDIAN]
    dominant_stall: str
    dominant_stall_pct: Annotated[Percent, MEDIAN]
    stall_barrier_pct: Annotated[Percent, MEDIAN]
    stall_branch_resolving_pct: Annotated[Percent, MEDIAN]
    stall_dispatch_stall_pct: Annotated[Percent, MEDIAN]
    stall_drain_pct: Annotated[Percent, MEDIAN]
    stall_imc_miss_pct: Annotated[Percent, MEDIAN]
    stall_lg_throttle_pct: Annotated[Percent, MEDIAN]
    stall_long_scoreboard_pct: Annotated[Percent, MEDIAN]
    stall_math_pipe_throttle_pct: Annotated[Percent, MEDIAN]
    stall_membar_pct: Annotated[Percent, MEDIAN]
    stall_mio_throttle_pct: Annotated[Percent, MEDIAN]
    stall_misc_pct: Annotated[Percent, MEDIAN]
    stall_no_instruction_pct: Annotated[Percent, MEDIAN]
    stall_not_selected_pct: Annotated[Percent, MEDIAN]
    stall_short_scoreboard_pct: Annotated[Percent, MEDIAN]
    stall_sleeping_pct: Annotated[Percent, MEDIAN]
    stall_tex_throttle_pct: Annotated[Percent, MEDIAN]
    stall_wait_pct: Annotated[Percent, MEDIAN]
    sm_pct: Annotated[Percent, MEDIAN]
    memory_pct: Annotated[Percent, MEDIAN]
    l1tex_pct: Annotated[Percent, MEDIAN]
    l2_pct: Annotated[Percent, MEDIAN]

    @property
    def smem_bytes(self) -> Bytes:
        """Shared memory per block, static plus dynamic."""
        return Bytes(self.static_smem_bytes + self.dynamic_smem_bytes)


def _merge(
    passes: Iterable[NcuPass],
) -> tuple[dict[str, int], dict[str, dict[str, float]], dict[str, list[float]]]:
    launches: dict[str, int] = {}
    merged: dict[str, dict[str, float]] = {}
    durations: dict[str, list[float]] = {}
    for one in passes:
        counts: dict[str, int] = {}
        totals: dict[str, dict[str, list[float]]] = {}
        for invocation in one.invocations:
            kernel = invocation.kernel
            counts[kernel] = counts.get(kernel, 0) + 1
            bucket = totals.setdefault(kernel, {})
            for metric, value in invocation.values.items():
                bucket.setdefault(metric, []).append(value)
        for kernel, bucket in totals.items():
            launches[kernel] = max(launches.get(kernel, 0), counts[kernel])
            target = merged.setdefault(kernel, {})
            for metric, samples in bucket.items():
                value = sum(samples) if _sums(metric) else _median(samples)
                if metric == _DURATION:
                    durations.setdefault(kernel, []).append(value)
                else:
                    target[metric] = value
    # Every pass re-reads the duration, so there is one sample per pass. The
    # consensus is their median, which is the statistic the spread beside it
    # measures. Taking the first pass's value instead would make the record depend
    # on the order the passes were handed in.
    for kernel, samples in durations.items():
        merged[kernel][_DURATION] = _median(samples)
    return launches, merged, durations


def _median(samples: Sequence[float]) -> float:
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _spread_pct(samples: Sequence[float]) -> Percent:
    return pct_of(max(samples) - min(samples), _median(samples))


def kernel_counters(
    passes: Sequence[NcuPass], *, required: Sequence[str] = REQUIRED_METRICS
) -> tuple[KernelCounters, ...]:
    """Merge passes into one record per kernel.

    Args:
        passes: One pass per table, in any order. The order does not reach the
            output: the duration is the median over the passes that reported it.
        required: Metrics every kernel must carry. Defaults to the union of
            :data:`NCU_TABLES`.

    Returns:
        One record per kernel, ordered by descending duration.

    Raises:
        ValueError: If no pass carried any launch, if a kernel is missing a
            required metric, or if the consensus duration of a kernel that
            launched is zero. Filling an absent counter with a zero would report a
            broken label as a free operation, and so would keeping a zero
            duration.
    """
    launches, merged, durations = _merge(passes)
    if not merged:
        raise ValueError("no kernel launches in any ncu pass")
    absent = {
        kernel: [m for m in required if m not in values]
        for kernel, values in merged.items()
    }
    broken = {k: v for k, v in absent.items() if v}
    if broken:
        kernel, missing = next(iter(sorted(broken.items())))
        raise ValueError(
            f"kernel {kernel!r} is missing {len(missing)} metrics, first "
            f"{missing[0]!r}; run every table in NCU_TABLES"
        )
    out: list[KernelCounters] = []
    for kernel, values in merged.items():
        duration_us = us_from_ns(Nanoseconds(values[_DURATION]))
        if duration_us <= 0.0:
            raise ValueError(
                f"ncu reported a zero duration for kernel {kernel!r} over "
                f"{len(durations[kernel])} passes; a kernel that launched and took "
                "no time is a broken profile, not a free operation"
            )
        read = Bytes(round(values["dram__bytes_read.sum"]))
        write = Bytes(round(values["dram__bytes_write.sum"]))
        load_reqs = round(values["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"])
        store_reqs = round(values["l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"])
        load_sectors = round(values["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"])
        store_sectors = round(values["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"])
        wavefronts = round(
            values["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"]
            + values["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"]
        )
        load_conflicts = round(
            values["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]
        )
        store_conflicts = round(
            values["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"]
        )
        stalls = {
            reason: Percent(values[stall_metric(reason)]) for reason in STALL_REASONS
        }
        dominant = max(STALL_REASONS, key=lambda reason: stalls[reason])
        out.append(
            KernelCounters(
                kernel=kernel,
                launch_count=Count(launches[kernel]),
                duration_us=duration_us,
                pass_duration_spread_pct=_spread_pct(durations[kernel]),
                dram_read_bytes=read,
                dram_write_bytes=write,
                dram_pct=Percent(
                    values["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
                ),
                achieved_gbs=gbs_from_bytes_us(Bytes(read + write), duration_us),
                global_load_request_count=Count(load_reqs),
                global_store_request_count=Count(store_reqs),
                global_load_sector_count=Count(load_sectors),
                global_store_sector_count=Count(store_sectors),
                sector_per_load_request_ratio=Ratio(
                    0.0 if load_reqs == 0 else load_sectors / load_reqs
                ),
                sector_per_store_request_ratio=Ratio(
                    0.0 if store_reqs == 0 else store_sectors / store_reqs
                ),
                wavefront_count=Count(wavefronts),
                shared_load_conflict_count=Count(load_conflicts),
                shared_store_conflict_count=Count(store_conflicts),
                conflict_per_wavefront_ratio=Ratio(
                    0.0
                    if wavefronts == 0
                    else (load_conflicts + store_conflicts) / wavefronts
                ),
                register_per_thread_count=Count(
                    round(values["launch__registers_per_thread"])
                ),
                static_smem_bytes=Bytes(
                    round(values["launch__shared_mem_per_block_static"])
                ),
                dynamic_smem_bytes=Bytes(
                    round(values["launch__shared_mem_per_block_dynamic"])
                ),
                theoretical_occupancy_pct=Percent(
                    values["sm__maximum_warps_per_active_cycle_pct"]
                ),
                achieved_occupancy_pct=Percent(
                    values["sm__warps_active.avg.pct_of_peak_sustained_active"]
                ),
                tensor_pipe_pct=Percent(
                    values[
                        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
                    ]
                ),
                inst_count=Count(round(values["sm__inst_executed.sum"])),
                active_thread_per_warp_ratio=Ratio(
                    values["smsp__thread_inst_executed_per_inst_executed.ratio"]
                ),
                block_count=Count(round(values["launch__grid_size"])),
                thread_per_block_count=Count(round(values["launch__block_size"])),
                wave_per_sm_ratio=Ratio(values["launch__waves_per_multiprocessor"]),
                issue_active_pct=Percent(values[_ISSUE_ACTIVE]),
                dominant_stall=dominant,
                dominant_stall_pct=stalls[dominant],
                stall_barrier_pct=stalls["barrier"],
                stall_branch_resolving_pct=stalls["branch_resolving"],
                stall_dispatch_stall_pct=stalls["dispatch_stall"],
                stall_drain_pct=stalls["drain"],
                stall_imc_miss_pct=stalls["imc_miss"],
                stall_lg_throttle_pct=stalls["lg_throttle"],
                stall_long_scoreboard_pct=stalls["long_scoreboard"],
                stall_math_pipe_throttle_pct=stalls["math_pipe_throttle"],
                stall_membar_pct=stalls["membar"],
                stall_mio_throttle_pct=stalls["mio_throttle"],
                stall_misc_pct=stalls["misc"],
                stall_no_instruction_pct=stalls["no_instruction"],
                stall_not_selected_pct=stalls["not_selected"],
                stall_short_scoreboard_pct=stalls["short_scoreboard"],
                stall_sleeping_pct=stalls["sleeping"],
                stall_tex_throttle_pct=stalls["tex_throttle"],
                stall_wait_pct=stalls["wait"],
                sm_pct=Percent(values[_SM_SOL]),
                memory_pct=Percent(values[_MEMORY_SOL]),
                l1tex_pct=Percent(values[_L1TEX_SOL]),
                l2_pct=Percent(values[_L2_SOL]),
            )
        )
    return tuple(sorted(out, key=lambda k: k.duration_us, reverse=True))


@dataclass(frozen=True)
class SpillCounters(PerfRecord):
    """Local-memory traffic for one kernel, over one capture window.

    Attributes:
        kernel: Demangled kernel name.
        launch_count: Launches profiled.
        duration_us: Summed kernel duration, on the same footing as
            :attr:`KernelCounters.duration_us`, so the two records can be checked
            against each other.
        local_load_sector_count: L1 sectors touched by local loads, which on these
            kernels is a spilled register being read back.
        local_store_sector_count: L1 sectors touched by local stores, which is a
            register being spilled.
    """

    kernel: str
    launch_count: Annotated[Count, SUM]
    duration_us: Annotated[Microseconds, SUM]
    local_load_sector_count: Annotated[Count, SUM]
    local_store_sector_count: Annotated[Count, SUM]

    @property
    def spill_sector_count(self) -> Count:
        """Local-memory sectors, load plus store."""
        return Count(self.local_load_sector_count + self.local_store_sector_count)

    @property
    def spilled(self) -> bool:
        """Whether the kernel touched local memory at all."""
        return self.spill_sector_count > 0


def spill_counters(one: NcuPass) -> tuple[SpillCounters, ...]:
    """Read local-memory sectors per kernel out of one :data:`SPILL_TABLE` pass.

    Args:
        one: The parsed spill pass.

    Returns:
        One record per kernel, ordered by descending duration.

    Raises:
        ValueError: If the pass carried no launch, or if a kernel is missing either
            counter. An absent spill counter filled with a zero would report a
            spilling kernel as clean, which is the one direction that turns a
            failing verdict into a passing one.
    """
    launches, merged, _durations = _merge((one,))
    if not merged:
        raise ValueError(f"no kernel launches in ncu pass {one.table!r}")
    out: list[SpillCounters] = []
    for kernel, values in sorted(merged.items()):
        absent = [
            metric
            for metric in (_DURATION, _LOCAL_LD, _LOCAL_ST)
            if metric not in values
        ]
        if absent:
            raise ValueError(
                f"kernel {kernel!r} is missing {absent[0]!r}; run SPILL_TABLE"
            )
        out.append(
            SpillCounters(
                kernel=kernel,
                launch_count=Count(launches[kernel]),
                duration_us=us_from_ns(Nanoseconds(values[_DURATION])),
                local_load_sector_count=Count(round(values[_LOCAL_LD])),
                local_store_sector_count=Count(round(values[_LOCAL_ST])),
            )
        )
    return tuple(sorted(out, key=lambda k: k.duration_us, reverse=True))
