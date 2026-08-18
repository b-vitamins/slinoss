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

One NCU pass per table. Counters from different passes describe different
executions, so they do not share a row without a stated disagreement:
``pass_duration_spread_pct`` carries the duration disagreement between passes,
which is the replay-stability signal for everything else in the record.
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
    "KernelCounters",
    "NcuInvocation",
    "NcuPass",
    "NcuTable",
    "kernel_counters",
    "metric_scale",
    "ncu_command",
    "parse_ncu_csv",
    "run_ncu",
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
            "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
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
)
"""The tables. Each is one NCU pass; every pass re-reads the duration so the
passes can be checked against each other."""

REQUIRED_METRICS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(metric for table in NCU_TABLES for metric in table.metrics)
)
"""Every metric :func:`kernel_counters` needs, in table order."""


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
        global_load_bytes: Bytes requested from global by load instructions.
        global_store_bytes: Bytes requested from global by store instructions.
        global_load_sector_count: L1 sectors touched by loads.
        global_store_sector_count: L1 sectors touched by stores.
        bytes_per_sector_ratio: Global bytes over global sectors. 32 is a fully
            coalesced access; below that the access pattern is wasting sectors.
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
    """

    kernel: str
    launch_count: Annotated[Count, SUM]
    duration_us: Annotated[Microseconds, SUM]
    pass_duration_spread_pct: Annotated[Percent, MEDIAN]
    dram_read_bytes: Annotated[Bytes, SUM]
    dram_write_bytes: Annotated[Bytes, SUM]
    dram_pct: Annotated[Percent, MEDIAN]
    achieved_gbs: Annotated[GBPerSecond, MEDIAN]
    global_load_bytes: Annotated[Bytes, SUM]
    global_store_bytes: Annotated[Bytes, SUM]
    global_load_sector_count: Annotated[Count, SUM]
    global_store_sector_count: Annotated[Count, SUM]
    bytes_per_sector_ratio: Annotated[Ratio, MEDIAN]
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
        load_bytes = round(values["l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"])
        store_bytes = round(values["l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"])
        load_sectors = round(values["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"])
        store_sectors = round(values["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"])
        sectors = load_sectors + store_sectors
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
                global_load_bytes=Bytes(load_bytes),
                global_store_bytes=Bytes(store_bytes),
                global_load_sector_count=Count(load_sectors),
                global_store_sector_count=Count(store_sectors),
                bytes_per_sector_ratio=Ratio(
                    0.0 if sectors == 0 else (load_bytes + store_bytes) / sectors
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
            )
        )
    return tuple(sorted(out, key=lambda k: k.duration_us, reverse=True))
