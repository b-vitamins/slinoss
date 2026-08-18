"""Report emission, and the agreement check that gates it.

Three independent clocks see the same work: CUDA events around the call, NSYS
tracing the launch stream, and NCU replaying each kernel. They must agree, and
when they do not the report refuses to emit rather than picking a favourite. The
failure mode this closes is a report that quotes a per-kernel breakdown summing
to more than the step it claims to break down.

The three are not interchangeable. NCU and NSYS both measure kernels and must
agree within tolerance. The CUDA-event wall covers kernels plus every gap between
them, so it is greater than or equal to the device sum; the excess is launch gap
and is reported as ``gap_pct`` rather than being absorbed. A negative gap is an
impossible timeline and fails the check.

Field names reach the table verbatim. A markdown header is a field name, so the
unit is in the header because it is in the name, and there is no formatting step
that can rename one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Final

from slinoss.perf.budget import BucketDelta, BudgetReport
from slinoss.perf.ceiling import Ceilings, ClassVerdict
from slinoss.perf.device import DeviceInfo
from slinoss.perf.dispersion import GrowthRow, RepeatRow
from slinoss.perf.memory import MemoryPeaks, SavedStorages
from slinoss.perf.ncu import KernelCounters
from slinoss.perf.nsys import NsysTrace
from slinoss.perf.timing import Throughput
from slinoss.perf.units import (
    INVARIANT,
    MEDIAN,
    Count,
    Microseconds,
    Percent,
    PerfRecord,
    Spread,
    pct_of,
)

__all__ = [
    "TOLERANCE_PCT",
    "Agreement",
    "AgreementError",
    "Report",
    "agreement",
    "json_text",
    "markdown",
    "payload",
    "write_report",
]

TOLERANCE_PCT: Final[Percent] = Percent(5.0)
"""Allowed disagreement between two clocks measuring the same work."""


class AgreementError(RuntimeError):
    """Raised instead of emitting a report whose clocks disagree."""


@dataclass(frozen=True)
class Agreement(PerfRecord):
    """The three-way cross-check, normalized to one iteration.

    Every duration here is per profiled iteration, so the three are directly
    comparable: the profiler sums cover the whole capture window and are divided
    by the iteration count that window contained.

    Attributes:
        label: What was measured.
        capture_iter_count: Iterations inside the profiler capture window.
        event_duration_us: Median per-iteration wall from CUDA events.
        nsys_kernel_sum_duration_us: NSYS kernel time per iteration.
        nsys_device_sum_duration_us: NSYS kernel, copy, and fill time per
            iteration.
        ncu_kernel_sum_duration_us: NCU kernel time per iteration.
        kernel_delta_pct: NCU against NSYS, relative to NSYS.
        gap_pct: Event wall minus device sum, as a percentage of the wall. This is
            launch gap and idle. Negative is an impossible timeline.
        tolerance_pct: The bar both checks are held to.
        agrees: True only if both checks pass.
        detail: Which check failed, or that both passed.
    """

    label: str
    capture_iter_count: Annotated[Count, INVARIANT]
    event_duration_us: Annotated[Microseconds, MEDIAN]
    nsys_kernel_sum_duration_us: Annotated[Microseconds, MEDIAN]
    nsys_device_sum_duration_us: Annotated[Microseconds, MEDIAN]
    ncu_kernel_sum_duration_us: Annotated[Microseconds, MEDIAN]
    kernel_delta_pct: Annotated[Percent, MEDIAN]
    gap_pct: Annotated[Percent, MEDIAN]
    tolerance_pct: Annotated[Percent, MEDIAN]
    agrees: bool
    detail: str


def agreement(
    label: str,
    *,
    event: Spread,
    trace: NsysTrace,
    kernels: Sequence[KernelCounters],
    capture_iters: int,
    tolerance_pct: Percent = TOLERANCE_PCT,
) -> Agreement:
    """Cross-check the CUDA-event wall against NSYS and NCU.

    Args:
        label: What was measured.
        event: Per-iteration wall dispersion from :func:`slinoss.perf.timing.measure`.
        trace: The NSYS trace of the capture window.
        kernels: The merged NCU counters for the same window.
        capture_iters: Iterations the capture window contained.
        tolerance_pct: Allowed disagreement.

    Returns:
        The check. Inspect ``agrees``; :func:`markdown` refuses a failing one.

    Raises:
        ValueError: If ``capture_iters`` is not positive, or if NCU contributed no
            kernels, which would make the check vacuous.
    """
    if capture_iters <= 0:
        raise ValueError(f"capture_iters must be positive, got {capture_iters}")
    if not kernels:
        raise ValueError("agreement needs at least one NCU kernel")
    per_iter = float(capture_iters)
    nsys_kernel = Microseconds(trace.kernel_sum_duration_us / per_iter)
    nsys_device = Microseconds(trace.device_sum_duration_us / per_iter)
    ncu_kernel = Microseconds(sum(k.duration_us for k in kernels) / per_iter)
    wall = event.median_duration_us
    delta = pct_of(ncu_kernel - nsys_kernel, nsys_kernel)
    gap = pct_of(wall - nsys_device, wall)
    kernels_agree = abs(delta) <= tolerance_pct
    timeline_sane = gap >= -tolerance_pct
    if kernels_agree and timeline_sane:
        detail = "ncu and nsys agree; event wall covers the device sum"
    elif not kernels_agree:
        detail = (
            f"ncu kernel sum {ncu_kernel:.3f} us and nsys kernel sum "
            f"{nsys_kernel:.3f} us differ by {delta:.2f}%"
        )
    else:
        detail = (
            f"event wall {wall:.3f} us is below the nsys device sum "
            f"{nsys_device:.3f} us by {-gap:.2f}%"
        )
    return Agreement(
        label=label,
        capture_iter_count=Count(capture_iters),
        event_duration_us=wall,
        nsys_kernel_sum_duration_us=nsys_kernel,
        nsys_device_sum_duration_us=nsys_device,
        ncu_kernel_sum_duration_us=ncu_kernel,
        kernel_delta_pct=delta,
        gap_pct=gap,
        tolerance_pct=tolerance_pct,
        agrees=kernels_agree and timeline_sane,
        detail=detail,
    )


@dataclass(frozen=True)
class Report:
    """Everything one measurement produced.

    Every part is optional except the title and the device, because a phase that
    has not yet built a fused path has nothing to put in some of them. An absent
    part prints as absent; it never prints as a zero.

    Attributes:
        title: Report heading.
        device: The part the numbers were taken on.
        agreement: The three-way check, or None if only one clock was run.
        budget: The closed budget tree.
        throughput: Token rates, one per measured configuration.
        ceilings: The measured DRAM and tensor ceilings.
        kernels: Merged NCU counters, longest first.
        trace: The NSYS trace.
        saved: Autograd saved-storage forensics.
        peaks: Allocator high-water marks.
        verdicts: Per-kernel class verdicts.
        deltas: Bucket-level comparison against a prior report.
        growth: Dispersion against the sample count.
        scatter: Run-to-run median scatter against the reported floor.
        notes: Free-form lines appended verbatim.
    """

    title: str
    device: DeviceInfo
    agreement: Agreement | None = None
    budget: BudgetReport | None = None
    throughput: tuple[Throughput, ...] = ()
    ceilings: Ceilings | None = None
    kernels: tuple[KernelCounters, ...] = ()
    trace: NsysTrace | None = None
    saved: SavedStorages | None = None
    peaks: MemoryPeaks | None = None
    verdicts: tuple[ClassVerdict, ...] = ()
    deltas: tuple[BucketDelta, ...] = ()
    growth: tuple[GrowthRow, ...] = ()
    scatter: RepeatRow | None = None
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def payload(obj: object) -> Any:
    """Convert a record tree to JSON-ready data, keeping every field name.

    Args:
        obj: A dataclass, sequence, mapping, or scalar.

    Returns:
        Nested dicts and lists. Field names are never rewritten, so a unit suffix
        and an ``est_`` prefix survive serialization.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: payload(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): payload(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [payload(item) for item in obj]
    return obj


def json_text(report: Report) -> str:
    """Serialize a report as JSON, field order preserved."""
    return json.dumps(payload(report), indent=2)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def _row(record: object, prefix: str = "") -> dict[str, str]:
    """Flatten one record's scalar fields, dotting into nested records.

    A nested field is named ``outer.inner``, so the leaf keeps its own unit
    suffix. Tuple fields are tables of their own and are skipped here.
    """
    out: dict[str, str] = {}
    if not is_dataclass(record) or isinstance(record, type):
        return out
    for field in fields(record):
        value = getattr(record, field.name)
        name = f"{prefix}{field.name}"
        if is_dataclass(value) and not isinstance(value, type):
            out.update(_row(value, prefix=f"{name}."))
        elif isinstance(value, (tuple, list)):
            continue
        else:
            out[name] = _fmt(value)
    return out


def _table(records: Sequence[object]) -> list[str]:
    rows = [_row(r) for r in records]
    rows = [r for r in rows if r]
    if not rows:
        return ["(none)", ""]
    headers = list(rows[0])
    other = next((r for r in rows[1:] if list(r) != headers), None)
    if other is not None:
        # Headers come from the first row, so a second shape would print under
        # the wrong columns and drop every field the first row does not have.
        raise ValueError(
            f"one table takes one record shape; got {headers} and {list(other)}"
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row.get(h, "") for h in headers) + " |" for row in rows]
    lines.append("")
    return lines


def _section(title: str, records: Sequence[object]) -> list[str]:
    return [f"## {title}", "", *_table(records)]


def markdown(report: Report, *, require_agreement: bool = True) -> str:
    """Render a report as markdown.

    Args:
        report: The report.
        require_agreement: If True, a report with no agreement check or a failing
            one raises instead of rendering. Pass False only for a measurement
            that ran a single clock, and expect the header to say so.

    Returns:
        The markdown text.

    Raises:
        AgreementError: If the check failed, or if it is required and absent.
    """
    check = report.agreement
    if check is not None and not check.agrees:
        raise AgreementError(
            f"{report.title}: clocks disagree beyond "
            f"{check.tolerance_pct:.1f}%: {check.detail}"
        )
    if check is None and require_agreement:
        raise AgreementError(
            f"{report.title}: no CUDA-event / NSYS / NCU cross-check; "
            f"pass require_agreement=False to emit a single-clock measurement"
        )
    device = report.device
    lines: list[str] = [
        f"# {report.title}",
        "",
        f"- device: {device.name}, capability {device.capability}, "
        f"{device.sm_count} SM",
        f"- clocks: {device.clocks.stamp}",
        f"- sharing: {device.sharing.stamp}",
        f"- smem opt-in per block: {device.smem_optin_per_block_bytes:,} bytes",
        f"- cross-check: {'not run' if check is None else check.detail}",
        "",
    ]
    if check is not None:
        lines += _section("cross-check", [check])
    if report.budget is not None:
        lines += [
            "## budget",
            "",
            f"- total_duration_us: {report.budget.total.median_duration_us:,.3f}",
            f"- spread_pct: {report.budget.total.spread_pct:,.3f}",
            f"- resolution_pct: {report.budget.total.resolution_pct:,.3f}",
            f"- coverage_pct: {report.budget.total.coverage_pct:,.3f}",
            f"- sample_count: {report.budget.total.sample_count}",
            "",
            *_table(report.budget.buckets),
        ]
    if report.throughput:
        lines += _section("throughput", report.throughput)
    if report.ceilings is not None:
        lines += _section("measured dram ceiling", [report.ceilings.dram])
        lines += _section("measured tensor ceiling", [report.ceilings.tensor])
    if report.verdicts:
        lines += _section("class verdicts", report.verdicts)
    if report.kernels:
        lines += _section("kernel counters", report.kernels)
    if report.trace is not None:
        lines += [
            "## gpu trace",
            "",
            f"- kernel_sum_duration_us: {report.trace.kernel_sum_duration_us:,.3f}",
            f"- memcpy_sum_duration_us: "
            f"{report.trace.memcpy_sum_duration_us:,.3f} "
            f"over {report.trace.memcpy_count} copies",
            f"- memset_sum_duration_us: "
            f"{report.trace.memset_sum_duration_us:,.3f} "
            f"over {report.trace.memset_count} fills",
            "",
            *_table(report.trace.kernels),
        ]
    if report.saved is not None:
        lines += _section("saved tensors", [report.saved])
        lines += _section("saved tensors by region", report.saved.regions)
    if report.peaks is not None:
        lines += _section("memory peaks", [report.peaks])
    if report.deltas:
        lines += _section("bucket deltas", report.deltas)
    if report.growth:
        lines += _section("dispersion against sample count", report.growth)
    if report.scatter is not None:
        lines += _section("run-to-run median scatter", [report.scatter])
    if report.notes:
        lines += ["## notes", "", *[f"- {note}" for note in report.notes], ""]
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    report: Report, base: Path, *, require_agreement: bool = True
) -> tuple[Path, Path]:
    """Write a report as markdown and JSON beside each other.

    Args:
        report: The report.
        base: Output path. The suffix is appended to the whole name, not
            substituted for its last suffix, so two bases differing only after a
            dot write two files.
        require_agreement: Passed to :func:`markdown`.

    Returns:
        The markdown path and the JSON path.

    Raises:
        AgreementError: Before writing anything, if the check fails. A refused
            report leaves no file, so a stale one cannot be mistaken for a fresh
            pass.
    """
    text = markdown(report, require_agreement=require_agreement)
    base.parent.mkdir(parents=True, exist_ok=True)
    # `with_suffix` would replace everything after the last dot, so bases like
    # `out/run.1` and `out/run.2` would both write `out/run.md` and the second
    # would silently overwrite the first.
    md = base.with_name(base.name + ".md")
    js = base.with_name(base.name + ".json")
    md.write_text(text)
    js.write_text(json_text(report) + "\n")
    return md, js
