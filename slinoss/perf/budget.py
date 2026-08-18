"""Budget taxonomy: a closed tree over the measured total.

Every node's children sum to the node exactly, because the remainder is a node.
A parent that is measured and has children gets an ``unattributed`` child equal
to the parent minus the sum of its children, so time no region covers is a row in
the table instead of a discrepancy nobody reads. The remainder is not clamped: a
negative one means two regions cover the same work, which is a defect in the
instrumentation and must be visible.

The measured total is the only total. A sum over regions is
``root_sum_duration_us`` and is a different name for a different quantity.

A declared bucket that reads exactly zero is a broken label, not a free
operation. :func:`assert_nonzero` is what turns that rule into a test; the
declared set is a parameter because it grows as the fused path lands, and a rule
that cannot be checked until the last phase is not a rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Annotated

from slinoss.perf.timing import UNATTRIBUTED, Timed, parent_of
from slinoss.perf.units import (
    CONFIDENCE_PCT,
    MEDIAN,
    SUM,
    Count,
    Microseconds,
    Percent,
    PerfRecord,
    Ratio,
    Spread,
    pct_of,
    ratio_of,
)

__all__ = [
    "CLOSURE_TOL_PCT",
    "STEP_BUCKETS",
    "UNATTRIBUTED",
    "BucketDelta",
    "BucketTiming",
    "BudgetReport",
    "assert_closed",
    "assert_nonzero",
    "budget",
    "compare",
    "rank",
]

STEP_BUCKETS: tuple[str, ...] = (
    "step.zero_grad",
    "step.forward",
    "step.backward",
    "step.clip",
    "step.optim",
)
"""The step-level taxonomy. Stable across phases; the sub-trees are not."""

CLOSURE_TOL_PCT: Percent = Percent(0.01)
"""How far a parent may differ from the sum of its children. Float noise only."""


@dataclass(frozen=True)
class BucketTiming(PerfRecord):
    """One node of the budget tree.

    Attributes:
        label: Dotted bucket label.
        parent: Dotted parent label, empty at the root.
        derived: True if this row is arithmetic over other rows rather than a
            measurement. Derived rows carry ``sample_count`` zero.
        median_duration_us: Median over the measured iterations, or the derived
            value.
        spread_pct: Full range of the samples over their median, zero on a derived
            row. Outlier exposure, not a floor.
        resolution_pct: Half-width of the median's confidence interval, zero on a
            derived row. Half of the floor a delta on this bucket must beat; the
            other half belongs to the run it is compared against.
        coverage_pct: Exact coverage of that interval, zero on a derived row.
            Below :data:`slinoss.perf.units.CONFIDENCE_PCT` the floor does not
            hold and no delta on this bucket resolves.
        sample_count: Samples behind the median, zero on a derived row.
        share_of_parent_pct: Median as a percentage of the parent's median.
        share_of_total_pct: Median as a percentage of the measured total.
    """

    label: str
    parent: str
    derived: bool
    median_duration_us: Annotated[Microseconds, MEDIAN]
    spread_pct: Annotated[Percent, MEDIAN]
    resolution_pct: Annotated[Percent, MEDIAN]
    coverage_pct: Annotated[Percent, MEDIAN]
    sample_count: Annotated[Count, SUM]
    share_of_parent_pct: Annotated[Percent, MEDIAN]
    share_of_total_pct: Annotated[Percent, MEDIAN]


@dataclass(frozen=True)
class BudgetReport(PerfRecord):
    """A closed budget tree over one measurement.

    Attributes:
        label: What was measured.
        clocks: Clock stamp of the run.
        total: The measured whole-call dispersion. The only total.
        buckets: Every node, parents before children.
    """

    label: str
    clocks: str
    total: Spread
    buckets: tuple[BucketTiming, ...]

    def get(self, label: str) -> BucketTiming:
        """Look up one bucket.

        Args:
            label: Dotted bucket label.

        Returns:
            The bucket.

        Raises:
            KeyError: If the label is absent.
        """
        for bucket in self.buckets:
            if bucket.label == label:
                return bucket
        raise KeyError(f"no bucket {label!r} in {self.label!r}")

    def labels(self) -> tuple[str, ...]:
        """Every bucket label, in emission order."""
        return tuple(b.label for b in self.buckets)

    def children(self, label: str) -> tuple[BucketTiming, ...]:
        """Direct children of one label. Empty for a leaf."""
        return tuple(b for b in self.buckets if b.parent == label)


def _ancestors(label: str) -> list[str]:
    parts = label.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def budget(timed: Timed) -> BudgetReport:
    """Build the closed budget tree for one measurement.

    Args:
        timed: The measurement, with its regions.

    Returns:
        The budget report. Parents are emitted before their children.

    Raises:
        ValueError: If a parent's median is zero, which makes its children's
            shares undefined and means the label is broken.
    """
    measured: dict[str, Spread] = {t.label: t.spread for t in timed.regions}
    nodes: list[str] = []
    for label in measured:
        for ancestor in [*_ancestors(label), label]:
            if ancestor not in nodes:
                nodes.append(ancestor)

    children: dict[str, list[str]] = {"": []}
    for label in nodes:
        children.setdefault(label, [])
        children.setdefault(parent_of(label), []).append(label)

    value: dict[str, Microseconds] = {}
    for label in sorted(nodes, key=lambda name: -name.count(".")):
        direct = measured.get(label)
        if direct is not None:
            value[label] = direct.median_duration_us
        else:
            value[label] = Microseconds(sum(value[c] for c in children[label]))
    value[""] = timed.total.median_duration_us

    for label in ["", *nodes]:
        kids = children[label]
        if label == "":
            # Always present, so an uninstrumented run reads 100% unattributed
            # rather than reporting an empty tree.
            pass
        elif not kids or label not in measured:
            continue
        remainder = Microseconds(value[label] - sum(value[c] for c in kids))
        leaf = UNATTRIBUTED if label == "" else f"{label}.{UNATTRIBUTED}"
        children[label].append(leaf)
        children[leaf] = []
        value[leaf] = remainder

    out: list[BucketTiming] = []

    def emit(label: str) -> None:
        direct = measured.get(label)
        parent = parent_of(label)
        parent_value = value[parent]
        if parent_value == 0.0:
            raise ValueError(f"bucket {parent or '<total>'!r} is zero; label is broken")
        out.append(
            BucketTiming(
                label=label,
                parent=parent,
                derived=direct is None,
                median_duration_us=value[label],
                spread_pct=direct.spread_pct if direct is not None else Percent(0.0),
                resolution_pct=(
                    direct.resolution_pct if direct is not None else Percent(0.0)
                ),
                coverage_pct=(
                    direct.coverage_pct if direct is not None else Percent(0.0)
                ),
                sample_count=direct.sample_count if direct is not None else Count(0),
                share_of_parent_pct=pct_of(value[label], parent_value),
                share_of_total_pct=pct_of(value[label], value[""]),
            )
        )
        for child in children[label]:
            emit(child)

    for root in children[""]:
        emit(root)
    return BudgetReport(
        label=timed.label,
        clocks=timed.clocks,
        total=timed.total,
        buckets=tuple(out),
    )


def assert_closed(report: BudgetReport, tol_pct: Percent = CLOSURE_TOL_PCT) -> None:
    """Check that every parent equals the sum of its children.

    Args:
        report: The budget report.
        tol_pct: Allowed disagreement, as a percentage of the parent.

    Raises:
        ValueError: On a node whose children do not sum to it, including a node
            that is zero while its children are not. A relative tolerance has no
            meaning against a zero parent, so the absolute comparison is made here
            rather than letting :func:`pct_of` raise about a denominator.
    """
    parents = [""] + [b.label for b in report.buckets]
    for parent in parents:
        kids = report.children(parent)
        if not kids:
            continue
        if parent == "":
            whole = report.total.median_duration_us
        else:
            whole = report.get(parent).median_duration_us
        got = sum(k.median_duration_us for k in kids)
        off = got != 0.0 if whole == 0.0 else abs(pct_of(got - whole, whole)) > tol_pct
        if off:
            raise ValueError(
                f"bucket {parent or '<total>'!r} is {whole:.3f} us and its "
                f"{len(kids)} children sum to {got:.3f} us"
            )


def assert_nonzero(report: BudgetReport, declared: Iterable[str]) -> None:
    """Check that every declared bucket exists and is not exactly zero.

    Args:
        report: The budget report.
        declared: Labels the path under test must populate.

    Raises:
        ValueError: Listing every missing or zero label.
    """
    present = dict(zip(report.labels(), report.buckets))
    missing = [label for label in declared if label not in present]
    zero = [
        label
        for label in declared
        if label in present and present[label].median_duration_us == 0.0
    ]
    if missing or zero:
        raise ValueError(
            f"declared buckets missing {sorted(missing)} and zero {sorted(zero)}; "
            f"a bucket that reads exactly zero is a broken label"
        )


@dataclass(frozen=True)
class BucketDelta(PerfRecord):
    """One bucket's change between two reports.

    ``resolved`` is the only field that licenses a claim. A change smaller than
    the resolution floor of either measurement is noise, whatever its sign.

    Attributes:
        label: Dotted bucket label.
        before_duration_us: Median in the earlier report.
        after_duration_us: Median in the later report.
        delta_pct: Change as a percentage of the earlier median.
        speedup_ratio: Earlier over later.
        floor_pct: Resolution floor this delta had to beat: the sum of the two
            runs' half-widths on this bucket, or on the whole call when the bucket
            is derived.
        resolved: True if ``delta_pct`` exceeds ``floor_pct`` and both intervals
            behind that floor reach nominal coverage.
    """

    label: str
    before_duration_us: Annotated[Microseconds, MEDIAN]
    after_duration_us: Annotated[Microseconds, MEDIAN]
    delta_pct: Annotated[Percent, MEDIAN]
    speedup_ratio: Annotated[Ratio, MEDIAN]
    floor_pct: Annotated[Percent, MEDIAN]
    resolved: bool


def _holds(coverage_pct: Percent) -> bool:
    """Whether a resolution floor over this interval means anything.

    This is :meth:`slinoss.perf.units.Spread.resolves` minus the comparison, which
    cannot be delegated here because a delta is judged against two measurements
    and each carries its own interval.

    Args:
        coverage_pct: Exact coverage of the interval behind the floor.

    Returns:
        True if the interval reaches nominal coverage.
    """
    return coverage_pct >= CONFIDENCE_PCT


def compare(before: BudgetReport, after: BudgetReport) -> tuple[BucketDelta, ...]:
    """Diff two budget reports over the buckets they share.

    Args:
        before: The earlier report.
        after: The later report.

    Returns:
        One delta per shared label, largest regression first.

    Raises:
        ValueError: If the reports share no bucket, which means they measured
            different things and must not be compared.
    """
    shared = [label for label in before.labels() if label in set(after.labels())]
    if not shared:
        raise ValueError(
            f"{before.label!r} and {after.label!r} share no bucket; "
            f"these are not two measurements of one thing"
        )
    # A half-width is not a range. Two medians are distinguishable only when they
    # lie further apart than the sum of their two half-widths, so the floor for a
    # delta is that sum. Taking the larger of the pair would license a claim at up
    # to twice the noise.
    whole = Percent(before.total.resolution_pct + after.total.resolution_pct)
    whole_holds = _holds(
        Percent(min(before.total.coverage_pct, after.total.coverage_pct))
    )
    out: list[BucketDelta] = []
    for label in shared:
        was_bucket = before.get(label)
        now_bucket = after.get(label)
        was = was_bucket.median_duration_us
        now = now_bucket.median_duration_us
        delta = pct_of(now - was, was)
        # A bucket is judged against its own floor, which is the noise the delta
        # actually had to beat. A derived bucket has none of its own, so it falls
        # back to the whole-call floor rather than resolving on nothing.
        if was_bucket.derived or now_bucket.derived:
            floor, holds = whole, whole_holds
        else:
            floor = Percent(was_bucket.resolution_pct + now_bucket.resolution_pct)
            holds = _holds(
                Percent(min(was_bucket.coverage_pct, now_bucket.coverage_pct))
            )
        out.append(
            BucketDelta(
                label=label,
                before_duration_us=was,
                after_duration_us=now,
                delta_pct=delta,
                speedup_ratio=ratio_of(was, now),
                floor_pct=floor,
                resolved=holds and abs(delta) > floor,
            )
        )
    return tuple(sorted(out, key=lambda d: -d.delta_pct))


def rank(deltas: Sequence[BucketDelta], top: int) -> tuple[BucketDelta, ...]:
    """The largest resolved regressions, worst first.

    Args:
        deltas: Output of :func:`compare`.
        top: How many to keep.

    Returns:
        Up to ``top`` resolved regressions. Unresolved deltas are dropped, since
        an unresolved regression is not a regression. The order is imposed here
        rather than inherited, so a hand-built sequence cannot make the list read
        worst-first when it is not.
    """
    kept = [d for d in deltas if d.resolved and d.delta_pct > 0.0]
    return tuple(sorted(kept, key=lambda d: -d.delta_pct))[:top]
