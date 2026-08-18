"""How a measurement's dispersion behaves, and whether its floor holds.

Two statistics are reported beside every median. Only one of them is a floor, and
which one is not a matter of taste:

- The full range grows with the sample count. A wider sample has more chance to
  contain the tail of the distribution, so the range is an outlier detector and
  not a bound on how well the median is pinned.
- The half-width of the median's confidence interval shrinks with the sample
  count. That is the bound, and it is the one :meth:`slinoss.perf.units.Spread.resolves`
  applies.

:func:`growth` shows the first behaviour from prefixes of one run. :func:`repeats`
checks the second against the thing it predicts: the scatter of medians over
independent measurements of identical work. Two medians are distinguishable only
when they lie further apart than the sum of their two half-widths, so a floor
under half that scatter is not a floor, and this is the only check that can say
so.

A within-run half-width does not bound a between-run difference, and on this
fleet it does not: repeated identical work scatters further than twice the floor
at several shapes. Both statistics above are within-run, so neither licenses a
delta taken from two separate runs, whatever the iteration count.
:func:`paired` is the way out. It compares two implementations inside one run,
alternating which goes first, and judges the per-iteration difference rather than
the difference of two medians. Whatever drifts between runs is common to both
arms of a pair and cancels; what is left is the difference, and its interval
comes off the same order statistics with no scale and no distribution assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Annotated

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
    median_ci,
    pct_of,
    ratio_of,
)

__all__ = [
    "GrowthRow",
    "PairedRow",
    "RepeatRow",
    "growth",
    "paired",
    "repeats",
]


@dataclass(frozen=True)
class GrowthRow(PerfRecord):
    """Every statistic over the first ``sample_count`` samples of one run.

    Attributes:
        sample_count: Prefix length.
        median_duration_us: Median over the prefix.
        min_duration_us: Fastest sample in the prefix.
        max_duration_us: Slowest sample in the prefix.
        spread_pct: Full range over the median.
        resolution_pct: Half-width of the median's confidence interval.
        coverage_pct: Exact coverage of that interval. The early rows carry the
            widest interval their samples admit, which is short of nominal, and
            this is the column that says so.
        resolves: Whether a delta of any size can resolve at this sample count.
    """

    sample_count: Annotated[Count, SUM]
    median_duration_us: Annotated[Microseconds, MEDIAN]
    min_duration_us: Annotated[Microseconds, MEDIAN]
    max_duration_us: Annotated[Microseconds, MEDIAN]
    spread_pct: Annotated[Percent, MEDIAN]
    resolution_pct: Annotated[Percent, MEDIAN]
    coverage_pct: Annotated[Percent, MEDIAN]
    resolves: bool


@dataclass(frozen=True)
class RepeatRow(PerfRecord):
    """Observed run-to-run scatter of the median, against the reported floor.

    Attributes:
        label: What was repeated.
        run_count: Independent measurements.
        sample_count: Samples in each of them.
        median_duration_us: Median of the per-run medians.
        min_duration_us: Fastest per-run median.
        max_duration_us: Slowest per-run median.
        scatter_pct: Range of the per-run medians over their median. This is the
            reproducibility the floor is supposed to predict.
        floor_pct: Largest half-width any of the runs reported.
        coverage_pct: Smallest coverage any of the runs reported.
        spread_pct: Largest full range any of the runs reported.
        floor_holds: Whether the floor covers the observed scatter. The scatter is
            a range between two medians and the floor is one median's half-width,
            so the comparison is against twice the floor: two medians are
            distinguishable only when they lie further apart than the sum of their
            two half-widths. False whenever a run's interval misses nominal
            coverage, since a floor that means nothing cannot hold.
    """

    label: str
    run_count: Annotated[Count, SUM]
    sample_count: Annotated[Count, SUM]
    median_duration_us: Annotated[Microseconds, MEDIAN]
    min_duration_us: Annotated[Microseconds, MEDIAN]
    max_duration_us: Annotated[Microseconds, MEDIAN]
    scatter_pct: Annotated[Percent, MEDIAN]
    floor_pct: Annotated[Percent, MEDIAN]
    coverage_pct: Annotated[Percent, MEDIAN]
    spread_pct: Annotated[Percent, MEDIAN]
    floor_holds: bool


@dataclass(frozen=True)
class PairedRow(PerfRecord):
    """Two implementations measured against each other inside one run.

    The samples are paired by iteration, so the statistic is the median of the
    per-iteration differences and not the difference of two medians. A clock
    excursion, a cache eviction, or another tenant arriving hits both arms of a
    pair and cancels out of the difference. Nothing between runs enters at all.

    The interval on that median is two of the differences themselves, so the
    verdict is a comparison against zero in the unit the differences are measured
    in. A speedup is claimed only when the whole interval sits on one side of
    zero.

    Attributes:
        label: What was compared.
        a_label: The baseline arm.
        b_label: The arm under test.
        sample_count: Pairs. One per iteration.
        a_median_duration_us: Median of the baseline arm.
        b_median_duration_us: Median of the arm under test.
        delta_median_duration_us: Median of ``b - a`` over the pairs. Negative
            means the arm under test is faster.
        delta_low_duration_us: Lower bound of the interval on that median.
        delta_high_duration_us: Upper bound of the interval on that median.
        coverage_pct: Exact coverage of that interval. Short of
            :data:`slinoss.perf.units.CONFIDENCE_PCT` nothing resolves, however
            far the interval sits from zero.
        delta_pct: The median difference over the baseline median.
        speedup_ratio: Baseline median over the median under test. Above one means
            faster.
        resolves: True only if the interval reaches nominal coverage and excludes
            zero. This is the only field that licenses a claim.
    """

    label: str
    a_label: str
    b_label: str
    sample_count: Annotated[Count, SUM]
    a_median_duration_us: Annotated[Microseconds, MEDIAN]
    b_median_duration_us: Annotated[Microseconds, MEDIAN]
    delta_median_duration_us: Annotated[Microseconds, MEDIAN]
    delta_low_duration_us: Annotated[Microseconds, MEDIAN]
    delta_high_duration_us: Annotated[Microseconds, MEDIAN]
    coverage_pct: Annotated[Percent, MEDIAN]
    delta_pct: Annotated[Percent, MEDIAN]
    speedup_ratio: Annotated[Ratio, MEDIAN]
    resolves: bool


def paired(
    label: str,
    a_label: str,
    a_samples: Sequence[Microseconds],
    b_label: str,
    b_samples: Sequence[Microseconds],
) -> PairedRow:
    """Judge two arms measured in the same iterations.

    Args:
        label: What was compared.
        a_label: The baseline arm.
        a_samples: Its per-iteration durations, in measurement order.
        b_label: The arm under test.
        b_samples: Its per-iteration durations, in the same order.

    Returns:
        The comparison. ``resolves`` is the verdict.

    Raises:
        ValueError: If either arm is empty, if the two arms carry different sample
            counts, which means they were not measured in the same iterations, or
            if either median is zero, which leaves the ratio and the percentage
            undefined.
    """
    if not a_samples or not b_samples:
        raise ValueError("paired needs at least one sample in each arm")
    if len(a_samples) != len(b_samples):
        raise ValueError(
            f"paired needs one sample per arm per iteration, got {len(a_samples)} "
            f"for {a_label!r} and {len(b_samples)} for {b_label!r}; these are not "
            f"pairs"
        )
    a_median = Microseconds(median(a_samples))
    b_median = Microseconds(median(b_samples))
    if a_median == 0.0 or b_median == 0.0:
        raise ValueError(
            f"paired needs a nonzero median in each arm, got {a_median} for "
            f"{a_label!r} and {b_median} for {b_label!r}"
        )
    deltas = [Microseconds(b - a) for a, b in zip(a_samples, b_samples)]
    low, high, coverage = median_ci(deltas)
    return PairedRow(
        label=label,
        a_label=a_label,
        b_label=b_label,
        sample_count=Count(len(deltas)),
        a_median_duration_us=a_median,
        b_median_duration_us=b_median,
        delta_median_duration_us=Microseconds(median(deltas)),
        delta_low_duration_us=low,
        delta_high_duration_us=high,
        coverage_pct=coverage,
        delta_pct=pct_of(median(deltas), a_median),
        speedup_ratio=ratio_of(a_median, b_median),
        # An interval straddling zero is consistent with no difference at all, so
        # the sign of the median it contains licenses nothing.
        resolves=coverage >= CONFIDENCE_PCT and (low > 0.0 or high < 0.0),
    )


def growth(samples: Sequence[Microseconds], stride: int) -> tuple[GrowthRow, ...]:
    """Summarize rising prefixes of one sample list.

    The prefixes share their samples, so the rows show the shape of each statistic
    against the sample count and not its own sampling error. For that, repeat the
    measurement and use :func:`repeats`.

    Args:
        samples: The samples, in measurement order. At least one.
        stride: Prefix stride. The whole list is always the last row, whether or
            not its length is a multiple of the stride.

    Returns:
        One row per prefix, shortest first.

    Raises:
        ValueError: If ``samples`` is empty or ``stride`` is not positive.
    """
    if not samples:
        raise ValueError("growth needs at least one sample")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    sizes = list(range(stride, len(samples) + 1, stride))
    if sizes[-1:] != [len(samples)]:
        sizes.append(len(samples))
    out: list[GrowthRow] = []
    for size in sizes:
        spread = Spread.of(samples[:size])
        out.append(
            GrowthRow(
                sample_count=spread.sample_count,
                median_duration_us=spread.median_duration_us,
                min_duration_us=spread.min_duration_us,
                max_duration_us=spread.max_duration_us,
                spread_pct=spread.spread_pct,
                resolution_pct=spread.resolution_pct,
                coverage_pct=spread.coverage_pct,
                # A delta this large is beyond any plausible floor, so the answer
                # is the coverage gate alone.
                resolves=spread.resolves(Percent(1e9)),
            )
        )
    return tuple(out)


def repeats(label: str, runs: Sequence[Spread]) -> RepeatRow:
    """Compare the scatter of independent medians against the reported floor.

    Args:
        label: What was repeated.
        runs: One spread per independent measurement of identical work, at least
            two.

    Returns:
        The comparison. ``floor_holds`` false means the floor is optimistic on
        this host at this shape and every delta judged against it is suspect.

    Raises:
        ValueError: If fewer than two runs are given, which leaves no scatter to
            measure.
    """
    if len(runs) < 2:
        raise ValueError(f"repeats needs at least two runs, got {len(runs)}")
    spread = Spread.of([r.median_duration_us for r in runs])
    floor = Percent(max(r.resolution_pct for r in runs))
    coverage = Percent(min(r.coverage_pct for r in runs))
    return RepeatRow(
        label=label,
        run_count=Count(len(runs)),
        sample_count=Count(min(r.sample_count for r in runs)),
        median_duration_us=spread.median_duration_us,
        min_duration_us=spread.min_duration_us,
        max_duration_us=spread.max_duration_us,
        scatter_pct=spread.spread_pct,
        floor_pct=floor,
        coverage_pct=coverage,
        spread_pct=Percent(max(r.spread_pct for r in runs)),
        floor_holds=coverage >= CONFIDENCE_PCT and spread.spread_pct <= 2.0 * floor,
    )
