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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from slinoss.perf.units import (
    CONFIDENCE_PCT,
    MEDIAN,
    SUM,
    Count,
    Microseconds,
    Percent,
    PerfRecord,
    Spread,
)

__all__ = [
    "GrowthRow",
    "RepeatRow",
    "growth",
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
