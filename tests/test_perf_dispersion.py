"""Dispersion against the sample count, and the floor against observed scatter.

Every sample list here is a literal, so each row is exact arithmetic over known
inputs and a failure names the statistic rather than the clock. The outlier series
is the point of the module: one 400 us stall pins the full range at every prefix
while the confidence interval on the median closes by two orders of magnitude over
the same samples.
"""

from __future__ import annotations

import pytest

from slinoss.perf.dispersion import growth, repeats
from slinoss.perf.units import (
    CONFIDENCE_PCT,
    MIN_RESOLVING_SAMPLES,
    Microseconds,
    Spread,
)


def us(*values: float) -> list[Microseconds]:
    """Microsecond samples from raw floats."""
    return [Microseconds(v) for v in values]


def outlier_series() -> list[Microseconds]:
    """Twenty samples around 100 us with one 400 us stall at index three.

    The stall lands inside the first prefix and every prefix after it, so the range
    is pinned by it throughout while the bulk closes around the median.
    """
    return us(
        100.0,
        102.0,
        98.0,
        400.0,
        101.0,
        99.0,
        100.0,
        103.0,
        97.0,
        100.0,
        102.0,
        98.0,
        101.0,
        99.0,
        100.0,
        103.0,
        97.0,
        101.0,
        99.0,
        100.0,
    )


def run_wide() -> Spread:
    """Six samples, median 100 us, range 8 us. The largest floor of the three."""
    return Spread.of(us(96.0, 99.0, 100.0, 100.0, 101.0, 104.0))


def run_tailed() -> Spread:
    """Ten samples, median 100 us, one 300 us stall. The largest range of the three."""
    return Spread.of(
        us(98.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 300.0)
    )


def run_at(median_us: float) -> Spread:
    """Seven samples spanning 2 us, with ``median_us`` as their median."""
    return Spread.of(
        us(*(median_us + d for d in (-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0)))
    )


# ---------------------------------------------------------------------------
# growth
# ---------------------------------------------------------------------------


def test_growth_rows_are_rising_prefixes_of_one_run() -> None:
    rows = growth(outlier_series(), 6)
    assert tuple(r.sample_count for r in rows) == (6, 12, 18, 20)
    # Each row summarizes a prefix, so the extremes are monotone in the count.
    assert tuple(r.min_duration_us for r in rows) == (98.0, 97.0, 97.0, 97.0)
    assert tuple(r.max_duration_us for r in rows) == (400.0, 400.0, 400.0, 400.0)
    assert tuple(r.median_duration_us for r in rows) == (100.5, 100.0, 100.0, 100.0)


def test_growth_ends_on_the_whole_list_without_duplicating_it() -> None:
    twelve = outlier_series()[:12]
    assert tuple(r.sample_count for r in growth(twelve, 6)) == (6, 12)
    fourteen = outlier_series()[:14]
    assert tuple(r.sample_count for r in growth(fourteen, 6)) == (6, 12, 14)
    # A stride past the end still reports the whole list, once.
    assert tuple(r.sample_count for r in growth(outlier_series()[:5], 50)) == (5,)


def test_the_range_holds_while_the_floor_falls() -> None:
    rows = growth(outlier_series(), 6)
    ranges = [r.spread_pct for r in rows]
    floors = [r.resolution_pct for r in rows]
    assert ranges == pytest.approx([100.0 * 302.0 / 100.5, 303.0, 303.0, 303.0])
    assert floors == pytest.approx([100.0 * 151.0 / 100.5, 2.0, 1.5, 1.0])
    # The stall stays inside the range at every count, so the range never falls.
    assert ranges == sorted(ranges)
    # It falls outside the confidence interval from twelve samples on, so the floor
    # never rises. Only the floor bounds what these samples can resolve.
    assert floors == sorted(floors, reverse=True)


def test_resolves_gates_on_the_sample_count_and_not_on_the_floor() -> None:
    rows = growth(outlier_series()[: MIN_RESOLVING_SAMPLES + 2], 1)
    assert tuple(r.sample_count for r in rows) == tuple(
        range(1, MIN_RESOLVING_SAMPLES + 3)
    )
    for row in rows:
        assert row.resolves is (row.sample_count >= MIN_RESOLVING_SAMPLES)
        assert row.resolves is (row.coverage_pct >= CONFIDENCE_PCT)
    # The four-sample row carries a floor of 149 percent and still refuses, so the
    # gate is the coverage the samples admit and not the width of the interval. The
    # floor is still printed, beside the coverage that says what it is worth.
    assert rows[3].resolution_pct > 100.0
    assert rows[3].coverage_pct == 87.5
    assert not rows[3].resolves


def test_growth_rejects_an_empty_sample_list() -> None:
    empty: list[Microseconds] = []
    with pytest.raises(ValueError, match="growth needs at least one sample"):
        growth(empty, 4)


@pytest.mark.parametrize("stride", [0, -1])
def test_growth_rejects_a_non_positive_stride(stride: int) -> None:
    with pytest.raises(ValueError, match="stride must be positive"):
        growth(outlier_series(), stride)


# ---------------------------------------------------------------------------
# repeats
# ---------------------------------------------------------------------------


def test_repeats_reduces_across_runs_by_the_worst_case() -> None:
    row = repeats("step", (run_wide(), run_tailed(), run_at(104.0)))
    assert row.label == "step"
    assert row.run_count == 3
    # The smallest count, because it is the weakest run behind the claim.
    assert row.sample_count == 6
    # The largest floor and the largest range, contributed by different runs.
    assert row.floor_pct == 4.0
    assert row.spread_pct == 202.0
    # The smallest coverage, from the six-sample run.
    assert row.coverage_pct == 96.875


def test_repeats_scatters_the_per_run_medians_and_not_their_samples() -> None:
    row = repeats("step", (run_wide(), run_tailed(), run_at(104.0)))
    # Medians 100, 100, and 104. The 300 us stall inside one run moves that run's
    # range and nothing else, so it stays out of the scatter entirely.
    assert row.median_duration_us == 100.0
    assert row.min_duration_us == 100.0
    assert row.max_duration_us == 104.0
    assert row.scatter_pct == 4.0


@pytest.mark.parametrize(
    ("median_us", "scatter_pct", "holds"),
    [(107.0, 7.0, True), (108.0, 8.0, True), (109.0, 9.0, False)],
)
def test_the_floor_holds_up_to_twice_the_scatter_and_no_further(
    median_us: float, scatter_pct: float, holds: bool
) -> None:
    # The floor is 4 percent in all three, and the scatter is a gap between two
    # medians that each carry a half-width, so the budget is 8. The boundary is
    # exact: a scatter equal to twice the floor holds and anything wider does not.
    row = repeats("step", (run_wide(), run_tailed(), run_at(median_us)))
    assert row.floor_pct == 4.0
    assert row.scatter_pct == scatter_pct
    assert row.floor_holds is holds


def test_the_floor_cannot_hold_below_nominal_coverage() -> None:
    # Five samples admit no interval at the nominal coverage, so there is no floor
    # for even a zero scatter to fall under.
    thin = [Spread.of(us(99.0, 100.0, 100.0, 100.0, 101.0)) for _ in range(2)]
    row = repeats("step", thin)
    assert row.floor_pct == 1.0
    assert row.coverage_pct < CONFIDENCE_PCT
    assert row.scatter_pct == 0.0
    assert not row.floor_holds


@pytest.mark.parametrize("count", [0, 1])
def test_repeats_rejects_fewer_than_two_runs(count: int) -> None:
    with pytest.raises(ValueError, match="repeats needs at least two runs"):
        repeats("step", [run_wide()] * count)
