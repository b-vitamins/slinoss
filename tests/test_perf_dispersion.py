"""Dispersion against the sample count, and the floor against observed scatter.

Every sample list here is a literal, so each row is exact arithmetic over known
inputs and a failure names the statistic rather than the clock. The outlier series
is the point of the module: one 400 us stall pins the full range at every prefix
while the confidence interval on the median closes by two orders of magnitude over
the same samples.

The paired series is the point of :func:`slinoss.perf.dispersion.paired`: an arm
drifting by 62 percent of its own median still resolves a 0.69 percent difference
against the other arm, because the drift is common to the pair and cancels out of
the difference. The position series is the case that drift argument does not cover.
A cost that follows the launch order rather than the arm is not common to the pair,
it splits the differences into one cluster per order, and pooling them puts the
interval across the gap between the two.
"""

from __future__ import annotations

import pytest

from slinoss.perf.dispersion import growth, paired, repeats
from slinoss.perf.units import (
    CONFIDENCE_PCT,
    MIN_RESOLVING_SAMPLES,
    Microseconds,
    Spread,
    median_ci,
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
    # The whole list is the last row once, whether or not the stride divides it.
    assert tuple(r.sample_count for r in growth(outlier_series()[:12], 6)) == (6, 12)
    assert tuple(r.sample_count for r in growth(outlier_series()[:14], 6)) == (
        6,
        12,
        14,
    )
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


def test_growth_rejects_arguments_that_define_no_prefix() -> None:
    empty: list[Microseconds] = []
    with pytest.raises(ValueError, match="growth needs at least one sample"):
        growth(empty, 4)
    for stride in (0, -1):
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
    # Medians 100, 100, and 104. The 300 us stall inside one run moves that run's
    # range and nothing else, so it stays out of the scatter entirely.
    assert row.median_duration_us == 100.0
    assert row.min_duration_us == 100.0
    assert row.max_duration_us == 104.0
    assert row.scatter_pct == 4.0


def test_the_floor_holds_up_to_twice_the_scatter_and_no_further() -> None:
    # The floor is 4 percent in all three, and the scatter is a gap between two
    # medians that each carry a half-width, so the budget is 8. The boundary is
    # exact: a scatter equal to twice the floor holds and anything wider does not.
    for median_us, scatter_pct, holds in (
        (107.0, 7.0, True),
        (108.0, 8.0, True),
        (109.0, 9.0, False),
    ):
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


def test_repeats_rejects_fewer_than_two_runs() -> None:
    for count in (0, 1):
        with pytest.raises(ValueError, match="repeats needs at least two runs"):
            repeats("step", [run_wide()] * count)


# ---------------------------------------------------------------------------
# paired
# ---------------------------------------------------------------------------


def drifting() -> list[Microseconds]:
    """Ten samples climbing from 100 to 190 us. Median 145, range 90."""
    return us(*(100.0 + 10.0 * i for i in range(10)))


def test_pairing_resolves_a_delta_far_below_either_arm_own_floor() -> None:
    slow = drifting()
    fast = us(*(v + 1.0 for v in slow))
    row = paired("scan", "reference", slow, "cute", fast)
    assert row.label == "scan"
    assert row.a_label == "reference"
    assert row.b_label == "cute"
    assert row.sample_count == 10
    assert row.a_median_duration_us == 145.0
    assert row.b_median_duration_us == 146.0
    assert row.delta_median_duration_us == 1.0
    assert row.delta_pct == pytest.approx(100.0 / 145.0)
    assert row.speedup_ratio == pytest.approx(145.0 / 146.0)
    assert row.resolves
    # Neither arm alone can see this. Each drifts across 62 percent of its own
    # median and carries a 24 percent floor, against a difference of 0.69 percent.
    # The drift is shared by the pair, so it leaves the difference untouched.
    own = Spread.of(slow)
    assert own.spread_pct == pytest.approx(100.0 * 90.0 / 145.0)
    assert own.resolution_pct == pytest.approx(100.0 * 35.0 / 145.0)
    assert not own.resolves(row.delta_pct)
    # The interval is two of the differences themselves, at rank two of ten: the
    # tightest interval reaching nominal coverage.
    assert row.coverage_pct == 97.8515625
    assert row.delta_low_duration_us == 1.0
    assert row.delta_high_duration_us == 1.0
    # Both launch orders read the same difference, so there is nothing to remove
    # and the correction is exactly zero.
    assert row.position_duration_us == 0.0


def test_a_pure_position_effect_resolves_nothing() -> None:
    flat = us(*([100.0] * 8))
    alternating = us(105.0, 95.0, 105.0, 95.0, 105.0, 95.0, 105.0, 95.0)
    row = paired("scan", "a", flat, "b", alternating)
    assert row.a_median_duration_us == 100.0
    assert row.b_median_duration_us == 100.0
    assert row.delta_median_duration_us == 0.0
    assert row.speedup_ratio == 1.0
    # The swing follows the launch order and nothing else: 5 us one way in the
    # iterations that ran a first, 5 us the other way in the iterations that ran b
    # first. That is the whole of the difference, so once it is removed the two
    # arms cost the same to the last bit.
    assert row.position_duration_us == 5.0
    assert row.delta_low_duration_us == 0.0
    assert row.delta_high_duration_us == 0.0
    assert not row.resolves
    # The verdict names no arm and prints no ratio, so it cannot be quoted as a
    # result. It does print what the order was worth.
    line = row.verdict()
    assert line == (
        "scan: no difference measured between a and b; the interval "
        "[0.000, 0.000] us at 99.219% coverage over 8 pairs does not exclude zero; "
        "position 5.000 us removed"
    )
    assert "beats" not in line
    assert "speedup" not in line


def test_a_difference_smaller_than_the_position_effect_still_resolves() -> None:
    # The arm under test costs 6 us more, and whichever arm runs second pays 18.5 us
    # on top of that. Ten pairs: the iterations that ran a first read +24.5 and the
    # iterations that ran b first read -12.5, two clusters 37 us apart with nothing
    # between them.
    a = us(*([100.0, 118.5] * 5))
    b = us(*([124.5, 106.0] * 5))
    raw = [Microseconds(y - x) for x, y in zip(a, b)]
    assert set(raw) == {24.5, -12.5}
    # Pooled, those differences are one sample list with a hole in the middle. The
    # interval on their median is two of them, so it reaches from one cluster to the
    # other, covers zero, and refuses a difference that every pair agrees on.
    low, high, _ = median_ci(raw)
    assert low == -12.5 and high == 24.5
    row = paired("scan", "a", a, "b", b)
    # Split by launch order the position term enters the two halves with opposite
    # signs, so it cancels in their mean and is what is left of their difference.
    assert row.position_duration_us == 18.5
    assert row.delta_median_duration_us == 6.0
    assert row.delta_low_duration_us == 6.0
    assert row.delta_high_duration_us == 6.0
    assert row.resolves
    # Each arm ran first half the time, so the two medians already carry the same
    # half of the position cost and the ratio never needed the correction.
    assert row.a_median_duration_us == 109.25
    assert row.b_median_duration_us == 115.25
    assert row.speedup_ratio == pytest.approx(109.25 / 115.25)
    assert row.delta_pct == pytest.approx(600.0 / 109.25)


def test_one_pair_admits_no_position_estimate() -> None:
    # One iteration ran one order, so there is no second order to difference it
    # against and the correction cannot be estimated. It is reported as zero rather
    # than guessed, and one pair resolves nothing anyway.
    row = paired("scan", "a", us(100.0), "b", us(90.0))
    assert row.position_duration_us == 0.0
    assert row.delta_median_duration_us == -10.0
    assert row.coverage_pct < CONFIDENCE_PCT
    assert not row.resolves


def test_a_consistent_difference_below_nominal_coverage_resolves_nothing() -> None:
    count = MIN_RESOLVING_SAMPLES - 1
    row = paired("scan", "a", us(*([100.0] * count)), "b", us(*([50.0] * count)))
    # Every pair agrees that the second arm is half the cost, and the interval is
    # one point wide. Too few pairs to bound a median at nominal coverage, so the
    # verdict refuses; the figures still print.
    assert row.delta_median_duration_us == -50.0
    assert row.delta_pct == -50.0
    assert row.speedup_ratio == 2.0
    assert row.coverage_pct < CONFIDENCE_PCT
    assert not row.resolves


def test_the_verdict_names_whichever_arm_is_faster() -> None:
    slow = drifting()
    line = paired(
        "scan", "reference", slow, "cute", us(*(v - 10.0 for v in slow))
    ).verdict()
    assert line.startswith("scan: cute beats reference by 10.000 us")
    assert "speedup_ratio 1.074" in line
    assert "[-10.000, -10.000] us at 97.852% coverage over 10 pairs" in line
    assert "excludes zero" in line
    # The baseline is named the same way when the baseline wins, so neither
    # direction reads more strongly than the other.
    other = paired("scan", "reference", slow, "cute", us(*(v + 10.0 for v in slow)))
    assert other.verdict().startswith("scan: reference beats cute by 10.000 us")


def test_paired_rejects_samples_it_cannot_pair() -> None:
    empty: list[Microseconds] = []
    one = us(100.0)
    for a, b in ((empty, one), (one, empty)):
        with pytest.raises(ValueError, match="at least one sample in each arm"):
            paired("scan", "a", a, "b", b)
    # Unequal counts mean the two arms did not run in the same iterations, so
    # element i of one is not the partner of element i of the other.
    with pytest.raises(ValueError, match="these are not pairs"):
        paired("scan", "a", us(100.0, 100.0, 100.0), "b", us(90.0, 90.0, 90.0, 90.0))
    # A zero median leaves the ratio and the percentage undefined.
    zero = us(0.0, 0.0)
    hundred = us(100.0, 100.0)
    for a, b in ((zero, hundred), (hundred, zero)):
        with pytest.raises(ValueError, match="nonzero median in each arm"):
            paired("scan", "a", a, "b", b)
