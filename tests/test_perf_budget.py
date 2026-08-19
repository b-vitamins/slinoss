"""The closed budget tree over one measured total.

Every record here is built from synthetic samples and ``RegionTiming`` literals
rather than by timing anything, so the tree arithmetic is exact and what fails is
the taxonomy, never the clock. Medians are the sample values verbatim; the
dispersion percentages are ``pct_of`` over binary floats, so those are compared
approximately.
"""

from __future__ import annotations

import pytest

from slinoss.perf.budget import (
    CLOSURE_TOL_PCT,
    STEP_BUCKETS,
    UNATTRIBUTED,
    BucketTiming,
    BudgetReport,
    assert_closed,
    assert_nonzero,
    budget,
    compare,
    rank,
)
from slinoss.perf.timing import RegionTiming, Timed, parent_of
from slinoss.perf.units import (
    CONFIDENCE_PCT,
    MIN_RESOLVING_SAMPLES,
    Count,
    Microseconds,
    Percent,
    Spread,
    pct_of,
)


def spread_of(us: float, *, pct: float = 4.0, count: int = 8) -> Spread:
    """A Spread over samples whose full range is ``pct`` of the median ``us``.

    Built through ``Spread.of``, so every derived field is real arithmetic. All but
    the two extreme samples sit on the median, so through eight samples the
    median's interval spans the whole range and ``resolution_pct`` is half of
    ``pct``; at nine it tightens and that identity fails. The default count is
    above ``MIN_RESOLVING_SAMPLES``, or nothing built from it would resolve.
    """
    half = us * pct / 200.0
    inner = [Microseconds(us)] * (count - 2)
    return Spread.of([Microseconds(us - half), *inner, Microseconds(us + half)])


def timed_of(
    total_us: float,
    regions: dict[str, float],
    *,
    label: str = "step",
    total_pct: float = 1.0,
    region_pct: float = 4.0,
    count: int = 8,
) -> Timed:
    """A measurement whose regions carry the given median durations.

    One sample count covers the total and every region, as a real loop gives each
    region one sample per iteration.
    """
    rows = tuple(
        RegionTiming(
            label=name,
            parent=parent_of(name),
            spread=spread_of(us, pct=region_pct, count=count),
            share_pct=pct_of(us, total_us) if total_us else Percent(0.0),
        )
        for name, us in regions.items()
    )
    roots = [r.spread.median_duration_us for r in rows if not r.parent]
    return Timed(
        label=label,
        timer="perf_counter",
        clocks="unlocked",
        total=spread_of(total_us, pct=total_pct, count=count),
        regions=rows,
        root_sum_duration_us=Microseconds(sum(roots)),
        timer_coverage_pct=Percent(99.0),
    )


def hand_bucket(label: str, us: float, whole: float) -> BucketTiming:
    """One bucket of a report that :func:`budget` would never build."""
    share = pct_of(us, whole)
    return BucketTiming(
        label=label,
        parent=parent_of(label),
        derived=True,
        median_duration_us=Microseconds(us),
        spread_pct=Percent(0.0),
        resolution_pct=Percent(0.0),
        coverage_pct=Percent(0.0),
        sample_count=Count(0),
        share_of_parent_pct=share,
        share_of_total_pct=share,
    )


def hand_report(
    total_us: float, buckets: tuple[BucketTiming, ...], *, label: str = "hand"
) -> BudgetReport:
    """A report assembled by hand, to drive the assertions in the checkers."""
    return BudgetReport(
        label=label, clocks="unlocked", total=spread_of(total_us), buckets=buckets
    )


def two_reports() -> tuple[BudgetReport, BudgetReport]:
    """Two comparable reports with four distinct per-bucket deltas."""
    before = budget(
        timed_of(100.0, {"step.a": 50.0, "step.b": 25.0}, label="before", total_pct=1.0)
    )
    after = budget(
        timed_of(105.0, {"step.a": 40.0, "step.b": 30.0}, label="after", total_pct=8.0)
    )
    return before, after


def test_step_taxonomy_is_a_flat_set_under_one_root() -> None:
    assert STEP_BUCKETS[0] == "step.zero_grad"
    assert {parent_of(name) for name in STEP_BUCKETS} == {"step"}
    assert UNATTRIBUTED not in STEP_BUCKETS


def test_derived_node_is_the_sum_of_its_children() -> None:
    report = budget(
        timed_of(100.0, {"step.forward.mixer": 30.0, "step.forward.ffn": 20.0})
    )
    assert report.labels() == (
        "step",
        "step.forward",
        "step.forward.mixer",
        "step.forward.ffn",
        "unattributed",
    )
    forward = report.get("step.forward")
    assert forward.derived
    assert forward.spread_pct == 0.0
    assert forward.resolution_pct == 0.0
    assert forward.coverage_pct == 0.0
    assert forward.sample_count == 0
    assert forward.median_duration_us == 50.0
    kids = report.children("step.forward")
    assert tuple(b.label for b in kids) == ("step.forward.mixer", "step.forward.ffn")
    assert forward.median_duration_us == sum(b.median_duration_us for b in kids)
    mixer = report.get("step.forward.mixer")
    assert not mixer.derived
    assert mixer.spread_pct == pytest.approx(4.0)
    assert mixer.resolution_pct == pytest.approx(2.0)
    assert mixer.coverage_pct >= CONFIDENCE_PCT
    assert mixer.sample_count == 8
    assert mixer.share_of_parent_pct == 60.0
    assert mixer.share_of_total_pct == 30.0
    assert report.children("step.forward.mixer") == ()
    assert_closed(report)


def test_measured_parent_gains_an_unattributed_child() -> None:
    report = budget(timed_of(100.0, {"step": 80.0, "step.a": 30.0, "step.b": 20.0}))
    assert report.labels() == (
        "step",
        "step.a",
        "step.b",
        "step.unattributed",
        "unattributed",
    )
    rest = report.get("step.unattributed")
    assert rest.derived
    assert rest.sample_count == 0
    assert rest.median_duration_us == 30.0
    assert rest.share_of_parent_pct == 37.5
    assert rest.share_of_total_pct == 30.0
    assert report.get("unattributed").median_duration_us == 20.0
    assert_closed(report)


def test_top_remainder_takes_the_whole_call_and_is_not_clamped() -> None:
    whole = budget(timed_of(100.0, {}))
    assert whole.labels() == ("unattributed",)
    rest = whole.get("unattributed")
    assert rest.derived
    assert rest.median_duration_us == 100.0
    assert rest.share_of_total_pct == 100.0
    assert_closed(whole)
    # Over-attribution is reported as a negative remainder. Clamping it to zero
    # would close the tree by hiding the overlap that caused it.
    over = budget(timed_of(100.0, {"a": 70.0, "b": 60.0}))
    negative = over.get("unattributed")
    assert negative.median_duration_us == -30.0
    assert negative.share_of_total_pct < 0.0
    assert_closed(over)


def test_budget_rejects_a_zero_parent() -> None:
    for total_us, regions, match in (
        (100.0, {"step": 0.0, "step.a": 0.0}, r"bucket 'step' is zero"),
        (0.0, {"a": 5.0}, r"bucket '<total>' is zero"),
    ):
        with pytest.raises(ValueError, match=match):
            budget(timed_of(total_us, regions))


def test_assert_closed_checks_every_parent_against_its_children() -> None:
    report = hand_report(
        100.0, (hand_bucket("step", 100.0, 100.0), hand_bucket("step.a", 10.0, 100.0))
    )
    with pytest.raises(ValueError, match=r"bucket 'step' is 100\.000 us"):
        assert_closed(report)
    assert CLOSURE_TOL_PCT == 0.01
    # 0.005 percent of the parent is float noise and passes at the default.
    assert_closed(hand_report(100.0, (hand_bucket("step", 100.005, 100.0),)))
    wide = hand_report(100.0, (hand_bucket("step", 100.02, 100.0),))
    with pytest.raises(ValueError, match=r"bucket '<total>' is 100\.000 us"):
        assert_closed(wide)
    assert_closed(wide, Percent(0.05))
    # A relative tolerance has no meaning against a zero parent. The check is made
    # absolutely there rather than letting pct_of raise about its denominator.
    zero_parent = hand_report(
        0.0, (hand_bucket("step", 0.0, 1.0), hand_bucket("step.a", 10.0, 1.0))
    )
    with pytest.raises(ValueError, match=r"bucket 'step' is 0\.000 us"):
        assert_closed(zero_parent)
    assert_closed(
        hand_report(
            0.0, (hand_bucket("step", 0.0, 1.0), hand_bucket("step.a", 0.0, 1.0))
        )
    )


def test_assert_nonzero_names_missing_and_zero_buckets() -> None:
    assert_nonzero(
        budget(timed_of(100.0, {"step.a": 30.0, "step.b": 20.0})),
        ("step", "step.a", "step.b", "unattributed"),
    )
    report = budget(timed_of(100.0, {"a": 5.0, "b": 0.0}))
    with pytest.raises(ValueError) as caught:
        assert_nonzero(report, ("a", "b", "step.absent"))
    message = str(caught.value)
    assert "missing ['step.absent']" in message
    assert "zero ['b']" in message


def test_get_rejects_an_absent_label() -> None:
    report = budget(timed_of(100.0, {"step.a": 30.0}))
    assert report.labels() == ("step", "step.a", "unattributed")
    assert tuple(b.label for b in report.children("step")) == ("step.a",)
    assert report.children("step.a") == ()
    with pytest.raises(KeyError, match="no bucket 'nope'"):
        report.get("nope")


def test_compare_orders_by_largest_regression() -> None:
    before, after = two_reports()
    deltas = compare(before, after)
    assert tuple(d.label for d in deltas) == (
        "unattributed",
        "step.b",
        "step",
        "step.a",
    )
    by_label = {d.label: d for d in deltas}
    faster = by_label["step.a"]
    assert faster.before_duration_us == 50.0
    assert faster.after_duration_us == 40.0
    assert faster.delta_pct == pytest.approx(-20.0)
    assert faster.speedup_ratio == pytest.approx(1.25)
    assert faster.resolved
    # The floor is the two runs' own half-widths summed, each half of a 4 percent
    # range.
    assert faster.floor_pct == pytest.approx(4.0)
    assert by_label["unattributed"].delta_pct == pytest.approx(40.0)
    # A derived bucket has no dispersion of its own, so it falls back to the sum of
    # the two totals' half-widths: half of the earlier 1 percent and half of the
    # later 8.
    derived = by_label["step"]
    assert derived.floor_pct == pytest.approx(4.5)
    assert derived.delta_pct == pytest.approx(-100.0 / 15.0)
    assert derived.resolved


def test_compare_sums_the_two_half_widths_rather_than_taking_the_larger() -> None:
    # Under the larger of the two half-widths a 3 percent move would read as a
    # result. Two medians are one measurement until they lie further apart than the
    # sum of their intervals' half-widths.
    before = budget(timed_of(100.0, {"step.a": 50.0}, label="before"))
    after = budget(timed_of(100.0, {"step.a": 51.5}, label="after"))
    assert before.get("step.a").resolution_pct == 2.0
    assert after.get("step.a").resolution_pct == pytest.approx(2.0)
    leaf = {d.label: d for d in compare(before, after)}["step.a"]
    assert leaf.delta_pct == pytest.approx(3.0)
    assert leaf.floor_pct == pytest.approx(4.0)
    assert not leaf.resolved


def test_compare_judges_a_bucket_against_its_own_dispersion() -> None:
    # The whole-call floor is 1 percent and the bucket's own is 32, so a 10 percent
    # bucket delta resolves against the total and not against itself. The bucket's
    # range stays at 4 percent, so the floor can only have come from the resolution.
    before = budget(timed_of(100.0, {"step.a": 50.0}, label="before", total_pct=1.0))
    after = BudgetReport(
        label="after",
        clocks="unlocked",
        total=spread_of(100.0, pct=1.0),
        buckets=tuple(
            BucketTiming(
                label=b.label,
                parent=b.parent,
                derived=b.derived,
                median_duration_us=Microseconds(55.0 if b.label == "step.a" else 45.0),
                spread_pct=b.spread_pct,
                resolution_pct=Percent(30.0) if not b.derived else Percent(0.0),
                coverage_pct=b.coverage_pct,
                sample_count=b.sample_count,
                share_of_parent_pct=b.share_of_parent_pct,
                share_of_total_pct=b.share_of_total_pct,
            )
            for b in before.buckets
        ),
    )
    assert after.get("step.a").spread_pct == 4.0
    by_label = {d.label: d for d in compare(before, after)}
    leaf = by_label["step.a"]
    assert leaf.delta_pct == pytest.approx(10.0)
    assert leaf.floor_pct == 32.0
    assert not leaf.resolved
    # The same 10 percent move on a derived sibling clears the whole-call floor.
    assert by_label["step"].floor_pct == 1.0
    assert by_label["step"].resolved


def test_compare_refuses_a_bucket_whose_own_interval_misses_coverage() -> None:
    # A region entered fewer times than the loop iterated carries fewer samples
    # than the total, so the whole-call gate does not cover it and its own coverage
    # must. A 50 percent move against a 4 percent floor still resolves nothing.
    before = budget(timed_of(100.0, {"step.a": 50.0}, label="before"))
    medians = {"step": 25.0, "step.a": 25.0, "unattributed": 75.0}
    after = BudgetReport(
        label="after",
        clocks="unlocked",
        total=spread_of(100.0),
        buckets=tuple(
            BucketTiming(
                label=b.label,
                parent=b.parent,
                derived=b.derived,
                median_duration_us=Microseconds(medians[b.label]),
                spread_pct=b.spread_pct,
                resolution_pct=b.resolution_pct,
                coverage_pct=(Percent(93.75) if not b.derived else b.coverage_pct),
                sample_count=Count(5) if not b.derived else b.sample_count,
                share_of_parent_pct=b.share_of_parent_pct,
                share_of_total_pct=b.share_of_total_pct,
            )
            for b in before.buckets
        ),
    )
    assert before.total.coverage_pct >= CONFIDENCE_PCT
    assert after.get("step.a").coverage_pct < CONFIDENCE_PCT
    by_label = {d.label: d for d in compare(before, after)}
    leaf = by_label["step.a"]
    assert leaf.delta_pct == pytest.approx(-50.0)
    assert leaf.floor_pct == pytest.approx(4.0)
    assert not leaf.resolved
    # Its derived parent is judged on the whole call, which is fully covered.
    assert by_label["step"].resolved


def test_compare_keeps_a_measured_zero_floor_rather_than_falling_back() -> None:
    # Identical samples put the floor at zero, and a measured bucket is judged on
    # that zero. Falling back to the whole-call floor here would discard the
    # sharpest measurement in the report.
    before = budget(
        timed_of(100.0, {"step.a": 50.0}, label="before", total_pct=8.0, region_pct=0.0)
    )
    after = budget(
        timed_of(100.0, {"step.a": 50.5}, label="after", total_pct=8.0, region_pct=0.0)
    )
    by_label = {d.label: d for d in compare(before, after)}
    leaf = by_label["step.a"]
    assert not before.get("step.a").derived
    assert before.get("step.a").resolution_pct == 0.0
    assert leaf.floor_pct == 0.0
    assert leaf.delta_pct == pytest.approx(1.0)
    assert leaf.resolved
    # The derived remainder moved by the same amount and does not resolve.
    rest = by_label["unattributed"]
    assert rest.floor_pct == 8.0
    assert rest.delta_pct == pytest.approx(-1.0)
    assert not rest.resolved


def test_compare_resolves_nothing_below_the_sample_count_minimum() -> None:
    # Under MIN_RESOLVING_SAMPLES no interval over the samples reaches the nominal
    # coverage, so the floor means nothing and licenses no claim at any size.
    thin = MIN_RESOLVING_SAMPLES - 1
    before = budget(timed_of(100.0, {"step.a": 50.0}, label="before", count=thin))
    after = budget(timed_of(100.0, {"step.a": 5.0}, label="after", count=thin))
    assert min(before.total.sample_count, after.total.sample_count) == thin
    assert before.total.coverage_pct < CONFIDENCE_PCT
    deltas = compare(before, after)
    assert max(abs(d.delta_pct) for d in deltas) == pytest.approx(90.0)
    assert all(abs(d.delta_pct) > d.floor_pct for d in deltas)
    assert not any(d.resolved for d in deltas)
    assert rank(deltas, 5) == ()


def test_compare_rejects_reports_with_no_shared_bucket() -> None:
    left = hand_report(100.0, (hand_bucket("a", 100.0, 100.0),), label="left")
    right = hand_report(100.0, (hand_bucket("b", 100.0, 100.0),), label="right")
    with pytest.raises(ValueError, match="share no bucket"):
        compare(left, right)


def test_rank_keeps_resolved_regressions_worst_first() -> None:
    deltas = compare(*two_reports())
    assert tuple(d.label for d in rank(deltas, 5)) == ("unattributed", "step.b")
    assert tuple(d.label for d in rank(deltas, 1)) == ("unattributed",)
    assert rank(deltas, 0) == ()
    # rank imposes the order. Truncating an unsorted sequence would drop the worst
    # regression and label the survivors "worst first".
    shuffled = tuple(reversed(deltas))
    assert tuple(d.label for d in rank(shuffled, 1)) == ("unattributed",)
