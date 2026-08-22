"""Field-name schema, record reduction, conversions, and spread.

Every raise in :mod:`slinoss.perf.units` is triggered here. The schema validates
in ``__init_subclass__``, so a rejection is a class-definition failure and each
rejection defines its own frozen dataclass inside a ``pytest.raises`` block.
Rejections are grouped by the rule they violate, since one rule is one failure
mode however many field types reach it.

``MIN_RESOLVING_SAMPLES`` is derived from ``CONFIDENCE_PCT`` through the
order-statistic coverage, so the tests assert that relation through the private
helpers that define it and never restate the constant.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Annotated, ClassVar

import pytest

from slinoss.perf import units as units_mod
from slinoss.perf.units import (
    CONFIDENCE_PCT,
    INVARIANT,
    MEDIAN,
    MIN_RESOLVING_SAMPLES,
    MODELLED,
    SUM,
    Bytes,
    Count,
    Microseconds,
    Milliseconds,
    Nanoseconds,
    Percent,
    PerfRecord,
    Spread,
    aggregate,
    gbs_from_bytes_us,
    median_ci,
    mib_from_bytes,
    ms_from_us,
    pct_of,
    ratio_of,
    tflops_from_flop_us,
    tps_from_tokens_us,
    us_from_ms,
    us_from_ns,
)


def us(*values: float) -> list[Microseconds]:
    """Microsecond samples from raw floats."""
    return [Microseconds(v) for v in values]


def stalled(count: int) -> list[Microseconds]:
    """Samples cycling over 100, 101, 102 us with one 1000 us stall.

    Args:
        count: Number of samples, at least four.

    Returns:
        The samples in measurement order. The stall sits at index 3, so every
        count used here holds exactly one outlier and the counts are prefixes of
        one series.
    """
    values = [100.0 + i % 3 for i in range(count)]
    values[3] = 1000.0
    return us(*values)


@dataclass(frozen=True)
class Sample(PerfRecord):
    """One field per reduction, one unmarked field, one nested record.

    Attributes:
        label: Unmarked, so INVARIANT by default.
        duration_us: Reduced by median.
        moved_bytes: Reduced by addition.
        sample_count: Reduced by addition.
        occupancy_pct: Marked INVARIANT.
        spread: Nested record; aggregation recurses into it.
    """

    label: str
    duration_us: Annotated[Microseconds, MEDIAN]
    moved_bytes: Annotated[Bytes, SUM]
    sample_count: Annotated[Count, SUM]
    occupancy_pct: Annotated[Percent, INVARIANT]
    spread: Spread


def sample(
    *,
    label: str = "scan",
    duration_us: float = 10.0,
    moved_bytes: int = 4,
    occupancy_pct: float = 50.0,
) -> Sample:
    """A :class:`Sample` whose nested spread comes from its own duration."""
    return Sample(
        label=label,
        duration_us=Microseconds(duration_us),
        moved_bytes=Bytes(moved_bytes),
        sample_count=Count(1),
        occupancy_pct=Percent(occupancy_pct),
        spread=Spread.of(us(duration_us)),
    )


@dataclass(frozen=True)
class Modelled(PerfRecord):
    """A modelled byte count beside a measured duration.

    Attributes:
        est_moved_bytes: Analytic byte count.
        duration_us: Measured duration.
    """

    est_moved_bytes: Annotated[Bytes, MODELLED, SUM]
    duration_us: Annotated[Microseconds, MEDIAN]


@dataclass(frozen=True)
class Sampled(PerfRecord):
    """A sample list beside a tuple of a non-unit type.

    Attributes:
        samples_duration_us: Tuple of a unit type, so the suffix rule applies and
            SUM concatenates.
        labels: Tuple of str, so no suffix obligation and no reduction marker.
    """

    samples_duration_us: Annotated[tuple[Microseconds, ...], SUM]
    labels: tuple[str, ...]


def sampled(*values: float, label: str = "scan") -> Sampled:
    """A :class:`Sampled` holding ``values`` in the order given."""
    return Sampled(samples_duration_us=tuple(us(*values)), labels=(label,))


@dataclass(frozen=True)
class Underscored(PerfRecord):
    """``_scratch_bytes`` claims a unit its type lacks and is skipped anyway.

    Attributes:
        duration_us: Measured duration.
        _scratch_bytes: Private, so outside the schema.
    """

    duration_us: Annotated[Microseconds, MEDIAN]
    _scratch_bytes: int = 0


# ---------------------------------------------------------------------------
# Exports and markers
# ---------------------------------------------------------------------------


def test_all_lists_every_public_name_and_nothing_dangling() -> None:
    # A unit type absent from __all__ is invisible to a star import while every
    # other unit is visible, which is worse than exporting none of them.
    missing = sorted(
        name
        for name, obj in vars(units_mod).items()
        if not name.startswith("_")
        and getattr(obj, "__module__", None) == units_mod.__name__
        and (isinstance(obj, type) or callable(obj) or hasattr(obj, "__supertype__"))
        and name not in units_mod.__all__
    )
    assert missing == []
    # Neither constant carries __module__, so the sweep above cannot see them.
    assert "CONFIDENCE_PCT" in units_mod.__all__
    assert "MIN_RESOLVING_SAMPLES" in units_mod.__all__
    dangling = [name for name in units_mod.__all__ if not hasattr(units_mod, name)]
    assert dangling == []
    # Marker identity is the whole of its meaning, so the repr names it.
    assert repr(MODELLED) == "<modelled>"
    assert repr(SUM) == "<sum>"
    assert repr(MEDIAN) == "<median>"
    assert repr(INVARIANT) == "<invariant>"


# ---------------------------------------------------------------------------
# Schema rejections
# ---------------------------------------------------------------------------


def test_rejects_a_name_and_a_unit_that_disagree() -> None:
    with pytest.raises(TypeError, match="'duration_ms' is Microseconds"):

        @dataclass(frozen=True)
        class WrongSuffix(PerfRecord):
            duration_ms: Annotated[Microseconds, MEDIAN]

    with pytest.raises(TypeError, match="'foo_bytes' claims unit '_bytes'"):

        @dataclass(frozen=True)
        class ClaimedByAnInt(PerfRecord):
            foo_bytes: int

    # A tuple of one unit type is a sample list and carries the unit exactly as a
    # scalar does.
    with pytest.raises(TypeError, match="'samples' is Microseconds"):

        @dataclass(frozen=True)
        class UnnamedSampleList(PerfRecord):
            samples: Annotated[tuple[Microseconds, ...], SUM]

    # A fixed-length tuple is a structure, not a sample list, so it does not
    # carry the unit of its first element and may not claim one by name.
    with pytest.raises(TypeError, match="'bounds_us' claims unit '_us'"):

        @dataclass(frozen=True)
        class ClaimedByAStructure(PerfRecord):
            bounds_us: Annotated[tuple[Microseconds, Percent], SUM]


def test_rejects_a_model_that_could_read_as_a_measurement() -> None:
    with pytest.raises(TypeError, match="must start with 'est_'"):

        @dataclass(frozen=True)
        class UnprefixedModel(PerfRecord):
            duration_us: Annotated[Microseconds, MODELLED, MEDIAN]

    with pytest.raises(TypeError, match="'est_duration_us' is named est_"):

        @dataclass(frozen=True)
        class UnmarkedEstimate(PerfRecord):
            est_duration_us: Annotated[Microseconds, MEDIAN]

    with pytest.raises(TypeError, match="a measured and a modelled Bytes"):

        @dataclass(frozen=True)
        class AdjacentColumns(PerfRecord):
            moved_bytes: Annotated[Bytes, SUM]
            est_moved_bytes: Annotated[Bytes, MODELLED, SUM]


def test_rejects_a_reduction_that_cannot_be_taken() -> None:
    with pytest.raises(TypeError, match="has a unit and no reduction"):

        @dataclass(frozen=True)
        class NoReduction(PerfRecord):
            duration_us: Microseconds

    with pytest.raises(TypeError, match="declares 2 reductions"):

        @dataclass(frozen=True)
        class TwoReductions(PerfRecord):
            duration_us: Annotated[Microseconds, MEDIAN, SUM]

    with pytest.raises(TypeError, match="the median of integers is not an integer"):

        @dataclass(frozen=True)
        class MedianOnACount(PerfRecord):
            launch_count: Annotated[Count, MEDIAN]

    with pytest.raises(TypeError, match="'samples_duration_us' is a tuple and MEDIAN"):

        @dataclass(frozen=True)
        class MedianOnASampleList(PerfRecord):
            samples_duration_us: Annotated[tuple[Microseconds, ...], MEDIAN]

    # The ban is on the tuple, not on its element type: the median of a list of
    # labels is a label nobody measured, and statistics.median would raise on it
    # at report time rather than here.
    with pytest.raises(TypeError, match="'labels' is a tuple and MEDIAN"):

        @dataclass(frozen=True)
        class MedianOnLabels(PerfRecord):
            labels: Annotated[tuple[str, ...], MEDIAN]


def test_requires_the_denominator_beside_a_count_and_a_floor() -> None:
    with pytest.raises(TypeError, match="conflict count and no wavefront_count"):

        @dataclass(frozen=True)
        class BareConflicts(PerfRecord):
            bank_conflict_count: Annotated[Count, SUM]

    # A half-width is only a floor at a stated coverage, so the pairing is
    # structural rather than a convention a new record can forget.
    with pytest.raises(TypeError, match="resolution floor and no coverage_pct"):

        @dataclass(frozen=True)
        class BareFloor(PerfRecord):
            resolution_pct: Annotated[Percent, MEDIAN]

    @dataclass(frozen=True)
    class Paired(PerfRecord):
        bank_conflict_count: Annotated[Count, SUM]
        wavefront_count: Annotated[Count, SUM]
        resolution_pct: Annotated[Percent, MEDIAN]
        coverage_pct: Annotated[Percent, MEDIAN]

    rec = Paired(
        bank_conflict_count=Count(2),
        wavefront_count=Count(8),
        resolution_pct=Percent(1.0),
        coverage_pct=Percent(96.0),
    )
    assert ratio_of(rec.bank_conflict_count, rec.wavefront_count) == 0.25
    assert rec.coverage_pct >= CONFIDENCE_PCT


# ---------------------------------------------------------------------------
# Schema acceptances
# ---------------------------------------------------------------------------


def test_a_sample_list_and_a_fixed_length_tuple_reduce_differently() -> None:
    # Definition-time acceptance is half the assertion. The reductions show the
    # sample list was validated and the tuple of str was left alone.
    assert Sampled._reductions == {"samples_duration_us": SUM, "labels": INVARIANT}

    @dataclass(frozen=True)
    class Bounds(PerfRecord):
        bounds: tuple[Microseconds, Percent]

    pair = (Microseconds(1.0), Percent(2.0))
    other = (Microseconds(3.0), Percent(2.0))
    # No suffix obligation, no reduction marker, and no elementwise reduction:
    # the pair reduces as a single INVARIANT value.
    assert aggregate([Bounds(bounds=pair), Bounds(bounds=pair)]).bounds == pair
    with pytest.raises(ValueError, match="'bounds' is INVARIANT"):
        aggregate([Bounds(bounds=pair), Bounds(bounds=other)])


def test_classvar_skips_validation_in_both_annotation_forms() -> None:
    # page_bytes claims a unit its type lacks; a validated field would raise.
    # dataclasses treats an unsubscripted ClassVar as a class variable, so the
    # schema must too, and get_origin is None for that form.
    @dataclass(frozen=True)
    class Subscripted(PerfRecord):
        page_bytes: ClassVar[int] = 4096
        duration_us: Annotated[Microseconds, MEDIAN]

    @dataclass(frozen=True)
    class Bare(PerfRecord):
        page_bytes: ClassVar = 4096
        duration_us: Annotated[Microseconds, MEDIAN]

    assert (Subscripted.page_bytes, Bare.page_bytes) == (4096, 4096)
    assert Subscripted(duration_us=Microseconds(1.0)).duration_us == 1.0
    assert [f.name for f in fields(Bare)] == ["duration_us"]


def test_underscored_field_skips_validation_and_reduces_as_invariant() -> None:
    low = Underscored(duration_us=Microseconds(1.0), _scratch_bytes=7)
    high = Underscored(duration_us=Microseconds(3.0), _scratch_bytes=7)
    out = aggregate([low, high])
    assert out.duration_us == 2.0
    assert out._scratch_bytes == 7
    other = Underscored(duration_us=Microseconds(3.0), _scratch_bytes=8)
    with pytest.raises(ValueError, match="'_scratch_bytes' is INVARIANT"):
        aggregate([low, other])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_reduces_field_by_field() -> None:
    out = aggregate(
        [
            sample(duration_us=10.0, moved_bytes=4),
            sample(duration_us=30.0, moved_bytes=5),
            sample(duration_us=20.0, moved_bytes=6),
        ]
    )
    assert out.duration_us == 20.0
    assert out.moved_bytes == 15
    assert out.sample_count == 3
    assert out.occupancy_pct == 50.0
    assert out.label == "scan"
    # A nested record recurses, and a Spread recurses through combine: the pooled
    # samples are the three parts' samples, and every statistic is recomputed from
    # them rather than reduced field by field.
    assert out.spread == Spread.of(us(10.0, 30.0, 20.0))
    assert out.spread.median_duration_us == 20.0
    assert out.spread.min_duration_us == 10.0
    assert out.spread.max_duration_us == 30.0
    assert out.spread.spread_pct == 100.0
    assert out.spread.resolution_pct == 50.0
    assert out.spread.coverage_pct == 75.0
    assert out.spread.sample_count == 3
    assert out.spread.samples_duration_us == (10.0, 30.0, 20.0)


def test_aggregate_combines_a_spread_from_the_samples_behind_its_parts() -> None:
    # Field by field would give min 2.5, max 4.5 and a 70 percent range over six
    # samples spanning 1 to 6, which is a record contradicting its own sample
    # list. It would also sum the counts to six and unlock resolves on two floors
    # that never held.
    parts = [Spread.of(us(1.0, 2.0, 3.0)), Spread.of(us(4.0, 5.0, 6.0))]
    out = aggregate(parts)
    assert out == Spread.of(us(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    assert out.min_duration_us == 1.0
    assert out.max_duration_us == 6.0
    assert out.sample_count == 6
    assert out.coverage_pct == units_mod._sign_coverage_pct(6, 1)
    assert out.resolves(Percent(1000.0))
    assert not parts[0].resolves(Percent(1000.0))


def test_spread_combine_needs_the_samples_behind_every_part() -> None:
    # A hand-built Spread carries no samples, so there is nothing to pool and no
    # honest way to reduce it.
    bare = Spread(
        median_duration_us=Microseconds(10.0),
        min_duration_us=Microseconds(9.0),
        max_duration_us=Microseconds(11.0),
        spread_pct=Percent(20.0),
        resolution_pct=Percent(10.0),
        coverage_pct=Percent(96.0),
        sample_count=Count(8),
    )
    with pytest.raises(ValueError, match="needs the samples behind every part"):
        aggregate([bare, Spread.of(us(10.0))])
    empty: list[Spread] = []
    with pytest.raises(ValueError, match="needs at least one sample"):
        Spread.combine(empty)


def test_aggregate_concatenates_sample_lists_and_holds_tuple_invariants() -> None:
    out = aggregate([sampled(10.0, 30.0), sampled(20.0)])
    assert out.samples_duration_us == (10.0, 30.0, 20.0)
    assert out.labels == ("scan",)
    with pytest.raises(ValueError, match="field 'labels' is INVARIANT"):
        aggregate([sampled(10.0), sampled(10.0, label="mixer")])


def test_aggregate_keeps_the_est_prefix() -> None:
    out = aggregate(
        [
            Modelled(est_moved_bytes=Bytes(2), duration_us=Microseconds(4.0)),
            Modelled(est_moved_bytes=Bytes(3), duration_us=Microseconds(6.0)),
        ]
    )
    assert out.est_moved_bytes == 5
    assert out.duration_us == 5.0
    assert [f.name for f in fields(out)] == ["est_moved_bytes", "duration_us"]


def test_aggregate_rejects_invariant_disagreement() -> None:
    with pytest.raises(ValueError, match="'occupancy_pct' is INVARIANT"):
        aggregate([sample(occupancy_pct=50.0), sample(occupancy_pct=60.0)])


def test_aggregate_rejects_a_malformed_sample_sequence() -> None:
    class Plain(PerfRecord):
        pass

    empty: list[Spread] = []
    with pytest.raises(ValueError, match="needs at least one sample"):
        aggregate(empty)
    mixed: list[PerfRecord] = [sample(), Spread.of(us(1.0))]
    with pytest.raises(ValueError, match="needs one record type"):
        aggregate(mixed)
    with pytest.raises(ValueError, match="Plain is not a dataclass"):
        aggregate([Plain()])


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def test_conversions_carry_the_base_they_declare() -> None:
    assert us_from_ns(Nanoseconds(1500.0)) == 1.5
    assert ms_from_us(Microseconds(2500.0)) == 2.5
    assert us_from_ms(Milliseconds(2.5)) == 2500.0
    assert tflops_from_flop_us(Count(2_000_000), Microseconds(1.0)) == 2.0
    assert tps_from_tokens_us(Count(1000), Microseconds(1000.0)) == pytest.approx(1e6)
    assert pct_of(25.0, 200.0) == 12.5
    assert ratio_of(3.0, 4.0) == 0.75
    # Mebibytes are 1024-based, so 10**6 bytes is under one.
    assert mib_from_bytes(Bytes(2**20)) == 1.0
    assert mib_from_bytes(Bytes(3 * 2**19)) == 1.5
    assert mib_from_bytes(Bytes(1_000_000)) < 1.0
    # 1e9 bytes in one second is 1 GB/s exactly. 2**30 bytes in that second is
    # more than 1, which is what distinguishes the 1000 base from 1024.
    assert gbs_from_bytes_us(Bytes(10**9), Microseconds(1e6)) == 1.0
    assert gbs_from_bytes_us(Bytes(2**30), Microseconds(1e6)) == pytest.approx(
        1.073741824
    )


def test_conversions_reject_a_denominator_that_is_not_positive() -> None:
    for duration in (0.0, -1.0):
        with pytest.raises(ValueError, match="duration_us must be positive"):
            gbs_from_bytes_us(Bytes(1), Microseconds(duration))
        with pytest.raises(ValueError, match="duration_us must be positive"):
            tflops_from_flop_us(Count(1), Microseconds(duration))
        with pytest.raises(ValueError, match="duration_us must be positive"):
            tps_from_tokens_us(Count(1), Microseconds(duration))
    with pytest.raises(ValueError, match="percent of zero is undefined"):
        pct_of(1.0, 0.0)
    with pytest.raises(ValueError, match="zero denominator is undefined"):
        ratio_of(1.0, 0.0)


# ---------------------------------------------------------------------------
# Resolution floor
# ---------------------------------------------------------------------------


def test_sign_coverage_is_the_binomial_tail() -> None:
    # Rank 1 spans every sample, so its coverage is 1 - 2 * 2**-count. One
    # sample bounds nothing, two bound the median half the time.
    assert units_mod._sign_coverage_pct(1, 1) == 0.0
    assert units_mod._sign_coverage_pct(2, 1) == 50.0
    assert units_mod._sign_coverage_pct(5, 1) == 93.75
    assert units_mod._sign_coverage_pct(6, 1) == 96.875
    # Rank 10 at thirty samples drops the tail from both ends.
    assert units_mod._sign_coverage_pct(30, 10) == pytest.approx(95.7226, abs=1e-4)


def test_sign_coverage_holds_past_the_float_range() -> None:
    # 2**count passes the largest float at 1024, so a float operand in the ratio
    # raises OverflowError there and the interval cannot be formed at all. The
    # coverage is a ratio of integers and has a value at any count.
    assert units_mod._sign_coverage_pct(1024, 1) == 100.0
    assert units_mod._sign_coverage_pct(4096, 1) == 100.0
    low, high, coverage = median_ci(us(*(float(i) for i in range(1024))))
    assert coverage >= CONFIDENCE_PCT
    assert (low, high) == (480.0, 543.0)


def test_median_ci_rank_is_the_tightest_interval_holding_the_confidence() -> None:
    # Coverage falls as the rank rises, so the tightest covering rank is the last
    # one before the first miss, at every count that has one.
    for count in range(MIN_RESOLVING_SAMPLES, 65):
        rank = units_mod._median_ci_rank(count)
        assert units_mod._sign_coverage_pct(count, rank) >= CONFIDENCE_PCT
        assert units_mod._sign_coverage_pct(count, rank + 1) < CONFIDENCE_PCT
    # Below the minimum no rank covers, so the search falls back to 1, the widest
    # interval the samples admit.
    for count in range(1, MIN_RESOLVING_SAMPLES):
        assert units_mod._median_ci_rank(count) == 1
    # The binomial tail steps, so one rank covers several counts.
    assert [units_mod._median_ci_rank(n) for n in (6, 8, 9, 11, 20, 30, 31, 40)] == [
        1,
        1,
        2,
        2,
        6,
        10,
        10,
        14,
    ]


def test_min_resolving_samples_is_derived_from_the_confidence() -> None:
    # The definition, not the literal: the widest interval reaches the nominal
    # coverage at this count and misses it at one sample fewer.
    assert units_mod._sign_coverage_pct(MIN_RESOLVING_SAMPLES, 1) >= CONFIDENCE_PCT
    assert units_mod._sign_coverage_pct(MIN_RESOLVING_SAMPLES - 1, 1) < CONFIDENCE_PCT


def test_median_ci_returns_two_of_the_samples() -> None:
    # Rank 2 of ten, so the bounds are the second and the ninth in order. They are
    # samples, not derived quantities, so they carry the unit and need no scale.
    # The input is unsorted, so this also fixes that order does not matter.
    low, high, coverage = median_ci(
        us(104.0, 92.0, 96.0, 100.0, 108.0, 100.0, 99.0, 101.0, 100.0, 100.0)
    )
    assert (low, high) == (96.0, 104.0)
    assert coverage == units_mod._sign_coverage_pct(10, 2)
    # Five samples admit no interval at the nominal coverage, so the widest one
    # they have is returned with the coverage it really has.
    low, high, coverage = median_ci(us(30.0, 10.0, 40.0, 100.0, 20.0))
    assert (low, high) == (10.0, 100.0)
    assert coverage == 93.75
    assert coverage < CONFIDENCE_PCT


def test_median_ci_is_the_one_definition_behind_a_spread() -> None:
    samples = stalled(20)
    low, high, coverage = median_ci(samples)
    s = Spread.of(samples)
    assert s.resolution_pct == pct_of(0.5 * (high - low), s.median_duration_us)
    assert s.coverage_pct == coverage


def test_empty_sample_lists_are_rejected() -> None:
    empty: list[Microseconds] = []
    with pytest.raises(ValueError, match="median_ci needs at least one sample"):
        median_ci(empty)
    with pytest.raises(ValueError, match="needs at least one sample"):
        Spread.of(empty)


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------


def test_spread_of_known_samples() -> None:
    # Median 30, max - min = 90, so the spread is 300 percent of the median.
    # Five samples take rank 1, so the interval is the whole range and the floor
    # is half of it.
    s = Spread.of(us(30.0, 10.0, 40.0, 100.0, 20.0))
    assert s.median_duration_us == 30.0
    assert s.min_duration_us == 10.0
    assert s.max_duration_us == 100.0
    assert s.spread_pct == 300.0
    assert s.resolution_pct == 150.0
    assert s.sample_count == 5
    # Five samples cannot reach the nominal coverage. The floor still prints,
    # labelled with the coverage it has, and it licenses nothing at any size.
    assert s.coverage_pct == 93.75
    assert s.coverage_pct < CONFIDENCE_PCT
    assert not s.resolves(Percent(1000.0))


def test_spread_resolution_at_twenty_samples_decides_what_resolves() -> None:
    # Twenty samples take rank 6, so the interval is [x_(6), x_(15)] = [98, 102]
    # and the half-width is 2 over a median of 100. Both tails fall outside it,
    # which is the whole difference from the range. Sorted here for legibility.
    s = Spread.of(us(*([1.0] + [98.0] * 5 + [100.0] * 8 + [102.0] * 5 + [1000.0])))
    assert s.median_duration_us == 100.0
    assert s.min_duration_us == 1.0
    assert s.max_duration_us == 1000.0
    assert s.spread_pct == 999.0
    assert s.resolution_pct == 2.0
    assert s.sample_count == 20
    # 1 - 2 * P(Bin(20, 1/2) < 6) = 1 - 43,400 / 1,048,576, the first rank at this
    # count whose interval still reaches nominal coverage.
    assert s.coverage_pct == pytest.approx(95.8610535, abs=1e-6)
    # The floor decides, not the range: a 3 percent claim resolves against a 999
    # percent range because the outliers lie outside the interval.
    assert s.resolves(Percent(3.0))
    assert not s.resolves(Percent(2.0))
    assert not s.resolves(Percent(1.0))
    # Magnitude decides, so a regression resolves on the same threshold.
    assert s.resolves(Percent(-3.0))
    assert not s.resolves(Percent(-1.0))


def test_spread_range_grows_while_the_floor_shrinks() -> None:
    # One stalled iteration is why both statistics exist. The range only grows
    # with the sample count, since it tracks the tail. The floor falls, since
    # rank 1 at six samples spans the stall and rank 6 at twenty does not. So a
    # 2 percent claim is unresolvable on six samples of this series and
    # resolvable on thirty of it.
    spreads = [Spread.of(stalled(count)) for count in (6, 10, 20, 30)]
    ranges = [s.spread_pct for s in spreads]
    floors = [s.resolution_pct for s in spreads]
    assert ranges == sorted(ranges)
    assert ranges[-1] > ranges[0]
    assert floors == sorted(floors, reverse=True)
    assert floors[0] > 400.0
    assert floors[-1] < 1.0
    assert spreads[-1].max_duration_us == 1000.0
    assert not spreads[0].resolves(Percent(2.0))
    assert spreads[-1].resolves(Percent(2.0))


def test_spread_keeps_the_samples_in_measurement_order() -> None:
    # Spread.of sorts a copy. The record keeps the order the timer produced, so
    # a reader can see drift that the median hides.
    raw = us(30.0, 10.0, 40.0, 100.0, 20.0)
    s = Spread.of(raw)
    assert s.samples_duration_us == (30.0, 10.0, 40.0, 100.0, 20.0)
    assert len(s.samples_duration_us) == s.sample_count
    assert raw == us(30.0, 10.0, 40.0, 100.0, 20.0)


def test_spread_resolution_is_zero_when_the_interval_collapses() -> None:
    # Rank 6 at twenty samples reads x_(6) and x_(15), both 100 here, so the
    # floor is zero while the range still exposes both outliers.
    s = Spread.of(us(*([1.0] + [100.0] * 18 + [500.0])))
    assert s.median_duration_us == 100.0
    assert s.spread_pct == 499.0
    assert s.resolution_pct == 0.0
    assert s.resolves(Percent(0.5))
    # A collapsed floor at nominal coverage admits any delta above it, and a zero
    # delta is not above zero.
    least = Spread.of(us(*[100.0] * MIN_RESOLVING_SAMPLES))
    assert least.resolution_pct == 0.0
    assert least.coverage_pct >= CONFIDENCE_PCT
    assert least.resolves(Percent(0.01))
    assert not least.resolves(Percent(0.0))


def test_spread_below_the_minimum_sample_count_resolves_nothing() -> None:
    # Identical samples put the floor at zero and the gate still refuses: no
    # interval over this few samples reaches CONFIDENCE_PCT, so the floor means
    # nothing and licenses nothing.
    for count in range(1, MIN_RESOLVING_SAMPLES):
        s = Spread.of(us(*[100.0] * count))
        assert s.resolution_pct == 0.0
        assert s.coverage_pct < CONFIDENCE_PCT
        assert not s.resolves(Percent(1000.0))
    # One sample bounds nothing, and a zero floor at zero coverage must not read
    # as a perfectly resolved measurement.
    one = Spread.of(us(7.0))
    assert one.median_duration_us == 7.0
    assert one.min_duration_us == 7.0
    assert one.max_duration_us == 7.0
    assert one.spread_pct == 0.0
    assert one.sample_count == 1
    assert one.samples_duration_us == (7.0,)
    assert one.coverage_pct == 0.0


def test_spread_of_a_zero_median() -> None:
    # A measurement that rounds to zero is still a measurement, and identical
    # samples span nothing, so both statistics are zero rather than undefined.
    s = Spread.of(us(0.0, 0.0))
    assert s.median_duration_us == 0.0
    assert s.spread_pct == 0.0
    assert s.resolution_pct == 0.0
    assert s.sample_count == 2
    # Samples that differ need a denominator, and a zero median is not one.
    with pytest.raises(ValueError, match="percent of zero is undefined"):
        Spread.of(us(0.0, 0.0, 1.0))


def test_resolves_gates_on_the_coverage_and_not_on_the_sample_count() -> None:
    # Spread.of always picks the tightest rank that still reaches nominal, so on its
    # own output the two gates agree and either would do. A record assembled
    # elsewhere can carry a nominal count over an interval that does not reach
    # nominal, and it is the interval that decides.
    thin = Spread(
        median_duration_us=Microseconds(100.0),
        min_duration_us=Microseconds(99.0),
        max_duration_us=Microseconds(101.0),
        spread_pct=Percent(2.0),
        resolution_pct=Percent(1.0),
        coverage_pct=Percent(80.0),
        sample_count=Count(30),
    )
    assert thin.sample_count > MIN_RESOLVING_SAMPLES
    assert not thin.resolves(Percent(50.0))
