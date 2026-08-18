"""Field-name schema, record reduction, conversions, and spread.

Every raise in :mod:`slinoss.perf.units` is triggered here. The schema validates
in ``__init_subclass__``, so a rejection is a class-definition failure and each
rejection test defines its own frozen dataclass inside the ``pytest.raises``
block.

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
# Markers
# ---------------------------------------------------------------------------


def test_all_lists_every_unit_type_record_and_conversion() -> None:
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


def test_all_exports_the_confidence_constants_and_nothing_dangling() -> None:
    # Neither constant carries __module__, so the sweep above cannot see them.
    assert "CONFIDENCE_PCT" in units_mod.__all__
    assert "MIN_RESOLVING_SAMPLES" in units_mod.__all__
    dangling = [name for name in units_mod.__all__ if not hasattr(units_mod, name)]
    assert dangling == []


def test_marker_repr_names_the_marker() -> None:
    assert repr(MODELLED) == "<modelled>"
    assert repr(SUM) == "<sum>"
    assert repr(MEDIAN) == "<median>"
    assert repr(INVARIANT) == "<invariant>"


# ---------------------------------------------------------------------------
# Schema rejections
# ---------------------------------------------------------------------------


def test_rejects_unit_field_without_the_matching_suffix() -> None:
    with pytest.raises(TypeError, match="'duration_ms' is Microseconds"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            duration_ms: Annotated[Microseconds, MEDIAN]


def test_rejects_suffix_claimed_by_a_non_unit_type() -> None:
    with pytest.raises(TypeError, match="'foo_bytes' claims unit '_bytes'"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            foo_bytes: int


def test_rejects_modelled_field_without_the_est_prefix() -> None:
    with pytest.raises(TypeError, match="must start with 'est_'"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            duration_us: Annotated[Microseconds, MODELLED, MEDIAN]


def test_rejects_est_prefix_without_the_modelled_marker() -> None:
    with pytest.raises(TypeError, match="'est_duration_us' is named est_"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            est_duration_us: Annotated[Microseconds, MEDIAN]


def test_rejects_unit_field_without_a_reduction() -> None:
    with pytest.raises(TypeError, match="has a unit and no reduction"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            duration_us: Microseconds


def test_rejects_two_reductions_on_one_field() -> None:
    with pytest.raises(TypeError, match="declares 2 reductions"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            duration_us: Annotated[Microseconds, MEDIAN, SUM]


def test_rejects_measured_and_modelled_of_one_unit() -> None:
    with pytest.raises(TypeError, match="a measured and a modelled Bytes"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            moved_bytes: Annotated[Bytes, SUM]
            est_moved_bytes: Annotated[Bytes, MODELLED, SUM]


def test_rejects_median_on_a_count() -> None:
    with pytest.raises(TypeError, match="the median of integers is not an integer"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            launch_count: Annotated[Count, MEDIAN]


def test_rejects_median_on_a_byte_count() -> None:
    with pytest.raises(TypeError, match="the median of integers is not an integer"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            moved_bytes: Annotated[Bytes, MEDIAN]


def test_rejects_sample_list_without_the_matching_suffix() -> None:
    # A tuple of a unit type carries the unit exactly as a scalar does.
    with pytest.raises(TypeError, match="'samples' is Microseconds"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            samples: Annotated[tuple[Microseconds, ...], SUM]


def test_rejects_sample_list_without_a_reduction() -> None:
    with pytest.raises(TypeError, match="has a unit and no reduction"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            samples_duration_us: tuple[Microseconds, ...]


def test_rejects_median_on_a_sample_list() -> None:
    with pytest.raises(TypeError, match="'samples_duration_us' is a tuple and MEDIAN"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            samples_duration_us: Annotated[tuple[Microseconds, ...], MEDIAN]


def test_rejects_median_on_a_tuple_of_a_non_unit_type() -> None:
    # The ban is on the tuple, not on its element type: the median of a list of
    # labels is a label nobody measured, and statistics.median would raise on it
    # at report time rather than here.
    with pytest.raises(TypeError, match="'labels' is a tuple and MEDIAN"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            labels: Annotated[tuple[str, ...], MEDIAN]


def test_rejects_suffix_claimed_by_a_fixed_length_tuple() -> None:
    # A fixed-length tuple is a structure, not a sample list, so it does not
    # carry the unit of its first element and may not claim one by name.
    with pytest.raises(TypeError, match="'bounds_us' claims unit '_us'"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            bounds_us: Annotated[tuple[Microseconds, Percent], SUM]


def test_rejects_conflict_count_without_a_wavefront_count() -> None:
    with pytest.raises(TypeError, match="conflict count and no wavefront_count"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            bank_conflict_count: Annotated[Count, SUM]


def test_rejects_a_resolution_floor_without_its_coverage() -> None:
    # A half-width is only a floor at a stated coverage, so the pairing is
    # structural rather than a convention a new record can forget.
    with pytest.raises(TypeError, match="resolution floor and no coverage_pct"):

        @dataclass(frozen=True)
        class Bad(PerfRecord):
            resolution_pct: Annotated[Percent, MEDIAN]


# ---------------------------------------------------------------------------
# Schema acceptances
# ---------------------------------------------------------------------------


def test_accepts_conflict_count_beside_its_wavefront_count() -> None:
    @dataclass(frozen=True)
    class Conflicts(PerfRecord):
        bank_conflict_count: Annotated[Count, SUM]
        wavefront_count: Annotated[Count, SUM]

    rec = Conflicts(bank_conflict_count=Count(2), wavefront_count=Count(8))
    assert ratio_of(rec.bank_conflict_count, rec.wavefront_count) == 0.25


def test_accepts_a_resolution_floor_beside_its_coverage() -> None:
    @dataclass(frozen=True)
    class Floor(PerfRecord):
        resolution_pct: Annotated[Percent, MEDIAN]
        coverage_pct: Annotated[Percent, MEDIAN]

    rec = Floor(resolution_pct=Percent(1.0), coverage_pct=Percent(96.0))
    assert rec.coverage_pct >= CONFIDENCE_PCT


def test_accepts_a_sample_list_named_for_its_unit() -> None:
    # Definition-time acceptance is the assertion. The reductions show the
    # sample list was validated and the tuple of str was left alone.
    assert Sampled._reductions == {"samples_duration_us": SUM, "labels": INVARIANT}


def test_accepts_a_fixed_length_tuple_as_one_structure() -> None:
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


def test_classvar_skips_validation() -> None:
    # page_bytes claims a unit its type lacks; a validated field would raise.
    @dataclass(frozen=True)
    class WithClassVar(PerfRecord):
        page_bytes: ClassVar[int] = 4096
        duration_us: Annotated[Microseconds, MEDIAN]

    assert WithClassVar.page_bytes == 4096
    assert WithClassVar(duration_us=Microseconds(1.0)).duration_us == 1.0


def test_bare_classvar_skips_validation() -> None:
    # dataclasses treats an unsubscripted ClassVar as a class variable, so the
    # schema must too. get_origin is None for this form.
    @dataclass(frozen=True)
    class WithBareClassVar(PerfRecord):
        page_bytes: ClassVar = 4096
        duration_us: Annotated[Microseconds, MEDIAN]

    assert WithBareClassVar.page_bytes == 4096
    assert [f.name for f in fields(WithBareClassVar)] == ["duration_us"]


def test_underscored_field_skips_validation_and_reduces_as_invariant() -> None:
    low = Underscored(duration_us=Microseconds(1.0), _scratch_bytes=7)
    high = Underscored(duration_us=Microseconds(3.0), _scratch_bytes=7)
    out = aggregate([low, high])
    assert out.duration_us == 2.0
    assert out._scratch_bytes == 7
    other = Underscored(duration_us=Microseconds(3.0), _scratch_bytes=8)
    with pytest.raises(ValueError, match="'_scratch_bytes' is INVARIANT"):
        aggregate([low, other])


def test_non_unit_field_without_a_marker_defaults_to_invariant() -> None:
    with pytest.raises(ValueError, match="field 'label' is INVARIANT"):
        aggregate([sample(label="scan"), sample(label="mixer")])


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


def test_combine_is_none_on_a_record_of_independent_fields() -> None:
    # Field by field is right unless the fields are functions of one sample list,
    # so the hook opts in and every other record ignores it.
    assert Sample.combine([sample(), sample()]) is None
    assert Sampled.combine([sampled(1.0)]) is None


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


def test_aggregate_rejects_a_spread_built_without_its_samples() -> None:
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


def test_spread_combine_rejects_no_runs() -> None:
    empty: list[Spread] = []
    with pytest.raises(ValueError, match="needs at least one sample"):
        Spread.combine(empty)


def test_aggregate_concatenates_sample_lists_under_sum() -> None:
    out = aggregate([sampled(10.0, 30.0), sampled(20.0)])
    assert out.samples_duration_us == (10.0, 30.0, 20.0)
    assert out.labels == ("scan",)


def test_aggregate_rejects_tuple_invariant_disagreement() -> None:
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


def test_aggregate_rejects_an_empty_sequence() -> None:
    empty: list[Spread] = []
    with pytest.raises(ValueError, match="needs at least one sample"):
        aggregate(empty)


def test_aggregate_rejects_mixed_record_types() -> None:
    mixed: list[PerfRecord] = [sample(), Spread.of(us(1.0))]
    with pytest.raises(ValueError, match="needs one record type"):
        aggregate(mixed)


def test_aggregate_rejects_a_non_dataclass_record() -> None:
    class Plain(PerfRecord):
        pass

    with pytest.raises(ValueError, match="Plain is not a dataclass"):
        aggregate([Plain()])


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def test_us_from_ns() -> None:
    assert us_from_ns(Nanoseconds(1500.0)) == 1.5


def test_ms_from_us() -> None:
    assert ms_from_us(Microseconds(2500.0)) == 2.5


def test_us_from_ms() -> None:
    assert us_from_ms(Milliseconds(2.5)) == 2500.0


def test_mib_from_bytes_is_1024_based() -> None:
    assert mib_from_bytes(Bytes(2**20)) == 1.0
    assert mib_from_bytes(Bytes(3 * 2**19)) == 1.5
    assert mib_from_bytes(Bytes(1_000_000)) < 1.0


def test_gbs_from_bytes_us_is_1000_based() -> None:
    # 1e9 bytes in one second is 1 GB/s exactly. 2**30 bytes in that second is
    # more than 1, which is what distinguishes the 1000 base from 1024.
    assert gbs_from_bytes_us(Bytes(10**9), Microseconds(1e6)) == 1.0
    assert gbs_from_bytes_us(Bytes(2**30), Microseconds(1e6)) == pytest.approx(
        1.073741824
    )


def test_tflops_from_flop_us() -> None:
    assert tflops_from_flop_us(Count(2_000_000), Microseconds(1.0)) == 2.0


def test_tps_from_tokens_us() -> None:
    assert tps_from_tokens_us(Count(1000), Microseconds(1000.0)) == pytest.approx(1e6)


def test_pct_of() -> None:
    assert pct_of(25.0, 200.0) == 12.5


def test_ratio_of() -> None:
    assert ratio_of(3.0, 4.0) == 0.75


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_gbs_from_bytes_us_rejects_a_nonpositive_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration_us must be positive"):
        gbs_from_bytes_us(Bytes(1), Microseconds(duration))


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_tflops_from_flop_us_rejects_a_nonpositive_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration_us must be positive"):
        tflops_from_flop_us(Count(1), Microseconds(duration))


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_tps_from_tokens_us_rejects_a_nonpositive_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration_us must be positive"):
        tps_from_tokens_us(Count(1), Microseconds(duration))


def test_pct_of_rejects_a_zero_whole() -> None:
    with pytest.raises(ValueError, match="percent of zero is undefined"):
        pct_of(1.0, 0.0)


def test_ratio_of_rejects_a_zero_denominator() -> None:
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


def test_median_ci_rank_tightens_as_samples_accumulate() -> None:
    # Under six samples no interval reaches the coverage, so the rank falls back
    # to 1, the widest the samples have. Rank 1 stays the tightest one that
    # covers until nine samples admit the next.
    assert [units_mod._median_ci_rank(n) for n in (1, 5, 6, 8)] == [1, 1, 1, 1]
    assert [units_mod._median_ci_rank(n) for n in (9, 10, 11)] == [2, 2, 2]
    assert units_mod._median_ci_rank(20) == 6
    # The binomial tail steps, so one rank covers several counts.
    assert [units_mod._median_ci_rank(n) for n in (30, 31)] == [10, 10]
    assert units_mod._median_ci_rank(40) == 14


@pytest.mark.parametrize("count", [6, 7, 10, 20, 30, 31, 40, 64])
def test_median_ci_rank_is_the_tightest_interval_holding_the_confidence(
    count: int,
) -> None:
    rank = units_mod._median_ci_rank(count)
    assert units_mod._sign_coverage_pct(count, rank) >= CONFIDENCE_PCT
    assert units_mod._sign_coverage_pct(count, rank + 1) < CONFIDENCE_PCT


def test_min_resolving_samples_is_derived_from_the_confidence() -> None:
    # The definition, not the literal: the widest interval reaches the nominal
    # coverage at this count and misses it at one sample fewer.
    assert units_mod._sign_coverage_pct(MIN_RESOLVING_SAMPLES, 1) >= CONFIDENCE_PCT
    assert units_mod._sign_coverage_pct(MIN_RESOLVING_SAMPLES - 1, 1) < CONFIDENCE_PCT


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
    # Five samples cannot reach the nominal coverage, and the floor says so on its
    # own row rather than printing as though it were nominal.
    assert s.coverage_pct == 93.75
    assert s.coverage_pct < CONFIDENCE_PCT
    assert not s.resolves(Percent(1000.0))


def test_spread_resolution_at_ten_samples() -> None:
    # Ten samples take rank 2, so the interval is [x_(2), x_(9)] = [96, 104] and
    # the half-width is 4 over a median of 100.
    s = Spread.of(us(100.0, 92.0, 108.0, 96.0, 104.0, 100.0, 99.0, 101.0, 100.0, 100.0))
    assert s.median_duration_us == 100.0
    assert s.min_duration_us == 92.0
    assert s.max_duration_us == 108.0
    assert s.spread_pct == 16.0
    assert s.resolution_pct == 4.0
    assert s.sample_count == 10
    assert s.coverage_pct == pytest.approx(97.8516, abs=1e-4)
    assert s.coverage_pct >= CONFIDENCE_PCT


def test_spread_resolution_at_twenty_samples() -> None:
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


def test_spread_of_one_sample_has_no_spread() -> None:
    s = Spread.of(us(7.0))
    assert s.median_duration_us == 7.0
    assert s.min_duration_us == 7.0
    assert s.max_duration_us == 7.0
    assert s.spread_pct == 0.0
    assert s.resolution_pct == 0.0
    assert s.sample_count == 1
    assert s.samples_duration_us == (7.0,)
    # One sample bounds nothing, and a zero floor at zero coverage must not read
    # as a perfectly resolved measurement.
    assert s.coverage_pct == 0.0
    assert not s.resolves(Percent(1000.0))


def test_spread_of_rejects_no_samples() -> None:
    empty: list[Microseconds] = []
    with pytest.raises(ValueError, match="needs at least one sample"):
        Spread.of(empty)


def test_spread_of_identical_zero_samples_has_no_spread() -> None:
    # A measurement that rounds to zero is still a measurement, and identical
    # samples span nothing, so both statistics are zero rather than undefined.
    s = Spread.of(us(0.0, 0.0))
    assert s.median_duration_us == 0.0
    assert s.spread_pct == 0.0
    assert s.resolution_pct == 0.0
    assert s.sample_count == 2


def test_spread_of_rejects_a_zero_median_under_a_nonzero_span() -> None:
    # Samples that differ need a denominator, and a zero median is not one.
    with pytest.raises(ValueError, match="percent of zero is undefined"):
        Spread.of(us(0.0, 0.0, 1.0))


def test_resolves_compares_the_delta_against_the_resolution_floor() -> None:
    s = Spread.of(us(*([1.0] + [98.0] * 5 + [100.0] * 8 + [102.0] * 5 + [1000.0])))
    assert s.resolution_pct == 2.0
    # The floor decides, not the range: a 3 percent claim resolves against a 999
    # percent range because the outliers lie outside the interval.
    assert s.spread_pct == 999.0
    assert s.resolves(Percent(3.0))
    assert not s.resolves(Percent(2.0))
    assert not s.resolves(Percent(1.0))
    # Magnitude decides, so a regression resolves on the same threshold.
    assert s.resolves(Percent(-3.0))
    assert not s.resolves(Percent(-1.0))


@pytest.mark.parametrize("count", range(1, MIN_RESOLVING_SAMPLES))
def test_resolves_is_false_below_the_minimum_sample_count(count: int) -> None:
    # Identical samples put the floor at zero and the gate still refuses: no
    # interval over this few samples reaches CONFIDENCE_PCT, so the floor means
    # nothing and licenses nothing.
    s = Spread.of(us(*[100.0] * count))
    assert s.resolution_pct == 0.0
    assert s.coverage_pct < CONFIDENCE_PCT
    assert not s.resolves(Percent(1000.0))


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


def test_a_sub_nominal_floor_carries_the_coverage_it_actually_has() -> None:
    # The floor still prints below the minimum, because suppressing it would hide
    # how wide the measurement was. It prints beside the coverage that says the
    # nominal figure was not reached, and it licenses nothing.
    s = Spread.of(us(99.0, 100.0, 101.0))
    assert s.resolution_pct == 1.0
    assert s.coverage_pct == 75.0
    assert s.coverage_pct < CONFIDENCE_PCT
    assert not s.resolves(Percent(50.0))


def test_resolves_admits_a_delta_above_a_collapsed_floor_at_the_minimum() -> None:
    s = Spread.of(us(*[100.0] * MIN_RESOLVING_SAMPLES))
    assert s.resolution_pct == 0.0
    assert s.coverage_pct >= CONFIDENCE_PCT
    assert s.resolves(Percent(0.01))
    # A zero delta is not larger than a zero floor.
    assert not s.resolves(Percent(0.0))
