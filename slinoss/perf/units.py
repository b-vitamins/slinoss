"""Unit-carrying field types, and the record schema that validates them.

A measurement report is only as good as its field names. This module makes the
naming rules mechanical rather than editorial:

- A value with a unit is annotated with that unit's type, and its field name ends
  with that unit's suffix. A field annotated ``Microseconds`` and named
  ``duration_ms`` fails at class definition.
- A modelled value is marked ``MODELLED`` and its name starts with ``est_``. The
  mark and the prefix travel together through every reduction, so aggregation
  cannot launder a model into a measurement.
- One record may not carry a measured and a modelled field of the same unit. A
  record is the unit of printing, so this is what keeps the two out of adjacent
  columns.
- A shared-memory conflict count is not comparable across kernels or shapes
  without its wavefront denominator, so a record declaring one must declare the
  other. A ``resolution_pct`` is not a floor without the coverage of the interval
  it came from, so the same pairing rule applies to it and ``coverage_pct``.
- An integer-backed unit does not take ``MEDIAN``. The median of an even number
  of integers is not an integer, and reducing it back to one would floor the
  value silently.
- A ``tuple[U, ...]`` of a unit type carries the same naming obligation as a
  scalar ``U``, so a field holding raw samples cannot drop the unit from its
  name. No tuple field takes ``MEDIAN``: a median across records is not
  elementwise, and the median of a list of tuples is a tuple nobody measured.
- A record whose fields are functions of one shared sample list reduces through
  ``combine``, not field by field. Reducing such fields separately produces a
  record that contradicts its own samples.

Violations raise at class definition time, not at report time.

Percent is the only proportion type. There is no fraction type, so no field can
be read off by 100x. Byte-to-mebibyte conversion is 1024-based and
byte-to-gigabyte-per-second conversion is 1000-based; both live here exactly
once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from math import comb
from statistics import median
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    NewType,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

__all__ = [
    "CONFIDENCE_PCT",
    "INVARIANT",
    "MEDIAN",
    "MIN_RESOLVING_SAMPLES",
    "MODELLED",
    "SUM",
    "Bytes",
    "Count",
    "GBPerSecond",
    "Mebibytes",
    "Megahertz",
    "Microseconds",
    "Milliseconds",
    "Nanoseconds",
    "Percent",
    "PerfRecord",
    "Ratio",
    "Spread",
    "TFlopsPerSecond",
    "TokensPerSecond",
    "aggregate",
    "gbs_from_bytes_us",
    "median_ci",
    "mib_from_bytes",
    "ms_from_us",
    "pct_of",
    "ratio_of",
    "tflops_from_flop_us",
    "tps_from_tokens_us",
    "us_from_ms",
    "us_from_ns",
]

# ---------------------------------------------------------------------------
# Unit types
#
# NewType over float and int, so the suffix rule is checked by name and the unit
# itself is checked by pyright: passing Milliseconds where Microseconds is
# expected is an error, and constructing either from a bare float requires
# naming the unit.
# ---------------------------------------------------------------------------

Nanoseconds = NewType("Nanoseconds", float)
Microseconds = NewType("Microseconds", float)
Milliseconds = NewType("Milliseconds", float)
Bytes = NewType("Bytes", int)
Mebibytes = NewType("Mebibytes", float)
GBPerSecond = NewType("GBPerSecond", float)
TFlopsPerSecond = NewType("TFlopsPerSecond", float)
TokensPerSecond = NewType("TokensPerSecond", float)
Megahertz = NewType("Megahertz", float)
Percent = NewType("Percent", float)
Ratio = NewType("Ratio", float)
Count = NewType("Count", int)

_SUFFIX: Final[dict[Any, str]] = {
    Nanoseconds: "_ns",
    Microseconds: "_us",
    Milliseconds: "_ms",
    Bytes: "_bytes",
    Mebibytes: "_mib",
    GBPerSecond: "_gbs",
    TFlopsPerSecond: "_tflops",
    TokensPerSecond: "_tps",
    Megahertz: "_mhz",
    Percent: "_pct",
    Ratio: "_ratio",
    Count: "_count",
}

_SUFFIXES: Final[tuple[str, ...]] = tuple(_SUFFIX.values())

_INT_BACKED: Final[frozenset[Any]] = frozenset(
    unit for unit in _SUFFIX if unit.__supertype__ is int
)

_EST_PREFIX: Final = "est_"


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


class _Marker:
    """An annotation marker. Identity is the whole of its meaning."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{self.name}>"


MODELLED: Final = _Marker("modelled")
"""The value comes from an analytic model, not from a counter."""

SUM: Final = _Marker("sum")
"""Reduce across samples by addition."""

MEDIAN: Final = _Marker("median")
"""Reduce across samples by median. The default for any duration."""

INVARIANT: Final = _Marker("invariant")
"""Reduce across samples by requiring equality. Disagreement is an error."""

_REDUCTIONS: Final[tuple[_Marker, ...]] = (SUM, MEDIAN, INVARIANT)


def _unwrap(hint: Any) -> tuple[Any, tuple[Any, ...]]:
    if get_origin(hint) is Annotated:
        base, *extras = get_args(hint)
        return base, tuple(extras)
    return hint, ()


def _element_unit(base: Any) -> Any | None:
    """The unit type a homogeneous ``tuple[U, ...]`` annotation holds, else None.

    A tuple of one unit type is a sample list. It carries the unit in the same
    way a scalar does, so the suffix rule applies to it; a heterogeneous or
    fixed-length tuple is a structure and does not.
    """
    if get_origin(base) is not tuple:
        return None
    args = get_args(base)
    if len(args) == 2 and args[1] is Ellipsis and args[0] in _SUFFIX:
        return args[0]
    return None


def _reduction(name: str, markers: Sequence[Any], required: bool) -> _Marker:
    found = [m for m in markers if m in _REDUCTIONS]
    if len(found) > 1:
        raise TypeError(f"field {name!r} declares {len(found)} reductions; declare one")
    if not found:
        if required:
            raise TypeError(
                f"field {name!r} has a unit and no reduction; "
                f"annotate it with SUM, MEDIAN, or INVARIANT"
            )
        return INVARIANT
    return found[0]


# ---------------------------------------------------------------------------
# Record base
# ---------------------------------------------------------------------------


class PerfRecord:
    """Base for every performance record. Validates its subclass's field names.

    Subclasses are frozen dataclasses. The validation runs at class creation, so
    a misnamed field is an import-time failure and never reaches a report.

    Raises:
        TypeError: On a unit-typed field whose name lacks the matching suffix, a
            field name claiming a unit its type does not have, a modelled field
            without the ``est_`` prefix or a measured field with it, a unit-typed
            field with no reduction marker or more than one, an integer-backed
            unit marked ``MEDIAN``, a tuple field marked ``MEDIAN``, a record
            carrying both a measured and a modelled field of one unit, a conflict
            count without a wavefront denominator, or a resolution floor without
            its interval's coverage.
    """

    _reductions: ClassVar[dict[str, _Marker]]

    @classmethod
    def combine(cls, samples: Sequence[Any]) -> Any:
        """How to reduce several of these into one, when field by field is wrong.

        Returning None accepts the field-by-field reduction, which is what a
        record of independent fields wants. A record whose fields are all
        functions of one shared sample list overrides this: reducing each field
        on its own would take the median of the minima and the median of the
        half-widths, and the result would contradict its own sample list.

        Args:
            samples: Records of this exact type, at least one.

        Returns:
            The combined record, or None to reduce field by field.
        """
        del samples
        return None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        hints = get_type_hints(cls, include_extras=True)
        reductions: dict[str, _Marker] = {}
        modelled_units: set[Any] = set()
        measured_units: set[Any] = set()
        for name, hint in hints.items():
            if hint is ClassVar or get_origin(hint) is ClassVar:
                continue
            if name.startswith("_"):
                continue
            base, markers = _unwrap(hint)
            element = _element_unit(base)
            carrier = base if element is None else element
            unit = _SUFFIX.get(carrier)
            if unit is not None:
                if not name.endswith(unit):
                    raise TypeError(
                        f"field {name!r} is {carrier.__name__} and must end in {unit!r}"
                    )
            else:
                claimed = [s for s in _SUFFIXES if name.endswith(s)]
                if claimed:
                    raise TypeError(
                        f"field {name!r} claims unit {claimed[0]!r} but is not a "
                        f"unit type"
                    )
            is_modelled = MODELLED in markers
            if is_modelled and not name.startswith(_EST_PREFIX):
                raise TypeError(f"modelled field {name!r} must start with 'est_'")
            if not is_modelled and name.startswith(_EST_PREFIX):
                raise TypeError(f"field {name!r} is named est_ but is not MODELLED")
            if unit is not None:
                (modelled_units if is_modelled else measured_units).add(carrier)
            how = _reduction(name, markers, required=unit is not None)
            if how is MEDIAN and get_origin(base) is tuple:
                raise TypeError(
                    f"field {name!r} is a tuple and MEDIAN; a median across "
                    f"records is not elementwise, so declare SUM or INVARIANT"
                )
            if how is MEDIAN and base in _INT_BACKED:
                raise TypeError(
                    f"field {name!r} is {base.__name__} and MEDIAN; the median "
                    f"of integers is not an integer, so declare SUM or INVARIANT"
                )
            reductions[name] = how
        both = modelled_units & measured_units
        if both:
            names = sorted(b.__name__ for b in both)
            raise TypeError(
                f"{cls.__name__} carries a measured and a modelled {names[0]}; "
                f"a model and a measurement do not share a table"
            )
        declares_conflicts = any(n.endswith("conflict_count") for n in reductions)
        if declares_conflicts and "wavefront_count" not in reductions:
            raise TypeError(
                f"{cls.__name__} declares a conflict count and no "
                f"wavefront_count; a raw conflict count is not comparable"
            )
        if "resolution_pct" in reductions and "coverage_pct" not in reductions:
            raise TypeError(
                f"{cls.__name__} declares a resolution floor and no coverage_pct; "
                f"a half-width whose interval misses nominal coverage is not a floor"
            )
        cls._reductions = reductions


R = TypeVar("R", bound=PerfRecord)


def aggregate(samples: Sequence[R]) -> R:
    """Reduce samples of one record type, field by field unless it says otherwise.

    Field names are reused verbatim, so an ``est_`` prefix survives aggregation
    by construction. This is the only reduction over records; there is no path
    that renames a field on the way out. A record type that defines
    :meth:`PerfRecord.combine` reduces through that instead, because its fields
    are not independent.

    Args:
        samples: One or more records of the same concrete type.

    Returns:
        A record of that type.

    Raises:
        ValueError: On an empty sequence, on mixed record types, or on a field
            marked ``INVARIANT`` whose samples disagree.
    """
    if not samples:
        raise ValueError("aggregate needs at least one sample")
    cls = type(samples[0])
    if any(type(s) is not cls for s in samples):
        raise ValueError(
            f"aggregate needs one record type, got {len(set(map(type, samples)))}"
        )
    if not is_dataclass(cls):
        raise ValueError(f"{cls.__name__} is not a dataclass")
    combined = cls.combine(samples)
    if combined is not None:
        return cast("R", combined)
    out: dict[str, Any] = {}
    for field in fields(cls):
        values = [getattr(s, field.name) for s in samples]
        how = cls._reductions.get(field.name, INVARIANT)
        first = values[0]
        if isinstance(first, PerfRecord):
            out[field.name] = aggregate(values)
        elif how is SUM and isinstance(first, tuple):
            # Summing sample lists concatenates them: the samples behind an
            # aggregate are every sample behind its parts.
            out[field.name] = tuple(item for value in values for item in value)
        elif how is SUM:
            out[field.name] = type(first)(sum(values))
        elif how is MEDIAN:
            out[field.name] = type(first)(median(values))
        else:
            if any(v != first for v in values):
                raise ValueError(
                    f"field {field.name!r} is INVARIANT and disagrees across "
                    f"{len(samples)} samples"
                )
            out[field.name] = first
    return cls(**out)


# ---------------------------------------------------------------------------
# Conversions
#
# One definition each. Mebibytes are 1024-based; GB/s is 1000-based. A zero
# denominator is a broken measurement and raises rather than propagating a nan.
# ---------------------------------------------------------------------------


def us_from_ns(value: Nanoseconds) -> Microseconds:
    """Nanoseconds to microseconds."""
    return Microseconds(value / 1e3)


def ms_from_us(value: Microseconds) -> Milliseconds:
    """Microseconds to milliseconds."""
    return Milliseconds(value / 1e3)


def us_from_ms(value: Milliseconds) -> Microseconds:
    """Milliseconds to microseconds. This is the CUDA-event unit conversion."""
    return Microseconds(value * 1e3)


def mib_from_bytes(value: Bytes) -> Mebibytes:
    """Bytes to mebibytes, 1024-based."""
    return Mebibytes(value / 2**20)


def gbs_from_bytes_us(moved: Bytes, duration: Microseconds) -> GBPerSecond:
    """Bytes moved over a duration, as gigabytes per second, 1000-based.

    Raises:
        ValueError: If the duration is not positive.
    """
    if duration <= 0.0:
        raise ValueError(f"duration_us must be positive, got {duration}")
    return GBPerSecond(moved / (duration * 1e3))


def tflops_from_flop_us(flop: Count, duration: Microseconds) -> TFlopsPerSecond:
    """Floating-point operations over a duration, as teraflop per second.

    Raises:
        ValueError: If the duration is not positive.
    """
    if duration <= 0.0:
        raise ValueError(f"duration_us must be positive, got {duration}")
    return TFlopsPerSecond(flop / (duration * 1e6))


def tps_from_tokens_us(tokens: Count, duration: Microseconds) -> TokensPerSecond:
    """Tokens over a duration, as tokens per second.

    Raises:
        ValueError: If the duration is not positive.
    """
    if duration <= 0.0:
        raise ValueError(f"duration_us must be positive, got {duration}")
    return TokensPerSecond(tokens * 1e6 / duration)


def pct_of(part: float, whole: float) -> Percent:
    """Part over whole, as a percentage.

    Raises:
        ValueError: If the whole is zero.
    """
    if whole == 0.0:
        raise ValueError("percent of zero is undefined")
    return Percent(100.0 * part / whole)


def ratio_of(numerator: float, denominator: float) -> Ratio:
    """Numerator over denominator.

    Raises:
        ValueError: If the denominator is zero.
    """
    if denominator == 0.0:
        raise ValueError("ratio with a zero denominator is undefined")
    return Ratio(numerator / denominator)


CONFIDENCE_PCT: Final[Percent] = Percent(95.0)
"""Nominal coverage of the interval that sets a spread's resolution floor."""


def _sign_coverage_pct(count: int, rank: int) -> Percent:
    """Exact coverage of ``[x_(rank), x_(count + 1 - rank)]`` for the median.

    However the samples are distributed, the number of them below the population
    median is binomial with probability one half, and the interval misses the
    median exactly when that number falls outside ``[rank, count - rank]``. So
    the coverage is a binomial tail and nothing here assumes normality, a
    variance, or independence beyond the samples being drawn from one
    distribution.
    """
    tail = sum(comb(count, i) for i in range(rank))
    return Percent(100.0 * (1.0 - 2.0 * tail / 2**count))


def _median_ci_rank(count: int) -> int:
    """Tightest order-statistic rank whose interval meets :data:`CONFIDENCE_PCT`.

    Coverage falls as the rank rises, so the search stops at the first rank that
    misses. Returns 1, the widest interval the samples admit, when no rank
    reaches the nominal coverage.
    """
    best = 1
    for rank in range(1, count // 2 + 1):
        if _sign_coverage_pct(count, rank) < CONFIDENCE_PCT:
            break
        best = rank
    return best


MIN_RESOLVING_SAMPLES: Final[int] = next(
    count for count in range(1, 65) if _sign_coverage_pct(count, 1) >= CONFIDENCE_PCT
)
"""Fewest samples that can bound a median at :data:`CONFIDENCE_PCT`.

Derived, not chosen. Below this the widest interval the samples admit still
misses the nominal coverage, so no delta resolves at any size.
"""


F = TypeVar("F", bound=float)


def median_ci(samples: Sequence[F]) -> tuple[F, F, Percent]:
    """Confidence interval on the median, read off the order statistics.

    The interval is two of the samples, so it carries their unit and needs no
    scale, no variance, and no distribution. The one below is the bound a delta
    is judged against; see :attr:`Spread.resolution_pct`.

    Args:
        samples: At least one sample. Order does not matter.

    Returns:
        The lower bound, the upper bound, and the exact coverage of that interval.
        The rank is the tightest one still reaching :data:`CONFIDENCE_PCT`, or 1
        when no rank reaches it, and the returned coverage is what it says.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("median_ci needs at least one sample")
    ordered = sorted(samples)
    count = len(ordered)
    rank = _median_ci_rank(count)
    return ordered[rank - 1], ordered[count - rank], _sign_coverage_pct(count, rank)


@dataclass(frozen=True)
class Spread(PerfRecord):
    """A timed quantity with the dispersion that bounds what it can resolve.

    Two dispersions are reported, because they answer different questions and
    only one of them is a floor.

    ``spread_pct`` is the full range over the median. It exposes outliers: one
    stalled iteration moves it and moves nothing else. It grows with the sample
    count, since a wider sample has more chance to contain the tail, so it is not
    a bound on how well the median is pinned and is not used as one.

    ``resolution_pct`` is that bound: the half-width of the distribution-free
    confidence interval on the median at :data:`CONFIDENCE_PCT`, read off the
    order statistics. It shrinks as samples accumulate, so paying for iterations
    buys resolution, and a lone outlier falls outside it. A change smaller than
    this is not a result.

    A half-width is not a range. Two medians differ by more than measurement
    noise when their separation exceeds the sum of their two half-widths, so a
    caller comparing a delta against a floor sums; it does not take the larger.

    ``coverage_pct`` is what the half-width is actually worth. Below
    :data:`MIN_RESOLVING_SAMPLES` samples no interval reaches nominal coverage,
    and the record then carries the widest interval the samples admit together
    with the coverage that interval really has. The floor still prints, labelled
    with what it means, and :meth:`resolves` refuses it.

    Nothing in this package reports a median without this record beside it.

    Attributes:
        median_duration_us: Median over the samples.
        min_duration_us: Fastest sample.
        max_duration_us: Slowest sample.
        spread_pct: ``(max - min) / median``, as a percentage.
        resolution_pct: Half-width of the median's confidence interval, as a
            percentage of the median. Zero when the interval collapses, which
            happens when the samples inside it agree.
        coverage_pct: Exact coverage of that interval. At or above
            :data:`CONFIDENCE_PCT` on any nominal sample count, below it when
            there are too few samples for any interval to reach nominal.
        sample_count: Number of samples.
        samples_duration_us: The samples themselves, in measurement order, so a
            reader can recompute every figure above and see any drift. Retained
            in full; a truncated sample list would read as a complete one. Tuple
            fields are tables of their own, so this reaches the JSON and stays
            out of every rendered row.
    """

    median_duration_us: Annotated[Microseconds, MEDIAN]
    min_duration_us: Annotated[Microseconds, MEDIAN]
    max_duration_us: Annotated[Microseconds, MEDIAN]
    spread_pct: Annotated[Percent, MEDIAN]
    resolution_pct: Annotated[Percent, MEDIAN]
    coverage_pct: Annotated[Percent, MEDIAN]
    sample_count: Annotated[Count, SUM]
    samples_duration_us: Annotated[tuple[Microseconds, ...], SUM] = ()

    @classmethod
    def of(cls, samples: Sequence[Microseconds]) -> Spread:
        """Summarize raw per-sample durations.

        Args:
            samples: Per-sample durations, at least one.

        Returns:
            The spread record.

        Raises:
            ValueError: If ``samples`` is empty, or if the samples differ and
                their median is zero, which leaves the spread undefined.
        """
        if not samples:
            raise ValueError("Spread.of needs at least one sample")
        ordered = sorted(samples)
        mid = Microseconds(median(ordered))
        low = ordered[0]
        high = ordered[-1]
        ci_low, ci_high, coverage = median_ci(ordered)
        width = ci_high - ci_low
        # Identical samples have zero spread whatever the median is, including a
        # median that rounds to zero. Only a nonzero span needs a denominator.
        return cls(
            median_duration_us=mid,
            min_duration_us=low,
            max_duration_us=high,
            spread_pct=Percent(0.0) if high == low else pct_of(high - low, mid),
            resolution_pct=Percent(0.0) if width == 0.0 else pct_of(0.5 * width, mid),
            coverage_pct=coverage,
            sample_count=Count(len(ordered)),
            samples_duration_us=tuple(samples),
        )

    @classmethod
    def combine(cls, samples: Sequence[Spread]) -> Spread:
        """Recompute one spread from every sample behind the parts.

        Every field here is a function of the sample list, so reducing them
        separately would emit a record that disagrees with its own samples: the
        median of the minima is not the minimum, and the median of two
        half-widths is not the half-width of the pooled interval. Summing the
        sample counts on top of that would lift the total past
        :data:`MIN_RESOLVING_SAMPLES` while every part's floor still failed to
        reach nominal coverage.

        Args:
            samples: Spreads to pool, at least one.

        Returns:
            The spread of the concatenated samples.

        Raises:
            ValueError: On an empty sequence, or if any part carries no samples,
                which leaves nothing to recompute from.
        """
        if not samples:
            raise ValueError("Spread.combine needs at least one sample")
        pooled: list[Microseconds] = []
        for part in samples:
            if not part.samples_duration_us:
                raise ValueError(
                    "Spread.combine needs the samples behind every part; one was "
                    "built without them and cannot be pooled"
                )
            pooled.extend(part.samples_duration_us)
        return cls.of(pooled)

    def resolves(self, delta_pct: Percent) -> bool:
        """Whether a delta of this size is larger than the measurement floor.

        Args:
            delta_pct: Magnitude of the claimed change, as a percentage.

        Returns:
            True if the delta exceeds ``resolution_pct``. False whenever
            ``coverage_pct`` is short of :data:`CONFIDENCE_PCT`, because a floor
            whose interval misses nominal coverage licenses nothing at any size.
            Below :data:`MIN_RESOLVING_SAMPLES` samples that is every interval, so
            the sample count needs no separate gate and does not get one.
        """
        if self.coverage_pct < CONFIDENCE_PCT:
            return False
        return abs(delta_pct) > self.resolution_pct
