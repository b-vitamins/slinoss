"""Region timers and the median-of-N measurement loop.

One CUDA-event pair per region per iteration, read after a single synchronize at
the end of the loop. No sample carries a synchronize's cost, and every sample is
one real call rather than a mean over an inner loop, so the dispersion in
:class:`~slinoss.perf.units.Spread` is the dispersion of the thing being
measured. A delta smaller than that spread is not a result.

A region is named by a dotted path and the path is the whole of its structure:
``forward.mixer.in_proj`` is a child of ``forward.mixer``. Nothing infers
parentage from call nesting, so a region can be opened anywhere the work happens.

One label yields at most one sample per iteration. A label opened several times in
one call contributes the sum of its intervals, because the question a budget
answers is how much of the step that label owns, not how long one of its visits
took. A per-visit median would understate a label entered once per chunk by the
chunk count.

Backward time is attributed to the same call site as the forward it belongs to.
:func:`call_region` aliases its tensor inputs and hooks the aliases and the
outputs; the backward region runs from the first output-gradient hook to the last
input-gradient hook. This is the only way one label covers both directions
without a second, hand-maintained list of backward region names that drifts.

A CUDA event records on the current stream, which belongs to the current device,
while the synchronize that resolves it names a device explicitly. On any ordinal
but the current one those are two different devices, and the pair then times the
host gap between the two records on an idle stream instead of the work: measured
at 2 us for a copy that took 497,634 us. The loop makes its device current and
then asserts the two ordinals agree, which is the defect itself rather than a
symptom of it.

The loop also reports what share of its own host wall the per-iteration events
account for. A low share is a fact about the work, not a broken timer: a loop
launching one small kernel per iteration spends most of its wall enqueueing, and
20 percent coverage is the honest reading. So the share is reported and not
enforced. Above 100 percent it is impossible, since the wall brackets every event
it is compared against, and that is enforced.

Comparing two implementations takes one loop, not two. :func:`measure_paired`
runs both arms inside every iteration and swaps which goes first on alternate
iterations, so each arm runs first exactly half the time and the verdict comes
off the per-iteration differences. Two separate loops cannot do this: their
medians scatter further than either loop's own floor, measured on this fleet at
0.234% against a 0.211% budget on a whole step and 21.250% against 10.198% on
the smallest shape. See :func:`slinoss.perf.dispersion.paired`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Any, Final, Protocol, TypeVar

import torch
from torch import Tensor

from slinoss.perf.device import ClockPolicy, clock_policy, device_ordinal
from slinoss.perf.dispersion import PairedRow, paired
from slinoss.perf.units import (
    INVARIANT,
    MEDIAN,
    Count,
    Microseconds,
    Milliseconds,
    Nanoseconds,
    Percent,
    PerfRecord,
    Spread,
    TokensPerSecond,
    pct_of,
    tps_from_tokens_us,
    us_from_ms,
    us_from_ns,
)

__all__ = [
    "MAX_TIMER_COVERAGE_PCT",
    "UNATTRIBUTED",
    "PairedMeasurement",
    "RegionRecorder",
    "RegionTiming",
    "Throughput",
    "Timed",
    "TimerError",
    "active_recorder",
    "call_region",
    "measure",
    "measure_paired",
    "on_device",
    "parent_of",
    "region",
]

UNATTRIBUTED = "unattributed"
"""Label for time or memory that belongs to no region."""

MAX_TIMER_COVERAGE_PCT: Final[Percent] = Percent(100.01)
"""Most of the loop's host wall the per-iteration events can account for.

The events tile the loop and the wall brackets all of them plus every boundary
between them, so their sum is bounded by it. A sum that exceeds the wall it sits
inside is measuring something outside the loop, whatever the durations look like.

The bound is not exactly 100% because the two quantities come off two clocks: a
CUDA event pair is timed by the GPU and the wall by the host, and the offset
between the crystals is a systematic ppm-scale term that a loop whose device work
fills its wall lands on. Measured on an A6000 over device-bound regions of 3.5 to
10.7 s, the event clock ran 4.14, 4.50, 8.96, 11.56, 11.83 and 13.06 ppm ahead of
the host clock, and the step-mode profile of the smallest shape failed a bound of
exactly 100% at 8 ppm over a 14.1 s loop. A hundredth of a percent is 100 ppm,
seven times the largest of those, and orders of magnitude below any event pair
that really covers work outside the loop.

There is no lower bound. A loop whose device work is shorter than the host cost of
enqueueing it covers a small fraction of its own wall and is measured correctly;
see :attr:`Timed.timer_coverage_pct`.
"""


class TimerError(RuntimeError):
    """Raised when a measurement's timer cannot be measuring what it names."""


T = TypeVar("T")

ENTER = "enter"
EXIT = "exit"


class _Sample(Protocol):
    """One resolved interval. Resolution is deferred until after a synchronize."""

    def resolve(self) -> Microseconds: ...


class _EventSample:
    """A CUDA-event pair. ``elapsed_time`` reports milliseconds."""

    __slots__ = ("_start", "_stop")

    def __init__(self, start: torch.cuda.Event, stop: torch.cuda.Event) -> None:
        self._start = start
        self._stop = stop

    def resolve(self) -> Microseconds:
        return us_from_ms(Milliseconds(self._start.elapsed_time(self._stop)))


class _HostSample:
    """A monotonic host-clock pair, for the CPU path."""

    __slots__ = ("_start_ns", "_stop_ns")

    def __init__(self, start_ns: int, stop_ns: int) -> None:
        self._start_ns = start_ns
        self._stop_ns = stop_ns

    def resolve(self) -> Microseconds:
        return us_from_ns(Nanoseconds(float(self._stop_ns - self._start_ns)))


@contextmanager
def on_device(device: torch.device) -> Iterator[None]:
    """Make ``device`` current for the body, or do nothing off CUDA.

    Every CUDA timer and every allocator probe in this package reads the current
    device implicitly while naming its device explicitly. Both must be the same
    device or the two halves measure different hardware.

    Args:
        device: The device the work runs on.

    Yields:
        None.
    """
    if device.type != "cuda":
        yield
        return
    with torch.cuda.device(device):
        yield


def _check_current_device(device: torch.device) -> None:
    """Reject a CUDA device that is not the current one.

    Every event in this package records implicitly on the current device and is
    resolved against a synchronize that names its device. Those must be one device.
    An index-less CUDA device means the current one, so it can never fail here.

    Args:
        device: The CUDA device being timed.

    Raises:
        TimerError: If the current device is a different ordinal.
    """
    current = torch.cuda.current_device()
    ordinal = device_ordinal(device)
    if current != ordinal:
        raise TimerError(
            f"timing {device} while cuda:{current} is current; the events would "
            f"record on cuda:{current} and resolve against a synchronize on "
            f"cuda:{ordinal}, which times the host gap between the two records"
        )


def parent_of(label: str) -> str:
    """The dotted parent of a region label, empty at the root."""
    return label.rpartition(".")[0]


def _check_label(label: str) -> None:
    """Reject a label the budget tree cannot represent.

    The tree appends ``UNATTRIBUTED`` as a leaf under every measured parent, so a
    measured region of that name would collide with a generated one and overwrite
    it. Rejecting the name is cheaper than detecting the collision later, when the
    two are indistinguishable.

    Args:
        label: Dotted region label.

    Raises:
        ValueError: If the label is empty or any segment is reserved.
    """
    if not label:
        raise ValueError("a region label must not be empty")
    if UNATTRIBUTED in label.split("."):
        raise ValueError(
            f"region label {label!r} uses the reserved segment {UNATTRIBUTED!r}"
        )


class RegionRecorder:
    """Collects per-iteration intervals for dotted region labels.

    Attributes:
        device: The device the regions run on. Selects the timer mechanism.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._is_cuda = device.type == "cuda"
        self._samples: dict[str, list[tuple[int, _Sample]]] = {}
        self._order: list[str] = []
        self._total: list[_Sample] = []
        self._fires: dict[str, dict[str, list[tuple[int, Any]]]] = {}
        self._stack: list[str] = []
        self._seq = 0
        self._iteration = -1

    def current_label(self) -> str:
        """The innermost open region, empty outside one.

        Memory forensics attributes a saved tensor to this label. Timing does not
        use it: the tree comes from the dotted path, not from call nesting.
        """
        return self._stack[-1] if self._stack else ""

    # -- mechanism ---------------------------------------------------------

    def _mark(self) -> Any:
        if self._is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter_ns()

    def _pair(self, start: Any, stop: Any) -> _Sample:
        if self._is_cuda:
            return _EventSample(start, stop)
        return _HostSample(int(start), int(stop))

    def _append(self, label: str, sample: _Sample) -> None:
        bucket = self._samples.get(label)
        if bucket is None:
            self._samples[label] = bucket = []
            self._order.append(label)
        bucket.append((self._iteration, sample))

    # -- recording ---------------------------------------------------------

    @contextmanager
    def iteration(self) -> Iterator[None]:
        """One measured call. Brackets the total and closes backward boundaries."""
        self._fires = {}
        self._iteration += 1
        start = self._mark()
        try:
            yield
        finally:
            self._total.append(self._pair(start, self._mark()))
            self._close_boundaries()

    @contextmanager
    def region(self, label: str) -> Iterator[None]:
        """Time one region for the current iteration.

        Args:
            label: Dotted region label. The path defines the tree.

        Raises:
            ValueError: If the label is empty or reserves a segment the budget
                tree generates for a remainder.
        """
        _check_label(label)
        self._stack.append(label)
        start = self._mark()
        try:
            yield
        finally:
            self._append(label, self._pair(start, self._mark()))
            self._stack.pop()

    def boundary(self, label: str, kind: str) -> None:
        """Record a backward-pass boundary fire.

        Args:
            label: Dotted region label.
            kind: ``enter`` on an output gradient, ``exit`` on an input gradient.
        """
        self._seq += 1
        sides = self._fires.setdefault(label, {ENTER: [], EXIT: []})
        sides[kind].append((self._seq, self._mark()))

    def _close_boundaries(self) -> None:
        for label, sides in self._fires.items():
            enters, exits = sides[ENTER], sides[EXIT]
            if not enters or not exits:
                continue
            first = min(enters, key=lambda fire: fire[0])[1]
            last = max(exits, key=lambda fire: fire[0])[1]
            self._append(label, self._pair(first, last))
        self._fires = {}

    # -- reading -----------------------------------------------------------

    def samples(self) -> dict[str, list[Microseconds]]:
        """Resolve every recorded interval, in first-seen label order.

        Intervals sharing a label and an iteration are summed, so the result holds
        one duration per iteration the label fired in. Synchronizes once. Nothing
        before this call reads an event.

        Returns:
            Label to per-iteration durations.
        """
        if self._is_cuda:
            torch.cuda.synchronize(self.device)
        out: dict[str, list[Microseconds]] = {}
        for label in self._order:
            per_iteration: dict[int, float] = {}
            for index, sample in self._samples[label]:
                per_iteration[index] = per_iteration.get(index, 0.0) + sample.resolve()
            out[label] = [Microseconds(v) for _, v in sorted(per_iteration.items())]
        return out

    def total_samples(self) -> list[Microseconds]:
        """Per-iteration whole-call durations. Synchronizes once."""
        if self._is_cuda:
            torch.cuda.synchronize(self.device)
        return [s.resolve() for s in self._total]


_ACTIVE: ContextVar[RegionRecorder | None] = ContextVar(
    "slinoss_perf_recorder", default=None
)


def active_recorder() -> RegionRecorder | None:
    """The recorder for the current context, or None outside a measurement."""
    return _ACTIVE.get()


@contextmanager
def region(label: str) -> Iterator[None]:
    """Time a region if a measurement is active, otherwise do nothing.

    The uninstrumented path costs one context-variable read, so this may stay in
    library code.

    Args:
        label: Dotted region label.
    """
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    with recorder.region(label):
        yield


def _tensors(obj: object) -> Iterator[Tensor]:
    """Every tensor in a return value, walking the standard containers.

    A named tuple is a tuple and a mapping is walked by value, so a callable
    returning either has its outputs hooked. Anything else yields nothing: the
    enter boundary then never fires and ``backward.<label>`` is absent rather than
    wrong, which is the same outcome as a call with no gradient at all.
    """
    if isinstance(obj, Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _tensors(item)


def _exit_alias(recorder: RegionRecorder, label: str, value: Tensor) -> Tensor:
    if not value.requires_grad:
        return value
    alias = value.view_as(value)
    alias.register_hook(lambda _grad: recorder.boundary(label, EXIT))
    return alias


def call_region(
    label: str,
    fn: Callable[..., T],
    *tensors: Tensor,
    **kwargs: Any,
) -> T:
    """Call ``fn`` inside ``forward.<label>``, and its backward inside
    ``backward.<label>``.

    The positional tensors are aliased so that the exit boundary fires inside this
    region's own subgraph rather than wherever the caller's tensor is next used.
    Keyword arguments pass through untouched and are not hooked.

    Args:
        label: Dotted region label, without a direction prefix.
        fn: The callable to time.
        *tensors: Tensor inputs. Aliased and hooked when they require grad.
        **kwargs: Passed through.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        ValueError: If the output requires grad while no positional tensor does.
            The enter boundary fires on an output gradient and the exit boundary
            on an input gradient, so that combination records an enter with no
            exit: ``backward.<label>`` is absent rather than zero, and an absent
            bucket is the one failure the closure and nonzero checks cannot see.
            A call whose output requires no grad has no backward at all and is
            not affected.
    """
    recorder = _ACTIVE.get()
    if recorder is None:
        return fn(*tensors, **kwargs)
    backward = f"backward.{label}"
    hooked = tuple(_exit_alias(recorder, backward, t) for t in tensors)
    with recorder.region(f"forward.{label}"):
        out = fn(*hooked, **kwargs)
    grad_out = [value for value in _tensors(out) if value.requires_grad]
    if grad_out and not any(t.requires_grad for t in tensors):
        raise ValueError(
            f"call_region({label!r}) returns a tensor requiring grad but got no "
            f"positional input that requires grad; backward.{label} would record "
            f"an enter with no exit and be absent from the budget"
        )
    for value in grad_out:
        value.register_hook(lambda _grad: recorder.boundary(backward, ENTER))
    return out


@dataclass(frozen=True)
class Throughput(PerfRecord):
    """Tokens per second, with the dispersion that bounds it.

    The rate and the duration it came from travel together, so a throughput
    figure can never be quoted without the dispersion that says whether a
    difference in it is real.

    Attributes:
        label: What was measured.
        token_count: Tokens per measured call.
        duration_us: Median duration of one call.
        spread_pct: Full range of that duration over its median.
        resolution_pct: Half of the floor a delta on this rate must beat; the
            other half belongs to the run it is compared against.
        coverage_pct: Exact coverage of the interval behind that half-width.
        throughput_tps: Tokens per second at the median duration.
    """

    label: str
    token_count: Annotated[Count, INVARIANT]
    duration_us: Annotated[Microseconds, MEDIAN]
    spread_pct: Annotated[Percent, MEDIAN]
    resolution_pct: Annotated[Percent, MEDIAN]
    coverage_pct: Annotated[Percent, MEDIAN]
    throughput_tps: Annotated[TokensPerSecond, MEDIAN]

    @classmethod
    def of(cls, label: str, tokens: Count, spread: Spread) -> Throughput:
        """Derive a throughput from a token count and a measured spread.

        Args:
            label: What was measured.
            tokens: Tokens per call.
            spread: The call's dispersion.

        Returns:
            The record.
        """
        return cls(
            label=label,
            token_count=tokens,
            duration_us=spread.median_duration_us,
            spread_pct=spread.spread_pct,
            resolution_pct=spread.resolution_pct,
            coverage_pct=spread.coverage_pct,
            throughput_tps=tps_from_tokens_us(tokens, spread.median_duration_us),
        )


@dataclass(frozen=True)
class RegionTiming(PerfRecord):
    """One region's dispersion, with its place in the tree.

    Attributes:
        label: Dotted region label.
        parent: Dotted parent label, empty at the root.
        spread: Per-iteration dispersion.
        share_pct: Median duration as a percentage of the measured total.
    """

    label: str
    parent: str
    spread: Spread
    share_pct: Annotated[Percent, MEDIAN]


@dataclass(frozen=True)
class Timed(PerfRecord):
    """The result of one measurement loop.

    ``total`` is the only whole-measurement duration in this package and it is
    always the wall interval around one call of the measured callable. A sum over
    regions is a different quantity with a different name, because a sum over
    regions is not a total: it omits whatever no region covers and double counts
    whatever two regions both cover.

    Attributes:
        label: What was measured.
        timer: ``cuda_event`` or ``perf_counter``.
        clocks: Clock stamp for the run.
        total: Per-iteration whole-call dispersion.
        regions: Every region, in first-seen order.
        root_sum_duration_us: Sum of the median durations of the root regions.
        timer_coverage_pct: Sum of the per-iteration event durations as a
            percentage of the host wall around the whole loop. Below 100 it says
            how much of the loop was host cost outside the timed interval, which is
            a property of the work; a launch-bound loop reads low and is measured
            correctly. A few ppm above 100 is the offset between the two clocks the
            quotient is taken across, and further than that is impossible; see
            :data:`MAX_TIMER_COVERAGE_PCT`.
    """

    label: str
    timer: str
    clocks: str
    total: Spread
    regions: tuple[RegionTiming, ...]
    root_sum_duration_us: Annotated[Microseconds, MEDIAN]
    timer_coverage_pct: Annotated[Percent, MEDIAN]

    def region(self, label: str) -> RegionTiming:
        """Look up one region.

        Args:
            label: Dotted region label.

        Returns:
            Its timing.

        Raises:
            KeyError: If no region carries that label.
        """
        for timing in self.regions:
            if timing.label == label:
                return timing
        raise KeyError(f"no region {label!r} in {self.label!r}")

    def resolves(self, delta_pct: Percent) -> bool:
        """Whether a whole-measurement delta of this size exceeds the spread."""
        return self.total.resolves(delta_pct)


def _prime(fn: Callable[[], object], device: torch.device, iters: int) -> None:
    """Run the warmup with a recorder active, into a recorder that is discarded.

    Whatever the timed loop does on its first iteration and never again -- taking
    the recording branch of :func:`region`, constructing the first timer of every
    region, aliasing and hooking in :func:`call_region` -- is paid by the first
    sample unless it has already happened. In :func:`measure_paired` the first
    iteration always runs the A arm first, so the whole of that one-off cost lands
    on one arm; it read as a 33.057% spread on a baseline whose own dispersion is
    a few tenths of a percent. A warmup that leaves the recorder inactive does not
    warm the thing being warmed.

    The samples are never resolved. This recorder exists to make the path warm,
    and reading it would need a synchronize the caller performs afterwards anyway.

    Args:
        fn: The callable being measured.
        device: The device the regions run on.
        iters: Warmup iterations.
    """
    recorder = RegionRecorder(device)
    token = _ACTIVE.set(recorder)
    try:
        for _ in range(iters):
            with recorder.iteration():
                fn()
    finally:
        _ACTIVE.reset(token)


def measure(
    fn: Callable[[], object],
    *,
    label: str,
    iters: int,
    warmup: int,
    device: torch.device,
    clocks: ClockPolicy | None = None,
) -> Timed:
    """Run ``fn`` and report the median and spread of its duration.

    Warmup calls run with a recorder active and their samples are discarded, so
    the first timed iteration is not the first call to enter the timing machinery;
    see :func:`_prime`. The whole call runs with ``device`` current, so the
    events, the synchronize, and the work are on one device.

    Args:
        fn: The callable to measure. Takes no arguments.
        label: What is being measured.
        iters: Timed calls. Each is one sample.
        warmup: Untimed calls first.
        device: Device to time on. ``cuda`` selects event pairs.
        clocks: Clock policy to stamp. Probed if omitted on a CUDA device.

    Returns:
        The measurement.

    Raises:
        ValueError: If ``iters`` is not positive or ``warmup`` is negative.
        TimerError: If the current CUDA device is not the one being timed, or if the
            per-iteration events sum past the host wall that brackets them by more
            than :data:`MAX_TIMER_COVERAGE_PCT`. Either way the events measure
            something other than what the result names, and they do it without
            failing.
    """
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    if warmup < 0:
        raise ValueError(f"warmup must not be negative, got {warmup}")
    is_cuda = device.type == "cuda"
    with on_device(device):
        if is_cuda:
            _check_current_device(device)
        if warmup:
            _prime(fn, device, warmup)
        if is_cuda:
            torch.cuda.synchronize(device)
        recorder = RegionRecorder(device)
        token = _ACTIVE.set(recorder)
        wall_start_ns = time.perf_counter_ns()
        try:
            for _ in range(iters):
                with recorder.iteration():
                    fn()
        finally:
            _ACTIVE.reset(token)
        # Both reads synchronize, so the wall closes after the last event has
        # landed and covers every sample it is compared against.
        totals = recorder.total_samples()
        region_samples = recorder.samples()
        wall_us = us_from_ns(Nanoseconds(float(time.perf_counter_ns() - wall_start_ns)))
    total = Spread.of(totals)
    coverage = pct_of(Microseconds(sum(totals)), wall_us)
    if coverage > MAX_TIMER_COVERAGE_PCT:
        raise TimerError(
            f"{label!r}: {iters} iteration events sum to {sum(totals):,.3f} us "
            f"inside a loop that took {wall_us:,.3f} us of host wall, which is "
            f"{coverage:.4f}% of it; the events do not fit in the interval that "
            f"brackets them, so they are not timing this loop"
        )
    if clocks is None:
        clocks = clock_policy(device_ordinal(device)) if is_cuda else None
    stamp = clocks.stamp if clocks is not None else "host clock"
    regions = _region_timings(region_samples, total)
    roots = [t.spread.median_duration_us for t in regions if not t.parent]
    return Timed(
        label=label,
        timer="cuda_event" if is_cuda else "perf_counter",
        clocks=stamp,
        total=total,
        regions=regions,
        root_sum_duration_us=Microseconds(sum(roots)),
        timer_coverage_pct=coverage,
    )


@dataclass(frozen=True)
class PairedMeasurement:
    """One loop that measured two arms against each other.

    Attributes:
        timed: The loop. Its two root regions are the arms, and its total covers
            both of them plus whatever sits between.
        comparison: The verdict on the per-iteration differences.
    """

    timed: Timed
    comparison: PairedRow


def measure_paired(
    a_label: str,
    a: Callable[[], object],
    b_label: str,
    b: Callable[[], object],
    *,
    label: str,
    iters: int,
    warmup: int,
    device: torch.device,
    clocks: ClockPolicy | None = None,
) -> PairedMeasurement:
    """Measure two arms in one loop and judge the difference.

    Each iteration runs both arms, in an order that swaps every iteration, so
    each arm runs first exactly half the time and neither pays the whole of
    whatever the first position costs. Every sample is still one real call.

    Args:
        a_label: Region label for the baseline arm.
        a: The baseline callable. Takes no arguments.
        b_label: Region label for the arm under test.
        b: The callable under test. Takes no arguments.
        label: What is being compared.
        iters: Timed iterations. Even, so the order swap balances. Each iteration
            yields one sample per arm.
        warmup: Untimed iterations first. Its parity does not matter: the swap
            balances over the timed iterations however many precede them. At
            least one, or the timing machinery's one-off cost is paid by the
            first timed iteration, which always runs ``a`` first; the swap cannot
            balance a cost that happens once.
        device: Device to time on.
        clocks: Clock policy to stamp. Probed if omitted on a CUDA device.

    Returns:
        The loop and the verdict.

    Raises:
        ValueError: If the two labels are equal, which would sum both arms into
            one region and compare it against itself, or if ``iters`` is odd,
            which leaves one iteration's order unbalanced by any other.
        TimerError: As :func:`measure`.
    """
    if a_label == b_label:
        raise ValueError(
            f"measure_paired needs two distinct labels; both arms are {a_label!r}, "
            f"so the recorder would sum them into one region"
        )
    if iters % 2 != 0:
        raise ValueError(
            f"measure_paired needs an even iters so each arm runs first exactly "
            f"half the time, got {iters}"
        )
    arms = ((a_label, a), (b_label, b))
    index = 0

    def body() -> None:
        nonlocal index
        order = arms if index % 2 == 0 else arms[::-1]
        index += 1
        for name, fn in order:
            with region(name):
                fn()

    timed = measure(
        body,
        label=label,
        iters=iters,
        warmup=warmup,
        device=device,
        clocks=clocks,
    )
    return PairedMeasurement(
        timed=timed,
        comparison=paired(
            label,
            a_label,
            timed.region(a_label).spread.samples_duration_us,
            b_label,
            timed.region(b_label).spread.samples_duration_us,
        ),
    )


def _region_timings(
    samples: Mapping[str, Sequence[Microseconds]], total: Spread
) -> tuple[RegionTiming, ...]:
    out: list[RegionTiming] = []
    for label, durations in samples.items():
        spread = Spread.of(durations)
        out.append(
            RegionTiming(
                label=label,
                parent=parent_of(label),
                spread=spread,
                share_pct=pct_of(spread.median_duration_us, total.median_duration_us),
            )
        )
    return tuple(out)
