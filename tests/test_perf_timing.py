"""Region timers and the median-of-N measurement loop.

Every measurement here runs on the CPU path, where a region is a monotonic
host-clock pair. Each timed callable does real work: ``Spread.of`` divides by the
median, so a region of exactly zero duration is undefined rather than fast.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from torch import Tensor

from slinoss.perf.device import ClockPolicy
from slinoss.perf.timing import (
    MAX_TIMER_COVERAGE_PCT,
    UNATTRIBUTED,
    RegionRecorder,
    Throughput,
    Timed,
    TimerError,
    active_recorder,
    call_region,
    measure,
    on_device,
    parent_of,
    region,
)
from slinoss.perf.units import Count, Megahertz, Microseconds, Percent, Spread

CPU = torch.device("cpu")
SLEEP_S = 0.001
WIDTH = 1024


def siblings() -> None:
    """Two sibling regions under one parent, each of nonzero duration."""
    with region("step.first"):
        time.sleep(SLEEP_S)
    with region("step.second"):
        time.sleep(SLEEP_S)


def spread_of(us: float, *, pct: float, count: int) -> Spread:
    """A Spread over samples whose full range is ``pct`` of the median ``us``.

    Built through ``Spread.of``, so every derived field is real arithmetic. All but
    the two extreme samples sit on the median, so through eight samples the
    median's interval spans the whole range and ``resolution_pct`` is half of
    ``pct``. Needs at least two samples.
    """
    half = us * pct / 200.0
    inner = [Microseconds(us)] * (count - 2)
    return Spread.of([Microseconds(us - half), *inner, Microseconds(us + half)])


def timed_literal() -> Timed:
    """A measurement with no regions, a 10 percent range and a 5 percent floor.

    The sample count is above ``MIN_RESOLVING_SAMPLES``, or the floor would
    license nothing whatever the delta.
    """
    return Timed(
        label="step",
        timer="perf_counter",
        clocks="host clock",
        total=spread_of(100.0, pct=10.0, count=8),
        regions=(),
        root_sum_duration_us=Microseconds(0.0),
        timer_coverage_pct=Percent(99.0),
    )


@pytest.mark.parametrize(
    ("label", "parent"),
    [
        ("forward.mixer.in_proj", "forward.mixer"),
        ("forward.mixer", "forward"),
        ("x", ""),
    ],
)
def test_parent_of(label: str, parent: str) -> None:
    assert parent_of(label) == parent


def test_region_outside_a_measurement_records_nothing() -> None:
    assert active_recorder() is None
    with region("orphan"):
        pass
    assert active_recorder() is None


def test_measure_on_cpu_records_sibling_regions() -> None:
    timed = measure(siblings, label="step", iters=3, warmup=0, device=CPU)
    assert timed.timer == "perf_counter"
    assert timed.clocks == "host clock"
    assert timed.total.sample_count == 3
    assert [t.label for t in timed.regions] == ["step.first", "step.second"]
    assert [t.parent for t in timed.regions] == ["step", "step"]
    for t in timed.regions:
        assert t.spread.sample_count == 3
        # Two equal sleeps in one call, so neither region owns the whole total
        # and neither is a per-unit fraction misread as a percentage.
        assert 10.0 < t.share_pct < 90.0
    # A sum over regions is not the total, and no region here is a root.
    assert timed.root_sum_duration_us == 0.0


@pytest.mark.parametrize(
    ("iters", "warmup", "match"),
    [
        (0, 0, "iters must be positive"),
        (-1, 0, "iters must be positive"),
        (1, -1, "warmup must not be negative"),
    ],
)
def test_measure_rejects_bad_counts(iters: int, warmup: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        measure(lambda: None, label="step", iters=iters, warmup=warmup, device=CPU)


@pytest.mark.parametrize(
    ("locked", "stamp"), [(True, "locked at 1410 MHz"), (False, "unlocked")]
)
def test_measure_stamps_the_given_clock_policy(locked: bool, stamp: str) -> None:
    policy = ClockPolicy(
        locked=locked,
        sm_clock_mhz=Megahertz(1410.0),
        max_sm_clock_mhz=Megahertz(1710.0),
        detail="supplied by the test",
    )
    timed = measure(
        siblings, label="step", iters=1, warmup=0, device=CPU, clocks=policy
    )
    assert timed.clocks == stamp


def test_warmup_calls_record_nothing() -> None:
    timed = measure(siblings, label="step", iters=2, warmup=3, device=CPU)
    assert timed.total.sample_count == 2
    assert [t.spread.sample_count for t in timed.regions] == [2, 2]


def test_timer_coverage_is_the_share_of_the_loop_wall_the_events_cover() -> None:
    # The iteration intervals tile the loop, so the only wall outside them is the
    # per-iteration boundary. On the host path the body is the whole cost and the
    # coverage is near the ceiling; nothing here asserts a lower bound, because a
    # loop whose body is cheaper than its own boundary reads low and is right.
    timed = measure(siblings, label="step", iters=3, warmup=0, device=CPU)
    assert 0.0 < timed.timer_coverage_pct <= MAX_TIMER_COVERAGE_PCT


def test_measure_refuses_events_that_outlast_the_loop_around_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The wall brackets every event it is compared against, so a sum past it is not
    # a slow measurement, it is a measurement of something else. A mark advancing a
    # second per call makes three iterations claim three seconds inside a loop that
    # took milliseconds.
    ticks = itertools.count(0, 1_000_000_000)

    def stub_mark(_self: RegionRecorder) -> int:
        return next(ticks)

    monkeypatch.setattr(RegionRecorder, "_mark", stub_mark)

    def body() -> None:
        time.sleep(SLEEP_S)

    with pytest.raises(TimerError, match=r"do not fit in the interval"):
        measure(body, label="step", iters=3, warmup=0, device=CPU)


def test_on_device_yields_without_selecting_a_device_off_cuda() -> None:
    # torch.cuda.device raises on a host with no driver, so the CPU path must not
    # reach it. That the body runs at all is the whole contract.
    calls: list[str] = []
    with on_device(CPU):
        calls.append("body")
    assert calls == ["body"]


def test_timed_region_lookup() -> None:
    timed = measure(siblings, label="step", iters=1, warmup=0, device=CPU)
    assert timed.region("step.second").label == "step.second"
    with pytest.raises(KeyError, match="no region 'absent'"):
        timed.region("absent")


def test_resolves_delegates_to_the_total_spread() -> None:
    timed = timed_literal()
    # The floor is the resolution, half the range the samples cover.
    assert timed.total.resolution_pct == 5.0
    assert timed.resolves(Percent(6.0))
    assert not timed.resolves(Percent(4.0))
    assert timed.resolves(Percent(-6.0)) == timed.total.resolves(Percent(-6.0))


def test_current_label_is_the_innermost_open_region() -> None:
    recorder = RegionRecorder(CPU)
    assert recorder.current_label() == ""
    with recorder.region("outer"):
        assert recorder.current_label() == "outer"
        with recorder.region("outer.inner"):
            assert recorder.current_label() == "outer.inner"
        assert recorder.current_label() == "outer"
    assert recorder.current_label() == ""
    # Labels resolve in first-seen order, which is close order, not open order.
    assert list(recorder.samples()) == ["outer.inner", "outer"]


def test_region_rejects_an_empty_label() -> None:
    recorder = RegionRecorder(CPU)
    with pytest.raises(ValueError, match="must not be empty"), recorder.region(""):
        pass


@pytest.mark.parametrize("label", ["unattributed", "step.unattributed"])
def test_region_rejects_the_reserved_segment(label: str) -> None:
    # The budget tree generates a remainder leaf of this name under every measured
    # parent, so a measured region of that name would be overwritten by one.
    recorder = RegionRecorder(CPU)
    with pytest.raises(ValueError, match="reserved segment"), recorder.region(label):
        pass


def test_region_accepts_a_label_that_merely_contains_the_reserved_word() -> None:
    recorder = RegionRecorder(CPU)
    with recorder.region(f"step.{UNATTRIBUTED}_scan"):
        time.sleep(SLEEP_S)
    assert list(recorder.samples()) == [f"step.{UNATTRIBUTED}_scan"]


def test_one_label_yields_one_sample_per_iteration() -> None:
    # A label entered once per chunk owns the sum of its visits, not the median of
    # one of them, so the sample count is the iteration count either way.
    def body() -> None:
        for _ in range(3):
            with region("step.chunk"):
                time.sleep(SLEEP_S)

    timed = measure(body, label="step", iters=2, warmup=0, device=CPU)
    chunk = timed.region("step.chunk")
    assert chunk.spread.sample_count == 2
    # Three sleeps summed, against a total that also contains loop overhead.
    assert chunk.spread.median_duration_us > 2.0 * SLEEP_S * 1e6
    assert chunk.share_pct <= 100.0


def test_region_samples_are_kept_in_measurement_order() -> None:
    # Each call sleeps longer than the last, so the samples increase only if they
    # are retained as measured. Sorting them would hide the drift, and dropping one
    # would still read as a complete list.
    calls: list[int] = []

    def body() -> None:
        calls.append(len(calls) + 1)
        with region("step.growing"):
            time.sleep(SLEEP_S * calls[-1])

    timed = measure(body, label="step", iters=3, warmup=0, device=CPU)
    spread = timed.region("step.growing").spread
    samples = spread.samples_duration_us
    assert spread.sample_count == 3
    assert len(samples) == 3
    assert samples[0] < samples[1] < samples[2]
    assert spread.min_duration_us == samples[0]
    assert spread.median_duration_us == samples[1]
    assert spread.max_duration_us == samples[2]
    # The whole-call samples are per iteration too, and each covers its region.
    assert len(timed.total.samples_duration_us) == 3
    assert all(
        total > part for total, part in zip(timed.total.samples_duration_us, samples)
    )


def test_call_region_records_forward_and_backward() -> None:
    x = torch.randn(WIDTH, requires_grad=True)

    def fn(t: Tensor) -> tuple[Tensor, Tensor]:
        return t * 2.0, t.sin()

    def body() -> None:
        first, second = call_region("mixer", fn, x)
        (first.sum() + second.sum()).backward()

    timed = measure(body, label="step", iters=2, warmup=1, device=CPU)
    assert [t.label for t in timed.regions] == ["forward.mixer", "backward.mixer"]
    # Two output-gradient hooks and one input-gradient hook per call collapse to
    # one backward interval per iteration.
    assert timed.region("backward.mixer").spread.sample_count == 2
    assert timed.region("forward.mixer").spread.sample_count == 2
    assert timed.region("backward.mixer").parent == "backward"


def test_call_region_rejects_a_grad_output_with_no_grad_input() -> None:
    # The gradient reaches ``weight``, not ``x``, so the exit boundary has nothing
    # to fire on and the backward bucket would be absent rather than zero.
    weight = torch.randn(WIDTH, requires_grad=True)
    x = torch.randn(WIDTH)

    def fn(t: Tensor) -> Tensor:
        return t * weight

    def body() -> None:
        call_region("free", fn, x).sum().backward()

    with pytest.raises(ValueError, match=r"backward\.free would record an enter"):
        measure(body, label="step", iters=1, warmup=0, device=CPU)


def test_call_region_records_no_backward_when_the_alias_is_unused() -> None:
    # ``x`` requires grad, so the guard passes, but ``fn`` drops it: the exit hook
    # sits on an alias no output depends on. One enter, no exit, no interval.
    weight = torch.randn(WIDTH, requires_grad=True)
    x = torch.randn(WIDTH, requires_grad=True)

    def fn(_t: Tensor) -> Tensor:
        return weight * 2.0

    def body() -> None:
        call_region("dropped", fn, x).sum().backward()

    timed = measure(body, label="step", iters=2, warmup=0, device=CPU)
    assert timed.region("forward.dropped").spread.sample_count == 2
    with pytest.raises(KeyError, match=r"no region 'backward\.dropped'"):
        timed.region("backward.dropped")


def test_call_region_hooks_outputs_inside_the_standard_containers() -> None:
    # A mixer returns a named tuple and a head often returns a mapping. Both are
    # walked, so the enter boundary fires and the backward interval is recorded
    # rather than the label going absent.
    x = torch.randn(WIDTH, requires_grad=True)

    def fn(t: Tensor) -> dict[str, list[Tensor]]:
        return {"out": [t * 2.0], "aux": [t.sin()]}

    def body() -> None:
        got = call_region("mapped", fn, x)
        (got["out"][0].sum() + got["aux"][0].sum()).backward()

    timed = measure(body, label="step", iters=2, warmup=0, device=CPU)
    assert timed.region("backward.mapped").spread.sample_count == 2


def test_call_region_records_no_backward_for_a_non_tensor_return() -> None:
    # A callable that reduces to a Python scalar has no gradient path through the
    # region, so there is nothing to hook and no backward interval to record. The
    # forward is still timed.
    x = torch.randn(WIDTH, requires_grad=True)

    def fn(t: Tensor) -> float:
        return float(t.detach().sum())

    def body() -> None:
        assert isinstance(call_region("scalar", fn, x), float)

    timed = measure(body, label="step", iters=2, warmup=0, device=CPU)
    assert [t.label for t in timed.regions] == ["forward.scalar"]


def test_call_region_outside_a_measurement_calls_through() -> None:
    x = torch.randn(WIDTH)

    def negate(t: Tensor) -> Tensor:
        return -t

    assert active_recorder() is None
    assert torch.equal(call_region("orphan", negate, x), -x)


def test_call_region_passes_kwargs_through() -> None:
    x = torch.ones(WIDTH)
    bias = torch.full((WIDTH,), 3.0)
    got: list[Tensor] = []

    def fn(t: Tensor, *, offset: Tensor) -> Tensor:
        return t + offset

    def body() -> None:
        got.append(call_region("bias", fn, x, offset=bias))

    timed = measure(body, label="step", iters=1, warmup=0, device=CPU)
    assert torch.equal(got[0], torch.full((WIDTH,), 4.0))
    assert timed.region("forward.bias").spread.sample_count == 1


def test_throughput_is_taken_from_the_spread() -> None:
    spread = spread_of(2000.0, pct=10.0, count=5)
    got = Throughput.of("step", Count(4096), spread)
    assert got.label == "step"
    assert got.token_count == 4096
    assert got.duration_us == spread.median_duration_us
    assert got.spread_pct == spread.spread_pct
    assert got.resolution_pct == spread.resolution_pct
    # The floor travels with the coverage it was computed at: five samples reach
    # neither, and a rate printed without the coverage would read as though the
    # floor were nominal.
    assert got.coverage_pct == spread.coverage_pct
    assert got.coverage_pct == 93.75
    assert got.throughput_tps == pytest.approx(4096 / 2000e-6)


@pytest.mark.cuda
def test_measure_on_cuda_uses_event_pairs() -> None:
    # The event path and its single trailing synchronize are unreachable on the
    # host path; this is the only test that covers them.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device("cuda")
    a = torch.randn(512, 512, device=dev)

    def body() -> None:
        with region("step.matmul"):
            for _ in range(8):
                _ = a @ a

    timed = measure(body, label="step", iters=3, warmup=1, device=dev)
    assert timed.timer == "cuda_event"
    assert timed.clocks != "host clock"
    assert timed.region("step.matmul").spread.sample_count == 3
    # Eight 512x512 matmuls per iteration cost the device less than the host spends
    # enqueueing them and creating the events, so the loop covers a fraction of its
    # own wall. That is the measurement, not a fault in it.
    assert 0.0 < timed.timer_coverage_pct <= MAX_TIMER_COVERAGE_PCT


@pytest.mark.cuda
def test_on_device_makes_the_named_device_current() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    with torch.cuda.device(0):
        assert torch.cuda.current_device() == 0
        with on_device(torch.device("cuda", 1)):
            assert torch.cuda.current_device() == 1
        assert torch.cuda.current_device() == 0


@pytest.mark.cuda
def test_measure_times_a_device_that_is_not_current() -> None:
    # measure makes its device current for the whole loop, so timing a device other
    # than the current one is not an error at the call site.
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    dev = torch.device("cuda", 1)
    a = torch.randn(2048, 2048, device=dev)

    def body() -> None:
        with region("step.matmul"):
            for _ in range(8):
                _ = a @ a

    with torch.cuda.device(0):
        timed = measure(body, label="step", iters=3, warmup=1, device=dev)
    assert timed.region("step.matmul").spread.median_duration_us > 0.0
    # 2048-cube matmuls are long enough that the device, not the host, owns the
    # wall. Under the wrong-device defect the events would time the host gap between
    # two records on an idle stream and this would read near zero.
    assert timed.timer_coverage_pct > 50.0


@pytest.mark.cuda
def test_measure_refuses_to_time_a_device_that_is_not_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A CUDA event records on the current device; the synchronize that resolves it
    # names its device. On two different ordinals the pair times the host gap
    # between the two record calls on an idle stream: 2 us for a copy that took
    # 497,634 us. Nothing in the durations says so, so the two ordinals are compared
    # directly. Dropping the device guard is the regression this catches.
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")

    @contextmanager
    def no_op(_device: torch.device) -> Iterator[None]:
        yield

    monkeypatch.setattr("slinoss.perf.timing.on_device", no_op)
    with torch.cuda.device(0), pytest.raises(TimerError, match=r"while cuda:0 is cur"):
        measure(
            lambda: None,
            label="step",
            iters=1,
            warmup=0,
            device=torch.device("cuda", 1),
        )
