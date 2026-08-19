"""Queried device state, measured ceilings, and the profiler capture window.

The clock and sharing probes are driven through an injected
:data:`slinoss.perf.device.SmiQuery` and
:data:`slinoss.perf.device.ComputeAppsQuery`, and the two readers behind them
through a monkeypatched ``subprocess.run``, so every parse branch runs without
``nvidia-smi``. The two ceilings need a GPU to measure but their refusal on a CPU
device does not, and the verdicts are pure arithmetic over hand-built ceilings.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence

import pytest
import torch

from slinoss.perf.capture import profiler_window
from slinoss.perf.ceiling import (
    CLASS_FLOOR_PCT,
    DRAM_BOUND,
    SERIAL_TINY,
    TENSOR_BOUND,
    TensorCeiling,
    ceilings,
    dram_ceiling,
    serial_verdict,
    tensor_ceiling,
    tensor_verdict,
)
from slinoss.perf.device import (
    ClockPolicy,
    ComputeAppsQuery,
    Contention,
    DeviceInfo,
    SmiQuery,
    clock_policy,
    compute_apps_query,
    contention,
    device_info,
    device_ordinal,
    require_cuda,
    smi_query,
)
from slinoss.perf.units import (
    Bytes,
    Count,
    Mebibytes,
    Megahertz,
    Microseconds,
    Percent,
    Spread,
    TFlopsPerSecond,
    gbs_from_bytes_us,
    tflops_from_flop_us,
)

CPU = torch.device("cpu")

FIELDS = (
    "clocks.sm",
    "clocks.max.sm",
    "clocks.applications.graphics",
    "clocks_throttle_reasons.applications_clocks_setting",
)


def _returning(line: str | None) -> SmiQuery:
    """An injected nvidia-smi reader that always answers with one line."""

    def query(fields: Sequence[str], index: int) -> str | None:
        del fields, index
        return line

    return query


def _apps(text: str | None) -> ComputeAppsQuery:
    """An injected compute-process reader that always answers with one block."""

    def query(index: int) -> str | None:
        del index
        return text

    return query


def _failed_run(
    failure: str,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A ``subprocess.run`` that fails the way ``failure`` names.

    Args:
        failure: ``missing`` raises OSError, ``timeout`` raises TimeoutExpired,
            ``nonzero`` exits 9, ``blank`` exits 0 with no output line.

    Returns:
        The replacement callable.
    """

    def fake_run(
        cmd: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        if failure == "missing":
            raise OSError("no such file")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=20)
        if failure == "nonzero":
            return subprocess.CompletedProcess(list(cmd), 9, "", "no devices")
        return subprocess.CompletedProcess(list(cmd), 0, "   \n", "")

    return fake_run


def _spread(median_us: float) -> Spread:
    """A Spread over eight samples spanning 2 percent of the median.

    Built through ``Spread.of``, so the dispersion fields are the arithmetic the
    pipeline performs rather than a claim about it.
    """
    half = median_us * 0.01
    inner = [Microseconds(median_us)] * 6
    return Spread.of(
        [Microseconds(median_us - half), *inner, Microseconds(median_us + half)]
    )


def _device(sm_count: int = 84) -> DeviceInfo:
    """A device record built from literals."""
    return DeviceInfo(
        name="Test Part",
        capability="8.6",
        sm_count=Count(sm_count),
        warp_thread_count=Count(32),
        max_threads_per_sm_count=Count(1536),
        regs_per_sm_count=Count(65536),
        smem_per_block_bytes=Bytes(49152),
        smem_optin_per_block_bytes=Bytes(101376),
        smem_per_sm_bytes=Bytes(102400),
        l2_bytes=Bytes(6291456),
        total_memory_bytes=Bytes(51041271808),
        clocks=ClockPolicy(
            locked=False,
            sm_clock_mhz=Megahertz(1740.0),
            max_sm_clock_mhz=Megahertz(1800.0),
            detail="fabricated",
        ),
        sharing=Contention(
            probed=True,
            foreign_process_count=Count(0),
            foreign_memory_mib=Mebibytes(0.0),
            utilization_pct=Percent(0.0),
            detail="fabricated",
        ),
    )


def _tensor(achieved_tflops: float) -> TensorCeiling:
    return TensorCeiling(
        label="8192x8192x8192 torch.bfloat16 gemm",
        flop_count=Count(1099511627776),
        duration=_spread(4200.0),
        achieved_tflops=TFlopsPerSecond(achieved_tflops),
    )


# ---------------------------------------------------------------------------
# clock_policy
#
# A failed or unparsed probe reports unlocked. That is the conservative
# direction: it keeps the spread discipline on rather than silently claiming a
# pinned clock.
# ---------------------------------------------------------------------------


def test_clock_policy_passes_the_fields_and_reports_a_missing_nvidia_smi() -> None:
    seen: list[tuple[tuple[str, ...], int]] = []

    def query(fields: Sequence[str], index: int) -> str | None:
        seen.append((tuple(fields), index))
        return None

    policy = clock_policy(3, query)
    assert seen == [(FIELDS, 3)]
    assert policy.detail == "nvidia-smi unavailable"
    assert not policy.locked
    assert policy.stamp == "unlocked"
    assert policy.sm_clock_mhz == 0.0
    assert policy.max_sm_clock_mhz == 0.0


def test_clock_policy_reports_an_active_applications_clock_as_locked() -> None:
    policy = clock_policy(0, _returning("1740, 1800, 1740, Active"))
    assert policy.locked
    assert policy.stamp == "locked at 1740 MHz"
    assert policy.sm_clock_mhz == 1740.0
    assert policy.max_sm_clock_mhz == 1800.0
    assert "clocks.sm=1740" in policy.detail
    assert "applications_clocks_setting=Active" in policy.detail


def test_clock_policy_reports_anything_else_as_unlocked() -> None:
    # Both halves of the condition: an inactive setting, and an active setting
    # over an applications clock of zero.
    for line in ("1740, 1800, 1740, Not Active", "1740, 1800, 0, Active"):
        policy = clock_policy(0, _returning(line))
        assert not policy.locked
        assert policy.stamp == "unlocked"


def test_clock_policy_reports_an_unparsed_line() -> None:
    # Too few fields and too many are both unparsed, so the field count is
    # compared for equality and not for a minimum.
    for line in ("1740, 1800", "1740, 1800, 1740, Active, extra"):
        policy = clock_policy(0, _returning(line))
        assert policy.detail.startswith("unparsed nvidia-smi output:")
        assert not policy.locked
        assert policy.sm_clock_mhz == 0.0


def test_clock_policy_parses_a_non_numeric_clock_as_zero() -> None:
    # A probe that answers [N/A] is a fact to report, not an exception to raise.
    policy = clock_policy(0, _returning("[N/A], 1800, 1740, Active"))
    assert policy.sm_clock_mhz == 0.0
    assert policy.max_sm_clock_mhz == 1800.0
    assert "clocks.sm=[N/A]" in policy.detail


# ---------------------------------------------------------------------------
# smi_query
# ---------------------------------------------------------------------------


def test_smi_query_builds_the_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[list[str], int]] = []

    def fake_run(
        cmd: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int)
        seen.append((list(cmd), timeout))
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(list(cmd), 0, "1740, 1800\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert smi_query(("clocks.sm", "clocks.max.sm"), 2) == "1740, 1800"
    assert seen == [
        (
            [
                "nvidia-smi",
                "--id=2",
                "--query-gpu=clocks.sm,clocks.max.sm",
                "--format=csv,noheader,nounits",
            ],
            20,
        )
    ]


def test_smi_query_returns_none_on_every_failed_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing binary, a timeout, a nonzero exit, and an answer with no line are
    # four distinct outcomes and one report: the probe did not run.
    for failure in ("missing", "timeout", "nonzero", "blank"):
        monkeypatch.setattr(subprocess, "run", _failed_run(failure))
        assert smi_query(FIELDS, 0) is None


# ---------------------------------------------------------------------------
# compute_apps_query and contention
#
# A shared device is the measurement condition that moved a median by 2.33x on
# this fleet, so it is stamped the way the clock state is. A probe that fails
# reports shared, never exclusive.
# ---------------------------------------------------------------------------


def test_compute_apps_query_builds_the_argv_and_reads_an_idle_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[list[str], int]] = []

    def fake_run(
        cmd: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int)
        seen.append((list(cmd), timeout))
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(list(cmd), 0, "17, 28\n99, 36918\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert compute_apps_query(1) == "17, 28\n99, 36918"
    assert seen == [
        (
            [
                "nvidia-smi",
                "--id=1",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            20,
        )
    ]

    # A device with no compute process prints nothing. That is not a failed probe,
    # and conflating the two would report an exclusive device as unknown.
    def idle_run(
        cmd: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(list(cmd), 0, "\n", "")

    monkeypatch.setattr(subprocess, "run", idle_run)
    assert compute_apps_query(0) == ""


def test_compute_apps_query_returns_none_on_a_failed_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for failure in ("nonzero", "missing", "timeout"):
        monkeypatch.setattr(subprocess, "run", _failed_run(failure))
        assert compute_apps_query(0) is None


def test_contention_reports_a_device_holding_only_this_process() -> None:
    got = contention(
        0, apps=_apps("4321, 512"), query=_returning("0, 512"), own_pid=4321
    )
    assert got.exclusive
    assert got.foreign_process_count == 0
    assert got.foreign_memory_mib == 0.0
    assert got.utilization_pct == 0.0
    assert got.stamp == "exclusive"
    assert "own pid 4321" in got.detail


def test_contention_counts_every_other_process_on_the_device() -> None:
    got = contention(
        0,
        apps=_apps("4321, 512\n3823071, 36918\n3270062, 28"),
        query=_returning("100, 37458"),
        own_pid=4321,
    )
    assert not got.exclusive
    assert got.foreign_process_count == 2
    assert got.foreign_memory_mib == 36946.0
    assert got.utilization_pct == 100.0
    assert got.stamp == (
        "shared with 2 processes holding 36,946 MiB at 100% utilization"
    )
    one = contention(0, apps=_apps("99, 28"), query=_returning("0, 44"), own_pid=4321)
    assert one.stamp == "shared with 1 process holding 28 MiB at 0% utilization"


def test_contention_reports_a_failed_probe_as_shared() -> None:
    # Unknown is not exclusive. Claiming an exclusive device off a probe that did
    # not run is the one direction that turns a contended median into a clean one.
    # Either half missing is a failed probe: an idle device whose utilization did
    # not read is still not a device this measurement had to itself.
    for apps, query in (
        (_apps(None), _returning("0, 44")),
        (_apps(""), _returning(None)),
    ):
        got = contention(0, apps=apps, query=query, own_pid=4321)
        assert not got.exclusive
        assert got.foreign_process_count == 0
        assert got.detail == "nvidia-smi unavailable"
        assert got.stamp == "sharing unknown"


def test_contention_reports_an_unparsed_probe_as_shared() -> None:
    # A process row and the utilization line are parsed separately, and too few
    # fields and too many are both unparsed.
    for apps, query in (
        (_apps("4321"), _returning("0, 44")),
        (_apps(""), _returning("0")),
        (_apps(""), _returning("0, 44, extra")),
    ):
        got = contention(0, apps=apps, query=query, own_pid=4321)
        assert not got.exclusive
        assert got.detail.startswith("unparsed nvidia-smi output:")
        assert got.stamp == "sharing unknown"


def test_contention_reads_a_non_numeric_field_as_zero() -> None:
    # A probe that answers [N/A] is a fact to report, not an exception to raise.
    got = contention(
        0,
        apps=_apps("99, [N/A]"),
        query=_returning("[N/A], [N/A]"),
        own_pid=4321,
    )
    assert not got.exclusive
    assert got.foreign_process_count == 1
    assert got.foreign_memory_mib == 0.0
    assert got.utilization_pct == 0.0


def test_contention_defaults_to_this_process_id() -> None:
    # The measuring process holds a context on the device it measures, so without
    # this exclusion every report would read as shared with itself.
    got = contention(0, apps=_apps(f"{os.getpid()}, 512"), query=_returning("0, 512"))
    assert got.exclusive


# ---------------------------------------------------------------------------
# DeviceInfo
# ---------------------------------------------------------------------------


def test_block_floor_count_is_twice_the_sm_count() -> None:
    assert _device(84).block_floor_count == 168
    assert _device(1).block_floor_count == 2


def test_device_info_refuses_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # Driven through the availability probe rather than skipped on a GPU host, or
    # the refusal would only ever be exercised where it cannot be reached.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device_info needs CUDA"):
        device_info()


def test_require_cuda_refuses_a_device_no_counter_exists_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One guard for all five drivers. The alternative is each of them permitting a
    # host device and then failing in whichever probe reaches for the ordinal
    # first, after the inputs are allocated and the warmup has run.
    for spec in ("cpu", "meta"):
        with pytest.raises(RuntimeError, match="is not a usable cuda device"):
            require_cuda(spec)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="is not a usable cuda device"):
        require_cuda("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_require_cuda_resolves_an_ordinal_a_report_can_name() -> None:
    # An index-less device carries index None at runtime and means the current
    # device, so both spellings must name an ordinal.
    for spec in ("cuda", "cuda:0"):
        device = require_cuda(spec)
        assert device.type == "cuda"
        assert device_ordinal(device) == 0


def test_device_info_refuses_an_ordinal_no_device_carries() -> None:
    # `device_ordinal` returns -1 for a CPU device, so a driver that permits
    # `--device cpu` arrives here with -1. Without this the call reaches
    # `get_device_properties(-1)` and fails on the ordinal rather than on the
    # reason, or reports the absence of CUDA on a host that has it.
    with pytest.raises(ValueError, match="device_info needs a cuda ordinal"):
        device_info(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_device_info_reads_the_part_it_runs_on() -> None:
    info = device_info(0)
    assert info.name
    assert info.capability.count(".") == 1
    assert info.sm_count > 0
    assert info.warp_thread_count == 32
    assert info.smem_optin_per_block_bytes >= info.smem_per_block_bytes
    assert info.total_memory_bytes > 0
    assert info.block_floor_count == 2 * info.sm_count
    # This process holds a context on the device, so the probe ran and excluded
    # it. Whether anything else is there is the host's business, not a property
    # the test can assert.
    assert info.sharing.detail != "nvidia-smi unavailable"
    assert info.sharing.foreign_memory_mib >= 0.0


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------


def test_the_ceilings_refuse_a_cpu_device() -> None:
    with pytest.raises(RuntimeError, match="dram_ceiling needs a CUDA device"):
        dram_ceiling(CPU)
    with pytest.raises(RuntimeError, match="tensor_ceiling needs a CUDA device"):
        tensor_ceiling(CPU)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_dram_ceiling_measures_a_copy() -> None:
    """The copy ceiling, taken from the fastest sample rather than the median.

    The estimator is pinned because reverting it is silent and harmful: a median
    absorbs foreign load into the denominator, which inflates every ratio built on
    the ceiling and turns a slow kernel into a passing one.
    """
    ceiling = dram_ceiling(
        torch.device("cuda"), requested_bytes=64 << 20, iters=3, warmup=1
    )
    assert ceiling.moved_bytes > 0
    assert ceiling.achieved_gbs > 0.0
    assert ceiling.duration.sample_count == 3
    assert "per buffer" in ceiling.label
    assert ceiling.achieved_gbs == pytest.approx(
        gbs_from_bytes_us(ceiling.moved_bytes, ceiling.duration.min_duration_us)
    )
    # Strict whenever any sample was slower than the fastest, which is the only
    # case where the two estimators differ at all.
    assert ceiling.achieved_gbs >= gbs_from_bytes_us(
        ceiling.moved_bytes, ceiling.duration.median_duration_us
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_tensor_ceiling_measures_a_gemm() -> None:
    """The GEMM ceiling, taken from the fastest sample. Same estimator, same why."""
    ceiling = tensor_ceiling(torch.device("cuda"), dim=1024, iters=3, warmup=1)
    assert ceiling.flop_count == 2 * 1024**3
    assert ceiling.achieved_tflops > 0.0
    assert ceiling.duration.sample_count == 3
    assert "bfloat16" in ceiling.label
    assert ceiling.achieved_tflops == pytest.approx(
        tflops_from_flop_us(ceiling.flop_count, ceiling.duration.min_duration_us)
    )
    assert ceiling.achieved_tflops >= tflops_from_flop_us(
        ceiling.flop_count, ceiling.duration.median_duration_us
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_ceilings_measures_both_and_reads_the_device() -> None:
    both = ceilings(torch.device("cuda"), iters=3, warmup=1)
    assert both.device.sm_count > 0
    assert both.dram.achieved_gbs > 0.0
    assert both.tensor.achieved_tflops > 0.0


# ---------------------------------------------------------------------------
# Class verdicts
# ---------------------------------------------------------------------------


def test_class_floor_pct_holds_the_declared_bars() -> None:
    assert CLASS_FLOOR_PCT[DRAM_BOUND] == 85.0
    assert CLASS_FLOOR_PCT[TENSOR_BOUND] == 70.0
    assert CLASS_FLOOR_PCT[SERIAL_TINY] == 2.0
    assert len(CLASS_FLOOR_PCT) == 3


def test_tensor_verdict_divides_by_the_measured_ceiling() -> None:
    verdict = tensor_verdict("gemm", TFlopsPerSecond(240.0), _tensor(300.0))
    assert verdict.declared == TENSOR_BOUND
    assert verdict.achieved_pct == 80.0
    assert verdict.required_pct == 70.0
    assert verdict.passed
    assert (
        tensor_verdict("gemm", TFlopsPerSecond(180.0), _tensor(300.0)).achieved_pct
        == 60.0
    )
    assert not tensor_verdict("gemm", TFlopsPerSecond(180.0), _tensor(300.0)).passed
    assert tensor_verdict("gemm", TFlopsPerSecond(210.0), _tensor(300.0)).passed


def test_serial_verdict_bounds_the_share_of_the_step() -> None:
    verdict = serial_verdict("norm", Percent(1.5))
    assert verdict.declared == SERIAL_TINY
    assert verdict.achieved_pct == 1.5
    assert verdict.required_pct == 2.0
    assert verdict.passed
    assert not serial_verdict("norm", Percent(3.0)).passed
    assert serial_verdict("norm", Percent(2.0)).passed


def test_serial_is_an_upper_bound_and_the_tensor_bar_is_a_floor() -> None:
    # Both achieve 1% of their bar's quantity: the serial kernel passes because its
    # bar is a ceiling on step share, the tensor kernel fails. The DRAM direction is
    # the same comparison against the time floor; see tests/test_perf_ceiling.py.
    assert serial_verdict("norm", Percent(1.0)).passed
    assert not tensor_verdict("gemm", TFlopsPerSecond(3.0), _tensor(300.0)).passed


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def test_the_capture_window_rejects_a_device_it_cannot_drain() -> None:
    # Without this the window fails inside `torch.cuda.device` with a message
    # about an unexpected device type, after the caller has already allocated the
    # inputs and run the whole warmup. The requirement belongs to the window, so
    # the window states it.
    # The guard raises on entry, so the body never runs.
    with (
        pytest.raises(ValueError, match="capture window needs a cuda device"),
        profiler_window(torch.device("cpu")),
    ):
        raise AssertionError("the window opened on a cpu device")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_profiler_window_closes_on_both_paths() -> None:
    # The closing synchronize and profiler stop run from a finally, or a failed
    # capture would leave the profiler collecting into the next one.
    dev = torch.device("cuda")
    with profiler_window(dev):
        torch.empty(16, device=dev).fill_(1.0)
    with pytest.raises(RuntimeError, match="body failed"), profiler_window(dev):
        raise RuntimeError("body failed")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_profiler_window_makes_its_device_current() -> None:
    # Both edges drain the device the work is on. On another ordinal the window
    # would close before the work it brackets had landed.
    other = torch.device("cuda", 1)
    with torch.cuda.device(0):
        with profiler_window(other):
            assert torch.cuda.current_device() == 1
        assert torch.cuda.current_device() == 0
