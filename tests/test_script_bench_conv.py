"""The conv bench driver: argument parsing, arm naming, and the exit status.

Every run is the smallest standard conv shape for two iterations, because what is
under test is the driver and not the operator. The driver refuses any device a
report cannot name, so every argv names a CUDA one.

Three collaborators are pinned rather than called. ``device_info`` shells out to
``nvidia-smi`` twice per call, ``clock_policy`` once more per measurement, and
``ceilings`` allocates two 512 MiB buffers and runs an 8192-cube GEMM. None of
them is the driver's own behaviour. The paired verdict is pinned too, in the test
that asserts an exit status: the null test resolves or refuses on the samples it is
given, never on whether the clock cooperated.

The dtype axis does not interact with anything the driver decides, so it is swept
once in ``tests/test_perf_workload.py`` and fixed at fp32 here. The verdict
wording belongs to :mod:`slinoss.perf.dispersion` and is pinned there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.bench import bench_conv
from scripts.bench.bench_conv import (
    arm_labels,
    bench,
    compare_backends,
    main,
    parse_args,
)
from slinoss import _C
from slinoss.perf import timing
from slinoss.perf.ceiling import Ceilings, DramCeiling, TensorCeiling
from slinoss.perf.device import ClockPolicy, Contention, DeviceInfo
from slinoss.perf.dispersion import paired
from slinoss.perf.report import rate_table
from slinoss.perf.timing import PairedMeasurement
from slinoss.perf.units import (
    Bytes,
    Count,
    GBPerSecond,
    Mebibytes,
    Megahertz,
    Microseconds,
    Percent,
    Spread,
    TFlopsPerSecond,
)
from slinoss.perf.workload import ConvInputs, ConvShape, conv_shape_by_name

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

CUDA = torch.device("cuda")

TINY = conv_shape_by_name("tiny")
"""The cheapest standard conv shape: ``B=1 T=256 D=16 W=4``."""

PAIRS = 8
"""Pairs behind a pinned verdict. Eight reaches nominal coverage, so the verdict
turns on whether the interval excludes zero and not on the pair count."""

MEASURE_PAIRED = bench_conv.measure_paired
"""The unpatched paired loop, held so a second pin in one test wraps the real one
rather than the previous stub."""


def _argv(out: Path, *extra: str) -> list[str]:
    """The cheapest legal run: fp32, two iterations, no warmup.

    Args:
        out: Report base path. Every caller uses ``tmp_path``.
        *extra: Arguments the test varies.

    Returns:
        The argument list. ``--iters 2`` is even, which the paired loop requires.
    """
    return [
        "--device",
        "cuda",
        "--dtype",
        "fp32",
        "--iters",
        "2",
        "--warmup",
        "0",
        "--out",
        str(out),
        *extra,
    ]


def _clocks() -> ClockPolicy:
    """A pinned clock, so every stamp in a report is a literal.

    Locking is denied on the verification fleet, so a real probe always stamps
    ``unlocked``. Claiming the opposite here is what proves a stamp carries the
    policy the run was handed rather than a default.
    """
    return ClockPolicy(
        locked=True,
        sm_clock_mhz=Megahertz(1740.0),
        max_sm_clock_mhz=Megahertz(1800.0),
        detail="fabricated",
    )


def _device() -> DeviceInfo:
    """A fabricated device record, shaped like an sm_86 part."""
    return DeviceInfo(
        name="Test Part",
        capability="8.6",
        sm_count=Count(84),
        warp_thread_count=Count(32),
        max_threads_per_sm_count=Count(1536),
        regs_per_sm_count=Count(65536),
        smem_per_block_bytes=Bytes(49152),
        smem_optin_per_block_bytes=Bytes(101376),
        smem_per_sm_bytes=Bytes(102400),
        l2_bytes=Bytes(6291456),
        total_memory_bytes=Bytes(51041271808),
        clocks=_clocks(),
        sharing=Contention(
            probed=True,
            foreign_process_count=Count(0),
            foreign_memory_mib=Mebibytes(0.0),
            utilization_pct=Percent(0.0),
            detail="fabricated",
        ),
    )


def _ceilings() -> Ceilings:
    """Fabricated ceilings. Both probes need CUDA and neither is under test."""
    duration = Spread.of([Microseconds(1000.0)])
    return Ceilings(
        device=_device(),
        dram=DramCeiling(
            label="fabricated copy",
            moved_bytes=Bytes(1 << 30),
            duration=duration,
            achieved_gbs=GBPerSecond(760.0),
        ),
        tensor=TensorCeiling(
            label="fabricated gemm",
            flop_count=Count(1 << 40),
            duration=duration,
            achieved_tflops=TFlopsPerSecond(120.0),
        ),
    )


@pytest.fixture(autouse=True)
def pinned_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the clock probe every measurement takes, for every test here."""
    monkeypatch.setattr(timing, "clock_policy", lambda _index: _clocks())


@pytest.fixture
def pinned_device(monkeypatch: pytest.MonkeyPatch) -> DeviceInfo:
    """Pin the report's device record for every entry point that builds one."""
    info = _device()
    monkeypatch.setattr(bench_conv, "device_info", lambda _index: info)
    return info


def _pin_ceilings(monkeypatch: pytest.MonkeyPatch) -> list[torch.device]:
    """Replace the measured ceilings with fabricated ones.

    Returns:
        The devices ``ceilings`` was asked for, appended in call order.
    """
    seen: list[torch.device] = []

    def patched(device: torch.device) -> Ceilings:
        seen.append(device)
        return _ceilings()

    monkeypatch.setattr(bench_conv, "ceilings", patched)
    return seen


def _count_input_sets(monkeypatch: pytest.MonkeyPatch) -> list[ConvInputs]:
    """Record every input set the driver builds.

    Returns:
        The input sets, appended in construction order.
    """
    built: list[ConvInputs] = []
    real = bench_conv.make_conv_inputs

    def patched(
        shape: ConvShape,
        device: torch.device,
        *,
        dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = True,
        seed: int = 0,
    ) -> ConvInputs:
        got = real(shape, device, dtype=dtype, requires_grad=requires_grad, seed=seed)
        built.append(got)
        return got

    monkeypatch.setattr(bench_conv, "make_conv_inputs", patched)
    return built


def _pin_comparison(
    monkeypatch: pytest.MonkeyPatch, *, a_us: float, b_us: float
) -> None:
    """Pin the paired verdict to :data:`PAIRS` literal differences.

    The loop still runs and the report still carries its real regions and rates;
    only the verdict comes from known samples. A null test that resolves because
    the clock drifted is not a test of the exit status.

    Args:
        monkeypatch: Patcher.
        a_us: Every baseline sample.
        b_us: Every sample from the arm under test.
    """
    real = MEASURE_PAIRED

    def patched(
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
        got = real(
            a_label,
            a,
            b_label,
            b,
            label=label,
            iters=iters,
            warmup=warmup,
            device=device,
            clocks=clocks,
        )
        return replace(
            got,
            comparison=paired(
                label,
                a_label,
                [Microseconds(a_us)] * PAIRS,
                b_label,
                [Microseconds(b_us)] * PAIRS,
            ),
        )

    monkeypatch.setattr(bench_conv, "measure_paired", patched)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_every_shape_in_both_modes_with_no_comparison() -> None:
    args = parse_args([])
    # None, not a list: main expands it to every standard shape.
    assert args.shape is None
    assert args.mode == "both"
    assert args.iters == 30
    assert args.warmup == 10
    assert args.dtype == "bf16"
    assert args.device == "cuda"
    assert args.backend is None
    assert args.against is None
    # Its own report base: the scan bench writes under out/bench-op, and one base
    # for both would have each run overwrite the other's reports.
    assert args.out == Path("out/bench-conv")
    assert args.no_ceilings is False
    # --shape appends, so a repeated flag is a list in command-line order.
    assert parse_args(["--shape", "standard", "--shape", "tiny"]).shape == [
        "standard",
        "tiny",
    ]
    # The registry rejects an unknown backend, so a choices list on either name
    # would have to be kept in step with it and would reject 'same' as well.
    named = parse_args(["--backend", "reference", "--against", "same"])
    assert (named.backend, named.against) == ("reference", "same")


def test_parse_args_rejects_a_value_outside_its_table() -> None:
    # argparse exits 2 rather than raising, so a typo in a sweep script stops the
    # run instead of silently benching the default configuration.
    for flag, value in (
        ("--shape", "huge"),
        ("--mode", "backward"),
        ("--dtype", "fp8"),
    ):
        with pytest.raises(SystemExit) as caught:
            parse_args([flag, value])
        assert caught.value.code == 2


# ---------------------------------------------------------------------------
# arm_labels
# ---------------------------------------------------------------------------


def test_arm_labels_name_each_backend_and_separate_the_two_null_arms() -> None:
    assert arm_labels("reference", "native") == ("conv-reference", "conv-native")
    # An unselected backend is the fastest registered one, named auto.
    assert arm_labels(None, "native") == ("conv-auto", "conv-native")
    # measure_paired refuses one label for both arms, and the null test still has
    # to be runnable.
    assert arm_labels("reference", "reference") == (
        "conv-reference-a",
        "conv-reference-b",
    )
    assert arm_labels(None, None) == ("conv-auto-a", "conv-auto-b")


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def test_bench_measures_a_forward_and_holds_no_saved_tensors(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    args = parse_args(_argv(tmp_path / "bench", "--backend", "reference"))
    report, rate = bench(TINY, "forward", args, CUDA, None)
    assert report.title == "bench: conv tiny forward"
    assert report.device is pinned_device
    assert rate.label == "conv tiny forward"
    assert rate.token_count == 256
    assert report.throughput == (rate,)
    assert report.budget is not None
    assert "conv.forward" in report.budget.labels()
    assert "conv.backward" not in report.budget.labels()
    # Nothing runs under grad mode, so there is no graph to probe.
    assert report.saved is None
    assert report.ceilings is None
    assert report.peaks is not None
    assert report.peaks.label == "conv tiny forward"
    assert report.notes == (
        "tiny: B=1 T=256 D=16 W=4",
        "mode=forward dtype=fp32 backend=reference",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    )


@pytest.mark.skipif(
    not _C.is_available(), reason=f"{_C.EXTENSION} is not built; run {_C.BUILD_COMMAND}"
)
def test_bench_measures_a_step_and_probes_what_autograd_holds(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    args = parse_args(_argv(tmp_path / "bench"))
    report, rate = bench(TINY, "step", args, CUDA, None)
    assert rate.label == "conv tiny step"
    assert report.budget is not None
    assert {"conv.forward", "conv.backward"} <= set(report.budget.labels())
    assert report.saved is not None
    assert report.saved.label == "conv tiny"
    # x, weight, bias, and the incoming window; the tap sum and the sigmoid are
    # recomputed in the backward, so a rematerializing backward keeps
    # derived_bytes at zero.
    assert report.saved.storage_count == 4
    assert report.saved.save_event_count == 4
    assert report.saved.derived_bytes == 0
    # The probe runs under a recorder, so no save reads as unattributed.
    assert [r.label for r in report.saved.regions] == ["conv.forward"]
    assert report.notes[1] == "mode=step dtype=fp32 backend=auto"


# ---------------------------------------------------------------------------
# compare_backends
# ---------------------------------------------------------------------------


def test_compare_backends_reports_one_rate_per_arm_from_one_input_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_device: DeviceInfo
) -> None:
    built = _count_input_sets(monkeypatch)
    args = parse_args(_argv(tmp_path / "bench", "--against", "reference"))
    report, row = compare_backends(TINY, "step", args, CUDA, None)
    # Two input sets would differ in address and in cache residency, and that
    # difference would be attributed to the backend.
    assert len(built) == 1
    assert report.title == "bench: conv tiny step paired"
    assert [r.label for r in report.throughput] == ["conv-auto", "conv-reference"]
    assert [r.token_count for r in report.throughput] == [256, 256]
    assert report.comparisons == (row,)
    assert (row.label, row.a_label, row.b_label) == (
        "conv tiny step paired",
        "conv-auto",
        "conv-reference",
    )
    # One sample per arm per iteration, and --iters 2.
    assert row.sample_count == 2
    assert report.budget is not None
    assert {
        "conv-auto.forward",
        "conv-auto.backward",
        "conv-reference.forward",
        "conv-reference.backward",
    } <= set(report.budget.labels())
    assert report.notes == (
        "tiny: B=1 T=256 D=16 W=4",
        "mode=step dtype=fp32",
        "arm a=conv-auto b=conv-reference, one loop, order swapped each iteration",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    )


def test_compare_backends_puts_the_named_backend_in_both_arms_of_a_null_test(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    argv = _argv(tmp_path / "bench", "--backend", "reference", "--against", "same")
    report, row = compare_backends(TINY, "forward", parse_args(argv), CUDA, None)
    assert (row.a_label, row.b_label) == ("conv-reference-a", "conv-reference-b")
    assert report.budget is not None
    labels = set(report.budget.labels())
    assert {"conv-reference-a.forward", "conv-reference-b.forward"} <= labels
    assert not [label for label in labels if label.endswith(".backward")]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_refuses_a_device_no_report_can_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both arcs of the shared guard: a host device, and a CUDA device on a host
    # without CUDA. Every report names the part its numbers came from, and off
    # CUDA there is no such part.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for spec in ("cpu", "cuda"):
        with pytest.raises(RuntimeError, match="is not a usable cuda device"):
            main(["--device", spec, "--out", str(tmp_path / "bench")])


def test_main_prints_one_rate_row_per_configuration_and_writes_both_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pinned_device: DeviceInfo
) -> None:
    out = tmp_path / "bench"
    code = main(_argv(out, "--shape", "tiny", "--mode", "both", "--no-ceilings"))
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        f"wrote {tmp_path / 'bench-tiny-forward.md'}",
        f"wrote {tmp_path / 'bench-tiny-step.md'}",
    ]
    assert lines[2] == ""
    # The header text is pinned in tests/test_perf_report.py; what belongs to the
    # driver is the column width it asked for.
    assert lines[3] == rate_table([], width=20)
    assert [line.split()[0] for line in lines[4:]] == ["tiny/forward", "tiny/step"]
    for mode in ("forward", "step"):
        assert (tmp_path / f"bench-tiny-{mode}.md").is_file()
        assert (tmp_path / f"bench-tiny-{mode}.json").is_file()


def test_main_benches_every_standard_shape_when_none_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pinned_device: DeviceInfo,
) -> None:
    # The standard set is replaced rather than run: 'long' is B=2 T=8192 D=576 on
    # the reference, and the driver's shape default is what is under test.
    monkeypatch.setattr(bench_conv, "CONV_SHAPES", (TINY,))
    code = main(_argv(tmp_path / "bench", "--mode", "forward", "--no-ceilings"))
    assert code == 0
    rows = capsys.readouterr().out.splitlines()[3:]
    assert [line.split()[0] for line in rows] == ["tiny/forward"]


def test_main_measures_the_ceilings_unless_told_not_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_device: DeviceInfo
) -> None:
    seen = _pin_ceilings(monkeypatch)
    measured = tmp_path / "measured" / "bench"
    assert main(_argv(measured, "--shape", "tiny", "--mode", "forward")) == 0
    assert seen == [CUDA]
    text = (tmp_path / "measured" / "bench-tiny-forward.md").read_text()
    # The DRAM ceiling is what the operator's declared class is measured against,
    # so declining it has to drop the section as well as the probe: a ceiling
    # heading over no measurement is a claim about the part that was never made.
    assert "## measured dram ceiling" in text
    seen.clear()
    skipped = tmp_path / "skipped" / "bench"
    argv = _argv(skipped, "--shape", "tiny", "--mode", "forward", "--no-ceilings")
    assert main(argv) == 0
    assert seen == []
    assert "ceiling" not in (tmp_path / "skipped" / "bench-tiny-forward.md").read_text()


def test_the_exit_status_says_whether_a_null_comparison_resolved_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pinned_device: DeviceInfo,
) -> None:
    def run(base: str, *extra: str) -> int:
        argv = _argv(
            tmp_path / base, "--shape", "tiny", "--mode", "forward", "--no-ceilings"
        )
        return main([*argv, *extra])

    # Every run below is handed a resolving verdict, so what the exit status turns
    # on is whether the two arms held the same backend and nothing else.
    _pin_comparison(monkeypatch, a_us=110.0, b_us=100.0)

    # Two distinct backends. A resolved difference there is the result the flag
    # exists to produce, so it is not an error.
    assert run("distinct", "--against", "reference") == 0
    capsys.readouterr()

    # One backend in both arms, named by --against same.
    assert run("resolved", "--against", "same") == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"wrote {tmp_path / 'resolved-tiny-forward-paired.md'}"
    assert lines[2] == rate_table([], width=48)
    assert [line.split()[0] for line in lines[3:5]] == [
        "tiny/forward/conv-auto-a",
        "tiny/forward/conv-auto-b",
    ]
    # The verdict wording is pinned in tests/test_perf_dispersion.py; what belongs
    # to the driver is which label it names and what it concludes from it.
    assert lines[6].startswith("conv tiny forward paired: ")
    assert lines[7] == (
        "both arms ran the same backend and ['conv tiny forward paired'] still "
        "resolve a difference; the comparison is measuring the arm order or the "
        "loop, not the backend"
    )

    # The same backend spelled out twice is the same null test, so it takes the
    # same exit status. Otherwise the check would depend on how it was written.
    spelled = run("spelled", "--backend", "reference", "--against", "reference")
    assert spelled == 1
    assert capsys.readouterr().out.splitlines()[3].split()[0] == (
        "tiny/forward/conv-reference-a"
    )

    # Equal samples resolve nothing, so the harness is intact and the driver says
    # nothing further.
    _pin_comparison(monkeypatch, a_us=100.0, b_us=100.0)
    assert run("null", "--against", "same") == 0
    assert len(capsys.readouterr().out.splitlines()) == 7
