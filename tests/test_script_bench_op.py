"""The bench driver: argument parsing, arm naming, and the exit status it returns.

Every run is the reference at the smallest standard shape for two iterations,
because what is under test is the driver and not the operator. The driver refuses
any device a report cannot name, so every argv names a CUDA one.

Three collaborators are pinned rather than called. ``device_info`` shells out to
``nvidia-smi`` twice per call, ``clock_policy`` once more per measurement, and
``ceilings`` allocates two 512 MiB buffers and runs an 8192-cube GEMM. None of
them is the driver's own behaviour. The paired verdict is pinned too, in the tests
that assert an exit status: the null test resolves or refuses on the samples it is
given, never on whether the clock cooperated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.bench import bench_op
from scripts.bench.bench_op import (
    arm_labels,
    bench,
    compare_backends,
    main,
    parse_args,
)
from slinoss.perf import arms, timing
from slinoss.perf.arms import op_arm
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
from slinoss.perf.workload import OPS, OpInputs, OpShape

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

CUDA = torch.device("cuda")

TINY = "tiny"
"""The cheapest standard shape, ``B=1 H=1 T=256 P=16 N=16 L=64`` for the scan. Every
family's table holds the name, at the same token count."""

PAIRS = 8
"""Pairs behind a pinned verdict. Eight reaches nominal coverage, so the verdict
turns on whether the interval excludes zero and not on the pair count."""

MEASURE_PAIRED = bench_op.measure_paired
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
    """Pin the report's device record for every entry point that builds one.

    Two ``nvidia-smi`` calls per report is the whole cost of the real one, and the
    record it returns is not the driver's own behaviour.
    """
    info = _device()
    monkeypatch.setattr(bench_op, "device_info", lambda _index: info)
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

    monkeypatch.setattr(bench_op, "ceilings", patched)
    return seen


def _count_input_sets(monkeypatch: pytest.MonkeyPatch) -> list[OpInputs]:
    """Record every scan input set the driver builds.

    Patched where the arm is allocated, since that is the one place a driver
    reaches an operator's inputs.

    Returns:
        The input sets, appended in construction order.
    """
    built: list[OpInputs] = []
    real = arms.make_inputs

    def patched(
        shape: OpShape,
        device: torch.device,
        *,
        dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = True,
        seed: int = 0,
    ) -> OpInputs:
        got = real(shape, device, dtype=dtype, requires_grad=requires_grad, seed=seed)
        built.append(got)
        return got

    monkeypatch.setattr(arms, "make_inputs", patched)
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

    monkeypatch.setattr(bench_op, "measure_paired", patched)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_every_shape_in_both_modes_with_no_comparison() -> None:
    args = parse_args([])
    assert args.op == "so3ssd"
    # None, not a list: main expands it to every standard shape.
    assert args.shape is None
    assert args.mode == "both"
    assert args.iters == 30
    assert args.warmup == 10
    assert args.dtype == "bf16"
    assert args.device == "cuda"
    assert args.backend is None
    assert args.against is None
    assert args.d_head == 0
    assert args.out == Path("out/bench-op")
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
        ("--op", "attention"),
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
    assert arm_labels("so3ssd", "reference", "cute") == (
        "so3ssd-reference",
        "so3ssd-cute",
    )
    # The operator leads, so a report holding two operators' regions still says
    # which arm each row came from.
    assert arm_labels("conv", "reference", "cuda") == ("conv-reference", "conv-cuda")
    # An unselected backend is the fastest registered one, named auto.
    assert arm_labels("so3ssd", None, "cute") == ("so3ssd-auto", "so3ssd-cute")
    # measure_paired refuses one label for both arms, and the null test still has
    # to be runnable.
    assert arm_labels("so3ssd", "reference", "reference") == (
        "so3ssd-reference-a",
        "so3ssd-reference-b",
    )
    assert arm_labels("so3ssd", None, None) == ("so3ssd-auto-a", "so3ssd-auto-b")


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def test_bench_measures_a_forward_and_holds_no_saved_tensors(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    args = parse_args(_argv(tmp_path / "bench", "--backend", "reference"))
    report, rate = bench(TINY, "forward", args, CUDA, None)
    assert report.title == "bench: so3ssd tiny forward"
    assert report.device is pinned_device
    assert rate.label == "so3ssd tiny forward"
    assert rate.token_count == 256
    assert report.throughput == (rate,)
    assert report.budget is not None
    assert "op.forward" in report.budget.labels()
    assert "op.backward" not in report.budget.labels()
    # Nothing runs under grad mode, so there is no graph to probe.
    assert report.saved is None
    assert report.ceilings is None
    assert report.peaks is not None
    assert report.peaks.label == "so3ssd tiny forward"
    assert report.notes == (
        "tiny: B=1 H=1 T=256 P=16 N=16 3N=48 L=64 G=1",
        "mode=forward dtype=fp32 backend=reference",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    )


def test_bench_measures_a_step_and_probes_what_autograd_holds(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    args = parse_args(_argv(tmp_path / "bench"))
    report, rate = bench(TINY, "step", args, CUDA, None)
    assert rate.label == "so3ssd tiny step"
    assert report.budget is not None
    assert {"op.forward", "op.backward"} <= set(report.budget.labels())
    assert report.saved is not None
    assert report.saved.label == "so3ssd tiny"
    # Five tensors per layer with no streaming carry, all of them declared inputs,
    # so a rematerializing backward keeps derived_bytes at zero.
    assert report.saved.storage_count == 5
    assert report.saved.save_event_count == 5
    assert report.saved.derived_bytes == 0
    # The probe runs under a recorder, so no save reads as unattributed.
    assert [r.label for r in report.saved.regions] == ["op.forward"]
    assert report.notes[1] == "mode=step dtype=fp32 backend=auto"


def test_bench_reaches_every_operator_under_its_own_region_prefix(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    """Every operator the profiler drivers dispatch on is also benchable.

    An operator only the profiler can reach is measured once under a profiler and
    never in the cheap loop. Forward only, and one shape: the axis under test is
    the dispatch, and the step arms are the expensive ones.
    """
    prefixes = {
        "so3ssd": "op",
        "conv": "conv",
        "scanprep": "prep",
        "block": "block",
        "mixer": "mixer",
        "xent": "xent",
    }
    assert sorted(prefixes) == sorted(OPS)
    for op in OPS:
        args = parse_args(_argv(tmp_path / "bench", "--op", op))
        report, rate = bench(TINY, "forward", args, CUDA, None)
        assert report.title == f"bench: {op} tiny forward"
        assert rate.label == f"{op} tiny forward"
        # One name per family table, and every family sees the same tokens at it.
        assert rate.token_count == 256
        assert report.notes[0].startswith("tiny: B=1 ")
        assert report.budget is not None
        assert f"{prefixes[op]}.forward" in report.budget.labels()
    # Argparse refuses an unknown name on the command line. The raise is what a
    # caller that is not argparse gets.
    with pytest.raises(ValueError, match="unknown op 'attention'"):
        op_arm("attention", TINY, CUDA, dtype=torch.float32, grads=False)


def test_the_conv_output_layout_reaches_the_arm_and_the_report(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    """A head-major conv is a different measurement, so the report names it.

    The layout changes the conv's store pattern and its wall. Two reports that do
    not say which one ran are not comparable, and the flag is silent on every
    other operator.
    """
    argv = _argv(tmp_path / "bench", "--op", "conv")
    head_major, _rate = bench(
        TINY, "forward", parse_args([*argv, "--d-head", "16"]), CUDA, None
    )
    assert head_major.notes[1] == "mode=forward dtype=fp32 backend=auto d_head=16"
    # Zero is token-major, and the note reads as every earlier report's did.
    token_major, _same = bench(TINY, "forward", parse_args(argv), CUDA, None)
    assert token_major.notes[1] == "mode=forward dtype=fp32 backend=auto"


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
    assert report.title == "bench: so3ssd tiny step paired"
    assert [r.label for r in report.throughput] == ["so3ssd-auto", "so3ssd-reference"]
    assert [r.token_count for r in report.throughput] == [256, 256]
    assert report.comparisons == (row,)
    assert (row.label, row.a_label, row.b_label) == (
        "so3ssd tiny step paired",
        "so3ssd-auto",
        "so3ssd-reference",
    )
    # One sample per arm per iteration, and --iters 2.
    assert row.sample_count == 2
    assert report.budget is not None
    assert {
        "so3ssd-auto.forward",
        "so3ssd-auto.backward",
        "so3ssd-reference.forward",
        "so3ssd-reference.backward",
    } <= set(report.budget.labels())
    assert report.notes == (
        "tiny: B=1 H=1 T=256 P=16 N=16 3N=48 L=64 G=1",
        "mode=step dtype=fp32",
        "arm a=so3ssd-auto b=so3ssd-reference, one loop, order swapped each iteration",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    )


def test_compare_backends_puts_the_named_backend_in_both_arms_of_a_null_test(
    tmp_path: Path, pinned_device: DeviceInfo
) -> None:
    argv = _argv(tmp_path / "bench", "--backend", "reference", "--against", "same")
    report, row = compare_backends(TINY, "forward", parse_args(argv), CUDA, None)
    assert (row.a_label, row.b_label) == ("so3ssd-reference-a", "so3ssd-reference-b")
    assert report.budget is not None
    labels = set(report.budget.labels())
    assert {"so3ssd-reference-a.forward", "so3ssd-reference-b.forward"} <= labels
    assert not [label for label in labels if label.endswith(".backward")]
    assert report.notes[2] == (
        "arm a=so3ssd-reference-a b=so3ssd-reference-b, one loop, "
        "order swapped each iteration"
    )


# ---------------------------------------------------------------------------
# main, without a comparison
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
    # The standard set is replaced rather than run: 'long' is B=2 H=12 T=8192 on
    # the reference, and the driver's shape default is what is under test.
    monkeypatch.setattr(bench_op, "SHAPE_NAMES", (TINY,))
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
    assert "## measured dram ceiling" in text
    assert "## measured tensor ceiling" in text
    # Two 512 MiB buffers and an 8192-cube GEMM per run, so a sweep can decline
    # them. Declining has to drop the section as well as the probe: a ceiling
    # heading over no measurement is a claim about the part that was never made.
    seen.clear()
    skipped = tmp_path / "skipped" / "bench"
    argv = _argv(skipped, "--shape", "tiny", "--mode", "forward", "--no-ceilings")
    assert main(argv) == 0
    assert seen == []
    assert "ceiling" not in (tmp_path / "skipped" / "bench-tiny-forward.md").read_text()


# ---------------------------------------------------------------------------
# main, comparing two arms in one loop
# ---------------------------------------------------------------------------


def test_main_compares_two_backends_in_one_loop_and_claims_nothing_of_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pinned_device: DeviceInfo
) -> None:
    out = tmp_path / "bench"
    code = main(
        _argv(
            out,
            "--shape",
            "tiny",
            "--mode",
            "forward",
            "--no-ceilings",
            "--against",
            "reference",
        )
    )
    # Two distinct backends, so the exit status carries no verdict either way.
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"wrote {tmp_path / 'bench-tiny-forward-paired.md'}"
    assert lines[1] == ""
    assert lines[2] == rate_table([], width=48)
    assert [line.split()[0] for line in lines[3:5]] == [
        "tiny/forward/so3ssd-auto",
        "tiny/forward/so3ssd-reference",
    ]
    assert lines[5] == ""
    assert lines[6].startswith("so3ssd tiny forward paired: ")
    assert len(lines) == 7
    assert (tmp_path / "bench-tiny-forward-paired.json").is_file()


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

    # One backend in both arms and a resolved difference: the comparison measured
    # the arm order or the loop.
    _pin_comparison(monkeypatch, a_us=110.0, b_us=100.0)
    assert run("resolved", "--against", "same") == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[6] == (
        "so3ssd tiny forward paired: so3ssd-auto-b beats so3ssd-auto-a by "
        "10.000 us (9.091%, speedup_ratio 1.100); the interval "
        "[-10.000, -10.000] us at 99.219% coverage over 8 pairs excludes zero"
    )
    assert lines[7] == (
        "both arms ran the same backend and ['so3ssd tiny forward paired'] still "
        "resolve a difference; the comparison is measuring the arm order or the "
        "loop, not the backend"
    )

    # Equal samples resolve nothing, so the harness is intact and the driver says
    # nothing further.
    _pin_comparison(monkeypatch, a_us=100.0, b_us=100.0)
    assert run("null", "--against", "same") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[6] == (
        "so3ssd tiny forward paired: no difference measured between so3ssd-auto-a "
        "and so3ssd-auto-b; the interval [0.000, 0.000] us at 99.219% coverage "
        "over 8 pairs does not exclude zero"
    )
    assert len(lines) == 7

    # Naming the same backend twice is the same null test as 'same', so it holds
    # the same exit status.
    _pin_comparison(monkeypatch, a_us=110.0, b_us=100.0)
    assert run("named", "--backend", "reference", "--against", "reference") == 1
    assert (
        capsys.readouterr()
        .out.splitlines()[-1]
        .startswith(
            "both arms ran the same backend and ['so3ssd tiny forward paired'] still "
        )
    )
