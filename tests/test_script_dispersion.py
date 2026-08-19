"""The dispersion driver: its command line, its printed table, and its exit code.

This pins ``scripts.perf.dispersion``, the driver. The library module it calls,
:mod:`slinoss.perf.dispersion`, is pinned by ``tests/test_perf_dispersion.py``.

Every run measures the smallest standard shape in float32 for one iteration,
because what is under test is the driver and not the operator. The driver refuses
any device a report cannot name, so every argv names a CUDA one.

No assertion reads a clock. ``growth`` and ``repeats`` are module attributes of the
driver, so a stub returns literal rows and every printed figure and both exit codes
are exact. Letting real timing scatter decide ``floor_holds`` would decide the exit
code by jitter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch

from scripts.perf import dispersion
from scripts.perf.dispersion import DTYPES, MODES, main, parse_args
from slinoss.perf import timing
from slinoss.perf.device import ClockPolicy, Contention, DeviceInfo
from slinoss.perf.dispersion import GrowthRow, RepeatRow
from slinoss.perf.units import (
    CONFIDENCE_PCT,
    MIN_RESOLVING_SAMPLES,
    Bytes,
    Count,
    Mebibytes,
    Megahertz,
    Microseconds,
    Percent,
    Spread,
)
from slinoss.perf.workload import SHAPES

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

HEADER = "sample_count    median_us  spread_pct  resolution_pct  coverage_pct  resolves"
"""The growth table's header row, verbatim. Six right-aligned columns, 77 wide."""


def argv(out: Path, *extra: str) -> list[str]:
    """The cheapest legal command line, with ``extra`` appended.

    One iteration of the smallest standard shape in float32, two runs, and a
    stride of one. Two runs is the fewest :func:`repeats` accepts
    and one is the smallest stride :func:`growth` accepts. Appended flags override
    the ones here, since argparse takes the last occurrence.

    Args:
        out: Report base path. The driver appends ``.md`` and ``.json`` to its name.
        extra: Flags appended verbatim.

    Returns:
        The argument list.
    """
    return [
        "--shape",
        "tiny",
        "--mode",
        "step",
        "--dtype",
        "fp32",
        "--device",
        "cuda",
        "--iters",
        "1",
        "--warmup",
        "0",
        "--stride",
        "1",
        "--repeat",
        "2",
        "--out",
        str(out),
        *extra,
    ]


def notes_of(base: Path) -> list[str]:
    """The note lines of the markdown report written at ``base``.

    Args:
        base: The ``--out`` path handed to the driver.

    Returns:
        One list entry per note, each still carrying its markdown bullet.
    """
    text = base.with_name(base.name + ".md").read_text()
    return text.split("## notes\n\n")[1].splitlines()


def fabricated_clocks() -> ClockPolicy:
    """A pinned clock, so the stamp in the notes is a literal.

    Locking is denied on the verification fleet, so a real probe always stamps
    ``unlocked``. Claiming the opposite here is what proves the note carries the
    policy it was handed rather than a default.
    """
    return ClockPolicy(
        locked=True,
        sm_clock_mhz=Megahertz(1740.0),
        max_sm_clock_mhz=Megahertz(1800.0),
        detail="fabricated",
    )


def fabricated_device() -> DeviceInfo:
    """A fixed device record, so no probe runs and the report header is constant."""
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
        clocks=fabricated_clocks(),
        sharing=Contention(
            probed=True,
            foreign_process_count=Count(0),
            foreign_memory_mib=Mebibytes(0.0),
            utilization_pct=Percent(0.0),
            detail="fabricated",
        ),
    )


@pytest.fixture(autouse=True)
def stub_device_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both ``nvidia-smi`` probes for every test in this module.

    The device record costs two subprocess calls, and ``measure`` probes the clock
    once per run on top of that. Neither reading is the driver's own behaviour, and
    fixing both is what makes the report header and the notes literals.
    """
    monkeypatch.setattr(dispersion, "device_info", lambda index: fabricated_device())
    monkeypatch.setattr(timing, "clock_policy", lambda index: fabricated_clocks())


def growth_row(*, sample_count: int, resolves: bool) -> GrowthRow:
    """One prefix summary with literal figures, so the printed row is exact."""
    return GrowthRow(
        sample_count=Count(sample_count),
        median_duration_us=Microseconds(1234.5),
        min_duration_us=Microseconds(1200.0),
        max_duration_us=Microseconds(1300.0),
        spread_pct=Percent(8.1),
        resolution_pct=Percent(0.25),
        coverage_pct=Percent(96.875),
        resolves=resolves,
    )


def growth_stub(
    rows: tuple[GrowthRow, ...],
) -> Callable[[Sequence[Microseconds], int], tuple[GrowthRow, ...]]:
    """A ``growth`` that ignores the samples and returns ``rows``."""

    def stub(samples: Sequence[Microseconds], stride: int) -> tuple[GrowthRow, ...]:
        del samples, stride
        return rows

    return stub


def repeat_row(*, scatter_pct: float, floor_holds: bool) -> RepeatRow:
    """A scatter row with literal figures. Two runs of eight samples.

    The floor is 2 percent, so the budget is 4 and the row's own verdict is
    consistent with its scatter at 3 percent and at 9.

    Args:
        scatter_pct: Range of the per-run medians over their median.
        floor_holds: The verdict the driver branches on.

    Returns:
        The row.
    """
    return RepeatRow(
        label="so3ssd tiny step",
        run_count=Count(2),
        sample_count=Count(8),
        median_duration_us=Microseconds(1234.5),
        min_duration_us=Microseconds(1230.0),
        max_duration_us=Microseconds(1240.0),
        scatter_pct=Percent(scatter_pct),
        floor_pct=Percent(2.0),
        coverage_pct=Percent(96.875),
        spread_pct=Percent(5.0),
        floor_holds=floor_holds,
    )


def repeats_stub(row: RepeatRow) -> Callable[[str, Sequence[Spread]], RepeatRow]:
    """A ``repeats`` that ignores the measured runs and returns ``row``."""

    def stub(label: str, runs: Sequence[Spread]) -> RepeatRow:
        del label, runs
        return row

    return stub


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_the_study_and_takes_every_flag() -> None:
    default = parse_args([])
    assert default.shape == "standard"
    assert default.mode == "step"
    assert default.iters == 30
    assert default.warmup == 10
    assert default.stride == 5
    assert default.repeat == 5
    assert default.dtype == "bf16"
    assert default.device == "cuda"
    assert default.backend is None
    assert default.out == Path("out/dispersion")
    args = parse_args(
        [
            "--shape",
            "ragged",
            "--mode",
            "forward",
            "--iters",
            "7",
            "--warmup",
            "3",
            "--stride",
            "2",
            "--repeat",
            "4",
            "--dtype",
            "fp16",
            "--device",
            "cuda:1",
            "--backend",
            "reference",
            "--out",
            "here/there",
        ]
    )
    assert args.shape == "ragged"
    assert args.mode == "forward"
    assert args.iters == 7
    assert args.warmup == 3
    assert args.stride == 2
    assert args.repeat == 4
    assert args.dtype == "fp16"
    assert args.device == "cuda:1"
    assert args.backend == "reference"
    assert args.out == Path("here/there")


def test_every_choice_the_driver_offers_selects_something() -> None:
    for shape in SHAPES:
        assert parse_args(["--shape", shape.name]).shape == shape.name
    for name in DTYPES:
        assert parse_args(["--dtype", name]).dtype == name
    for name in MODES:
        assert parse_args(["--mode", name]).mode == name
    # The choices are the table's keys, sorted, so a name the driver offers can
    # never miss a dtype to select.
    assert sorted(DTYPES) == ["bf16", "fp16", "fp32"]
    assert DTYPES["bf16"] == torch.bfloat16
    assert DTYPES["fp16"] == torch.float16
    assert DTYPES["fp32"] == torch.float32
    assert MODES == ("forward", "step")
    for flag in ("--shape", "--mode", "--dtype"):
        with pytest.raises(SystemExit) as caught:
            parse_args([flag, "nonesuch"])
        assert caught.value.code == 2


# ---------------------------------------------------------------------------
# main: the guards
# ---------------------------------------------------------------------------


def test_main_refuses_a_device_no_report_can_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Both arcs of the shared guard: a host device, and a CUDA device on a host
    # without CUDA. Forced, so the refusal is reached on a GPU host too.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for spec in ("cpu", "cuda"):
        with pytest.raises(RuntimeError, match="is not a usable cuda device"):
            main(["--device", spec, "--out", str(tmp_path / "dispersion")])


def test_main_refuses_a_study_shape_no_statistic_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Both statistics reject these, but only after every run has been measured.
    # A stub on either would hide the guard, so neither is patched: the refusal
    # has to arrive before the measurement to be observable at all.
    def unreachable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("the study ran before its shape was checked")

    monkeypatch.setattr(dispersion, "measure", unreachable)
    for flag, value, message in (
        ("--repeat", "1", "--repeat needs at least two runs, got 1"),
        ("--stride", "0", "--stride must be positive, got 0"),
        ("--stride", "-1", "--stride must be positive, got -1"),
    ):
        with pytest.raises(ValueError, match=message):
            main(argv(tmp_path / "dispersion", flag, value))


# ---------------------------------------------------------------------------
# main: the report
# ---------------------------------------------------------------------------


def test_main_writes_the_markdown_and_the_json_and_says_where(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "nested" / "dispersion"
    main(argv(base))
    md = tmp_path / "nested" / "dispersion.md"
    assert md.exists()
    assert (tmp_path / "nested" / "dispersion.json").exists()
    assert capsys.readouterr().out.splitlines()[0] == f"wrote {md}"
    assert md.read_text().splitlines()[0] == "# dispersion: so3ssd tiny step"
    assert notes_of(base) == [
        "- tiny: B=1 H=1 T=256 P=16 N=16 3N=48 L=64",
        "- mode=step dtype=fp32 backend=default",
        "- iters=1 warmup=0 repeat=2",
        # The stamp is the policy the run was handed, not the state of this host.
        "- timer=cuda_event clocks=locked at 1740 MHz",
        f"- confidence_pct={CONFIDENCE_PCT:.1f} "
        f"min_resolving_samples={MIN_RESOLVING_SAMPLES}",
        "- growth rows are prefixes of run 0 and share their samples",
    ]


def test_the_mode_and_the_backend_reach_the_title_and_the_notes(tmp_path: Path) -> None:
    base = tmp_path / "dispersion"
    main(argv(base, "--mode", "forward", "--backend", "reference"))
    text = base.with_name("dispersion.md").read_text()
    assert text.splitlines()[0] == "# dispersion: so3ssd tiny forward"
    assert "- mode=forward dtype=fp32 backend=reference" in notes_of(base)


# ---------------------------------------------------------------------------
# main: the printed table
# ---------------------------------------------------------------------------


def test_each_growth_row_prints_its_prefix_under_the_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = (
        growth_row(sample_count=3, resolves=False),
        growth_row(sample_count=6, resolves=True),
    )
    monkeypatch.setattr(dispersion, "growth", growth_stub(rows))
    main(argv(tmp_path / "dispersion"))
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == ""
    assert lines[2] == HEADER
    assert lines[2].split() == [
        "sample_count",
        "median_us",
        "spread_pct",
        "resolution_pct",
        "coverage_pct",
        "resolves",
    ]
    assert lines[3].split() == ["3", "1,234.500", "8.100", "0.250", "96.875", "no"]
    assert lines[4].split() == ["6", "1,234.500", "8.100", "0.250", "96.875", "yes"]
    # Every field is right-aligned in its own column, so a row is exactly as wide as
    # the header and the table cannot shear.
    assert [len(line) for line in lines[2:5]] == [len(HEADER)] * 3


def test_the_scatter_line_states_the_scatter_the_floor_and_the_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    row = repeat_row(scatter_pct=3.0, floor_holds=True)
    monkeypatch.setattr(dispersion, "repeats", repeats_stub(row))
    main(argv(tmp_path / "dispersion", "--repeat", "3"))
    lines = capsys.readouterr().out.splitlines()
    # One per-run median line follows it, for each of the three runs.
    assert lines[-4] == (
        "2 runs of 8 samples: median-to-median scatter 3.000%, floor 2.000% at "
        "96.875% coverage, budget 4.000%, widest range 5.000%"
    )
    # Those medians are measured, so the assertion is on the labels and the unit.
    runs = lines[-3:]
    assert [line.split(":")[0] for line in runs] == ["  run 0", "  run 1", "  run 2"]
    assert all(line.endswith(" us") for line in runs)


# ---------------------------------------------------------------------------
# main: the exit code
# ---------------------------------------------------------------------------


def test_the_exit_code_follows_the_floor_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    failing = repeat_row(scatter_pct=9.0, floor_holds=False)
    monkeypatch.setattr(dispersion, "repeats", repeats_stub(failing))
    assert main(argv(tmp_path / "failing")) == 1
    assert capsys.readouterr().out.splitlines()[-1] == (
        "floor 2.000% at 96.875% coverage does not cover the observed scatter "
        "9.000%; no delta measured against it is a result"
    )
    holding = repeat_row(scatter_pct=3.0, floor_holds=True)
    monkeypatch.setattr(dispersion, "repeats", repeats_stub(holding))
    assert main(argv(tmp_path / "holding")) == 0
    assert "does not cover" not in capsys.readouterr().out
