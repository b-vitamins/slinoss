"""The Mamba2 bench driver: its command line, its two group rows, and its output.

``mamba-ssm`` is never imported. ``load_scan`` is the seam: every test that runs
the driver replaces it with a stub taking Mamba2's six positional arguments and
returning ``(batch, seqlen, nheads, headdim)`` differentiable in all five inputs,
which is everything the driver requires of it. The absent-package path is driven
by putting a module with no ``__path__`` under ``mamba_ssm`` in
:data:`sys.modules`, so this file behaves the same whether or not the package is
installed on the host.

Every run is eight tokens, because what is under test is the driver and not either
operator. The driver refuses any device a report cannot name, so every argv names a
CUDA one.

Three more substitutions:

- ``SHAPES`` and ``shape_by_name`` are replaced by :data:`SMALL`, which carries two
  heads. At one head the two group configurations are one configuration, and the
  pair of reported rows is the claim this driver exists to make.
- ``device_info`` is replaced by a fabricated record. Reading it costs two
  ``nvidia-smi`` calls per report and none of them is the driver's behaviour.
- ``clock_policy`` is pinned, so the stamp every measurement carries is a literal.
"""

from __future__ import annotations

import json
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor

from scripts.bench import bench_mamba
from scripts.bench.bench_mamba import (
    compare_so3ssd,
    group_counts,
    load_scan,
    main,
    make_inputs,
    parse_args,
    runner,
)
from slinoss.perf import timing
from slinoss.perf.device import ClockPolicy, Contention, DeviceInfo
from slinoss.perf.report import rate_table
from slinoss.perf.timing import measure
from slinoss.perf.units import Bytes, Count, Mebibytes, Megahertz, Percent
from slinoss.perf.workload import OpShape

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

CUDA = torch.device("cuda")

SMALL = OpShape("small", bsz=1, heads=2, seq=8, rows=16, lanes=16, chunk=4)
"""Two whole chunks at the smallest legal row and lane counts, over two heads."""

ITERS = 2
"""Timed iterations. Even, as the paired loop requires, and the fewest that yield
a dispersion at all."""

DELAY_US = 40_000.0
"""Host delay in the mamba stub for the orientation test. Two orders above the
warm reference at this shape, so the order of the two medians is not a race."""

LEAF = "mamba_ssm.ops.triton.ssd_combined"
"""Module the driver imports the chunked scan from."""


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def scan_stub(*, delay_us: float = 0.0) -> Callable[..., Tensor]:
    """Build a stand-in for ``mamba_chunk_scan_combined``.

    The driver calls it as ``scan(x, dt, A, B, C, chunk)``. Every input is read, so
    ``torch.autograd.grad`` finds all five in the graph.

    Args:
        delay_us: Host delay per call, so one arm of a pair can be made the slower
            of the two.

    Returns:
        The callable. Returns ``(batch, seqlen, nheads, headdim)`` in ``x``'s
        dtype.
    """

    def scan(
        x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk: int
    ) -> Tensor:
        if delay_us > 0.0:
            time.sleep(delay_us / 1e6)
        state = (B * C).sum(dim=(-2, -1))
        y = x * dt.unsqueeze(-1) * A.unsqueeze(-1) + state[..., None, None]
        return y.to(x.dtype)

    return scan


def fabricated_clocks() -> ClockPolicy:
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


def fabricated_device() -> DeviceInfo:
    """A device record no probe was run for."""
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
            probed=False,
            foreign_process_count=Count(0),
            foreign_memory_mib=Mebibytes(0.0),
            utilization_pct=Percent(0.0),
            detail="fabricated",
        ),
    )


def small_by_name(name: str) -> OpShape:
    """The only shape the driver may look up while stubbed."""
    assert name == SMALL.name
    return SMALL


@pytest.fixture(autouse=True)
def pinned_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the clock probe every measurement takes, for every test here."""
    monkeypatch.setattr(timing, "clock_policy", lambda _index: fabricated_clocks())


def install(
    monkeypatch: pytest.MonkeyPatch, scan: Callable[..., Tensor] | None = None
) -> Callable[..., Tensor]:
    """Point the driver at a stub scan and the small shape.

    Args:
        monkeypatch: The patcher.
        scan: Stand-in for ``mamba_chunk_scan_combined``. Built with no delay if
            omitted.

    Returns:
        The installed scan.
    """
    fn = scan_stub() if scan is None else scan
    monkeypatch.setattr(bench_mamba, "load_scan", lambda: fn)
    monkeypatch.setattr(bench_mamba, "device_info", lambda _index: fabricated_device())
    monkeypatch.setattr(bench_mamba, "SHAPES", (SMALL,))
    monkeypatch.setattr(bench_mamba, "shape_by_name", small_by_name)
    return fn


def argv(out: Path, *rest: str, device: str = "cuda", iters: int = ITERS) -> list[str]:
    """The command line every run shares.

    Args:
        out: Directory the reports land in.
        *rest: Extra flags, appended so they override.
        device: ``--device`` value.
        iters: ``--iters`` value.

    Returns:
        The argument vector.
    """
    return [
        "--device",
        device,
        "--dtype",
        "fp32",
        "--iters",
        str(iters),
        "--warmup",
        "0",
        "--out",
        str(out / "bench-mamba"),
        *rest,
    ]


def notes_of(path: Path) -> list[str]:
    """The notes of one written report."""
    return json.loads(path.read_text())["notes"]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_every_shape_both_modes_and_both_group_kinds() -> None:
    args = parse_args([])
    # None for the two repeatable flags, resolved by main to every shape and to
    # both group kinds. A default list would be appended to, not replaced.
    assert args.shape is None
    assert args.groups is None
    assert args.mode == "both"
    assert args.iters == 30
    assert args.warmup == 10
    assert args.dtype == "bf16"
    assert args.device == "cuda"
    assert args.backend is None
    assert args.against_so3ssd is False
    assert args.out == Path("out/bench-mamba")


def test_parse_args_repeats_the_shape_and_the_group_flags() -> None:
    args = parse_args(
        [
            "--shape",
            "tiny",
            "--shape",
            "standard",
            "--groups",
            "one",
            "--groups",
            "heads",
        ]
    )
    assert args.shape == ["tiny", "standard"]
    assert args.groups == ["one", "heads"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--shape", "huge"),
        ("--mode", "backward"),
        ("--groups", "all"),
        ("--dtype", "fp64"),
    ],
)
def test_parse_args_rejects_a_value_outside_the_choices(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as err:
        parse_args([flag, value])
    # Exit 2, so a typo in a bench command cannot quietly bench something else.
    assert err.value.code == 2


# ---------------------------------------------------------------------------
# load_scan
# ---------------------------------------------------------------------------


def test_load_scan_returns_the_mamba2_chunked_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan = scan_stub()
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", LEAF):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(
        sys.modules[LEAF], "mamba_chunk_scan_combined", scan, raising=False
    )
    assert load_scan() is scan


def test_load_scan_names_the_package_instead_of_raising_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A module with no __path__ is not a package, so every submodule import under
    # it fails whether or not mamba-ssm is installed on the host. Any cached
    # submodule would be found without consulting the parent, so they go first.
    for name in ("mamba_ssm.ops", "mamba_ssm.ops.triton", LEAF):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mamba_ssm", types.ModuleType("mamba_ssm"))
    with pytest.raises(SystemExit) as err:
        load_scan()
    # A string exit code prints the message and stops; the import error is kept as
    # the cause rather than reaching the terminal as a traceback.
    assert isinstance(err.value.code, str)
    assert err.value.code.startswith("bench_mamba needs mamba-ssm: ")
    assert isinstance(err.value.__cause__, ImportError)
    assert "mamba_ssm" in str(err.value.__cause__)


# ---------------------------------------------------------------------------
# make_inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groups", [1, 2])
def test_make_inputs_matches_the_mamba2_tensor_contract(groups: int) -> None:
    got = make_inputs(SMALL, groups, CUDA, dtype=torch.float32, requires_grad=False)
    lead = (SMALL.bsz, SMALL.seq)
    assert tuple(got.x.shape) == (*lead, SMALL.heads, SMALL.rows)
    assert tuple(got.dt.shape) == (*lead, SMALL.heads)
    assert tuple(got.A.shape) == (SMALL.heads,)
    # groups=1 shares B and C across heads and moves fewer bytes; groups=heads
    # gives every head its own, which is what the SO(3) operator does.
    assert tuple(got.B.shape) == (*lead, groups, SMALL.d_state)
    assert tuple(got.C.shape) == (*lead, groups, SMALL.d_state)
    assert tuple(got.dy.shape) == (*lead, SMALL.heads, SMALL.rows)
    for t in got:
        assert t.is_contiguous()


def test_make_inputs_pins_dt_and_the_decay_rate_under_a_low_precision_dtype() -> None:
    got = make_inputs(SMALL, 1, CUDA, dtype=torch.bfloat16, requires_grad=False)
    assert got.x.dtype == torch.bfloat16
    assert got.B.dtype == torch.bfloat16
    assert got.C.dtype == torch.bfloat16
    assert got.dy.dtype == torch.bfloat16
    # Mamba2 requires float32 for dt and A whatever x, B, and C are.
    assert got.dt.dtype == torch.float32
    assert got.A.dtype == torch.float32


def test_make_inputs_keeps_the_state_decay_non_positive() -> None:
    # A is a decay rate: positive entries would grow the state without bound and
    # measure a kernel that cannot run in training.
    got = make_inputs(SMALL, 1, CUDA, dtype=torch.float32, requires_grad=False)
    assert torch.all(got.A <= 0.0)


def test_make_inputs_carries_gradients_on_the_five_differentiable_inputs() -> None:
    got = make_inputs(SMALL, 2, CUDA, dtype=torch.float32, requires_grad=True)
    assert got.differentiable == (got.x, got.dt, got.A, got.B, got.C)
    assert all(t.requires_grad for t in got.differentiable)
    # The output-gradient seed is preallocated and is not a graph input, so the
    # backward measurement contains no allocation of its own.
    assert not got.dy.requires_grad


def test_make_inputs_takes_no_gradients_when_asked_not_to() -> None:
    got = make_inputs(SMALL, 2, CUDA, dtype=torch.float32, requires_grad=False)
    assert not any(t.requires_grad for t in got.differentiable)


def test_make_inputs_is_reproducible_from_the_seed() -> None:
    # Two runs of a bench must compare the same numbers, or the delta includes the
    # inputs.
    first = make_inputs(
        SMALL, 2, CUDA, dtype=torch.float32, requires_grad=False, seed=7
    )
    same = make_inputs(SMALL, 2, CUDA, dtype=torch.float32, requires_grad=False, seed=7)
    other = make_inputs(
        SMALL, 2, CUDA, dtype=torch.float32, requires_grad=False, seed=8
    )
    for a, b in zip(first, same):
        assert torch.equal(a, b)
    assert not torch.equal(first.x, other.x)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def test_runner_times_the_forward_alone_without_grads() -> None:
    inputs = make_inputs(SMALL, 2, CUDA, dtype=torch.float32, requires_grad=False)
    timed = measure(
        runner(scan_stub(), inputs, SMALL.chunk, grads=False),
        label="mamba",
        iters=ITERS,
        warmup=0,
        device=CUDA,
    )
    assert [t.label for t in timed.regions] == ["mamba.forward"]
    assert timed.region("mamba.forward").spread.sample_count == ITERS


def test_runner_times_the_forward_and_the_backward_under_grads() -> None:
    inputs = make_inputs(SMALL, 2, CUDA, dtype=torch.float32, requires_grad=True)
    timed = measure(
        runner(scan_stub(), inputs, SMALL.chunk, grads=True),
        label="mamba",
        iters=ITERS,
        warmup=0,
        device=CUDA,
    )
    assert [t.label for t in timed.regions] == ["mamba.forward", "mamba.backward"]
    # torch.autograd.grad, so nothing accumulates into a .grad buffer and no
    # aten::fill_ can contaminate the backward bucket.
    assert all(t.grad is None for t in inputs.differentiable)


def test_runner_takes_a_prefix_so_two_arms_stay_apart() -> None:
    inputs = make_inputs(SMALL, 2, CUDA, dtype=torch.float32, requires_grad=False)
    timed = measure(
        runner(scan_stub(), inputs, SMALL.chunk, grads=False, prefix="mamba-g2"),
        label="mamba",
        iters=1,
        warmup=0,
        device=CUDA,
    )
    assert [t.label for t in timed.regions] == ["mamba-g2.forward"]


# ---------------------------------------------------------------------------
# group_counts
# ---------------------------------------------------------------------------


def test_group_counts_resolve_the_kinds_in_the_order_requested() -> None:
    assert group_counts(SMALL, ["heads", "one"]) == (2, 1)
    assert group_counts(SMALL, ["one", "heads"]) == (1, 2)


def test_group_counts_collapse_to_one_configuration_at_a_single_head() -> None:
    # At heads=1 the two kinds are one configuration. Measuring both would time
    # one thing twice and let the second report overwrite the first.
    one_head = OpShape("one-head", bsz=1, heads=1, seq=8, rows=16, lanes=16, chunk=4)
    assert group_counts(one_head, ["heads", "one"]) == (1,)


# ---------------------------------------------------------------------------
# main: the plain path
# ---------------------------------------------------------------------------


def test_main_reports_each_group_configuration_as_its_own_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    assert main(argv(tmp_path, "--mode", "forward")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"wrote {tmp_path / 'bench-mamba-small-g2-forward.md'}"
    assert lines[1] == f"wrote {tmp_path / 'bench-mamba-small-g1-forward.md'}"
    assert lines[2] == ""
    # The header text is pinned in tests/test_perf_report.py; what belongs to the
    # driver is the column width it asked for.
    assert lines[3] == rate_table([], width=28)
    assert [line.split()[0] for line in lines[4:]] == [
        "small/g2/forward",
        "small/g1/forward",
    ]


def test_main_writes_a_markdown_and_a_json_report_per_group_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    assert main(argv(tmp_path, "--mode", "forward")) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "bench-mamba-small-g1-forward.json",
        "bench-mamba-small-g1-forward.md",
        "bench-mamba-small-g2-forward.json",
        "bench-mamba-small-g2-forward.md",
    ]
    # Two separate reports, each naming its own group count. One file for both
    # would have the second configuration overwrite the first.
    assert [
        notes_of(tmp_path / f"bench-mamba-small-g{count}-forward.json")[1]
        for count in (2, 1)
    ] == [
        "mamba2 ngroups=2 headdim=16 dstate=48",
        "mamba2 ngroups=1 headdim=16 dstate=48",
    ]


def test_main_notes_the_shape_the_group_count_the_mode_and_the_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    assert main(argv(tmp_path, "--mode", "forward", "--groups", "heads")) == 0
    assert notes_of(tmp_path / "bench-mamba-small-g2-forward.json") == [
        "small: B=1 H=2 T=8 P=16 N=16 3N=48 L=4",
        "mamba2 ngroups=2 headdim=16 dstate=48",
        "mode=forward dtype=fp32",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    ]


def test_main_probes_the_saved_set_in_step_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    assert main(argv(tmp_path, "--mode", "step", "--groups", "heads")) == 0
    saved = json.loads((tmp_path / "bench-mamba-small-g2-step.json").read_text())[
        "saved"
    ]
    assert saved["label"] == "mamba small"
    # The probe runs under a recorder, so every save attributes to the region it
    # was taken in instead of reading unattributed.
    labels = [region["label"] for region in saved["regions"]]
    assert labels[0] == "mamba.forward"
    assert "unattributed" not in labels
    assert saved["input_bytes"] > 0


def test_main_omits_the_saved_set_in_forward_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    assert main(argv(tmp_path, "--mode", "forward", "--groups", "one")) == 0
    # A forward builds no graph, so there is no saved set. An absent part prints
    # as absent and never as a zero.
    data = json.loads((tmp_path / "bench-mamba-small-g1-forward.json").read_text())
    assert data["saved"] is None
    text = (tmp_path / "bench-mamba-small-g1-forward.md").read_text()
    assert "## saved tensors" not in text


def test_main_benches_the_shape_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    argument = argv(tmp_path, "--shape", "small", "--mode", "step", "--groups", "one")
    assert main(argument) == 0
    assert (tmp_path / "bench-mamba-small-g1-step.md").exists()


def test_main_measures_both_modes_when_asked_for_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    assert main(argv(tmp_path, "--mode", "both", "--groups", "heads")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[4:]] == [
        "small/g2/forward",
        "small/g2/step",
    ]


def test_main_refuses_a_device_that_is_not_cuda(tmp_path: Path) -> None:
    # Nothing is stubbed here, which is the point: every report names the part the
    # numbers came from and the host is not one. The refusal lands before the
    # mamba-ssm import and before any allocation.
    with pytest.raises(RuntimeError, match="'cpu' is not a usable cuda device"):
        main(argv(tmp_path, device="cpu", iters=1))


def test_main_refuses_cuda_when_cuda_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="'cuda' is not a usable cuda device"):
        main(argv(tmp_path, iters=1))


# ---------------------------------------------------------------------------
# compare_so3ssd
# ---------------------------------------------------------------------------


def test_a_speedup_ratio_above_one_means_the_operator_beat_mamba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = install(monkeypatch, scan_stub(delay_us=DELAY_US))
    # The only test here that warms up. First-call cost on the SO(3) arm is two
    # orders above its warm cost and would decide the order of the medians.
    args = parse_args(argv(tmp_path, "--warmup", "1"))
    _, row = compare_so3ssd(scan, SMALL, SMALL.heads, "forward", args, CUDA)
    # Mamba2 is arm a, so the ratio is mamba over so3ssd: above one is the SO(3)
    # operator winning, and the delay is in the mamba arm.
    assert (row.a_label, row.b_label) == ("mamba-g2", "so3ssd-auto")
    assert row.a_median_duration_us > row.b_median_duration_us
    assert row.speedup_ratio == pytest.approx(
        row.a_median_duration_us / row.b_median_duration_us
    )
    assert row.speedup_ratio > 1.0
    # Two pairs cannot reach nominal coverage, so the row licenses nothing however
    # far apart the medians land.
    assert not row.resolves


def test_compare_so3ssd_names_the_requested_backend_in_the_arm_b_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = install(monkeypatch)
    args = parse_args(argv(tmp_path, "--backend", "reference"))
    _, row = compare_so3ssd(scan, SMALL, 1, "step", args, CUDA)
    assert (row.a_label, row.b_label) == ("mamba-g1", "so3ssd-reference")


def test_compare_so3ssd_reports_a_rate_for_each_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = install(monkeypatch)
    args = parse_args(argv(tmp_path))
    report, row = compare_so3ssd(scan, SMALL, SMALL.heads, "forward", args, CUDA)
    assert report.title == "bench: mamba g2 vs so3ssd small forward paired"
    assert [rate.label for rate in report.throughput] == ["mamba-g2", "so3ssd-auto"]
    assert all(rate.token_count == SMALL.token_count for rate in report.throughput)
    assert report.comparisons == (row,)
    # Two prefixes, so the tree keeps the arms apart. One prefix would sum both
    # forwards into one region describing neither operator.
    assert report.budget is not None
    labels = set(report.budget.labels())
    assert {"mamba-g2.forward", "so3ssd-auto.forward"} <= labels
    assert not [label for label in labels if label.endswith(".backward")]


def test_compare_so3ssd_notes_that_each_arm_holds_its_own_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = install(monkeypatch)
    args = parse_args(argv(tmp_path))
    report, _ = compare_so3ssd(scan, SMALL, SMALL.heads, "forward", args, CUDA)
    # The two operators take different tensors, so the arms cannot share inputs
    # the way two backends of one operator do, and the peak belongs to neither.
    assert report.notes == (
        "small: B=1 H=2 T=8 P=16 N=16 3N=48 L=4",
        "mamba2 ngroups=2 headdim=16 dstate=48",
        "mode=forward dtype=fp32",
        "arm a=mamba-g2 b=so3ssd-auto, one loop, order swapped each iteration",
        "each arm holds its own inputs; the memory peak covers both",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    )


# ---------------------------------------------------------------------------
# main: the paired path against the SO(3) operator
# ---------------------------------------------------------------------------


def test_against_so3ssd_writes_one_paired_report_per_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    status = main(
        argv(
            tmp_path,
            "--against-so3ssd",
            "--mode",
            "both",
            "--groups",
            "heads",
            "--groups",
            "one",
        )
    )
    assert status == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "bench-mamba-small-g1-forward-paired.json",
        "bench-mamba-small-g1-forward-paired.md",
        "bench-mamba-small-g1-step-paired.json",
        "bench-mamba-small-g1-step-paired.md",
        "bench-mamba-small-g2-forward-paired.json",
        "bench-mamba-small-g2-forward-paired.md",
        "bench-mamba-small-g2-step-paired.json",
        "bench-mamba-small-g2-step-paired.md",
    ]
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        f"wrote {tmp_path / 'bench-mamba-small-g2-forward-paired.md'}",
        f"wrote {tmp_path / 'bench-mamba-small-g2-step-paired.md'}",
        f"wrote {tmp_path / 'bench-mamba-small-g1-forward-paired.md'}",
        f"wrote {tmp_path / 'bench-mamba-small-g1-step-paired.md'}",
    ]


def test_against_so3ssd_prints_a_rate_per_arm_and_one_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    argument = argv(
        tmp_path, "--against-so3ssd", "--mode", "forward", "--groups", "one"
    )
    assert main(argument) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"wrote {tmp_path / 'bench-mamba-small-g1-forward-paired.md'}"
    assert lines[1] == ""
    assert lines[2] == rate_table([], width=52)
    assert [line.split()[0] for line in lines[3:5]] == [
        "small/g1/forward/mamba-g1",
        "small/g1/forward/so3ssd-auto",
    ]
    assert lines[5] == ""
    assert lines[6].startswith(
        "mamba g1 vs so3ssd small forward paired: no difference measured between "
        "mamba-g1 and so3ssd-auto; the interval "
    )
    # Two pairs license nothing, so the driver must not print a winner.
    assert "beats" not in lines[6]
    assert len(lines) == 7


def test_against_so3ssd_refuses_an_odd_iteration_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The order of the two arms swaps every iteration, so an odd count leaves one
    # iteration's order unbalanced and the difference carries a position bias.
    install(monkeypatch)
    argument = argv(tmp_path, "--against-so3ssd", "--groups", "one", iters=3)
    with pytest.raises(ValueError, match="needs an even iters"):
        main(argument)
