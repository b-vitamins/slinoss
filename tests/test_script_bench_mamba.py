"""The Mamba2 bench driver: its command line, its two group rows, and its output.

No test of the driver imports ``mamba-ssm``. ``load_scan`` is one seam: every test
that runs the driver replaces it with a stub taking Mamba2's six positional
arguments and returning ``(batch, seqlen, nheads, headdim)`` differentiable in all
five inputs, which is everything the driver requires of it. ``parameter_counts``
is the other, replaced by :func:`fabricated_params`, since counting a parameter
builds the module. The absent-package path is driven by putting a module with no
``__path__`` under ``mamba_ssm`` in :data:`sys.modules`, so this file behaves the
same whether or not the package is installed on the host. One test outside the
driver path counts the real parameters and skips when the package is absent.

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
    Parameters,
    chunk_variants,
    compare_so3ssd,
    group_counts,
    load_scan,
    main,
    make_inputs,
    mamba_arithmetic,
    mamba_chunk,
    mamba_tag,
    mapping_of,
    parameter_counts,
    parse_args,
    runner,
    seq_variants,
    so3ssd_arithmetic,
    unbuilt_stage_blocker,
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

PARAMS = OpShape("params", bsz=1, heads=2, seq=32, rows=16, lanes=16, chunk=16)
"""The same layer at the shortest chunk a mixer config admits. :data:`SMALL` is
four, which is under that floor, and a parameter count needs a config."""

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


def fabricated_params(shape: OpShape, groups: int) -> tuple[Parameters, Parameters]:
    """Parameter counts no module was built for.

    The real count instantiates both layers, which imports ``mamba_ssm.modules`` and
    is what :func:`test_the_parameter_counts_come_from_the_two_shipped_modules`
    covers. Every driver test replaces it, so the seam holds and the printed block is
    a literal.
    """
    del shape, groups
    return (Parameters("mamba2-stub", 11), Parameters("slinoss-stub", 13))


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
    monkeypatch.setattr(bench_mamba, "parameter_counts", fabricated_params)
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
    assert args.seq is None
    assert args.mode == "both"
    assert args.iters == 30
    assert args.warmup == 10
    assert args.dtype == "bf16"
    assert args.device == "cuda"
    assert args.backend is None
    assert args.against_so3ssd is False
    assert args.out == Path("out/bench-mamba")
    repeated = parse_args(
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
    assert repeated.shape == ["tiny", "standard"]
    assert repeated.groups == ["one", "heads"]
    assert parse_args(["--seq", "512", "--seq", "1024"]).seq == [512, 1024]


def test_parse_args_rejects_a_value_outside_the_choices() -> None:
    # Exit 2, so a typo in a bench command cannot quietly bench something else.
    for flag, value in (
        ("--shape", "huge"),
        ("--mode", "backward"),
        ("--groups", "all"),
        ("--dtype", "fp64"),
    ):
        with pytest.raises(SystemExit) as err:
            parse_args([flag, value])
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


def test_make_inputs_matches_the_mamba2_tensor_contract() -> None:
    lead = (SMALL.bsz, SMALL.seq)
    # groups=1 shares B and C across heads and moves fewer bytes; groups=heads
    # gives every head its own, which is what the SO(3) operator does.
    for groups in (1, 2):
        got = make_inputs(SMALL, groups, CUDA, dtype=torch.float32, requires_grad=False)
        assert tuple(got.x.shape) == (*lead, SMALL.heads, SMALL.rows)
        assert tuple(got.dt.shape) == (*lead, SMALL.heads)
        assert tuple(got.A.shape) == (SMALL.heads,)
        assert tuple(got.B.shape) == (*lead, groups, SMALL.d_state)
        assert tuple(got.C.shape) == (*lead, groups, SMALL.d_state)
        assert tuple(got.dy.shape) == (*lead, SMALL.heads, SMALL.rows)
        for t in got:
            assert t.is_contiguous()
    low = make_inputs(SMALL, 1, CUDA, dtype=torch.bfloat16, requires_grad=False)
    assert low.x.dtype == torch.bfloat16
    assert low.B.dtype == torch.bfloat16
    assert low.C.dtype == torch.bfloat16
    assert low.dy.dtype == torch.bfloat16
    # Mamba2 requires float32 for dt and A whatever x, B, and C are.
    assert low.dt.dtype == torch.float32
    assert low.A.dtype == torch.float32


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
    plain = make_inputs(SMALL, 2, CUDA, dtype=torch.float32, requires_grad=False)
    assert not any(t.requires_grad for t in plain.differentiable)


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
    # A prefix renames the region, so one loop can hold two arms.
    prefixed = measure(
        runner(scan_stub(), inputs, SMALL.chunk, grads=False, prefix="mamba-g2"),
        label="mamba",
        iters=1,
        warmup=0,
        device=CUDA,
    )
    assert [t.label for t in prefixed.regions] == ["mamba-g2.forward"]


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


# ---------------------------------------------------------------------------
# group_counts
# ---------------------------------------------------------------------------


def test_group_counts_resolve_the_kinds_in_the_order_requested() -> None:
    assert group_counts(SMALL, ["heads", "one"]) == (2, 1)
    assert group_counts(SMALL, ["one", "heads"]) == (1, 2)
    # At heads=1 the two kinds are one configuration. Measuring both would time
    # one thing twice and let the second report overwrite the first.
    one_head = OpShape("one-head", bsz=1, heads=1, seq=8, rows=16, lanes=16, chunk=4)
    assert group_counts(one_head, ["heads", "one"]) == (1,)


# ---------------------------------------------------------------------------
# seq_variants
# ---------------------------------------------------------------------------


def test_seq_variants_move_only_the_sequence_length() -> None:
    # No lengths means the shape's own, so the default command is unchanged.
    assert seq_variants(SMALL, ()) == (SMALL,)
    variants = seq_variants(SMALL, [4, 16])
    assert [v.seq for v in variants] == [4, 16]
    # T reaches the name, so two lengths write two reports instead of one
    # overwriting the other.
    assert [v.name for v in variants] == ["small-t4", "small-t16"]
    # Everything the ratio depends on besides T is held, or the sweep would be a
    # sweep over geometries.
    for got in variants:
        assert (got.bsz, got.heads, got.rows, got.lanes, got.chunk, got.groups) == (
            SMALL.bsz,
            SMALL.heads,
            SMALL.rows,
            SMALL.lanes,
            SMALL.chunk,
            SMALL.groups,
        )
    with pytest.raises(ValueError, match="must be positive"):
        seq_variants(SMALL, [0])


# ---------------------------------------------------------------------------
# chunk_variants and the per-arm chunk
# ---------------------------------------------------------------------------


def test_chunk_variants_move_only_the_chunk_length() -> None:
    # No lengths means the shape's own, so the default command is unchanged.
    assert chunk_variants(SMALL, ()) == (SMALL,)
    variants = chunk_variants(SMALL, [4, 16])
    assert [v.chunk for v in variants] == [4, 16]
    # L reaches the name, so two lengths write two reports.
    assert [v.name for v in variants] == ["small-l4", "small-l16"]
    for got in variants:
        assert (got.bsz, got.heads, got.seq, got.rows, got.lanes, got.groups) == (
            SMALL.bsz,
            SMALL.heads,
            SMALL.seq,
            SMALL.rows,
            SMALL.lanes,
            SMALL.groups,
        )
    # A non-power-of-two is refused here rather than after the inputs are allocated.
    with pytest.raises(ValueError, match="power of two"):
        chunk_variants(SMALL, [48])


def test_the_mamba_arm_takes_its_own_chunk_and_says_so_only_when_it_differs() -> None:
    assert mamba_chunk(SMALL, None) == SMALL.chunk
    assert mamba_chunk(SMALL, 256) == 256
    with pytest.raises(ValueError, match="power of two"):
        mamba_chunk(SMALL, 48)
    # Iso-chunk is the matched run, and its labels are what they were before per-arm
    # tiling existed. A differing length has to reach the label, or two runs that
    # differ only in it would collide on one report name.
    assert mamba_tag(SMALL, SMALL.chunk) == ""
    assert mamba_tag(SMALL, 256) == "-l256"


def test_the_mamba_flop_follows_the_arm_that_chunk_was_given_to() -> None:
    # The score and the diagonal terms are linear in the chunk, so a sweep that left
    # the count at the shape's own default would report one flop for four tilings.
    own = mamba_arithmetic(SMALL, SMALL.heads)
    assert mamba_arithmetic(SMALL, SMALL.heads, SMALL.chunk) == own
    assert (
        mamba_arithmetic(SMALL, SMALL.heads, 2 * SMALL.chunk).step_flop > own.step_flop
    )
    # The mapping states both tilings, because the ratio is read off that line.
    assert (
        "chunk_size=8 against L=4, per-arm tiling" in mapping_of(SMALL, 1, 8).describe()
    )


def test_main_holds_one_shape_at_each_requested_sequence_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    argument = argv(tmp_path, "--mode", "forward", "--groups", "one", "--seq", "4")
    assert main([*argument, "--seq", "8"]) == 0
    assert capsys.readouterr().out.splitlines()[:2] == [
        f"wrote {tmp_path / 'bench-mamba-small-t4-g1-forward.md'}",
        f"wrote {tmp_path / 'bench-mamba-small-t8-g1-forward.md'}",
    ]
    assert notes_of(tmp_path / "bench-mamba-small-t4-g1-forward.json")[0] == (
        "small-t4: B=1 H=2 T=4 P=16 N=16 3N=48 L=4 G=1"
    )


# ---------------------------------------------------------------------------
# main: the plain path
# ---------------------------------------------------------------------------


def test_main_reports_each_group_configuration_as_its_own_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    both = tmp_path / "both"
    assert main(argv(both, "--mode", "forward")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"wrote {both / 'bench-mamba-small-g2-forward.md'}"
    assert lines[1] == f"wrote {both / 'bench-mamba-small-g1-forward.md'}"
    assert lines[2] == ""
    # The header text is pinned in tests/test_perf_report.py; what belongs to the
    # driver is the column width it asked for.
    assert lines[3] == rate_table([], width=28)
    assert [line.split()[0] for line in lines[4:]] == [
        "small/g2/forward",
        "small/g1/forward",
    ]
    # Two configurations, two pairs of files. One base name for both would have the
    # second configuration overwrite the first.
    assert sorted(p.name for p in both.iterdir()) == [
        "bench-mamba-small-g1-forward.json",
        "bench-mamba-small-g1-forward.md",
        "bench-mamba-small-g2-forward.json",
        "bench-mamba-small-g2-forward.md",
    ]
    assert notes_of(both / "bench-mamba-small-g2-forward.json") == [
        "small: B=1 H=2 T=8 P=16 N=16 3N=48 L=4 G=1",
        "mamba2 ngroups=2 headdim=16 dstate=48",
        "mode=forward dtype=fp32",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    ]
    assert (
        notes_of(both / "bench-mamba-small-g1-forward.json")[1]
        == "mamba2 ngroups=1 headdim=16 dstate=48"
    )
    # A named shape is resolved through shape_by_name instead of the default set.
    named = tmp_path / "named"
    assert (
        main(argv(named, "--shape", "small", "--mode", "step", "--groups", "one")) == 0
    )
    assert (named / "bench-mamba-small-g1-step.md").exists()


def test_the_saved_set_is_probed_in_step_mode_and_absent_in_forward_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    assert main(argv(tmp_path, "--mode", "both", "--groups", "heads")) == 0
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()[4:]] == [
        "small/g2/forward",
        "small/g2/step",
    ]
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
    # A forward builds no graph, so there is no saved set. An absent part prints
    # as absent and never as a zero.
    data = json.loads((tmp_path / "bench-mamba-small-g2-forward.json").read_text())
    assert data["saved"] is None
    text = (tmp_path / "bench-mamba-small-g2-forward.md").read_text()
    assert "## saved tensors" not in text


def test_main_refuses_a_device_no_report_can_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing is stubbed, which is the point: every report names the part the
    # numbers came from and the host is not one. Both arcs of the guard land before
    # the mamba-ssm import and before any allocation.
    with pytest.raises(RuntimeError, match="'cpu' is not a usable cuda device"):
        main(argv(tmp_path, device="cpu", iters=1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="'cuda' is not a usable cuda device"):
        main(argv(tmp_path, iters=1))


def test_a_stage_with_no_kernel_refuses_the_comparison() -> None:
    # The failure mode this guards is silent: every stage has a reference path, so an
    # unbuilt extension produces a slower arm and not an error, and the ratio is then
    # a number about the build. Only the scan and the convolution have host kernels
    # to be missing, so the host path names the other two and no more.
    host = unbuilt_stage_blocker(torch.device("cpu"))
    assert host is not None
    assert "build_ext" in host
    assert unbuilt_stage_blocker(torch.device("cuda")) is None


# ---------------------------------------------------------------------------
# What the comparison holds equal
# ---------------------------------------------------------------------------


def test_the_mapping_equalizes_the_state_elements_a_head_carries() -> None:
    mapped = mapping_of(SMALL, 2)
    assert (mapped.headdim, mapped.dstate, mapped.ngroups, mapped.chunk) == (
        16,
        48,
        2,
        4,
    )
    # dstate is 3N and not N. Matching the lane count instead hands Mamba2 a third
    # of the state, and a ratio measured against that is an artefact of the mapping.
    assert mapped.dstate == SMALL.d_state == 3 * SMALL.lanes
    assert mapped.state_elems == SMALL.rows * SMALL.d_state
    # The group count moves the score's cost, not the state a head carries.
    assert mapping_of(SMALL, 1).state_elems == mapped.state_elems


def test_the_counted_flop_of_both_sides_puts_the_so3_operator_above_mamba() -> None:
    theirs = mamba_arithmetic(SMALL, SMALL.heads)
    ours = so3ssd_arithmetic(SMALL)
    assert (theirs.label, ours.label) == ("mamba-g2", "so3ssd")
    for count in (theirs, ours):
        assert count.step_flop == count.forward_flop + count.backward_flop
        # Both sides recompute in the backward, so it is the larger of the two.
        assert count.backward_flop > count.forward_flop
    # At equal state per head the SO(3) operator does the greater arithmetic: its
    # score is per head where Mamba2 shares one across a group, and its forcing has
    # two taps. The mapping is not neutral, and the counts state which way.
    assert ours.step_flop > theirs.step_flop
    # Only the score is shared across a group, so the group count moves their total
    # and nothing else does.
    assert mamba_arithmetic(SMALL, 1).step_flop < theirs.step_flop


def test_the_parameter_counts_come_from_the_two_shipped_modules() -> None:
    pytest.importorskip("mamba_ssm.modules.mamba2")
    # Neither operator holds a parameter: both take tensors a projection produced.
    # The comparable count is the layer's, so both layers are built, and built on
    # the host: building on the device under measurement would leave two layers of
    # parameters in its allocator.
    before = torch.cuda.memory_allocated()
    theirs, ours = parameter_counts(PARAMS, 1)
    assert torch.cuda.memory_allocated() == before
    assert (theirs.label, ours.label) == ("mamba2-g1", "slinoss-mixer")
    assert theirs.elements > 0
    assert ours.elements > 0
    # Both layers project B and C per group, so a matched run compares two counts
    # that moved together.
    wider_theirs, wider_ours = parameter_counts(PARAMS, 2)
    assert wider_theirs.elements > theirs.elements
    assert wider_ours.elements > ours.elements


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
    _, face = compare_so3ssd(scan, SMALL, SMALL.heads, "forward", args, CUDA)
    row = face.row
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
    # The arm medians are read out of the same loop as the verdict, so the slower
    # arm is the slower arm in both.
    mamba, ours = face.arms
    assert (mamba.label, ours.label) == (row.a_label, row.b_label)
    assert mamba.total_us > ours.total_us
    assert 0.0 < mamba.forward_us <= mamba.total_us
    # A forward builds no graph, so neither arm has a backward region to read.
    assert mamba.backward_us is None
    assert ours.backward_us is None


def test_compare_so3ssd_reports_a_rate_and_a_region_for_each_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = install(monkeypatch)
    args = parse_args(argv(tmp_path))
    report, face = compare_so3ssd(scan, SMALL, SMALL.heads, "forward", args, CUDA)
    assert report.title == "bench: mamba g2 vs so3ssd small forward paired"
    assert [rate.label for rate in report.throughput] == ["mamba-g2", "so3ssd-auto"]
    assert all(rate.token_count == SMALL.token_count for rate in report.throughput)
    assert report.comparisons == (face.row,)
    # Two prefixes, so the tree keeps the arms apart. One prefix would sum both
    # forwards into one region describing neither operator.
    assert report.budget is not None
    labels = set(report.budget.labels())
    assert {"mamba-g2.forward", "so3ssd-auto.forward"} <= labels
    assert not [label for label in labels if label.endswith(".backward")]
    notes = report.notes
    assert notes[0] == "small: B=1 H=2 T=8 P=16 N=16 3N=48 L=4 G=1"
    # What was held equal is in the notes, so the ratio can be read against the
    # geometry, the arithmetic and the parameter counts that produced it.
    assert notes[1:7] == face.lines()
    assert notes[1] == (
        "mapping: headdim=16 dstate=48 ngroups=2 chunk_size=4, "
        "768 state elements per head on both sides"
    )
    assert notes[2].startswith("mamba-g2 flop: forward ")
    assert notes[3].startswith("so3ssd flop: forward ")
    assert notes[4] == "parameters: mamba2-stub 11, slinoss-stub 13"
    assert notes[5].startswith("mamba-g2: total ")
    assert notes[6].startswith("so3ssd-auto: total ")
    # The two operators take different tensors, so the arms cannot share inputs
    # the way two backends of one operator do, and the peak belongs to neither.
    assert notes[7:] == (
        "so3ssd n_groups=1",
        "mode=forward dtype=fp32",
        "arm a=mamba-g2 b=so3ssd-auto, one loop, order swapped each iteration",
        "each arm holds its own inputs; the memory peak covers both",
        "iters=2 warmup=0",
        "timer=cuda_event clocks=locked at 1740 MHz",
    )
    # A requested backend is named in the arm-b label, so the report says which
    # implementation the number belongs to.
    named = parse_args(argv(tmp_path, "--backend", "reference"))
    _, named_face = compare_so3ssd(scan, SMALL, 1, "step", named, CUDA)
    assert (named_face.row.a_label, named_face.row.b_label) == (
        "mamba-g1",
        "so3ssd-reference",
    )
    # A step is the only mode with a backward to read, and both arms have one.
    assert all(arm.backward_us is not None for arm in named_face.arms)


# ---------------------------------------------------------------------------
# main: the paired path against the SO(3) operator
# ---------------------------------------------------------------------------


def test_against_so3ssd_prints_a_rate_per_arm_and_one_verdict_per_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch)
    one = tmp_path / "one"
    assert (
        main(argv(one, "--against-so3ssd", "--mode", "forward", "--groups", "one")) == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"wrote {one / 'bench-mamba-small-g1-forward-paired.md'}"
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
    # Under every verdict, what was held equal: the mapping, both counted flop
    # totals, the parameter counts, and each arm's medians. A ratio printed alone
    # cannot be judged.
    assert [line[:2] for line in lines[7:]] == ["  "] * 6
    block = [line[2:] for line in lines[7:]]
    assert block[0].startswith("mapping: headdim=16 dstate=48 ngroups=1 chunk_size=4")
    assert [line.split(":")[0] for line in block] == [
        "mapping",
        "mamba-g1 flop",
        "so3ssd flop",
        "parameters",
        "mamba-g1",
        "so3ssd-auto",
    ]
    assert len(lines) == 13

    # Every mode of every group configuration is its own paired report, so one
    # sweep cannot have the last configuration overwrite the rest.
    every = tmp_path / "every"
    argument = argv(
        every,
        "--against-so3ssd",
        "--mode",
        "both",
        "--groups",
        "heads",
        "--groups",
        "one",
    )
    assert main(argument) == 0
    assert sorted(p.name for p in every.iterdir()) == [
        "bench-mamba-small-g1-forward-paired.json",
        "bench-mamba-small-g1-forward-paired.md",
        "bench-mamba-small-g1-step-paired.json",
        "bench-mamba-small-g1-step-paired.md",
        "bench-mamba-small-g2-forward-paired.json",
        "bench-mamba-small-g2-forward-paired.md",
        "bench-mamba-small-g2-step-paired.json",
        "bench-mamba-small-g2-step-paired.md",
    ]
    assert capsys.readouterr().out.splitlines()[:4] == [
        f"wrote {every / 'bench-mamba-small-g2-forward-paired.md'}",
        f"wrote {every / 'bench-mamba-small-g2-step-paired.md'}",
        f"wrote {every / 'bench-mamba-small-g1-forward-paired.md'}",
        f"wrote {every / 'bench-mamba-small-g1-step-paired.md'}",
    ]


def test_against_so3ssd_refuses_an_odd_iteration_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The order of the two arms swaps every iteration, so an odd count leaves one
    # iteration's order unbalanced and the difference carries a position bias.
    install(monkeypatch)
    argument = argv(tmp_path, "--against-so3ssd", "--groups", "one", iters=3)
    with pytest.raises(ValueError, match="needs an even iters"):
        main(argument)
