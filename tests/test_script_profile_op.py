"""The profile driver: argv, the three-clock wiring, and the refusal path.

The driver's job is orchestration, so what is pinned here is what it hands to
whom: the same argv to both profilers, the event total to the cross-check, the
capture iteration count as the divisor, and nothing at all to a file when the
clocks disagree.

Both profiler binaries and the device query are replaced by fakes: neither ``ncu``
nor ``nsys`` runs under the suite, and the query shells out to ``nvidia-smi``. The
workload, the timer, the budget tree, the cross-check, and the emission are real.
The fabricated profiler sums are deliberately tiny against the measured wall, so
the timeline check passes on any part and the test turns on the wiring rather than
on the clock.

The time floor is fabricated too, and backwards: its fitted law puts the one owned
kernel at a known share of the floor, so a driver scoring against the copy ceiling
instead would land on a different number. The spill record beside it is what turns
that verdict from passing to failed.

The operator axis does not interact with the skip flags, the cross-check, or the
refusal paths: ``--op`` selects which workload the three clocks run over, and they
run over whichever one that is. So it is swept once, against the report it names
and the argv it forwards, and not crossed with the rest.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from scripts.perf import profile_op
from slinoss.perf.ceiling import (
    DRAM_BOUND,
    Ceilings,
    CopySample,
    DramCeiling,
    DramTimeFloor,
    TensorCeiling,
)
from slinoss.perf.device import ClockPolicy, Contention, DeviceInfo
from slinoss.perf.ncu import (
    NCU_TABLES,
    SPILL_TABLE,
    KernelCounters,
    NcuPass,
    NcuTable,
    SpillCounters,
)
from slinoss.perf.nsys import NsysKernel, NsysTrace
from slinoss.perf.report import AgreementError
from slinoss.perf.units import (
    Bytes,
    Count,
    GBPerSecond,
    Mebibytes,
    Megahertz,
    Microseconds,
    Percent,
    Ratio,
    Spread,
    TFlopsPerSecond,
)
from slinoss.perf.workload import OPS, conv_shape_by_name

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

PASS_TABLES = (*NCU_TABLES, SPILL_TABLE)
"""Every pass the driver runs, in order: the counter tables, then the spill table.
The spill pass is not optional, because the spill rule fails a kernel whatever its
percentage and a pass never run must not read as clean."""

TABLE_NAMES = tuple(table.name for table in PASS_TABLES)
FAKE_SUM_US = 12.0
"""Fabricated profiler total, in microseconds. Far below any real CPU wall, so
the event-covers-device check has room on every host."""

FLOOR_READ_BYTES = 6 << 20
FLOOR_WRITE_BYTES = 2 << 20
FLOOR_MOVED_BYTES = FLOOR_READ_BYTES + FLOOR_WRITE_BYTES
"""DRAM traffic the fabricated kernel moves, read plus write. Over
:data:`FAKE_SUM_US` this is 699.05 GB/s, so the record is one rate rather than a
duration and a byte count that contradict each other.

Above the fabricated L2 as well, which is what makes the kernel judgeable at all: a
launch whose traffic fits in cache gets no bandwidth verdict."""

FLOOR_FIXED_US = 0.16
"""Fabricated size-independent term of the time law. Nonzero, because the verdict
charges it once per launch and a zero would hide that it is charged at all."""

FLOOR_ACHIEVED_PCT = 88.0
"""Share of the floor the owned kernel reaches, by construction. Above the 85%
bar, so the spill rule is the only thing that can fail the default fixture."""

OWNED = "kernel_cutlass_chunk_scan_fwd_kernel_bf16_Ampere_0"
"""A profiled kernel under the symbol NCU reports. The declaration table matches
the function name inside the mangled symbol, so the fabricated name carries the
mangling rather than the bare table key."""

FOREIGN = "void at::native::vectorized_elementwise_kernel<4, ...>(int, ...)"
"""A kernel this repo does not compile. It gets no verdict, and the report says
which kernels were left unjudged rather than omitting them."""


def spread(median_us: float) -> Spread:
    """A one-sample dispersion at ``median_us``."""
    return Spread.of([Microseconds(median_us)])


def device_record() -> DeviceInfo:
    """A part the numbers could have come from. Values are not asserted."""
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


def ceiling_records() -> Ceilings:
    """Measured ceilings, fabricated. The driver only forwards them."""
    return Ceilings(
        device=device_record(),
        dram=DramCeiling(
            label="device-to-device copy, 512 MiB per buffer",
            moved_bytes=Bytes(1073741824),
            duration=spread(1400.0),
            achieved_gbs=GBPerSecond(767.0),
        ),
        tensor=TensorCeiling(
            label="8192x8192x8192 torch.bfloat16 gemm",
            flop_count=Count(1099511627776),
            duration=spread(4200.0),
            achieved_tflops=TFlopsPerSecond(261.8),
        ),
    )


def floor_record() -> DramTimeFloor:
    """A fitted time law, fabricated backwards from the kernel it judges.

    The two samples lie exactly on ``t = c + bytes/B`` for the ``c`` and ``B`` that
    put the floor for :data:`FLOOR_MOVED_BYTES` at :data:`FLOOR_ACHIEVED_PCT` of
    :data:`FAKE_SUM_US`, so the fit recovers those terms and the residual is zero.
    The arithmetic of the verdict itself belongs to
    :func:`slinoss.perf.ceiling.dram_floor_verdict`; a known percentage here is
    what shows the driver judging against the floor rather than against the copy
    ceiling, which scores the same kernel differently.

    Returns:
        The fitted floor.
    """
    target_us = FAKE_SUM_US * FLOOR_ACHIEVED_PCT / 100.0
    slope = (target_us - FLOOR_FIXED_US) / FLOOR_MOVED_BYTES
    l2 = device_record().l2_bytes
    samples: list[CopySample] = []
    # Both footprints are several L2 sizes per buffer, which is what the real sweep
    # requires of them; a copy that fits in L2 measures the cache, not DRAM.
    for multiple in (2.0, 4.0):
        moved = int(2 * multiple * l2)
        duration_us = FLOOR_FIXED_US + slope * moved
        samples.append(
            CopySample(
                moved_bytes=Bytes(moved),
                duration=spread(duration_us),
                achieved_gbs=GBPerSecond(moved / (1e3 * duration_us)),
                l2_multiple_ratio=Ratio(multiple),
            )
        )
    return DramTimeFloor.of("fabricated sweep", samples, l2_bytes=l2)


def spill_record(*, kernel: str = OWNED, sector_count: int = 0) -> SpillCounters:
    """Local-memory traffic for one kernel. Zero sectors is a clean kernel.

    Args:
        kernel: Symbol the record is keyed to.
        sector_count: Sectors on each side, load and store.

    Returns:
        The record.
    """
    return SpillCounters(
        kernel=kernel,
        launch_count=Count(1),
        duration_us=Microseconds(FAKE_SUM_US),
        local_load_sector_count=Count(sector_count),
        local_store_sector_count=Count(sector_count),
    )


def trace_record(*, kernel_sum_us: float = FAKE_SUM_US) -> NsysTrace:
    """A launch stream whose sums are ``kernel_sum_us`` in total."""
    return NsysTrace(
        label="fabricated",
        report_path="fabricated.nsys-rep",
        kernel_sum_duration_us=Microseconds(kernel_sum_us),
        memcpy_sum_duration_us=Microseconds(0.0),
        memset_sum_duration_us=Microseconds(0.0),
        memcpy_count=Count(0),
        memset_count=Count(0),
        kernels=(
            NsysKernel(
                kernel="scan",
                launch_count=Count(1),
                duration_us=Microseconds(kernel_sum_us),
                duration=spread(kernel_sum_us),
                share_pct=Percent(100.0),
            ),
        ),
    )


def counter_record(
    *,
    duration_us: float = FAKE_SUM_US,
    kernel: str = OWNED,
    read_bytes: int = FLOOR_READ_BYTES,
    write_bytes: int = FLOOR_WRITE_BYTES,
) -> KernelCounters:
    """Merged counters for one kernel, summing to ``duration_us``.

    The traffic and the duration are one rate, so the DRAM verdict reads off a
    record that does not contradict itself. The traffic is a parameter because the
    audit decides on it: below L2 there is no verdict to read.
    """
    return KernelCounters(
        kernel=kernel,
        launch_count=Count(1),
        duration_us=Microseconds(duration_us),
        pass_duration_spread_pct=Percent(0.4),
        dram_read_bytes=Bytes(read_bytes),
        dram_write_bytes=Bytes(write_bytes),
        dram_pct=Percent(83.0),
        achieved_gbs=GBPerSecond((read_bytes + write_bytes) / (1e3 * duration_us)),
        global_load_request_count=Count(1 << 18),
        global_store_request_count=Count(1 << 17),
        global_load_sector_count=Count(1 << 20),
        global_store_sector_count=Count(1 << 19),
        sector_per_load_request_ratio=Ratio(4.0),
        sector_per_store_request_ratio=Ratio(4.0),
        wavefront_count=Count(4096),
        shared_load_conflict_count=Count(0),
        shared_store_conflict_count=Count(0),
        conflict_per_wavefront_ratio=Ratio(0.0),
        register_per_thread_count=Count(96),
        static_smem_bytes=Bytes(0),
        dynamic_smem_bytes=Bytes(65536),
        theoretical_occupancy_pct=Percent(50.0),
        achieved_occupancy_pct=Percent(47.0),
        tensor_pipe_pct=Percent(72.0),
        inst_count=Count(1 << 20),
        active_thread_per_warp_ratio=Ratio(32.0),
        block_count=Count(336),
        thread_per_block_count=Count(256),
        wave_per_sm_ratio=Ratio(4.0),
        issue_active_pct=Percent(12.0),
        dominant_stall="long_scoreboard",
        dominant_stall_pct=Percent(61.0),
        stall_barrier_pct=Percent(1.0),
        stall_branch_resolving_pct=Percent(0.5),
        stall_dispatch_stall_pct=Percent(0.2),
        stall_drain_pct=Percent(0.1),
        stall_imc_miss_pct=Percent(0.3),
        stall_lg_throttle_pct=Percent(0.4),
        stall_long_scoreboard_pct=Percent(61.0),
        stall_math_pipe_throttle_pct=Percent(2.0),
        stall_membar_pct=Percent(0.1),
        stall_mio_throttle_pct=Percent(3.0),
        stall_misc_pct=Percent(0.6),
        stall_no_instruction_pct=Percent(1.5),
        stall_not_selected_pct=Percent(4.0),
        stall_short_scoreboard_pct=Percent(5.0),
        stall_sleeping_pct=Percent(0.0),
        stall_tex_throttle_pct=Percent(0.0),
        stall_wait_pct=Percent(7.0),
        sm_pct=Percent(18.0),
        memory_pct=Percent(44.0),
        l1tex_pct=Percent(31.0),
        l2_pct=Percent(27.0),
    )


class Recorder:
    """What the driver handed to each fake."""

    def __init__(self) -> None:
        self.nsys_argv: list[list[str]] = []
        self.nsys_base: list[Path] = []
        self.ncu_argv: list[list[str]] = []
        self.ncu_tables: list[str] = []
        self.ncu_binary: list[str] = []
        self.ncu_extra: list[tuple[str, ...]] = []
        self.nsys_binary: list[str] = []


def patch_externals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kernel_sum_us: float = FAKE_SUM_US,
    ncu_sum_us: float = FAKE_SUM_US,
    missing_metrics: Sequence[str] = (),
    spill_sector_count: int = 0,
) -> Recorder:
    """Replace the two profiler drivers and the three device measurements.

    Args:
        monkeypatch: Patch scope.
        kernel_sum_us: NSYS kernel total for the whole capture window.
        ncu_sum_us: NCU kernel total for the whole capture window. Equal to
            ``kernel_sum_us`` means the two profilers agree exactly.
        missing_metrics: Metrics every NCU pass reports as absent.
        spill_sector_count: Local-memory sectors the profiled kernel touched, on
            each side. Nonzero is a spill, which the spill rule fails.

    Returns:
        The recorder holding what each fake was called with.
    """
    seen = Recorder()

    def fake_nsys(
        argv: Sequence[str], base: Path, *, label: str, nsys: str
    ) -> NsysTrace:
        seen.nsys_argv.append(list(argv))
        seen.nsys_base.append(base)
        seen.nsys_binary.append(nsys)
        return trace_record(kernel_sum_us=kernel_sum_us)

    def fake_ncu(
        table: NcuTable, argv: Sequence[str], *, ncu: str, extra: Sequence[str] = ()
    ) -> NcuPass:
        seen.ncu_argv.append(list(argv))
        seen.ncu_tables.append(table.name)
        seen.ncu_binary.append(ncu)
        seen.ncu_extra.append(tuple(extra))
        return NcuPass(
            table=table.name,
            command=(ncu, *argv),
            invocations=(),
            missing_metrics=tuple(missing_metrics),
        )

    def fake_counters(passes: Sequence[NcuPass]) -> tuple[KernelCounters, ...]:
        # The spill pass feeds the spill rule instead of the merge, so what reaches
        # the merge is the counter tables and no more: a spill duration folded in
        # here would double one kernel's time against the other two clocks.
        assert [one.table for one in passes] == [one.name for one in NCU_TABLES]
        return (counter_record(duration_us=ncu_sum_us),)

    def fake_spills(one: NcuPass) -> tuple[SpillCounters, ...]:
        # Keyed off the local-memory pass and no other. A record read from a
        # counter table would report every spilling kernel as clean.
        assert one.table == SPILL_TABLE.name
        return (spill_record(sector_count=spill_sector_count),)

    monkeypatch.setattr(profile_op, "run_nsys", fake_nsys)
    monkeypatch.setattr(profile_op, "run_ncu", fake_ncu)
    monkeypatch.setattr(profile_op, "kernel_counters", fake_counters)
    monkeypatch.setattr(profile_op, "spill_counters", fake_spills)
    monkeypatch.setattr(profile_op, "ceilings", lambda _device: ceiling_records())
    monkeypatch.setattr(profile_op, "dram_time_floor", lambda _device: floor_record())
    monkeypatch.setattr(profile_op, "device_info", lambda _ordinal: device_record())
    return seen


def argv_for(out: Path, *extra: str) -> list[str]:
    """The cheapest run that still exercises every stage."""
    return [
        "--shape",
        "tiny",
        "--mode",
        "forward",
        "--device",
        "cuda",
        "--iters",
        "2",
        "--warmup",
        "0",
        "--event-iters",
        "2",
        "--out",
        str(out),
        *extra,
    ]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_the_defaults_are_the_ones_the_report_stamps() -> None:
    args = profile_op.parse_args([])
    # The scan is the first entry of the registry and the default, so the command
    # that profiled the scan before the conv existed still profiles the scan.
    assert args.op == OPS[0] == "so3ssd"
    assert args.shape == "standard"
    assert args.mode == "step"
    assert args.iters == 3
    assert args.warmup == 5
    assert args.event_iters == 30
    assert args.dtype == "bf16"
    assert args.device == "cuda"
    assert args.backend is None
    assert args.kernel is None
    assert args.ncu == "ncu"
    assert args.nsys == "nsys"
    assert args.python == sys.executable
    assert args.out == Path("out/profile-op")
    assert args.skip_ncu is False
    assert args.skip_nsys is False
    named = profile_op.parse_args(
        ["--python", "/usr/bin/python3", "--ncu", "/opt/ncu", "--nsys", "/opt/nsys"]
    )
    assert named.python == "/usr/bin/python3"
    assert profile_op.target_argv(named)[0] == "/usr/bin/python3"
    assert (named.ncu, named.nsys) == ("/opt/ncu", "/opt/nsys")


def test_parse_args_rejects_a_value_outside_its_table() -> None:
    # argparse exits 2 rather than raising, so a typo in a sweep script stops the
    # run instead of silently profiling the default shape.
    for flag, value in (
        ("--op", "mamba"),
        ("--shape", "enormous"),
        ("--mode", "backward"),
        ("--dtype", "fp8"),
    ):
        with pytest.raises(SystemExit) as exc:
            profile_op.parse_args([flag, value])
        assert exc.value.code == 2
    for op in OPS:
        assert profile_op.parse_args(["--op", op]).op == op


# ---------------------------------------------------------------------------
# target_argv
# ---------------------------------------------------------------------------


def test_the_target_argv_carries_every_argument_the_target_needs() -> None:
    args = profile_op.parse_args(
        ["--shape", "long", "--mode", "step", "--iters", "4", "--warmup", "1"]
    )
    assert profile_op.target_argv(args) == [
        sys.executable,
        str(profile_op.TARGET),
        "--shape",
        "long",
        "--mode",
        "step",
        "--iters",
        "4",
        "--warmup",
        "1",
        "--dtype",
        "bf16",
        "--device",
        "cuda",
    ]
    # The profilers run a second process, so the target path is resolved from this
    # module's location rather than from the working directory.
    assert profile_op.TARGET.name == "profile_target.py"
    assert profile_op.TARGET.parent == Path(profile_op.__file__).parent
    # An operator and a backend are named only when one is asked for; a default
    # operator and an absent backend are what the target already does, so the
    # quoted command stays what a reader would have to type.
    auto = profile_op.target_argv(profile_op.parse_args([]))
    named = profile_op.target_argv(profile_op.parse_args(["--backend", "cute"]))
    assert "--op" not in auto
    assert "--backend" not in auto
    assert named[-2:] == ["--backend", "cute"]
    conv = profile_op.target_argv(profile_op.parse_args(["--op", "conv"]))
    assert conv[-2:] == ["--op", "conv"]
    # The output layout has to reach the profiled process. Without it the counters
    # would describe one layout and the event wall the other, and the cross-check
    # compares the two.
    assert "--d-head" not in auto
    layout = profile_op.target_argv(
        profile_op.parse_args(["--op", "conv", "--d-head", "48"])
    )
    assert layout[-2:] == ["--d-head", "48"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_refuses_a_device_the_report_cannot_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both arcs of the shared guard: a host device, and a CUDA device on a host
    # without CUDA. Refused before the inputs are allocated.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for spec in ("cpu", "cuda"):
        with pytest.raises(RuntimeError, match="is not a usable cuda device"):
            profile_op.main(["--device", spec])


def test_main_cross_checks_three_clocks_and_writes_both_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = patch_externals(monkeypatch)
    out = tmp_path / "prof"
    assert profile_op.main(argv_for(out)) == 0

    md = tmp_path / "prof-tiny-forward.md"
    js = tmp_path / "prof-tiny-forward.json"
    assert md.is_file()
    assert js.is_file()
    text = md.read_text()
    assert "## cross-check" in text
    assert "ncu and nsys agree" in text

    # One NSYS run and one NCU run per pass, all on the same argv and unnarrowed.
    assert len(seen.nsys_argv) == 1
    assert tuple(seen.ncu_tables) == TABLE_NAMES
    assert seen.ncu_argv == [seen.nsys_argv[0]] * len(TABLE_NAMES)
    assert seen.nsys_binary == ["nsys"]
    assert seen.ncu_binary == ["ncu"] * len(TABLE_NAMES)
    assert seen.ncu_extra == [()] * len(TABLE_NAMES)

    printed = capsys.readouterr().out.splitlines()
    assert printed[: len(TABLE_NAMES)] == [f"ncu {n}: 0 launches" for n in TABLE_NAMES]
    assert printed[-2:] == [f"wrote {md}", f"wrote {js}"]


def test_every_profiled_kernel_this_repo_compiles_lands_a_class_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The taxonomy reaches the report, and an unjudged kernel is named.

    What is pinned is the wiring, not the arithmetic: the comparison against the
    floor belongs to :mod:`slinoss.perf.ceiling` and is tested there. A verdict
    that is computed and never emitted leaves the class rule unchecked by the
    tooling that exists to check it, which is what this closes.
    """
    # Both kernels at the fabricated duration, and the trace told the window held
    # both, so the owned kernel's verdict is the same arithmetic as one kernel
    # alone and the three clocks still agree.
    patch_externals(monkeypatch, kernel_sum_us=2 * FAKE_SUM_US)

    def two_kernels(_passes: Sequence[NcuPass]) -> tuple[KernelCounters, ...]:
        return (counter_record(), counter_record(kernel=FOREIGN))

    monkeypatch.setattr(profile_op, "kernel_counters", two_kernels)
    assert profile_op.main(argv_for(tmp_path / "prof")) == 0
    doc = json.loads((tmp_path / "prof-tiny-forward.json").read_text())
    # One verdict, for the one kernel this repo owns.
    assert [v["kernel"] for v in doc["verdicts"]] == [OWNED]
    one = doc["verdicts"][0]
    assert one["declared"] == DRAM_BOUND
    assert one["required_pct"] == 85.0
    # The floor at the kernel's own traffic, not the rate of the largest copy the
    # device can run: the copy ceiling would have scored the same record 91.1%.
    assert one["achieved_pct"] == pytest.approx(FLOOR_ACHIEVED_PCT, abs=5e-4)
    assert one["passed"] is True
    text = (tmp_path / "prof-tiny-forward.md").read_text()
    assert "## class verdicts" in text
    assert f"unjudged kernels, not compiled by this repo: {FOREIGN}" in text
    assert "spill rule" not in text


def test_a_spill_fails_a_dram_bound_kernel_whatever_percentage_it_reached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The spill pass reaches the rule, and the report names what it failed.

    A spill moves the counted bytes as well as the duration, so the percentage
    stops ordering two configurations of one kernel by speed. The verdict therefore
    keeps the figure it reached and fails anyway.
    """
    patch_externals(monkeypatch, spill_sector_count=1)
    assert profile_op.main(argv_for(tmp_path / "prof")) == 0
    doc = json.loads((tmp_path / "prof-tiny-forward.json").read_text())
    one = doc["verdicts"][0]
    assert one["kernel"] == OWNED
    assert one["achieved_pct"] == pytest.approx(FLOOR_ACHIEVED_PCT, abs=5e-4)
    assert one["required_pct"] == 85.0
    assert one["passed"] is False
    text = (tmp_path / "prof-tiny-forward.md").read_text()
    assert f"failed by the spill rule, whatever the percentage: {OWNED}" in text


def test_a_kernel_whose_traffic_stays_inside_l2_gets_no_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """What the smallest shape reads, where the DRAM counters describe the cache.

    A percentage here would divide by a floor extrapolated below every swept
    footprint, off traffic that reports what L2 did not already hold rather than what
    the kernel did. The counters are still emitted; only the verdict is withheld, and
    the report names the kernel it was withheld for.
    """
    patch_externals(monkeypatch)

    def inside_l2(_passes: Sequence[NcuPass]) -> tuple[KernelCounters, ...]:
        return (counter_record(read_bytes=0, write_bytes=1 << 10),)

    monkeypatch.setattr(profile_op, "kernel_counters", inside_l2)
    assert profile_op.main(argv_for(tmp_path / "prof")) == 0
    doc = json.loads((tmp_path / "prof-tiny-forward.json").read_text())
    assert doc["verdicts"] == []
    text = (tmp_path / "prof-tiny-forward.md").read_text()
    assert "## class verdicts" not in text
    assert "## kernel counters" in text
    assert (
        "no dram verdict, per-launch traffic within L2 so the counters describe "
        f"the cache: {OWNED}" in text
    )


def test_naming_a_kernel_narrows_every_pass_and_drops_the_cross_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--kernel`` is what makes a per-kernel question answerable.

    A narrowed capture holds a subset of the launches the trace holds, so the three
    clocks cannot be made to agree; asking them to would fail on the narrowing.
    """
    seen = patch_externals(monkeypatch)
    argv = argv_for(tmp_path / "prof", "--kernel", "chunk_scan_fwd")
    assert profile_op.main(argv) == 0
    # Every pass, not just the counter tables: a spill question asked of one kernel
    # is the reason the flag exists.
    narrow = ("--kernel-name", "regex:chunk_scan_fwd")
    assert tuple(seen.ncu_tables) == TABLE_NAMES
    assert seen.ncu_extra == [narrow] * len(TABLE_NAMES)
    text = (tmp_path / "prof-tiny-forward.md").read_text()
    assert "## cross-check" not in text
    assert "cross-check skipped" in text
    assert "ncu narrowed to kernels matching 'chunk_scan_fwd'" in text
    # The counters and the verdicts still land; they cover what was profiled.
    assert "## kernel counters" in text
    assert "## class verdicts" in text


def test_the_cross_check_divides_by_the_capture_iters_and_uses_the_event_total(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_externals(monkeypatch)
    out = tmp_path / "prof"
    assert profile_op.main(argv_for(out, "--iters", "5")) == 0
    doc = json.loads((tmp_path / "prof-tiny-forward.json").read_text())
    check = doc["agreement"]
    assert check["capture_iter_count"] == 5
    # Both figures come from the same measured total, so equality here is what
    # rules out a region timing being passed where the loop total belongs.
    assert check["event_duration_us"] == doc["throughput"][0]["duration_us"]
    # The fabricated sums are per capture window, so the per-iteration figures
    # are the totals over the divisor.
    assert check["nsys_kernel_sum_duration_us"] == pytest.approx(FAKE_SUM_US / 5)
    assert check["ncu_kernel_sum_duration_us"] == pytest.approx(FAKE_SUM_US / 5)
    assert check["kernel_delta_pct"] == 0.0
    assert check["agrees"] is True


def test_a_report_whose_clocks_disagree_never_reaches_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # NCU sees twice what NSYS saw. A stale report that survives a failed run is
    # indistinguishable from a fresh pass, so nothing is written at all.
    patch_externals(monkeypatch, kernel_sum_us=2.0, ncu_sum_us=4.0)
    out = tmp_path / "prof"
    with pytest.raises(AgreementError, match="differ by"):
        profile_op.main(argv_for(out))
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_main_raises_when_an_ncu_table_reports_a_metric_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A metric absent from every row means the name is wrong for this driver
    # version, which would otherwise emit a table of zeros.
    patch_externals(monkeypatch, missing_metrics=("dram__bytes.sum",))
    with pytest.raises(ValueError, match=r"dram__bytes\.sum") as exc:
        profile_op.main(argv_for(tmp_path / "prof"))
    assert TABLE_NAMES[0] in str(exc.value)


def test_skipping_a_profiler_emits_without_it_and_says_the_check_was_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A cross-check needs all three clocks, so any skip drops it, and the report
    # says so rather than omitting the section silently.
    no_ncu = patch_externals(monkeypatch)
    assert profile_op.main(argv_for(tmp_path / "a" / "prof", "--skip-ncu")) == 0
    text = (tmp_path / "a" / "prof-tiny-forward.md").read_text()
    assert no_ncu.ncu_tables == []
    assert "## cross-check" not in text
    assert "cross-check skipped" in text
    assert "## gpu trace" in text

    no_nsys = patch_externals(monkeypatch)
    assert profile_op.main(argv_for(tmp_path / "b" / "prof", "--skip-nsys")) == 0
    text = (tmp_path / "b" / "prof-tiny-forward.md").read_text()
    assert no_nsys.nsys_argv == []
    assert "## cross-check" not in text
    assert "cross-check skipped" in text
    assert "## kernel counters" in text

    neither = patch_externals(monkeypatch)
    argv = argv_for(tmp_path / "c" / "prof", "--skip-ncu", "--skip-nsys")
    assert profile_op.main(argv) == 0
    text = (tmp_path / "c" / "prof-tiny-forward.md").read_text()
    assert neither.ncu_tables == []
    assert neither.nsys_argv == []
    # The event wall survives both skips.
    assert "## budget" in text
    assert "## throughput" in text
    assert "cross-check skipped" in text


def test_step_mode_measures_the_backward_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_externals(monkeypatch)
    out = tmp_path / "prof"
    argv = argv_for(out)
    argv[argv.index("forward")] = "step"
    assert profile_op.main(argv) == 0
    text = (tmp_path / "prof-tiny-step.md").read_text()
    assert "profile: so3ssd tiny step" in text
    assert "| op.backward | op |" in text
    assert "mode=step dtype=bf16 backend=auto" in text


def test_the_conv_operator_profiles_the_conv_and_names_it_in_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = patch_externals(monkeypatch)
    out = tmp_path / "prof"
    argv = argv_for(out, "--op", "conv")
    argv[argv.index("forward")] = "step"
    assert profile_op.main(argv) == 0
    doc = json.loads((tmp_path / "prof-tiny-step.json").read_text())
    # The title carries the operator, because the report base does not: one --out
    # at one shape and one mode is one file whichever operator produced it.
    assert doc["title"] == "profile: conv tiny step"
    assert doc["throughput"][0]["label"] == "conv tiny step"
    tiny = conv_shape_by_name("tiny")
    assert doc["throughput"][0]["token_count"] == tiny.token_count
    text = (tmp_path / "prof-tiny-step.md").read_text()
    assert "| conv.backward | conv |" in text
    assert tiny.describe() in text
    # The event bench and the profiled window ran one workload, so the forwarded
    # argv names the operator the event wall measured.
    assert seen.nsys_argv[0][-2:] == ["--op", "conv"]


def test_the_notes_quote_the_command_the_profilers_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = patch_externals(monkeypatch)
    out = tmp_path / "prof"
    assert profile_op.main(argv_for(out, "--backend", "reference")) == 0
    text = (tmp_path / "prof-tiny-forward.md").read_text()
    assert "target: " + " ".join(seen.nsys_argv[0]) in text
    assert "--backend reference" in text
    assert "event iters=2 capture iters=2" in text
    assert "nsys report: fabricated.nsys-rep" in text
    # Both fitted terms and the residual travel with the verdicts they scored: a
    # floor extrapolated far below its sweep is worth no more than its residual,
    # and a reader cannot see that from the percentage alone.
    assert (
        "dram floor: c=0.160 us B=806.60 GB/s max residual=0.00% l2=6291456 B" in text
    )
