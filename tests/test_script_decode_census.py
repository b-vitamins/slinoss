"""The decode census: its staged step, its byte model, and its two reductions.

This pins ``scripts.perf.decode_census``. The one thing the census cannot get
wrong is which program it measured, so the first test runs the staged
decomposition and :meth:`slinoss.mixer.SLinOSSMixer.step` from identical states
and holds every output and every carry bitwise equal over several steps. A stage
that drifted from the routed program fails here rather than mis-attributing a
kernel in a report.

No external profiler runs and no figure here is a timing. Two properties are about
the device and are observed on it: which stages launch a kernel, read off the
in-process torch profiler, and what a step costs once its host program is deleted,
which is a CUDA-graph replay held only to its own arithmetic. The step itself needs
a device:
every operand between the two projections is a column band, and
:func:`slinoss._guard.check_pitched` holds a band to a device rule, so a CPU step
raises from the first consumer that checks one. The tests that only construct a
program stay on the host. The two reductions are driven with fixture text in NCU's
and NSYS's own formats, and the DRAM floor is fitted to a law stated in this module,
so every verdict a row carries is exact rather than a property of the host.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest
import torch
from torch.autograd import DeviceType

from scripts.perf.decode_census import (
    ALL_STAGES,
    CARRY_CONV,
    CARRY_KEYS,
    CENSUS_TABLE,
    CONTAMINATION_CEILING_PCT,
    COPY_ONLY_STAGES,
    FUSION_CANDIDATE,
    IN_PROJ,
    KEY_CONV,
    OUT_PROJ,
    PREP,
    RECURRENCE,
    SECTOR_BYTES,
    STAGE_ORDER,
    TAIL,
    UNJUDGED,
    VALUE_CONV,
    VERDICT_METRICS,
    Site,
    build_layer,
    cell_key,
    contamination_residual_pct,
    launch_census,
    launch_table,
    loop_wall_us,
    main,
    parse_args,
    payload,
    prime,
    registry_names,
    replay_timing,
    require_kernel_path,
    resolved_backends,
    stage_program,
    stage_rows,
    sum_by_site,
    target_argv,
    traffic_table,
    write_cell,
)
from slinoss.mixer import SLinOSSMixer
from slinoss.perf.ceiling import DRAM_BOUND, CopySample, DramTimeFloor
from slinoss.perf.ncu import NcuPass, NcuTable, parse_ncu_csv
from slinoss.perf.units import Bytes, GBPerSecond, Microseconds, Ratio, Spread
from slinoss.state import MixerState

SHAPE: Final = "tiny"
"""The smallest registered decode geometry: one layer, the shortest launch chain.

Every test here is about a composition or a reduction and neither is a function of
width, so the cheapest geometry that still exercises both convolutions is the right
one."""

BATCH: Final = 2

HOST: Final = torch.device("cpu")
"""Where a program is constructed. Construction launches nothing."""

DEVICE: Final = torch.device("cuda")
"""Where a program runs.

float32 rather than a low-precision dtype: the composition is held bitwise, and
every figure the byte model states is a shape times an element size, so the widest
supported activation is the one that leaves nothing to a tolerance."""

STEPS: Final = 3
"""Steps the composition is held over.

More than one, because the failure mode is a carry advanced in the wrong order:
that reproduces the first step exactly and diverges on the second."""


# ---------------------------------------------------------------------------
# the staged step is the routed step
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_staged_program_reproduces_the_routed_step() -> None:
    # The census attributes a kernel to a stage, so the stages must be the routed
    # program and not a paraphrase of it. Bitwise, not close: both paths call the
    # same backends on the same operands in the same order, so any difference here
    # is a difference in the program.
    staged = build_layer(SHAPE, BATCH, DEVICE, dtype=torch.float32)
    routed = build_layer(SHAPE, BATCH, DEVICE, dtype=torch.float32)
    for _ in range(STEPS):
        staged.run()
        expected = routed.mixer.step(routed.x, routed.state)
        assert torch.equal(staged.output(), expected)
        for name in ("conv", "keys", "ssm", "b_prev", "u_prev"):
            assert torch.equal(
                getattr(staged.state, name), getattr(routed.state, name)
            ), f"carry {name} diverged"


def test_stage_order_is_the_call_order_and_the_carries_land_last() -> None:
    # A carry advances a window only after the last read of the window it replaces,
    # so its position in the order is part of the program rather than a detail.
    assert set(STAGE_ORDER) == set(FUSION_CANDIDATE)
    assert len(STAGE_ORDER) == len(FUSION_CANDIDATE)
    assert STAGE_ORDER[0] == IN_PROJ
    assert STAGE_ORDER[-2:] == (CARRY_CONV, CARRY_KEYS)
    assert STAGE_ORDER.index(RECURRENCE) < STAGE_ORDER.index(TAIL)
    assert STAGE_ORDER.index(TAIL) < STAGE_ORDER.index(OUT_PROJ)
    program = build_layer(SHAPE, BATCH, HOST, dtype=torch.float32)
    assert tuple(stage.name for stage in program.stages) == STAGE_ORDER
    assert all(stage.candidate for stage in program.stages)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_only_the_two_carries_are_declared_kernelless() -> None:
    # The declaration decides which stages NCU is asked about, and asking about a
    # kernelless one loses the whole cell: NCU emits no CSV and exits zero. So the
    # property is a launch count and it is observed as one, per stage, rather than
    # read back off the constant that declares it. A memcpy is a device event too and
    # is told apart by its name, which is the distinction NCU itself draws.
    #
    # Counted off the device events rather than off a host op's kernel list: the
    # extension conv and every CuTe entry point launch outside the dispatcher, so
    # their kernels correlate to no host op and a kernel-list count reports every one
    # of them as kernelless.
    program = build_layer(SHAPE, BATCH, DEVICE, dtype=torch.float32)
    program.run()
    torch.cuda.synchronize()
    observed: dict[str, int] = {}
    for stage in program.stages:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            stage.run()
            torch.cuda.synchronize()
        events = prof.events()
        assert events is not None
        observed[stage.name] = sum(
            1
            for event in events
            if event.device_type == DeviceType.CUDA
            and not event.key.startswith("Memcpy")
        )
    kernelless = {name for name, count in observed.items() if count == 0}
    assert kernelless == COPY_ONLY_STAGES
    assert {stage.name for stage in program.stages if stage.copy_only} == kernelless
    # The recurrence stage is the one that launches twice, and the step's whole
    # kernel count is what the launch census has to reproduce.
    assert observed[RECURRENCE] == 2
    assert sum(observed.values()) == 8


def test_stage_program_guards_every_disagreement_with_the_state() -> None:
    program = build_layer(SHAPE, BATCH, HOST, dtype=torch.float32)
    mixer, state = program.mixer, program.state
    width = mixer.config.d_model
    with pytest.raises(ValueError, match="expected"):
        stage_program(mixer, torch.zeros(BATCH, 1, width + 1), state)
    with pytest.raises(ValueError, match="census subject is T=1"):
        stage_program(mixer, torch.zeros(BATCH, 2, width), state)
    with pytest.raises(ValueError, match="batch"):
        stage_program(mixer, torch.zeros(BATCH + 1, 1, width), state)
    with pytest.raises(ValueError, match="cast the module, not the state"):
        stage_program(mixer, torch.zeros(BATCH, 1, width, dtype=torch.float64), state)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_operands_refuse_a_slot_no_stage_has_filled() -> None:
    # A stage measured in isolation reads what the routed step handed it, and
    # priming is what makes that true. An unprimed program refuses rather than
    # reporting a byte count for operands that do not exist yet.
    program = build_layer(SHAPE, BATCH, DEVICE, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="run the whole program once"):
        program.stage(RECURRENCE).operands()
    with pytest.raises(RuntimeError, match="run the whole program once"):
        program.output()
    program.run()
    assert int(program.stage(RECURRENCE).operands().total_bytes) > 0


def test_prime_needs_at_least_one_whole_step() -> None:
    program = build_layer(SHAPE, BATCH, HOST, dtype=torch.float32)
    with pytest.raises(ValueError, match="at least one whole step"):
        prime(program, warmup=0)


def test_stage_lookup_names_what_it_has() -> None:
    program = build_layer(SHAPE, BATCH, HOST, dtype=torch.float32)
    with pytest.raises(KeyError, match="no stage"):
        program.stage("tail_fusion")


# ---------------------------------------------------------------------------
# the compulsory byte model
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_recurrence_compulsory_bytes_match_the_operator_traffic_model() -> None:
    # The operator boundary documents its own per-call traffic. The census counts
    # bytes off the live tensors instead of restating that model, so the two must
    # agree or one of them is wrong about the shapes.
    program = build_layer(SHAPE, BATCH, DEVICE, dtype=torch.float32)
    program.run()
    cfg = program.mixer.config
    heads, rows = cfg.n_heads, cfg.d_head
    state_dim, groups = cfg.d_state, cfg.n_groups
    activation = program.x.element_size()
    carried = program.state.ssm.element_size()
    tokens = BATCH * heads * rows * activation
    band = BATCH * groups * state_dim * activation
    resident = BATCH * heads * rows * state_dim * carried
    operands = program.stage(RECURRENCE).operands()
    assert int(operands.read_bytes) == (
        tokens + 16 * BATCH * heads + 32 * BATCH * heads + band + band + resident
    )
    assert int(operands.write_bytes) == tokens + resident + band + tokens


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_compulsory_bytes_count_each_distinct_operand_once() -> None:
    # A byte read twice by one kernel is a cache question. The compulsory figure is
    # one pass over each distinct operand, so a repeated label would double-count
    # and a missing one would put the traffic ratio above one for free.
    program = build_layer(SHAPE, BATCH, DEVICE, dtype=torch.float32)
    program.run()
    for stage in program.stages:
        operands = stage.operands()
        reads = [label for label, _ in operands.reads]
        writes = [label for label, _ in operands.writes]
        assert len(reads) == len(set(reads)), stage.name
        assert len(writes) == len(set(writes)), stage.name
        assert int(operands.total_bytes) == int(operands.read_bytes) + int(
            operands.write_bytes
        )
    # The recurrence reads and writes one buffer, which is the whole point of the
    # T=1 boundary: it advances the carry in the caller's storage.
    recurrence = program.stage(RECURRENCE).operands()
    assert "state.ssm" in dict(recurrence.reads)
    assert "state.ssm" in dict(recurrence.writes)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_a_carry_moves_the_window_it_replaces_and_nothing_else() -> None:
    program = build_layer(SHAPE, BATCH, DEVICE, dtype=torch.float32)
    program.run()
    carry = program.stage(CARRY_CONV).operands()
    assert int(carry.read_bytes) == int(carry.write_bytes)
    assert int(carry.write_bytes) == (
        program.state.conv.numel() * program.state.conv.element_size()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_a_layer_without_key_convolution_has_two_empty_stages() -> None:
    # The key convolution is configurable, and a stage that does nothing must
    # report no operands rather than a byte count NCU will never account for.
    device = DEVICE
    program = build_layer(SHAPE, BATCH, device, dtype=torch.float32)
    keyless = dataclasses.replace(program.mixer.config, key_conv=False)
    mixer = SLinOSSMixer(keyless, device=device)
    state = MixerState.allocate(keyless, BATCH, device=device, dtype=torch.float32)
    plain = stage_program(mixer, program.x, state)
    plain.run()
    assert int(plain.stage(KEY_CONV).operands().total_bytes) == 0
    assert int(plain.stage(CARRY_KEYS).operands().total_bytes) == 0
    assert int(plain.stage(RECURRENCE).operands().total_bytes) > 0


# ---------------------------------------------------------------------------
# the anti-void check
# ---------------------------------------------------------------------------


def test_require_kernel_path_refuses_a_reference_decode() -> None:
    # A registry whose kernel import failed answers with the reference, and the only
    # symptom is a slower program reported as this one. The host path is that
    # condition by construction, so it is what the refusal is tested against.
    with pytest.raises(RuntimeError, match="decode resolved to 'reference'"):
        require_kernel_path("cpu", torch.float32)
    # The provenance block prints both maps and a registry in one but not the other
    # is a hole in it, so the key set is held against the registries themselves and
    # every answer against what that registry actually holds.
    registries = registry_names()
    resolved = resolved_backends("cpu", torch.float32)
    assert set(resolved) == set(registries)
    assert all(resolved[name] in registries[name] for name in registries)
    assert resolved["decode"] == "reference"


# ---------------------------------------------------------------------------
# the launch census
# ---------------------------------------------------------------------------

GEMM: Final = "cutlass_gemm_tn"
CONV: Final = "conv1d_fwd_kernel"
DECODE: Final = "decode_fwd_kernel"

TRACE_CSV: Final = """Start (ns),Duration (ns),Name
1000,400,cutlass_gemm_tn
1600,200,conv1d_fwd_kernel
2000,300,decode_fwd_kernel
2500,100,[CUDA memcpy DtoD]
3000,400,cutlass_gemm_tn
3600,200,conv1d_fwd_kernel
4000,300,decode_fwd_kernel
4500,100,[CUDA memcpy DtoD]
"""
"""Two identical steps of three kernels and one copy each, with stated gaps.

Span 1000 to 4600 ns and 2000 ns busy, so 1600 ns of idle over seven gaps whose
longest is the 400 ns between the two steps. Kernels hold 1800 of the 2000 busy
ns, which is the total every kernel share is a fraction of."""

TRACE_REPS: Final = 2
TRACE_WALL_US: Final = Microseconds(2.0)
TRACE_KERNEL_SHARE_PCT: Final = 90.0


def test_launch_census_divides_the_window_and_counts_the_copies() -> None:
    got = launch_census(
        TRACE_CSV, step_wall_us=TRACE_WALL_US, reps=TRACE_REPS, report_path="out/x.rep"
    )
    assert int(got.reps) == TRACE_REPS
    # Per step, which is per layer: the subject is one layer.
    assert float(got.launches_per_step) == pytest.approx(3.0)
    assert float(got.copies_per_step) == pytest.approx(1.0)
    assert float(got.kernel_us) == pytest.approx(0.9)
    assert float(got.copy_us) == pytest.approx(0.1)
    assert float(got.device_us) == pytest.approx(1.0)
    # A carry between same-dtype contiguous buffers is a copy and not a kernel, so
    # the two device totals differ by exactly the copy.
    assert float(got.device_us) - float(got.kernel_us) == pytest.approx(
        float(got.copy_us)
    )
    assert float(got.idle_us) == pytest.approx(0.8)
    assert float(got.idle_pct) == pytest.approx(1600 / 3600 * 100)
    assert float(got.gaps_per_step) == pytest.approx(3.5)
    # Not divided: the longest gap is one interval, not a rate.
    assert float(got.max_gap_us) == pytest.approx(0.4)
    assert float(got.per_launch_idle_us) == pytest.approx(0.8 / 3.0)
    # What the step costs beyond the device.
    assert float(got.host_us) == pytest.approx(1.0)


def test_launch_census_rows_are_per_kernel_and_ordered_by_cost() -> None:
    got = launch_census(TRACE_CSV, step_wall_us=TRACE_WALL_US, reps=TRACE_REPS)
    assert [row.kernel for row in got.rows] == [GEMM, DECODE, CONV]
    assert [float(row.launches_per_step) for row in got.rows] == [1.0, 1.0, 1.0]
    assert float(got.rows[0].duration_us) == pytest.approx(0.4)
    # The share is of device time, so the kernel rows leave the copy's share out.
    assert sum(float(row.share_pct) for row in got.rows) == pytest.approx(
        TRACE_KERNEL_SHARE_PCT
    )
    rendered = launch_table(got)
    assert GEMM in rendered
    assert "device total" in rendered


def test_launch_census_refuses_a_window_it_cannot_divide_by() -> None:
    with pytest.raises(ValueError, match="reps must be positive"):
        launch_census(TRACE_CSV, step_wall_us=TRACE_WALL_US, reps=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_replay_prices_launches_against_device_time_not_the_eager_wall() -> None:
    # The eager wall carries the dispatcher and the Python between kernels, and a
    # launch count cannot buy either back. Only replay isolates what the launches
    # themselves cost, so that is where the per-launch figure comes from. The
    # arithmetic is held exactly; the timing is held only to being positive, since
    # this runs on whatever card the suite is on.
    program = build_layer(SHAPE, 4, DEVICE, dtype=torch.float32)
    prime(program, warmup=2)
    census = launch_census(TRACE_CSV, step_wall_us=TRACE_WALL_US, reps=TRACE_REPS)
    got = replay_timing(program, census, iters=8, warmup=4, device=DEVICE)
    assert got.error == ""
    assert float(got.wall_us) > 0.0
    assert float(got.idle_us) == pytest.approx(
        max(0.0, float(got.wall_us) - float(census.device_us))
    )
    assert float(got.per_launch_idle_us) == pytest.approx(
        float(got.idle_us) / float(census.launches_per_step)
    )
    # The two loop walls run the same step on the same host, so replay cannot cost
    # more host time than eager: replay deletes the dispatcher and keeps the launches.
    assert float(got.eager_loop_wall_us) > 0.0
    assert float(got.replay_loop_wall_us) > 0.0
    assert float(got.replay_loop_wall_us) < float(got.eager_loop_wall_us)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_the_loop_wall_is_per_step_and_not_per_loop() -> None:
    # A host cost of a known size, so the division is observable: four sleeps of two
    # milliseconds read as one sleep, not as the loop's eight. The figure this guards
    # is the eager host cost, which a per-sample median misreports by a factor of
    # fifteen whenever the host runs ahead of the device.
    delay_s = 0.002
    got = loop_wall_us(lambda: time.sleep(delay_s), iters=4, device=DEVICE)
    assert float(got) == pytest.approx(delay_s * 1e6, rel=0.5)


def test_the_loop_wall_refuses_an_empty_loop() -> None:
    with pytest.raises(ValueError, match="iters must be positive"):
        loop_wall_us(lambda: None, iters=0, device=HOST)


# ---------------------------------------------------------------------------
# the traffic census
# ---------------------------------------------------------------------------

FIXED_US: Final = 4.5036
RATE_GBS: Final = 684.898
"""The law the fixture floor is fitted to, so every verdict below is exact."""

L2_BYTES: Final = Bytes(6 * 1024 * 1024)


def law_us(moved: int) -> float:
    """The fixture law's duration for a copy of ``moved`` bytes.

    Args:
        moved: Bytes crossing DRAM, read plus write.

    Returns:
        Microseconds.
    """
    return FIXED_US + moved / (RATE_GBS * 1e3)


def sample(moved: int) -> CopySample:
    """One copy sample sitting exactly on :func:`law_us`.

    Args:
        moved: Bytes crossing DRAM, read plus write.

    Returns:
        The sample.
    """
    duration = Microseconds(law_us(moved))
    return CopySample(
        moved_bytes=Bytes(moved),
        duration=Spread.of((duration,)),
        achieved_gbs=GBPerSecond(moved / duration / 1e3),
        l2_multiple_ratio=Ratio(moved / 2 / int(L2_BYTES)),
    )


@pytest.fixture
def floor() -> DramTimeFloor:
    """The fixture floor, fitted to two points on :func:`law_us`."""
    return DramTimeFloor.of(
        "fixture",
        (sample(12 * 1024 * 1024), sample(384 * 1024 * 1024)),
        l2_bytes=L2_BYTES,
    )


def test_the_fixture_floor_recovers_the_law_it_was_fitted_to(
    floor: DramTimeFloor,
) -> None:
    # Two points fit the two terms exactly, so a residual here would be a defect in
    # the fit and every verdict below would rest on it.
    assert float(floor.fixed_duration_us) == pytest.approx(FIXED_US, rel=1e-6)
    assert float(floor.asymptotic_gbs) == pytest.approx(RATE_GBS, rel=1e-6)
    assert float(floor.max_residual_pct) == pytest.approx(0.0, abs=1e-6)


HEADER: Final = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
    '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"'
)


def record(launch: int, kernel: str, metric: str, unit: str, value: float) -> str:
    """One CSV record in NCU's column order.

    Args:
        launch: Launch id, which is what groups metrics into one invocation.
        kernel: Kernel name.
        metric: Metric name.
        unit: Metric unit, verbatim as NCU prints it.
        value: Display value, in that unit.

    Returns:
        One line, newline terminated.
    """
    return (
        f'"{launch}","4711","python3","host","{kernel}",'
        f'"2026-09-05 09:14:03","1","7","","{metric}","{unit}","{value}"\n'
    )


UNITS: Final[Mapping[str, str]] = {
    "gpu__time_duration.sum": "usecond",
    "launch__grid_size": "",
    "launch__block_size": "",
    "launch__registers_per_thread": "register/thread",
    "dram__bytes_read.sum": "byte",
    "dram__bytes_write.sum": "byte",
    "lts__t_sectors_op_read.sum": "sector",
    "lts__t_sectors_op_write.sum": "sector",
    "lts__t_sector_op_read_hit_rate.pct": "%",
    "lts__t_sector_op_write_hit_rate.pct": "%",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "request",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": "request",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "sector",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "sector",
}
"""The unit NCU prints beside each census metric. Every one must be requested."""


def census_pass(
    launches: Sequence[tuple[str, Mapping[str, float]]],
    *,
    table: NcuTable = CENSUS_TABLE,
    metrics: Sequence[str] | None = None,
) -> NcuPass:
    """Parse a synthetic census pass over the given launches.

    Args:
        launches: ``(kernel, metric -> display value)`` per launch, in launch order.
            A metric the mapping omits is emitted at zero.
        table: The table whose metrics were requested.
        metrics: Metrics actually emitted, defaulting to the table's own. Emitting
            fewer is how a missing metric is staged.

    Returns:
        The parsed pass.
    """
    emitted = tuple(table.metrics if metrics is None else metrics)
    text = HEADER + "\n"
    for index, (kernel, values) in enumerate(launches):
        for metric in emitted:
            text += record(
                index, kernel, metric, UNITS[metric], values.get(metric, 0.0)
            )
    return parse_ncu_csv(text, table.metrics, table=table.name)


def test_every_census_metric_carries_a_unit_the_parser_knows() -> None:
    # A unit the parser cannot scale raises, and a metric requested but never
    # emitted lands in missing_metrics. Both are fixture defects that would read as
    # device facts, so the mapping is held complete against the table.
    assert set(UNITS) == set(CENSUS_TABLE.metrics)
    assert set(VERDICT_METRICS) <= set(CENSUS_TABLE.metrics)


def counters(
    *,
    duration_us: float,
    read_bytes: float,
    write_bytes: float,
    grid: int = 96,
    block: int = 256,
    registers: int = 64,
    read_hit_pct: float = 0.0,
    write_hit_pct: float = 0.0,
    load_requests: float = 0.0,
    load_sectors: float = 0.0,
    store_requests: float = 0.0,
    store_sectors: float = 0.0,
    local_load_sectors: float = 0.0,
    local_store_sectors: float = 0.0,
    unexplained_bytes: float = 0.0,
) -> dict[str, float]:
    """One launch's display values, with the L2 sectors its DRAM traffic implies.

    The sectors are derived from the byte counts rather than stated, so a launch
    built here is uncontaminated by construction unless ``unexplained_bytes`` asks
    for otherwise. The read side is derived through the read miss rate and the write
    side without one, which is the write-back model the census uses; a launch built
    here therefore sits at a zero residual whatever its hit rates are, so a hit rate
    is free to differ between the two sides and an argument swap in the join is
    visible.

    Args:
        duration_us: Kernel duration.
        read_bytes: DRAM bytes read.
        write_bytes: DRAM bytes written.
        grid: Blocks launched.
        block: Threads per block.
        registers: Registers per thread.
        read_hit_pct: L2 read hit rate.
        write_hit_pct: L2 write hit rate. Carried, not used to derive sectors.
        load_requests: Global load requests.
        load_sectors: Global load sectors.
        store_requests: Global store requests.
        store_sectors: Global store sectors.
        local_load_sectors: Local-memory load sectors, which is a register spill.
        local_store_sectors: Local-memory store sectors, the other half of one.
        unexplained_bytes: DRAM bytes to report that the sectors do not account
            for, which is what a co-resident process's traffic looks like. Added to
            the read side only, since that is the side the miss-rate bound holds.

    Returns:
        Metric name to display value.
    """
    read_miss = 1.0 - read_hit_pct / 100.0
    return {
        "gpu__time_duration.sum": duration_us,
        "launch__grid_size": grid,
        "launch__block_size": block,
        "launch__registers_per_thread": registers,
        "dram__bytes_read.sum": read_bytes + unexplained_bytes,
        "dram__bytes_write.sum": write_bytes,
        "lts__t_sectors_op_read.sum": read_bytes / SECTOR_BYTES / read_miss,
        "lts__t_sectors_op_write.sum": write_bytes / SECTOR_BYTES,
        "lts__t_sector_op_read_hit_rate.pct": read_hit_pct,
        "lts__t_sector_op_write_hit_rate.pct": write_hit_pct,
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": load_requests,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": load_sectors,
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": store_requests,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": store_sectors,
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": local_load_sectors,
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": local_store_sectors,
    }


def test_stage_rows_judge_a_kernel_against_the_floor_at_its_own_footprint(
    floor: DramTimeFloor,
) -> None:
    # The denominator is a copy of the kernel's own size, not the largest copy the
    # device can run. A kernel exactly on the law is at 100% of it.
    moved = 8 * 1024 * 1024
    one = census_pass(
        [
            (
                DECODE,
                counters(
                    duration_us=law_us(moved),
                    read_bytes=moved * 0.75,
                    write_bytes=moved * 0.25,
                ),
            )
        ]
    )
    rows = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        one,
        compulsory=Bytes(moved),
        reps=1,
        floor=floor,
    )
    assert len(rows) == 1
    row = rows[0]
    assert float(row.traffic_ratio) == pytest.approx(1.0)
    assert float(row.floor_pct) == pytest.approx(100.0, rel=1e-6)
    assert row.roofline_class == f"{DRAM_BOUND} pass"
    assert float(row.achieved_gbs) == pytest.approx(float(row.floor_gbs), rel=1e-6)
    assert not row.contaminated
    assert not row.spilled
    assert row.register_per_thread_count == 64
    assert row.site == Site(kernel=DECODE, grid=96, block=256)


def test_the_contamination_flag_trips_at_the_ceiling_and_not_below(
    floor: DramTimeFloor,
) -> None:
    # `dram__bytes` is device-wide, so a co-resident process's traffic lands in this
    # kernel's row and the residual against its own L2 sectors is the only detector.
    # Held on both sides of the ceiling, since a row that never trips it is a row
    # that would report a neighbour's bytes as this kernel's.
    moved = 8 * 1024 * 1024

    def row_at(extra_pct: float) -> tuple[float, bool]:
        one = census_pass(
            [
                (
                    DECODE,
                    counters(
                        duration_us=law_us(moved),
                        read_bytes=moved,
                        write_bytes=0.0,
                        unexplained_bytes=moved * extra_pct / 100.0,
                    ),
                )
            ]
        )
        got = stage_rows(
            RECURRENCE,
            FUSION_CANDIDATE[RECURRENCE],
            one,
            compulsory=Bytes(moved),
            reps=1,
            floor=floor,
        )[0]
        return float(got.residual_pct), got.contaminated

    # The residual is a share of the reported total, not of the explained part, so
    # the extra traffic that straddles a 10% ceiling is 9% on one side and 14% on the
    # other. 11% reads 9.91% and would sit below it.
    below_pct, below = row_at(9.0)
    above_pct, above = row_at(14.0)
    assert below_pct == pytest.approx(9.0 / 1.09, rel=1e-3)
    assert above_pct == pytest.approx(14.0 / 1.14, rel=1e-3)
    assert below_pct < CONTAMINATION_CEILING_PCT < above_pct
    assert not below
    assert above


def test_a_site_reports_its_own_store_coalescing_and_both_spill_halves(
    floor: DramTimeFloor,
) -> None:
    # The store side is where this operator's defect lived: sectors per store request
    # read 10.7059 against a 4.0 ideal before the transpose. A census blind to the
    # store term cannot see that class, so both store counters and both halves of the
    # spill are held, and each is given a value the load side does not share.
    moved = 4 * 1024 * 1024
    one = census_pass(
        [
            (
                DECODE,
                counters(
                    duration_us=law_us(moved),
                    read_bytes=moved / 2,
                    write_bytes=moved / 2,
                    load_requests=1000.0,
                    load_sectors=4000.0,
                    store_requests=800.0,
                    store_sectors=8565.0,
                    local_load_sectors=0.0,
                    local_store_sectors=48.0,
                ),
            )
        ]
    )
    row = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        one,
        compulsory=Bytes(moved),
        reps=1,
        floor=floor,
    )[0]
    assert float(row.sectors_per_load_request) == pytest.approx(4.0)
    assert float(row.sectors_per_store_request) == pytest.approx(10.70625)
    # Local stores alone are a spill. Counting loads alone would call this clean.
    assert row.local_sector_count == 48
    assert row.spilled


def test_stage_rows_divide_the_window_but_judge_on_its_whole_sum(
    floor: DramTimeFloor,
) -> None:
    # Every reported byte and microsecond is per step; the verdict is taken on the
    # window's own sums, because the fitted fixed term is charged once per launch
    # and folding it in once would score a multi-launch kernel low.
    moved = 4 * 1024 * 1024
    launch = counters(
        duration_us=law_us(moved), read_bytes=moved / 2, write_bytes=moved / 2
    )
    rows = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        census_pass([(DECODE, launch)] * 3),
        compulsory=Bytes(moved),
        reps=3,
        floor=floor,
    )
    row = rows[0]
    assert float(row.launches_per_step) == pytest.approx(1.0)
    assert float(row.duration_us) == pytest.approx(law_us(moved))
    assert int(row.dram_bytes) == pytest.approx(moved, rel=1e-6)
    assert float(row.traffic_ratio) == pytest.approx(1.0)
    assert float(row.floor_pct) == pytest.approx(100.0, rel=1e-6)


def test_a_footprint_below_the_cache_is_unjudged_not_passed(
    floor: DramTimeFloor,
) -> None:
    # `footprint > L2` is not the crossover condition and the traffic ratio is. A
    # stage that moved less than its operands demand was served by cache, and no
    # bandwidth verdict exists for it at that shape.
    compulsory = 8 * 1024 * 1024
    moved = int(compulsory * 0.967)
    one = census_pass(
        [(DECODE, counters(duration_us=law_us(moved), read_bytes=moved, write_bytes=0))]
    )
    rows = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        one,
        compulsory=Bytes(compulsory),
        reps=1,
        floor=floor,
    )
    assert float(rows[0].traffic_ratio) == pytest.approx(0.967, rel=1e-3)
    assert rows[0].roofline_class == UNJUDGED
    # The percentage still prints. It is the class that is withheld, so a reader
    # cannot quote a bandwidth share as a verdict.
    assert float(rows[0].floor_pct) > 0.0


def test_a_site_that_reached_dram_for_nothing_is_unjudged(
    floor: DramTimeFloor,
) -> None:
    # The extreme of the cache-served case, and a real one: a stage whose operands
    # are a few kilobytes is served entirely out of L2 and its DRAM counters read
    # zero. The floor is a time per byte moved, so at zero bytes there is no
    # denominator and the row carries no class rather than a division.
    compulsory = 24 * 1024
    one = census_pass(
        [(DECODE, counters(duration_us=2.7, read_bytes=0.0, write_bytes=0.0))]
    )
    rows = stage_rows(
        PREP,
        FUSION_CANDIDATE[PREP],
        one,
        compulsory=Bytes(compulsory),
        reps=1,
        floor=floor,
    )
    assert float(rows[0].traffic_ratio) == 0.0
    assert rows[0].roofline_class == UNJUDGED
    assert float(rows[0].floor_pct) == 0.0
    assert float(rows[0].achieved_gbs) == 0.0
    assert float(rows[0].floor_gbs) == 0.0
    assert not rows[0].contaminated


def test_one_site_cannot_buy_a_verdict_for_another(floor: DramTimeFloor) -> None:
    # The recurrence stage launches two kernels and one of them moves nothing. On a
    # stage-level ratio the big site's 1.0012x declared the small one DRAM-bound at
    # 197.57% of a floor it never reached, which is the kernel-wide-average failure
    # in miniature. Each site is tested against the stage's compulsory figure on its
    # own traffic.
    compulsory = 8 * 1024 * 1024
    carrier = counters(
        duration_us=law_us(compulsory),
        read_bytes=compulsory,
        write_bytes=0.0,
        grid=18432,
    )
    bystander = counters(
        duration_us=2.6, read_bytes=0.0, write_bytes=170.0, grid=128, registers=16
    )
    rows = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        census_pass([(DECODE, carrier), (DECODE, bystander)]),
        compulsory=Bytes(compulsory),
        reps=1,
        floor=floor,
    )
    by_grid = {row.site.grid: row for row in rows}
    assert float(by_grid[18432].traffic_ratio) == pytest.approx(1.0)
    assert by_grid[18432].roofline_class == f"{DRAM_BOUND} pass"
    assert float(by_grid[128].traffic_ratio) < 1e-4
    assert by_grid[128].roofline_class == UNJUDGED


def test_stage_rows_refuse_a_pass_missing_a_metric_a_verdict_rests_on(
    floor: DramTimeFloor,
) -> None:
    kept = tuple(m for m in CENSUS_TABLE.metrics if m != "dram__bytes_read.sum")
    one = census_pass(
        [(DECODE, counters(duration_us=1.0, read_bytes=0.0, write_bytes=1024.0))],
        metrics=kept,
    )
    assert "dram__bytes_read.sum" in one.missing_metrics
    with pytest.raises(RuntimeError, match="did not report"):
        stage_rows(
            RECURRENCE,
            FUSION_CANDIDATE[RECURRENCE],
            one,
            compulsory=Bytes(1024),
            reps=1,
            floor=floor,
        )


def test_stage_rows_refuse_a_window_they_cannot_divide_by(
    floor: DramTimeFloor,
) -> None:
    one = census_pass(
        [(DECODE, counters(duration_us=1.0, read_bytes=1024.0, write_bytes=0.0))]
    )
    with pytest.raises(ValueError, match="reps must be positive"):
        stage_rows(
            RECURRENCE,
            FUSION_CANDIDATE[RECURRENCE],
            one,
            compulsory=Bytes(1024),
            reps=0,
            floor=floor,
        )


def test_sum_by_site_separates_one_kernel_name_at_two_geometries() -> None:
    # The value and the key convolution are one kernel launched twice at two grids,
    # and so are the two projections. A kernel-wide average over the pair would
    # hide whichever of them is the outlier.
    value = counters(duration_us=4.0, read_bytes=2048.0, write_bytes=1024.0, grid=72)
    keys = counters(duration_us=1.0, read_bytes=512.0, write_bytes=256.0, grid=12)
    sites = sum_by_site(census_pass([(CONV, value), (CONV, keys)]))
    assert set(sites) == {
        Site(kernel=CONV, grid=72, block=256),
        Site(kernel=CONV, grid=12, block=256),
    }
    assert sites[Site(CONV, 72, 256)]["launches"] == 1.0
    assert sites[Site(CONV, 72, 256)]["dram__bytes_read.sum"] == pytest.approx(2048.0)
    assert sites[Site(CONV, 12, 256)]["dram__bytes_read.sum"] == pytest.approx(512.0)


def test_sum_by_site_sums_counters_and_averages_rates() -> None:
    # A counter over a launch adds across launches; a hit rate and a register count
    # do not, and summing either reports a figure no launch had.
    launch = counters(
        duration_us=2.0,
        read_bytes=1024.0,
        write_bytes=1024.0,
        read_hit_pct=50.0,
        write_hit_pct=50.0,
        registers=48,
    )
    site = sum_by_site(census_pass([(DECODE, launch)] * 4))[Site(DECODE, 96, 256)]
    assert site["launches"] == 4.0
    assert site["dram__bytes_read.sum"] == pytest.approx(4 * 1024.0)
    assert site["lts__t_sector_op_read_hit_rate.pct"] == pytest.approx(50.0)
    assert site["launch__registers_per_thread"] == pytest.approx(48.0)
    assert site["launch__grid_size"] == pytest.approx(96.0)


def test_a_site_reports_its_own_coalescing_and_its_own_spill(
    floor: DramTimeFloor,
) -> None:
    # A ratio taken over heterogeneous requests hides its outlier, so sectors per
    # request is a per-site figure. A spilled kernel is reported spilled rather than
    # scored as if the local traffic were not there.
    moved = 1 << 20
    one = census_pass(
        [
            (
                DECODE,
                counters(
                    duration_us=law_us(moved),
                    read_bytes=moved,
                    write_bytes=0.0,
                    load_requests=1000.0,
                    load_sectors=12000.0,
                    local_load_sectors=64.0,
                ),
            )
        ]
    )
    row = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        one,
        compulsory=Bytes(moved),
        reps=1,
        floor=floor,
    )[0]
    assert float(row.sectors_per_load_request) == pytest.approx(12.0)
    assert float(row.sectors_per_store_request) == 0.0
    assert row.local_sector_count == 64
    assert row.spilled


# ---------------------------------------------------------------------------
# contamination
# ---------------------------------------------------------------------------


def test_the_residual_is_zero_when_the_l2_traffic_explains_the_dram_bytes() -> None:
    # A kernel cannot read DRAM it did not miss on, and a line it dirtied leaves on
    # eviction whether the store hit or not. Both terms carry distinct values and
    # distinct hit rates, so no argument of the join is interchangeable with another:
    # a clean pass sits at zero and either swap moves off it.
    read_sectors, write_sectors = 4096.0, 512.0
    read_hit, write_hit = 25.0, 90.0
    moved = Bytes(
        int(SECTOR_BYTES * (read_sectors * (1.0 - read_hit / 100.0) + write_sectors))
    )
    assert float(
        contamination_residual_pct(
            moved, read_sectors, write_sectors, read_hit, write_hit
        )
    ) == pytest.approx(0.0, abs=1e-9)
    sectors_swapped = contamination_residual_pct(
        moved, write_sectors, read_sectors, read_hit, write_hit
    )
    rates_swapped = contamination_residual_pct(
        moved, read_sectors, write_sectors, write_hit, read_hit
    )
    assert float(sectors_swapped) == pytest.approx(-25.0)
    assert float(rates_swapped) == pytest.approx(74.29, abs=0.01)


def test_the_residual_names_traffic_that_is_not_this_kernels() -> None:
    # dram__bytes is device-wide, so a co-resident process's traffic inflates it
    # while nothing in the kernel's own L2 counters moves. That gap is the residual.
    sectors = 1024.0
    own = SECTOR_BYTES * sectors
    got = contamination_residual_pct(Bytes(int(own * 2)), sectors, 0.0, 0.0, 0.0)
    assert float(got) == pytest.approx(50.0)
    assert float(got) > CONTAMINATION_CEILING_PCT


def test_a_write_back_hit_still_leaves_for_dram() -> None:
    # The failure this closes: the miss-rate form applied to the write side declared
    # the recurrence kernel void at 49.79% while its DRAM figure agreed with the
    # compulsory model to 0.1%. L2 is write-back, so a store that finds its line
    # resident counts as a hit and the dirty line still goes out. These are that
    # kernel's own counters at B 128: 4,511,230 write sectors at a 99.91% hit rate
    # against 141,843,072 DRAM bytes written, which the miss rate explains 129,920 of.
    got = contamination_residual_pct(Bytes(141_843_072), 0.0, 4_511_230.0, 0.0, 99.91)
    assert float(got) == pytest.approx(-1.77, abs=0.01)
    assert float(got) < CONTAMINATION_CEILING_PCT
    stale = SECTOR_BYTES * 4_511_230.0 * (1.0 - 99.91 / 100.0)
    assert stale / 141_843_072 < 0.01


def test_the_residual_is_zero_on_a_kernel_that_moved_nothing() -> None:
    # Zero over zero is not a hundred percent contaminated.
    assert float(contamination_residual_pct(Bytes(0), 0.0, 0.0, 0.0, 0.0)) == 0.0


def test_a_contaminated_row_is_reported_void(floor: DramTimeFloor) -> None:
    moved = 1 << 20
    launch = counters(duration_us=law_us(moved), read_bytes=moved, write_bytes=0.0)
    launch["lts__t_sectors_op_read.sum"] /= 4.0
    row = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        census_pass([(DECODE, launch)]),
        compulsory=Bytes(moved),
        reps=1,
        floor=floor,
    )[0]
    assert float(row.residual_pct) == pytest.approx(75.0)
    assert row.contaminated


# ---------------------------------------------------------------------------
# artifacts, rendering, and the command line
# ---------------------------------------------------------------------------


def test_write_cell_lands_whole_and_leaves_no_partial_file(tmp_path: Path) -> None:
    # A census that dies mid-shape has to leave the cells it banked readable.
    key = cell_key("traffic", "acceptance", 128, "bf16")
    assert key == "traffic-acceptance-b128-bf16"
    path = write_cell(tmp_path / "cells", key, {"rows": [1, 2], "floor": None})
    assert path.name == f"{key}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "rows": [1, 2],
        "floor": None,
    }
    assert not list(path.parent.glob("*.tmp"))


def test_payload_reaches_every_nested_record(floor: DramTimeFloor) -> None:
    moved = 1 << 20
    one = census_pass(
        [(DECODE, counters(duration_us=law_us(moved), read_bytes=moved, write_bytes=0))]
    )
    rows = stage_rows(
        RECURRENCE,
        FUSION_CANDIDATE[RECURRENCE],
        one,
        compulsory=Bytes(moved),
        reps=1,
        floor=floor,
    )
    data = payload({"rows": rows, "floor": floor})
    assert data["rows"][0]["site"]["kernel"] == DECODE
    assert data["floor"]["asymptotic_gbs"] == pytest.approx(RATE_GBS, rel=1e-6)
    assert json.loads(json.dumps(data))["rows"][0]["stage"] == RECURRENCE


def test_traffic_table_renders_one_row_per_site(floor: DramTimeFloor) -> None:
    moved = 1 << 20
    one = census_pass(
        [
            (
                CONV,
                counters(duration_us=law_us(moved), read_bytes=moved, write_bytes=0),
            ),
            (
                CONV,
                counters(
                    duration_us=law_us(moved // 4),
                    read_bytes=moved // 4,
                    write_bytes=0,
                    grid=12,
                ),
            ),
        ]
    )
    rows = stage_rows(
        VALUE_CONV,
        FUSION_CANDIDATE[VALUE_CONV],
        one,
        compulsory=Bytes(moved),
        reps=1,
        floor=floor,
    )
    lines = traffic_table(rows).strip().splitlines()
    assert len(lines) == 2 + len(rows)
    assert lines[0].count("|") == lines[1].count("|") == lines[2].count("|")
    assert FUSION_CANDIDATE[VALUE_CONV] in lines[2]


def test_target_argv_hands_the_child_the_window_the_parent_divides_by() -> None:
    # A parent that divided by a different window than the child ran would scale
    # every per-step figure by that ratio and say nothing about it.
    args = parse_args(
        [
            "--pass",
            "traffic",
            "--shape",
            "wide",
            "--batch",
            "32",
            "--capture-iters",
            "7",
        ]
    )
    argv = target_argv(args, PREP)
    assert argv[1].endswith("decode_census.py")
    assert argv[argv.index("--pass") + 1] == "target"
    assert argv[argv.index("--stage") + 1] == PREP
    assert argv[argv.index("--capture-iters") + 1] == "7"
    assert argv[argv.index("--batch") + 1] == "32"
    assert argv[argv.index("--shape") + 1] == "wide"
    whole = target_argv(args, ALL_STAGES)
    assert whole[whole.index("--stage") + 1] == ALL_STAGES


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--batch", "0"),
        ("--iters", "0"),
        ("--capture-iters", "0"),
        ("--warmup", "0"),
    ],
)
def test_main_refuses_an_axis_no_measurement_can_use(flag: str, value: str) -> None:
    # Each of the four is a divisor or a precondition, and a zero in any of them
    # produces a report rather than an error unless it is refused here.
    with pytest.raises(ValueError, match=flag):
        main(["--pass", "provenance", flag, value])


def test_the_thresholds_are_stated_not_derived() -> None:
    # Both are thresholds rather than measurements. They are held here so an edit
    # shows up as a test change rather than as a silent shift in every verdict.
    assert CONTAMINATION_CEILING_PCT == 10.0
    assert SECTOR_BYTES == 32
    assert "unjudged" in UNJUDGED
