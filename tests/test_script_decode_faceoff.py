"""The decode faceoff driver: its grid, its matching, its state model, its verdict.

No test here builds either architecture or takes a timed sample. What is under test is
the part of the driver that decides what would be measured and what a measurement means,
which is the part that can make a table dishonest while every kernel runs correctly:
a grid that enumerates a shape one side cannot run, a verdict quantified over the cells
that happened to survive, a ``d_state`` mismatch that never reaches the page.

One test per failure mode. A threshold is exercised at the value itself and at the first
value past it, because inclusive and exclusive are different code paths; a second ratio
inside the same interval is the same path and is not a test.

The state model is checked against :meth:`slinoss.state.MixerState.allocate` rather than
against arithmetic copied out of the driver, so a shape the driver got wrong fails here
instead of agreeing with itself. Mamba3's side has no allocator to check against on a host
without the package, so its terms are spelled out from the shapes
``allocate_inference_cache`` returns.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Final

import pytest
import torch

from scripts.perf.decode_faceoff import (
    BANK_SCHEMA,
    CLOSE_SETTLE_S,
    COMPETITIVE,
    D_HEAD,
    DECISIVE,
    DISCLOSURES,
    DOMINATES,
    EAGER,
    EXCLUSIVE_WITNESS,
    FIT_AGREEMENT_PCT,
    FP32_DISCLOSURE,
    G1,
    GHEAD,
    GMID,
    GRAPH,
    GRAPH_CELLS_PER_PROCESS,
    KERNEL_PATH,
    MAMBA_D_STATES,
    MAMBA_PACKAGE,
    MARGIN_PCT,
    MATCHED_PROVENANCE,
    MEASURED_HEAD_ROWS,
    MIMO,
    MIXED_PATH,
    NEITHER,
    NESTED,
    NO_WITNESS,
    PRIOR_DRAM_RATE_GBS,
    RECURRENCE,
    REFERENCE,
    REFERENCE_PATH,
    REPLAY_IDLE_PCT_HIGH,
    RESIDENCY_REPLICATES,
    RESIDENCY_WITNESS,
    ROUTING_DISCLOSURE,
    SAMPLE_COUNT_DISCLOSURE_LABEL,
    SISO,
    SOURCES_ORIGIN,
    TIMER_QUANTUM_US,
    VOID_SUFFIX,
    WHOLE_STEP,
    Cell,
    FloorPair,
    GridPoint,
    Liveness,
    MatchReport,
    Resolved,
    StateBytes,
    Task,
    Witness,
    admit,
    agrees_within_half_widths,
    cell_from_record,
    cell_record,
    closing_probe,
    competitor_provenance,
    crossover,
    decisiveness,
    default_resource,
    defers,
    dependency_set,
    dispersion_lines,
    enumerate_grid,
    file_manifest,
    fit_cross_check,
    floor_pair,
    git_commit,
    group_count,
    group_verdicts,
    installed_version,
    judgeable,
    kernel_gate,
    literal_head_rejection,
    main,
    mamba_rejection,
    mamba_state_bytes,
    match_shapes,
    measure_replicated,
    moved_bytes,
    nearest_mamba_d_state,
    order_statistic_us,
    parse_args,
    path_class,
    poisons,
    provenance,
    read_bank,
    read_voids,
    regime,
    render,
    rope_angles,
    sample_void,
    slinoss_rejection,
    slinoss_state_bytes,
    source_digest,
    tasks,
    unjudged,
    unresolved,
    verdict,
    witness_mark,
    write_cell,
    write_void,
)
from slinoss.config import SLinOSSConfig
from slinoss.perf.ceiling import DramTimeFloor
from slinoss.perf.device import Contention
from slinoss.perf.units import (
    Bytes,
    Count,
    GBPerSecond,
    Mebibytes,
    Microseconds,
    Percent,
)
from slinoss.state import MixerState

PRIMARY: Final[tuple[int, ...]] = (1, 8, 32, 64, 128)
"""The driver's primary batches, spelled here so a test asserts against a literal."""


def flat(value: float) -> dict[int, float]:
    """One ratio at every primary batch.

    Args:
        value: The ratio.

    Returns:
        The map :func:`verdict` consumes.
    """
    return dict.fromkeys(PRIMARY, value)


def a_config(*, d_state: int = 96, key_conv: bool = True) -> SLinOSSConfig:
    """A legal one-layer shape, narrow enough to allocate on a CPU.

    Args:
        d_state: ``3N``.
        key_conv: Whether the key convolution runs.

    Returns:
        The config. ``d_model 256`` at ``expand 2`` and ``d_head 64`` is eight heads, so
        the three sharing cases are three distinct group counts.
    """
    return SLinOSSConfig(
        d_model=256,
        d_state=d_state,
        expand=2.0,
        d_head=D_HEAD,
        n_groups=1,
        d_conv=4,
        key_conv=key_conv,
        n_layers=1,
        vocab_size=None,
    )


def a_point(batch: int) -> GridPoint:
    """The fabricated cell's shape at one batch. The nearest matched pair."""
    return GridPoint(
        d_model=256,
        dtype_name="bf16",
        batch=batch,
        slinoss_d_state=144,
        mamba_d_state=128,
        sharing=G1,
        mode=SISO,
    )


def a_fit(*, gbs: float = 684.640, fixed_us: float = 4.1877) -> DramTimeFloor:
    """A fitted DRAM time law, with no samples behind it.

    Args:
        gbs: The asymptotic rate.
        fixed_us: The size-independent term.

    Returns:
        The law. Constructed rather than measured, so the floor arithmetic is under test and
        not the card.
    """
    return DramTimeFloor(
        label="fabricated",
        l2_bytes=Bytes(6 * 1024 * 1024),
        fixed_duration_us=Microseconds(fixed_us),
        asymptotic_gbs=GBPerSecond(gbs),
        max_residual_pct=Percent(0.2),
        copies=(),
    )


def a_floor(*, available: bool = True) -> FloorPair:
    """A fabricated floor pair, with literal bytes and literal floors.

    Args:
        available: Whether the row carries a floor at all.

    Returns:
        The pair. SLinOSS moves more bytes and sits closer to its floor, which is the case
        a report has to state rather than average away.
    """
    if not available:
        return FloorPair(
            available=False,
            slinoss_moved_bytes=1_000,
            slinoss_floor_us=Microseconds(0.0),
            slinoss_x_floor=0.0,
            mamba_moved_bytes=900,
            mamba_floor_us=Microseconds(0.0),
            mamba_x_floor=0.0,
            detail="no floor: the row is host",
        )
    return FloorPair(
        available=True,
        slinoss_moved_bytes=1_000,
        slinoss_floor_us=Microseconds(50.0),
        slinoss_x_floor=2.0,
        mamba_moved_bytes=900,
        mamba_floor_us=Microseconds(40.0),
        mamba_x_floor=5.0,
        detail="slinoss is 2.00x its own floor; mamba3 is 5.00x its own floor",
    )


def a_cell(
    *,
    batch: int = 128,
    ratio: float = 0.5,
    resolved: Resolved | None = None,
    floor: FloorPair | None = None,
    slinoss_us: float = 100.0,
    resolution_pct: float = 0.5,
    mamba_resolution_pct: float = 0.4,
    witness: Witness | None = None,
    boundary: str = RECURRENCE,
    execution: str = GRAPH,
    iters: int = 1_000,
    samples: tuple[float, ...] = (),
) -> Cell:
    """A fabricated measured cell, for the parts of the driver downstream of timing.

    Args:
        batch: The cell's batch.
        ratio: SLinOSS over Mamba3.
        resolved: What each stage resolved to. A live kernel path by default.
        floor: The floor pair. One that applies, by default.
        slinoss_us: SLinOSS's median. Mamba3's follows from the ratio.
        resolution_pct: Half-width on SLinOSS's median, as a percent of it.
        mamba_resolution_pct: Half-width on Mamba3's median, as a percent of it.
        witness: How the row earned its card. Unwitnessed by default.
        boundary: The measured boundary.
        execution: Eager or graph.
        iters: Timed iterations behind the two medians. Zero marks a record written
            before the field existed.
        samples: SLinOSS's per-iteration samples. Mamba3's are the same divided by the
            ratio, so the two arms differ in the block under test. Empty by default, which
            is what a record written before the field carries.

    Returns:
        The cell. Every duration is a literal, so a table test asserts on the driver's
        formatting rather than on a machine.
    """
    point = a_point(batch)
    return Cell(
        point=point,
        boundary=boundary,
        execution=execution,
        resolved=resolved
        or Resolved(names=("decode=cute",), path=KERNEL_PATH, detail="live"),
        regime="dram",
        slinoss_duration_us=Microseconds(slinoss_us),
        slinoss_resolution_pct=Percent(resolution_pct),
        slinoss_spread_pct=Percent(2.0),
        slinoss_samples_duration_us=tuple(Microseconds(one) for one in samples),
        mamba_duration_us=Microseconds(slinoss_us / ratio),
        mamba_resolution_pct=Percent(mamba_resolution_pct),
        mamba_spread_pct=Percent(1.5),
        mamba_samples_duration_us=tuple(Microseconds(one / ratio) for one in samples),
        ratio=ratio,
        paired_delta_us=Microseconds(-100.0),
        paired_low_us=Microseconds(-102.0),
        paired_high_us=Microseconds(-98.0),
        paired_resolves=True,
        match=match_shapes(
            point.config, dtype=torch.bfloat16, batch=batch, mamba_d_state=128
        ),
        floor=a_floor() if floor is None else floor,
        witness=NO_WITNESS if witness is None else witness,
        iters=iters,
    )


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


def test_the_grid_emits_only_shapes_legal_on_both_sides() -> None:
    """Nothing downstream re-checks a shape, so an illegal cell would reach a build."""
    grid = enumerate_grid()
    assert grid.points
    for point in grid.points:
        assert (
            slinoss_rejection(
                d_model=point.d_model,
                d_state=point.slinoss_d_state,
                d_head=D_HEAD,
                n_groups=point.n_groups,
            )
            is None
        )
        assert (
            mamba_rejection(
                d_model=point.d_model,
                d_state=point.mamba_d_state,
                d_head=D_HEAD,
                n_groups=point.n_groups,
            )
            is None
        )


def test_the_grid_names_every_refusal_rather_than_dropping_it() -> None:
    """A refusal that never reaches the table is a verdict over an unstated set.

    ``3N 128`` is illegal for SLinOSS and legal for Mamba3, and ``d_state 96`` is the
    reverse, so one enumeration exercises a refusal from each side.
    """
    grid = enumerate_grid(
        d_models=(512,),
        dtype_names=("bf16",),
        batches=(1,),
        slinoss_d_states=(128, 144),
        mamba_d_states=(96, 128),
        sharings=(G1,),
        modes=(SISO,),
    )
    refused = " ".join(one.detail for one in grid.rejections)
    assert "d_state 128 is not a positive multiple of 48" in refused
    assert "mamba3_step_fn asserts" in refused
    assert {point.slinoss_d_state for point in grid.points} == {144}
    assert {point.mamba_d_state for point in grid.points} == {128}


def test_an_illegal_d_state_is_refused_by_name_on_each_side() -> None:
    """Each side's rule refuses the other side's legal width, and says which field."""
    sl = slinoss_rejection(d_model=512, d_state=128, d_head=D_HEAD, n_groups=1)
    assert sl is not None
    assert "d_state 128" in sl
    assert "multiple of 48" in sl
    m3 = mamba_rejection(d_model=512, d_state=144, d_head=D_HEAD, n_groups=1)
    assert m3 is not None
    assert "d_state 144" in m3
    assert not MAMBA_D_STATES & {96, 144, 192}


def test_a_d_head_off_the_measured_mma_list_is_refused_by_name() -> None:
    """The config's multiple-of-16 rule is necessary and not sufficient.

    ``d_head 32`` is a multiple of 16 and outside the measured N-mode list, so a grid
    trusting the config alone would enumerate a shape the tiled MMA refuses to compile.
    """
    assert 32 % 16 == 0
    assert 32 not in MEASURED_HEAD_ROWS
    off_list = slinoss_rejection(d_model=512, d_state=96, d_head=32, n_groups=1)
    assert off_list is not None
    assert "d_head 32" in off_list
    assert "MMA N-mode" in off_list
    off_multiple = slinoss_rejection(d_model=512, d_state=96, d_head=24, n_groups=1)
    assert off_multiple is not None
    assert "multiple of 16" in off_multiple


def test_every_grid_point_carries_the_shipped_head_width() -> None:
    """Compiling is forward legality and is not trainability, and the two part at
    ``d_head 128``: at ``3N = 144`` that width has no legal backward at the shipped
    ``chunk_size 64``. A grid point carrying it would owe the chunk size at which the shape
    trains, so the guard is that no grid point carries it at all."""
    assert 128 in MEASURED_HEAD_ROWS
    for point in enumerate_grid().points:
        assert point.config.d_head == D_HEAD
        assert f"/P={D_HEAD}/" in point.shape_class


def test_a_group_count_that_does_not_divide_the_heads_is_refused_on_both_sides() -> (
    None
):
    """A group holds a whole number of heads, and Mamba3 admits only 1 or nheads."""
    sl = slinoss_rejection(d_model=512, d_state=96, d_head=D_HEAD, n_groups=5)
    assert sl is not None
    assert "n_groups 5" in sl
    m3 = mamba_rejection(d_model=512, d_state=128, d_head=D_HEAD, n_groups=4)
    assert m3 is not None
    assert "ngroups 4" in m3
    assert mamba_rejection(d_model=512, d_state=128, d_head=D_HEAD, n_groups=16) is None


def test_the_literal_four_head_shape_is_refused_with_the_d_head_it_forces() -> None:
    """``H4/G1`` is a sharing case here, and the reading is stated, not assumed."""
    for d_model in (512, 1024, 2048):
        detail = literal_head_rejection(d_model)
        assert detail is not None
        assert f"d_head {round(2.0 * d_model) // 4}" in detail
        assert f"sharing case at d_head {D_HEAD}" in detail
    assert literal_head_rejection(128) is None


def test_an_unknown_axis_raises_rather_than_enumerating_nothing() -> None:
    """An empty grid reads as a grid with no legal cell, which is a different claim."""
    with pytest.raises(ValueError, match="unknown dtype"):
        enumerate_grid(dtype_names=("fp16",))
    with pytest.raises(ValueError, match="unknown sharing"):
        enumerate_grid(sharings=("H4",))
    with pytest.raises(ValueError, match="unknown mode"):
        enumerate_grid(modes=("mimo4",))


def test_group_count_spells_the_three_sharing_cases() -> None:
    """One pair for the layer, one per four heads, one per head; never zero."""
    assert group_count(G1, 32) == 1
    assert group_count(GMID, 32) == 8
    assert group_count(GHEAD, 32) == 32
    assert group_count(GMID, 2) == 1
    with pytest.raises(ValueError, match="unknown sharing"):
        group_count("G2", 32)


def test_the_mimo_mode_carries_the_rank_and_siso_does_not() -> None:
    """``mimo_rank`` sizes ``k_state``, so a mode that lost it would understate the state
    bytes it is compared on."""
    siso = GridPoint(
        d_model=512,
        dtype_name="bf16",
        batch=1,
        slinoss_d_state=144,
        mamba_d_state=128,
        sharing=G1,
        mode=SISO,
    )
    mimo = GridPoint(
        d_model=512,
        dtype_name="bf16",
        batch=1,
        slinoss_d_state=144,
        mamba_d_state=128,
        sharing=G1,
        mode=MIMO,
    )
    assert siso.rank == 1
    assert mimo.rank == 4
    assert (
        mamba_state_bytes(
            d_model=512,
            d_state=128,
            d_head=D_HEAD,
            dtype=torch.bfloat16,
            rank=mimo.rank,
        ).total_bytes
        > mamba_state_bytes(
            d_model=512,
            d_state=128,
            d_head=D_HEAD,
            dtype=torch.bfloat16,
            rank=siso.rank,
        ).total_bytes
    )


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_the_match_report_states_the_d_state_mismatch() -> None:
    """The nearest pair is still a mismatch, and the row says so with both widths."""
    report = match_shapes(
        a_config(d_state=144), dtype=torch.bfloat16, batch=8, mamba_d_state=128
    )
    assert report.d_state_matched is False
    assert "d_state NOT matched" in report.detail
    assert "3N=144" in report.detail
    assert "d_state=128" in report.detail
    assert report.d_state_ratio == pytest.approx(144 / 128)
    assert "State bytes per token per layer" in report.detail
    assert report.held == ("d_model", "dtype", "batch", "layers", "expand", "d_head")


def test_the_match_report_reports_an_unmatched_parameter_count() -> None:
    """Parameter count is held equal, so a difference is a line on the row."""
    unequal = match_shapes(
        a_config(d_state=144),
        dtype=torch.bfloat16,
        batch=8,
        mamba_d_state=128,
        slinoss_param_count=1_000,
        mamba_param_count=1_200,
    )
    assert unequal.param_matched is False
    assert "Parameter count NOT matched" in unequal.detail
    equal = match_shapes(
        a_config(d_state=144),
        dtype=torch.bfloat16,
        batch=8,
        mamba_d_state=128,
        slinoss_param_count=1_000,
        mamba_param_count=1_000,
    )
    assert equal.param_matched is True
    assert "Parameter count NOT matched" not in equal.detail


def test_nearest_mamba_d_state_breaks_the_absolute_tie_by_ratio() -> None:
    """96 is 32 from both 64 and 128; a difference would tie and pick arbitrarily."""
    assert abs(96 - 64) == abs(96 - 128)
    assert nearest_mamba_d_state(96) == 128
    assert nearest_mamba_d_state(144) == 128
    assert nearest_mamba_d_state(48) == 64
    with pytest.raises(ValueError, match="must be positive"):
        nearest_mamba_d_state(0)
    with pytest.raises(ValueError, match="no candidate"):
        nearest_mamba_d_state(96, legal=())


def test_the_match_report_is_a_named_record_not_a_positional_tuple() -> None:
    """Every row's provenance is read by field name downstream."""
    report = match_shapes(a_config(), dtype=torch.bfloat16, batch=1, mamba_d_state=128)
    assert isinstance(report, MatchReport)
    assert report.d_model == 256
    assert report.dtype == str(torch.bfloat16)
    assert report.batch == 1
    assert report.layers == 1


# --------------------------------------------------------------------------
# State bytes per token
# --------------------------------------------------------------------------


def test_slinoss_state_bytes_are_the_allocated_buffers() -> None:
    """Checked against the allocator, not against arithmetic the driver also owns."""
    config = a_config(d_state=144)
    state = MixerState.allocate(config, 1, device="cpu", dtype=torch.bfloat16)
    counted = slinoss_state_bytes(config, dtype=torch.bfloat16)
    allocated = sum(
        buffer.numel() * buffer.element_size()
        for buffer in (state.ssm, state.conv, state.keys, state.b_prev, state.u_prev)
    )
    assert int(counted.total_bytes) == allocated
    assert int(counted.recurrent_bytes) == state.ssm.numel() * 4
    assert counted.conv_buffer_count == 2


def test_a_state_term_the_step_never_reads_is_left_out_of_the_count() -> None:
    """``keys`` is allocated whether or not the key convolution runs, and counts only
    when it does; the divergence from the allocator there is deliberate."""
    off = a_config(key_conv=False)
    counted = slinoss_state_bytes(off, dtype=torch.bfloat16)
    state = MixerState.allocate(off, 1, device="cpu", dtype=torch.bfloat16)
    keys_bytes = state.keys.numel() * state.keys.element_size()
    assert keys_bytes > 0
    assert int(counted.total_bytes) + keys_bytes == sum(
        buffer.numel() * buffer.element_size()
        for buffer in (state.ssm, state.conv, state.keys, state.b_prev, state.u_prev)
    )
    assert counted.conv_buffer_count == 1


def test_mamba_state_bytes_are_the_documented_shapes_and_carry_no_conv() -> None:
    """Two float32 pins that do not shrink at bf16, and zero convolution state."""
    counted = mamba_state_bytes(
        d_model=256, d_state=128, d_head=D_HEAD, dtype=torch.bfloat16, rank=4
    )
    heads = 512 // D_HEAD
    assert int(counted.recurrent_bytes) == 4 * heads * D_HEAD * 128
    assert int(counted.carry_bytes) == (
        4 * heads * rope_angles(128) + 2 * 4 * heads * 128 + 2 * heads * D_HEAD
    )
    assert int(counted.conv_bytes) == 0
    assert counted.conv_buffer_count == 0
    at_fp32 = mamba_state_bytes(
        d_model=256, d_state=128, d_head=D_HEAD, dtype=torch.float32, rank=4
    )
    assert at_fp32.recurrent_bytes == counted.recurrent_bytes


def test_rope_angles_decrement_an_odd_span_before_halving() -> None:
    """The one Mamba3 state term that is not a product of the widths."""
    assert rope_angles(128) == 32
    assert rope_angles(64) == 16
    assert rope_angles(50, fraction=0.5) == 12
    assert rope_angles(128, fraction=1.0) == 64


def test_the_state_bytes_record_is_the_sum_of_its_terms() -> None:
    """The total is reported beside the itemization, so a term added to one and not the
    other would be invisible."""
    for counted in (
        slinoss_state_bytes(a_config(), dtype=torch.bfloat16),
        mamba_state_bytes(
            d_model=256, d_state=64, d_head=D_HEAD, dtype=torch.bfloat16, rank=1
        ),
    ):
        assert isinstance(counted, StateBytes)
        assert int(counted.total_bytes) == (
            int(counted.recurrent_bytes)
            + int(counted.conv_bytes)
            + int(counted.carry_bytes)
        )


# --------------------------------------------------------------------------
# Compulsory traffic and the floor
# --------------------------------------------------------------------------


def test_the_recurrence_boundary_moves_no_weights_and_the_whole_step_does() -> None:
    """Charging the parameter map to the recurrence would credit SLinOSS with a floor it
    never pays there, and dropping it from the whole step would flatter both sides."""
    state = slinoss_state_bytes(a_config(), dtype=torch.bfloat16)
    recurrence = moved_bytes(
        boundary=RECURRENCE,
        param_bytes=4_000,
        state=state,
        batch=8,
        d_model=256,
        dtype=torch.bfloat16,
    )
    whole = moved_bytes(
        boundary=WHOLE_STEP,
        param_bytes=4_000,
        state=state,
        batch=8,
        d_model=256,
        dtype=torch.bfloat16,
    )
    assert recurrence.weight_bytes == 0
    assert recurrence.activation_bytes == 0
    assert whole.weight_bytes == 4_000
    assert whole.activation_bytes == 2 * 8 * 256 * 2
    assert recurrence.state_bytes == whole.state_bytes == 2 * 8 * int(state.total_bytes)
    assert whole.total_bytes == sum(whole[:3])
    assert recurrence.total_bytes == sum(recurrence[:3])


def test_traffic_refuses_an_unknown_boundary_and_an_empty_batch() -> None:
    """A zero batch would divide the floor by nothing and report an infinite ratio."""
    state = slinoss_state_bytes(a_config(), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unknown boundary"):
        moved_bytes(
            boundary="both",
            param_bytes=0,
            state=state,
            batch=1,
            d_model=256,
            dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="batch must be positive"):
        moved_bytes(
            boundary=WHOLE_STEP,
            param_bytes=0,
            state=state,
            batch=0,
            d_model=256,
            dtype=torch.bfloat16,
        )


def test_no_floor_is_offered_below_the_cache_crossover_or_without_a_fit() -> None:
    """The two ways a floor can fail to exist read differently, and neither may print a
    number: a footprint under L2 gets no roofline verdict, and a fit from another process
    does not price this card."""
    traffic = moved_bytes(
        boundary=WHOLE_STEP,
        param_bytes=1_000,
        state=slinoss_state_bytes(a_config(), dtype=torch.bfloat16),
        batch=1,
        d_model=256,
        dtype=torch.bfloat16,
    )
    for where, expected in (
        ("host", "the row is host"),
        ("sub_l2", "the row is sub_l2"),
    ):
        pair = floor_pair(
            regime_name=where,
            slinoss_bytes=traffic,
            mamba_bytes=traffic,
            slinoss_duration_us=Microseconds(100.0),
            mamba_duration_us=Microseconds(100.0),
            fit=a_fit(),
        )
        assert not pair.available
        assert expected in pair.detail
        assert pair.slinoss_x_floor == 0.0
    unfitted = floor_pair(
        regime_name="dram",
        slinoss_bytes=traffic,
        mamba_bytes=traffic,
        slinoss_duration_us=Microseconds(100.0),
        mamba_duration_us=Microseconds(100.0),
        fit=None,
    )
    assert not unfitted.available
    assert "none was fitted in this process" in unfitted.detail
    with pytest.raises(ValueError, match="unknown regime"):
        floor_pair(
            regime_name="l2",
            slinoss_bytes=traffic,
            mamba_bytes=traffic,
            slinoss_duration_us=Microseconds(100.0),
            mamba_duration_us=Microseconds(100.0),
            fit=None,
        )


def test_each_side_is_priced_against_its_own_footprint() -> None:
    """One shared floor would credit whichever side moves fewer bytes, so the two sides get
    two floors and the report states both."""
    fit = a_fit()
    small = moved_bytes(
        boundary=RECURRENCE,
        param_bytes=0,
        state=StateBytes(
            recurrent_bytes=Bytes(1_000),
            conv_bytes=Bytes(0),
            carry_bytes=Bytes(0),
            total_bytes=Bytes(1_000),
            conv_buffer_count=0,
        ),
        batch=32,
        d_model=256,
        dtype=torch.bfloat16,
    )
    large = moved_bytes(
        boundary=RECURRENCE,
        param_bytes=0,
        state=StateBytes(
            recurrent_bytes=Bytes(4_000),
            conv_bytes=Bytes(0),
            carry_bytes=Bytes(0),
            total_bytes=Bytes(4_000),
            conv_buffer_count=0,
        ),
        batch=32,
        d_model=256,
        dtype=torch.bfloat16,
    )
    pair = floor_pair(
        regime_name="dram",
        slinoss_bytes=large,
        mamba_bytes=small,
        slinoss_duration_us=Microseconds(20.0),
        mamba_duration_us=Microseconds(20.0),
        fit=fit,
    )
    assert pair.available
    assert pair.slinoss_moved_bytes > pair.mamba_moved_bytes
    assert float(pair.slinoss_floor_us) > float(pair.mamba_floor_us)
    # Equal durations, unequal footprints: the side moving more bytes is nearer its floor.
    assert pair.slinoss_x_floor < pair.mamba_x_floor
    assert "slinoss is" in pair.detail and "mamba3 is" in pair.detail


def test_the_fit_is_cross_checked_against_the_prior_one_and_never_replaced_by_it() -> (
    None
):
    """A rate is a property of the card, so a disagreement past the tolerance indicts the
    samples; the prior fit is printed beside this run's and never substituted for it."""
    assert "none taken in this process" in fit_cross_check(None)
    agrees = fit_cross_check(a_fit(gbs=PRIOR_DRAM_RATE_GBS))
    assert "agrees with" in agrees
    assert str(PRIOR_DRAM_RATE_GBS) in agrees
    assert "DISAGREES WITH" in fit_cross_check(
        a_fit(gbs=PRIOR_DRAM_RATE_GBS * (1.0 - 2.0 * FIT_AGREEMENT_PCT / 100.0))
    )


# --------------------------------------------------------------------------
# What ran, and in what regime
# --------------------------------------------------------------------------


def test_path_class_routes_by_the_operator_stage() -> None:
    """A native conv around a reference scan is still a measurement of torch."""
    assert path_class({"conv": "native", "scan": "cute", "tail": "cute"}) == KERNEL_PATH
    assert (
        path_class({"conv": "native", "scan": REFERENCE, "tail": "cute"})
        == REFERENCE_PATH
    )
    assert path_class({"conv": REFERENCE, "scan": "cute", "tail": "cute"}) == MIXED_PATH
    assert path_class({"decode": REFERENCE}) == REFERENCE_PATH
    # A record banked before routing names the operator stage chunked_scan, and a fallback
    # there is still
    # a fallback of the operator under test.
    assert path_class({"conv": "native", "chunked_scan": REFERENCE}) == REFERENCE_PATH


def test_the_two_boundaries_cross_out_of_l2_at_different_batches() -> None:
    """One crossover would grant the whole step a roofline verdict it has not earned, or
    withhold one the decode kernel has: the decode boundary moves state only and crosses at
    batch 2.8 to 5.7, the whole step streams weights and crosses at 10.6 to 24.4."""
    assert crossover(RECURRENCE) == (2.8, 5.7)
    assert crossover(WHOLE_STEP) == (10.6, 24.4)
    with pytest.raises(ValueError, match="unknown boundary"):
        crossover("both")
    # Batch 8 is above the decode crossover and below the whole step's, which is the whole
    # reason the two are separate.
    assert regime(batch=8, execution=GRAPH, boundary=RECURRENCE) == "dram"
    assert regime(batch=8, execution=GRAPH, boundary=WHOLE_STEP) == "sub_l2"
    # An omitted boundary withholds the verdict rather than granting it.
    assert regime(batch=8, execution=GRAPH) == "sub_l2"


def test_regime_separates_host_bound_from_sub_l2_and_dram() -> None:
    """Batch 1 eager is a Python comparison; batch 1 replayed is a cache-resident one."""
    assert regime(batch=1, execution=EAGER, boundary=WHOLE_STEP) == "host"
    assert regime(batch=1, execution=GRAPH, boundary=WHOLE_STEP) == "sub_l2"
    assert regime(batch=1, execution=GRAPH, boundary=RECURRENCE) == "sub_l2"
    assert regime(batch=32, execution=EAGER, boundary=WHOLE_STEP) == "dram"
    assert regime(batch=128, execution=GRAPH, boundary=WHOLE_STEP) == "dram"


def test_default_resource_names_the_regime_and_owes_a_profile_above_it() -> None:
    """A resource string carries its own provenance; above the whole step's crossover none
    is known and the empty string is what says so, while the decode kernel's own limit is
    already profiled and is named."""
    assert "host enqueue" in default_resource(
        path=KERNEL_PATH, batch=1, execution=EAGER, boundary=WHOLE_STEP
    )
    assert "weight bytes" in default_resource(
        path=KERNEL_PATH, batch=8, execution=GRAPH, boundary=WHOLE_STEP
    )
    assert (
        default_resource(
            path=KERNEL_PATH, batch=128, execution=GRAPH, boundary=WHOLE_STEP
        )
        == ""
    )
    named = default_resource(
        path=KERNEL_PATH, batch=128, execution=GRAPH, boundary=RECURRENCE
    )
    assert "DRAM bandwidth" in named
    assert "98.94%" in named and "not by this run" in named
    # The superseded profile has to stay named, and named as void, so a reader who has the
    # old 52.77% figure in hand learns it was retired rather than silently not finding it.
    assert "52.77%" in named and "void" in named
    assert "reference path" in default_resource(
        path=REFERENCE_PATH, batch=128, execution=GRAPH, boundary=WHOLE_STEP
    )


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def test_dominates_is_inclusive_at_the_ratio_and_lost_just_past_it() -> None:
    """Exactly 0.90 dominates; one point a hair above it does not."""
    at = verdict(
        flat(0.90),
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
    )
    assert at.word == DOMINATES
    assert at.caveats == ()
    past = flat(0.90)
    past[64] = 0.9001
    lost = verdict(past, shape_class="bf16/class", boundary=RECURRENCE, execution=GRAPH)
    assert lost.word == COMPETITIVE
    assert lost.worst_batch == 64


def test_competitive_is_inclusive_at_the_geomean_and_at_the_worst_point() -> None:
    """Both thresholds are boundaries, each exercised at the value and past it."""
    at_geomean = verdict(
        flat(1.10),
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
    )
    assert at_geomean.word == COMPETITIVE
    assert at_geomean.geomean_ratio == pytest.approx(1.10)
    past_geomean = verdict(
        flat(1.1001),
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        limiting_resource="named",
    )
    assert past_geomean.word == NEITHER

    at_worst = flat(0.95)
    at_worst[128] = 1.20
    on_the_line = verdict(
        at_worst, shape_class="bf16/class", boundary=RECURRENCE, execution=GRAPH
    )
    assert on_the_line.word == COMPETITIVE
    past_worst = dict(at_worst)
    past_worst[128] = 1.2001
    beyond = verdict(
        past_worst,
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        limiting_resource="named",
    )
    assert beyond.word == NEITHER
    assert beyond.geomean_ratio <= 1.10


def test_a_missing_primary_batch_refuses_every_positive_word() -> None:
    """Quantifying over the cells that happened to run is the failure mode."""
    partial = flat(0.5)
    del partial[128]
    out = verdict(
        partial, shape_class="bf16/class", boundary=RECURRENCE, execution=GRAPH
    )
    assert out.word == NEITHER
    assert out.missing_batches == (128,)
    assert "no verdict is quantified" in out.detail
    assert "[128]" in out.detail


def test_a_reference_path_refuses_dominates_however_fast_it_ran() -> None:
    """A win recorded while SLinOSS ran torch is a fact about torch."""
    out = verdict(
        flat(0.1),
        shape_class="fp32/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        path=REFERENCE_PATH,
    )
    assert out.word == COMPETITIVE
    assert "did not run its kernel" in out.detail
    assert any("FLOAT32 IS NO LONGER A MIXED PATH" in caveat for caveat in out.caveats)


def test_neither_carries_the_gap_and_says_when_no_resource_is_named() -> None:
    """A gap without a limiting resource is not a report, and the table says which. The
    whole step above its crossover has no profiled limit, so it is where the refusal shows;
    the decode kernel's own limit is profiled and is named instead of refused."""
    slow = flat(1.5)
    slow[128] = 3.0
    unnamed = verdict(
        slow, shape_class="bf16/class", boundary=WHOLE_STEP, execution=GRAPH
    )
    assert unnamed.word == NEITHER
    assert unnamed.worst_batch == 128
    assert unnamed.gap_pct > 0.0
    assert "UNNAMED" in unnamed.detail
    profiled = verdict(
        slow, shape_class="bf16/class", boundary=RECURRENCE, execution=GRAPH
    )
    assert "UNNAMED" not in profiled.detail
    assert "DRAM bandwidth" in profiled.detail
    named = verdict(
        slow,
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        limiting_resource="DRAM bytes at 91% of the measured floor",
    )
    assert "DRAM bytes at 91%" in named.detail
    assert "UNNAMED" not in named.detail


def test_a_whole_step_verdict_carries_the_convolution_disclosure() -> None:
    """SLinOSS pays two convolutions Mamba3 does not have, whichever side wins."""
    out = verdict(
        flat(1.05),
        shape_class="bf16/class",
        boundary=WHOLE_STEP,
        execution=EAGER,
    )
    assert any("two causal convolutions" in caveat for caveat in out.caveats)
    assert any("host enqueue" in caveat for caveat in out.caveats)


def test_a_verdict_over_nothing_or_over_a_nonpositive_ratio_raises() -> None:
    """Neither leaves a geometric mean defined, so neither may return a word."""
    with pytest.raises(ValueError, match="no primary batch"):
        verdict({7: 0.5}, shape_class="c", boundary=RECURRENCE, execution=GRAPH)
    with pytest.raises(ValueError, match="not positive"):
        verdict(
            flat(0.0),
            shape_class="c",
            boundary=RECURRENCE,
            execution=GRAPH,
        )
    with pytest.raises(ValueError, match="unknown path"):
        verdict(
            flat(0.5),
            shape_class="c",
            boundary=RECURRENCE,
            execution=GRAPH,
            path="fast",
        )


def test_a_non_primary_batch_does_not_move_the_verdict() -> None:
    """A stray cell that moved a geometric mean would move a verdict nobody
    quantified over."""
    ratios = flat(0.5)
    ratios[7] = 99.0
    out = verdict(
        ratios, shape_class="bf16/class", boundary=RECURRENCE, execution=GRAPH
    )
    assert out.word == DOMINATES
    assert out.batches == PRIMARY


# --------------------------------------------------------------------------
# The instrument gate
# --------------------------------------------------------------------------


def test_two_medians_inside_one_tick_print_the_clock_and_are_refused() -> None:
    """A ratio between two adjacent timer ticks is the clock's step reported as a
    difference, and no iteration count moves it, so the row may not enter a verdict."""
    row = a_cell(batch=1, slinoss_us=14.336, ratio=14.336 / 13.312)
    reason = unresolved(row)
    assert "within one 1.024 us timer tick" in reason
    assert "no iteration count moves it" in reason
    assert f"{MARGIN_PCT:.0f}% margin" in reason


def test_a_tie_between_two_long_medians_is_a_measured_tie_and_stays_judged() -> None:
    """One tick is negligible against a long row, so two arms that come out equal there
    measured a tie. Refusing it would drop the rows where slinoss is level and keep the
    ones where it wins, which flatters slinoss."""
    assert unresolved(a_cell(batch=64, slinoss_us=169.984, ratio=1.0)) == ""


def test_a_gap_wider_than_the_tick_is_a_measured_gap_however_coarse_the_clock() -> None:
    """Both conditions are needed. A short row where one tick is over the margin still
    measured its gap when the two medians are many ticks apart."""
    row = a_cell(batch=8, slinoss_us=15.360, ratio=0.6)
    assert 100.0 * TIMER_QUANTUM_US * (1.0 / 15.360 + 1.0 / 25.600) > MARGIN_PCT
    assert unresolved(row) == ""


def test_half_widths_that_sum_past_the_margin_refuse_the_row_and_name_its_count() -> (
    None
):
    """A row whose own uncertainty exceeds the margin the three words turn on cannot
    adjudicate them, and the reason has to carry the count so a second sample at another count
    can be set against it. What it must not do is promise that a higher count closes the band:
    twenty times the iterations narrowed the widest band 1.5x against the 4.5x that promise
    implies, so the reason states the count and names the disclosure instead."""
    row = a_cell(batch=8, slinoss_us=15.360, ratio=0.75, resolution_pct=21.0, iters=50)
    reason = unresolved(row)
    assert "half-widths sum to 21.40%" in reason
    assert "at 50 timed iterations" in reason
    assert "no count is asserted to close the band" in reason
    assert SAMPLE_COUNT_DISCLOSURE_LABEL in reason
    assert "a sample count short" not in reason
    assert "at an unrecorded count" in unresolved(row._replace(iters=0))


def test_the_gate_holds_the_row_to_the_margin_and_not_to_a_constant_of_its_own() -> (
    None
):
    """The band is compared against the tightest margin the vocabulary discriminates, so a
    caller who states a different margin gets a different answer at the same row. A second
    threshold would drift from the one the words are defined by."""
    row = a_cell(batch=8, slinoss_us=15.360, ratio=0.75, resolution_pct=8.0)
    assert unresolved(row) == ""
    assert "half-widths sum to 8.40%" in unresolved(row, margin_pct=8.0)


def test_a_refused_primary_batch_refuses_dominates_and_keeps_the_mean() -> None:
    """A refused row was measured, so the class keeps its coverage and loses only the
    universal claim, which cannot rest on a row nobody judged."""
    ratios = flat(0.5)
    del ratios[1]
    out = verdict(
        ratios,
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        refused={1: "not judged: the two medians lie within one tick"},
    )
    assert out.word == COMPETITIVE
    assert out.refused_batches == (1,)
    assert out.batches == (8, 32, 64, 128)
    assert out.geomean_ratio == pytest.approx(0.5)
    assert "were measured and not judged" in out.detail
    assert any("within one tick" in caveat for caveat in out.caveats)


def test_a_refused_batch_is_not_a_missing_batch() -> None:
    """A missing batch is a hole in coverage and refuses every positive word; a refused one
    is a measured row and refuses only the universal claim. Merging them would either
    discard a covered class or let an unjudged row carry one."""
    ratios = flat(0.5)
    del ratios[1]
    hole = verdict(
        ratios, shape_class="bf16/class", boundary=RECURRENCE, execution=GRAPH
    )
    assert hole.missing_batches == (1,)
    assert "no verdict is quantified" in hole.detail
    covered = verdict(
        ratios,
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        refused={1: "not judged: the clock"},
    )
    assert covered.missing_batches == ()
    assert "no verdict is quantified" not in covered.detail


def test_a_row_that_was_not_judged_may_not_move_a_number_either() -> None:
    """A batch that appears in both maps is refused, not averaged: a row excluded from the
    word and included in the mean would move the verdict it was excluded from."""
    out = verdict(
        flat(0.5),
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        refused={1: "not judged: the clock"},
    )
    assert out.batches == (8, 32, 64, 128)
    assert out.refused_batches == (1,)


def test_a_class_the_instrument_emptied_says_no_comparison_was_attempted() -> None:
    """Every row measured and none judged is not the same as no row measured, and the
    eager class's regime reason must not be printed over a class the clock refused."""
    out = unjudged(
        shape_class="bf16/class",
        boundary=RECURRENCE,
        execution=GRAPH,
        refused={1: "not judged: the clock", 8: "not judged: the band"},
    )
    assert out.word == NEITHER
    assert "NO COMPARISON WAS ATTEMPTED" in out.detail
    assert "none was fit to be judged" in out.detail
    assert out.refused_batches == (1, 8)
    assert out.missing_batches == ()
    assert "host-bound eager" not in out.detail


def test_group_verdicts_routes_the_instrument_gate_apart_from_the_regime_rule() -> None:
    """The regime rule and the clock are different refusals with different consequences,
    so a class carrying one of each keeps its coverage and loses its universal claim."""
    cells = [a_cell(batch=batch, ratio=0.5) for batch in (8, 32, 64, 128)]
    cells.insert(0, a_cell(batch=1, slinoss_us=14.336, ratio=14.336 / 13.312))
    grouped = group_verdicts(cells)
    assert len(grouped) == 1
    assert grouped[0].refused_batches == (1,)
    assert grouped[0].batches == (8, 32, 64, 128)
    assert grouped[0].word == COMPETITIVE
    assert grouped[0].missing_batches == ()


def test_group_verdicts_takes_the_worst_path_in_the_class() -> None:
    """One reference cell makes the class's mean a statement about torch, so it cannot
    be averaged away by the kernel cells beside it."""
    cells = [a_cell(batch=batch) for batch in (1, 8, 32, 64)]
    cells.append(
        a_cell(
            batch=128,
            resolved=Resolved(
                names=("decode=reference",), path=REFERENCE_PATH, detail="fell back"
            ),
        )
    )
    grouped = group_verdicts(cells)
    assert len(grouped) == 1
    assert grouped[0].path == REFERENCE_PATH
    assert grouped[0].word == COMPETITIVE


def test_group_verdicts_keys_on_the_boundary_and_the_execution() -> None:
    """The two boundaries are limited by different resources and are never blended."""
    cells = [a_cell(batch=batch) for batch in PRIMARY]
    whole = [cell._replace(boundary=WHOLE_STEP, execution=EAGER) for cell in cells]
    grouped = group_verdicts(cells + whole)
    assert len(grouped) == 2
    assert {(one.boundary, one.execution) for one in grouped} == {
        (RECURRENCE, GRAPH),
        (WHOLE_STEP, EAGER),
    }


def test_a_whole_step_gap_is_priced_by_the_rows_own_floor() -> None:
    """The boundary with no NCU profile still names its resource, and factors it.

    The whole-step boundary is the one :func:`default_resource` owes a string and cannot
    supply, so before this it printed UNNAMED and the verdict was incomplete by its own
    rule. The floor here is built so the two terms multiply to the measured ratio exactly:
    that is the property the string claims, so a test that fabricated an inconsistent floor
    would assert on the formatting and not on the arithmetic.
    """
    consistent = FloorPair(
        available=True,
        slinoss_moved_bytes=800,
        slinoss_floor_us=Microseconds(100.0),
        slinoss_x_floor=1.2,
        mamba_moved_bytes=500,
        mamba_floor_us=Microseconds(62.5),
        mamba_x_floor=1.28,
        detail="fabricated",
    )
    # Rising with the batch, so the worst row is the batch-128 one: below the crossover the
    # resource is weight streaming and the factorization does not apply.
    cells = [
        a_cell(
            batch=batch,
            ratio=1.1 + 0.1 * index,
            boundary=WHOLE_STEP,
            floor=consistent,
        )
        for index, batch in enumerate(PRIMARY)
    ]
    one = group_verdicts(cells)[0]
    assert one.word == NEITHER
    assert "UNNAMED" not in one.detail
    assert "DRAM bandwidth" in one.limiting_resource
    assert "1.6000x (800 against 500 compulsory bytes)" in one.limiting_resource
    assert "0.9375x" in one.limiting_resource
    assert "1.5000x reproduces the measurement to +0.00%" in one.limiting_resource
    # The fleet profile of the decode kernel is stronger evidence than a one-row
    # factorization, so the recurrence boundary must not fall through to this path.
    recurrence = group_verdicts(
        [cell._replace(boundary=RECURRENCE) for cell in cells],
    )[0]
    assert "factors into a byte term" not in recurrence.limiting_resource
    assert "669.75 GB/s" in recurrence.limiting_resource


def test_a_row_with_no_floor_leaves_the_resource_unnamed() -> None:
    """No floor, no invented resource: the verdict says a profile is owed."""
    cells = [
        a_cell(
            batch=batch,
            ratio=1.1 + 0.1 * index,
            boundary=WHOLE_STEP,
            floor=a_floor(available=False),
        )
        for index, batch in enumerate(PRIMARY)
    ]
    one = group_verdicts(cells)[0]
    assert one.word == NEITHER
    assert one.limiting_resource == ""
    assert "UNNAMED" in one.detail


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def a_liveness(*, loaded: bool = True, live: bool = True) -> Liveness:
    """A fabricated liveness proof, with one registry line.

    Args:
        loaded: Whether a decode kernel backend is registered at all.
        live: Whether every registry resolved to a kernel at the measured dtype.

    Returns:
        The proof.
    """
    return Liveness(
        lines=("decode.names()=('cute', 'reference')  resolve->cute  prio10",),
        live=live,
        loaded=loaded,
        recurrence_live=loaded,
        slinoss_package="/tmp/slinoss/__init__.py",
        torch_version="2.7.1+cu126",
        detail="passed",
    )


def test_the_table_carries_the_proof_the_backends_and_both_state_columns() -> None:
    """A table without the resolved backends is discarded, so it is a column."""
    live = a_liveness()
    cell = a_cell()
    body = "\n".join(render([cell], group_verdicts([cell]), live))
    assert "liveness proof" in body
    assert "decode.names()" in body
    assert "decode=cute" in body
    assert "median over per-iteration CUDA-event samples" in body
    assert "two causal convolutions" in body
    assert "FLOAT32 IS NO LONGER A MIXED PATH" in body
    assert "d_state NOT matched" in body
    assert f"{int(cell.match.slinoss_state_bytes.total_bytes):,}" in body
    assert f"{int(cell.match.mamba_state_bytes.total_bytes):,}" in body
    assert " 2/0" in body
    assert "sl_xfl" in body and "m3_xfl" in body
    assert "   2.00    5.00" in body


def test_a_win_carries_both_sides_distance_from_their_own_floor() -> None:
    """A measured win with no floor beside it is a number without a bound, and a row with
    no floor may not print one."""
    live = a_liveness()
    ahead = a_cell(ratio=0.5)
    behind = a_cell(ratio=1.5)
    unfitted = a_cell(ratio=0.5, floor=a_floor(available=False))
    body = "\n".join(render([ahead], group_verdicts([ahead]), live, fit=a_fit()))
    assert "slinoss is 2.00x its own floor" in body
    assert "mamba3 is 5.00x its own floor" in body
    assert "dram fit here" in body and "agrees with the prior fit" in body
    # A loss reads off the ratio, so the floor sentence is not repeated for it.
    lost = "\n".join(render([behind], group_verdicts([behind]), live, fit=a_fit()))
    # The floor line's own wording, not the phrase: a disclosure quotes a floor ratio too.
    assert "x its own floor" not in lost
    blank = "\n".join(render([unfitted], group_verdicts([unfitted]), live))
    assert "no floor: the row is host" in blank
    assert "      -       -" in blank
    assert "none taken in this process" in blank


def test_a_refused_row_reaches_the_table_and_quotes_its_own_medians() -> None:
    """A refused cell that is documented is a result; a silently dropped one is a hole. The
    reason is per row and not per class, because it quotes that row's two medians."""
    live = a_liveness()
    clock = a_cell(batch=1, slinoss_us=14.336, ratio=14.336 / 13.312)
    band = a_cell(batch=8, slinoss_us=15.360, ratio=0.75, resolution_pct=21.0, iters=50)
    body = "\n".join(render([clock, band], group_verdicts([clock, band]), live))
    assert f"jdg=n {RECURRENCE}/{GRAPH}/B=1: " in body
    assert f"jdg=n {RECURRENCE}/{GRAPH}/B=8: " in body
    assert "14.336 against 13.312 us" in body
    assert "at 50 timed iterations" in body


def test_the_iteration_count_prints_per_row_and_a_dash_where_none_was_stored() -> None:
    """The half-width a row is judged on is a function of its own iteration count, so a
    table holding rows taken at two counts has to say per row which one it carries."""
    live = a_liveness()
    counted = a_cell(batch=64, iters=1_200)
    older = a_cell(batch=128, iters=0)
    body = "\n".join(render([counted, older], group_verdicts([counted, older]), live))
    assert "    it " in body
    assert " 1,200 " in body
    assert "  64      -" not in body
    assert " 128      - " in body
    assert "timed iterations behind the row's two medians" in body


def test_no_disclosure_quotes_a_latency_from_another_torch_or_another_host() -> None:
    """Every figure in the table comes from the measuring process. The retired baseline
    numbers are the ones most likely to creep back in as a reference point, so they are
    named here."""
    text = " ".join(DISCLOSURES)
    for retired in ("2,636", "628", "7,693", "277,055", "276,107", "6,085", "71.16"):
        assert retired not in text
    assert "NO FIGURE IN THIS TABLE CROSSES A HOST OR A TORCH VERSION" in text
    assert f"{REPLAY_IDLE_PCT_HIGH}%" in text
    assert "bytes-and-kernel-quality" in text


def test_a_whole_step_row_says_which_operator_it_reached() -> None:
    """SLinOSSMixer.step routes T=1 to the decode boundary, and every whole-step figure
    taken before it did is void rather than superseded, so the row says which it is."""
    live = a_liveness()
    whole = a_cell()._replace(boundary=WHOLE_STEP)
    body = "\n".join(render([whole], group_verdicts([whole]), live))
    assert "ROUTED: SLinOSSMixer.step reaches the decode boundary at T=1" in body
    caveats = group_verdicts([whole])[0].caveats
    assert any("ROUTED" in one for one in caveats)
    # The recurrence row is the boundary itself and carries no routing caveat.
    assert not any("ROUTED" in one for one in group_verdicts([a_cell()])[0].caveats)


def a_probe(*, foreign_mib: float = 0.0, utilization: float = 0.0) -> Contention:
    """One injected closing contention probe."""
    return Contention(
        probed=True,
        foreign_process_count=Count(0 if foreign_mib == 0.0 else 1),
        foreign_memory_mib=Mebibytes(foreign_mib),
        utilization_pct=Percent(utilization),
        detail="injected",
    )


def test_a_foreign_job_that_lands_mid_run_voids_the_run_and_its_verdict() -> None:
    """The gate is armed once, before the first sample. A card grabbed after that produces
    a table that reads clean, so the closing probe has to be able to void it."""
    live = a_liveness()
    cell = a_cell(ratio=0.5)
    verdicts = group_verdicts([cell])
    assert sample_void(a_probe()) == ""
    dirty = sample_void(a_probe(foreign_mib=36_855.0, utilization=100.0))
    assert dirty.startswith("VOID:")
    assert "36,855 MiB at 100% utilization" in dirty
    body = "\n".join(render([cell], verdicts, live, void=dirty))
    assert dirty in body
    assert DOMINATES not in body.rsplit("VOID:", 1)[-1]
    # A resident context too small to hold a workload does not void; requiring zero would
    # make the gate unopenable.
    assert sample_void(a_probe(foreign_mib=64.0)) == ""


def test_a_render_that_took_no_sample_is_not_voided_by_a_tenant() -> None:
    """The postcondition is about samples taken here, and a render out of a bank has none.

    Measured: with the whole grid banked, every render on a shared card printed VOID and
    dropped its verdicts, so no table could be produced whenever any tenant held memory. The
    rows it prints carry the witness of the window that measured them, so the closing probe
    has nothing to say about them.
    """
    tenant = a_probe(foreign_mib=36_855.0, utilization=100.0)
    assert sample_void(tenant, sampled=False) == ""
    assert sample_void(tenant).startswith("VOID:")


def test_an_unbuilt_tree_closes_the_gate_and_a_dtype_gap_does_not() -> None:
    """Measured on this card: the decode registry resolves cute at float32 while so3ssd
    resolves reference, so a dtype gap is per boundary and cannot close the whole run. Only
    an unregistered kernel means the tree did not build."""
    assert kernel_gate(a_liveness()) == ""
    assert kernel_gate(a_liveness(live=False)) == ""
    closed = kernel_gate(a_liveness(loaded=False))
    assert "build_ext --inplace" in closed


def test_no_eager_row_is_judged_and_the_batch_was_the_wrong_proxy() -> None:
    """Measured: the eager arms sit on flat host floors of 160-163 us against 28-29 us, so
    an eager batch-8 row read 8.53x its own DRAM floor while its graph twin read 1.04x. The
    crossover called that row DRAM-bound, so the batch is not the discriminant and no eager
    row is judged."""
    assert judgeable(batch=1, execution=EAGER, boundary=RECURRENCE)
    assert judgeable(batch=1, execution=GRAPH, boundary=RECURRENCE) == ""
    # The row the earlier rule judged, and the one that falsified it.
    assert "host floor" in judgeable(batch=128, execution=EAGER, boundary=RECURRENCE)
    # Each boundary quotes its own measured floors, not the other's.
    assert "160-163" in judgeable(batch=128, execution=EAGER, boundary=RECURRENCE)
    assert "620-643" in judgeable(batch=128, execution=EAGER, boundary=WHOLE_STEP)
    assert "host floor" in judgeable(batch=8, execution=EAGER, boundary=RECURRENCE)
    # The whole step crosses later, so an eager row there is host-bound by regime as well.
    assert "host-bound" in judgeable(batch=8, execution=EAGER, boundary=WHOLE_STEP)
    # Nothing about the graph rows moves.
    for batch in (1, 8, 32, 64, 128):
        assert judgeable(batch=batch, execution=GRAPH, boundary=WHOLE_STEP) == ""

    live = a_liveness()
    unfit = a_cell(batch=1, ratio=0.1)._replace(execution=EAGER, regime="host")
    fit = a_cell(batch=128, ratio=0.1)._replace(execution=EAGER)
    only = group_verdicts([unfit])
    assert only[0].word == NEITHER
    assert "NO COMPARISON WAS ATTEMPTED" in only[0].detail
    assert only[0].batches == ()
    # Both eager rows are out, so the class is quantified over nothing however fast it ran.
    mixed = group_verdicts([unfit, fit])
    assert mixed[0].batches == ()
    assert mixed[0].word == NEITHER
    body = "\n".join(render([unfit, fit], mixed, live))
    assert "jdg=n:" in body
    assert "host-bound at the recurrence boundary" in body
    assert "measured host floor" in body


def test_the_closing_probe_waits_for_the_reading_to_become_attributable() -> None:
    """Measured: the instant the last region ends the probe reads 93-100 percent
    utilization with zero foreign processes, which is this run's own work, so probing then
    voids every run against itself."""
    order: list[str] = []

    def sync(device: torch.device) -> None:
        order.append(f"sync {device}")

    def rest(seconds: float) -> None:
        order.append(f"rest {seconds}")

    def probe(ordinal: int) -> Contention:
        order.append(f"probe {ordinal}")
        return a_probe()

    out = closing_probe(
        0,
        device=torch.device("cpu"),
        settle_s=CLOSE_SETTLE_S,
        probe=probe,
        rest=rest,
        sync=sync,
    )
    assert order == ["sync cpu", f"rest {CLOSE_SETTLE_S}", "probe 0"]
    assert sample_void(out) == ""
    # Four times the 0.25 s the reading took to settle, and no threshold moved to get it.
    assert CLOSE_SETTLE_S >= 1.0


def test_the_table_states_the_interpreter_and_bounds_it_without_correcting_it() -> None:
    """A torch minor moves host dispatch 5-17% and the toolkit moves device time 1.2%, so
    the interpreter is a stamped term. The deltas bound the confound and adjust nothing."""
    text = " ".join(DISCLOSURES)
    assert "2.6.0+cu124" in text and "2.7.1+cu126" in text
    assert "not a conversion factor" in text
    assert "cannot manufacture a win" in text
    assert "capture both sides or neither" in text
    assert "20 to 42 percent" in text
    assert "8.53x" in text


def test_a_render_takes_no_sample_and_so_waits_for_no_card() -> None:
    """The idle-card gate protects samples. A render computes every number from the bank and
    takes none, so blocking it on a busy card withheld the table for 600 seconds and then
    printed nothing at all. The flag admits no contended sample: it admits no sample."""
    assert parse_args([]).render_only is False
    assert parse_args(["--render-only"]).render_only is True
    text = parse_args(["--render-only"]).__dict__
    assert text["graph_cells_per_process"] == GRAPH_CELLS_PER_PROCESS
    # Not a way to measure on a busy card: the two flags are independent, and the render one
    # measures nothing whatever the graph budget says.
    assert parse_args(["--render-only", "--graph-cells-per-process", "4"]).render_only


def test_the_smoke_pass_enumerates_and_discloses_without_a_device(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Construction, enumeration and the state model need no idle card, and the driver
    has to be runnable before one is free."""
    status = main(
        [
            "--smoke",
            "--d-model",
            "512",
            "--dtype",
            "bf16",
            "--batch",
            "1",
            "128",
            "--d-state",
            "144",
            "--mamba-d-state",
            "128",
            "--sharing",
            G1,
            "--mode",
            SISO,
        ]
    )
    assert status == 0
    out = capsys.readouterr().out
    assert "enumerated 2 cells" in out
    assert "sl_state=" in out
    assert "m3_state=" in out
    assert "cvbuf 2/0" in out
    assert "regime rec host/sub_l2 step host/sub_l2" in out
    assert "regime rec dram/dram step dram/dram" in out
    assert "FLOAT32 IS NO LONGER A MIXED PATH" in out
    assert "no short convolution" in out
    assert "ROUTED: SLinOSSMixer.step reaches the decode boundary at T=1" in out
    assert "ONE GRAPH CELL PER PROCESS" in out
    assert "NEVER ACROSS TREES" in out
    assert "is_outproj_norm" in out


def test_a_quantile_is_an_order_statistic_and_never_a_value_between_two_samples() -> (
    None
):
    """An interpolated quantile of a bimodal sample names a latency the loop never ran, and
    bimodal is what this instrument produces: host run-ahead splits the samples into two
    modes and the midpoint between them is a duration of nothing.

    Ten distinct samples, out of order, so the rank rule itself is pinned. Every other rule
    that lands on an observed sample disagrees here: linear interpolation reads 1.9, 5.5 and
    9.1, and rounding ``q*(n-1)`` reads 2 at the tenth percentile.
    """
    samples = tuple(
        Microseconds(one) for one in (7.0, 2.0, 9.0, 4.0, 1.0, 10.0, 5.0, 3.0, 8.0, 6.0)
    )
    observed = set(samples)
    quantiles = [order_statistic_us(samples, q) for q in (0.0, 0.1, 0.5, 0.9, 1.0)]
    assert all(one in observed for one in quantiles)
    assert quantiles == [1.0, 1.0, 5.0, 9.0, 10.0]


def test_a_quantile_of_no_samples_raises_rather_than_reading_zero() -> None:
    """A zero prints in the same column as a duration and would read as a fast row."""
    with pytest.raises(ValueError, match="no samples"):
        order_statistic_us((), 0.5)
    with pytest.raises(ValueError, match="outside"):
        order_statistic_us((Microseconds(1.0),), 1.5)


def test_the_dispersion_block_prints_each_arms_own_order_statistics() -> None:
    """One block built off one arm's samples would print the same row twice under two
    names, and the pair's asymmetry is the whole point of the comparison."""
    cell = a_cell(samples=(10.0, 20.0, 30.0, 40.0, 90.0), ratio=0.5, slinoss_us=30.0)
    lines = dispersion_lines([cell])
    slinoss_row = next(one for one in lines if " slinoss " in one)
    mamba_row = next(one for one in lines if " mamba3 " in one)
    # p50 of five samples is the third: ceil(0.5*5) - 1 = 2.
    assert "30.000" in slinoss_row
    assert "10.000" in slinoss_row
    assert "90.000" in slinoss_row
    # Mamba3's samples are SLinOSS's over the ratio, so every figure doubles.
    assert "60.000" in mamba_row
    assert "180.000" in mamba_row


def test_a_row_that_banked_no_samples_is_counted_rather_than_dropped() -> None:
    """A block silently shorter than the table it sits under reads as a block over all of
    it, and a reader would take the recomputation as covering rows it never saw."""
    lines = dispersion_lines([a_cell(samples=(1.0, 2.0, 3.0)), a_cell(batch=32)])
    assert any("1 of 2 rows banked no samples" in one for one in lines)


def test_a_block_over_rows_that_banked_nothing_is_empty_rather_than_a_header() -> None:
    """A header over no rows claims a recomputation that is not there."""
    assert dispersion_lines([a_cell(), a_cell(batch=32)]) == ()


def test_the_banked_record_carries_every_sample_the_row_measured() -> None:
    """The summary floats are three reductions of the samples; without the samples in the
    artifact, no reader can recompute them or see drift across the loop."""
    cell = a_cell(samples=(10.0, 20.0, 30.0), ratio=0.5, slinoss_us=20.0)
    record = cell_record(cell, stored={"schema": BANK_SCHEMA})
    assert record["slinoss_samples_duration_us"] == [10.0, 20.0, 30.0]
    assert record["mamba_samples_duration_us"] == [20.0, 40.0, 60.0]
    back = cell_from_record(record)
    assert back.slinoss_samples_duration_us == cell.slinoss_samples_duration_us
    assert back.mamba_samples_duration_us == cell.mamba_samples_duration_us


def test_a_record_written_before_the_samples_existed_loads_with_none_of_them() -> None:
    """On the rule `iters` set: an absent field reads as absent, never as a value derived
    from the summary, because a list synthesized off a median would read as measured."""
    record = cell_record(
        a_cell(samples=(10.0,), slinoss_us=10.0), stored={"schema": BANK_SCHEMA}
    )
    del record["slinoss_samples_duration_us"]
    del record["mamba_samples_duration_us"]
    back = cell_from_record(record)
    assert back.slinoss_samples_duration_us == ()
    assert back.mamba_samples_duration_us == ()
    assert back.slinoss_duration_us == pytest.approx(10.0)


# --------------------------------------------------------------------------
# The bank, the order and the witness
# --------------------------------------------------------------------------


def a_witness(*, stamp: str = EXCLUSIVE_WITNESS, foreign_mib: float = 0.0) -> Witness:
    """A fabricated witness, with the window count its stamp implies.

    Args:
        stamp: :data:`EXCLUSIVE_WITNESS` or :data:`RESIDENCY_WITNESS`.
        foreign_mib: Foreign memory at admission.

    Returns:
        The witness.
    """
    replicates = 1 if stamp == EXCLUSIVE_WITNESS else RESIDENCY_REPLICATES
    return Witness(
        stamp=stamp,
        foreign_mib=foreign_mib,
        replicates=replicates,
        agrees=True,
        detail=f"{stamp} card, {foreign_mib:,.0f} MiB foreign memory, "
        f"{replicates} window{'s' if replicates > 1 else ''}",
    )


def test_an_interrupted_run_loses_at_most_the_cell_in_flight(
    tmp_path: Path,
) -> None:
    """A 360-cell run that either completes or yields nothing wastes every short window. So
    a cell is banked the moment it is measured, under a temporary name and renamed, and the
    next invocation reads what is there and skips it."""
    bank = str(tmp_path)
    stored = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    first = a_cell(batch=8, witness=a_witness())
    second = a_cell(batch=128, witness=a_witness())
    write_cell(bank, first, stored=stored)
    write_cell(bank, second, stored=stored)
    read, refused = read_bank(bank, stored=stored)
    assert refused == ()
    assert set(read) == {
        Task(point=first.point, boundary=RECURRENCE, execution=GRAPH).key,
        Task(point=second.point, boundary=RECURRENCE, execution=GRAPH).key,
    }
    # No temporary left behind for the next invocation to read as a banked cell.
    assert [name for name in os.listdir(bank) if not name.endswith(".json")] == []


def test_a_banked_cell_carries_every_number_the_table_prints(
    tmp_path: Path,
) -> None:
    """A bank that dropped a field would silently print a different table for a resumed run
    than for a single-pass one, so the round trip is asserted field by field."""
    bank = str(tmp_path)
    stored = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    cell = a_cell(batch=64, ratio=0.83, witness=a_witness(stamp=RESIDENCY_WITNESS))
    write_cell(bank, cell, stored=stored)
    read, _ = read_bank(bank, stored=stored)
    back = read[Task(point=cell.point, boundary=RECURRENCE, execution=GRAPH).key]
    assert back == cell


def test_a_banked_cell_from_another_host_or_torch_is_refused_not_read(
    tmp_path: Path,
) -> None:
    """A bank is the one path by which a figure could cross a card or a torch version
    without anyone noticing, because it is read from disk rather than measured."""
    bank = str(tmp_path)
    mine = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    theirs = dict(mine, device="NVIDIA A100-SXM4-80GB")
    write_cell(bank, a_cell(witness=a_witness()), stored=theirs)
    read, refused = read_bank(bank, stored=mine)
    assert read == {}
    assert len(refused) == 1
    assert "provenance differs on device" in refused[0]
    older = dict(mine, schema=BANK_SCHEMA + 1)
    write_cell(bank, a_cell(batch=64, witness=a_witness()), stored=older)
    read, refused = read_bank(bank, stored=mine)
    assert read == {}
    assert any("schema" in one for one in refused)


def a_competitor_tree(root: str, *, step_body: str = "STEP = 1\n") -> str:
    """Write a stand-in for the resolved competitor package.

    Args:
        root: Package directory to create.
        step_body: Contents of the module standing in for the step kernel, which is the file
            an edit to the competitor would land in.

    Returns:
        ``root``.
    """
    os.makedirs(os.path.join(root, "modules"), exist_ok=True)
    for relative, body in (
        ("__init__.py", ""),
        (os.path.join("modules", "__init__.py"), ""),
        (os.path.join("modules", "mamba3.py"), step_body),
    ):
        with open(os.path.join(root, relative), "w") as handle:
            handle.write(body)
    return root


def a_resolved_competitor(
    monkeypatch: pytest.MonkeyPatch, root: str
) -> types.ModuleType:
    """Put a package on :data:`sys.modules` so the driver resolves it as the competitor.

    The driver reads the competitor's location from the import machinery and not from an
    argument, so a test that wants to control which tree answers has to answer as that tree.
    """
    module = types.ModuleType(MAMBA_PACKAGE)
    module.__file__ = os.path.join(root, "__init__.py")
    monkeypatch.setitem(sys.modules, MAMBA_PACKAGE, module)
    return module


def test_a_competitor_source_edit_refuses_a_banked_competitor_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Digesting our tree alone let a banked Mamba3 figure survive a Mamba3 source edit: the
    cell is a comparison, so half of it went unkeyed. One byte of the competitor's step module
    must refuse the record the way a kernel edit refuses ours."""
    root = a_competitor_tree(str(tmp_path / "mamba_ssm"))
    a_resolved_competitor(monkeypatch, root)
    bank = str(tmp_path / "bank")
    os.makedirs(bank)
    before = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    assert before["mamba_package"] == root
    write_cell(bank, a_cell(witness=a_witness()), stored=before)
    read, refused = read_bank(bank, stored=before)
    assert len(read) == 1 and refused == ()

    a_competitor_tree(root, step_body="STEP = 2\n")
    after = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    assert after["mamba_sources"] != before["mamba_sources"]
    read, refused = read_bank(bank, stored=after)
    assert read == {}
    assert len(refused) == 1
    assert "provenance differs on mamba_sources" in refused[0]


def test_a_competitor_resolved_from_another_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two copies of one competitor revision at two paths are two dependency sets, and the
    field names what answered, so the directory is keyed as well as the digest."""
    first = a_competitor_tree(str(tmp_path / "one" / "mamba_ssm"))
    second = a_competitor_tree(str(tmp_path / "two" / "mamba_ssm"))
    bank = str(tmp_path / "bank")
    os.makedirs(bank)
    a_resolved_competitor(monkeypatch, first)
    mine = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    write_cell(bank, a_cell(witness=a_witness()), stored=mine)
    a_resolved_competitor(monkeypatch, second)
    theirs = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    assert theirs["mamba_sources"] == mine["mamba_sources"]
    read, refused = read_bank(bank, stored=theirs)
    assert read == {}
    assert "provenance differs on mamba_package" in refused[0]


def test_the_dependency_set_is_recorded_and_not_keyed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MIMO and SISO paths differ in exactly the dependency mapping, so keying on it would
    refuse a MIMO cell on a SISO render while blending nothing. It is recorded to be read.
    The boundary is asserted in both directions: the unkeyed field does not refuse and the
    keyed one does."""
    root = a_competitor_tree(str(tmp_path / "mamba_ssm"))
    a_resolved_competitor(monkeypatch, root)
    bank = str(tmp_path / "bank")
    os.makedirs(bank)
    mine = provenance(a_liveness(), device_name="NVIDIA RTX A6000")
    assert "deps" not in MATCHED_PROVENANCE
    assert "competitor_origin" not in MATCHED_PROVENANCE
    elsewhere = dict(
        mine,
        deps=dict(mine["deps"], triton="absent"),
        competitor_origin=dict(mine["competitor_origin"], route="sys.modules"),
    )
    write_cell(bank, a_cell(witness=a_witness()), stored=elsewhere)
    read, refused = read_bank(bank, stored=mine)
    assert len(read) == 1 and refused == ()
    # And the record still carries what it was not keyed on, or there is nothing to read.
    assert mine["deps"]["python"] == sys.executable
    assert mine["deps"]["prefix"] == sys.prefix
    assert mine["deps"]["torch"] == str(torch.__version__)
    assert set(mine["deps"]) == {
        "python",
        "prefix",
        "torch",
        "triton",
        "apache-tvm-ffi",
    }


def test_the_competitor_origin_records_every_file_it_digested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest says two trees differ; the manifest says which file, which is what makes the
    disposable copy auditable against the immutable tree it came from."""
    root = a_competitor_tree(str(tmp_path / "mamba_ssm"))
    a_resolved_competitor(monkeypatch, root)
    found = competitor_provenance()
    origin = found["origin"]
    assert origin["route"] == "sys.modules"
    assert origin["copied_from"] == SOURCES_ORIGIN
    assert origin["attested"] == "process"
    assert origin["file_count"] == 3
    assert [one["path"] for one in origin["files"]] == [
        "__init__.py",
        "modules/__init__.py",
        "modules/mamba3.py",
    ]
    # sha256 of the file itself, not of the walk: an empty file hashes to the empty digest.
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert origin["files"][0]["sha256"] == empty
    assert origin["files"][-1]["sha256"] != empty
    assert len({one["sha256"] for one in origin["files"]}) == 2


def test_a_copy_with_no_git_says_so_rather_than_inventing_a_commit(
    tmp_path: Path,
) -> None:
    """Both the copy that answers and the .sources tree it came from are bare file trees, so
    the honest answer is absent, and a manifest is then the only origin there is. A copy that
    does carry a repository names its commit: the search runs upward because a package
    directory sits inside its repository, not beside it."""
    root = a_competitor_tree(str(tmp_path / "bare" / "mamba_ssm"))
    assert git_commit(root) == "absent"
    repository = os.path.join(str(tmp_path / "bare"), ".git")
    os.makedirs(os.path.join(repository, "refs", "heads"))
    with open(os.path.join(repository, "HEAD"), "w") as handle:
        handle.write("ref: refs/heads/main\n")
    with open(os.path.join(repository, "refs", "heads", "main"), "w") as handle:
        handle.write("0f6e5d4c3b2a19080706050403020100fedcba98\n")
    assert git_commit(root) == "0f6e5d4c3b2a19080706050403020100fedcba98"


def test_an_unimportable_competitor_is_named_and_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that cannot resolve the competitor cannot validate a banked competitor cell,
    so it reports the failure in the keyed field and refuses the bank rather than matching a
    placeholder against a real digest."""
    monkeypatch.delitem(sys.modules, MAMBA_PACKAGE, raising=False)
    monkeypatch.setattr(sys, "path", [])
    found = competitor_provenance()
    assert found["package"] == "unimportable"
    assert found["digest"] == "unreadable"
    assert found["origin"]["file_count"] == 0
    assert found["origin"]["route"] == "no spec on sys.path"
    assert found["origin"]["git"] == "absent"


def test_a_manifest_ignores_caches_and_orders_by_path(
    tmp_path: Path,
) -> None:
    """A bytecode write must not change the record of unchanged source, and the walk order
    differs by filesystem, so the manifest is sorted and __pycache__ is excluded."""
    root = a_competitor_tree(str(tmp_path / "mamba_ssm"))
    cache = os.path.join(root, "__pycache__")
    os.makedirs(cache)
    with open(os.path.join(cache, "__init__.cpython-312.py"), "w") as handle:
        handle.write("noise = 1\n")
    paths = [one["path"] for one in file_manifest(root)]
    assert paths == sorted(paths)
    assert all("__pycache__" not in one for one in paths)
    assert file_manifest(os.path.join(root, "does-not-exist")) == ()


def test_the_dependency_set_needs_no_import_to_report_a_missing_package() -> None:
    """Importing to read a version is a side effect on the thing being measured, and the
    competitor's own dependencies refuse to import outside their pin set."""
    assert dependency_set()["torch"] == str(torch.__version__)
    absent = installed_version("no-such-distribution-xyz", "no_such_module_xyz")
    assert absent == "absent"
    assert "no_such_module_xyz" not in sys.modules


def test_the_decisive_order_completes_a_class_before_it_starts_another() -> None:
    """A verdict needs every primary batch, so an order that spreads cells across classes
    yields a hundred incomplete classes. The class fields sort first and the batch last."""
    grid = enumerate_grid(
        d_models=[512, 1024],
        dtype_names=["bf16", "fp32"],
        batches=list(PRIMARY),
        slinoss_d_states=[96, 144],
        mamba_d_states=[64, 128],
        sharings=[G1],
        modes=[SISO],
    )
    order = tasks(
        grid,
        boundaries=[RECURRENCE, WHOLE_STEP],
        executions=[GRAPH, EAGER],
        order=DECISIVE,
    )
    head = order[0]
    assert head.boundary == RECURRENCE
    assert head.execution == GRAPH
    assert head.point.dtype_name == "bf16"
    # The closest pair the grid offers, which is the pair the brief names.
    assert (head.point.slinoss_d_state, head.point.mamba_d_state) == (144, 128)
    seen: list[str] = []
    for one in order:
        name = f"{one.boundary}/{one.execution}/{one.point.shape_class}"
        if not seen or seen[-1] != name:
            assert name not in seen, f"class {name} was reopened after being left"
            seen.append(name)
    batches = [
        one.point.batch
        for one in order
        if f"{one.boundary}/{one.execution}/{one.point.shape_class}"
        == f"{head.boundary}/{head.execution}/{head.point.shape_class}"
    ]
    # Above the crossover first, largest first; batch 1 is under it and comes last.
    assert batches == [128, 64, 32, 8, 1]
    assert tasks(
        grid, boundaries=[RECURRENCE], executions=[GRAPH], order=NESTED
    ) == tuple(
        Task(point=point, boundary=RECURRENCE, execution=GRAPH) for point in grid.points
    )
    with pytest.raises(ValueError, match="unknown order"):
        tasks(grid, boundaries=[RECURRENCE], executions=[GRAPH], order="soonest")


def test_the_eager_rows_of_a_class_sort_after_its_graph_rows() -> None:
    """An eager host-bound row cannot be judged, so it must not be measured before a row
    that can."""
    graph = Task(point=a_point(1), boundary=RECURRENCE, execution=GRAPH)
    eager = Task(point=a_point(1), boundary=RECURRENCE, execution=EAGER)
    assert decisiveness(graph) < decisiveness(eager)
    step = Task(point=a_point(128), boundary=WHOLE_STEP, execution=GRAPH)
    assert decisiveness(graph) < decisiveness(step)


def test_a_card_running_foreign_compute_is_refused_and_the_thresholds_do_not_move() -> (
    None
):
    """The residency witness is a heavier requirement, not a lower bar. A card above the
    utilization ceiling is refused outright, which is the case it must never admit."""
    assert admit(a_probe())[0] == EXCLUSIVE_WITNESS
    assert admit(a_probe())[1] == 1
    assert admit(a_probe(foreign_mib=511.0))[0] == EXCLUSIVE_WITNESS
    resident = admit(a_probe(foreign_mib=3464.0))
    assert resident[0] == RESIDENCY_WITNESS
    assert resident[1] == RESIDENCY_REPLICATES == 2
    assert "Neither threshold is relaxed" in resident[2]
    busy = admit(a_probe(foreign_mib=3464.0, utilization=100.0))
    assert busy[0] == ""
    assert "above the 5% utilization ceiling" in busy[2]
    # Utilization is the discriminant, not memory: a card at 6% with no memory is still
    # refused, so nothing here admits a contended sample.
    assert admit(a_probe(utilization=6.0))[0] == ""
    assert admit(a_probe(foreign_mib=3464.0), exclusive_only=True)[0] == ""


def test_two_windows_agree_exactly_at_the_sum_of_their_half_widths() -> None:
    """Inclusive at the boundary and exclusive past it are different code paths."""
    # Half-widths of 3.0 us and zero, so the boundary is a literal and sits above the tick.
    assert agrees_within_half_widths(100.0, 3.0, 103.0, 0.0)
    assert not agrees_within_half_widths(100.0, 3.0, 103.01, 0.0)
    assert agrees_within_half_widths(100.0, 0.0, 100.0, 0.0)


def test_the_agreement_reach_never_falls_below_one_timer_tick() -> None:
    """A half-width narrower than the timer's step claims a resolution the timer does not
    have, and two medians on adjacent ticks are one measurement, not a disagreement. This
    was measured: the first banked window discarded a cell on a gap of exactly one tick."""
    # The case from that window: 54.272 against 53.248 at half-widths summing to 0.512.
    assert agrees_within_half_widths(54.272, 0.5, 53.248, 0.46)
    # One tick is the floor, not a licence: two ticks apart still disagrees.
    assert not agrees_within_half_widths(54.272, 0.5, 52.224, 0.46)
    assert TIMER_QUANTUM_US == 1.024
    # And the floor does not shrink a reach that is already wider than a tick.
    assert not agrees_within_half_widths(100.0, 3.0, 104.0, 0.0)


def test_a_cell_whose_windows_disagree_is_discarded_not_averaged(
    tmp_path: Path,
) -> None:
    """Averaging two windows that disagree hides the thing the second window was taken to
    detect, so the cell is dropped, unbanked, and measured again next time."""
    windows = iter(
        [
            a_cell(slinoss_us=100.0, resolution_pct=0.5),
            a_cell(slinoss_us=140.0, resolution_pct=0.5),
        ]
    )

    def next_window(*args: object, **kwargs: object) -> Cell:
        return next(windows)

    cell, lost = measure_replicated(
        a_point(128),
        boundary=RECURRENCE,
        execution=GRAPH,
        device=torch.device("cpu"),
        iters=2,
        warmup=0,
        witness_stamp=RESIDENCY_WITNESS,
        replicates=RESIDENCY_REPLICATES,
        foreign_mib=3464.0,
        ordinal=0,
        probe=lambda _: a_probe(foreign_mib=3464.0),
        rest=lambda _: None,
        measure=next_window,
    )
    assert cell is None
    assert "slinoss disagreed across the two windows" in lost
    assert "timer step, whichever is larger" in lost
    assert "not averaged" in lost
    read, _ = read_bank(str(tmp_path), stored={"schema": BANK_SCHEMA})
    assert read == {}


def test_two_windows_that_agree_earn_a_residency_row_and_it_says_so() -> None:
    """The row is the first window with the second recorded as corroboration, and it is
    stamped distinctly from an exclusive row rather than blended with one."""
    windows = iter(
        [
            a_cell(slinoss_us=100.0, resolution_pct=0.5),
            a_cell(slinoss_us=100.4, resolution_pct=0.5),
        ]
    )

    def next_window(*args: object, **kwargs: object) -> Cell:
        return next(windows)

    cell, lost = measure_replicated(
        a_point(128),
        boundary=RECURRENCE,
        execution=GRAPH,
        device=torch.device("cpu"),
        iters=2,
        warmup=0,
        witness_stamp=RESIDENCY_WITNESS,
        replicates=RESIDENCY_REPLICATES,
        foreign_mib=3464.0,
        ordinal=0,
        probe=lambda _: a_probe(foreign_mib=3464.0),
        rest=lambda _: None,
        measure=next_window,
    )
    assert lost == ""
    assert cell is not None
    assert cell.slinoss_duration_us == 100.0
    assert cell.witness.stamp == RESIDENCY_WITNESS
    assert cell.witness.replicates == 2
    assert cell.witness.foreign_mib == 3464.0
    assert witness_mark(cell.witness) == "resi2"
    assert witness_mark(NO_WITNESS) == "-"
    body = "\n".join(render([cell], group_verdicts([cell]), a_liveness()))
    assert "wit=resi2" in body
    assert "2 windows agreed" in body
    assert "THE CARD IS A PROPERTY OF THE ROW" in body


def test_a_card_that_changes_between_windows_discards_the_cell() -> None:
    """Two windows are evidence only if they are the same card twice."""

    def next_window(*args: object, **kwargs: object) -> Cell:
        return a_cell()

    cell, lost = measure_replicated(
        a_point(128),
        boundary=RECURRENCE,
        execution=GRAPH,
        device=torch.device("cpu"),
        iters=2,
        warmup=0,
        witness_stamp=RESIDENCY_WITNESS,
        replicates=RESIDENCY_REPLICATES,
        foreign_mib=3464.0,
        ordinal=0,
        probe=lambda _: a_probe(foreign_mib=9000.0, utilization=100.0),
        rest=lambda _: None,
        measure=next_window,
    )
    assert cell is None
    assert "the card changed between windows" in lost


def test_a_residency_run_stands_on_its_admitted_memory_and_voids_on_a_new_tenant() -> (
    None
):
    """Under residency the closing probe cannot be quiet, so the closing condition is the
    one it was admitted under: no foreign compute and no new tenant."""
    admitted = a_witness(stamp=RESIDENCY_WITNESS, foreign_mib=3464.0)
    assert sample_void(a_probe(foreign_mib=3464.0), witness=admitted) == ""
    assert sample_void(a_probe(foreign_mib=3900.0), witness=admitted) == ""
    grown = sample_void(a_probe(foreign_mib=12000.0), witness=admitted)
    assert "VOID" in grown and "new tenant" not in grown
    assert "admitted on residency" in grown
    busy = sample_void(a_probe(foreign_mib=3464.0, utilization=100.0), witness=admitted)
    assert "VOID" in busy
    # An exclusive run is unchanged: any foreign memory over the floor still voids it.
    assert "VOID" in sample_void(a_probe(foreign_mib=3464.0))


def test_a_kernel_edit_empties_the_bank(tmp_path: object) -> None:
    """A bank keyed on host and torch alone would merge two kernels into one table.

    The decode kernel gained an addressing transpose that moved it from 357.64 to 669.75
    GB/s with no interface change, so the digest is over source and not over an interface.
    """
    root = os.path.join(str(tmp_path), "slinoss")
    os.makedirs(os.path.join(root, "ops", "decode", "cute"))
    package = os.path.join(root, "__init__.py")
    with open(package, "w") as handle:
        handle.write("VERSION = 1\n")
    kernel = os.path.join(root, "ops", "decode", "cute", "step.py")
    with open(kernel, "w") as handle:
        handle.write("stride = 1\n")
    before = source_digest(package)
    with open(kernel, "w") as handle:
        handle.write("stride = 2\n")
    assert source_digest(package) != before
    # A bytecode write is not a source change.
    os.makedirs(os.path.join(root, "__pycache__"))
    with open(os.path.join(root, "__pycache__", "x.pyc"), "wb") as binary:
        binary.write(b"\x00")
    after = source_digest(package)
    with open(kernel, "w") as handle:
        handle.write("stride = 1\n")
    assert source_digest(package) == before
    assert after != before
    assert source_digest(os.path.join(str(tmp_path), "absent", "__init__.py")) == (
        "unreadable"
    )


def test_the_second_graph_cell_of_a_process_is_deferred_and_eager_never_is() -> None:
    """Two decode graph captures in one process fire a device-side assert."""
    graph = Task(point=a_point(128), boundary=RECURRENCE, execution=GRAPH)
    eager = Task(point=a_point(128), boundary=RECURRENCE, execution=EAGER)
    assert not defers(graph, graphed=0)
    assert defers(graph, graphed=1)
    # Eager builds no capture, so the hazard does not reach it at any count.
    assert not defers(eager, graphed=9)
    # The budget is a number and not a boolean: a run that accepts the hazard says so.
    assert not defers(graph, graphed=1, budget=2)


def test_a_device_assert_voids_the_process_and_an_ordinary_error_voids_the_cell() -> (
    None
):
    """A device assert poisons the context, so later cells in that process are void even
    when they produce numbers."""
    assert poisons(
        RuntimeError(
            "CUDA error: device-side assert triggered\nAssertion `srcIndex < "
            "srcSelectDimSize` failed."
        )
    )
    assert poisons(RuntimeError("indexSelectSmallIndex: srcIndex < srcSelectDimSize"))
    assert poisons(RuntimeError("an illegal memory access was encountered"))
    # A shape refusal or an allocator failure is a fact about the cell, not the context.
    assert not poisons(ValueError("d_head 256 is not on the measured MMA list"))
    assert not poisons(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))


def test_a_void_marker_is_a_record_and_the_cell_stays_pending(tmp_path: object) -> None:
    """A cell that died is documented, not skipped: a silently dropped cell is a hole."""
    bank = str(tmp_path)
    stored = {
        "schema": BANK_SCHEMA,
        "torch": "2.7.1+cu126",
        "device": "NVIDIA RTX A6000",
        "slinoss_package": "/tree/slinoss/__init__.py",
        "slinoss_sources": "0123456789abcdef",
    }
    key = Task(point=a_point(128), boundary=RECURRENCE, execution=GRAPH).key
    written = write_void(bank, key, "device-side assert triggered", stored=stored)
    assert written.endswith(VOID_SUFFIX)
    banked, refused = read_bank(bank, stored=stored)
    # Neither a cell nor a refusal: the marker is a third thing.
    assert banked == {}
    assert refused == ()
    voids = read_voids(bank)
    assert len(voids) == 1
    assert key in voids[0]
    assert "device-side assert" in voids[0]


def test_a_whole_step_row_names_the_decode_stage_and_a_banked_one_still_reads() -> None:
    """Routing moved the whole step onto the decode kernel; the pre-routing stage name
    stays classifiable so an old record does not silently reclassify."""
    routed = {"conv": "native", "prep": "cute", "decode": "cute", "tail": "cute"}
    assert path_class(routed) == KERNEL_PATH
    assert path_class({**routed, "conv": REFERENCE}) == MIXED_PATH
    assert path_class({**routed, "decode": REFERENCE}) == REFERENCE_PATH
    pre = {"conv": "native", "prep": "cute", "chunked_scan": REFERENCE, "tail": "cute"}
    assert path_class(pre) == REFERENCE_PATH
    assert "2.5281x" in ROUTING_DISCLOSURE
    assert "void" in ROUTING_DISCLOSURE
    # A disclosure that is written and never wired in is not disclosed, so the count is
    # asserted and each entry appears once.
    assert len(DISCLOSURES) == len(set(DISCLOSURES)) == 15
    assert any(SAMPLE_COUNT_DISCLOSURE_LABEL in one for one in DISCLOSURES)
    assert any("BOTH TREES COUNT" in one for one in DISCLOSURES)
    # The claim that fp32 makes the whole step a mixed path died with routing.
    assert not any("PRE-ROUTING" in one for one in DISCLOSURES)
    assert "false in this tree" in FP32_DISCLOSURE
