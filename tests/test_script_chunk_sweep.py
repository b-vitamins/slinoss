"""The chunk sweep's arithmetic: its legality map, its byte model, its probe.

This pins ``scripts.perf.chunk_sweep``. Nothing here reads a clock or launches a
kernel. The three computed modes are pure host arithmetic over the shipped layout
functions and the shipped shapes, so every figure a test asserts is exact, and the
one measured mode is left to the driver.

The tests that evaluate a real kernel budget need the CuTe layout modules and carry
the ``cute`` mark; the rest need neither a device nor the DSL. Capacity is a
literal wherever a verdict depends on it: querying the device would make the
verdicts a property of the host.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("cutlass")

from scripts.perf.chunk_sweep import (
    CANDIDATES,
    FLAT,
    INVERSE,
    LAUNCH_METRICS,
    MODES,
    RESIDENT_TARGET,
    SHIFTED,
    ArenaKernel,
    ArenaRow,
    Geometry,
    OccupancyRow,
    arena_rows,
    budget_at,
    flop_terms,
    geometry_of,
    legal_chunks,
    parse_args,
    prefix_probe,
    refusing_extent,
    residency_at,
    resident_chunks,
    step_model,
    traffic_terms,
)
from slinoss.config import HEAD_MULTIPLE, MAX_CHUNK, MIN_CHUNK, STATE_MULTIPLE
from slinoss.perf.workload import SHAPES, shape_by_name

CUTE = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
"""The tests that import the kernels' own layout functions."""

CAPACITY = 101376
"""The sm_86 opt-in carveout, as a literal. Every verdict here is judged against it."""

ACCEPTANCE = Geometry(bsz=4, heads=18, groups=1, seqlen=2048, rows=64, dim=240)
"""The geometry the whole-step attribution defaults to."""

COUNTED_FLOP: dict[str, int] = {
    "increment_passing_fwd": 30_720,
    "chunk_scan_fwd": 69_632,
    "start_passing_bwd": 30_720,
    "chunk_input_bwd": 217_088,
    "chunk_vector_bwd": 296_960,
}
"""Flop per token per head at :data:`ACCEPTANCE` and ``L 64``, from the counter.

``sm__inst_executed_pipe_tensor.sum`` a launch on an RTX A6000, at 4,096 flop a
warp-level MMA over ``B*T*H = 147,456``, which is the counter over 36: 1,105,920,
2,506,752, 1,105,920, 7,815,168 and 10,690,560 warp-inst. The counter is the whole
GEMM census, no kernel in the tree emitting ``hfma``, ``hadd`` or ``hmul`` at all.

The first two are post-fusion. One forcing column replaced two in the increment
pass, and one score column and one diagonal replaced two of each in the scan, so
those two counters halved every term but the scan's chunk offset.

Keyed by launch. ``chunk_start_bwd`` is not among them: the backward fused that GEMM
into ``start_passing_bwd``, and :mod:`slinoss.perf.coverage` declares the fused-away
name targeted.
"""

COUNTED_STEP = 645_120
"""Sum of :data:`COUNTED_FLOP`. By hand, 23,224,320 warp-inst over 36."""


def fixed(*, at_widths: int, at_floor: int) -> ArenaKernel:
    """A kernel whose budget is a literal at the geometry and at the narrowest widths.

    :func:`refusing_extent` reaches five verdicts by re-evaluating one layout function
    at four extent combinations. A shipped function reaches at most two of them at any
    one geometry, so the arcs are covered by a stub that answers each combination
    independently.

    Args:
        at_widths: Bytes at ``P`` and ``3N`` of the geometry, whatever ``L``.
        at_floor: Bytes at :data:`HEAD_MULTIPLE` and :data:`STATE_MULTIPLE`.

    Returns:
        The kernel. Narrowing one extent alone yields the mean of the two.
    """

    def nbytes(chunk: int, rows: int, dim: int) -> int:
        del chunk
        narrow_rows = rows == HEAD_MULTIPLE
        narrow_dim = dim == STATE_MULTIPLE
        if narrow_rows and narrow_dim:
            return at_floor
        if narrow_rows or narrow_dim:
            return (at_widths + at_floor) // 2
        return at_widths

    return ArenaKernel(name="stub", nbytes=nbytes, knob=lambda c, p, d: "-")


def unit_transform(seqlen: int, *, decay: float, angle: float) -> torch.Tensor:
    """A ``(1,1,T,4)`` transform packing one constant rotation and one constant decay.

    Args:
        seqlen: ``T``.
        decay: ``ls`` at every step. Non-positive, as I1 requires.
        angle: Magnitude of ``w`` at every step, on the x axis.

    Returns:
        The transform at float64, the oracle width the probe is handed.
    """
    out = torch.zeros(1, 1, seqlen, 4, dtype=torch.float64)
    out[..., 0] = angle
    out[..., 3] = decay
    return out


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_parse_args_defaults_the_acceptance_sweep_and_takes_every_mode() -> None:
    default = parse_args([])
    assert default.mode == "arena"
    assert default.shape == "acceptance"
    assert default.chunks == list(CANDIDATES)
    # Zero queries the device; both denominators absent leaves the model timeless.
    assert default.capacity == 0
    assert default.dram_gbs is None
    assert default.peak_tflops is None
    for name in MODES:
        assert parse_args(["--mode", name]).mode == name
    assert MODES == (
        "arena",
        "traffic",
        "numerics",
        "step",
        "op",
        "occupancy",
        "launch",
    )
    # The op mode pairs against a fixed length, so the default has to be one the
    # acceptance arena admits or every pair loses its baseline arm.
    assert default.op_ref == 64
    args = parse_args(
        [
            "--mode",
            "traffic",
            "--shape",
            "wide",
            "--chunks",
            "32",
            "64",
            "--capacity",
            "65536",
            "--dram-gbs",
            "685.22",
            "--peak-tflops",
            "112.0",
        ]
    )
    assert args.mode == "traffic"
    assert args.shape == "wide"
    assert args.chunks == [32, 64]
    assert args.capacity == 65536
    assert args.dram_gbs == pytest.approx(685.22)
    assert args.peak_tflops == pytest.approx(112.0)
    with pytest.raises(SystemExit) as caught:
        parse_args(["--mode", "nonesuch"])
    assert caught.value.code == 2


def test_the_geometry_is_a_named_shape_with_the_chunk_length_dropped() -> None:
    shape = shape_by_name("acceptance")
    geo = geometry_of(shape)
    assert geo == ACCEPTANCE
    assert geo.dim == shape.d_state == 240
    assert geo.fold == 18
    # Every shape's fold divides, so no shape resolves to a fractional fold.
    for other in SHAPES:
        assert other.heads % other.groups == 0
        assert geometry_of(other).fold == other.heads // other.groups
    # A ragged tail is a whole chunk: 2004 at 64 is 32 chunks and a remainder.
    assert geo.chunks(64) == 32
    assert geometry_of(shape_by_name("ragged")).chunks(64) == 32
    assert ACCEPTANCE.describe() == (
        "B=4 H=18 G=1 T=2048 P=64 3N=240 fold=18 itemsize=2"
    )


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def test_a_refusal_names_the_extent_that_narrowing_would_relieve() -> None:
    fits = fixed(at_widths=CAPACITY, at_floor=CAPACITY)
    assert refusing_extent(fits, 64, ACCEPTANCE, CAPACITY) == ""
    # Over capacity at the narrowest legal widths: L alone carries the excess.
    only_chunk = fixed(at_widths=4 * CAPACITY, at_floor=2 * CAPACITY)
    assert refusing_extent(only_chunk, 64, ACCEPTANCE, CAPACITY) == "chunk"
    # Narrowing either extent alone suffices. 3N is tried first, so it is named.
    either = fixed(at_widths=2 * CAPACITY, at_floor=0)
    assert refusing_extent(either, 64, ACCEPTANCE, CAPACITY) == "dim"
    # Neither alone suffices, but both together do.
    both = fixed(at_widths=4 * CAPACITY, at_floor=CAPACITY)
    assert refusing_extent(both, 64, ACCEPTANCE, CAPACITY) == "rows+dim"


def test_a_width_the_launch_falls_back_from_refuses_nothing() -> None:
    # Three kernels get a row per block width. Two of them narrow the block when the
    # wide arena does not fit, so a wide row over capacity is not a verdict on ``L``
    # and must not remove the length or the residency the narrow block does reach.
    def row(name: str, *, nbytes: int, refused: str, binding: bool) -> ArenaRow:
        return ArenaRow(
            kernel=name,
            chunk=64,
            knob="-",
            smem_bytes=nbytes,
            capacity_pct=100.0 * nbytes / CAPACITY,
            resident=residency_at(nbytes, CAPACITY),
            floor_bytes=nbytes,
            refused_by=refused,
            binding=binding,
        )

    narrow = row("k/w4", nbytes=40_000, refused="", binding=True)
    wide = row("k/w8", nbytes=2 * CAPACITY, refused="rows+dim", binding=False)
    assert legal_chunks([narrow, wide]) == (64,)
    assert resident_chunks([narrow, wide], RESIDENT_TARGET) == (64,)
    # The same row binding does refuse it, so the filter is the flag and not the name.
    binds = wide._replace(binding=True)
    assert legal_chunks([narrow, binds]) == ()
    assert resident_chunks([narrow, binds], RESIDENT_TARGET) == ()


@CUTE
@pytest.mark.cute
def test_every_arena_row_agrees_with_the_carveout_it_was_judged_against() -> None:
    from slinoss._cute import smem_residency

    rows = arena_rows(ACCEPTANCE, CANDIDATES, CAPACITY)
    assert {row.chunk for row in rows} == set(CANDIDATES)
    for row in rows:
        assert row.smem_bytes > 0
        assert row.capacity_pct == pytest.approx(100.0 * row.smem_bytes / CAPACITY)
        # The report must not claim a residency the launch bound will not ask
        # for, so it is held against the launch's own helper, not to a divide.
        assert row.resident == smem_residency(row.smem_bytes, capacity=CAPACITY)
        # A verdict and its bytes cannot disagree: one is derived from the other.
        assert bool(row.refused_by) == (row.smem_bytes > CAPACITY)
        assert row.floor_bytes <= row.smem_bytes
        assert (row.refused_by == "chunk") == (row.floor_bytes > CAPACITY)
    fitting = {row.chunk for row in rows} - {
        row.chunk
        for row in rows
        if row.binding and (row.refused_by or row.refused_slice)
    }
    assert legal_chunks(rows) == tuple(
        sorted(c for c in fitting if MIN_CHUNK <= c <= MAX_CHUNK)
    )
    short = {
        row.chunk for row in rows if row.resident < RESIDENT_TARGET and row.binding
    }
    assert set(resident_chunks(rows, RESIDENT_TARGET)) == set(CANDIDATES) - short


@CUTE
@pytest.mark.cute
def test_a_length_every_arena_fits_is_still_refused_by_a_k_slice_that_misses() -> None:
    """The two refusals are on two axes, and the arena alone is not the map.

    ``L = 48`` fits every kernel's carveout at every shipped shape and does not run.
    Three launches cap a block at 32 and 48 does not divide it: ``chunk_scan_fwd``
    its score columns, ``chunk_input_bwd`` its target tokens, ``chunk_vector_bwd``
    its source tokens. Reading only the byte column reported 48 as legal, and the
    operator raised from
    :func:`slinoss.ops.so3ssd.cute.guard.check_extents` on the first launch.
    """
    rows = arena_rows(ACCEPTANCE, [48, 64, 80], CAPACITY)
    missed = {
        (row.kernel, row.chunk, row.slice_width)
        for row in rows
        if row.refused_slice and row.binding
    }
    assert missed == {
        ("chunk_scan_fwd/w4", 48, 32),
        ("chunk_input_bwd/w4", 48, 32),
        ("chunk_input_bwd/w4", 80, 32),
        ("chunk_vector_bwd/fold1/w8", 80, 32),
        ("chunk_vector_bwd/fold18/w8", 80, 32),
    }
    fits = {(row.kernel, row.chunk) for row in rows if row.smem_bytes <= CAPACITY}
    # Every L = 48 refusal fits its arena, so the byte column acquits what the
    # slice refuses and the arena axis alone cannot produce this answer.
    assert {(k, c) for k, c, _ in missed if c == 48} <= fits
    # At L = 80 the two axes disagree in both directions on the same length: the
    # target-token block misses at a width that fits, the source-token block
    # misses at a width that does not.
    assert ("chunk_input_bwd/w4", 80) in fits
    assert ("chunk_vector_bwd/fold18/w8", 80) not in fits
    assert 48 not in legal_chunks(rows)
    assert 64 in legal_chunks(rows)
    # A row that declares no slice divides nothing and refuses nothing.
    assert all(row.slice_width > 0 for row in rows)


def test_an_occupancy_row_names_the_resource_that_sets_its_residency() -> None:
    """The arena mode cannot reach this verdict, so the counter row has to.

    Host arithmetic over the layout functions prices shared memory and is blind to the
    register file. A length that frees enough shared memory for a second block and
    still runs one is the case the two columns exist to separate, and a report that
    printed only the shared limit would call that a residency step.
    """

    def occ(smem_limit: int, register_limit: int) -> OccupancyRow:
        return OccupancyRow(
            kernel="k",
            chunk=32,
            smem_bytes=48_000,
            registers=168,
            smem_limit=smem_limit,
            register_limit=register_limit,
            waves=8.57,
            blocks=2880,
            threads=256,
        )

    assert occ(2, 3).resident == 2
    assert occ(2, 3).bound_by == "smem"
    assert occ(3, 2).resident == 2
    assert occ(3, 2).bound_by == "regs"
    assert occ(2, 2).bound_by == "both"
    # The counters are requested in print order and carry no duration: a launch
    # configuration is not a measurement of time and must not be read as one.
    assert LAUNCH_METRICS[:4] == (
        "launch__shared_mem_per_block",
        "launch__registers_per_thread",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_registers",
    )
    assert not any("time" in metric for metric in LAUNCH_METRICS)


@CUTE
@pytest.mark.cute
def test_the_residency_bar_is_the_granular_one_not_the_capacity_divided() -> None:
    from slinoss._cute import smem_residency

    # k blocks pay k reservations against a capacity that has one subtracted
    # already, and each is rounded up to a granule. Both corrections are needed:
    # a plain divide reads one block high in a 512 B band under the two-block
    # bar, which is where every kernel in the tree sits.
    assert budget_at(2, CAPACITY) == 50176
    assert CAPACITY // 2 == 50688
    assert residency_at(50176, CAPACITY) == 2
    assert residency_at(50177, CAPACITY) == 1
    assert residency_at(50688, CAPACITY) == 1
    # Each bar is the largest budget that reaches it, so one byte more loses a
    # block. One block is the floor and has no bar above it to fall from.
    assert residency_at(budget_at(1, CAPACITY) + 1, CAPACITY) == 1
    for blocks in (2, 3, 4, 5, 6):
        budget = budget_at(blocks, CAPACITY)
        assert residency_at(budget, CAPACITY) == blocks
        assert residency_at(budget + 1, CAPACITY) == blocks - 1
    # Against the launch bound, which queries the device for the same carveout.
    for nbytes in (1, 4096, 33024, 50176, 50177, 101376):
        assert residency_at(nbytes, CAPACITY) == smem_residency(
            nbytes, capacity=CAPACITY
        )


# ---------------------------------------------------------------------------
# Traffic and arithmetic
# ---------------------------------------------------------------------------


def test_no_global_memory_term_grows_with_the_chunk_length() -> None:
    # The claim the byte model rests on: the L x L score never reaches global
    # memory, so every term is flat, 1/L, or the one extra row a shifted span
    # reads. A term that grew with L would put a minimum inside the range.
    here = traffic_terms(ACCEPTANCE, 64)
    there = traffic_terms(ACCEPTANCE, 128)
    assert len(here) == len(there)
    assert {t.side for t in here} == {"read", "write"}
    assert {t.scaling for t in here} == {FLAT, INVERSE, SHIFTED}
    for near, far in zip(here, there, strict=True):
        assert (near.kernel, near.tensor, near.side) == (
            far.kernel,
            far.tensor,
            far.side,
        )
        if near.scaling == FLAT:
            assert far.nbytes == near.nbytes
        elif near.scaling == INVERSE:
            assert 2 * far.nbytes == near.nbytes
        else:
            assert near.nbytes // 2 < far.nbytes < near.nbytes
    # The backward reads the chunk start states the forward left, so no launch in
    # the model re-runs a forward kernel and no term is counted twice.
    assert not [t.kernel for t in here if "[remat]" in t.kernel]


@CUTE
@pytest.mark.cute
def test_the_arithmetic_is_the_counted_arithmetic_of_the_launches_that_run() -> None:
    terms = flop_terms(ACCEPTANCE, 64)
    assert {term.kernel: term.flop for term in terms} == COUNTED_FLOP
    assert sum(term.flop for term in terms) == COUNTED_STEP
    # Launch order, and the launch names: a term billed to a kernel that never runs
    # is arithmetic no profile can be scored against.
    assert [term.kernel for term in terms] == list(COUNTED_FLOP)
    for term in terms:
        assert term.flop > 0
    lanes = ACCEPTANCE.bsz * ACCEPTANCE.heads * ACCEPTANCE.seqlen
    assert lanes == 147_456
    model = step_model(ACCEPTANCE, 64, dram_gbs=None, peak_tflops=None)
    assert model.flop == lanes * COUNTED_STEP
    # 95.13 GFLOP a step. The forward tap fusion took 10.26 GFLOP off the 105.39 the
    # two-tap tree paid, which is 9.74% of the step's arithmetic.
    assert model.flop == 95_126_814_720


@CUTE
@pytest.mark.cute
def test_no_kernel_pays_P_and_3N_at_one_rate() -> None:
    # A ``(P + 3N)`` coefficient is symmetric under the swap and the operator is
    # not: the input backward pays ``P`` twice for every once it pays ``3N``, and
    # the vector backward pays ``3N`` twice. Both geometries are legal, ``3N``
    # being a multiple of 48 and ``P`` of HEAD_MULTIPLE either way.
    wide_state = Geometry(bsz=4, heads=18, groups=1, seqlen=2048, rows=48, dim=96)
    wide_head = Geometry(bsz=4, heads=18, groups=1, seqlen=2048, rows=96, dim=48)
    here = {term.kernel: term.flop for term in flop_terms(wide_state, 64)}
    there = {term.kernel: term.flop for term in flop_terms(wide_head, 64)}
    # By hand, not by helper: ``8*P*3N + 4*mma_rows(L)*3N + 8*mma_rows(L)*P``, at
    # ``mma_rows(64) == 64``.
    assert here["chunk_input_bwd"] == 8 * 48 * 96 + 4 * 64 * 96 + 8 * 64 * 48
    assert there["chunk_input_bwd"] == 8 * 96 * 48 + 4 * 64 * 48 + 8 * 64 * 96
    assert here["chunk_input_bwd"] != there["chunk_input_bwd"]
    # ``6*P*3N + 4*mma_rows(L)*P*tiles + 8*mma_rows(L)*3N``: the vector backward's
    # score carries no state extent and the lane tile is a grid axis, so ``3N``
    # sets how many times it is recomputed. Two tiles at 96 against one at 48.
    assert here["chunk_vector_bwd"] == 6 * 48 * 96 + 4 * 64 * 48 * 2 + 8 * 64 * 96
    assert there["chunk_vector_bwd"] == 6 * 96 * 48 + 4 * 64 * 96 * 1 + 8 * 64 * 48
    assert here["chunk_vector_bwd"] != there["chunk_vector_bwd"]
    # The flat parts contract over ``P * 3N`` and are symmetric, so the asymmetry
    # above is the L-linear part alone and nothing else.
    for kernel in ("increment_passing_fwd", "chunk_scan_fwd", "start_passing_bwd"):
        assert here[kernel] == there[kernel]


@CUTE
@pytest.mark.cute
def test_the_padded_chunk_mode_holds_the_arithmetic_flat_below_one_M_tile() -> None:
    # ``L`` is the M mode of every GEMM the scan and the two chunk-local backward
    # kernels issue, and M is the one mode the atom rounds up, so a chunk under one
    # M tile executes a whole tile of work. Counted, chunk_input_bwd runs 2,656
    # tensor warp-inst a block at ``L 32`` where the unpadded form predicts 1,328.
    per_block = {
        chunk: next(
            term.flop
            for term in flop_terms(ACCEPTANCE, chunk)
            if term.kernel == "chunk_input_bwd"
        )
        * chunk
        for chunk in (32, 64)
    }
    assert per_block[32] == 2_656 * 4096
    assert per_block[64] == 3_392 * 4096
    # Twice the blocks at half the chunk against a block that shrank by less than
    # half, so the per-token cost rises as the chunk falls. ``L 128`` is not pinned:
    # the counter reads 8,768 warp-inst there against 9,728 from this form, and the
    # 960 are unexplained.
    assert per_block[32] // 32 > per_block[64] // 64


def test_the_step_model_takes_the_larger_floor_and_omits_what_it_lacks() -> None:
    timeless = step_model(ACCEPTANCE, 64, dram_gbs=None, peak_tflops=None)
    terms = traffic_terms(ACCEPTANCE, 64)
    assert timeless.read_bytes == sum(t.nbytes for t in terms if t.side == "read")
    assert timeless.write_bytes == sum(t.nbytes for t in terms if t.side == "write")
    assert timeless.total_bytes == timeless.read_bytes + timeless.write_bytes
    assert timeless.intensity == pytest.approx(timeless.flop / timeless.total_bytes)
    assert timeless.dram_us is None
    assert timeless.tensor_us is None
    assert timeless.model_us is None
    # A model time appears only when both denominators are measured, and it is the
    # larger of the two floors.
    timed = step_model(ACCEPTANCE, 64, dram_gbs=685.22, peak_tflops=112.0)
    dram_us = timed.dram_us
    tensor_us = timed.tensor_us
    assert dram_us is not None and tensor_us is not None
    assert dram_us == pytest.approx(timed.total_bytes / (1e3 * 685.22))
    assert tensor_us == pytest.approx(timed.flop / 112e6)
    assert timed.model_us == max(dram_us, tensor_us)
    # Bytes fall with L monotonically. Arithmetic does not rise with it: L is the M
    # mode of three kernels and the atom rounds M up to one tile, so below 64 the
    # work is flat in L and the per-token cost falls to a minimum at one tile.
    models = [
        step_model(ACCEPTANCE, chunk, dram_gbs=685.22, peak_tflops=112.0)
        for chunk in CANDIDATES
    ]
    bytes_ = [model.total_bytes for model in models]
    flops = [model.flop for model in models]
    assert bytes_ == sorted(bytes_, reverse=True)
    assert flops != sorted(flops)
    assert min(models, key=lambda model: model.flop).chunk == 64


# ---------------------------------------------------------------------------
# The invariant probe
# ---------------------------------------------------------------------------


def test_the_prefix_probe_reads_the_product_before_it_is_renormalized() -> None:
    from slinoss.ops.so3ssd.reference import quat_exp, quat_prefix_scan

    trans = unit_transform(256, decay=-0.25, angle=0.5)
    decay, drift = prefix_probe(trans, 64)
    # I1: the chunk-local decay is exp(2 * sum ls) over the chunk, at its end.
    assert decay == pytest.approx(math.exp(2.0 * -0.25 * 64), rel=1e-6)
    # I5: the drift is of the unnormalized product at the pinned width, so it is
    # nonzero, and it is orders below the bf16 operand epsilon.
    assert 0.0 < drift < 1e-4
    # quat_prefix_scan divides by the norm, so its output carries the effect of
    # the renormalization and not the drift the renormalization absorbs.
    folded = trans.unflatten(2, (-1, 64))
    scanned = quat_prefix_scan(quat_exp(0.5 * folded[..., :3].to(torch.float32)))
    assert (scanned.norm(dim=-1) - 1.0).abs().max().item() < drift
    # A longer prefix drifts further, which is what makes the cadence a question.
    assert prefix_probe(trans, 128)[1] > drift
    # A ragged tail is dropped, so a partial chunk cannot set either reading.
    assert prefix_probe(unit_transform(200, decay=-0.25, angle=0.5), 64) == (
        decay,
        drift,
    )
