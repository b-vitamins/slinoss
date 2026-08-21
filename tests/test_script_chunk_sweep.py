"""The chunk sweep's arithmetic: its legality map, its byte model, its probe.

This pins ``scripts.perf.chunk_sweep``. Nothing here reads a clock or launches a
kernel. The three computed modes are pure host arithmetic over the shipped layout
functions and the shipped shapes, so every figure a test asserts is exact, and the
one measured mode is left to the driver.

The two tests that evaluate a real kernel budget need the CuTe layout modules and
carry the ``cute`` mark; the rest need neither a device nor the DSL. Capacity is a
literal wherever a verdict depends on it: querying the device would make the
verdicts a property of the host.
"""

from __future__ import annotations

import math

import pytest
import torch

from scripts.perf.chunk_sweep import (
    CANDIDATES,
    FLAT,
    INVERSE,
    MODES,
    RESIDENT_TARGET,
    SHIFTED,
    ArenaKernel,
    Geometry,
    arena_rows,
    flop_terms,
    geometry_of,
    legal_chunks,
    parse_args,
    prefix_probe,
    refusing_extent,
    resident_chunks,
    step_model,
    traffic_terms,
)
from slinoss.config import HEAD_MULTIPLE, MAX_CHUNK, MIN_CHUNK, STATE_MULTIPLE
from slinoss.perf.workload import SHAPES, shape_by_name

CUTE = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
"""The two tests that import the kernels' own layout functions."""

CAPACITY = 101376
"""The sm_86 opt-in carveout, as a literal. Every verdict here is judged against it."""

ACCEPTANCE = Geometry(bsz=4, heads=18, groups=1, seqlen=2048, rows=64, dim=240)
"""The geometry the whole-step attribution defaults to."""


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
    assert MODES == ("arena", "traffic", "numerics", "step", "op")
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


@CUTE
@pytest.mark.cute
def test_every_arena_row_agrees_with_the_carveout_it_was_judged_against() -> None:
    rows = arena_rows(ACCEPTANCE, CANDIDATES, CAPACITY)
    assert {row.chunk for row in rows} == set(CANDIDATES)
    for row in rows:
        assert row.smem_bytes > 0
        assert row.capacity_pct == pytest.approx(100.0 * row.smem_bytes / CAPACITY)
        assert row.resident == CAPACITY // row.smem_bytes
        # A verdict and its bytes cannot disagree: one is derived from the other.
        assert bool(row.refused_by) == (row.smem_bytes > CAPACITY)
        assert row.floor_bytes <= row.smem_bytes
        assert (row.refused_by == "chunk") == (row.floor_bytes > CAPACITY)
    fitting = {row.chunk for row in rows} - {
        row.chunk for row in rows if row.refused_by
    }
    assert legal_chunks(rows) == tuple(
        sorted(c for c in fitting if MIN_CHUNK <= c <= MAX_CHUNK)
    )
    short = {row.chunk for row in rows if row.resident < RESIDENT_TARGET}
    assert set(resident_chunks(rows, RESIDENT_TARGET)) == set(CANDIDATES) - short


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
def test_the_arithmetic_is_affine_in_the_chunk_length() -> None:
    terms = flop_terms(ACCEPTANCE)
    for term in terms:
        assert term.flat > 0
        assert term.linear >= 0
        assert term.flops(64) == term.flat + 64 * term.linear
        assert term.flops(0) == term.flat
    carries = {term.kernel for term in terms if term.linear > 0}
    # Only the score and the diagonal contract over the chunk.
    assert carries == {"chunk_scan_fwd", "chunk_input_bwd", "chunk_vector_bwd"}


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
    # Bytes fall and arithmetic rises with L, in both directions, monotonically.
    models = [
        step_model(ACCEPTANCE, chunk, dram_gbs=685.22, peak_tflops=112.0)
        for chunk in CANDIDATES
    ]
    bytes_ = [model.total_bytes for model in models]
    flops = [model.flop for model in models]
    assert bytes_ == sorted(bytes_, reverse=True)
    assert flops == sorted(flops)


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
