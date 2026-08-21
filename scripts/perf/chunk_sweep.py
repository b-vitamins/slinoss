"""Chunk length as a free parameter: what refuses it, what it costs, what it buys.

``L`` is not tuned. It is pinned by the shared-memory arena of two kernels, and every
other consequence of the choice follows from that. Four modes separate the questions
so that a computed row and a measured row never share a table::

    python3 scripts/perf/chunk_sweep.py --mode arena
    python3 scripts/perf/chunk_sweep.py --mode traffic
    python3 scripts/perf/chunk_sweep.py --mode numerics
    python3 scripts/perf/chunk_sweep.py --mode step

``arena`` is the legality map. The shipped layout functions are evaluated at each
candidate ``L`` against the device's carveout and against half of it, which is what
two resident blocks need. Host arithmetic over the same functions the kernels call,
so there is one description of each budget and nothing here can drift from it. For
every refused ``L`` the report names which extent carries the excess, by
re-evaluating the same function at the narrowest legal ``P`` and ``3N``: an ``L``
whose footprint exceeds capacity at the narrowest widths is refused by ``L`` alone.

``traffic`` is the analytic byte model with ``L`` free. Each term is classified by
what it does under a doubling of ``L``, measured on the term itself rather than
declared. Compulsory traffic only: one pass per operand per launch, at the ``L + 1``
rows a shifted span reads. Tile re-reads above one lane tile, band re-reads across
the heads of a group, and cache hits are not modelled, and none of the three depends
on ``L``. Bytes and FLOPs are reported; a time appears only when ``--dram-gbs`` and
``--peak-tflops`` supply a measured denominator, and it is labelled a model.

``numerics`` is what a longer prefix does to the invariants. I1 bounds the chunk
decay, I2 the per-step angle, I5 renormalizes the quaternion prefix once per chunk,
so the drift a single renormalization has to absorb grows with ``L``. The probe
reports the drift before that renormalization and the error of three arms against a
float64 oracle, which separates the prefix error from the operand error. Measured.

``step`` is the whole-step time at each legal ``L``, with the per-kernel attribution
and a parity check against the reference at the same ``L``, so a faster wrong answer
is caught.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from slinoss.config import HEAD_MULTIPLE, MAX_CHUNK, MIN_CHUNK, STATE_MULTIPLE
from slinoss.ops.so3ssd.reference import SO3SSDResult
from slinoss.perf.workload import OpShape, shape_by_name

MODES = ("arena", "traffic", "numerics", "step")

CANDIDATES: tuple[int, ...] = (16, 32, 64, 128, 256)
"""Chunk lengths the sweep asks about.

Powers of two, since :class:`slinoss.config.SLinOSSConfig` admits no other, spanning
:data:`slinoss.config.MIN_CHUNK` to one doubling past
:data:`slinoss.config.MAX_CHUNK`. The last entry is outside the config's range and is
reported rather than omitted: whether the arithmetic or the range refuses it first is
the question the mode exists to answer.
"""

RESIDENT_TARGET = 2
"""Blocks per multiprocessor the carveout is also checked against.

One block of :data:`slinoss.ops.so3ssd.cute.common.THREADS` threads occupies 128 of
an sm_86 multiprocessor's 1,536 thread slots, so a kernel that fits once and not
twice is bounded by its arena and not by its register count.
"""

WIDE = torch.float64
"""The oracle width. Every arm of the numerics mode is cast from one draw at this
width, so the arms differ by rounding and not by their inputs."""


@dataclass(frozen=True)
class Geometry:
    """The shape held fixed while ``L`` varies.

    Attributes:
        bsz: ``B``.
        heads: ``H``.
        groups: ``G``. Divides ``heads``.
        seqlen: ``T``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. 2 for the tensor-core atom.
    """

    bsz: int
    heads: int
    groups: int
    seqlen: int
    rows: int
    dim: int
    itemsize: int = 2

    @property
    def fold(self) -> int:
        """Heads one block of the vector backward walks, ``H // G``."""
        return self.heads // self.groups

    def chunks(self, chunk: int) -> int:
        """Chunks a sequence of this length splits into at ``chunk``."""
        return -(-self.seqlen // chunk)

    def describe(self) -> str:
        """One line for a report header."""
        return (
            f"B={self.bsz} H={self.heads} G={self.groups} T={self.seqlen} "
            f"P={self.rows} 3N={self.dim} fold={self.fold} itemsize={self.itemsize}"
        )


def geometry_of(shape: OpShape) -> Geometry:
    """The sweep geometry a named shape resolves to, with ``L`` dropped.

    Args:
        shape: A shape from :data:`slinoss.perf.workload.SHAPES`.

    Returns:
        Its geometry. The sweep supplies ``L``.
    """
    return Geometry(
        bsz=shape.bsz,
        heads=shape.heads,
        groups=shape.groups,
        seqlen=shape.seq,
        rows=shape.rows,
        dim=shape.d_state,
    )


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


class ArenaKernel(NamedTuple):
    """One kernel's shared-memory budget as a function of the three extents.

    Attributes:
        name: Kernel name, with the fold appended where the budget depends on it.
        nbytes: ``(chunk, rows, dim) -> bytes``, the shipped layout function.
        knob: ``(chunk, rows, dim) -> str``, the slice width that layout chose.
    """

    name: str
    nbytes: Callable[[int, int, int], int]
    knob: Callable[[int, int, int], str]


def arena_kernels(fold: int) -> tuple[ArenaKernel, ...]:
    """Every kernel on the CuTe path that allocates shared memory.

    The backward's state passing and the boundary allocate none, so ``L`` cannot
    refuse them and they do not appear.

    Args:
        fold: Heads one vector-backward block walks. A fold above one adds the
            cross-head accumulator to that kernel's arena, so both folds appear
            when they differ.

    Returns:
        One entry per kernel, forward launches first.
    """
    from slinoss.ops.so3ssd.cute.bwd.chunk_input import input_smem_bytes, lblock
    from slinoss.ops.so3ssd.cute.bwd.chunk_start import start_smem_bytes
    from slinoss.ops.so3ssd.cute.bwd.chunk_vector import vblock, vector_smem_bytes
    from slinoss.ops.so3ssd.cute.fwd.chunk_scan import nblock, scan_smem_bytes
    from slinoss.ops.so3ssd.cute.fwd.increment_passing import (
        SPLIT,
        fused_kblock,
        fused_smem_bytes,
    )

    def input_knob(chunk: int, rows: int, dim: int) -> str:
        # The map asks about chunk lengths the kernel refuses, and at those there is no
        # lane block to name: ``lblock`` returns a block that fits or raises. The bytes
        # column still reports the narrowest block's cost, which is what the refusal is
        # judged on.
        try:
            return f"lblk={lblock(chunk, rows, dim)}"
        except ValueError:
            return "lblk=none"

    def vector(f: int) -> ArenaKernel:
        return ArenaKernel(
            name=f"chunk_vector_bwd/fold{f}",
            nbytes=lambda c, p, d: vector_smem_bytes(c, p, d, f, vblock(c, p, d, f)),
            knob=lambda c, p, d: f"span={vblock(c, p, d, f)}",
        )

    entries = [
        ArenaKernel(
            name="increment_passing_fwd",
            # The band width is fixed, so the arena follows ``L`` and ``P`` alone:
            # the band is what makes the operand tiles independent of ``3N``. The
            # slice is the widest the residency admits, which is where ``L`` enters
            # twice, once through the chunk-sized tiles and once through the slice
            # the budget left for them.
            nbytes=lambda c, p, d: fused_smem_bytes(
                c, p, SPLIT, kblk=fused_kblock(c, p, SPLIT)
            ),
            knob=lambda c, p, d: f"span={SPLIT} kblk={fused_kblock(c, p, SPLIT)}",
        ),
        ArenaKernel(
            name="chunk_scan_fwd",
            nbytes=scan_smem_bytes,
            knob=lambda c, p, d: f"nblk={nblock(c)}",
        ),
        ArenaKernel(
            name="chunk_start_bwd",
            nbytes=start_smem_bytes,
            knob=lambda c, p, d: "-",
        ),
        ArenaKernel(
            name="chunk_input_bwd",
            nbytes=input_smem_bytes,
            knob=input_knob,
        ),
    ]
    entries.extend(vector(f) for f in dict.fromkeys((1, fold)))
    return tuple(entries)


class ArenaRow(NamedTuple):
    """One kernel at one chunk length.

    Attributes:
        kernel: Kernel name.
        chunk: ``L``.
        knob: The slice width that layout chose.
        smem_bytes: Bytes the arena spans.
        capacity_pct: Those bytes over the carveout.
        resident: Blocks the carveout holds, floor of the ratio.
        floor_bytes: The same function at the narrowest legal ``P`` and ``3N``, so
            the part of the footprint ``L`` alone forces.
        refused_by: Empty when the kernel fits. Otherwise the extent carrying the
            excess: ``chunk`` when the narrowest widths already exceed capacity,
            ``dim`` or ``rows`` when narrowing that one alone suffices, ``rows+dim``
            when neither alone does.
    """

    kernel: str
    chunk: int
    knob: str
    smem_bytes: int
    capacity_pct: float
    resident: int
    floor_bytes: int
    refused_by: str


def refusing_extent(
    kernel: ArenaKernel, chunk: int, geo: Geometry, capacity: int
) -> str:
    """Which of the three extents carries a kernel's excess over capacity.

    Args:
        kernel: The kernel.
        chunk: ``L``.
        geo: The geometry.
        capacity: Carveout in bytes.

    Returns:
        The extent name, or the empty string when the kernel fits.
    """
    if kernel.nbytes(chunk, geo.rows, geo.dim) <= capacity:
        return ""
    if kernel.nbytes(chunk, HEAD_MULTIPLE, STATE_MULTIPLE) > capacity:
        return "chunk"
    if kernel.nbytes(chunk, geo.rows, STATE_MULTIPLE) <= capacity:
        return "dim"
    if kernel.nbytes(chunk, HEAD_MULTIPLE, geo.dim) <= capacity:
        return "rows"
    return "rows+dim"


def arena_rows(
    geo: Geometry, chunks: Sequence[int], capacity: int
) -> tuple[ArenaRow, ...]:
    """The legality map.

    Args:
        geo: The geometry.
        chunks: Chunk lengths to ask about.
        capacity: Carveout in bytes.

    Returns:
        One row per kernel per chunk length, grouped by chunk length.
    """
    kernels = arena_kernels(geo.fold)
    rows: list[ArenaRow] = []
    for chunk in chunks:
        for kernel in kernels:
            nbytes = kernel.nbytes(chunk, geo.rows, geo.dim)
            rows.append(
                ArenaRow(
                    kernel=kernel.name,
                    chunk=chunk,
                    knob=kernel.knob(chunk, geo.rows, geo.dim),
                    smem_bytes=nbytes,
                    capacity_pct=100.0 * nbytes / capacity,
                    resident=capacity // nbytes,
                    floor_bytes=kernel.nbytes(chunk, HEAD_MULTIPLE, STATE_MULTIPLE),
                    refused_by=refusing_extent(kernel, chunk, geo, capacity),
                )
            )
    return tuple(rows)


def legal_chunks(rows: Sequence[ArenaRow]) -> tuple[int, ...]:
    """Chunk lengths every kernel fits and the config admits.

    Args:
        rows: The legality map.

    Returns:
        The admitted lengths, ascending.
    """
    refused = {row.chunk for row in rows if row.refused_by}
    return tuple(
        sorted(
            chunk
            for chunk in {row.chunk for row in rows}
            if chunk not in refused and MIN_CHUNK <= chunk <= MAX_CHUNK
        )
    )


def resident_chunks(rows: Sequence[ArenaRow], target: int) -> tuple[int, ...]:
    """Chunk lengths at which every kernel reaches ``target`` resident blocks.

    Args:
        rows: The legality map.
        target: Blocks required.

    Returns:
        The lengths, ascending.
    """
    short = {row.chunk for row in rows if row.resident < target}
    return tuple(sorted({row.chunk for row in rows} - short))


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------

FLAT = "flat"
INVERSE = "1/L"
SHIFTED = "(L+1)/L"
LINEAR = "L"
MIXED = "mixed"

Term = tuple[str, str, Callable[[int], int]]


class TrafficTerm(NamedTuple):
    """One tensor one launch touches once.

    Attributes:
        kernel: Kernel name, as the source spells it.
        tensor: Tensor name, as the kernel's own docstring spells it.
        side: ``read`` or ``write``.
        nbytes: Bytes at the chunk length asked for.
        scaling: What the term does under a doubling of ``L``, from the term itself.
    """

    kernel: str
    tensor: str
    side: str
    nbytes: int
    scaling: str


def _scaling(fn: Callable[[int], int], chunk: int) -> str:
    """Classify a term by evaluating it at ``chunk`` and at twice ``chunk``."""
    here, there = fn(chunk), fn(2 * chunk)
    if here == there:
        return FLAT
    if 2 * there == here:
        return INVERSE
    if there == 2 * here:
        return LINEAR
    if here > there:
        return SHIFTED
    return MIXED


def traffic_terms(geo: Geometry, chunk: int) -> tuple[TrafficTerm, ...]:
    """Every global-memory term of one whole step of one operator call.

    Seven launches: the forward's two, then the backward's five. No forward kernel
    appears twice; the backward reads the chunk start states the forward left.

    ``U`` and ``B`` are read over ``L + 1`` rows per chunk: the two-tap forcing at a
    chunk's first token reads the previous chunk's last row. ``C`` carries no tap and
    is read over ``L``.

    Args:
        geo: The geometry.
        chunk: ``L``.

    Returns:
        Every term, in launch order.
    """
    g = geo

    def rowwise(_: int) -> int:
        return g.bsz * g.heads * g.seqlen * g.rows * g.itemsize

    def rowwise_shifted(c: int) -> int:
        return rowwise(c) * (c + 1) // c

    def band(_: int) -> int:
        return g.bsz * g.groups * g.seqlen * g.dim * g.itemsize

    def band_shifted(c: int) -> int:
        return band(c) * (c + 1) // c

    def trans(_: int) -> int:
        return g.bsz * g.heads * g.seqlen * 4 * 4

    def taps(_: int) -> int:
        return g.bsz * g.heads * g.seqlen * 2 * 4 * 4

    def buffer(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * g.rows * g.dim * 4

    # A (B,H,C,P,3N) state a recurrence stored rather than reads: the store narrows
    # to the width its GEMM consumers stage it at. Only zstart is one on this arm.
    # The shipped backward's fused launch writes dinc at this width too; the unfused
    # chunk-start pair modelled below overwrites dzstart in place, so both are
    # float32 there.
    def stored(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * g.rows * g.dim * g.itemsize

    def cquat(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * 4 * 4

    def cscale(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * 4

    def state(_: int) -> int:
        return g.bsz * g.heads * g.rows * g.dim * 4

    def carry_u(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * g.rows * 4

    def carry_b(c: int) -> int:
        return g.bsz * g.groups * g.chunks(c) * g.dim * 4

    def dlogp(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * c * 4

    def dchunk_rot(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * 9 * 4

    def edge_row(c: int) -> int:
        return g.bsz * g.heads * g.chunks(c) * g.rows * g.itemsize

    def edge_band(c: int) -> int:
        return g.bsz * g.groups * g.chunks(c) * g.dim * g.itemsize

    # The fused prologue's increment never reaches memory, so ``inc`` appears on
    # neither side. Every band of one head rereads ``U``, ``trans``, and ``K``, and
    # that duplication is not counted: these are compulsory bytes, and the bands of
    # one head are co-resident by the launch order, so the rereads are L2 traffic
    # rather than DRAM traffic. Which they are on the part is a measurement, and
    # ``scripts/perf/profile_increment_passing_fwd.py`` is where it is taken.
    prologue: tuple[Term, ...] = (
        ("U", "read", rowwise_shifted),
        ("trans", "read", trans),
        ("K", "read", taps),
        ("B", "read", band_shifted),
        ("zstart", "write", stored),
        ("state", "write", state),
        ("cquat", "write", cquat),
        ("cscale", "write", cscale),
    )
    launches: tuple[tuple[str, tuple[Term, ...]], ...] = (
        ("increment_passing_fwd", prologue),
        (
            "chunk_scan_fwd",
            (
                ("U", "read", rowwise_shifted),
                ("trans", "read", trans),
                ("K", "read", taps),
                ("B", "read", band_shifted),
                ("C", "read", band),
                ("zstart", "read", stored),
                ("y", "write", rowwise),
            ),
        ),
        (
            "chunk_start_bwd",
            (
                ("dy", "read", rowwise),
                ("trans", "read", trans),
                ("C", "read", band),
                ("dzstart", "write", buffer),
            ),
        ),
        (
            "state_passing_bwd",
            (
                ("dzstart", "read", buffer),
                ("cquat", "read", cquat),
                ("cscale", "read", cscale),
                ("dinc", "write", buffer),
            ),
        ),
        (
            "chunk_input_bwd",
            (
                ("dy", "read", rowwise),
                ("U", "read", rowwise_shifted),
                ("trans", "read", trans),
                ("K", "read", taps),
                ("B", "read", band_shifted),
                ("B_tap", "read", band),
                ("C", "read", band),
                ("dinc", "read", buffer),
                ("zstart", "read", stored),
                ("dU", "write", rowwise),
                ("carry_u", "write", carry_u),
                ("dlogp", "write", dlogp),
                ("dchunk_rot", "write", dchunk_rot),
                ("dchunk_scale", "write", cscale),
            ),
        ),
        (
            "chunk_vector_bwd",
            (
                ("dy", "read", rowwise),
                ("U", "read", rowwise_shifted),
                ("B", "read", band_shifted),
                ("C", "read", band),
                ("trans", "read", trans),
                ("K", "read", taps),
                ("dinc", "read", buffer),
                ("zstart", "read", stored),
                ("dlogp", "read", dlogp),
                ("dchunk_rot", "read", dchunk_rot),
                ("dchunk_scale", "read", cscale),
                ("dB", "write", band),
                ("dC", "write", band),
                ("dtrans", "write", trans),
                ("dK", "write", taps),
                ("carry_b", "write", carry_b),
            ),
        ),
        (
            "boundary_bwd",
            (
                ("carry_u", "read", carry_u),
                ("carry_b", "read", carry_b),
                ("dU_edge", "read", edge_row),
                ("dB_edge", "read", edge_band),
                ("dU_edge", "write", edge_row),
                ("dB_edge", "write", edge_band),
            ),
        ),
    )
    return tuple(
        TrafficTerm(
            kernel=kernel,
            tensor=tensor,
            side=side,
            nbytes=fn(chunk),
            scaling=_scaling(fn, chunk),
        )
        for kernel, terms in launches
        for tensor, side, fn in terms
    )


class FlopTerm(NamedTuple):
    """One kernel's arithmetic, per token per head, affine in ``L``.

    Attributes:
        kernel: Kernel the arithmetic runs in.
        form: The GEMM form, as ``docs/kernels.md`` names it.
        flat: The part independent of ``L``, in flop per token per head.
        linear: The coefficient of ``L``, in flop per token per head per unit ``L``.
    """

    kernel: str
    form: str
    flat: int
    linear: int

    def flops(self, chunk: int) -> int:
        """Flop per token per head at one chunk length."""
        return self.flat + self.linear * chunk


def flop_terms(geo: Geometry) -> tuple[FlopTerm, ...]:
    """The arithmetic of one whole step, per token per head.

    Only the score and the diagonal carry ``L``: both contract over the chunk, so
    each token pays ``L``. The increment, the offset and the state terms contract
    over ``P`` or ``3N`` and are flat.

    The coefficients reproduce the per-launch figures the kernels' own docstrings
    state at ``standard``: 906 MFLOP for the increment, 2.87 GFLOP for the scan, 604
    MFLOP for the start, 3.54 MFLOP per block for the input backward and 4.03 MFLOP
    per block for the vector backward.

    Args:
        geo: The geometry, for ``P`` and ``3N``.

    Returns:
        One term per kernel, in launch order.
    """
    from slinoss.ops.so3ssd.cute.mma import mma_rows

    p, d = geo.rows, geo.dim
    return (
        FlopTerm("increment_passing_fwd", "increment", 4 * p * d, 0),
        FlopTerm("chunk_scan_fwd", "offset+score+diagonal", 2 * p * d, 4 * (p + d)),
        FlopTerm("chunk_start_bwd", "offset transpose", 2 * d * mma_rows(p), 0),
        FlopTerm("chunk_input_bwd", "all forms", 8 * p * d, 6 * (p + d)),
        FlopTerm("chunk_vector_bwd", "all forms", 6 * p * d, 8 * (p + d)),
    )


class StepModel(NamedTuple):
    """One chunk length's predicted step cost. Arithmetic, not a measurement.

    Attributes:
        chunk: ``L``.
        read_bytes: Compulsory read traffic of one step of one operator call.
        write_bytes: Compulsory write traffic of the same step.
        flop: Arithmetic of the same step.
        intensity: ``flop`` over total bytes.
        dram_us: Bytes over the supplied bandwidth, or None when none was supplied.
        tensor_us: Flop over the supplied peak, or None.
    """

    chunk: int
    read_bytes: int
    write_bytes: int
    flop: int
    intensity: float
    dram_us: float | None
    tensor_us: float | None

    @property
    def total_bytes(self) -> int:
        """Read plus write."""
        return self.read_bytes + self.write_bytes

    @property
    def model_us(self) -> float | None:
        """The larger of the two floors, or None when either is missing."""
        if self.dram_us is None or self.tensor_us is None:
            return None
        return max(self.dram_us, self.tensor_us)


def step_model(
    geo: Geometry, chunk: int, *, dram_gbs: float | None, peak_tflops: float | None
) -> StepModel:
    """Predict one chunk length's step cost from the two floors.

    Args:
        geo: The geometry.
        chunk: ``L``.
        dram_gbs: Measured achievable bandwidth, or None to omit the byte floor.
        peak_tflops: Measured tensor peak, or None to omit the arithmetic floor.

    Returns:
        The prediction.
    """
    terms = traffic_terms(geo, chunk)
    read = sum(t.nbytes for t in terms if t.side == "read")
    write = sum(t.nbytes for t in terms if t.side == "write")
    lanes = geo.bsz * geo.heads * geo.seqlen
    flop = lanes * sum(t.flops(chunk) for t in flop_terms(geo))
    return StepModel(
        chunk=chunk,
        read_bytes=read,
        write_bytes=write,
        flop=flop,
        intensity=flop / (read + write),
        dram_us=None if dram_gbs is None else (read + write) / (1e3 * dram_gbs),
        tensor_us=None if peak_tflops is None else flop / (1e6 * peak_tflops),
    )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

ARMS: tuple[tuple[str, torch.dtype, str], ...] = (
    ("ref/f32", torch.float32, "reference"),
    ("ref/bf16", torch.bfloat16, "reference"),
    ("cute/bf16", torch.bfloat16, "cute"),
)
"""Arm label, operand dtype, backend.

``ref/f32`` isolates the prefix and the accumulation from the operand width.
``ref/bf16`` adds the operand width every kernel inherits. ``cute/bf16`` is the
kernel path, and only reaches a chunk length the arena admits.
"""


class NumericsRow(NamedTuple):
    """One arm at one chunk length, against the float64 oracle.

    Attributes:
        chunk: ``L``.
        arm: What ran, or the refusal.
        y_rel: Largest absolute error on ``y`` over the oracle's largest magnitude,
            so a near-zero element cannot inflate it.
        state_rel: The same on the closing state.
        decay_min: Smallest chunk-local decay ``exp(2*lp)`` any chunk reaches. I1
            holds this at or below one and it falls as ``L`` grows.
        prefix_drift: Largest departure from unit norm of a chunk-local quaternion
            prefix, before I5 renormalizes. What one renormalization must absorb.
    """

    chunk: int
    arm: str
    y_rel: float
    state_rel: float
    decay_min: float
    prefix_drift: float


def _probe_shape(geo: Geometry, chunk: int, name: str) -> OpShape:
    """An :class:`slinoss.perf.workload.OpShape` for one draw at this geometry."""
    return OpShape(
        name=name,
        bsz=geo.bsz,
        heads=geo.heads,
        seq=geo.seqlen,
        rows=geo.rows,
        lanes=geo.dim // 3,
        chunk=chunk,
        groups=geo.groups,
    )


def _oracle_inputs(
    geo: Geometry, device: torch.device, *, seed: int
) -> tuple[torch.Tensor, ...]:
    """One draw at :data:`WIDE`, from which every arm is cast.

    Args:
        geo: The probe geometry.
        device: Where to allocate.
        seed: Generator seed.

    Returns:
        ``(U, trans, K, B, C)`` at :data:`WIDE`, none requiring gradients.
    """
    from slinoss.perf.workload import make_inputs

    shape = _probe_shape(geo, MIN_CHUNK, "probe")
    got = make_inputs(shape, device, dtype=WIDE, requires_grad=False, seed=seed)
    return (got.U, got.trans.to(WIDE), got.K.to(WIDE), got.B, got.C)


def prefix_probe(trans: torch.Tensor, chunk: int) -> tuple[float, float]:
    """The two invariant readings a longer prefix moves.

    The drift is the unnormalized Hillis-Steele product's departure from unit norm
    at float32, the pinned width.
    :func:`slinoss.ops.so3ssd.reference.quat_prefix_scan` renormalizes at the end, so
    reading its output would report I5's effect and not its input.

    Args:
        trans: ``(B,H,T,4)`` at :data:`WIDE`, packing ``(w_x, w_y, w_z, ls)``.
        chunk: ``L``. A ragged tail is dropped: a partial chunk has a shorter prefix
            and cannot set the maximum.

    Returns:
        The smallest chunk-local decay and the largest pre-renormalization drift.
    """
    from slinoss.ops.so3ssd.reference import quat_exp, quat_mul

    kept = trans.shape[2] // chunk * chunk
    folded = trans[:, :, :kept].unflatten(2, (-1, chunk))
    decay = torch.exp(2.0 * folded[..., 3].cumsum(-1)).min().item()
    out = quat_exp(0.5 * folded[..., :3].to(torch.float32))
    step = 1
    while step < chunk:
        out = torch.cat(
            [out[..., :step, :], quat_mul(out[..., step:, :], out[..., :-step, :])],
            dim=-2,
        )
        step *= 2
    drift = (out.norm(dim=-1) - 1.0).abs().max().item()
    return float(decay), float(drift)


def numerics_rows(
    geo: Geometry, chunks: Sequence[int], device: torch.device, *, seed: int
) -> tuple[NumericsRow, ...]:
    """Error against the float64 oracle at each chunk length.

    A chunk length an arm refuses yields a row naming the refusal, never a figure.

    Args:
        geo: The probe geometry.
        chunks: Chunk lengths to ask about.
        device: Where to run.
        seed: Generator seed.

    Returns:
        One row per arm per chunk length.
    """
    wide = _oracle_inputs(geo, device, seed=seed)
    rows: list[NumericsRow] = []
    for chunk in chunks:
        decay_min, drift = prefix_probe(wide[1], chunk)
        oracle = _run_arm(wide, chunk, "reference")
        for arm, dtype, backend in ARMS:
            try:
                got = _run_arm(_cast_arm(wide, dtype), chunk, backend)
            except (ValueError, TypeError, RuntimeError) as exc:
                text = str(exc).splitlines()[0][:36]
                rows.append(
                    NumericsRow(chunk, f"{arm} refused: {text}", 0.0, 0.0, 0.0, 0.0)
                )
                continue
            rows.append(
                NumericsRow(
                    chunk=chunk,
                    arm=arm,
                    y_rel=_rel(got.y, oracle.y),
                    state_rel=_rel(got.state, oracle.state),
                    decay_min=decay_min,
                    prefix_drift=drift,
                )
            )
    return tuple(rows)


def _cast_arm(
    wide: Sequence[torch.Tensor], dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    """Cast one oracle draw to an arm's widths.

    ``trans`` and ``K`` land at float32 whatever the operand width. I4 pins them, so
    a low-precision transition is a defect and not an arm.

    Args:
        wide: ``(U, trans, K, B, C)`` at :data:`WIDE`.
        dtype: Operand dtype for ``U``, ``B`` and ``C``.

    Returns:
        The arm's five tensors.
    """
    U, trans, K, B, C = wide
    return (
        U.to(dtype),
        trans.to(torch.float32),
        K.to(torch.float32),
        B.to(dtype),
        C.to(dtype),
    )


def _run_arm(args: Sequence[torch.Tensor], chunk: int, backend: str) -> SO3SSDResult:
    """One arm's forward.

    Args:
        args: The arm's five tensors, ``(U, trans, K, B, C)``.
        chunk: ``L``.
        backend: ``reference`` or ``cute``.

    Returns:
        The result.
    """
    from slinoss.ops.so3ssd import so3ssd

    U, trans, K, B, C = args
    return so3ssd(U, trans, K, B, C, chunk, backend=backend)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest absolute error over the largest magnitude of the reference."""
    scale = want.abs().max().to(WIDE).clamp_min(torch.finfo(torch.float32).tiny)
    return float(((got.to(WIDE) - want.to(WIDE)).abs().max() / scale).item())


def parity(geo: Geometry, chunk: int, device: torch.device, *, seed: int) -> float:
    """Largest relative departure of the CuTe forward from the reference.

    Both arms run at bfloat16 on the same draw, so this is the kernel path against
    the shape it implements and not a width study. The guard on the step mode: a
    chunk length that times faster and answers differently is not a result.

    ``P``, ``3N`` and ``G`` are what select a kernel's tiling and its arena, and the
    caller narrows ``B`` and ``T``: the reference is a torch scan and does not fit
    the measured batch at the measured length.

    Args:
        geo: The geometry to check at. Carries the measured ``P``, ``3N`` and ``G``.
        chunk: ``L``.
        device: Where to run.
        seed: Generator seed.

    Returns:
        The larger of the ``y`` and closing-state relative errors.
    """
    from slinoss.perf.workload import make_inputs

    got = make_inputs(
        _probe_shape(geo, chunk, "parity"),
        device,
        dtype=torch.bfloat16,
        requires_grad=False,
        seed=seed,
    )
    args = (got.U, got.trans, got.K, got.B, got.C)
    fast = _run_arm(args, chunk, "cute")
    slow = _run_arm(args, chunk, "reference")
    return max(_rel(fast.y, slow.y), _rel(fast.state, slow.state))


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class StepRow(NamedTuple):
    """One measured step at one chunk length.

    Attributes:
        chunk: ``L``.
        median_us: Median over the timed launches.
        min_us: Fastest launch.
        max_us: Slowest launch.
        spread_pct: Range over the median.
        tokens_per_s: ``B*T`` over the median.
        cute_us: Device time in this package's own kernels, per step.
        parity_rel: Largest relative departure of the CuTe forward from the
            reference at this chunk length.
        foreign_processes: Compute processes on the device other than this one, at
            probe time. Anything above zero makes the row contaminated and not a
            measurement of this geometry alone.
    """

    chunk: int
    median_us: float
    min_us: float
    max_us: float
    spread_pct: float
    tokens_per_s: float
    cute_us: float
    parity_rel: float
    foreign_processes: int


def step_rows(
    args: argparse.Namespace, geo: Geometry, chunks: Sequence[int], device: torch.device
) -> tuple[tuple[StepRow, ...], dict[int, tuple[tuple[str, float, float], ...]]]:
    """Measure the whole step at each chunk length, with the per-kernel table.

    The parity check runs before the timing at each chunk length, so a wrong answer
    is reported even when the timing is contaminated.

    Args:
        args: The command line. Its geometry flags feed
            :func:`scripts.perf.attribute_step.build_config`.
        geo: The operator geometry the parity check runs at.
        chunks: Chunk lengths to measure.
        device: Where to measure.

    Returns:
        One row per chunk length, and the per-kernel rows keyed by chunk length.
    """
    from torch.profiler import ProfilerActivity, profile

    from scripts.perf.attribute_step import build_config, build_step, device_rows
    from slinoss.perf.device import contention, device_ordinal
    from slinoss.perf.timing import measure

    index = max(device_ordinal(device), 0)
    checked = Geometry(
        bsz=args.parity_batch,
        heads=geo.heads,
        groups=geo.groups,
        seqlen=args.parity_seq,
        rows=geo.rows,
        dim=geo.dim,
    )
    rows: list[StepRow] = []
    kernels: dict[int, tuple[tuple[str, float, float], ...]] = {}
    for chunk in chunks:
        rel = parity(checked, chunk, device, seed=args.seed)
        inner = argparse.Namespace(**vars(args))
        inner.mode = "step"
        inner.chunk = chunk
        step = build_step(inner, build_config(inner), device)
        timed = measure(
            step,
            label=f"step/L{chunk}",
            iters=args.iters,
            warmup=args.warmup,
            device=device,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiled:
            for _ in range(args.profile_iters):
                step()
            torch.cuda.synchronize(device)
        table = tuple(device_rows(profiled, args.profile_iters))
        kernels[chunk] = table
        probe = contention(index)
        total = timed.total
        rows.append(
            StepRow(
                chunk=chunk,
                median_us=float(total.median_duration_us),
                min_us=float(total.min_duration_us),
                max_us=float(total.max_duration_us),
                spread_pct=float(total.spread_pct),
                tokens_per_s=1e6
                * args.batch
                * args.seqlen
                / float(total.median_duration_us),
                cute_us=sum(us for name, us, _ in table if "kernel_cutlass_" in name),
                parity_rel=rel,
                foreign_processes=int(probe.foreign_process_count),
            )
        )
        del step
        torch.cuda.empty_cache()
    return tuple(rows), kernels


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="arena")
    parser.add_argument("--shape", default="acceptance", help="Geometry held fixed.")
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=list(CANDIDATES),
        help="Chunk lengths to ask about. Powers of two.",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=0,
        help="Carveout in bytes. Queried from the DSL when zero.",
    )
    parser.add_argument(
        "--dram-gbs",
        type=float,
        default=None,
        help="Measured achievable bandwidth, for the model's byte floor.",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="Measured tensor peak, for the model's arithmetic floor.",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--prefill", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--d-state", type=int, default=240)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--rows", type=int, default=12, help="Kernels listed.")
    parser.add_argument("--probe-seq", type=int, default=1024)
    parser.add_argument("--probe-heads", type=int, default=2)
    parser.add_argument(
        "--parity-batch",
        type=int,
        default=1,
        help="Batch of the step mode's parity draw. The reference is a torch scan.",
    )
    parser.add_argument("--parity-seq", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def resolve_device(spec: str) -> torch.device:
    """Resolve a device and make it current.

    The DSL exports a tensor through DLPack and refuses an export whose ordinal is
    not the current device, so an explicit ``cuda:N`` has to move the context and not
    only the allocations.

    Args:
        spec: Device string, ``cuda`` or ``cuda:N``.

    Returns:
        The device.
    """
    from slinoss.perf.device import device_ordinal, require_cuda

    device = require_cuda(spec)
    index = device_ordinal(device)
    if index >= 0:
        torch.cuda.set_device(index)
    return device


def resolve_capacity(requested: int) -> int:
    """The carveout the legality map is judged against.

    Args:
        requested: Bytes from the command line, or zero to query the DSL.

    Returns:
        Capacity in bytes.
    """
    if requested > 0:
        return requested
    from slinoss._cute import smem_capacity

    return smem_capacity()


def print_arena(geo: Geometry, args: argparse.Namespace) -> None:
    """Print the legality map."""
    capacity = resolve_capacity(args.capacity)
    rows = arena_rows(geo, args.chunks, capacity)
    print(f"computed  geometry {geo.describe()}")
    print(
        f"carveout {capacity:,} B   {RESIDENT_TARGET} resident blocks need "
        f"{capacity // RESIDENT_TARGET:,} B   config admits L in "
        f"[{MIN_CHUNK}, {MAX_CHUNK}]"
    )
    print()
    print(
        f"{'kernel':26s} {'L':>5s} {'knob':>10s} {'bytes':>10s} {'cap_pct':>8s} "
        f"{'resident':>9s} {'floor_bytes':>12s}  refused_by"
    )
    for row in rows:
        admitted = MIN_CHUNK <= row.chunk <= MAX_CHUNK
        verdict = row.refused_by or ("" if admitted else "config")
        print(
            f"{row.kernel:26s} {row.chunk:5d} {row.knob:>10s} "
            f"{row.smem_bytes:10,d} {row.capacity_pct:8.2f} {row.resident:9d} "
            f"{row.floor_bytes:12,d}  {verdict}"
        )
    print()
    print(f"legal L: {list(legal_chunks(rows))}")
    print(
        f"L at {RESIDENT_TARGET} resident blocks everywhere: "
        f"{list(resident_chunks(rows, RESIDENT_TARGET))}"
    )


def print_traffic(geo: Geometry, args: argparse.Namespace) -> None:
    """Print the analytic byte and arithmetic model."""
    base = args.chunks[len(args.chunks) // 2]
    terms = traffic_terms(geo, base)
    print(f"computed  geometry {geo.describe()}")
    print()
    print(f"per-term traffic at L={base}, one step of one operator call")
    print(f"{'kernel':26s} {'tensor':12s} {'side':6s} {'MB':>10s}  scaling")
    for term in terms:
        print(
            f"{term.kernel:26s} {term.tensor:12s} {term.side:6s} "
            f"{term.nbytes / 1e6:10.2f}  {term.scaling}"
        )
    print()
    print(f"per-kernel totals at L={base}")
    print(f"{'kernel':26s} {'read_MB':>10s} {'write_MB':>10s} {'total_MB':>10s}")
    for kernel in dict.fromkeys(term.kernel for term in terms):
        read = sum(t.nbytes for t in terms if t.kernel == kernel and t.side == "read")
        write = sum(t.nbytes for t in terms if t.kernel == kernel and t.side == "write")
        print(
            f"{kernel:26s} {read / 1e6:10.2f} {write / 1e6:10.2f} "
            f"{(read + write) / 1e6:10.2f}"
        )
    print()
    print("arithmetic per token per head, affine in L")
    print(f"{'kernel':26s} {'form':22s} {'flat':>10s} {'per_L':>10s}")
    for flop in flop_terms(geo):
        print(f"{flop.kernel:26s} {flop.form:22s} {flop.flat:10,d} {flop.linear:10,d}")
    print()
    print("step model over L")
    header = (
        f"{'L':>5s} {'read_MB':>10s} {'write_MB':>10s} {'total_MB':>10s} "
        f"{'GFLOP':>10s} {'flop/byte':>10s}"
    )
    if args.dram_gbs is not None:
        header += f" {'dram_us':>10s}"
    if args.peak_tflops is not None:
        header += f" {'tensor_us':>10s}"
    if args.dram_gbs is not None and args.peak_tflops is not None:
        header += f" {'model_us':>10s}"
    print(header)
    models = [
        step_model(geo, chunk, dram_gbs=args.dram_gbs, peak_tflops=args.peak_tflops)
        for chunk in args.chunks
    ]
    for model in models:
        line = (
            f"{model.chunk:5d} {model.read_bytes / 1e6:10.2f} "
            f"{model.write_bytes / 1e6:10.2f} {model.total_bytes / 1e6:10.2f} "
            f"{model.flop / 1e9:10.2f} {model.intensity:10.2f}"
        )
        for value in (model.dram_us, model.tensor_us, model.model_us):
            if value is not None:
                line += f" {value:10.1f}"
        print(line)
    print()
    print(f"model byte minimum at L={min(models, key=lambda m: m.total_bytes).chunk}")
    timed = [m for m in models if m.model_us is not None]
    if timed:
        best = min(timed, key=lambda m: m.model_us or 0.0)
        print(f"model time minimum at L={best.chunk}, from a supplied denominator")


def print_numerics(geo: Geometry, args: argparse.Namespace) -> None:
    """Print the invariant probe and the error against the float64 oracle."""
    device = resolve_device(args.device)
    probe = Geometry(
        bsz=1,
        heads=args.probe_heads,
        groups=1,
        seqlen=args.probe_seq,
        rows=HEAD_MULTIPLE,
        dim=STATE_MULTIPLE,
    )
    print(f"measured  probe {probe.describe()}  oracle float64 reference")
    print(f"the geometry the other modes hold fixed: {geo.describe()}")
    print()
    print(
        f"{'L':>5s} {'arm':46s} {'y_rel':>12s} {'state_rel':>12s} "
        f"{'decay_min':>12s} {'prefix_drift':>13s}"
    )
    for row in numerics_rows(probe, args.chunks, device, seed=args.seed):
        print(
            f"{row.chunk:5d} {row.arm:46s} {row.y_rel:12.3e} {row.state_rel:12.3e} "
            f"{row.decay_min:12.3e} {row.prefix_drift:13.3e}"
        )


def print_step(geo: Geometry, args: argparse.Namespace) -> None:
    """Print the measured step at each legal chunk length."""
    from slinoss.perf.device import device_info, device_ordinal

    device = resolve_device(args.device)
    capacity = resolve_capacity(args.capacity)
    admitted = legal_chunks(arena_rows(geo, args.chunks, capacity))
    asked = [c for c in args.chunks if c in admitted]
    refused = [c for c in args.chunks if c not in admitted]
    info = device_info(max(device_ordinal(device), 0))
    print(f"measured  {info.name}  clocks {info.clocks.detail}")
    print(f"geometry {geo.describe()}  layers {args.layers}")
    if refused:
        print(f"not measured, refused at this geometry: {refused}")
    print()
    rows, kernels = step_rows(args, geo, asked, device)
    print(
        f"{'L':>5s} {'median_us':>12s} {'min_us':>12s} {'max_us':>12s} "
        f"{'spread_pct':>11s} {'tokens_per_s':>13s} {'cute_us':>12s} "
        f"{'parity_rel':>12s} {'foreign':>8s}"
    )
    for row in rows:
        print(
            f"{row.chunk:5d} {row.median_us:12,.1f} {row.min_us:12,.1f} "
            f"{row.max_us:12,.1f} {row.spread_pct:11.2f} {row.tokens_per_s:13,.0f} "
            f"{row.cute_us:12,.1f} {row.parity_rel:12.3e} "
            f"{row.foreign_processes:8d}"
        )
    for chunk in asked:
        print()
        print(f"per-kernel at L={chunk}")
        print(f"{'kernel':64s} {'us/iter':>12s} {'calls':>8s}")
        for name, us, calls in kernels[chunk][: args.rows]:
            print(f"{name[:64]:64s} {us:12,.1f} {calls:8,.1f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one mode.

    Args:
        argv: Command line, or None for ``sys.argv``.

    Returns:
        Process exit status.

    Raises:
        ValueError: If a requested chunk length is not a positive power of two. No
            layout function and no config accepts one.
    """
    args = parse_args(argv)
    for chunk in args.chunks:
        if chunk <= 0 or chunk & (chunk - 1):
            raise ValueError(f"chunk lengths must be powers of two, got {chunk}")
    geo = geometry_of(shape_by_name(args.shape))
    if args.mode == "arena":
        print_arena(geo, args)
    elif args.mode == "traffic":
        print_traffic(geo, args)
    elif args.mode == "numerics":
        print_numerics(geo, args)
    else:
        print_step(geo, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
