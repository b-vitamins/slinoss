"""Every dense GEMM in a training step, by role, shape, layout and achieved rate.

`attribute_step.py` says how much of a step the GEMM class costs. It cannot say
which GEMM: cuBLAS names a kernel by tile and transpose case, never by call site,
one kernel serves several call sites, and one call site changes kernel with its
shape. This driver names them.

The roles are the linear maps of the stack: the mixer's input and output
projection, the feed-forward's gate, up and out, and the language-model head.
Each contributes three GEMMs per step, and the three carry different operand
layouts:

    fwd    y(T,O)  = x(T,I)    @ W(O,I)^T   both operands k-major   tn
    dgrad  dx(T,I) = dy(T,O)   @ W(O,I)     W is n-major            nn
    wgrad  dW(O,I) = dy(T,O)^T @ x(T,I)     neither is k-major      nt

A map that stores its weight `(I,O)` instead swaps the first two cases and
transposes the third, so the stored orientation is part of the role table.

Recorded operand shapes do not separate these: `x @ W^T` and `dy @ W` reach the
profiler as one pair of extents, and only the transpose case tells them apart. So
the case is read off the kernel name, which is cuBLAS's own account of what it
ran, and a launch whose extents match a role but whose kernel names no case is
reported unassigned rather than absorbed.

`census` attributes a profiled training step: one row per distinct GEMM, its
per-launch median over every launch in the profile, and its per-step total.
`isolated` times the same extents and layouts alone, which is the harness a
change is judged in. `alternatives` runs the paired experiments a diagnosis rests
on: the head against an aligned vocabulary, three ways; every weight gradient
against its transposed orientation, and every map's three stages against the
weight stored the other way; and the input projection's width against the next
tile multiple.

    python3 scripts/perf/gemm_census.py --mode census
    python3 scripts/perf/gemm_census.py --mode isolated
    python3 scripts/perf/gemm_census.py --mode alternatives

Every mode measures the tensor-core ceiling first, in the same process, on the
same device, and divides by it. That ceiling is a square GEMM at its own
footprint: a row can exceed it, since a shape with a smaller working set holds
clocks better than an 8192-cube does, and a row above 100% is a statement about
the denominator rather than about the hardware. Clocks cannot be locked on this
fleet, and both the clock policy and what else was resident are stamped.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, cast

import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from scripts.perf.attribute_step import DTYPE, build_config, build_step
from slinoss.config import SLinOSSConfig
from slinoss.mixer import ProjectionLayout
from slinoss.perf.ceiling import (
    CLASS_FLOOR_PCT,
    TENSOR_BOUND,
    TensorCeiling,
    tensor_ceiling,
)
from slinoss.perf.device import clock_policy, contention, device_ordinal, require_cuda
from slinoss.perf.timing import measure, measure_paired
from slinoss.perf.units import (
    INVARIANT,
    MEDIAN,
    SUM,
    Count,
    Microseconds,
    Percent,
    PerfRecord,
    Spread,
    TFlopsPerSecond,
    pct_of,
    tflops_from_flop_us,
)

MODES = ("census", "isolated", "alternatives")

FWD = "fwd"
DGRAD = "dgrad"
WGRAD = "wgrad"

TN = "tn"
NN = "nn"
NT = "nt"
LAYOUTS = (TN, NN, NT)
"""cuBLAS's transpose cases, in the order the three stages produce them."""

MATMUL_OPS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm")
"""Aten operators whose device work is a dense GEMM.

`aten::linear` and `aten::matmul` are dispatchers: they appear in a trace as
parents of one of these and launch nothing themselves.
"""

VECTOR_COUNT = 8
"""Elements a sixteen-byte tensor-core operand load covers at bf16.

cuBLAS reads its kernel's alignment off the extents, so an extent that is not a
multiple of this drops the whole GEMM onto an `align1` kernel built for a
narrower load.
"""

TILE_COUNT = 128
"""Output columns the widest bf16 Ampere GEMM tile covers.

The unit an extent is rounded to when an experiment pays for a whole tile rather
than for alignment alone.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="census")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--d-state", type=int, default=240)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument(
        "--iters",
        type=int,
        default=6,
        help="Profiled steps, in census mode. Every layer's launches are samples "
        "of their row, so a handful of steps resolves every row except the "
        "head's, which is called once a step.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Timed launches per arm, in isolated and alternatives mode.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.set_defaults(prefill=64)
    return parser.parse_args(argv)


def step_args(args: argparse.Namespace) -> argparse.Namespace:
    """The namespace `attribute_step` builds its training step from.

    The census is of the step that driver attributes, so the geometry and the
    callable come from it rather than from a second definition here.

    Args:
        args: This driver's command line.

    Returns:
        A namespace carrying `attribute_step`'s fields.
    """
    return argparse.Namespace(
        mode="step",
        batch=args.batch,
        seqlen=args.seqlen,
        prefill=args.prefill,
        d_model=args.d_model,
        d_state=args.d_state,
        d_head=args.d_head,
        chunk=args.chunk,
        groups=args.groups,
        layers=args.layers,
        vocab=args.vocab,
        iters=args.iters,
        warmup=args.warmup,
        device=args.device,
    )


@dataclass(frozen=True)
class LinearMap:
    """One weight matrix of the stack, and how often a step applies it.

    Attributes:
        name: Call site, as the module tree names it.
        fan_in: Contraction extent of the weight, `I`.
        fan_out: Output extent of the weight, `O`.
        call_count: Applications per step.
        transposed: Whether the weight is stored `(I,O)` rather than `(O,I)`.
    """

    name: str
    fan_in: int
    fan_out: int
    call_count: int
    transposed: bool = False


def linear_maps(config: SLinOSSConfig) -> tuple[LinearMap, ...]:
    """Every weight matrix a training step multiplies by, from the configuration.

    The widths come from the configuration and from
    :class:`slinoss.mixer.ProjectionLayout`, which is where the projection's
    padded width is decided, so a change to either reaches this table without an
    edit here. A test holds the table against the shapes a built stack carries.

    Args:
        config: The geometry. `vocab_size` decides whether the head exists.

    Returns:
        One entry per weight, in module order.
    """
    layers = config.n_layers
    width = ProjectionLayout.from_config(config).width
    maps = [
        LinearMap("in_proj", config.d_model, width, layers),
        LinearMap("out_proj", config.d_inner, config.d_model, layers),
        LinearMap("ffn_gate", config.d_model, config.d_ffn, layers),
        LinearMap("ffn_up", config.d_model, config.d_ffn, layers),
        LinearMap("ffn_out", config.d_ffn, config.d_model, layers),
    ]
    padded_vocab = config.padded_vocab_size
    if padded_vocab is not None:
        maps.append(LinearMap("head", config.d_model, padded_vocab, 1))
    return tuple(maps)


def ragged_extents(m: int, n: int, k: int) -> tuple[str, ...]:
    """Extents that are not a whole number of :data:`VECTOR_COUNT` elements.

    Which of them costs the kernel its wide load depends on the transpose case,
    so all three are named and the kernel's own `align` suffix says which one
    bit.

    Args:
        m: Output rows.
        n: Output columns.
        k: Reduction extent.

    Returns:
        The names of the ragged extents, in `M`, `N`, `K` order.
    """
    return tuple(
        name
        for name, extent in (("M", m), ("N", n), ("K", k))
        if extent % VECTOR_COUNT != 0
    )


@dataclass(frozen=True)
class GemmShape:
    """One distinct GEMM of a step: its extents, its layout, and its call count.

    Attributes:
        label: Call sites this shape covers, joined by `+` when two share it.
        stage: `fwd`, `dgrad` or `wgrad`.
        m: Rows of the output.
        n: Columns of the output.
        k: Reduction extent.
        layout: One of :data:`LAYOUTS`.
        call_count: Launches per step.
    """

    label: str
    stage: str
    m: int
    n: int
    k: int
    layout: str
    call_count: int

    @property
    def key(self) -> tuple[int, int, int, str]:
        """What makes two GEMMs one call as far as cuBLAS is concerned."""
        return (self.m, self.n, self.k, self.layout)

    @property
    def merge_key(self) -> tuple[str, int, int, int, str]:
        """What makes two GEMMs one row.

        The stage joins the shape, so a row stays one stage of one call site even
        where cuBLAS runs the same kernel for another. Two rows over one key share
        the kernel and the per-launch median and split the step total by their own
        call counts.
        """
        return (self.stage, *self.key)

    @property
    def flop_count(self) -> Count:
        """Floating-point operations in one launch."""
        return Count(2 * self.m * self.n * self.k)


def gemm_shapes(config: SLinOSSConfig, tokens: int) -> tuple[GemmShape, ...]:
    """The three GEMMs of every linear map, merged where two share a shape.

    Two call sites of one shape, one layout and one stage are one cuBLAS call and
    one profiler row. The census reports them as one row and names both, since
    nothing measurable separates them. Two different stages on one shape stay two
    rows: the stage is what the caller is reading the table for.

    Args:
        config: The geometry.
        tokens: `B*T`, the batch-flattened token count.

    Returns:
        One entry per distinct GEMM and stage.
    """
    merged: dict[tuple[str, int, int, int, str], GemmShape] = {}
    for one in linear_maps(config):
        calls = one.call_count
        # The stored orientation decides which operand is k-major in the two
        # forward-shaped GEMMs and which way round the third comes out.
        fwd_layout, dgrad_layout = (NN, TN) if one.transposed else (TN, NN)
        rows, cols = (
            (one.fan_in, one.fan_out) if one.transposed else (one.fan_out, one.fan_in)
        )
        for shape in (
            GemmShape(
                one.name, FWD, tokens, one.fan_out, one.fan_in, fwd_layout, calls
            ),
            GemmShape(
                one.name, DGRAD, tokens, one.fan_in, one.fan_out, dgrad_layout, calls
            ),
            GemmShape(one.name, WGRAD, rows, cols, tokens, NT, calls),
        ):
            found = merged.get(shape.merge_key)
            merged[shape.merge_key] = (
                shape
                if found is None
                else GemmShape(
                    label=f"{found.label}+{shape.label}",
                    stage=found.stage,
                    m=found.m,
                    n=found.n,
                    k=found.k,
                    layout=found.layout,
                    call_count=found.call_count + shape.call_count,
                )
            )
    return tuple(merged.values())


@dataclass(frozen=True)
class GemmRow(PerfRecord):
    """One GEMM's measured position against the tensor-core ceiling.

    Attributes:
        label: Call sites the row covers.
        stage: `fwd`, `dgrad` or `wgrad`.
        layout: cuBLAS's transpose case.
        kernel: Kernel that ran, or empty when the shape was timed directly.
        ragged: Extents that are not a whole alignment vector.
        m_count: Rows of the output.
        n_count: Columns of the output.
        k_count: Reduction extent.
        call_count: Launches per step.
        flop_count: Floating-point operations in one launch.
        duration: Per-launch dispersion.
        step_duration_us: Every launch of this row in one step, at the median.
        achieved_tflops: One launch's flop over its median duration.
        ceiling_pct: That rate over the measured square-GEMM ceiling.
    """

    label: str
    stage: str
    layout: str
    kernel: str
    ragged: tuple[str, ...]
    m_count: Annotated[Count, INVARIANT]
    n_count: Annotated[Count, INVARIANT]
    k_count: Annotated[Count, INVARIANT]
    call_count: Annotated[Count, SUM]
    flop_count: Annotated[Count, INVARIANT]
    duration: Spread
    step_duration_us: Annotated[Microseconds, MEDIAN]
    achieved_tflops: Annotated[TFlopsPerSecond, MEDIAN]
    ceiling_pct: Annotated[Percent, MEDIAN]

    @classmethod
    def of(
        cls,
        shape: GemmShape,
        kernel: str,
        samples: Sequence[Microseconds],
        ceiling: TensorCeiling,
    ) -> GemmRow:
        """Score measured launches of one shape.

        Args:
            shape: What ran.
            kernel: Kernel name, or empty.
            samples: Per-launch durations.
            ceiling: The square-GEMM ceiling, measured on the same device in the
                same process.

        Returns:
            The row.
        """
        spread = Spread.of(samples)
        rate = tflops_from_flop_us(shape.flop_count, spread.median_duration_us)
        return cls(
            label=shape.label,
            stage=shape.stage,
            layout=shape.layout,
            kernel=kernel,
            ragged=ragged_extents(shape.m, shape.n, shape.k),
            m_count=Count(shape.m),
            n_count=Count(shape.n),
            k_count=Count(shape.k),
            call_count=Count(shape.call_count),
            flop_count=shape.flop_count,
            duration=spread,
            step_duration_us=Microseconds(shape.call_count * spread.median_duration_us),
            achieved_tflops=rate,
            ceiling_pct=pct_of(rate, ceiling.achieved_tflops),
        )


def _launched(event: object) -> tuple[tuple[str, float], ...]:
    """Device kernels one profiler event launched, with their durations in us."""
    kernels = cast("Sequence[Any]", getattr(event, "kernels", ()))
    return tuple((str(one.name), float(one.duration)) for one in kernels)


def layout_of(kernel: str) -> str:
    """cuBLAS's transpose case, from its kernel name.

    Every cuBLAS and CUTLASS tiled GEMM name carries the case as a standalone
    `tn`, `nn` or `nt` token. A name without one is not a tiled GEMM -- a gemv, a
    split-k reduction, a workspace memset -- and the caller keeps it out of the
    row it appeared under.

    Args:
        kernel: Kernel name as the profiler reports it.

    Returns:
        The case, or an empty string.
    """
    tokens = kernel.replace(">", "_").replace("<", "_").replace("(", "_").split("_")
    for one in LAYOUTS:
        if one in tokens:
            return one
    return ""


def extents_of(shapes: Sequence[Any]) -> tuple[int, int, int] | None:
    """`(m, n, k)` of a matmul operator, from its recorded input shapes.

    An `addmm` carries its bias first, so both forms end with the two matrices.
    Anything whose last two operands are not a conformable pair is not a plain
    GEMM.

    Args:
        shapes: Recorded input shapes.

    Returns:
        The extents, or None.
    """
    matrices = [tuple(int(dim) for dim in one) for one in shapes if len(one) == 2]
    if len(matrices) < 2:
        return None
    lhs, rhs = matrices[-2], matrices[-1]
    if lhs[1] != rhs[0]:
        return None
    return lhs[0], rhs[1], lhs[1]


@dataclass
class Launches:
    """Every launch the profile recorded under one key.

    Attributes:
        kernels: Distinct kernel names seen, in trace order. More than one means
            the key covers two kernels and its median mixes them.
        samples: Per-launch durations, in trace order.
    """

    kernels: list[str] = field(default_factory=list)
    samples: list[Microseconds] = field(default_factory=list)

    def add(self, kernel: str, duration: float) -> None:
        """Record one launch."""
        if kernel not in self.kernels:
            self.kernels.append(kernel)
        self.samples.append(Microseconds(duration))

    @property
    def name(self) -> str:
        """The kernel, or all of them joined, so a mixed key cannot read clean."""
        return "+".join(self.kernels)


class Profiled(Protocol):
    """What a census reads off a finished profile: its operator events.

    Named as a protocol rather than taken as :class:`torch.profiler.profile` so the
    reader can be exercised against a recorded trace without a device.
    """

    def events(self) -> Sequence[Any] | None:
        """Operator events, or nothing if the profile recorded none."""
        ...


def observed(profiled: Profiled) -> dict[tuple[int, int, int, str], Launches]:
    """Per-launch GEMM durations from a profile, keyed by extents and case.

    Every launch is kept rather than summed, so a row carries the dispersion of
    the launches behind it instead of a mean of unknown scatter. A kernel a
    matmul launched that is not a tiled GEMM is keyed under its own name, where
    it cannot hide inside a GEMM's duration.

    Args:
        profiled: A finished profile taken with `record_shapes=True`.

    Returns:
        Key to launches. A key whose extents are zero is a non-GEMM kernel, or a
        GEMM whose operand shapes were not recorded.
    """
    out: dict[tuple[int, int, int, str], Launches] = {}
    for event in profiled.events() or ():
        if str(getattr(event, "name", "")) not in MATMUL_OPS:
            continue
        kernels = _launched(event)
        if not kernels:
            continue
        extents = extents_of(cast("Sequence[Any]", getattr(event, "input_shapes", ())))
        for kernel, duration in kernels:
            layout = layout_of(kernel)
            key = (
                (*extents, layout)
                if extents is not None and layout
                else (0, 0, 0, kernel)
            )
            out.setdefault(key, Launches()).add(kernel, duration)
    return out


@dataclass(frozen=True)
class Census:
    """A profiled step's GEMMs, assigned to roles.

    Attributes:
        rows: One per distinct GEMM and stage.
        unassigned: Lines describing device GEMM work no role claimed, and every
            disagreement between the role table and the trace.
    """

    rows: tuple[GemmRow, ...]
    unassigned: tuple[str, ...]


def census(
    shapes: Sequence[GemmShape],
    profiled: Profiled,
    iters: int,
    ceiling: TensorCeiling,
) -> Census:
    """Assign a profile's GEMM launches to roles and score them.

    Args:
        shapes: The role table.
        profiled: A finished profile of `iters` steps.
        iters: Steps inside it.
        ceiling: The square-GEMM ceiling.

    Returns:
        The census.
    """
    seen = observed(profiled)
    rows: list[GemmRow] = []
    loose: list[str] = []
    # One shape can be two roles: a weight stored transposed puts its dgrad on the
    # shape another map's forward runs, and cuBLAS launches one kernel for both.
    # The launch check is over the group, and each row scores the shared median
    # against its own call count.
    claims: dict[tuple[int, int, int, str], list[GemmShape]] = {}
    for shape in shapes:
        claims.setdefault(shape.key, []).append(shape)
    for key, group in claims.items():
        named = "+".join(f"{one.label}.{one.stage}" for one in group)
        found = seen.pop(key, None)
        if found is None:
            loose.append(f"{named}: no launch matched {key}")
            continue
        calls = sum(one.call_count for one in group)
        expected = calls * iters
        if len(found.samples) != expected:
            loose.append(
                f"{named}: {len(found.samples)} launches, "
                f"{expected} expected from {calls} calls over {iters} steps"
            )
        if len(found.kernels) > 1:
            loose.append(
                f"{named}: {len(found.kernels)} kernels under one key, {found.name}"
            )
        rows.extend(
            GemmRow.of(one, found.name, found.samples, ceiling) for one in group
        )
    for (m, n, k, tag), launches in seen.items():
        per_step = sum(launches.samples) / iters
        what = f"{m}x{n}x{k} {tag}" if m else tag[:56]
        loose.append(
            f"unclaimed {what}: {len(launches.samples)} launches, "
            f"{per_step:,.1f} us per step"
        )
    return Census(rows=tuple(rows), unassigned=tuple(loose))


def operands(shape: GemmShape, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """The two operands and the output of one GEMM, in the layout it runs in.

    The layout is built rather than declared: `tn` transposes a contiguous
    `(n,k)` weight, `nt` transposes a contiguous `(k,m)` cotangent, and the
    output is a fresh contiguous `(m,n)`, which is what autograd hands cuBLAS.

    Args:
        shape: What to allocate for.
        device: Where to allocate.

    Returns:
        `(a, b, out)` for `torch.mm(a, b, out=out)`.

    Raises:
        ValueError: On an unknown layout.
    """
    m, n, k = shape.m, shape.n, shape.k
    if shape.layout == TN:
        a = torch.randn(m, k, dtype=DTYPE, device=device)
        b = torch.randn(n, k, dtype=DTYPE, device=device).t()
    elif shape.layout == NN:
        a = torch.randn(m, k, dtype=DTYPE, device=device)
        b = torch.randn(k, n, dtype=DTYPE, device=device)
    elif shape.layout == NT:
        a = torch.randn(k, m, dtype=DTYPE, device=device).t()
        b = torch.randn(k, n, dtype=DTYPE, device=device)
    else:
        raise ValueError(f"unknown layout {shape.layout!r}")
    return a, b, torch.empty(m, n, dtype=DTYPE, device=device)


def isolated(
    shapes: Sequence[GemmShape],
    device: torch.device,
    ceiling: TensorCeiling,
    *,
    iters: int,
    warmup: int,
) -> tuple[GemmRow, ...]:
    """Time every shape alone, one at a time.

    Operands are freed between shapes: the head's cotangent alone is most of a
    gigabyte, and holding every row's operands at once would time them under
    memory pressure the step does not have. An isolated loop runs one GEMM back
    to back with nothing between, which holds clocks differently than a step
    does, so these figures compare with each other and not with a census row.

    Args:
        shapes: The role table.
        device: Device to time on.
        ceiling: The square-GEMM ceiling.
        iters: Timed launches per shape.
        warmup: Untimed launches first.

    Returns:
        One row per shape.
    """
    rows: list[GemmRow] = []
    for shape in shapes:
        a, b, out = operands(shape, device)
        timed = measure(
            lambda a=a, b=b, out=out: torch.mm(a, b, out=out),
            label=f"{shape.label}.{shape.stage}",
            iters=iters,
            warmup=warmup,
            device=device,
        )
        rows.append(GemmRow.of(shape, "", timed.total.samples_duration_us, ceiling))
        del a, b, out
        torch.cuda.empty_cache()
    return tuple(rows)


@dataclass(frozen=True)
class Arm:
    """One call form, and the work it does.

    Attributes:
        label: Region label, distinct within an experiment.
        run: The call. Takes no arguments.
        flop_count: Floating-point operations per call. The two arms of an
            experiment need not agree: padding an extent buys alignment and pays
            in work, and what decides such an experiment is the wall clock, not
            the rate.
    """

    label: str
    run: Callable[[], object]
    flop_count: Count


@dataclass(frozen=True)
class Experiment:
    """Two call forms of one GEMM, measured against each other.

    Attributes:
        label: What is being decided.
        first: The form the library uses today.
        second: The alternative.
        parity: Largest absolute difference between the two forms' results at the
            operand dtype, or None where the two do not compute the same thing. A
            form that reorders a reduction is not required to agree bit for bit,
            and this says by how much it does not.
    """

    label: str
    first: Arm
    second: Arm
    parity: Callable[[], float] | None = None


def aligned_down(extent: int) -> int:
    """`extent` rounded down to a whole number of alignment vectors."""
    return extent - extent % VECTOR_COUNT


def tiled_up(extent: int) -> int:
    """`extent` rounded up to a whole tile of columns."""
    return -(-extent // TILE_COUNT) * TILE_COUNT


def head_experiments(
    config: SLinOSSConfig, tokens: int, device: torch.device
) -> Iterator[Experiment]:
    """The head's three GEMMs, each against a form that pays for alignment.

    A vocabulary that is not a multiple of the alignment vector puts all three
    head GEMMs on an `align1` kernel. Three ways out are priced from the raw
    `vocab_size`, which is what the alignment is worth; the library takes the
    third by default, so this measures a decision already made rather than an
    open one.

    Splitting the forward at the aligned boundary leaves the aligned bulk on a
    wide-load kernel and finishes the remaining columns separately. It is
    measured twice, into a tight output and into one whose row pitch is a whole
    tile, because the extent and the pitch gate the kernel choice separately and
    a split into a tight output buys nothing.

    Staging the cotangent into an aligned pitch does the same for the two
    backward GEMMs. One staged copy serves both, so the pair is measured together
    against the pair before, and the copy is inside the arm that pays for it.

    Padding the head's output width fixes all three at once, which is what
    `SLinOSSConfig.vocab_pad_multiple` does. It is measured to price it and
    nothing else.

    Args:
        config: Supplies `d_model` and `vocab_size`.
        tokens: `B*T`.
        device: Where to allocate.

    Yields:
        One experiment per GEMM and per way out.

    Raises:
        ValueError: If the geometry carries no head.
    """
    vocab = config.vocab_size
    if vocab is None:
        raise ValueError("no head to measure")
    dm = config.d_model
    pad = tiled_up(vocab)
    cut = aligned_down(vocab)
    flop = Count(2 * tokens * vocab * dm)
    pad_flop = Count(2 * tokens * pad * dm)

    x = torch.randn(tokens, dm, dtype=DTYPE, device=device)
    w = torch.randn(vocab, dm, dtype=DTYPE, device=device)
    wp = torch.randn(pad, dm, dtype=DTYPE, device=device)
    logits = torch.empty(tokens, vocab, dtype=DTYPE, device=device)
    logits_pad = torch.empty(tokens, pad, dtype=DTYPE, device=device)
    # Row slices of a row-major weight are contiguous with the same pitch, so
    # neither half of the split copies anything.
    front, rest = w[:cut], w[cut:]

    def one_gemm() -> None:
        torch.mm(x, w.t(), out=logits)

    def split_into(buffer: Tensor) -> Callable[[], None]:
        bulk, tail = buffer[:, :cut], buffer[:, cut:vocab]

        def run() -> None:
            torch.mm(x, front.t(), out=bulk)
            torch.mm(x, rest.t(), out=tail)

        return run

    tight_split = split_into(logits)
    padded_split = split_into(logits_pad)

    def split_parity() -> float:
        one_gemm()
        padded_split()
        return float((logits.float() - logits_pad[:, :vocab].float()).abs().max())

    yield Experiment(
        f"head.fwd M={tokens} N={vocab} K={dm} tn, split into a tight pitch",
        Arm("one_gemm", one_gemm, flop),
        Arm(f"split_{cut}_{vocab - cut}", tight_split, flop),
    )
    yield Experiment(
        f"head.fwd M={tokens} N={vocab} K={dm} tn, split into a pitch of {pad}",
        Arm("one_gemm", one_gemm, flop),
        Arm(f"split_{cut}_{vocab - cut}_pitch_{pad}", padded_split, flop),
        split_parity,
    )
    yield Experiment(
        f"head.fwd M={tokens} N={vocab} K={dm} tn, against a padded vocabulary",
        Arm(f"vocab_{vocab}", one_gemm, flop),
        Arm(f"vocab_{pad}", lambda: torch.mm(x, wp.t(), out=logits_pad), pad_flop),
    )
    # Every buffer above stays live: the closures that hold them are the arms of
    # experiments already yielded, and a generator resumes after its consumer has
    # finished with them, not before.
    dy = torch.randn(tokens, vocab, dtype=DTYPE, device=device)
    dyp = torch.randn(tokens, pad, dtype=DTYPE, device=device)
    dx = torch.empty(tokens, dm, dtype=DTYPE, device=device)
    staged = torch.zeros(tokens, pad, dtype=DTYPE, device=device)
    head = staged[:, :vocab]

    def dgrad() -> None:
        torch.mm(dy, w, out=dx)

    def stage_then_dgrad() -> None:
        head.copy_(dy)
        torch.mm(staged, wp, out=dx)

    yield Experiment(
        f"head.dgrad M={tokens} N={dm} K={vocab} nn, stage to an aligned pitch",
        Arm("one_gemm", dgrad, flop),
        Arm("stage_then_gemm", stage_then_dgrad, pad_flop),
    )
    yield Experiment(
        f"head.dgrad M={tokens} N={dm} K={vocab} nn, against a padded vocabulary",
        Arm(f"vocab_{vocab}", dgrad, flop),
        Arm(f"vocab_{pad}", lambda: torch.mm(dyp, wp, out=dx), pad_flop),
    )

    dw = torch.empty(vocab, dm, dtype=DTYPE, device=device)
    dwp = torch.empty(pad, dm, dtype=DTYPE, device=device)

    def wgrad() -> None:
        torch.mm(dy.t(), x, out=dw)

    def stage_then_wgrad() -> None:
        head.copy_(dy)
        torch.mm(staged.t(), x, out=dwp)
        dw.copy_(dwp[:vocab])

    yield Experiment(
        f"head.wgrad M={vocab} N={dm} K={tokens} nt, stage to an aligned pitch",
        Arm("one_gemm", wgrad, flop),
        Arm("stage_then_gemm", stage_then_wgrad, pad_flop),
    )
    yield Experiment(
        f"head.wgrad M={vocab} N={dm} K={tokens} nt, against a padded vocabulary",
        Arm(f"vocab_{vocab}", wgrad, flop),
        Arm(f"vocab_{pad}", lambda: torch.mm(dyp.t(), x, out=dwp), pad_flop),
    )

    def backward_pair() -> None:
        dgrad()
        wgrad()

    def staged_backward_pair() -> None:
        head.copy_(dy)
        torch.mm(staged, wp, out=dx)
        torch.mm(staged.t(), x, out=dwp)
        dw.copy_(dwp[:vocab])

    yield Experiment(
        f"head backward pair K={vocab} and M={vocab}, one staged copy for both",
        Arm("two_gemms", backward_pair, Count(2 * flop)),
        Arm("stage_then_two_gemms", staged_backward_pair, Count(2 * pad_flop)),
    )


def wgrad_experiments(
    config: SLinOSSConfig, tokens: int, device: torch.device
) -> Iterator[Experiment]:
    """Every weight gradient against the same contraction, transposed.

    `dW = dy^T x` and `dW^T = x^T dy` are one contraction with the output's two
    extents exchanged. Both are `nt`, both do the same work, and cuBLAS tiles
    them differently: the weight extents are small, the reduction is the token
    count, and the orientation decides how many tiles the grid holds and whether
    one wave fills the device. The alternative arm produces the transpose, so a
    map that wants it stores its weight that way; a transposing copy afterwards
    would give the time back.

    Args:
        config: The geometry.
        tokens: `B*T`.
        device: Where to allocate.

    Yields:
        One experiment per map, the head excluded: its own experiments cover it.
    """
    for one in linear_maps(config):
        if one.name == "head":
            continue
        dy = torch.randn(tokens, one.fan_out, dtype=DTYPE, device=device)
        x = torch.randn(tokens, one.fan_in, dtype=DTYPE, device=device)
        dw = torch.empty(one.fan_out, one.fan_in, dtype=DTYPE, device=device)
        dwt = torch.empty(one.fan_in, one.fan_out, dtype=DTYPE, device=device)
        flop = Count(2 * one.fan_out * one.fan_in * tokens)
        yield Experiment(
            f"{one.name}.wgrad M={one.fan_out} N={one.fan_in} K={tokens} nt, "
            f"against the transposed orientation",
            Arm("direct", lambda dy=dy, x=x, dw=dw: torch.mm(dy.t(), x, out=dw), flop),
            Arm(
                "transposed",
                lambda dy=dy, x=x, dwt=dwt: torch.mm(x.t(), dy, out=dwt),
                flop,
            ),
        )
        del dy, x, dw, dwt
        torch.cuda.empty_cache()


def orientation_experiments(
    config: SLinOSSConfig, tokens: int, device: torch.device
) -> Iterator[Experiment]:
    """Each map's three GEMMs with its weight stored `(O,I)` against `(I,O)`.

    Which way a weight is stored fixes all three of its transpose cases at once:
    `(O,I)` gives `tn`, `nn`, `nt`, and `(I,O)` gives `nn`, `tn`, `nt` with the
    weight gradient's two output extents exchanged. Judging one stage alone would
    buy a weight gradient and hand the time back in the forward, so each arm runs
    all three and the sum decides. This is the experiment a layout change is
    settled by; :func:`wgrad_experiments` says which stage moved.

    Args:
        config: The geometry.
        tokens: `B*T`.
        device: Where to allocate.

    Yields:
        One experiment per map, the head excluded: an extent of `vocab_size` is
        ragged in either orientation, so no orientation reaches a wide load.
    """
    for one in linear_maps(config):
        if one.name != "head":
            yield _orientations(one, tokens, device)
            torch.cuda.empty_cache()


def _orientations(one: LinearMap, tokens: int, device: torch.device) -> Experiment:
    """One map's three GEMMs in both weight orientations.

    Its own scope, so the two closures hold this map's operands and no later
    map's.

    Args:
        one: The map.
        tokens: `B*T`.
        device: Where to allocate.

    Returns:
        The experiment.
    """
    fan_in, fan_out = one.fan_in, one.fan_out
    x = torch.randn(tokens, fan_in, dtype=DTYPE, device=device)
    dy = torch.randn(tokens, fan_out, dtype=DTYPE, device=device)
    y = torch.empty(tokens, fan_out, dtype=DTYPE, device=device)
    dx = torch.empty(tokens, fan_in, dtype=DTYPE, device=device)
    weight = torch.randn(fan_out, fan_in, dtype=DTYPE, device=device)
    stored = torch.randn(fan_in, fan_out, dtype=DTYPE, device=device)
    dw = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
    dws = torch.empty(fan_in, fan_out, dtype=DTYPE, device=device)

    def out_by_in() -> None:
        torch.mm(x, weight.t(), out=y)
        torch.mm(dy, weight, out=dx)
        torch.mm(dy.t(), x, out=dw)

    def in_by_out() -> None:
        torch.mm(x, stored, out=y)
        torch.mm(dy, stored.t(), out=dx)
        torch.mm(x.t(), dy, out=dws)

    flop = Count(3 * 2 * fan_out * fan_in * tokens)
    return Experiment(
        f"{one.name} all three stages, weight ({fan_out},{fan_in}) against "
        f"({fan_in},{fan_out})",
        Arm(f"stored_{fan_out}x{fan_in}", out_by_in, flop),
        Arm(f"stored_{fan_in}x{fan_out}", in_by_out, flop),
    )


def width_experiments(
    config: SLinOSSConfig, tokens: int, device: torch.device
) -> Iterator[Experiment]:
    """The input projection's three GEMMs at its width and at the next tile.

    The projection's width is the band total plus the padding that keeps every
    band offset on a sector, so above that bound it is free. Rounding it to a
    whole tile removes a partial tile and pays for the columns it adds. Both arms
    run at their own flop and the wall clock decides.

    Args:
        config: The geometry.
        tokens: `B*T`.
        device: Where to allocate.

    Yields:
        One experiment per stage, or nothing if the width is already a whole
        tile.
    """
    width = ProjectionLayout.from_config(config).width
    wider = tiled_up(width)
    if wider == width:
        return
    dm = config.d_model
    for stage in (FWD, DGRAD, WGRAD):
        arms: list[Arm] = []
        for extent in (width, wider):
            shape = {
                FWD: GemmShape("", FWD, tokens, extent, dm, TN, 1),
                DGRAD: GemmShape("", DGRAD, tokens, dm, extent, NN, 1),
                WGRAD: GemmShape("", WGRAD, extent, dm, tokens, NT, 1),
            }[stage]
            a, b, out = operands(shape, device)
            arms.append(
                Arm(
                    f"width_{extent}",
                    lambda a=a, b=b, out=out: torch.mm(a, b, out=out),
                    shape.flop_count,
                )
            )
        yield Experiment(
            f"in_proj.{stage} width {width} against {wider}", arms[0], arms[1]
        )
        del arms
        torch.cuda.empty_cache()


def run_experiments(
    experiments: Iterator[Experiment],
    device: torch.device,
    ceiling: TensorCeiling,
    *,
    iters: int,
    warmup: int,
) -> None:
    """Measure each experiment's two arms in one loop and print the verdict.

    One loop with the arms swapping order every iteration is the only comparison
    this driver makes: other jobs share the device, and a delta taken from two
    separate loops carries whatever changed between them.

    Args:
        experiments: What to decide.
        device: Device to time on.
        ceiling: The square-GEMM ceiling, for the two achieved rates.
        iters: Timed iterations. Rounded up to even, as pairing requires.
        warmup: Untimed iterations.
    """
    pairs = iters + iters % 2
    for one in experiments:
        parity = None if one.parity is None else one.parity()
        measured = measure_paired(
            one.first.label,
            one.first.run,
            one.second.label,
            one.second.run,
            label=one.label,
            iters=pairs,
            warmup=warmup,
            device=device,
        )
        print(one.label)
        for arm in (one.first, one.second):
            spread = measured.timed.region(arm.label).spread
            rate = tflops_from_flop_us(arm.flop_count, spread.median_duration_us)
            print(
                f"  {arm.label:22s} {spread.median_duration_us:10,.1f} us  "
                f"range {spread.spread_pct:5,.2f}%  half-width "
                f"{spread.resolution_pct:5,.2f}%  {rate:7,.1f} TFLOPS  "
                f"{pct_of(rate, ceiling.achieved_tflops):6,.1f}% of ceiling"
            )
        print(f"  {measured.comparison.verdict()}")
        if parity is not None:
            print(f"  largest absolute difference between the results {parity:g}")
        print()
        torch.cuda.empty_cache()


def algorithm(kernel: str) -> str:
    """The kernel name with the launcher template around it removed.

    What the row is read for is the tile, the stage count, the transpose case and
    the alignment, and on a CUTLASS kernel those sit inside
    `void cutlass::Kernel2<...>`, past the width a table can hold. The wrapper is
    dropped rather than the tail truncated.

    Args:
        kernel: Kernel name as the profiler reports it.

    Returns:
        The innermost template argument, or the name unchanged.
    """
    head, _, rest = kernel.partition("<")
    if not rest or not head.startswith("void "):
        return kernel
    return rest.rpartition(">")[0] or rest


def print_rows(rows: Sequence[GemmRow], ceiling: TensorCeiling) -> None:
    """Print the census table and the class totals.

    Backward rows print first. Two thirds of the class is `dgrad` and `wgrad`,
    and a table sorted by duration alone buries that.

    Args:
        rows: The rows.
        ceiling: The denominator every percentage came from.
    """
    floor = CLASS_FLOOR_PCT[TENSOR_BOUND]
    order = {DGRAD: 0, WGRAD: 1, FWD: 2}
    print(
        f"{'role':24s} {'stage':6s} {'lay':4s} {'M':>7s} {'N':>7s} {'K':>7s} "
        f"{'calls':>5s} {'us/call':>9s} {'range':>7s} {'us/step':>9s} "
        f"{'TFLOPS':>7s} {'ceil':>7s} {'ragged':7s} kernel"
    )
    for row in sorted(rows, key=lambda r: (order.get(r.stage, 3), -r.step_duration_us)):
        print(
            f"{row.label[:24]:24s} {row.stage:6s} {row.layout:4s} {row.m_count:7,d} "
            f"{row.n_count:7,d} {row.k_count:7,d} {row.call_count:5,d} "
            f"{row.duration.median_duration_us:9,.1f} "
            f"{row.duration.spread_pct:6,.2f}% {row.step_duration_us:9,.1f} "
            f"{row.achieved_tflops:7,.1f} {row.ceiling_pct:6,.1f}%"
            f"{' ' if row.ceiling_pct >= floor else '*'} "
            f"{','.join(row.ragged) or '-':7s} {algorithm(row.kernel)}"
        )
    stages: dict[str, float] = {}
    for row in rows:
        stages[row.stage] = stages.get(row.stage, 0.0) + row.step_duration_us
    total = Microseconds(sum(stages.values()))
    flop = Count(sum(row.flop_count * row.call_count for row in rows))
    rate = tflops_from_flop_us(flop, total)
    below = [f"{row.label}.{row.stage}" for row in rows if row.ceiling_pct < floor]
    print()
    for stage in (DGRAD, WGRAD, FWD):
        if stage in stages:
            print(
                f"{stage:6s} {stages[stage] / 1000.0:8,.3f} ms per step, "
                f"{pct_of(stages[stage], total):5,.1f}% of the class"
            )
    print(
        f"class  {total / 1000.0:8,.3f} ms per step over "
        f"{sum(row.call_count for row in rows):,d} launches, {flop / 1e12:,.3f} Tflop, "
        f"{rate:,.1f} TFLOPS, {pct_of(rate, ceiling.achieved_tflops):,.1f}% of the "
        f"ceiling"
    )
    print(f"below the {floor:,.0f}% floor: {', '.join(below) if below else 'none'}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one mode and print its table.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If the geometry carries no head, since the census is of a
            training step and that step reads logits.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    index = device_ordinal(device)
    config = build_config(step_args(args))
    if config.vocab_size is None:
        raise ValueError("the census is of a training step, which needs a vocabulary")
    tokens = args.batch * args.seqlen
    shapes = gemm_shapes(config, tokens)
    before = contention(index)
    ceiling = tensor_ceiling(device)
    print(
        f"device {index}  mode {args.mode}  {config.n_layers} layers  "
        f"tokens {tokens:,d}  vocab {config.vocab_size:,d}  {DTYPE}  "
        f"{clock_policy(index).stamp}"
    )
    print(
        f"ceiling {ceiling.achieved_tflops:,.1f} TFLOPS from a {ceiling.label}, "
        f"range {ceiling.duration.spread_pct:,.2f}% over "
        f"{ceiling.duration.sample_count:,d} launches"
    )
    print(f"before {before.stamp}")
    print()

    if args.mode == "isolated":
        print_rows(
            isolated(shapes, device, ceiling, iters=args.samples, warmup=args.warmup),
            ceiling,
        )
    elif args.mode == "alternatives":
        for experiments in (
            head_experiments(config, tokens, device),
            wgrad_experiments(config, tokens, device),
            orientation_experiments(config, tokens, device),
            width_experiments(config, tokens, device),
        ):
            run_experiments(
                experiments, device, ceiling, iters=args.samples, warmup=args.warmup
            )
    else:
        step = build_step(step_args(args), config, device)
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize(device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as profiled:
            for _ in range(args.iters):
                step()
            torch.cuda.synchronize(device)
        result = census(shapes, profiled, args.iters, ceiling)
        print_rows(result.rows, ceiling)
        for line in result.unassigned:
            print(line)

    print()
    print(f"after {contention(index).stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
