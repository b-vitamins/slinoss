"""Choose a launch geometry by measurement, and persist what was measured.

Sweeps the geometries a block kernel declares in :mod:`slinoss.autotune`, at one
problem shape, on the device in front of it, and writes the winner to the user's
tuning cache. Nothing here decides anything from a model: a candidate displaces
the default only when a paired comparison against the default resolves and is
faster, which is :attr:`slinoss.perf.dispersion.PairedRow.resolves`.

The measurement is :mod:`slinoss.perf`'s, unchanged. Both arms of every candidate
run inside one :func:`slinoss.perf.timing.measure_paired` loop, so a clock
excursion or another tenant hits both arms of a pair and mostly cancels; the
durations stored in a record are the min, median and max of that loop's own
samples for the arm they describe.

Both arms enter :func:`slinoss.autotune.pinned` per call, which clears the
resolution memo, so both pay one context entry and one first-miss resolution per
call. That cost is identical in the two arms and cancels out of the per-iteration
difference the verdict is taken on. What the hook costs on the steady-state path
is a separate question, answered by ``--host-overhead``.

The shape a record is addressed by is read back from the kernel, not derived here:
:func:`slinoss.autotune.witnessed` runs the arm once and reports which keys
resolved. The parameter-gradient tail is launched over the row count of the grid
that produced its partials, so its key is a function of the geometry the reducing
kernel ran at. That is also why the kernels are swept in the order of
:data:`KERNEL_ORDER` and why each record is installed as soon as it is written: a
``rmsnorm_dweight`` record is measured against whatever ``rmsnorm_bwd`` resolves
to, so the reducing kernel is tuned first and its winner is in force.

A geometry the device will not launch is not a hole in the sweep. Each candidate
runs once outside the timing loop; a refusal is recorded as
:meth:`slinoss.autotune.Attempt.refused` carrying the reason, and so is a
candidate whose output disagrees with the default's beyond :data:`AGREEMENT_TOL`.

A per-kernel win is not the claim. ``--verify`` times the whole block step, all
three kernels forward and their pullbacks, with the shape's records in force
against an empty cache: two arms differing by selection alone, no pin, the same
callable the bench times.

    python3 scripts/perf/tune.py --kernel swiglu_fwd --d-model 8192 --tokens long
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from slinoss import autotune
from slinoss.autotune import AGREEMENT_TOL, Attempt, Record, ShapeKey, Variants
from slinoss.ops.block import rmsnorm, rmsnorm_residual, swiglu
from slinoss.perf.ceiling import DramTimeFloor, dram_time_floor
from slinoss.perf.device import (
    clock_policy,
    contention,
    device_info,
    device_ordinal,
    require_cuda,
)
from slinoss.perf.dispersion import PairedRow
from slinoss.perf.report import Report, write_report
from slinoss.perf.timing import measure, measure_paired
from slinoss.perf.units import CONFIDENCE_PCT, Bytes, Microseconds
from slinoss.perf.workload import (
    SHAPE_NAMES,
    BlockInputs,
    BlockShape,
    OpShape,
    block_step,
    make_block_inputs,
    shape_by_name,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

BACKEND = "cute"
"""The only block backend with a launch geometry. The reference path has none."""

D_MODEL_GEOMETRY = {
    288: (12, 48),
    384: (12, 64),
    576: (24, 48),
    1152: (48, 48),
    4096: (128, 64),
    8192: (256, 64),
}
"""``(heads, rows)`` realizing each swept ``d_model``.

:func:`slinoss.perf.workload.layer_config` reads ``d_model`` off ``H*P/2``, so a
width is requested by naming the scan that implies it rather than by a second
number that can disagree with it. Every ``rows`` here is a multiple of
:data:`slinoss.config.HEAD_MULTIPLE`.
"""

KERNEL_ORDER = (
    "rmsnorm_fwd",
    "rmsnorm_residual_fwd",
    "swiglu_fwd",
    "rmsnorm_bwd",
    "rmsnorm_residual_bwd",
    "swiglu_bwd",
    "rmsnorm_dweight",
)
"""Sweep order. ``rmsnorm_dweight`` is last because its shape key depends on the
grid ``rmsnorm_bwd`` runs at, so the reducing kernel is tuned first."""


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
#
# One callable per kernel, built from the public op the way slinoss.perf.workload
# builds the bench arms, so what is tuned is what ships. A backward arm builds its
# graph once, outside the callable, and retains it: the pullback is the launch
# under test and the forward that produced the graph is not in the timed region.


def _rmsnorm_fwd(inputs: BlockInputs, shape: BlockShape) -> Callable[[], object]:
    def run() -> object:
        with torch.no_grad():
            return rmsnorm(
                inputs.prehead, inputs.weight, eps=shape.eps, backend=BACKEND
            )

    return run


def _rmsnorm_residual_fwd(
    inputs: BlockInputs, shape: BlockShape
) -> Callable[[], object]:
    def run() -> object:
        with torch.no_grad():
            return rmsnorm_residual(
                inputs.x,
                inputs.residual,
                inputs.weight,
                eps=shape.eps,
                backend=BACKEND,
            )

    return run


def _swiglu_fwd(inputs: BlockInputs, shape: BlockShape) -> Callable[[], object]:
    del shape

    def run() -> object:
        with torch.no_grad():
            return swiglu(inputs.gate, inputs.up, backend=BACKEND)

    return run


def _rmsnorm_bwd(inputs: BlockInputs, shape: BlockShape) -> Callable[[], object]:
    head = rmsnorm(inputs.prehead, inputs.weight, eps=shape.eps, backend=BACKEND)
    targets = (inputs.prehead, inputs.weight)

    def run() -> object:
        return torch.autograd.grad(head, targets, inputs.dprehead, retain_graph=True)

    return run


def _rmsnorm_residual_bwd(
    inputs: BlockInputs, shape: BlockShape
) -> Callable[[], object]:
    out = rmsnorm_residual(
        inputs.x, inputs.residual, inputs.weight, eps=shape.eps, backend=BACKEND
    )
    targets = (inputs.x, inputs.residual, inputs.weight)
    cotangents = (inputs.dnormed, inputs.dresidual)

    def run() -> object:
        return torch.autograd.grad(
            (out.normed, out.residual), targets, cotangents, retain_graph=True
        )

    return run


def _swiglu_bwd(inputs: BlockInputs, shape: BlockShape) -> Callable[[], object]:
    del shape
    out = swiglu(inputs.gate, inputs.up, backend=BACKEND)
    targets = (inputs.gate, inputs.up)

    def run() -> object:
        return torch.autograd.grad(out, targets, inputs.dout, retain_graph=True)

    return run


# ---------------------------------------------------------------------------
# Analytic traffic
# ---------------------------------------------------------------------------
#
# Each of these restates one bullet of the traffic its kernel's module docstring
# declares, at operand itemsize i, and nothing else. The roofline model is
# slinoss.perf.ceiling's; this only says how many bytes to hand it.


def _norm_fwd_bytes(shape: BlockShape, itemsize: int) -> Bytes:
    """``2*D*i`` per row, plus the float32 weight once."""
    width = shape.width
    return Bytes(int(shape.token_count) * 2 * width * itemsize + 4 * width)


def _residual_fwd_bytes(shape: BlockShape, itemsize: int) -> Bytes:
    """``D*(i_x + 4 + 4 + i_normed)`` per row plus the weight: ``x``, the float32
    residual in, the float32 sum out, the normed output."""
    width = shape.width
    return Bytes(int(shape.token_count) * width * (2 * itemsize + 8) + 4 * width)


def _swiglu_fwd_bytes(shape: BlockShape, itemsize: int) -> Bytes:
    """``3*numel*i``: one read of each operand, one write."""
    return Bytes(3 * int(shape.token_count) * shape.hidden * itemsize)


def _swiglu_bwd_bytes(shape: BlockShape, itemsize: int) -> Bytes:
    """``5*numel*i``: the cotangent, both operands, both gradients."""
    return Bytes(5 * int(shape.token_count) * shape.hidden * itemsize)


def _undeclared(shape: BlockShape, itemsize: int) -> Bytes:
    """No analytic traffic, because it is not a function of the shape alone.

    Every reducing backward writes ``4*D`` float32 of partials per block of its
    grid and the reduction reads them back, so its traffic moves with the grid --
    which is one of the axes under test. One figure per shape would be wrong for
    every candidate but one, and a roofline percentage against it would rank
    candidates by that error. The parameter-gradient tail reads the same buffer and
    inherits the same dependence.
    """
    del shape, itemsize
    return Bytes(0)


@dataclass(frozen=True)
class Arm:
    """How one registered kernel is exercised and judged.

    Attributes:
        kernel: Registered name, and the record key.
        build: Builds the callable that launches it, from the inputs and shape.
        traffic: Analytic DRAM traffic of one launch, or zero when it is not a
            function of the shape alone. See :func:`_undeclared`.
        extent: The row length the launch runs at, for the printed table: the
            residual stream for the norms, the FFN hidden for the activation.
    """

    kernel: str
    build: Callable[[BlockInputs, BlockShape], Callable[[], object]]
    traffic: Callable[[BlockShape, int], Bytes]
    extent: Callable[[BlockShape], int]


ARMS: dict[str, Arm] = {
    arm.kernel: arm
    for arm in (
        Arm("rmsnorm_fwd", _rmsnorm_fwd, _norm_fwd_bytes, lambda s: s.width),
        Arm(
            "rmsnorm_residual_fwd",
            _rmsnorm_residual_fwd,
            _residual_fwd_bytes,
            lambda s: s.width,
        ),
        Arm("swiglu_fwd", _swiglu_fwd, _swiglu_fwd_bytes, lambda s: s.hidden),
        Arm("rmsnorm_bwd", _rmsnorm_bwd, _undeclared, lambda s: s.width),
        Arm(
            "rmsnorm_residual_bwd",
            _rmsnorm_residual_bwd,
            _undeclared,
            lambda s: s.width,
        ),
        Arm("swiglu_bwd", _swiglu_bwd, _swiglu_bwd_bytes, lambda s: s.hidden),
        # Shares the plain norm's pullback: that launch is what produces the
        # partials this kernel reduces, so it is how the tail is reached at all.
        Arm("rmsnorm_dweight", _rmsnorm_bwd, _undeclared, lambda s: s.width),
    )
}


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _flat(value: object) -> tuple[Tensor, ...]:
    """Every tensor an arm returned, in order. Anything else is dropped."""
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, tuple | list):
        return tuple(t for item in value for t in _flat(item))
    return ()


def _disagreement(got: Sequence[Tensor], want: Sequence[Tensor]) -> float:
    """Largest relative difference across a pair of arm outputs, in float64.

    Args:
        got: The candidate's outputs.
        want: The default's outputs, same count and shapes.

    Returns:
        The largest ``|got - want| / max|want|`` over the pair, or ``inf`` if the
        two arms returned different counts or shapes, which is not a difference in
        rounding and must not read as agreement.
    """
    if len(got) != len(want):
        return float("inf")
    worst = 0.0
    for a, b in zip(got, want):
        if a.shape != b.shape:
            return float("inf")
        reference = b.detach().double()
        scale = max(float(reference.abs().max()), torch.finfo(torch.float64).tiny)
        worst = max(worst, float((a.detach().double() - reference).abs().max()) / scale)
    return worst


def _pin(
    kernel: str, geometry: Sequence[int], body: Callable[[], object]
) -> Callable[[], object]:
    """``body`` with one kernel's geometry forced. Both arms are built this way, so
    the pin's own cost is in both and cancels in the difference."""

    def run() -> object:
        with autotune.pinned({kernel: geometry}):
            return body()

    return run


@dataclass(frozen=True)
class Probe:
    """One candidate measured against the default.

    Attributes:
        attempt: The candidate as the sweep found it, measured or refused.
        comparison: The paired verdict, absent when the candidate never ran.
    """

    attempt: Attempt
    comparison: PairedRow | None = None


def _probe(
    variants: Variants[tuple[int, ...]],
    geometry: Sequence[int],
    build: Callable[[], Callable[[], object]],
    reference: Sequence[Tensor],
    tol: float,
    *,
    label: str,
    iters: int,
    warmup: int,
    device: torch.device,
) -> Probe:
    """Measure one candidate against the kernel's default, in one paired loop.

    Args:
        variants: The kernel's declaration.
        geometry: The candidate.
        build: Builds a fresh arm callable. Called twice, once per arm, so the two
            arms hold separate retained graphs and neither reuses the other's.
        reference: The default's outputs, for the agreement check.
        tol: Largest relative disagreement admitted.
        label: What is being compared.
        iters: Timed iterations. Even.
        warmup: Untimed iterations.
        device: Device to time on.

    Returns:
        The probe. A candidate the device refused, or one that disagreed with the
        default, carries a refusal and no comparison.
    """
    base = _pin(variants.kernel, variants.default, build())
    candidate = _pin(variants.kernel, geometry, build())
    try:
        got = _flat(candidate())
        torch.cuda.synchronize(device)
    except Exception as failure:
        # A launch the device refuses is a fact about the candidate, not an error in
        # the sweep. Resynchronize so a sticky context surfaces here rather than as
        # a wrong duration three candidates later.
        torch.cuda.synchronize(device)
        return Probe(Attempt.refused(geometry, f"{type(failure).__name__}: {failure}"))
    error = _disagreement(got, reference)
    del got
    if not error < tol:
        return Probe(
            Attempt.refused(
                geometry, f"disagrees with the default by {error:.3e}, over {tol:.1e}"
            )
        )
    measured = measure_paired(
        "default",
        base,
        "candidate",
        candidate,
        label=label,
        iters=iters,
        warmup=warmup,
        device=device,
    )
    samples = measured.timed.region("candidate").spread.samples_duration_us
    return Probe(Attempt.of(geometry, samples), measured.comparison)


@dataclass(frozen=True)
class Sweep:
    """Every candidate of one kernel at one shape, and the record it produced.

    Attributes:
        record: What was written. Its winner is the default when nothing beat it.
        default: The default geometry measured on its own, which is what a process
            with no cache pays.
        verdict: The winner's own paired comparison against the default, absent
            when the default held.
        rows: One paired verdict per candidate that ran, in probe order.
        floor_us: Time floor at this kernel's analytic traffic, or None when the
            traffic is not a function of the shape alone or no floor was fitted.
    """

    record: Record
    default: Attempt
    verdict: PairedRow | None
    rows: tuple[PairedRow, ...]
    floor_us: Microseconds | None


def _sweep(
    variants: Variants[tuple[int, ...]],
    key: ShapeKey,
    build: Callable[[], Callable[[], object]],
    *,
    label: str,
    iters: int,
    warmup: int,
    device: torch.device,
    tol: float,
    conditions: str,
    floor_us: Microseconds | None,
) -> Sweep:
    """Measure every candidate of one kernel at one shape and pick the winner.

    Args:
        variants: The kernel's declaration.
        key: The shape the kernel resolved at, read back from it.
        build: Builds a fresh arm callable.
        label: Report label for the comparisons.
        iters: Timed iterations per candidate. Even.
        warmup: Untimed iterations per candidate.
        device: Device to time on.
        tol: Largest relative disagreement a candidate may show.
        conditions: Clock and sharing stamp, stored with the record.
        floor_us: Time floor at the analytic traffic, or None.

    Returns:
        The sweep.
    """
    baseline = measure(
        _pin(variants.kernel, variants.default, build()),
        label=f"{label} default",
        iters=iters,
        warmup=warmup,
        device=device,
    )
    default = Attempt.of(variants.default, baseline.total.samples_duration_us)
    reference = _flat(_pin(variants.kernel, variants.default, build())())
    torch.cuda.synchronize(device)

    probes: list[Probe] = []
    for geometry in variants.candidates:
        if tuple(geometry) == tuple(variants.default):
            continue
        probes.append(
            _probe(
                variants,
                geometry,
                build,
                reference,
                tol,
                label=f"{label} {tuple(geometry)}",
                iters=iters,
                warmup=warmup,
                device=device,
            )
        )
    del reference

    # Two hurdles, and a candidate clears both or the default holds.
    #
    # The paired verdict licenses the difference between the two arms of one loop.
    # It is not by itself a statement about deployment: in a loop that alternates
    # arms, a geometry can resolve faster than the default while its own median is
    # worse than the default's when the default runs alone, which is what a process
    # without a cache pays. Records selected on the paired verdict alone made the
    # whole block step measurably slower at three of the swept shapes. So the
    # candidate's own samples must also beat the solo baseline.
    #
    # Among the survivors, the fastest median; a tie keeps the earlier candidate,
    # which is declaration order.
    beats = [
        probe
        for probe in probes
        if probe.comparison is not None
        and probe.comparison.resolves
        and probe.comparison.delta_median_duration_us < 0.0
        and probe.attempt.median_duration_us < default.median_duration_us
    ]
    best = min(beats, key=lambda p: p.attempt.median_duration_us) if beats else None
    winner = best.attempt if best is not None else default
    others = [
        attempt
        for attempt in (default, *(probe.attempt for probe in probes))
        if attempt.geometry != winner.geometry
    ]
    others.sort(key=lambda a: (not a.measured, a.median_duration_us))
    torch_version, cutlass_version = autotune.versions()
    return Sweep(
        record=Record(
            kernel=variants.kernel,
            shape=key,
            device=autotune.device_key(device_ordinal(device)),
            winner=winner,
            runners_up=tuple(others),
            repeat_count=iters,
            probe_count=len(probes) + 1,
            torch_version=torch_version,
            cutlass_version=cutlass_version,
            conditions=conditions,
        ),
        default=default,
        verdict=best.comparison if best is not None else None,
        rows=tuple(p.comparison for p in probes if p.comparison is not None),
        floor_us=floor_us,
    )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _installed(
    entries: Sequence[Record], body: Callable[[], object]
) -> Callable[[], object]:
    """``body`` with exactly ``entries`` in force, chosen per call.

    Per call because the two arms alternate inside one paired loop and each needs
    its own table. Both arms therefore pay one table build and one first-miss
    resolution per launch site, and that cost cancels in the per-iteration
    difference; only the geometries differ.
    """

    def run() -> object:
        autotune.install(entries)
        return body()

    return run


def _end_to_end(
    inputs: BlockInputs,
    shape: BlockShape,
    entries: Sequence[Record],
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> PairedRow:
    """The whole block step, tuned against untuned, at one shape.

    The arm is :func:`slinoss.perf.workload.block_step`: all three kernels forward
    and their pullbacks, the same callable the bench times. Nothing is pinned. The
    tuned arm reaches its geometries the way a user's process would, through the
    cache, so what this measures is selection and not a hand-set constant.

    Args:
        inputs: Block inputs, differentiable ones requiring grad.
        shape: The problem size.
        entries: The records measured at this shape.
        iters: Timed iterations. Even.
        warmup: Untimed iterations.
        device: Device to time on.

    Returns:
        The paired verdict. ``resolves`` is what licenses a speedup claim, and a
        negative ``delta_median_duration_us`` is the tuned arm being faster.
    """
    measured = measure_paired(
        "untuned",
        _installed((), block_step(inputs, shape, backend=BACKEND)),
        "tuned",
        _installed(entries, block_step(inputs, shape, backend=BACKEND)),
        label=f"block step {shape.name}",
        iters=iters,
        warmup=warmup,
        device=device,
    )
    return measured.comparison


# ---------------------------------------------------------------------------
# Host overhead
# ---------------------------------------------------------------------------


def select_overhead_us(
    variants: Variants[tuple[int, ...]], key: ShapeKey, index: int, calls: int
) -> tuple[float, float]:
    """Host cost of one steady-state resolution, and of the read it replaced.

    The hook replaced a module-constant read at each launch site, so that read is
    the baseline and the difference is what dispatch pays per call for being
    tunable. Both loops run with the memo warm, which is the steady state; the
    first resolution of a shape is a different quantity and happens once per shape
    per process.

    Args:
        variants: The kernel's declaration.
        key: The shape to resolve at.
        index: CUDA device ordinal.
        calls: Calls per loop.

    Returns:
        Microseconds per :meth:`slinoss.autotune.Variants.select`, then
        microseconds per attribute read.
    """
    variants.select(key.rows, key.width, key.itemsize, index, key.extents)
    start = time.perf_counter_ns()
    for _ in range(calls):
        variants.select(key.rows, key.width, key.itemsize, index, key.extents)
    hooked = (time.perf_counter_ns() - start) / calls / 1e3
    start = time.perf_counter_ns()
    for _ in range(calls):
        _ = variants.default
    plain = (time.perf_counter_ns() - start) / calls / 1e3
    return hooked, plain


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def block_shape(d_model: int, tokens: OpShape) -> BlockShape:
    """The block of a layer ``d_model`` wide, at one batch and sequence length.

    Args:
        d_model: Residual-stream width. One of :data:`D_MODEL_GEOMETRY`.
        tokens: The bench shape naming the batch and sequence length.

    Returns:
        The block shape.

    Raises:
        KeyError: If no head geometry realizes ``d_model``.
    """
    heads, rows = D_MODEL_GEOMETRY[d_model]
    return BlockShape(
        OpShape(
            name=f"{tokens.name}-d{d_model}",
            bsz=tokens.bsz,
            heads=heads,
            seq=tokens.seq,
            rows=rows,
            lanes=tokens.lanes,
            chunk=tokens.chunk,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        nargs="+",
        choices=list(KERNEL_ORDER),
        default=list(KERNEL_ORDER),
        help="Registered kernels to sweep. Always swept in KERNEL_ORDER, whatever "
        "order they are given in, because one key depends on another kernel's grid.",
    )
    parser.add_argument(
        "--d-model",
        nargs="+",
        type=int,
        choices=sorted(D_MODEL_GEOMETRY),
        default=sorted(D_MODEL_GEOMETRY),
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        choices=list(SHAPE_NAMES),
        default=["standard", "long"],
        help="Bench shapes naming the batch and sequence length. Only B and T reach "
        "a block kernel, so two names carrying one pair are one sweep.",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, cuda or cuda:N. There is no host path: a geometry is "
        "chosen for the part it was measured on.",
    )
    parser.add_argument(
        "--floor",
        action="store_true",
        help="Fit the copy time law once and report each winner against the floor "
        "at its own analytic traffic. Costs one bandwidth sweep of the device.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After sweeping a shape, time the whole block step with that shape's "
        "records in force against an empty cache. This is the end-to-end claim: "
        "the arms differ by selection alone, nothing is pinned.",
    )
    parser.add_argument(
        "--host-overhead",
        type=int,
        default=0,
        metavar="CALLS",
        help="Measure the per-call host cost the resolution hook adds, over this "
        "many calls. Zero skips it.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help=f"Tuning cache to merge into. Defaults to ${autotune.CACHE_ENV}, else "
        f"the per-user cache directory.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/tune"))
    return parser.parse_args(argv)


HEADER = (
    f"{'kernel':<21} {'shape':<18} {'extent':>6} {'probes':>6} "
    f"{'base_min':>9} {'base_med':>9} {'base_max':>9} "
    f"{'win_min':>9} {'win_med':>9} {'win_max':>9} "
    f"{'solo_pct':>8} {'pair_pct':>8} {'floor_pct':>9}  geometry"
)


def _row(kernel: str, shape: BlockShape, extent: int, sweep: Sweep) -> str:
    """One table line: the default, the winner, and the two deltas separately.

    ``solo_pct`` is each arm against the other's own samples, which are measured in
    different loops and carry no interval. ``pair_pct`` is the winner's own paired
    verdict, which is the licensed figure and is empty when the default held. Both
    are printed because a displacement had to clear both.
    """
    base, winner = sweep.default, sweep.record.winner
    solo = (
        100.0
        * (winner.median_duration_us - base.median_duration_us)
        / base.median_duration_us
    )
    pair = (
        f"{sweep.verdict.delta_pct:>8,.2f}"
        if sweep.verdict is not None
        else f"{'--':>8}"
    )
    floor = (
        f"{100.0 * sweep.floor_us / winner.median_duration_us:>9,.1f}"
        if sweep.floor_us is not None
        else f"{'--':>9}"
    )
    return (
        f"{kernel:<21} {shape.name:<18} {extent:>6} {sweep.record.probe_count:>6} "
        f"{base.min_duration_us:>9,.2f} {base.median_duration_us:>9,.2f} "
        f"{base.max_duration_us:>9,.2f} "
        f"{winner.min_duration_us:>9,.2f} {winner.median_duration_us:>9,.2f} "
        f"{winner.max_duration_us:>9,.2f} {solo:>8,.2f} {pair} {floor}  "
        f"{tuple(winner.geometry)}"
    )


VERIFY_HEADER = (
    f"{'block step':<21} {'shape':<18} {'untuned':>10} {'tuned':>10} "
    f"{'delta_us':>9} {'ci_low':>9} {'ci_high':>9} {'delta_pct':>9} "
    f"{'speedup':>8} {'kept':>6}"
)


def kept(row: PairedRow) -> bool:
    """Whether a shape's records earned the file.

    A record only exists to make a real forward and backward faster, so the whole
    block step has to resolve faster with the shape's records in force. An interval
    straddling zero is consistent with no difference at all, and a positive one is
    a regression the cache would then ship.

    Args:
        row: The end-to-end verdict, tuned against untuned.

    Returns:
        True if the records may be written.
    """
    return row.resolves and row.delta_median_duration_us < 0.0


def _verify_row(shape: BlockShape, row: PairedRow) -> str:
    """One end-to-end line, and whether the shape's records were written."""
    return (
        f"{'tuned vs untuned':<21} {shape.name:<18} "
        f"{row.a_median_duration_us:>10,.2f} {row.b_median_duration_us:>10,.2f} "
        f"{row.delta_median_duration_us:>9,.2f} {row.delta_low_duration_us:>9,.2f} "
        f"{row.delta_high_duration_us:>9,.2f} {row.delta_pct:>9,.2f} "
        f"{row.speedup_ratio:>8,.3f} {kept(row)!s:>6}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep, persist, and print.

    Returns:
        Process exit status. Nonzero if a requested kernel is not registered.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If an arm never launched the kernel it is supposed to measure.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    ordinal = device_ordinal(device)
    dtype = DTYPES[args.dtype]
    tol = AGREEMENT_TOL[dtype]
    declared = autotune.registered()
    wanted = [name for name in KERNEL_ORDER if name in set(args.kernel)]
    missing = [name for name in wanted if name not in declared]
    if missing:
        print(f"not registered: {missing}; import the op package that declares them")
        return 1

    conditions = f"{clock_policy(ordinal).stamp}, {contention(ordinal).stamp}"
    floor: DramTimeFloor | None = dram_time_floor(device) if args.floor else None

    # One sweep per (B,T): those two extents are all a block kernel sees of a shape.
    tokens = list(
        {
            (shape_by_name(n).bsz, shape_by_name(n).seq): shape_by_name(n)
            for n in args.tokens
        }.values()
    )
    sweeps: list[tuple[str, BlockShape, int, Sweep]] = []
    comparisons: list[PairedRow] = []
    verified: list[tuple[BlockShape, PairedRow]] = []
    dropped: list[str] = []
    overhead: tuple[str, float, float] | None = None
    written: Path | None = None
    # Accumulated in memory rather than read back per kernel: under --verify a
    # shape's records are provisional until the block step has been timed with them,
    # and only what survives that reaches the file.
    table = autotune.load(args.cache)

    print(HEADER, flush=True)
    for width in sorted(args.d_model):
        for token_shape in tokens:
            shape = block_shape(width, token_shape)
            held = table
            inputs = make_block_inputs(shape, device, dtype=dtype, requires_grad=True)
            for name in wanted:
                arm = ARMS[name]
                variants = declared[name]

                def build(
                    arm: Arm = arm,
                    shape: BlockShape = shape,
                    inputs: BlockInputs = inputs,
                ) -> Callable[[], object]:
                    return arm.build(inputs, shape)

                with autotune.witnessed() as seen:
                    build()()
                    torch.cuda.synchronize(device)
                keys = dict(seen)
                if name not in keys:
                    raise ValueError(
                        f"the {name} arm resolved {sorted(keys)} and not {name}; it "
                        f"does not launch the kernel it is supposed to measure"
                    )
                moved = arm.traffic(shape, dtype.itemsize)
                sweep = _sweep(
                    variants,
                    keys[name],
                    build,
                    label=f"{name} {shape.name}",
                    iters=args.iters,
                    warmup=args.warmup,
                    device=device,
                    tol=tol,
                    conditions=conditions,
                    floor_us=(
                        floor.floor_us(moved) if floor is not None and moved else None
                    ),
                )
                extent = arm.extent(shape)
                sweeps.append((name, shape, extent, sweep))
                comparisons.extend(sweep.rows)
                # Installed per kernel, so a later kernel resolves against the
                # winners already measured: rmsnorm_dweight is launched over the grid
                # rmsnorm_bwd ran at.
                table = autotune.merge(table, [sweep.record])
                autotune.install(table)
                print(_row(name, shape, extent, sweep), flush=True)
                if args.host_overhead and overhead is None:
                    overhead = (
                        name,
                        *select_overhead_us(
                            variants, keys[name], ordinal, args.host_overhead
                        ),
                    )
            if args.verify:
                row = _end_to_end(
                    inputs,
                    shape,
                    tuple(s.record for _, at, _, s in sweeps if at.name == shape.name),
                    iters=args.iters,
                    warmup=args.warmup,
                    device=device,
                )
                verified.append((shape, row))
                if not kept(row):
                    # Every one of this shape's records goes. The set was measured
                    # together and its effect was measured together; keeping the
                    # subset whose own probe looked best would be selecting on the
                    # statistic that just failed.
                    table = held
                    dropped.append(shape.name)
                # Each verify arm installed its own table, so the last one to run is
                # still in force. Restore what will be written, which the next
                # shape's key derivation resolves against.
                autotune.install(table)
                print(_verify_row(shape, row), flush=True)
            written = autotune.save(table, args.cache)
            del inputs
            torch.cuda.empty_cache()

    notes = [
        f"dtype={args.dtype} iters={args.iters} warmup={args.warmup} backend={BACKEND}",
        f"agreement_tol={tol:.3e} confidence_pct={CONFIDENCE_PCT:.1f}",
        f"conditions: {conditions}",
        f"cache: {written}" if written is not None else "no sweep ran",
        "a candidate displaces the default only where its paired comparison against "
        "the default resolves faster and its own median also beats the default's "
        "solo median; the paired verdict alone is a statement about a loop that "
        "alternates arms, not about deployment",
        "base and win are each arm's own samples, measured in different loops, so "
        "solo_pct carries no interval; pair_pct is the winner's own paired verdict "
        "and is the licensed figure",
    ]
    if floor is not None:
        notes.append(
            f"floor law: c={floor.fixed_duration_us:,.3f} us "
            f"B={floor.asymptotic_gbs:,.1f} GB/s "
            f"max_residual_pct={floor.max_residual_pct:,.2f}"
        )
        notes.append(
            "floor_pct is absent for both reducing backwards and for the "
            "parameter-gradient tail: their traffic moves with the grid, which is "
            "an axis under test"
        )
    if overhead is not None:
        kernel, hooked, plain = overhead
        notes.append(
            f"host overhead on {kernel}: select {hooked:.3f} us/call against "
            f"{plain:.3f} us/call for the constant read it replaced, "
            f"{hooked - plain:+.3f} us/call over {args.host_overhead:,} calls"
        )
    if verified:
        notes.append(
            "block step rows compare the whole three-kernel forward and pullback "
            "with this shape's records in force against an empty cache; both arms "
            "install a table per call, so selection is the only difference"
        )
        notes.append(
            "a shape's records reach the file only where that row resolves faster; "
            f"not written: {dropped if dropped else 'none'}"
        )
    notes.extend(dict.fromkeys(shape.describe() for _, shape, _, _ in sweeps))

    report = Report(
        title=f"tune: block launch geometry, {args.dtype}",
        device=device_info(ordinal),
        comparisons=(*comparisons, *(row for _, row in verified)),
        notes=tuple(notes),
    )
    md, _ = write_report(report, args.out, require_agreement=False)
    print()
    print(HEADER)
    for name, shape, extent, sweep in sweeps:
        print(_row(name, shape, extent, sweep))
    if verified:
        print()
        print(VERIFY_HEADER)
        for at, row in verified:
            print(_verify_row(at, row))
    print()
    for note in notes[:8]:
        print(note)
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
