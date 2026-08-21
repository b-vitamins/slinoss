"""Where the end-to-end layer comparison's time goes, on both arms, and the Amdahl
ceiling that follows.

``scripts/bench/bench_mamba.py --end-to-end`` reports one ratio between the whole
Mamba2 layer and the whole SO(3) mixer. This says what that ratio is made of: a
device-time partition of each arm's forward and backward, stage by stage, and the
end-to-end ratio that would follow if the SO(3) scan operator alone were faster.

    python3 scripts/perf/attribute_e2e.py --mamba-chunk 256

Both arms are built by ``bench_mamba``'s own constructors and run inside one
:func:`slinoss.perf.timing.measure_paired` loop, so the headline ratio here is the
one that driver reports and not a second definition of it.

Two instruments, and they answer different questions:

- The region timers give each arm's total and its forward/backward split. That is
  wall time on the stream and it is what the ratio is taken over.
- The profiler gives the per-kernel device-time partition, which is what attributes
  a stage. Kernel names separate every stage of the SO(3) mixer except the two
  projections: both are cuBLAS GEMMs. Those are split by grouping the ``aten::mm``
  and ``aten::addmm`` operator rows by input shape, which names the six GEMMs of a
  step uniquely, and cross-checked against the GEMM-class kernel total.

A profiled interval runs long, so every share is taken against the profiled arm's
own kernel sum and every absolute is reported beside the unprofiled event time the
same arm was measured at.

The Amdahl table holds everything but the scan fixed and moves the scan alone. Its
last row is a scan that costs nothing, which is the upper bound on what operator
work can reach.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from scripts.bench.bench_mamba import (
    DTYPES,
    block_runner,
    block_stream,
    d_model_of,
    fused_path_blocker,
    make_mamba_block,
    mamba_chunk,
    parameter_counts,
)
from slinoss.perf.device import device_info, device_ordinal, require_cuda
from slinoss.perf.memory import (
    SavedTensorProbe,
    memory_peaks,
    reset_memory_peaks,
)
from slinoss.perf.timing import measure, measure_paired
from slinoss.perf.workload import layer_config, shape_by_name

GEMM_OPS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm")
"""Operator names whose device time is a projection's GEMM."""

GEMM_MARKERS = (
    "cutlass::Kernel",
    "ampere_",
    "sm80_",
    "sm86_",
    "gemm",
    "gemv",
    "nn_128x128",
    "cublas",
)
"""Kernel-name fragments of a cuBLAS or CUTLASS GEMM.

``kernel_cutlass_`` is the CuTe DSL's own prefix and is not one of these: it is
matched by a stage rule ahead of this table, so a DSL kernel is never counted as a
GEMM.
"""

GLUE_MARKERS = (
    "elementwise_kernel",
    "reduce_kernel",
    "vectorized_",
    "fill_",
    "Memcpy",
    "Memset",
    "unrolled_",
    "CatArrayBatched",
    "index_select",
    "copy_",
    "at::native",
)
"""Kernel-name fragments of aten glue and data movement."""

SO3SSD_KERNELS = (
    "chunk_increment_fwd",
    "increment_passing_fwd",
    "state_passing_fwd",
    "chunk_scan_fwd",
    "chunk_input_bwd",
    "chunk_vector_bwd",
    "start_passing_bwd",
    "state_passing_bwd",
    "boundary_bwd",
    "chunk_prefix_bwd",
    "chunk_start_bwd",
    # The head-sum close of dB, dC and the b carry, launched by
    # chunk_vector_backward whenever the head-sum depth exceeds one. It is the
    # scan operator's kernel and not the tail's, whose own name is mixer_tail.
    "vector_reduce",
)
"""Every kernel the SO(3) scan operator can launch, forward and backward.

``chunk_start_bwd``, ``chunk_increment_fwd`` and ``state_passing_fwd`` are fused
into ``start_passing_bwd`` and ``increment_passing_fwd`` respectively, and
``state_passing_bwd`` runs only when ``dy`` is absent. All four are listed so that a
launch of one would be attributed rather than fall through to ``other``: their
presence in a profile of the mixer means the cute arms are not the ones running.
"""

SHARED_REDUCE = "reduce_rows"
"""The one kernel three stages launch.

``reduce_partials`` closes a slot buffer for the tail's two parameter gradients, for
scanprep's ``dparam_bias``, and for the scan's ``dK`` lane tiles. One symbol, three
call sites, and the profiler groups by name, so the row cannot be attributed to a
stage by name alone. It is reported as its own stage rather than charged to one of
the three.
"""

MIXER_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reduce_rows", (SHARED_REDUCE,)),
    ("so3ssd", SO3SSD_KERNELS),
    ("conv1d", ("conv1d_fwd", "conv1d_bwd", "conv1d_reduce_parts")),
    ("scanprep", ("scanprep_fwd", "scanprep_bwd")),
    ("tail", ("mixer_tail_fwd", "mixer_tail_bwd", "rmsnorm")),
    ("proj", GEMM_MARKERS),
    ("glue", GLUE_MARKERS),
)
"""Stage against the kernel-name fragments that select it, first match winning.

``reduce_rows`` is matched first because three stages launch it and none of them may
claim it. This package's own kernels are matched before the GEMM and glue tables,
which are keyed on fragments general enough to catch one otherwise.
"""

MAMBA_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "scan",
        (
            "chunk_cumsum",
            "chunk_state",
            "chunk_scan",
            "bmm_chunk",
            "state_passing",
            "ssd",
            "_chunk_",
        ),
    ),
    ("norm", ("layer_norm", "layernorm", "rms_norm", "rmsnorm")),
    ("conv", ("conv", "cudnn", "implicit", "depthwise", "dgrad", "wgrad")),
    ("act", ("silu", "swish", "sigmoid")),
    ("proj", GEMM_MARKERS),
    ("glue", GLUE_MARKERS),
)
"""Stage against the kernel-name fragments that select it, first match winning.

Mamba2's scan is Triton and its kernels carry their Python function names, so the
scan, the gated norm and the convolution separate by name. The convolution rule is
last of the three named stages because ``conv`` is a fragment of a cuDNN symbol that
a Triton kernel cannot carry, while ``implicit`` and the two gradient suffixes are
how cuDNN names the algorithm rather than the operator.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="acceptance")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--mamba-chunk", type=int, default=256)
    parser.add_argument(
        "--mamba-path",
        choices=["fused", "unfused"],
        default="unfused",
        help="Which Mamba2 forward the baseline arm takes. The fused path needs the "
        "causal_conv1d package and raises without it.",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--rows", type=int, default=80)
    # A CuTe DSL runtime symbol carries its whole traced signature, so a narrow
    # column truncates two specializations of one kernel to the same string.
    parser.add_argument("--name-width", type=int, default=130)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def classify(name: str, table: Sequence[tuple[str, tuple[str, ...]]]) -> str:
    """The stage of a kernel, by name.

    Args:
        name: Kernel, memcpy or memset name as the profiler reports it.
        table: Stage rules, first match winning.

    Returns:
        The stage, or ``other`` if no rule matches.
    """
    lowered = name.lower()
    for stage, needles in table:
        if any(needle.lower() in lowered for needle in needles):
            return stage
    return "other"


def so3ssd_kernel(name: str) -> str:
    """Which SO(3) kernel a symbol is, or the empty string.

    Args:
        name: Kernel symbol.

    Returns:
        The base name from :data:`SO3SSD_KERNELS`, or ``""``.
    """
    for base in SO3SSD_KERNELS:
        if base in name:
            return base
    return ""


def _self_us(row: object) -> float:
    """One row's own device microseconds, under either attribute name."""
    for attribute in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(row, attribute, None)
        if value is not None:
            return float(value)
    raise AttributeError("no self device time on a profiler row")


def direction(name: str) -> str:
    """Which pass a kernel belongs to, read off its name.

    Every kernel this package compiles and every Triton kernel of Mamba2's scan
    carries the direction in its own name. A cuBLAS GEMM does not, and is reported
    as ``-``: the GEMM split comes from the shape-grouped operator rows instead.

    Args:
        name: Kernel symbol.

    Returns:
        ``fwd``, ``bwd``, or ``-``.
    """
    lowered = name.lower()
    if "_bwd" in lowered or "backward" in lowered:
        return "bwd"
    if "_fwd" in lowered or "forward" in lowered:
        return "fwd"
    return "-"


class KernelRow(NamedTuple):
    """One kernel's device cost per iteration.

    Attributes:
        name: Kernel, memcpy or memset name.
        us: Device microseconds per iteration.
        calls: Launches per iteration.
        stage: Stage the name was classified into.
        pass_: Direction the name declares, or ``-``.
    """

    name: str
    us: float
    calls: float
    stage: str
    pass_: str


class GemmRow(NamedTuple):
    """One GEMM operator's device cost per iteration, keyed by its operand shapes.

    Attributes:
        shapes: The operator's input shapes as the profiler recorded them.
        us: Device microseconds per iteration.
        calls: Calls per iteration.
    """

    shapes: str
    us: float
    calls: float


def kernel_rows(
    profiled: profile, iters: int, table: Sequence[tuple[str, tuple[str, ...]]]
) -> list[KernelRow]:
    """Per-kernel device time, descending, with each kernel's stage.

    A user annotation is typed as a device row and carries the time of the kernels
    inside it, so counting one would count that range twice.

    Args:
        profiled: A finished profile.
        iters: Iterations inside it.
        table: Stage rules.

    Returns:
        One row per kernel.
    """
    rows = [
        KernelRow(
            name=row.key,
            us=_self_us(row) / iters,
            calls=row.count / iters,
            stage=classify(row.key, table),
            pass_=direction(row.key),
        )
        for row in profiled.key_averages()
        if row.device_type == DeviceType.CUDA
        and not getattr(row, "is_user_annotation", False)
    ]
    return sorted(rows, key=lambda row: -row.us)


def gemm_rows(profiled: profile, iters: int) -> list[GemmRow]:
    """Per-GEMM device time, keyed by operand shape, descending.

    The two projections launch cuBLAS kernels that no name distinguishes, so the
    split comes from the operator rows instead: one row per distinct operand shape,
    carrying the device time of the kernels correlated with it.

    Args:
        profiled: A finished profile, taken with ``record_shapes``.
        iters: Iterations inside it.

    Returns:
        One row per distinct GEMM shape.
    """
    rows = [
        GemmRow(
            shapes=f"{row.key} {row.input_shapes}",
            us=_self_us(row) / iters,
            calls=row.count / iters,
        )
        for row in profiled.key_averages(group_by_input_shape=True)
        if row.device_type == DeviceType.CPU and row.key in GEMM_OPS
    ]
    return sorted(rows, key=lambda row: -row.us)


def event_us(step: Callable[[], object], iters: int, device: torch.device) -> float:
    """Microseconds per call, from a device event pair around ``iters`` calls.

    Args:
        step: The callable to time.
        iters: Calls inside the interval.
        device: Device to time on.

    Returns:
        Microseconds per call.
    """
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(iters):
        step()
    stop.record()
    torch.cuda.synchronize(device)
    return 1000.0 * start.elapsed_time(stop) / iters


class ArmProfile(NamedTuple):
    """One arm's device-time partition.

    Attributes:
        label: Arm label.
        wall_us: Unprofiled event time of one call.
        bracket_us: Event time of one call inside the profiler.
        kernels: Per-kernel rows, descending.
        gemms: Per-GEMM-shape rows, descending.
    """

    label: str
    wall_us: float
    bracket_us: float
    kernels: tuple[KernelRow, ...]
    gemms: tuple[GemmRow, ...]

    @property
    def device_us(self) -> float:
        """Sum of the per-kernel device time. The partition's total."""
        return sum(row.us for row in self.kernels)

    def by_stage(self) -> dict[str, float]:
        """Device microseconds per stage."""
        out: dict[str, float] = {}
        for row in self.kernels:
            out[row.stage] = out.get(row.stage, 0.0) + row.us
        return out

    def by_stage_pass(self) -> dict[tuple[str, str], float]:
        """Device microseconds per stage and declared direction."""
        out: dict[tuple[str, str], float] = {}
        for row in self.kernels:
            key = (row.stage, row.pass_)
            out[key] = out.get(key, 0.0) + row.us
        return out

    def stage_us(self, stage: str) -> float:
        """Device microseconds of one stage, zero if it launched nothing."""
        return self.by_stage().get(stage, 0.0)


def profile_arm(
    label: str,
    step: Callable[[], object],
    args: argparse.Namespace,
    device: torch.device,
    table: Sequence[tuple[str, tuple[str, ...]]],
) -> ArmProfile:
    """Event-time one arm, then profile it and partition its device time.

    Args:
        label: Arm label.
        step: The callable. One call is one forward and backward.
        args: The command line.
        device: Device to time on.
        table: Stage rules.

    Returns:
        The partition.

    Raises:
        ValueError: If the profile recorded no device work, so the partition is of
            nothing.
    """
    for _ in range(args.warmup):
        step()
    wall = event_us(step, args.profile_iters, device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as profiled:
        bracket = event_us(step, args.profile_iters, device)
    rows = kernel_rows(profiled, args.profile_iters, table)
    if not rows:
        raise ValueError(f"{label}: the profile recorded no device work")
    return ArmProfile(
        label=label,
        wall_us=wall,
        bracket_us=bracket,
        kernels=tuple(rows),
        gemms=tuple(gemm_rows(profiled, args.profile_iters)),
    )


def print_arm(arm: ArmProfile, args: argparse.Namespace) -> None:
    """Print one arm's stage totals, its GEMM split, and its kernel rows."""
    total = arm.device_us
    print()
    print(f"=== {arm.label} ===")
    print(
        f"step {arm.wall_us:,.1f} us unprofiled, {arm.bracket_us:,.1f} us inside the "
        f"profiler; {len(arm.kernels)} kernels sum to {total:,.1f} us of device time, "
        f"{100.0 * total / arm.bracket_us:,.2f}% of the profiled interval"
    )
    gap = arm.bracket_us - total
    print(
        f"the complement is {gap:,.1f} us, {100.0 * gap / arm.bracket_us:,.2f}% of the "
        f"profiled interval: launch gap and host cost the kernels do not cover"
    )
    print()
    per_pass = arm.by_stage_pass()
    print(
        f"{'stage':12s} {'us/iter':>10s} {'share':>8s} {'fwd us':>10s} "
        f"{'bwd us':>10s} {'undecl us':>10s}"
    )
    for stage, us in sorted(arm.by_stage().items(), key=lambda entry: -entry[1]):
        print(
            f"{stage:12s} {us:10,.1f} {100.0 * us / total:7,.2f}% "
            f"{per_pass.get((stage, 'fwd'), 0.0):10,.1f} "
            f"{per_pass.get((stage, 'bwd'), 0.0):10,.1f} "
            f"{per_pass.get((stage, '-'), 0.0):10,.1f}"
        )
    print(f"{'TOTAL':12s} {total:10,.1f} {100.0:7,.2f}%")
    print(
        "a direction column is read off the kernel's own name; a cuBLAS GEMM "
        "declares none and lands under undecl"
    )
    print()
    print("GEMM operator rows, grouped by operand shape:")
    print(f"{'operator and shapes':78s} {'us/iter':>10s} {'calls':>7s}")
    for row in arm.gemms:
        print(f"{row.shapes[:78]:78s} {row.us:10,.1f} {row.calls:7,.1f}")
    print(f"{'GEMM operator sum':78s} {sum(r.us for r in arm.gemms):10,.1f}")
    print(f"{'gemm-class kernel sum (cross-check)':78s} {arm.stage_us('proj'):10,.1f}")
    print()
    width = args.name_width
    print(
        f"{'kernel':{width}s} {'us/iter':>10s} {'share':>8s} {'calls':>7s} "
        f"{'pass':>5s}  stage"
    )
    for row in arm.kernels[: args.rows]:
        print(
            f"{row.name[:width]:{width}s} {row.us:10,.1f} "
            f"{100.0 * row.us / total:7,.2f}% {row.calls:7,.1f} "
            f"{row.pass_:>5s}  {row.stage}"
        )


def print_so3ssd(arm: ArmProfile) -> None:
    """Print the SO(3) scan operator's own kernels, descending."""
    rows = [(so3ssd_kernel(row.name), row) for row in arm.kernels]
    picked = [(base, row) for base, row in rows if base]
    total = sum(row.us for _, row in picked)
    print()
    print("=== so3ssd operator, by launched kernel ===")
    print(f"{'kernel':26s} {'us/iter':>10s} {'share of op':>12s} {'calls':>7s}")
    for base, row in sorted(picked, key=lambda entry: -entry[1].us):
        print(
            f"{base:26s} {row.us:10,.1f} {100.0 * row.us / total:11,.2f}% "
            f"{row.calls:7,.1f}"
        )
    print(f"{'TOTAL':26s} {total:10,.1f} {100.0:11,.2f}%")
    launched = {base for base, _ in picked}
    absent = [base for base in SO3SSD_KERNELS if base not in launched]
    print(f"launched {len(launched)} of {len(SO3SSD_KERNELS)} declared kernels")
    if absent:
        print(f"never launched: {', '.join(absent)}")
    shared = arm.stage_us(SHARED_REDUCE)
    print(
        f"operator time is bounded, not exact: {total:,.1f} us charging "
        f"{SHARED_REDUCE} nowhere, {total + shared:,.1f} us charging all "
        f"{shared:,.1f} us of it to the operator. Three stages launch that one "
        f"symbol and the profiler groups by name, so no name-based split exists."
    )


def amdahl(
    baseline_us: float,
    mixer_us: float,
    operator_us: float,
    targets: Sequence[float],
) -> None:
    """Print the end-to-end ratio at each hypothetical operator time.

    Everything but the operator is held at what it measured. The ratio is the
    baseline's time over the mixer's, so above one means the mixer is faster.

    Args:
        baseline_us: The Mamba2 layer's measured step time.
        mixer_us: The SO(3) mixer's measured step time.
        operator_us: The scan operator's measured share of ``mixer_us``.
        targets: Hypothetical operator times, descending.
    """
    rest = mixer_us - operator_us
    print()
    print("=== Amdahl: the operator alone moves, everything else holds ===")
    print(
        f"baseline {baseline_us:,.1f} us   mixer {mixer_us:,.1f} us   "
        f"operator {operator_us:,.1f} us   rest {rest:,.1f} us "
        f"({100.0 * rest / mixer_us:,.2f}% of the mixer)"
    )
    print()
    print(
        f"{'operator us':>12s} {'mixer us':>12s} {'ratio':>8s} {'vs 2.0x':>9s} "
        f"{'op speedup':>11s}"
    )
    for target in targets:
        total = rest + target
        ratio = baseline_us / total
        print(
            f"{target:12,.0f} {total:12,.1f} {ratio:7,.3f}x "
            f"{ratio - 2.0:+8,.3f} {operator_us / target:10,.2f}x"
        )
    ceiling = baseline_us / rest
    print(
        f"{0:12,.0f} {rest:12,.1f} {ceiling:7,.3f}x {ceiling - 2.0:+8,.3f} {'inf':>10s}"
    )
    print()
    print(
        f"free-operator ceiling {ceiling:,.3f}x: the hard upper bound on operator "
        f"work alone"
    )
    for bar in (2.0, 2.5):
        need = baseline_us / bar - rest
        if need > 0.0:
            print(
                f"{bar:,.1f}x needs the operator at {need:,.1f} us, a "
                f"{operator_us / need:,.2f}x speedup on it"
            )
        else:
            print(
                f"{bar:,.1f}x is unreachable by operator work: it needs the mixer "
                f"under {baseline_us / bar:,.1f} us and the rest alone is "
                f"{rest:,.1f} us, over by {rest - baseline_us / bar:,.1f} us"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Measure both arms, partition both, and print the Amdahl table.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        SystemExit: If the fused Mamba2 path is asked for and cannot run.
    """
    from slinoss.mixer import SLinOSSMixer

    args = parse_args(argv)
    device = require_cuda(args.device)
    shape = shape_by_name(args.shape)
    dtype = DTYPES[args.dtype]
    mem_eff = args.mamba_path == "fused"
    mchunk = mamba_chunk(shape, args.mamba_chunk)
    config = layer_config(shape, groups=args.groups)

    print(f"device {device_ordinal(device)}  {device_info(device_ordinal(device))}")
    print(shape.describe())
    print(
        f"iso axes: batch {shape.bsz}  seq {shape.seq}  d_model {d_model_of(shape)}  "
        f"heads {shape.heads}  headdim {shape.rows}  dstate {shape.d_state}  "
        f"groups {args.groups}  dtype {args.dtype}"
    )
    print(
        f"tiling: mamba2 chunk_size={mchunk}  so3ssd L={shape.chunk}  "
        f"mamba2 path={args.mamba_path}"
    )
    blocker = fused_path_blocker()
    print(f"fused Mamba2 path: {'available' if blocker is None else blocker}")
    theirs_p, ours_p = parameter_counts(shape, args.groups)
    print(
        f"parameters: {theirs_p.label} {theirs_p.elements:,}  "
        f"{ours_p.label} {ours_p.elements:,}  "
        f"({100.0 * (ours_p.elements - theirs_p.elements) / theirs_p.elements:+,.2f}%)"
    )

    a_label = "mamba2"
    b_label = "mixer"
    theirs = make_mamba_block(
        shape,
        args.groups,
        device=device,
        dtype=dtype,
        mem_eff_path=mem_eff,
        chunk=mchunk,
    )
    ours = SLinOSSMixer(config, device=device, dtype=dtype)
    x, dy = block_stream(shape, device, dtype=dtype, requires_grad=True)
    their_x = x.detach().clone().requires_grad_(True)
    our_x = x.detach().clone().requires_grad_(True)
    their_step = block_runner(theirs, their_x, dy, grads=True, prefix=a_label)
    our_step = block_runner(ours, our_x, dy, grads=True, prefix=b_label)

    print()
    print("gradient requirements:")
    for label, module, leaf in ((a_label, theirs, their_x), (b_label, ours, our_x)):
        params = [p for p in module.parameters() if p.requires_grad]
        print(
            f"  {label}: input requires_grad={leaf.requires_grad}, "
            f"{len(params)} parameters require grad, "
            f"{sum(p.numel() for p in params):,} elements, "
            f"dtypes {sorted({str(p.dtype) for p in module.parameters()})}"
        )

    reset_memory_peaks(device)
    out = measure_paired(
        a_label,
        their_step,
        b_label,
        our_step,
        label=f"mamba2 layer vs slinoss mixer {shape.name} step paired",
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    print()
    print("=== paired region timing, one loop, order swapped each iteration ===")
    print(out.comparison.verdict())
    print(f"resolves={out.comparison.resolves}")
    print()
    print(f"{'region':22s} {'median us':>12s} {'spread %':>10s} {'share %':>9s}")
    for timing in out.timed.regions:
        print(
            f"{timing.label:22s} {float(timing.spread.median_duration_us):12,.1f} "
            f"{float(timing.spread.spread_pct):10,.3f} {float(timing.share_pct):9,.2f}"
        )
    a_total = float(out.timed.region(a_label).spread.median_duration_us)
    b_total = float(out.timed.region(b_label).spread.median_duration_us)
    print()
    print(f"end-to-end ratio {a_total / b_total:,.4f}x, region timers, one hold")
    print(memory_peaks("paired", device))

    probes: dict[str, Any] = {}
    for label, module, leaf, step in (
        (a_label, theirs, their_x, their_step),
        (b_label, ours, our_x, our_step),
    ):
        probe = SavedTensorProbe()
        with probe:
            measure(step, label=f"{label} saved", iters=1, warmup=0, device=device)
        # The leaf and the parameters are the declared inputs, so whatever is left
        # is an activation the graph holds rather than a tensor the caller owned.
        probes[label] = probe.report(label, (leaf, *module.parameters()))

    arms: dict[str, ArmProfile] = {}
    for label, step, table in (
        (a_label, their_step, MAMBA_STAGES),
        (b_label, our_step, MIXER_STAGES),
    ):
        arms[label] = profile_arm(label, step, args, device, table)
        print_arm(arms[label], args)

    print_so3ssd(arms[b_label])

    print()
    print("=== autograd saved-tensor traffic ===")
    print(
        f"{'arm':10s} {'storages':>9s} {'saves':>7s} {'saved MiB':>11s} "
        f"{'input MiB':>11s} {'derived MiB':>12s}"
    )
    for label, saved in probes.items():
        mib = 1024.0 * 1024.0
        print(
            f"{label:10s} {int(saved.storage_count):9,d} "
            f"{int(saved.save_event_count):7,d} {saved.saved_bytes / mib:11,.2f} "
            f"{saved.input_bytes / mib:11,.2f} {saved.derived_bytes / mib:12,.2f}"
        )

    mixer_arm, mamba_arm = arms[b_label], arms[a_label]
    operator_us = mixer_arm.stage_us("so3ssd")
    # The Amdahl arithmetic runs on the region-timed totals, which is what the
    # ratio is taken over, with the operator's share taken from the profiled
    # partition and rescaled to the unprofiled total. A share measured inside the
    # profiler and an absolute measured outside it are two different intervals.
    scale = b_total / mixer_arm.device_us
    print()
    print(
        f"operator device time {operator_us:,.1f} us of {mixer_arm.device_us:,.1f} us "
        f"profiled kernel sum, {100.0 * operator_us / mixer_arm.device_us:,.2f}%; "
        f"scaled onto the {b_total:,.1f} us region total by {scale:,.4f} gives "
        f"{operator_us * scale:,.1f} us"
    )
    amdahl(
        a_total,
        b_total,
        operator_us * scale,
        (3000.0, 2385.0, 2000.0, 1500.0, 1200.0),
    )
    print()
    print("same table on the profiled device-time partition, unscaled:")
    amdahl(
        mamba_arm.device_us,
        mixer_arm.device_us,
        operator_us,
        (3000.0, 2385.0, 2000.0, 1500.0, 1200.0),
    )
    # The ceiling is baseline / (mixer - operator), so charging the ambiguous
    # reduce_rows to the operator raises it. Both ends are stated because the
    # symbol has three call sites and the profiler cannot split them.
    shared = mixer_arm.stage_us(SHARED_REDUCE) * scale
    if shared > 0.0:
        low = a_total / (b_total - operator_us * scale)
        high = a_total / (b_total - operator_us * scale - shared)
        print()
        print(
            f"free-operator ceiling bracket from the ambiguous {SHARED_REDUCE} "
            f"({shared:,.1f} us scaled): {low:,.3f}x charging it to the rest, "
            f"{high:,.3f}x charging it to the operator"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
