"""Counter profile of ``start_passing_bwd_kernel`` against the pair it replaces.

The fusion is judged on the pair's total time and the pair's total bytes, never on
the fused kernel's own percentage of its own floor: a kernel can sit at its floor
and still be the slower of two ways to compute the same thing. So one driver runs
both arms under one floor fit, one warmup policy, and one capture window.
``--arm fused`` launches :func:`start_passing_backward`; ``--arm pair`` launches
:func:`chunk_start_backward` and then :func:`state_passing_backward`, and NCU is
narrowed to both kernels so the printed blocks add up to the arm.

Two modes in one file, for the reason ``scripts/perf/profile_chunk_start_bwd.py``
has two: one kernel has one warmup policy and one capture window, and a second file
would only let them drift. The default drives; ``--window`` is the process NCU
attaches to.

    python3 scripts/perf/profile_start_passing_bwd.py --arm fused --shape standard
    python3 scripts/perf/profile_start_passing_bwd.py --arm pair --rows 64 \
        --lanes 80 --heads 18 --groups 1

Two byte counts are printed. ``issued`` charges every per-head operand once per lane
band and the readout band once per head, which is what the blocks request.
``unique`` charges each of them once, which is what DRAM owes if L2 holds them
across the bands and heads that share them. The measured read count between the two
is the reuse the launch order reaches.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence

import torch

from slinoss.ops.so3ssd.cute.bwd.chunk_start import chunk_start_backward
from slinoss.ops.so3ssd.cute.bwd.start_passing import SPLIT, start_passing_backward
from slinoss.ops.so3ssd.cute.bwd.state_passing import state_passing_backward
from slinoss.ops.so3ssd.cute.common import WARPS
from slinoss.perf.capture import profiler_window
from slinoss.perf.ceiling import dram_floor_verdict, dram_time_floor
from slinoss.perf.device import (
    compute_apps_query,
    device_ordinal,
    require_cuda,
    smi_selector,
)
from slinoss.perf.ncu import (
    NCU_TABLES,
    SPILL_TABLE,
    NcuPass,
    SpillCounters,
    kernel_counters,
    run_ncu,
    spill_counters,
)
from slinoss.perf.timing import measure, on_device
from slinoss.perf.units import Bytes, Count
from slinoss.perf.workload import SHAPE_NAMES, OpShape, make_inputs, shape_by_name

KERNELS = {
    "fused": "start_passing_bwd",
    "pair": "chunk_start_bwd|state_passing_bwd",
}
"""Regex NCU narrows to, per arm. The mangled symbol carries the constexpr suffix."""

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

GEOMETRY = {
    "bsz": "Batch, B.",
    "heads": "Heads, H.",
    "seq": "Tokens, T. Sets the chunk count with the chunk length.",
    "rows": "Rows per head, P.",
    "lanes": "Lanes per head, N. The state width is 3N.",
    "chunk": "Chunk length, L.",
}
"""Overridable :class:`slinoss.perf.workload.OpShape` fields, name to help text.

Every field is here, unlike the GEMM-only drivers: both arms run once per step, so
``bsz`` and ``seq`` scale one launch rather than the launch count."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(KERNELS), default="fused")
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="standard")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    for field, helptext in GEOMETRY.items():
        parser.add_argument(f"--{field}", type=int, default=None, help=helptext)
    parser.add_argument(
        "--groups",
        type=int,
        default=None,
        help="Groups, G. Divides H. Defaults to H, which is fold one.",
    )
    parser.add_argument(
        "--span",
        type=int,
        default=SPLIT,
        help="Lane band width the fused arm splits 3N into. Ignored by the pair.",
    )
    parser.add_argument(
        "--warps",
        type=int,
        default=WARPS,
        help="Warps per block of the fused kernel. Warps past the first four go to "
        "the tile's N mode, which halves both accumulators at unchanged shared "
        "bytes. Ignored by the pair.",
    )
    parser.add_argument(
        "--resident",
        type=int,
        default=None,
        help="Blocks per SM the fused launch bound asks for. Defaults to "
        "RESIDENT_MAX. The cap it puts on the register file is what decides "
        "whether the residency is reached, so it is priced rather than assumed.",
    )
    parser.add_argument(
        "--seed-state",
        action="store_true",
        help="Supply a final-state cotangent, the has_dstate variant.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Launches inside the capture window, and the divisor that puts the "
        "counter sums on a per-launch footing.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--event-iters", type=int, default=30)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument(
        "--window",
        action="store_true",
        help="Run as the profiler's target: warm up, then launch inside the "
        "capture window. Emits nothing.",
    )
    return parser.parse_args(argv)


def requested_shape(args: argparse.Namespace) -> OpShape:
    """The named shape, with any geometry override applied.

    Args:
        args: The parsed command line.

    Returns:
        The shape. Unchanged, and under its own name, when no override was given.
        ``--groups``, ``--span`` and the arm are not fields of the shape, so they
        rename it and nothing else.

    Raises:
        KeyError: If ``--shape`` is not one of
            :data:`slinoss.perf.workload.SHAPES`.
    """
    shape = shape_by_name(args.shape)
    given = {f: getattr(args, f) for f in GEOMETRY if getattr(args, f) is not None}
    suffix = "".join(f"-{f}{v}" for f, v in sorted(given.items()))
    if args.groups is not None:
        suffix += f"-g{args.groups}"
    if args.arm == "fused" and args.span != SPLIT:
        suffix += f"-span{args.span}"
    if args.arm == "fused" and args.warps != WARPS:
        suffix += f"-w{args.warps}"
    if args.arm == "fused" and args.resident is not None:
        suffix += f"-r{args.resident}"
    if not suffix:
        return shape
    return dataclasses.replace(shape, name=f"{shape.name}{suffix}", **given)


def build_runner(
    shape: OpShape,
    groups: int,
    device: torch.device,
    dtype: torch.dtype,
    arm: str,
    span: int,
    warps: int,
    resident: int | None,
    seed_state: bool,
) -> Callable[[], None]:
    """Allocate one input set and return the callable that launches the arm.

    ``B`` and ``C`` come out of :func:`slinoss.perf.workload.make_inputs` at ``G ==
    H``. A fold above one needs fewer groups than heads, so the readout band is
    reallocated here at ``(B,G,T,3N)`` rather than sliced: a slice of the head mode
    is a view whose pitch is the parent's, and the budget the fold is being measured
    for is read off the layout.

    ``cquat`` is normalized and ``cscale`` is a sigmoid, so both satisfy I1. Neither
    is read off a forward: this driver measures traffic and occupancy, and the
    parity file is where the values have to be the pipeline's.

    Args:
        shape: The problem size.
        groups: ``G``. Divides ``shape.heads``.
        device: Where to allocate.
        dtype: Activation dtype.
        arm: ``fused`` or ``pair``.
        span: Lane band width for the fused arm.
        warps: Block width for the fused arm.
        resident: Launch bound for the fused arm, or None for its default.
        seed_state: Whether to supply a final-state cotangent.

    Returns:
        The callable, allocating its outputs per call as the host entries do.

    Raises:
        ValueError: If ``groups`` does not divide ``shape.heads``.
    """
    if shape.heads % groups:
        raise ValueError(f"groups {groups} must divide heads {shape.heads}")
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=False)
    gen = torch.Generator(device=device).manual_seed(1)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(*size, dtype=torch.float32, device=device, generator=gen)

    if groups == shape.heads:
        vecc = inputs.C
    else:
        vecc = torch.randn(
            shape.bsz,
            groups,
            shape.seq,
            shape.d_state,
            dtype=dtype,
            device=device,
            generator=gen,
        )

    chunks = -(-shape.seq // shape.chunk)
    cquat = randn(shape.bsz, shape.heads, chunks, 4)
    cquat = cquat / cquat.norm(dim=-1, keepdim=True)
    cscale = randn(shape.bsz, shape.heads, chunks).sigmoid()
    dstate = randn(shape.bsz, shape.heads, shape.rows, shape.d_state)
    seed = dstate if seed_state else None

    def fused() -> None:
        start_passing_backward(
            inputs.dy,
            inputs.trans,
            vecc,
            cquat,
            cscale,
            shape.chunk,
            seed,
            span=span,
            warps=warps,
            resident=resident,
        )

    def pair() -> None:
        dzstart = chunk_start_backward(inputs.dy, inputs.trans, vecc, shape.chunk)
        state_passing_backward(dzstart, cquat, cscale, seed)

    return fused if arm == "fused" else pair


def target_argv(args: argparse.Namespace) -> list[str]:
    """The argv NCU attaches to. Carries every override forward."""
    argv = [
        sys.executable,
        __file__,
        "--window",
        "--arm",
        args.arm,
        "--shape",
        args.shape,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--span",
        str(args.span),
        "--warps",
        str(args.warps),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]
    for field in GEOMETRY:
        value = getattr(args, field)
        if value is not None:
            argv += [f"--{field}", str(value)]
    if args.groups is not None:
        argv += ["--groups", str(args.groups)]
    if args.resident is not None:
        argv += ["--resident", str(args.resident)]
    if args.seed_state:
        argv += ["--seed-state"]
    return argv


def analytic_bytes(
    shape: OpShape, groups: int, span: int, itemsize: int, arm: str
) -> tuple[Bytes, Bytes]:
    """Traffic one call of the arm moves, operand by operand.

    The fused arm reads every per-head operand once per lane band, because a band
    holds a slice of the state and needs the whole cotangent of ``y`` to form it.
    The pair reads each once and pays a full ``(B,H,C,P,3N)`` round trip instead.

    Args:
        shape: The problem size.
        groups: ``G``.
        span: Lane band width.
        itemsize: Bytes per activation element.
        arm: ``fused`` or ``pair``.

    Returns:
        ``(issued, unique)``. The first charges a shared operand once per block that
        requests it, the second once per distinct buffer.
    """
    chunks = -(-shape.seq // shape.chunk)
    tokens = shape.bsz * shape.seq
    lead = shape.bsz * shape.heads
    dy = tokens * shape.heads * shape.rows * itemsize
    trans = tokens * shape.heads * 4 * 4
    readout = tokens * shape.d_state * itemsize
    plane = lead * chunks * shape.rows * shape.d_state * 4
    state = lead * shape.rows * shape.d_state * 4
    transition = lead * chunks * 4 * 4 + lead * chunks * 4
    if arm == "fused":
        bands = -(-shape.d_state // span)
        per_head = dy + trans + transition
        rest = plane + state
        return (
            Bytes(bands * per_head + readout * shape.heads + rest),
            Bytes(per_head + readout * groups + rest),
        )
    common = dy + trans + 3 * plane + state + transition
    return (
        Bytes(common + readout * shape.heads),
        Bytes(common + readout * groups),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, profile, score, and print.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If ``--iters`` is not positive, or if a counter table returned
            no value for a metric, which means the name is wrong for this driver.
    """
    args = parse_args(argv)
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    device = require_cuda(args.device)
    shape = requested_shape(args)
    dtype = DTYPES[args.dtype]
    groups = shape.heads if args.groups is None else args.groups

    if args.window:
        runner = build_runner(
            shape,
            groups,
            device,
            dtype,
            args.arm,
            args.span,
            args.warps,
            args.resident,
            args.seed_state,
        )
        with on_device(device):
            for _ in range(args.warmup):
                runner()
            with profiler_window(device):
                for _ in range(args.iters):
                    runner()
        return 0

    ordinal = device_ordinal(device)
    before = compute_apps_query(smi_selector(ordinal))
    runner = build_runner(
        shape,
        groups,
        device,
        dtype,
        args.arm,
        args.span,
        args.warps,
        args.resident,
        args.seed_state,
    )
    timed = measure(
        runner,
        label=f"{args.arm} {shape.name}",
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
    )
    floor = dram_time_floor(device)
    issued, unique = analytic_bytes(
        shape, groups, args.span, torch.finfo(dtype).bits // 8, args.arm
    )
    del runner

    narrow = ("--kernel-name", f"regex:{KERNELS[args.arm]}")
    passes: list[NcuPass] = []
    spills: tuple[SpillCounters, ...] = ()
    for table in (*NCU_TABLES, SPILL_TABLE):
        one = run_ncu(table, target_argv(args), ncu=args.ncu, extra=narrow)
        if one.missing_metrics:
            raise ValueError(
                f"ncu table {table.name!r} returned no value for "
                f"{list(one.missing_metrics)}; the metric names are wrong for "
                f"this driver"
            )
        if table is SPILL_TABLE:
            spills = spill_counters(one)
        else:
            passes.append(one)
    after = compute_apps_query(smi_selector(ordinal))

    print(
        f"arm          {args.arm} span {args.span} warps {args.warps} "
        f"resident {args.resident}"
    )
    print(f"shape        {shape.describe()}")
    print(f"groups       {groups} fold {shape.heads // groups}")
    print(f"dtype        {args.dtype}")
    print(f"clocks       {timed.clocks}")
    print(f"smi before   {before}")
    print(f"smi after    {after}")
    print(
        f"floor        c={floor.fixed_duration_us:.3f} us "
        f"B={floor.asymptotic_gbs:.2f} GB/s "
        f"residual={floor.max_residual_pct:.2f}% l2={floor.l2_bytes} B"
    )
    print(
        f"model        issued {issued / 1e6:.2f} MB {floor.floor_us(issued):.1f} us  "
        f"unique {unique / 1e6:.2f} MB {floor.floor_us(unique):.1f} us"
    )
    print(
        f"call wall    med={timed.total.median_duration_us:.1f} us "
        f"min={timed.total.min_duration_us:.1f} "
        f"max={timed.total.max_duration_us:.1f} "
        f"resolution={timed.total.resolution_pct:.2f}%"
    )

    arm_us = 0.0
    arm_bytes = 0
    for counters in kernel_counters(passes):
        launches = counters.launch_count
        moved = Bytes(counters.dram_read_bytes + counters.dram_write_bytes)
        verdict = dram_floor_verdict(
            counters.kernel,
            moved_bytes=moved,
            launch_count=Count(launches),
            duration_us=counters.duration_us,
            floor=floor,
        )
        per_launch = Bytes(moved // launches)
        arm_us += counters.duration_us / launches
        arm_bytes += per_launch
        print(f"kernel       {counters.kernel}")
        print(f"  launches   {launches}")
        print(f"  us/launch  {counters.duration_us / launches:.1f}")
        print(f"  pass spread {counters.pass_duration_spread_pct:.2f}%")
        print(f"  MB/launch  {per_launch / 1e6:.2f}")
        print(f"  read MB    {counters.dram_read_bytes / launches / 1e6:.2f}")
        print(f"  write MB   {counters.dram_write_bytes / launches / 1e6:.2f}")
        print(f"  GB/s       {counters.achieved_gbs:.1f}")
        print(f"  dram_pct   {counters.dram_pct:.1f}%")
        print(f"  floor      {floor.floor_us(per_launch):.1f} us/launch")
        print(
            f"  class      {verdict.achieved_pct:.1f}% of {verdict.required_pct:.0f}%"
        )
        print(f"  regs       {counters.register_per_thread_count}")
        print(f"  smem       {counters.smem_bytes} B/block")
        print(
            f"  blocks     {counters.block_count} x {counters.thread_per_block_count}"
        )
        print(
            f"  occ        theo={counters.theoretical_occupancy_pct:.1f}% "
            f"achieved={counters.achieved_occupancy_pct:.1f}%"
        )
        print(f"  issue      {counters.issue_active_pct:.2f}%")
        print(
            f"  stall      {counters.dominant_stall} "
            f"{counters.dominant_stall_pct:.1f}% "
            f"long_scoreboard={counters.stall_long_scoreboard_pct:.1f}% "
            f"short_scoreboard={counters.stall_short_scoreboard_pct:.1f}% "
            f"lg_throttle={counters.stall_lg_throttle_pct:.1f}% "
            f"mio_throttle={counters.stall_mio_throttle_pct:.1f}% "
            f"barrier={counters.stall_barrier_pct:.1f}% "
            f"not_selected={counters.stall_not_selected_pct:.1f}%"
        )
        print(
            f"  conflicts  ld={counters.shared_load_conflict_count} "
            f"st={counters.shared_store_conflict_count} "
            f"wavefronts={counters.wavefront_count} "
            f"per_wavefront={counters.conflict_per_wavefront_ratio:.4f}"
        )
        print(
            f"  sol        sm={counters.sm_pct:.1f}% mem={counters.memory_pct:.1f}% "
            f"l1tex={counters.l1tex_pct:.1f}% l2={counters.l2_pct:.1f}% "
            f"tensor={counters.tensor_pipe_pct:.1f}%"
        )

    # The arm is what the two figures below judge: a fusion is worth its complexity
    # only if the sum over the arm's kernels falls, whatever any one of them does
    # against its own floor.
    print(f"arm total    {arm_us:.1f} us {arm_bytes / 1e6:.2f} MB")
    print(f"arm floor    {floor.floor_us(Bytes(arm_bytes)):.1f} us")

    for record in spills:
        launches = record.launch_count
        print(f"spill        {record.kernel}")
        print(f"  ld sectors {record.local_load_sector_count / launches:.0f}/launch")
        print(f"  st sectors {record.local_store_sector_count / launches:.0f}/launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
