"""Counter profile of ``chunk_vector_bwd_kernel``, scored against its own floor.

The so3ssd CuTe backend registers the reference backward, so no operator any other
driver in this directory runs launches this kernel, and the class audit judges only
what a capture contained. This driver reaches it directly.

Two modes in one file, for the reason
``scripts/perf/profile_chunk_input_bwd.py`` has two: one kernel has one warmup
policy and one capture window, and a second file would only let them drift. The
default drives; ``--window`` is the process NCU attaches to.

``--rows``, ``--lanes``, ``--heads`` and ``--chunk`` override the named shape's
per-block geometry. ``--groups`` sets ``G``, and the fold ``H // G`` with it, which
is what decides whether the float32 readout sum is allocated at all. ``--splits``
sets how many blocks that fold is shared over, which trades the in-block fold's
local traffic against a workspace partial. ``--warps`` sets the block width, which
is the lever on how many warps are resident per scheduler. The overrides rename the
shape, so a figure taken under one cannot be read as a figure at the name it
started from.

    python3 scripts/perf/profile_chunk_vector_bwd.py --shape standard
    python3 scripts/perf/profile_chunk_vector_bwd.py --rows 64 --lanes 80 --heads 18
    python3 scripts/perf/profile_chunk_vector_bwd.py --rows 64 --lanes 80 --heads 18 \
        --groups 1 --splits 6
    python3 scripts/perf/profile_chunk_vector_bwd.py --shape standard --warps 8
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence

import torch

from slinoss.ops.so3ssd.cute.bwd.chunk_vector import (
    chunk_vector_backward,
    partial_bytes,
    vector_splits,
)
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
    NcuTable,
    SpillCounters,
    kernel_counters,
    run_ncu,
    spill_counters,
)
from slinoss.perf.timing import measure, on_device
from slinoss.perf.units import Bytes, Count
from slinoss.perf.workload import SHAPE_NAMES, OpShape, make_inputs, shape_by_name

KERNEL = "chunk_vector_bwd|vector_reduce_kernel|reduce_rows_kernel"
"""Regex NCU narrows to. The mangled symbol carries the constexpr suffix.

The operator is more than one launch: the main kernel writes float32 workspace
partials, ``slinoss._reduce.reduce_rows_kernel`` sums the lane-tile slots of
``dtrans`` and ``dK``, and
``slinoss.ops.so3ssd.cute.bwd.chunk_vector.vector_reduce_kernel`` sums the
head-shard slots of ``dB``, ``dC`` and the carry. Narrowing to the first alone
credits a candidate that moves traffic into the others.
"""

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

GEOMETRY = {
    "rows": "Rows per head, P.",
    "lanes": "Lanes per head, N. The state width is 3N.",
    "heads": "Heads, H.",
    "chunk": "Chunk length, L.",
}
"""Overridable :class:`slinoss.perf.workload.OpShape` fields, name to help text.

``bsz`` and ``seq`` are absent: they scale the launch count and not the block, and
a figure per launch is what the class is read from."""

LOCAL_TABLE = NcuTable(
    "local",
    (
        "gpu__time_duration.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum",
    ),
)
"""Where the spill lands after L1.

:data:`slinoss.perf.ncu.SPILL_TABLE` says a spill happened. This says how much of
it L1 absorbed, which is the difference between a spill that costs registers and
one that costs device bandwidth."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--splits",
        type=int,
        default=None,
        help="Partial depth of the head sum, S. Divides the fold H // G. Defaults "
        "to the operator's own choice.",
    )
    parser.add_argument(
        "--warps",
        type=int,
        default=None,
        help=f"Block width in warps. Defaults to {WARPS}, the operator's own.",
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
        ``--groups``, ``--splits`` and ``--warps`` are not fields of the shape, so
        they rename it and nothing else.

    Raises:
        KeyError: If ``--shape`` is not one of
            :data:`slinoss.perf.workload.SHAPES`.
    """
    shape = shape_by_name(args.shape)
    given = {f: getattr(args, f) for f in GEOMETRY if getattr(args, f) is not None}
    suffix = "".join(f"-{f}{v}" for f, v in sorted(given.items()))
    if args.groups is not None:
        suffix += f"-g{args.groups}"
    if args.splits is not None:
        suffix += f"-s{args.splits}"
    if args.warps is not None:
        suffix += f"-w{args.warps}"
    if not suffix:
        return shape
    return dataclasses.replace(shape, name=f"{shape.name}{suffix}", **given)


def build_runner(
    shape: OpShape,
    groups: int,
    device: torch.device,
    dtype: torch.dtype,
    splits: int | None = None,
    warps: int = WARPS,
) -> Callable[[], None]:
    """Allocate one input set and return the callable that launches the kernel.

    ``dinc`` and ``zstart`` are the state-passing backward's outputs and ``dlogp``,
    ``dchunk_rot`` and ``dchunk_scale`` the chunk-input backward's. Their values do
    not reach a counter, so they are drawn from a generator rather than
    rematerialized through the two stages that produce them.

    ``B`` and ``C`` come out of :func:`slinoss.perf.workload.make_inputs` at ``G ==
    H``. A fold above one needs fewer groups than heads, so they are reallocated
    here at ``(B,G,T,3N)`` rather than sliced: a slice of the head mode is a view
    whose pitch is the parent's, and the budget the fold is being measured for is
    read off the layout.

    Args:
        shape: The problem size.
        groups: ``G``. Divides ``shape.heads``.
        device: Where to allocate.
        dtype: Activation dtype.
        splits: Partial depth of the head sum. ``None`` leaves the choice to the
            operator, which is the configuration a step runs.
        warps: Block width in warps.

    Returns:
        The callable, allocating its five outputs per call as the host entry does.

    Raises:
        ValueError: If ``groups`` does not divide ``shape.heads``, or if ``warps``
            is not a width the tiling admits.
    """
    if shape.heads % groups:
        raise ValueError(f"groups {groups} must divide heads {shape.heads}")
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=False)
    chunks = -(-shape.seq // shape.chunk)
    gen = torch.Generator(device=device).manual_seed(1)

    def randn(*size: int, dt: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    state = (shape.bsz, shape.heads, chunks, shape.rows, shape.d_state)
    dinc = randn(*state)
    zstart = randn(*state)
    dlogp = randn(shape.bsz, shape.heads, chunks, shape.chunk)
    dchunk_rot = randn(shape.bsz, shape.heads, chunks, 3, 3)
    dchunk_scale = randn(shape.bsz, shape.heads, chunks)
    if groups == shape.heads:
        vecb, vecc = inputs.B, inputs.C
    else:
        band = (shape.bsz, groups, shape.seq, shape.d_state)
        vecb = randn(*band, dt=dtype)
        vecc = randn(*band, dt=dtype)

    def run() -> None:
        chunk_vector_backward(
            inputs.dy,
            inputs.U,
            inputs.trans,
            inputs.K,
            vecb,
            vecc,
            dinc,
            zstart,
            dlogp,
            dchunk_rot,
            dchunk_scale,
            shape.chunk,
            splits=splits,
            warps=warps,
        )

    return run


def target_argv(args: argparse.Namespace) -> list[str]:
    """The argv NCU attaches to. Carries every geometry override forward."""
    argv = [
        sys.executable,
        __file__,
        "--window",
        "--shape",
        args.shape,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
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
    if args.splits is not None:
        argv += ["--splits", str(args.splits)]
    if args.warps is not None:
        argv += ["--warps", str(args.warps)]
    return argv


def local_sectors(one: NcuPass) -> dict[tuple[str, str], float]:
    """Sum the local-memory lookup metrics per kernel over the profiled launches.

    Args:
        one: The parsed ``local`` pass.

    Returns:
        Kernel and metric name to summed sector count, empty for a metric this
        driver does not carry. Kept separate by kernel because the pass covers
        both launches and one total would hide spill moved between them.
    """
    out: dict[tuple[str, str], float] = {}
    for invocation in one.invocations:
        for metric, value in invocation.values.items():
            if metric.endswith(".sum") and "local" in metric:
                key = (invocation.kernel, metric)
                out[key] = out.get(key, 0.0) + value
    return out


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
    warps = WARPS if args.warps is None else args.warps

    if args.window:
        runner = build_runner(shape, groups, device, dtype, args.splits, warps)
        with on_device(device):
            for _ in range(args.warmup):
                runner()
            with profiler_window(device):
                for _ in range(args.iters):
                    runner()
        return 0

    ordinal = device_ordinal(device)
    before = compute_apps_query(smi_selector(ordinal))
    runner = build_runner(shape, groups, device, dtype, args.splits, warps)
    label = f"chunk_vector_bwd {shape.name}"
    timed = measure(
        runner,
        label=label,
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
    )
    floor = dram_time_floor(device)
    del runner

    tables = (*NCU_TABLES, SPILL_TABLE, LOCAL_TABLE)
    narrow = ("--kernel-name", f"regex:{KERNEL}")
    passes: list[NcuPass] = []
    spills: tuple[SpillCounters, ...] = ()
    local: dict[tuple[str, str], float] = {}
    for table in tables:
        one = run_ncu(table, target_argv(args), ncu=args.ncu, extra=narrow)
        if one.missing_metrics:
            raise ValueError(
                f"ncu table {table.name!r} returned no value for "
                f"{list(one.missing_metrics)}; the metric names are wrong for "
                f"this driver"
            )
        if table is SPILL_TABLE:
            spills = spill_counters(one)
        elif table is LOCAL_TABLE:
            local = local_sectors(one)
        else:
            passes.append(one)
    after = compute_apps_query(smi_selector(ordinal))

    fold = shape.heads // groups
    shards = vector_splits(fold, args.splits)
    workspace = partial_bytes(
        shape.bsz,
        groups,
        shape.seq,
        -(-shape.seq // shape.chunk),
        shape.d_state,
        shards,
        DTYPES[args.dtype].itemsize,
    )
    print(f"shape        {shape.describe()}")
    print(f"groups       {groups} fold {fold}")
    print(f"splits       {shards} in-block fold {fold // shards}")
    print(f"workspace    {workspace / 1e6:.2f} MB of head-sum partials")
    print(f"width        {warps} warps, {warps * 32} threads")
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
        f"call wall    med={timed.total.median_duration_us:.1f} us "
        f"min={timed.total.min_duration_us:.1f} "
        f"max={timed.total.max_duration_us:.1f} "
        f"resolution={timed.total.resolution_pct:.2f}%"
    )

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
        print(f"kernel       {counters.kernel}")
        print(f"  launches   {launches}")
        print(f"  us/launch  {counters.duration_us / launches:.1f}")
        print(f"  pass spread {counters.pass_duration_spread_pct:.2f}%")
        print(f"  MB/launch  {per_launch / 1e6:.2f}")
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

    for record in spills:
        launches = record.launch_count
        print(f"spill        {record.kernel}")
        print(f"  ld sectors {record.local_load_sector_count / launches:.0f}/launch")
        print(f"  st sectors {record.local_store_sector_count / launches:.0f}/launch")
        print(
            f"  ld MB      {record.local_load_sector_count * 32 / launches / 1e6:.2f}"
        )
        print(
            f"  st MB      {record.local_store_sector_count * 32 / launches / 1e6:.2f}"
        )
    for (kernel, metric), value in sorted(local.items()):
        print(f"local        {kernel} {metric} {value:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
