"""Counter profile of ``state_passing_bwd_kernel``, scored against its own floor.

The so3ssd CuTe backend registers the reference backward, so no operator any other
driver in this directory runs launches this kernel, and the class audit judges only
what a capture contained. This driver reaches it directly.

Two modes in one file, for the reason
``scripts/perf/profile_chunk_input_bwd.py`` has two: one kernel has one warmup
policy and one capture window, and a second file would only let them drift. The
default drives; ``--window`` is the process NCU attaches to.

``--rows``, ``--lanes``, ``--heads`` and ``--chunk`` override the named shape's
geometry. Every one of them moves this kernel's traffic and its grid, since the
launch is a bijection from threads onto the ``B*H*P*N`` independent 3-vectors and
the chunk count is the trip count of the recurrence. The overrides rename the
shape, so a figure taken under one cannot be read as a figure at the name it
started from.

    python3 scripts/perf/profile_state_passing_bwd.py --shape standard
    python3 scripts/perf/profile_state_passing_bwd.py --rows 64 --lanes 80 --heads 18

``--seed-state`` selects the ``has_dstate`` variant and ``--no-readout`` the
``has_dzstart`` one. The step path supplies neither a final-state cotangent nor an
absent ``dy``, so the default is the compiled variant a training step launches.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence

import torch

from slinoss.ops.so3ssd.cute.bwd.state_passing import state_passing_backward
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
from slinoss.perf.workload import SHAPE_NAMES, OpShape, shape_by_name

KERNEL = "state_passing_bwd"
"""Regex NCU narrows to. The mangled symbol carries the constexpr suffix."""

GEOMETRY = {
    "bsz": "Batch, B.",
    "heads": "Heads, H.",
    "seq": "Tokens, T. Sets the chunk count with the chunk length.",
    "rows": "Rows per head, P.",
    "lanes": "Lanes per head, N. The state width is 3N.",
    "chunk": "Chunk length, L.",
}
"""Overridable :class:`slinoss.perf.workload.OpShape` fields, name to help text.

Unlike the two GEMM backward drivers this carries ``bsz`` and ``seq``: this kernel
runs once per step whatever they are, so they scale one launch rather than the
launch count."""

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
    parser.add_argument("--device", default="cuda")
    for field, helptext in GEOMETRY.items():
        parser.add_argument(f"--{field}", type=int, default=None, help=helptext)
    parser.add_argument(
        "--seed-state",
        action="store_true",
        help="Supply a final-state cotangent, the has_dstate variant.",
    )
    parser.add_argument(
        "--no-readout",
        action="store_true",
        help="Drop the readout cotangent, the variant an absent dy leaves.",
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

    Raises:
        KeyError: If ``--shape`` is not one of
            :data:`slinoss.perf.workload.SHAPES`.
    """
    shape = shape_by_name(args.shape)
    given = {f: getattr(args, f) for f in GEOMETRY if getattr(args, f) is not None}
    if not given:
        return shape
    suffix = "".join(f"-{f}{v}" for f, v in sorted(given.items()))
    return dataclasses.replace(shape, name=f"{shape.name}{suffix}", **given)


def build_runner(
    shape: OpShape, device: torch.device, *, seed_state: bool, readout: bool
) -> Callable[[], None]:
    """Allocate one input set and return the callable that launches the kernel.

    ``dzstart`` is the chunk-start backward's output and the two chunk transitions
    are the forward's. Their values do not reach a counter, so they are drawn from a
    generator rather than rematerialized through the stages that produce them. The
    quaternion is normalized anyway, because I1 holds it to unit norm and a kernel
    read against a non-unit one would be reading a shape the operator cannot
    produce.

    ``dzstart`` is consumed in place, so one buffer serves every launch: the kernel
    reads what the previous launch wrote. The chunk decay is in ``(0, 1]``, so the
    recurrence cannot grow the buffer past float32 range over a measurement's worth
    of launches.

    Args:
        shape: The problem size.
        device: Where to allocate.
        seed_state: Supply the final-state cotangent.
        readout: Whether ``dzstart`` carries the readout cotangent on entry.

    Returns:
        The callable. Allocates ``dz0`` per call as the host entry does.
    """
    gen = torch.Generator(device=device).manual_seed(1)
    chunks = -(-shape.seq // shape.chunk)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(*size, dtype=torch.float32, device=device, generator=gen)

    dzstart = randn(shape.bsz, shape.heads, chunks, shape.rows, shape.d_state)
    cquat = randn(shape.bsz, shape.heads, chunks, 4)
    cquat = cquat / cquat.norm(dim=-1, keepdim=True)
    cscale = randn(shape.bsz, shape.heads, chunks).sigmoid()
    dstate = randn(shape.bsz, shape.heads, shape.rows, shape.d_state)

    def run() -> None:
        state_passing_backward(
            dzstart,
            cquat,
            cscale,
            dstate if seed_state else None,
            has_dzstart=readout,
        )

    return run


def target_argv(args: argparse.Namespace) -> list[str]:
    """The argv NCU attaches to. Carries every override forward."""
    argv = [
        sys.executable,
        __file__,
        "--window",
        "--shape",
        args.shape,
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
    if args.seed_state:
        argv.append("--seed-state")
    if args.no_readout:
        argv.append("--no-readout")
    return argv


def local_sectors(one: NcuPass) -> dict[str, float]:
    """Sum the local-memory lookup metrics over the profiled launches.

    Args:
        one: The parsed ``local`` pass.

    Returns:
        Metric name to summed sector count, empty for a metric this driver does
        not carry.
    """
    out: dict[str, float] = {}
    for invocation in one.invocations:
        for metric, value in invocation.values.items():
            if metric.endswith(".sum") and "local" in metric:
                out[metric] = out.get(metric, 0.0) + value
    return out


def analytic_bytes(shape: OpShape, *, seed_state: bool, readout: bool) -> Bytes:
    """Traffic one launch moves, operand by operand, every tensor float32.

    ``dzstart`` is read and written over once, the readout variant deciding the
    read; ``dz0`` is written; the two chunk transitions are read once per chunk and
    head. Nothing is read twice, so this is the floor's byte count.

    Args:
        shape: The problem size.
        seed_state: Whether the final-state cotangent is read.
        readout: Whether ``dzstart`` is read as well as written.

    Returns:
        Bytes moved per launch.
    """
    chunks = -(-shape.seq // shape.chunk)
    lead = shape.bsz * shape.heads
    state = lead * shape.rows * shape.d_state * 4
    total = state * chunks * (2 if readout else 1)
    total += state
    total += state if seed_state else 0
    total += lead * chunks * 4 * 4
    total += lead * chunks * 4
    return Bytes(total)


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
    readout = not args.no_readout

    if args.window:
        runner = build_runner(
            shape, device, seed_state=args.seed_state, readout=readout
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
    runner = build_runner(shape, device, seed_state=args.seed_state, readout=readout)
    label = f"state_passing_bwd {shape.name}"
    timed = measure(
        runner,
        label=label,
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
    )
    floor = dram_time_floor(device)
    model = analytic_bytes(shape, seed_state=args.seed_state, readout=readout)
    del runner

    tables = (*NCU_TABLES, SPILL_TABLE, LOCAL_TABLE)
    narrow = ("--kernel-name", f"regex:{KERNEL}")
    passes: list[NcuPass] = []
    spills: tuple[SpillCounters, ...] = ()
    local: dict[str, float] = {}
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

    print(f"shape        {shape.describe()}")
    print(f"variant      has_dstate={args.seed_state} has_dzstart={readout}")
    print(f"clocks       {timed.clocks}")
    print(f"smi before   {before}")
    print(f"smi after    {after}")
    print(
        f"floor        c={floor.fixed_duration_us:.3f} us "
        f"B={floor.asymptotic_gbs:.2f} GB/s "
        f"residual={floor.max_residual_pct:.2f}% l2={floor.l2_bytes} B"
    )
    print(f"model        {model / 1e6:.2f} MB/launch {floor.floor_us(model):.1f} us")
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
    for metric, value in sorted(local.items()):
        print(f"local        {metric} {value:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
