"""Counter profile of ``increment_passing_fwd_kernel`` against the pair it replaces.

The fusion is judged on the pair's total time and the pair's total bytes, never on
the fused kernel's own percentage of its own floor: deleting a round trip takes bytes
out of the numerator, so the faster arm can score lower. One driver runs both arms
under one floor fit, one warmup policy, and one capture window. ``--arm fused``
launches :func:`increment_passing_forward`; ``--arm pair`` launches
:func:`chunk_increment_forward` and then :func:`state_passing_forward`, and NCU is
narrowed to both kernels so the printed blocks add up to the arm.

    python3 scripts/perf/profile_increment_passing_fwd.py --arm fused \
        --shape acceptance
    python3 scripts/perf/profile_increment_passing_fwd.py --sweep --shape acceptance
    CUTE_DSL_LINEINFO=1 python3 scripts/perf/profile_increment_passing_fwd.py \
        --source --shape acceptance

``--source`` ranks the kernel's own source lines by instruction count. It needs
``CUTE_DSL_LINEINFO=1`` in the environment, which the profiled target inherits;
without it NCU correlates nothing and the pass raises rather than printing an empty
table.

``--sweep`` is the tiling curve, and it runs in one process on purpose: the SM clock
moves with what else is on the part, so rows measured in separate processes are not
comparable to each other. The pair is timed first and again last, which brackets any
drift across the fused rows between two readings of the same baseline. Each row
carries the arena it allocates and the blocks per SM that arena permits, since
residency is what feeds the DRAM pipe and shared memory is what bounds it.

No initial state and no streaming carry-in, because that is the variant a step
compiles. :func:`slinoss.perf.workload.step` calls
:func:`slinoss.ops.so3ssd.so3ssd` with five positional tensors and no keyword, so
:class:`slinoss.ops.so3ssd.interface.SO3SSDFunction` hands
:func:`slinoss.ops.so3ssd.cute.forward.so3ssd_fwd_cute` ``z0=None``,
``u_prev=None`` and ``b_prev=None``, and the launch is
``has_z0=False has_prev=False``. The backward's rebuild at
:func:`slinoss.ops.so3ssd.cute.backward.so3ssd_bwd_cute` is the only other caller
and a step never reaches it: the forward saved the prologue. ``has_z0`` and
``has_prev`` are constexpr, so each is a separate compiled kernel and not a flag on
one: a counter crosses between this driver and ``scripts/perf/profile_op.py --mode
step`` only while the variant line below matches. ``--seed-state`` selects the
seeded kernel; nothing here reaches the streaming one.

Two byte counts are printed. ``issued`` charges every per-head operand once per band
of the state, which is what the blocks request. ``unique`` charges it once, which is
what DRAM owes if L2 holds it across the bands that share it. The measured read count
between the two is the reuse the launch order reaches.

The cycle table is separate from the eight standard ones. A duration in
microseconds is a duration at whatever clock the part happened to be running, so
``sm__cycles_active`` is collected beside it and
``gpc__cycles_elapsed.avg.per_second`` beside that: two runs agreeing in cycles and
differing in microseconds differ in contention and not in the kernel.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence

import torch

from slinoss._cute import smem_capacity, smem_residency
from slinoss.ops.so3ssd.cute.fwd.chunk_increment import chunk_increment_forward
from slinoss.ops.so3ssd.cute.fwd.increment_passing import (
    RESIDENT_MAX,
    SPLIT,
    fused_kblock,
    fused_smem_bytes,
    increment_passing_forward,
)
from slinoss.ops.so3ssd.cute.fwd.state_passing import state_passing_forward
from slinoss.ops.so3ssd.cute.mma import MMA_TILE_K, MMA_TILE_N, WARPS, WARPS_WIDE
from slinoss.perf.capture import profiler_window
from slinoss.perf.ceiling import dram_floor_verdict, dram_time_floor
from slinoss.perf.device import (
    compute_apps_query,
    device_info,
    device_ordinal,
    require_cuda,
    smi_selector,
)
from slinoss.perf.ncu import (
    NCU_TABLES,
    SPILL_TABLE,
    NcuPass,
    NcuTable,
    SourcePass,
    SpillCounters,
    kernel_counters,
    run_ncu,
    run_source,
    spill_counters,
)
from slinoss.perf.timing import measure, on_device
from slinoss.perf.units import Bytes, Count
from slinoss.perf.workload import SHAPE_NAMES, OpShape, make_inputs, shape_by_name

KERNELS = {
    "fused": "increment_passing_fwd",
    "pair": "chunk_increment_fwd|state_passing_fwd",
}
"""Regex NCU narrows to, per arm. The mangled symbol carries the constexpr suffix."""

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

CYCLES = NcuTable(
    "cycles",
    (
        "gpu__time_duration.sum",
        "sm__cycles_active.avg",
        "gpc__cycles_elapsed.avg.per_second",
    ),
)
"""Cycles and the clock they were counted at, as a ninth pass.

Not folded into :data:`slinoss.perf.ncu.NCU_TABLES`, which every driver runs: this
pair answers whether two runs of one kernel differed in contention, which is a
question about a comparison rather than about a kernel's class.
"""

GEOMETRY = {
    "bsz": "Batch, B.",
    "heads": "Heads, H.",
    "seq": "Tokens, T. Sets the chunk count with the chunk length.",
    "rows": "Rows per head, P.",
    "lanes": "Lanes per head, N. The state width is 3N.",
    "chunk": "Chunk length, L.",
    "groups": "Groups, G. Divides H. The fold is H // G.",
}
"""Overridable :class:`slinoss.perf.workload.OpShape` fields, name to help text.

Every field is here: both arms run once per step, so ``bsz`` and ``seq`` scale one
launch rather than the launch count."""


def legal_spans(dim: int, rows: int, threads: int) -> tuple[int, ...]:
    """Every band width the fused launch can cover exactly, narrowest first.

    A multiple of 3 and of :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N` that
    divides ``dim`` and splits its 3-vectors evenly over the block. Enumerated rather
    than listed, so the sweep is the whole space at the shape and not three rows
    somebody wrote down.

    Args:
        dim: ``3N``.
        rows: ``P``.
        threads: Block width.

    Returns:
        The widths, ascending.
    """
    return tuple(
        span
        for span in range(MMA_TILE_N, dim + 1, MMA_TILE_N)
        if span % 3 == 0 and dim % span == 0 and (rows * span // 3) % threads == 0
    )


def legal_slices(chunk: int) -> tuple[int, ...]:
    """Every K slice width the chunk divides, narrowest first.

    The slice is the only per-slice allocation, so it is the lever on the arena and
    therefore on residency. Above the atom's own K extent it is still legal and costs
    shared bytes for one fewer barrier pair per chunk, which is the trade the sweep
    prices.

    Args:
        chunk: ``L``.
    """
    return tuple(
        kblk for kblk in range(MMA_TILE_K, chunk + 1, MMA_TILE_K) if chunk % kblk == 0
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(KERNELS), default="fused")
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    for field, helptext in GEOMETRY.items():
        parser.add_argument(f"--{field}", type=int, default=None, help=helptext)
    parser.add_argument(
        "--span",
        type=int,
        default=SPLIT,
        help="Band of 3N one block contracts and carries. Ignored by the pair.",
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
        "--kblk",
        type=int,
        default=0,
        help="K extent of one slice of the fused kernel. Zero takes the shipped "
        "default, the widest slice the residency admits. Ignored by the pair.",
    )
    parser.add_argument(
        "--resident",
        type=int,
        default=None,
        help="Blocks per SM the fused launch bound asks for. Defaults to "
        "RESIDENT_MAX capped by the arena. The cap it puts on the register file is "
        "what decides whether the residency is reached, so it is priced rather than "
        "assumed.",
    )
    parser.add_argument(
        "--seed-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Supply an initial state, the has_z0 variant. Off by default because a "
        "step supplies none; the module docstring says why.",
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
        "--sweep",
        action="store_true",
        help="Time the pair, every legal fused tiling, and the pair again, in this "
        "process. No profiler: this mode is the shape of the curve.",
    )
    parser.add_argument(
        "--residents",
        default="0",
        help="Comma-separated launch bounds to cross the sweep with. 0 means the "
        "default the arena permits.",
    )
    parser.add_argument(
        "--source",
        action="store_true",
        help="Rank the kernel's own source lines by instruction count instead of "
        "collecting the counter tables. Needs CUTE_DSL_LINEINFO=1 in this "
        "environment, which the target inherits.",
    )
    parser.add_argument(
        "--source-top",
        type=int,
        default=24,
        help="Source lines to print, hottest first.",
    )
    parser.add_argument(
        "--report",
        default="increment_passing_fwd",
        help="Where the source pass writes its NCU report.",
    )
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
        The tiling knobs are not fields of the shape, so they rename it and nothing
        else.

    Raises:
        KeyError: If ``--shape`` is not one of
            :data:`slinoss.perf.workload.SHAPES`.
    """
    shape = shape_by_name(args.shape)
    given = {f: getattr(args, f) for f in GEOMETRY if getattr(args, f) is not None}
    suffix = "".join(f"-{f}{v}" for f, v in sorted(given.items()))
    if args.arm == "fused" and not args.sweep:
        if args.span != SPLIT:
            suffix += f"-span{args.span}"
        if args.warps != WARPS:
            suffix += f"-w{args.warps}"
        if args.kblk:
            suffix += f"-k{args.kblk}"
        if args.resident is not None:
            suffix += f"-r{args.resident}"
    if not suffix:
        return shape
    return dataclasses.replace(shape, name=f"{shape.name}{suffix}", **given)


@dataclasses.dataclass(frozen=True)
class Tiling:
    """One fused configuration, and what it costs in shared memory.

    Attributes:
        span: Band of ``3N`` one block owns.
        warps: Warps per block.
        kblk: K extent of one slice.
        resident: Launch bound asked for, or None for the arena's own cap.
        arena_bytes: Shared memory the block allocates.
        residency: Blocks per SM that arena permits, driver reservation included.
    """

    span: int
    warps: int
    kblk: int
    resident: int | None
    arena_bytes: int
    residency: int

    def describe(self) -> str:
        """One column group for a sweep row."""
        asked = "auto" if self.resident is None else str(self.resident)
        return (
            f"span {self.span:3d} warps {self.warps} kblk {self.kblk:2d} "
            f"bound {asked:>4s} arena {self.arena_bytes:6,d} B "
            f"residency {self.residency}"
        )


def tiling(
    shape: OpShape,
    itemsize: int,
    *,
    span: int,
    warps: int,
    kblk: int,
    resident: int = 0,
) -> Tiling:
    """Fill in a tiling's shared-memory columns from the layouts.

    Args:
        shape: The problem size.
        itemsize: Bytes per activation element.
        span: Band of ``3N`` one block owns.
        warps: Warps per block.
        kblk: K extent of one slice. Zero takes the shipped default,
            :func:`fused_kblock`.
        resident: Launch bound to ask for. Zero means the arena's own cap.

    Returns:
        The tiling, with the arena and the residency computed.
    """
    width = kblk or fused_kblock(shape.chunk, shape.rows, span, itemsize)
    nbytes = fused_smem_bytes(shape.chunk, shape.rows, span, itemsize, kblk=width)
    return Tiling(
        span=span,
        warps=warps,
        kblk=width,
        resident=resident or None,
        arena_bytes=nbytes,
        residency=min(RESIDENT_MAX, smem_residency(nbytes)),
    )


def sweep_tilings(
    shape: OpShape, itemsize: int, residents: Sequence[int]
) -> list[Tiling]:
    """Every legal fused configuration at one shape, in report order.

    Args:
        shape: The problem size.
        itemsize: Bytes per activation element.
        residents: Launch bounds to cross with. Zero means the arena's own cap.

    Returns:
        The configurations, band width slowest-varying.
    """
    out: list[Tiling] = []
    for warps in range(WARPS, WARPS_WIDE + 1, WARPS):
        for span in legal_spans(shape.d_state, shape.rows, warps * 32):
            for kblk in legal_slices(shape.chunk):
                for asked in residents:
                    out.append(
                        tiling(
                            shape,
                            itemsize,
                            span=span,
                            warps=warps,
                            kblk=kblk,
                            resident=asked,
                        )
                    )
    return sorted(out, key=lambda t: (t.span, t.warps, t.kblk, t.resident or 0))


def build_runner(
    shape: OpShape,
    device: torch.device,
    dtype: torch.dtype,
    arm: str,
    one: Tiling,
    seed_state: bool,
) -> Callable[[], None]:
    """Allocate one input set and return the callable that launches the arm.

    Both arms read the same inputs and allocate their own outputs per call, as the
    host entries do. Nothing is read off a forward: this driver measures traffic and
    occupancy, and the parity file is where the values have to be the pipeline's.

    Args:
        shape: The problem size.
        device: Where to allocate.
        dtype: Activation dtype.
        arm: ``fused`` or ``pair``.
        one: The fused tiling. Ignored by the pair.
        seed_state: Whether to supply an initial state.

    Returns:
        The callable.
    """
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=False)
    z0 = (
        torch.zeros(
            shape.bsz,
            shape.heads,
            shape.rows,
            shape.d_state,
            dtype=torch.float32,
            device=device,
        )
        if seed_state
        else None
    )

    def fused() -> None:
        increment_passing_forward(
            inputs.U,
            inputs.trans,
            inputs.K,
            inputs.B,
            shape.chunk,
            z0=z0,
            span=one.span,
            warps=one.warps,
            kblk=one.kblk,
            resident=one.resident,
        )

    def pair() -> None:
        out = chunk_increment_forward(
            inputs.U, inputs.trans, inputs.K, inputs.B, shape.chunk
        )
        state_passing_forward(out.inc, out.cquat, out.cscale, z0)

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
        "--kblk",
        str(args.kblk),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]
    for field in GEOMETRY:
        value = getattr(args, field)
        if value is not None:
            argv += [f"--{field}", str(value)]
    if args.resident is not None:
        argv += ["--resident", str(args.resident)]
    # Spelled either way rather than only when true, so the child cannot pick up a
    # different default and compile a different kernel than the one reported.
    argv += ["--seed-state" if args.seed_state else "--no-seed-state"]
    return argv


def variant_line(seed_state: bool) -> str:
    """The compile-time variant, as the ``variant`` report row.

    Two constexpr flags select the kernel. ``has_prev`` is always false here: the
    driver supplies no streaming carry-in, and neither does a step.

    Args:
        seed_state: Whether an initial state is supplied.
    """
    return f"variant      has_z0={seed_state} has_prev=False"


def analytic_bytes(
    shape: OpShape, span: int, itemsize: int, arm: str
) -> tuple[Bytes, Bytes]:
    """Traffic one call of the arm moves, operand by operand.

    The fused arm reads every per-head operand once per band of the state, because a
    band carries a slice of the state and needs the whole of ``u`` and the whole
    transform to form it. The pair reads each once and pays a full ``(B,H,C,P,3N)``
    round trip instead: the increment writes the plane, the recurrence reads it, and
    ``zstart`` overwrites it.

    Args:
        shape: The problem size.
        span: Band width.
        itemsize: Bytes per activation element.
        arm: ``fused`` or ``pair``.

    Returns:
        ``(issued, unique)``. The first charges a shared operand once per block that
        requests it, the second once per distinct buffer.
    """
    chunks = -(-shape.seq // shape.chunk)
    tokens = shape.bsz * shape.seq
    lead = shape.bsz * shape.heads
    weights = tokens * shape.heads * shape.rows * itemsize
    trans = tokens * shape.heads * 4 * 4
    taps = tokens * shape.heads * 2 * 4 * 4
    forcing = tokens * shape.d_state * itemsize
    plane = lead * chunks * shape.rows * shape.d_state * 4
    # The fused arm stores ``zstart`` at the activation dtype. The pair cannot: its
    # ``zstart`` is a view of the float32 increment buffer the recurrence read.
    stored = lead * chunks * shape.rows * shape.d_state * itemsize
    state = lead * shape.rows * shape.d_state * 4
    transition = lead * chunks * 4 * 4 + lead * chunks * 4
    carry = shape.bsz * shape.groups * shape.d_state * itemsize
    carry += shape.bsz * shape.heads * shape.rows * itemsize
    written = state + transition + carry
    per_head = weights + trans + taps
    if arm == "fused":
        bands = -(-shape.d_state // span)
        return (
            Bytes(bands * per_head + forcing * shape.heads + stored + written),
            Bytes(per_head + forcing * shape.groups + stored + written),
        )
    # Three passes over the plane: the increment writes it, the recurrence reads it,
    # and the recurrence writes zstart over it. All three are float32, because the
    # pair's zstart is a view of the increment buffer the recurrence read.
    return (
        Bytes(per_head + forcing * shape.heads + 3 * plane + written),
        Bytes(per_head + forcing * shape.groups + 3 * plane + written),
    )


def run_sweep(args: argparse.Namespace, shape: OpShape, device: torch.device) -> int:
    """Time the pair, every fused tiling, and the pair again, in this process.

    Returns:
        Process exit status.

    Raises:
        ValueError: If ``--residents`` is not a comma-separated list of integers.
    """
    dtype = DTYPES[args.dtype]
    itemsize = torch.finfo(dtype).bits // 8
    residents = [int(one) for one in args.residents.split(",")]
    baseline = tiling(shape, itemsize, span=SPLIT, warps=WARPS, kblk=0)
    rows = sweep_tilings(shape, itemsize, residents)
    floor = dram_time_floor(device)
    print(f"shape        {shape.describe()}  {args.dtype}")
    print(variant_line(args.seed_state))
    print(
        f"floor        c={floor.fixed_duration_us:.3f} us "
        f"B={floor.asymptotic_gbs:.2f} GB/s l2={floor.l2_bytes} B"
    )
    for arm in ("pair", "fused"):
        issued, unique = analytic_bytes(shape, SPLIT, itemsize, arm)
        print(
            f"model {arm:5s}  issued {issued / 1e6:8.2f} MB "
            f"{floor.floor_us(issued):7.1f} us  "
            f"unique {unique / 1e6:8.2f} MB {floor.floor_us(unique):7.1f} us"
        )

    def time_one(arm: str, one: Tiling, label: str) -> None:
        runner = build_runner(shape, device, dtype, arm, one, args.seed_state)
        timed = measure(
            runner,
            label=label,
            iters=args.event_iters,
            warmup=args.warmup,
            device=device,
        )
        spread = timed.total
        del runner
        torch.cuda.empty_cache()
        knobs = "" if arm == "pair" else f"  {one.describe()}"
        print(
            f"{label:14s} med={spread.median_duration_us:8.1f} us "
            f"min={spread.min_duration_us:8.1f} max={spread.max_duration_us:8.1f} "
            f"resolution={spread.resolution_pct:5.2f}%  {timed.clocks}{knobs}"
        )

    time_one("pair", baseline, "pair-before")
    for index, one in enumerate(rows):
        if one.arena_bytes > smem_capacity():
            print(f"fused-{index:02d}      skipped: {one.describe()} over capacity")
            continue
        time_one("fused", one, f"fused-{index:02d}")
    time_one("pair", baseline, "pair-after")
    return 0


def print_source(passed: SourcePass, top: int) -> None:
    """Print the per-line attribution, hottest line first.

    Ranked by instruction count rather than by LSU count, because the discriminant
    is total instructions: an arm that moves work off the port and onto another pipe
    has been measured to lose. The LSU column is beside it, not instead of it.

    The module column is the traced entry module on every row and not the file the
    line is in, which is an NVVM property; see
    :func:`slinoss.perf.ncu.parse_source_csv`. Intersect the line numbers with the
    location set of the modules the kernel traces before naming a site.

    Args:
        passed: The collected pass.
        top: Rows to print.
    """
    attributed = sum(one.inst_count for one in passed.lines)
    lsu = sum(one.lsu_inst_count for one in passed.lines)
    excess = sum(one.shared_wavefront_excess_count for one in passed.lines)
    print(f"source       {passed.report}")
    print(f"  modules    {sorted({one.entry_module for one in passed.lines})}")
    print(
        f"  attributed {attributed:,} inst  {lsu:,} LSU  "
        f"excess shared wavefronts {excess:,}"
    )
    print(f"  unattributed {passed.unattributed_inst_count:,} inst")
    for one in sorted(passed.lines, key=lambda row: -row.inst_count)[:top]:
        reason, samples = max(
            one.stall_samples.items(), key=lambda item: item[1], default=("", 0)
        )
        print(
            f"  line {one.line:5d} inst {one.inst_count:11,} "
            f"lsu {one.lsu_inst_count:10,} "
            f"wf {one.shared_wavefront_count:,}/"
            f"{one.shared_wavefront_ideal_count:,} "
            f"samples {one.sample_count:,} top {reason} {samples:,}"
        )
        codes = " ".join(f"{name} {n:,}" for name, n in one.opcode_inst.items())
        widths = " ".join(f"{bits}b {n:,}" for bits, n in one.access_bit_inst.items())
        if codes or widths:
            print(f"               {codes}   [{widths}]")


def run_source_mode(args: argparse.Namespace) -> int:
    """Collect the source pass and print it.

    Args:
        args: The parsed command line.

    Returns:
        Process exit status.
    """
    passed = run_source(
        target_argv(args),
        report=args.report,
        ncu=args.ncu,
        extra=("--kernel-name", f"regex:{KERNELS[args.arm]}"),
    )
    print(f"arm          {args.arm}")
    print(variant_line(args.seed_state))
    print_source(passed, args.source_top)
    return 0


def print_cycles(passes: Sequence[NcuPass]) -> None:
    """Print the cycle count and the clock it was counted at, per kernel.

    Args:
        passes: The parsed :data:`CYCLES` pass, or an empty sequence.
    """
    for one in passes:
        totals: dict[str, tuple[float, float, int]] = {}
        for invocation in one.invocations:
            cycles = invocation.values.get("sm__cycles_active.avg", 0.0)
            hertz = invocation.values.get("gpc__cycles_elapsed.avg.per_second", 0.0)
            seen = totals.get(invocation.kernel, (0.0, 0.0, 0))
            totals[invocation.kernel] = (
                seen[0] + cycles,
                max(seen[1], hertz),
                seen[2] + 1,
            )
        for kernel, (cycles, hertz, launches) in totals.items():
            print(f"cycles       {kernel}")
            print(f"  sm cycles  {cycles / launches:,.0f}/launch")
            print(f"  gpc clock  {hertz / 1e9:.4f} GHz")


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
    itemsize = torch.finfo(dtype).bits // 8
    one = tiling(
        shape,
        itemsize,
        span=args.span,
        warps=args.warps,
        kblk=args.kblk,
        resident=args.resident or 0,
    )

    if args.window:
        runner = build_runner(shape, device, dtype, args.arm, one, args.seed_state)
        with on_device(device):
            for _ in range(args.warmup):
                runner()
            with profiler_window(device):
                for _ in range(args.iters):
                    runner()
        return 0

    if args.sweep:
        return run_sweep(args, shape, device)

    if args.source:
        return run_source_mode(args)

    ordinal = device_ordinal(device)
    info = device_info(ordinal)
    before = compute_apps_query(smi_selector(ordinal))
    runner = build_runner(shape, device, dtype, args.arm, one, args.seed_state)
    timed = measure(
        runner,
        label=f"{args.arm} {shape.name}",
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
    )
    floor = dram_time_floor(device)
    issued, unique = analytic_bytes(shape, args.span, itemsize, args.arm)
    del runner
    torch.cuda.empty_cache()

    narrow = ("--kernel-name", f"regex:{KERNELS[args.arm]}")
    passes: list[NcuPass] = []
    cycles: list[NcuPass] = []
    spills: tuple[SpillCounters, ...] = ()
    for table in (*NCU_TABLES, SPILL_TABLE, CYCLES):
        pass_ = run_ncu(table, target_argv(args), ncu=args.ncu, extra=narrow)
        if pass_.missing_metrics:
            raise ValueError(
                f"ncu table {table.name!r} returned no value for "
                f"{list(pass_.missing_metrics)}; the metric names are wrong for "
                f"this driver"
            )
        if table is SPILL_TABLE:
            spills = spill_counters(pass_)
        elif table is CYCLES:
            cycles.append(pass_)
        else:
            passes.append(pass_)
    after = compute_apps_query(smi_selector(ordinal))

    print(f"arm          {args.arm}")
    print(variant_line(args.seed_state))
    if args.arm == "fused":
        print(f"tiling       {one.describe()}")
    print(f"shape        {shape.describe()}")
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
    warp_ceiling = info.max_threads_per_sm_count / info.warp_thread_count
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
        # Blocks per SM is not a counter. The theoretical occupancy is the warp
        # census the launch configuration permits, so dividing it by the block's own
        # warps is what NCU knows about residency.
        blocks_per_sm = (
            counters.theoretical_occupancy_pct
            / 100.0
            * warp_ceiling
            / (counters.thread_per_block_count / info.warp_thread_count)
        )
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
        print(f"  per SM     {blocks_per_sm:.2f} blocks")
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
    print_cycles(cycles)

    for record in spills:
        launches = record.launch_count
        print(f"spill        {record.kernel}")
        print(f"  ld sectors {record.local_load_sector_count / launches:.0f}/launch")
        print(f"  st sectors {record.local_store_sector_count / launches:.0f}/launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
