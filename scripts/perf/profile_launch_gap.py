"""Where a paired step's event interval goes, split three ways per arm.

The event-bracketed step of the acceptance shape reads longer than the sum of its
kernels, and the difference grew while the kernel sum fell. A subtraction cannot
say what the difference is: device idle between launches, host time the device
waited through, and the placement of the two CUDA events themselves all land in
the same residual. This driver separates them, for both arms of the Mamba2
face-off, from a timeline rather than by subtraction.

Four quantities per arm, three clocks:

1. The CUDA-event interval around the arm, from :func:`measure_paired`. The number
   every ratio in this project is stated in.
2. The push-to-pop NVTX interval, from NSYS. What the host thread spent enqueueing.
3. The device span, first launch of a step to last, from the NSYS timeline. Split
   into busy and idle by :func:`slinoss.perf.nsys.occupancy`, by union over the
   timeline, so a gap is a measured gap and not a remainder.
4. The sum of ``gpu__time_duration.sum`` over every launch, from NCU. Kernel time
   with no gap and no launch cost in it at all.

    PYTHONPATH=$PWD python3 scripts/perf/profile_launch_gap.py --shape acceptance

Each arm is traced in its own process, so every device operation in the window
belongs to that arm and the steps are cut out by
:func:`slinoss.perf.nsys.repeat_windows`. An NVTX range cannot do that job: the
projection is per thread, and the autograd engine runs the backward on its own
worker thread, so a range pushed around a step projects the forward's launches and
none of the backward's. The range is still pushed, for its push-to-pop host
interval, which does cover the blocking ``grad`` call.

``--arm`` selects what the traced process runs, and is also how NCU sees one arm at
a time: NCU replays each kernel in isolation, so per-launch duration does not depend
on the interleave, while the event loop does and runs both arms in one process with
the order swapped every iteration.

``--serialize-with`` holds a lock file while each workload process runs, for a
device several measurements share. Off by default.

This driver does not gate on the three clocks agreeing. Every other report in this
package refuses to emit past a five percent disagreement; the disagreement is the
subject here, so it is reported with its parts named instead.

The Mamba arm takes its group count from ``--groups``, not from the shape:
``bench_mamba`` does the same, and a driver that silently used the shape's own
would compare two group counts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from scripts.bench.bench_mamba import DTYPES, load_scan
from scripts.bench.bench_mamba import make_inputs as mamba_inputs
from scripts.bench.bench_mamba import runner as mamba_runner
from slinoss.graph import capture
from slinoss.perf.capture import profiler_window
from slinoss.perf.device import device_info, device_ordinal, require_cuda
from slinoss.perf.ncu import NcuTable, run_ncu
from slinoss.perf.nsys import (
    GpuEvent,
    NvtxSpan,
    Occupancy,
    nsys_report_texts,
    occupancy,
    parse_gpu_events,
    parse_nvtx_projection,
    repeat_windows,
)
from slinoss.perf.timing import measure_paired, on_device
from slinoss.perf.tools import resolve_tool
from slinoss.perf.workload import SHAPE_NAMES, OpShape, shape_by_name
from slinoss.perf.workload import forward_only as so3ssd_forward_only
from slinoss.perf.workload import make_inputs as so3ssd_inputs
from slinoss.perf.workload import step as so3ssd_step

MODES = ("forward", "step")
ARMS = ("both", "a", "b")
DURATION = NcuTable("duration", ("gpu__time_duration.sum",))
"""The one metric this driver reads. A kernel sum, per launch, in nanoseconds."""

GPU_TRACE = "cuda_gpu_trace"
NVTX_PROJ = "nvtx_gpu_proj_trace"


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------


def arm_labels(shape: OpShape, groups: int, backend: str | None) -> tuple[str, str]:
    """The two region labels, spelled exactly as ``bench_mamba`` spells them.

    Args:
        shape: The problem size. Unused by the labels; taken so a caller cannot
            pass a group count from one shape and a label from another.
        groups: Mamba2 group count.
        backend: SO(3) backend name, or None for the fastest registered one.

    Returns:
        The baseline label and the label under test.
    """
    del shape
    return f"mamba-g{groups}", f"so3ssd-{backend or 'auto'}"


def nvtx_wrap(name: str, fn: Callable[[], None]) -> Callable[[], None]:
    """Push an NVTX range around ``fn``.

    The range is the only thing that ties a device operation to an arm. Kernel
    names cannot: both arms launch elementwise ``aten`` kernels, and a name in both
    arms cannot be attributed by name.

    Args:
        name: Range name. The arm's region label, so the timeline and the event
            tree carry one vocabulary.
        fn: The arm.

    Returns:
        The wrapped arm.
    """

    def run() -> None:
        torch.cuda.nvtx.range_push(name)
        try:
            fn()
        finally:
            torch.cuda.nvtx.range_pop()

    return run


def build_arms(
    args: argparse.Namespace, device: torch.device
) -> tuple[str, Callable[[], None], str, Callable[[], None]]:
    """Both arms of the face-off at one shape.

    Not NVTX-wrapped: the range has to go outside a graph capture, or it is recorded
    into the graph and pushed once at capture rather than once per replay.

    Args:
        args: Parsed command line.
        device: Device to allocate on.

    Returns:
        The baseline label and callable, then the label and callable under test.
    """
    shape = shape_by_name(args.shape)
    groups = 1 if args.groups == "one" else shape.heads
    dtype = DTYPES[args.dtype]
    grads = args.mode == "step"
    a_label, b_label = arm_labels(shape, groups, args.backend)
    mamba = mamba_inputs(shape, groups, device, dtype=dtype, requires_grad=grads)
    ours = so3ssd_inputs(shape, device, dtype=dtype, requires_grad=grads)
    a = mamba_runner(load_scan(), mamba, shape.chunk, grads=grads, prefix=a_label)
    b = (
        so3ssd_step(ours, shape.chunk, backend=args.backend, prefix=b_label)
        if grads
        else so3ssd_forward_only(
            ours, shape.chunk, backend=args.backend, prefix=b_label
        )
    )
    return a_label, a, b_label, b


def graphed(fn: Callable[[], None]) -> Callable[[], None]:
    """Record ``fn`` into a CUDA graph and return a callable that replays it.

    The arm closes over its own tensors, so the graph takes no arguments and a
    replay reads the buffers the closure already held.

    A replay is one launch. What it removes from a step is the per-launch host cost
    and whatever device idle waiting on the host produced; the kernels, their order,
    and their durations are unchanged. So the difference a graph makes is an upper
    bound on what launch cost was worth, and no evidence about the kernels.

    Args:
        fn: The arm, taking no arguments.

    Returns:
        The replay.

    Raises:
        RuntimeError: If the capture recorded no work, or compiled anything. The
            second means the arm had not been warmed until every executor was
            traced, and tracing inside a capture is host work a graph cannot hold.
    """
    step = capture(lambda: fn())
    return lambda: step()


def run_traced(args: argparse.Namespace) -> int:
    """The process a profiler attaches to, and the event loop when none does.

    Warmup runs before the capture window opens, so no compilation and no
    allocator growth reaches a counter or a timeline.

    ``--arm both`` runs the paired loop and writes its event numbers to
    ``--json``. ``--arm a`` and ``--arm b`` run one arm with no recorder, which is
    what an NCU pass needs: one arm's launches and nothing else.

    ``--graph`` replaces both arms with a CUDA graph replay of the same step. Both
    arms or neither: a graph on one side of a face-off is not a measurement of the
    face-off. The regions the workload records inside its own step do not survive
    the capture, so the paired loop then reports the arm totals and nothing under
    them.

    Args:
        args: Parsed command line.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device, or if a
            capture recorded no work or compiled during recording.
    """
    device = require_cuda(args.device)
    shape = shape_by_name(args.shape)
    a_label, a, b_label, b = build_arms(args, device)
    with on_device(device):
        # Twice: once on the raw arm, because a capture of a step that has not
        # compiled every kernel it launches is refused, and once on whatever the arm
        # ended up being, so the allocator has settled around it.
        for _ in range(args.warmup):
            a()
            b()
        if args.graph:
            a, b = graphed(a), graphed(b)
        a, b = nvtx_wrap(a_label, a), nvtx_wrap(b_label, b)
        for _ in range(args.warmup):
            a()
            b()
        if args.arm != "both":
            one = a if args.arm == "a" else b
            with profiler_window(device):
                for _ in range(args.iters):
                    one()
            return 0
        with profiler_window(device):
            out = measure_paired(
                a_label,
                a,
                b_label,
                b,
                label=f"launch-gap {shape.name} {args.mode}",
                iters=args.iters,
                warmup=0,
                device=device,
            )
    if args.json is not None:
        payload: dict[str, Any] = {
            "shape": shape.describe(),
            "mode": args.mode,
            "dtype": args.dtype,
            "groups": args.groups,
            "graph": args.graph,
            "iters": args.iters,
            "warmup": args.warmup,
            "device": asdict(device_info(device_ordinal(device))),
            "clocks": out.timed.clocks,
            "timer_coverage_pct": float(out.timed.timer_coverage_pct),
            "loop": asdict(out.timed.total),
            "arms": {
                label: {
                    part: asdict(out.timed.region(name).spread)
                    for part, name in (
                        ("total", label),
                        ("forward", f"{label}.forward"),
                        ("backward", f"{label}.backward"),
                    )
                    if _has_region(out.timed, name)
                }
                for label in (a_label, b_label)
            },
            "speedup_ratio": float(out.comparison.speedup_ratio),
            "verdict": out.comparison.verdict(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def _has_region(timed: Any, name: str) -> bool:
    """Whether a measurement carries a region, without raising to find out."""
    try:
        timed.region(name)
    except KeyError:
        return False
    return True


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSplit:
    """One arm's step, split across the three clocks that measured it.

    Every duration is per step: the traced loop's per-arm totals divided by the
    iterations it ran, so the figures are comparable against a single step's event
    interval and against each other.

    Attributes:
        label: Arm label.
        step_count: Steps the trace held for this arm.
        event_us: Median CUDA-event interval around the arm, from the untraced
            loop. None when no event loop was run.
        host_us: Push-to-pop host interval per step.
        device_span_us: First launch to last, per step.
        busy_us: Device executing, per step. A union over the timeline.
        idle_us: Device idle strictly between the arm's own launches, per step.
        launch_count: Device operations per step. An integer: every step of one arm
            launches the same sequence, which is what the segmentation checked.
        max_gap_us: Longest single idle interval inside any one of the arm's steps.
            Not a per-step figure: one settling step can hold it alone, which is
            what ``step_idle_us`` is beside it to show.
        step_idle_us: Idle inside each step, in trace order.
        between_steps_us: Last launch of a step to first launch of the next, per
            boundary. The traced loop's own cost, not the step's.
        projected_op_count: Device operations NSYS projected into the arm's NVTX
            range, per step. Below ``launch_count`` when a range's launches were not
            all made by the thread that pushed it.
        ncu_kernel_us: Sum of ``gpu__time_duration.sum`` over one step's launches,
            or None when the NCU pass was skipped.
        ncu_launch_count: Launches NCU profiled per step, or None.
    """

    label: str
    step_count: int
    event_us: float | None
    host_us: float
    device_span_us: float
    busy_us: float
    idle_us: float
    launch_count: int
    max_gap_us: float
    step_idle_us: tuple[float, ...]
    between_steps_us: float
    projected_op_count: float
    ncu_kernel_us: float | None = None
    ncu_launch_count: float | None = None

    @property
    def outside_span_us(self) -> float | None:
        """Event interval the device span does not cover.

        The two CUDA events bracket the arm on the host, so this holds the host
        time before the first launch reached the device, the time after the last
        kernel retired, and the placement of the events themselves. Negative is
        possible and is not an error: the stop event resolves once the stream
        drains past it, and a device that is still behind the host at the pop puts
        the last kernel's tail outside the pair.
        """
        if self.event_us is None:
            return None
        return self.event_us - self.device_span_us

    @property
    def idle_per_launch_us(self) -> float:
        """Idle divided by the gaps that could hold it, ``launches - 1``."""
        gaps = self.launch_count - 1
        return self.idle_us / gaps if gaps > 0 else 0.0

    @property
    def kernel_us(self) -> float:
        """Kernel time NCU measured, or the traced busy time when NCU was skipped.

        The two disagree by whatever the timeline's own resolution costs. NCU is
        preferred where it ran, because a launch it profiled is a launch it timed in
        isolation with the caches and clocks it was told to leave alone.
        """
        return self.busy_us if self.ncu_kernel_us is None else self.ncu_kernel_us


def split_arm(
    label: str,
    events: Sequence[GpuEvent],
    spans: Sequence[NvtxSpan],
    *,
    event_us: float | None,
    steps: int,
) -> tuple[ArmSplit, tuple[Occupancy, ...]]:
    """Split one arm's traced steps against its event interval.

    The trace holds one arm and nothing else, so the timeline is cut into steps by
    repetition index rather than by NVTX projection. Occupancy is taken per step, so
    the wait between two steps is not counted as idle inside either one.

    Args:
        label: Arm label. Also the NVTX range name the host interval is read from.
        events: The whole timeline of a single-arm traced run.
        spans: Every projected range of that run.
        event_us: Median CUDA-event interval for one step of this arm, or None.
        steps: Steps the traced loop ran.

    Returns:
        The per-step split, and the occupancy of each step.

    Raises:
        KeyError: If no range carries this label. Without it there is no host
            interval, and the arm being traced cannot be confirmed from the trace.
    """
    mine = [s for s in spans if s.name == label]
    if not mine:
        raise KeyError(f"no NVTX range named {label!r} in the trace")
    windows = repeat_windows(events, steps)
    parts = tuple(
        occupancy(f"{label}[{index}]", window) for index, window in enumerate(windows)
    )
    between = [
        float(windows[i + 1][0].start_us) - max(e.end_us for e in windows[i])
        for i in range(len(windows) - 1)
    ]
    return (
        ArmSplit(
            label=label,
            step_count=steps,
            event_us=event_us,
            host_us=median(float(s.host_duration_us) for s in mine),
            device_span_us=median(float(p.span_us) for p in parts),
            busy_us=median(float(p.busy_us) for p in parts),
            idle_us=median(float(p.idle_us) for p in parts),
            launch_count=len(windows[0]),
            max_gap_us=max(float(p.max_gap_us) for p in parts),
            step_idle_us=tuple(float(p.idle_us) for p in parts),
            between_steps_us=median(between) if between else 0.0,
            projected_op_count=median(float(s.gpu_op_count) for s in mine),
        ),
        parts,
    )


def median(values: Iterable[float]) -> float:
    """Middle value, or the mean of the two middle values.

    The median and not the mean, because the first traced step carries a cost no
    later step does and a mean would spread it over all of them.

    Args:
        values: At least one value.

    Returns:
        The median.

    Raises:
        ValueError: If no values were given.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median of no values")
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def target_argv(args: argparse.Namespace, *, arm: str, iters: int) -> list[str]:
    """The traced process's own command line.

    ``--serialize-with`` prefixes ``flock``, and it prefixes the target rather than
    the profiler: the lock is then held while the workload runs and released before
    the report export, which needs no device. A step time measured while another
    process holds the same device is not a measurement of this step.

    Args:
        args: Parsed command line.
        arm: ``both``, ``a`` or ``b``.
        iters: Iterations inside the capture window.

    Returns:
        The argv, this script re-invoked in ``--traced`` mode.
    """
    lock = ["flock", str(args.serialize_with)] if args.serialize_with else []
    return [
        *lock,
        sys.executable,
        str(Path(__file__).resolve()),
        "--traced",
        "--arm",
        arm,
        "--shape",
        args.shape,
        "--mode",
        args.mode,
        "--groups",
        args.groups,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--iters",
        str(iters),
        "--warmup",
        str(args.warmup),
        *(("--backend", args.backend) if args.backend else ()),
        *(("--graph",) if args.graph else ()),
    ]


def event_numbers(args: argparse.Namespace) -> dict[str, Any]:
    """Run the untraced paired loop in its own process and read its JSON.

    Its own process, because a profiler attached to this one would perturb the
    interval this driver exists to explain, and because the traced runs must not
    inherit an allocator this one grew.

    Args:
        args: Parsed command line.

    Returns:
        The payload :func:`run_traced` wrote.

    Raises:
        RuntimeError: If the loop exited nonzero.
    """
    path = args.out.with_name(args.out.name + "-events.json")
    argv = [*target_argv(args, arm="both", iters=args.iters), "--json", str(path)]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError(f"event loop exited {done.returncode}: " + " | ".join(tail))
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--mode", choices=MODES, default="step")
    parser.add_argument(
        "--groups",
        choices=("one", "heads"),
        default="one",
        help="Mamba2 group count. `one` shares B and C across heads, which is "
        "Mamba2's own default and the harder number to beat.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default=None)
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Iterations of the untraced event loop. Even, as the order swap needs.",
    )
    parser.add_argument(
        "--trace-iters",
        type=int,
        default=4,
        help="Steps inside each arm's NSYS window. One arm per trace, so no swap.",
    )
    parser.add_argument(
        "--ncu-iters",
        type=int,
        default=1,
        help="Steps per NCU pass. One is enough: the pass reports every launch.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Replay both arms from a CUDA graph instead of launching them. Bounds "
        "what per-launch cost is worth: the kernels are unchanged, the launches go.",
    )
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument(
        "--serialize-with",
        type=Path,
        default=None,
        help="Lock file to hold with flock(1) while each workload process runs, so "
        "two measurements on one device do not overlap. Off by default.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/launch-gap"))
    parser.add_argument(
        "--traced",
        action="store_true",
        help="Run the workload instead of driving profilers. Set by the driver.",
    )
    parser.add_argument("--arm", choices=ARMS, default="both")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def ncu_kernel_sums(
    args: argparse.Namespace, ncu_path: str
) -> dict[str, tuple[float, float]]:
    """Per-arm kernel time and launch count, per step, from NCU.

    One pass per arm. NCU replays each kernel in isolation, so the interleave the
    event loop needs is irrelevant here, and one arm per pass removes the need to
    attribute a launch to an arm by its name -- which cannot be done for the
    elementwise kernels both arms launch.

    Args:
        args: Parsed command line.
        ncu_path: Resolved ``ncu`` binary.

    Returns:
        Arm key (``a`` or ``b``) to summed microseconds and launch count per step.
    """
    out: dict[str, tuple[float, float]] = {}
    for arm in ("a", "b"):
        pass_ = run_ncu(
            DURATION,
            target_argv(args, arm=arm, iters=args.ncu_iters),
            ncu=ncu_path,
        )
        total_ns = sum(
            invocation.values.get("gpu__time_duration.sum", 0.0)
            for invocation in pass_.invocations
        )
        out[arm] = (
            total_ns / 1e3 / args.ncu_iters,
            len(pass_.invocations) / args.ncu_iters,
        )
    return out


def report(
    args: argparse.Namespace,
    events_payload: dict[str, Any],
    splits: Sequence[ArmSplit],
) -> None:
    """Print the decomposition.

    Args:
        args: Parsed command line.
        events_payload: What the untraced loop wrote.
        splits: One per arm.
    """
    print(events_payload["shape"])
    launch = "cuda graph replay, both arms" if args.graph else "eager launches"
    print(f"mode={args.mode} dtype={args.dtype} groups={args.groups}  {launch}")
    device = events_payload["device"]
    print(f"device {device['name']} sm_{device['capability'].replace('.', '')}")
    print(f"clocks {events_payload['clocks']}")
    # A contended device produces a number, not a measurement, and the only
    # defence is to print what the probe found beside the numbers it gated.
    print(f"sharing {device['sharing']['detail']}")
    print(
        f"event loop: iters={args.iters} warmup={args.warmup}  "
        f"timer coverage {events_payload['timer_coverage_pct']:.3f}%"
    )
    loop = events_payload["loop"]
    print(
        f"loop iteration: {loop['median_duration_us']:,.1f} us median  "
        f"spread {loop['spread_pct']:.3f}%  "
        f"resolution {loop['resolution_pct']:.3f}%"
    )
    print()
    head = (
        f"{'arm':<18}{'event':>11}{'host':>11}{'dev span':>11}{'busy':>11}"
        f"{'ncu':>11}{'idle':>9}{'outside':>10}{'launch':>8}{'us/gap':>9}"
    )
    print(head)
    print("-" * len(head))
    for split in splits:
        event = "-" if split.event_us is None else f"{split.event_us:,.1f}"
        outside = (
            "-" if split.outside_span_us is None else f"{split.outside_span_us:,.1f}"
        )
        ncu = "-" if split.ncu_kernel_us is None else f"{split.ncu_kernel_us:,.1f}"
        print(
            f"{split.label:<18}{event:>11}{split.host_us:>11,.1f}"
            f"{split.device_span_us:>11,.1f}{split.busy_us:>11,.1f}{ncu:>11}"
            f"{split.idle_us:>9,.1f}{outside:>10}"
            f"{split.launch_count:>8d}{split.idle_per_launch_us:>9,.2f}"
        )
    print()
    print(
        "event: CUDA-event interval, untraced paired loop, median of "
        f"{args.iters // 2} steps per arm"
    )
    print(
        "host, dev span, busy, idle, launch: single-arm NSYS trace, median of "
        f"{args.trace_iters} steps"
    )
    print("ncu: sum of gpu__time_duration.sum over one step's launches")
    print("outside: event interval less the device span. Host time, and the events.")
    print()
    for split in splits:
        counts = f"{split.launch_count} device ops per step"
        if split.ncu_launch_count is not None:
            counts += f", {split.ncu_launch_count:,.1f} of them kernels NCU profiled"
        idle = " ".join(f"{value:,.1f}" for value in split.step_idle_us)
        print(
            f"{split.label}: {counts}; between steps "
            f"{split.between_steps_us:,.1f} us; NVTX projected "
            f"{split.projected_op_count:,.1f} of the {split.launch_count} ops"
        )
        print(
            f"{split.label}: idle per step [{idle}] us; longest single gap "
            f"{split.max_gap_us:,.1f} us"
        )
    print()
    print(events_payload["verdict"])


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the three clocks and print the decomposition.

    Returns:
        Process exit status.

    Raises:
        ValueError: If ``--iters`` is odd, which leaves one iteration's arm order
            unbalanced by any other, or if ``--trace-iters`` is below two, which
            leaves no boundary to measure the between-step wait across.
    """
    args = parse_args(argv)
    if args.traced:
        return run_traced(args)
    if args.iters % 2 != 0:
        raise ValueError(
            f"--iters must be even so the arm order swaps, got {args.iters}"
        )
    if args.trace_iters < 2:
        raise ValueError(f"--trace-iters must be at least 2, got {args.trace_iters}")
    # Both profilers are resolved before any workload runs: a profiler that is not
    # installed is an environment defect, and finding out after the event loop has
    # been paid for wastes the measurement it would have gated.
    nsys_path = resolve_tool(args.nsys)
    ncu_path = None if args.skip_ncu else resolve_tool(args.ncu)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    payload = event_numbers(args)
    labels = list(payload["arms"])
    sums = ncu_kernel_sums(args, ncu_path) if ncu_path is not None else {}
    splits: list[ArmSplit] = []
    steps: dict[str, list[dict[str, Any]]] = {}
    for key, label in zip(("a", "b"), labels, strict=True):
        base = args.out.with_name(f"{args.out.name}-{key}")
        texts = nsys_report_texts(
            target_argv(args, arm=key, iters=args.trace_iters),
            base,
            (GPU_TRACE, NVTX_PROJ),
            nsys=nsys_path,
            trace="cuda,nvtx",
        )
        for name in (GPU_TRACE, NVTX_PROJ):
            base.with_name(f"{base.name}-{name}.csv").write_text(
                texts[name], encoding="utf-8"
            )
        split, parts = split_arm(
            label,
            parse_gpu_events(texts[GPU_TRACE]),
            parse_nvtx_projection(texts[NVTX_PROJ]),
            event_us=payload["arms"][label]["total"]["median_duration_us"],
            steps=args.trace_iters,
        )
        if key in sums:
            kernel_us, launches = sums[key]
            split = replace(split, ncu_kernel_us=kernel_us, ncu_launch_count=launches)
        splits.append(split)
        # Per step, not only the median: an arm whose first traced step holds one
        # large gap and whose later steps hold none has a median that describes the
        # steady state and a maximum that describes neither.
        steps[label] = [asdict(part) for part in parts]
    report(args, payload, splits)
    summary = args.out.with_name(args.out.name + "-split.json")
    summary.write_text(
        json.dumps(
            {
                "events": payload,
                "arms": [asdict(s) for s in splits],
                "steps": steps,
                "trace_iters": args.trace_iters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
