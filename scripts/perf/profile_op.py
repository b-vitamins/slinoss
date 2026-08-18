"""Per-kernel profile of the SO(3) scan, cross-checked three ways.

Runs three clocks over the same workload and refuses to emit if they disagree:

1. CUDA events in this process, giving the per-iteration wall and its spread.
2. NSYS over ``scripts/perf/profile_target.py``, giving the launch stream.
3. NCU over the same target, one pass per counter table, giving the counters.

NCU is slow: it replays every kernel once per pass, six passes deep. Keep
``--iters`` small; the wall comes from the event bench, not from the profiler.

    python3 scripts/perf/profile_op.py --shape standard --mode step
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.ceiling import ceilings
from slinoss.perf.device import device_info, device_ordinal
from slinoss.perf.ncu import NCU_TABLES, NcuPass, kernel_counters, run_ncu
from slinoss.perf.nsys import run_nsys
from slinoss.perf.report import Report, agreement, write_report
from slinoss.perf.timing import Throughput, measure
from slinoss.perf.workload import (
    SHAPES,
    forward_only,
    make_inputs,
    shape_by_name,
    step,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "step")
TARGET = Path(__file__).with_name("profile_target.py")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=[s.name for s in SHAPES], default="standard")
    parser.add_argument("--mode", choices=MODES, default="step")
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Iterations inside the capture window, and the divisor that puts "
        "the profiler sums on a per-iteration footing.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--event-iters", type=int, default=30)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, default=Path("out/profile-op"))
    parser.add_argument(
        "--skip-ncu",
        action="store_true",
        help="Emit without the counter tables. The cross-check cannot run, and "
        "the report says so.",
    )
    parser.add_argument(
        "--skip-nsys",
        action="store_true",
        help="Emit without the launch stream. Same consequence.",
    )
    return parser.parse_args(argv)


def target_argv(args: argparse.Namespace) -> list[str]:
    """The command the profilers run."""
    argv = [
        args.python,
        str(TARGET),
        "--shape",
        args.shape,
        "--mode",
        args.mode,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    if args.backend is not None:
        argv += ["--backend", args.backend]
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, profile, cross-check, and emit.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If CUDA is unavailable.
        ValueError: If NCU returned a table with a metric absent, which means the
            metric name is wrong for this driver version.
    """
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("profile_op needs CUDA")
    shape = shape_by_name(args.shape)
    grads = args.mode == "step"
    inputs = make_inputs(shape, device, dtype=DTYPES[args.dtype], requires_grad=grads)
    runner = (
        step(inputs, shape.chunk, backend=args.backend)
        if grads
        else forward_only(inputs, shape.chunk, backend=args.backend)
    )
    label = f"so3ssd {shape.name} {args.mode}"
    timed = measure(
        runner,
        label=label,
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
    )
    tree = budget(timed)
    assert_closed(tree)
    limits = ceilings(device)
    del inputs, runner

    argv_target = target_argv(args)
    notes = [
        shape.describe(),
        f"mode={args.mode} dtype={args.dtype} backend={args.backend or 'auto'}",
        f"event iters={args.event_iters} capture iters={args.iters}",
        f"timer={timed.timer} clocks={timed.clocks}",
        "target: " + " ".join(argv_target),
    ]

    trace = None
    if not args.skip_nsys:
        base = args.out.with_name(f"{args.out.name}-{shape.name}-{args.mode}")
        base.parent.mkdir(parents=True, exist_ok=True)
        trace = run_nsys(argv_target, base, label=label, nsys=args.nsys)
        notes.append(f"nsys report: {trace.report_path}")

    passes: list[NcuPass] = []
    if not args.skip_ncu:
        for table in NCU_TABLES:
            one = run_ncu(table, argv_target, ncu=args.ncu)
            if one.missing_metrics:
                raise ValueError(
                    f"ncu table {table.name!r} returned no value for "
                    f"{list(one.missing_metrics)}; the metric names are wrong for "
                    f"this driver"
                )
            passes.append(one)
            print(f"ncu {table.name}: {len(one.invocations)} launches")

    kernels = kernel_counters(passes) if passes else ()
    check = None
    if trace is not None and kernels:
        check = agreement(
            label,
            event=timed.total,
            trace=trace,
            kernels=kernels,
            capture_iters=args.iters,
        )
    else:
        notes.append(
            "cross-check skipped: it needs the event wall, the nsys trace, and "
            "the ncu tables together"
        )

    report = Report(
        title=f"profile: {label}",
        device=device_info(device_ordinal(device)),
        agreement=check,
        budget=tree,
        throughput=(Throughput.of(label, shape.token_count, timed.total),),
        ceilings=limits,
        kernels=kernels,
        trace=trace,
        notes=tuple(notes),
    )
    md, js = write_report(
        report,
        args.out.with_name(f"{args.out.name}-{shape.name}-{args.mode}"),
        require_agreement=check is not None,
    )
    print(f"wrote {md}")
    print(f"wrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
