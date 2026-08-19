"""Per-kernel profile of one operator, cross-checked three ways.

Runs three clocks over the same workload and refuses to emit if they disagree:

1. CUDA events in this process, giving the per-iteration wall and its spread.
2. NSYS over ``scripts/perf/profile_target.py``, giving the launch stream.
3. NCU over the same target, one pass per counter table, giving the counters.

NCU is slow: it replays every kernel once per pass, eight passes deep. Keep
``--iters`` small; the wall comes from the event bench, not from the profiler.

    python3 scripts/perf/profile_op.py --shape standard --mode step

``--op`` picks the operator, out of :data:`slinoss.perf.workload.OPS`. The report
base is named from ``--out``, the shape, and the mode, so a second operator at one
shape and mode needs its own ``--out`` or it overwrites the first.

    python3 scripts/perf/profile_op.py --op conv --shape standard --mode step \\
        --out out/profile-conv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.ceiling import ceilings
from slinoss.perf.declared import ClassAudit, class_audit
from slinoss.perf.device import device_info, device_ordinal, require_cuda
from slinoss.perf.ncu import NCU_TABLES, NcuPass, kernel_counters, run_ncu
from slinoss.perf.nsys import run_nsys
from slinoss.perf.report import Report, agreement, write_report
from slinoss.perf.timing import Throughput, measure
from slinoss.perf.workload import (
    BLOCK,
    CONV,
    MIXER,
    OPS,
    SCANPREP,
    SHAPE_NAMES,
    BlockShape,
    ConvShape,
    MixerShape,
    OpShape,
    PrepShape,
    block_forward_only,
    block_shape_by_name,
    block_step,
    conv_forward_only,
    conv_shape_by_name,
    conv_step,
    forward_only,
    make_block_inputs,
    make_conv_inputs,
    make_inputs,
    make_mixer_inputs,
    make_prep_inputs,
    mixer_forward_only,
    mixer_shape_by_name,
    mixer_step,
    prep_forward_only,
    prep_shape_by_name,
    prep_step,
    shape_by_name,
    step,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "step")
TARGET = Path(__file__).with_name("profile_target.py")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPS, default=OPS[0])
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="standard")
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
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, cuda or cuda:N. There is no host path: every "
        "report names the part the numbers came from.",
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument(
        "--d-head",
        type=int,
        default=0,
        help="Rows per head for the conv output layout, or 0 for token-major. "
        "Forwarded to the target, so the counters and the event wall cover one "
        "layout. Ignored by every other operator.",
    )
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
    # An operator and a backend are named only when one is asked for, so the
    # quoted command is what a reader would have to type and no more.
    if args.op != OPS[0]:
        argv += ["--op", args.op]
    if args.backend is not None:
        argv += ["--backend", args.backend]
    # The layout has to reach the profiled process, or the counters describe the
    # token-major kernel while the event wall describes the head-major one.
    if args.d_head:
        argv += ["--d-head", str(args.d_head)]
    return argv


def build_workload(
    args: argparse.Namespace, device: torch.device
) -> tuple[
    OpShape | ConvShape | PrepShape | BlockShape | MixerShape, Callable[[], None]
]:
    """Resolve the shape and build the event-bench runner for ``--op``.

    The same dispatch runs in :mod:`scripts.perf.profile_target`, so the event
    wall and the profiled window cover one workload.

    Args:
        args: Parsed command line.
        device: Device to allocate on.

    Returns:
        The shape record and the workload callable.
    """
    grads = args.mode == "step"
    dtype = DTYPES[args.dtype]
    if args.op == CONV:
        conv_shape = conv_shape_by_name(args.shape)
        conv = make_conv_inputs(
            conv_shape,
            device,
            dtype=dtype,
            requires_grad=grads,
            d_head=args.d_head or None,
        )
        runner = (
            conv_step(conv, backend=args.backend)
            if grads
            else conv_forward_only(conv, backend=args.backend)
        )
        return conv_shape, runner
    if args.op == SCANPREP:
        prep_shape = prep_shape_by_name(args.shape)
        prep = make_prep_inputs(prep_shape, device, dtype=dtype, requires_grad=grads)
        return prep_shape, (
            prep_step(prep, prep_shape, backend=args.backend)
            if grads
            else prep_forward_only(prep, prep_shape, backend=args.backend)
        )
    if args.op == BLOCK:
        block_shape = block_shape_by_name(args.shape)
        block = make_block_inputs(block_shape, device, dtype=dtype, requires_grad=grads)
        return block_shape, (
            block_step(block, block_shape, backend=args.backend)
            if grads
            else block_forward_only(block, block_shape, backend=args.backend)
        )
    if args.op == MIXER:
        mixer_shape = mixer_shape_by_name(args.shape)
        mixer = make_mixer_inputs(mixer_shape, device, dtype=dtype, requires_grad=grads)
        return mixer_shape, (
            mixer_step(mixer, mixer_shape, backend=args.backend)
            if grads
            else mixer_forward_only(mixer, mixer_shape, backend=args.backend)
        )
    shape = shape_by_name(args.shape)
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=grads)
    return shape, (
        step(inputs, shape.chunk, backend=args.backend)
        if grads
        else forward_only(inputs, shape.chunk, backend=args.backend)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, profile, cross-check, and emit.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If NCU returned a table with a metric absent, which means the
            metric name is wrong for this driver version, or if a profiled kernel
            this repo compiles declares no class.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    shape, runner = build_workload(args, device)
    label = f"{args.op} {shape.name} {args.mode}"
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
    # The inputs live in the runner's closure and the profilers need the memory
    # for a second process at the same shape.
    del runner

    argv_target = target_argv(args)
    notes = [
        shape.describe(),
        f"mode={args.mode} dtype={args.dtype} backend={args.backend or 'auto'}"
        + (f" d_head={args.d_head}" if args.d_head else ""),
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

    audit: ClassAudit | None = None
    if kernels:
        audit = class_audit(
            kernels,
            limits=limits,
            step_duration_us=timed.total.median_duration_us,
            capture_iters=args.iters,
        )
        if audit.unjudged:
            notes.append(
                "unjudged kernels, not compiled by this repo: "
                + ", ".join(audit.unjudged)
            )

    report = Report(
        title=f"profile: {label}",
        device=device_info(device_ordinal(device)),
        agreement=check,
        budget=tree,
        throughput=(Throughput.of(label, shape.token_count, timed.total),),
        ceilings=limits,
        kernels=kernels,
        verdicts=() if audit is None else audit.verdicts,
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
