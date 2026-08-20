"""Per-kernel profile of one operator, cross-checked three ways.

Runs three clocks over the same workload and refuses to emit if they disagree:

1. CUDA events in this process, giving the per-iteration wall and its spread.
2. NSYS over ``scripts/perf/profile_target.py``, giving the launch stream.
3. NCU over the same target, one pass per counter table, giving the counters.

NCU is slow: it replays every kernel once per pass, nine passes deep. Keep
``--iters`` small; the wall comes from the event bench, not from the profiler.

A DRAM-bound kernel is scored against a time floor measured in this process at
the kernel's own traffic, not against the rate of the largest copy the device can
run. The ninth pass is the local-memory one, and it is not optional: the spill
rule fails a kernel whatever its percentage, so a pass that was never run must
not read as clean.

A failing audit exits nonzero. The class floor, the spill rule, the occupancy rule
and the block-count floor are gates, so a violation is an exit status a sweep or a
CI step can act on rather than a line in a file nobody read. The report is written
either way: a refused emission would leave the failing measurement unreadable.

An audit that judged nothing exits nonzero too, and this is not the same rule. Every
gate above is a statement about a kernel the capture held, so a capture holding no
kernel passes all of them: a conv audit exited zero having judged nothing, the
compiled extension never having been built in that environment, so the operator
resolved to its reference and thirteen torch kernels were reported as unjudged. Three
checks close that, in the order they can be answered:

1. Dispatch, before any profiler runs. Every registry the operator selects through is
   asked what it resolves for this device and dtype, and a reference answer ends the
   run: nine NCU passes over a torch path cost the same as nine over a kernel path.
2. The profiler paths, also before the workload. A profiler that is not installed is
   an environment defect, and finding that out after the event bench and the ceiling
   fits wastes the measurement it would have gated.
3. Coverage, after the audit. What was judged is compared against the kernels
   :data:`slinoss.perf.coverage.COVERAGE` says the arm launches, so a capture short
   by one launch is an exit status and not a shorter table.

The tree the measured package came out of is stamped in the report beside those
three. It is not a fourth check: calling a tree wrong needs a declared expected
tree, and nothing here records one.

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
from collections.abc import Sequence
from pathlib import Path

import torch

from slinoss.perf.arms import op_arm
from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.ceiling import ceilings, dram_time_floor
from slinoss.perf.coverage import (
    MODES,
    TARGETED,
    coverage_verdict,
    tree_provenance,
    unreachable,
)
from slinoss.perf.declared import FloorAudit, floor_audit
from slinoss.perf.device import device_info, device_ordinal, require_cuda
from slinoss.perf.dispatch import dispatch_verdict
from slinoss.perf.ncu import (
    NCU_TABLES,
    SPILL_TABLE,
    NcuPass,
    SpillCounters,
    kernel_counters,
    run_ncu,
    spill_counters,
)
from slinoss.perf.nsys import run_nsys
from slinoss.perf.report import Report, agreement, write_report
from slinoss.perf.timing import Throughput, measure
from slinoss.perf.tools import resolve_tool
from slinoss.perf.traffic import traffic_mix
from slinoss.perf.workload import OPS, SHAPE_NAMES

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
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
    parser.add_argument(
        "--ncu",
        default="ncu",
        help="Path to ncu, or a bare name resolved through PATH and then the CUDA "
        "bin directories. Resolved before the workload is allocated, so a profiler "
        "that is not installed costs no measurement.",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="NCU kernel-name regex. Narrows every pass to the matching "
        "kernels, which is what makes a per-kernel question answerable without "
        "replaying the whole step. The cross-check and the class audit then "
        "cover only what was profiled, and the report says so.",
    )
    parser.add_argument("--nsys", default="nsys", help="Path to nsys. See --ncu.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, default=Path("out/profile-op"))
    parser.add_argument(
        "--skip-ncu",
        action="store_true",
        help="Emit without the counter tables. The cross-check cannot run, no "
        "kernel is judged, and the run exits nonzero on the coverage rule: a "
        "report with no verdict in it is not a pass.",
    )
    parser.add_argument(
        "--skip-nsys",
        action="store_true",
        help="Emit without the launch stream. The cross-check cannot run; the "
        "counters and the audit are unaffected.",
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


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, profile, cross-check, emit, and judge.

    Returns:
        Process exit status: zero when the operator dispatched to its kernels, the
        capture held every kernel the arm launches, and every one of them cleared
        every rule the audit applied. One when any of the three failed, the run that
        judged nothing included.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ToolNotFoundError: If a profiler that was not skipped is on neither PATH nor
            any CUDA bin directory. Raised before the workload is allocated.
        ValueError: If NCU returned a table with a metric absent, which means the
            metric name is wrong for this driver version, or if a profiled kernel
            this repo compiles declares no class.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    tree = tree_provenance(Path(__file__))
    # Before the workload and before the profilers. A reference dispatch launches no
    # declared kernel, so every rule below it would hold over torch's kernels, and a
    # profiler that is not installed cannot judge what the bench would measure.
    chosen = dispatch_verdict(
        args.op,
        device_type=device.type,
        dtype=DTYPES[args.dtype],
        backend=args.backend,
    )
    if not chosen.passed:
        print(f"dispatch failure: {chosen.detail}")
        return 1
    ncu_path = None if args.skip_ncu else resolve_tool(args.ncu)
    nsys_path = None if args.skip_nsys else resolve_tool(args.nsys)
    arm = op_arm(
        args.op,
        args.shape,
        device,
        dtype=DTYPES[args.dtype],
        grads=args.mode == "step",
        d_head=args.d_head or None,
    )
    shape = arm.shape
    runner = arm.run(args.backend, arm.prefix)
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
    floor = dram_time_floor(device)
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
        f"dram floor: c={floor.fixed_duration_us:.3f} us "
        f"B={floor.asymptotic_gbs:.2f} GB/s "
        f"max residual={floor.max_residual_pct:.2f}% "
        f"l2={floor.l2_bytes} B",
        f"dispatch: {chosen.detail}",
        f"profilers: ncu={ncu_path or 'skipped'} nsys={nsys_path or 'skipped'}",
        f"tree: package={tree.package_root} driver={tree.driver_root} "
        f"same={tree.same_tree} extension={tree.extension} {tree.extension_stamp}",
    ]
    # Read every run, not only the run that adds one: an excuse nobody reads is how a
    # kernel stops being profiled at all.
    notes += [
        f"declared and driven elsewhere: {one.kernel} by {one.driver}, {one.reason}"
        for one in TARGETED
    ]
    orphans = unreachable()
    if orphans:
        notes.append(
            "declared and reached by nothing, so the class is a claim with no gate: "
            + ", ".join(orphans)
        )

    trace = None
    if nsys_path is not None:
        base = args.out.with_name(f"{args.out.name}-{shape.name}-{args.mode}")
        base.parent.mkdir(parents=True, exist_ok=True)
        trace = run_nsys(argv_target, base, label=label, nsys=nsys_path)
        notes.append(f"nsys report: {trace.report_path}")

    passes: list[NcuPass] = []
    spills: tuple[SpillCounters, ...] = ()
    narrow = () if args.kernel is None else ("--kernel-name", f"regex:{args.kernel}")
    if narrow:
        notes.append(
            f"ncu narrowed to kernels matching {args.kernel!r}; the cross-check "
            f"and the class audit cover only those"
        )
    if ncu_path is not None:
        for table in (*NCU_TABLES, SPILL_TABLE):
            one = run_ncu(table, argv_target, ncu=ncu_path, extra=narrow)
            if one.missing_metrics:
                raise ValueError(
                    f"ncu table {table.name!r} returned no value for "
                    f"{list(one.missing_metrics)}; the metric names are wrong for "
                    f"this driver"
                )
            print(f"ncu {table.name}: {len(one.invocations)} launches")
            # The spill pass feeds the spill rule, not the counter merge: its
            # duration column exists only to key the records to kernels.
            if table is SPILL_TABLE:
                spills = spill_counters(one)
            else:
                passes.append(one)

    kernels = kernel_counters(passes) if passes else ()
    check = None
    # A narrowed capture holds a subset of the launches the trace holds, so the
    # three clocks cannot be made to agree and asking them to would fail on the
    # narrowing rather than on a disagreement.
    if trace is not None and kernels and not narrow:
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

    audit: FloorAudit | None = None
    if kernels:
        audit = floor_audit(
            kernels,
            floor=floor,
            spills=spills,
            step_duration_us=timed.total.median_duration_us,
            capture_iters=args.iters,
            # The part the ceilings were measured on, so the block floor and the
            # denominators come from one device record.
            device=limits.device,
        )
        if audit.unjudged:
            notes.append(
                "unjudged kernels, not compiled by this repo: "
                + ", ".join(audit.unjudged)
            )
        if audit.spilled:
            notes.append(
                "failed by the spill rule, whatever the percentage: "
                + ", ".join(audit.spilled)
            )
        if audit.cached:
            notes.append(
                "no dram verdict, per-launch traffic within L2 so the counters "
                "describe the cache: " + ", ".join(audit.cached)
            )
        notes += [f"audit failure: {line}" for line in audit.failures]

    # Judged, not captured: a kernel the capture held and the audit could not judge is
    # not covered, and the whole point of the rule is that the count is of verdicts.
    covered = coverage_verdict(
        args.op,
        args.mode,
        () if audit is None else audit.judged,
        narrowed=bool(narrow),
    )
    notes.append(f"coverage: {covered.detail}")

    report = Report(
        title=f"profile: {label}",
        device=device_info(device_ordinal(device)),
        agreement=check,
        budget=tree,
        throughput=(Throughput.of(label, shape.token_count, timed.total),),
        ceilings=limits,
        kernels=kernels,
        verdicts=() if audit is None else audit.verdicts,
        geometry=() if audit is None else audit.geometry,
        traffic=traffic_mix(kernels),
        coverage=covered,
        dispatch=chosen,
        provenance=tree,
        spills=spills,
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
    failures = (
        [] if audit is None else [f"audit failure: {one}" for one in audit.failures]
    )
    if not covered.passed:
        failures.append(f"coverage failure: {covered.detail}")
    for line in failures:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
