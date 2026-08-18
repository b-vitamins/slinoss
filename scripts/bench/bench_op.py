"""Bench the SO(3) scan operator with CUDA events.

One event pair per region per iteration, so the reported spread is the real
run-to-run dispersion and a delta smaller than it is not a result. No profiler is
attached, so this is the cheap measurement to run before and after every change.

Emits one report per shape and mode under ``--out``, and a summary table on
stdout. The reports carry no cross-check, because only one clock ran; the
per-kernel picture and the three-way agreement come from
``scripts/perf/profile_op.py``.

    python3 scripts/bench/bench_op.py --shape standard --mode both
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.ceiling import Ceilings, ceilings
from slinoss.perf.device import device_info, device_ordinal
from slinoss.perf.memory import (
    SavedStorages,
    SavedTensorProbe,
    memory_peaks,
    reset_memory_peaks,
)
from slinoss.perf.report import Report, write_report
from slinoss.perf.timing import Throughput, measure
from slinoss.perf.workload import (
    SHAPES,
    OpShape,
    forward_only,
    make_inputs,
    shape_by_name,
    step,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "step")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        choices=[s.name for s in SHAPES],
        help="Shape to bench. Repeatable. Defaults to every standard shape.",
    )
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--out", type=Path, default=Path("out/bench-op"))
    parser.add_argument(
        "--no-ceilings",
        action="store_true",
        help="Skip the measured DRAM and tensor ceilings.",
    )
    return parser.parse_args(argv)


def _saved(
    shape: OpShape,
    device: torch.device,
    dtype: torch.dtype,
    backend: str | None,
) -> SavedStorages:
    """Probe what autograd holds for one forward at this shape.

    Runs under a recorder so each save attributes to the region it was taken in.
    Without one every row would read ``unattributed``, which says nothing about
    which part of the graph holds the bytes.
    """
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        measure(
            step(inputs, shape.chunk, backend=backend),
            label=f"so3ssd {shape.name} saved",
            iters=1,
            warmup=0,
            device=device,
        )
    return probe.report(f"so3ssd {shape.name}", inputs.differentiable)


def bench(
    shape: OpShape,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    limits: Ceilings | None,
) -> tuple[Report, Throughput]:
    """Measure one shape in one mode and build its report."""
    dtype = DTYPES[args.dtype]
    grads = mode == "step"
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=grads)
    runner = (
        step(inputs, shape.chunk, backend=args.backend)
        if grads
        else forward_only(inputs, shape.chunk, backend=args.backend)
    )
    label = f"so3ssd {shape.name} {mode}"
    reset_memory_peaks(device)
    timed = measure(
        runner,
        label=label,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    peaks = memory_peaks(label, device)
    tree = budget(timed)
    assert_closed(tree)
    rate = Throughput.of(label, shape.token_count, timed.total)
    report = Report(
        title=f"bench: {label}",
        device=device_info(device_ordinal(device)),
        budget=tree,
        throughput=(rate,),
        ceilings=limits,
        saved=_saved(shape, device, dtype, args.backend) if grads else None,
        peaks=peaks,
        notes=(
            shape.describe(),
            f"mode={mode} dtype={args.dtype} backend={args.backend or 'auto'}",
            f"iters={args.iters} warmup={args.warmup}",
            f"timer={timed.timer} clocks={timed.clocks}",
        ),
    )
    return report, rate


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is CUDA and CUDA is unavailable.
    """
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda needs CUDA")
    shapes = [shape_by_name(n) for n in (args.shape or [s.name for s in SHAPES])]
    modes = MODES if args.mode == "both" else (args.mode,)
    limits = None if args.no_ceilings else ceilings(device)
    rows: list[tuple[str, Throughput]] = []
    for shape in shapes:
        for mode in modes:
            report, rate = bench(shape, mode, args, device, limits)
            base = args.out.with_name(f"{args.out.name}-{shape.name}-{mode}")
            md, _ = write_report(report, base, require_agreement=False)
            rows.append((f"{shape.name}/{mode}", rate))
            print(f"wrote {md}")
    print()
    print(
        f"{'config':<20} {'duration_us':>14} {'spread_pct':>11} "
        f"{'resolution_pct':>15} {'coverage_pct':>13} {'tps':>14}"
    )
    for name, rate in rows:
        print(
            f"{name:<20} {rate.duration_us:>14,.3f} {rate.spread_pct:>11,.3f} "
            f"{rate.resolution_pct:>15,.3f} {rate.coverage_pct:>13,.3f} "
            f"{rate.throughput_tps:>14,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
