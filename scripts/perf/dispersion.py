"""Measure the harness itself: which dispersion statistic bounds a delta.

Repeats one workload unchanged and reports two things:

1. How each statistic moves with the sample count, from prefixes of the first run.
2. Whether the resolution floor covers the scatter of medians across the runs.

Nothing here compares two implementations. A floor that fails the second check
invalidates every delta measured against it, so this is run once per host and
after any change to the timing path.

    python3 scripts/perf/dispersion.py --shape standard --mode step
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from slinoss.perf.device import device_info, device_ordinal
from slinoss.perf.dispersion import growth, repeats
from slinoss.perf.report import Report, write_report
from slinoss.perf.timing import measure
from slinoss.perf.units import CONFIDENCE_PCT, MIN_RESOLVING_SAMPLES
from slinoss.perf.workload import SHAPES, forward_only, make_inputs, shape_by_name, step

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "step")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=[s.name for s in SHAPES], default="standard")
    parser.add_argument("--mode", choices=MODES, default="step")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5, help="Prefix stride.")
    parser.add_argument("--repeat", type=int, default=5, help="Independent runs.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--out", type=Path, default=Path("out/dispersion"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study.

    Returns:
        Process exit status. Nonzero if the floor did not cover the scatter.

    Raises:
        RuntimeError: If the requested device is CUDA and CUDA is unavailable.
    """
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda needs CUDA")
    shape = shape_by_name(args.shape)
    dtype = DTYPES[args.dtype]
    grads = args.mode == "step"
    label = f"so3ssd {shape.name} {args.mode}"
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=grads)
    body = (
        step(inputs, shape.chunk, backend=args.backend)
        if grads
        else forward_only(inputs, shape.chunk, backend=args.backend)
    )
    # Warmup runs before every measurement, so each is an independent run of the
    # same steady state rather than a later slice of one long run.
    runs = [
        measure(
            body,
            label=f"{label} run {index}",
            iters=args.iters,
            warmup=args.warmup,
            device=device,
        )
        for index in range(args.repeat)
    ]
    rows = growth(runs[0].total.samples_duration_us, args.stride)
    scatter = repeats(label, [t.total for t in runs])
    report = Report(
        title=f"dispersion: {label}",
        device=device_info(device_ordinal(device)),
        growth=rows,
        scatter=scatter,
        notes=(
            shape.describe(),
            f"mode={args.mode} dtype={args.dtype} backend={args.backend or 'default'}",
            f"iters={args.iters} warmup={args.warmup} repeat={args.repeat}",
            f"timer={runs[0].timer} clocks={runs[0].clocks}",
            f"confidence_pct={CONFIDENCE_PCT:.1f} "
            f"min_resolving_samples={MIN_RESOLVING_SAMPLES}",
            "growth rows are prefixes of run 0 and share their samples",
        ),
    )
    md, _ = write_report(report, args.out, require_agreement=False)
    print(f"wrote {md}")
    print()
    print(
        f"{'sample_count':>12} {'median_us':>12} {'spread_pct':>11} "
        f"{'resolution_pct':>15} {'coverage_pct':>13} {'resolves':>9}"
    )
    for row in rows:
        print(
            f"{row.sample_count:>12} {row.median_duration_us:>12,.3f} "
            f"{row.spread_pct:>11,.3f} {row.resolution_pct:>15,.3f} "
            f"{row.coverage_pct:>13,.3f} {'yes' if row.resolves else 'no':>9}"
        )
    print()
    # Two medians separated by less than the sum of their half-widths are one
    # measurement, so the scatter is judged against twice the floor.
    print(
        f"{scatter.run_count} runs of {scatter.sample_count} samples: "
        f"median-to-median scatter {scatter.scatter_pct:,.3f}%, "
        f"floor {scatter.floor_pct:,.3f}% at {scatter.coverage_pct:,.3f}% coverage, "
        f"budget {2.0 * scatter.floor_pct:,.3f}%, "
        f"widest range {scatter.spread_pct:,.3f}%"
    )
    for index, run in enumerate(runs):
        print(f"  run {index}: median {run.total.median_duration_us:,.3f} us")
    if not scatter.floor_holds:
        print(
            f"floor {scatter.floor_pct:,.3f}% at {scatter.coverage_pct:,.3f}% "
            f"coverage does not cover the observed scatter "
            f"{scatter.scatter_pct:,.3f}%; no delta measured against it is a result"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
