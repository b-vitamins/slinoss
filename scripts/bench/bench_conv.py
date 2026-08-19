"""Bench the causal depthwise conv1d with CUDA events.

One event pair per region per iteration, so the reported spread is the real
run-to-run dispersion and a delta smaller than it is not a result. No profiler is
attached, so this is the cheap measurement to run before and after every change.

Emits one report per shape and mode under ``--out``, and a summary table on
stdout. The reports carry no cross-check, because only one clock ran; the
per-kernel picture and the three-way agreement come from
``scripts/perf/profile_op.py --op conv``.

    python3 scripts/bench/bench_conv.py --shape standard --mode both

The operator is declared ``DRAM-bound`` in both directions. This script measures
the wall and the DRAM ceiling; the achieved bandwidth that the declaration is
held to is a counter, and it comes from the NCU report.

Two backends are compared in one loop, not in two runs. Run-to-run medians on a
shared host scatter further than either run's own floor, so a difference of two
separately measured medians resolves nothing; ``--against`` pairs the two arms
inside every iteration instead and judges the per-iteration difference. Both arms
read the same input tensors.

    python3 scripts/bench/bench_conv.py --shape standard --mode step \\
        --backend reference --against native

``--against same`` runs the identical backend in both arms. That is the null test
of the comparison itself: a verdict of ``resolves`` there is a broken harness, not
a speedup.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.ceiling import Ceilings, ceilings
from slinoss.perf.device import device_info, device_ordinal, require_cuda
from slinoss.perf.dispersion import PairedRow
from slinoss.perf.memory import (
    SavedStorages,
    SavedTensorProbe,
    memory_peaks,
    reset_memory_peaks,
)
from slinoss.perf.report import Report, rate_table, write_report
from slinoss.perf.timing import Throughput, measure, measure_paired
from slinoss.perf.workload import (
    CONV_SHAPES,
    ConvShape,
    conv_forward_only,
    conv_shape_by_name,
    conv_step,
    make_conv_inputs,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "step")
SAME = "same"
"""``--against`` value that puts the same backend in both arms. The null test."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        choices=[s.name for s in CONV_SHAPES],
        help="Shape to bench. Repeatable. Defaults to every standard shape.",
    )
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, cuda or cuda:N. There is no host path: every "
        "report names the part the numbers came from.",
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument(
        "--against",
        default=None,
        help=(
            f"Second backend, measured against --backend inside one loop. "
            f"{SAME!r} runs --backend in both arms, which is the null test of the "
            f"comparison itself. Needs an even --iters."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("out/bench-conv"))
    parser.add_argument(
        "--no-ceilings",
        action="store_true",
        help="Skip the measured DRAM and tensor ceilings.",
    )
    return parser.parse_args(argv)


def _saved(
    shape: ConvShape,
    device: torch.device,
    dtype: torch.dtype,
    backend: str | None,
) -> SavedStorages:
    """Probe what autograd holds for one forward at this shape.

    Runs under a recorder so each save attributes to the region it was taken in.
    Without one every row would read ``unattributed``, which says nothing about
    which part of the graph holds the bytes.
    """
    inputs = make_conv_inputs(shape, device, dtype=dtype, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        measure(
            conv_step(inputs, backend=backend),
            label=f"conv {shape.name} saved",
            iters=1,
            warmup=0,
            device=device,
        )
    return probe.report(f"conv {shape.name}", inputs.differentiable)


def bench(
    shape: ConvShape,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    limits: Ceilings | None,
) -> tuple[Report, Throughput]:
    """Measure one shape in one mode and build its report."""
    dtype = DTYPES[args.dtype]
    grads = mode == "step"
    inputs = make_conv_inputs(shape, device, dtype=dtype, requires_grad=grads)
    runner = (
        conv_step(inputs, backend=args.backend)
        if grads
        else conv_forward_only(inputs, backend=args.backend)
    )
    label = f"conv {shape.name} {mode}"
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


def arm_labels(a: str | None, b: str | None) -> tuple[str, str]:
    """Region prefixes for two backends measured in one loop.

    Args:
        a: Baseline backend name, or None for the fastest registered one.
        b: Backend name for the arm under test.

    Returns:
        The two labels, in arm order. Equal backends get a side suffix, because
        :func:`slinoss.perf.timing.measure_paired` refuses one label for both arms
        and the null test still has to be runnable.
    """
    labels = [f"conv-{name or 'auto'}" for name in (a, b)]
    if labels[0] == labels[1]:
        return f"{labels[0]}-a", f"{labels[1]}-b"
    return labels[0], labels[1]


def compare_backends(
    shape: ConvShape,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    limits: Ceilings | None,
) -> tuple[Report, PairedRow]:
    """Measure two backends against each other in one loop at one shape and mode.

    Args:
        shape: The problem size.
        mode: ``forward`` or ``step``.
        args: Parsed command line. ``--backend`` is the baseline arm and
            ``--against`` the arm under test.
        device: Device to time on.
        limits: Measured ceilings, or None.

    Returns:
        The report and the verdict on the per-iteration differences.
    """
    dtype = DTYPES[args.dtype]
    grads = mode == "step"
    against = args.backend if args.against == SAME else args.against
    a_label, b_label = arm_labels(args.backend, against)
    # One input set for both arms. Two would differ in address and in cache
    # residency, and that difference would be attributed to the backend.
    inputs = make_conv_inputs(shape, device, dtype=dtype, requires_grad=grads)

    def arm(backend: str | None, prefix: str) -> Callable[[], None]:
        if grads:
            return conv_step(inputs, backend=backend, prefix=prefix)
        return conv_forward_only(inputs, backend=backend, prefix=prefix)

    label = f"conv {shape.name} {mode} paired"
    reset_memory_peaks(device)
    out = measure_paired(
        a_label,
        arm(args.backend, a_label),
        b_label,
        arm(against, b_label),
        label=label,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    peaks = memory_peaks(label, device)
    tree = budget(out.timed)
    assert_closed(tree)
    report = Report(
        title=f"bench: {label}",
        device=device_info(device_ordinal(device)),
        budget=tree,
        throughput=tuple(
            Throughput.of(name, shape.token_count, out.timed.region(name).spread)
            for name in (a_label, b_label)
        ),
        comparisons=(out.comparison,),
        ceilings=limits,
        peaks=peaks,
        notes=(
            shape.describe(),
            f"mode={mode} dtype={args.dtype}",
            f"arm a={a_label} b={b_label}, one loop, order swapped each iteration",
            f"iters={args.iters} warmup={args.warmup}",
            f"timer={out.timed.timer} clocks={out.timed.clocks}",
        ),
    )
    return report, out.comparison


def _run_comparisons(
    shapes: Sequence[ConvShape],
    modes: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    limits: Ceilings | None,
) -> int:
    """Run every paired comparison and print the verdicts.

    Returns:
        Process exit status. Nonzero when both arms held the same backend and a
        verdict still resolved a difference: that is the comparison measuring the
        arm order or the loop rather than the backend.
    """
    null_test = args.against in (SAME, args.backend)
    rates: list[tuple[str, Throughput]] = []
    verdicts: list[PairedRow] = []
    for shape in shapes:
        for mode in modes:
            report, row = compare_backends(shape, mode, args, device, limits)
            base = args.out.with_name(f"{args.out.name}-{shape.name}-{mode}-paired")
            md, _ = write_report(report, base, require_agreement=False)
            rates += [
                (f"{shape.name}/{mode}/{rate.label}", rate)
                for rate in report.throughput
            ]
            verdicts.append(row)
            print(f"wrote {md}")
    print()
    print(rate_table(rates, width=48))
    print()
    for row in verdicts:
        print(row.verdict())
    if null_test:
        broken = [row.label for row in verdicts if row.resolves]
        if broken:
            print(
                f"both arms ran the same backend and {broken} still resolve a "
                f"difference; the comparison is measuring the arm order or the loop, "
                f"not the backend"
            )
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench.

    Returns:
        Process exit status. Nonzero when a null comparison resolves a difference.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    names = args.shape or [s.name for s in CONV_SHAPES]
    shapes = [conv_shape_by_name(n) for n in names]
    modes = MODES if args.mode == "both" else (args.mode,)
    limits = None if args.no_ceilings else ceilings(device)
    if args.against is not None:
        return _run_comparisons(shapes, modes, args, device, limits)
    rows: list[tuple[str, Throughput]] = []
    for shape in shapes:
        for mode in modes:
            report, rate = bench(shape, mode, args, device, limits)
            base = args.out.with_name(f"{args.out.name}-{shape.name}-{mode}")
            md, _ = write_report(report, base, require_agreement=False)
            rows.append((f"{shape.name}/{mode}", rate))
            print(f"wrote {md}")
    print()
    print(rate_table(rows, width=20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
