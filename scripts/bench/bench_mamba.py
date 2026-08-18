"""Bench Mamba2's chunked scan under the same harness, as the external floor.

Same timer, same iteration count, same spread discipline, same report schema as
``bench_op.py``. A comparison run under two different harnesses compares the
harnesses.

Two group configurations are measured, because they are two different claims:

- ``groups=heads`` gives every head its own ``B`` and ``C``, which is what the
  SO(3) operator does, so the two move the same bytes.
- ``groups=1`` shares ``B`` and ``C`` across heads, which is Mamba2's own default
  and moves fewer bytes. It is the harder number to beat and it is reported
  rather than omitted.

Requires ``mamba-ssm``. Absent, the script exits with a message naming the
package instead of a traceback.

    python3 scripts/bench/bench_mamba.py --shape standard --mode both

``--against-so3ssd`` runs both operators inside one loop and judges the
per-iteration difference. Two separate runs cannot be subtracted: their medians
scatter further than either run's own floor. This is the only comparison against
the floor that resolves anything.

    python3 scripts/bench/bench_mamba.py --shape standard --mode step \\
        --groups heads --against-so3ssd
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.device import device_info, device_ordinal
from slinoss.perf.dispersion import PairedRow
from slinoss.perf.memory import (
    SavedStorages,
    SavedTensorProbe,
    memory_peaks,
    reset_memory_peaks,
)
from slinoss.perf.report import Report, rate_table, write_report
from slinoss.perf.timing import Throughput, measure, measure_paired, region
from slinoss.perf.workload import SHAPES, OpShape, shape_by_name
from slinoss.perf.workload import forward_only as so3ssd_forward_only
from slinoss.perf.workload import make_inputs as so3ssd_inputs
from slinoss.perf.workload import step as so3ssd_step

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "step")


def load_scan() -> Callable[..., Any]:
    """Import Mamba2's chunked scan.

    Returns:
        ``mamba_chunk_scan_combined``.

    Raises:
        SystemExit: If ``mamba-ssm`` is not installed.
    """
    try:
        from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore[import-not-found]
            mamba_chunk_scan_combined,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"bench_mamba needs mamba-ssm: {exc}") from exc
    return mamba_chunk_scan_combined


class MambaInputs(NamedTuple):
    """Inputs to ``mamba_chunk_scan_combined``.

    Attributes:
        x: ``(batch, seqlen, nheads, headdim)``.
        dt: ``(batch, seqlen, nheads)``.
        A: ``(nheads,)``, float32. Mamba2 requires float32 here.
        B: ``(batch, seqlen, ngroups, dstate)``.
        C: ``(batch, seqlen, ngroups, dstate)``.
        dy: Output-gradient seed, shaped like ``x``.
    """

    x: Tensor
    dt: Tensor
    A: Tensor
    B: Tensor
    C: Tensor
    dy: Tensor

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """The five tensors gradients are taken with respect to."""
        return (self.x, self.dt, self.A, self.B, self.C)


def make_inputs(
    shape: OpShape,
    groups: int,
    device: torch.device,
    *,
    dtype: torch.dtype,
    requires_grad: bool,
    seed: int = 0,
) -> MambaInputs:
    """Build Mamba2 inputs matching one SO(3) shape.

    ``headdim`` is the SO(3) row count and ``dstate`` is its ``3N``, so the state
    per head is the same size in both operators.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*size: int, dt: torch.dtype = dtype) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    x = randn(shape.bsz, shape.seq, shape.heads, shape.rows)
    return MambaInputs(
        x=x.requires_grad_(requires_grad),
        dt=randn(shape.bsz, shape.seq, shape.heads, dt=torch.float32).requires_grad_(
            requires_grad
        ),
        A=(-randn(shape.heads, dt=torch.float32).abs()).requires_grad_(requires_grad),
        B=randn(shape.bsz, shape.seq, groups, shape.d_state).requires_grad_(
            requires_grad
        ),
        C=randn(shape.bsz, shape.seq, groups, shape.d_state).requires_grad_(
            requires_grad
        ),
        dy=randn(shape.bsz, shape.seq, shape.heads, shape.rows),
    )


def runner(
    scan: Callable[..., Any],
    inputs: MambaInputs,
    chunk: int,
    *,
    grads: bool,
    prefix: str = "mamba",
) -> Callable[[], None]:
    """Build the timed callable for one mode.

    Args:
        scan: ``mamba_chunk_scan_combined``.
        inputs: Its inputs.
        chunk: Chunk length.
        grads: Whether to run the backward.
        prefix: Region label prefix. Two arms measured in one loop need two
            prefixes; see :func:`slinoss.perf.workload.forward_only`.

    Returns:
        The callable.
    """

    def forward() -> Tensor:
        return scan(inputs.x, inputs.dt, inputs.A, inputs.B, inputs.C, chunk)

    if not grads:

        def run_forward() -> None:
            with torch.no_grad(), region(f"{prefix}.forward"):
                forward()

        return run_forward

    def run_step() -> None:
        with region(f"{prefix}.forward"):
            y = forward()
        with region(f"{prefix}.backward"):
            torch.autograd.grad(y, inputs.differentiable, inputs.dy)

    return run_step


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
    parser.add_argument(
        "--groups",
        action="append",
        choices=["heads", "one"],
        help="Group configuration. Repeatable. Defaults to both.",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--against-so3ssd",
        action="store_true",
        help=(
            "Measure the SO(3) operator against Mamba2 inside one loop and judge "
            "the per-iteration difference. Needs an even --iters."
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="SO(3) backend for the comparison arm. Default is the fastest one.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/bench-mamba"))
    return parser.parse_args(argv)


def group_counts(shape: OpShape, kinds: Sequence[str]) -> tuple[int, ...]:
    """Resolve group kinds to distinct group counts, in the order requested.

    Args:
        shape: The shape being benched. ``heads`` resolves against its head count.
        kinds: ``heads``, ``one``, or both.

    Returns:
        The group counts to measure. ``heads`` and ``one`` are the same
        configuration at ``heads == 1``, and one configuration is measured once.
    """
    counts = [shape.heads if kind == "heads" else 1 for kind in kinds]
    return tuple(dict.fromkeys(counts))


def _saved(
    scan: Callable[..., Any],
    shape: OpShape,
    groups: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SavedStorages:
    """Probe what Mamba2's graph holds for one forward and backward.

    Runs under a recorder so each save attributes to the region it was taken in.
    Without one every row would read ``unattributed``.
    """
    inputs = make_inputs(shape, groups, device, dtype=dtype, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        measure(
            runner(scan, inputs, shape.chunk, grads=True),
            label=f"mamba {shape.name} saved",
            iters=1,
            warmup=0,
            device=device,
        )
    return probe.report(f"mamba {shape.name}", inputs.differentiable)


def compare_so3ssd(
    scan: Callable[..., Any],
    shape: OpShape,
    groups: int,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Report, PairedRow]:
    """Measure the SO(3) operator against Mamba2 in one loop at one configuration.

    Mamba2 is the baseline arm, so ``speedup_ratio`` above one means the SO(3)
    operator is the faster of the two.

    Args:
        scan: ``mamba_chunk_scan_combined``.
        shape: The problem size. Both arms carry the same state size per head.
        groups: Mamba2 group count.
        mode: ``forward`` or ``step``.
        args: Parsed command line.
        device: Device to time on.

    Returns:
        The report and the verdict on the per-iteration differences.
    """
    dtype = DTYPES[args.dtype]
    grads = mode == "step"
    a_label = f"mamba-g{groups}"
    b_label = f"so3ssd-{args.backend or 'auto'}"
    mamba = make_inputs(shape, groups, device, dtype=dtype, requires_grad=grads)
    ours = so3ssd_inputs(shape, device, dtype=dtype, requires_grad=grads)
    label = f"mamba g{groups} vs so3ssd {shape.name} {mode} paired"
    reset_memory_peaks(device)
    out = measure_paired(
        a_label,
        runner(scan, mamba, shape.chunk, grads=grads, prefix=a_label),
        b_label,
        (
            so3ssd_step(ours, shape.chunk, backend=args.backend, prefix=b_label)
            if grads
            else so3ssd_forward_only(
                ours, shape.chunk, backend=args.backend, prefix=b_label
            )
        ),
        label=label,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
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
        peaks=memory_peaks(label, device),
        notes=(
            shape.describe(),
            f"mamba2 ngroups={groups} headdim={shape.rows} dstate={shape.d_state}",
            f"mode={mode} dtype={args.dtype}",
            f"arm a={a_label} b={b_label}, one loop, order swapped each iteration",
            # The two operators take different tensors, so the arms cannot share
            # inputs the way two backends of one operator do. The peak is the sum of
            # both arms' live tensors and belongs to neither.
            "each arm holds its own inputs; the memory peak covers both",
            f"iters={args.iters} warmup={args.warmup}",
            f"timer={out.timed.timer} clocks={out.timed.clocks}",
        ),
    )
    return report, out.comparison


def _run_comparisons(
    scan: Callable[..., Any],
    shapes: Sequence[OpShape],
    modes: Sequence[str],
    wanted: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    """Run every paired comparison against Mamba2 and print the verdicts.

    Returns:
        Process exit status.
    """
    rates: list[tuple[str, Throughput]] = []
    verdicts: list[PairedRow] = []
    for shape in shapes:
        for groups in group_counts(shape, wanted):
            for mode in modes:
                report, row = compare_so3ssd(scan, shape, groups, mode, args, device)
                base = args.out.with_name(
                    f"{args.out.name}-{shape.name}-g{groups}-{mode}-paired"
                )
                md, _ = write_report(report, base, require_agreement=False)
                rates += [
                    (f"{shape.name}/g{groups}/{mode}/{rate.label}", rate)
                    for rate in report.throughput
                ]
                verdicts.append(row)
                print(f"wrote {md}")
    print()
    print(rate_table(rates, width=52))
    print()
    for row in verdicts:
        print(row.verdict())
    return 0


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
    scan = load_scan()
    dtype = DTYPES[args.dtype]
    shapes = [shape_by_name(n) for n in (args.shape or [s.name for s in SHAPES])]
    modes = MODES if args.mode == "both" else (args.mode,)
    wanted = args.groups or ["heads", "one"]
    if args.against_so3ssd:
        return _run_comparisons(scan, shapes, modes, wanted, args, device)
    info = device_info(device_ordinal(device))
    rows: list[tuple[str, Throughput]] = []
    for shape in shapes:
        # At heads=1 the two group kinds resolve to the same configuration. Running
        # both would time one thing twice, print it under one label, and have the
        # second report overwrite the first.
        for groups in group_counts(shape, wanted):
            for mode in modes:
                grads = mode == "step"
                inputs = make_inputs(
                    shape, groups, device, dtype=dtype, requires_grad=grads
                )
                label = f"mamba {shape.name} g{groups} {mode}"
                reset_memory_peaks(device)
                timed = measure(
                    runner(scan, inputs, shape.chunk, grads=grads),
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
                    device=info,
                    budget=tree,
                    throughput=(rate,),
                    saved=_saved(scan, shape, groups, device, dtype) if grads else None,
                    peaks=peaks,
                    notes=(
                        shape.describe(),
                        f"mamba2 ngroups={groups} headdim={shape.rows} "
                        f"dstate={shape.d_state}",
                        f"mode={mode} dtype={args.dtype}",
                        f"iters={args.iters} warmup={args.warmup}",
                        f"timer={timed.timer} clocks={timed.clocks}",
                    ),
                )
                base = args.out.with_name(
                    f"{args.out.name}-{shape.name}-g{groups}-{mode}"
                )
                md, _ = write_report(report, base, require_agreement=False)
                rows.append((f"{shape.name}/g{groups}/{mode}", rate))
                print(f"wrote {md}")
    print()
    print(rate_table(rows, width=28))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
