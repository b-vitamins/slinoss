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
from slinoss.perf.memory import (
    SavedStorages,
    SavedTensorProbe,
    memory_peaks,
    reset_memory_peaks,
)
from slinoss.perf.report import Report, write_report
from slinoss.perf.timing import Throughput, measure, region
from slinoss.perf.workload import SHAPES, OpShape, shape_by_name

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
) -> Callable[[], None]:
    """Build the timed callable for one mode."""

    def forward() -> Tensor:
        return scan(inputs.x, inputs.dt, inputs.A, inputs.B, inputs.C, chunk)

    if not grads:

        def run_forward() -> None:
            with torch.no_grad(), region("mamba.forward"):
                forward()

        return run_forward

    def run_step() -> None:
        with region("mamba.forward"):
            y = forward()
        with region("mamba.backward"):
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
    print(
        f"{'config':<28} {'duration_us':>14} {'spread_pct':>11} "
        f"{'resolution_pct':>15} {'coverage_pct':>13} {'tps':>14}"
    )
    for name, rate in rows:
        print(
            f"{name:<28} {rate.duration_us:>14,.3f} {rate.spread_pct:>11,.3f} "
            f"{rate.resolution_pct:>15,.3f} {rate.coverage_pct:>13,.3f} "
            f"{rate.throughput_tps:>14,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
