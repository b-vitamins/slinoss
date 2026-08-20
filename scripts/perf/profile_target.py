"""The process an external profiler attaches to. Not a bench.

Warmup, allocation, and any first-call compilation happen before the capture
window opens, so nothing but the measured iterations reaches a counter. The
window is :func:`slinoss.perf.capture.profiler_window`, which is what
``--capture-range=cudaProfilerApi`` and ``--profile-from-start off`` key on.

``--op`` selects the operator. It is the whole registry the profiler drivers
dispatch on: one target process per operator would be one warmup policy and one
capture window per operator, and they would drift.

Run it through ``scripts/perf/profile_op.py``; it is invoked directly only to
check that the target itself works.

    python3 scripts/perf/profile_target.py --shape standard --mode step --iters 3
    python3 scripts/perf/profile_target.py --op conv --shape standard --mode step
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import torch

from slinoss.perf.arms import op_arm
from slinoss.perf.capture import profiler_window
from slinoss.perf.coverage import MODES
from slinoss.perf.device import require_cuda
from slinoss.perf.timing import on_device
from slinoss.perf.workload import OPS, SHAPE_NAMES

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# MODES is imported, not restated: the coverage rule is keyed by mode, so a mode
# this target accepts and the table does not know would be judged against no entry.


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
        help="Iterations inside the capture window. Must match the driver's "
        "capture_iters, or every per-iteration figure is wrong by that factor.",
    )
    parser.add_argument("--warmup", type=int, default=5)
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
        "Nonzero makes the conv write y head-major, which is the layout the scan "
        "reads U in. Ignored by every other operator.",
    )
    return parser.parse_args(argv)


def build_runner(args: argparse.Namespace, device: torch.device) -> Callable[[], None]:
    """Allocate the inputs and return the callable the window wraps.

    Args:
        args: Parsed command line.
        device: Device to allocate on.

    Returns:
        The workload callable, region-labelled ``op.*`` for the scan, ``conv.*``
        for the conv, ``prep.*`` for the frontier, ``block.*`` for the block and
        ``mixer.*`` for the fused tail.
    """
    arm = op_arm(
        args.op,
        args.shape,
        device,
        dtype=DTYPES[args.dtype],
        grads=args.mode == "step",
        d_head=args.d_head or None,
    )
    return arm.run(args.backend, arm.prefix)


def main(argv: Sequence[str] | None = None) -> int:
    """Warm up, then run the measured iterations inside the capture window.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If ``--iters`` is not positive or ``--warmup`` is negative.
    """
    args = parse_args(argv)
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    # `range(-1)` is empty, so a negative warmup would run none and profile the
    # first call, which is the one thing the window exists to exclude. The bench
    # path rejects the same value in `measure`; both paths reject it or the same
    # flag over the same workload means two different things.
    if args.warmup < 0:
        raise ValueError(f"--warmup must not be negative, got {args.warmup}")
    device = require_cuda(args.device)
    runner = build_runner(args, device)
    with on_device(device):
        for _ in range(args.warmup):
            runner()
        with profiler_window(device):
            for _ in range(args.iters):
                runner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
