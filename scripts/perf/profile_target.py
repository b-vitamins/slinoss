"""The process an external profiler attaches to. Not a bench.

Warmup, allocation, and any first-call compilation happen before the capture
window opens, so nothing but the measured iterations reaches a counter. The
window is :func:`slinoss.perf.capture.profiler_window`, which is what
``--capture-range=cudaProfilerApi`` and ``--profile-from-start off`` key on.

Run it through ``scripts/perf/profile_op.py``; it is invoked directly only to
check that the target itself works.

    python3 scripts/perf/profile_target.py --shape standard --mode step --iters 3
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from slinoss.perf.capture import profiler_window
from slinoss.perf.timing import on_device
from slinoss.perf.workload import (
    SHAPES,
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
    parser.add_argument("--shape", choices=[s.name for s in SHAPES], default="standard")
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Warm up, then run the measured iterations inside the capture window.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is CUDA and CUDA is unavailable.
        ValueError: If ``--iters`` is not positive.
    """
    args = parse_args(argv)
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda needs CUDA")
    shape = shape_by_name(args.shape)
    grads = args.mode == "step"
    inputs = make_inputs(shape, device, dtype=DTYPES[args.dtype], requires_grad=grads)
    runner = (
        step(inputs, shape.chunk, backend=args.backend)
        if grads
        else forward_only(inputs, shape.chunk, backend=args.backend)
    )
    with on_device(device):
        for _ in range(args.warmup):
            runner()
        with profiler_window(device):
            for _ in range(args.iters):
                runner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
