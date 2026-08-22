"""Paired A/B of the global load width in the two token-major staging helpers.

:func:`slinoss.ops.so3ssd.cute.table.stage_trans` and
:func:`slinoss.ops.so3ssd.cute.table.stage_chunk` read a float32 row of ``trans``
and ``K`` per token. This tree reads each row at
:func:`slinoss.ops.so3ssd.cute.table._segment_run` width; the baseline here is the
scalar form it replaced, one component per load, restated in this file so that no
second copy of the helper lives under ``slinoss/``.

Both forms trace in one process. A helper called at trace time is not part of the
executor cache key, so the arm is selected by rebinding the name every consumer
module bound at import, tracing with the cache empty, and keeping the two resulting
executor sets. Swapping the cache between calls is what the paired loop alternates.

``--mode target`` is the process a profiler attaches to; ``--arm`` picks which form
runs inside the window, and the sector counters are collected over it. The counters
are per-launch counts, so they are read across two invocations without a pairing.

    python3 scripts/perf/ab_stage_width.py --mode parity --shape acceptance
    python3 scripts/perf/ab_stage_width.py --mode paired --shape acceptance
    python3 scripts/perf/ab_stage_width.py --mode paired --null --shape acceptance
    python3 scripts/perf/ab_stage_width.py --mode target --arm scalar --shape standard
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Final

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss import _cute
from slinoss._cute import select
from slinoss.ops.so3ssd import so3ssd
from slinoss.ops.so3ssd.cute import table
from slinoss.perf.capture import profiler_window
from slinoss.perf.device import clock_policy, contention, device_ordinal, require_cuda
from slinoss.perf.timing import measure_paired, on_device
from slinoss.perf.workload import (
    SHAPE_NAMES,
    OpInputs,
    OpShape,
    make_inputs,
    shape_by_name,
)

DTYPES: Final = {"bf16": torch.bfloat16, "fp16": torch.float16}

CONSUMERS: Final = (
    "slinoss.ops.so3ssd.cute.table",
    "slinoss.ops.so3ssd.cute.bwd.chunk_start",
    "slinoss.ops.so3ssd.cute.bwd.chunk_input",
    "slinoss.ops.so3ssd.cute.bwd.chunk_vector",
    "slinoss.ops.so3ssd.cute.fwd.chunk_scan",
    "slinoss.ops.so3ssd.cute.fwd.chunk_increment",
    "slinoss.ops.so3ssd.cute.fwd.increment_passing",
)
"""Every module holding a binding of either helper, ``table`` included.

Each consumer imported the name, so rebinding it on ``table`` alone reaches only
``table``'s own internal call. ``chunk_increment`` is here for completeness; it
does not launch in the shipped step.
"""

ARMS: Final = ("wide", "scalar")

_WIDE: Final = (table.stage_trans, table.stage_chunk)
"""The forms in the tree, captured before any rebinding."""


@cute.jit
def scalar_trans(
    gtrans: cute.Tensor,
    strans: cute.Tensor,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    """:func:`slinoss.ops.so3ssd.cute.table.stage_trans` at one component a load.

    Args:
        gtrans: ``(T, 4)`` float32 view of ``trans`` for one ``(b, h)``.
        strans: ``(4, L)`` float32 shared tile, written.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
    """
    zero = cutlass.Float32(0.0)
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            inside = token < valid
            pos = t0 + cutlass.min(token, valid - 1)
            for j in cutlass.range_constexpr(4):
                strans[j, token] = select(inside, gtrans[pos, j], zero)


@cute.jit
def scalar_chunk(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    strans: cute.Tensor,
    stap: cute.Tensor,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    """:func:`slinoss.ops.so3ssd.cute.table.stage_chunk` at one component a load.

    Args:
        gtrans: ``(T, 4)`` float32 view of ``trans`` for one ``(b, h)``.
        gtap: ``(T, 2, 4)`` float32 view of ``K`` for one ``(b, h)``.
        strans: ``(4, L)`` float32 shared tile, written.
        stap: ``(8, L)`` float32 shared tile, written.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
    """
    scalar_trans(gtrans, strans, t0, valid, tid, threads, chunk)
    zero = cutlass.Float32(0.0)
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            inside = token < valid
            pos = t0 + cutlass.min(token, valid - 1)
            for tap in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(4):
                    stap[4 * tap + j, token] = select(inside, gtap[pos, tap, j], zero)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("parity", "paired", "target"), default="parity"
    )
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=ARMS, default="wide")
    parser.add_argument(
        "--null",
        action="store_true",
        help="Measure one arm against itself, which is the control on the loop.",
    )
    parser.add_argument("--event-iters", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args(argv)


def modules() -> tuple[ModuleType, ...]:
    """Import every consumer once and return them in :data:`CONSUMERS` order."""
    return tuple(importlib.import_module(one) for one in CONSUMERS)


def rebind(held: Sequence[ModuleType], arm: str) -> None:
    """Point every consumer's two staging names at one arm's forms.

    Args:
        held: The consumer modules.
        arm: ``wide`` for the forms in the tree, ``scalar`` for this file's.
    """
    trans, chunk = _WIDE if arm == "wide" else (scalar_trans, scalar_chunk)
    # Through the name, not the attribute: a module's staging names are not part of
    # its declared surface, so an assignment to one does not type-check.
    forms = (("stage_trans", trans), ("stage_chunk", chunk))
    for one in held:
        for name, form in forms:
            if hasattr(one, name):
                setattr(one, name, form)


def compile_arm(
    held: Sequence[ModuleType], arm: str, work: Callable[[], None]
) -> dict[Any, Any]:
    """Trace one arm with the cache empty and return the executors it left.

    Args:
        held: The consumer modules.
        arm: Which arm to trace.
        work: One pass over the workload, enough to reach every launch.

    Returns:
        The executor cache the pass built.
    """
    rebind(held, arm)
    _cute._EXECUTORS.clear()
    work()
    torch.cuda.synchronize()
    return dict(_cute._EXECUTORS)


def install(held: Sequence[ModuleType], arm: str, snapshot: dict[Any, Any]) -> None:
    """Make one arm the live one: its bindings and its executors.

    The rebinding matters only if a key is missing from the snapshot and retraces.

    Args:
        held: The consumer modules.
        arm: Which arm to install.
        snapshot: That arm's executors, from :func:`compile_arm`.
    """
    rebind(held, arm)
    _cute._EXECUTORS.clear()
    _cute._EXECUTORS.update(snapshot)


def build_step(
    inputs: OpInputs, shape: OpShape
) -> tuple[Callable[[], tuple[Tensor, ...]], Callable[[], None]]:
    """A forward and backward over one allocated input set, twice over.

    Args:
        inputs: The operator inputs, differentiable.
        shape: The shape record, for the chunk length.

    Returns:
        A callable returning ``y`` and the five gradients, and one returning
        nothing, which is what the paired loop times.
    """
    targets = tuple(inputs.differentiable)

    def call() -> tuple[Tensor, ...]:
        y = so3ssd(inputs.U, inputs.trans, inputs.K, inputs.B, inputs.C, shape.chunk).y
        grads = torch.autograd.grad(y, targets, inputs.dy)
        return (y.detach(), *grads)

    def run() -> None:
        y = so3ssd(inputs.U, inputs.trans, inputs.K, inputs.B, inputs.C, shape.chunk).y
        torch.autograd.grad(y, targets, inputs.dy)

    return call, run


def parity(left: Sequence[Tensor], right: Sequence[Tensor]) -> tuple[bool, str]:
    """Compare two output sets bitwise.

    Args:
        left: Baseline outputs, ``y`` first.
        right: Arm outputs, in the same order.

    Returns:
        Whether every tensor is bitwise equal, and the per-tensor detail.
    """
    names = ("y", "dU", "dtrans", "dK", "dB", "dC")
    parts: list[str] = []
    clean = True
    for name, a, b in zip(names, left, right):
        same = torch.equal(a, b)
        clean = clean and same
        if same:
            parts.append(f"{name} equal")
        else:
            gap = (a.float() - b.float()).abs().max().item()
            parts.append(f"{name} DIFFERS max {gap:.3e}")
    return clean, "  ".join(parts)


def run_parity(args: argparse.Namespace, device: torch.device) -> int:
    """Trace both arms and compare their outputs bitwise.

    Args:
        args: The parsed command line.
        device: Where to allocate.

    Returns:
        Zero when every output is bitwise equal.
    """
    shape = shape_by_name(args.shape)
    inputs = make_inputs(shape, device, dtype=DTYPES[args.dtype], requires_grad=True)
    held = modules()
    call, run = build_step(inputs, shape)
    snapshots = {one: compile_arm(held, one, run) for one in ARMS}
    outs: dict[str, tuple[Tensor, ...]] = {}
    for one in ARMS:
        install(held, one, snapshots[one])
        outs[one] = call()
    torch.cuda.synchronize(device)
    clean, detail = parity(outs["scalar"], outs["wide"])
    print(f"device       {device} ord {device_ordinal(device)}")
    print(f"shape        {shape.describe()}")
    print("executors    " + "  ".join(f"{k} {len(v)}" for k, v in snapshots.items()))
    print(f"parity       {'bitwise' if clean else 'DIFFERS'}")
    print(f"detail       {detail}")
    return 0 if clean else 1


def run_paired(args: argparse.Namespace, device: torch.device) -> int:
    """Measure both arms in one order-swapped loop over one input set.

    Args:
        args: The parsed command line.
        device: Where to allocate and time.

    Returns:
        Zero when the outputs agree bitwise.
    """
    shape = shape_by_name(args.shape)
    inputs = make_inputs(shape, device, dtype=DTYPES[args.dtype], requires_grad=True)
    held = modules()
    call, run = build_step(inputs, shape)
    snapshots = {one: compile_arm(held, one, run) for one in ARMS}
    base = args.arm if args.null else "scalar"

    def arm_runner(which: str) -> Callable[[], None]:
        def go() -> None:
            install(held, which, snapshots[which])
            run()

        return go

    install(held, base, snapshots[base])
    left = call()
    install(held, args.arm, snapshots[args.arm])
    right = call()
    clean, detail = parity(left, right)
    torch.cuda.synchronize(device)
    out = measure_paired(
        "base",
        arm_runner(base),
        "arm",
        arm_runner(args.arm),
        label=f"stage.{args.shape}",
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
        clocks=clock_policy(device_ordinal(device)),
    )
    print(f"device       {device} ord {device_ordinal(device)}")
    print(f"shape        {shape.describe()}")
    print(
        f"arms         base {base}  arm {args.arm}{'  (null control)' if args.null else ''}"
    )
    print(f"clocks       {out.timed.clocks}")
    print(f"contention   {contention(device_ordinal(device))}")
    print(f"parity       {'bitwise' if clean else 'DIFFERS'}  {detail}")
    print(f"verdict      {out.comparison.verdict()}")
    print(
        f"medians      base {out.comparison.a_median_duration_us:,.3f} us  "
        f"arm {out.comparison.b_median_duration_us:,.3f} us"
    )
    return 0 if clean else 1


def run_target(args: argparse.Namespace, device: torch.device) -> int:
    """Run one arm inside the capture window and nothing else.

    Args:
        args: The parsed command line.
        device: Where to allocate and run.

    Returns:
        Zero.
    """
    shape = shape_by_name(args.shape)
    inputs = make_inputs(shape, device, dtype=DTYPES[args.dtype], requires_grad=True)
    held = modules()
    _, run = build_step(inputs, shape)
    install(held, args.arm, compile_arm(held, args.arm, run))
    with on_device(device):
        for _ in range(args.warmup):
            run()
        with profiler_window(device):
            for _ in range(args.iters):
                run()
        torch.cuda.synchronize(device)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the two staging load widths, or be the profiler's target.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    if args.mode == "parity":
        return run_parity(args, device)
    if args.mode == "paired":
        return run_paired(args, device)
    return run_target(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
