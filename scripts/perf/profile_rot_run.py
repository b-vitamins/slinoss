"""Measure and gate :data:`slinoss.ops.so3ssd.cute.table.ROT_RUN`.

``stage_rotated``'s per-thread run width is a module constant read at trace time,
so it is not part of the executor cache key in :mod:`slinoss._cute`. Two run
widths therefore compile into two executor sets inside one process, and the
constant is the only thing that differs between them: no call site moves, and the
backward kernels pick the width up without a file edit. That is what makes an
honest paired comparison possible on kernels this lane may not edit.

Four modes:

``decide`` records every ``_rot_run`` call made while tracing and prints the width
each staging geometry resolved to. Nothing is timed, so it needs no quiet device.

``count`` reports ``slinoss.perf.ncu``-free instruction facts: the executor count
and the compiled launch names, to confirm both arms built.

``parity`` runs the step once and writes ``y`` and the five gradients, for a
``torch.equal`` comparison against another run's dump. Two dumps from the same
seed and shape must agree bit for bit or the arm is refused.

``paired`` alternates the two executor sets inside one
:func:`slinoss.perf.timing.measure_paired` loop with the launch order swapped
every iteration, and reports the interval. ``--null`` runs the baseline against
itself through the same swap machinery, which is the control the interval is read
against.

    CUDA_VISIBLE_DEVICES=0 python3 scripts/perf/profile_rot_run.py \
        --mode paired --shape acceptance --part step
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch

from slinoss import _cute
from slinoss.ops.so3ssd.cute import table
from slinoss.perf.capture import profiler_window
from slinoss.perf.device import clock_policy, device_ordinal, require_cuda
from slinoss.perf.timing import measure_paired, on_device
from slinoss.perf.workload import (
    SHAPE_NAMES,
    OpInputs,
    forward_only,
    make_inputs,
    shape_by_name,
    step,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

BACKEND = "cute"

GRAD_NAMES = ("dU", "dtrans", "dK", "dB", "dC")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("decide", "count", "parity", "paired", "compare", "target"),
        default="decide",
    )
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--part", choices=("step", "forward"), default="step")
    parser.add_argument(
        "--rot-run", type=int, default=0, help="0 leaves the file's value."
    )
    parser.add_argument("--base-run", type=int, default=2)
    parser.add_argument("--arm-run", type=int, default=4)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--null", action="store_true", help="Baseline against itself.")
    parser.add_argument("--out", default="")
    parser.add_argument("--left", default="")
    parser.add_argument("--right", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build(args: argparse.Namespace, device: torch.device) -> tuple[OpInputs, int]:
    """Inputs and chunk length at one shape."""
    shape = shape_by_name(args.shape)
    inputs = make_inputs(shape, device, dtype=DTYPES[args.dtype], seed=0)
    return inputs, shape.chunk


def callable_for(
    inputs: OpInputs, chunk: int, part: str, prefix: str
) -> Callable[[], None]:
    """The timed callable for one part of the operator."""
    if part == "forward":
        return forward_only(inputs, chunk, backend=BACKEND, prefix=prefix)
    return step(inputs, chunk, backend=BACKEND, prefix=prefix)


def compile_arm(run: int, work: Callable[[], None]) -> dict[Any, Any]:
    """Compile every launcher ``work`` reaches at one run width.

    Rebinds ``table.ROT_RUN``, empties the executor cache, runs ``work`` once, and
    returns the executor set that built. The constant is read while tracing, so a
    cache left populated would return the other arm's code under the same key and
    the loop would measure one arm twice.

    Args:
        run: Run width to trace at.
        work: The callable to compile through.

    Returns:
        The executor set, keyed as :mod:`slinoss._cute` keys it.
    """
    table.ROT_RUN = run
    _cute._EXECUTORS = {}
    work()
    torch.cuda.synchronize()
    return _cute._EXECUTORS


def install(executors: dict[Any, Any]) -> None:
    """Make ``executors`` the cache the next launch looks in."""
    _cute._EXECUTORS = executors


def call_site() -> str:
    """The first frame outside ``table.py``, as ``module:line``.

    The trace enters ``_rot_run`` from ``stage_rotated``, so the nearest frame in
    another module is the kernel that staged. This is an exact attribution of a
    staging geometry to a call site, which an NCU source page cannot give: one
    ``.file`` per module means a line number there names no module.
    """
    frame = inspect.currentframe()
    here = table.__file__
    root = here.rsplit("/", 1)[0]
    while frame is not None:
        name = frame.f_code.co_filename
        if name != here and name.startswith(root):
            return f"{name[len(root) + 1 :]}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


def decide(args: argparse.Namespace, device: torch.device) -> int:
    """Print the run width every staging geometry resolved to."""
    inputs, chunk = build(args, device)
    seen: list[tuple[str, int, int, int, int]] = []
    original = table._rot_run

    def recording(threads: Any, span: Any, lanes: Any) -> int:
        got = original(threads, span, lanes)
        seen.append((call_site(), int(threads), int(span), int(lanes), int(got)))
        return got

    table._rot_run = recording
    if args.rot_run:
        table.ROT_RUN = args.rot_run
    try:
        _cute._EXECUTORS = {}
        callable_for(inputs, chunk, args.part, "op")()
        torch.cuda.synchronize()
    finally:
        table._rot_run = original
    print(f"shape {args.shape}  part {args.part}  ROT_RUN {table.ROT_RUN}")
    print(f"{'call site':<44} threads   span  lanes   run  count")
    tally: dict[tuple[str, int, int, int, int], int] = {}
    for row in seen:
        tally[row] = tally.get(row, 0) + 1
    for (site, threads, span, lanes, run), count in sorted(tally.items()):
        print(f"{site:<44} {threads:>7} {span:>6} {lanes:>6} {run:>5} {count:>6}")
    wide = sum(c for key, c in tally.items() if key[4] == table.ROT_RUN)
    print(f"call sites {len(tally)}  traces {len(seen)}  at wide run {wide}")
    return 0


def count(args: argparse.Namespace, device: torch.device) -> int:
    """Report the executor count and launch names each arm builds."""
    inputs, chunk = build(args, device)
    for run in (args.base_run, args.arm_run):
        executors = compile_arm(run, callable_for(inputs, chunk, args.part, f"r{run}"))
        names = sorted(_cute.compiled_launches())
        print(f"run {run}: executors {len(executors)}  launches {len(names)}")
        for name in names:
            print(f"    {name}")
    return 0


def target(args: argparse.Namespace, device: torch.device) -> int:
    """Run one iteration inside a profiler window at one run width.

    What ``scripts/perf/profile_target.py`` is, plus the run width, so a counter
    pass can be collected twice from one file. Differencing two file versions
    instead shifts every line number in ``table.py`` and the source page's line
    key stops meaning the same site on both sides.
    """
    inputs, chunk = build(args, device)
    if args.rot_run:
        table.ROT_RUN = args.rot_run
    _cute._EXECUTORS = {}
    work = callable_for(inputs, chunk, args.part, "op")
    with on_device(device):
        for _ in range(args.warmup):
            work()
        with profiler_window(device):
            work()
    return 0


def parity(args: argparse.Namespace, device: torch.device) -> int:
    """Write ``y`` and the five gradients at one run width."""
    inputs, chunk = build(args, device)
    if args.rot_run:
        table.ROT_RUN = args.rot_run
    _cute._EXECUTORS = {}
    from slinoss.ops.so3ssd import so3ssd

    y = so3ssd(
        inputs.U, inputs.trans, inputs.K, inputs.B, inputs.C, chunk, backend=BACKEND
    ).y
    grads = torch.autograd.grad(y, inputs.differentiable, inputs.dy)
    torch.cuda.synchronize()
    payload = {"y": y.detach().cpu()}
    for name, grad in zip(GRAD_NAMES, grads, strict=True):
        payload[name] = grad.detach().cpu()
    rot = getattr(table, "ROT_RUN", 0)
    print(f"shape {args.shape}  ROT_RUN {rot}  y {tuple(y.shape)} {y.dtype}")
    if args.out:
        torch.save(payload, args.out)
        print(f"wrote {args.out}")
    return 0


def compare(args: argparse.Namespace) -> int:
    """Compare two parity dumps tensor by tensor.

    Returns:
        0 when every tensor is bitwise equal, 1 otherwise.
    """
    left = torch.load(args.left, weights_only=True)
    right = torch.load(args.right, weights_only=True)
    bad = 0
    print(f"{'tensor':<8} {'equal':<6} {'max abs diff':>14} {'ulp-ish':>10}")
    for name in ("y", *GRAD_NAMES):
        a, b = left[name], right[name]
        same = bool(torch.equal(a, b))
        wide_a, wide_b = a.double(), b.double()
        diff = (wide_a - wide_b).abs()
        scale = wide_a.abs().clamp_min(1e-30)
        print(
            f"{name:<8} {same!s:<6} {diff.max().item():>14.6e} "
            f"{(diff / scale).max().item():>10.3e}"
        )
        bad += 0 if same else 1
    print("bitwise equal" if bad == 0 else f"NOT bitwise: {bad} tensors differ")
    return 0 if bad == 0 else 1


def paired(args: argparse.Namespace, device: torch.device) -> int:
    """Alternate two run widths in one loop and print the interval."""
    inputs, chunk = build(args, device)
    base_run, arm_run = args.base_run, args.arm_run
    base_label, arm_label = f"run{base_run}", f"run{arm_run}"
    base_work = callable_for(inputs, chunk, args.part, base_label)
    arm_work = callable_for(inputs, chunk, args.part, arm_label)
    base_set = compile_arm(base_run, base_work)
    arm_set = compile_arm(base_run if args.null else arm_run, arm_work)
    if len(base_set) != len(arm_set):
        print(f"WARNING executor counts differ: {len(base_set)} vs {len(arm_set)}")
    built = (len(base_set), len(arm_set))

    # The width is restored per call as well as per compile: a key neither warmup
    # reached would otherwise compile mid-loop at whichever width was set last,
    # and one arm would carry the other's code with nothing in the interval to
    # show it.
    def run_base() -> None:
        table.ROT_RUN = base_run
        install(base_set)
        base_work()

    def run_arm() -> None:
        table.ROT_RUN = base_run if args.null else arm_run
        install(arm_set)
        arm_work()

    got = measure_paired(
        base_label,
        run_base,
        arm_label,
        run_arm,
        label=f"so3ssd.{args.part} ROT_RUN {base_run} vs {arm_run}"
        + (" (null)" if args.null else ""),
        iters=args.iters,
        warmup=args.warmup,
        device=device,
        clocks=clock_policy(device_ordinal(device)),
    )
    row = got.comparison
    if (len(base_set), len(arm_set)) != built:
        print(
            f"WARNING an arm compiled inside the loop: {built} -> "
            f"{(len(base_set), len(arm_set))}"
        )
    print(f"shape {args.shape}  part {args.part}  iters {args.iters}")
    print(row.verdict())
    print(
        f"a {row.a_median_duration_us:.3f} us  b {row.b_median_duration_us:.3f} us  "
        f"delta {row.delta_median_duration_us:+.3f} "
        f"[{row.delta_low_duration_us:+.3f}, {row.delta_high_duration_us:+.3f}] us  "
        f"pct {row.delta_pct:+.3f}%  speedup {row.speedup_ratio:.4f}  "
        f"position {row.position_duration_us:+.3f} us  resolves {row.resolves}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch on ``--mode``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    if args.mode == "compare":
        return compare(args)
    device = torch.device("cuda", device_ordinal(require_cuda(args.device)))
    torch.cuda.set_device(device)
    if args.mode == "decide":
        return decide(args, device)
    if args.mode == "count":
        return count(args, device)
    if args.mode == "parity":
        return parity(args, device)
    if args.mode == "target":
        return target(args, device)
    return paired(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
