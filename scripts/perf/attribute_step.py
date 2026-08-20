"""Where a step's device time goes, by kernel and by class.

The per-kernel benches say how close one kernel is to its own ceiling. This says
which kernel is worth the work: it attributes every microsecond of device time in a
whole step of the stack, so a kernel that is 2x off its roofline and 1% of the step
is not mistaken for the one that is 49% of it.

    python3 scripts/perf/attribute_step.py --mode step
    python3 scripts/perf/attribute_step.py --mode forward
    python3 scripts/perf/attribute_step.py --mode decode

Only device-side rows are summed. An operator row in a profiler table carries the
device time of its children as well as its own, so summing operator rows double
counts; the rows here are kernels, memcpies and memsets, and they partition the
step.

The class of a kernel is decided by its name, against the table in :data:`CLASSES`.
Anything no rule matches is reported under ``other``, with its name, so nothing is
absorbed silently.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from slinoss.config import SLinOSSConfig
from slinoss.ops.xent import cross_entropy
from slinoss.perf.device import device_ordinal, require_cuda
from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

MODES = ("forward", "step", "decode")
DTYPE = torch.bfloat16
"""The dtype the kernel path runs. float32 falls back to the reference scan."""

CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loss", ("SoftMax", "nll_loss", "cross_entropy", "xent_")),
    ("cute", ("kernel_cutlass_",)),
    ("gemm", ("cutlass::Kernel", "ampere_", "sm80_", "sm86_", "gemm", "gemv")),
    ("optim", ("multi_tensor_apply", "adamw", "fused_adam")),
    ("memory", ("Memcpy", "Memset")),
    ("elementwise", ("elementwise_kernel", "reduce_kernel", "fill_", "vectorized_")),
)
"""Class name against the substrings that select it, first match winning.

``loss`` is what turns logits into a scalar, whichever kernel does it, so it is
matched ahead of ``cute``: the class is a stage of the step and a fusion that moved
the stage into another class would report as a stage that vanished. ``cute`` is this
package's own kernels. ``gemm`` is cuBLAS and CUTLASS. ``elementwise`` is the aten
glue that a fused kernel would have absorbed, and is the class a fusion is supposed
to shrink.
"""

GLUE = ("elementwise", "memory", "other")
"""The classes that are cost rather than work: aten glue and data movement."""

MAX_COVERAGE_PCT = 102.0
"""Most of the bracketing interval the rows may sum to.

One stream, so the kernels cannot overlap and their sum cannot exceed the interval
they run in. Two percent of slack for the profiler's own per-kernel cost, which
lands inside the profiled interval and not in the one it is compared against.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="step")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--prefill", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--d-state", type=int, default=240)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rows", type=int, default=24, help="Kernels listed.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SLinOSSConfig:
    """The geometry the step runs at."""
    return SLinOSSConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        d_head=args.d_head,
        n_groups=args.groups,
        chunk_size=args.chunk,
        n_layers=args.layers,
        ffn_ratio=4.0,
        vocab_size=args.vocab,
    )


def build_step(
    args: argparse.Namespace, config: SLinOSSConfig, device: torch.device
) -> Callable[[], object]:
    """The callable one iteration profiles.

    ``forward`` is inference over a full sequence, ``step`` is that plus the
    backward and a fused optimizer update, ``decode`` is one token per sequence
    against a state a prefill has already advanced.

    Args:
        args: The command line.
        config: The geometry.
        device: Where to run.

    Returns:
        The callable. Takes no arguments.
    """
    torch.manual_seed(0)
    stack = SLinOSSStack(config, device=device).to(DTYPE)
    vocab = config.vocab_size
    assert vocab is not None

    if args.mode == "decode":
        state = StackState.allocate(config, args.batch, device=device, dtype=DTYPE)
        stack(torch.randint(0, vocab, (args.batch, args.prefill), device=device), state)
        token = torch.randint(0, vocab, (args.batch, 1), device=device)
        return lambda: stack(token, state)

    ids = torch.randint(0, vocab, (args.batch, args.seqlen), device=device)
    if args.mode == "forward":

        def forward() -> object:
            with torch.no_grad():
                return stack(ids)

        return forward

    labels = torch.randint(0, vocab, (args.batch, args.seqlen), device=device)
    optimizer = torch.optim.AdamW(stack.parameters(), lr=0.0)

    def train() -> object:
        optimizer.zero_grad(set_to_none=False)
        logits = stack(ids)
        # Classes come from the config, never from the logits' last extent: an
        # aligned head pads its output width past the vocabulary, and a pad
        # column is not a class a label indexes. The fused operator takes the
        # padded width and the class count separately, so the flatten is the
        # only reshape and no slice narrows the operand.
        loss = cross_entropy(logits.flatten(0, 1), labels.flatten(), classes=vocab)
        loss.backward()
        optimizer.step()
        return loss.detach()

    return train


def classify(name: str) -> str:
    """The class of a kernel, by name.

    Args:
        name: Kernel, memcpy or memset name as the profiler reports it.

    Returns:
        The class, or ``other`` if no rule matches.
    """
    lowered = name.lower()
    for label, needles in CLASSES:
        if any(needle.lower() in lowered for needle in needles):
            return label
    return "other"


def _self_us(row: object) -> float:
    """One row's own device microseconds, under either attribute name."""
    for attribute in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(row, attribute, None)
        if value is not None:
            return float(value)
    raise AttributeError("no self device time on a profiler row")


def device_rows(profiled: profile, iters: int) -> list[tuple[str, float, float]]:
    """Per-kernel name, microseconds per iteration, and calls per iteration.

    A user annotation is typed as a device row and carries the device time of the
    kernels inside it, so counting one would count that range twice. The optimizer's
    own annotation is such a range.

    Args:
        profiled: A finished profile.
        iters: Iterations inside it.

    Returns:
        One entry per kernel, descending by time.
    """
    rows = [
        (row.key, _self_us(row) / iters, row.count / iters)
        for row in profiled.key_averages()
        if row.device_type == DeviceType.CUDA
        and not getattr(row, "is_user_annotation", False)
    ]
    return sorted(rows, key=lambda entry: -entry[1])


def event_us(step: Callable[[], object], iters: int, device: torch.device) -> float:
    """Microseconds per call, from a device event pair around ``iters`` calls.

    The profiler is not the reference for how long the step takes. This is what the
    per-kernel rows are checked against: they cover an interval, and a sum over them
    that exceeds it is double counting rather than a slower step.

    Args:
        step: The callable to time.
        iters: Calls inside the interval.
        device: Device to time on.

    Returns:
        Microseconds per call.
    """
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(iters):
        step()
    stop.record()
    torch.cuda.synchronize(device)
    return 1000.0 * start.elapsed_time(stop) / iters


def main(argv: Sequence[str] | None = None) -> int:
    """Profile one mode and print the attribution.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If no device work was recorded, or if the rows sum past
            :data:`MAX_COVERAGE_PCT` of the interval that brackets them. Either way
            the attribution is not of this step, and it is wrong without failing.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    config = build_config(args)
    step = build_step(args, config, device)
    for _ in range(args.warmup):
        step()
    wall = event_us(step, args.iters, device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiled:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize(device)

    rows = device_rows(profiled, args.iters)
    total = sum(us for _, us, _ in rows)
    if total <= 0.0:
        raise ValueError("the profile recorded no device work")
    coverage = 100.0 * total / wall
    if coverage > MAX_COVERAGE_PCT:
        raise ValueError(
            f"the {len(rows)} device rows sum to {total:,.3f} us per iteration inside "
            f"an interval of {wall:,.3f} us, which is {coverage:,.2f}% of it; a row is "
            f"counted twice"
        )
    by_class: dict[str, float] = {}
    for name, us, _ in rows:
        by_class[classify(name)] = by_class.get(classify(name), 0.0) + us

    print(f"device {device_ordinal(device)}  mode {args.mode}  iters {args.iters}")
    print(
        f"geometry {config.n_layers} layers  d_model {config.d_model}  "
        f"3N {config.d_state}  d_head {config.d_head}  chunk {config.chunk_size}  "
        f"heads {config.n_heads}  groups {config.n_groups}  batch {args.batch}"
    )
    print(
        f"device time {total / 1000.0:,.3f} ms per iteration over {len(rows)} kernels, "
        f"{coverage:,.2f}% of the {wall / 1000.0:,.3f} ms the step takes unprofiled"
    )
    print()
    for label, us in sorted(by_class.items(), key=lambda entry: -entry[1]):
        print(f"{label:12s} {us / 1000.0:10,.3f} ms  {100.0 * us / total:6,.2f}%")
    glue = sum(by_class.get(label, 0.0) for label in GLUE)
    print(f"{'glue':12s} {glue / 1000.0:10,.3f} ms  {100.0 * glue / total:6,.2f}%")
    print()
    print(f"{'kernel':64s} {'ms/iter':>10s} {'share':>8s} {'calls':>8s}  class")
    for name, us, calls in rows[: args.rows]:
        print(
            f"{name[:64]:64s} {us / 1000.0:10,.3f} {100.0 * us / total:7,.2f}% "
            f"{calls:8,.1f}  {classify(name)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
