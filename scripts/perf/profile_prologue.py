"""What the chunk-boundary prologue costs at the model geometry, both ways.

The forward's ``increment_passing_fwd`` produces ``cquat``, ``cscale`` and
``zstart``, and the backward reads all three. Holding them across the step deletes
the launch that would rebuild them there and pays in live activation. The trade has
two units, device time and allocator bytes, and neither is readable off the other,
so both are measured here in one process at the geometry the stack trains at.

    python3 scripts/perf/profile_prologue.py

The policy is not a flag. Which one the tree holds is read off the launch counts: a
step that launches ``increment_passing_fwd`` twice per layer is rematerializing, and
one that launches it once per layer is not. A driver that took the policy as a flag
would label whichever tree it was pointed at.

Peak allocated bytes is what the memory side is judged on, not the saved-set total.
A tensor the saved set holds and something else keeps alive anyway costs no peak,
and the peak is what runs out. The saved set is reported beside it because the two
disagreeing is the interesting case: bytes saved and not peaked are bytes some other
consumer already held.

Each repeat is a whole measurement, timed loop and profile both, so every figure
printed twice is a figure measured twice.

Both policies measured back to back, on one A6000 at unlocked clocks sharing the
part with two contexts holding 6,564 MiB at 0% utilization, two repeats each, 13
layers, d_model 576, 3N 240, d_head 64, chunk 64, 18 heads, batch 4, seqlen 2048,
bfloat16. Recompute: step 220.378 and 220.795 ms, peak allocated 9,275.99 MiB, peak
reserved 9,366.00 MiB, prologue 20.722 and 20.720 ms per step over 52 launches.
Saving all three, which is what the tree holds: step 210.442 and 210.752 ms, peak
allocated 11,187.88 MiB, peak reserved 11,328.00 MiB, prologue 10.333 and 10.335 ms
over 26. The trade is 9.99 ms of a 220.4 ms step, 4.53%, against 1,911.89 MiB of
peak, 20.61%, and 191 MiB of peak per millisecond bought. Half-widths were 0.07% to
0.19%, so the delta resolves by two orders of magnitude.

Per-launch time does not move: ``state_passing_fwd`` 428.8 us against 429.1,
``chunk_increment_fwd`` 367.9 against 366.7. The policy deletes launches; it does not
make the surviving ones faster, which is why the whole win is readable off the launch
counts alone.

Both arms above predate the fusion of the increment into the recurrence, so their
prologue is two launches a layer where the tree now makes one. The trade is unchanged
in kind: the fusion halves the time term and moves no byte of the memory term. Same
geometry, same part, both trees measured in one session, two repeats each, saving all
three either way. The pair: step 207.625 and 208.072 ms, prologue 10.314 and 10.325 ms
per step over 26 launches, 4.98% of device time. Fused: step 202.839 and 203.393 ms,
prologue 5.357 and 5.398 ms over 13 launches at 412.0 and 415.2 us a call, 2.64%. Peak
allocated 11,187.88 MiB, peak reserved 11,328.00 MiB, and the saved set 419 storages
over 7,227.12 MiB, all three the same to the byte in both trees: the increment buffer
the fusion deleted was never live at the instant the peak is taken.

The retained tensors are 1,755.6 MiB of the 1,911.89 MiB the peak rose by: 13 layers
of ``zstart`` at 135.00 MiB, ``cquat`` at 0.036, ``cscale`` at 0.009. The remaining
156.3 MiB is not a tensor. The peak is a maximum over a step and the two policies do
not reach it at the same instant, which is the reason the memory side is measured
rather than added up. The saved set rose by less than either figure, 5,687.65 MiB
over 386 storages to 7,227.12 MiB over 419.

Saving less is not available, and the fusion is what settles it: one launch produces
all three, so any subset leaves that launch in the backward and returns none of its
time for whatever share of the 1,755 MiB the subset holds. Every partial policy is
dominated, so the choice is the two arms above and nothing between them.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity, profile

from scripts.perf.attribute_step import DTYPE, build_config, build_step, device_rows
from slinoss.perf.device import (
    clock_policy,
    contention,
    device_ordinal,
    require_cuda,
    smi_selector,
)
from slinoss.perf.memory import SavedTensorProbe, memory_peaks, reset_memory_peaks
from slinoss.perf.timing import measure
from slinoss.perf.units import mib_from_bytes, pct_of

PROLOGUE = ("increment_passing_fwd",)
"""Kernels the recompute policy relaunches inside the backward."""

BOUNDARY = ("chunk_start_bwd", "state_passing_bwd")
"""The rest of the chunk-boundary region, for scale in the same table."""

RECOMPUTE_RATIO = 1.5
"""Launches per layer above which a prologue kernel is being rematerialized.

Halfway between the one launch per layer the forward makes and the two a
rematerializing backward makes, so the reading does not turn on an exact count that
a fused or split launch would change.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--d-state", type=int, default=240)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Timed steps per repeat. Each is one sample of the step duration and "
        "one sample of nothing else: the allocator peak is a maximum over the loop.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=3,
        help="Profiled steps per repeat, for the per-kernel rows.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Whole measurements. Two is the floor: one is not a measurement.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def step_args(args: argparse.Namespace) -> argparse.Namespace:
    """The namespace ``attribute_step`` builds its training step from.

    The step priced here is the step that driver attributes, so the geometry and the
    callable come from it rather than from a second definition.

    Args:
        args: This driver's command line.

    Returns:
        A namespace carrying ``attribute_step``'s fields.
    """
    return argparse.Namespace(
        mode="step",
        batch=args.batch,
        seqlen=args.seqlen,
        prefill=0,
        d_model=args.d_model,
        d_state=args.d_state,
        d_head=args.d_head,
        chunk=args.chunk,
        groups=args.groups,
        layers=args.layers,
        vocab=args.vocab,
        iters=args.profile_iters,
        warmup=args.warmup,
        device=args.device,
    )


@dataclass(frozen=True)
class KernelUse:
    """One kernel's whole contribution to a step, over every symbol it compiles to.

    A kernel with a compile-time variant reaches the profiler as two symbols, and a
    table that read one row would miss half the launches. The needle claims both.

    Attributes:
        needle: Kernel name as the source spells it.
        duration_us: Device time per step, summed over the matching symbols.
        call_count: Launches per step, summed the same way.
        symbol_count: Distinct symbols the needle matched. Zero means the step never
            launched it.
    """

    needle: str
    duration_us: float
    call_count: float
    symbol_count: int

    @property
    def call_duration_us(self) -> float:
        """Device time per launch, or zero when nothing launched."""
        return self.duration_us / self.call_count if self.call_count else 0.0


def kernel_use(
    rows: Sequence[tuple[str, float, float]], needles: Sequence[str]
) -> tuple[KernelUse, ...]:
    """Sum the per-kernel rows a needle names.

    Args:
        rows: Per-kernel name, microseconds per iteration, calls per iteration, as
            :func:`scripts.perf.attribute_step.device_rows` returns.
        needles: Kernel names to claim, as the source spells them.

    Returns:
        One record per needle, in the order given.
    """
    return tuple(
        KernelUse(
            needle=needle,
            duration_us=sum(us for name, us, _ in rows if needle in name),
            call_count=sum(calls for name, _, calls in rows if needle in name),
            symbol_count=sum(1 for name, _, _ in rows if needle in name),
        )
        for needle in needles
    )


def policy(prologue: Sequence[KernelUse], layers: int) -> str:
    """Which prologue policy the tree holds, read off the launch counts.

    Args:
        prologue: Use records for :data:`PROLOGUE`.
        layers: Layers in the stack. One launch per layer is the forward's own.

    Returns:
        A line naming the policy and the counts it was read from.
    """
    counts = ", ".join(f"{u.needle} {u.call_count:,.1f}" for u in prologue)
    relaunched = [u.needle for u in prologue if u.call_count > RECOMPUTE_RATIO * layers]
    verdict = "recompute" if relaunched else "saved"
    return f"policy {verdict} over {layers} layers: {counts}"


def print_rows(uses: Sequence[KernelUse], total_us: float) -> None:
    """Print one per-kernel table.

    Args:
        uses: What to print, in table order.
        total_us: Device time per step over every kernel, the share denominator.
    """
    for use in uses:
        print(
            f"  {use.needle:22s} {use.duration_us / 1000.0:8,.3f} ms/step "
            f"{use.call_duration_us:9,.1f} us/call {use.call_count:7,.1f} calls "
            f"{pct_of(use.duration_us, total_us):6,.2f}% {use.symbol_count} symbols"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Price the prologue policy the tree holds and print the report.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If a profiled repeat recorded no device work, which leaves every
            share denominator zero.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    index = device_ordinal(device)
    config = build_config(step_args(args))
    step = build_step(step_args(args), config, device)

    print(
        f"device {index} {smi_selector(index)} {torch.cuda.get_device_name(index)} "
        f"{clock_policy(index).stamp}"
    )
    print(
        f"geometry {config.n_layers} layers  d_model {config.d_model}  "
        f"3N {config.d_state}  d_head {config.d_head}  chunk {config.chunk_size}  "
        f"heads {config.n_heads}  batch {args.batch}  seqlen {args.seqlen}  {DTYPE}"
    )
    before = contention(index)
    print(f"before {before.stamp}")
    print(f"       {before.detail}")

    for repeat in range(1, args.repeats + 1):
        reset_memory_peaks(device)
        timed = measure(
            step,
            label="step",
            iters=args.iters,
            warmup=args.warmup,
            device=device,
        )
        peaks = memory_peaks("step", device)
        spread = timed.total
        print()
        print(
            f"repeat {repeat}  step {spread.median_duration_us / 1000.0:,.3f} ms  "
            f"range {spread.spread_pct:,.2f}%  half-width {spread.resolution_pct:,.2f}%"
            f"  over {spread.sample_count:,d} steps  {timed.clocks}"
        )
        print(
            f"  peak allocated {mib_from_bytes(peaks.peak_allocated_bytes):11,.2f} MiB"
            f"  reserved {mib_from_bytes(peaks.peak_reserved_bytes):11,.2f} MiB"
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiled:
            for _ in range(args.profile_iters):
                step()
            torch.cuda.synchronize(device)
        rows = device_rows(profiled, args.profile_iters)
        total_us = sum(us for _, us, _ in rows)
        if total_us <= 0.0:
            raise ValueError("the profile recorded no device work")
        prologue = kernel_use(rows, PROLOGUE)
        boundary = kernel_use(rows, BOUNDARY)
        print(
            f"  device {total_us / 1000.0:,.3f} ms/step over {len(rows)} kernels, "
            f"{pct_of(total_us, spread.median_duration_us):,.2f}% of the step"
        )
        print_rows((*prologue, *boundary), total_us)
        prologue_us = sum(use.duration_us for use in prologue)
        print(
            f"  prologue {prologue_us / 1000.0:,.3f} ms/step, "
            f"{pct_of(prologue_us, total_us):,.2f}% of device time"
        )
        print(f"  {policy(prologue, config.n_layers)}")

    probe = SavedTensorProbe()
    with probe:
        step()
    torch.cuda.synchronize(device)
    saved = probe.report("stack step")
    print()
    print(
        f"saved set {saved.storage_count:,d} storages  {saved.saved_mib:,.2f} MiB  "
        f"{saved.save_event_count:,d} save events"
    )
    after = contention(index)
    print(f"after {after.stamp}")
    print(f"      {after.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
