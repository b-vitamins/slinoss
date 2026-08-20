"""Build an ahead-of-time payload, and measure what it costs to start without one.

A cold process compiles every executor its first step needs. ``build`` runs the
steps once with the executor cache empty, exports whatever they compiled, and
writes the manifest; ``cold`` runs one step and reports where the seconds before it
went.

    python3 scripts/aot/payload.py build
    python3 scripts/aot/payload.py cold --mode step
    python3 scripts/aot/payload.py cold --mode step --payload strict

The key set is discovered, not listed. ``build`` reads
:func:`slinoss._cute.compiled_launches`, so a kernel that gains a
:class:`cutlass.Constexpr`, or a launcher reached only by the backward, enters the
payload the moment a step reaches it. A hand-written list would drift silently and
the drift would show up as a fallback compile.

``build`` verifies in child processes by default: one ``cold --payload strict`` per
mode, which raises on a key the payload does not hold. It fails loudly, so it is
the whole CI entry point. A build host with no device cannot run the discovery
step at all -- ``build`` needs one.
"""

import time

_START = time.perf_counter()
"""Process start, as close to it as an import can get.

Read before torch is imported, because importing torch is most of the wall time a
cold process spends before its first step and hiding it inside the setup number
would misattribute it.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from slinoss import aot
from slinoss._cute import cache_events, compiled_launches, executor_count
from slinoss.config import SLinOSSConfig
from slinoss.perf.device import contention, device_ordinal, require_cuda
from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

_IMPORTED = time.perf_counter()
"""End of this module's imports."""

MODES = ("forward", "step", "decode")
"""What a step can be. ``step`` is forward, backward and an optimizer update."""

DTYPE = torch.bfloat16
"""The dtype the kernel path runs. float32 falls back to the reference scan."""

PAYLOAD_MODES = ("none", "load", "strict")
"""Whether ``cold`` consults a payload, and whether a miss is fatal."""


def add_geometry(parser: argparse.ArgumentParser) -> None:
    """Add the geometry flags.

    The defaults are the acceptance geometry. Both subcommands take them, and a
    payload built at one geometry serves another only where the launch keys agree,
    so the two must be given the same flags.

    Args:
        parser: Parser to add to.
    """
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
    parser.add_argument("--device", default="cuda")


GEOMETRY = (
    "batch",
    "seqlen",
    "prefill",
    "d_model",
    "d_state",
    "d_head",
    "chunk",
    "groups",
    "layers",
    "vocab",
    "device",
)
"""The geometry flags, by attribute name, for passing on to a child process."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Compile, export and verify a payload.")
    build.add_argument(
        "--out",
        default=str(aot.PAYLOAD_DIR),
        help="Payload directory. Defaults to the one the package loads from.",
    )
    build.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    build.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the child-process check that the payload covers every mode.",
    )
    add_geometry(build)

    cold = sub.add_parser("cold", help="Time one process to the end of its first step.")
    cold.add_argument("--mode", choices=MODES, default="step")
    cold.add_argument("--payload", choices=PAYLOAD_MODES, default="none")
    cold.add_argument("--payload-path", default=str(aot.PAYLOAD_DIR))
    add_geometry(cold)
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
    mode: str, args: argparse.Namespace, config: SLinOSSConfig, device: torch.device
) -> Callable[[], object]:
    """The callable one step calls once.

    ``forward`` is inference over a full sequence, ``step`` is that plus the
    backward and a fused optimizer update, ``decode`` is one token per sequence
    against a state a prefill has already advanced. The prefill is part of building
    the decode step, so its compiles land in the setup interval rather than in the
    decode one.

    Args:
        mode: One of :data:`MODES`.
        args: The geometry.
        config: The geometry, resolved.
        device: Where to run.

    Returns:
        The callable. Takes no arguments.
    """
    torch.manual_seed(0)
    stack = SLinOSSStack(config, device=device).to(DTYPE)
    vocab = config.vocab_size
    assert vocab is not None

    if mode == "decode":
        state = StackState.allocate(config, args.batch, device=device, dtype=DTYPE)
        stack(torch.randint(0, vocab, (args.batch, args.prefill), device=device), state)
        token = torch.randint(0, vocab, (args.batch, 1), device=device)
        return lambda: stack(token, state)

    ids = torch.randint(0, vocab, (args.batch, args.seqlen), device=device)
    if mode == "forward":

        def forward() -> object:
            with torch.no_grad():
                return stack(ids)

        return forward

    labels = torch.randint(0, vocab, (args.batch, args.seqlen), device=device)
    optimizer = torch.optim.AdamW(stack.parameters(), lr=0.0)

    def train() -> object:
        optimizer.zero_grad(set_to_none=False)
        logits = stack(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten()
        )
        loss.backward()
        optimizer.step()
        return loss.detach()

    return train


def _physical(device: torch.device) -> int:
    """The ``nvidia-smi`` index of a torch device.

    ``CUDA_VISIBLE_DEVICES`` renumbers torch's ordinals and not the driver's, so
    probing the torch ordinal would report a different part than the step ran on.

    Args:
        device: The device.

    Returns:
        The driver index, or the torch ordinal if the variable is unset or names
        devices by UUID.
    """
    ordinal = device_ordinal(device)
    visible = [
        item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    ]
    if ordinal < len(visible) and visible[ordinal].isdigit():
        return int(visible[ordinal])
    return ordinal


def _stamp(device: torch.device) -> None:
    """Print what else was on the device.

    A cold-start number is host time, so a foreign process moves it through the
    launches the step still makes. Reported rather than assumed away.

    Args:
        device: The device the step ran on.
    """
    index = _physical(device)
    seen = contention(index)
    print(
        f"device {index}  exclusive {seen.exclusive}  "
        f"foreign {seen.foreign_process_count} processes "
        f"{seen.foreign_memory_mib:,.0f} MiB"
    )


def cold(args: argparse.Namespace) -> int:
    """Time one process from its own start to the end of its first step.

    Args:
        args: The command line.

    Returns:
        Process exit status.

    Raises:
        FileNotFoundError: If a payload was asked for and the directory holds none.
        ValueError: If a strict payload was asked for and the step compiled
            anything. Strict mode raises on the miss itself; this catches a compile
            that reached the DSL by some other path.
    """
    setup0 = time.perf_counter()
    device = require_cuda(args.device)
    loaded = None
    if args.payload != "none":
        loaded = aot.use(args.payload_path, strict=args.payload == "strict")
        if loaded is None:
            raise FileNotFoundError(f"no payload at {args.payload_path} to load")
    config = build_config(args)
    step = build_step(args.mode, args, config, device)
    torch.cuda.synchronize(device)
    setup1 = time.perf_counter()
    during_setup = cache_events()
    step()
    torch.cuda.synchronize(device)
    ended = time.perf_counter()
    events = cache_events()

    if args.payload == "strict" and events.compiled:
        raise ValueError(
            f"a strict payload was loaded and the process still compiled "
            f"{events.compiled} executors"
        )
    total = ended - _START
    compile_s = events.compile_us / 1e6
    print(
        f"mode {args.mode}  payload {args.payload}"
        + (f" ({len(loaded)} entries)" if loaded is not None else "")
    )
    print(
        f"geometry {config.n_layers} layers  d_model {config.d_model}  "
        f"3N {config.d_state}  d_head {config.d_head}  chunk {config.chunk_size}  "
        f"heads {config.n_heads}  groups {config.n_groups}  batch {args.batch}  "
        f"seqlen {args.seqlen}"
    )
    _stamp(device)
    print(f"import      {_IMPORTED - _START:9.3f} s")
    print(f"setup       {setup1 - setup0:9.3f} s")
    print(f"first step  {ended - setup1:9.3f} s")
    print(f"total       {total:9.3f} s  process start to end of first step")
    print(
        f"compile     {compile_s:9.3f} s  "
        f"{during_setup.compile_us / 1e6:.3f} s in setup, "
        f"{(events.compile_us - during_setup.compile_us) / 1e6:.3f} s in the step"
    )
    print(f"other       {total - compile_s:9.3f} s  total less compile")
    print(
        f"executors {executor_count()}  compiled {events.compiled}  "
        f"cache hits {events.hits}  payload hits {events.payload_hits}  "
        f"payload misses {events.payload_misses}  "
        f"dsl hits {events.dsl_hits}  dsl misses {events.dsl_misses}"
    )
    return 0


def _child_argv(mode: str, args: argparse.Namespace, out: Path) -> list[str]:
    """The command line one verification child runs.

    ``sys.executable`` and ``__file__`` rather than ``-m``, so the child does not
    depend on the repository root being importable as a package.

    Args:
        mode: The mode to verify.
        args: The parent's command line.
        out: Payload directory.

    Returns:
        The argv vector.
    """
    argv = [
        sys.executable,
        __file__,
        "cold",
        "--mode",
        mode,
        "--payload",
        "strict",
        "--payload-path",
        str(out),
    ]
    for name in GEOMETRY:
        argv += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
    return argv


def verify(modes: Sequence[str], args: argparse.Namespace, out: Path) -> None:
    """Run one strict-payload child per mode.

    The parent has every executor in its own cache, so it cannot tell a payload hit
    from a cache hit. A child process can: it loads the payload, and a key the
    payload does not hold raises instead of compiling.

    Args:
        modes: Modes to verify.
        args: The parent's command line.
        out: Payload directory.

    Raises:
        RuntimeError: If a child fails. Its output is printed first.
    """
    for mode in modes:
        print(f"--- verify {mode}", flush=True)
        done = subprocess.run(_child_argv(mode, args, out), check=False)
        if done.returncode != 0:
            raise RuntimeError(
                f"the payload at {out} does not cover mode {mode}: the strict child "
                f"exited {done.returncode}"
            )


def build(args: argparse.Namespace) -> int:
    """Compile every launch the modes reach, export it, and write the manifest.

    Args:
        args: The command line.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If a mode compiled nothing, which means the kernel path was
            not taken and the payload would be silently short. Or if verification
            fails.
    """
    device = require_cuda(args.device)
    out = Path(args.out)
    config = build_config(args)
    for mode in args.modes:
        before = cache_events()
        step = build_step(mode, args, config, device)
        step()
        torch.cuda.synchronize(device)
        after = cache_events()
        gained = after.compiled - before.compiled
        if gained == 0 and before.compiled == 0:
            raise RuntimeError(
                f"mode {mode} compiled no executor; the kernel path was not taken"
            )
        print(
            f"{mode:8s} compiled {gained:4d} in "
            f"{(after.compile_us - before.compile_us) / 1e6:8.3f} s",
            flush=True,
        )

    launches = compiled_launches()
    manifest = aot.build(launches, path=out)
    total = sum((out / entry.file).stat().st_size for entry in manifest.entries)
    print(f"\npayload {out}")
    for field, value in manifest.identity._asdict().items():
        print(f"  {field:14s} {value}")
    print(
        f"  entries        {len(manifest.entries)} from {len(launches)} launches, "
        f"{total / 1024.0:,.1f} KiB"
    )
    _stamp(device)
    if not args.no_verify:
        verify(args.modes, args, out)
        print(
            f"\nverified {len(manifest.entries)} entries over {len(args.modes)} modes"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a subcommand.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    return cold(args) if args.command == "cold" else build(args)


if __name__ == "__main__":
    raise SystemExit(main())
