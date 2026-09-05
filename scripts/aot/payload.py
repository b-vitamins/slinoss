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
cell, which raises on a key the payload does not hold. It fails loudly, so it is
the whole CI entry point. A build host with no device cannot run the discovery
step at all -- ``build`` needs one.

What one run of a step discovers is one cell of the reachable set, and the decode
step's set has two axes. ``decode_fwd`` is compiled with
``(THREADS, row_group(N), N // row_group(N))`` and declares its operands' dtypes, so
it is specialized on the activation dtype and on ``N``; ``decode_carry`` is compiled
with ``(THREADS,)`` alone, so it is specialized on the dtype and one entry serves
every width. Nothing else is an axis: no extent, stride or pitch reaches a key, so
one entry serves every batch, head count, grouping and row count. ``build`` therefore
walks :data:`DECODE_WIDTHS` x :data:`DTYPES` for the decode mode -- 27 entries, and
around 670 KiB -- rather than the one cell its geometry flags name. ``--widths``
narrows the ladder for a deployment that knows its width; the default never does,
because a payload short a cell fails by compiling rather than by raising.

A decode cell runs against a state no prefill advanced. The prefill runs the chunked
scan, whose shared memory grows with ``d_state`` and stops two rungs short of the
decode row walk's 384, so a walk that prefilled would build a payload narrower than
the kernel it is building it for. ``cold`` prefills by default, because there the
prefill is the setup a first decode step is measured against; ``--no-prefill`` is what
a verification child passes.
"""

import time

_START = time.perf_counter()
"""Process start, as close to it as an import can get.

Read before torch is imported, because importing torch is most of the wall time a
cold process spends before its first step and hiding it inside the setup number
would misattribute it.
"""

import argparse
import itertools
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from slinoss import aot
from slinoss._cute import cache_events, compiled_launches, executor_count
from slinoss._precision import KERNEL_DTYPES
from slinoss.config import STATE_MULTIPLE, SLinOSSConfig
from slinoss.ops.xent import cross_entropy
from slinoss.perf.device import (
    contention,
    device_ordinal,
    require_cuda,
    smi_selector,
)
from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

_IMPORTED = time.perf_counter()
"""End of this module's imports."""

MODES = ("forward", "step", "decode")
"""What a step can be. ``step`` is forward, backward and an optimizer update."""

DTYPE = torch.bfloat16
"""The dtype the whole-sequence kernel path runs. float32 falls back to the
reference scan, so ``forward`` and ``step`` are built at this width alone."""

DTYPES = {str(dtype).removeprefix("torch."): dtype for dtype in KERNEL_DTYPES}
"""Activations a rowwise kernel reads, by flag spelling.

Derived from :data:`slinoss._precision.KERNEL_DTYPES` rather than listed, so a dtype
the registry gains is a dtype the payload gains. The decode step is rowwise at every
one of them, which is why its ladder has a dtype axis and the scan's does not.
"""

DECODE_WIDTHS = tuple(range(STATE_MULTIPLE, 8 * STATE_MULTIPLE + 1, STATE_MULTIPLE))
"""Every ``d_state`` a decode payload covers by default: 48 through 384.

A multiple of :data:`slinoss.config.STATE_MULTIPLE` because ``d_state`` is ``3N``
with ``N`` a multiple of half a warp, and stopping at 384 because the row walk holds
``N`` 3-vectors per state row in shared memory and 384 is the last width that fits.
The step is the multiple and not a coarser one: ``N`` sets ``row_group(N)`` and
``N // row_group(N)``, both compile-time, so every rung is its own key.
"""

PAYLOAD_MODES = ("none", "load", "strict")
"""Whether ``cold`` consults a payload, and whether a miss is fatal."""


class Cell(NamedTuple):
    """One point of the set a mode's payload has to cover.

    Attributes:
        d_state: Per-head state width ``3N``.
        dtype: Activation dtype, spelled as a key of :data:`DTYPES`.
    """

    d_state: int
    dtype: str

    def label(self) -> str:
        """This cell, for a progress line."""
        return f"3N {self.d_state:<4d} {self.dtype}"


def add_geometry(parser: argparse.ArgumentParser) -> None:
    """Add the geometry flags.

    The defaults are the acceptance geometry. Both subcommands take them, and a
    payload built at one geometry serves another only where the launch keys agree,
    so the two must be given the same flags.

    ``--d-state`` and ``--dtype`` are the two of these that reach a decode launch
    key. ``build`` overrides both per cell; ``cold`` runs the one it is given, which
    is how a verification child checks a cell rather than the parent's geometry.

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
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")


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
    "dtype",
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
        "--widths",
        nargs="+",
        type=int,
        default=list(DECODE_WIDTHS),
        help=(
            "d_state values the decode payload covers. Defaults to every legal "
            "width, since a payload short a width compiles there rather than "
            "raising. Narrow it only for a deployment that runs one."
        ),
    )
    build.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the child-process check that the payload covers every cell.",
    )
    add_geometry(build)

    cold = sub.add_parser("cold", help="Time one process to the end of its first step.")
    cold.add_argument("--mode", choices=MODES, default="step")
    cold.add_argument("--payload", choices=PAYLOAD_MODES, default="none")
    cold.add_argument("--payload-path", default=str(aot.PAYLOAD_DIR))
    cold.add_argument(
        "--no-prefill",
        action="store_true",
        help=(
            "Decode against a state no prefill advanced. The prefill runs the "
            "chunked scan, which does not fit at the widest d_state the decode "
            "kernel serves; skipping it is what lets a wide cell be checked at all."
        ),
    )
    add_geometry(cold)
    return parser.parse_args(argv)


def cells(mode: str, args: argparse.Namespace) -> tuple[Cell, ...]:
    """The set one mode's payload has to cover.

    The decode step is rowwise and runs at every dtype in :data:`DTYPES` and every
    width in ``--widths``, and each pair is its own launch key. The scan is neither:
    float32 falls back to the reference, and a chunked scan's keys carry the chunk
    rather than ``N``, so ``forward`` and ``step`` are the one cell their flags name.

    Args:
        mode: One of :data:`MODES`.
        args: The command line.

    Returns:
        The cells, in build order.
    """
    if mode != "decode":
        return (Cell(d_state=args.d_state, dtype=args.dtype),)
    widths = getattr(args, "widths", None) or [args.d_state]
    return tuple(
        Cell(d_state=width, dtype=name)
        for width, name in itertools.product(widths, DTYPES)
    )


def build_config(args: argparse.Namespace, d_state: int | None = None) -> SLinOSSConfig:
    """The geometry the step runs at.

    Args:
        args: The command line.
        d_state: Width to use instead of ``args.d_state``, for one cell of a ladder.

    Returns:
        The config.
    """
    return SLinOSSConfig(
        d_model=args.d_model,
        d_state=args.d_state if d_state is None else d_state,
        d_head=args.d_head,
        n_groups=args.groups,
        chunk_size=args.chunk,
        n_layers=args.layers,
        ffn_ratio=4.0,
        vocab_size=args.vocab,
    )


def build_step(
    mode: str,
    args: argparse.Namespace,
    config: SLinOSSConfig,
    device: torch.device,
    dtype: torch.dtype = DTYPE,
    prefill: bool = True,
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
        dtype: Activation dtype. One of :data:`DTYPES`. The scan has a kernel at
            :data:`DTYPE` alone; the decode step has one at every width.
        prefill: Advance the state through ``--prefill`` tokens of the chunked scan
            first. Ignored outside ``decode``. False decodes against the allocated
            zero state, which reaches the same two launchers -- no launch key carries
            a value -- and reaches them at every ``d_state`` the decode kernel serves
            rather than only at those the scan's shared budget also admits.

    Returns:
        The callable. Takes no arguments.
    """
    torch.manual_seed(0)
    stack = SLinOSSStack(config, device=device).to(dtype)
    vocab = config.vocab_size
    assert vocab is not None

    if mode == "decode":
        state = StackState.allocate(config, args.batch, device=device, dtype=dtype)
        if prefill:
            ids = torch.randint(0, vocab, (args.batch, args.prefill), device=device)
            stack(ids, state)
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


def _stamp(device: torch.device) -> None:
    """Print what else was on the device.

    A cold-start number is host time, so a foreign process moves it through the
    launches the step still makes. Reported rather than assumed away.

    Args:
        device: The device the step ran on.
    """
    ordinal = device_ordinal(device)
    seen = contention(ordinal)
    print(
        f"device {smi_selector(ordinal)}  exclusive {seen.exclusive}  "
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
    step = build_step(
        args.mode,
        args,
        config,
        device,
        DTYPES[args.dtype],
        prefill=not args.no_prefill,
    )
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
        f"mode {args.mode}  dtype {args.dtype}  payload {args.payload}"
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


def _child_argv(
    mode: str, args: argparse.Namespace, out: Path, cell: Cell
) -> list[str]:
    """The command line one verification child runs.

    ``sys.executable`` and ``__file__`` rather than ``-m``, so the child does not
    depend on the repository root being importable as a package.

    Args:
        mode: The mode to verify.
        args: The parent's command line.
        out: Payload directory.
        cell: The cell to verify. Overrides the parent's width and dtype, so a
            child checks the cell it was given rather than the parent's geometry.

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
    # The same omission the build made. A decode payload holds decode keys, so a
    # child that prefilled would demand the scan's keys of it and a strict load would
    # refuse them -- and at the widest two widths the scan does not fit at all.
    if mode == "decode":
        argv.append("--no-prefill")
    overridden = {"d_state": cell.d_state, "dtype": cell.dtype}
    for name in GEOMETRY:
        value = overridden.get(name, getattr(args, name))
        argv += [f"--{name.replace('_', '-')}", str(value)]
    return argv


def verify(modes: Sequence[str], args: argparse.Namespace, out: Path) -> None:
    """Run one strict-payload child per cell of every mode.

    The parent has every executor in its own cache, so it cannot tell a payload hit
    from a cache hit. A child process can: it loads the payload, and a key the
    payload does not hold raises instead of compiling.

    Per cell rather than per mode. A decode payload's whole claim is that it covers
    a set of widths and dtypes, and one child at one width verifies one of them.

    Args:
        modes: Modes to verify.
        args: The parent's command line.
        out: Payload directory.

    Raises:
        RuntimeError: If a child fails. Its output is printed first.
    """
    for mode in modes:
        for cell in cells(mode, args):
            print(f"--- verify {mode} {cell.label()}", flush=True)
            done = subprocess.run(_child_argv(mode, args, out, cell), check=False)
            if done.returncode != 0:
                raise RuntimeError(
                    f"the payload at {out} does not cover mode {mode} at "
                    f"{cell.label()}: the strict child exited {done.returncode}"
                )


def _covers(manifest: aot.Manifest, expected: int) -> None:
    """Refuse a decode payload short a cell.

    At least one ``decode_fwd`` key per cell walked and one ``decode_carry`` key per
    dtype, which is what the two launchers' compile-time arguments make them. Counted
    rather than matched key by key, so the check does not restate the key format and
    cannot drift from it: an equal total with the wrong split still fails, because a
    width that reached no forward kernel leaves the forward count short.

    A lower bound and not an equality because the manifest is written from the
    process's whole executor cache. This entry point runs one build per process, where
    the bound is tight; a caller that built twice in one process, or that ran other
    decode work first, inflates the count and is not accused of a shortfall for it.

    Args:
        manifest: What the build wrote.
        expected: Decode cells walked.

    Raises:
        RuntimeError: On a forward or carry count below the walked set's.
    """
    keys = [entry.key for entry in manifest.entries]
    forward = len({key for key in keys if "decode_fwd" in key})
    carry = len({key for key in keys if "decode_carry" in key})
    if forward < expected or carry < len(DTYPES):
        raise RuntimeError(
            f"the payload holds {forward} decode_fwd and {carry} decode_carry keys "
            f"for {expected} cells over {len(DTYPES)} dtypes; a cell that exported "
            f"no key is a shape a loaded payload compiles rather than refuses"
        )


def build(args: argparse.Namespace) -> int:
    """Compile every launch the modes reach, export it, and write the manifest.

    A mode is walked cell by cell: one step per ``(d_state, dtype)`` for ``decode``,
    one for each of the others. The executor cache spans the walk, so a key two cells
    share is compiled once and exported once. A decode cell skips the prefill, so the
    walk's reach is the decode kernel's and not the chunked scan's.

    Args:
        args: The command line.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If a mode compiled nothing, which means the kernel path was
            not taken and the payload would be silently short. If the decode entry
            count is below the walked cell set's. Or if verification fails.
    """
    device = require_cuda(args.device)
    out = Path(args.out)
    decode_cells = 0
    for mode in args.modes:
        for cell in cells(mode, args):
            decode_cells += mode == "decode"
            before = cache_events()
            config = build_config(args, cell.d_state)
            step = build_step(
                mode,
                args,
                config,
                device,
                DTYPES[cell.dtype],
                prefill=mode != "decode",
            )
            step()
            torch.cuda.synchronize(device)
            after = cache_events()
            gained = after.compiled - before.compiled
            if gained == 0 and before.compiled == 0:
                raise RuntimeError(
                    f"mode {mode} compiled no executor; the kernel path was not taken"
                )
            print(
                f"{mode:8s} {cell.label():<18s} compiled {gained:4d} in "
                f"{(after.compile_us - before.compile_us) / 1e6:8.3f} s",
                flush=True,
            )

    launches = compiled_launches()
    manifest = aot.build(launches, path=out)
    if decode_cells:
        _covers(manifest, decode_cells)
    total = sum((out / entry.file).stat().st_size for entry in manifest.entries)
    print(f"\npayload {out}")
    for field, value in manifest.identity._asdict().items():
        print(f"  {field:14s} {value}")
    print(
        f"  entries        {len(manifest.entries)} from {len(launches)} launches, "
        f"{total / 1024.0:,.1f} KiB, {total:,} bytes"
    )
    _stamp(device)
    if not args.no_verify:
        verify(args.modes, args, out)
        checked = sum(len(cells(mode, args)) for mode in args.modes)
        print(
            f"\nverified {len(manifest.entries)} entries over {checked} cells of "
            f"{len(args.modes)} modes"
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
