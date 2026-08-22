"""Paired A/B and address-level census of ``chunk_vector_bwd_kernel``.

``scripts/perf/profile_chunk_vector_bwd.py`` scores the kernel against its own
floor. This driver answers the two questions that one does not: which SASS site
holds a stall, and whether a candidate form beats the form in the tree by more
than the launch order is worth.

``--census`` collects :data:`slinoss.perf.ncu.SOURCE_TABLE` and reads its source
page at instruction granularity. Address is the only unambiguous key on a CuTe DSL
kernel, so the site map is keyed by it and the line number beside it is the block's,
not a file's.

``--pair`` measures two host wrappers in one
:func:`slinoss.perf.timing.measure_paired` loop over one allocated input set, and
compares every output bitwise. The arm under test is this tree's
``chunk_vector_backward``; the baseline is whatever module ``--baseline`` names,
which is how a candidate is measured against the form it replaces without a switch
inside the kernel. ``--null`` runs this tree against itself, which is the control
that says what the loop resolves at all.

    python3 scripts/perf/ab_chunk_vector_bwd.py --census --shape acceptance
    python3 scripts/perf/ab_chunk_vector_bwd.py --pair --shape acceptance \\
        --baseline slinoss.ops.so3ssd.cute.bwd.chunk_vector_base
    python3 scripts/perf/ab_chunk_vector_bwd.py --pair --null --shape acceptance
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Final, NamedTuple

import torch
from torch import Tensor

from slinoss.ops.so3ssd.cute.mma import WARPS_WIDE
from slinoss.perf.capture import profiler_window
from slinoss.perf.device import clock_policy, contention, device_ordinal, require_cuda
from slinoss.perf.ncu import (
    LSU_OPCODES,
    SOURCE_TABLE,
    SOURCE_VIEW,
    export_flags,
    import_command,
    ncu_command,
    pcsamp_metric,
    report_file,
)
from slinoss.perf.timing import measure_paired
from slinoss.perf.tools import resolve_tool
from slinoss.perf.workload import SHAPE_NAMES, OpShape, make_inputs, shape_by_name

KERNEL: Final = "chunk_vector_bwd"
"""Regex NCU narrows to. The main launch alone: the site map is inside it."""

DTYPES: Final = {"bf16": torch.bfloat16, "fp16": torch.float16}

MIO_OPCODES: Final = frozenset(("LDS", "LDSM", "STS", "SHFL", "MUFU", "ATOMS"))
"""Opcode classes that queue on the MIO pipe, without the modifier suffix.

``MUFU`` is the non-obvious member and the reason this set is not
:data:`slinoss.perf.ncu.LSU_OPCODES`: a transcendental issues to the MIO queue and
occupies no LSU slot, while a global access occupies the LSU port and no MIO slot.
``mio_throttle`` is a queue-full stall, so the set that prices it is this one.
"""

_ADDRESS: Final = "Address"
_LINE_NO: Final = "Line No"
_SOURCE: Final = "Source"
_FILE_PATH: Final = "File Path"
_FUNCTION_NAME: Final = "Function Name"
_INST: Final = "inst_executed"
_SAMPLES: Final = "smsp__pcsamp_sample_count"

_OPCODE: Final = re.compile(r"^\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)")


class Site(NamedTuple):
    """One SASS instruction of one kernel, with the counters correlated to it.

    Attributes:
        kernel: Kernel name, from the source page's ``Function Name``.
        address: SASS address, the only unambiguous key on a DSL kernel.
        line: Line number of the block the instruction sits under. The file is the
            traced entry module, so this is not per-file attribution.
        sass: The instruction text as NCU printed it.
        inst_count: Warp-instructions executed here.
        sample_count: PC samples taken here, issuing or not.
        stall_samples: Stall reason to not-issued PC samples.
    """

    kernel: str
    address: str
    line: int
    sass: str
    inst_count: int
    sample_count: int
    stall_samples: dict[str, int]

    @property
    def opcode(self) -> str:
        """Opcode class, predicate and modifier suffix removed."""
        matched = _OPCODE.match(self.sass)
        return "" if matched is None else matched.group(1).partition(".")[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--warps", type=int, default=WARPS_WIDE)
    parser.add_argument("--census", action="store_true", help="Collect the site map.")
    parser.add_argument("--pair", action="store_true", help="Measure two arms.")
    parser.add_argument(
        "--window",
        action="store_true",
        help="Run as the profiler's target: warm up, then launch inside the "
        "capture window. Emits nothing.",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="Module holding the baseline ``chunk_vector_backward``. Empty with "
        "--pair means --null.",
    )
    parser.add_argument(
        "--null",
        action="store_true",
        help="Both arms are this tree's wrapper. The control on the loop.",
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--event-iters", type=int, default=60)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--report", default="/tmp/cvb-source")
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Read the report already on disk instead of collecting one.",
    )
    return parser.parse_args(argv)


def build_runner(
    entry: Callable[..., object],
    shape: OpShape,
    device: torch.device,
    dtype: torch.dtype,
    splits: int | None,
    warps: int,
) -> tuple[Callable[[], object], Callable[[], None]]:
    """Allocate one input set and bind one host wrapper over it.

    Two arms of a pair share the input set, so the difference cannot be an address
    or a cache residency. The returned pair is the value-returning call, for
    parity, and the value-discarding call, for the timing loop.

    Args:
        entry: A ``chunk_vector_backward``, from this tree or from a baseline
            module.
        shape: The problem size.
        device: Where to allocate.
        dtype: Activation dtype.
        splits: Head-sum partial depth, or None for the operator's own.
        warps: Block width in warps.

    Returns:
        ``(call, run)``. ``call`` returns the named output tuple; ``run`` drops it.
    """
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=False)
    chunks = -(-shape.seq // shape.chunk)
    gen = torch.Generator(device=device).manual_seed(1)

    def randn(*size: int, dt: torch.dtype = torch.float32) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    state = (shape.bsz, shape.heads, chunks, shape.rows, shape.d_state)
    dinc = randn(*state, dt=dtype)
    zstart = randn(*state, dt=dtype)
    dlogp = randn(shape.bsz, shape.heads, chunks, shape.chunk)
    dchunk_rot = randn(shape.bsz, shape.heads, chunks, 3, 3)
    dchunk_scale = randn(shape.bsz, shape.heads, chunks)

    def call() -> object:
        return entry(
            inputs.dy,
            inputs.U,
            inputs.trans,
            inputs.K,
            inputs.B,
            inputs.C,
            dinc,
            zstart,
            dlogp,
            dchunk_rot,
            dchunk_scale,
            shape.chunk,
            splits=splits,
            warps=warps,
        )

    def run() -> None:
        call()

    return call, run


def parity(a: object, b: object) -> tuple[bool, str]:
    """Compare two output tuples bitwise, field by field.

    Args:
        a: The baseline arm's output.
        b: The arm under test's output.

    Returns:
        ``(clean, line)``. ``clean`` is True only when every field matches bit for
        bit. The line names each field, its worst absolute difference, and the
        count of differing elements.
    """
    fields = getattr(type(a), "_fields", ())
    clean = True
    parts: list[str] = []
    for name in fields:
        left = getattr(a, name)
        right = getattr(b, name)
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            continue
        same = torch.equal(left, right)
        clean = clean and same
        if same:
            parts.append(f"{name} bitwise")
            continue
        diff = (left.float() - right.float()).abs()
        parts.append(
            f"{name} max {diff.max().item():.3e} over "
            f"{int((left != right).sum().item()):,} of {left.numel():,}"
        )
    return clean, "  ".join(parts)


def _cell(row: Sequence[str], columns: dict[str, int], metric: str) -> int:
    """One metric cell of one source-page row, as an integer.

    Args:
        row: The row, restored to its header's width.
        columns: Metric name to column index for the block the row belongs to.
        metric: The metric to read.

    Returns:
        The value, rounded. Zero when the block dropped the column, which NCU does
        for a kernel with no traffic of that kind, and zero when the cell is not a
        number.
    """
    index = columns.get(metric, -1)
    if index < 0:
        return 0
    try:
        return round(float(row[index].strip().replace(",", "")))
    except ValueError:
        return 0


def parse_sites(text: str, *, reasons: Sequence[str]) -> tuple[Site, ...]:
    """Read a source page at instruction granularity.

    :func:`slinoss.perf.ncu.parse_source_csv` merges every instruction under its
    line, which on a DSL kernel merges two traced files that share a line number.
    This keeps the address, which nothing merges.

    Args:
        text: Stdout of :func:`slinoss.perf.ncu.import_command` with
            ``page="source"`` and ``print_source=SOURCE_VIEW``.
        reasons: Stall reasons to carry per site.

    Returns:
        One record per instruction, in page order.

    Raises:
        ValueError: If the page holds no instruction row, which is a page collected
            without ``CUTE_DSL_LINEINFO=1`` or imported at the wrong page.
    """
    out: list[Site] = []
    columns: dict[str, int] = {}
    sass_column = -1
    fields = 0
    kernel = ""
    line = 0
    for row in csv.reader(io.StringIO(text)):
        if len(row) == 2 and row[0] == _FILE_PATH:
            columns, line = {}, 0
            continue
        if len(row) == 2 and row[0] == _FUNCTION_NAME:
            kernel, columns, line = row[1], {}, 0
            continue
        if row and row[0] == _LINE_NO:
            columns = {}
            for index, name in enumerate(row):
                columns.setdefault(name, index)
            wide = [i for i, name in enumerate(row) if name == _SOURCE]
            sass_column = wide[1] if len(wide) > 1 else -1
            fields = len(row)
            line = 0
            continue
        if not columns or not any(cell.strip() for cell in row):
            continue
        # A line-aggregate row carries the high-level source text, which NCU quotes
        # without escaping the quotes inside it, so csv splits or merges the row on
        # the text's own punctuation and no column right of it decodes. Only the
        # line number is read off such a row, and it is column zero, ahead of the
        # damage. An instruction row carries no high-level text and cannot split.
        if row[0].strip().isdigit():
            line = int(row[0].strip())
            continue
        extra = len(row) - fields
        if extra > 0:
            row = [row[0], ",".join(row[1 : 2 + extra]), *row[2 + extra :]]
        if extra < 0:
            continue
        address = row[columns[_ADDRESS]].strip()
        if not address.startswith("0x"):
            continue

        out.append(
            Site(
                kernel=kernel,
                address=address,
                line=line,
                sass=row[sass_column].strip() if sass_column >= 0 else "",
                inst_count=_cell(row, columns, _INST),
                sample_count=_cell(row, columns, _SAMPLES),
                stall_samples={
                    reason: _cell(row, columns, pcsamp_metric(reason))
                    for reason in reasons
                },
            )
        )
    if not out:
        raise ValueError(
            "no instruction row in the ncu source page; collect with "
            "CUTE_DSL_LINEINFO=1 and import with page='source'"
        )
    return tuple(out)


def print_opcodes(sites: Sequence[Site], launches: int) -> None:
    """Print the opcode census, MIO and LSU subtotals beside it.

    Args:
        sites: Every instruction of the window.
        launches: Launches in the window, the divisor onto a per-launch footing.
    """
    total: dict[str, list[int]] = {}
    for one in sites:
        seen = total.setdefault(one.opcode, [0, 0])
        seen[0] += one.inst_count
        seen[1] += one.sample_count
    inst = sum(v[0] for v in total.values())
    samples = sum(v[1] for v in total.values())
    mio = sum(v[0] for k, v in total.items() if k in MIO_OPCODES)
    lsu = sum(v[0] for k, v in total.items() if k in LSU_OPCODES)
    print(f"opcodes      {inst / launches:,.0f} inst/launch  {samples:,} samples")
    print(
        f"  mio        {mio / launches:,.0f} inst/launch "
        f"({100.0 * mio / inst:.2f}%)   lsu {lsu / launches:,.0f} "
        f"({100.0 * lsu / inst:.2f}%)"
    )
    ranked = sorted(total.items(), key=lambda kv: -kv[1][0])
    for opcode, (count, sampled) in ranked[:20]:
        tag = (
            "mio" if opcode in MIO_OPCODES else ("lsu" if opcode in LSU_OPCODES else "")
        )
        print(
            f"  {opcode:8s} {count / launches:12,.0f} inst "
            f"{100.0 * count / inst:6.2f}%  samp {100.0 * sampled / samples:6.2f}% "
            f"{tag}"
        )


def print_sites(sites: Sequence[Site], reason: str, top: int, launches: int) -> None:
    """Print the site map, ranked by one stall reason's not-issued samples.

    Args:
        sites: Every instruction of the window.
        reason: The stall reason to rank by.
        top: Rows to print.
        launches: Launches in the window.
    """
    total = sum(one.stall_samples.get(reason, 0) for one in sites)
    allsamp = sum(one.sample_count for one in sites)
    print(f"sites        {reason} {total:,} samples, {allsamp:,} over every reason")
    ranked = sorted(sites, key=lambda one: -one.stall_samples.get(reason, 0))
    for one in ranked[:top]:
        held = one.stall_samples.get(reason, 0)
        if held == 0:
            break
        print(
            f"  {one.address} line {one.line:5d} "
            f"{held:7,} ({100.0 * held / total:5.2f}% of {reason}, "
            f"{100.0 * held / allsamp:5.2f}% of all) "
            f"inst {one.inst_count / launches:9,.0f}  {one.sass[:56]}"
        )


def target_argv(args: argparse.Namespace) -> list[str]:
    """The command NCU attaches to.

    Args:
        args: The parsed command line.

    Returns:
        The argv, this file in ``--window`` mode at the same geometry.
    """
    out = [
        sys.executable,
        __file__,
        "--window",
        "--shape",
        args.shape,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--warps",
        str(args.warps),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]
    if args.splits is not None:
        out += ["--splits", str(args.splits)]
    return out


def run_census(args: argparse.Namespace, device: torch.device) -> int:
    """Collect the source pass and print the census.

    Args:
        args: The parsed command line.
        device: The device, for the header.

    Returns:
        Process exit status.
    """
    reasons = (
        "barrier",
        "mio_throttle",
        "short_scoreboard",
        "long_scoreboard",
        "wait",
        "math_pipe_throttle",
    )
    binary = resolve_tool(args.ncu)
    if not args.reuse:
        _run(
            ncu_command(
                SOURCE_TABLE,
                target_argv(args),
                ncu=binary,
                extra=(
                    *export_flags(args.report),
                    "--kernel-name",
                    f"regex:{KERNEL}",
                ),
            )
        )
    written = report_file(args.report)
    text = _run(
        import_command(written, ncu=binary, page="source", print_source=SOURCE_VIEW)
    )
    sites = parse_sites(text, reasons=reasons)
    kernels = sorted({one.kernel for one in sites})
    print(f"device       {device} ord {device_ordinal(device)}")
    print(f"shape        {args.shape} warps {args.warps} splits {args.splits}")
    print(f"report       {written}")
    print(f"kernels      {kernels}")
    print_opcodes(sites, args.iters)
    for one in reasons:
        print_sites(sites, one, args.top, args.iters)
    return 0


def _run(command: Sequence[str]) -> str:
    """Run one NCU invocation and return its stdout.

    :func:`slinoss.perf.ncu.run_source` would do both invocations, and its parse
    raises on this kernel's page: a line-aggregate row whose source text holds a
    quote arrives one cell short of its header, which
    :func:`slinoss.perf.ncu._aligned_row` refuses. The address-level read here does
    not need that row's metrics.

    Args:
        command: The argv.

    Returns:
        Stdout.

    Raises:
        RuntimeError: If the invocation exits nonzero.
    """
    done = subprocess.run(list(command), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{command[0]} failed: {done.stderr[-2000:]}")
    return done.stdout


def run_pair(args: argparse.Namespace, device: torch.device) -> int:
    """Measure two arms in one loop and compare their outputs bitwise.

    Args:
        args: The parsed command line.
        device: Where to allocate and time.

    Returns:
        Process exit status. Nonzero when the arms disagree bitwise.
    """
    from slinoss.ops.so3ssd.cute.bwd.chunk_vector import chunk_vector_backward

    shape = shape_by_name(args.shape)
    dtype = DTYPES[args.dtype]
    if args.null or not args.baseline:
        base_entry: Callable[..., object] = chunk_vector_backward
        base_label = "null"
    else:
        base_entry = importlib.import_module(args.baseline).chunk_vector_backward
        base_label = "base"
    a_call, a_run = build_runner(
        base_entry, shape, device, dtype, args.splits, args.warps
    )
    b_call, b_run = build_runner(
        chunk_vector_backward, shape, device, dtype, args.splits, args.warps
    )
    clean, detail = parity(a_call(), b_call())
    torch.cuda.synchronize(device)
    out = measure_paired(
        base_label,
        a_run,
        "arm",
        b_run,
        label=f"cvb.{args.shape}",
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
        clocks=clock_policy(device_ordinal(device)),
    )
    print(f"device       {device} ord {device_ordinal(device)}")
    print(f"shape        {shape.describe()}")
    print(f"baseline     {args.baseline or 'this tree (null control)'}")
    print(f"clocks       {out.timed.clocks}")
    print(f"contention   {contention(device_ordinal(device))}")
    print(f"parity       {'bitwise' if clean else 'DIFFERS'}  {detail}")
    print(f"verdict      {out.comparison.verdict()}")
    print(
        f"medians      {base_label} {out.comparison.a_median_duration_us:,.3f} us  "
        f"arm {out.comparison.b_median_duration_us:,.3f} us"
    )
    return 0 if clean else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Census, pair, or run as the profiler's target.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If no mode was asked for.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    if args.window:
        from slinoss.ops.so3ssd.cute.bwd.chunk_vector import chunk_vector_backward

        _, run = build_runner(
            chunk_vector_backward,
            shape_by_name(args.shape),
            device,
            DTYPES[args.dtype],
            args.splits,
            args.warps,
        )
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize(device)
        with profiler_window(device):
            for _ in range(args.iters):
                run()
        torch.cuda.synchronize(device)
        return 0
    if args.census:
        return run_census(args, device)
    if args.pair:
        return run_pair(args, device)
    raise ValueError("nothing to do: pass --census, --pair, or --window")


if __name__ == "__main__":
    raise SystemExit(main())
