"""Paired A/B, counter table, and address-level census of ``start_passing_bwd_kernel``.

``scripts/perf/profile_start_passing_bwd.py`` scores the kernel against its own DRAM
floor and against the pair it replaced. This driver answers the three questions that
one does not: which SASS site holds a stall, what the launch's own resource and
instruction counters read on each side of a change, and whether a candidate form
beats the form in the tree by more than the launch order is worth.

``--census`` collects :data:`slinoss.perf.ncu.SOURCE_TABLE` and reads its source page
at instruction granularity. Address is the only unambiguous key on a CuTe DSL kernel,
so the site map is keyed by it and the line number beside it is the block's, not a
file's.

``--counters`` collects :data:`COUNTER_TABLE`, the resource and traffic counters an
arm on this kernel is adjudicated on. Registers and the local sector pair are read
together, because a register jump reports as LSU instructions rather than as spill.

``--pair`` measures two host wrappers in one
:func:`slinoss.perf.timing.measure_paired` loop over one allocated input set and
compares every output bitwise. Side A is the baseline ``--baseline`` names and side B
is this tree, so a harness that built the base on both sides reads as a null.
``--null`` is that control on purpose: both arms are this tree.

    python3 scripts/perf/ab_start_passing_bwd.py --census --shape acceptance --groups 1
    python3 scripts/perf/ab_start_passing_bwd.py --counters --shape acceptance \\
        --groups 1 --baseline slinoss.ops.so3ssd.cute.bwd.start_passing_base
    python3 scripts/perf/ab_start_passing_bwd.py --pair --shape acceptance --groups 1 \\
        --baseline slinoss.ops.so3ssd.cute.bwd.start_passing_base
    python3 scripts/perf/ab_start_passing_bwd.py --pair --null --shape acceptance
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

from slinoss.ops.so3ssd.cute.bwd.start_passing import SPLIT
from slinoss.ops.so3ssd.cute.mma import WARPS_WIDE
from slinoss.perf.capture import profiler_window
from slinoss.perf.device import clock_policy, contention, device_ordinal, require_cuda
from slinoss.perf.ncu import (
    LSU_OPCODES,
    SOURCE_TABLE,
    SOURCE_VIEW,
    NcuTable,
    export_flags,
    import_command,
    ncu_command,
    pcsamp_metric,
    report_file,
    run_ncu,
)
from slinoss.perf.timing import measure_paired
from slinoss.perf.tools import resolve_tool
from slinoss.perf.workload import SHAPE_NAMES, OpShape, make_inputs, shape_by_name

KERNEL: Final = "start_passing_bwd"
"""Regex NCU narrows to. The fused launch alone, never its prefix producer."""

DTYPES: Final = {"bf16": torch.bfloat16, "fp16": torch.float16}

ENTRY: Final = "start_passing_backward"
"""Host entry both sides of a pair are bound over, in the tree and in a baseline."""

COUNTER_TABLE: Final = NcuTable(
    "counters",
    (
        "gpu__time_duration.sum",
        "sm__cycles_active.avg",
        "sm__cycles_elapsed.sum",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_warps",
        "sm__inst_executed.sum",
        "sm__inst_executed_pipe_lsu.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ),
)
"""Counters an arm on this kernel is adjudicated on, in one pass.

``sm__cycles_active`` and the clock are here because NCU's reported duration has
disagreed in sign with the cycle count twice. The local sector pair is here because a
register jump does not announce itself as spill: it reports as LSU instructions, so
the register count and the instruction count are read in the same pass.
"""

MIO_OPCODES: Final = frozenset(("LDS", "LDSM", "STS", "SHFL", "MUFU", "ATOMS"))
"""Opcode classes that queue on the MIO pipe, without the modifier suffix.

``MUFU`` is the non-obvious member and the reason this set is not
:data:`slinoss.perf.ncu.LSU_OPCODES`: a transcendental issues to the MIO queue and
occupies no LSU slot, while a global access occupies the LSU port and no MIO slot.
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
        path: File the block was traced from. A DSL kernel inlines several files
            into one SASS body, and their line numbers collide, so a line number
            without this is not an attribution.
        address: SASS address, the only unambiguous key on a DSL kernel.
        line: Line number of the block the instruction sits under.
        sass: The instruction text as NCU printed it.
        inst_count: Warp-instructions executed here, in one launch. The source page
            emits one section per launch, so this is already per-launch and must not
            be divided by the window's launch count.
        sample_count: PC samples taken here, issuing or not.
        stall_samples: Stall reason to not-issued PC samples.
    """

    kernel: str
    path: str
    address: str
    line: int
    launch: int
    sass: str
    inst_count: int
    sample_count: int
    stall_samples: dict[str, int]

    @property
    def site(self) -> str:
        """File basename and line, the key a source edit is made against."""
        return f"{self.path.rpartition('/')[2]}:{self.line}"

    @property
    def opcode(self) -> str:
        """Opcode class, predicate and modifier suffix removed."""
        matched = _OPCODE.match(self.sass)
        return "" if matched is None else matched.group(1).partition(".")[0]

    @property
    def width(self) -> str:
        """Access width suffix of a memory instruction, or the empty string."""
        matched = _OPCODE.match(self.sass)
        if matched is None:
            return ""
        parts = matched.group(1).split(".")
        for one in parts[1:]:
            if one in ("16", "32", "64", "128", "U8", "S8", "U16", "S16"):
                return one
        return "32" if parts[0] in LSU_OPCODES else ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--groups",
        type=int,
        default=None,
        help="Groups, G. Divides H. Defaults to the shape's own.",
    )
    parser.add_argument("--span", type=int, default=SPLIT)
    parser.add_argument("--warps", type=int, default=WARPS_WIDE)
    parser.add_argument("--resident", type=int, default=None)
    parser.add_argument(
        "--seed-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Supply a final-state cotangent, the variant a step compiles.",
    )
    parser.add_argument("--census", action="store_true", help="Collect the site map.")
    parser.add_argument(
        "--counters", action="store_true", help="Collect COUNTER_TABLE for one side."
    )
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
        help=f"Module holding the baseline ``{ENTRY}``. Empty with --pair means "
        f"--null; empty with --census or --counters means this tree.",
    )
    parser.add_argument(
        "--null",
        action="store_true",
        help="Both arms are this tree's wrapper. The control on the loop.",
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--event-iters", type=int, default=600)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--report", default="/tmp/spb-source")
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Read the report already on disk instead of collecting one.",
    )
    return parser.parse_args(argv)


def resolve_entry(module: str) -> Callable[..., object]:
    """The host entry of one side, from a named module or from this tree.

    Args:
        module: Dotted module path, or the empty string for this tree.

    Returns:
        The ``start_passing_backward`` that module exposes.
    """
    if not module:
        from slinoss.ops.so3ssd.cute.bwd.start_passing import start_passing_backward

        return start_passing_backward
    return getattr(importlib.import_module(module), ENTRY)


def build_runner(
    entry: Callable[..., object],
    shape: OpShape,
    groups: int,
    device: torch.device,
    dtype: torch.dtype,
    span: int,
    warps: int,
    resident: int | None,
    seed_state: bool,
) -> tuple[Callable[[], object], Callable[[], None]]:
    """Allocate one input set and bind one host wrapper over it.

    Two arms of a pair share the input set, so the difference cannot be an address
    or a cache residency.

    ``cquat`` is normalized and ``cscale`` is a sigmoid, so both satisfy I1. Neither
    is read off a forward: this driver measures traffic and instructions, and the
    parity files are where the values have to be the pipeline's.

    Args:
        entry: A ``start_passing_backward``, from this tree or from a baseline.
        shape: The problem size.
        groups: ``G``. Divides ``shape.heads``.
        device: Where to allocate.
        dtype: Activation dtype.
        span: Lane band width.
        warps: Block width in warps.
        resident: Launch bound, or None for the kernel's default.
        seed_state: Whether to supply a final-state cotangent.

    Returns:
        ``(call, run)``. ``call`` returns the named output tuple; ``run`` drops it.

    Raises:
        ValueError: If ``groups`` does not divide ``shape.heads``.
    """
    if shape.heads % groups:
        raise ValueError(f"groups {groups} must divide heads {shape.heads}")
    inputs = make_inputs(shape, device, dtype=dtype, requires_grad=False)
    gen = torch.Generator(device=device).manual_seed(1)

    def randn(*size: int) -> Tensor:
        return torch.randn(*size, dtype=torch.float32, device=device, generator=gen)

    if groups == shape.heads:
        vecc = inputs.C
    else:
        vecc = torch.randn(
            shape.bsz,
            groups,
            shape.seq,
            shape.d_state,
            dtype=dtype,
            device=device,
            generator=gen,
        )

    chunks = -(-shape.seq // shape.chunk)
    cquat = randn(shape.bsz, shape.heads, chunks, 4)
    cquat = cquat / cquat.norm(dim=-1, keepdim=True)
    cscale = randn(shape.bsz, shape.heads, chunks).sigmoid()
    dstate = randn(shape.bsz, shape.heads, shape.rows, shape.d_state)
    seed = dstate if seed_state else None

    def call() -> object:
        return entry(
            inputs.dy,
            inputs.trans,
            vecc,
            cquat,
            cscale,
            shape.chunk,
            seed,
            span=span,
            warps=warps,
            resident=resident,
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
        bit. The line names each field, its worst absolute difference, and the count
        of differing elements.
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
    path = ""
    line = 0
    launch = -1
    for row in csv.reader(io.StringIO(text)):
        if len(row) == 2 and row[0] == _FILE_PATH:
            path, columns, line = row[1], {}, 0
            continue
        if len(row) == 2 and row[0] == _FUNCTION_NAME:
            kernel, columns, line = row[1], {}, 0
            launch += 1
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
        # the text's own punctuation and no column right of it decodes. Only the line
        # number is read off such a row, and it is column zero, ahead of the damage.
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
                path=path,
                address=address,
                line=line,
                launch=max(launch, 0),
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


def print_widths(sites: Sequence[Site], launches: int) -> None:
    """Print the memory-access width histogram, which prices a packing arm.

    Args:
        sites: Every instruction of the window.
        launches: Launches in the window.
    """
    total: dict[tuple[str, str], int] = {}
    for one in sites:
        if one.opcode not in LSU_OPCODES:
            continue
        key = (one.opcode, one.width)
        total[key] = total.get(key, 0) + one.inst_count
    print("widths       lsu instructions per launch, by opcode and access width")
    for (opcode, width), count in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"  {opcode:6s} .{width:4s} {count / launches:12,.0f}")


def print_lines(sites: Sequence[Site], launches: int, top: int) -> None:
    """Print the instruction census keyed by file and line.

    Args:
        sites: Every instruction of the window.
        launches: Launches in the window.
        top: Rows to print.
    """
    total: dict[str, list[int]] = {}
    for one in sites:
        seen = total.setdefault(one.site, [0, 0, 0])
        seen[0] += one.inst_count
        seen[1] += one.sample_count
        if one.opcode in LSU_OPCODES:
            seen[2] += one.inst_count
    inst = sum(v[0] for v in total.values())
    samples = sum(v[1] for v in total.values())
    print(f"sources      {len(total)} distinct, {inst / launches:,.0f} inst/launch")
    ranked = sorted(total.items(), key=lambda kv: -kv[1][0])
    for site, (count, sampled, lsu) in ranked[:top]:
        print(
            f"  {site:34s} {count / launches:12,.0f} inst "
            f"{100.0 * count / inst:6.2f}%  lsu {lsu / launches:11,.0f} "
            f"samp {100.0 * sampled / samples:6.2f}%"
        )


def print_phases(sites: Sequence[Site], launches: int, reasons: Sequence[str]) -> None:
    """Print the instruction census segmented at every ``BAR.SYNC``.

    Line attribution is useless on this kernel: every inlined helper is reported
    under the launch function's own line numbers, and two blocks holding 55% of the
    issue land on a decorator. Address order survives, and ``BAR.SYNC`` cuts it into
    the phases the source is written in, so a phase is the smallest unit an arm can
    be aimed at.

    Args:
        sites: Every instruction of the window.
        launches: Launches in the window.
        reasons: Stall reasons to report per phase.
    """
    ordered = sorted(
        (one for one in sites if one.launch == 0), key=lambda one: int(one.address, 16)
    )
    phases: list[list[Site]] = [[]]
    for one in ordered:
        phases[-1].append(one)
        if one.opcode == "BAR":
            phases.append([])
    inst = sum(one.inst_count for one in ordered)
    samples = sum(one.sample_count for one in ordered)
    print(
        f"phases       {len(phases)} between {sum(1 for o in ordered if o.opcode == 'BAR')}"
        f" BAR.SYNC sites, {inst:,} inst/launch over {len(ordered)} addresses"
    )
    for index, phase in enumerate(phases):
        if not phase:
            continue
        count = sum(one.inst_count for one in phase)
        lsu = sum(one.inst_count for one in phase if one.opcode in LSU_OPCODES)
        sampled = sum(one.sample_count for one in phase)
        mix: dict[str, int] = {}
        for one in phase:
            mix[one.opcode] = mix.get(one.opcode, 0) + one.inst_count
        top = " ".join(
            f"{k}:{v:,}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])[:6]
        )
        held = {
            reason: sum(one.stall_samples.get(reason, 0) for one in phase)
            for reason in reasons
        }
        worst = max(held.items(), key=lambda kv: kv[1])
        print(
            f"  phase {index:2d} {phase[0].address}..{phase[-1].address} "
            f"{count:12,} inst {100.0 * count / inst:6.2f}%  lsu {lsu:10,}  "
            f"samp {100.0 * sampled / samples:6.2f}%  top-stall {worst[0]} "
            f"{100.0 * worst[1] / max(sampled, 1):5.1f}%"
        )
        print(f"             {top}")


def print_sites(sites: Sequence[Site], reason: str, top: int) -> None:
    """Print the site map, ranked by one stall reason's not-issued samples.

    Rows are merged over the window's launches by address, because the source page
    emits one row per launch per address and one launch's share of a stall is not a
    figure worth printing five times.

    Args:
        sites: Every instruction of the window.
        reason: The stall reason to rank by.
        top: Rows to print.
    """
    merged: dict[str, tuple[Site, int, int, int]] = {}
    for one in sites:
        first, held, sampled, inst = merged.get(one.address, (one, 0, 0, 0))
        merged[one.address] = (
            first,
            held + one.stall_samples.get(reason, 0),
            sampled + one.sample_count,
            max(inst, one.inst_count),
        )
    total = sum(row[1] for row in merged.values())
    allsamp = sum(row[2] for row in merged.values())
    if total == 0:
        print(f"sites        {reason} no samples")
        return
    print(f"sites        {reason} {total:,} samples, {allsamp:,} over every reason")
    ranked = sorted(merged.values(), key=lambda row: -row[1])
    for first, held, _, inst in ranked[:top]:
        if held == 0:
            break
        print(
            f"  {first.address} {first.site:30s} "
            f"{held:7,} ({100.0 * held / total:5.2f}% of {reason}, "
            f"{100.0 * held / allsamp:5.2f}% of all) "
            f"inst/launch {inst:9,}  {first.sass[:52]}"
        )


def target_argv(args: argparse.Namespace) -> list[str]:
    """The command NCU attaches to.

    Args:
        args: The parsed command line.

    Returns:
        The argv, this file in ``--window`` mode at the same geometry and on the
        same side. ``--baseline`` is forwarded, so a counter pass can profile the
        baseline module rather than only this tree.
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
        "--span",
        str(args.span),
        "--warps",
        str(args.warps),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]
    if args.groups is not None:
        out += ["--groups", str(args.groups)]
    if args.resident is not None:
        out += ["--resident", str(args.resident)]
    if args.baseline:
        out += ["--baseline", args.baseline]
    out += ["--seed-state" if args.seed_state else "--no-seed-state"]
    return out


def _run(command: Sequence[str]) -> str:
    """Run one NCU invocation and return its stdout.

    :func:`slinoss.perf.ncu.run_source` would do both invocations, and its parse
    raises on this kernel's page for the reason
    :func:`scripts.perf.ab_chunk_vector_bwd._run` records: a line-aggregate row whose
    source text holds a quote arrives one cell short of its header.

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


def header(args: argparse.Namespace, device: torch.device, shape: OpShape) -> None:
    """Print the stamp every mode carries.

    Args:
        args: The parsed command line.
        device: The device.
        shape: The problem size.
    """
    print(f"device       {device} ord {device_ordinal(device)}")
    print(f"shape        {shape.describe()}")
    print(
        f"config       span {args.span} warps {args.warps} resident {args.resident} "
        f"seed_state {args.seed_state} groups {args.groups}"
    )
    print(f"side         {args.baseline or 'this tree'}")
    print(f"contention   {contention(device_ordinal(device))}")


def run_census(args: argparse.Namespace, device: torch.device, shape: OpShape) -> int:
    """Collect the source pass and print the census.

    Args:
        args: The parsed command line.
        device: The device, for the header.
        shape: The problem size.

    Returns:
        Process exit status.
    """
    reasons = (
        "barrier",
        "mio_throttle",
        "short_scoreboard",
        "long_scoreboard",
        "lg_throttle",
        "wait",
        "no_instruction",
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
    # The page emits one section per launch, so the instruction counts sum over the
    # window and the divisor is the section count, not the requested iteration count.
    launches = 1 + max(one.launch for one in sites)
    header(args, device, shape)
    print(f"report       {written}")
    print(f"kernels      {sorted({one.kernel for one in sites})}")
    print(f"sections     {launches} launch(es), {args.iters} requested")
    print(f"files        {sorted({one.path.rpartition('/')[2] for one in sites})}")
    print_opcodes(sites, launches)
    print_widths(sites, launches)
    print_lines(sites, launches, args.top)
    print_phases(sites, launches, reasons)
    for one in reasons:
        print_sites(sites, one, args.top)
    return 0


def run_counters(args: argparse.Namespace, device: torch.device, shape: OpShape) -> int:
    """Collect :data:`COUNTER_TABLE` for one side and print it.

    Args:
        args: The parsed command line.
        device: The device, for the header.
        shape: The problem size.

    Returns:
        Process exit status.

    Raises:
        ValueError: If the pass returned no value for a requested metric, which
            means the name is wrong for this device.
    """
    one = run_ncu(
        COUNTER_TABLE,
        target_argv(args),
        ncu=args.ncu,
        extra=("--kernel-name", f"regex:{KERNEL}"),
    )
    if one.missing_metrics:
        raise ValueError(
            f"ncu returned no value for {list(one.missing_metrics)}; the metric "
            f"names are wrong for this device"
        )
    header(args, device, shape)
    print(f"launches     {len(one.invocations)}")
    for call in one.invocations:
        print(f"launch       {call.launch_id}")
        for metric in COUNTER_TABLE.metrics:
            value = call.values.get(metric)
            if value is None:
                continue
            print(f"  {metric:66s} {value:20,.3f}")
    return 0


def run_pair(args: argparse.Namespace, device: torch.device, shape: OpShape) -> int:
    """Measure two arms in one loop and compare their outputs bitwise.

    Side A is the baseline and side B is this tree, so a harness that resolved the
    same entry twice reads as the null it would then be.

    Args:
        args: The parsed command line.
        device: Where to allocate and time.
        shape: The problem size.

    Returns:
        Process exit status. Nonzero when the arms disagree bitwise.
    """
    dtype = DTYPES[args.dtype]
    groups = shape.groups if args.groups is None else args.groups
    arm_entry = resolve_entry("")
    if args.null or not args.baseline:
        base_entry = arm_entry
        base_label = "null"
    else:
        base_entry = resolve_entry(args.baseline)
        base_label = "base"
    bound = (shape, groups, device, dtype, args.span, args.warps, args.resident)
    a_call, a_run = build_runner(base_entry, *bound, args.seed_state)
    b_call, b_run = build_runner(arm_entry, *bound, args.seed_state)
    clean, detail = parity(a_call(), b_call())
    torch.cuda.synchronize(device)
    out = measure_paired(
        base_label,
        a_run,
        "arm",
        b_run,
        label=f"spb.{shape.name}",
        iters=args.event_iters,
        warmup=args.warmup,
        device=device,
        clocks=clock_policy(device_ordinal(device)),
    )
    header(args, device, shape)
    print(f"baseline     {args.baseline or 'this tree (null control)'}")
    print(f"clocks       {out.timed.clocks}")
    print(f"parity       {'bitwise' if clean else 'DIFFERS'}  {detail}")
    print(f"verdict      {out.comparison.verdict()}")
    print(
        f"medians      {base_label} {out.comparison.a_median_duration_us:,.3f} us  "
        f"arm {out.comparison.b_median_duration_us:,.3f} us"
    )
    print(f"pairs        {args.event_iters}")
    return 0 if clean else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Census, counters, pair, or run as the profiler's target.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If no mode was asked for.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    shape = shape_by_name(args.shape)
    groups = shape.groups if args.groups is None else args.groups
    if args.window:
        _, run = build_runner(
            resolve_entry(args.baseline),
            shape,
            groups,
            device,
            DTYPES[args.dtype],
            args.span,
            args.warps,
            args.resident,
            args.seed_state,
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
        return run_census(args, device, shape)
    if args.counters:
        return run_counters(args, device, shape)
    if args.pair:
        return run_pair(args, device, shape)
    raise ValueError("nothing to do: pass --census, --counters, --pair, or --window")


if __name__ == "__main__":
    raise SystemExit(main())
