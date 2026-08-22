"""Shared-memory bank-conflict census: per kernel over the tree, per SASS site.

``docs/kernels.md`` holds a conflict to be a bug. Nothing measured which kernel
carries the conflicts, which access makes them, or what a conflict costs. This
driver answers the first two and bounds the third.

``--kernels`` runs the ``shared`` table of :data:`slinoss.perf.ncu.NCU_TABLES` over
the whole step and ranks every kernel by ABSOLUTE conflict wavefronts. Rate is
reported beside the count and is not the ranking key: a high rate on a small launch
buys nothing, and a fix that cuts wavefronts while raising the rate is still a win.

``--census`` collects :data:`slinoss.perf.ncu.SOURCE_TABLE` and reads its source page
at instruction granularity, keyed by SASS address. Address is the only unambiguous
key on a CuTe DSL kernel: the DSL emits one ``.file`` per module, so a line number
names a traced block and not a file.
:func:`slinoss.perf.ncu.parse_source_csv` merges instructions under their line and
this does not, which is why the parse is here and not shared.

Two columns decide everything and no other driver accumulates them:
``memory_l1_wavefronts_shared`` and ``memory_l1_wavefronts_shared_ideal``. Their
difference is the conflict replay of one instruction, and the PC-sample column
beside it is the only pricing input. A site with excess wavefronts and no sample
share is an unpriced arm, not a small one, and the census says so per site.

``--census`` needs ``CUTE_DSL_LINEINFO=1`` in this environment, which the profiled
target inherits. Without it the source page holds no instruction row and the parse
raises rather than reporting an empty census.

    CUTE_DSL_LINEINFO=1 CUDA_VISIBLE_DEVICES=0 \\
        python3 scripts/perf/profile_bank_conflicts.py --kernels --shape acceptance
    CUTE_DSL_LINEINFO=1 CUDA_VISIBLE_DEVICES=0 \\
        python3 scripts/perf/profile_bank_conflicts.py --census --shape acceptance \\
        --kernel chunk_vector_bwd

The census this driver was written for, at the acceptance shape on sm_86. Five
kernels carry the class and the other seven are at zero:

    kernel                  excess/launch   of tree   of own wavefronts
    chunk_vector_bwd            7,612,416    44.05%              11.67%
    chunk_input_bwd             5,013,504    29.01%              21.59%
    increment_passing_fwd       1,658,880     9.60%              16.94%
    start_passing_bwd           1,658,880     9.60%              18.70%
    chunk_scan_fwd              1,336,320     7.73%              10.46%

REFUSED, and the arithmetic that refused it. 8,146,944 of the 17,280,000 is
``STS.64`` at two-way, forced by the odd 16-byte segment count ``smem_pitch``
returns: an 8-byte phase is four rows of four pairs, its pair-bank index is
``2 * Ps * r + k mod 16``, and that is a bijection only at ``Ps = 2 mod 4``. The
same odd ``Ps`` is what makes ``LDSM.16.M88.4``'s eight-consecutive-row phase a
bijection, measured at degree exactly 1.0000 over 329 sites and 43,831,296
wavefronts. Flipping the pitch buys 8.1 M and pays 43.8 M. The remaining 9.1 M is
the thread map, not the pitch: the same opcode at the same width measures 1.0000 in
one kernel and 1.32 to 1.40 in another. Neither half is priced. Conflicting sites
carry 2.4% to 4.2% of their kernel's samples, 0.00% to 0.03% of
``short_scoreboard``, and one site in the tree clears a 0.5% sample floor.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

import torch

from slinoss.perf.coverage import MODES
from slinoss.perf.device import contention, device_ordinal
from slinoss.perf.ncu import (
    NCU_TABLES,
    SOURCE_TABLE,
    SOURCE_VIEW,
    STALL_REASONS,
    NcuPass,
    NcuTable,
    export_flags,
    import_command,
    ncu_command,
    pcsamp_metric,
    report_file,
    run_ncu,
)
from slinoss.perf.tools import resolve_tool
from slinoss.perf.workload import OPS, SHAPE_NAMES, shape_by_name

TARGET: Final = Path(__file__).with_name("profile_target.py")
"""The process NCU attaches to. One warmup policy and one capture window."""

SHARED_TABLE: Final[NcuTable] = next(
    table for table in NCU_TABLES if table.name == "shared"
)
"""The per-launch conflict table, selected rather than restated.

Restating its metrics here would let the per-kernel census and
:class:`slinoss.perf.ncu.KernelCounters` disagree about what a conflict is."""

REASONS: Final = (
    "barrier",
    "mio_throttle",
    "short_scoreboard",
    "long_scoreboard",
    "wait",
)
"""Stall reasons carried per site.

``short_scoreboard`` and ``mio_throttle`` are the two a conflict can move: a replayed
wavefront holds the MIO queue and lengthens the shared dependency. A conflict arm
that moves neither has not been priced, whatever the count did."""

MIO_OPCODES: Final = frozenset(("LDS", "LDSM", "STS", "SHFL", "MUFU", "ATOMS"))
"""Opcodes that queue on MIO. Only ``LDS``, ``LDSM`` and ``STS`` can conflict."""

SHARED_OPCODES: Final = frozenset(("LDS", "LDSM", "STS"))
"""Opcodes addressing shared memory, and so the only carriers of a conflict."""

_ADDRESS: Final = "Address"
_LINE_NO: Final = "Line No"
_SOURCE: Final = "Source"
_FILE_PATH: Final = "File Path"
_FUNCTION_NAME: Final = "Function Name"
_INST: Final = "inst_executed"
_SAMPLES: Final = "smsp__pcsamp_sample_count"
_WIDTH: Final = "memory_access_size_type"
_WAVEFRONTS: Final = "memory_l1_wavefronts_shared"
_IDEAL: Final = "memory_l1_wavefronts_shared_ideal"

_OPCODE: Final = re.compile(r"^\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)")

_LD_WAVEFRONTS: Final = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
_ST_WAVEFRONTS: Final = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
_LD_CONFLICTS: Final = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
_ST_CONFLICTS: Final = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"


class KernelRow(NamedTuple):
    """One kernel's conflict account over the profiled window.

    Attributes:
        kernel: Demangled kernel name.
        launches: Launches profiled.
        duration_us: Summed duration over those launches.
        load_wavefronts: Shared load wavefronts.
        store_wavefronts: Shared store wavefronts.
        load_conflicts: Bank conflicts on loads.
        store_conflicts: Bank conflicts on stores.
    """

    kernel: str
    launches: int
    duration_us: float
    load_wavefronts: int
    store_wavefronts: int
    load_conflicts: int
    store_conflicts: int

    @property
    def wavefronts(self) -> int:
        """Shared wavefronts, load plus store."""
        return self.load_wavefronts + self.store_wavefronts

    @property
    def conflicts(self) -> int:
        """Bank conflicts, load plus store. The ranking key."""
        return self.load_conflicts + self.store_conflicts

    @property
    def per_wavefront(self) -> float:
        """Conflicts over wavefronts. Zero for a kernel with no shared access."""
        return self.conflicts / self.wavefronts if self.wavefronts else 0.0


class Site(NamedTuple):
    """One SASS instruction, with the conflict and sample counters correlated to it.

    Attributes:
        kernel: Kernel name, from the source page's ``Function Name``.
        address: SASS address. The only unambiguous key on a DSL kernel.
        line: Line of the traced block, not of a file this record names.
        sass: The instruction text as NCU printed it.
        width: Access width in bits, 0 for an instruction touching no memory.
        inst_count: Warp-instructions executed here.
        wavefronts: Shared wavefronts L1 served for them.
        ideal: Wavefronts a conflict-free layout would have needed.
        sample_count: PC samples taken here, issuing or not.
        stall_samples: Stall reason to not-issued PC samples.
    """

    kernel: str
    address: str
    line: int
    sass: str
    width: int
    inst_count: int
    wavefronts: int
    ideal: int
    sample_count: int
    stall_samples: dict[str, int]

    @property
    def opcode(self) -> str:
        """Opcode class, predicate and modifier suffix removed."""
        matched = _OPCODE.match(self.sass)
        return "" if matched is None else matched.group(1).partition(".")[0]

    @property
    def excess(self) -> int:
        """Wavefronts a conflict-free layout would not have needed."""
        return self.wavefronts - self.ideal

    @property
    def degree(self) -> float:
        """Wavefronts per ideal wavefront. 1.0 is conflict-free, 2.0 is two-way."""
        return self.wavefronts / self.ideal if self.ideal else 0.0

    @property
    def per_inst(self) -> float:
        """Wavefronts per warp-instruction. 32 lanes of 4 bytes is 1 when clean."""
        return self.wavefronts / self.inst_count if self.inst_count else 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", action="store_true", help="Per-kernel ranking.")
    parser.add_argument("--census", action="store_true", help="Per-site census.")
    parser.add_argument("--op", choices=OPS, default=OPS[0])
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--mode", choices=MODES, default="step")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--kernel",
        default="",
        help="Regex NCU narrows to. Empty profiles every launch of the step, "
        "which is what the per-kernel ranking needs.",
    )
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report", default="/tmp/banks-source")
    parser.add_argument("--top", type=int, default=28)
    parser.add_argument(
        "--floor",
        type=float,
        default=0.5,
        help="Sample share in percent below which a site's excess is unpriced.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Read the report already on disk instead of collecting one.",
    )
    return parser.parse_args(argv)


def target_argv(args: argparse.Namespace) -> list[str]:
    """The command NCU attaches to.

    Args:
        args: The parsed command line.

    Returns:
        The argv, :data:`TARGET` at the same geometry.
    """
    argv = [
        args.python,
        str(TARGET),
        "--shape",
        args.shape,
        "--mode",
        args.mode,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    if args.op != OPS[0]:
        argv += ["--op", args.op]
    return argv


def kernel_rows(one: NcuPass) -> tuple[KernelRow, ...]:
    """Fold one ``shared`` pass onto one row per kernel.

    :func:`slinoss.perf.ncu.kernel_counters` needs every table in
    :data:`slinoss.perf.ncu.NCU_TABLES` and refuses a single pass, so the fold is
    here. Counts sum over launches; the ranking is a total and not a rate.

    Args:
        one: The parsed pass.

    Returns:
        One row per kernel, ranked by absolute conflict wavefronts.
    """
    held: dict[str, list[float]] = {}
    duration = SHARED_TABLE.metrics[0]
    for launch in one.invocations:
        row = held.setdefault(launch.kernel, [0.0] * 6)
        row[0] += 1
        row[1] += launch.values.get(duration, 0.0) / 1000.0
        row[2] += launch.values.get(_LD_WAVEFRONTS, 0.0)
        row[3] += launch.values.get(_ST_WAVEFRONTS, 0.0)
        row[4] += launch.values.get(_LD_CONFLICTS, 0.0)
        row[5] += launch.values.get(_ST_CONFLICTS, 0.0)
    out = [
        KernelRow(
            kernel=name,
            launches=int(row[0]),
            duration_us=row[1],
            load_wavefronts=round(row[2]),
            store_wavefronts=round(row[3]),
            load_conflicts=round(row[4]),
            store_conflicts=round(row[5]),
        )
        for name, row in held.items()
    ]
    return tuple(sorted(out, key=lambda r: -r.conflicts))


def print_kernels(rows: Sequence[KernelRow]) -> None:
    """Print the per-kernel ranking, ranked by absolute conflict wavefronts.

    Args:
        rows: The rows, already ranked.
    """
    conflicts = sum(row.conflicts for row in rows) or 1
    wavefronts = sum(row.wavefronts for row in rows) or 1
    duration = sum(row.duration_us for row in rows) or 1.0
    print(
        f"tree         {len(rows)} kernels  {conflicts:,} conflicts  "
        f"{wavefronts:,} wavefronts  {duration:,.1f} us"
    )
    print(
        f"  {'kernel':34s} {'launch':>6s} {'us':>9s} {'conflicts':>13s} "
        f"{'share':>7s} {'wavefronts':>13s} {'per wf':>7s} {'ld/st':>15s}"
    )
    for row in rows:
        print(
            f"  {row.kernel[:34]:34s} {row.launches:6d} {row.duration_us:9.1f} "
            f"{row.conflicts:13,} {100.0 * row.conflicts / conflicts:6.2f}% "
            f"{row.wavefronts:13,} {row.per_wavefront:7.4f} "
            f"{row.load_conflicts:7,}/{row.store_conflicts:,}"
        )


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


def parse_sites(text: str, *, reasons: Sequence[str] = REASONS) -> tuple[Site, ...]:
    """Read a source page at instruction granularity, carrying the wavefronts.

    Args:
        text: Stdout of :func:`slinoss.perf.ncu.import_command` with ``page="source"``
            and ``print_source=SOURCE_VIEW``.
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
                address=address,
                line=line,
                sass=row[sass_column].strip() if sass_column >= 0 else "",
                width=_cell(row, columns, _WIDTH),
                inst_count=_cell(row, columns, _INST),
                wavefronts=_cell(row, columns, _WAVEFRONTS),
                ideal=_cell(row, columns, _IDEAL),
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


def print_totals(sites: Sequence[Site], launches: int) -> None:
    """Print the window's shared totals and the conflict share of the instruction mix.

    Args:
        sites: Every instruction of the window.
        launches: Launches in the window, the divisor onto a per-launch footing.
    """
    shared = [one for one in sites if one.wavefronts]
    inst = sum(one.inst_count for one in sites)
    samples = sum(one.sample_count for one in sites) or 1
    wavefronts = sum(one.wavefronts for one in shared)
    ideal = sum(one.ideal for one in shared)
    conflicted = [one for one in shared if one.excess > 0]
    print(
        f"shared       {len(shared):,} accessing sites, {len(conflicted):,} "
        f"conflicting  {wavefronts / launches:,.0f} wavefronts/launch  ideal "
        f"{ideal / launches:,.0f}  excess {(wavefronts - ideal) / launches:,.0f} "
        f"({100.0 * (wavefronts - ideal) / (ideal or 1):.2f}%)"
    )
    print(
        f"instructions {inst / launches:,.0f}/launch  shared "
        f"{sum(one.inst_count for one in shared) / launches:,.0f}  "
        f"samples on conflicting sites "
        f"{100.0 * sum(one.sample_count for one in conflicted) / samples:.2f}% of "
        f"{samples:,}"
    )


def print_reasons(sites: Sequence[Site]) -> None:
    """Print each stall reason's total and the conflicting sites' share of it.

    This is the pricing instrument, and the only one. A replayed wavefront holds the
    MIO queue and lengthens the shared dependency, so a conflict that costs time
    shows up as ``mio_throttle`` where it issues or ``short_scoreboard`` where the
    value is consumed. A conflict class holding neither is a count without a cost,
    whatever a per-wavefront rate would say about it.

    Args:
        sites: Every instruction of the window.
    """
    samples = sum(one.sample_count for one in sites) or 1
    conflicted = [one for one in sites if one.excess > 0]
    shared = [one for one in sites if one.wavefronts]
    print(f"stalls       {samples:,} samples, share of each reason by site class")
    print(
        f"  {'reason':18s} {'samples':>10s} {'of all':>8s} {'conflict':>9s} {'shared':>8s}"
    )
    for reason in REASONS:
        total = sum(one.stall_samples.get(reason, 0) for one in sites)
        bad = sum(one.stall_samples.get(reason, 0) for one in conflicted)
        held = sum(one.stall_samples.get(reason, 0) for one in shared)
        print(
            f"  {reason:18s} {total:10,} {100.0 * total / samples:7.2f}% "
            f"{100.0 * bad / (total or 1):8.2f}% {100.0 * held / (total or 1):7.2f}%"
        )


def print_opcode_widths(sites: Sequence[Site], launches: int) -> None:
    """Print the conflict account by opcode and access width.

    The width decides the modulus: a 32-bit access is one phase of 32 lanes over 32
    banks, a 64-bit access two phases of 16, a 128-bit access four phases of 8 over
    16-byte segments. A pitch is conflict-free against one modulus at a time, so a
    diagnosis that does not name the width has not named a stride.

    Args:
        sites: Every instruction of the window.
        launches: Launches in the window.
    """
    held: dict[tuple[str, int], list[int]] = {}
    for one in sites:
        if not one.wavefronts:
            continue
        row = held.setdefault((one.opcode, one.width), [0, 0, 0, 0, 0])
        row[0] += one.inst_count
        row[1] += one.wavefronts
        row[2] += one.ideal
        row[3] += one.sample_count
        row[4] += 1
    if not held:
        return
    excess = sum(row[1] - row[2] for row in held.values()) or 1
    samples = sum(one.sample_count for one in sites) or 1
    print("class        opcode x width, ranked by absolute excess wavefronts")
    print(
        f"  {'opcode':7s} {'bits':>5s} {'sites':>6s} {'inst/launch':>12s} "
        f"{'wf/launch':>12s} {'excess':>12s} {'share':>7s} {'degree':>7s} "
        f"{'wf/inst':>8s} {'samp':>7s}"
    )
    ranked = sorted(held.items(), key=lambda kv: -(kv[1][1] - kv[1][2]))
    for (opcode, width), row in ranked:
        gap = row[1] - row[2]
        print(
            f"  {opcode:7s} {width:5d} {row[4]:6,} {row[0] / launches:12,.0f} "
            f"{row[1] / launches:12,.0f} {gap / launches:12,.0f} "
            f"{100.0 * gap / excess:6.2f}% {row[1] / (row[2] or 1):7.4f} "
            f"{row[1] / (row[0] or 1):8.4f} {100.0 * row[3] / samples:6.2f}%"
        )


def print_pricing(sites: Sequence[Site], floor: float) -> None:
    """Split the excess into the part that carries samples and the part that does not.

    A conflict counter gives count. Only a sample share gives time. The split is the
    refusal test: excess concentrated on sites below the floor is unpriced, and
    multiplying it by any per-wavefront rate would be the flat-rate arithmetic this
    program keeps falsifying.

    Args:
        sites: Every instruction of the window.
        floor: Sample share in percent at or above which a site counts as priced.
    """
    samples = sum(one.sample_count for one in sites) or 1
    conflicted = [one for one in sites if one.excess > 0]
    excess = sum(one.excess for one in conflicted) or 1
    priced = [one for one in conflicted if 100.0 * one.sample_count / samples >= floor]
    addresses = {one.address for one in priced}
    rest = [one for one in conflicted if one.address not in addresses]
    held = sum(one.excess for one in priced)
    print(
        f"pricing      floor {floor:.2f}% of samples  {len(priced):,} of "
        f"{len(conflicted):,} conflicting sites priced, carrying "
        f"{100.0 * held / excess:.2f}% of the excess"
    )
    print(
        f"  priced     excess {held:,}  samples "
        f"{100.0 * sum(one.sample_count for one in priced) / samples:.2f}%"
    )
    print(
        f"  unpriced   excess {excess - held:,}  samples "
        f"{100.0 * sum(one.sample_count for one in rest) / samples:.2f}%"
    )


def print_sites(sites: Sequence[Site], top: int, launches: int) -> None:
    """Print the conflicting sites, ranked by absolute excess wavefronts.

    Args:
        sites: Every instruction of the window.
        top: Rows to print.
        launches: Launches in the window.
    """
    samples = sum(one.sample_count for one in sites) or 1
    conflicted = [one for one in sites if one.excess > 0]
    excess = sum(one.excess for one in conflicted) or 1
    print(f"sites        {len(conflicted):,} conflicting, ranked by excess wavefronts")
    head = "  ".join(f"{reason[:6]:>6s}" for reason in REASONS)
    print(
        f"  {'address':12s} {'line':>5s} {'excess/l':>10s} {'share':>7s} "
        f"{'degree':>7s} {'inst/l':>10s} {'wf/inst':>8s} {'samp':>6s}  {head}  sass"
    )
    for one in sorted(conflicted, key=lambda s: -s.excess)[:top]:
        cells = "  ".join(f"{one.stall_samples.get(r, 0):6,}" for r in REASONS)
        print(
            f"  {one.address:12s} {one.line:5d} {one.excess / launches:10,.0f} "
            f"{100.0 * one.excess / excess:6.2f}% {one.degree:7.4f} "
            f"{one.inst_count / launches:10,.0f} {one.per_inst:8.4f} "
            f"{100.0 * one.sample_count / samples:5.2f}%  {cells}  {one.sass[:44]}"
        )


def print_clean(sites: Sequence[Site], top: int, launches: int) -> None:
    """Print the largest conflict-free shared sites, for contrast.

    A clean site with more wavefronts than the worst conflicting one bounds what the
    conflict class can be worth: the port carries both.

    Args:
        sites: Every instruction of the window.
        top: Rows to print.
        launches: Launches in the window.
    """
    clean = [one for one in sites if one.wavefronts and one.excess <= 0]
    if not clean:
        return
    samples = sum(one.sample_count for one in sites) or 1
    print(f"clean        {len(clean):,} conflict-free shared sites, largest first")
    for one in sorted(clean, key=lambda s: -s.wavefronts)[:top]:
        print(
            f"  {one.address:12s} {one.line:5d} wf {one.wavefronts / launches:10,.0f} "
            f"inst {one.inst_count / launches:10,.0f} {one.per_inst:7.4f}/inst "
            f"w{one.width:4d} samp {100.0 * one.sample_count / samples:5.2f}%  "
            f"{one.sass[:44]}"
        )


def _run(command: Sequence[str]) -> str:
    """Run one NCU invocation and return its stdout.

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


def header(args: argparse.Namespace) -> None:
    """Print what the numbers came from, contention included.

    Args:
        args: The parsed command line.
    """
    ordinal = device_ordinal(torch.device(args.device))
    print(f"device       {args.device} ord {ordinal}")
    print(f"shape        {shape_by_name(args.shape).describe()}")
    print(f"mode         {args.mode}  iters {args.iters}  op {args.op}")
    print(f"contention   {contention(ordinal)}")


def run_kernels(args: argparse.Namespace) -> int:
    """Rank every kernel of the step by absolute conflict wavefronts.

    Args:
        args: The parsed command line.

    Returns:
        Process exit status.
    """
    narrow = () if not args.kernel else ("--kernel-name", f"regex:{args.kernel}")
    one = run_ncu(
        SHARED_TABLE, target_argv(args), ncu=resolve_tool(args.ncu), extra=narrow
    )
    if one.missing_metrics:
        raise ValueError(f"ncu returned no value for {list(one.missing_metrics)}")
    header(args)
    print(f"launches     {len(one.invocations)}")
    print_kernels(kernel_rows(one))
    return 0


def run_census(args: argparse.Namespace) -> int:
    """Collect the source pass and print the per-site census.

    Args:
        args: The parsed command line.

    Returns:
        Process exit status.
    """
    binary = resolve_tool(args.ncu)
    narrow = () if not args.kernel else ("--kernel-name", f"regex:{args.kernel}")
    if not args.reuse:
        _run(
            ncu_command(
                SOURCE_TABLE,
                target_argv(args),
                ncu=binary,
                extra=(*export_flags(args.report), *narrow),
            )
        )
    written = report_file(args.report)
    text = _run(
        import_command(written, ncu=binary, page="source", print_source=SOURCE_VIEW)
    )
    sites = parse_sites(text)
    header(args)
    print(f"report       {written}")
    print(f"kernels      {sorted({one.kernel for one in sites})}")
    print(f"reasons      {[r for r in REASONS if r in STALL_REASONS]}")
    print_totals(sites, args.iters)
    print_reasons(sites)
    print_opcode_widths(sites, args.iters)
    print_pricing(sites, args.floor)
    print_sites(sites, args.top, args.iters)
    print_clean(sites, args.top, args.iters)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Rank the kernels or census the sites.

    Returns:
        Process exit status.

    Raises:
        ValueError: If no mode was asked for.
    """
    args = parse_args(argv)
    if args.kernels:
        return run_kernels(args)
    if args.census:
        return run_census(args)
    raise ValueError("nothing to do: pass --kernels or --census")


if __name__ == "__main__":
    raise SystemExit(main())
