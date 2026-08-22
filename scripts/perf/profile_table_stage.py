"""Address-keyed source census for the ``table.py`` staging passes.

``slinoss.perf.ncu.parse_source_csv`` aggregates a source page by line, and a line
number is ambiguous on a CuTe DSL kernel: NVVM emits one ``.file`` per module while
keeping every traced file's line numbers, so two traced files that share a line
number merge into one record. This driver keys the same page by instruction address,
which is unique, and reports the opcode and access width beside the PC-sample share
so a site can be priced rather than guessed at.

Needs ``CUTE_DSL_LINEINFO=1`` in this environment; the profiled target inherits it.

    CUTE_DSL_LINEINFO=1 CUDA_VISIBLE_DEVICES=0 python3 \
        scripts/perf/profile_table_stage.py --kernel chunk_vector_bwd \
        --shape acceptance --ncu /usr/local/cuda/bin/ncu

``--opcode`` narrows the per-address listing to one opcode class, which is how the
widen sites are found. ``--json`` writes the whole address table so two collections
can be differenced; a differential is the only unambiguous attribution of an
instruction to a source site on a DSL kernel.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from slinoss.perf.ncu import (
    LSU_OPCODES,
    SOURCE_TABLE,
    SOURCE_VIEW,
    STALL_REASONS,
    export_flags,
    import_command,
    ncu_command,
    pcsamp_metric,
    report_file,
    resolve_tool,
)
from slinoss.perf.workload import SHAPE_NAMES

KERNELS: Mapping[str, str] = {
    "chunk_scan_fwd": "chunk_scan_fwd_kernel",
    "increment_passing_fwd": "increment_passing_fwd_kernel",
    "chunk_vector_bwd": "chunk_vector_bwd_kernel",
    "chunk_input_bwd": "chunk_input_bwd_kernel",
    "chunk_start_bwd": "chunk_start_bwd_kernel",
}

MODE: Mapping[str, str] = {
    "chunk_scan_fwd": "step",
    "increment_passing_fwd": "step",
    "chunk_vector_bwd": "step",
    "chunk_input_bwd": "step",
    "chunk_start_bwd": "step",
}

TABLE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "slinoss"
    / "ops"
    / "so3ssd"
    / "cute"
    / "table.py"
)
"""The file the spans below are read from. Parsed, not imported: the parent process
only reads a CSV, and importing the module would pull the DSL into it."""


def _regions() -> tuple[tuple[str, int, int], ...]:
    """Line span of every top-level function of ``table.py``, read from the file.

    A span is a claim about ``table.py`` only. Every other module the kernel traces
    overlaps these numbers, so a row inside a span is a candidate site and not a
    finding; confirm it with ``--json`` and a differential.

    The spans are derived rather than written down because a hardcoded table is a
    claim about one file version that goes stale silently. It also cannot survive a
    differential across two versions: the renumbering makes one span name two
    functions. Comparing one file at two settings, which is what ``--rot-run`` does,
    is the form that keeps the spans meaningful.

    Returns:
        ``(qualified name, first line, last line)`` in file order. The first line is
        the decorator where there is one, since a decorated definition attributes to
        it.
    """
    text = TABLE_SOURCE.read_text(encoding="utf-8")
    heads = [
        (min([node.lineno, *(d.lineno for d in node.decorator_list)]), node.name)
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef)
    ]
    ends = [line - 1 for line, _ in heads[1:]] + [len(text.splitlines())]
    return tuple(
        (f"table.{name}", line, end)
        for (line, name), end in zip(heads, ends, strict=True)
    )


REGIONS: tuple[tuple[str, int, int], ...] = _regions()
"""Line spans of the ``table.py`` this driver imports. See :func:`_regions`."""

_LINE_NO = "Line No"
_ADDRESS = "Address"
_SOURCE = "Source"
_FILE_PATH = "File Path"
_FUNCTION_NAME = "Function Name"
_INST = "inst_executed"
_WIDTH = "memory_access_size_type"
_WAVEFRONTS = "memory_l1_wavefronts_shared"
_IDEAL = "memory_l1_wavefronts_shared_ideal"
_SAMPLES = "smsp__pcsamp_sample_count"

_OPCODE = re.compile(r"^\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)")


@dataclass
class Site:
    """One SASS instruction, keyed by its address.

    Attributes:
        address: Instruction address, the only unambiguous key on the page.
        kernel: Demangled kernel name.
        line: Line number NCU correlated, of an unknown module.
        opcode: Opcode with its modifier suffix dropped.
        sass: Raw SASS text.
        inst: Warp-instructions executed, summed over profiled launches.
        bits: Access width in bits, or 0 for a non-access.
        samples: PC samples at this address.
        stalls: Stall samples by reason.
    """

    address: str
    kernel: str
    line: int
    opcode: str
    sass: str
    inst: int
    bits: int
    samples: int
    stalls: dict[str, int] = field(default_factory=dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="chunk_vector_bwd")
    parser.add_argument("--shape", choices=SHAPE_NAMES, default="acceptance")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--report", default="")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--rot-run",
        type=int,
        default=0,
        help="Collect through profile_rot_run.py at this stage_rotated run width.",
    )
    parser.add_argument(
        "--opcode",
        default="",
        help="Narrow the per-address listing to this opcode class.",
    )
    parser.add_argument(
        "--json", default="", help="Write the whole address table here."
    )
    parser.add_argument(
        "--import-only",
        default="",
        help="Skip collection and re-read this report instead.",
    )
    return parser.parse_args(argv)


def target_argv(args: argparse.Namespace) -> list[str]:
    """The argv NCU attaches to.

    ``--rot-run`` swaps in ``profile_rot_run.py``, which runs the same window with
    ``table.ROT_RUN`` set. Two collections from one file version keep every line
    number meaning the same site, which differencing two file versions does not.
    """
    here = os.path.dirname(__file__)
    if args.rot_run:
        return [
            sys.executable,
            os.path.join(here, "profile_rot_run.py"),
            "--mode",
            "target",
            "--shape",
            args.shape,
            "--dtype",
            args.dtype,
            "--part",
            MODE[args.kernel],
            "--rot-run",
            str(args.rot_run),
            "--warmup",
            "3",
        ]
    return [
        sys.executable,
        os.path.join(here, "profile_target.py"),
        "--op",
        "so3ssd",
        "--shape",
        args.shape,
        "--dtype",
        args.dtype,
        "--mode",
        MODE[args.kernel],
        "--iters",
        "1",
    ]


def _aligned(row: Sequence[str], fields: int) -> list[str]:
    """Restore a row NCU split on an unescaped quote in its source text."""
    extra = len(row) - fields
    if extra <= 0:
        return list(row)
    return [row[0], ",".join(row[1 : 2 + extra]), *row[2 + extra :]]


def _int(cell: str) -> int:
    text = cell.strip().replace(",", "")
    if not text or text in {"-", "n/a", "N/A"}:
        return 0
    try:
        return round(float(text))
    except ValueError:
        return 0


def _columns(header: Sequence[str]) -> tuple[dict[str, int], int]:
    columns: dict[str, int] = {}
    for index, name in enumerate(header):
        columns.setdefault(name, index)
    sass = [i for i, name in enumerate(header) if name == _SOURCE]
    return columns, sass[1] if len(sass) > 1 else -1


def parse_addresses(text: str) -> tuple[list[Site], dict[str, int]]:
    """Key a source page by instruction address.

    Args:
        text: Stdout of an ``--page source`` import with SASS printed.

    Returns:
        The sites, and per-kernel launch counts (blocks seen).

    Raises:
        ValueError: If the page holds no block.
    """
    reader = csv.reader(io.StringIO(text))
    columns: dict[str, int] = {}
    sass_column = -1
    fields = 0
    kernel = ""
    line = 0
    blocks = 0
    sites: dict[str, Site] = {}
    launches: dict[str, int] = {}
    for row in reader:
        if len(row) == 2 and row[0] == _FILE_PATH:
            blocks += 1
            columns, line = {}, 0
            continue
        if len(row) == 2 and row[0] == _FUNCTION_NAME:
            kernel, columns, line = row[1], {}, 0
            launches[kernel] = launches.get(kernel, 0) + 1
            continue
        if row and row[0] == _LINE_NO:
            columns, sass_column = _columns(row)
            fields = len(row)
            line = 0
            continue
        if not columns or not any(cell.strip() for cell in row):
            continue
        row = _aligned(row, fields)
        number = row[columns[_LINE_NO]].strip()
        if number.isdigit():
            line = int(number)
            continue
        address = row[columns[_ADDRESS]].strip()
        if not address.startswith("0x"):
            continue
        sass = row[sass_column] if 0 <= sass_column < len(row) else ""
        matched = _OPCODE.match(sass)
        opcode = matched.group(1).partition(".")[0] if matched else ""
        width = row[columns[_WIDTH]].strip() if _WIDTH in columns else ""
        key = f"{kernel}@{address}"
        site = sites.get(key)
        if site is None:
            site = sites[key] = Site(
                address=address,
                kernel=kernel,
                line=line,
                opcode=opcode,
                sass=sass.strip(),
                inst=0,
                bits=int(width) if width.isdigit() else 0,
                samples=0,
            )
        site.inst += _int(row[columns[_INST]])
        if _SAMPLES in columns:
            site.samples += _int(row[columns[_SAMPLES]])
        for reason in STALL_REASONS:
            name = pcsamp_metric(reason)
            if name in columns:
                got = _int(row[columns[name]])
                if got:
                    site.stalls[reason] = site.stalls.get(reason, 0) + got
    if blocks == 0:
        raise ValueError(
            "no source block; the target needs CUTE_DSL_LINEINFO=1 in its environment"
        )
    return list(sites.values()), launches


def region_of(line: int) -> str:
    """The ``table.py`` region a line number falls in, or an empty string."""
    for name, low, high in REGIONS:
        if low <= line <= high:
            return name
    return ""


def collect(args: argparse.Namespace) -> str:
    """Run the source pass and import its page.

    Returns:
        The imported CSV text.

    Raises:
        RuntimeError: If either NCU invocation exits nonzero.
    """
    binary = resolve_tool(args.ncu)
    if args.import_only:
        written = report_file(args.import_only)
    else:
        stem = f"table-stage-{args.kernel}-{args.shape}"
        if args.rot_run:
            stem = f"{stem}-run{args.rot_run}"
        report = args.report or os.path.join(tempfile.gettempdir(), stem)
        command = ncu_command(
            SOURCE_TABLE,
            target_argv(args),
            ncu=binary,
            extra=(
                *export_flags(report),
                "--kernel-name",
                f"regex:{KERNELS[args.kernel]}",
                "--launch-count",
                "1",
            ),
        )
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(
                f"source pass exited {done.returncode}: {done.stderr[-2000:]}"
            )
        written = report_file(report)
    read = subprocess.run(
        import_command(written, ncu=binary, page="source", print_source=SOURCE_VIEW),
        capture_output=True,
        text=True,
        check=False,
    )
    if read.returncode != 0:
        raise RuntimeError(f"import exited {read.returncode}: {read.stderr[-2000:]}")
    return read.stdout


def report(args: argparse.Namespace, sites: Sequence[Site]) -> None:
    """Print the opcode census, the region census, and the per-address listing."""
    total_inst = sum(one.inst for one in sites)
    total_samples = sum(one.samples for one in sites)
    total_lsu = sum(one.inst for one in sites if one.opcode in LSU_OPCODES)
    print(f"kernel {args.kernel}  shape {args.shape}  addresses {len(sites)}")
    print(f"inst {total_inst:,}  lsu {total_lsu:,}  samples {total_samples:,}")

    print("\nopcode                inst        share    samples     share")
    by_opcode: dict[str, list[int]] = {}
    for one in sites:
        got = by_opcode.setdefault(one.opcode, [0, 0])
        got[0] += one.inst
        got[1] += one.samples
    for opcode, (inst, samples) in sorted(by_opcode.items(), key=lambda kv: -kv[1][0])[
        : args.top
    ]:
        print(
            f"{opcode:<18} {inst:>12,} {pct(inst, total_inst):>8}  "
            f"{samples:>10,} {pct(samples, total_samples):>8}"
        )

    print("\nregion                            inst      share     samples    share")
    by_region: dict[str, list[int]] = {}
    for one in sites:
        got = by_region.setdefault(
            region_of(one.line) or "(outside table.py spans)", [0, 0]
        )
        got[0] += one.inst
        got[1] += one.samples
    for name, (inst, samples) in sorted(by_region.items(), key=lambda kv: -kv[1][1]):
        print(
            f"{name:<30} {inst:>12,} {pct(inst, total_inst):>8} "
            f"{samples:>10,} {pct(samples, total_samples):>8}"
        )

    listing = [one for one in sites if not args.opcode or one.opcode == args.opcode]
    listing.sort(key=lambda one: (-one.samples, -one.inst))
    print("\naddress             line  region                 opcode      bits")
    print("        inst  samples   share  top stall")
    for one in listing[: args.top]:
        stall = max(one.stalls.items(), key=lambda kv: kv[1], default=("", 0))
        print(
            f"{one.address:<18} {one.line:>5}  {region_of(one.line) or '-':<22} "
            f"{one.opcode:<10} {one.bits or '':>4} {one.inst:>12,} "
            f"{one.samples:>8,} {pct(one.samples, total_samples):>7}  "
            f"{stall[0]}:{stall[1]}"
        )
        print(f"    {one.sass[:120]}")


def pct(part: int, whole: int) -> str:
    """``part`` as a percentage of ``whole``, or ``-`` when the whole is zero."""
    return "-" if whole <= 0 else f"{100.0 * part / whole:.2f}%"


def main(argv: Sequence[str] | None = None) -> int:
    """Collect, parse, and print.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    text = collect(args)
    sites, launches = parse_addresses(text)
    print(f"launches {launches}")
    report(args, sites)
    if args.json:
        with open(args.json, "w", encoding="ascii") as out:
            json.dump(
                [
                    {
                        "address": one.address,
                        "kernel": one.kernel,
                        "line": one.line,
                        "opcode": one.opcode,
                        "sass": one.sass,
                        "inst": one.inst,
                        "bits": one.bits,
                        "samples": one.samples,
                        "stalls": one.stalls,
                    }
                    for one in sites
                ],
                out,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
