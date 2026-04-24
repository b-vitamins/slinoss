#!/usr/bin/env python3
"""Run Nsight Compute sections for a perf profiling script and summarize them."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable

PROFILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROFILE_DIR.parent.parent

KERNEL_SCRIPTS = {
    "scanprep_fwd": "profile_scanprep_fwd.py",
    "scanprep_bwd": "profile_scanprep_bwd.py",
}

NUM_RE = r"([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:[eE][+-]?\d+)?)"
KERNEL_HEADER_RE = re.compile(
    r"^  (?P<kernel_name>\S.*?), Context \d+, Stream \d+, Device \d+, CC [\d.]+$",
    flags=re.MULTILINE,
)


def _find_match(patterns: Iterable[str], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match
    return None


def _parse_float(text: str) -> float:
    return float(text.replace(",", ""))


def _parse_metric_value(text: str, metric_name: str) -> float | None:
    pattern = rf"^\s*{re.escape(metric_name)}\s+(?:\S+\s+)?{NUM_RE}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return _parse_float(match.group(1))


def _parse_percent(text: str, patterns: Iterable[str]) -> float | None:
    match = _find_match(patterns, text)
    if not match:
        return None
    return _parse_float(match.group(1))


def _parse_unit_value(
    text: str,
    patterns: Iterable[str],
    *,
    unit_to_gbs: bool = False,
) -> float | None:
    match = _find_match(patterns, text)
    if not match:
        return None
    value = _parse_float(match.group(1))
    unit = (
        match.group(2).lower()
        if match.lastindex is not None and match.lastindex >= 2 and match.group(2)
        else ""
    )
    if not unit_to_gbs:
        return value
    if unit in ("gb/s", "gbyte/s", "gbytes/s") or unit == "":
        return value
    if unit in ("mb/s", "mbyte/s", "mbytes/s"):
        return value / 1024.0
    if unit in ("kb/s", "kbyte/s", "kbytes/s"):
        return value / (1024.0 * 1024.0)
    if unit in ("b/s", "bytes/s"):
        return value / (1024.0 * 1024.0 * 1024.0)
    return value


def _parse_unit_value_table(
    text: str,
    metric_name: str,
    *,
    unit_to_gbs: bool = False,
) -> float | None:
    pattern = rf"^\s*{re.escape(metric_name)}\s+(\S+)\s+{NUM_RE}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    unit = match.group(1).lower()
    value = _parse_float(match.group(2))
    if not unit_to_gbs:
        return value
    unit = unit.replace("bytes", "byte")
    unit = unit.replace("gbyte", "gb")
    unit = unit.replace("mbyte", "mb")
    unit = unit.replace("kbyte", "kb")
    unit = unit.replace("byte", "b")
    if unit in ("gb/s", "gbs", "gbps"):
        return value
    if unit in ("mb/s", "mbs", "mbps"):
        return value / 1024.0
    if unit in ("kb/s", "kbs", "kbps"):
        return value / (1024.0 * 1024.0)
    if unit in ("b/s", "bs", "bps"):
        return value / (1024.0 * 1024.0 * 1024.0)
    return value


def _parse_size_kib(text: str, patterns: Iterable[str]) -> float | None:
    match = _find_match(patterns, text)
    if not match:
        return None
    value = _parse_float(match.group(1))
    unit = (
        match.group(2).lower()
        if match.lastindex is not None and match.lastindex >= 2 and match.group(2)
        else ""
    )
    unit = unit.replace("/block", "")
    if unit in ("kib", "kb") or unit == "":
        return value
    if unit in ("mib", "mb"):
        return value * 1024.0
    if unit in ("bytes", "byte", "b"):
        return value / 1024.0
    return value


def _parse_size_kib_table(text: str, metric_name: str) -> float | None:
    pattern = rf"^\s*{re.escape(metric_name)}\s+(\S+)\s+{NUM_RE}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    unit = match.group(1).lower().replace("/block", "")
    value = _parse_float(match.group(2))
    if unit in ("kib", "kb", "kbyte"):
        return value
    if unit in ("mib", "mb", "mbyte"):
        return value * 1024.0
    if unit in ("bytes", "byte", "b"):
        return value / 1024.0
    return value


def _duration_to_ms(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ("ms", "msec", "millisecond", "milliseconds"):
        return value
    if unit in ("us", "usec", "microsecond", "microseconds"):
        return value / 1000.0
    if unit in ("ns", "nsec", "nanosecond", "nanoseconds"):
        return value / 1_000_000.0
    if unit in ("s", "sec", "second", "seconds"):
        return value * 1000.0
    return value


def _parse_duration_ms(text: str) -> float | None:
    pattern = rf"^\s*Duration\s+(\S+)\s+{NUM_RE}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raw_match = _find_match(
            [rf"gpu__time_duration\.sum\s+{NUM_RE}\s*([A-Za-z]+)?"],
            text,
        )
        if not raw_match:
            return None
        value = _parse_float(raw_match.group(1))
        unit = raw_match.group(2) or "ms"
        return _duration_to_ms(value, unit)
    return _duration_to_ms(_parse_float(match.group(2)), match.group(1))


def _launch_label(kernel_name: str, launch_index: int) -> str:
    if "kernel_cutlass_kernel_" in kernel_name:
        return "main"
    return f"launch_{launch_index}"


def _fmt_pct(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None else "N/A"


def _fmt_gbs(value: float | None) -> str:
    return f"{value:.2f} GB/s" if value is not None else "N/A"


def _fmt_kib(value: float | None) -> str:
    return f"{value:.2f} KiB" if value is not None else "N/A"


def _fmt_num(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def _fmt_int(value: float | None) -> str:
    return f"{int(value):,}" if value is not None else "N/A"


def _run_ncu(
    section: str,
    script: Path,
    script_args: list[str],
    *,
    ncu: str,
    python: str,
) -> str:
    cmd = [
        ncu,
        "--section",
        section,
        "--print-details",
        "all",
        "--profile-from-start",
        "off",
        python,
        str(script),
        *script_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"NCU failed for section {section}:\n{output}")
    return output


def run_ncu_sections(
    sections: Iterable[str],
    script: Path,
    script_args: list[str],
    *,
    ncu: str,
    python: str,
) -> str:
    cmd = [ncu]
    for section in sections:
        cmd.extend(("--section", section))
    cmd.extend(
        [
            "--print-details",
            "all",
            "--profile-from-start",
            "off",
            python,
            str(script),
            *script_args,
        ]
    )
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            "NCU failed for sections "
            f"{', '.join(str(section) for section in sections)}:\n{output}"
        )
    return output


def _resolve_script(kernel: str | None) -> Path:
    if kernel is None:
        return PROFILE_DIR / KERNEL_SCRIPTS["scanprep_bwd"]
    if kernel.endswith(".py"):
        return Path(kernel)
    script_name = KERNEL_SCRIPTS.get(kernel)
    if script_name is None:
        choices = ", ".join(sorted(KERNEL_SCRIPTS))
        raise SystemExit(f"Unknown kernel key: {kernel}. Available: {choices}")
    return PROFILE_DIR / script_name


def _print_commands(
    *,
    sections: list[str],
    ncu: str,
    python: str,
    script: Path,
    script_args: list[str],
) -> None:
    print("\nCommands used (repro)\n")
    for section in sections:
        cmd = (
            f"  - {ncu} --section {section} --print-details all "
            f"--profile-from-start off {python} {script}"
        )
        if script_args:
            cmd = f"{cmd} {' '.join(script_args)}"
        print(cmd)


def parse_ncu_summary(output: str) -> dict[str, float | None]:
    dram_pct = _parse_percent(
        output,
        [
            rf"dram__throughput(?:\.avg)?\.pct_of_peak_sustained_elapsed\s+{NUM_RE}",
            rf"DRAM Throughput[^\n]*?{NUM_RE}\s*%",
        ],
    )
    if dram_pct is None:
        dram_pct = _parse_metric_value(output, "DRAM Throughput")
    dram_read = _parse_unit_value(
        output,
        [
            rf"dram__bytes_read\.sum\.per_second\s+{NUM_RE}\s*([A-Za-z/]+)?",
            rf"Requested Global Load Throughput[^\n]*?{NUM_RE}\s*([A-Za-z/]+)?",
        ],
        unit_to_gbs=True,
    )
    if dram_read is None:
        dram_read = _parse_unit_value_table(
            output,
            "dram__bytes_read.sum.per_second",
            unit_to_gbs=True,
        )
    dram_write = _parse_unit_value(
        output,
        [
            rf"dram__bytes_write\.sum\.per_second\s+{NUM_RE}\s*([A-Za-z/]+)?",
            rf"Requested Global Store Throughput[^\n]*?{NUM_RE}\s*([A-Za-z/]+)?",
        ],
        unit_to_gbs=True,
    )
    if dram_write is None:
        dram_write = _parse_unit_value_table(
            output,
            "dram__bytes_write.sum.per_second",
            unit_to_gbs=True,
        )
    bank_ld = _parse_metric_value(
        output,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    )
    bank_st = _parse_metric_value(
        output,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    )
    achieved_occ = _parse_metric_value(output, "Achieved Occupancy")
    if achieved_occ is None:
        achieved_occ = _parse_percent(
            output,
            [
                rf"Achieved Occupancy[^\n]*?{NUM_RE}\s*%",
                rf"sm__warps_active\.avg\.pct_of_peak_sustained_active\s+{NUM_RE}",
            ],
        )
    no_eligible = _parse_metric_value(output, "No Eligible")
    regs = _parse_metric_value(output, "Registers Per Thread")
    if regs is None:
        regs = _parse_unit_value(
            output,
            [rf"launch__registers_per_thread\s+{NUM_RE}"],
        )
    smem_dynamic_kib = _parse_size_kib_table(
        output,
        "Dynamic Shared Memory Per Block",
    )
    if smem_dynamic_kib is None:
        smem_dynamic_kib = _parse_size_kib(
            output,
            [
                rf"launch__dynamic_shared_mem_per_block\s+{NUM_RE}\s*([A-Za-z]+)?",
                rf"launch__shared_mem_per_block\s+{NUM_RE}\s*([A-Za-z]+)?",
            ],
        )
    smem_static_kib = _parse_size_kib_table(
        output,
        "Static Shared Memory Per Block",
    )
    if smem_static_kib is None:
        smem_static_kib = _parse_size_kib(
            output,
            [rf"launch__static_shared_mem_per_block\s+{NUM_RE}\s*([A-Za-z]+)?"],
        )
    smem_driver_kib = _parse_size_kib_table(
        output,
        "Driver Shared Memory Per Block",
    )
    if smem_driver_kib is None:
        smem_driver_kib = _parse_size_kib(
            output,
            [rf"launch__driver_shared_mem_per_block\s+{NUM_RE}\s*([A-Za-z]+)?"],
        )
    smem_total_kib = (
        (smem_dynamic_kib or 0.0) + (smem_static_kib or 0.0) + (smem_driver_kib or 0.0)
        if (
            smem_dynamic_kib is not None
            or smem_static_kib is not None
            or smem_driver_kib is not None
        )
        else None
    )
    duration_ms = _parse_duration_ms(output)
    elapsed_cycles = _parse_metric_value(output, "Elapsed Cycles")
    total_dram = (
        dram_read + dram_write
        if dram_read is not None and dram_write is not None
        else None
    )
    bank_total = (
        (bank_ld or 0.0) + (bank_st or 0.0)
        if bank_ld is not None or bank_st is not None
        else None
    )
    return {
        "dram_pct": dram_pct,
        "dram_read_gbs": dram_read,
        "dram_write_gbs": dram_write,
        "dram_total_gbs": total_dram,
        "achieved_occupancy": achieved_occ,
        "no_eligible": no_eligible,
        "bank_conflicts": bank_total,
        "registers_per_thread": regs,
        "smem_dynamic_kib": smem_dynamic_kib,
        "smem_static_kib": smem_static_kib,
        "smem_driver_kib": smem_driver_kib,
        "smem_total_kib": smem_total_kib,
        "duration_ms": duration_ms,
        "elapsed_cycles": elapsed_cycles,
    }


def parse_ncu_launch_summaries(output: str) -> list[dict[str, object]]:
    matches = list(KERNEL_HEADER_RE.finditer(output))
    if not matches:
        summary: dict[str, object] = dict(parse_ncu_summary(output))
        summary["launch_index"] = 0
        summary["kernel_name"] = None
        summary["kernel_label"] = "launch_0"
        summary["duration_source"] = "ncu_replay_profiled"
        return [summary]

    launches: list[dict[str, object]] = []
    for launch_index, match in enumerate(matches):
        next_start = (
            matches[launch_index + 1].start()
            if launch_index + 1 < len(matches)
            else len(output)
        )
        kernel_name = match.group("kernel_name").strip()
        segment = output[match.start() : next_start]
        summary = dict(parse_ncu_summary(segment))
        summary["launch_index"] = launch_index
        summary["kernel_name"] = kernel_name
        summary["kernel_label"] = _launch_label(kernel_name, launch_index)
        summary["duration_source"] = "ncu_replay_profiled"
        launches.append(summary)
    return launches


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nsight Compute on a perf profiling script and summarize "
            "bandwidth, occupancy, and stall metrics."
        )
    )
    parser.add_argument(
        "kernel",
        nargs="?",
        default="scanprep_bwd",
        help="Kernel key or path to a profile_*.py script.",
    )
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="List available kernel keys."
    )
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.list:
        for key in sorted(KERNEL_SCRIPTS):
            print(key)
        return 0

    script = _resolve_script(args.kernel)
    if not script.exists():
        raise SystemExit(f"Profile script not found: {script}")

    script_args = [arg for arg in args.script_args if arg != "--"]
    sections = [
        "MemoryWorkloadAnalysis_Tables",
        "LaunchStats",
        "SchedulerStats",
        "WarpStateStats",
        "InstructionStats",
        "SpeedOfLight",
    ]
    outputs: dict[str, str] = {}
    for section in sections:
        outputs[section] = _run_ncu(
            section,
            script,
            script_args,
            ncu=args.ncu,
            python=args.python,
        )
        if args.verbose:
            print(outputs[section])

    mem_output = outputs["MemoryWorkloadAnalysis_Tables"]
    launch_output = outputs["LaunchStats"]
    scheduler_output = outputs["SchedulerStats"]
    warp_output = outputs["WarpStateStats"]
    instruction_output = outputs["InstructionStats"]
    sol_output = outputs["SpeedOfLight"]

    dram_pct = _parse_percent(
        mem_output,
        [
            rf"dram__throughput(?:\.avg)?\.pct_of_peak_sustained_elapsed\s+{NUM_RE}",
            rf"DRAM Throughput[^\n]*?{NUM_RE}\s*%",
        ],
    )
    if dram_pct is None:
        dram_pct = _parse_metric_value(mem_output, "DRAM Throughput")
    dram_read = _parse_unit_value(
        mem_output,
        [
            rf"dram__bytes_read\.sum\.per_second\s+{NUM_RE}\s*([A-Za-z/]+)?",
            rf"Requested Global Load Throughput[^\n]*?{NUM_RE}\s*([A-Za-z/]+)?",
        ],
        unit_to_gbs=True,
    )
    if dram_read is None:
        dram_read = _parse_unit_value_table(
            mem_output,
            "dram__bytes_read.sum.per_second",
            unit_to_gbs=True,
        )
    dram_write = _parse_unit_value(
        mem_output,
        [
            rf"dram__bytes_write\.sum\.per_second\s+{NUM_RE}\s*([A-Za-z/]+)?",
            rf"Requested Global Store Throughput[^\n]*?{NUM_RE}\s*([A-Za-z/]+)?",
        ],
        unit_to_gbs=True,
    )
    if dram_write is None:
        dram_write = _parse_unit_value_table(
            mem_output,
            "dram__bytes_write.sum.per_second",
            unit_to_gbs=True,
        )
    avg_load_sector_bytes = _parse_metric_value(
        mem_output,
        "Average Bytes Per Sector For Global Loads",
    )
    avg_store_sector_bytes = _parse_metric_value(
        mem_output,
        "Average Bytes Per Sector For Global Stores",
    )

    sh_pct = _parse_percent(
        mem_output,
        [
            rf"l1tex__data_pipe_lsu_wavefronts_mem_shared\.sum\.pct_of_peak_sustained_elapsed\s+{NUM_RE}",
            rf"Shared Memory[^\n]*?{NUM_RE}\s*%",
        ],
    )
    if sh_pct is None:
        sh_pct = _parse_metric_value(
            mem_output,
            "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed",
        )
    sh_ld_pct = _parse_percent(
        mem_output,
        [
            rf"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld\.sum\.pct_of_peak_sustained_elapsed\s+{NUM_RE}",
        ],
    )
    sh_st_pct = _parse_percent(
        mem_output,
        [
            rf"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st\.sum\.pct_of_peak_sustained_elapsed\s+{NUM_RE}",
        ],
    )
    bank_ld = _parse_metric_value(
        mem_output,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    )
    bank_st = _parse_metric_value(
        mem_output,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    )

    regs = _parse_metric_value(launch_output, "Registers Per Thread")
    if regs is None:
        regs = _parse_unit_value(
            launch_output,
            [rf"launch__registers_per_thread\s+{NUM_RE}"],
        )
    smem_dynamic_kib = _parse_size_kib_table(
        launch_output,
        "Dynamic Shared Memory Per Block",
    )
    if smem_dynamic_kib is None:
        smem_dynamic_kib = _parse_size_kib(
            launch_output,
            [rf"launch__shared_mem_per_block\s+{NUM_RE}\s*([A-Za-z]+)?"],
        )
    smem_static_kib = _parse_size_kib_table(
        launch_output,
        "Static Shared Memory Per Block",
    )
    smem_driver_kib = _parse_size_kib_table(
        launch_output,
        "Driver Shared Memory Per Block",
    )
    block_limit_sm = _parse_metric_value(launch_output, "Block Limit SM")
    block_limit_registers = _parse_metric_value(
        launch_output,
        "Block Limit Registers",
    )
    block_limit_shared_mem = _parse_metric_value(
        launch_output,
        "Block Limit Shared Mem",
    )
    block_limit_warps = _parse_metric_value(launch_output, "Block Limit Warps")
    waves_per_sm = _parse_metric_value(launch_output, "Waves Per SM")
    theo_occ = _parse_metric_value(launch_output, "Theoretical Occupancy")
    if theo_occ is None:
        theo_occ = _parse_percent(
            launch_output,
            [
                rf"Theoretical Occupancy[^\n]*?{NUM_RE}\s*%",
                rf"launch__occupancy_limit_active_warps\s+{NUM_RE}",
            ],
        )
    achieved_occ = _parse_metric_value(launch_output, "Achieved Occupancy")
    if achieved_occ is None:
        achieved_occ = _parse_percent(
            launch_output,
            [
                rf"Achieved Occupancy[^\n]*?{NUM_RE}\s*%",
                rf"sm__warps_active\.avg\.pct_of_peak_sustained_active\s+{NUM_RE}",
            ],
        )

    no_eligible = _parse_metric_value(scheduler_output, "No Eligible")
    eligible_warps = _parse_metric_value(
        scheduler_output,
        "Eligible Warps Per Scheduler",
    )
    active_warps = _parse_metric_value(
        scheduler_output,
        "Active Warps Per Scheduler",
    )
    issued_warps = _parse_metric_value(
        scheduler_output,
        "Issued Warp Per Scheduler",
    )

    warp_cycles_per_issue = _parse_metric_value(
        warp_output,
        "Warp Cycles Per Issued Instruction",
    )
    active_threads = _parse_metric_value(warp_output, "Avg. Active Threads Per Warp")
    not_pred_threads = _parse_metric_value(
        warp_output,
        "Avg. Not Predicated Off Threads Per Warp",
    )
    stall_items = [
        ("Long scoreboard", _parse_metric_value(warp_output, "Stall Long Scoreboard")),
        (
            "Short scoreboard",
            _parse_metric_value(warp_output, "Stall Short Scoreboard"),
        ),
        ("Barrier", _parse_metric_value(warp_output, "Stall Barrier")),
        ("Wait", _parse_metric_value(warp_output, "Stall Wait")),
        ("MIO throttle", _parse_metric_value(warp_output, "Stall MIO Throttle")),
        ("Not selected", _parse_metric_value(warp_output, "Stall Not Selected")),
        (
            "Math pipe throttle",
            _parse_metric_value(warp_output, "Stall Math Pipe Throttle"),
        ),
    ]

    instructions_executed = _parse_metric_value(
        instruction_output,
        "Instructions Executed",
    )
    if instructions_executed is None:
        instructions_executed = _parse_metric_value(
            instruction_output,
            "Executed Instructions",
        )
    tensor_pct = _parse_percent(
        sol_output,
        [
            rf"sm__pipe_tensor_cycles_active\.avg\.pct_of_peak_sustained_active\s+{NUM_RE}",
            rf"sm__pipe_tensor_active\.avg\.pct_of_peak_sustained_active\s+{NUM_RE}",
            rf"SM:\s*Pipe Tensor Cycles Active[^\n]*?{NUM_RE}\s*%",
        ],
    )
    if tensor_pct is None:
        tensor_pct = _parse_metric_value(sol_output, "SM: Pipe Tensor Cycles Active")

    total_dram = (
        dram_read + dram_write
        if dram_read is not None and dram_write is not None
        else None
    )
    bank_total = (
        (bank_ld or 0.0) + (bank_st or 0.0)
        if bank_ld is not None or bank_st is not None
        else None
    )

    print("Global-memory (DRAM) throughput\n")
    print(f"  - DRAM throughput (pct of peak): {_fmt_pct(dram_pct)}")
    print(f"  - dram__bytes_read.sum.per_second: {_fmt_gbs(dram_read)}")
    print(f"  - dram__bytes_write.sum.per_second: {_fmt_gbs(dram_write)}")
    print(f"  - Total DRAM BW (read+write): {_fmt_gbs(total_dram)}")
    if avg_load_sector_bytes is not None or avg_store_sector_bytes is not None:
        load = (
            f"{avg_load_sector_bytes:.2f}/32"
            if avg_load_sector_bytes is not None
            else "N/A"
        )
        store = (
            f"{avg_store_sector_bytes:.2f}/32"
            if avg_store_sector_bytes is not None
            else "N/A"
        )
        print(f"  - Avg bytes/sector: loads {load}, stores {store}")
    else:
        print("  - Avg bytes/sector: loads N/A, stores N/A")

    print("\nShared-memory throughput\n")
    print(f"  - Shared-memory throughput (pct of peak): {_fmt_pct(sh_pct)}")
    if sh_ld_pct is not None or sh_st_pct is not None:
        print(
            "  - Shared-memory load/store pct of peak: "
            f"{_fmt_pct(sh_ld_pct)} / {_fmt_pct(sh_st_pct)}"
        )
    print(f"  - Bank conflicts (total): {_fmt_int(bank_total)}")

    print("\nSM occupancy / launch stats\n")
    print(f"  - Registers per thread: {_fmt_int(regs)}")
    print(f"  - Dynamic shared memory per block: {_fmt_kib(smem_dynamic_kib)}")
    print(f"  - Static shared memory per block: {_fmt_kib(smem_static_kib)}")
    print(f"  - Driver shared memory per block: {_fmt_kib(smem_driver_kib)}")
    print(f"  - Block limit SM: {_fmt_num(block_limit_sm)}")
    print(f"  - Block limit registers: {_fmt_num(block_limit_registers)}")
    print(f"  - Block limit shared mem: {_fmt_num(block_limit_shared_mem)}")
    print(f"  - Block limit warps: {_fmt_num(block_limit_warps)}")
    print(f"  - Waves per SM: {_fmt_num(waves_per_sm)}")
    print(f"  - Theoretical occupancy: {_fmt_pct(theo_occ)}")
    print(f"  - Achieved occupancy: {_fmt_pct(achieved_occ)}")

    print("\nScheduler / issue\n")
    print(f"  - No Eligible: {_fmt_pct(no_eligible)}")
    print(f"  - Active warps / scheduler: {_fmt_num(active_warps)}")
    print(f"  - Eligible warps / scheduler: {_fmt_num(eligible_warps)}")
    print(f"  - Issued warps / scheduler: {_fmt_num(issued_warps)}")

    print("\nWarp execution / stalls\n")
    print(f"  - Warp cycles / issued inst: {_fmt_num(warp_cycles_per_issue)}")
    if active_threads is not None:
        print(
            "  - Avg active threads / warp: "
            f"{active_threads:.2f} / 32 = {active_threads / 32.0 * 100.0:.1f}%"
        )
    else:
        print("  - Avg active threads / warp: N/A")
    if not_pred_threads is not None:
        print(
            "  - Avg not-predicated-off threads / warp: "
            f"{not_pred_threads:.2f} / 32 = {not_pred_threads / 32.0 * 100.0:.1f}%"
        )
    else:
        print("  - Avg not-predicated-off threads / warp: N/A")
    for label, value in stall_items:
        print(f"  - Stall {label}: {_fmt_num(value)} inst/issue")

    print("\nInstruction / tensor mix\n")
    print(f"  - Instructions executed: {_fmt_int(instructions_executed)}")
    print(f"  - Tensor pipe active: {_fmt_pct(tensor_pct)}")

    _print_commands(
        sections=sections,
        ncu=args.ncu,
        python=args.python,
        script=script,
        script_args=script_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
