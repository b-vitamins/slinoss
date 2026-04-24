#!/usr/bin/env python3
"""Run Nsight Systems CUDA traces and summarize hot kernel launch timings."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
from typing import Iterable

PROFILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROFILE_DIR.parent.parent


def _parse_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text.replace(",", ""))


def _launch_label(kernel_name: str, launch_index: int) -> str:
    if "kernel_cutlass_kernel_" in kernel_name:
        return "main"
    return f"launch_{launch_index}"


def parse_nsys_cuda_gpu_trace_csv(csv_text: str) -> list[dict[str, object]]:
    """Parse `nsys stats --report cuda_gpu_trace --format csv` output."""

    trace_rows: list[dict[str, str]] = []
    for row in csv.DictReader(csv_text.splitlines()):
        name = row.get("Name")
        duration_ns = _parse_float(row.get("Duration (ns)"))
        start_ns = _parse_float(row.get("Start (ns)"))
        registers_per_thread = _parse_float(row.get("Reg/Trd"))
        if (
            not name
            or duration_ns is None
            or start_ns is None
            or registers_per_thread is None
        ):
            continue
        trace_rows.append(row)

    trace_rows.sort(key=lambda row: float(row["Start (ns)"].replace(",", "")))

    launches: list[dict[str, object]] = []
    for launch_index, row in enumerate(trace_rows):
        kernel_name = row["Name"]
        duration_ns = _parse_float(row.get("Duration (ns)"))
        start_ns = _parse_float(row.get("Start (ns)"))
        registers_per_thread = _parse_float(row.get("Reg/Trd"))
        dynamic_smem_mb = _parse_float(row.get("DymSMem (MB)"))
        static_smem_mb = _parse_float(row.get("StcSMem (MB)"))
        launches.append(
            {
                "launch_index": launch_index,
                "kernel_name": kernel_name,
                "kernel_label": _launch_label(kernel_name, launch_index),
                "start_ns": start_ns,
                "duration_ms": (
                    None if duration_ns is None else float(duration_ns) / 1_000_000.0
                ),
                "duration_source": "nsys_cuda_gpu_trace_hot",
                "registers_per_thread": registers_per_thread,
                "smem_dynamic_kib": (
                    None if dynamic_smem_mb is None else dynamic_smem_mb * 1024.0
                ),
                "smem_static_kib": (
                    None if static_smem_mb is None else static_smem_mb * 1024.0
                ),
            }
        )
    return launches


def run_nsys_cuda_gpu_trace(
    script: Path,
    script_args: Iterable[str],
    *,
    nsys: str,
    python: str,
    output_base: Path,
) -> list[dict[str, object]]:
    """Profile one CUDA profiler capture range and return chronological kernels."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    profile_cmd = [
        nsys,
        "profile",
        "--trace=cuda",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        str(output_base),
        python,
        str(script),
        *list(script_args),
    ]
    profile_proc = subprocess.run(
        profile_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    profile_output = (profile_proc.stdout or "") + (profile_proc.stderr or "")
    if profile_proc.returncode != 0:
        raise RuntimeError(f"NSYS profile failed:\n{profile_output}")

    report_path = output_base.with_suffix(".nsys-rep")
    stats_base = output_base.with_name(f"{output_base.name}_stats")
    stats_cmd = [
        nsys,
        "stats",
        "--report",
        "cuda_gpu_trace",
        "--format",
        "csv",
        "--force-export=true",
        "--output",
        str(stats_base),
        str(report_path),
    ]
    stats_proc = subprocess.run(
        stats_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stats_output = (stats_proc.stdout or "") + (stats_proc.stderr or "")
    if stats_proc.returncode != 0:
        raise RuntimeError(f"NSYS stats failed:\n{stats_output}")

    csv_path = Path(f"{stats_base}_cuda_gpu_trace.csv")
    return parse_nsys_cuda_gpu_trace_csv(csv_path.read_text())
