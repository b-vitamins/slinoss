#!/usr/bin/env python3
"""Benchmark and Nsight-Compute profile the canonical CuTe training kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, cast

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import dtype_from_str, ensure_cuda  # noqa: E402
from _training import DEFAULT_TRAINING_PERF_CONFIG  # noqa: E402
from _ncu_kernels import (  # noqa: E402
    DEFAULT_V2_BATCH,
    DEFAULT_V2_BC_GROUPS,
    DEFAULT_V2_CHUNK,
    DEFAULT_V2_HEADS,
    DEFAULT_V2_N,
    DEFAULT_V2_P,
    DEFAULT_V2_T,
    KERNEL_ORDER,
    MixerTailPerfConfig,
    ScanPrepPerfConfig,
    V2KernelPerfConfig,
    build_kernel_runner,
)
import ncu_report  # noqa: E402

NCU_REPORT = cast(Any, ncu_report)


NCU_SECTIONS = (
    "MemoryWorkloadAnalysis_Tables",
    "LaunchStats",
    "SchedulerStats",
    "WarpStateStats",
    "InstructionStats",
    "SpeedOfLight",
)

DEFAULT_DTYPE = (
    "fp16"
    if DEFAULT_TRAINING_PERF_CONFIG.dtype == torch.float16
    else "bf16"
    if DEFAULT_TRAINING_PERF_CONFIG.dtype == torch.bfloat16
    else "fp32"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List supported kernels.")
    parser.add_argument(
        "--kernel",
        action="append",
        choices=KERNEL_ORDER,
        help="Limit the report to one or more kernels.",
    )
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=1,
        help="Warm launches to run before the profiled NCU launch.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("fp16", "bf16", "fp32"), default=DEFAULT_DTYPE
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_V2_BATCH)
    parser.add_argument("--heads", type=int, default=DEFAULT_V2_HEADS)
    parser.add_argument("--T", type=int, default=DEFAULT_V2_T)
    parser.add_argument("--N", type=int, default=DEFAULT_V2_N)
    parser.add_argument("--P", type=int, default=DEFAULT_V2_P)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_V2_CHUNK)
    parser.add_argument("--bc-groups", type=int, default=DEFAULT_V2_BC_GROUPS)
    parser.add_argument("--scanprep-batch", type=int, default=DEFAULT_V2_BATCH)
    parser.add_argument("--scanprep-heads", type=int, default=DEFAULT_V2_HEADS)
    parser.add_argument("--scanprep-T", type=int, default=DEFAULT_V2_T)
    parser.add_argument("--scanprep-N", type=int, default=DEFAULT_V2_N)
    parser.add_argument("--scanprep-P", type=int, default=DEFAULT_V2_P)
    parser.add_argument(
        "--scanprep-bc-groups",
        type=int,
        default=DEFAULT_V2_BC_GROUPS,
    )
    parser.add_argument(
        "--scanprep-dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--run-kernel",
        choices=KERNEL_ORDER,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _make_v2_cfg(args: argparse.Namespace) -> V2KernelPerfConfig:
    return V2KernelPerfConfig(
        batch=args.batch,
        heads=args.heads,
        T=args.T,
        N=args.N,
        P=args.P,
        chunk_size=args.chunk_size,
        bc_groups=args.bc_groups,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )


def _make_scanprep_cfg(args: argparse.Namespace) -> ScanPrepPerfConfig:
    return ScanPrepPerfConfig(
        batch=args.scanprep_batch,
        heads=args.scanprep_heads,
        T=args.scanprep_T,
        N=args.scanprep_N,
        P=args.scanprep_P,
        bc_groups=args.scanprep_bc_groups,
        dtype=dtype_from_str(args.scanprep_dtype),
        device=args.device,
        seed=args.seed,
    )


def _make_mixer_tail_cfg(args: argparse.Namespace) -> MixerTailPerfConfig:
    return MixerTailPerfConfig(
        batch=args.batch,
        heads=args.heads,
        T=args.T,
        P=args.P,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )


def _benchmark_kernel(
    runner,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeat: int,
) -> dict[str, float | list[float]]:
    for _ in range(max(0, int(warmup))):
        runner.prepare()
        runner.launch()
    torch.cuda.synchronize(device)

    stream = torch.cuda.current_stream(device=device)
    samples_ms: list[float] = []
    for _ in range(max(1, int(repeat))):
        launch_samples: list[float] = []
        for _ in range(max(1, int(iterations))):
            runner.prepare()
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            runner.launch()
            end.record(stream)
            torch.cuda.synchronize(device)
            launch_samples.append(float(start.elapsed_time(end)))
        samples_ms.append(float(statistics.fmean(launch_samples)))
    return {
        "samples_ms": samples_ms,
        "mean_ms": float(statistics.fmean(samples_ms)),
        "median_ms": float(statistics.median(samples_ms)),
        "stdev_ms": float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
    }


def _bench_gbs(effective_bytes: int, mean_ms: float) -> float:
    if mean_ms <= 0.0:
        return 0.0
    return float(effective_bytes) / (mean_ms * 1_000_000.0)


def _run_kernel_once(
    runner,
    *,
    device: torch.device,
    warmup: int,
) -> None:
    for _ in range(max(0, int(warmup))):
        runner.prepare()
        runner.launch()
    torch.cuda.synchronize(device)
    runner.prepare()
    torch.cuda.profiler.start()
    runner.launch()
    torch.cuda.synchronize(device)
    torch.cuda.profiler.stop()


def _kernel_cli_args(args: argparse.Namespace, kernel_name: str) -> list[str]:
    return [
        "--run-kernel",
        kernel_name,
        "--profile-warmup",
        str(int(args.profile_warmup)),
        "--device",
        args.device,
        "--seed",
        str(int(args.seed)),
        "--dtype",
        args.dtype,
        "--batch",
        str(int(args.batch)),
        "--heads",
        str(int(args.heads)),
        "--T",
        str(int(args.T)),
        "--N",
        str(int(args.N)),
        "--P",
        str(int(args.P)),
        "--chunk-size",
        str(int(args.chunk_size)),
        "--bc-groups",
        str(int(args.bc_groups)),
        "--scanprep-batch",
        str(int(args.scanprep_batch)),
        "--scanprep-heads",
        str(int(args.scanprep_heads)),
        "--scanprep-T",
        str(int(args.scanprep_T)),
        "--scanprep-N",
        str(int(args.scanprep_N)),
        "--scanprep-P",
        str(int(args.scanprep_P)),
        "--scanprep-bc-groups",
        str(int(args.scanprep_bc_groups)),
        "--scanprep-dtype",
        args.scanprep_dtype,
    ]


def _format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _format_gbs(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} GB/s"


def _format_bank(value: float | None) -> str:
    return "N/A" if value is None else str(int(round(value)))


def _format_regs(value: float | None) -> str:
    return "N/A" if value is None else str(int(round(value)))


def _format_kib(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} KiB"


def main() -> int:
    args = _parse_args()
    if args.list:
        for name in KERNEL_ORDER:
            print(name)
        return 0

    ensure_cuda(args.device)
    v2_cfg = _make_v2_cfg(args)
    scanprep_cfg = _make_scanprep_cfg(args)
    mixer_tail_cfg = _make_mixer_tail_cfg(args)
    device = torch.device(args.device)

    if args.run_kernel is not None:
        runner = build_kernel_runner(
            args.run_kernel,
            v2_cfg=v2_cfg,
            scanprep_cfg=scanprep_cfg,
            mixer_tail_cfg=mixer_tail_cfg,
        )
        _run_kernel_once(
            runner,
            device=device,
            warmup=args.profile_warmup,
        )
        return 0

    requested_set = set(args.kernel or KERNEL_ORDER)
    requested = [name for name in KERNEL_ORDER if name in requested_set]

    rows: list[dict[str, object]] = []
    for kernel_name in requested:
        torch.cuda.empty_cache()
        runner = build_kernel_runner(
            kernel_name,
            v2_cfg=v2_cfg,
            scanprep_cfg=scanprep_cfg,
            mixer_tail_cfg=mixer_tail_cfg,
        )
        bench = _benchmark_kernel(
            runner,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeat=args.repeat,
        )
        ncu_output = NCU_REPORT.run_ncu_sections(
            # One NCU invocation per kernel keeps the parser compact and avoids
            # the multi-kernel wall of output the user wants to avoid.
            NCU_SECTIONS,
            Path(__file__).resolve(),
            _kernel_cli_args(args, runner.name),
            ncu=args.ncu,
            python=args.python,
        )
        ncu_summary = NCU_REPORT.parse_ncu_summary(ncu_output)
        bench_ms = float(cast(float, bench["mean_ms"]))
        bench_gbs = _bench_gbs(runner.effective_bytes, bench_ms)
        label = runner.name if runner.note is None else f"{runner.name} ({runner.note})"
        print(
            f"- {label}: {bench_ms:.4f} ms, bench {bench_gbs:.1f} GB/s; "
            f"NCU DRAM {_format_pct(ncu_summary['dram_pct'])} "
            f"({_format_gbs(ncu_summary['dram_total_gbs'])}), "
            f"occ {_format_pct(ncu_summary['achieved_occupancy'])}, "
            f"no-eligible {_format_pct(ncu_summary['no_eligible'])}, "
            f"bank conflicts {_format_bank(ncu_summary['bank_conflicts'])}, "
            f"regs/thread {_format_regs(ncu_summary['registers_per_thread'])}, "
            f"smem {_format_kib(ncu_summary['smem_total_kib'])} "
            f"(dyn {_format_kib(ncu_summary['smem_dynamic_kib'])}, "
            f"static {_format_kib(ncu_summary['smem_static_kib'])}, "
            f"driver {_format_kib(ncu_summary['smem_driver_kib'])})."
        )
        rows.append(
            {
                "kernel": runner.name,
                "label": label,
                "note": runner.note,
                "effective_bytes": int(runner.effective_bytes),
                "bench": bench,
                "bench_gbs": bench_gbs,
                "ncu": ncu_summary,
            }
        )
        del runner
        torch.cuda.empty_cache()

    if args.json_out is not None:
        payload = {
            "device_name": torch.cuda.get_device_name(0),
            "v2x2ssd_config": {
                "batch": v2_cfg.batch,
                "heads": v2_cfg.heads,
                "bc_groups": v2_cfg.resolved_bc_groups,
                "T": v2_cfg.T,
                "N": v2_cfg.N,
                "P": v2_cfg.P,
                "chunk_size": v2_cfg.chunk_size,
                "dtype": str(v2_cfg.dtype),
                "device": v2_cfg.device,
                "seed": v2_cfg.seed,
            },
            "scanprep_config": {
                "batch": scanprep_cfg.batch,
                "heads": scanprep_cfg.heads,
                "bc_groups": scanprep_cfg.resolved_bc_groups,
                "T": scanprep_cfg.T,
                "N": scanprep_cfg.N,
                "P": scanprep_cfg.P,
                "dtype": str(scanprep_cfg.dtype),
                "device": scanprep_cfg.device,
                "seed": scanprep_cfg.seed,
            },
            "mixer_tail_config": {
                "batch": mixer_tail_cfg.batch,
                "heads": mixer_tail_cfg.heads,
                "T": mixer_tail_cfg.T,
                "P": mixer_tail_cfg.P,
                "hidden_dim": mixer_tail_cfg.hidden_dim,
                "dtype": str(mixer_tail_cfg.dtype),
                "d_skip_dtype": str(mixer_tail_cfg.d_skip_dtype),
                "device": mixer_tail_cfg.device,
                "seed": mixer_tail_cfg.seed,
                "eps": mixer_tail_cfg.eps,
                "warps_per_block": mixer_tail_cfg.warps_per_block,
            },
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
