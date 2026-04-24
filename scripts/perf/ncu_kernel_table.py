#!/usr/bin/env python3
"""Benchmark and Nsight-Compute profile the canonical CuTe training kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
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
    FfnPerfConfig,
    KERNEL_ORDER,
    MixerTailPerfConfig,
    ScanPrepPerfConfig,
    V2KernelPerfConfig,
    build_kernel_runner,
)
import ncu_report  # noqa: E402
import nsys_report  # noqa: E402

NCU_REPORT = cast(Any, ncu_report)
NSYS_REPORT = cast(Any, nsys_report)


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
    parser.add_argument("--nsys", default="nsys")
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
    parser.add_argument(
        "--hot-launch-timing",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Use Nsight Systems for per-CUDA-launch hot timings. "
            "'auto' profiles only logical kernels that NCU reports as multi-launch."
        ),
    )
    parser.add_argument(
        "--nsys-output-dir",
        type=Path,
        default=None,
        help="Keep NSYS timing artifacts under this directory instead of a temp dir.",
    )
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
    parser.add_argument("--ffn-d-model", type=int, default=FfnPerfConfig().d_model)
    parser.add_argument(
        "--ffn-hidden-dim", type=int, default=FfnPerfConfig().hidden_dim
    )
    parser.add_argument(
        "--ffn-kind",
        choices=("gelu", "swiglu"),
        default=FfnPerfConfig().kind,
    )
    parser.add_argument("--ffn-eps", type=float, default=FfnPerfConfig().eps)
    parser.add_argument(
        "--ffn-warps-per-block",
        type=int,
        default=FfnPerfConfig().warps_per_block,
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


def _make_ffn_cfg(args: argparse.Namespace) -> FfnPerfConfig:
    return FfnPerfConfig(
        batch=args.batch,
        T=args.T,
        d_model=args.ffn_d_model,
        hidden_dim=args.ffn_hidden_dim,
        kind=args.ffn_kind,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
        eps=args.ffn_eps,
        warps_per_block=args.ffn_warps_per_block,
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
        "--ffn-d-model",
        str(int(args.ffn_d_model)),
        "--ffn-hidden-dim",
        str(int(args.ffn_hidden_dim)),
        "--ffn-kind",
        args.ffn_kind,
        "--ffn-eps",
        str(float(args.ffn_eps)),
        "--ffn-warps-per-block",
        str(int(args.ffn_warps_per_block)),
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


def _format_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f} ms"


def _summary_float(summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _representative_launch(
    launches: list[dict[str, object]],
) -> dict[str, object] | None:
    for launch in launches:
        if launch.get("kernel_label") == "main":
            return launch
    return launches[-1] if launches else None


def _hot_launch_timing_requested(
    mode: str,
    ncu_launches: list[dict[str, object]],
) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    return len(ncu_launches) > 1


def _timing_output_base(
    *,
    args: argparse.Namespace,
    kernel_name: str,
    temp_dir: tempfile.TemporaryDirectory[str] | None,
) -> Path:
    if args.nsys_output_dir is not None:
        return args.nsys_output_dir / f"{kernel_name}_nsys"
    if temp_dir is None:
        raise RuntimeError("Temporary NSYS directory was not initialized")
    return Path(temp_dir.name) / f"{kernel_name}_nsys"


def _hot_duration_ms(summary: dict[str, object]) -> float | None:
    return _summary_float(summary, "duration_ms")


def _hot_total_ms(launches: list[dict[str, object]]) -> float | None:
    durations = [_hot_duration_ms(launch) for launch in launches]
    if not durations or any(duration is None for duration in durations):
        return None
    return float(sum(cast(float, duration) for duration in durations))


def _event_normalized_hot_launches(
    launches: list[dict[str, object]],
    *,
    event_total_ms: float,
) -> list[dict[str, object]]:
    hot_total_ms = _hot_total_ms(launches)
    normalized: list[dict[str, object]] = []
    for launch in launches:
        normalized_launch = dict(launch)
        duration_ms = _hot_duration_ms(launch)
        if hot_total_ms is not None and hot_total_ms > 0.0 and duration_ms is not None:
            fraction = duration_ms / hot_total_ms
            normalized_launch["event_fraction"] = fraction
            normalized_launch["event_estimate_ms"] = event_total_ms * fraction
            normalized_launch["event_estimate_source"] = (
                "cuda_event_total_scaled_by_nsys_trace_fraction"
            )
        else:
            normalized_launch["event_fraction"] = None
            normalized_launch["event_estimate_ms"] = None
            normalized_launch["event_estimate_source"] = None
        normalized.append(normalized_launch)
    return normalized


def _launch_by_index(
    launches: list[dict[str, object]],
    launch_index: int,
) -> dict[str, object] | None:
    for launch in launches:
        if int(cast(int, launch["launch_index"])) == launch_index:
            return launch
    return None


def _format_ncu_counter_suffix(summary: dict[str, object]) -> str:
    return (
        f"NCU DRAM {_format_pct(_summary_float(summary, 'dram_pct'))} "
        f"({_format_gbs(_summary_float(summary, 'dram_total_gbs'))}), "
        f"occ {_format_pct(_summary_float(summary, 'achieved_occupancy'))}, "
        f"no-eligible {_format_pct(_summary_float(summary, 'no_eligible'))}, "
        f"bank conflicts {_format_bank(_summary_float(summary, 'bank_conflicts'))}, "
        f"regs/thread {_format_regs(_summary_float(summary, 'registers_per_thread'))}, "
        f"smem {_format_kib(_summary_float(summary, 'smem_total_kib'))} "
        f"(dyn {_format_kib(_summary_float(summary, 'smem_dynamic_kib'))}, "
        f"static {_format_kib(_summary_float(summary, 'smem_static_kib'))}, "
        f"driver {_format_kib(_summary_float(summary, 'smem_driver_kib'))})."
    )


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
    ffn_cfg = _make_ffn_cfg(args)
    device = torch.device(args.device)

    if args.run_kernel is not None:
        runner = build_kernel_runner(
            args.run_kernel,
            v2_cfg=v2_cfg,
            scanprep_cfg=scanprep_cfg,
            mixer_tail_cfg=mixer_tail_cfg,
            ffn_cfg=ffn_cfg,
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
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.nsys_output_dir is None and args.hot_launch_timing != "never":
        temp_dir = tempfile.TemporaryDirectory(prefix="slinoss-nsys-")
    try:
        for kernel_name in requested:
            torch.cuda.empty_cache()
            runner = build_kernel_runner(
                kernel_name,
                v2_cfg=v2_cfg,
                scanprep_cfg=scanprep_cfg,
                mixer_tail_cfg=mixer_tail_cfg,
                ffn_cfg=ffn_cfg,
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
            ncu_launches = cast(
                list[dict[str, object]],
                NCU_REPORT.parse_ncu_launch_summaries(ncu_output),
            )
            hot_launches: list[dict[str, object]] = []
            if _hot_launch_timing_requested(args.hot_launch_timing, ncu_launches):
                hot_launches = cast(
                    list[dict[str, object]],
                    NSYS_REPORT.run_nsys_cuda_gpu_trace(
                        Path(__file__).resolve(),
                        _kernel_cli_args(args, runner.name),
                        nsys=args.nsys,
                        python=args.python,
                        output_base=_timing_output_base(
                            args=args,
                            kernel_name=runner.name,
                            temp_dir=temp_dir,
                        ),
                    ),
                )
            representative_launch = _representative_launch(ncu_launches)
            ncu_summary = (
                representative_launch
                if representative_launch is not None
                else cast(dict[str, object], NCU_REPORT.parse_ncu_summary(ncu_output))
            )
            bench_ms = float(cast(float, bench["mean_ms"]))
            bench_gbs = _bench_gbs(runner.effective_bytes, bench_ms)
            normalized_hot_launches = _event_normalized_hot_launches(
                hot_launches,
                event_total_ms=bench_ms,
            )
            label = (
                runner.name if runner.note is None else f"{runner.name} ({runner.note})"
            )
            if len(ncu_launches) <= 1:
                print(
                    f"- {label}: {bench_ms:.4f} ms CUDA-event hot path, "
                    f"bench {bench_gbs:.1f} GB/s; "
                    f"{_format_ncu_counter_suffix(ncu_summary)}"
                )
            else:
                hot_total = _hot_total_ms(hot_launches)
                hot_suffix = (
                    f"; NSYS trace launch-sum {_format_ms(hot_total)}, "
                    "split normalized to CUDA-event total"
                    if hot_launches
                    else ""
                )
                print(
                    f"- {label}: {bench_ms:.4f} ms CUDA-event hot path, "
                    f"bench {bench_gbs:.1f} GB/s; "
                    f"CUDA launches {len(ncu_launches)}{hot_suffix}; "
                    "NCU counters below use profiled replay, not hot timing."
                )
                for launch in ncu_launches:
                    launch_index = int(cast(int, launch["launch_index"]))
                    launch_label = str(launch["kernel_label"])
                    ncu_duration_ms = _summary_float(launch, "duration_ms")
                    hot_launch = _launch_by_index(
                        normalized_hot_launches,
                        launch_index,
                    )
                    hot_duration = None
                    event_estimate = None
                    event_fraction = None
                    if hot_launch is not None:
                        hot_duration = _hot_duration_ms(hot_launch)
                        event_estimate = _summary_float(
                            hot_launch,
                            "event_estimate_ms",
                        )
                        event_fraction = _summary_float(
                            hot_launch,
                            "event_fraction",
                        )
                    fraction_text = (
                        "N/A"
                        if event_fraction is None
                        else f"{event_fraction * 100.0:.1f}%"
                    )
                    print(
                        f"  - launch[{launch_index}] {launch_label}: "
                        f"event-est {_format_ms(event_estimate)} ({fraction_text}), "
                        f"NSYS trace {_format_ms(hot_duration)}, "
                        f"NCU replay {_format_ms(ncu_duration_ms)}; "
                        f"{_format_ncu_counter_suffix(launch)}"
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
                    "ncu_launches": ncu_launches,
                    "hot_launches": normalized_hot_launches,
                    "hot_launch_total_ms": _hot_total_ms(hot_launches),
                }
            )
            del runner
            torch.cuda.empty_cache()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

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
            "ffn_config": {
                "batch": ffn_cfg.batch,
                "T": ffn_cfg.T,
                "d_model": ffn_cfg.d_model,
                "hidden_dim": ffn_cfg.hidden_dim,
                "projected_dim": ffn_cfg.projected_dim,
                "kind": ffn_cfg.kind,
                "dtype": str(ffn_cfg.dtype),
                "device": ffn_cfg.device,
                "seed": ffn_cfg.seed,
                "eps": ffn_cfg.eps,
                "warps_per_block": ffn_cfg.warps_per_block,
            },
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
