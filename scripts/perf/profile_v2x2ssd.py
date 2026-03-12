#!/usr/bin/env python3
"""Profile the canonical v2x2ssd kernels with PyTorch profiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_CHUNK,
    DEFAULT_DTYPE,
    DEFAULT_HEADS,
    DEFAULT_N,
    DEFAULT_P,
    DEFAULT_T,
    DIRECTIONS,
    STAGES,
    PerfConfig,
    build_callable,
    dtype_from_str,
    ensure_cuda,
    format_header,
    seed_all,
)
from slinoss.perf import PerfRecorder  # noqa: E402
from slinoss.perf.budget import summarize_named_samples  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="chunk_scan")
    parser.add_argument("--direction", choices=DIRECTIONS, default="backward")
    parser.add_argument("--backend", choices=("reference", "cute"), default="cute")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--P", type=int, default=DEFAULT_P)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default=DEFAULT_DTYPE,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup steps to run before active profiling steps.",
    )
    parser.add_argument(
        "--active",
        type=int,
        default=4,
        help="Profiled steps to record.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of profiler rows to print.",
    )
    parser.add_argument(
        "--sort-by",
        default="self_cuda_time_total",
        help="Profiler sort key, e.g. self_cuda_time_total or cuda_time_total.",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Optional chrome trace output path.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON summary of event-timed regions.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    seed_all(args.seed)

    cfg = PerfConfig(
        batch=args.batch,
        heads=args.heads,
        T=args.T,
        N=args.N,
        P=args.P,
        chunk_size=args.chunk_size,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )
    fn = build_callable(
        cfg, stage=args.stage, direction=args.direction, backend=args.backend
    )

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=0, warmup=int(args.warmup), active=int(args.active), repeat=1
    )
    total_steps = int(args.warmup) + int(args.active)
    captures: list[dict[str, Any]] = []
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        acc_events=True,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            recorder = PerfRecorder(device=cfg.torch_device)
            with recorder.capture_step():
                fn()
            captures.append(recorder.steps[-1])
            prof.step()

    if args.trace_out is not None:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace_out))

    print(format_header(cfg))
    print(
        f"profile: stage={args.stage} direction={args.direction} backend={args.backend}"
    )
    print(
        prof.key_averages().table(
            sort_by=args.sort_by,
            row_limit=args.top_k,
        )
    )
    active_captures = captures[-int(args.active) :]
    region_samples = [capture["regions_ms"] for capture in active_captures]
    region_summaries = summarize_named_samples(region_samples)
    if region_summaries:
        print()
        print("| region | mean_ms | median_ms | stdev_ms |")
        print("| --- | ---: | ---: | ---: |")
        for region, stats in sorted(region_summaries.items()):
            print(
                f"| {region} | {float(stats['mean_ms']):.6f} | "
                f"{float(stats['median_ms']):.6f} | {float(stats['stdev_ms']):.6f} |"
            )
    if args.trace_out is not None:
        print(f"trace: {args.trace_out}")
    if args.json_out is not None:
        payload = {
            "header": format_header(cfg),
            "profile": {
                "stage": args.stage,
                "direction": args.direction,
                "backend": args.backend,
            },
            "regions": region_summaries,
            "trace_out": None if args.trace_out is None else str(args.trace_out),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
