#!/usr/bin/env python3
"""Benchmark steady-state nextchar decode on the persistent AR path."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, cast

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from _common import dtype_from_str, ensure_cuda  # noqa: E402
from _nextchar import NextCharPerfConfig  # noqa: E402
from _nextchar_decode import (  # noqa: E402
    benchmark_decode_mode,
    build_decode_model,
    estimate_lower_bound,
    resolve_peak_spec,
    summary_speedup,
)
from slinoss.perf.schema import validate_nextchar_decode_bench_payload  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cute", "reference"), default="cute")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--expand", type=float, default=2.0)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--active-tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--peak-preset",
        default="auto",
        choices=("auto", "none", "h100_sxm5_fp16", "rtx3060_fp16"),
    )
    parser.add_argument("--peak-bw-gbps", type=float, default=None)
    parser.add_argument("--peak-tc-tflops", type=float, default=None)
    parser.add_argument("--peak-simt-tflops", type=float, default=None)
    parser.add_argument("--launch-floor-us", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _make_cfg(args: argparse.Namespace, *, batch_size: int) -> NextCharPerfConfig:
    return NextCharPerfConfig(
        batch_size=batch_size,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        d_head=args.d_head,
        d_conv=args.d_conv,
        chunk_size=args.chunk_size,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )


def _parse_batch_sizes(spec: str) -> list[int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one batch size.")
    return values


def _row(
    *,
    batch_size: int,
    persistent: dict[str, object],
    eager: dict[str, object],
    lower_bound: dict[str, object] | None,
) -> dict[str, object]:
    persistent_summary = cast(dict[str, float], persistent["summary"])
    lower_us = None
    efficiency = None
    if lower_bound is not None:
        lower_us = float(cast(float, lower_bound["t_lower_us"]))
        persistent_us = float(persistent_summary["mean_us_per_token"])
        efficiency = None if persistent_us <= 0.0 else lower_us / persistent_us
    return {
        "batch_size": batch_size,
        "persistent": persistent,
        "eager": eager,
        "speedup_vs_eager": summary_speedup(persistent, eager),
        "lower_bound": lower_bound,
        "efficiency": efficiency,
    }


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    rows: list[dict[str, object]] = []
    device = torch.device(args.device)
    peak = resolve_peak_spec(
        device=device,
        preset=args.peak_preset,
        peak_bw_gbps=args.peak_bw_gbps,
        peak_tc_tflops=args.peak_tc_tflops,
        peak_simt_tflops=args.peak_simt_tflops,
        launch_floor_us=args.launch_floor_us,
    )

    for batch_size in batch_sizes:
        cfg = _make_cfg(args, batch_size=batch_size)
        persistent = benchmark_decode_mode(
            cfg,
            backend=args.backend,
            mode="persistent",
            warmup_tokens=args.warmup_tokens,
            active_tokens=args.active_tokens,
            repeat=args.repeat,
        )
        eager = benchmark_decode_mode(
            cfg,
            backend=args.backend,
            mode="eager",
            warmup_tokens=args.warmup_tokens,
            active_tokens=args.active_tokens,
            repeat=args.repeat,
        )
        lower_bound = None
        if peak is not None:
            model = build_decode_model(cfg, backend=args.backend)
            lower_bound = estimate_lower_bound(
                model,
                batch_size=batch_size,
                peak=peak,
            ).to_dict()
        rows.append(
            _row(
                batch_size=batch_size,
                persistent=persistent,
                eager=eager,
                lower_bound=lower_bound,
            )
        )
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "| B | persistent us/token | eager us/token | speedup | t_lower us/token | efficiency |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        row_dict = cast(dict[str, object], row)
        persistent = cast(dict[str, object], row_dict["persistent"])
        eager = cast(dict[str, object], row_dict["eager"])
        persistent_summary = cast(dict[str, float], persistent["summary"])
        eager_summary = cast(dict[str, float], eager["summary"])
        persistent_us = float(persistent_summary["mean_us_per_token"])
        eager_us = float(eager_summary["mean_us_per_token"])
        lower = cast(dict[str, object] | None, row_dict["lower_bound"])
        lower_us = (
            float(cast(float, lower["t_lower_us"]))
            if isinstance(lower, dict)
            else float("nan")
        )
        efficiency = row_dict["efficiency"]
        lower_text = "-" if not isinstance(lower, dict) else f"{lower_us:.3f}"
        eff_text = (
            "-" if efficiency is None else f"{float(cast(float, efficiency)):.3f}"
        )
        print(
            f"| {int(cast(int, row_dict['batch_size']))} | {persistent_us:.3f} | "
            f"{eager_us:.3f} | {float(cast(float, row_dict['speedup_vs_eager'])):.3f}x | "
            f"{lower_text} | {eff_text} |"
        )

    payload: dict[str, Any] = {
        "kind": "bench_nextchar_decode",
        "schema_version": 1,
        "backend": args.backend,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else str(device)
        ),
        "peak": None if peak is None else peak.to_dict(),
        "rows": rows,
    }
    validate_nextchar_decode_bench_payload(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
