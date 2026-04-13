#!/usr/bin/env python3
"""Benchmark one-token mixer decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import cast

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import benchmark, dtype_from_str, ensure_cuda, seed_all  # noqa: E402
from slinoss.layers import SLinOSSMixer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--expand", type=float, default=2.0)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _run_case(args: argparse.Namespace, *, batch_size: int) -> dict[str, object]:
    dtype = dtype_from_str(args.dtype)
    seed_all(args.seed)
    mixer = SLinOSSMixer(
        args.d_model,
        d_state=args.d_state,
        expand=args.expand,
        d_head=args.d_head,
        d_conv=args.d_conv,
        chunk_size=args.chunk_size,
        device=args.device,
        dtype=dtype,
    ).eval()
    x = mixer.in_proj.weight.new_empty((batch_size, args.d_model)).normal_()
    state = mixer.init_decode_state(batch_size, device=args.device, dtype=dtype)

    def fn() -> object:
        with torch.no_grad():
            return mixer.step(x, state, inplace=True)

    stats = benchmark(
        fn,
        warmup=args.warmup,
        iterations=args.iterations,
        repeat=args.repeat,
    )
    mean_ms = cast(float, stats["mean_ms"])
    min_ms = cast(float, stats["min_ms"])
    max_ms = cast(float, stats["max_ms"])
    samples_ms = cast(list[float], stats["samples_ms"])
    return {
        "batch_size": batch_size,
        "mean_us": mean_ms * 1000.0,
        "min_us": min_ms * 1000.0,
        "max_us": max_ms * 1000.0,
        "samples_us": [sample * 1000.0 for sample in samples_ms],
    }


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    batch_sizes = [int(part) for part in args.batch_sizes.split(",") if part]
    results = [_run_case(args, batch_size=batch_size) for batch_size in batch_sizes]
    for result in results:
        print(
            f"B={result['batch_size']}: "
            f"{result['mean_us']:.3f} us/step "
            f"[{result['min_us']:.3f}, {result['max_us']:.3f}]"
        )
    payload = {
        "kind": "bench_mixer_step",
        "schema_version": 1,
        "dtype": args.dtype,
        "device": args.device,
        "d_model": args.d_model,
        "d_state": args.d_state,
        "expand": args.expand,
        "d_head": args.d_head,
        "d_conv": args.d_conv,
        "chunk_size": args.chunk_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeat": args.repeat,
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
