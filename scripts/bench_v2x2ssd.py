#!/usr/bin/env python3
"""Benchmark the staged v2x2ssd forward path.

This is a development benchmark, not an experiment harness. It compares the
reference and CuTe forward paths, or any individual forward stage, on the
canonical v2x2ssd contract.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import cast

import torch


def _load_ops() -> dict[str, Callable]:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute
    from slinoss.ops.v2x2ssd.cute.kernels.fwd import (
        chunk_increment_cute,
        chunk_scan_cute,
        state_passing_cute,
    )
    from slinoss.ops.v2x2ssd.reference import (
        chunk_increment as ref_chunk_increment,
    )
    from slinoss.ops.v2x2ssd.reference import chunk_scan as ref_chunk_scan
    from slinoss.ops.v2x2ssd.reference import state_passing as ref_state_passing

    return {
        "v2x2ssd": v2x2ssd,
        "v2x2ssd_cute": v2x2ssd_cute,
        "chunk_increment_cute": chunk_increment_cute,
        "chunk_scan_cute": chunk_scan_cute,
        "state_passing_cute": state_passing_cute,
        "ref_chunk_increment": ref_chunk_increment,
        "ref_chunk_scan": ref_chunk_scan,
        "ref_state_passing": ref_state_passing,
    }


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _time_once(fn, iterations: int) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / max(1, iterations))

    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    ended = time.perf_counter()
    return (ended - started) * 1000.0 / max(1, iterations)


def _benchmark(
    fn, *, warmup: int, iterations: int, repeat: int
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    samples = [_time_once(fn, iterations) for _ in range(repeat)]
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _make_inputs(
    args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor]:
    radius = 0.6 + 0.35 * torch.rand((args.batch, args.heads, args.T), device=device)
    angle = (2.0 * math.pi) * torch.rand(
        (args.batch, args.heads, args.T), device=device
    ) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn(
            (args.batch, args.heads, args.T, 2), device=device, dtype=torch.float32
        )
        + 1j
        * torch.randn(
            (args.batch, args.heads, args.T, 2), device=device, dtype=torch.float32
        )
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn(
        (args.batch, args.heads, args.T, args.P), device=device, dtype=torch.float32
    )
    B = (
        torch.randn(
            (args.batch, args.heads, args.T, 2 * args.N),
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    )
    C = (
        torch.randn(
            (args.batch, args.heads, args.T, 2 * args.N),
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    )
    initial_states = torch.randn(
        (args.batch, args.heads, args.P, 2 * args.N),
        device=device,
        dtype=torch.float32,
    )
    b_prev = (
        torch.randn(
            (args.batch, args.heads, args.N), device=device, dtype=torch.float32
        )
        + 1j
        * torch.randn(
            (args.batch, args.heads, args.N), device=device, dtype=torch.float32
        )
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=torch.float32)
    U_prev = torch.randn(
        (args.batch, args.heads, args.P), device=device, dtype=torch.float32
    )
    return {
        "U": U,
        "M": M,
        "K": K,
        "B": B,
        "C": C,
        "initial_states": initial_states,
        "B_prev": B_prev,
        "U_prev": U_prev,
    }


def _build_stage_callable(args: argparse.Namespace, backend: str):
    ops = _load_ops()
    inputs = _make_inputs(args, torch.device(args.device))
    U = inputs["U"]
    M = inputs["M"]
    K = inputs["K"]
    B = inputs["B"]
    C = inputs["C"]
    initial_states = inputs["initial_states"]
    B_prev = inputs["B_prev"]
    U_prev = inputs["U_prev"]
    chunk_size = args.chunk_size

    if args.stage == "full":
        if backend == "reference":
            fn = partial(
                ops["v2x2ssd"],
                U,
                M,
                K,
                B,
                C,
                chunk_size=chunk_size,
                initial_states=initial_states,
                B_prev=B_prev,
                U_prev=U_prev,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        else:
            fn = partial(
                ops["v2x2ssd_cute"],
                U,
                M,
                K,
                B,
                C,
                chunk_size=chunk_size,
                initial_states=initial_states,
                B_prev=B_prev,
                U_prev=U_prev,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        fn()
        return fn

    if args.stage == "chunk_increment":
        if backend == "reference":
            fn = partial(
                ops["ref_chunk_increment"],
                U,
                M,
                K,
                B,
                B_prev=B_prev,
                U_prev=U_prev,
                T=args.T,
                chunk_size=chunk_size,
                compute_dtype=torch.float32,
            )
        else:
            fn = partial(
                ops["chunk_increment_cute"],
                U,
                M,
                K,
                B,
                chunk_size=chunk_size,
                B_prev=B_prev,
                U_prev=U_prev,
                compute_dtype=torch.float32,
            )
        fn()
        return fn

    inc_ref, m_ref = ops["ref_chunk_increment"](
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=args.T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    inc_cute, m_cute = ops["chunk_increment_cute"](
        U,
        M,
        K,
        B,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    if args.stage == "state_passing":
        if backend == "reference":
            fn = partial(
                ops["ref_state_passing"],
                inc_ref,
                m_ref,
                initial_states=initial_states,
                compute_dtype=torch.float32,
            )
        else:
            fn = partial(
                ops["state_passing_cute"],
                inc_cute,
                m_cute,
                initial_states=initial_states.to(dtype=torch.float32).contiguous(),
            )
        fn()
        return fn

    starts_ref, _ = ops["ref_state_passing"](
        inc_ref,
        m_ref,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )
    starts_cute, _ = ops["state_passing_cute"](
        inc_cute,
        m_cute,
        initial_states=initial_states.to(dtype=torch.float32).contiguous(),
    )

    if backend == "reference":
        fn = partial(
            ops["ref_chunk_scan"],
            U,
            M,
            K,
            B,
            C,
            starts_ref,
            B_prev=B_prev,
            U_prev=U_prev,
            T=args.T,
            chunk_size=chunk_size,
            output_dtype=torch.float32,
            compute_dtype=torch.float32,
        )
    else:
        fn = partial(
            ops["chunk_scan_cute"],
            U,
            M,
            K,
            B,
            C,
            starts_cute,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            output_dtype=torch.float32,
            compute_dtype=torch.float32,
        )
    fn()
    return fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("full", "chunk_increment", "state_passing", "chunk_scan"),
        default="full",
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "cute", "both"),
        default="both",
    )
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--P", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    backends = ("reference", "cute") if args.backend == "both" else (args.backend,)
    rows: list[dict[str, float | str | list[float]]] = []
    for backend in backends:
        fn = _build_stage_callable(args, backend)
        stats = _benchmark(
            fn,
            warmup=args.warmup,
            iterations=args.iterations,
            repeat=args.repeat,
        )
        row = {
            "backend": backend,
            "stage": args.stage,
            "batch": args.batch,
            "heads": args.heads,
            "T": args.T,
            "N": args.N,
            "P": args.P,
            "chunk_size": args.chunk_size,
            **stats,
        }
        rows.append(row)

    print("| backend | stage | mean_ms | median_ms | stdev_ms |")
    print("| --- | --- | ---: | ---: | ---: |")
    for row in rows:
        mean_ms = cast(float, row["mean_ms"])
        median_ms = cast(float, row["median_ms"])
        stdev_ms = cast(float, row["stdev_ms"])
        print(
            f"| {row['backend']} | {row['stage']} | "
            f"{mean_ms:.6f} | {median_ms:.6f} | {stdev_ms:.6f} |"
        )
    if len(rows) == 2:
        speedup = cast(float, rows[0]["mean_ms"]) / cast(float, rows[1]["mean_ms"])
        print(
            f"\nspeedup ({rows[0]['backend']} / {rows[1]['backend']}): {speedup:.3f}x"
        )

    if args.json_out is not None:
        payload = {
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
