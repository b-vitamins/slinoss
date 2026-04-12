#!/usr/bin/env python3
"""Benchmark nextchar-style training with end-to-end and stage/kernel budgets."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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

from slinoss.perf.budget import (  # noqa: E402
    build_tree,
    summarize_budget_samples,
    summarize_cache_samples,
    summarize_named_samples,
    summarize_scalar_samples,
)
from _common import (  # noqa: E402
    benchmark_instrumented,
    build_callable,
    dtype_from_str,
    ensure_cuda,
    seed_all,
    PerfConfig,
)
from _nextchar import (  # noqa: E402
    DEFAULT_NEXTCHAR_PERF_CONFIG,
    NextCharBenchFixture,
    NextCharPerfConfig,
    build_bench_fixture,
    run_bench_step,
)
from slinoss.perf.schema import validate_nextchar_bench_payload  # noqa: E402


def _summarize_byte_samples(samples: list[float]) -> dict[str, float]:
    return {
        key.replace("_ms", "_bytes"): value
        for key, value in summarize_scalar_samples(samples).items()
    }


def _summarize_memory_samples(
    samples: list[dict[str, int]],
) -> dict[str, dict[str, float]]:
    labels = ("peak_allocated_bytes", "peak_reserved_bytes")
    return {
        label: _summarize_byte_samples(
            [float(sample.get(label, 0)) for sample in samples]
        )
        for label in labels
    }


def _bytes_to_mib(num_bytes: float) -> float:
    return float(num_bytes) / float(1024**2)


def _parse_args() -> argparse.Namespace:
    default_cfg = DEFAULT_NEXTCHAR_PERF_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("reference", "cute", "both"), default="both"
    )
    parser.add_argument("--batch-size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--block-size", type=int, default=default_cfg.block_size)
    parser.add_argument("--vocab-size", type=int, default=default_cfg.vocab_size)
    parser.add_argument("--d-model", type=int, default=default_cfg.d_model)
    parser.add_argument("--n-layers", type=int, default=default_cfg.n_layers)
    parser.add_argument("--d-state", type=int, default=default_cfg.d_state)
    parser.add_argument("--expand", type=float, default=default_cfg.expand)
    parser.add_argument("--d-head", type=int, default=default_cfg.d_head)
    parser.add_argument("--d-conv", type=int, default=default_cfg.d_conv)
    parser.add_argument("--chunk-size", type=int, default=default_cfg.chunk_size)
    parser.add_argument("--lr", type=float, default=default_cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=default_cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=default_cfg.grad_clip)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default=default_cfg.device)
    parser.add_argument("--seed", type=int, default=default_cfg.seed)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--workload-repeat",
        type=int,
        default=3,
        help="Independent clean end-to-end repeats per backend/case.",
    )
    parser.add_argument("--stage-warmup", type=int, default=2)
    parser.add_argument("--stage-iterations", type=int, default=8)
    parser.add_argument("--stage-repeat", type=int, default=3)
    parser.add_argument(
        "--suite",
        choices=("single", "training"),
        default="single",
        help="single=current shape only, training=default + tail-batch cases",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _make_nextchar_cfg(args: argparse.Namespace) -> NextCharPerfConfig:
    return NextCharPerfConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        d_head=args.d_head,
        d_conv=args.d_conv,
        chunk_size=args.chunk_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )


def _make_stage_cfg(cfg: NextCharPerfConfig) -> PerfConfig:
    return PerfConfig(
        batch=cfg.batch_size,
        heads=cfg.n_heads,
        T=cfg.block_size,
        N=cfg.d_state,
        P=cfg.d_head,
        chunk_size=cfg.chunk_size,
        dtype=cfg.dtype,
        device=cfg.device,
        seed=cfg.seed,
    )


def _make_case_cfgs(
    cfg: NextCharPerfConfig,
    *,
    suite: str,
) -> dict[str, NextCharPerfConfig]:
    if suite == "single":
        return {"default": cfg}
    if suite == "training":
        half_batch = max(1, cfg.batch_size // 2)
        return {
            "default": cfg,
            "tail_half": replace(cfg, batch_size=half_batch),
            "tail_one": replace(cfg, batch_size=1),
        }
    raise ValueError(f"Unsupported suite: {suite}")


def _summarize_workload(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    warmup_steps: int,
    steps: int,
    repeat: int,
    fixture: NextCharBenchFixture,
) -> dict[str, object]:
    result = run_bench_step(
        cfg,
        backend=backend,
        warmup=warmup_steps,
        steps=steps,
        repeat=repeat,
        fixture=fixture,
    )
    cold = result["cold_profile"]
    warm_steps = result["warm_profile"]
    tokens_per_step = int(result["tokens_per_step"])
    cold_memory = dict(result["cold_memory"])
    warm_memory = list(result["warm_memory"])
    step_total_samples = [float(ms) for ms in result["warm_step_ms"]]
    repeat_step_samples = [float(ms) for ms in result["warm_repeat_step_mean_ms"]]

    warm_region_samples = [step["regions_ms"] for step in warm_steps]
    warm_cache_samples = [step["cache_events"] for step in warm_steps]

    warm_regions = summarize_named_samples(warm_region_samples)
    warm_budget = summarize_budget_samples(warm_region_samples)
    warm_tree = build_tree(warm_budget)
    tokens_per_s_samples = [
        (1000.0 * tokens_per_step / ms) if ms > 0.0 else 0.0
        for ms in step_total_samples
    ]
    repeat_tokens_per_s_samples = [
        (1000.0 * tokens_per_step / ms) if ms > 0.0 else 0.0
        for ms in repeat_step_samples
    ]
    tokens_per_s_stats = {
        key.replace("_ms", ""): value
        for key, value in summarize_scalar_samples(tokens_per_s_samples).items()
    }
    repeat_tokens_per_s_stats = {
        key.replace("_ms", ""): value
        for key, value in summarize_scalar_samples(repeat_tokens_per_s_samples).items()
    }

    cold_budget = summarize_budget_samples([cold["regions_ms"]])
    cold_tree = build_tree(cold_budget)

    return {
        "backend": backend,
        "config": cfg.perf_config_dict,
        "tokens_per_step": tokens_per_step,
        "methodology": {
            "timing": "cuda_event_per_step",
            "deterministic_fixture": True,
            "fixture_model_seed": int(result["fixture"]["model_seed"]),
            "fixture_batch_seed": int(result["fixture"]["batch_seed"]),
            "batch_count": int(result["fixture"]["batch_count"]),
            "warmup_steps": int(warmup_steps),
            "steps_per_repeat": int(steps),
            "workload_repeat": int(result["repeat_count"]),
            "profile_replay": "single_post_clean_replay",
        },
        "cold": {
            "regions": summarize_named_samples([cold["regions_ms"]]),
            "budget": cold_budget,
            "tree": cold_tree,
            "cache_events": summarize_cache_samples([cold["cache_events"]]),
            "memory": _summarize_memory_samples([cold_memory]),
        },
        "warm": {
            "step": summarize_scalar_samples(step_total_samples),
            "repeat_step": summarize_scalar_samples(repeat_step_samples),
            "tokens_per_s": tokens_per_s_stats,
            "repeat_tokens_per_s": repeat_tokens_per_s_stats,
            "regions": warm_regions,
            "budget": warm_budget,
            "tree": warm_tree,
            "cache_events": summarize_cache_samples(warm_cache_samples),
            "memory": _summarize_memory_samples(warm_memory),
        },
    }


def _summarize_stage_suite(
    stage_cfg: PerfConfig,
    *,
    backend_choice: str,
    warmup: int,
    iterations: int,
    repeat: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    backends = ("reference", "cute") if backend_choice == "both" else (backend_choice,)
    for direction in ("forward", "backward"):
        for stage in ("chunk_increment", "state_passing", "chunk_scan"):
            for backend in backends:
                fn = build_callable(
                    stage_cfg,
                    stage=stage,
                    direction=direction,
                    backend=backend,
                )
                stats = benchmark_instrumented(
                    fn,
                    device=stage_cfg.torch_device,
                    warmup=warmup,
                    iterations=iterations,
                    repeat=repeat,
                )
                rows.append(
                    {
                        "direction": direction,
                        "stage": stage,
                        "backend": backend,
                        "summary": {
                            key: value
                            for key, value in stats.items()
                            if key
                            not in {
                                "samples_ms",
                                "region_samples",
                                "region_summaries",
                                "cache_events",
                            }
                        },
                        "regions": stats["region_summaries"],
                        "cache_events": stats["cache_events"],
                    }
                )
    return {
        "config": {
            "batch": stage_cfg.batch,
            "heads": stage_cfg.heads,
            "T": stage_cfg.T,
            "N": stage_cfg.N,
            "P": stage_cfg.P,
            "chunk_size": stage_cfg.chunk_size,
            "dtype": str(stage_cfg.dtype),
            "device": stage_cfg.device,
        },
        "rows": rows,
    }


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    seed_all(args.seed)

    nextchar_cfg = _make_nextchar_cfg(args)
    backends = ("reference", "cute") if args.backend == "both" else (args.backend,)

    cases: dict[str, dict[str, Any]] = {}
    for case_name, case_cfg in _make_case_cfgs(nextchar_cfg, suite=args.suite).items():
        stage_cfg = _make_stage_cfg(case_cfg)
        fixture = build_bench_fixture(
            case_cfg,
            total_batches=1 + int(args.warmup_steps) + int(args.steps),
        )
        workload: dict[str, Any] = {
            backend: _summarize_workload(
                case_cfg,
                backend=backend,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
                repeat=args.workload_repeat,
                fixture=fixture,
            )
            for backend in backends
        }
        stage_suite = _summarize_stage_suite(
            stage_cfg,
            backend_choice=args.backend,
            warmup=args.stage_warmup,
            iterations=args.stage_iterations,
            repeat=args.stage_repeat,
        )
        cases[case_name] = {
            "config": case_cfg.perf_config_dict,
            "workload": workload,
            "stage_suite": stage_suite,
        }

    payload = {
        "kind": "bench_nextchar",
        "schema_version": 1,
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "suite": args.suite,
        "cases": cases,
    }
    validate_nextchar_bench_payload(payload)

    for case_name, case_payload in cases.items():
        for backend in backends:
            warm = case_payload["workload"][backend]["warm"]
            step_mean = float(warm["step"]["mean_ms"])
            tps_mean = float(warm["tokens_per_s"]["mean"])
            peak_alloc_mib = _bytes_to_mib(
                float(warm["memory"]["peak_allocated_bytes"]["mean_bytes"])
            )
            peak_res_mib = _bytes_to_mib(
                float(warm["memory"]["peak_reserved_bytes"]["mean_bytes"])
            )
            print(
                f"{case_name}/{backend}: step_mean_ms={step_mean:.6f} "
                f"tokens_per_s={tps_mean:.2f} "
                f"peak_allocated_mib={peak_alloc_mib:.2f} "
                f"peak_reserved_mib={peak_res_mib:.2f}"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
