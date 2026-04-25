#!/usr/bin/env python3
"""Benchmark block-based training throughput and logical op budgets."""

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

from _common import (  # noqa: E402
    PerfConfig,
    benchmark_instrumented,
    build_v2x2ssd_callable,
    dtype_from_str,
    ensure_cuda,
    seed_all,
)
from _training import (  # noqa: E402
    DEFAULT_TRAINING_PERF_CONFIG,
    TrainingBenchFixture,
    TrainingPerfConfig,
    build_bench_fixture,
    run_bench_step,
)
from slinoss.perf.budget import (  # noqa: E402
    build_tree,
    summarize_budget_samples,
    summarize_cache_samples,
    summarize_named_samples,
    summarize_scalar_samples,
)
from slinoss.perf.schema import validate_training_bench_payload  # noqa: E402


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
    default_cfg = DEFAULT_TRAINING_PERF_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cute",),
        default="cute",
    )
    parser.add_argument("--batch-size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--seq-len", type=int, default=default_cfg.seq_len)
    parser.add_argument("--vocab-size", type=int, default=default_cfg.vocab_size)
    parser.add_argument("--d-model", type=int, default=default_cfg.d_model)
    parser.add_argument("--n-layers", type=int, default=default_cfg.n_layers)
    parser.add_argument("--d-state", type=int, default=default_cfg.mixer.d_state)
    parser.add_argument("--expand", type=float, default=default_cfg.mixer.expand)
    parser.add_argument("--d-head", type=int, default=default_cfg.mixer.d_head)
    parser.add_argument("--d-conv", type=int, default=default_cfg.mixer.d_conv)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=default_cfg.mixer.chunk_size,
    )
    parser.add_argument(
        "--bc-groups",
        type=int,
        default=default_cfg.mixer.bc_groups,
    )
    parser.add_argument(
        "--ffn-expand",
        type=float,
        default=(0.0 if default_cfg.ffn is None else default_cfg.ffn.expand),
    )
    parser.add_argument("--lr", type=float, default=default_cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=default_cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=default_cfg.grad_clip)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
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
    parser.add_argument("--v2-warmup", type=int, default=2)
    parser.add_argument("--v2-iterations", type=int, default=8)
    parser.add_argument("--v2-repeat", type=int, default=3)
    parser.add_argument(
        "--suite",
        choices=("single", "training"),
        default="single",
        help="single=current shape only, training=default + tail-batch cases",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _make_training_cfg(args: argparse.Namespace) -> TrainingPerfConfig:
    ffn = None
    default_ffn = DEFAULT_TRAINING_PERF_CONFIG.ffn
    if float(args.ffn_expand) > 0.0 and default_ffn is not None:
        ffn = replace(
            default_ffn,
            expand=float(args.ffn_expand),
        )
    return TrainingPerfConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        mixer=replace(
            DEFAULT_TRAINING_PERF_CONFIG.mixer,
            d_state=args.d_state,
            expand=args.expand,
            d_head=args.d_head,
            d_conv=args.d_conv,
            chunk_size=args.chunk_size,
            bc_groups=args.bc_groups,
        ),
        ffn=ffn,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )


def _make_v2_cfg(cfg: TrainingPerfConfig) -> PerfConfig:
    return PerfConfig(
        batch=cfg.batch_size,
        heads=cfg.n_heads,
        T=cfg.seq_len,
        N=cfg.mixer.d_state,
        P=cfg.mixer.d_head,
        chunk_size=cfg.mixer.chunk_size,
        dtype=cfg.dtype,
        device=cfg.device,
        seed=cfg.seed,
    )


def _make_case_cfgs(
    cfg: TrainingPerfConfig,
    *,
    suite: str,
) -> dict[str, TrainingPerfConfig]:
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
    cfg: TrainingPerfConfig,
    *,
    backend: str,
    warmup_steps: int,
    steps: int,
    repeat: int,
    fixture: TrainingBenchFixture,
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
            "warm_execution": str(result["warm_execution_mode"]),
            "profile_execution": "eager_single_post_bench_replay",
            "memory_measurement": "bench_path_step_peaks",
            "memory_forensics": "use profile_training_memory.py for eager attribution",
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


def _summarize_v2x2ssd_suite(
    v2_cfg: PerfConfig,
    *,
    backend_choice: str,
    warmup: int,
    iterations: int,
    repeat: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    backends = (backend_choice,)
    for direction in ("forward", "backward"):
        for backend in backends:
            fn = build_v2x2ssd_callable(
                v2_cfg,
                direction=direction,
                backend=backend,
            )
            stats = benchmark_instrumented(
                fn,
                device=v2_cfg.torch_device,
                warmup=warmup,
                iterations=iterations,
                repeat=repeat,
            )
            rows.append(
                {
                    "direction": direction,
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
            "batch": v2_cfg.batch,
            "heads": v2_cfg.heads,
            "T": v2_cfg.T,
            "N": v2_cfg.N,
            "P": v2_cfg.P,
            "chunk_size": v2_cfg.chunk_size,
            "dtype": str(v2_cfg.dtype),
            "device": v2_cfg.device,
        },
        "rows": rows,
    }


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    seed_all(args.seed)

    training_cfg = _make_training_cfg(args)
    backends = (args.backend,)

    cases: dict[str, dict[str, Any]] = {}
    for case_name, case_cfg in _make_case_cfgs(training_cfg, suite=args.suite).items():
        v2_cfg = _make_v2_cfg(case_cfg)
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
        v2x2ssd_suite = _summarize_v2x2ssd_suite(
            v2_cfg,
            backend_choice=args.backend,
            warmup=args.v2_warmup,
            iterations=args.v2_iterations,
            repeat=args.v2_repeat,
        )
        cases[case_name] = {
            "config": case_cfg.perf_config_dict,
            "workload": workload,
            "v2x2ssd_suite": v2x2ssd_suite,
        }

    payload = {
        "kind": "bench_training",
        "schema_version": 2,
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "suite": args.suite,
        "cases": cases,
    }
    validate_training_bench_payload(payload)

    for case_name, case_payload in cases.items():
        for backend in backends:
            warm = case_payload["workload"][backend]["warm"]
            step_mean = float(warm["step"]["mean_ms"])
            tps_mean = float(warm["tokens_per_s"]["mean"])
            peak_alloc_mib = _bytes_to_mib(
                float(warm["memory"]["peak_allocated_bytes"]["mean_bytes"])
            )
            peak_reserved_mib = _bytes_to_mib(
                float(warm["memory"]["peak_reserved_bytes"]["mean_bytes"])
            )
            print(
                f"{case_name}/{backend}: step_mean_ms={step_mean:.6f} "
                f"tokens_per_s={tps_mean:.2f} "
                f"peak_allocated_mib_primary={peak_alloc_mib:.2f} "
                f"peak_reserved_mib_context={peak_reserved_mib:.2f}"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
