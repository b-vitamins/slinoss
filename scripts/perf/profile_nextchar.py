#!/usr/bin/env python3
"""Profile nextchar-style training with aligned budget labels."""

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

from _common import dtype_from_str, ensure_cuda, seed_all  # noqa: E402
from _nextchar import (  # noqa: E402
    NextCharPerfConfig,
    build_model,
    attach_workload_timers,
    random_batch,
    run_train_step,
)
from slinoss.perf.budget import (  # noqa: E402
    build_tree,
    summarize_budget_samples,
    summarize_named_samples,
)
from slinoss.perf.schema import validate_nextchar_profile_payload  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "cute"), default="cute")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--sort-by", default="self_cuda_time_total")
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _make_cfg(args: argparse.Namespace) -> NextCharPerfConfig:
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


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    seed_all(args.seed)

    cfg = _make_cfg(args)
    model, optimizer = build_model(cfg, backend=args.backend)
    handles = attach_workload_timers(model)
    captures: list[dict[str, Any]] = []
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(
            wait=0, warmup=args.warmup, active=args.active, repeat=1
        )
        total_steps = int(args.warmup) + int(args.active)
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            acc_events=True,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(total_steps):
                xb, yb = random_batch(cfg)
                from slinoss.perf import PerfRecorder

                recorder = PerfRecorder(device=cfg.torch_device)
                with recorder.capture_step():
                    run_train_step(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)
                captures.append(recorder.steps[-1])
                prof.step()
        if args.trace_out is not None:
            args.trace_out.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(args.trace_out))
        active_captures = captures[-int(args.active) :]
        region_samples = [capture["regions_ms"] for capture in active_captures]
        summaries = summarize_named_samples(region_samples)
        budget = summarize_budget_samples(region_samples)
        tree = build_tree(budget)

        print(
            prof.key_averages().table(
                sort_by=args.sort_by,
                row_limit=args.top_k,
            )
        )
        payload = {
            "kind": "profile_nextchar",
            "schema_version": 1,
            "backend": args.backend,
            "config": cfg.perf_config_dict,
            "regions": summaries,
            "budget": budget,
            "tree": tree,
            "trace_out": None if args.trace_out is None else str(args.trace_out),
        }
        validate_nextchar_profile_payload(payload)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            print(f"json: {args.json_out}")
        if args.trace_out is not None:
            print(f"trace: {args.trace_out}")
    finally:
        for handle in handles:
            handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
