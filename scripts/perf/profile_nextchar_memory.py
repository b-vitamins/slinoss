#!/usr/bin/env python3
"""Run eager memory forensics for one nextchar training step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import dtype_from_str, ensure_cuda, seed_all  # noqa: E402
from _nextchar import (  # noqa: E402
    DEFAULT_NEXTCHAR_PERF_CONFIG,
    NextCharPerfConfig,
    build_model,
    random_batch,
    run_train_step_profiled,
)
from slinoss.perf import (  # noqa: E402
    EagerMemoryForensics,
    PerfRecorder,
    allocator_snapshot_metadata,
    current_memory_stats,
    peak_memory_stats,
    reset_peak_memory_stats,
)
from slinoss.perf.budget import (  # noqa: E402
    build_tree,
    summarize_budget_samples,
    summarize_named_samples,
)
from slinoss.perf.schema import validate_nextchar_memory_payload  # noqa: E402


def _parse_args() -> argparse.Namespace:
    default_cfg = DEFAULT_NEXTCHAR_PERF_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "cute"), default="cute")
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
    parser.add_argument("--bc-groups", type=int, default=default_cfg.bc_groups)
    parser.add_argument("--lr", type=float, default=default_cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=default_cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=default_cfg.grad_clip)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default=default_cfg.device)
    parser.add_argument("--seed", type=int, default=default_cfg.seed)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Eager warmup steps before the measured step.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--allocator-snapshot-out", type=Path, default=None)
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
        bc_groups=args.bc_groups,
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
    model, optimizer = build_model(cfg, backend=args.backend, instrumented=True)

    for _ in range(max(0, int(args.warmup_steps))):
        xb, yb = random_batch(cfg)
        run_train_step_profiled(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)

    recorder = PerfRecorder(device=cfg.torch_device)
    forensics = EagerMemoryForensics(device=cfg.torch_device)
    allocator_snapshot, snapshot_ctx = allocator_snapshot_metadata(
        device=cfg.torch_device,
        out_path=args.allocator_snapshot_out,
    )

    xb, yb = random_batch(cfg)
    baseline_memory = current_memory_stats(cfg.torch_device)
    reset_peak_memory_stats(cfg.torch_device)
    with snapshot_ctx:
        with forensics.capture():
            with recorder.capture_step():
                run_train_step_profiled(
                    model,
                    optimizer,
                    xb,
                    yb,
                    grad_clip=cfg.grad_clip,
                )

    capture = recorder.steps[-1]
    region_samples = [capture["regions_ms"]]
    step_memory = {
        **peak_memory_stats(cfg.torch_device),
        **{
            f"end_{key}": value
            for key, value in current_memory_stats(cfg.torch_device).items()
        },
    }
    payload = {
        "kind": "profile_nextchar_memory",
        "schema_version": 1,
        "backend": args.backend,
        "config": cfg.perf_config_dict,
        "methodology": {
            "execution": "eager_training_step",
            "baseline_scope": "warmed_model_plus_inputs",
            "warmup_steps": int(args.warmup_steps),
            "top_k": int(args.top_k),
            "memory_metric_primary": "peak_allocated_bytes",
            "allocator_snapshot_requested": bool(args.allocator_snapshot_out),
        },
        "baseline_memory": baseline_memory,
        "step_memory": step_memory,
        "regions": summarize_named_samples(region_samples),
        "budget": summarize_budget_samples(region_samples),
        "tree": build_tree(summarize_budget_samples(region_samples)),
        "cache_events": capture["cache_events"],
        "top_region_exit_allocated": forensics.top_region_exit_allocated(
            top_k=args.top_k
        ),
        "saved_tensors_by_region": forensics.saved_tensors_by_region(),
        "saved_tensors_summary": forensics.saved_tensors_summary(),
        "allocator_snapshot": allocator_snapshot,
    }
    validate_nextchar_memory_payload(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")
    if bool(payload["allocator_snapshot"]["captured"]):
        print(f"allocator_snapshot: {args.allocator_snapshot_out}")
    elif args.allocator_snapshot_out is not None:
        print(
            "allocator_snapshot: unavailable "
            f"({payload['allocator_snapshot'].get('reason', 'capture_failed')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
