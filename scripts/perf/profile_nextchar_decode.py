#!/usr/bin/env python3
"""Profile steady-state nextchar decode."""

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

from _common import dtype_from_str, ensure_cuda  # noqa: E402
from _nextchar import NextCharPerfConfig  # noqa: E402
from _nextchar_decode import profile_decode_trace  # noqa: E402
from slinoss.perf.schema import validate_nextchar_decode_profile_payload  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cute", "reference"), default="cute")
    parser.add_argument("--mode", choices=("persistent", "eager"), default="persistent")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--active-tokens", type=int, default=64)
    parser.add_argument("--sort-by", default="self_cuda_time_total")
    parser.add_argument("--top-k", type=int, default=40)
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
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    result = profile_decode_trace(
        _make_cfg(args),
        backend=args.backend,
        mode=args.mode,
        warmup_tokens=args.warmup_tokens,
        active_tokens=args.active_tokens,
        trace_out=None if args.trace_out is None else str(args.trace_out),
        sort_by=args.sort_by,
        top_k=args.top_k,
    )
    print(result["table"])
    payload = {
        "kind": "profile_nextchar_decode",
        "schema_version": 1,
        "backend": args.backend,
        "mode": args.mode,
        "config": _make_cfg(args).perf_config_dict,
        "warmup_tokens": args.warmup_tokens,
        "active_tokens": args.active_tokens,
        "trace_out": None if args.trace_out is None else str(args.trace_out),
    }
    validate_nextchar_decode_profile_payload(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")
    if args.trace_out is not None:
        print(f"trace: {args.trace_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
