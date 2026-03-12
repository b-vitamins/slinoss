#!/usr/bin/env python3
"""Compare two nextchar bench payloads and rank perf deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slinoss.perf.compare import compare_budget_trees, rank_budget_deltas  # noqa: E402
from slinoss.perf.schema import validate_nextchar_bench_payload  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--backend", choices=("reference", "cute"), default="cute")
    parser.add_argument("--case", default="default")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    validate_nextchar_bench_payload(payload)
    return payload


def _case_workload(
    payload: dict[str, Any], *, case: str, backend: str
) -> dict[str, Any]:
    cases = payload["cases"]
    if case not in cases:
        raise ValueError(f"Missing case {case!r}. Available: {sorted(cases)}")
    workloads = cases[case]["workload"]
    if backend not in workloads:
        raise ValueError(
            f"Missing backend {backend!r} in case {case!r}. Available: {sorted(workloads)}"
        )
    return workloads[backend]


def main() -> int:
    args = _parse_args()
    before = _load_payload(args.before)
    after = _load_payload(args.after)
    before_workload = _case_workload(before, case=args.case, backend=args.backend)
    after_workload = _case_workload(after, case=args.case, backend=args.backend)

    before_tree = before_workload["warm"]["tree"]
    after_tree = after_workload["warm"]["tree"]
    rows = compare_budget_trees(before_tree, after_tree)
    ranked = rank_budget_deltas(rows, top_k=args.top_k)

    before_step = float(before_workload["warm"]["step"]["mean_ms"])
    after_step = float(after_workload["warm"]["step"]["mean_ms"])
    before_tps = float(before_workload["warm"]["tokens_per_s"]["mean"])
    after_tps = float(after_workload["warm"]["tokens_per_s"]["mean"])

    print(f"backend={args.backend} case={args.case}")
    print(
        "step_mean_ms "
        f"{before_step:.6f} -> {after_step:.6f}  "
        f"delta={after_step - before_step:+.6f}"
    )
    print(
        "tokens_per_s "
        f"{before_tps:.2f} -> {after_tps:.2f}  "
        f"delta={after_tps - before_tps:+.2f}"
    )

    print("\nTop Regressions")
    for row in ranked["regressions"]:
        print(
            f"{row['label']}: {row['before_ms']:.6f} -> {row['after_ms']:.6f} "
            f"({row['delta_ms']:+.6f} ms, {row['delta_pct']:+.2f}%)"
        )

    print("\nTop Improvements")
    for row in ranked["improvements"]:
        print(
            f"{row['label']}: {row['before_ms']:.6f} -> {row['after_ms']:.6f} "
            f"({row['delta_ms']:+.6f} ms, {row['delta_pct']:+.2f}%)"
        )

    payload = {
        "kind": "compare_nextchar_perf",
        "schema_version": 1,
        "backend": args.backend,
        "case": args.case,
        "before": str(args.before),
        "after": str(args.after),
        "step_mean_ms": {
            "before": before_step,
            "after": after_step,
            "delta": after_step - before_step,
        },
        "tokens_per_s": {
            "before": before_tps,
            "after": after_tps,
            "delta": after_tps - before_tps,
        },
        "ranked": ranked,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\njson: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
