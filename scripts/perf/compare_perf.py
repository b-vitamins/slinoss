#!/usr/bin/env python3
"""Compare two training bench payloads and rank perf deltas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slinoss.perf.compare import compare_budget_trees, rank_budget_deltas  # noqa: E402
from slinoss.perf.schema import validate_training_bench_payload  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--backend", choices=("cute",), default="cute")
    parser.add_argument("--case", default="default")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    validate_training_bench_payload(payload)
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


def _select_step_summary(workload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    warm = workload["warm"]
    repeat_summary = warm.get("repeat_step")
    if isinstance(repeat_summary, dict):
        return repeat_summary, "repeat_step"
    return warm["step"], "step"


def _select_tps_summary(workload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    warm = workload["warm"]
    repeat_summary = warm.get("repeat_tokens_per_s")
    if isinstance(repeat_summary, dict):
        return repeat_summary, "repeat_tokens_per_s"
    return warm["tokens_per_s"], "tokens_per_s"


def _select_memory_summary(
    workload: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any] | None:
    warm = workload.get("warm")
    if not isinstance(warm, dict):
        return None
    memory = warm.get("memory")
    if not isinstance(memory, dict):
        return None
    summary = memory.get(label)
    return summary if isinstance(summary, dict) else None


def _bytes_to_mib(num_bytes: float) -> float:
    return float(num_bytes) / float(1024**2)


def _ci95_halfwidth(summary: dict[str, Any], *, key: str) -> float | None:
    count = int(float(summary.get("num_samples", 0.0)))
    if count <= 1:
        return None
    stdev = float(summary.get(key, 0.0))
    return 1.96 * stdev / math.sqrt(count)


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

    before_step_summary, before_step_basis = _select_step_summary(before_workload)
    after_step_summary, after_step_basis = _select_step_summary(after_workload)
    before_tps_summary, before_tps_basis = _select_tps_summary(before_workload)
    after_tps_summary, after_tps_basis = _select_tps_summary(after_workload)
    before_step = float(before_step_summary["mean_ms"])
    after_step = float(after_step_summary["mean_ms"])
    before_tps = float(before_tps_summary["mean"])
    after_tps = float(after_tps_summary["mean"])
    before_step_ci = _ci95_halfwidth(before_step_summary, key="stdev_ms")
    after_step_ci = _ci95_halfwidth(after_step_summary, key="stdev_ms")
    before_tps_ci = _ci95_halfwidth(before_tps_summary, key="stdev")
    after_tps_ci = _ci95_halfwidth(after_tps_summary, key="stdev")

    print(f"backend={args.backend} case={args.case}")
    print(
        (
            f"step_mean_ms[{before_step_basis}] "
            if before_step_basis == after_step_basis
            else f"step_mean_ms[{before_step_basis}->{after_step_basis}] "
        )
        + f"{before_step:.6f}"
        + (f" +- {before_step_ci:.6f}" if before_step_ci is not None else "")
        + f" -> {after_step:.6f}"
        + (f" +- {after_step_ci:.6f}" if after_step_ci is not None else "")
        + "  "
        f"delta={after_step - before_step:+.6f}"
    )
    print(
        (
            f"tokens_per_s[{before_tps_basis}] "
            if before_tps_basis == after_tps_basis
            else f"tokens_per_s[{before_tps_basis}->{after_tps_basis}] "
        )
        + f"{before_tps:.2f}"
        + (f" +- {before_tps_ci:.2f}" if before_tps_ci is not None else "")
        + f" -> {after_tps:.2f}"
        + (f" +- {after_tps_ci:.2f}" if after_tps_ci is not None else "")
        + "  "
        f"delta={after_tps - before_tps:+.2f}"
    )
    for label in ("peak_allocated_bytes", "peak_reserved_bytes"):
        before_memory = _select_memory_summary(before_workload, label=label)
        after_memory = _select_memory_summary(after_workload, label=label)
        if before_memory is None or after_memory is None:
            continue
        before_value = float(before_memory["mean_bytes"])
        after_value = float(after_memory["mean_bytes"])
        before_ci = _ci95_halfwidth(before_memory, key="stdev_bytes")
        after_ci = _ci95_halfwidth(after_memory, key="stdev_bytes")
        print(
            f"{label} "
            f"{_bytes_to_mib(before_value):.2f} MiB"
            + (f" +- {_bytes_to_mib(before_ci):.2f}" if before_ci is not None else "")
            + f" -> {_bytes_to_mib(after_value):.2f} MiB"
            + (f" +- {_bytes_to_mib(after_ci):.2f}" if after_ci is not None else "")
            + "  "
            f"delta={_bytes_to_mib(after_value - before_value):+.2f} MiB"
        )
    combined_step_ci = None
    if before_step_ci is not None and after_step_ci is not None:
        combined_step_ci = math.sqrt(before_step_ci**2 + after_step_ci**2)
        if abs(after_step - before_step) <= combined_step_ci:
            print(
                "step_delta_note "
                f"delta is within combined 95% repeat CI ({combined_step_ci:.6f} ms)"
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
        "kind": "compare_training_perf",
        "schema_version": 1,
        "backend": args.backend,
        "case": args.case,
        "before": str(args.before),
        "after": str(args.after),
        "step_basis": {"before": before_step_basis, "after": after_step_basis},
        "tokens_per_s_basis": {
            "before": before_tps_basis,
            "after": after_tps_basis,
        },
        "step_mean_ms": {
            "before": before_step,
            "after": after_step,
            "delta": after_step - before_step,
            "before_ci95_halfwidth": before_step_ci,
            "after_ci95_halfwidth": after_step_ci,
            "combined_ci95_halfwidth": combined_step_ci,
        },
        "tokens_per_s": {
            "before": before_tps,
            "after": after_tps,
            "delta": after_tps - before_tps,
            "before_ci95_halfwidth": before_tps_ci,
            "after_ci95_halfwidth": after_tps_ci,
        },
        "memory": {
            label: {
                "before": float(before_summary["mean_bytes"]),
                "after": float(after_summary["mean_bytes"]),
                "delta": float(after_summary["mean_bytes"])
                - float(before_summary["mean_bytes"]),
                "before_ci95_halfwidth": _ci95_halfwidth(
                    before_summary, key="stdev_bytes"
                ),
                "after_ci95_halfwidth": _ci95_halfwidth(
                    after_summary, key="stdev_bytes"
                ),
            }
            for label in ("peak_allocated_bytes", "peak_reserved_bytes")
            for before_summary, after_summary in [
                (
                    _select_memory_summary(before_workload, label=label),
                    _select_memory_summary(after_workload, label=label),
                )
            ]
            if before_summary is not None and after_summary is not None
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
