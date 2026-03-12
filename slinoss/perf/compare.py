"""Comparison helpers for perf harness payloads."""

from __future__ import annotations

from typing import Any


def flatten_tree_stats(tree: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    stats = tree.get("__stats__")
    if prefix and isinstance(stats, dict):
        flat[prefix] = float(stats.get("mean_ms", 0.0))
    for key, child in tree.items():
        if key == "__stats__" or not isinstance(child, dict):
            continue
        child_prefix = f"{prefix}.{key}" if prefix else key
        flat.update(flatten_tree_stats(child, child_prefix))
    return flat


def compare_budget_trees(
    before_tree: dict[str, Any],
    after_tree: dict[str, Any],
) -> list[dict[str, float | str]]:
    before = flatten_tree_stats(before_tree)
    after = flatten_tree_stats(after_tree)
    labels = sorted(set(before) | set(after))
    rows: list[dict[str, float | str]] = []
    for label in labels:
        before_ms = float(before.get(label, 0.0))
        after_ms = float(after.get(label, 0.0))
        delta_ms = after_ms - before_ms
        delta_pct = (100.0 * delta_ms / before_ms) if before_ms > 0.0 else 0.0
        rows.append(
            {
                "label": label,
                "before_ms": before_ms,
                "after_ms": after_ms,
                "delta_ms": delta_ms,
                "delta_pct": delta_pct,
            }
        )
    return rows


def rank_budget_deltas(
    rows: list[dict[str, float | str]],
    *,
    top_k: int,
) -> dict[str, list[dict[str, float | str]]]:
    regressions = sorted(
        [row for row in rows if float(row["delta_ms"]) > 0.0],
        key=lambda row: float(row["delta_ms"]),
        reverse=True,
    )[:top_k]
    improvements = sorted(
        [row for row in rows if float(row["delta_ms"]) < 0.0],
        key=lambda row: float(row["delta_ms"]),
    )[:top_k]
    return {"regressions": regressions, "improvements": improvements}
