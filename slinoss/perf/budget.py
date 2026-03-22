"""Budget aggregation helpers for performance harnesses."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def summarize_scalar_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stdev_ms": 0.0,
            "num_samples": 0.0,
        }
    return {
        "mean_ms": float(statistics.fmean(samples)),
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "stdev_ms": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "num_samples": float(len(samples)),
    }


def summarize_named_samples(
    samples: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    labels = sorted({label for sample in samples for label in sample})
    by_label: dict[str, list[float]] = {label: [] for label in labels}
    for sample in samples:
        for label in labels:
            by_label[label].append(float(sample.get(label, 0.0)))
    return {
        label: summarize_scalar_samples(values) for label, values in by_label.items()
    }


def summarize_cache_samples(
    samples: list[dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
    for sample in samples:
        for label, counts in sample.items():
            totals[label]["hits"] += int(counts.get("hits", 0))
            totals[label]["misses"] += int(counts.get("misses", 0))
    return {label: dict(counts) for label, counts in sorted(totals.items())}


def summarize_budget_samples(
    samples: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    derived_samples = [derive_nextchar_budget(sample) for sample in samples]
    return summarize_named_samples(derived_samples)


def derive_nextchar_budget(sample: dict[str, float]) -> dict[str, float]:
    sample = {label: float(value) for label, value in sample.items()}

    def _sum(*labels: str) -> float:
        return sum(sample.get(label, 0.0) for label in labels)

    def _scanprep_total(direction: str) -> float:
        return sample.get(
            f"{direction}.mixer.scanprep.total",
            _sum(
                f"{direction}.mixer.scanprep.pack_u",
                f"{direction}.mixer.scanprep.bc_norm",
                f"{direction}.mixer.scanprep.coefficients",
                f"{direction}.mixer.scanprep.pack_bc",
            ),
        )

    out = dict(sample)
    out["step.total"] = sample.get(
        "step.total",
        _sum(
            "step.zero_grad",
            "step.forward_loss",
            "step.backward",
            "step.clip",
            "step.optim",
        ),
    )
    out["forward.total"] = sample.get("step.forward_loss", 0.0)
    out["backward.total"] = sample.get("step.backward", 0.0)

    out["forward.v2x2ssd.total"] = sample.get("forward.v2x2ssd.total", 0.0)
    out["backward.v2x2ssd.total"] = sample.get("backward.v2x2ssd.total", 0.0)
    out["forward.other.total"] = out["forward.total"] - out["forward.v2x2ssd.total"]
    out["backward.other.total"] = out["backward.total"] - out["backward.v2x2ssd.total"]

    out["forward.embed.token"] = sample.get("forward.embed.token", 0.0)
    out["forward.embed.pos"] = sample.get("forward.embed.pos", 0.0)
    out["forward.embed.total"] = _sum(
        "forward.embed.token",
        "forward.embed.pos",
    )
    out["backward.embed.token"] = sample.get("backward.embed.token", 0.0)
    out["backward.embed.pos"] = sample.get("backward.embed.pos", 0.0)
    out["backward.embed.total"] = _sum(
        "backward.embed.token",
        "backward.embed.pos",
    )

    out["forward.norms.pre_mixer"] = sample.get("forward.norms.pre_mixer", 0.0)
    out["forward.norms.pre_ffn"] = sample.get("forward.norms.pre_ffn", 0.0)
    out["forward.norms.final"] = sample.get("forward.norms.final", 0.0)
    out["forward.norms.total"] = _sum(
        "forward.norms.pre_mixer",
        "forward.norms.pre_ffn",
        "forward.norms.final",
    )
    out["backward.norms.pre_mixer"] = sample.get("backward.norms.pre_mixer", 0.0)
    out["backward.norms.pre_ffn"] = sample.get("backward.norms.pre_ffn", 0.0)
    out["backward.norms.final"] = sample.get("backward.norms.final", 0.0)
    out["backward.norms.total"] = _sum(
        "backward.norms.pre_mixer",
        "backward.norms.pre_ffn",
        "backward.norms.final",
    )

    out["forward.mixer.in_proj"] = sample.get("forward.mixer.in_proj", 0.0)
    out["forward.mixer.dw_conv"] = sample.get("forward.mixer.dw_conv", 0.0)
    out["forward.mixer.dw_conv_activation"] = sample.get(
        "forward.mixer.dw_conv_activation", 0.0
    )
    out["forward.mixer.bc_emit"] = sample.get(
        "forward.mixer.bc_emit",
        sample.get("forward.mixer.bc_proj", 0.0),
    )
    out["forward.mixer.scanprep.total"] = _scanprep_total("forward")
    out["forward.mixer.scanprep.pack_u"] = sample.get(
        "forward.mixer.scanprep.pack_u", 0.0
    )
    out["forward.mixer.scanprep.bc_norm"] = sample.get(
        "forward.mixer.scanprep.bc_norm", 0.0
    )
    out["forward.mixer.scanprep.coefficients"] = sample.get(
        "forward.mixer.scanprep.coefficients", 0.0
    )
    out["forward.mixer.scanprep.pack_bc"] = sample.get(
        "forward.mixer.scanprep.pack_bc", 0.0
    )
    out["forward.mixer.gate_skip"] = sample.get("forward.mixer.gate_skip", 0.0)
    out["forward.mixer.out_proj"] = sample.get("forward.mixer.out_proj", 0.0)
    out["forward.mixer.total"] = (
        out["forward.mixer.in_proj"]
        + out["forward.mixer.dw_conv"]
        + out["forward.mixer.dw_conv_activation"]
        + out["forward.mixer.bc_emit"]
        + out["forward.mixer.scanprep.total"]
        + out["forward.mixer.gate_skip"]
        + out["forward.mixer.out_proj"]
    )
    out["backward.mixer.in_proj"] = sample.get("backward.mixer.in_proj", 0.0)
    out["backward.mixer.dw_conv"] = sample.get("backward.mixer.dw_conv", 0.0)
    out["backward.mixer.dw_conv_activation"] = sample.get(
        "backward.mixer.dw_conv_activation", 0.0
    )
    out["backward.mixer.bc_emit"] = sample.get(
        "backward.mixer.bc_emit",
        sample.get("backward.mixer.bc_proj", 0.0),
    )
    out["backward.mixer.scanprep.total"] = _scanprep_total("backward")
    out["backward.mixer.scanprep.pack_u"] = sample.get(
        "backward.mixer.scanprep.pack_u", 0.0
    )
    out["backward.mixer.scanprep.bc_norm"] = sample.get(
        "backward.mixer.scanprep.bc_norm", 0.0
    )
    out["backward.mixer.scanprep.coefficients"] = sample.get(
        "backward.mixer.scanprep.coefficients", 0.0
    )
    out["backward.mixer.scanprep.pack_bc"] = sample.get(
        "backward.mixer.scanprep.pack_bc", 0.0
    )
    out["backward.mixer.gate_skip"] = sample.get("backward.mixer.gate_skip", 0.0)
    out["backward.mixer.out_proj"] = sample.get("backward.mixer.out_proj", 0.0)
    out["backward.mixer.total"] = (
        out["backward.mixer.in_proj"]
        + out["backward.mixer.dw_conv"]
        + out["backward.mixer.dw_conv_activation"]
        + out["backward.mixer.bc_emit"]
        + out["backward.mixer.scanprep.total"]
        + out["backward.mixer.gate_skip"]
        + out["backward.mixer.out_proj"]
    )

    out["forward.ffn.total"] = sample.get("forward.ffn", 0.0)
    out["backward.ffn.total"] = sample.get("backward.ffn", 0.0)
    out["forward.residual.mixer"] = sample.get("forward.residual.mixer", 0.0)
    out["forward.residual.ffn"] = sample.get("forward.residual.ffn", 0.0)
    out["forward.residual.total"] = _sum(
        "forward.residual.mixer",
        "forward.residual.ffn",
    )
    out["backward.residual.mixer"] = sample.get("backward.residual.mixer", 0.0)
    out["backward.residual.ffn"] = sample.get("backward.residual.ffn", 0.0)
    out["backward.residual.total"] = _sum(
        "backward.residual.mixer",
        "backward.residual.ffn",
    )

    out["forward.head.logits"] = sample.get("forward.head.logits", 0.0)
    out["forward.head.loss"] = sample.get("forward.head.loss", 0.0)
    out["forward.head.total"] = _sum(
        "forward.head.logits",
        "forward.head.loss",
    )
    out["backward.head.logits"] = sample.get("backward.head.logits", 0.0)
    out["backward.head.loss"] = sample.get("backward.head.loss", 0.0)
    out["backward.head.total"] = _sum(
        "backward.head.logits",
        "backward.head.loss",
    )

    out["forward.other.unattributed"] = out["forward.other.total"] - (
        out["forward.embed.total"]
        + out["forward.norms.total"]
        + out["forward.mixer.total"]
        + out["forward.ffn.total"]
        + out["forward.residual.total"]
        + out["forward.head.total"]
    )
    out["backward.other.unattributed"] = out["backward.other.total"] - (
        out["backward.embed.total"]
        + out["backward.norms.total"]
        + out["backward.mixer.total"]
        + out["backward.ffn.total"]
        + out["backward.residual.total"]
        + out["backward.head.total"]
    )

    out["forward.v2x2ssd.chunk_increment.total"] = sample.get(
        "forward.v2x2ssd.chunk_increment.total", 0.0
    )
    out["forward.v2x2ssd.state_passing.total"] = sample.get(
        "forward.v2x2ssd.state_passing.total", 0.0
    )
    out["forward.v2x2ssd.chunk_scan.total"] = sample.get(
        "forward.v2x2ssd.chunk_scan.total", 0.0
    )
    out["forward.v2x2ssd.stage_sum"] = (
        out["forward.v2x2ssd.chunk_increment.total"]
        + out["forward.v2x2ssd.state_passing.total"]
        + out["forward.v2x2ssd.chunk_scan.total"]
    )
    out["forward.v2x2ssd.overhead"] = (
        out["forward.v2x2ssd.total"] - out["forward.v2x2ssd.stage_sum"]
    )

    out["backward.v2x2ssd.chunk_increment.total"] = sample.get(
        "backward.v2x2ssd.chunk_increment.total", 0.0
    )
    out["backward.v2x2ssd.state_passing.total"] = sample.get(
        "backward.v2x2ssd.state_passing.total", 0.0
    )
    out["backward.v2x2ssd.chunk_scan.total"] = sample.get(
        "backward.v2x2ssd.chunk_scan.total", 0.0
    )
    out["backward.v2x2ssd.stage_sum"] = (
        out["backward.v2x2ssd.chunk_increment.total"]
        + out["backward.v2x2ssd.state_passing.total"]
        + out["backward.v2x2ssd.chunk_scan.total"]
    )
    out["backward.v2x2ssd.overhead"] = (
        out["backward.v2x2ssd.total"] - out["backward.v2x2ssd.stage_sum"]
    )

    out["backward.v2x2ssd.chunk_increment.kernel_sum"] = _sum(
        "backward.v2x2ssd.chunk_increment.db",
        "backward.v2x2ssd.chunk_increment.du",
        "backward.v2x2ssd.chunk_increment.boundary",
        "backward.v2x2ssd.chunk_increment.param_scan",
    )
    out["backward.v2x2ssd.chunk_increment.overhead"] = (
        out["backward.v2x2ssd.chunk_increment.total"]
        - out["backward.v2x2ssd.chunk_increment.kernel_sum"]
    )
    out["backward.v2x2ssd.state_passing.kernel_sum"] = _sum(
        "backward.v2x2ssd.state_passing.state",
        "backward.v2x2ssd.state_passing.m",
    )
    out["backward.v2x2ssd.state_passing.overhead"] = (
        out["backward.v2x2ssd.state_passing.total"]
        - out["backward.v2x2ssd.state_passing.kernel_sum"]
    )
    out["backward.v2x2ssd.chunk_scan.kernel_sum"] = _sum(
        "backward.v2x2ssd.chunk_scan.dz0",
        "backward.v2x2ssd.chunk_scan.du",
        "backward.v2x2ssd.chunk_scan.db",
        "backward.v2x2ssd.chunk_scan.dcdr",
        "backward.v2x2ssd.chunk_scan.param_scan",
    )
    out["backward.v2x2ssd.chunk_scan.overhead"] = (
        out["backward.v2x2ssd.chunk_scan.total"]
        - out["backward.v2x2ssd.chunk_scan.kernel_sum"]
    )

    return out


def build_tree(
    summaries: dict[str, dict[str, float]],
    *,
    root_total_label: str = "step.total",
) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for label, stats in summaries.items():
        parts = label.split(".")
        if len(parts) > 1 and parts[-1] == "total":
            path = parts[:-1]
        else:
            path = parts
        node = tree
        for part in path:
            node = node.setdefault(part, {})
        node["__stats__"] = dict(stats)

    root_total = summaries.get(root_total_label, {}).get("mean_ms", 0.0)
    _annotate_tree(tree, parent_total=root_total, root_total=root_total)
    return tree


def _annotate_tree(
    node: dict[str, Any],
    *,
    parent_total: float,
    root_total: float,
) -> None:
    node_stats = node.get("__stats__")
    current_total = parent_total
    if isinstance(node_stats, dict):
        mean_ms = float(node_stats.get("mean_ms", 0.0))
        node_stats["percent_of_parent_mean"] = (
            (100.0 * mean_ms / parent_total) if parent_total > 0.0 else 0.0
        )
        node_stats["percent_of_step_mean"] = (
            (100.0 * mean_ms / root_total) if root_total > 0.0 else 0.0
        )
        current_total = mean_ms

    for key, child in node.items():
        if key == "__stats__" or not isinstance(child, dict):
            continue
        _annotate_tree(child, parent_total=current_total, root_total=root_total)
