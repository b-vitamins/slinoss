"""Disjoint optimizer parameter groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

__all__ = [
    "GROUPS",
    "GroupPolicy",
    "classify",
    "group_counts",
    "parameter_groups",
]

GROUPS = ("embedding", "unembedding", "hidden", "scalar", "ssm")
"""Group names, in report order."""


@dataclass(frozen=True)
class GroupPolicy:
    """Learning-rate and decay policy."""

    lr: float
    embedding_lr: float
    weight_decay: float = 0.1

    def __post_init__(self) -> None:
        for name in ("lr", "embedding_lr"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must not be negative, got {self.weight_decay}"
            )

    def rate(self, group: str) -> float:
        """Learning rate for ``group``."""
        if group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
        if group == "embedding":
            return self.embedding_lr
        return self.lr

    def decay(self, group: str) -> float:
        """Weight decay for ``group``."""
        if group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
        return self.weight_decay if group in {"unembedding", "hidden"} else 0.0


def classify(name: str, param: Tensor) -> str:
    """Return the unique optimizer group for ``param``."""
    if name.startswith("embedding."):
        return "embedding"
    if name.startswith("head."):
        return "unembedding"
    in_mixer = ".mixer." in f".{name}"
    flagged: Any = getattr(param, "_no_weight_decay", False)
    if in_mixer and bool(flagged):
        return "ssm"
    if "norm" in name or param.ndim == 1:
        return "scalar"
    return "hidden"


def group_counts(model: nn.Module) -> dict[str, int]:
    """Trainable parameter count per group, zeros included."""
    counts: dict[str, int] = dict.fromkeys(GROUPS, 0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            counts[classify(name, param)] += param.numel()
    return counts


def parameter_groups(model: nn.Module, policy: GroupPolicy) -> list[dict[str, Any]]:
    """Build disjoint optimizer groups; reject parameters reachable twice."""
    members: dict[str, list[Tensor]] = {group: [] for group in GROUPS}
    first: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if not param.requires_grad:
            continue
        if id(param) in first:
            raise ValueError(
                f"{name} and {first[id(param)]} are one parameter; the group rule "
                f"cannot route a tied weight"
            )
        first[id(param)] = name
        members[classify(name, param)].append(param)
    return [
        {
            "name": group,
            "params": params,
            "lr": policy.rate(group),
            "weight_decay": policy.decay(group),
        }
        for group, params in members.items()
        if params
    ]
