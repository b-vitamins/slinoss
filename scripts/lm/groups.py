"""The five parameter groups, and the rule that puts a parameter in exactly one.

    embedding     the token table                      own rate, no decay
    unembedding   the head                             base rate, decay
    hidden        two-dimensional projection weights    base rate, decay
    scalar        norm gains, biases, anything 1-D      base rate, no decay
    ssm           the state-space parameters            0.1x rate, no decay

The state-space group is separated because a recurrence's transition parameters sit inside a
scan: a rate that suits a projection drives them past the stability the scan assumes, and
decay on them shrinks the dynamics rather than a weight. Two things route a parameter there,
and both are needed. A ``_no_weight_decay`` attribute is what the mixers in this tree and in
``mamba_ssm`` already set, so a baseline's own declaration is honoured without a per-baseline
table. :data:`SSM_LEAVES` names the rest by leaf, because a transition parameter that carries
no flag is still a transition parameter.

Two named deviations from a rule that would route by rank alone. A norm gain goes to
``scalar`` even at rank two, because ``SLinOSSMixer.norm_weight`` is ``(H,P)`` and is a gain.
The depthwise convolution kernel goes to ``hidden`` at rank two, matching ``mamba_ssm``, which
flags ``A_log`` and ``D`` and leaves ``conv1d.weight`` in the decayed group.

The partition is the point. Upstream trainers on this axis lose parameters to a group rule
that overlaps or misses, so :func:`parameter_groups` checks that the groups cover every
trainable parameter exactly once and refuses to build an optimizer otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

__all__ = [
    "GROUPS",
    "SSM_LEAVES",
    "GroupPolicy",
    "classify",
    "group_counts",
    "parameter_groups",
]

GROUPS = ("embedding", "unembedding", "hidden", "scalar", "ssm")
"""Group names, in report order."""

SSM_LEAVES = frozenset({"param_bias", "d_skip"})
"""Mixer leaves that are state-space parameters whatever their rank.

``param_bias`` carries the transition and the taps; ``d_skip`` is the direct path's gain and
already declares ``_no_weight_decay``. Named as well as flagged so the routing does not
depend on an attribute another tree might drop.
"""


@dataclass(frozen=True)
class GroupPolicy:
    """Rate and decay per group.

    Attributes:
        lr: Base rate, already transferred to this arm's width and batch.
        embedding_lr: Rate for the token table, transferred by the same factor.
        ssm_multiplier: Multiple of ``lr`` the state-space group runs at.
        weight_decay: Decay for the two decayed groups.
    """

    lr: float
    embedding_lr: float
    ssm_multiplier: float = 0.1
    weight_decay: float = 0.1

    def __post_init__(self) -> None:
        for name in ("lr", "embedding_lr", "ssm_multiplier"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must not be negative, got {self.weight_decay}"
            )

    def rate(self, group: str) -> float:
        """The rate one group runs at.

        Args:
            group: Group name.

        Returns:
            The rate.

        Raises:
            ValueError: On an unknown group.
        """
        if group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
        if group == "embedding":
            return self.embedding_lr
        if group == "ssm":
            return self.lr * self.ssm_multiplier
        return self.lr

    def decay(self, group: str) -> float:
        """The decay one group carries.

        Args:
            group: Group name.

        Returns:
            The decay.

        Raises:
            ValueError: On an unknown group.
        """
        if group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
        return self.weight_decay if group in {"unembedding", "hidden"} else 0.0


def classify(name: str, param: Tensor) -> str:
    """Which group a parameter belongs to.

    Args:
        name: Dotted parameter name, as :meth:`torch.nn.Module.named_parameters` gives it.
        param: The parameter, for its rank and its flags.

    Returns:
        One of :data:`GROUPS`.
    """
    leaf = name.rsplit(".", 1)[-1]
    if "embedding" in name:
        return "embedding"
    if name.startswith("head."):
        return "unembedding"
    in_mixer = ".mixer." in f".{name}"
    flagged: Any = getattr(param, "_no_weight_decay", False)
    if in_mixer and (bool(flagged) or leaf in SSM_LEAVES):
        return "ssm"
    if "norm" in name or param.ndim == 1:
        return "scalar"
    return "hidden"


def group_counts(model: nn.Module) -> dict[str, int]:
    """Trainable parameters per group.

    Args:
        model: The model.

    Returns:
        A count per name in :data:`GROUPS`, zeros included. An arm whose mixer carries no
        state-space parameter reports ``ssm`` at zero, which is correct rather than
        missing.
    """
    counts: dict[str, int] = dict.fromkeys(GROUPS, 0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            counts[classify(name, param)] += param.numel()
    return counts


def parameter_groups(model: nn.Module, policy: GroupPolicy) -> list[dict[str, Any]]:
    """Build the optimizer's param groups.

    Args:
        model: The model.
        policy: Rates and decay.

    Returns:
        One dict per non-empty group, carrying ``name``, ``params``, ``lr`` and
        ``weight_decay``. ``lr`` is the group's rate at the peak; the trainer scales every
        group by one schedule factor, so the ratios between groups hold at every step.

    Raises:
        ValueError: On a parameter reachable under two names. A tied weight has two names
            and one identity, so the rule would route it by whichever name came first and
            the choice would be invisible. Nothing in this scaffold ties, and a baseline
            that does has to say which group it wants rather than inherit an accident.
            Found by walking with ``remove_duplicate=False``, since the default walk hides
            a tie by yielding the shared parameter once, under its first name.
    """
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
