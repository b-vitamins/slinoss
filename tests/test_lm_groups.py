"""The five parameter groups: a partition, and the right parameter in each.

Two things are checked and they are different. That the rule is a partition -- every trainable
parameter in exactly one group, the union the whole set -- is what stops a parameter from
silently training at no rate or at two. That the state-space parameters land in ``ssm`` is what
the group exists for: a transition parameter at a projection's rate leaves the scan's stability
assumption, and decayed it shrinks the dynamics rather than a weight.

Routing is checked on the real parameter names, because that is what it reads. A rule verified
against invented names would pass while ``mixer_norm_weight`` went to ``ssm`` for containing
``mixer``.
"""

from __future__ import annotations

import pytest
import torch

from scripts.lm.groups import (
    GROUPS,
    GroupPolicy,
    classify,
    group_counts,
    parameter_groups,
)
from scripts.lm.mixers import REGISTRY
from scripts.lm.model import build_model, layer_factories, scaffold_config
from slinoss import SLinOSSStack

D_MODEL = 64
N_LAYERS = 2
VOCAB = 64
MAX_LENGTH = 32


def _model(mixer: str, overrides: tuple[str, ...] = ()) -> SLinOSSStack:
    """A two-layer stack with one mixer swapped into every block."""
    torch.manual_seed(0)
    config = scaffold_config(d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB)
    resolved = REGISTRY.resolve(mixer, overrides)
    return build_model(
        config, layer_factories(resolved.factory, N_LAYERS), max_length=MAX_LENGTH
    )


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("embedding.weight", "embedding"),
        ("head.weight", "unembedding"),
        ("norm_weight", "scalar"),
        ("blocks.0.mixer_norm_weight", "scalar"),
        ("blocks.0.ffn_norm_weight", "scalar"),
        ("blocks.0.ffn_gate.weight", "hidden"),
        ("blocks.0.ffn_out_weight", "hidden"),
        ("blocks.0.mixer.in_proj.weight", "hidden"),
        ("blocks.0.mixer.out_proj.weight", "hidden"),
        ("blocks.0.mixer.conv_weight", "hidden"),
        ("blocks.0.mixer.norm_weight", "scalar"),
        ("blocks.0.mixer.param_bias", "ssm"),
    ],
)
def test_the_rule_routes_the_real_names(name: str, group: str) -> None:
    """Each name the scaffold and the mixer produce, and the group it belongs to.

    ``mixer_norm_weight`` is the one that catches a substring test written as ``mixer`` rather
    than ``.mixer.``: it is a block-level norm gain, not a state-space parameter. The
    convolution kernel is rank two and goes to ``hidden``, matching ``mamba_ssm``, which
    flags its transition and its skip and leaves the kernel decayed.
    """
    assert classify(name, torch.zeros(4, 4)) == group


def test_a_flagged_parameter_reaches_the_ssm_group() -> None:
    """A baseline's own ``_no_weight_decay`` declaration is honoured without a per-arm table.

    Both routes are needed: the flag covers what a mixer declares, and the leaf names cover
    a transition parameter that declares nothing.
    """
    plain = torch.zeros(8)
    flagged = torch.zeros(8)
    flagged._no_weight_decay = True  # type: ignore[attr-defined]
    assert classify("blocks.0.mixer.A_log", plain) == "scalar"
    assert classify("blocks.0.mixer.A_log", flagged) == "ssm"
    assert classify("blocks.0.mixer.d_skip", plain) == "ssm"


def test_the_flag_only_counts_inside_a_mixer() -> None:
    """A flag on a scaffold parameter does not move it into the recurrence's group.

    The scaffold is shared across arms and its rates are what make them comparable; a
    baseline setting a flag on something outside its own mixer must not move it.
    """
    flagged = torch.zeros(4, 4)
    flagged._no_weight_decay = True  # type: ignore[attr-defined]
    assert classify("blocks.0.ffn_out_weight", flagged) == "hidden"


def test_the_groups_partition_every_trainable_parameter() -> None:
    """The union is the whole set and the intersections are empty.

    Checked by count rather than by inspection: a parameter routed nowhere would leave the
    sum short, and one routed twice would leave it long.
    """
    model = _model("slinoss")
    counts = group_counts(model)
    assert set(counts) == set(GROUPS)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(counts.values()) == total
    assert counts["ssm"] > 0
    assert counts["embedding"] == VOCAB * D_MODEL


def test_transition_embedding_is_ssm_not_token_embedding() -> None:
    """A shared word in a leaf name must not override its explicit SSM policy."""
    model = _model("slinoss")
    found = [
        (name, classify(name, param))
        for name, param in model.named_parameters()
        if "transition_embedding" in name
    ]
    assert found
    assert all(group == "ssm" for _, group in found)


def test_an_arm_with_no_state_space_parameter_reports_zero() -> None:
    """``conv`` carries no transition, so ``ssm`` is empty and the optimizer omits it.

    Zero rather than missing: a group of no parameters would make AdamW raise, and a count
    silently absent would hide an arm whose recurrence never reached its own rate.
    """
    model = _model("conv")
    assert group_counts(model)["ssm"] == 0
    groups = parameter_groups(model, GroupPolicy(lr=1e-3, embedding_lr=1e-2))
    assert {group["name"] for group in groups} == {
        "embedding",
        "unembedding",
        "hidden",
        "scalar",
    }


def test_the_policy_puts_the_state_space_group_at_a_tenth_with_no_decay() -> None:
    """The rates and decays the protocol specifies, read off the policy rather than typed."""
    policy = GroupPolicy(
        lr=1e-3, embedding_lr=0.1, ssm_multiplier=0.1, weight_decay=0.1
    )
    assert policy.rate("hidden") == pytest.approx(1e-3)
    assert policy.rate("unembedding") == pytest.approx(1e-3)
    assert policy.rate("scalar") == pytest.approx(1e-3)
    assert policy.rate("embedding") == pytest.approx(0.1)
    assert policy.rate("ssm") == pytest.approx(1e-4)
    assert policy.decay("hidden") == pytest.approx(0.1)
    assert policy.decay("unembedding") == pytest.approx(0.1)
    assert policy.decay("embedding") == 0.0
    assert policy.decay("scalar") == 0.0
    assert policy.decay("ssm") == 0.0
    with pytest.raises(ValueError, match="group must be one of"):
        policy.rate("mixer")


def test_the_policy_refuses_a_non_positive_rate() -> None:
    """A zero rate would train nothing and report the run as trained."""
    with pytest.raises(ValueError, match="lr must be positive"):
        GroupPolicy(lr=0.0, embedding_lr=0.1)
    with pytest.raises(ValueError, match="weight_decay must not be negative"):
        GroupPolicy(lr=1e-3, embedding_lr=0.1, weight_decay=-0.1)


def test_a_tied_weight_is_refused_rather_than_routed_by_accident() -> None:
    """One parameter under two names would be routed by whichever name came first.

    Nothing in this scaffold ties. A baseline that does has to say which group it wants,
    because the choice changes its rate and its decay and would otherwise be invisible.

    The guard only works on a walk with ``remove_duplicate=False``: the default walk yields a
    shared parameter once, under its first name, so a tie is already resolved by the time the
    rule sees it. A guard written on the default walk would never fire.
    """
    model = _model("conv")
    head = model.head
    assert head is not None
    embedding = model.embedding
    assert embedding is not None
    head.weight = embedding.weight
    with pytest.raises(ValueError, match="are one parameter"):
        parameter_groups(model, GroupPolicy(lr=1e-3, embedding_lr=1e-2))


def test_a_frozen_parameter_is_in_no_group() -> None:
    """A parameter with no gradient is not the optimizer's, so it is counted nowhere."""
    model = _model("conv")
    embedding = model.embedding
    assert embedding is not None
    embedding.weight.requires_grad_(False)
    assert group_counts(model)["embedding"] == 0
    groups = parameter_groups(model, GroupPolicy(lr=1e-3, embedding_lr=1e-2))
    assert "embedding" not in {group["name"] for group in groups}
