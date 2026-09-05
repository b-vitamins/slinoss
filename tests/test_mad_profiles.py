"""Atomic MAD profile contracts.

These tests are intentionally about resolved structures, not merely parser defaults: a
paper-reconstruction label is useful only if it cannot coexist with a conflicting
scaffold or optimizer setting, and it must not imply evidence the upstream release lacks.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from scripts.mad.model import BottleneckModel, CausalModel, ModelConfig, RMSNorm, SwiGLU
from scripts.mad.profiles import KLA_PAPER_V2
from scripts.mad.run import _resolved_profile, build_parser
from scripts.mad.tasks import TASKS


def _paper_config(*, bottleneck: bool) -> ModelConfig:
    """Small-vocabulary model with every scaffold field from the locked profile."""
    return ModelConfig(
        vocab_size=16,
        task_length=32,
        observed_width=32,
        bottleneck=bottleneck,
        **KLA_PAPER_V2.model_args(),
    )


def test_kla_paper_v2_is_the_exact_locked_protocol() -> None:
    """Paper-fixed settings resolve together, with reconstruction status explicit."""
    profile = KLA_PAPER_V2
    assert profile.locked
    assert profile.contract_status == "textual-reconstruction"
    assert not profile.published_table_eligible
    assert any("encoder MLP" in item for item in profile.limitations)
    assert (profile.d_model, profile.n_layers) == (128, 1)
    assert (profile.epochs, profile.patience, profile.eval_every) == (750, 70, 10)
    assert (profile.batch_size, profile.lr, profile.weight_decay) == (172, 1e-3, 0.0)
    assert (profile.schedule, profile.precision, profile.grad_clip) == (
        "none",
        "fp32",
        5.0,
    )
    assert profile.drop_last
    assert profile.decoder_widths == (240, 120)


@pytest.mark.parametrize(
    "flag",
    (
        ["--batch-size", "128"],
        ["--precision", "bf16"],
        ["--d-model", "256"],
        ["--no-drop-last"],
    ),
)
def test_kla_paper_v2_refuses_a_conflicting_override(flag: list[str]) -> None:
    """A command cannot retain the published label after changing its contract."""
    parser = build_parser()
    args = parser.parse_args(["--profile", "kla-paper-v2", *flag])
    with pytest.raises(SystemExit, match="2"):
        _resolved_profile(args, parser)


def test_legacy_hybrid_remains_explicitly_overrideable() -> None:
    """Old commands can still be replayed, but under an honest profile name."""
    parser = build_parser()
    args = parser.parse_args(
        ["--profile", "legacy-hybrid", "--batch-size", "64", "--precision", "bf16"]
    )
    profile, _, train = _resolved_profile(args, parser)
    assert not profile.locked
    assert (train["batch_size"], train["precision"]) == (64, "bf16")


def test_kla_scaffold_details_are_structural_and_recordable() -> None:
    """The profile builds the repository scaffold plus paper decoder widths."""
    causal = CausalModel(_paper_config(bottleneck=False), lambda _d, _t: nn.Identity())
    assert causal.head.bias is None
    channel = causal.encoder[1].inner
    assert isinstance(channel, SwiGLU)
    assert channel.fused_input
    assert channel.w12.weight.shape == (2 * 341, 128)

    bottleneck = BottleneckModel(
        _paper_config(bottleneck=True), lambda _d, _t: nn.Identity()
    )
    assert isinstance(bottleneck.encoder_norm, RMSNorm)
    first, second = bottleneck.decoder[1], bottleneck.decoder[4]
    assert isinstance(first, nn.Linear) and first.weight.shape == (240, 128)
    assert isinstance(second, nn.Linear) and second.weight.shape == (120, 240)
    assert bottleneck.head.weight.shape == (16, 120)
    assert bottleneck.head.bias is not None

    position = torch.arange(32, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, 128, 2, dtype=torch.float32) * (-math.log(10000.0) / 128)
    )
    expected = torch.zeros(32, 128)
    expected[:, 0::2] = torch.sin(position * frequency)
    expected[:, 1::2] = torch.cos(position * frequency)
    torch.testing.assert_close(bottleneck.positions, expected)


def test_task_baselines_match_mad_lab_with_the_documented_paper_correction() -> None:
    """Table 8's noisy vocab 16 is impossible; official MAD-Lab correctly uses 32."""
    compact = {
        name: (spec.vocab_size, spec.seq_len, spec.num_train, spec.num_test, spec.extra)
        for name, spec in TASKS.items()
    }
    assert compact == {
        "icr": (16, 128, 12800, 1280, {"multi_query": True}),
        "nicr": (
            32,
            128,
            12800,
            1280,
            {"multi_query": True, "noise_vocab_size": 16, "frac_noise": 0.2},
        ),
        "ficr": (
            16,
            128,
            12800,
            1280,
            {"multi_query": True, "k_motif_size": 3, "v_motif_size": 3},
        ),
        "mem": (256, 32, 256, 1280, {}),
        "comp": (16, 32, 12800, 1280, {}),
        "sc": (16, 256, 12800, 1280, {"num_tokens_to_copy": 16}),
    }
