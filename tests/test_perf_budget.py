from __future__ import annotations

import pytest

from slinoss.perf.budget import build_tree, derive_nextchar_budget


def test_derive_nextchar_budget_builds_expected_aggregates() -> None:
    sample = {
        "step.forward_loss": 10.0,
        "step.backward": 20.0,
        "step.clip": 1.0,
        "step.optim": 2.0,
        "step.zero_grad": 0.5,
        "forward.v2x2ssd.total": 3.0,
        "backward.v2x2ssd.total": 8.0,
        "forward.embed.token": 1.0,
        "forward.embed.pos": 0.5,
        "backward.embed.token": 0.75,
        "backward.embed.pos": 0.25,
        "forward.norms.pre_mixer": 0.1,
        "forward.norms.pre_ffn": 0.2,
        "forward.norms.final": 0.3,
        "backward.norms.pre_mixer": 0.4,
        "backward.norms.pre_ffn": 0.5,
        "backward.norms.final": 0.6,
        "forward.mixer.in_proj": 0.7,
        "forward.mixer.dw_conv": 0.8,
        "forward.mixer.bc_emit": 0.9,
        "forward.mixer.dw_conv_activation": 0.25,
        "forward.mixer.scanprep.total": 2.25,
        "forward.mixer.scanprep.pack_u": 0.4,
        "forward.mixer.scanprep.bc_norm": 0.15,
        "forward.mixer.scanprep.coefficients": 1.0,
        "forward.mixer.scanprep.pack_bc": 0.7,
        "forward.mixer.gate_skip": 1.2,
        "backward.mixer.in_proj": 1.3,
        "backward.mixer.dw_conv": 1.4,
        "backward.mixer.bc_emit": 1.5,
        "backward.mixer.dw_conv_activation": 0.45,
        "backward.mixer.scanprep.total": 3.65,
        "backward.mixer.scanprep.pack_u": 0.5,
        "backward.mixer.scanprep.bc_norm": 0.35,
        "backward.mixer.scanprep.coefficients": 1.6,
        "backward.mixer.scanprep.pack_bc": 1.2,
        "backward.mixer.gate_skip": 1.8,
        "forward.ffn": 0.9,
        "backward.ffn": 1.9,
        "forward.residual.mixer": 0.11,
        "forward.residual.ffn": 0.12,
        "backward.residual.mixer": 0.21,
        "backward.residual.ffn": 0.22,
        "forward.head.logits": 0.4,
        "forward.head.loss": 0.2,
        "backward.head.logits": 0.6,
        "backward.head.loss": 0.7,
        "forward.v2x2ssd.chunk_increment.total": 1.0,
        "forward.v2x2ssd.state_passing.total": 0.5,
        "forward.v2x2ssd.chunk_scan.total": 1.25,
        "backward.v2x2ssd.chunk_increment.total": 2.0,
        "backward.v2x2ssd.state_passing.total": 1.0,
        "backward.v2x2ssd.chunk_scan.total": 3.0,
        "backward.v2x2ssd.chunk_increment.db": 0.4,
        "backward.v2x2ssd.chunk_increment.du": 0.5,
        "backward.v2x2ssd.chunk_increment.boundary": 0.3,
        "backward.v2x2ssd.chunk_increment.param_scan": 0.2,
        "backward.v2x2ssd.state_passing.state": 0.4,
        "backward.v2x2ssd.state_passing.m": 0.2,
        "backward.v2x2ssd.chunk_scan.dz0": 0.3,
        "backward.v2x2ssd.chunk_scan.du": 0.4,
        "backward.v2x2ssd.chunk_scan.db": 0.5,
        "backward.v2x2ssd.chunk_scan.dcdr": 0.6,
        "backward.v2x2ssd.chunk_scan.param_scan": 0.7,
    }
    derived = derive_nextchar_budget(sample)

    assert derived["step.total"] == 33.5
    assert derived["forward.other.total"] == 7.0
    assert derived["backward.other.total"] == 12.0
    assert derived["forward.v2x2ssd.stage_sum"] == pytest.approx(2.75)
    assert derived["forward.v2x2ssd.overhead"] == pytest.approx(0.25)
    assert derived["backward.v2x2ssd.stage_sum"] == pytest.approx(6.0)
    assert derived["backward.v2x2ssd.overhead"] == pytest.approx(2.0)
    assert derived["backward.v2x2ssd.chunk_increment.kernel_sum"] == pytest.approx(1.4)
    assert derived["backward.v2x2ssd.chunk_increment.overhead"] == pytest.approx(0.6)
    assert derived["backward.v2x2ssd.state_passing.kernel_sum"] == pytest.approx(0.6)
    assert derived["backward.v2x2ssd.state_passing.overhead"] == pytest.approx(0.4)
    assert derived["backward.v2x2ssd.chunk_scan.kernel_sum"] == pytest.approx(2.5)
    assert derived["backward.v2x2ssd.chunk_scan.overhead"] == pytest.approx(0.5)
    assert derived["forward.embed.total"] == 1.5
    assert derived["backward.embed.total"] == 1.0
    assert derived["forward.norms.total"] == pytest.approx(0.6)
    assert derived["backward.norms.total"] == pytest.approx(1.5)
    assert derived["forward.mixer.scanprep.total"] == pytest.approx(2.25)
    assert derived["backward.mixer.scanprep.total"] == pytest.approx(3.65)
    assert derived["forward.mixer.total"] == pytest.approx(6.1)
    assert derived["backward.mixer.total"] == pytest.approx(10.1)
    assert derived["forward.residual.total"] == pytest.approx(0.23)
    assert derived["backward.residual.total"] == pytest.approx(0.43)
    assert derived["forward.head.total"] == pytest.approx(0.6)
    assert derived["backward.head.total"] == pytest.approx(1.3)
    assert derived["forward.other.unattributed"] == pytest.approx(-2.93)
    assert derived["backward.other.unattributed"] == pytest.approx(-4.23)

    tree = build_tree(
        {
            "step.total": {"mean_ms": 33.5},
            "forward.total": {"mean_ms": 10.0},
            "forward.v2x2ssd.total": {"mean_ms": 4.0},
            "forward.v2x2ssd.chunk_scan.total": {"mean_ms": 1.5},
        }
    )
    assert "step" in tree
    assert "forward" in tree
    assert tree["forward"]["__stats__"]["mean_ms"] == 10.0
    assert tree["forward"]["v2x2ssd"]["__stats__"]["mean_ms"] == 4.0
    assert tree["forward"]["v2x2ssd"]["chunk_scan"]["__stats__"][
        "percent_of_parent_mean"
    ] == pytest.approx(37.5)
