from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from scripts.harness.prior_bc_so3 import (
    Mamba3BCMixer,
    OldSLinOSSBCMixer,
    _mamba3_vectors,
    _old_slinoss_vectors,
)
from slinoss import SLinOSSMixerConfig


def test_old_so3_bc_restricts_exactly_to_old_complex_parameterization() -> None:
    torch.manual_seed(1)
    amplitude = torch.randn(2, 3, 5, 16, dtype=torch.float64)
    phase = torch.randn_like(amplitude)
    height = torch.zeros_like(amplitude)
    raw = torch.stack((amplitude, phase, height), dim=-2).flatten(-2)

    got = _old_slinoss_vectors(raw)
    amp = F.softplus(amplitude.float())
    angle = math.pi * torch.tanh(phase.float())
    pair = amp.unsqueeze(-1) * torch.stack((angle.cos(), angle.sin()), dim=-1)
    row_rms = pair.square().sum(-1).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    want_xy = pair / row_rms.unsqueeze(-1)
    got_lanes = got.unflatten(-1, (16, 3))

    torch.testing.assert_close(got_lanes[..., :2], want_xy.to(got.dtype))
    assert not bool(got_lanes[..., 2].any())


def test_old_so3_bc_has_unit_rms_lane_magnitude() -> None:
    raw = torch.randn(2, 3, 5, 48)
    value = _old_slinoss_vectors(raw).unflatten(-1, (16, 3))
    lane_rms = value.float().square().sum(-1).mean(-1).sqrt()
    torch.testing.assert_close(lane_rms, torch.ones_like(lane_rms))


def test_mamba3_bc_is_rmsnorm_then_group_broadcast_then_bias() -> None:
    torch.manual_seed(2)
    raw = torch.randn(2, 2, 7, 48)
    weight = torch.randn(48)
    bias = torch.randn(4, 48)
    got = _mamba3_vectors(
        raw,
        weight,
        bias,
        heads_per_group=2,
        eps=1e-5,
    )
    normalized = F.rms_norm(raw, (48,), weight, 1e-5)
    want = normalized.repeat_interleave(2, dim=1) + bias[None, :, None, :]
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("kind", ["old", "mamba3"])
def test_prior_bc_mixer_initialization_and_backward(kind: str) -> None:
    cfg = SLinOSSMixerConfig(
        d_model=32,
        d_state=48,
        expand=2.0,
        d_head=16,
        n_groups=2,
        chunk_size=16,
        key_conv=False,
        bias=False,
    )
    cls = OldSLinOSSBCMixer if kind == "old" else Mamba3BCMixer
    torch.manual_seed(3)
    mixer = cls(cfg).double()
    rows = mixer.in_proj.weight[mixer.layout.b_off : mixer.layout.params_off]
    assert float(rows.detach().abs().max()) <= 1.0 / math.sqrt(cfg.d_model)

    x = torch.randn(2, 16, cfg.d_model, dtype=torch.float64, requires_grad=True)
    out = mixer(x)
    out.square().mean().backward()
    assert bool(torch.isfinite(out).all())
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert mixer.in_proj.weight.grad is not None
    assert bool(torch.isfinite(mixer.in_proj.weight.grad).all())


def test_mamba3_bc_affine_parameters_match_source_initialization() -> None:
    cfg = SLinOSSMixerConfig(
        d_model=32,
        d_state=48,
        expand=2.0,
        d_head=16,
        n_groups=2,
        chunk_size=16,
        key_conv=False,
        bias=False,
    )
    mixer = Mamba3BCMixer(cfg)
    for value in (
        mixer.B_norm_weight,
        mixer.C_norm_weight,
        mixer.B_bias,
        mixer.C_bias,
    ):
        torch.testing.assert_close(value, torch.ones_like(value))
