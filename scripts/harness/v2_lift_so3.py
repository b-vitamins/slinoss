"""Experimental SO(3) lift of the production v2x2 SLinOSS learning geometry."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from slinoss import SLinOSSMixer, SLinOSSMixerConfig
from slinoss.config import ROTATION_CHART_SCALE_MAX
from slinoss.ops.conv import causal_conv1d
from slinoss.ops.scanprep import scanprep
from slinoss.ops.so3ssd import so3ssd


def _polar_vectors(raw: Tensor, *, eps: float = 1.0e-8) -> Tensor:
    """Lift old amplitude/phase rows to amplitude/azimuth/height 3-vectors."""
    lanes, remainder = divmod(raw.shape[-1], 3)
    if remainder:
        raise ValueError(f"B/C width must contain three fields per lane, got {raw.shape[-1]}")
    fields = raw.unflatten(-1, (3, lanes)).float()
    amplitude = F.softplus(fields[..., 0, :])
    azimuth = math.pi * torch.tanh(fields[..., 1, :])
    height = torch.tanh(fields[..., 2, :])
    radius = (1.0 - height.square()).clamp_min(0.0).sqrt()
    vectors = amplitude.unsqueeze(-1) * torch.stack(
        (radius * azimuth.cos(), radius * azimuth.sin(), height), dim=-1
    )
    row_rms = vectors.square().sum(-1).mean(-1, keepdim=True).clamp_min(eps).sqrt()
    return (vectors / row_rms.unsqueeze(-1)).flatten(-2).to(raw.dtype).contiguous()


class V2LiftSO3Mixer(SLinOSSMixer):
    """Current SO(3) scan with the learning geometry used by old v2x2ssd."""

    def __init__(self, config: SLinOSSMixerConfig) -> None:
        if config.key_conv:
            raise ValueError("the production v2x2ssd mixer convolved U only")
        super().__init__(config)
        self.transition_tangent_scale = 1.0 / math.sqrt(config.d_model)
        with torch.no_grad():
            # v2x2ssd left every emitted stream live at framework fan-in scale.
            self.in_proj.reset_parameters()
            self.d_skip.normal_()
            nn.init.kaiming_uniform_(
                self.conv_weight.view(config.d_inner, 1, config.d_conv),
                a=math.sqrt(5.0),
            )
            bound = 1.0 / math.sqrt(config.d_conv)
            if self.conv_bias is not None:
                self.conv_bias.uniform_(-bound, bound)
            self.norm_weight.fill_(1.0)
            self.out_proj.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        cfg, layout = self.config, self.layout
        projected = F.linear(x, self.in_proj.weight, self.in_proj.bias)
        if layout.pad_width:
            projected = F.pad(projected, (0, layout.pad_width))
        step = causal_conv1d(
            layout.value(projected),
            self.conv_weight,
            self.conv_bias,
            activation=True,
            d_head=cfg.d_head,
        )
        B = _polar_vectors(layout.b(projected))
        C = _polar_vectors(layout.c(projected))
        params = scanprep(
            layout.params(projected) * self.transition_tangent_scale,
            self.transition_bias,
            heads=cfg.n_heads,
            w_max=ROTATION_CHART_SCALE_MAX,
        )
        z0 = self.initial_state.unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()
        scan = so3ssd(
            step.y,
            params.trans,
            params.K,
            B,
            C,
            cfg.chunk_size,
            z0=z0,
        )

        mixed = scan.y.float() + self.d_skip[None, :, None, None] * step.y.float()
        gate = layout.gate(projected).unflatten(-1, (cfg.n_heads, cfg.d_head))
        gate = gate.permute(0, 2, 1, 3).float()
        mixed = mixed * F.silu(gate)
        mixed = mixed.permute(0, 2, 1, 3).flatten(-2)
        mixed = F.rms_norm(
            mixed,
            (cfg.d_inner,),
            self.norm_weight.flatten(),
            cfg.norm_eps,
        )
        return F.linear(mixed.to(x.dtype), self.out_proj.weight, self.out_proj.bias)

    def step(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError("the bounded training diagnostic has no decode path")


def build_v2_lift_so3(d_model: int, **settings: Any) -> nn.Module:
    return V2LiftSO3Mixer(SLinOSSMixerConfig(d_model=d_model, **settings))


__all__ = ["V2LiftSO3Mixer", "build_v2_lift_so3"]
