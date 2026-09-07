"""Source-faithful B/C frontends on the R10 SO(3) recurrence.

These are bounded diagnostic mixers.  They deliberately leave the SO(3)
transition, cyclic initial state, value path, tail, and output projection alone.
Only the B/C projection initialization and realization follow the named prior.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from slinoss import SLinOSSMixer, SLinOSSMixerConfig
from slinoss.config import ROTATION_CHART_SCALE_MAX
from slinoss.ops.conv import causal_conv1d
from slinoss.ops.mixer import mixer_tail
from slinoss.ops.scanprep import scanprep
from slinoss.ops.so3ssd import so3ssd

_OLD_BC_EPS = 1.0e-8


def _old_slinoss_vectors(raw: Tensor, *, eps: float = _OLD_BC_EPS) -> Tensor:
    """Natural three-dimensional lift of old SLinOSS polar B/C.

    Each group emits one positive amplitude, azimuth, and height per SO(3)
    lane.  The resulting 3-vectors are normalized by the RMS of their lane
    magnitudes, exactly as old SLinOSS normalizes complex-pair magnitudes.
    """
    lanes, remainder = divmod(raw.shape[-1], 3)
    if remainder:
        raise ValueError(
            f"B/C width must contain three fields per lane, got {raw.shape[-1]}"
        )
    fields = raw.unflatten(-1, (3, lanes)).float()
    amplitude = F.softplus(fields[..., 0, :])
    azimuth = math.pi * torch.tanh(fields[..., 1, :])
    height = torch.tanh(fields[..., 2, :])
    planar = (1.0 - height.square()).clamp_min(0.0).sqrt()
    vectors = amplitude.unsqueeze(-1) * torch.stack(
        (planar * azimuth.cos(), planar * azimuth.sin(), height), dim=-1
    )
    magnitude_square = vectors.square().sum(-1, dtype=torch.float32)
    lane_rms = magnitude_square.mean(-1, keepdim=True).clamp_min(eps).sqrt()
    return (vectors / lane_rms.unsqueeze(-1)).flatten(-2).to(raw.dtype).contiguous()


def _mamba3_vectors(
    raw: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    heads_per_group: int,
    eps: float,
) -> Tensor:
    """Mamba3's B/C RMSNorm, group broadcast, and per-head +1 bias."""
    dtype = raw.dtype
    value = raw.float()
    scale = torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    value = value * scale * weight.float()
    value = value.repeat_interleave(heads_per_group, dim=1)
    return (value + bias.float()[None, :, None, :]).to(dtype).contiguous()


class _R10BCMixer(SLinOSSMixer):
    """Common R10 body with the prior-specific B/C realization left abstract."""

    def __init__(self, config: SLinOSSMixerConfig) -> None:
        if config.key_conv:
            raise ValueError("old SLinOSS and Mamba3 do not convolve B/C")
        if config.bias:
            raise ValueError("the source B/C projections are bias-free")
        super().__init__(config)
        self.transition_tangent_scale = 1.0 / math.sqrt(config.d_model)
        with torch.no_grad():
            # super() deliberately shrinks these rows for L2 normalization.
            # Both sources instead leave their fused linear at nn.Linear's
            # Kaiming-uniform (a=sqrt(5)) initialization.
            rows = self.in_proj.weight[self.layout.b_off : self.layout.params_off]
            nn.init.kaiming_uniform_(rows, a=math.sqrt(5.0))

    def _vectors(self, b_raw: Tensor, c_raw: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

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
        B, C = self._vectors(layout.b(projected), layout.c(projected))
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
        tail = mixer_tail(
            scan.y,
            step.y,
            layout.gate(projected),
            self.d_skip,
            self.norm_weight,
            eps=cfg.norm_eps,
        )
        return F.linear(tail, self.out_proj.weight, self.out_proj.bias)


class OldSLinOSSBCMixer(_R10BCMixer):
    """R10 with the faithful SO(3) lift of old v2x2 SLinOSS B/C."""

    def _vectors(self, b_raw: Tensor, c_raw: Tensor) -> tuple[Tensor, Tensor]:
        return _old_slinoss_vectors(b_raw), _old_slinoss_vectors(c_raw)


class Mamba3BCMixer(_R10BCMixer):
    """R10 with Mamba3's exact B/C frontend and initialization."""

    def __init__(self, config: SLinOSSMixerConfig) -> None:
        super().__init__(config)
        factory = {
            "device": self.in_proj.weight.device,
            "dtype": self.in_proj.weight.dtype,
        }
        self.B_norm_weight = nn.Parameter(torch.ones(config.d_state, **factory))
        self.C_norm_weight = nn.Parameter(torch.ones(config.d_state, **factory))
        self.B_bias = nn.Parameter(
            torch.ones(
                config.n_heads,
                config.d_state,
                device=factory["device"],
                dtype=torch.float32,
            )
        )
        self.C_bias = nn.Parameter(
            torch.ones(
                config.n_heads,
                config.d_state,
                device=factory["device"],
                dtype=torch.float32,
            )
        )

    def _vectors(self, b_raw: Tensor, c_raw: Tensor) -> tuple[Tensor, Tensor]:
        heads_per_group = self.config.heads_per_group
        return (
            _mamba3_vectors(
                b_raw,
                self.B_norm_weight,
                self.B_bias,
                heads_per_group=heads_per_group,
                eps=self.config.norm_eps,
            ),
            _mamba3_vectors(
                c_raw,
                self.C_norm_weight,
                self.C_bias,
                heads_per_group=heads_per_group,
                eps=self.config.norm_eps,
            ),
        )


def _config(d_model: int, settings: dict[str, object]) -> SLinOSSMixerConfig:
    exact = dict(settings)
    exact["key_conv"] = False
    exact["bias"] = False
    return SLinOSSMixerConfig(d_model=d_model, **exact)


def build_old_slinoss_bc(d_model: int, **settings: object) -> nn.Module:
    return OldSLinOSSBCMixer(_config(d_model, settings))


def build_mamba3_bc(d_model: int, **settings: object) -> nn.Module:
    return Mamba3BCMixer(_config(d_model, settings))


__all__ = [
    "Mamba3BCMixer",
    "OldSLinOSSBCMixer",
    "build_mamba3_bc",
    "build_old_slinoss_bc",
]
