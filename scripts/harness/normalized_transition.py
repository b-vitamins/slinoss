"""Shared bounded diagnostic for a fan-in-normalized token transition."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from slinoss import SLinOSSMixer, SLinOSSMixerConfig


class _PackedRowScale(nn.Module):
    def __init__(
        self,
        rows: int,
        start: int,
        stop: int,
        factor: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        scale = torch.ones(rows, 1, device=device, dtype=dtype)
        scale[start:stop] = factor
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, weight: Tensor) -> Tensor:
        return weight * self.scale


class NormalizedTransitionMixer(SLinOSSMixer):
    """SLinOSS with token-transition projection normalized by input fan-in."""

    def __init__(self, config: SLinOSSMixerConfig) -> None:
        super().__init__(config)
        factor = 1.0 / math.sqrt(config.d_model)
        weight = self.in_proj.weight
        parametrize.register_parametrization(
            self.in_proj,
            "weight",
            _PackedRowScale(
                weight.shape[0],
                self.layout.params_off,
                self.layout.out_features,
                factor,
                device=weight.device,
                dtype=weight.dtype,
            ),
        )
        self.transition_tangent_scale = factor


__all__ = ["NormalizedTransitionMixer"]
