"""State-tracking registrations for bounded LM diagnostic candidates."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from scripts.harness import slinoss_defaults
from scripts.harness.v2_lift_so3 import build_v2_lift_so3
from scripts.state_tracking.mixers import MixerEntry, register
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
    """SLinOSS with its token-transition projection normalized by input fan-in."""

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


def _build(d_model: int, **settings: Any) -> nn.Module:
    return NormalizedTransitionMixer(
        SLinOSSMixerConfig(d_model=d_model, **settings)
    )


register(
    "slinoss-normalized-transition",
    MixerEntry(_build, "unused", slinoss_defaults(144)),
)


def _build_v2_lift(d_model: int, **settings: Any) -> nn.Module:
    return build_v2_lift_so3(d_model, **settings)


_v2_lift_defaults = slinoss_defaults(144)
_v2_lift_defaults["key_conv"] = False
register(
    "slinoss-v2-lift-so3",
    MixerEntry(_build_v2_lift, "unused", _v2_lift_defaults),
)
