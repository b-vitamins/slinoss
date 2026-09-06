"""State-tracking registrations for bounded LM diagnostic candidates."""

from __future__ import annotations

from typing import Any

from torch import nn

from scripts.harness import slinoss_defaults
from scripts.harness.normalized_transition import NormalizedTransitionMixer
from scripts.harness.v2_lift_so3 import build_v2_lift_so3
from scripts.state_tracking.mixers import MixerEntry, register
from slinoss import SLinOSSMixerConfig


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
