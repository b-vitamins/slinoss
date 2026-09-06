"""MAD registration for the LM-selected normalized-transition candidate."""

from __future__ import annotations

from typing import Any

from torch import nn

from scripts.harness import slinoss_defaults
from scripts.harness.normalized_transition import NormalizedTransitionMixer
from scripts.mad.mixers import MixerEntry, register
from slinoss import SLinOSSMixerConfig


def _build(d_model: int, **settings: Any) -> nn.Module:
    return NormalizedTransitionMixer(SLinOSSMixerConfig(d_model=d_model, **settings))


register(
    "slinoss-normalized-transition",
    MixerEntry(_build, "unused", slinoss_defaults(144)),
)
