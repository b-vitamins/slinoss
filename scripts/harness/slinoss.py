"""Shared SLinOSS constructor plumbing for experimental registries."""

from __future__ import annotations

from typing import Any

from torch import nn

__all__ = ["build_slinoss", "slinoss_defaults"]


def build_slinoss(d_model: int, **settings: Any) -> nn.Module:
    """Build the mixer from its mixer-only configuration."""
    from slinoss import SLinOSSMixer, SLinOSSMixerConfig

    return SLinOSSMixer(SLinOSSMixerConfig(d_model=d_model, **settings))


def slinoss_defaults(d_state: int, *, d_head: int | None = None) -> dict[str, Any]:
    """Return every configurable mixer default, with the axis's state geometry."""
    from slinoss import SLinOSSMixerConfig

    return {
        "d_state": d_state,
        "expand": SLinOSSMixerConfig.expand,
        "d_head": SLinOSSMixerConfig.d_head if d_head is None else d_head,
        "n_groups": SLinOSSMixerConfig.n_groups,
        "chunk_size": SLinOSSMixerConfig.chunk_size,
        "d_conv": SLinOSSMixerConfig.d_conv,
        "key_conv": SLinOSSMixerConfig.key_conv,
        "bias": SLinOSSMixerConfig.bias,
        "conv_bias": SLinOSSMixerConfig.conv_bias,
        "norm_eps": SLinOSSMixerConfig.norm_eps,
    }
