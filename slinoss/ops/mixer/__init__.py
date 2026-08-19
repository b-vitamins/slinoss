"""Fused rowwise mixer tail."""

from slinoss.ops.mixer.backends import (
    Backend,
    MixerBackend,
    MixerBackward,
    MixerForward,
    get,
    names,
    register,
    resolve,
)
from slinoss.ops.mixer.interface import MixerTailFunction, mixer_tail
from slinoss.ops.mixer.reference import (
    MixerTailGrads,
    mixer_tail_bwd_ref,
    mixer_tail_ref,
)

__all__ = [
    "Backend",
    "MixerBackend",
    "MixerBackward",
    "MixerForward",
    "MixerTailFunction",
    "MixerTailGrads",
    "get",
    "mixer_tail",
    "mixer_tail_bwd_ref",
    "mixer_tail_ref",
    "names",
    "register",
    "resolve",
]
