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
    as_head_major,
    as_token_major,
    mixer_tail_bwd_ref,
    mixer_tail_ref,
    tail_shape,
)

__all__ = [
    "Backend",
    "MixerBackend",
    "MixerBackward",
    "MixerForward",
    "MixerTailFunction",
    "MixerTailGrads",
    "as_head_major",
    "as_token_major",
    "get",
    "mixer_tail",
    "mixer_tail_bwd_ref",
    "mixer_tail_ref",
    "names",
    "register",
    "resolve",
    "tail_shape",
]
