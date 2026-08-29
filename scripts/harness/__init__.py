"""Pieces every measurement axis shares.

A mixer registry and the two non-recurrent controls, lifted out of
``scripts/state_tracking/mixers.py`` so that an axis contributes its own defaults and
nothing else. Two axes that build their own attention are two axes whose numbers cannot be
compared, and the arithmetic of a shared scaffold is the whole point of an arm.

Nothing here imports an optional dependency at module scope.
"""

from scripts.harness.controls import CausalAttention, CausalConv, Rotary
from scripts.harness.registry import (
    Mixer,
    MixerEntry,
    MixerFactory,
    Registry,
    load_module,
)

__all__ = [
    "CausalAttention",
    "CausalConv",
    "Mixer",
    "MixerEntry",
    "MixerFactory",
    "Registry",
    "Rotary",
    "load_module",
]
