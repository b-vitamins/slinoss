"""CuTe kernels for the mixer tail.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.mixer.reference` and needs neither the DSL nor a GPU.
"""

from slinoss.ops.mixer.cute.tail import (
    ROWS,
    THREADS,
    MixerTailGrads,
    mixer_tail,
    mixer_tail_backward,
    mixer_tail_forward,
)

__all__ = [
    "ROWS",
    "THREADS",
    "MixerTailGrads",
    "mixer_tail",
    "mixer_tail_backward",
    "mixer_tail_forward",
]
