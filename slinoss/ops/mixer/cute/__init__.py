"""CuTe kernels for the mixer tail.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.mixer.reference` and needs neither the DSL nor a GPU. The public
differentiable entry point is :func:`slinoss.ops.mixer.mixer_tail`, which dispatches
here through the registry.
"""

from slinoss.ops.mixer.cute.tail import (
    ROWS,
    THREADS,
    mixer_tail_backward,
    mixer_tail_forward,
)

__all__ = [
    "ROWS",
    "THREADS",
    "mixer_tail_backward",
    "mixer_tail_forward",
]
