"""CuTe kernels for the parameter frontier.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.scanprep.reference` and needs neither the DSL nor a GPU. The
public differentiable entry point is :func:`slinoss.ops.scanprep.scanprep`, which
dispatches here through the registry.
"""

from slinoss.ops.scanprep.cute.frontier import (
    THREADS,
    TILE_TOKENS,
    scanprep_backward,
    scanprep_forward,
)

__all__ = [
    "THREADS",
    "TILE_TOKENS",
    "scanprep_backward",
    "scanprep_forward",
]
