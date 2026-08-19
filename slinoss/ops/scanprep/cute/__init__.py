"""CuTe kernels for the parameter maps.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.scanprep.reference` and needs neither the DSL nor a GPU. The
public differentiable entry point is :func:`slinoss.ops.scanprep.scanprep`, which
dispatches here through the registry.
"""

from slinoss.ops.scanprep.cute.maps import (
    THREADS,
    scanprep_backward,
    scanprep_forward,
)

__all__ = [
    "THREADS",
    "scanprep_backward",
    "scanprep_forward",
]
