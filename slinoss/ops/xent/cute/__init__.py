"""CuTe kernels for the fused cross entropy.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.xent.reference` and needs neither the DSL nor a GPU. The public
differentiable entry point is :func:`slinoss.ops.xent.cross_entropy`, which
dispatches here through the registry.
"""

from slinoss.ops.xent.cute.loss import (
    NO_COLUMN,
    XENT_THREADS,
    xent_backward,
    xent_forward,
)

__all__ = [
    "NO_COLUMN",
    "XENT_THREADS",
    "xent_backward",
    "xent_forward",
]
