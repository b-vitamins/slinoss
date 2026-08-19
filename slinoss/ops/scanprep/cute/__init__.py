"""CuTe kernels for the parameter maps.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.scanprep.reference` and needs neither the DSL nor a GPU.
"""

from slinoss.ops.scanprep.cute.maps import (
    KERNEL_DTYPES,
    THREADS,
    ScanGrads,
    scanprep,
    scanprep_backward,
    scanprep_forward,
)

__all__ = [
    "KERNEL_DTYPES",
    "THREADS",
    "ScanGrads",
    "scanprep",
    "scanprep_backward",
    "scanprep_forward",
]
