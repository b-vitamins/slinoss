"""Parameter maps: rotation vector, log-scale, first-order-hold taps."""

from slinoss.ops.scanprep.backends import (
    Backend,
    ScanPrepBackend,
    ScanPrepBackward,
    ScanPrepForward,
    get,
    names,
    register,
    resolve,
)
from slinoss.ops.scanprep.interface import ScanPrepFunction, scanprep
from slinoss.ops.scanprep.reference import (
    ScanGrads,
    ScanParams,
    bounded_logscale,
    bounded_rotvec,
    scanprep_bwd_ref,
    scanprep_ref,
)

__all__ = [
    "Backend",
    "ScanGrads",
    "ScanParams",
    "ScanPrepBackend",
    "ScanPrepBackward",
    "ScanPrepForward",
    "ScanPrepFunction",
    "bounded_logscale",
    "bounded_rotvec",
    "get",
    "names",
    "register",
    "resolve",
    "scanprep",
    "scanprep_bwd_ref",
    "scanprep_ref",
]
