"""The scan's parameter frontier: bounded maps, tap packing, ``B``/``C`` permute."""

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
    LS_COLUMN,
    PARAM_COLS,
    ROTVEC_COLUMNS,
    TAP_COLUMNS,
    ScanGrads,
    ScanParams,
    bounded_logscale,
    bounded_rotvec,
    pack_params,
    scanprep_bwd_ref,
    scanprep_ref,
)

__all__ = [
    "LS_COLUMN",
    "PARAM_COLS",
    "ROTVEC_COLUMNS",
    "TAP_COLUMNS",
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
    "pack_params",
    "register",
    "resolve",
    "scanprep",
    "scanprep_bwd_ref",
    "scanprep_ref",
]
