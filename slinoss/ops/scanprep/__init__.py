"""Operator surface for SLinOSS scan preparation."""

from __future__ import annotations

from typing import Any

from .reference import (
    SLinOSSScanPrepCoefficients,
    build_transition_from_polar,
    foh_taps_from_polar,
    principal_angle,
)


def scanprep_cute(*args: Any, **kwargs: Any):
    """Lazily import the CuTe scanprep path so base installs stay importable."""
    from .cute import scanprep_cute as _scanprep_cute

    return _scanprep_cute(*args, **kwargs)


__all__ = [
    "SLinOSSScanPrepCoefficients",
    "principal_angle",
    "build_transition_from_polar",
    "foh_taps_from_polar",
    "scanprep_cute",
]
