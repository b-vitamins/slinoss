"""Operator surface for SLinOSS scan preparation."""

from __future__ import annotations

from typing import Any, Callable, cast

import torch

from .reference import (
    SLinOSSScanPrepCoefficients,
    build_transition_from_polar,
    foh_taps_from_polar,
    principal_angle,
)


def scanprep_cute(*args: Any, **kwargs: Any):
    """Compiler boundary for the CuTe scanprep JIT/runtime path."""
    from . import cute

    return cute.scanprep_cute(*args, **kwargs)


scanprep_cute = cast(Callable[..., Any], torch.compiler.disable(scanprep_cute))


__all__ = [
    "SLinOSSScanPrepCoefficients",
    "principal_angle",
    "build_transition_from_polar",
    "foh_taps_from_polar",
    "scanprep_cute",
]
