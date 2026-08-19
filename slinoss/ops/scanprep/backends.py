"""Backend registry for the bounded parameter maps.

The lookup itself is :class:`slinoss._registry.Registry`, which every operator
shares. This module holds only what is scanprep's own: the two call signatures and
which implementations exist.

The CuTe backend registers only when the DSL imports and a CUDA device is visible.
Importing the DSL on a host without one raises, and an operator that cannot be
imported on a CPU host is not an operator, so the import is guarded and a tree with
no DSL resolves to the reference everywhere.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from slinoss._precision import KERNEL_DTYPES, SUPPORTED_DTYPES
from slinoss._registry import Backend, Registry
from slinoss.ops.scanprep.reference import (
    ScanGrads,
    ScanParams,
    scanprep_bwd_ref,
    scanprep_ref,
)

__all__ = [
    "Backend",
    "ScanPrepBackend",
    "ScanPrepBackward",
    "ScanPrepForward",
    "get",
    "names",
    "register",
    "resolve",
]

REFERENCE = "reference"
CUTE = "cute"


class ScanPrepForward(Protocol):
    """Forward signature every backend implements."""

    def __call__(
        self,
        w_raw: Tensor,
        ls_raw: Tensor,
        tap_raw: Tensor,
        /,
        *,
        w_max: float,
    ) -> ScanParams: ...


class ScanPrepBackward(Protocol):
    """Backward signature every backend implements.

    ``tap_raw`` is absent: the tap map is the identity, so its pullback reads only
    ``dK`` and the forward saves it for nobody.
    """

    def __call__(
        self,
        dtrans: Tensor,
        dK: Tensor,
        w_raw: Tensor,
        ls_raw: Tensor,
        /,
        *,
        w_max: float,
    ) -> ScanGrads: ...


ScanPrepBackend = Backend[ScanPrepForward, ScanPrepBackward]

_REGISTRY: Registry[ScanPrepForward, ScanPrepBackward] = Registry("scanprep")

register = _REGISTRY.register
names = _REGISTRY.names
get = _REGISTRY.get
resolve = _REGISTRY.resolve


register(
    Backend(
        name=REFERENCE,
        forward=scanprep_ref,
        backward=scanprep_bwd_ref,
        device_types=("cpu", "cuda"),
        dtypes=SUPPORTED_DTYPES,
        priority=0,
    )
)


def _register_cute() -> None:
    """Register the CuTe backend if this host can run it."""
    if not torch.cuda.is_available():
        return
    try:
        from slinoss.ops.scanprep.cute.maps import scanprep_backward, scanprep_forward
    except ImportError:
        return
    register(
        Backend(
            name=CUTE,
            forward=scanprep_forward,
            backward=scanprep_backward,
            device_types=("cuda",),
            dtypes=KERNEL_DTYPES,
            priority=10,
        )
    )


_register_cute()
