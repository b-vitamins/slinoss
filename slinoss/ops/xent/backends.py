"""Backend registry for the fused cross entropy.

The lookup itself is :class:`slinoss._registry.Registry`, which every operator
shares. This module holds only what is this operator's own: the two call signatures
and which implementations exist.

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
from slinoss.ops.xent.reference import (
    XentGrads,
    XentState,
    xent_bwd_ref,
    xent_ref,
)

__all__ = [
    "Backend",
    "XentBackend",
    "XentBackward",
    "XentForward",
    "get",
    "names",
    "register",
    "resolve",
]

REFERENCE = "reference"
CUTE = "cute"


class XentForward(Protocol):
    """Forward signature every backend implements.

    ``classes`` is keyword-only and separate from the operand width, which is what
    keeps a padded class axis out of the partition function.
    """

    def __call__(
        self,
        logits: Tensor,
        labels: Tensor,
        /,
        *,
        classes: int,
    ) -> XentState: ...


class XentBackward(Protocol):
    """Backward signature every backend implements.

    ``lse`` is the forward's own output rather than a recomputation: 4 B per row
    against a second pass over the logits.
    """

    def __call__(
        self,
        dloss: Tensor,
        logits: Tensor,
        labels: Tensor,
        lse: Tensor,
        /,
        *,
        classes: int,
    ) -> XentGrads: ...


XentBackend = Backend[XentForward, XentBackward]

_REGISTRY: Registry[XentForward, XentBackward] = Registry("cross_entropy")

register = _REGISTRY.register
names = _REGISTRY.names
get = _REGISTRY.get
resolve = _REGISTRY.resolve

register(
    Backend(
        name=REFERENCE,
        forward=xent_ref,
        backward=xent_bwd_ref,
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
        from slinoss.ops.xent.cute.loss import xent_backward, xent_forward
    except ImportError:
        return
    register(
        Backend(
            name=CUTE,
            forward=xent_forward,
            backward=xent_backward,
            device_types=("cuda",),
            dtypes=KERNEL_DTYPES,
            priority=10,
        )
    )


_register_cute()
