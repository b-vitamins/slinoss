"""Backend registry for the block norm and activation.

The lookup itself is :class:`slinoss._registry.Registry`, which every operator
shares. This module holds only what is the block's own: the three pairs of call
signatures and which implementations exist.

Three families, so three registries. The plain norm, the norm fused with the
residual add, and the activation take different operands and are different
kernels, so a name resolved for one says nothing about the others. The lookup
functions carry the family name -- ``rmsnorm_resolve``,
``rmsnorm_residual_resolve``, ``swiglu_resolve`` -- rather than three shadowing
copies of one ``resolve``: with one set of names per module the last binding wins
and two of the three families become unreachable, and a prefixed name also makes
a call site say which operator it is resolving.

The CuTe backends register only when the DSL imports and a CUDA device is visible.
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
from slinoss.ops.block.reference import (
    NormResidual,
    NormResidualGrads,
    RMSNormGrads,
    SwiGLUGrads,
    rmsnorm_bwd_ref,
    rmsnorm_ref,
    rmsnorm_residual_bwd_ref,
    rmsnorm_residual_ref,
    swiglu_bwd_ref,
    swiglu_ref,
)

__all__ = [
    "Backend",
    "NormResidualBackend",
    "NormResidualBackward",
    "NormResidualForward",
    "RMSNormBackend",
    "RMSNormBackward",
    "RMSNormForward",
    "SwiGLUBackend",
    "SwiGLUBackward",
    "SwiGLUForward",
    "rmsnorm_get",
    "rmsnorm_names",
    "rmsnorm_register",
    "rmsnorm_residual_get",
    "rmsnorm_residual_names",
    "rmsnorm_residual_register",
    "rmsnorm_residual_resolve",
    "rmsnorm_resolve",
    "swiglu_get",
    "swiglu_names",
    "swiglu_register",
    "swiglu_resolve",
]

REFERENCE = "reference"
CUTE = "cute"


class RMSNormForward(Protocol):
    """Forward signature every plain-norm backend implements."""

    def __call__(self, x: Tensor, weight: Tensor, /, *, eps: float) -> Tensor: ...


class RMSNormBackward(Protocol):
    """Backward signature every plain-norm backend implements.

    The two operands and the cotangent, and nothing else: the row scale is
    recomputed, so no forward intermediate crosses the boundary.
    """

    def __call__(
        self,
        dout: Tensor,
        x: Tensor,
        weight: Tensor,
        /,
        *,
        eps: float,
    ) -> RMSNormGrads: ...


class NormResidualForward(Protocol):
    """Forward signature every fused add-and-norm backend implements.

    ``residual`` is None for the first block of a stack. That is a distinct case
    rather than a zero tensor: a zero add costs a whole-tensor read and write.
    """

    def __call__(
        self,
        x: Tensor,
        residual: Tensor | None,
        weight: Tensor,
        /,
        *,
        eps: float,
    ) -> NormResidual: ...


class NormResidualBackward(Protocol):
    """Backward signature every fused add-and-norm backend implements.

    Both cotangents are optional and either may be absent, because the forward
    has two outputs and a caller need not consume both. The wide sum the forward
    returned is not an argument: the backward re-forms it from ``x`` and
    ``residual``.
    """

    def __call__(
        self,
        dnormed: Tensor | None,
        dresidual: Tensor | None,
        x: Tensor,
        residual: Tensor | None,
        weight: Tensor,
        /,
        *,
        eps: float,
    ) -> NormResidualGrads: ...


class SwiGLUForward(Protocol):
    """Forward signature every activation backend implements."""

    def __call__(self, gate: Tensor, up: Tensor, /) -> Tensor: ...


class SwiGLUBackward(Protocol):
    """Backward signature every activation backend implements.

    The logistic of the gate appears in both gradients and in the forward, and is
    recomputed here rather than saved.
    """

    def __call__(self, dout: Tensor, gate: Tensor, up: Tensor, /) -> SwiGLUGrads: ...


RMSNormBackend = Backend[RMSNormForward, RMSNormBackward]
NormResidualBackend = Backend[NormResidualForward, NormResidualBackward]
SwiGLUBackend = Backend[SwiGLUForward, SwiGLUBackward]

_RMSNORM: Registry[RMSNormForward, RMSNormBackward] = Registry("rmsnorm")
_RESIDUAL: Registry[NormResidualForward, NormResidualBackward] = Registry(
    "rmsnorm_residual"
)
_SWIGLU: Registry[SwiGLUForward, SwiGLUBackward] = Registry("swiglu")

rmsnorm_register = _RMSNORM.register
rmsnorm_names = _RMSNORM.names
rmsnorm_get = _RMSNORM.get
rmsnorm_resolve = _RMSNORM.resolve

rmsnorm_residual_register = _RESIDUAL.register
rmsnorm_residual_names = _RESIDUAL.names
rmsnorm_residual_get = _RESIDUAL.get
rmsnorm_residual_resolve = _RESIDUAL.resolve

swiglu_register = _SWIGLU.register
swiglu_names = _SWIGLU.names
swiglu_get = _SWIGLU.get
swiglu_resolve = _SWIGLU.resolve


rmsnorm_register(
    Backend(
        name=REFERENCE,
        forward=rmsnorm_ref,
        backward=rmsnorm_bwd_ref,
        device_types=("cpu", "cuda"),
        dtypes=SUPPORTED_DTYPES,
        priority=0,
    )
)

rmsnorm_residual_register(
    Backend(
        name=REFERENCE,
        forward=rmsnorm_residual_ref,
        backward=rmsnorm_residual_bwd_ref,
        device_types=("cpu", "cuda"),
        dtypes=SUPPORTED_DTYPES,
        priority=0,
    )
)

swiglu_register(
    Backend(
        name=REFERENCE,
        forward=swiglu_ref,
        backward=swiglu_bwd_ref,
        device_types=("cpu", "cuda"),
        dtypes=SUPPORTED_DTYPES,
        priority=0,
    )
)


def _register_cute() -> None:
    """Register the CuTe backends if this host can run them.

    One guard for all three families: the kernels sit in one package behind one
    DSL import, so either every family gets its kernel backend or none does.
    """
    if not torch.cuda.is_available():
        return
    try:
        from slinoss.ops.block.cute.act import swiglu_backward, swiglu_forward
        from slinoss.ops.block.cute.norm import (
            rmsnorm_backward,
            rmsnorm_forward,
            rmsnorm_residual_backward,
            rmsnorm_residual_forward,
        )
    except ImportError:
        return
    rmsnorm_register(
        Backend(
            name=CUTE,
            forward=rmsnorm_forward,
            backward=rmsnorm_backward,
            device_types=("cuda",),
            dtypes=KERNEL_DTYPES,
            priority=10,
        )
    )
    rmsnorm_residual_register(
        Backend(
            name=CUTE,
            forward=rmsnorm_residual_forward,
            backward=rmsnorm_residual_backward,
            device_types=("cuda",),
            dtypes=KERNEL_DTYPES,
            priority=10,
        )
    )
    swiglu_register(
        Backend(
            name=CUTE,
            forward=swiglu_forward,
            backward=swiglu_backward,
            device_types=("cuda",),
            dtypes=KERNEL_DTYPES,
            priority=10,
        )
    )


_register_cute()
