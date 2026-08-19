"""CuTe kernels for the block norm and activation. Forward only.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.block.reference` and needs neither the DSL nor a GPU.
"""

from slinoss.ops.block.cute.act import (
    ACT_THREADS,
    VECTOR_BYTES,
    swiglu_forward,
    swiglu_fwd,
    swiglu_fwd_kernel,
)
from slinoss.ops.block.cute.norm import (
    NORM_THREADS,
    PARTIAL_TILE,
    SCALE_TILE,
    WARPS,
    norm_smem_bytes,
    rmsnorm_forward,
    rmsnorm_fwd,
    rmsnorm_fwd_kernel,
    rmsnorm_residual_forward,
    rmsnorm_residual_fwd,
    rmsnorm_residual_fwd_kernel,
)

__all__ = [
    "ACT_THREADS",
    "NORM_THREADS",
    "PARTIAL_TILE",
    "SCALE_TILE",
    "VECTOR_BYTES",
    "WARPS",
    "norm_smem_bytes",
    "rmsnorm_forward",
    "rmsnorm_fwd",
    "rmsnorm_fwd_kernel",
    "rmsnorm_residual_forward",
    "rmsnorm_residual_fwd",
    "rmsnorm_residual_fwd_kernel",
    "swiglu_forward",
    "swiglu_fwd",
    "swiglu_fwd_kernel",
]
