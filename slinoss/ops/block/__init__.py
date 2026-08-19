"""Fused norm and activation for the block."""

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
    "NormResidual",
    "NormResidualGrads",
    "RMSNormGrads",
    "SwiGLUGrads",
    "rmsnorm_bwd_ref",
    "rmsnorm_ref",
    "rmsnorm_residual_bwd_ref",
    "rmsnorm_residual_ref",
    "swiglu_bwd_ref",
    "swiglu_ref",
]
