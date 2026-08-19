"""Fused norm and activation for the block."""

from slinoss.ops.block.reference import (
    NormResidual,
    rmsnorm_ref,
    rmsnorm_residual_ref,
    swiglu_ref,
)

__all__ = ["NormResidual", "rmsnorm_ref", "rmsnorm_residual_ref", "swiglu_ref"]
