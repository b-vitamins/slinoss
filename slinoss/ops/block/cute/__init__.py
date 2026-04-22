"""CuTe rowwise helpers for the block FFN training path.

This package owns the non-GEMM pieces around the FFN residual branch:

- RMSNorm forward/backward
- GELU/SwiGLU activation forward/backward

Vendor GEMMs remain responsible for the FFN projections.
"""

from .activation import (
    _ffn_activation_bwd_cute_prevalidated,
    _ffn_activation_fwd_cute_prevalidated,
)
from .norm import BackwardOutputs, _ffn_rmsnorm_bwd_cute_prevalidated
from .norm import _ffn_rmsnorm_fwd_cute_prevalidated

__all__ = [
    "BackwardOutputs",
    "_ffn_activation_bwd_cute_prevalidated",
    "_ffn_activation_fwd_cute_prevalidated",
    "_ffn_rmsnorm_bwd_cute_prevalidated",
    "_ffn_rmsnorm_fwd_cute_prevalidated",
]
