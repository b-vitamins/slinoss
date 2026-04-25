"""CuTe-facing API surface for the mixer-tail rowwise backend.

This package owns the fused rowwise stage that surrounds the output projection:

- forward: skip/add + SiLU gate + RMSNorm
- backward: rowwise gradients for scan output, gate, skip, and norm weight

Ahead-of-time export/load helpers are available under
:mod:`slinoss.ops.mixer.cute.aot`.
"""

from . import aot
from .bwd import BackwardOutputs, _mixer_tail_rowwise_bwd_cute_prevalidated
from .fwd import _mixer_tail_rowwise_fwd_cute_prevalidated

__all__ = [
    "BackwardOutputs",
    "_mixer_tail_rowwise_bwd_cute_prevalidated",
    "_mixer_tail_rowwise_fwd_cute_prevalidated",
    "aot",
]
