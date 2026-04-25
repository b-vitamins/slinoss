"""CuTe-facing API surface for the v2x2ssd operator.

The public tensor contract matches ``slinoss.ops.v2x2ssd.reference`` exactly:

- ``U``: ``(B, H, T, P)``
- ``M``: ``(B, H, T, 2)``
- ``K``: ``(B, H, T, 2, 2)``
- ``B, C``: ``(B, H, T, D)`` with ``D = 2N`` interleaved complex lanes
- state: ``(B, H, P, D)``

The public CuTe path is the combined ``v2x2ssd_cute`` operator. Its host
boundary launches the internal forward kernels as one compiled call site, and
the backward path rematerializes forward boundary metadata through a private
compiled launcher.

Ahead-of-time export/load helpers are available under
:mod:`slinoss.ops.v2x2ssd.cute.aot`.
Autotuning helpers are available under :mod:`slinoss.ops.v2x2ssd.cute.tuning`.
"""

from . import aot
from . import tuning
from .api import v2x2ssd_cute
from .step import mixer_decode_step_cute

__all__ = ["aot", "tuning", "v2x2ssd_cute", "mixer_decode_step_cute"]
