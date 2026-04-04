"""CuTe-facing API surface for the v2x2ssd operator.

The public tensor contract matches ``slinoss.ops.v2x2ssd.reference`` exactly:

- ``U``: ``(B, H, T, P)``
- ``M``: ``(B, H, T, 2)``
- ``K``: ``(B, H, T, 2, 2)``
- ``B, C``: ``(B, H, T, D)`` with ``D = 2N`` interleaved complex lanes
- state: ``(B, H, P, D)``

The staged forward decomposition remains:

- ``chunk_increment``
- ``state_passing``
- ``chunk_scan``

Ahead-of-time export/load helpers live under :mod:`slinoss.ops.v2x2ssd.cute.aot`.
Autotuning helpers live under :mod:`slinoss.ops.v2x2ssd.cute.tuning`.
"""

from . import aot
from . import tuning
from .api import v2x2ssd_cute
from .decode import mixer_decode_step_cute

__all__ = ["aot", "tuning", "v2x2ssd_cute", "mixer_decode_step_cute"]
