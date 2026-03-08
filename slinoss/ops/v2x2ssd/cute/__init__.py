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
"""

from .api import v2x2ssd_cute

__all__ = ["v2x2ssd_cute"]
