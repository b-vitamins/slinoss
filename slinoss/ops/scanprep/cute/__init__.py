"""CuTe-facing API surface for the ``scanprep`` operator.

The public tensor contract matches ``slinoss.ops.scanprep.reference`` exactly.
Ahead-of-time export/load helpers are available under
:mod:`slinoss.ops.scanprep.cute.aot`.
"""

from . import aot
from .api import scanprep_cute

__all__ = ["aot", "scanprep_cute"]
