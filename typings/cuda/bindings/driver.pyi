"""Partial stub for ``cuda.bindings.driver``.

The module is compiled and ships no types, so the driver handle a kernel launch
takes reads as ``Any`` and cannot be named in an annotation. One class is declared
here, because it is the one this package puts in a signature: every launcher takes
a stream, and without a type for it nothing distinguishes a stream from an int.

``__getattr__`` keeps the stub the size of that claim. Do not enumerate the driver
API here.
"""

from typing import Any

class CUstream:
    """An opaque CUDA stream handle, built from ``torch``'s stream pointer."""

    def __init__(self, handle: int, /) -> None: ...

def __getattr__(name: str) -> Any: ...
