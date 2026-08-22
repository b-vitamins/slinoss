"""Partial stub for ``cutlass.cute.nvgpu``. See ``cutlass/__init__.pyi``.

The subpackage needs a file of its own because a package's ``__getattr__`` does not
resolve a submodule import: ``from cutlass.cute.nvgpu import cpasync`` is a module
lookup on the filesystem, not an attribute lookup on ``cutlass.cute``.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
