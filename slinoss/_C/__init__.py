"""Bindings for the native causal conv1d extension.

Importing this package never fails. The extension is compiled out of tree, so a
tree that has not been built still imports and still runs the pure-PyTorch
reference; :func:`extension` is what raises, and it names the build command.

The import is dynamic because the module does not exist until the build runs, so
a static import would be an unresolvable reference in every unbuilt checkout.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["BUILD_COMMAND", "EXTENSION", "extension", "is_available"]

EXTENSION = "slinoss._C._conv1d"
"""Import path of the compiled module."""

BUILD_COMMAND = "python3 setup.py build_ext --inplace"
"""What builds it, from the repository root."""


def _load() -> tuple[ModuleType | None, ImportError | None]:
    """Import the extension, reporting the failure rather than raising it."""
    try:
        return importlib.import_module(EXTENSION), None
    except ImportError as error:
        return None, error


_MODULE, _ERROR = _load()


def is_available() -> bool:
    """Whether the compiled extension imported."""
    return _MODULE is not None


def extension() -> ModuleType:
    """The compiled extension.

    Returns:
        The module named by :data:`EXTENSION`.

    Raises:
        RuntimeError: If it is not built. The message carries the build command
            and the original import error, because a stale build and an absent
            one fail the same way and only the error tells them apart.
    """
    if _MODULE is None:
        raise RuntimeError(
            f"{EXTENSION} is not built; run {BUILD_COMMAND!r} from the "
            f"repository root. Import error: {_ERROR}"
        )
    return _MODULE
