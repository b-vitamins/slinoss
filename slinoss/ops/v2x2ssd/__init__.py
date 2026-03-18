"""Reference implementations for the v2x2 SSD operator."""

from __future__ import annotations

from typing import Any

from .reference import (
    chunk_increment,
    chunk_scan,
    state_passing,
    v2x2ssm,
    v2x2ssd,
    v2x2ssd_ref,
)


def v2x2ssd_cute(*args: Any, **kwargs: Any):
    """Lazily import the CuTe scan path so base installs stay importable."""
    from .cute import v2x2ssd_cute as _v2x2ssd_cute

    return _v2x2ssd_cute(*args, **kwargs)


__all__ = [
    "v2x2ssm",
    "v2x2ssd_ref",
    "chunk_increment",
    "state_passing",
    "chunk_scan",
    "v2x2ssd",
    "v2x2ssd_cute",
]
