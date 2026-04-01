"""Reference implementations for the v2x2 SSD operator."""

from __future__ import annotations

from typing import Any, Callable, cast

import torch

from .reference import (
    chunk_increment,
    chunk_scan,
    state_passing,
    v2x2ssm,
    v2x2ssd,
    v2x2ssd_ref,
)


def _v2x2ssd_cute_impl(*args: Any, **kwargs: Any):
    """Compiler boundary for the CuTe scan JIT/runtime path."""
    from .cute import v2x2ssd_cute as _v2x2ssd_cute

    return _v2x2ssd_cute(*args, **kwargs)


v2x2ssd_cute = cast(
    Callable[..., Any],
    torch.compiler.disable(_v2x2ssd_cute_impl),
)


__all__ = [
    "v2x2ssm",
    "v2x2ssd_ref",
    "chunk_increment",
    "state_passing",
    "chunk_scan",
    "v2x2ssd",
    "v2x2ssd_cute",
]
