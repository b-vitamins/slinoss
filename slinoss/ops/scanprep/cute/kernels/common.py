"""Shared CuTe kernel typing helpers for the ``scanprep`` stack."""

from typing import Any, cast

import cutlass.cute as cute

from slinoss.ops._cute_common import _launchable, _llvm_ptr, _make_layout, _size

_cute_struct = cast(Any, getattr(cute, "struct"))
_cute_cosize = cast(Any, getattr(cute, "cosize"))


def _struct():
    return _cute_struct


def _cosize(layout):
    return _cute_cosize(layout)


__all__ = [
    "_cosize",
    "_launchable",
    "_llvm_ptr",
    "_make_layout",
    "_size",
    "_struct",
]
