"""Shared CuTe kernel typing helpers for the ``scanprep`` stack."""

from typing import Any, cast

import cutlass.cute as cute

_cute_make_layout = cast(Any, getattr(cute, "make_layout"))
_cute_size = cast(Any, getattr(cute, "size"))
_cute_struct = cast(Any, getattr(cute, "struct"))
_cute_cosize = cast(Any, getattr(cute, "cosize"))


def _make_layout(*args, **kwargs):
    return _cute_make_layout(*args, **kwargs)


def _size(*args, **kwargs):
    return _cute_size(*args, **kwargs)


def _struct():
    return _cute_struct


def _cosize(layout):
    return _cute_cosize(layout)


def _launchable(kernel_call):
    return cast(Any, kernel_call)


def _llvm_ptr(value):
    return cast(Any, value).llvm_ptr


__all__ = [
    "_cosize",
    "_launchable",
    "_llvm_ptr",
    "_make_layout",
    "_size",
    "_struct",
]
