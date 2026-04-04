from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, TypeAlias
from importlib import import_module

import torch


TensorSpec: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]


def _cute_module():
    return import_module("cutlass.cute")


def _default_cute_cache_dir() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return root / "slinoss" / "cute-dsl"


def ensure_cute_runtime_env() -> None:
    os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")
    cache_dir = os.environ.get("CUTE_DSL_CACHE_DIR")
    if not cache_dir:
        cache_dir = str(_default_cute_cache_dir())
        os.environ["CUTE_DSL_CACHE_DIR"] = cache_dir
    Path(cache_dir).expanduser().mkdir(parents=True, exist_ok=True)


def _compact_stride_order(stride: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        mode
        for mode, _ in sorted(
            enumerate(int(step) for step in stride),
            key=lambda item: (-item[1], item[0]),
        )
    )


def make_runtime_tensor_spec_view(
    tensor: torch.Tensor,
    spec: TensorSpec,
) -> torch.Tensor:
    shape, stride = spec
    return torch.as_strided(tensor, size=shape, stride=stride)


def _spec_stride_order(spec: TensorSpec) -> tuple[int, ...]:
    _shape, stride = spec
    return _compact_stride_order(tuple(int(step) for step in stride))


def _deduce_leading_dim(tensor: torch.Tensor) -> int | None:
    stride_one_modes = [i for i, step in enumerate(tensor.stride()) if int(step) == 1]
    if len(stride_one_modes) == 1:
        return int(stride_one_modes[0])
    return None


def to_cute_static_tensor(
    tensor: torch.Tensor,
    *,
    align: int,
):
    ensure_cute_runtime_env()
    cute = _cute_module()
    return cute.runtime.from_dlpack(
        tensor,
        assumed_align=int(align),
        use_32bit_stride=True,
        enable_tvm_ffi=True,
    )


def to_cute_layout_dynamic_tensor(
    tensor: torch.Tensor,
    *,
    align: int,
):
    cute_tensor = to_cute_static_tensor(tensor, align=align)
    leading_dim = _deduce_leading_dim(tensor)
    if leading_dim is None:
        return cute_tensor.mark_layout_dynamic()
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def to_cute_compact_dynamic_tensor(
    tensor: torch.Tensor,
    *,
    align: int,
    dynamic_modes: Iterable[int],
    stride_order: tuple[int, ...] | None = None,
):
    cute_tensor = to_cute_static_tensor(tensor, align=align)
    if stride_order is None:
        stride_order = _compact_stride_order(
            tuple(int(step) for step in tensor.stride())
        )
    for mode in dynamic_modes:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=int(mode),
            stride_order=stride_order,
        )
    return cute_tensor


def to_cute_runtime_tensor_view(
    tensor: torch.Tensor,
    spec: TensorSpec,
    *,
    align: int,
    dynamic_modes: Iterable[int],
    compact: bool = True,
):
    runtime_view = make_runtime_tensor_spec_view(tensor, spec)
    if not compact:
        return to_cute_layout_dynamic_tensor(runtime_view, align=align)
    return to_cute_compact_dynamic_tensor(
        runtime_view,
        align=align,
        dynamic_modes=dynamic_modes,
        stride_order=_spec_stride_order(spec),
    )


__all__ = [
    "ensure_cute_runtime_env",
    "make_runtime_tensor_spec_view",
    "to_cute_runtime_tensor_view",
    "to_cute_compact_dynamic_tensor",
    "to_cute_layout_dynamic_tensor",
    "to_cute_static_tensor",
]
