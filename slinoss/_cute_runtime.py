from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from typing import TypeAlias, TypeVar

import torch


TensorSpec: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
_T = TypeVar("_T")
_CACHED_TENSOR_STREAMS: dict[int, torch.cuda.Stream] = {}


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


def make_runtime_tensor_spec_view(
    tensor: torch.Tensor,
    spec: TensorSpec,
) -> torch.Tensor:
    shape, stride = spec
    return torch.as_strided(tensor, size=shape, stride=stride)


def record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
    if not torch.cuda.is_available():
        return
    streams: dict[torch.device, torch.cuda.Stream] = {}
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is None or tensor.device.type != "cuda":
            continue
        ident = id(tensor)
        if ident in seen:
            continue
        stream = streams.get(tensor.device)
        if stream is None:
            stream = torch.cuda.current_stream(device=tensor.device)
            streams[tensor.device] = stream
        tensor.record_stream(stream)
        seen.add(ident)


def _is_current_stream_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def prepare_cached_tensors_on_current_stream(
    *tensors: torch.Tensor | None,
) -> None:
    if not torch.cuda.is_available():
        return
    streams: dict[torch.device, torch.cuda.Stream] = {}
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is None or tensor.device.type != "cuda":
            continue
        ident = id(tensor)
        if ident in seen:
            continue
        stream = streams.get(tensor.device)
        if stream is None:
            stream = torch.cuda.current_stream(device=tensor.device)
            streams[tensor.device] = stream
        previous_stream = _CACHED_TENSOR_STREAMS.get(ident)
        if _is_current_stream_capturing():
            _CACHED_TENSOR_STREAMS[ident] = stream
            seen.add(ident)
            continue
        if previous_stream is not None and previous_stream != stream:
            stream.wait_stream(previous_stream)
        tensor.record_stream(stream)
        _CACHED_TENSOR_STREAMS[ident] = stream
        seen.add(ident)


def launch_tvm_ffi_on_current_stream(
    compiled: Callable[..., _T],
    *runtime_args: object,
) -> _T:
    tensor_args = tuple(arg for arg in runtime_args if isinstance(arg, torch.Tensor))
    result = compiled(*runtime_args)
    record_tensors_on_current_stream(
        *tensor_args,
    )
    return result


__all__ = [
    "ensure_cute_runtime_env",
    "launch_tvm_ffi_on_current_stream",
    "make_runtime_tensor_spec_view",
    "prepare_cached_tensors_on_current_stream",
    "record_tensors_on_current_stream",
]
