"""Hardware fingerprint helpers for CuTe forward autotuning."""

from __future__ import annotations

from importlib import import_module

import torch

from .types import HardwareFingerprint


def _cutlass_version() -> str:
    try:
        cutlass = import_module("cutlass")
    except Exception:
        return "unknown"
    return str(getattr(cutlass, "__version__", "unknown"))


def _cuda_runtime_version() -> str:
    try:
        cudart = import_module("cuda.bindings.runtime")

        status, version = cudart.cudaRuntimeGetVersion()
        if int(status) != 0:
            return "unknown"
        major = int(version) // 1000
        minor = (int(version) % 1000) // 10
        return f"{major}.{minor}"
    except Exception:
        return str(torch.version.cuda or "unknown")


def current_hardware_fingerprint(
    *, device_index: int | None = None
) -> HardwareFingerprint:
    """Return a stable fingerprint for the active CUDA device."""

    if not torch.cuda.is_available():
        return HardwareFingerprint(
            arch_tag="cpu",
            device_name="cpu",
            sm_major=0,
            sm_minor=0,
            multiprocessor_count=0,
            shared_memory_per_block_optin=0,
            total_memory_bytes=0,
            cuda_runtime_version=str(torch.version.cuda or "none"),
            torch_cuda_version=str(torch.version.cuda or "none"),
            cutlass_version=_cutlass_version(),
        )

    if device_index is None:
        device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(int(device_index))
    return HardwareFingerprint(
        arch_tag=f"sm_{props.major}{props.minor}",
        device_name=str(props.name),
        sm_major=int(props.major),
        sm_minor=int(props.minor),
        multiprocessor_count=int(getattr(props, "multi_processor_count", 0)),
        shared_memory_per_block_optin=int(
            getattr(props, "shared_memory_per_block_optin", 0)
        ),
        total_memory_bytes=int(getattr(props, "total_memory", 0)),
        cuda_runtime_version=_cuda_runtime_version(),
        torch_cuda_version=str(torch.version.cuda or "unknown"),
        cutlass_version=_cutlass_version(),
    )


__all__ = ["current_hardware_fingerprint"]
