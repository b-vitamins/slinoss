"""CUDA-event benchmarking helpers for forward autotuning."""

from __future__ import annotations

from collections.abc import Callable

import torch


def benchmark_cuda_callable(
    run_once: Callable[[], None],
    *,
    device: torch.device,
    warmup_iterations: int,
    timed_iterations: int,
    before_iteration: Callable[[], None] | None = None,
) -> float:
    """Return mean kernel wall time in milliseconds for ``run_once``.

    ``before_iteration`` is executed before each trial but excluded from the
    measured region. This keeps zeroing/reset work from distorting candidate
    selection for stateful kernels.
    """

    if device.type != "cuda":
        raise ValueError("benchmark_cuda_callable requires a CUDA device.")

    resolved_device = torch.device(
        device.type,
        torch.cuda.current_device() if device.index is None else int(device.index),
    )
    torch.cuda.synchronize(resolved_device)

    for _ in range(int(warmup_iterations)):
        if before_iteration is not None:
            before_iteration()
        run_once()
    torch.cuda.synchronize(resolved_device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(int(timed_iterations)):
        if before_iteration is not None:
            before_iteration()
        start_event.record()
        run_once()
        end_event.record()
        end_event.synchronize()
        total_ms += float(start_event.elapsed_time(end_event))
    return total_ms / float(int(timed_iterations))


__all__ = ["benchmark_cuda_callable"]
