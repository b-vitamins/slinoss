"""The capture window an external profiler attaches to.

Both profiler drivers run the target with the capture range set to the CUDA
profiler API, so only the work inside this window is profiled. Warmup,
compilation, and allocator growth happen outside it and never enter a counter.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from slinoss.perf.timing import on_device

__all__ = ["profiler_window"]


@contextmanager
def profiler_window(device: torch.device) -> Iterator[None]:
    """Bracket the region ``ncu`` and ``nsys`` capture.

    Synchronizes on both edges so the window contains whole kernels, and makes
    ``device`` current so both edges drain the device the work is on.

    Args:
        device: The device the profiled work runs on.

    Yields:
        None.

    Raises:
        ValueError: If the device is not CUDA. Both edges synchronize a CUDA
            device and the counters come from one, so there is no window to open
            anywhere else. Checked on entry, before the caller's body runs.
    """
    if device.type != "cuda":
        raise ValueError(f"the capture window needs a cuda device, got {device}")
    with on_device(device):
        torch.cuda.synchronize(device)
        torch.cuda.profiler.start()
        try:
            yield
        finally:
            torch.cuda.synchronize(device)
            torch.cuda.profiler.stop()
