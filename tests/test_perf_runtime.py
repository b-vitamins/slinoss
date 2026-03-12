from __future__ import annotations

import threading

import torch

from slinoss.perf import (
    PerfRecorder,
    call_region,
    current_step,
    note_cache_event,
    record_region,
)


def test_perf_recorder_captures_forward_and_backward_regions() -> None:
    recorder = PerfRecorder(device=torch.device("cpu"))
    x = torch.randn(8, requires_grad=True)
    with recorder.capture_step():
        with record_region("step.total"):
            y = call_region("unit.linear", lambda x_: x_ * 2.0, x)
            loss = y.square().sum()
            with record_region("step.backward"):
                loss.backward()

    capture = recorder.steps[-1]
    regions = capture["regions_ms"]
    assert "step.total" in regions
    assert "step.backward" in regions
    assert "forward.unit.linear" in regions
    assert "backward.unit.linear" in regions
    assert regions["forward.unit.linear"] >= 0.0
    assert regions["backward.unit.linear"] >= 0.0


def test_perf_recorder_active_step_visible_across_threads() -> None:
    recorder = PerfRecorder(device=torch.device("cpu"))
    seen: list[bool] = []

    def worker() -> None:
        seen.append(current_step() is not None)
        note_cache_event("cache.unit", hit=True)

    with recorder.capture_step():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    capture = recorder.steps[-1]
    assert seen == [True]
    assert capture["cache_events"] == {"cache.unit": {"hits": 1, "misses": 0}}
