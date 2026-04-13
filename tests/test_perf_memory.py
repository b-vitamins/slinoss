from __future__ import annotations

import torch

from slinoss.perf import EagerMemoryForensics, PerfRecorder, record_region


def test_eager_memory_forensics_tracks_saved_tensors_and_region_exits() -> None:
    tracker = EagerMemoryForensics(device=torch.device("cpu"))
    recorder = PerfRecorder(device=torch.device("cpu"))
    x = torch.randn(8, requires_grad=True)

    with tracker.capture():
        with recorder.capture_step():
            with record_region("step.total"):
                with record_region("forward.unit.square"):
                    y = x.square().sum()
                with record_region("step.backward"):
                    y.backward()

    saved_rows = tracker.saved_tensors_by_region()
    assert saved_rows
    assert saved_rows[0]["label"] == "forward.unit.square"
    assert int(saved_rows[0]["unique_saved_bytes"]) > 0
    assert int(saved_rows[0]["save_event_count"]) > 0

    top_regions = tracker.top_region_exit_allocated(top_k=4)
    labels = {str(row["label"]) for row in top_regions}
    assert "forward.unit.square" in labels
    assert "step.backward" in labels
