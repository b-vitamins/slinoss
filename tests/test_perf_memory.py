"""Memory forensics: the saved set autograd holds, and the allocator peaks.

The probe runs over a real CPU autograd graph, so the counts are what autograd
actually keeps rather than what a fabricated tensor list claims. Deduplication is
by storage identity, which is the whole point: autograd keeps a storage alive, not
a view.
"""

from __future__ import annotations

import pytest
import torch

from slinoss.perf.memory import (
    MemoryPeaks,
    SavedStorages,
    SavedTensorProbe,
    memory_peaks,
    peak_window,
    reset_memory_peaks,
    saved_storage_bytes,
)
from slinoss.perf.timing import UNATTRIBUTED, measure, region
from slinoss.perf.units import Bytes, Count

CPU = torch.device("cpu")
FLOAT32_BYTES = 4


def _saved(saved_bytes: int) -> SavedStorages:
    """A saved-storage record with a known byte total."""
    return SavedStorages(
        label="hand built",
        storage_count=Count(1),
        save_event_count=Count(1),
        saved_bytes=Bytes(saved_bytes),
        input_bytes=Bytes(0),
        derived_bytes=Bytes(saved_bytes),
        regions=(),
    )


def _probe_two_views() -> SavedStorages:
    """Probe a graph that saves two views of one buffer, several times each."""
    buf = torch.randn(8, requires_grad=True)
    lo = buf[:4]
    hi = buf[4:]
    probe = SavedTensorProbe()
    with probe:
        out = (lo * lo).sum() + (hi * hi).sum()
    assert out.requires_grad
    return probe.report("two views", inputs=(buf,))


# ---------------------------------------------------------------------------
# SavedTensorProbe
# ---------------------------------------------------------------------------


def test_probe_splits_input_bytes_from_derived_bytes() -> None:
    x = torch.randn(4, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        y = x.exp()
        out = (y * x).sum()
    assert out.requires_grad
    saved = probe.report("exp then mul", inputs=(x,))
    assert saved.label == "exp then mul"
    assert saved.storage_count == 2
    assert saved.input_bytes == 4 * FLOAT32_BYTES
    assert saved.derived_bytes == 4 * FLOAT32_BYTES
    assert saved.saved_bytes == saved.input_bytes + saved.derived_bytes


def test_probe_with_no_declared_inputs_calls_everything_derived() -> None:
    x = torch.randn(4, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        out = (x * x).sum()
    assert out.requires_grad
    saved = probe.report("undeclared")
    assert saved.input_bytes == 0
    assert saved.derived_bytes == saved.saved_bytes
    assert saved.saved_bytes == 4 * FLOAT32_BYTES


def test_probe_observes_without_changing_the_graph() -> None:
    x = torch.randn(4, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        out = (x * x).sum()
    out.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, 2.0 * x.detach())


def test_probe_dedupes_by_storage_identity() -> None:
    # Two views of one buffer are one storage, so they cost one buffer.
    saved = _probe_two_views()
    assert saved.storage_count == 1
    assert saved.saved_bytes == 8 * FLOAT32_BYTES
    assert saved.input_bytes == 8 * FLOAT32_BYTES
    assert saved.derived_bytes == 0


def test_save_event_count_exceeds_storage_count() -> None:
    # One buffer saved by several nodes: the events count each save, the storages
    # count the buffer once.
    saved = _probe_two_views()
    assert saved.save_event_count > saved.storage_count


def test_probe_attributes_saves_to_the_open_region() -> None:
    x = torch.randn(4, requires_grad=True)
    probe = SavedTensorProbe()

    def body() -> None:
        with region("x"), probe:
            out = (x * x).sum()
        assert out.requires_grad

    measure(body, label="probe", iters=1, warmup=0, device=CPU)
    saved = probe.report("regioned", inputs=(x,))
    assert [r.label for r in saved.regions] == ["x"]
    assert saved.regions[0].storage_count == saved.storage_count
    assert saved.regions[0].save_event_count == saved.save_event_count
    assert saved.regions[0].saved_bytes == saved.saved_bytes


def test_probe_labels_saves_outside_a_region_unattributed() -> None:
    # A blank label prints as a blank table cell, which reads as a bug in the
    # table rather than as a save taken outside every region.
    x = torch.randn(4, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        out = (x * x).sum()
    assert out.requires_grad
    assert [r.label for r in probe.report("bare").regions] == [UNATTRIBUTED]


def test_probe_labels_saves_inside_a_measurement_but_outside_a_region() -> None:
    x = torch.randn(4, requires_grad=True)
    probe = SavedTensorProbe()

    def body() -> None:
        with probe:
            out = (x * x).sum()
        assert out.requires_grad

    measure(body, label="probe", iters=1, warmup=0, device=CPU)
    assert [r.label for r in probe.report("unregioned").regions] == [UNATTRIBUTED]


def test_probe_reports_nothing_for_an_empty_graph() -> None:
    probe = SavedTensorProbe()
    with probe:
        pass
    saved = probe.report("empty")
    assert saved.storage_count == 0
    assert saved.save_event_count == 0
    assert saved.saved_bytes == 0
    assert saved.regions == ()


# ---------------------------------------------------------------------------
# SavedStorages
# ---------------------------------------------------------------------------


def test_saved_mib_is_1024_based() -> None:
    assert _saved(2 * 1024 * 1024).saved_mib == 2.0
    assert _saved(1_000_000).saved_mib != 1.0
    assert _saved(0).saved_mib == 0.0


# ---------------------------------------------------------------------------
# saved_storage_bytes
# ---------------------------------------------------------------------------


def test_saved_storage_bytes_counts_aliases_once() -> None:
    buf = torch.empty(1024, dtype=torch.float32)
    total = 1024 * FLOAT32_BYTES
    assert saved_storage_bytes([buf]) == total
    assert saved_storage_bytes([buf[:512], buf[512:], buf]) == total


def test_saved_storage_bytes_sums_distinct_storages() -> None:
    a = torch.empty(1024, dtype=torch.float32)
    b = torch.empty(16, dtype=torch.float32)
    assert saved_storage_bytes([a, b]) == (1024 + 16) * FLOAT32_BYTES


def test_saved_storage_bytes_of_nothing_is_zero() -> None:
    assert saved_storage_bytes([]) == 0


# ---------------------------------------------------------------------------
# Allocator peaks
# ---------------------------------------------------------------------------


def test_memory_peaks_on_cpu_are_zero() -> None:
    # The CUDA caching allocator is the only instrumented allocator, so a CPU
    # device reports zeros rather than a fabricated figure.
    peaks = memory_peaks("cpu window", CPU)
    assert peaks.label == "cpu window"
    assert peaks.peak_allocated_bytes == 0
    assert peaks.peak_reserved_bytes == 0


def test_reset_memory_peaks_on_cpu_is_a_noop() -> None:
    reset_memory_peaks(CPU)
    assert memory_peaks("after reset", CPU).peak_allocated_bytes == 0


def test_peak_window_on_cpu_fills_the_sink_on_exit() -> None:
    with peak_window("cpu window", CPU) as sink:
        assert sink == []
    assert len(sink) == 1
    assert sink[0].label == "cpu window"
    assert sink[0].peak_allocated_bytes == 0


def test_peak_window_fills_the_sink_even_when_the_body_raises() -> None:
    sink: list[MemoryPeaks] = []
    with pytest.raises(RuntimeError, match="body failed"), peak_window("w", CPU) as got:
        sink = got
        raise RuntimeError("body failed")
    assert len(sink) == 1
    assert sink[0].label == "w"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_memory_peaks_on_cuda_read_the_allocator() -> None:
    device = torch.device("cuda")
    reset_memory_peaks(device)
    buf = torch.empty(1 << 20, dtype=torch.float32, device=device)
    wanted = buf.numel() * buf.element_size()
    peaks = memory_peaks("cuda window", device)
    assert peaks.peak_allocated_bytes >= wanted
    assert peaks.peak_reserved_bytes >= peaks.peak_allocated_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_reset_memory_peaks_on_cuda_clears_the_marks() -> None:
    device = torch.device("cuda")
    big = torch.empty(1 << 22, dtype=torch.float32, device=device)
    high = big.numel() * big.element_size()
    del big
    before = memory_peaks("before reset", device)
    assert before.peak_allocated_bytes >= high
    reset_memory_peaks(device)
    after = memory_peaks("after reset", device)
    assert after.peak_allocated_bytes < before.peak_allocated_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_peak_window_on_cuda_captures_the_body_high_water_mark() -> None:
    device = torch.device("cuda")
    with peak_window("cuda window", device) as sink:
        tmp = torch.empty(1 << 20, dtype=torch.float32, device=device)
        wanted = tmp.numel() * tmp.element_size()
        del tmp
    assert sink[0].peak_allocated_bytes >= wanted
