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
    # Declaring no input makes the same saved set wholly derived.
    bare = probe.report("undeclared")
    assert bare.saved_bytes == saved.saved_bytes
    assert bare.input_bytes == 0
    assert bare.derived_bytes == bare.saved_bytes
    # The hooks pass the tensor through, so the backward is the one the graph
    # would have run unprobed.
    out.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, (1.0 + x.detach()) * x.detach().exp())


def test_probe_dedupes_by_storage_identity() -> None:
    # Two views of one buffer are one storage, so they cost one buffer.
    buf = torch.randn(8, requires_grad=True)
    lo = buf[:4]
    hi = buf[4:]
    probe = SavedTensorProbe()
    with probe:
        out = (lo * lo).sum() + (hi * hi).sum()
    assert out.requires_grad
    saved = probe.report("two views", inputs=(buf,))
    assert saved.storage_count == 1
    assert saved.saved_bytes == 8 * FLOAT32_BYTES
    assert saved.input_bytes == 8 * FLOAT32_BYTES
    assert saved.derived_bytes == 0
    # The events count each save of that one buffer, the storages count it once.
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
    # table rather than as a save taken outside every region. Both ways of having
    # no region are covered: no recorder at all, and a recorder with nothing open.
    x = torch.randn(4, requires_grad=True)
    bare = SavedTensorProbe()
    with bare:
        out = (x * x).sum()
    assert out.requires_grad
    assert [r.label for r in bare.report("bare").regions] == [UNATTRIBUTED]

    unregioned = SavedTensorProbe()

    def body() -> None:
        with unregioned:
            inner = (x * x).sum()
        assert inner.requires_grad

    measure(body, label="probe", iters=1, warmup=0, device=CPU)
    assert [r.label for r in unregioned.report("unregioned").regions] == [UNATTRIBUTED]


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


def test_saved_storage_bytes_counts_each_storage_once() -> None:
    buf = torch.empty(1024, dtype=torch.float32)
    total = 1024 * FLOAT32_BYTES
    assert saved_storage_bytes([buf]) == total
    assert saved_storage_bytes([buf[:512], buf[512:], buf]) == total
    other = torch.empty(16, dtype=torch.float32)
    assert saved_storage_bytes([buf, other]) == total + 16 * FLOAT32_BYTES
    assert saved_storage_bytes([]) == 0


# ---------------------------------------------------------------------------
# Allocator peaks
# ---------------------------------------------------------------------------


def test_allocator_peaks_on_cpu_are_zero() -> None:
    # The CUDA caching allocator is the only instrumented allocator, so a CPU
    # device reports zeros rather than a fabricated figure, and the reset is a noop.
    peaks = memory_peaks("cpu window", CPU)
    assert peaks.label == "cpu window"
    assert peaks.peak_allocated_bytes == 0
    assert peaks.peak_reserved_bytes == 0
    reset_memory_peaks(CPU)
    assert memory_peaks("after reset", CPU).peak_allocated_bytes == 0
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
def test_allocator_peaks_on_cuda_read_and_clear_the_marks() -> None:
    device = torch.device("cuda")
    reset_memory_peaks(device)
    big = torch.empty(1 << 22, dtype=torch.float32, device=device)
    high = big.numel() * big.element_size()
    before = memory_peaks("cuda window", device)
    assert before.peak_allocated_bytes >= high
    # Reserved above allocated is fragmentation, and never the other way round.
    assert before.peak_reserved_bytes >= before.peak_allocated_bytes
    # The mark survives the free, so it is a high-water mark and not a live total.
    del big
    assert memory_peaks("after free", device).peak_allocated_bytes >= high
    reset_memory_peaks(device)
    assert (
        memory_peaks("after reset", device).peak_allocated_bytes
        < before.peak_allocated_bytes
    )
    with peak_window("cuda window", device) as sink:
        tmp = torch.empty(1 << 20, dtype=torch.float32, device=device)
        wanted = tmp.numel() * tmp.element_size()
        del tmp
    assert sink[0].peak_allocated_bytes >= wanted
