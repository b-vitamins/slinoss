"""The stream a CuTe launch runs on.

``launch()`` with no ``stream`` goes to the legacy default stream. That is not the
stream torch produced the operands on, and it is not capturable: inside
``torch.cuda.graph`` the launch runs instead of being recorded, so the graph replays
as a no-op and the caller reads the eager result and sees nothing wrong. Two
properties are pinned here, one static and one on the device.

The static one is a source scan rather than a signature check because the defect a
signature cannot see is a launcher that takes a stream and launches without it. The
device one uses a probe kernel with an in-place effect on a buffer allocated outside
the graph pool: a sentinel is the only way to tell a recorded launch from one that
ran, since a graph that recorded nothing leaves the eagerly computed result in place
and replay appears to agree.
"""

import re
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute

from slinoss._cute import Stream, jit_launch

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

N = 128
"""Probe length. One block, one thread per element."""

PACKAGE = Path(__file__).resolve().parents[1] / "slinoss"


@cute.kernel
def _double_kernel(gx: cute.Tensor) -> None:
    tid, _, _ = cute.arch.thread_idx()
    gx[tid] = gx[tid] * 2.0


@cute.jit
def _double_launch(gx: cute.Tensor, stream: Stream, n: cutlass.Constexpr) -> None:
    _double_kernel(gx).launch(grid=(1, 1, 1), block=(n, 1, 1), stream=stream)


def _call_args(text: str, open_paren: int) -> str:
    """The text between ``open_paren`` and the paren that closes it."""
    depth, index = 1, open_paren + 1
    while depth:
        depth += (text[index] == "(") - (text[index] == ")")
        index += 1
    return text[open_paren:index]


def _enclosing_def(text: str, position: int) -> str:
    """The signature of the last ``def`` opened before ``position``."""
    start = text.rindex("\ndef ", 0, position)
    return _call_args(text, text.index("(", start))


def test_every_launch_hands_the_stream_it_was_given() -> None:
    """No shipped launch reaches the default stream."""
    seen = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text()
        for match in re.finditer(r"\.launch\(", text):
            seen += 1
            line = text[: match.start()].count("\n") + 1
            where = f"{path.relative_to(PACKAGE.parent)}:{line}"
            assert "stream=stream" in _call_args(text, match.end() - 1), where
            assert "stream: Stream" in _enclosing_def(text, match.start()), where
    assert seen >= 20, f"the scan found {seen} launches; it should find every one"


def test_a_capture_records_the_launch_instead_of_running_it() -> None:
    """The property the stream argument exists for."""
    x = torch.ones(N, device="cuda")
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        jit_launch(_double_launch, (x,), (N,))
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()

    x.fill_(1.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        jit_launch(_double_launch, (x,), (N,))
    during = float(x[0])
    graph.replay()
    torch.cuda.synchronize()

    assert during == 1.0, "the launch ran during capture instead of being recorded"
    assert float(x[0]) == 2.0, "the graph replayed without the launch in it"
