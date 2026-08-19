"""The profiler target: its command line, and what reaches the capture window.

The runner factories and the window are replaced with counting stubs, so what is
under test is the driver's own control flow -- which factory it builds, how many
calls precede the window, and how many run inside it -- and not the operator.
:func:`slinoss.perf.capture.profiler_window` needs a CUDA device it can drain, so
it is replaced by a null window that records both of its edges. The driver refuses
any device a report cannot name, so every test names a CUDA one.

Ordering is the point. A warmup call inside the window puts first-call
compilation and allocator growth into a counter, and the only evidence of that
is the position of a runner call relative to the window's ``enter``.

The operator axis does not interact with the warmup, iteration, or device axes:
``--op`` selects which factory is built, and the window and the guards run over
whichever one that is. So it is swept once, against the workload it selects, and
not crossed with the rest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import pytest
import torch

from scripts.perf import profile_target
from slinoss.perf.workload import (
    OPS,
    SHAPES,
    ConvInputs,
    OpInputs,
    conv_shape_by_name,
    shape_by_name,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

SMALL: Final = shape_by_name("tiny")
"""The smallest standard shape the driver accepts: B=1 H=1 T=256 P=16 N=16 L=64."""

SMALL_CONV: Final = conv_shape_by_name("tiny")
"""The same name under the conv shape table: B=1 T=256 D=16 W=4."""


def argv(*extra: str) -> list[str]:
    """Driver argv at the smallest shape, one measured iteration.

    Args:
        *extra: Flags appended after these. Argparse keeps the last occurrence,
            so any of them can be overridden.

    Returns:
        The argument list.
    """
    return [
        "--shape",
        SMALL.name,
        "--device",
        "cuda",
        "--dtype",
        "fp32",
        "--warmup",
        "0",
        "--iters",
        "1",
        *extra,
    ]


@dataclass
class Trace:
    """What the driver did, in order.

    Attributes:
        events: ``run`` per runner call, ``enter`` and ``exit`` per capture
            window, in call order. A warmup call and a measured call differ only
            in their position relative to ``enter``.
        factories: One name per runner built, out of ``step``, ``forward``,
            ``conv-step``, and ``conv-forward``.
        chunks: Chunk length handed to each factory, or None for the conv, which
            has no chunk.
        backends: Backend name handed to each factory.
        inputs: Operator inputs handed to each factory.
        devices: Device each window opened on.
    """

    events: list[str] = field(default_factory=list)
    factories: list[str] = field(default_factory=list)
    chunks: list[int | None] = field(default_factory=list)
    backends: list[str | None] = field(default_factory=list)
    inputs: list[OpInputs | ConvInputs] = field(default_factory=list)
    devices: list[torch.device] = field(default_factory=list)


@pytest.fixture
def trace(monkeypatch: pytest.MonkeyPatch) -> Trace:
    """Count runner calls and window edges instead of running either.

    Every name is patched in the driver's namespace, which is where it imported
    them and therefore where it looks them up. The stub runner does no work, so
    the counts are the driver's control flow and nothing else.

    Args:
        monkeypatch: Patcher, undone after the test.

    Returns:
        The trace the stubs write to.
    """
    out = Trace()

    def factory(name: str) -> Callable[..., Callable[[], None]]:
        def make(
            inputs: OpInputs | ConvInputs,
            chunk: int | None = None,
            *,
            backend: str | None = None,
        ) -> Callable[[], None]:
            out.factories.append(name)
            out.chunks.append(chunk)
            out.backends.append(backend)
            out.inputs.append(inputs)

            def run() -> None:
                out.events.append("run")

            return run

        return make

    @contextmanager
    def window(device: torch.device) -> Iterator[None]:
        out.devices.append(device)
        out.events.append("enter")
        try:
            yield
        finally:
            out.events.append("exit")

    monkeypatch.setattr(profile_target, "step", factory("step"))
    monkeypatch.setattr(profile_target, "forward_only", factory("forward"))
    monkeypatch.setattr(profile_target, "conv_step", factory("conv-step"))
    monkeypatch.setattr(profile_target, "conv_forward_only", factory("conv-forward"))
    monkeypatch.setattr(profile_target, "profiler_window", window)
    return out


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_a_step_on_the_standard_shape_on_cuda() -> None:
    got = profile_target.parse_args([])
    # The scan is the first entry of the registry and the default, so the command
    # that profiled the scan before the conv existed still profiles the scan.
    assert got.op == OPS[0] == "so3ssd"
    assert got.shape == "standard"
    assert got.mode == "step"
    assert got.iters == 3
    assert got.warmup == 5
    assert got.dtype == "bf16"
    assert got.device == "cuda"
    assert got.backend is None
    given = profile_target.parse_args(
        ["--iters", "8", "--warmup", "2", "--device", "cuda:1", "--backend", "cute"]
    )
    assert given.iters == 8
    assert given.warmup == 2
    assert given.device == "cuda:1"
    assert given.backend == "cute"


def test_every_choice_the_driver_offers_selects_something() -> None:
    assert profile_target.MODES == ("forward", "step")
    assert sorted(profile_target.DTYPES) == ["bf16", "fp16", "fp32"]
    assert profile_target.DTYPES["bf16"] is torch.bfloat16
    assert profile_target.DTYPES["fp16"] is torch.float16
    assert profile_target.DTYPES["fp32"] is torch.float32
    assert OPS == ("so3ssd", "conv", "scanprep", "block", "mixer")
    for op in OPS:
        assert profile_target.parse_args(["--op", op]).op == op
    for shape in SHAPES:
        assert profile_target.parse_args(["--shape", shape.name]).shape == shape.name
    for mode in profile_target.MODES:
        assert profile_target.parse_args(["--mode", mode]).mode == mode
    for name in sorted(profile_target.DTYPES):
        assert profile_target.parse_args(["--dtype", name]).dtype == name
    # Argparse exits 2 rather than raising, so a mistyped shape stops the run
    # instead of silently profiling the default one.
    for flags in (
        ["--op", "mamba"],
        ["--shape", "huge"],
        ["--mode", "train"],
        ["--dtype", "fp8"],
        ["--iters", "many"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            profile_target.parse_args(flags)
        assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_refuses_a_run_no_counter_can_read(
    trace: Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The driver divides the profiler sums by the iteration count, so zero
    # iterations inside the window is a division by zero downstream, not an empty
    # run.
    for iters in (0, -1):
        with pytest.raises(ValueError, match=f"--iters must be positive, got {iters}"):
            profile_target.main(argv("--iters", str(iters)))
    # `range(-1)` is empty, so a negative warmup would silently profile the first
    # call. The bench path rejects the same value inside `measure`; the same flag
    # over the same workload cannot mean two things.
    with pytest.raises(ValueError, match="--warmup must not be negative, got -1"):
        profile_target.main(argv("--warmup", "-1"))
    # Both arcs of the device guard: a host device, and a CUDA device on a host
    # without CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for spec in ("cpu", "cuda"):
        with pytest.raises(RuntimeError, match="is not a usable cuda device"):
            profile_target.main(argv("--device", spec))
    # Every refusal arrives before anything is allocated and before a window opens.
    assert trace.events == []
    assert trace.factories == []


def test_the_capture_window_wraps_exactly_the_measured_iterations(
    trace: Trace,
) -> None:
    assert profile_target.main(argv("--warmup", "2", "--iters", "3")) == 0
    # Every warmup call before the single window, every measured call inside it.
    # A count is not enough: two runs inside and one outside would count the same.
    assert trace.events == ["run", "run", "enter", "run", "run", "run", "exit"]
    assert trace.events.count("enter") == 1
    assert trace.events.index("enter") == 2
    # Zero warmup leaves nothing in front of the window.
    assert profile_target.main(argv("--warmup", "0", "--iters", "1")) == 0
    assert trace.events[7:] == ["enter", "run", "exit"]


def test_the_step_path_builds_gradient_carrying_inputs(trace: Trace) -> None:
    assert profile_target.main(argv("--mode", "step")) == 0
    assert trace.factories == ["step"]
    inputs = trace.inputs[0]
    assert all(t.requires_grad for t in inputs.differentiable)
    # The output-gradient seed is preallocated and takes no gradient, so the
    # backward inside the window allocates nothing of its own.
    assert not inputs.dy.requires_grad


def test_the_forward_path_builds_inputs_without_gradients(trace: Trace) -> None:
    assert profile_target.main(argv("--mode", "forward")) == 0
    assert trace.factories == ["forward"]
    assert not any(t.requires_grad for t in trace.inputs[0].differentiable)


def test_the_shape_the_device_and_the_backend_reach_the_runner(trace: Trace) -> None:
    assert profile_target.main(argv()) == 0
    assert trace.devices == [torch.device("cuda")]
    assert trace.chunks == [SMALL.chunk]
    inputs = trace.inputs[0]
    assert isinstance(inputs, OpInputs)
    lead = (SMALL.bsz, SMALL.heads, SMALL.seq)
    assert tuple(inputs.U.shape) == (*lead, SMALL.rows)
    assert tuple(inputs.B.shape) == (*lead, SMALL.d_state)
    # None means the fastest registered backend, which is the profiled path.
    assert profile_target.main(argv("--backend", "reference")) == 0
    assert trace.backends == [None, "reference"]


def test_the_requested_dtype_reaches_the_inputs(trace: Trace) -> None:
    for index, (name, dtype) in enumerate(
        (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32))
    ):
        assert profile_target.main(argv("--dtype", name)) == 0
        inputs = trace.inputs[index]
        assert isinstance(inputs, OpInputs)
        assert inputs.U.dtype == dtype
        assert inputs.B.dtype == dtype
        # I4: trans and K are float32 whatever U, B, and C are.
        assert inputs.trans.dtype == torch.float32
        assert inputs.K.dtype == torch.float32


def test_the_conv_operator_builds_the_conv_workload_at_the_named_shape(
    trace: Trace,
) -> None:
    assert profile_target.main(argv("--op", "conv", "--mode", "step")) == 0
    # Only the conv factory, so nothing allocates a scan input set the profiler
    # would count as the conv's memory traffic.
    assert trace.factories == ["conv-step"]
    # The conv has no chunk. A driver that passed the scan shape's chunk to it
    # would be reading the wrong shape table.
    assert trace.chunks == [None]
    assert trace.devices == [torch.device("cuda")]
    inputs = trace.inputs[0]
    assert isinstance(inputs, ConvInputs)
    assert tuple(inputs.x.shape) == (
        SMALL_CONV.bsz,
        SMALL_CONV.seq,
        SMALL_CONV.channels,
    )
    assert tuple(inputs.weight.shape) == (SMALL_CONV.channels, SMALL_CONV.width)
    assert tuple(inputs.initial_state.shape) == SMALL_CONV.state_shape
    assert {t.dtype for t in inputs.tensors} == {torch.float32}
    assert all(t.requires_grad for t in inputs.differentiable)
    # The output-gradient seed is preallocated and takes no gradient, so the
    # backward inside the window allocates nothing of its own.
    assert not inputs.dy.requires_grad
    # Token-major by default, which is the layout every earlier report was taken at.
    assert inputs.d_head is None
    assert tuple(inputs.dy.shape) == tuple(inputs.x.shape)
    # Forward mode drops the graph, and a named backend reaches the factory.
    assert profile_target.main(argv("--op", "conv", "--backend", "reference")) == 0
    assert trace.factories[1] == "conv-step"
    assert trace.backends == [None, "reference"]
    assert profile_target.main(argv("--op", "conv", "--mode", "forward")) == 0
    assert trace.factories[2] == "conv-forward"
    assert not any(t.requires_grad for t in trace.inputs[2].differentiable)


def test_the_conv_output_layout_reaches_the_seed_and_not_the_scan(
    trace: Trace,
) -> None:
    # The seed's shape is the only place the layout is visible to the runner, so a
    # flag that reached the forward and left dy token-major would raise inside the
    # window rather than here.
    assert profile_target.main(argv("--op", "conv", "--d-head", "16")) == 0
    conv = trace.inputs[0]
    assert isinstance(conv, ConvInputs)
    assert conv.d_head == 16
    assert tuple(conv.dy.shape) == (
        SMALL_CONV.bsz,
        SMALL_CONV.channels // 16,
        SMALL_CONV.seq,
        16,
    )
    # The scan takes no such flag, and passing one leaves its inputs untouched.
    assert profile_target.main(argv("--d-head", "16")) == 0
    assert isinstance(trace.inputs[1], OpInputs)
