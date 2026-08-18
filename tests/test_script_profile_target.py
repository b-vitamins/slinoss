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
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import pytest
import torch

from scripts.perf import profile_target
from slinoss.perf.workload import SHAPES, OpInputs, shape_by_name

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

SMALL: Final = shape_by_name("tiny")
"""The smallest standard shape the driver accepts: B=1 H=1 T=256 P=8 N=16 L=64."""


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
        factories: ``step`` or ``forward`` per runner built.
        chunks: Chunk length handed to each factory.
        backends: Backend name handed to each factory.
        inputs: Operator inputs handed to each factory.
        devices: Device each window opened on.
    """

    events: list[str] = field(default_factory=list)
    factories: list[str] = field(default_factory=list)
    chunks: list[int] = field(default_factory=list)
    backends: list[str | None] = field(default_factory=list)
    inputs: list[OpInputs] = field(default_factory=list)
    devices: list[torch.device] = field(default_factory=list)


@pytest.fixture
def trace(monkeypatch: pytest.MonkeyPatch) -> Trace:
    """Count runner calls and window edges instead of running either.

    All three names are patched in the driver's namespace, which is where it
    imported them and therefore where it looks them up. The stub runner does no
    work, so the counts are the driver's control flow and nothing else.

    Args:
        monkeypatch: Patcher, undone after the test.

    Returns:
        The trace the stubs write to.
    """
    out = Trace()

    def factory(name: str) -> Callable[..., Callable[[], None]]:
        def make(
            inputs: OpInputs, chunk: int, *, backend: str | None = None
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
    monkeypatch.setattr(profile_target, "profiler_window", window)
    return out


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_a_step_on_the_standard_shape_on_cuda() -> None:
    got = profile_target.parse_args([])
    assert got.shape == "standard"
    assert got.mode == "step"
    assert got.iters == 3
    assert got.warmup == 5
    assert got.dtype == "bf16"
    assert got.device == "cuda"
    assert got.backend is None


def test_the_choice_tables_are_the_modes_and_the_dtypes_the_operator_supports() -> None:
    assert profile_target.MODES == ("forward", "step")
    assert sorted(profile_target.DTYPES) == ["bf16", "fp16", "fp32"]
    assert profile_target.DTYPES["bf16"] is torch.bfloat16
    assert profile_target.DTYPES["fp16"] is torch.float16
    assert profile_target.DTYPES["fp32"] is torch.float32


@pytest.mark.parametrize("name", [s.name for s in SHAPES])
def test_parse_args_accepts_every_standard_shape_name(name: str) -> None:
    assert profile_target.parse_args(["--shape", name]).shape == name


@pytest.mark.parametrize("mode", profile_target.MODES)
def test_parse_args_accepts_both_modes(mode: str) -> None:
    assert profile_target.parse_args(["--mode", mode]).mode == mode


@pytest.mark.parametrize("name", sorted(profile_target.DTYPES))
def test_parse_args_accepts_every_dtype_name(name: str) -> None:
    assert profile_target.parse_args(["--dtype", name]).dtype == name


def test_parse_args_reads_the_counts_the_device_and_the_backend() -> None:
    got = profile_target.parse_args(
        ["--iters", "8", "--warmup", "2", "--device", "cuda:1", "--backend", "cute"]
    )
    assert got.iters == 8
    assert got.warmup == 2
    assert got.device == "cuda:1"
    assert got.backend == "cute"


@pytest.mark.parametrize(
    "flags",
    [
        ["--shape", "huge"],
        ["--mode", "train"],
        ["--dtype", "fp8"],
        ["--iters", "many"],
    ],
)
def test_parse_args_rejects_a_value_outside_its_choices(flags: list[str]) -> None:
    # Argparse exits 2 rather than raising, so a mistyped shape stops the run
    # instead of silently profiling the default one.
    with pytest.raises(SystemExit) as excinfo:
        profile_target.parse_args(flags)
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iters", [0, -1])
def test_main_rejects_a_non_positive_iters(trace: Trace, iters: int) -> None:
    # The driver divides the profiler sums by this count, so zero iterations
    # inside the window is a division by zero downstream, not an empty run.
    with pytest.raises(ValueError, match=f"--iters must be positive, got {iters}"):
        profile_target.main(argv("--iters", str(iters)))
    # Refused before anything was allocated or built.
    assert trace.events == []
    assert trace.factories == []


def test_main_rejects_a_negative_warmup(trace: Trace) -> None:
    # `range(-1)` is empty, so this would silently profile the first call. The
    # bench path rejects the same value inside `measure`; the same flag over the
    # same workload cannot mean two things.
    with pytest.raises(ValueError, match="--warmup must not be negative, got -1"):
        profile_target.main(argv("--warmup", "-1"))
    assert trace.events == []
    assert trace.factories == []


@pytest.mark.parametrize("spec", ["cpu", "cuda"])
def test_main_rejects_a_device_no_counter_exists_on(
    trace: Trace, monkeypatch: pytest.MonkeyPatch, spec: str
) -> None:
    # Both arcs of the shared guard: a host device, and a CUDA device on a host
    # without CUDA. Either way nothing is allocated and no window opens.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="is not a usable cuda device"):
        profile_target.main(argv("--device", spec))
    assert trace.events == []
    assert trace.factories == []


def test_main_returns_zero_on_the_step_path(trace: Trace) -> None:
    assert profile_target.main(argv("--mode", "step")) == 0
    assert trace.factories == ["step"]


def test_main_returns_zero_on_the_forward_path(trace: Trace) -> None:
    assert profile_target.main(argv("--mode", "forward")) == 0
    assert trace.factories == ["forward"]


def test_the_capture_window_wraps_exactly_the_measured_iterations(
    trace: Trace,
) -> None:
    assert profile_target.main(argv("--warmup", "2", "--iters", "3")) == 0
    # Every warmup call before the single window, every measured call inside it.
    # A count is not enough: two runs inside and one outside would count the same.
    assert trace.events == ["run", "run", "enter", "run", "run", "run", "exit"]
    assert trace.events.count("enter") == 1
    assert trace.events.index("enter") == 2


def test_no_warmup_leaves_nothing_in_front_of_the_window(trace: Trace) -> None:
    assert profile_target.main(argv("--warmup", "0", "--iters", "1")) == 0
    assert trace.events == ["enter", "run", "exit"]


def test_the_window_opens_on_the_device_the_work_runs_on(trace: Trace) -> None:
    assert profile_target.main(argv()) == 0
    assert trace.devices == [torch.device("cuda")]


def test_the_step_path_builds_gradient_carrying_inputs(trace: Trace) -> None:
    assert profile_target.main(argv("--mode", "step")) == 0
    inputs = trace.inputs[0]
    assert all(t.requires_grad for t in inputs.differentiable)
    # The output-gradient seed is preallocated and takes no gradient, so the
    # backward inside the window allocates nothing of its own.
    assert not inputs.dy.requires_grad


def test_the_forward_path_builds_inputs_without_gradients(trace: Trace) -> None:
    assert profile_target.main(argv("--mode", "forward")) == 0
    inputs = trace.inputs[0]
    assert not any(t.requires_grad for t in inputs.differentiable)


def test_the_shape_name_selects_the_chunk_and_the_input_shapes(trace: Trace) -> None:
    assert profile_target.main(argv()) == 0
    assert trace.chunks == [SMALL.chunk]
    inputs = trace.inputs[0]
    lead = (SMALL.bsz, SMALL.heads, SMALL.seq)
    assert tuple(inputs.U.shape) == (*lead, SMALL.rows)
    assert tuple(inputs.B.shape) == (*lead, SMALL.d_state)


@pytest.mark.parametrize(
    ("flags", "want"), [([], None), (["--backend", "reference"], "reference")]
)
def test_the_backend_reaches_the_runner_factory(
    trace: Trace, flags: list[str], want: str | None
) -> None:
    # None means the fastest registered backend, which is the profiled path.
    assert profile_target.main(argv(*flags)) == 0
    assert trace.backends == [want]


@pytest.mark.parametrize(
    ("name", "dtype"),
    [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)],
)
def test_the_requested_dtype_reaches_the_inputs(
    trace: Trace, name: str, dtype: torch.dtype
) -> None:
    assert profile_target.main(argv("--dtype", name)) == 0
    inputs = trace.inputs[0]
    assert inputs.U.dtype == dtype
    assert inputs.B.dtype == dtype
    # I4: trans and K are float32 whatever U, B, and C are.
    assert inputs.trans.dtype == torch.float32
    assert inputs.K.dtype == torch.float32
