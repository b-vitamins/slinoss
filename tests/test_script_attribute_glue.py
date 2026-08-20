"""The glue attribution's trace reader.

What can break here is the linkage, not the arithmetic: a kernel reaches its
launch through a correlation id, its operator through an external id, and, when
the launch is a pullback's, the forward line through the node's sequence number. A
reader that loses any of the three produces a full table of rows with no site,
which looks like a finished measurement. So the traces here are literal and tiny,
and every assertion is exact.

No device. :func:`scripts.perf.attribute_glue.launch_costs` reads a file, and the
profiler that writes that file is torch's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.perf.attribute_glue import (
    UNATTRIBUTED,
    Cost,
    Site,
    launch_costs,
    parse_args,
    project_frames,
    report,
    short_node,
)

STACK = (
    "<built-in method run_backward of torch._C._EngineBase object>;"
    "torch/autograd/graph.py(856): _engine_run_backward;"
    "slinoss/mixer.py(573): backward;"
    "slinoss/stack.py(206): forward;"
    "scripts/perf/attribute_step.py(164): train;"
)
"""One recorded stack: two project frames between library and harness frames."""

ENGINE_STACK = (
    "<built-in method run_backward of torch._C._EngineBase object>;"
    "torch/autograd/graph.py(856): _engine_run_backward;"
    "torch/_tensor.py(575): backward;"
    "scripts/perf/attribute_step.py(164): train;"
)
"""What the engine records for a pullback: no line of this package on it."""


def _trace(events: list[dict[str, Any]], tmp_path: Path) -> Path:
    """Write ``events`` as a chrome trace and return its path."""
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def _linked(*, correlation: int, external: int, dur: float) -> list[dict[str, Any]]:
    """A kernel, the runtime call that launched it, and the operator around it."""
    return [
        {
            "cat": "cpu_op",
            "name": "aten::mul",
            "args": {"External id": external, "Call stack": STACK},
        },
        {
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "args": {"correlation": correlation, "External id": external},
        },
        {
            "cat": "kernel",
            "name": "elementwise_kernel",
            "dur": dur,
            "args": {"correlation": correlation},
        },
    ]


def _pullback(*, forward: bool) -> list[dict[str, Any]]:
    """A forward operator, the node evaluating its pullback, and that launch.

    The launching operator's own stack is the engine's, so the site can only come
    from the forward operator the node's sequence number names. Without ``forward``
    the node carries no sequence number and there is no forward operator, which is
    the shape of an accumulation.
    """
    sequence = 80
    node: dict[str, Any] = {
        "cat": "cpu_op",
        "name": "autograd::engine::evaluate_function: torch::autograd::CopySlices",
        "tid": 2,
        "ts": 100.0,
        "dur": 50.0,
        "args": {"External id": 12, "Fwd thread id": 1, "Call stack": ENGINE_STACK},
    }
    forwards: list[dict[str, Any]] = []
    if forward:
        node["args"]["Sequence number"] = sequence
        forwards = [
            {
                "cat": "cpu_op",
                "name": "aten::copy_",
                "tid": 1,
                "ts": 10.0,
                "dur": 1.0,
                "args": {
                    "External id": 11,
                    "Fwd thread id": 0,
                    "Sequence number": sequence,
                    "Call stack": STACK,
                },
            }
        ]
    return [
        *forwards,
        node,
        {
            "cat": "cpu_op",
            "name": "aten::copy_",
            "tid": 2,
            "ts": 120.0,
            "dur": 5.0,
            "args": {"External id": 13, "Call stack": ENGINE_STACK},
        },
        {
            "cat": "cuda_runtime",
            "name": "cudaMemcpyAsync",
            "args": {"correlation": 21, "External id": 13},
        },
        {
            "cat": "gpu_memcpy",
            "name": "Memcpy DtoD (Device -> Device)",
            "dur": 400.0,
            "args": {"correlation": 21},
        },
    ]


def test_a_kernel_reaches_the_line_that_launched_it(tmp_path: Path) -> None:
    """Correlation to launch, external id to operator, operator to stack.

    Two launches of one kernel from one line sum into one row, so the row carries
    the whole cost of a site rather than one launch of it.
    """
    events = _linked(correlation=7, external=3, dur=100.0) + _linked(
        correlation=8, external=3, dur=300.0
    )
    (cost,) = launch_costs(_trace(events, tmp_path), iters=2)
    assert cost.site.kernel == "elementwise_kernel"
    assert cost.site.operator == "aten::mul"
    assert cost.site.node == ""
    assert cost.site.frames[0] == "slinoss/mixer.py(573): backward"
    assert cost.us == 200.0
    assert cost.calls == 1.0


def test_a_pullback_reaches_the_forward_line_it_is_the_pullback_of(
    tmp_path: Path,
) -> None:
    """The engine launches it, so the only line that can be fixed is the forward's.

    Without the node's sequence number every backward kernel reports no site, and
    the backward is where the glue is.
    """
    events = _linked(correlation=7, external=3, dur=100.0) + _pullback(forward=True)
    trace = _trace(events, tmp_path)
    memcpy = next(
        cost
        for cost in launch_costs(trace, iters=1)
        if cost.site.kernel.startswith("Memcpy")
    )
    assert memcpy.site.operator == "aten::copy_"
    assert memcpy.site.node.endswith("torch::autograd::CopySlices")
    assert memcpy.site.frames == (
        "slinoss/mixer.py(573): backward",
        "slinoss/stack.py(206): forward",
    )
    assert memcpy.us == 400.0


def test_a_pullback_with_no_forward_operator_still_names_its_node(
    tmp_path: Path,
) -> None:
    """An accumulation has no forward operator, and the node is the whole answer.

    Reporting nothing would leave the largest counts in the table with no cause.
    """
    events = _linked(correlation=7, external=3, dur=100.0) + _pullback(forward=False)
    memcpy = next(
        cost
        for cost in launch_costs(_trace(events, tmp_path), iters=1)
        if cost.site.kernel.startswith("Memcpy")
    )
    assert memcpy.site.frames == ()
    assert short_node(memcpy.site.node) == "CopySlices"


def test_an_uncorrelated_kernel_is_reported_rather_than_dropped(
    tmp_path: Path,
) -> None:
    """A kernel whose launch is absent still carries its microseconds.

    Dropping it would make the listed classes sum below the class total that
    ``attribute_step.py`` reports, and the gap would read as a fusion.
    """
    events = [
        *_linked(correlation=7, external=3, dur=100.0),
        {"cat": "kernel", "name": "orphan_kernel", "dur": 50.0, "args": {}},
    ]
    costs = launch_costs(_trace(events, tmp_path), iters=1)
    orphan = next(cost for cost in costs if cost.site.kernel == "orphan_kernel")
    assert orphan.site.operator == UNATTRIBUTED
    assert orphan.site.frames == ()
    assert orphan.us == 50.0


def test_a_trace_that_correlates_nothing_raises(tmp_path: Path) -> None:
    """Every row unattributed is a broken reader, not a step with no operators."""
    events = [{"cat": "kernel", "name": "k", "dur": 1.0, "args": {"correlation": 1}}]
    with pytest.raises(ValueError, match="correlates no device event"):
        launch_costs(_trace(events, tmp_path), iters=1)


def test_the_stack_keeps_project_frames_innermost_first() -> None:
    """Library frames and the harness's own frames are not launch sites."""
    assert project_frames(STACK) == (
        "slinoss/mixer.py(573): backward",
        "slinoss/stack.py(206): forward",
    )
    assert project_frames(None) == ()


COSTS = (
    Cost(site=Site("kernel_cutlass_scan", "aten::mm", "", ()), us=100.0, calls=1.0),
    Cost(site=Site("elementwise_kernel", "aten::mul", "", ()), us=25.0, calls=2.0),
)
"""One site of a class the glue table excludes, one of a class it includes."""


def test_an_empty_class_covers_every_class(capsys: pytest.CaptureFixture[str]) -> None:
    """``--classes ''`` is how a shell passes no class, and it means all of them.

    A set holding the empty name selects nothing, which prints a table of zero
    rows summing to zero milliseconds and reads as a step with no glue in it.
    """
    report(COSTS, parse_args(["--classes", ""]))
    out = capsys.readouterr().out
    assert "the listed classes are 0.125 ms, 100.00% of it" in out
    assert "kernel_cutlass_scan" in out
    assert "elementwise_kernel" in out


def test_a_class_no_kernel_can_be_in_is_refused() -> None:
    """A misspelt class would print an empty table rather than fail."""
    with pytest.raises(ValueError, match="no such class"):
        report(COSTS, parse_args(["--classes", "elementwize"]))
