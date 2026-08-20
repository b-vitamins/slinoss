"""Which line launches each glue kernel.

``scripts/perf/attribute_step.py`` says how much device time each class costs.
This says where the cost comes from: one row per kernel and launch site, so a
glue kernel is bound to the line that has to change for it to go away.

    python3 scripts/perf/attribute_glue.py
    python3 scripts/perf/attribute_glue.py --classes elementwise memory other
    python3 scripts/perf/attribute_glue.py --classes '' --rows 60

The binding is the profiler's own, read off an exported trace rather than off
:meth:`torch.profiler.profile.events`: a kernel carries the correlation id of the
runtime call that launched it, that call carries the external id of the innermost
CPU operator running, and that operator carries the Python stack ``with_stack``
records. ``FunctionEvent.stack`` is empty in torch 2.10 whatever ``with_stack``
says, so the trace is the only place the stack survives, and it survives there
only under a verbose experimental config.

A backward kernel is launched from the autograd engine's C++, so its own stack
holds no line of this package. It is followed one step further: the engine records
the node it is evaluating, the node carries the sequence number of the forward
operator that built it, and that operator carries the forward line. The reported
site of a backward kernel is therefore the forward line whose pullback launches
it, which is the line a fix has to change.

A kernel whose launch or whose operator is missing is reported under
``<unattributed>`` rather than dropped.

A frame is ``file(line): function``, and the line is where the function is defined
rather than the statement running in it: that is what the profiler records. A site
therefore names a function, and the operator column says which of its statements.

``with_stack`` costs Python tracing on every call, so the milliseconds here run
long. Read the ranking and the call counts; ``attribute_step.py`` is what reports
a step's time.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch.profiler import ProfilerActivity, _ExperimentalConfig, profile

from scripts.perf.attribute_step import (
    CLASSES,
    GLUE,
    build_config,
    build_step,
    classify,
)
from slinoss.perf.device import device_ordinal, require_cuda

MODES = ("forward", "step", "decode")

CLASS_NAMES = frozenset(label for label, _ in CLASSES) | {"other"}
"""Every class :func:`classify` can return. ``other`` is its fallback."""

DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
"""Trace categories that spend device time. Together they partition the step."""

LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
"""Trace categories that launch device work."""

UNATTRIBUTED = "<unattributed>"
"""Operator name for a kernel whose launch or whose operator is not in the trace."""

PROJECT_MARKERS = ("slinoss/", "scripts/")
"""Path fragments that make a stack frame this project's rather than a library's."""

SKIP_MARKERS = ("attribute_glue.py", "attribute_step.py")
"""Frames of the harness itself. They launch nothing under measurement."""

ENGINE_PREFIX = "autograd::engine::evaluate_function: "
"""What the autograd engine names the operator it wraps a node's evaluation in."""

NODE_WIDTH = 22
"""Columns the autograd node's name is truncated to."""


class Site(NamedTuple):
    """One kernel's launch site.

    Attributes:
        kernel: Kernel, memcpy or memset name as the trace reports it.
        operator: CPU operator the launch was correlated with, or
            :data:`UNATTRIBUTED`.
        node: Autograd node the launch ran inside, or ``""`` for a forward launch.
        frames: Project stack frames, innermost first, each ``file(line): func``.
            The forward line for a backward launch. Empty when no frame of this
            package is on either stack.
    """

    kernel: str
    operator: str
    node: str
    frames: tuple[str, ...]


class Cost(NamedTuple):
    """What one site costs per iteration.

    Attributes:
        site: The site.
        us: Microseconds of device time per iteration.
        calls: Launches per iteration.
    """

    site: Site
    us: float
    calls: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="step")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--prefill", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--d-state", type=int, default=240)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rows", type=int, default=40, help="Sites listed.")
    parser.add_argument(
        "--frames",
        type=int,
        default=3,
        help="Project frames printed per site, innermost first.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(GLUE),
        help="Classes the table covers. An empty list covers every class.",
    )
    parser.add_argument("--name-width", type=int, default=46)
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Where to write the exported trace. A temporary file by default.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def project_frames(call_stack: str | None) -> tuple[str, ...]:
    """This project's frames of one recorded stack, innermost first.

    The trace records frames innermost first, separated by semicolons. Library
    and harness frames are dropped: what a fix has to touch is a line of this
    package.

    Args:
        call_stack: The ``Call stack`` argument of a CPU operator, or None.

    Returns:
        The project's frames, innermost first.
    """
    if not call_stack:
        return ()
    return tuple(
        frame
        for frame in call_stack.split(";")
        if frame
        and any(marker in frame for marker in PROJECT_MARKERS)
        and not any(marker in frame for marker in SKIP_MARKERS)
    )


def _by_key(events: Iterable[dict[str, Any]], key: str) -> dict[int, dict[str, Any]]:
    """Index events by an integer argument, last occurrence winning."""
    indexed: dict[int, dict[str, Any]] = {}
    for event in events:
        value = (event.get("args") or {}).get(key)
        if value is not None:
            indexed[int(value)] = event
    return indexed


def _is_node(event: dict[str, Any]) -> bool:
    """Whether a CPU operator is the engine evaluating one autograd node.

    By name rather than by ``Fwd thread id``: an accumulation has no forward
    operator and carries neither that argument nor a sequence number, and it is the
    node behind the largest launch count in the table.
    """
    return str(event.get("name", "")).startswith(ENGINE_PREFIX)


def forward_operators(events: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index the operators that build autograd nodes, by sequence number.

    Innermost wins: ``aten::linear`` and the ``aten::addmm`` inside it can share a
    sequence number, and the inner one names the operation whose pullback runs.

    Args:
        events: Trace events.

    Returns:
        Sequence number against the forward operator that carries it.
    """
    return _by_key(
        (
            event
            for event in events
            if event.get("cat") == "cpu_op" and not _is_node(event)
        ),
        "Sequence number",
    )


def backward_nodes(
    events: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group the autograd engine's node evaluations by thread, ordered by start.

    Args:
        events: Trace events.

    Returns:
        Thread id against its node evaluations, ascending by timestamp.
    """
    threads: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("cat") == "cpu_op" and _is_node(event):
            threads.setdefault(int(event.get("tid", 0)), []).append(event)
    for nodes in threads.values():
        nodes.sort(key=lambda event: float(event.get("ts", 0.0)))
    return threads


def enclosing_node(
    threads: dict[int, list[dict[str, Any]]], operator: dict[str, Any]
) -> dict[str, Any] | None:
    """The innermost autograd node ``operator`` ran inside, if any.

    Nodes nest, so the innermost container is the last one to start that has not
    ended. Anything that starts later and does not contain the operator is a
    sibling.

    Args:
        threads: Output of :func:`backward_nodes`.
        operator: The CPU operator that launched the kernel.

    Returns:
        The node, or None for a forward launch.
    """
    start = float(operator.get("ts", 0.0))
    for node in reversed(threads.get(int(operator.get("tid", 0)), [])):
        node_start = float(node.get("ts", 0.0))
        if node_start > start:
            continue
        if start <= node_start + float(node.get("dur", 0.0)):
            return node
    return None


def site_of(
    kernel: str,
    operator: dict[str, Any] | None,
    forwards: dict[int, dict[str, Any]],
    threads: dict[int, list[dict[str, Any]]],
) -> Site:
    """Bind one device event to a line, following a backward to its forward.

    Args:
        kernel: Kernel, memcpy or memset name.
        operator: The CPU operator correlated with the launch, or None.
        forwards: Output of :func:`forward_operators`.
        threads: Output of :func:`backward_nodes`.

    Returns:
        The site. ``frames`` is the forward line when the launch is a pullback's,
        and empty when neither stack holds a line of this package.
    """
    if operator is None:
        return Site(kernel=kernel, operator=UNATTRIBUTED, node="", frames=())
    frames = project_frames((operator.get("args") or {}).get("Call stack"))
    name = str(operator.get("name"))
    if frames:
        return Site(kernel=kernel, operator=name, node="", frames=frames)
    node = enclosing_node(threads, operator)
    if node is None:
        return Site(kernel=kernel, operator=name, node="", frames=())
    sequence = (node.get("args") or {}).get("Sequence number")
    source = forwards.get(int(sequence)) if sequence is not None else None
    return Site(
        kernel=kernel,
        operator=name,
        node=str(node.get("name")),
        frames=project_frames((source.get("args") or {}).get("Call stack"))
        if source
        else (),
    )


def launch_costs(trace: Path, iters: int) -> list[Cost]:
    """Per-site device microseconds and launch counts, descending by time.

    Args:
        trace: An exported chrome trace of a profile taken with ``with_stack``.
        iters: Iterations inside it.

    Returns:
        One entry per distinct site.

    Raises:
        ValueError: If the trace holds no device event, or if no device event
            reaches an operator. Either way the table would say nothing.
    """
    events = json.loads(trace.read_text())["traceEvents"]
    launches = _by_key(
        (event for event in events if event.get("cat") in LAUNCH_CATEGORIES),
        "correlation",
    )
    operators = _by_key(
        (event for event in events if event.get("cat") == "cpu_op"), "External id"
    )
    forwards = forward_operators(events)
    threads = backward_nodes(events)
    tally: dict[Site, list[float]] = {}
    attributed = 0
    for event in events:
        if event.get("cat") not in DEVICE_CATEGORIES:
            continue
        correlation = (event.get("args") or {}).get("correlation")
        launch = launches.get(int(correlation)) if correlation is not None else None
        external = (launch.get("args") or {}).get("External id") if launch else None
        operator = operators.get(int(external)) if external is not None else None
        if operator is not None:
            attributed += 1
        site = site_of(str(event.get("name")), operator, forwards, threads)
        entry = tally.setdefault(site, [0.0, 0.0])
        entry[0] += float(event.get("dur", 0.0))
        entry[1] += 1.0
    if not tally:
        raise ValueError(f"{trace} holds no device event")
    if attributed == 0:
        raise ValueError(f"{trace} correlates no device event with an operator")
    costs = [
        Cost(site=site, us=total / iters, calls=count / iters)
        for site, (total, count) in tally.items()
    ]
    return sorted(costs, key=lambda cost: -cost.us)


def short_node(node: str) -> str:
    """The node's own name, without the engine's and the namespace's prefixes.

    Args:
        node: ``Site.node``.

    Returns:
        The last ``::``-separated segment, truncated to :data:`NODE_WIDTH`.
    """
    return node.removeprefix(ENGINE_PREFIX).rsplit("::", 1)[-1][:NODE_WIDTH]


def report(costs: Sequence[Cost], args: argparse.Namespace) -> None:
    """Print the class total and one row per site.

    Args:
        costs: Every site, descending by time.
        args: The command line. ``classes``, ``rows``, ``frames`` and
            ``name_width`` decide what is printed.

    Raises:
        ValueError: If a requested class is not one :func:`classify` assigns. It
            would select nothing, which reads as a step that spends no time there.
    """
    total = sum(cost.us for cost in costs)
    # An empty name is how a shell passes --classes '', and it means every class.
    wanted = {name for name in args.classes if name} or None
    if wanted is not None and not wanted <= CLASS_NAMES:
        raise ValueError(
            f"no such class {sorted(wanted - CLASS_NAMES)}; "
            f"classify assigns {sorted(CLASS_NAMES)}"
        )
    picked = [
        cost for cost in costs if wanted is None or classify(cost.site.kernel) in wanted
    ]
    picked_us = sum(cost.us for cost in picked)
    print(
        f"correlated device time {total / 1000.0:,.3f} ms per iteration over "
        f"{len(costs)} sites; the listed classes are {picked_us / 1000.0:,.3f} ms, "
        f"{100.0 * picked_us / total:,.2f}% of it"
    )
    print("with_stack is on, so these milliseconds are traced and run long")
    print("a node names the pullback that launched the kernel; the site is forward")
    print("a frame's line is the function's definition, not the statement running")
    print()
    width = args.name_width
    print(
        f"{'kernel':{width}s} {'ms/iter':>9s} {'share':>7s} {'calls':>7s}  "
        f"{'operator':26s} {'node':{NODE_WIDTH}s} site"
    )
    for cost in picked[: args.rows]:
        frames = cost.site.frames[: args.frames]
        print(
            f"{cost.site.kernel[:width]:{width}s} {cost.us / 1000.0:9,.3f} "
            f"{100.0 * cost.us / total:6,.2f}% {cost.calls:7,.1f}  "
            f"{cost.site.operator[:26]:26s} "
            f"{short_node(cost.site.node):{NODE_WIDTH}s} "
            f"{' <- '.join(frames) if frames else '-'}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Profile one mode and print each glue kernel's launch site.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    config = build_config(args)
    step = build_step(args, config, device)
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        # Without verbose the exported trace carries the recorded frames only as
        # python_function events, and no operator carries a Call stack argument.
        experimental_config=_ExperimentalConfig(verbose=True),
    ) as profiled:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize(device)

    trace = args.trace
    if trace is None:
        # The trace outlives the handle: export_chrome_trace opens the path
        # itself, so a live handle would be a second one on the same file.
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            trace = Path(handle.name)
    profiled.export_chrome_trace(str(trace))
    print(f"device {device_ordinal(device)}  mode {args.mode}  iters {args.iters}")
    print(
        f"geometry {config.n_layers} layers  d_model {config.d_model}  "
        f"3N {config.d_state}  d_head {config.d_head}  chunk {config.chunk_size}  "
        f"heads {config.n_heads}  groups {config.n_groups}  batch {args.batch}"
    )
    report(launch_costs(trace, args.iters), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
