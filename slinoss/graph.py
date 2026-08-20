"""CUDA-graph capture of one step.

A decode step is a few thousand launches over buffers whose addresses never
change, and each launch costs microseconds of host work the device waits through.
Capturing the step and replaying it removes all of that: one launch of a recorded
graph, no Python, no dispatcher, no per-launch driver call.

    step = capture_decode(stack, state)
    logits = step(token)               # advances state in place, as the eager step does

The contract capture imposes is that every address is fixed. That is why the state
containers are frozen and written in place, why the step's inputs are copied into
buffers this module owns, and why the outputs are views into the graph's private
pool that the next replay overwrites. A caller who needs an output to outlive the
next replay clones it.

Three things a caller must know about the general :func:`capture`. The function is
run before it is recorded, ``warmup`` times, so a function with a side effect has
that effect ``warmup`` times before the graph exists; :func:`capture_decode`
restores the state it warmed on for exactly this reason. The function may not
synchronize: a host read of a device value raises during capture rather than
recording anything. And a function that trains returns its loss detached, since a
returned loss keeps its autograd graph alive for as long as the step exists, and the
``AccumulateGrad`` nodes in it belong to the capture stream rather than to the stream
a later eager backward runs on.

A capture that records nothing is an error here rather than a warning. It is the
one failure that is otherwise silent: an empty graph replays as a no-op, the
output buffer still holds what the warmup left in it, and a comparison against the
eager result passes.

A capture that compiles is an error for the same reason. Tracing a CuTe entry point
is host work, and host work inside a capture is not recorded: whatever it computes
happens once, at capture time, and never on a replay. :func:`capture` counts the
executors compiled across the recording and refuses a capture that grew the count.
An ahead-of-time payload removes the compile; it does not remove the warmup, which
the allocator needs.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

__all__ = ["GraphedStep", "capture", "capture_decode"]

WARMUP = 3
"""Warmup iterations before a capture.

Enough for the allocator to reach a steady state and for every kernel in the step
to have compiled, since a CuTe entry point traces on its first call and tracing
inside a capture is host work the graph cannot record. :func:`capture` enforces the
second half: a recording that compiles anything raises.
"""


def _compiled() -> int | None:
    """Executors this process has compiled, or None if the DSL is not installed.

    Imported here rather than at module scope: this module is importable, and every
    reference-path test of it runs, on a tree with no CuTe DSL.

    Returns:
        The count, or None.
    """
    try:
        from slinoss._cute import cache_events
    except ImportError:
        return None
    return cache_events().compiled


@dataclass(frozen=True)
class GraphedStep:
    """A step recorded once and replayed per call.

    Attributes:
        graph: The recorded graph.
        inputs: The buffers the step reads its arguments from, in the order
            :func:`capture` was given them. A caller may write into these instead
            of passing arguments.
        outputs: Whatever the captured function returned, holding views into the
            graph's private memory pool. Overwritten by the next replay.
    """

    graph: torch.cuda.CUDAGraph
    inputs: tuple[Tensor, ...]
    outputs: Any

    def __call__(self, *given: Tensor) -> Any:
        """Replay the step over ``given``.

        Args:
            *given: One tensor per captured input, matching it in shape and dtype.
                An argument that is already the captured buffer is not copied.

        Returns:
            :attr:`outputs`.

        Raises:
            ValueError: On the wrong number of arguments, or on one whose shape or
                dtype is not the captured buffer's. Both would otherwise reach
                ``copy_``, which broadcasts and casts rather than refusing.
        """
        if len(given) != len(self.inputs):
            raise ValueError(
                f"the step captured {len(self.inputs)} inputs and got {len(given)}"
            )
        for index, (static, value) in enumerate(zip(self.inputs, given, strict=True)):
            if value.shape != static.shape or value.dtype is not static.dtype:
                raise ValueError(
                    f"input {index} is {tuple(value.shape)} {value.dtype} and the "
                    f"captured buffer is {tuple(static.shape)} {static.dtype}"
                )
            if value.data_ptr() != static.data_ptr():
                static.copy_(value)
        self.graph.replay()
        return self.outputs


def capture(
    fn: Callable[..., Any],
    *inputs: Tensor,
    warmup: int = WARMUP,
    share: GraphedStep | None = None,
) -> GraphedStep:
    """Record ``fn`` over copies of ``inputs``.

    Args:
        fn: The step. Called with one buffer per entry of ``inputs``. Runs
            ``warmup`` times before it is recorded, so its side effects happen
            that many times; must not synchronize the device.
        *inputs: The step's arguments. Copied, so the caller's tensors are never
            the graph's and later writes to them do not reach a replay.
        warmup: Iterations before capture. At least one.
        share: An earlier step to take the memory pool from, so two graphs that are
            never live at once reserve one pool rather than two. None for a fresh
            pool.

    Returns:
        The :class:`GraphedStep`.

    Raises:
        ValueError: If ``warmup`` is not positive, if an input is not on CUDA, or
            if the inputs span two devices.
        RuntimeError: If the capture recorded no work, or compiled anything.
    """
    if warmup < 1:
        raise ValueError(f"warmup must be positive, got {warmup}")
    devices = {t.device for t in inputs}
    if any(d.type != "cuda" for d in devices):
        raise ValueError(f"capture needs cuda inputs, got {sorted(map(str, devices))}")
    if len(devices) > 1:
        raise ValueError(f"one device only, got {sorted(map(str, devices))}")
    device = devices.pop() if devices else torch.device("cuda")

    statics = tuple(t.clone() for t in inputs)
    with torch.cuda.device(device):
        # Warmup on a side stream, as capture requires: the allocator serves the
        # capture from a private pool, and blocks it hands out on the default
        # stream during warmup are not in it.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn(*statics)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        compiled = _compiled()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*CUDA Graph is empty.*")
                with torch.cuda.graph(
                    graph, pool=None if share is None else share.graph.pool()
                ):
                    outputs = fn(*statics)
        except UserWarning as exc:
            raise RuntimeError(
                "the capture recorded no work. A launch reaches the graph only if "
                "it goes to the stream being captured, so a kernel launched on the "
                "default stream, or on any other stream, runs instead"
            ) from exc
    after = _compiled()
    if compiled is not None and after is not None and after != compiled:
        raise RuntimeError(
            f"the capture compiled {after - compiled} executors. Tracing is "
            f"host work and the graph did not record it, so the launch it traced is "
            f"not in the graph; raise warmup above {warmup} or load a payload"
        )
    return GraphedStep(graph=graph, inputs=statics, outputs=outputs)


def _restore(state: StackState, saved: StackState) -> None:
    """Copy ``saved`` back into ``state``'s own buffers.

    In place, field by field: rebinding a field would leave the graph writing
    memory the caller no longer reads.
    """
    for layer, snapshot in zip(state.layers, saved.layers, strict=True):
        layer.conv.copy_(snapshot.conv)
        layer.ssm.copy_(snapshot.ssm)
        layer.b_prev.copy_(snapshot.b_prev)
        layer.u_prev.copy_(snapshot.u_prev)


def capture_decode(
    stack: SLinOSSStack,
    state: StackState,
    *,
    warmup: int = WARMUP,
    share: GraphedStep | None = None,
) -> GraphedStep:
    """Record one decode step of ``stack`` over ``state``.

    The step is one token per sequence, so the captured input is ``(B,1)`` and the
    output is ``(B,1,padded_vocab_size)``. ``state`` is advanced in place by every
    replay, and is the state the graph holds: a replay against a different state is
    not possible, since the graph records these addresses.

    Sampling stays outside. It branches on the host, it draws from a generator whose
    state a replay would not advance, and it is not where the launches are.

    Args:
        stack: A stack with an embedding and a head, so the step maps ids to logits.
        state: The state to advance. Restored to its entry values before this
            returns, so the warmup's tokens do not remain in it.
        warmup: See :func:`capture`.
        share: See :func:`capture`.

    Returns:
        The :class:`GraphedStep`. Its input buffer holds zeros.

    Raises:
        ValueError: On a stack with no head. The depth of ``state`` against the
            stack's is the stack's own check.
        RuntimeError: If the capture recorded no work.
    """
    if stack.head is None or stack.embedding is None:
        raise ValueError("a decode step needs a stack with a head; set vocab_size")
    token = torch.zeros(state.batch, 1, dtype=torch.int64, device=state.device)
    saved = state.clone()
    try:
        step = capture(lambda ids: stack(ids, state), token, warmup=warmup, share=share)
    finally:
        _restore(state, saved)
    return step
