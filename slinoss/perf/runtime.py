"""Low-overhead runtime instrumentation for benchmarked model regions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from time import perf_counter
import threading
from typing import Any, Callable, Iterator, TypeAlias, TypeVar, cast

import torch
from torch.autograd.profiler import record_function

T = TypeVar("T")
Stamp: TypeAlias = torch.cuda.Event | float
StampedBoundary: TypeAlias = tuple[int, Stamp]

_ACTIVE_STEPS: list["_PerfStep"] = []
_ACTIVE_STEPS_LOCK = threading.RLock()


def current_step() -> "_PerfStep | None":
    with _ACTIVE_STEPS_LOCK:
        return _ACTIVE_STEPS[-1] if _ACTIVE_STEPS else None


class _PerfStep:
    def __init__(self, *, device: torch.device | None) -> None:
        self.device = device
        self._timed_regions: list[tuple[str, Stamp, Stamp]] = []
        self._backward_tokens: dict[
            tuple[str, int], dict[str, list[StampedBoundary]]
        ] = {}
        self._next_token_id = 0
        self._sequence_id = 0
        self._cache_events: dict[str, dict[str, int]] = {}
        self._resolved_regions: dict[str, float] | None = None

    @property
    def uses_cuda_events(self) -> bool:
        return (
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    def add_region(self, label: str, start: Stamp, end: Stamp) -> None:
        self._timed_regions.append((label, start, end))

    def next_backward_token(self) -> int:
        token_id = self._next_token_id
        self._next_token_id += 1
        return token_id

    def record_backward_boundary(self, label: str, token_id: int, *, kind: str) -> None:
        key = (label, int(token_id))
        events = self._backward_tokens.setdefault(key, {"enter": [], "exit": []})
        seq = self._sequence_id
        self._sequence_id += 1
        if self.uses_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device=self.device))
            events[kind].append((seq, event))
        else:
            events[kind].append((seq, perf_counter()))

    def note_cache_event(self, label: str, *, hit: bool) -> None:
        bucket = self._cache_events.setdefault(label, {"hits": 0, "misses": 0})
        bucket["hits" if hit else "misses"] += 1

    def finalize(self) -> dict[str, Any]:
        if self._resolved_regions is not None:
            return {
                "regions_ms": dict(self._resolved_regions),
                "cache_events": {
                    label: dict(counts) for label, counts in self._cache_events.items()
                },
            }

        if self.uses_cuda_events:
            torch.cuda.synchronize(self.device)

        regions: dict[str, float] = {}
        for label, start, end in self._timed_regions:
            if self.uses_cuda_events:
                elapsed = float(
                    cast(torch.cuda.Event, start).elapsed_time(
                        cast(torch.cuda.Event, end)
                    )
                )
            else:
                elapsed = float(cast(float, end) - cast(float, start)) * 1000.0
            regions[label] = regions.get(label, 0.0) + elapsed

        for (label, _token_id), events in self._backward_tokens.items():
            enters = events["enter"]
            exits = events["exit"]
            if not enters or not exits:
                continue
            if self.uses_cuda_events:
                _, start_event = min(enters, key=lambda item: item[0])
                _, end_event = max(exits, key=lambda item: item[0])
                elapsed = float(
                    cast(torch.cuda.Event, start_event).elapsed_time(
                        cast(torch.cuda.Event, end_event)
                    )
                )
            else:
                elapsed = (
                    max(cast(float, timestamp) for _, timestamp in exits)
                    - min(cast(float, timestamp) for _, timestamp in enters)
                ) * 1000.0
            regions[label] = regions.get(label, 0.0) + elapsed

        self._resolved_regions = regions
        return {
            "regions_ms": dict(regions),
            "cache_events": {
                label: dict(counts) for label, counts in self._cache_events.items()
            },
        }


class PerfRecorder:
    """Collect per-step CUDA-event budgets for instrumented regions."""

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        self.device = None if device is None else torch.device(device)
        self.steps: list[dict[str, Any]] = []

    @contextmanager
    def capture_step(self) -> Iterator[_PerfStep]:
        step = _PerfStep(device=self.device)
        with _ACTIVE_STEPS_LOCK:
            _ACTIVE_STEPS.append(step)
        try:
            yield step
        finally:
            with _ACTIVE_STEPS_LOCK:
                active = _ACTIVE_STEPS.pop()
            if active is not step:
                raise RuntimeError("PerfRecorder active-step stack corrupted.")
            self.steps.append(step.finalize())


@contextmanager
def record_region(label: str) -> Iterator[None]:
    step = current_step()
    if step is None:
        yield
        return

    if step.uses_cuda_events:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with record_function(label):
            start.record(torch.cuda.current_stream(device=step.device))
            try:
                yield
            finally:
                end.record(torch.cuda.current_stream(device=step.device))
                step.add_region(label, start, end)
        return

    start_t = perf_counter()
    with record_function(label):
        try:
            yield
        finally:
            end_t = perf_counter()
            step.add_region(label, start_t, end_t)


def note_cache_event(label: str, *, hit: bool) -> None:
    step = current_step()
    if step is not None:
        step.note_cache_event(label, hit=hit)


class _BackwardRegionEnter(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        tensor: torch.Tensor,
        label: str,
        token_id: int,
    ) -> torch.Tensor:
        ctx.label = label
        ctx.token_id = int(token_id)
        return tensor

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        step = current_step()
        if step is not None:
            step.record_backward_boundary(ctx.label, ctx.token_id, kind="enter")
        return grad_output, None, None


class _BackwardRegionExit(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        tensor: torch.Tensor,
        label: str,
        token_id: int,
    ) -> torch.Tensor:
        ctx.label = label
        ctx.token_id = int(token_id)
        return tensor

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        step = current_step()
        if step is not None:
            step.record_backward_boundary(ctx.label, ctx.token_id, kind="exit")
        return grad_output, None, None


def _tree_map_tensors(obj: T, fn: Callable[[torch.Tensor], torch.Tensor]) -> T:
    if isinstance(obj, torch.Tensor):
        return fn(obj)  # type: ignore[return-value]
    if isinstance(obj, tuple):
        return tuple(_tree_map_tensors(item, fn) for item in obj)  # type: ignore[return-value]
    if isinstance(obj, list):
        return [_tree_map_tensors(item, fn) for item in obj]  # type: ignore[return-value]
    if isinstance(obj, dict):
        return {key: _tree_map_tensors(value, fn) for key, value in obj.items()}  # type: ignore[return-value]
    if is_dataclass(obj) and not isinstance(obj, type):
        values = {
            field.name: _tree_map_tensors(getattr(obj, field.name), fn)
            for field in fields(obj)
        }
        return type(obj)(**values)  # type: ignore[return-value]
    return obj


def call_region(
    label: str,
    fn: Callable[..., T],
    *args: Any,
    capture_backward: bool = True,
    **kwargs: Any,
) -> T:
    step = current_step()
    if step is None:
        return fn(*args, **kwargs)

    token_id = step.next_backward_token()
    backward_label = f"backward.{label}"

    if capture_backward:
        args = tuple(
            _tree_map_tensors(
                arg,
                lambda t: (
                    cast(
                        torch.Tensor,
                        _BackwardRegionExit.apply(t, backward_label, token_id),
                    )
                    if t.requires_grad
                    else t
                ),
            )
            for arg in args
        )
        kwargs = {
            key: _tree_map_tensors(
                value,
                lambda t: (
                    cast(
                        torch.Tensor,
                        _BackwardRegionExit.apply(t, backward_label, token_id),
                    )
                    if t.requires_grad
                    else t
                ),
            )
            for key, value in kwargs.items()
        }

    with record_region(f"forward.{label}"):
        out = fn(*args, **kwargs)

    if not capture_backward:
        return out

    return _tree_map_tensors(
        out,
        lambda t: (
            cast(
                torch.Tensor,
                _BackwardRegionEnter.apply(t, backward_label, token_id),
            )
            if t.requires_grad
            else t
        ),
    )


def attach_module_timer(
    module: torch.nn.Module,
    label: str,
    *,
    capture_forward: bool = True,
    capture_backward: bool = True,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach forward/backward timers to a module for active perf steps."""

    fwd_stack: list[tuple[_PerfStep, Stamp]] = []
    bwd_stack: list[tuple[_PerfStep, Stamp]] = []

    def _stamp(step: _PerfStep) -> Stamp:
        if step.uses_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device=step.device))
            return event
        return perf_counter()

    def _forward_pre(_module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
        step = current_step()
        if step is None:
            return
        fwd_stack.append((step, _stamp(step)))

    def _forward_post(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if not fwd_stack:
            return
        step, start = fwd_stack.pop()
        end = _stamp(step)
        step.add_region(f"forward.{label}", start, end)

    def _backward_pre(
        _module: torch.nn.Module,
        _grad_output: Any,
    ) -> Any:
        step = current_step()
        if step is None:
            return None
        bwd_stack.append((step, _stamp(step)))
        return None

    def _backward_post(
        _module: torch.nn.Module,
        _grad_input: Any,
        _grad_output: Any,
    ) -> Any:
        if not bwd_stack:
            return None
        step, start = bwd_stack.pop()
        end = _stamp(step)
        step.add_region(f"backward.{label}", start, end)
        return None

    handles: list[torch.utils.hooks.RemovableHandle] = []
    if capture_forward:
        handles.append(module.register_forward_pre_hook(_forward_pre))
        handles.append(module.register_forward_hook(_forward_post))
    if capture_backward:
        handles.append(module.register_full_backward_pre_hook(_backward_pre))
        handles.append(module.register_full_backward_hook(_backward_post))
    return handles
