"""Low-overhead runtime instrumentation for benchmarked model regions."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass
from time import perf_counter
import threading
from typing import Any, Callable, Iterator, Protocol, TypeAlias, TypeVar, cast

import torch
from torch.autograd.profiler import record_function

T = TypeVar("T")
Stamp: TypeAlias = torch.cuda.Event | float
StampedBoundary: TypeAlias = tuple[int, Stamp]

_IS_PROFILER_ENABLED = getattr(torch.autograd.profiler, "_is_profiler_enabled", None)

_ACTIVE_STEPS: list["_PerfStep"] = []
_ACTIVE_STEPS_LOCK = threading.RLock()
_ACTIVE_REGION_OBSERVERS: list["_RegionObserver"] = []
_ACTIVE_REGION_OBSERVERS_LOCK = threading.RLock()


class _RegionObserver(Protocol):
    def region_enter(self, label: str) -> None: ...

    def region_exit(self, label: str) -> None: ...


def current_step() -> "_PerfStep | None":
    try:
        return _ACTIVE_STEPS[-1]
    except IndexError:
        return None


@contextmanager
def register_region_observer(observer: _RegionObserver) -> Iterator[None]:
    with _ACTIVE_REGION_OBSERVERS_LOCK:
        _ACTIVE_REGION_OBSERVERS.append(observer)
    try:
        yield
    finally:
        with _ACTIVE_REGION_OBSERVERS_LOCK:
            for idx in range(len(_ACTIVE_REGION_OBSERVERS) - 1, -1, -1):
                if _ACTIVE_REGION_OBSERVERS[idx] is observer:
                    del _ACTIVE_REGION_OBSERVERS[idx]
                    break
            else:
                raise RuntimeError("PerfRecorder region observer stack corrupted.")


def _snapshot_region_observers() -> tuple[_RegionObserver, ...]:
    with _ACTIVE_REGION_OBSERVERS_LOCK:
        return tuple(_ACTIVE_REGION_OBSERVERS)


def _notify_region_enter(label: str) -> None:
    for observer in _snapshot_region_observers():
        observer.region_enter(label)


def _notify_region_exit(label: str) -> None:
    for observer in _snapshot_region_observers():
        observer.region_exit(label)


def _profiler_enabled() -> bool:
    if callable(_IS_PROFILER_ENABLED):
        return bool(_IS_PROFILER_ENABLED())
    if _IS_PROFILER_ENABLED is None:
        return True
    return bool(_IS_PROFILER_ENABLED)


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
    has_region_observers = bool(_snapshot_region_observers())
    if step is None and not has_region_observers:
        yield
        return

    use_record_function = _profiler_enabled()
    maybe_record = record_function(label) if use_record_function else nullcontext()

    if step is None:
        _notify_region_enter(label)
        with maybe_record:
            try:
                yield
            finally:
                _notify_region_exit(label)
        return

    if step.uses_cuda_events:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _notify_region_enter(label)
        with maybe_record:
            start.record(torch.cuda.current_stream(device=step.device))
            try:
                yield
            finally:
                end.record(torch.cuda.current_stream(device=step.device))
                step.add_region(label, start, end)
                _notify_region_exit(label)
        return

    start_t = perf_counter()
    _notify_region_enter(label)
    with maybe_record:
        try:
            yield
        finally:
            end_t = perf_counter()
            step.add_region(label, start_t, end_t)
            _notify_region_exit(label)


def note_cache_event(label: str, *, hit: bool) -> None:
    step = current_step()
    if step is not None:
        step.note_cache_event(label, hit=hit)


def _make_boundary_hook(
    *, backward_label: str, token_id: int, kind: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _hook(grad: torch.Tensor) -> torch.Tensor:
        step = current_step()
        if step is not None:
            step.record_backward_boundary(backward_label, token_id, kind=kind)
        return grad

    return _hook


def _alias_with_backward_exit_hook(
    tensor: torch.Tensor,
    *,
    backward_label: str,
    token_id: int,
) -> torch.Tensor:
    if not tensor.requires_grad:
        return tensor
    aliased = torch.ops.aten.alias.default(tensor)
    aliased.register_hook(
        _make_boundary_hook(
            backward_label=backward_label,
            token_id=token_id,
            kind="exit",
        )
    )
    return aliased


def _iter_unique_tensors(
    obj: Any, seen: set[int] | None = None
) -> Iterator[torch.Tensor]:
    if seen is None:
        seen = set()
    if isinstance(obj, torch.Tensor):
        obj_id = id(obj)
        if obj_id not in seen:
            seen.add(obj_id)
            yield obj
        return
    if isinstance(obj, tuple):
        for item in obj:
            yield from _iter_unique_tensors(item, seen)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_unique_tensors(item, seen)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_unique_tensors(value, seen)
        return
    if is_dataclass(obj) and not isinstance(obj, type):
        for field in fields(obj):
            yield from _iter_unique_tensors(getattr(obj, field.name), seen)
        return


def _attach_backward_enter_hooks(
    out: Any,
    *,
    backward_label: str,
    token_id: int,
) -> None:
    hook = _make_boundary_hook(
        backward_label=backward_label,
        token_id=token_id,
        kind="enter",
    )
    for tensor in _iter_unique_tensors(out):
        if tensor.requires_grad:
            tensor.register_hook(hook)


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

    forward_label = f"forward.{label}"
    if not capture_backward:
        with record_region(forward_label):
            return fn(*args, **kwargs)

    token_id = step.next_backward_token()
    backward_label = f"backward.{label}"

    if not kwargs and len(args) == 1 and isinstance(args[0], torch.Tensor):
        arg0 = _alias_with_backward_exit_hook(
            args[0],
            backward_label=backward_label,
            token_id=token_id,
        )
        with record_region(forward_label):
            out = fn(arg0)
        _attach_backward_enter_hooks(
            out,
            backward_label=backward_label,
            token_id=token_id,
        )
        return out

    if (
        not kwargs
        and len(args) == 2
        and isinstance(args[0], torch.Tensor)
        and isinstance(args[1], torch.Tensor)
    ):
        arg0 = _alias_with_backward_exit_hook(
            args[0],
            backward_label=backward_label,
            token_id=token_id,
        )
        arg1 = _alias_with_backward_exit_hook(
            args[1],
            backward_label=backward_label,
            token_id=token_id,
        )
        with record_region(forward_label):
            out = fn(arg0, arg1)
        _attach_backward_enter_hooks(
            out,
            backward_label=backward_label,
            token_id=token_id,
        )
        return out

    args = tuple(
        _tree_map_tensors(
            arg,
            lambda t: _alias_with_backward_exit_hook(
                t,
                backward_label=backward_label,
                token_id=token_id,
            ),
        )
        for arg in args
    )
    kwargs = {
        key: _tree_map_tensors(
            value,
            lambda t: _alias_with_backward_exit_hook(
                t,
                backward_label=backward_label,
                token_id=token_id,
            ),
        )
        for key, value in kwargs.items()
    }

    with record_region(forward_label):
        out = fn(*args, **kwargs)

    _attach_backward_enter_hooks(
        out,
        backward_label=backward_label,
        token_id=token_id,
    )
    return out


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
