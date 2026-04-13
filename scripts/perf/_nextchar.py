from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import os
import statistics
from time import perf_counter
from typing import Any, Callable, TypeAlias, TypeVar

import torch
from torch.nn import functional as F

from _nextchar_model import NextCharLM, configure_optim
from _profiled_nextchar_model import ProfiledNextCharLM
from slinoss.layers import SLinOSSMixer
from slinoss.layers.backend import (
    AutoCConv1dBackend,
    AutoMixerDecodeBackend,
    AutoScanPrepBackend,
    CuteScanBackend,
    ReferenceCConv1dBackend,
    ReferenceMixerDecodeBackend,
    ReferenceScanBackend,
    ReferenceScanPrepBackend,
)
from slinoss.perf import PerfRecorder, call_region, record_region

T = TypeVar("T")
NextCharModel: TypeAlias = NextCharLM | ProfiledNextCharLM


def _cross_entropy_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _step_memory_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


@dataclass(frozen=True)
class NextCharPerfConfig:
    batch_size: int = 4
    block_size: int = 2048
    vocab_size: int = 256
    d_model: int = 736
    n_layers: int = 3
    d_state: int = 128
    expand: float = 2.0
    d_head: int = 64
    d_conv: int = 4
    chunk_size: int = 32
    bc_groups: int | None = None
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    seed: int = 0

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def n_heads(self) -> int:
        d_inner = SLinOSSMixer._quantize_inner_dim(
            self.expand * self.d_model,
            self.d_head,
        )
        return d_inner // self.d_head

    @property
    def resolved_bc_groups(self) -> int:
        return self.n_heads if self.bc_groups is None else int(self.bc_groups)

    @property
    def perf_config_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dtype"] = str(self.dtype)
        data["bc_groups"] = self.resolved_bc_groups
        return data


@dataclass(frozen=True)
class NextCharBenchFixture:
    initial_state: dict[str, torch.Tensor]
    batches: list[tuple[torch.Tensor, torch.Tensor]]
    model_seed: int
    batch_seed: int


# Largest training-suite shape verified to complete on the local RTX 3060 12GB.
DEFAULT_NEXTCHAR_PERF_CONFIG = NextCharPerfConfig()


def build_model(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    instrumented: bool = False,
) -> tuple[NextCharModel, torch.optim.Optimizer]:
    model_cls = ProfiledNextCharLM if instrumented else NextCharLM
    model = model_cls(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        expand=cfg.expand,
        d_head=cfg.d_head,
        d_conv=cfg.d_conv,
        chunk_size=cfg.chunk_size,
        bc_groups=cfg.resolved_bc_groups,
    ).to(device=cfg.torch_device, dtype=cfg.dtype)
    model.perf_trainable_params = tuple(
        p for p in model.parameters() if p.requires_grad
    )
    optimizer = configure_optim(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    _configure_backend(model, backend=backend)
    return model, optimizer


def _configure_backend(model: NextCharModel, *, backend: str) -> None:
    if backend not in ("reference", "cute"):
        raise ValueError(f"Unsupported backend: {backend}")
    scan_backend = (
        ReferenceScanBackend(compute_dtype=torch.float32)
        if backend == "reference"
        else CuteScanBackend(compute_dtype=torch.float32)
    )
    scanprep_backend = (
        ReferenceScanPrepBackend() if backend == "reference" else AutoScanPrepBackend()
    )
    cconv_backend = (
        ReferenceCConv1dBackend() if backend == "reference" else AutoCConv1dBackend()
    )
    decode_backend = (
        ReferenceMixerDecodeBackend()
        if backend == "reference"
        else AutoMixerDecodeBackend()
    )
    for module in model.modules():
        if isinstance(module, SLinOSSMixer):
            module.scan_backend = scan_backend
            module.scanprep.backend = scanprep_backend
            module.cconv_backend = cconv_backend
            module.decode_backend = decode_backend


def random_batch(cfg: NextCharPerfConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.block_size),
        device=cfg.torch_device,
        dtype=torch.long,
    )
    y = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.block_size),
        device=cfg.torch_device,
        dtype=torch.long,
    )
    return x, y


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    if device.index is not None:
        return [int(device.index)]
    return [int(torch.cuda.current_device())]


def _seed_fixture_rng(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_bench_fixture(
    cfg: NextCharPerfConfig,
    *,
    total_batches: int,
) -> NextCharBenchFixture:
    devices = _rng_devices(cfg.torch_device)
    model_seed = int(cfg.seed)
    batch_seed = int(cfg.seed) + 1
    with torch.random.fork_rng(devices=devices):
        _seed_fixture_rng(model_seed)
        model, optimizer = build_model(cfg, backend="reference", instrumented=False)
        initial_state = deepcopy(model.state_dict())
        del model, optimizer

        _seed_fixture_rng(batch_seed)
        batches = [random_batch(cfg) for _ in range(total_batches)]

    return NextCharBenchFixture(
        initial_state=initial_state,
        batches=batches,
        model_seed=model_seed,
        batch_seed=batch_seed,
    )


def _clip_params(model: NextCharModel) -> tuple[torch.nn.Parameter, ...]:
    clip_params = model.perf_trainable_params
    if not clip_params:
        clip_params = tuple(p for p in model.parameters() if p.requires_grad)
        model.perf_trainable_params = clip_params
    return clip_params


def _materialized_param_grads(model: NextCharModel) -> tuple[torch.Tensor, ...]:
    return tuple(param.grad for param in model.parameters() if param.grad is not None)


def _zero_param_grads_in_place(grads: tuple[torch.Tensor, ...]) -> None:
    if not grads:
        return
    if len(grads) == 1:
        grads[0].zero_()
        return
    torch._foreach_zero_(grads)


class _NextCharCudaGraphTrainer:
    def __init__(
        self,
        model: NextCharModel,
        optimizer: torch.optim.Optimizer,
        *,
        xb: torch.Tensor,
        yb: torch.Tensor,
        grad_clip: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = float(grad_clip)
        self.device = xb.device
        self.static_xb = xb.clone()
        self.static_yb = yb.clone()
        self.graph = torch.cuda.CUDAGraph()
        self.static_logits: torch.Tensor | None = None
        self.static_loss: torch.Tensor | None = None
        self._grad_buffers: tuple[torch.Tensor, ...] = ()
        self._capture()

    def _run_body(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(self.static_xb)
        loss = _cross_entropy_logits(logits, self.static_yb)
        loss.backward()
        return logits, loss

    def _capture(self) -> None:
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(stream):
            # Match the captured path's persistent grad-buffer behavior during warmup.
            self.optimizer.zero_grad(set_to_none=False)
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=False)
                self._run_body()
        torch.cuda.current_stream(device=self.device).wait_stream(stream)
        self.optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(self.graph):
            self.static_logits, self.static_loss = self._run_body()
        self._grad_buffers = _materialized_param_grads(self.model)

    def step(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.static_xb.copy_(xb)
        self.static_yb.copy_(yb)
        _zero_param_grads_in_place(self._grad_buffers)
        self.graph.replay()
        torch.nn.utils.clip_grad_norm_(
            _clip_params(self.model),
            max_norm=self.grad_clip,
            foreach=xb.device.type == "cuda",
        )
        self.optimizer.step()
        if self.static_logits is None or self.static_loss is None:
            raise RuntimeError("CUDA graph did not materialize logits/loss.")
        return self.static_logits, self.static_loss


def run_train_step_clean(
    model: NextCharModel,
    optimizer: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    grad_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = _cross_entropy_logits(logits, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        _clip_params(model),
        max_norm=grad_clip,
        foreach=xb.device.type == "cuda",
    )
    optimizer.step()
    return logits, loss


def run_train_step_profiled(
    model: NextCharModel,
    optimizer: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    grad_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with record_region("step.total"):
        with record_region("step.zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with record_region("step.forward_loss"):
            logits = model(xb)
            loss = call_region(
                "head.loss",
                _cross_entropy_logits,
                logits,
                yb,
            )
        with record_region("step.backward"):
            loss.backward()
        with record_region("step.clip"):
            torch.nn.utils.clip_grad_norm_(
                _clip_params(model),
                max_norm=grad_clip,
                foreach=xb.device.type == "cuda",
            )
        with record_region("step.optim"):
            optimizer.step()
    return logits, loss


def _time_step(
    fn: Callable[[], T],
    *,
    device: torch.device,
) -> tuple[float, T]:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        stream = torch.cuda.current_stream(device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        out = fn()
        end.record(stream)
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end)), out

    started = perf_counter()
    out = fn()
    ended = perf_counter()
    return (ended - started) * 1000.0, out


def _time_step_with_memory(
    fn: Callable[[], T],
    *,
    device: torch.device,
) -> tuple[float, T, dict[str, int]]:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        stream = torch.cuda.current_stream(device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        out = fn()
        end.record(stream)
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end)), out, _step_memory_stats(device)

    started = perf_counter()
    out = fn()
    ended = perf_counter()
    return (ended - started) * 1000.0, out, _step_memory_stats(device)


def _run_profiled_sequence(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    initial_state: dict[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, optimizer = build_model(cfg, backend=backend, instrumented=True)
    model.load_state_dict(initial_state)
    model.perf_trainable_params = tuple(
        p for p in model.parameters() if p.requires_grad
    )

    cold_recorder = PerfRecorder(device=cfg.torch_device)
    cold_xb, cold_yb = batches[0]
    with cold_recorder.capture_step():
        logits, loss = run_train_step_profiled(
            model, optimizer, cold_xb, cold_yb, grad_clip=cfg.grad_clip
        )
    del logits, loss
    cold = cold_recorder.steps[-1]

    for xb, yb in batches[1 : 1 + warmup]:
        recorder = PerfRecorder(device=cfg.torch_device)
        with recorder.capture_step():
            run_train_step_profiled(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)

    warm_steps: list[dict[str, Any]] = []
    for xb, yb in batches[1 + warmup : 1 + warmup + steps]:
        recorder = PerfRecorder(device=cfg.torch_device)
        with recorder.capture_step():
            run_train_step_profiled(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)
        warm_steps.append(recorder.steps[-1])

    return cold, warm_steps


def _run_clean_sequence(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    initial_state: dict[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    steps: int,
) -> tuple[float, dict[str, int], list[float], list[dict[str, int]]]:
    use_cuda_graph = (
        backend == "cute"
        and cfg.torch_device.type == "cuda"
        and os.environ.get("SLINOSS_DISABLE_CUDA_GRAPH_BENCH") != "1"
    )
    model, optimizer = build_model(cfg, backend=backend, instrumented=False)
    model.load_state_dict(initial_state)
    model.perf_trainable_params = tuple(
        p for p in model.parameters() if p.requires_grad
    )

    cold_xb, cold_yb = batches[0]
    cold_step_ms, (logits, loss), cold_memory = _time_step_with_memory(
        lambda: run_train_step_clean(
            model, optimizer, cold_xb, cold_yb, grad_clip=cfg.grad_clip
        ),
        device=cfg.torch_device,
    )
    del logits, loss

    if use_cuda_graph:
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        del model, optimizer

        model, optimizer = build_model(cfg, backend=backend, instrumented=False)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        model.perf_trainable_params = tuple(
            p for p in model.parameters() if p.requires_grad
        )

        capture_xb, capture_yb = batches[1]
        trainer = _NextCharCudaGraphTrainer(
            model,
            optimizer,
            xb=capture_xb,
            yb=capture_yb,
            grad_clip=cfg.grad_clip,
        )

        for xb, yb in batches[1 : 1 + warmup]:
            trainer.step(xb, yb)

        warm_step_ms: list[float] = []
        warm_memory: list[dict[str, int]] = []
        for xb, yb in batches[1 + warmup : 1 + warmup + steps]:
            step_ms, _, step_memory = _time_step_with_memory(
                lambda xb=xb, yb=yb: trainer.step(xb, yb),
                device=cfg.torch_device,
            )
            warm_step_ms.append(step_ms)
            warm_memory.append(step_memory)

        return cold_step_ms, cold_memory, warm_step_ms, warm_memory

    for xb, yb in batches[1 : 1 + warmup]:
        run_train_step_clean(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)

    warm_step_ms: list[float] = []
    warm_memory: list[dict[str, int]] = []
    for xb, yb in batches[1 + warmup : 1 + warmup + steps]:
        step_ms, _, step_memory = _time_step_with_memory(
            lambda xb=xb, yb=yb: run_train_step_clean(
                model, optimizer, xb, yb, grad_clip=cfg.grad_clip
            ),
            device=cfg.torch_device,
        )
        warm_step_ms.append(step_ms)
        warm_memory.append(step_memory)

    return cold_step_ms, cold_memory, warm_step_ms, warm_memory


def run_bench_step(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    warmup: int,
    steps: int,
    repeat: int = 1,
    fixture: NextCharBenchFixture | None = None,
) -> dict[str, Any]:
    total_batches = 1 + int(warmup) + int(steps)
    fixture = (
        build_bench_fixture(cfg, total_batches=total_batches)
        if fixture is None
        else fixture
    )
    if len(fixture.batches) < total_batches:
        raise ValueError(
            f"Fixture has {len(fixture.batches)} batches, expected at least {total_batches}."
        )

    cold_step_ms = 0.0
    cold_memory: dict[str, int] = {}
    warm_step_ms: list[float] = []
    warm_memory: list[dict[str, int]] = []
    warm_repeat_step_mean_ms: list[float] = []
    repeat_count = max(1, int(repeat))
    for repeat_idx in range(repeat_count):
        (
            repeat_cold_ms,
            repeat_cold_memory,
            repeat_warm_step_ms,
            repeat_warm_memory,
        ) = _run_clean_sequence(
            cfg,
            backend=backend,
            initial_state=fixture.initial_state,
            batches=fixture.batches,
            warmup=warmup,
            steps=steps,
        )
        if repeat_idx == 0:
            cold_step_ms = repeat_cold_ms
            cold_memory = repeat_cold_memory
        warm_step_ms.extend(repeat_warm_step_ms)
        warm_memory.extend(repeat_warm_memory)
        if repeat_warm_step_ms:
            warm_repeat_step_mean_ms.append(
                float(statistics.fmean(repeat_warm_step_ms))
            )
        else:
            warm_repeat_step_mean_ms.append(0.0)

    cold_profile, warm_profile = _run_profiled_sequence(
        cfg,
        backend=backend,
        initial_state=fixture.initial_state,
        batches=fixture.batches,
        warmup=warmup,
        steps=steps,
    )

    return {
        "cold_step_ms": cold_step_ms,
        "cold_memory": cold_memory,
        "warm_step_ms": warm_step_ms,
        "warm_memory": warm_memory,
        "warm_repeat_step_mean_ms": warm_repeat_step_mean_ms,
        "cold_profile": cold_profile,
        "warm_profile": warm_profile,
        "repeat_count": repeat_count,
        "fixture": {
            "model_seed": fixture.model_seed,
            "batch_seed": fixture.batch_seed,
            "batch_count": len(fixture.batches),
        },
        "tokens_per_step": cfg.batch_size * cfg.block_size,
    }
