from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
import statistics
from time import perf_counter
from typing import Any, Callable, TypeAlias, TypeVar

import torch
from torch.nn import functional as F

from _profiled_training_model import ProfiledTrainingLM
from _training_model import TrainingLM, configure_optim
from slinoss.blocks import SLinOSSBlockConfig, SLinOSSMixerConfig, SLinOSSStackConfig
from slinoss.layers import (
    AutoCConv1dBackend,
    AutoMixerDecodeBackend,
    AutoScanPrepBackend,
    CuteScanBackend,
    SLinOSSMLPConfig,
    SLinOSSMixer,
)
from slinoss.perf import PerfRecorder, call_region, record_region

T = TypeVar("T")
TrainingModel: TypeAlias = TrainingLM | ProfiledTrainingLM


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


def _block_dict(block: SLinOSSBlockConfig, *, n_heads: int) -> dict[str, Any]:
    ffn = None
    if block.ffn is not None:
        ffn = {
            "kind": block.ffn.kind,
            "hidden_dim": block.ffn.hidden_dim,
            "expand": block.ffn.expand,
            "multiple_of": block.ffn.multiple_of,
            "bias": block.ffn.bias,
        }
    mixer = {
        "d_state": block.mixer.d_state,
        "expand": block.mixer.expand,
        "d_head": block.mixer.d_head,
        "d_conv": block.mixer.d_conv,
        "chunk_size": block.mixer.chunk_size,
        "bc_groups": n_heads
        if block.mixer.bc_groups is None
        else int(block.mixer.bc_groups),
        "dt_min": block.mixer.dt_min,
        "dt_max": block.mixer.dt_max,
        "dt_init_floor": block.mixer.dt_init_floor,
        "gamma_min": block.mixer.gamma_min,
        "gamma_max": block.mixer.gamma_max,
        "theta_init_min": block.mixer.theta_init_min,
        "theta_init_max": block.mixer.theta_init_max,
        "r_min": block.mixer.r_min,
        "r_max": block.mixer.r_max,
        "bc_gain_max": block.mixer.bc_gain_max,
        "eps": block.mixer.eps,
        "n_heads": n_heads,
    }
    return {
        "d_model": block.d_model,
        "norm_kind": block.norm_kind,
        "norm_eps": block.norm_eps,
        "residual_in_fp32": block.residual_in_fp32,
        "residual_dropout": block.residual_dropout,
        "mixer": mixer,
        "ffn": ffn,
    }


@dataclass(frozen=True)
class TrainingPerfConfig:
    batch_size: int = 4
    seq_len: int = 2048
    vocab_size: int = 256
    d_model: int = 576
    n_layers: int = 13
    mixer: SLinOSSMixerConfig = field(
        default_factory=lambda: SLinOSSMixerConfig(chunk_size=64, bc_groups=1)
    )
    ffn: SLinOSSMLPConfig | None = field(
        default_factory=lambda: SLinOSSMLPConfig(
            kind="gelu",
            expand=4.0,
            multiple_of=1,
            bias=True,
        )
    )
    norm_kind: str = "rmsnorm"
    norm_eps: float = 1.0e-5
    residual_in_fp32: bool = True
    residual_dropout: float = 0.0
    final_norm_kind: str = "rmsnorm"
    final_norm_eps: float = 1.0e-5
    gradient_checkpointing: bool = False
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    seed: int = 0

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def n_heads(self) -> int:
        d_inner = SLinOSSMixer._quantize_inner_dim(
            self.mixer.expand * self.d_model,
            self.mixer.d_head,
        )
        return d_inner // self.mixer.d_head

    @property
    def resolved_bc_groups(self) -> int:
        return (
            self.n_heads if self.mixer.bc_groups is None else int(self.mixer.bc_groups)
        )

    @property
    def block_config(self) -> SLinOSSBlockConfig:
        return SLinOSSBlockConfig(
            d_model=self.d_model,
            mixer=self.mixer,
            ffn=self.ffn,
            norm_kind=self.norm_kind,  # type: ignore[arg-type]
            norm_eps=self.norm_eps,
            residual_in_fp32=self.residual_in_fp32,
            residual_dropout=self.residual_dropout,
        )

    @property
    def stack_config(self) -> SLinOSSStackConfig:
        return SLinOSSStackConfig.uniform(
            self.block_config,
            n_layers=self.n_layers,
            final_norm_kind=self.final_norm_kind,  # type: ignore[arg-type]
            final_norm_eps=self.final_norm_eps,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    @property
    def perf_config_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "dtype": str(self.dtype),
            "device": self.device,
            "seed": self.seed,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "block": _block_dict(self.block_config, n_heads=self.n_heads),
            "stack": {
                "n_layers": self.n_layers,
                "final_norm_kind": self.final_norm_kind,
                "final_norm_eps": self.final_norm_eps,
                "gradient_checkpointing": self.gradient_checkpointing,
            },
        }


@dataclass(frozen=True)
class TrainingBenchFixture:
    initial_state: dict[str, torch.Tensor]
    batches: list[tuple[torch.Tensor, torch.Tensor]]
    model_seed: int
    batch_seed: int


DEFAULT_TRAINING_PERF_CONFIG = TrainingPerfConfig()


def build_model(
    cfg: TrainingPerfConfig,
    *,
    backend: str,
    instrumented: bool = False,
) -> tuple[TrainingModel, torch.optim.Optimizer]:
    model_cls = ProfiledTrainingLM if instrumented else TrainingLM
    model = model_cls(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        stack_config=cfg.stack_config,
    ).to(device=cfg.torch_device, dtype=cfg.dtype)
    model.perf_trainable_params = tuple(
        param for param in model.parameters() if param.requires_grad
    )
    optimizer = configure_optim(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    _configure_backend(model, backend=backend)
    return model, optimizer


def _configure_backend(model: TrainingModel, *, backend: str) -> None:
    if backend != "cute":
        raise ValueError(f"Unsupported backend: {backend}")
    scan_backend = CuteScanBackend(compute_dtype=torch.float32)
    scanprep_backend = AutoScanPrepBackend()
    cconv_backend = AutoCConv1dBackend()
    decode_backend = AutoMixerDecodeBackend()
    for module in model.modules():
        if isinstance(module, SLinOSSMixer):
            module.scan_backend = scan_backend
            module.scanprep.backend = scanprep_backend
            module.cconv_backend = cconv_backend
            module.decode_backend = decode_backend


def random_batch(cfg: TrainingPerfConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.seq_len),
        device=cfg.torch_device,
        dtype=torch.long,
    )
    y = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.seq_len),
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
    cfg: TrainingPerfConfig,
    *,
    total_batches: int,
) -> TrainingBenchFixture:
    devices = _rng_devices(cfg.torch_device)
    model_seed = int(cfg.seed)
    batch_seed = int(cfg.seed) + 1
    with torch.random.fork_rng(devices=devices):
        _seed_fixture_rng(model_seed)
        model, optimizer = build_model(cfg, backend="cute", instrumented=False)
        initial_state = deepcopy(model.state_dict())
        del model, optimizer

        _seed_fixture_rng(batch_seed)
        batches = [random_batch(cfg) for _ in range(total_batches)]

    return TrainingBenchFixture(
        initial_state=initial_state,
        batches=batches,
        model_seed=model_seed,
        batch_seed=batch_seed,
    )


def _clip_params(model: TrainingModel) -> tuple[torch.nn.Parameter, ...]:
    clip_params = model.perf_trainable_params
    if not clip_params:
        clip_params = tuple(
            param for param in model.parameters() if param.requires_grad
        )
        model.perf_trainable_params = clip_params
    return clip_params


def _materialized_param_grads(model: TrainingModel) -> tuple[torch.Tensor, ...]:
    return tuple(param.grad for param in model.parameters() if param.grad is not None)


def _zero_param_grads_in_place(grads: tuple[torch.Tensor, ...]) -> None:
    if not grads:
        return
    if len(grads) == 1:
        grads[0].zero_()
        return
    torch._foreach_zero_(grads)


class _TrainingCudaGraphTrainer:
    def __init__(
        self,
        model: TrainingModel,
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
    model: TrainingModel,
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
    model: TrainingModel,
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
            loss = call_region("head.loss", _cross_entropy_logits, logits, yb)
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
    cfg: TrainingPerfConfig,
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
        param for param in model.parameters() if param.requires_grad
    )

    cold_recorder = PerfRecorder(device=cfg.torch_device)
    cold_xb, cold_yb = batches[0]
    with cold_recorder.capture_step():
        logits, loss = run_train_step_profiled(
            model,
            optimizer,
            cold_xb,
            cold_yb,
            grad_clip=cfg.grad_clip,
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
    cfg: TrainingPerfConfig,
    *,
    backend: str,
    initial_state: dict[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    steps: int,
) -> tuple[float, dict[str, int], list[float], list[dict[str, int]], str]:
    use_cuda_graph = (
        backend == "cute"
        and cfg.torch_device.type == "cuda"
        and os.environ.get("SLINOSS_DISABLE_CUDA_GRAPH_BENCH") != "1"
    )
    model, optimizer = build_model(cfg, backend=backend, instrumented=False)
    model.load_state_dict(initial_state)
    model.perf_trainable_params = tuple(
        param for param in model.parameters() if param.requires_grad
    )

    cold_xb, cold_yb = batches[0]
    cold_step_ms, (logits, loss), cold_memory = _time_step_with_memory(
        lambda: run_train_step_clean(
            model,
            optimizer,
            cold_xb,
            cold_yb,
            grad_clip=cfg.grad_clip,
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
            param for param in model.parameters() if param.requires_grad
        )

        capture_xb, capture_yb = batches[1]
        trainer = _TrainingCudaGraphTrainer(
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

        return cold_step_ms, cold_memory, warm_step_ms, warm_memory, "cuda_graph_replay"

    for xb, yb in batches[1 : 1 + warmup]:
        run_train_step_clean(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)

    warm_step_ms: list[float] = []
    warm_memory: list[dict[str, int]] = []
    for xb, yb in batches[1 + warmup : 1 + warmup + steps]:
        step_ms, _, step_memory = _time_step_with_memory(
            lambda xb=xb, yb=yb: run_train_step_clean(
                model,
                optimizer,
                xb,
                yb,
                grad_clip=cfg.grad_clip,
            ),
            device=cfg.torch_device,
        )
        warm_step_ms.append(step_ms)
        warm_memory.append(step_memory)

    return cold_step_ms, cold_memory, warm_step_ms, warm_memory, "eager"


def run_bench_step(
    cfg: TrainingPerfConfig,
    *,
    backend: str,
    warmup: int,
    steps: int,
    repeat: int = 1,
    fixture: TrainingBenchFixture | None = None,
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
    warm_execution_mode = "eager"
    repeat_count = max(1, int(repeat))
    for repeat_idx in range(repeat_count):
        (
            repeat_cold_ms,
            repeat_cold_memory,
            repeat_warm_step_ms,
            repeat_warm_memory,
            repeat_warm_execution_mode,
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
            warm_execution_mode = repeat_warm_execution_mode
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
        "warm_execution_mode": warm_execution_mode,
        "tokens_per_step": cfg.batch_size * cfg.seq_len,
    }


__all__ = [
    "DEFAULT_TRAINING_PERF_CONFIG",
    "TrainingBenchFixture",
    "TrainingPerfConfig",
    "build_bench_fixture",
    "build_model",
    "random_batch",
    "run_bench_step",
    "run_train_step_clean",
    "run_train_step_profiled",
]
