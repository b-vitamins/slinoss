from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
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
    AutoScanPrepBackend,
    CuteScanBackend,
    ReferenceCConv1dBackend,
    ReferenceScanBackend,
    ReferenceScanPrepBackend,
)
from slinoss.perf import PerfRecorder, call_region, record_region

T = TypeVar("T")
NextCharModel: TypeAlias = NextCharLM | ProfiledNextCharLM


def _cross_entropy_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@dataclass(frozen=True)
class NextCharPerfConfig:
    batch_size: int = 12
    block_size: int = 128
    vocab_size: int = 256
    d_model: int = 96
    n_layers: int = 2
    d_state: int = 16
    expand: int = 2
    d_head: int = 32
    d_conv: int = 4
    chunk_size: int = 32
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
        return (self.expand * self.d_model) // self.d_head

    @property
    def perf_config_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dtype"] = str(self.dtype)
        return data


@dataclass(frozen=True)
class NextCharBenchFixture:
    initial_state: dict[str, torch.Tensor]
    batches: list[tuple[torch.Tensor, torch.Tensor]]
    model_seed: int
    batch_seed: int


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
    for module in model.modules():
        if isinstance(module, SLinOSSMixer):
            module.backend = scan_backend
            module.scanprep.backend = scanprep_backend
            module.cconv_backend = cconv_backend


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
) -> tuple[float, list[float]]:
    model, optimizer = build_model(cfg, backend=backend, instrumented=False)
    model.load_state_dict(initial_state)
    model.perf_trainable_params = tuple(
        p for p in model.parameters() if p.requires_grad
    )

    cold_xb, cold_yb = batches[0]
    cold_step_ms, (logits, loss) = _time_step(
        lambda: run_train_step_clean(
            model, optimizer, cold_xb, cold_yb, grad_clip=cfg.grad_clip
        ),
        device=cfg.torch_device,
    )
    del logits, loss

    for xb, yb in batches[1 : 1 + warmup]:
        run_train_step_clean(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)

    warm_step_ms: list[float] = []
    for xb, yb in batches[1 + warmup : 1 + warmup + steps]:
        step_ms, _ = _time_step(
            lambda xb=xb, yb=yb: run_train_step_clean(
                model, optimizer, xb, yb, grad_clip=cfg.grad_clip
            ),
            device=cfg.torch_device,
        )
        warm_step_ms.append(step_ms)

    return cold_step_ms, warm_step_ms


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
    warm_step_ms: list[float] = []
    warm_repeat_step_mean_ms: list[float] = []
    repeat_count = max(1, int(repeat))
    for repeat_idx in range(repeat_count):
        repeat_cold_ms, repeat_warm_step_ms = _run_clean_sequence(
            cfg,
            backend=backend,
            initial_state=fixture.initial_state,
            batches=fixture.batches,
            warmup=warmup,
            steps=steps,
        )
        if repeat_idx == 0:
            cold_step_ms = repeat_cold_ms
        warm_step_ms.extend(repeat_warm_step_ms)
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
        "warm_step_ms": warm_step_ms,
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
