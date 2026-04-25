from __future__ import annotations

import math
import statistics
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute  # noqa: E402
from slinoss.perf import PerfRecorder, record_region  # noqa: E402
from slinoss.perf.budget import summarize_cache_samples, summarize_named_samples  # noqa: E402

DEFAULT_BATCH = 16
DEFAULT_HEADS = 4
DEFAULT_T = 2048
DEFAULT_N = 48
DEFAULT_P = 64
DEFAULT_CHUNK = 64
DEFAULT_DTYPE = "fp16"
DIRECTIONS = ("forward", "backward")


@dataclass(frozen=True)
class PerfConfig:
    batch: int = DEFAULT_BATCH
    heads: int = DEFAULT_HEADS
    T: int = DEFAULT_T
    N: int = DEFAULT_N
    P: int = DEFAULT_P
    chunk_size: int = DEFAULT_CHUNK
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    seed: int = 0

    @property
    def D(self) -> int:
        return 2 * int(self.N)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


def dtype_from_str(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def ensure_cuda(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)


def format_header(cfg: PerfConfig) -> str:
    return (
        f"B={cfg.batch} H={cfg.heads} T={cfg.T} N={cfg.N} "
        f"P={cfg.P} L={cfg.chunk_size} dtype={_dtype_name(cfg.dtype)}"
    )


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype)


def benchmark(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    repeat: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    samples = [_time_once(fn, iterations) for _ in range(repeat)]
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark_instrumented(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        recorder = PerfRecorder(device=device)
        with recorder.capture_step():
            fn()

    samples: list[float] = []
    region_samples: list[dict[str, float]] = []
    cache_samples: list[dict[str, dict[str, int]]] = []
    for _ in range(repeat):
        sample_ms, sample_regions, sample_caches = _time_once_instrumented(
            fn, iterations=iterations, device=device
        )
        samples.append(sample_ms)
        region_samples.extend(sample_regions)
        cache_samples.extend(sample_caches)

    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "region_summaries": summarize_named_samples(region_samples),
        "cache_events": summarize_cache_samples(cache_samples),
        "region_samples": region_samples,
    }


def _time_once(fn: Callable[[], object], iterations: int) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / max(1, iterations))

    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    ended = time.perf_counter()
    return (ended - started) * 1000.0 / max(1, iterations)


def _time_once_instrumented(
    fn: Callable[[], object],
    *,
    iterations: int,
    device: torch.device,
) -> tuple[float, list[dict[str, float]], list[dict[str, dict[str, int]]]]:
    region_samples: list[dict[str, float]] = []
    cache_samples: list[dict[str, dict[str, int]]] = []
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device=device))
        for _ in range(iterations):
            recorder = PerfRecorder(device=device)
            with recorder.capture_step():
                fn()
            capture = recorder.steps[-1]
            region_samples.append(capture["regions_ms"])
            cache_samples.append(capture["cache_events"])
        end.record(torch.cuda.current_stream(device=device))
        torch.cuda.synchronize(device)
        return (
            float(start.elapsed_time(end) / max(1, iterations)),
            region_samples,
            cache_samples,
        )

    started = time.perf_counter()
    for _ in range(iterations):
        recorder = PerfRecorder(device=device)
        with recorder.capture_step():
            fn()
        capture = recorder.steps[-1]
        region_samples.append(capture["regions_ms"])
        cache_samples.append(capture["cache_events"])
    ended = time.perf_counter()
    return (
        (ended - started) * 1000.0 / max(1, iterations),
        region_samples,
        cache_samples,
    )


def make_inputs(cfg: PerfConfig) -> dict[str, torch.Tensor]:
    device = cfg.torch_device
    batch, heads, T, N, P = cfg.batch, cfg.heads, cfg.T, cfg.N, cfg.P

    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=cfg.dtype)
    B = torch.randn((batch, heads, T, 2 * N), device=device, dtype=cfg.dtype) * 0.1
    C = torch.randn((batch, heads, T, 2 * N), device=device, dtype=cfg.dtype) * 0.1
    initial_states = torch.randn(
        (batch, heads, P, 2 * N), device=device, dtype=cfg.dtype
    )

    b_prev = (
        torch.randn((batch, heads, N), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, N), device=device, dtype=torch.float32)
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=cfg.dtype)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=cfg.dtype)
    return {
        "U": U.contiguous(),
        "M": M,
        "K": K,
        "B": B.contiguous(),
        "C": C.contiguous(),
        "initial_states": initial_states.contiguous(),
        "B_prev": B_prev.contiguous(),
        "U_prev": U_prev.contiguous(),
    }


def build_v2x2ssd_callable(
    cfg: PerfConfig,
    *,
    direction: str,
    backend: str,
) -> Callable[[], object]:
    if direction == "forward":
        fn = _build_forward_callable(cfg, backend=backend)
    elif direction == "backward":
        fn = _build_backward_callable(cfg, backend=backend)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    region_label = f"{direction}.v2x2ssd.total"

    def wrapped() -> object:
        with record_region(region_label):
            return fn()

    return wrapped


def _build_forward_callable(cfg: PerfConfig, *, backend: str) -> Callable[[], object]:
    tensors = make_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    C = tensors["C"]
    initial_states = tensors["initial_states"]
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    if backend == "reference":
        fn = partial(
            v2x2ssd,
            U,
            M,
            K,
            B,
            C,
            chunk_size=cfg.chunk_size,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
    else:
        fn = partial(
            v2x2ssd_cute,
            U,
            M,
            K,
            B,
            C,
            chunk_size=cfg.chunk_size,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
    fn()
    return fn


def _build_backward_callable(cfg: PerfConfig, *, backend: str) -> Callable[[], object]:
    tensors = make_inputs(cfg)
    return _build_full_backward_callable(cfg, tensors=tensors, backend=backend)


def _build_full_backward_callable(
    cfg: PerfConfig,
    *,
    tensors: dict[str, torch.Tensor],
    backend: str,
) -> Callable[[], object]:
    dY = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )

    def fn() -> None:
        U = _clone_requires_grad(tensors["U"])
        M = _clone_requires_grad(tensors["M"])
        K = _clone_requires_grad(tensors["K"])
        B = _clone_requires_grad(tensors["B"])
        C = _clone_requires_grad(tensors["C"])
        if backend == "reference":
            initial_states = _clone_requires_grad(tensors["initial_states"])
            B_prev = _clone_requires_grad(tensors["B_prev"])
            U_prev = _clone_requires_grad(tensors["U_prev"])
            d_final = torch.randn(
                (cfg.batch, cfg.heads, cfg.P, cfg.D),
                device=cfg.torch_device,
                dtype=torch.float32,
            )
            dB_last = torch.randn(
                (cfg.batch, cfg.heads, cfg.D),
                device=cfg.torch_device,
                dtype=torch.float32,
            )
            dU_last = torch.randn(
                (cfg.batch, cfg.heads, cfg.P),
                device=cfg.torch_device,
                dtype=torch.float32,
            )
            Y, final_state, B_last, U_last = v2x2ssd(
                U,
                M,
                K,
                B,
                C,
                chunk_size=cfg.chunk_size,
                initial_states=initial_states,
                B_prev=B_prev,
                U_prev=U_prev,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
            loss = (
                (Y * dY).sum()
                + (final_state * d_final).sum()
                + (B_last * dB_last).sum()
                + (U_last * dU_last).sum()
            )
            torch.autograd.grad(loss, (U, M, K, B, C, initial_states, B_prev, U_prev))
            return

        Y = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=cfg.chunk_size,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        loss = (Y * dY).sum()
        torch.autograd.grad(loss, (U, M, K, B, C))

    fn()
    return fn


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _clone_requires_grad(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)
