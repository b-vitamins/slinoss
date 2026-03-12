from __future__ import annotations

import math
import statistics
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute  # noqa: E402
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import (  # noqa: E402
    compile_chunk_scan_bwd_kernels,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment import (  # noqa: E402
    chunk_increment_bwd_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import (  # noqa: E402
    compile_state_passing_bwd_kernels,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (  # noqa: E402
    chunk_increment_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import chunk_scan_cute  # noqa: E402
from slinoss.ops.v2x2ssd.cute.kernels.fwd.state_passing import (  # noqa: E402
    state_passing_cute,
)
from slinoss.ops.v2x2ssd.reference import (  # noqa: E402
    chunk_increment as ref_chunk_increment,
)
from slinoss.ops.v2x2ssd.reference import chunk_scan as ref_chunk_scan  # noqa: E402
from slinoss.ops.v2x2ssd.reference import state_passing as ref_state_passing  # noqa: E402

DEFAULT_BATCH = 16
DEFAULT_HEADS = 4
DEFAULT_T = 2048
DEFAULT_N = 48
DEFAULT_P = 64
DEFAULT_CHUNK = 64
DEFAULT_DTYPE = "fp16"
STAGES = ("chunk_increment", "state_passing", "chunk_scan", "full")
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
    fn: Callable[[], None],
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


def _time_once(fn: Callable[[], None], iterations: int) -> float:
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


def build_callable(
    cfg: PerfConfig,
    *,
    stage: str,
    direction: str,
    backend: str,
) -> Callable[[], None]:
    if direction == "forward":
        return _build_forward_callable(cfg, stage=stage, backend=backend)
    if direction == "backward":
        return _build_backward_callable(cfg, stage=stage, backend=backend)
    raise ValueError(f"Unsupported direction: {direction}")


def _build_forward_callable(
    cfg: PerfConfig, *, stage: str, backend: str
) -> Callable[[], None]:
    tensors = make_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    C = tensors["C"]
    initial_states = tensors["initial_states"]
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    if stage == "full":
        fn = partial(
            v2x2ssd if backend == "reference" else v2x2ssd_cute,
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

    if stage == "chunk_increment":
        fn = partial(
            ref_chunk_increment if backend == "reference" else chunk_increment_cute,
            U,
            M,
            K,
            B,
            B_prev=B_prev,
            U_prev=U_prev,
            T=cfg.T if backend == "reference" else None,
            chunk_size=cfg.chunk_size,
            compute_dtype=torch.float32,
        )
        if backend == "cute":
            fn = partial(
                chunk_increment_cute,
                U,
                M,
                K,
                B,
                chunk_size=cfg.chunk_size,
                B_prev=B_prev,
                U_prev=U_prev,
                compute_dtype=torch.float32,
            )
        fn()
        return fn

    inc_ref, m_ref = ref_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=cfg.T,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
    )
    inc_cute, m_cute = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    if stage == "state_passing":
        fn = partial(
            ref_state_passing if backend == "reference" else state_passing_cute,
            inc_ref if backend == "reference" else inc_cute,
            m_ref if backend == "reference" else m_cute,
            initial_states=initial_states,
            compute_dtype=torch.float32,
        )
        fn()
        return fn

    starts_ref, _ = ref_state_passing(
        inc_ref,
        m_ref,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )
    starts_cute, _ = state_passing_cute(
        inc_cute,
        m_cute,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )

    if stage != "chunk_scan":
        raise ValueError(f"Unsupported forward stage: {stage}")

    fn = partial(
        ref_chunk_scan if backend == "reference" else chunk_scan_cute,
        U,
        M,
        K,
        B,
        C,
        starts_ref if backend == "reference" else starts_cute,
        B_prev=B_prev,
        U_prev=U_prev,
        T=cfg.T if backend == "reference" else None,
        chunk_size=cfg.chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    if backend == "cute":
        fn = partial(
            chunk_scan_cute,
            U,
            M,
            K,
            B,
            C,
            starts_cute,
            chunk_size=cfg.chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            output_dtype=torch.float32,
            compute_dtype=torch.float32,
        )
    fn()
    return fn


def _build_backward_callable(
    cfg: PerfConfig, *, stage: str, backend: str
) -> Callable[[], None]:
    tensors = make_inputs(cfg)
    if stage == "full":
        return _build_full_backward_callable(cfg, tensors=tensors, backend=backend)
    if stage == "chunk_increment":
        return _build_chunk_increment_backward_callable(
            cfg, tensors=tensors, backend=backend
        )
    if stage == "state_passing":
        return _build_state_passing_backward_callable(
            cfg, tensors=tensors, backend=backend
        )
    if stage == "chunk_scan":
        return _build_chunk_scan_backward_callable(
            cfg, tensors=tensors, backend=backend
        )
    raise ValueError(f"Unsupported backward stage: {stage}")


def _build_chunk_increment_backward_callable(
    cfg: PerfConfig,
    *,
    tensors: dict[str, torch.Tensor],
    backend: str,
) -> Callable[[], None]:
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    inc, m_chunk = ref_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=cfg.T,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)

    if backend == "reference":

        def fn() -> None:
            U_ref = _clone_requires_grad(U)
            M_ref = _clone_requires_grad(M)
            K_ref = _clone_requires_grad(K)
            B_ref = _clone_requires_grad(B)
            B_prev_ref = _clone_requires_grad(B_prev)
            U_prev_ref = _clone_requires_grad(U_prev)
            inc_ref, m_ref = ref_chunk_increment(
                U_ref,
                M_ref,
                K_ref,
                B_ref,
                B_prev=B_prev_ref,
                U_prev=U_prev_ref,
                T=cfg.T,
                chunk_size=cfg.chunk_size,
                compute_dtype=torch.float32,
            )
            loss = (inc_ref * d_inc).sum() + (m_ref * d_m_chunk).sum()
            torch.autograd.grad(
                loss, (U_ref, M_ref, K_ref, B_ref, B_prev_ref, U_prev_ref)
            )

        fn()
        return fn

    def fn() -> None:
        chunk_increment_bwd_cute(
            U,
            M,
            K,
            B,
            d_inc=d_inc,
            d_m_chunk=d_m_chunk,
            chunk_size=cfg.chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
        )

    fn()
    return fn


def _build_state_passing_backward_callable(
    cfg: PerfConfig,
    *,
    tensors: dict[str, torch.Tensor],
    backend: str,
) -> Callable[[], None]:
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    initial_states = tensors["initial_states"]
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    inc, m_chunk = ref_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=cfg.T,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts_ref, final_ref = ref_state_passing(
        inc,
        m_chunk,
        initial_states=initial_states.to(dtype=torch.float32),
        compute_dtype=torch.float32,
    )
    d_chunk_starts = torch.randn_like(chunk_starts_ref)
    d_final = torch.randn_like(final_ref)

    if backend == "reference":

        def fn() -> None:
            inc_ref = _clone_requires_grad(inc)
            m_ref = _clone_requires_grad(m_chunk)
            initial_ref = _clone_requires_grad(initial_states.to(dtype=torch.float32))
            starts_ref, final_state_ref = ref_state_passing(
                inc_ref,
                m_ref,
                initial_states=initial_ref,
                compute_dtype=torch.float32,
            )
            loss = (starts_ref * d_chunk_starts).sum() + (
                final_state_ref * d_final
            ).sum()
            torch.autograd.grad(loss, (inc_ref, m_ref, initial_ref))

        fn()
        return fn

    chunk_starts_cute, _ = state_passing_cute(
        inc,
        m_chunk,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )
    _, _, _, _, _, launch_pipeline = compile_state_passing_bwd_kernels(
        chunk_starts_cute,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
        return_launchers=True,
    )
    launch_pipeline()
    return launch_pipeline


def _build_chunk_scan_backward_callable(
    cfg: PerfConfig,
    *,
    tensors: dict[str, torch.Tensor],
    backend: str,
) -> Callable[[], None]:
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    C = tensors["C"]
    initial_states = tensors["initial_states"]
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    inc_ref, m_ref = ref_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=cfg.T,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
    )
    starts_ref, _ = ref_state_passing(
        inc_ref,
        m_ref,
        initial_states=initial_states.to(dtype=torch.float32),
        compute_dtype=torch.float32,
    )
    dY = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P), device=U.device, dtype=torch.float32
    )

    if backend == "reference":

        def fn() -> None:
            U_ref = _clone_requires_grad(U)
            M_ref = _clone_requires_grad(M)
            K_ref = _clone_requires_grad(K)
            B_ref = _clone_requires_grad(B)
            C_ref = _clone_requires_grad(C)
            starts_ref_req = _clone_requires_grad(starts_ref)
            B_prev_ref = _clone_requires_grad(B_prev)
            U_prev_ref = _clone_requires_grad(U_prev)
            Y_ref = ref_chunk_scan(
                U_ref,
                M_ref,
                K_ref,
                B_ref,
                C_ref,
                starts_ref_req,
                B_prev=B_prev_ref,
                U_prev=U_prev_ref,
                T=cfg.T,
                chunk_size=cfg.chunk_size,
                output_dtype=torch.float32,
                compute_dtype=torch.float32,
            )
            loss = (Y_ref * dY).sum()
            torch.autograd.grad(
                loss,
                (
                    U_ref,
                    M_ref,
                    K_ref,
                    B_ref,
                    C_ref,
                    starts_ref_req,
                    B_prev_ref,
                    U_prev_ref,
                ),
            )

        fn()
        return fn

    inc_cute, m_cute = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )
    starts_cute, _ = state_passing_cute(
        inc_cute,
        m_cute,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )
    compiled = compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        starts_cute,
        dY,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
        return_launchers=True,
    )
    launch_overlapped = compiled[-1]

    def fn() -> None:
        launch_overlapped()

    fn()
    return fn


def _build_full_backward_callable(
    cfg: PerfConfig,
    *,
    tensors: dict[str, torch.Tensor],
    backend: str,
) -> Callable[[], None]:
    dY = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    d_final = torch.randn(
        (cfg.batch, cfg.heads, cfg.P, cfg.D),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    dB_last = torch.randn(
        (cfg.batch, cfg.heads, cfg.D), device=cfg.torch_device, dtype=torch.float32
    )
    dU_last = torch.randn(
        (cfg.batch, cfg.heads, cfg.P), device=cfg.torch_device, dtype=torch.float32
    )

    op = v2x2ssd if backend == "reference" else v2x2ssd_cute

    def fn() -> None:
        U = _clone_requires_grad(tensors["U"])
        M = _clone_requires_grad(tensors["M"])
        K = _clone_requires_grad(tensors["K"])
        B = _clone_requires_grad(tensors["B"])
        C = _clone_requires_grad(tensors["C"])
        initial_states = _clone_requires_grad(tensors["initial_states"])
        B_prev = _clone_requires_grad(tensors["B_prev"])
        U_prev = _clone_requires_grad(tensors["U_prev"])
        Y, final_state, B_last, U_last = op(
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
