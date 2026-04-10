from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
import statistics
import sys
from typing import Literal, cast

import torch

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _nextchar import NextCharPerfConfig, build_model  # noqa: E402
from _nextchar_model import NextCharLM  # noqa: E402
from slinoss.models import NextCharBlock, NextCharDecodeState  # noqa: E402

DecodeMode = Literal["persistent", "eager"]


@dataclass(frozen=True)
class PeakSpec:
    name: str
    bandwidth_bytes_per_s: float
    tc_flops_per_s: float
    simt_flops_per_s: float
    launch_floor_us: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LowerBoundEstimate:
    bytes_hbm: float
    flops_tc: float
    flops_simt: float
    launches: int
    bw_us: float
    tc_us: float
    simt_us: float
    launch_us: float
    t_lower_us: float
    dominant: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_PEAK_PRESETS: dict[str, PeakSpec] = {
    "h100_sxm5_fp16": PeakSpec(
        name="NVIDIA H100 SXM5 FP16",
        bandwidth_bytes_per_s=3.35e12,
        tc_flops_per_s=1.979e15,
        simt_flops_per_s=6.7e13,
        launch_floor_us=1.0,
        source="NVIDIA H100 Tensor Core GPU product page",
    ),
    "rtx3060_fp16": PeakSpec(
        name="NVIDIA GeForce RTX 3060 FP16",
        bandwidth_bytes_per_s=3.60e11,
        tc_flops_per_s=1.02e14,
        simt_flops_per_s=1.27e13,
        launch_floor_us=2.0,
        source="device-derived bandwidth + GA106 peak-throughput preset",
    ),
}


def _auto_peak_spec(device: torch.device) -> PeakSpec | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    if "H100" in name:
        return _PEAK_PRESETS["h100_sxm5_fp16"]
    if "RTX 3060" in name:
        return _PEAK_PRESETS["rtx3060_fp16"]
    props = torch.cuda.get_device_properties(device)
    bandwidth = (
        2.0
        * float(props.memory_clock_rate)
        * 1.0e3
        * (float(props.memory_bus_width) / 8.0)
    )
    return PeakSpec(
        name=f"{name} (auto)",
        bandwidth_bytes_per_s=bandwidth,
        tc_flops_per_s=0.0,
        simt_flops_per_s=0.0,
        launch_floor_us=2.0,
        source="device-derived bandwidth only; compute peaks unspecified",
    )


def resolve_peak_spec(
    *,
    device: torch.device,
    preset: str,
    peak_bw_gbps: float | None,
    peak_tc_tflops: float | None,
    peak_simt_tflops: float | None,
    launch_floor_us: float | None,
) -> PeakSpec | None:
    if preset == "none":
        return None
    if preset == "auto":
        base = _auto_peak_spec(device)
        if base is None:
            return None
    else:
        try:
            base = _PEAK_PRESETS[preset]
        except KeyError as exc:
            raise ValueError(
                f"Unknown peak preset {preset!r}. Available: {sorted(_PEAK_PRESETS)}"
            ) from exc
    return PeakSpec(
        name=base.name,
        bandwidth_bytes_per_s=(
            base.bandwidth_bytes_per_s
            if peak_bw_gbps is None
            else float(peak_bw_gbps) * 1.0e9
        ),
        tc_flops_per_s=(
            base.tc_flops_per_s
            if peak_tc_tflops is None
            else float(peak_tc_tflops) * 1.0e12
        ),
        simt_flops_per_s=(
            base.simt_flops_per_s
            if peak_simt_tflops is None
            else float(peak_simt_tflops) * 1.0e12
        ),
        launch_floor_us=(
            base.launch_floor_us if launch_floor_us is None else float(launch_floor_us)
        ),
        source=base.source,
    )


def build_decode_model(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
) -> NextCharLM:
    model, optimizer = build_model(cfg, backend=backend, instrumented=False)
    del optimizer
    return cast(NextCharLM, model.eval())


def random_decode_tokens(
    cfg: NextCharPerfConfig,
    *,
    total_tokens: int,
) -> torch.Tensor:
    return torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, total_tokens),
        device=cfg.torch_device,
        dtype=torch.long,
    )


def _decode_one_mode(
    model: NextCharLM,
    token: torch.Tensor,
    state: NextCharDecodeState,
    *,
    mode: DecodeMode,
) -> tuple[torch.Tensor, NextCharDecodeState]:
    if mode == "persistent":
        return model.decode_one(token, state)
    if mode == "eager":
        return model._decode_one_eager_inplace(token, state)
    raise ValueError(f"Unsupported decode mode: {mode}")


def _time_decode_tokens(
    model: NextCharLM,
    tokens: torch.Tensor,
    *,
    mode: DecodeMode,
    warmup_tokens: int,
) -> tuple[float, torch.Tensor]:
    batch, total_tokens = map(int, tokens.shape)
    state = model.init_decode_state(
        batch,
        device=tokens.device,
        dtype=model.token_embed.weight.dtype,
    )
    logits = tokens.new_empty((batch, 0), dtype=model.token_embed.weight.dtype)
    with torch.no_grad():
        for t in range(warmup_tokens):
            logits, state = _decode_one_mode(model, tokens[:, t], state, mode=mode)
        if tokens.device.type == "cuda":
            stream = torch.cuda.current_stream(device=tokens.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for t in range(warmup_tokens, total_tokens):
                logits, state = _decode_one_mode(model, tokens[:, t], state, mode=mode)
            end.record(stream)
            torch.cuda.synchronize(tokens.device)
            return float(start.elapsed_time(end)), logits

        started = __import__("time").perf_counter()
        for t in range(warmup_tokens, total_tokens):
            logits, state = _decode_one_mode(model, tokens[:, t], state, mode=mode)
        ended = __import__("time").perf_counter()
        return (ended - started) * 1000.0, logits


def benchmark_decode_mode(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    mode: DecodeMode,
    warmup_tokens: int,
    active_tokens: int,
    repeat: int,
) -> dict[str, object]:
    total_tokens = int(warmup_tokens) + int(active_tokens)
    if total_tokens <= 0:
        raise ValueError("warmup_tokens + active_tokens must be positive.")
    if total_tokens > cfg.block_size:
        raise ValueError(
            f"Need block_size >= {total_tokens} for decode benchmark. Got {cfg.block_size}."
        )
    model = build_decode_model(cfg, backend=backend)
    samples_us_per_token: list[float] = []
    logits_ref: torch.Tensor | None = None
    for rep in range(repeat):
        torch.manual_seed(int(cfg.seed) + rep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed) + rep)
        tokens = random_decode_tokens(cfg, total_tokens=total_tokens)
        elapsed_ms, logits = _time_decode_tokens(
            model,
            tokens,
            mode=mode,
            warmup_tokens=warmup_tokens,
        )
        logits_ref = logits
        samples_us_per_token.append(
            (elapsed_ms * 1000.0) / max(1, cfg.batch_size * active_tokens)
        )
    del model
    gc.collect()
    if cfg.torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "backend": backend,
        "mode": mode,
        "config": cfg.perf_config_dict,
        "warmup_tokens": int(warmup_tokens),
        "active_tokens": int(active_tokens),
        "repeat": int(repeat),
        "summary": {
            "num_samples": len(samples_us_per_token),
            "mean_us_per_token": statistics.fmean(samples_us_per_token),
            "median_us_per_token": statistics.median(samples_us_per_token),
            "min_us_per_token": min(samples_us_per_token),
            "max_us_per_token": max(samples_us_per_token),
            "stdev_us_per_token": (
                statistics.stdev(samples_us_per_token)
                if len(samples_us_per_token) > 1
                else 0.0
            ),
        },
        "last_logits_shape": None if logits_ref is None else list(logits_ref.shape),
    }


def estimate_lower_bound(
    model: NextCharLM,
    *,
    batch_size: int,
    peak: PeakSpec,
) -> LowerBoundEstimate:
    dtype_bytes = int(model.token_embed.weight.element_size())
    d_model = int(model.token_embed.embedding_dim)
    vocab_size = int(model.token_embed.num_embeddings)
    blocks = cast(list[NextCharBlock], list(model.blocks))
    n_layers = len(blocks)
    if not blocks:
        raise ValueError("Model must have at least one block.")
    mixer = blocks[0].mixer
    d_inner = int(mixer.d_inner)
    d_head = int(mixer.d_head)
    d_state = int(mixer.d_state)
    n_heads = int(mixer.n_heads)
    d_conv = int(mixer.d_conv)
    hidden = int(blocks[0].ff.fc1.out_features)

    bytes_hbm = 0
    bytes_hbm += batch_size * d_model * dtype_bytes  # token embedding rows
    bytes_hbm += d_model * dtype_bytes  # single position row
    state_bytes = (
        batch_size
        * n_layers
        * (
            d_inner * max(d_conv - 1, 0) * dtype_bytes
            + n_heads * d_head * (2 * d_state) * dtype_bytes
            + n_heads * (2 * d_state) * dtype_bytes
            + n_heads * d_head * dtype_bytes
        )
    )
    bytes_hbm += 2 * state_bytes
    bytes_hbm += batch_size * vocab_size * dtype_bytes  # logits write

    flops_tc = 0.0
    flops_tc += 2.0 * batch_size * d_model * vocab_size
    for block in blocks:
        flops_tc += 2.0 * batch_size * d_model * block.mixer.in_proj.out_features
        flops_tc += 2.0 * batch_size * d_inner * d_model
        flops_tc += 2.0 * batch_size * d_model * hidden
        flops_tc += 2.0 * batch_size * hidden * d_model

    # Lower-bound proxy for elementwise / recurrent work. This intentionally
    # undercounts rather than overclaims throughput.
    flops_simt = 0.0
    flops_simt += 6.0 * batch_size * d_model  # final norm
    for _block in blocks:
        flops_simt += 6.0 * batch_size * d_model  # norm1
        flops_simt += 6.0 * batch_size * d_model  # norm2
        flops_simt += batch_size * d_inner * (2.0 * d_conv + 6.0)  # dw conv + silu
        flops_simt += batch_size * n_heads * (40.0 * d_state)  # scanprep scalar work
        flops_simt += (
            batch_size * n_heads * d_head * (24.0 * d_state + 8.0)
        )  # recurrent update + gate
        flops_simt += 4.0 * batch_size * d_model  # residual adds
        flops_simt += 8.0 * batch_size * hidden  # GELU

    # The persistent decode path issues one host-visible graph replay per token.
    launches = 1
    bw_us = (bytes_hbm / peak.bandwidth_bytes_per_s) * 1.0e6
    tc_us = (
        0.0 if peak.tc_flops_per_s <= 0.0 else (flops_tc / peak.tc_flops_per_s) * 1.0e6
    )
    simt_us = (
        0.0
        if peak.simt_flops_per_s <= 0.0
        else (flops_simt / peak.simt_flops_per_s) * 1.0e6
    )
    launch_us = float(launches) * float(peak.launch_floor_us)
    terms: dict[str, float] = {
        "bytes_hbm": bw_us,
        "flops_tc": tc_us,
        "flops_simt": simt_us,
        "launches": launch_us,
    }
    dominant = max(terms, key=lambda label: terms[label])
    t_lower_us = terms[dominant]
    return LowerBoundEstimate(
        bytes_hbm=float(bytes_hbm),
        flops_tc=float(flops_tc),
        flops_simt=float(flops_simt),
        launches=int(launches),
        bw_us=float(bw_us),
        tc_us=float(tc_us),
        simt_us=float(simt_us),
        launch_us=float(launch_us),
        t_lower_us=float(t_lower_us),
        dominant=str(dominant),
    )


def profile_decode_trace(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    mode: DecodeMode,
    warmup_tokens: int,
    active_tokens: int,
    trace_out: str | None,
    sort_by: str,
    top_k: int,
) -> dict[str, object]:
    total_tokens = int(warmup_tokens) + int(active_tokens)
    if total_tokens > cfg.block_size:
        raise ValueError(
            f"Need block_size >= {total_tokens} for decode profile. Got {cfg.block_size}."
        )
    model = build_decode_model(cfg, backend=backend)
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    tokens = random_decode_tokens(cfg, total_tokens=total_tokens)
    state = model.init_decode_state(
        cfg.batch_size,
        device=cfg.torch_device,
        dtype=model.token_embed.weight.dtype,
    )
    with torch.no_grad():
        for t in range(warmup_tokens):
            _, state = _decode_one_mode(model, tokens[:, t], state, mode=mode)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cfg.torch_device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for t in range(warmup_tokens, total_tokens):
                with torch.autograd.profiler.record_function("decode.one"):
                    _, state = _decode_one_mode(model, tokens[:, t], state, mode=mode)
        if trace_out is not None:
            prof.export_chrome_trace(trace_out)
        table = prof.key_averages().table(sort_by=sort_by, row_limit=top_k)
    return {
        "backend": backend,
        "mode": mode,
        "config": cfg.perf_config_dict,
        "warmup_tokens": int(warmup_tokens),
        "active_tokens": int(active_tokens),
        "table": table,
        "trace_out": trace_out,
    }


def summary_speedup(
    faster: dict[str, object],
    slower: dict[str, object],
) -> float:
    faster_summary = cast(dict[str, float], faster["summary"])
    slower_summary = cast(dict[str, float], slower["summary"])
    fast_us = float(faster_summary["mean_us_per_token"])
    slow_us = float(slower_summary["mean_us_per_token"])
    return math.inf if fast_us <= 0.0 else slow_us / fast_us
