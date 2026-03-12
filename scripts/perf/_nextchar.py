from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import warnings

import torch
from torch.nn import functional as F

from _nextchar_model import NextCharLM, configure_optim
from slinoss.layers import SLinOSSMixer
from slinoss.layers.backend import CuteScanBackend, ReferenceScanBackend
from slinoss.perf import PerfRecorder, attach_module_timer, call_region, record_region

warnings.filterwarnings(
    "ignore",
    message=(
        "Full backward hook is firing when gradients are computed with respect "
        "to module outputs since no inputs require gradients.*"
    ),
)


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


def build_model(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
) -> tuple[NextCharLM, torch.optim.Optimizer]:
    model = NextCharLM(
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
    optimizer = configure_optim(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    _configure_backend(model, backend=backend)
    return model, optimizer


def _configure_backend(model: NextCharLM, *, backend: str) -> None:
    if backend not in ("reference", "cute"):
        raise ValueError(f"Unsupported backend: {backend}")
    backend_obj = (
        ReferenceScanBackend(compute_dtype=torch.float32)
        if backend == "reference"
        else CuteScanBackend(compute_dtype=torch.float32)
    )
    for module in model.modules():
        if isinstance(module, SLinOSSMixer):
            module.backend = backend_obj


def attach_workload_timers(
    model: NextCharLM,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    handles.extend(
        attach_module_timer(
            model.token_embed,
            "embed.token",
            capture_forward=False,
            capture_backward=True,
        )
    )
    handles.extend(
        attach_module_timer(
            model.norm_f,
            "norms.final",
            capture_forward=False,
            capture_backward=True,
        )
    )
    handles.extend(
        attach_module_timer(
            model.lm_head,
            "head.logits",
            capture_forward=False,
            capture_backward=True,
        )
    )
    for block in model.blocks:
        handles.extend(
            attach_module_timer(
                block.norm1,
                "norms.pre_mixer",
                capture_forward=False,
                capture_backward=True,
            )
        )
        handles.extend(
            attach_module_timer(
                block.norm2,
                "norms.pre_ffn",
                capture_forward=False,
                capture_backward=True,
            )
        )
        handles.extend(
            attach_module_timer(
                block.ff,
                "ffn",
                capture_forward=False,
                capture_backward=True,
            )
        )
        handles.extend(
            attach_module_timer(
                block.mixer.in_proj,
                "mixer.in_proj",
                capture_forward=False,
                capture_backward=True,
            )
        )
        handles.extend(
            attach_module_timer(
                block.mixer.dw_conv,
                "mixer.dw_conv",
                capture_forward=False,
                capture_backward=True,
            )
        )
        handles.extend(
            attach_module_timer(
                block.mixer.bc_proj,
                "mixer.bc_proj",
                capture_forward=False,
                capture_backward=True,
            )
        )
        handles.extend(
            attach_module_timer(
                block.mixer.out_proj,
                "mixer.out_proj",
                capture_forward=False,
                capture_backward=True,
            )
        )
    return handles


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


def run_train_step(
    model: NextCharLM,
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
                lambda logits_, targets_: F.cross_entropy(
                    logits_.reshape(-1, logits_.shape[-1]), targets_.reshape(-1)
                ),
                logits,
                yb,
            )
        with record_region("step.backward"):
            loss.backward()
        with record_region("step.clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        with record_region("step.optim"):
            optimizer.step()
    return logits, loss


def run_bench_step(
    cfg: NextCharPerfConfig,
    *,
    backend: str,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    model, optimizer = build_model(cfg, backend=backend)
    handles = attach_workload_timers(model)
    try:
        cold_recorder = PerfRecorder(device=cfg.torch_device)
        xb, yb = random_batch(cfg)
        with cold_recorder.capture_step():
            logits, loss = run_train_step(
                model, optimizer, xb, yb, grad_clip=cfg.grad_clip
            )
        cold = cold_recorder.steps[-1]
        del logits, loss

        for _ in range(warmup):
            xb, yb = random_batch(cfg)
            recorder = PerfRecorder(device=cfg.torch_device)
            with recorder.capture_step():
                run_train_step(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)

        warm_steps: list[dict[str, Any]] = []
        for _ in range(steps):
            xb, yb = random_batch(cfg)
            recorder = PerfRecorder(device=cfg.torch_device)
            with recorder.capture_step():
                run_train_step(model, optimizer, xb, yb, grad_clip=cfg.grad_clip)
            warm_steps.append(recorder.steps[-1])
    finally:
        for handle in handles:
            handle.remove()
    return {
        "cold": cold,
        "warm_steps": warm_steps,
        "tokens_per_step": cfg.batch_size * cfg.block_size,
    }
