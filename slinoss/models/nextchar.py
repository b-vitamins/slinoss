"""Shared nextchar model with decode support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.layers import AutoMixerDecodeBackend, CuteMixerDecodeBackend, SLinOSSMixer
from slinoss.layers.state import SLinOSSMixerState
from slinoss.ops.decode_linear import decode_linear


def configure_optim(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "bias" not in name and "norm" not in name.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = any(p.is_cuda for p in decay) or any(p.is_cuda for p in no_decay)
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(0.9, 0.95),
        fused=use_fused,
    )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, *, mult: int = 4) -> None:
        super().__init__()
        hidden = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    def decode_one(self, x: torch.Tensor) -> torch.Tensor:
        hidden = decode_linear(x, self.fc1)
        return decode_linear(F.gelu(hidden, approximate="tanh"), self.fc2)


@dataclass
class NextCharDecodeState:
    """Persistent decode state for ``NextCharLM``."""

    layers: list[SLinOSSMixerState]
    position: int = 0
    position_buffer: torch.Tensor | None = None
    _engine: object | None = field(default=None, repr=False, compare=False)

    def clone(self) -> "NextCharDecodeState":
        return NextCharDecodeState(
            layers=[layer.clone() for layer in self.layers],
            position=int(self.position),
            position_buffer=(
                None if self.position_buffer is None else self.position_buffer.clone()
            ),
        )

    def detach(self) -> "NextCharDecodeState":
        return NextCharDecodeState(
            layers=[layer.detach() for layer in self.layers],
            position=int(self.position),
            position_buffer=(
                None if self.position_buffer is None else self.position_buffer.detach()
            ),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NextCharDecodeState":
        return NextCharDecodeState(
            layers=[layer.to(device=device, dtype=dtype) for layer in self.layers],
            position=int(self.position),
            position_buffer=(
                None
                if self.position_buffer is None
                else self.position_buffer.to(device=device)
            ),
        )


class NextCharBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = SLinOSSMixer(
            d_model,
            d_state=d_state,
            expand=expand,
            d_head=d_head,
            d_conv=d_conv,
            chunk_size=chunk_size,
            normalize_bc=True,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm1 = self.norm1(x)
        x = x + self.mixer(norm1)
        norm2 = self.norm2(x)
        x = x + self.ff(norm2)
        return x

    def init_decode_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSMixerState:
        return self.mixer.init_decode_state(batch_size, device=device, dtype=dtype)

    def decode_one_inplace(
        self,
        x: torch.Tensor,
        state: SLinOSSMixerState,
    ) -> torch.Tensor:
        norm1 = self.norm1(x)
        x = x + self.mixer._step_inplace(norm1, state)
        norm2 = self.norm2(x)
        x = x + self.ff.decode_one(norm2)
        return x


def _copy_mixer_state_(dst: SLinOSSMixerState, src: SLinOSSMixerState) -> None:
    if dst.conv is not None and src.conv is not None:
        dst.conv.copy_(src.conv)
    if dst.scan.state is not None and src.scan.state is not None:
        dst.scan.state.copy_(src.scan.state)
    if dst.scan.b_prev is not None and src.scan.b_prev is not None:
        dst.scan.b_prev.copy_(src.scan.b_prev)
    if dst.scan.u_prev is not None and src.scan.u_prev is not None:
        dst.scan.u_prev.copy_(src.scan.u_prev)


def _restore_decode_state_(dst: NextCharDecodeState, src: NextCharDecodeState) -> None:
    for dst_layer, src_layer in zip(dst.layers, src.layers, strict=True):
        _copy_mixer_state_(dst_layer, src_layer)
    dst.position = int(src.position)
    if dst.position_buffer is not None and src.position_buffer is not None:
        dst.position_buffer.copy_(src.position_buffer)


class _NextCharCudaGraphDecodeEngine:
    """Fixed-shape CUDA graph replay for one-token decode."""

    def __init__(
        self,
        model: "NextCharLM",
        state: NextCharDecodeState,
        *,
        batch_size: int,
    ) -> None:
        self.model = model
        self._state_ref = weakref.ref(state)
        self.batch_size = int(batch_size)
        self.device = model.token_embed.weight.device
        self.idx_buffer = torch.zeros(
            (self.batch_size,), device=self.device, dtype=torch.long
        )
        self.graph = torch.cuda.CUDAGraph()
        self.static_logits: torch.Tensor | None = None
        self._capture(state)

    @staticmethod
    def supported(model: "NextCharLM", *, batch_size: int) -> bool:
        if model.token_embed.weight.device.type != "cuda":
            return False
        if model.token_embed.weight.dtype not in (torch.float16, torch.bfloat16):
            return False
        if batch_size not in (1, 2, 4, 8, 16):
            return False
        for block in cast(list[NextCharBlock], list(model.blocks)):
            if not isinstance(
                block.mixer.decode_backend,
                (AutoMixerDecodeBackend, CuteMixerDecodeBackend),
            ):
                return False
            if not block.mixer._supports_cute_decode(
                batch_size=batch_size,
                device=model.token_embed.weight.device,
                dtype=model.token_embed.weight.dtype,
            ):
                return False
        return True

    def _get_captured_state(self) -> NextCharDecodeState:
        state = self._state_ref()
        if state is None:
            raise RuntimeError("Decode graph state has been released.")
        return state

    def _run_body(self, state: NextCharDecodeState) -> torch.Tensor:
        if state.position_buffer is None:
            raise RuntimeError("Decode state is missing a position buffer.")
        x = self.model.token_embed(self.idx_buffer)
        pos = self.model.pos_embed[0].index_select(0, state.position_buffer)
        x = x + pos.expand(self.batch_size, -1)
        for block, layer_state in zip(
            cast(list[NextCharBlock], list(self.model.blocks)),
            state.layers,
            strict=True,
        ):
            x = block.decode_one_inplace(x, layer_state)
        x = self.model.norm_f(x)
        return decode_linear(x, self.model.lm_head)

    def _capture(self, state: NextCharDecodeState) -> None:
        snapshot = state.clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                _restore_decode_state_(state, snapshot)
                self.static_logits = self._run_body(state)
        _restore_decode_state_(state, snapshot)
        torch.cuda.current_stream(device=self.device).wait_stream(stream)
        with torch.cuda.graph(self.graph):
            self.static_logits = self._run_body(state)
        _restore_decode_state_(state, snapshot)

    def decode_one(
        self,
        idx: torch.Tensor,
        state: NextCharDecodeState,
    ) -> tuple[torch.Tensor, NextCharDecodeState]:
        if self._get_captured_state() is not state:
            raise RuntimeError("Decode graph engine is bound to a different state.")
        if state.position_buffer is None:
            raise RuntimeError("Decode state is missing a position buffer.")
        idx_c = idx.to(device=self.device, dtype=torch.long, non_blocking=True)
        self.idx_buffer.copy_(idx_c)
        state.position_buffer.fill_(int(state.position))
        self.graph.replay()
        state.position += 1
        if self.static_logits is None:
            raise RuntimeError("Decode graph did not materialize logits.")
        return self.static_logits.clone(), state


class NextCharLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.block_size = int(block_size)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, self.block_size, d_model))
        self.blocks = nn.ModuleList(
            [
                NextCharBlock(
                    d_model,
                    d_state=d_state,
                    expand=expand,
                    d_head=d_head,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.perf_trainable_params: tuple[torch.nn.Parameter, ...] = ()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _add_pos_embed(self, x: torch.Tensor, T: int) -> torch.Tensor:
        return x + self.pos_embed[:, :T, :]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (batch, T), got {tuple(idx.shape)}.")
        if idx.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {idx.shape[1]} exceeds block_size {self.block_size}."
            )
        x = self.token_embed(idx)
        x = self._add_pos_embed(x, int(idx.shape[1]))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def init_decode_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> NextCharDecodeState:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive. Got {batch_size}.")
        if device is None:
            device = self.token_embed.weight.device
        if dtype is None:
            dtype = self.token_embed.weight.dtype
        return NextCharDecodeState(
            layers=[
                block.init_decode_state(batch_size, device=device, dtype=dtype)
                for block in cast(list[NextCharBlock], list(self.blocks))
            ],
            position=0,
            position_buffer=torch.zeros((1,), device=device, dtype=torch.long),
        )

    def _decode_one_eager_inplace(
        self,
        idx: torch.Tensor,
        state: NextCharDecodeState,
    ) -> tuple[torch.Tensor, NextCharDecodeState]:
        if state.position >= self.block_size:
            raise ValueError(
                f"decode position {state.position} exceeds block_size {self.block_size}."
            )
        x = self.token_embed(idx)
        pos = self.pos_embed[:, state.position : state.position + 1, :][:, 0, :]
        x = x + pos
        for block, layer_state in zip(
            cast(list[NextCharBlock], list(self.blocks)),
            state.layers,
            strict=True,
        ):
            x = block.decode_one_inplace(x, layer_state)
        x = self.norm_f(x)
        logits = decode_linear(x, self.lm_head)
        state.position += 1
        if state.position_buffer is not None:
            state.position_buffer.fill_(int(state.position))
        return logits, state

    @torch.no_grad()
    def decode_one(
        self,
        idx: torch.Tensor,
        state: NextCharDecodeState | None = None,
    ) -> tuple[torch.Tensor, NextCharDecodeState]:
        if idx.ndim == 2:
            if idx.shape[1] != 1:
                raise ValueError(
                    "decode_one expects (batch,) or (batch, 1) token ids. "
                    f"Got {tuple(idx.shape)}."
                )
            idx = idx[:, 0]
        elif idx.ndim != 1:
            raise ValueError(
                f"decode_one expects (batch,) or (batch, 1); got {tuple(idx.shape)}."
            )

        if state is None:
            state = self.init_decode_state(
                int(idx.shape[0]),
                device=self.token_embed.weight.device,
                dtype=self.token_embed.weight.dtype,
            )
        if state.position >= self.block_size:
            raise ValueError(
                f"decode position {state.position} exceeds block_size {self.block_size}."
            )

        if _NextCharCudaGraphDecodeEngine.supported(self, batch_size=int(idx.shape[0])):
            engine = state._engine
            if not isinstance(engine, _NextCharCudaGraphDecodeEngine):
                engine = _NextCharCudaGraphDecodeEngine(
                    self,
                    state,
                    batch_size=int(idx.shape[0]),
                )
                state._engine = engine
            return engine.decode_one(idx, state)

        return self._decode_one_eager_inplace(idx, state)


__all__ = [
    "FeedForward",
    "NextCharBlock",
    "NextCharDecodeState",
    "NextCharLM",
    "configure_optim",
]
