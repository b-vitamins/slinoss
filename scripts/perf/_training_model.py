"""Thin LM shells built compositionally from SLinOSS blocks and stacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast
import weakref

import torch
from torch import nn

from slinoss.blocks import (
    SLinOSSBlock,
    SLinOSSStack,
    SLinOSSStackConfig,
    SLinOSSStackState,
)
from slinoss.ops.decode import decode_linear


def configure_optim(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "bias" not in name and "norm" not in name.lower():
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = (
        any(param.is_cuda for param in decay)
        or any(param.is_cuda for param in no_decay)
    ) and not any(torch.is_complex(param) for param in model.parameters())
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(0.9, 0.95),
        fused=use_fused,
    )


@dataclass
class TrainingDecodeState:
    """Persistent decode state for the perf-harness LM shell."""

    backbone: SLinOSSStackState
    position: int = 0
    position_buffer: torch.Tensor | None = None
    _engine: object | None = field(default=None, repr=False, compare=False)

    def copy_(self, other: "TrainingDecodeState") -> "TrainingDecodeState":
        self.backbone.copy_(other.backbone)
        self.position = int(other.position)
        if self.position_buffer is not None and other.position_buffer is not None:
            self.position_buffer.copy_(other.position_buffer)
        return self

    def clone(self) -> "TrainingDecodeState":
        return TrainingDecodeState(
            backbone=self.backbone.clone(),
            position=int(self.position),
            position_buffer=(
                None if self.position_buffer is None else self.position_buffer.clone()
            ),
        )

    def detach(self) -> "TrainingDecodeState":
        return TrainingDecodeState(
            backbone=self.backbone.detach(),
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
    ) -> "TrainingDecodeState":
        return TrainingDecodeState(
            backbone=self.backbone.to(device=device, dtype=dtype),
            position=int(self.position),
            position_buffer=(
                None
                if self.position_buffer is None
                else self.position_buffer.to(device=device)
            ),
        )


def _restore_decode_state_(
    dst: TrainingDecodeState,
    src: TrainingDecodeState,
) -> None:
    dst.copy_(src)


class _TrainingCudaGraphDecodeEngine:
    """Fixed-shape CUDA graph replay for one-token decode."""

    def __init__(
        self,
        model: "TrainingLM",
        state: TrainingDecodeState,
        *,
        batch_size: int,
    ) -> None:
        self.model = model
        self._state_ref = weakref.ref(state)
        self.batch_size = int(batch_size)
        self.device = model.token_embed.weight.device
        self.dtype = model.token_embed.weight.dtype
        self.idx_buffer = torch.zeros(
            (self.batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.static_logits: torch.Tensor | None = None
        self._capture(state)

    @staticmethod
    def supported(model: "TrainingLM", *, batch_size: int) -> bool:
        if model.token_embed.weight.device.type != "cuda":
            return False
        if model.token_embed.weight.dtype not in (torch.float16, torch.bfloat16):
            return False
        if batch_size not in (1, 2, 4, 8, 16):
            return False
        for block in cast(list[SLinOSSBlock], list(model.backbone.blocks)):
            if not block.mixer._supports_fast_decode(
                batch_size=batch_size,
                device=model.token_embed.weight.device,
                dtype=model.token_embed.weight.dtype,
            ):
                return False
        return True

    def _get_captured_state(self) -> TrainingDecodeState:
        state = self._state_ref()
        if state is None:
            raise RuntimeError("Decode graph state has been released.")
        return state

    def _run_body(self, state: TrainingDecodeState) -> torch.Tensor:
        if state.position_buffer is None:
            raise RuntimeError("Decode state is missing a position buffer.")
        return self.model._decode_token_inplace(
            self.idx_buffer,
            state,
            position_buffer=state.position_buffer,
        )

    def _capture(self, state: TrainingDecodeState) -> None:
        snapshot = state.clone()
        stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(device=self.device)
        stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _restore_decode_state_(state, snapshot)
                    self.static_logits = self._run_body(state)
            _restore_decode_state_(state, snapshot)
            current_stream.wait_stream(stream)
            with torch.cuda.graph(self.graph):
                self.static_logits = self._run_body(state)
        finally:
            current_stream.wait_stream(stream)
            _restore_decode_state_(state, snapshot)

    def decode_one(
        self,
        idx: torch.Tensor,
        state: TrainingDecodeState,
    ) -> tuple[torch.Tensor, TrainingDecodeState]:
        if self._get_captured_state() is not state:
            raise RuntimeError("Decode graph engine is bound to a different state.")
        if state.position_buffer is None:
            raise RuntimeError("Decode state is missing a position buffer.")
        idx_c = idx.to(device=self.device, dtype=torch.long, non_blocking=True)
        state.position_buffer.fill_(int(state.position))
        self.idx_buffer.copy_(idx_c)
        self.graph.replay()
        state.position += 1
        state.position_buffer.fill_(int(state.position))
        if self.static_logits is None:
            raise RuntimeError("Decode graph did not materialize logits.")
        return self.static_logits.clone(), state


class TrainingLM(nn.Module):
    """A minimal LM shell composed from token embeddings, a stack, and a head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        stack_config: SLinOSSStackConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(stack_config.d_model)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.token_embed = nn.Embedding(vocab_size, self.d_model, **factory_kwargs)
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.seq_len, self.d_model, **factory_kwargs)
        )
        self.backbone = SLinOSSStack(stack_config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False, **factory_kwargs)
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

    def _add_pos_embed(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x + self.pos_embed[:, :seq_len, :]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (batch, T), got {tuple(idx.shape)}.")
        if idx.shape[1] > self.seq_len:
            raise ValueError(
                f"Sequence length {idx.shape[1]} exceeds seq_len {self.seq_len}."
            )
        hidden = self.token_embed(idx)
        hidden = self._add_pos_embed(hidden, int(idx.shape[1]))
        hidden = cast(torch.Tensor, self.backbone(hidden))
        return self.lm_head(hidden)

    def init_decode_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TrainingDecodeState:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive. Got {batch_size}.")
        if device is None:
            device = self.token_embed.weight.device
        if dtype is None:
            dtype = self.token_embed.weight.dtype
        return TrainingDecodeState(
            backbone=self.backbone.init_decode_state(
                batch_size,
                device=device,
                dtype=dtype,
            ),
            position=0,
            position_buffer=torch.zeros((1,), device=device, dtype=torch.long),
        )

    def _decode_token_inplace(
        self,
        idx: torch.Tensor,
        state: TrainingDecodeState,
        *,
        position_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(idx.shape[0])
        hidden = self.token_embed(idx)
        if position_buffer is None:
            pos = self.pos_embed[:, state.position : state.position + 1, :][:, 0, :]
            pos = pos.expand(batch_size, -1)
        else:
            pos = self.pos_embed[0].index_select(0, position_buffer)
        hidden = hidden + pos
        next_backbone_hidden, next_backbone = self.backbone.step(
            hidden,
            state.backbone,
            inplace=True,
        )
        if next_backbone is not state.backbone:
            state.backbone.copy_(next_backbone)
        return decode_linear(next_backbone_hidden, self.lm_head)

    def _decode_one_eager_inplace(
        self,
        idx: torch.Tensor,
        state: TrainingDecodeState,
    ) -> tuple[torch.Tensor, TrainingDecodeState]:
        if state.position >= self.seq_len:
            raise ValueError(
                f"decode position {state.position} exceeds seq_len {self.seq_len}."
            )
        logits = self._decode_token_inplace(idx, state)
        state.position += 1
        if state.position_buffer is not None:
            state.position_buffer.fill_(int(state.position))
        return logits, state

    @torch.no_grad()
    def decode_one(
        self,
        idx: torch.Tensor,
        state: TrainingDecodeState | None = None,
    ) -> tuple[torch.Tensor, TrainingDecodeState]:
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
        if state.position >= self.seq_len:
            raise ValueError(
                f"decode position {state.position} exceeds seq_len {self.seq_len}."
            )

        if _TrainingCudaGraphDecodeEngine.supported(self, batch_size=int(idx.shape[0])):
            engine = state._engine
            if not isinstance(engine, _TrainingCudaGraphDecodeEngine):
                engine = _TrainingCudaGraphDecodeEngine(
                    self,
                    state,
                    batch_size=int(idx.shape[0]),
                )
                state._engine = engine
            return engine.decode_one(idx, state)

        return self._decode_one_eager_inplace(idx, state)


__all__ = [
    "TrainingDecodeState",
    "TrainingLM",
    "configure_optim",
]
