"""Instrumented nextchar model used only by perf harnesses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.layers import SLinOSSMixer
from slinoss.perf import call_region


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
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, *, mult: int = 4) -> None:
        super().__init__()
        hidden = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


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
        norm1 = call_region("norms.pre_mixer", self.norm1, x, capture_backward=False)
        x = call_region("residual.mixer", torch.add, x, self.mixer(norm1))
        norm2 = call_region("norms.pre_ffn", self.norm2, x, capture_backward=False)
        x = call_region(
            "residual.ffn",
            torch.add,
            x,
            call_region("ffn", self.ff, norm2, capture_backward=False),
        )
        return x


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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (batch, T), got {tuple(idx.shape)}.")
        if idx.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {idx.shape[1]} exceeds block_size {self.block_size}."
            )
        x = call_region(
            "embed.token",
            self.token_embed,
            idx,
            capture_backward=False,
        )
        x = call_region(
            "embed.pos",
            lambda x_: x_ + self.pos_embed[:, : idx.shape[1], :],
            x,
        )
        for block in self.blocks:
            x = block(x)
        x = call_region("norms.final", self.norm_f, x, capture_backward=False)
        return call_region("head.logits", self.lm_head, x, capture_backward=False)
