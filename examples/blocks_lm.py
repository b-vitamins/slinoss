#!/usr/bin/env python3
"""Minimal causal LM composed from SLinOSS blocks and stacks."""

from __future__ import annotations

import torch
from torch import nn

from slinoss.blocks import (
    SLinOSSBlockConfig,
    SLinOSSMixerConfig,
    SLinOSSStack,
    SLinOSSStackConfig,
)
from slinoss.layers import SLinOSSMLPConfig


class ExampleBlockLM(nn.Module):
    """Small example showing how to compose embeddings, a stack, and a head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        stack_config: SLinOSSStackConfig,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(stack_config.d_model)
        self.token_embed = nn.Embedding(vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, self.seq_len, self.d_model))
        self.backbone = SLinOSSStack(stack_config)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected input_ids shape (batch, T), got {tuple(input_ids.shape)}."
            )
        if input_ids.shape[1] > self.seq_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds seq_len {self.seq_len}."
            )
        hidden = (
            self.token_embed(input_ids) + self.pos_embed[:, : input_ids.shape[1], :]
        )
        hidden = self.backbone(hidden)
        return self.lm_head(hidden)


def build_demo_model() -> ExampleBlockLM:
    block = SLinOSSBlockConfig(
        d_model=256,
        mixer=SLinOSSMixerConfig(
            d_state=64,
            expand=2.0,
            d_head=64,
            d_conv=4,
            chunk_size=32,
        ),
        ffn=SLinOSSMLPConfig(kind="swiglu", expand=8.0 / 3.0, multiple_of=128),
        norm_kind="rmsnorm",
        residual_in_fp32=True,
    )
    stack = SLinOSSStackConfig.uniform(block, n_layers=6, final_norm_kind="rmsnorm")
    return ExampleBlockLM(vocab_size=4096, seq_len=1024, stack_config=stack)


def main() -> int:
    model = build_demo_model()
    x = torch.randint(0, 4096, (2, 16))
    logits = model(x)
    print(f"logits shape: {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
