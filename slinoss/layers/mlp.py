"""Reusable channel-mixing layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.ops.decode import decode_linear

from ._validation import _require

MLPKind = Literal["swiglu", "gelu"]
SLinOSSMLPKind = MLPKind


def _resolve_hidden_dim(
    d_model: int,
    *,
    hidden_dim: int | None,
    expand: float,
    multiple_of: int,
) -> int:
    _require(d_model > 0, f"d_model must be positive. Got {d_model}.")
    _require(multiple_of > 0, f"multiple_of must be positive. Got {multiple_of}.")
    if hidden_dim is None:
        _require(expand > 0.0, f"expand must be positive. Got {expand}.")
        hidden_dim = int(d_model * expand)
    _require(hidden_dim > 0, f"hidden_dim must be positive. Got {hidden_dim}.")
    return ((int(hidden_dim) + multiple_of - 1) // multiple_of) * multiple_of


@dataclass(frozen=True, slots=True)
class SLinOSSMLPConfig:
    """Typed configuration for the canonical SLinOSS channel mixer."""

    kind: MLPKind = "swiglu"
    hidden_dim: int | None = None
    expand: float = 8.0 / 3.0
    multiple_of: int = 128
    bias: bool = False

    def resolve_hidden_dim(self, d_model: int) -> int:
        return _resolve_hidden_dim(
            d_model,
            hidden_dim=self.hidden_dim,
            expand=self.expand,
            multiple_of=self.multiple_of,
        )


class SLinOSSMLP(nn.Module):
    """Canonical channel-mixing layer used by SLinOSS blocks."""

    def __init__(
        self,
        d_model: int,
        *,
        config: SLinOSSMLPConfig | None = None,
        kind: MLPKind | None = None,
        hidden_dim: int | None = None,
        expand: float | None = None,
        multiple_of: int | None = None,
        bias: bool | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = SLinOSSMLPConfig(
                kind="swiglu" if kind is None else kind,
                hidden_dim=hidden_dim,
                expand=(8.0 / 3.0) if expand is None else expand,
                multiple_of=128 if multiple_of is None else multiple_of,
                bias=False if bias is None else bias,
            )
        else:
            _require(kind is None, "Pass either config or explicit kind, not both.")
            _require(
                hidden_dim is None,
                "Pass either config or explicit hidden_dim, not both.",
            )
            _require(
                expand is None,
                "Pass either config or explicit expand, not both.",
            )
            _require(
                multiple_of is None,
                "Pass either config or explicit multiple_of, not both.",
            )
            _require(bias is None, "Pass either config or explicit bias, not both.")

        self.d_model = int(d_model)
        self.config = config
        self.kind: MLPKind = config.kind
        self.hidden_dim = config.resolve_hidden_dim(self.d_model)

        factory_kwargs = {"device": device, "dtype": dtype}
        if self.kind == "swiglu":
            self.in_proj = nn.Linear(
                self.d_model,
                2 * self.hidden_dim,
                bias=config.bias,
                **factory_kwargs,
            )
        elif self.kind == "gelu":
            self.in_proj = nn.Linear(
                self.d_model,
                self.hidden_dim,
                bias=config.bias,
                **factory_kwargs,
            )
        else:
            raise ValueError(f"Unsupported MLP kind: {self.kind!r}.")
        self.out_proj = nn.Linear(
            self.hidden_dim,
            self.d_model,
            bias=config.bias,
            **factory_kwargs,
        )

    @classmethod
    def from_config(
        cls,
        d_model: int,
        config: SLinOSSMLPConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SLinOSSMLP":
        return cls(
            d_model,
            config=config,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {int(x.shape[-1])}."
            )
        if self.kind == "swiglu":
            value, gate = self.in_proj(x).chunk(2, dim=-1)
            return self.out_proj(value * F.silu(gate))
        return self.out_proj(F.gelu(self.in_proj(x), approximate="tanh"))

    def step(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_one(x)

    def decode_one(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected decode input shape (batch, {self.d_model}), "
                f"got {tuple(x.shape)}."
            )
        projected = decode_linear(x, self.in_proj)
        if self.kind == "swiglu":
            value, gate = projected.chunk(2, dim=-1)
            hidden = value * F.silu(gate)
        else:
            hidden = F.gelu(projected, approximate="tanh")
        return decode_linear(hidden, self.out_proj)


__all__ = [
    "MLPKind",
    "SLinOSSMLP",
    "SLinOSSMLPConfig",
    "SLinOSSMLPKind",
]
