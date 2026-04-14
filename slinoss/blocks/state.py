"""State containers for SLinOSS blocks and stacks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from slinoss.layers.state import SLinOSSMixerState


@dataclass
class SLinOSSBlockState:
    """Streaming state for one SLinOSS block."""

    mixer: SLinOSSMixerState = field(default_factory=SLinOSSMixerState)

    def copy_(self, other: "SLinOSSBlockState") -> "SLinOSSBlockState":
        self.mixer.copy_(other.mixer)
        return self

    def clone(self) -> "SLinOSSBlockState":
        return SLinOSSBlockState(mixer=self.mixer.clone())

    def detach(self) -> "SLinOSSBlockState":
        return SLinOSSBlockState(mixer=self.mixer.detach())

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SLinOSSBlockState":
        return SLinOSSBlockState(mixer=self.mixer.to(device=device, dtype=dtype))


@dataclass
class SLinOSSStackState:
    """Streaming state for a stack of SLinOSS blocks."""

    layers: list[SLinOSSBlockState] = field(default_factory=list)

    def copy_(self, other: "SLinOSSStackState") -> "SLinOSSStackState":
        for dst, src in zip(self.layers, other.layers, strict=True):
            dst.copy_(src)
        return self

    def clone(self) -> "SLinOSSStackState":
        return SLinOSSStackState(layers=[layer.clone() for layer in self.layers])

    def detach(self) -> "SLinOSSStackState":
        return SLinOSSStackState(layers=[layer.detach() for layer in self.layers])

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SLinOSSStackState":
        return SLinOSSStackState(
            layers=[layer.to(device=device, dtype=dtype) for layer in self.layers]
        )


__all__ = ["SLinOSSBlockState", "SLinOSSStackState"]
