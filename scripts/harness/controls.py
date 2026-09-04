"""The two non-recurrent mixers that bracket a recurrence.

``CausalAttention`` is the strongest thing that carries no fixed-size state: it reads the
whole prefix, so it is the ceiling on any task whose answer is in the prefix and the floor on
any task whose answer needs a state carried past the context. ``CausalConv`` has a receptive
field of ``d_conv`` positions per layer, so it is the star-free floor: a task it solves is
not measuring a carried state.

Both are transcribed once and shared, so an axis that wants a control does not build one.

Registration is the caller's, not this module's: the defaults an axis wants differ, and a
control registered here would fix them for every axis.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

__all__ = ["CausalAttention", "CausalConv", "Rotary"]


class Rotary(nn.Module):
    """Rotary position code, the half-split convention.

    The tables are built once at ``max_length`` and sliced, so no arm pays a rebuild per
    step. A batch wider than ``max_length`` is a configuration error, not a resize: an
    evaluation past the trained length is the measurement, and silently extending the
    tables would hide which arm was asked to extrapolate.

    Args:
        d_head: Channels per head. Even.
        max_length: Longest sequence the tables cover.
        base: Frequency base.

    Raises:
        ValueError: On an odd ``d_head``, which the half split cannot pair.
    """

    def __init__(self, d_head: int, max_length: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {d_head}")
        freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        angle = torch.outer(torch.arange(max_length, dtype=torch.float32), freq)
        doubled = torch.cat([angle, angle], dim=-1)
        self.register_buffer("cos", doubled.cos(), persistent=False)
        self.register_buffer("sin", doubled.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Rotate every head's channels by its position's angle.

        Args:
            x: ``(B,H,T,P)``.

        Returns:
            The same shape and dtype.

        Raises:
            ValueError: When ``T`` is over the table's length.
        """
        length = x.shape[-2]
        table = cast("Tensor", self.cos)
        if length > table.shape[0]:
            raise ValueError(
                f"sequence of {length} is over the rotary table's {table.shape[0]}"
            )
        cos = table[:length].to(x.dtype)
        sin = cast("Tensor", self.sin)[:length].to(x.dtype)
        half = x.shape[-1] // 2
        flipped = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + flipped * sin


class CausalAttention(nn.Module):
    """Causal multi-head attention with rotary positions.

    Projections carry no bias and the attention is
    :func:`torch.nn.functional.scaled_dot_product_attention` rather than a fused kernel.

    Args:
        d_model: Stream width.
        max_length: Longest sequence, for the rotary tables.
        n_heads: Heads. Divides ``d_model``.
        rotary: Whether to rotate. Without it the mixer is position-blind and a scaffold
            that carries no positional embedding collapses on any task whose answer
            depends on order.

    Raises:
        ValueError: When ``n_heads`` does not divide ``d_model``.
    """

    def __init__(
        self, d_model: int, max_length: int, n_heads: int = 16, rotary: bool = True
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"n_heads {n_heads} does not divide d_model {d_model}")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = (
            Rotary(d_model // n_heads, max_length) if rotary else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Attend over the prefix.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``.
        """
        bsz, length, _ = x.shape
        qkv = self.qkv(x).unflatten(-1, (3, self.n_heads, -1)).permute(2, 0, 3, 1, 4)
        query, key, value = (self.rotary(qkv[0]), self.rotary(qkv[1]), qkv[2])
        out = nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        return self.out_proj(out.transpose(1, 2).reshape(bsz, length, -1))


class CausalConv(nn.Module):
    """Causal depthwise convolution over ``d_conv`` taps, gated.

    Args:
        d_model: Stream width.
        d_conv: Taps.
        expand: Inner width multiplier.

    Raises:
        ValueError: On fewer than one tap.
    """

    def __init__(self, d_model: int, d_conv: int = 4, expand: float = 2.0) -> None:
        super().__init__()
        if d_conv < 1:
            raise ValueError(f"d_conv must be positive, got {d_conv}")
        inner = round(expand * d_model)
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner, bias=True)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Convolve over the prefix.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``.
        """
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        padded = nn.functional.pad(value.transpose(1, 2), (self.d_conv - 1, 0))
        mixed = self.conv(padded).transpose(1, 2)
        return self.out_proj(mixed * nn.functional.silu(gate))
