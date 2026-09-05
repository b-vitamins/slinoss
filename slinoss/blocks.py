"""Pre-norm residual block with an SLinOSS mixer and SwiGLU FFN."""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn

from slinoss._precision import Float32Module
from slinoss.config import SLinOSSConfig
from slinoss.mixer import SLinOSSMixer
from slinoss.ops.block import rmsnorm_residual, swiglu
from slinoss.state import MixerState

__all__ = ["BlockOutput", "SLinOSSBlock"]


class BlockOutput(NamedTuple):
    """What one block hands to the next.

    Attributes:
        hidden: ``(B,T,d_model)`` FFN branch output in the activation dtype. Not
            yet added to ``residual``; the next fused norm adds it.
        residual: ``(B,T,d_model)`` float32 stream through the block's input and
            the mixer branch.
    """

    hidden: Tensor
    residual: Tensor


class SLinOSSBlock(Float32Module):
    """Two fused residual norms around a mixer and a SwiGLU FFN."""

    _float32_names = ("mixer_norm_weight", "ffn_norm_weight")

    def __init__(
        self,
        config: SLinOSSConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mixer_norm_weight = nn.Parameter(
            torch.ones(config.d_model, device=device, dtype=torch.float32)
        )
        self.mixer = SLinOSSMixer(config, device=device, dtype=dtype)
        self.ffn_norm_weight = nn.Parameter(
            torch.ones(config.d_model, device=device, dtype=torch.float32)
        )
        self.ffn_gate = nn.Linear(
            config.d_model, config.d_ffn, bias=config.bias, device=device, dtype=dtype
        )
        self.ffn_up = nn.Linear(
            config.d_model, config.d_ffn, bias=config.bias, device=device, dtype=dtype
        )
        self.ffn_out = nn.Linear(
            config.d_ffn,
            config.d_model,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        residual_scale = 1.0 / math.sqrt(2.0 * config.n_layers)
        with torch.no_grad():
            self.mixer.out_proj.weight.mul_(residual_scale)
            self.ffn_out.weight.mul_(residual_scale)
            bias = cast(Tensor | None, self.ffn_out.bias)
            if bias is not None:
                bias.zero_()

    def forward(
        self,
        x: Tensor,
        residual: Tensor | None = None,
        state: MixerState | None = None,
    ) -> BlockOutput:
        """Run both branches over one sequence, or continue one from ``state``.

        The composition is stated once for both paths. Only the mixer call differs:
        with a state it is :meth:`slinoss.SLinOSSMixer.step`, which advances the
        state in place.

        Args:
            x: ``(B,T,d_model)`` branch input, bf16/fp16/fp32.
            residual: ``(B,T,d_model)`` incoming stream, or None for the first
                block of a stack. Float32 when it comes from another block.
            state: This layer's decode state, or None to mix a whole sequence.

        Returns:
            A :class:`BlockOutput`.

        Raises:
            ValueError: From a consumer's guard, on a device, shape, or layout
                its operand rule refuses.
            TypeError: From a consumer's guard, on an unsupported dtype.
        """
        # Decode takes no gradient. The mixer's step records nothing whatever the
        # caller's grad mode, so a block that recorded the norms and the FFN around
        # it would grow a graph per token that reaches no mixer parameter.
        with torch.set_grad_enabled(state is None and torch.is_grad_enabled()):
            eps = self.config.norm_eps
            pre = rmsnorm_residual(x, residual, self.mixer_norm_weight, eps=eps)
            mixed = (
                self.mixer(pre.normed)
                if state is None
                else self.mixer.step(pre.normed, state)
            )
            post = rmsnorm_residual(mixed, pre.residual, self.ffn_norm_weight, eps=eps)
            gated = swiglu(self.ffn_gate(post.normed), self.ffn_up(post.normed))
            return BlockOutput(
                hidden=self.ffn_out(gated),
                residual=post.residual,
            )
