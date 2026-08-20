"""The stack: an optional embedding, ``n_layers`` blocks, a final fused norm, an
optional head.

    r = None
    for each block:  h, r = block(h, r)
    out = W_head norm(h, r)

One fused norm per branch and one more at the end. Every block returns its branch
output unadded, so the final norm is the add the last block did not do rather than
an extra pass over the stream.

The stream is float32 from the first block's norm to the last, which is what
:func:`slinoss.ops.block.rmsnorm_residual` returns at every activation dtype. The
head reads the normed output in the activation dtype, so nothing but the stream is
wide.

``vocab_size`` decides both ends together. With it the stack takes token ids and
returns logits; without it the stack takes and returns activations, and embeds
into a larger model that owns those two layers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from slinoss._precision import LOW_PRECISION_DTYPES
from slinoss.blocks import BlockOutput, SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.ops.block import rmsnorm_residual

__all__ = ["SLinOSSStack"]


class SLinOSSStack(nn.Module):
    """``n_layers`` :class:`slinoss.SLinOSSBlock` under one config.

    The final norm weight is float32 at every module dtype and stays float32
    through a module-wide cast, for the reason the block's two are (I4).

    CUDA, for the reason the mixer is.

    Initialization is the framework default everywhere except the final norm
    weight, which is ones, and the blocks, which own their own. No depth scaling
    and no weight tying: the head is its own parameter.

    Args:
        config: Shape and parameterization contract. ``n_layers`` sets the depth
            and ``vocab_size`` decides whether the embedding and the head exist.
        device: Device for every parameter.
        dtype: Dtype for every parameter except the norm weights.
    """

    def __init__(
        self,
        config: SLinOSSConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding: nn.Embedding | None = (
            nn.Embedding(config.vocab_size, config.d_model, device=device, dtype=dtype)
            if config.vocab_size is not None
            else None
        )
        self.blocks = nn.ModuleList(
            SLinOSSBlock(config, device=device, dtype=dtype)
            for _ in range(config.n_layers)
        )
        self.norm_weight = nn.Parameter(
            torch.empty(config.d_model, device=device, dtype=torch.float32)
        )
        self.head: nn.Linear | None = (
            nn.Linear(
                config.d_model,
                config.vocab_size,
                bias=config.bias,
                device=device,
                dtype=dtype,
            )
            if config.vocab_size is not None
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place, every block's included.

        Called by the constructor.
        """
        with torch.no_grad():
            self.norm_weight.fill_(1.0)
            if self.embedding is not None:
                self.embedding.reset_parameters()
            if self.head is not None:
                self.head.reset_parameters()
            for module in self.blocks:
                cast("SLinOSSBlock", module).reset_parameters()

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> SLinOSSStack:
        """Apply ``fn`` to every parameter, then undo a demoted norm weight.

        Each block undoes its own two; this handles the final one. A widening cast
        is left alone.

        Args:
            fn: The per-tensor operation :meth:`torch.nn.Module._apply` applies.
            recurse: Whether to descend into submodules.

        Returns:
            This module.
        """
        super()._apply(fn, recurse)
        if self.norm_weight.dtype in LOW_PRECISION_DTYPES:
            self.norm_weight.data = self.norm_weight.data.to(torch.float32)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Run every block over one sequence.

        Args:
            x: ``(B,T)`` integer token ids when ``vocab_size`` is set, otherwise
                ``(B,T,d_model)`` activations in bf16/fp16/fp32.

        Returns:
            ``(B,T,vocab_size)`` logits when ``vocab_size`` is set, otherwise
            ``(B,T,d_model)`` normed activations. Both in the dtype the last GEMM
            produces.

        Raises:
            ValueError: On a rank the configured input form does not admit, or
                from a consumer's guard on a device, shape, or layout its operand
                rule refuses.
            TypeError: From a consumer's guard, on an unsupported dtype.
        """
        if self.embedding is not None:
            if x.ndim != 2:
                raise ValueError(f"expected (B,T) token ids, got {tuple(x.shape)}")
            hidden = self.embedding(x)
        else:
            if x.ndim != 3 or x.shape[-1] != self.config.d_model:
                raise ValueError(
                    f"expected (B,T,{self.config.d_model}), got {tuple(x.shape)}"
                )
            hidden = x

        residual: Tensor | None = None
        for module in self.blocks:
            out = cast("BlockOutput", cast("SLinOSSBlock", module)(hidden, residual))
            hidden, residual = out.hidden, out.residual

        normed = rmsnorm_residual(
            hidden, residual, self.norm_weight, eps=self.config.norm_eps
        ).normed
        if self.head is None:
            return normed
        return cast("Tensor", self.head(normed))
