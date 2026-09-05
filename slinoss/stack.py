"""SLinOSS block stack with optional token embedding and vocabulary head."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import linear

from slinoss._precision import Float32Module, cast_opt, cast_to
from slinoss.blocks import BlockOutput, SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.ops.block import rmsnorm_residual
from slinoss.state import MixerState, StackState

__all__ = ["SLinOSSStack"]

_HeadGrads = tuple[Tensor, Tensor, Tensor | None, None]


class _PaddedHeadFunction(torch.autograd.Function):
    """Aligned head whose padded logits and gradients are constants/zeros."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, normed: Tensor, weight: Tensor, bias: Tensor | None, vocab: int
    ) -> Tensor:
        logits = linear(normed, weight, bias)
        # The mask, not an accumulator: 112 KiB of constant at the reference
        # geometry, against a GEMM whose alignment is what the padding buys.
        logits.narrow(-1, vocab, logits.shape[-1] - vocab).fill_(
            torch.finfo(logits.dtype).min
        )
        ctx.save_for_backward(normed, weight)
        ctx.vocab = vocab
        ctx.has_bias = bias is not None
        return logits

    @staticmethod
    def backward(ctx: Any, dlogits: Tensor) -> _HeadGrads:  # type: ignore[override]
        normed, weight = ctx.saved_tensors
        vocab: int = ctx.vocab
        pad = weight.shape[0] - vocab
        flat = dlogits.flatten(0, -2)
        # The vocabulary axis is contracted here, so a padding column would reach
        # every output. Zero it out of a copy of the weight rather than out of a copy
        # of the cotangent: the same products over 14 times fewer bytes at the
        # reference geometry, and a cast is already a copy.
        masked = (
            weight.clone()
            if weight.dtype is dlogits.dtype
            else weight.to(dlogits.dtype)
        )
        masked.narrow(0, vocab, pad).zero_()
        dnormed = flat @ masked
        # One buffer for both bands: the GEMM covers it and the padding rows, which
        # hold nothing but the padding columns' contribution, are then zeroed.
        dweight = torch.empty(weight.shape, dtype=dlogits.dtype, device=dlogits.device)
        torch.mm(flat.t(), cast_to(normed.flatten(0, -2), dlogits.dtype), out=dweight)
        dweight.narrow(0, vocab, pad).zero_()
        dbias: Tensor | None = None
        if ctx.has_bias:
            dbias = torch.empty(
                weight.shape[0], dtype=dlogits.dtype, device=dlogits.device
            )
            torch.sum(flat, 0, out=dbias)
            dbias.narrow(0, vocab, pad).zero_()
        return (
            cast_to(dnormed.view(normed.shape), normed.dtype),
            cast_to(dweight, weight.dtype),
            cast_opt(dbias, weight.dtype),
            None,
        )


class SLinOSSStack(Float32Module):
    """A stack of :class:`SLinOSSBlock` modules under one config."""

    _float32_names = ("norm_weight",)

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
            torch.ones(config.d_model, device=device, dtype=torch.float32)
        )
        padded_vocab = config.padded_vocab_size
        self.head: nn.Linear | None = (
            nn.Linear(
                config.d_model,
                padded_vocab,
                bias=config.bias,
                device=device,
                dtype=dtype,
            )
            if padded_vocab is not None
            else None
        )

    def forward(self, x: Tensor, state: StackState | None = None) -> Tensor:
        """Run every block over one sequence, or continue one from ``state``.

        With a state each block continues its own layer and advances it in place, so
        prefill and decode are this method at two sequence lengths.

        Args:
            x: ``(B,T)`` integer token ids when ``vocab_size`` is set, otherwise
                ``(B,T,d_model)`` activations in bf16/fp16/fp32.
            state: Decode state for the whole stack, or None to run whole
                sequences.

        Returns:
            ``(B,T,padded_vocab_size)`` logits when ``vocab_size`` is set,
            otherwise ``(B,T,d_model)`` normed activations. Both in the dtype the
            last GEMM produces.

            The first ``vocab_size`` columns are the logits and are bit-identical
            to an unpadded head's over the same weight rows. The rest hold
            ``finfo(dtype).min``, and hold it as a constant: exactly zero under a
            softmax at every supported dtype, below every reachable logit so no
            argmax and no sample can land on one, and gradient-transparent, so a
            cotangent placed on one reaches neither head gradient and the head's
            rows past ``vocab_size`` come back with exactly zero. They are not
            outputs and carry no meaning.

        Raises:
            ValueError: On a rank the configured input form does not admit, on a
                state whose depth is not this stack's, or from a consumer's guard
                on a device, shape, or layout its operand rule refuses.
            TypeError: From a consumer's guard, on an unsupported dtype.
        """
        layers: tuple[MixerState | None, ...] = (
            (None,) * len(self.blocks) if state is None else state.layers
        )
        if len(layers) != len(self.blocks):
            raise ValueError(
                f"state has depth {len(layers)} and the stack has "
                f"{len(self.blocks)} layers"
            )

        # The embedding, the final norm and the head are outside every block, so the
        # block's own grad gate does not cover them. See SLinOSSBlock.forward.
        with torch.set_grad_enabled(state is None and torch.is_grad_enabled()):
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
            for module, layer in zip(self.blocks, layers, strict=True):
                out = cast(
                    "BlockOutput",
                    cast("SLinOSSBlock", module)(hidden, residual, layer),
                )
                hidden, residual = out.hidden, out.residual

            normed = rmsnorm_residual(
                hidden, residual, self.norm_weight, eps=self.config.norm_eps
            ).normed
            if self.head is None:
                return normed
            vocab = self.config.vocab_size
            if vocab is None or self.head.out_features == vocab:
                return cast("Tensor", self.head(normed))
            return cast(
                "Tensor",
                _PaddedHeadFunction.apply(
                    normed, self.head.weight, self.head.bias, vocab
                ),
            )
