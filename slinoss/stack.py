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

The head is :attr:`slinoss.SLinOSSConfig.padded_vocab_size` wide, not
``vocab_size``: all three of its GEMMs read their operand alignment off that
width, so an unaligned one costs every one of them its wide load and half its MMA
K-extent. The columns past ``vocab_size`` carry ``finfo(dtype).min``, which is
zero under every softmax and unreachable by every argmax. The embedding is a
gather and is not padded.

A :class:`slinoss.StackState` threads decode through the same loop: each block
continues its own layer and advances it in place, so a prefill and a single-token
step are one call at two sequence lengths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import linear

from slinoss._precision import LOW_PRECISION_DTYPES, cast_opt, cast_to
from slinoss.blocks import BlockOutput, SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.ops.block import rmsnorm_residual
from slinoss.state import MixerState, StackState

__all__ = ["SLinOSSStack"]

_HeadGrads = tuple[Tensor, Tensor, Tensor | None, None]


class _PaddedHeadFunction(torch.autograd.Function):
    """The head GEMM and the constant it writes past ``vocab_size``, as one node.

    The padding columns are constants, so a cotangent on one must reach neither
    head gradient. Autograd states that by recording the write and clearing the
    padding band of the cotangent in the pullback, and clearing a band of a tensor
    it does not own makes it clone the whole logit block first. Measured at the
    reference geometry: one 1.65 GB device-to-device copy, 2.416 ms, a third of the
    step's glue, and 2,424 MiB of peak against 1,166 MiB, the clone and the cleared
    cotangent both being the size of the logit block.

    The write is not recorded here, and the pullback clears nothing on the cotangent.
    It keeps the padding out of each gradient where that gradient reads it, and every
    contraction stays at the padded width while it does, because that width is what
    aligns them: narrowing the vocabulary axis of ``dnormed`` costs 4.05 ms of the
    4.62 ms the aligned one takes, and narrowing ``dbias`` 0.61 ms of 1.21 ms.
    ``dweight`` and ``dbias`` reduce over tokens, so a padding column reaches only a
    padding row of either, and that band is overwritten with zero. ``dnormed``
    contracts the vocabulary axis, so it reads a copy of the weight whose padding rows
    are zero, which costs 0.13 ms.

    Every product is then the recorded version's and every sum is over the same terms
    in the same order, because a zero padding row contributes exactly the zero a
    cleared padding column contributed. Measured at the reference geometry over a
    cotangent drawn on every column, padding included: all three gradients bitwise
    the recorded version's at bf16, float32 and float64.

    No :func:`torch.amp.custom_fwd`. Autocast rewrites the GEMM's operands where it
    is issued, and the gradients are cast back to the parameter dtypes on the way
    out, so nothing here needs the eager cast that would also demote ``normed``.
    """

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


class SLinOSSStack(nn.Module):
    """``n_layers`` :class:`slinoss.SLinOSSBlock` under one config.

    The final norm weight is float32 at every module dtype and stays float32
    through a module-wide cast, for the reason the block's two are (I4).

    CUDA, for the reason the mixer is.

    Initialization is the framework default everywhere except the final norm
    weight, which is ones, and the blocks, which own their own. No depth scaling
    and no weight tying: the head is its own parameter, and the embedding stays
    ``vocab_size`` rows while the head is
    :attr:`slinoss.SLinOSSConfig.padded_vocab_size`.

    The head's rows past ``vocab_size`` are left at the framework default. Their
    value never reaches an output, because :meth:`forward` overwrites their columns
    with ``finfo(dtype).min``, and their gradient is exactly zero, because the
    pullback of that write contracts the live columns only. Both are
    :class:`_PaddedHeadFunction`.

    Args:
        config: Shape and parameterization contract. ``n_layers`` sets the depth,
            ``vocab_size`` decides whether the embedding and the head exist, and
            ``vocab_pad_multiple`` sets how much wider than ``vocab_size`` the head
            is.
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
