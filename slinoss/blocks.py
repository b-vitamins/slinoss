"""The residual block: fused pre-norm, the mixer, fused pre-norm, a SwiGLU FFN.

    normed, r = norm(x, r);   m = mixer(normed)
    normed, r = norm(m, r);   h = W_o swiglu(W_g normed, W_u normed)

The block hands back ``(h, r)``: the branch output unadded and the stream it
belongs to. The add is the next fused norm's first operation, so the stream is
touched once per norm instead of once per add and again per norm, and a stack
carries one pass over it per branch. The last branch is added by the stack's
final norm.

The stream is float32 at every activation dtype, which is what the fused norm
returns. A deep stack therefore accumulates wide and narrows only where a GEMM
reads.

The FFN projection is two GEMMs over two weights rather than one GEMM over a
fused ``[gate | up]`` weight. :func:`slinoss.ops.block.swiglu` takes contiguous
operands and the two halves of a fused projection output are column bands, so one
GEMM buys either a copy of both halves or a pitched path through the activation
kernel. Slicing one fused weight is not a third option: each slice's pullback
allocates a full-width zero buffer and the two are summed, which is an allocation
per step that the two-weight form does not make. What the two-weight form costs
instead is a second read of ``normed``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn

from slinoss._precision import LOW_PRECISION_DTYPES
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


class SLinOSSBlock(nn.Module):
    """Pre-norm residual block around :class:`slinoss.SLinOSSMixer`.

    Two fused residual norms, the mixer, and a SwiGLU FFN of width
    :attr:`slinoss.SLinOSSConfig.d_ffn`. Both norm weights are float32 at every
    module dtype and stay float32 through a module-wide cast, because the kernel
    backend refuses a low-precision norm weight (I4).

    CUDA, for the same reason the mixer is: its operands are column bands of one
    projection and the band rule is a device rule.

    Initialization is the framework default everywhere except the norm weights,
    which are ones, and the mixer, which owns its own. No depth scaling.

    Args:
        config: Shape and parameterization contract.
        device: Device for every parameter.
        dtype: Dtype for every parameter except the two norm weights.
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
        self.mixer_norm_weight = nn.Parameter(
            torch.empty(config.d_model, device=device, dtype=torch.float32)
        )
        self.mixer = SLinOSSMixer(config, device=device, dtype=dtype)
        self.ffn_norm_weight = nn.Parameter(
            torch.empty(config.d_model, device=device, dtype=torch.float32)
        )
        self.ffn_gate = nn.Linear(
            config.d_model, config.d_ffn, bias=config.bias, device=device, dtype=dtype
        )
        self.ffn_up = nn.Linear(
            config.d_model, config.d_ffn, bias=config.bias, device=device, dtype=dtype
        )
        self.ffn_out = nn.Linear(
            config.d_ffn, config.d_model, bias=config.bias, device=device, dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place, the mixer's included.

        Called by the constructor.
        """
        with torch.no_grad():
            self.mixer_norm_weight.fill_(1.0)
            self.ffn_norm_weight.fill_(1.0)
            self.mixer.reset_parameters()
            self.ffn_gate.reset_parameters()
            self.ffn_up.reset_parameters()
            self.ffn_out.reset_parameters()

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> SLinOSSBlock:
        """Apply ``fn`` to every parameter, then undo a demoted norm weight.

        ``block.to(torch.bfloat16)`` is how the module is meant to reach a kernel
        dtype, and the two norm weights are the parameters that cannot follow. A
        widening cast is left alone, so a float64 module keeps a float64 oracle
        end to end.

        Args:
            fn: The per-tensor operation :meth:`torch.nn.Module._apply` applies.
            recurse: Whether to descend into submodules.

        Returns:
            This module.
        """
        super()._apply(fn, recurse)
        for weight in (self.mixer_norm_weight, self.ffn_norm_weight):
            if weight.dtype in LOW_PRECISION_DTYPES:
                weight.data = weight.data.to(torch.float32)
        return self

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
                hidden=cast("Tensor", self.ffn_out(gated)), residual=post.residual
            )
