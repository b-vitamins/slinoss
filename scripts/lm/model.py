"""The published KLA language-model scaffold: embedding, mixer blocks, norm, head.

KLA follows Mamba's *fused-MLP mixer*: the expansion, gate and channel mixing live inside
the sequence mixer. There is no second transformer-style FFN after it. This is independently
identified by the paper's dimensions: a 12-layer 8-head attention control at width 496 has
44.3M total parameters, and width 1360 has 177.9M, only under the mixer-only scaffold. Adding
an external FFN makes both published 45M/180M configurations impossible.

Every block is therefore exactly ``x = x + mixer(rmsnorm(x))``. A per-layer factory list
supports the published hybrid without a second build path. The token table is stored in
bfloat16 and cast on lookup; every mixer, norm, residual and head computation remains fp32.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from scripts.harness import MixerFactory
from slinoss import SLinOSSMixer
from slinoss.config import VOCAB_MULTIPLE
from slinoss.ops.block import rmsnorm

__all__ = [
    "LMConfig",
    "MixerLM",
    "MixerResidualBlock",
    "build_model",
    "layer_factories",
    "mixer_parameters",
    "non_embedding_parameters",
    "parameter_count",
]


@dataclass(frozen=True)
class LMConfig:
    """Configuration consumed by the external mixer-only LM scaffold."""

    d_model: int
    n_layers: int
    vocab_size: int
    bias: bool = False
    norm_eps: float = 1e-5
    vocab_pad_multiple: int = VOCAB_MULTIPLE

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.norm_eps <= 0.0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.vocab_pad_multiple < 1:
            raise ValueError(
                f"vocab_pad_multiple must be positive, got {self.vocab_pad_multiple}"
            )
        if (
            self.vocab_pad_multiple != 1
            and self.vocab_pad_multiple % VOCAB_MULTIPLE != 0
        ):
            raise ValueError(
                "vocab_pad_multiple must be 1 or a multiple of "
                f"{VOCAB_MULTIPLE}, got {self.vocab_pad_multiple}"
            )

    @property
    def padded_vocab_size(self) -> int:
        """Vocabulary width rounded up to ``vocab_pad_multiple``."""
        return -(-self.vocab_size // self.vocab_pad_multiple) * self.vocab_pad_multiple


class MixerResidualBlock(nn.Module):
    """One pre-norm residual mixer and no external FFN."""

    def __init__(self, d_model: int, mixer: nn.Module, *, norm_eps: float) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.mixer = mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the published fused-mixer block."""
        return x + self.mixer(rmsnorm(x, self.norm_weight, eps=self.norm_eps))


class MixerLM(nn.Module):
    """Untied embedding/head around a uniform list of mixer-residual blocks."""

    def __init__(self, config: LMConfig, mixers: Sequence[nn.Module]) -> None:
        super().__init__()
        if len(mixers) != config.n_layers:
            raise ValueError(f"{len(mixers)} mixers for {config.n_layers} layers")
        self.config = config
        padded_vocab_size = config.padded_vocab_size
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, dtype=torch.bfloat16
        )
        self.blocks = nn.ModuleList(
            MixerResidualBlock(config.d_model, mixer, norm_eps=config.norm_eps)
            for mixer in mixers
        )
        self.norm_weight = nn.Parameter(torch.ones(config.d_model, dtype=torch.float32))
        self.head = nn.Linear(
            config.d_model,
            padded_vocab_size,
            bias=config.bias,
            dtype=torch.float32,
        )

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> MixerLM:
        """Move/cast the model while preserving the protocol's parameter dtypes."""
        super()._apply(fn, recurse)
        self.embedding.weight.data = self.embedding.weight.data.to(torch.bfloat16)
        self.norm_weight.data = self.norm_weight.data.to(torch.float32)
        for block in self.blocks:
            cast("MixerResidualBlock", block).norm_weight.data = cast(
                "MixerResidualBlock", block
            ).norm_weight.data.to(torch.float32)
        return self

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return causal next-token logits, with padded columns made unreachable."""
        if ids.ndim != 2:
            raise ValueError(f"expected (B,T) token ids, got {tuple(ids.shape)}")
        hidden = self.embedding(ids).to(self.head.weight.dtype)
        for block in self.blocks:
            hidden = cast("MixerResidualBlock", block)(hidden)
        hidden = rmsnorm(hidden, self.norm_weight, eps=self.config.norm_eps)
        logits = self.head(hidden)
        vocab = self.config.vocab_size
        if logits.shape[-1] == vocab:
            return logits
        padding = logits.new_full(
            (*logits.shape[:-1], logits.shape[-1] - vocab),
            torch.finfo(logits.dtype).min,
        )
        return torch.cat((logits[..., :vocab], padding), dim=-1)


def layer_factories(
    factory: MixerFactory, n_layers: int, final: MixerFactory | None = None
) -> list[MixerFactory]:
    """One factory per layer, with an optional different last one.

    Args:
        factory: The mixer every layer gets.
        n_layers: Blocks.
        final: Mixer for the last block only, or None to leave it as the rest. This is the
            hybrid arm: a stack of one mixer whose final layer is another.

    Returns:
        A list of length ``n_layers``.

    Raises:
        ValueError: On a non-positive depth.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    factories = [factory] * n_layers
    if final is not None:
        factories[-1] = final
    return factories


def build_model(
    config: LMConfig,
    factories: Sequence[MixerFactory],
    *,
    max_length: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> MixerLM:
    """Build the scaffold and swap in one mixer per layer.

    Construction is on the host and the move is one call at the end, so no arm is ever half
    on the device: a factory that allocates on the default device would otherwise put the
    mixer somewhere the scaffold is not.

    Call under a seeded generator. Every parameter here is a torch default draw or a
    mixer's own initialization, so the seed has to be set first for an arm to reproduce.

    Args:
        config: Scaffold shape.
        factories: One mixer factory per layer, from :func:`layer_factories`.
        max_length: Longest sequence the arm will run, passed to each factory.
        device: Destination device.
        dtype: Destination dtype for every parameter but the norm gains, which
            :meth:`slinoss.SLinOSSStack._apply` keeps in float32.

    Returns:
        The stack.

    Raises:
        ValueError: When the factory count is not the depth, or when a swapped-in
            :class:`slinoss.SLinOSSMixer` was built at another width than the scaffold's.
    """
    if len(factories) != config.n_layers:
        raise ValueError(f"{len(factories)} factories for {config.n_layers} layers")
    mixers: list[nn.Module] = []
    for factory in factories:
        mixer = factory(config.d_model, max_length)
        if isinstance(mixer, SLinOSSMixer) and mixer.config.d_model != config.d_model:
            raise ValueError(
                f"mixer built at d_model {mixer.config.d_model} and the scaffold is "
                f"{config.d_model}"
            )
        mixers.append(mixer)
    model = MixerLM(config, mixers)
    return model.to(device=device, dtype=dtype or torch.float32)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters.

    Args:
        model: The model.

    Returns:
        The count. The head's padding columns are parameters and are counted: they are
        allocated, they are optimized, and they hold no output.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def non_embedding_parameters(model: MixerLM) -> int:
    """Trainable parameters outside the token table and the head.

    Args:
        model: The stack.

    Returns:
        The count. This is what arms are matched on: the embedding and the head scale with
        the vocabulary and are identical across arms at one width, so including them would
        let a wider mixer hide behind a shared table.
    """
    total = parameter_count(model)
    for module in (model.embedding, model.head):
        total -= sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total


def mixer_parameters(model: MixerLM) -> int:
    """Trainable parameters inside the mixers only.

    Args:
        model: The stack.

    Returns:
        The count over every block's ``mixer``. The rest of the scaffold is shared across
        arms, so this separates what the recurrence contributes from what the norms, the
        FFN, the embedding and the head contribute.
    """
    total = 0
    for module in model.blocks:
        block = cast("MixerResidualBlock", module)
        total += sum(p.numel() for p in block.mixer.parameters() if p.requires_grad)
    return total
