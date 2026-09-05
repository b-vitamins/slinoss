"""The scaffold every arm on this axis is scored on.

`structured-linear-cdes`'s ``models/mamba.py`` and ``models/deltanet2.py`` carry the same
block around different mixers, and `expressive-sparse-state-space-model`'s ``models/pdssm.py``
carries its own copy of the same shape; that block is transcribed here once so a baseline
is a mixer and nothing else. It is post-norm:

    y = mixer(x)
    y = y + x
    if use_glu: y = y + glu(post_linear(y))
    y = norm(y)
    y = dropout(y)

and the model is ``nn.Embedding`` -> blocks -> ``nn.Linear``, logits at every position.

Three properties of it are load-bearing and are the reason it is transcribed rather than
rewritten.

The embedding parameter is named ``embedding``. Upstream's optimizer splits its two weight
decay groups on the substring ``embedding`` appearing in a parameter's name, so renaming
the attribute silently moves the token table from zero decay to 1e-2.

The norm closes the block rather than opening it. A post-norm stack rescales the residual
stream after every mixer, so a recurrence's output cannot dominate the skip by growing;
under pre-norm it can. On a task whose answer is a state carried unchanged for hundreds of
steps that is not a cosmetic difference, and it is the shape every published number on
this axis was measured under.

There is no initialization pass. No upstream tree on this axis touches a default: the
embedding is ``N(0,1)``, the linears are Kaiming-uniform, the norm is unit-affine. A
mixer's own initialization is the mixer's business and runs inside its constructor.

One divergence: ``use_glu`` defaults to False, as every one of
`expressive-sparse-state-space-model`'s twenty experiment configs sets it, and the flag is
kept because `structured-linear-cdes` exposes it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

MixerFactory = Callable[[int, int], nn.Module]
"""``(d_model, max_length) -> module``. Maps ``(B,T,d_model)`` to ``(B,T,d_model)``.

The max length is passed because a mixer may need to size a positional term or a cache;
it is the widest batch the arm can produce, not the width of any one batch."""


@dataclass(frozen=True)
class ModelConfig:
    """Scaffold shape.

    Attributes:
        input_vocab_size: Number of input symbols; upstream's ``data_dim``.
        output_vocab_size: Number of classifier classes; upstream's ``label_dim``.
            Defaults to ``input_vocab_size`` for the released regular tasks.
        max_length: Widest batch the arm can produce, passed to the mixer factory.
        d_model: Residual width. Upstream's ``model_dim``, 128.
        n_layers: Blocks. Upstream's ``num_blocks``, 2.
        dropout: Dropout after each block's norm. Upstream's ``dropout_rate``, 0.01.
        use_glu: Whether each block carries the gated-linear branch.
    """

    input_vocab_size: int
    max_length: int
    d_model: int = 128
    n_layers: int = 2
    dropout: float = 0.01
    use_glu: bool = False
    output_vocab_size: int | None = None

    def __post_init__(self) -> None:
        if self.output_vocab_size is None:
            object.__setattr__(self, "output_vocab_size", self.input_vocab_size)
        for name in (
            "input_vocab_size",
            "output_vocab_size",
            "max_length",
            "d_model",
            "n_layers",
        ):
            value = getattr(self, name)
            if value is None or value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def vocab_size(self) -> int:
        """The legacy shared size, refusing asymmetric input/output vocabularies."""
        if self.input_vocab_size != self.output_vocab_size:
            raise ValueError(
                "vocab_size is ambiguous; use input_vocab_size or output_vocab_size"
            )
        return self.input_vocab_size


class Block(nn.Module):
    """One post-norm block.

    Args:
        mixer: The sequence mixer, already built.
        d_model: Residual width.
        dropout: Dropout probability.
        use_glu: Whether to carry the gated-linear branch.
    """

    def __init__(
        self, mixer: nn.Module, d_model: int, dropout: float, use_glu: bool
    ) -> None:
        super().__init__()
        self.mixer = mixer
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p=dropout)
        self.use_glu = use_glu
        # Twice the width in: glu halves it again.
        self.post_linear = nn.Linear(d_model, 2 * d_model) if use_glu else None

    def forward(self, x: Tensor) -> Tensor:
        """Mix, add, optionally gate, normalize, drop.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``.
        """
        y = self.mixer(x) + x
        if self.post_linear is not None:
            y = y + nn.functional.glu(self.post_linear(y), dim=-1)
        return self.drop(self.norm(y))


class StateTracker(nn.Module):
    """Embedding, a stack of blocks, a linear head.

    Args:
        config: Scaffold shape.
        factory: Builds one mixer per block. Called ``n_layers`` times, so a factory that
            returns a shared module would tie the layers; each call must build.
    """

    def __init__(self, config: ModelConfig, factory: MixerFactory) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.input_vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            Block(
                factory(config.d_model, config.max_length),
                config.d_model,
                config.dropout,
                config.use_glu,
            )
            for _ in range(config.n_layers)
        )
        if config.output_vocab_size is None:  # narrowed by ModelConfig.__post_init__
            raise AssertionError("resolved output vocabulary is missing")
        self.head = nn.Linear(config.d_model, config.output_vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Logits at every position.

        Args:
            tokens: ``(B,T)`` int64 in ``[0, input_vocab_size)``.

        Returns:
            ``(B,T,output_vocab_size)``. Every position, never just the last: the loss
            selects with the batch's mask.
        """
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build_model(config: ModelConfig, factory: MixerFactory) -> StateTracker:
    """Build the scaffold.

    Call under a seeded generator: every parameter here is a torch default draw, so the
    seed has to be set first for an arm to reproduce.

    Args:
        config: Scaffold shape.
        factory: Mixer factory.

    Returns:
        The model, on the CPU in float32.
    """
    return StateTracker(config, factory)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters.

    Args:
        model: The model.

    Returns:
        The count. Reported in every record: two mixers matched on ``d_model`` are not
        matched on parameters, and a win at unequal count is not a win.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mixer_parameters(model: StateTracker) -> int:
    """Parameters inside the mixers only.

    Args:
        model: The model.

    Returns:
        The count over every block's ``mixer`` submodule. The scaffold is shared across
        arms, so this is the number that separates what the recurrence contributes from
        what the embedding, the norms and the head contribute.
    """
    total = 0
    for module in model.blocks:
        block = cast(Block, module)
        total += sum(p.numel() for p in block.mixer.parameters() if p.requires_grad)
    return total


def device_of(model: nn.Module) -> torch.device:
    """Where the model's first parameter lives.

    Args:
        model: The model.

    Returns:
        The device.

    Raises:
        ValueError: On a model with no parameters, which cannot be trained.
    """
    for param in model.parameters():
        return param.device
    raise ValueError("model has no parameters")
