"""The backbone MQAR is scored on.

Two pre-norm blocks over a token embedding, then a tied linear head. The block is the
usual pre-norm form with one twist that must be reproduced exactly: the residual stream
is carried *beside* the hidden state rather than added into it, so a layer norm never sees
its own output added back. Written out, one block is::

    dropped  = dropout(hidden)
    residual = dropped + residual          # or dropped, at layer 0
    hidden   = sequence_mixer(norm1(residual))
    dropped  = dropout(hidden)
    residual = dropped + residual
    hidden   = state_mixer(norm2(residual))

and the backbone closes with ``ln_f(drop_f(hidden) + residual)``. Layer 0's first dropout
uses ``embed_dropout``; every other dropout uses ``resid_dropout``, which is 0 in every
published MQAR config.

The published MQAR configs set the state mixer to identity, so the block is a bare
sequence mixer with two norms. That is deliberate on their part: it isolates the mixer,
which is the whole point of the benchmark. ``state_mixer="mlp"`` is available for the
scaffold ablation and matches upstream's own default block.

Initialization follows upstream: normal at ``init_std`` for every linear weight and every
embedding, zeros for every bias, then a second draw at ``init_std / sqrt(2 * n_layers)``
for any parameter whose name ends in ``out_proj.weight``, ``fc2.weight`` or
``output_linear.0.weight`` -- the GPT-2 residual-depth rescale. The word embedding is the
one exception: ``embedding_init_type`` draws it at construction and marks it, and the
blanket pass then leaves it alone. That is upstream's own mechanism, and one of the two
current MQAR sweeps runs it at ``spherical``, so it is a setting rather than a default.

Three divergences from upstream, all deliberate:

1. Upstream has no opt-out for linear layers, so its init overwrites whatever a mixer set
   up for itself. :func:`protect` marks a subtree as owning its own initialization and
   this walk skips it. A mixer with a nontrivial parameterization is otherwise measured
   with that parameterization erased.
2. Upstream applies the init twice (once in the backbone, once in the language model). The
   end state is the same distribution either way; only which draw lands where differs.
   Applied once here.
3. Upstream hardcodes ``device='cuda'`` in the embedding and builds position ids on that
   device, so it cannot run on CPU. Position ids follow the input's device here.

Upstream config fields with no counterpart here, each because no MQAR config selects it:
``multiplier`` (asserted 1 by the discrete-input model), ``pad_vocab_size_multiple``,
``word_embed_proj_dim``, ``drop_path`` and its ``StochasticDepth``, ``block_type``'s two
Mamba branches (whose init pass differs), the GLU state mixer, and the ``mse`` and
``ce_embed`` losses of the continuous-input model. Selecting one upstream would change a
number, so each is named here rather than silently absent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MixerFactory = Callable[[int, int, int], nn.Module]
"""``(d_model, layer_idx, max_length) -> module`` taking and returning ``(B, T, d_model)``."""

RESCALED_SUFFIXES = ("out_proj.weight", "fc2.weight", "output_linear.0.weight")
"""Parameter-name suffixes that take the residual-depth rescale."""

STATE_MIXERS = ("identity", "mlp")
"""Admissible state mixers. Upstream's GLU is not ported; no MQAR config selects it."""

EMBEDDING_INITS = ("default", "spherical", "normal")
"""Admissible word-embedding draws.

``default`` leaves the table to :func:`initialize`, which is normal at ``init_std``.
``spherical`` puts every row on the unit sphere, so a component's scale is
``1 / sqrt(d_model)``; ``normal`` draws at unit variance. Both mark the table, so the
blanket pass cannot undo them.
"""


@dataclass(frozen=True)
class ModelConfig:
    """Backbone shape.

    Attributes:
        vocab_size: Token count; must match the pool's.
        d_model: Width.
        n_layers: Block count. 2 in every published MQAR config.
        max_length: Longest sequence the mixers are built for.
        max_position_embeddings: Learned absolute position table size. 0 means no
            position embedding at all, which is what every recurrent mixer is measured
            under; attention is measured with it set to ``max_length``.
        learnable_word_embeddings: Train the word embedding, and tie the head to it.
        embedding_init_type: One of :data:`EMBEDDING_INITS`. The published figure-2 sweep
            runs the default; the modern reproduction with the filler off runs
            ``spherical``.
        state_mixer: One of :data:`STATE_MIXERS`.
        hidden_mult: Width multiple of the ``mlp`` state mixer.
        embed_dropout: Dropout on the embedding, layer 0 only.
        resid_dropout: Dropout everywhere else.
        layer_norm_epsilon: Epsilon of the final norm. Upstream leaves the per-block
            norms at torch's default 1e-5 and only passes this to the final norm; that
            asymmetry is reproduced.
        init_std: Standard deviation of the initial normal draws.
    """

    vocab_size: int = 8192
    d_model: int = 128
    n_layers: int = 2
    max_length: int = 64
    max_position_embeddings: int = 0
    learnable_word_embeddings: bool = True
    embedding_init_type: str = "default"
    state_mixer: str = "identity"
    hidden_mult: int = 4
    embed_dropout: float = 0.1
    resid_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_length < 1:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.state_mixer not in STATE_MIXERS:
            raise ValueError(
                f"state_mixer must be one of {STATE_MIXERS}, got {self.state_mixer!r}"
            )
        if self.embedding_init_type not in EMBEDDING_INITS:
            raise ValueError(
                f"embedding_init_type must be one of {EMBEDDING_INITS}, got "
                f"{self.embedding_init_type!r}"
            )
        if 0 < self.max_position_embeddings < self.max_length:
            raise ValueError(
                f"max_position_embeddings {self.max_position_embeddings} is shorter than "
                f"max_length {self.max_length}; positions would index out of range"
            )


class MLP(nn.Module):
    """``fc2(gelu(fc1(x)))`` at width ``d_model * hidden_mult``, both linears biased."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d_model)`` in and out."""
        return self.fc2(F.gelu(self.fc1(x)))


class Embeddings(nn.Module):
    """Word embedding, plus a learned absolute position embedding when enabled.

    A non-default word draw is taken here, between the two tables, because that is where
    upstream takes it and it consumes the global generator.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        _draw_embedding(self.word_embeddings, config.embedding_init_type)
        if not config.learnable_word_embeddings:
            self.word_embeddings.weight.requires_grad = False
        self.max_position_embeddings = config.max_position_embeddings
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.d_model)
            if config.max_position_embeddings > 0
            else None
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """``(B, T)`` int64 in, ``(B, T, d_model)`` out."""
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            positions = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
            embeddings = embeddings + self.position_embeddings(positions)
        return embeddings


class Block(nn.Module):
    """One pre-norm block carrying the residual stream beside the hidden state."""

    def __init__(
        self, config: ModelConfig, layer_idx: int, mixer: MixerFactory
    ) -> None:
        super().__init__()
        self.sequence_mixer = mixer(config.d_model, layer_idx, config.max_length)
        self.state_mixer = (
            MLP(config.d_model, config.hidden_mult)
            if config.state_mixer == "mlp"
            else nn.Identity()
        )
        first = config.embed_dropout if layer_idx == 0 else config.resid_dropout
        self.dropout1 = nn.Dropout(first)
        self.dropout2 = nn.Dropout(config.resid_dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, hidden: Tensor, residual: Tensor | None) -> tuple[Tensor, Tensor]:
        """Advance the hidden state and the residual stream by one block.

        Args:
            hidden: ``(B, T, d_model)``. At layer 0 this is the embedding.
            residual: ``(B, T, d_model)``, or None at layer 0.

        Returns:
            ``(hidden, residual)``, both ``(B, T, d_model)``.
        """
        dropped = self.dropout1(hidden)
        # A separate name for the stream: a module call is untyped, so reassigning the
        # optional parameter would widen it back to Tensor | None at every use.
        stream: Tensor = dropped if residual is None else dropped + residual
        hidden = self.sequence_mixer(self.norm1(stream.to(self.norm1.weight.dtype)))
        dropped = self.dropout2(hidden)
        stream = dropped + stream
        hidden = self.state_mixer(self.norm2(stream.to(self.norm2.weight.dtype)))
        return hidden, stream


class LanguageModel(nn.Module):
    """The scored model: embeddings, blocks, final norm, tied head.

    Attributes:
        config: The shape it was built from.
    """

    def __init__(self, config: ModelConfig, mixer: MixerFactory) -> None:
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            Block(config, index, mixer) for index in range(config.n_layers)
        )
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        initialize(self, config.n_layers, config.init_std)
        if config.learnable_word_embeddings:
            # Tying after the init pass is upstream's order: the head's own draw is
            # discarded and the embedding's draw is what both use.
            self.lm_head.weight = self.embeddings.word_embeddings.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """``(B, T)`` int64 in, ``(B, T, vocab_size)`` logits out."""
        hidden = self.embeddings(input_ids)
        residual: Tensor | None = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual)
        assert residual is not None  # n_layers >= 1 is enforced by the config
        stream: Tensor = self.drop_f(hidden) + residual
        return self.lm_head(self.ln_f(stream.to(self.ln_f.weight.dtype)))


def protect(module: nn.Module) -> nn.Module:
    """Mark a subtree as owning its own initialization.

    :func:`initialize` skips a marked module and everything under it. Without this the
    backbone's blanket normal draw overwrites a mixer's parameterization, which is the
    difference between measuring a mixer and measuring its shape.

    Args:
        module: The subtree root.

    Returns:
        ``module``, for use inline in a factory.
    """
    # object.__setattr__ rather than plain assignment: nn.Module.__setattr__ is typed for
    # tensors and modules, and routing a bool through it is a type error for no gain.
    object.__setattr__(module, "_no_reinit", True)
    return module


def _draw_embedding(embedding: nn.Embedding, init_type: str) -> None:
    """Draw a word table at ``init_type`` and mark it against the blanket pass.

    ``default`` returns without touching the table or the generator, leaving it to
    :func:`initialize`. Upstream's padding-index branch is unreachable -- no MQAR config
    passes a padding index -- so token 0 takes the same draw as every other row.

    Args:
        embedding: The word table.
        init_type: One of :data:`EMBEDDING_INITS`.
    """
    if init_type == "default":
        return
    with torch.no_grad():
        if init_type == "spherical":
            vectors = torch.randn_like(embedding.weight)
            embedding.weight.copy_(
                vectors / torch.norm(vectors, p=2, dim=1, keepdim=True)
            )
        else:
            nn.init.normal_(embedding.weight, mean=0.0, std=1.0)
    protect(embedding)


def initialize(model: nn.Module, n_layers: int, init_std: float) -> None:
    """Apply upstream's initialization to every unprotected subtree.

    Args:
        model: Root.
        n_layers: Block count, for the residual-depth rescale.
        init_std: Standard deviation of the first draw.
    """
    for _, module in _unprotected(model):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=init_std)
    rescaled = init_std / math.sqrt(2 * n_layers)
    for prefix, module in _unprotected(model):
        for name, parameter in module.named_parameters(recurse=False):
            if (prefix + name).endswith(RESCALED_SUFFIXES):
                nn.init.normal_(parameter, mean=0.0, std=rescaled)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters, counting a tied weight once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _unprotected(
    module: nn.Module, prefix: str = ""
) -> Iterator[tuple[str, nn.Module]]:
    """Walk the tree, pruning any subtree marked by :func:`protect`."""
    if getattr(module, "_no_reinit", False):
        return
    yield prefix, module
    for name, child in module.named_children():
        yield from _unprotected(child, f"{prefix}{name}.")
