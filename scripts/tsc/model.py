"""The scaffold every arm is scored on: the reference's, with the mixer swapped.

Not :class:`slinoss.SLinOSSStack`. The bars in :data:`scripts.tsc.protocol.REFERENCE` were
produced on the LinOSS scaffold, so that is what this axis holds fixed, and an arm's number is
comparable to them only if the scaffold around its mixer is the same one::

    Linear(input_dim -> hidden_dim)                       per timepoint
    blocks x [ BatchNorm -> mixer -> gelu -> dropout
               -> GLU -> dropout -> add the input ]
    mean over the sequence
    Linear(hidden_dim -> classes)
    softmax

Four details in that stack are the kind that silently move a number, and each is here on
purpose.

    the norm is a batch norm       over batch and time, no affine. Not a layer norm and not a
                                   pre-norm residual: the skip is taken before the norm and
                                   the norm sees the block input, so the residual stream is
                                   unnormalized.
    gelu is the tanh form          ``jax.nn.gelu`` defaults to ``approximate=True``, so the
                                   exact erf form is a different scaffold.
    the head emits probabilities   the reference applies softmax in the model and its loss is
                                   ``-sum(y log(p + 1e-8))``. That epsilon floors the loss at
                                   ``-log(1e-8)`` and bounds the gradient on a confidently
                                   wrong example, which a fused cross entropy does not.
                                   :func:`scripts.tsc.train.loss_on` reproduces it.
    the pool is unmasked           a missing value anywhere in an instance makes the whole
                                   prediction NaN. The reference does not mask and neither
                                   does this; :func:`scripts.tsc.train.check_finite` refuses
                                   such a corpus up front instead.

The mixer is swapped through :meth:`torch.nn.Module.add_module` for every arm including the
reference baseline, so no arm gets a scaffold built around it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from scripts.harness import MixerFactory

__all__ = [
    "GLU",
    "Block",
    "ModelConfig",
    "Scaffold",
    "build_model",
    "mixer_parameters",
    "parameter_count",
]


@dataclass(frozen=True)
class ModelConfig:
    """The scaffold's shape.

    Attributes:
        input_dim: Channels the encoder reads, the time channel included when the dataset's
            setting asks for it.
        hidden_dim: Stream width, the reference's ``H``. Also the mixer's width.
        classes: Head columns.
        blocks: Residual blocks.
        drop_rate: Dropout inside a block, applied twice.
        norm_eps: Batch norm epsilon.
        norm_momentum: Torch's momentum, the weight on the *batch* statistic. The reference's
            equinox default is 0.99 on the running statistic, which is this.

    Raises:
        ValueError: On a non-positive size, or a rate outside ``[0, 1)``.
    """

    input_dim: int
    hidden_dim: int
    classes: int
    blocks: int
    drop_rate: float = 0.05
    norm_eps: float = 1e-5
    norm_momentum: float = 0.01

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "classes", "blocks"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1), got {self.drop_rate}")


class GLU(nn.Module):
    """The reference's gated linear unit: ``w1(x) * sigmoid(w2(x))``.

    Two biased projections of the same width, which is one more parameter block than the
    common half-width gate. Transcribed rather than substituted because the parameter count
    per block is what :func:`mixer_parameters` is subtracted from.

    Args:
        width: Input and output width.

    Raises:
        ValueError: On a non-positive width.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(f"width must be positive, got {width}")
        self.value = nn.Linear(width, width, bias=True)
        self.gate = nn.Linear(width, width, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Gate.

        Args:
            x: ``(..., width)``.

        Returns:
            The same shape.
        """
        return self.value(x) * torch.sigmoid(self.gate(x))


class Block(nn.Module):
    """One residual block: norm, mix, activate, gate, add.

    The mixer is a placeholder at construction and :func:`build_model` replaces it. A block
    that built its own mixer would give the arm under test a different construction path from
    a baseline's.

    Args:
        config: The scaffold's shape.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(
            config.hidden_dim,
            eps=config.norm_eps,
            momentum=config.norm_momentum,
            affine=False,
        )
        self.mixer: nn.Module = nn.Identity()
        self.drop = nn.Dropout(config.drop_rate)
        self.glu = GLU(config.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Run the block.

        Args:
            x: ``(B,L,H)``.

        Returns:
            ``(B,L,H)``.
        """
        # The norm is channel-wise over batch and time, so the stream is transposed into
        # (B,H,L) for it and back. The skip is the block input, before the norm.
        normed = self.norm(x.transpose(1, 2)).transpose(1, 2)
        mixed = self.drop(nn.functional.gelu(self.mixer(normed), approximate="tanh"))
        return x + self.drop(self.glu(mixed))


class Scaffold(nn.Module):
    """Encoder, blocks, mean pool, head, softmax.

    Args:
        config: The scaffold's shape.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.blocks))
        self.head = nn.Linear(config.hidden_dim, config.classes, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch of series.

        Args:
            x: ``(B,L,input_dim)``.

        Returns:
            ``(B,classes)`` probabilities, summing to one along the last axis. Probabilities
            and not logits: see the module docstring.

        Raises:
            ValueError: On a last axis that is not ``input_dim``, or on a rank other than 3.
                A ``(B,L)`` batch would broadcast through the encoder and train.
        """
        if x.dim() != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"input is {tuple(x.shape)}, expected (B,L,{self.config.input_dim})"
            )
        hidden = self.encoder(x)
        for block in self.blocks:
            hidden = block(hidden)
        return torch.softmax(self.head(hidden.mean(dim=1)), dim=-1)


def build_model(
    config: ModelConfig,
    factories: Sequence[MixerFactory],
    *,
    max_length: int,
    device: torch.device | str | None = None,
) -> Scaffold:
    """Build the scaffold and swap in one mixer per block.

    Each mixer is moved to the destination as it is built and the whole model is moved again at
    the end, so a factory that allocates on the default device, or an operator that registers a
    buffer during the probe, cannot leave an arm half on the card.

    Call under a seeded generator: every parameter here is a torch default draw or a mixer's
    own initialization, so the seed has to be set before this returns for an arm to reproduce.

    Args:
        config: The scaffold's shape.
        factories: One factory per block, each ``(d_model, max_length) -> module`` mapping
            ``(B,L,H)`` to ``(B,L,H)``.
        max_length: Longest sequence the arm will run, passed to each factory.
        device: Destination.

    Returns:
        The model.

    Raises:
        ValueError: When the factory count is not the block count, or when a factory returns
            a module that does not preserve the stream's shape at a one-position probe. The
            probe is two forward passes on a ``(1,2,H)`` tensor and it catches the whole
            class of mixer wired at the wrong width, which otherwise surfaces as a matmul
            error thousands of steps into a lane.
    """
    if len(factories) != config.blocks:
        raise ValueError(f"{len(factories)} factories for {config.blocks} blocks")
    model = Scaffold(config).to(device=device)
    probe = torch.zeros(1, 2, config.hidden_dim, device=device)
    for block, factory in zip(model.blocks, factories, strict=True):
        # Moved before it is probed, not after the loop: this axis's own mixer is a CUDA-only
        # operator that refuses a host tensor, so a probe on the host cannot pass for it at all.
        mixer = factory(config.hidden_dim, max_length).to(device=device)
        with torch.no_grad():
            shape = tuple(mixer(probe).shape)
        if shape != probe.shape:
            raise ValueError(f"mixer maps {tuple(probe.shape)} to {shape}")
        # Through the module API, not by assignment: the block declares its placeholder and
        # what goes in is whatever the registry built.
        block.add_module("mixer", mixer)
    return model.to(device=device)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters.

    Args:
        model: The model.

    Returns:
        The count.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def mixer_parameters(model: Scaffold) -> int:
    """Trainable parameters inside the mixers only.

    Args:
        model: The model.

    Returns:
        The count over every block's mixer. The encoder, the norms, the GLUs and the head are
        shared across arms at one width, so this is what separates the recurrence's
        contribution from the scaffold's -- the split the ablations are about.
    """
    total = 0
    for module in model.blocks:
        block = cast("Block", module)
        total += sum(p.numel() for p in block.mixer.parameters() if p.requires_grad)
    return total
