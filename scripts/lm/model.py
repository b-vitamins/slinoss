"""The scaffold every arm is scored on: :class:`slinoss.SLinOSSStack` with the mixer swapped.

The tree already ships the whole language model -- embedding, ``n_layers`` pre-norm blocks
with a SwiGLU FFN, one fused final norm, a head padded to a tensor-core multiple -- so this
module writes no scaffold. It replaces one attribute:

    stack.blocks[i].mixer = factory(d_model, max_length)

:meth:`slinoss.SLinOSSBlock.forward` calls ``self.mixer(normed)`` on the whole-sequence path,
so any module mapping ``(B,T,d_model)`` to ``(B,T,d_model)`` slots in. Every arm therefore
shares the scaffold bit for bit: the same fused residual norms, the same float32 stream
between blocks, the same FFN orientation, the same padded head. A difference between two arms
is the mixer, and the parameter-count difference between them is the mixer's too, which is
what makes :mod:`scripts.lm.sizing` well posed.

The swap is uniform. The slinoss arm also goes through it, building its mixer from the
registry like any baseline, rather than keeping the one the block constructed. A special case
for the arm under test is how a harness stops being a comparison; the cost is one discarded
mixer's worth of transient host memory per layer.

A per-layer list rather than one factory, because the hybrid arm is a per-layer choice: eleven
layers of one mixer and a twelfth of another is a list and needs no second code path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from scripts.harness import MixerFactory
from slinoss import SLinOSSBlock, SLinOSSConfig, SLinOSSMixer, SLinOSSStack

__all__ = [
    "build_model",
    "layer_factories",
    "mixer_parameters",
    "non_embedding_parameters",
    "parameter_count",
    "scaffold_config",
]


def scaffold_config(
    *,
    d_model: int,
    n_layers: int,
    vocab_size: int,
    ffn_ratio: float = 4.0,
    d_state: int = 48,
    d_head: int = 64,
    bias: bool = False,
    norm_eps: float = 1e-5,
) -> SLinOSSConfig:
    """The config the scaffold reads.

    Only ``d_model``, ``n_layers``, ``vocab_size``, ``ffn_ratio``, ``bias`` and
    ``norm_eps`` reach the scaffold. The mixer fields are here because
    :class:`slinoss.SLinOSSConfig` validates them all at construction and the block needs a
    complete config to build the mixer it is about to have replaced; their values do not
    reach any arm, so they sit at the narrowest legal setting.

    Args:
        d_model: Residual width.
        n_layers: Blocks.
        vocab_size: Tokens the embedding gathers and classes the head carries meaning on.
        ffn_ratio: FFN hidden width as a multiple of ``d_model``.
        d_state: Unused by the scaffold; the narrowest legal width by default.
        d_head: Unused by the scaffold, but it must divide ``d_inner``.
        bias: Bias on the FFN and the head.
        norm_eps: RMS norm epsilon.

    Returns:
        The config.

    Raises:
        ValueError: From :class:`slinoss.SLinOSSConfig`.
    """
    return SLinOSSConfig(
        d_model=d_model,
        d_state=d_state,
        d_head=d_head,
        n_layers=n_layers,
        ffn_ratio=ffn_ratio,
        bias=bias,
        norm_eps=norm_eps,
        vocab_size=vocab_size,
    )


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
    config: SLinOSSConfig,
    factories: Sequence[MixerFactory],
    *,
    max_length: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> SLinOSSStack:
    """Build the scaffold and swap in one mixer per layer.

    Construction is on the host and the move is one call at the end, so no arm is ever half
    on the device: a factory that allocates on the default device would otherwise put the
    mixer somewhere the scaffold is not.

    Call under a seeded generator. Every parameter here is a torch default draw or a
    mixer's own initialization, so the seed has to be set first for an arm to reproduce.

    Args:
        config: Scaffold shape, from :func:`scaffold_config`.
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
    stack = SLinOSSStack(config, device="cpu", dtype=torch.float32)
    for module, factory in zip(stack.blocks, factories, strict=True):
        block = cast("SLinOSSBlock", module)
        mixer = factory(config.d_model, max_length)
        if isinstance(mixer, SLinOSSMixer) and mixer.config.d_model != config.d_model:
            raise ValueError(
                f"mixer built at d_model {mixer.config.d_model} and the scaffold is "
                f"{config.d_model}"
            )
        # Through the module API, not by assignment: the block declares its own mixer as a
        # SLinOSSMixer, and what goes in here is any mixer the registry builds.
        block.add_module("mixer", mixer)
    return cast("SLinOSSStack", stack.to(device=device, dtype=dtype))


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters.

    Args:
        model: The model.

    Returns:
        The count. The head's padding columns are parameters and are counted: they are
        allocated, they are optimized, and they hold no output.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def non_embedding_parameters(model: SLinOSSStack) -> int:
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
        if module is not None:
            total -= sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total


def mixer_parameters(model: SLinOSSStack) -> int:
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
        block = cast("SLinOSSBlock", module)
        total += sum(p.numel() for p in block.mixer.parameters() if p.requires_grad)
    return total
