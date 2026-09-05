"""The mixers a language-modelling arm can name.

    name      what it is                                    package
    slinoss   the tree's mixer                              this tree
    gpt       causal multi-head attention with rotary       this tree
    conv      causal depthwise convolution, gated           this tree
    mamba2    Mamba-2                                       mamba_ssm
    mamba3    Mamba-3                                       flash-linear-attention
    gdn2      Gated DeltaNet 2                              flash-linear-attention

The three in-tree entries import nothing optional. The three baselines import inside their
build, so this module is importable, and every arm's settings are readable, on a host where
neither package is installed; the import error arrives when an arm actually asks for one.

``gpt`` is the attention control and the hybrid's other half. It is not a ceiling: a
fixed-depth transformer reads the whole prefix, so it wins wherever the answer is in the
context and cannot carry a state past it. ``conv`` is the star-free floor and exists to say
what a task does not need.

Parity with the baselines follows the standing rule: one B/C group, matched state width, and
the same optimizer for every arm. The state widths default to a matched 96 -- ``d_state`` is
``3N`` here so 96 is ``N = 32`` -- and are overridable per arm through ``--set``.

Baselines whose package is not installed reach the registry the same way anything else does:
a module of its own that calls :meth:`scripts.harness.Registry.register`, imported by
``--mixer-module``. Nothing is guessed by fallback. A name that resolves to a class this
harness has not verified the calling convention of would report a number for a program nobody
chose, so an absent package raises.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from scripts.harness import CausalAttention, CausalConv, MixerEntry, Registry

__all__ = ["REGISTRY", "Unwrap"]

REGISTRY = Registry("lm")
"""The language-modelling axis's mixers."""


class Unwrap(nn.Module):
    """Take the hidden state out of a mixer that returns a tuple.

    The linear-attention layers return ``(output, attentions, cache)``, and the scaffold's
    block wants one tensor. Wrapping rather than adapting per baseline keeps the parameter
    leaf names intact, which is what :func:`scripts.lm.groups.classify` routes on.

    Args:
        inner: The mixer.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        """Run the inner mixer and keep its first output.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``.
        """
        out = self.inner(x)
        return out[0] if isinstance(out, tuple) else out


def _build_slinoss(d_model: int, max_length: int, **settings: Any) -> nn.Module:
    """The tree's mixer.

    Args:
        d_model: Stream width.
        max_length: Declared model context. This is constructor metadata and is
            never inferred from an observed batch width.
        **settings: :class:`slinoss.SLinOSSConfig` mixer fields.

    Returns:
        A :class:`slinoss.SLinOSSMixer`.
    """
    from slinoss import SLinOSSConfig, SLinOSSMixer

    return SLinOSSMixer(
        SLinOSSConfig(d_model=d_model, context_length=max_length, **settings)
    )


def _build_mamba2(d_model: int, **settings: Any) -> nn.Module:
    """Mamba-2, from ``mamba_ssm``.

    Args:
        d_model: Stream width.
        **settings: ``d_state``, ``d_conv``, ``expand``, ``headdim``, ``ngroups``.

    Returns:
        The layer. Its forward already returns one tensor.

    Raises:
        ModuleNotFoundError: When ``mamba_ssm`` is not installed.
    """
    from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore[import-not-found]

    return Mamba2(d_model=d_model, **settings)


def _build_mamba3(d_model: int, **settings: Any) -> nn.Module:
    """Mamba-3, from ``flash-linear-attention``.

    Args:
        d_model: Stream width.
        **settings: ``state_size``, ``expand``, ``head_dim``, ``n_groups``, ``chunk_size``.

    Returns:
        The layer, wrapped in :class:`Unwrap`. CUDA only, from its own guard.

    Raises:
        ModuleNotFoundError: When ``fla`` is not installed.
    """
    from fla.layers.mamba3 import Mamba3  # type: ignore[import-not-found]

    return Unwrap(Mamba3(hidden_size=d_model, **settings))


def _build_gdn2(d_model: int, **settings: Any) -> nn.Module:
    """Gated DeltaNet 2, from ``flash-linear-attention``.

    Args:
        d_model: Stream width.
        **settings: ``expand_v``, ``head_dim``, ``num_heads``, ``conv_size``,
            ``use_short_conv``, ``allow_neg_eigval``.

    Returns:
        The layer, wrapped in :class:`Unwrap`.

    Raises:
        ModuleNotFoundError: When ``fla`` is not installed.
    """
    from fla.layers.gdn2 import GatedDeltaNet2  # type: ignore[import-not-found]

    return Unwrap(GatedDeltaNet2(hidden_size=d_model, **settings))


def _slinoss_defaults() -> dict[str, Any]:
    """The mixer's own defaults, read off its config rather than restated.

    ``d_state`` has none there and is named here: it is ``3N`` with ``N`` a multiple of 16,
    and 96 is ``N = 32``, the state width the baselines default to.

    Returns:
        Every mixer field an arm may move.
    """
    from slinoss import SLinOSSConfig

    return {
        "d_state": 96,
        "expand": SLinOSSConfig.expand,
        "d_head": SLinOSSConfig.d_head,
        "n_groups": SLinOSSConfig.n_groups,
        "chunk_size": SLinOSSConfig.chunk_size,
        "d_conv": SLinOSSConfig.d_conv,
        "key_conv": SLinOSSConfig.key_conv,
        "forcing_init": SLinOSSConfig.forcing_init,
        "init_span": SLinOSSConfig.init_span,
        "init_period_context_scale": SLinOSSConfig.init_period_context_scale,
        "init_decay_context_scale": SLinOSSConfig.init_decay_context_scale,
        "w_max": SLinOSSConfig.w_max,
        "bias": SLinOSSConfig.bias,
        "conv_bias": SLinOSSConfig.conv_bias,
    }


def _register_builtins() -> None:
    """Register every mixer this module defines.

    Called at import. Only the slinoss entry reads a default off another module, and
    :class:`slinoss.SLinOSSConfig` imports torch and nothing optional.
    """
    REGISTRY.register(
        "slinoss", MixerEntry(_build_slinoss, "required", _slinoss_defaults())
    )
    REGISTRY.register(
        "gpt", MixerEntry(CausalAttention, "required", {"n_heads": 8, "rotary": True})
    )
    REGISTRY.register(
        "conv", MixerEntry(CausalConv, "unused", {"d_conv": 4, "expand": 2.0})
    )
    REGISTRY.register(
        "mamba2",
        MixerEntry(
            _build_mamba2,
            "unused",
            {
                "d_state": 96,
                "d_conv": 4,
                "expand": 2,
                "headdim": 64,
                "ngroups": 1,
            },
        ),
    )
    REGISTRY.register(
        "mamba3",
        MixerEntry(
            _build_mamba3,
            "unused",
            {
                "state_size": 96,
                "expand": 2,
                "head_dim": 64,
                "n_groups": 1,
                "chunk_size": 64,
            },
        ),
    )
    REGISTRY.register(
        "gdn2",
        MixerEntry(
            _build_gdn2,
            "unused",
            {
                "expand_v": 1.0,
                "head_dim": 64,
                "num_heads": 8,
                "conv_size": 4,
                "use_short_conv": True,
                "allow_neg_eigval": False,
            },
        ),
    )


_register_builtins()
