"""The mixers a time series classification arm can name.

    name          what it is                                       package
    linoss_im     the reference recurrence, implicit scheme         this tree
    linoss_imex   the reference recurrence, IMEX scheme             this tree
    slinoss       the tree's mixer                                  this tree
    attention     causal multi-head attention with rotary           this tree
    conv          causal depthwise convolution, gated               this tree
    mamba2        Mamba-2                                           mamba_ssm
    mamba3        Mamba-3                                           flash-linear-attention
    gdn2          Gated DeltaNet 2                                  flash-linear-attention

The three in-tree recurrences import nothing optional. The baselines import inside their build,
so this module is importable and every arm's settings are readable on a host where none of
those packages exist; the import error arrives when an arm asks for one.

All five in-tree entries are causal, including the two controls. The task is not: the head
mean-pools the whole sequence, so a bidirectional mixer would read the suffix and would not be
comparable to the reference or to any arm here. Causality is the axis's fixed choice.

The widths this axis runs at are small -- the published configs use ``hidden_dim`` 16, 64 and
128 -- so head-based mixers default to head counts that divide all three. ``slinoss`` defaults
to ``d_head`` 16 rather than 64 for the same reason: at ``hidden_dim`` 16 the inner width is 32
and a 64-row head does not fit.

:func:`paper_overrides` is how the published per-dataset state width reaches the reference
baseline. It applies to the two ``linoss`` names only. Mapping the reference's oscillator count
onto another mixer's state width is a modelling decision and not the protocol's, so every other
mixer takes its width from ``--set`` and from nowhere else.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from scripts.harness import CausalAttention, CausalConv, MixerEntry, Registry
from scripts.tsc.linoss import LinOSSRecurrence
from scripts.tsc.protocol import Setting

__all__ = ["REGISTRY", "Unwrap", "paper_overrides"]

REGISTRY = Registry("tsc")
"""The time series classification axis's mixers."""

_LINOSS_NAMES = ("linoss_im", "linoss_imex")
"""The names whose state width the published configs specify."""


class Unwrap(nn.Module):
    """Take the hidden state out of a mixer that returns a tuple.

    The linear-attention baselines return ``(output, attentions, cache)`` and the scaffold's
    block wants one tensor. Wrapping rather than adapting per baseline keeps the parameter leaf
    names intact.

    Args:
        inner: The mixer.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: Any) -> Any:
        """Run the inner mixer and keep its first output.

        Args:
            x: ``(B,L,d_model)``.

        Returns:
            ``(B,L,d_model)``.
        """
        out = self.inner(x)
        return out[0] if isinstance(out, tuple) else out


def _build_linoss_im(d_model: int, **settings: Any) -> nn.Module:
    """The reference recurrence under the implicit scheme.

    Args:
        d_model: Stream width.
        **settings: ``ssm_dim``.

    Returns:
        A :class:`scripts.tsc.linoss.LinOSSRecurrence`.
    """
    return LinOSSRecurrence(d_model, discretization="IM", **settings)


def _build_linoss_imex(d_model: int, **settings: Any) -> nn.Module:
    """The reference recurrence under the IMEX scheme.

    Args:
        d_model: Stream width.
        **settings: ``ssm_dim``.

    Returns:
        A :class:`scripts.tsc.linoss.LinOSSRecurrence`.
    """
    return LinOSSRecurrence(d_model, discretization="IMEX", **settings)


def _build_slinoss(d_model: int, **settings: Any) -> nn.Module:
    """The tree's mixer.

    Args:
        d_model: Stream width.
        **settings: :class:`slinoss.SLinOSSConfig` mixer fields.

    Returns:
        A :class:`slinoss.SLinOSSMixer`.
    """
    from slinoss import SLinOSSConfig, SLinOSSMixer

    return SLinOSSMixer(SLinOSSConfig(d_model=d_model, **settings))


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

    ``d_state`` has no default there and is named here: it is ``3N`` with ``N`` a multiple of
    16, so 96 is ``N = 32``. ``d_head`` is narrowed from the config's 64 because this axis runs
    at ``hidden_dim`` 16.

    Returns:
        Every mixer field an arm may move.
    """
    from slinoss import SLinOSSConfig

    return {
        "d_state": 96,
        "expand": SLinOSSConfig.expand,
        "d_head": 16,
        "n_groups": SLinOSSConfig.n_groups,
        "chunk_size": SLinOSSConfig.chunk_size,
        "d_conv": SLinOSSConfig.d_conv,
        "key_conv": SLinOSSConfig.key_conv,
        "init_span": SLinOSSConfig.init_span,
        "w_max": SLinOSSConfig.w_max,
        "bias": SLinOSSConfig.bias,
        "conv_bias": SLinOSSConfig.conv_bias,
    }


def paper_overrides(mixer: str, setting: Setting) -> list[str]:
    """The mixer settings the published per-dataset config fixes.

    Only the reference recurrence has any: its ``ssm_dim`` is the published ``ssm_dim``, and
    the discretization is already in the mixer's name. Every other mixer returns nothing,
    because there is no protocol-sanctioned translation from an oscillator count to another
    mixer's state width and inventing one silently would make a baseline's number a harness
    choice.

    Args:
        mixer: Registry name.
        setting: The dataset's published setting.

    Returns:
        ``key=value`` strings for :meth:`scripts.harness.Registry.resolve`, possibly empty.
    """
    if mixer in _LINOSS_NAMES:
        return [f"ssm_dim={setting.ssm_dim}"]
    return []


def _register_builtins() -> None:
    """Register every mixer this module defines.

    Called at import. Only the slinoss entry reads defaults off another module, and
    :class:`slinoss.SLinOSSConfig` imports torch and nothing optional.
    """
    REGISTRY.register(
        "linoss_im", MixerEntry(_build_linoss_im, "unused", {"ssm_dim": 64})
    )
    REGISTRY.register(
        "linoss_imex", MixerEntry(_build_linoss_imex, "unused", {"ssm_dim": 64})
    )
    REGISTRY.register(
        "slinoss", MixerEntry(_build_slinoss, "unused", _slinoss_defaults())
    )
    REGISTRY.register(
        "attention",
        MixerEntry(CausalAttention, "required", {"n_heads": 4, "rotary": True}),
    )
    REGISTRY.register(
        "conv", MixerEntry(CausalConv, "unused", {"d_conv": 4, "expand": 2.0})
    )
    REGISTRY.register(
        "mamba2",
        MixerEntry(
            _build_mamba2,
            "unused",
            {"d_state": 96, "d_conv": 4, "expand": 2, "headdim": 16, "ngroups": 1},
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
                "head_dim": 16,
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
                "head_dim": 16,
                "num_heads": 4,
                "conv_size": 4,
                "use_short_conv": True,
                "allow_neg_eigval": False,
            },
        ),
    )


_register_builtins()
