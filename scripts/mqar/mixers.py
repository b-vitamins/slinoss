"""Name-keyed sequence-mixer registry.

A mixer is a module taking and returning ``(B, T, d_model)``. Three are registered here:
``slinoss``, and the two controls the published protocol is built out of, ``attention``
and ``conv``. Anything else slots in from outside the tree: call :func:`register` in your
own module and point ``--mixer-module`` at it.

The registry keys per-layer, not per-model. ``resolve(["conv", "slinoss"])`` builds layer
0 as the conv and layer 1 as slinoss, cycling if the model is deeper than the list. That
is upstream's ``Hybrid``, and it is not a detail: every architecture in zoology's current
MQAR config is a short conv followed by the architecture under test, so a slinoss number
compared against those is only comparable if it is built the same way.

Settings are typed by their default. ``--set d_state=192`` when one mixer is named,
``--set slinoss.d_state=192`` when several are. An unknown key is an error, not a
silently ignored extra.

Init ownership: ``attention`` and ``conv`` take the backbone's blanket normal draw, which
is what upstream measures them under. ``slinoss`` is wrapped in
:func:`scripts.mqar.model.protect`, so it keeps its own parameterization -- without that
the backbone would overwrite the horizon grid with a normal draw and the measurement
would be of a shape rather than of a mixer.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NamedTuple, cast

import torch
from torch import Tensor, nn

from scripts.mqar.model import MixerFactory, protect

Setting = bool | int | float | str
"""A mixer setting. Its type is fixed by the registered default."""

BuildFn = Callable[..., nn.Module]
"""Constructor whose context arguments are governed by :class:`MixerEntry`."""


class Mixer(NamedTuple):
    """A resolved per-layer mixer cycle.

    Attributes:
        name: The cycle, joined with ``+``. Goes into the record verbatim.
        factory: What :class:`scripts.mqar.model.LanguageModel` calls per layer.
        settings: Merged settings per entry name, for the record.
        contracts: Declared context-consumption policy per entry name.
        constructions: Effective configuration and context of every built layer.
    """

    name: str
    factory: MixerFactory
    settings: dict[str, dict[str, Setting]]
    contracts: dict[str, dict[str, str]]
    constructions: list[dict[str, Any]]


@dataclass(frozen=True)
class MixerEntry:
    """One registered mixer.

    Attributes:
        build: Constructor.
        layer_index_policy: Whether ``build`` consumes the layer index.
        max_length_policy: Whether ``build`` consumes the maximum sequence length.
        defaults: Every admissible setting, with its default. The types here are what
            command-line overrides are coerced to.
    """

    build: BuildFn
    layer_index_policy: Literal["required", "unused"]
    max_length_policy: Literal["required", "unused"]
    defaults: dict[str, Setting]

    def __post_init__(self) -> None:
        for name, policy in (
            ("layer_index_policy", self.layer_index_policy),
            ("max_length_policy", self.max_length_policy),
        ):
            if policy not in {"required", "unused"}:
                raise ValueError(
                    f"{name} must be 'required' or 'unused', got {policy!r}"
                )


REGISTRY: dict[str, MixerEntry] = {}
"""Name to entry. Populated at import with the three builtins."""


def register(
    name: str,
    build: BuildFn,
    defaults: dict[str, Setting],
    *,
    layer_index_policy: Literal["required", "unused"],
    max_length_policy: Literal["required", "unused"],
) -> None:
    """Add a mixer.

    Args:
        name: Registry key. Must be free, and must not contain ``.`` or ``+``, which the
            override and cycle syntaxes reserve.
        build: Constructor.
        defaults: Admissible settings and their defaults.
        layer_index_policy: Whether the constructor consumes its layer index.
        max_length_policy: Whether the constructor consumes its maximum length.

    Raises:
        KeyError: If the name is taken.
        ValueError: If the name uses a reserved character.
    """
    if name in REGISTRY:
        raise KeyError(f"mixer {name} is already registered")
    if "." in name or "+" in name:
        raise ValueError(f"mixer name {name!r} must not contain '.' or '+'")
    REGISTRY[name] = MixerEntry(
        build=build,
        layer_index_policy=layer_index_policy,
        max_length_policy=max_length_policy,
        defaults=dict(defaults),
    )


def resolve(names: Sequence[str], overrides: Iterable[str] = ()) -> Mixer:
    """Build the per-layer factory for a mixer cycle.

    Args:
        names: One name per layer position, cycled if the model is deeper.
        overrides: ``key=value`` when ``names`` holds one distinct entry, or
            ``entry.key=value`` in general.

    Returns:
        A :class:`Mixer`.

    Raises:
        KeyError: On an unregistered name or an unknown setting.
        ValueError: On an empty cycle, or an unscoped override under several entries.
    """
    if not names:
        raise ValueError("a mixer cycle needs at least one name")
    for name in names:
        if name not in REGISTRY:
            raise KeyError(f"no mixer {name}; registered: {sorted(REGISTRY)}")
    scoped = _scope_overrides(names, overrides)
    settings = {name: settings_from(name, scoped[name]) for name in scoped}
    cycle = tuple(names)
    contracts = {
        name: {
            "layer_index_policy": REGISTRY[name].layer_index_policy,
            "max_length_policy": REGISTRY[name].max_length_policy,
        }
        for name in dict.fromkeys(cycle)
    }
    constructions: list[dict[str, Any]] = []

    def factory(d_model: int, layer_idx: int, max_length: int) -> nn.Module:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        if max_length < 1:
            raise ValueError(f"max_length must be positive, got {max_length}")
        name = cycle[layer_idx % len(cycle)]
        entry = REGISTRY[name]
        context: list[int] = []
        if entry.layer_index_policy == "required":
            context.append(layer_idx)
        if entry.max_length_policy == "required":
            context.append(max_length)
        module = entry.build(d_model, *context, **settings[name])
        config = getattr(module, "config", None)
        effective = (
            asdict(cast(Any, config))
            if is_dataclass(config)
            else {"d_model": d_model, **settings[name]}
        )
        constructions.append(
            {
                "entry": name,
                "module": f"{type(module).__module__}.{type(module).__qualname__}",
                "effective_config": effective,
                "context": {
                    "layer_index_supplied": layer_idx,
                    "layer_index_policy": entry.layer_index_policy,
                    "layer_index_consumed": (
                        layer_idx if entry.layer_index_policy == "required" else None
                    ),
                    "max_length_supplied": max_length,
                    "max_length_policy": entry.max_length_policy,
                    "max_length_consumed": (
                        max_length if entry.max_length_policy == "required" else None
                    ),
                },
                "initialization": (
                    "mixer_constructor; protected from scaffold reinitialization"
                    if getattr(module, "_no_reinit", False)
                    else "scaffold blanket normal draw after construction"
                ),
            }
        )
        return module

    return Mixer(
        name="+".join(cycle),
        factory=factory,
        settings=settings,
        contracts=contracts,
        constructions=constructions,
    )


def settings_from(name: str, overrides: Iterable[str]) -> dict[str, Setting]:
    """Merge ``key=value`` overrides into one entry's defaults.

    Args:
        name: Registered mixer name.
        overrides: Unscoped ``key=value`` strings.

    Returns:
        Every setting the entry declares, overrides applied.

    Raises:
        KeyError: On an unregistered name or an undeclared key.
        ValueError: On a malformed override or an uncoercible value.
    """
    if name not in REGISTRY:
        raise KeyError(f"no mixer {name}; registered: {sorted(REGISTRY)}")
    merged = dict(REGISTRY[name].defaults)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, text = item.split("=", 1)
        if key not in merged:
            raise KeyError(
                f"mixer {name} has no setting {key}; declared: {sorted(merged)}"
            )
        merged[key] = _coerce(key, merged[key], text)
    return merged


def load_module(spec: str) -> ModuleType:
    """Import a module so its :func:`register` calls run.

    Args:
        spec: A dotted module name, or a path to a ``.py`` file.

    Returns:
        The imported module.

    Raises:
        FileNotFoundError: If a path was given and does not exist.
        ImportError: If a dotted name does not import.
    """
    path = Path(spec)
    if path.suffix == ".py":
        if not path.is_file():
            raise FileNotFoundError(f"no mixer module at {path}")
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load a module from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


class CausalAttention(nn.Module):
    """Upstream's ``MHA``: one fused qkv projection, additive mask, output projection.

    The mask adds ``-10000`` rather than ``-inf`` above the diagonal, and the softmax
    scale multiplies the keys rather than the queries. Both are upstream's, kept because
    this is the reference point every published MQAR figure is drawn against.
    """

    def __init__(
        self, d_model: int, num_heads: int, bias: bool, dropout: float
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} is not a multiple of {num_heads} heads"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d_model)`` in and out."""
        batch, length, _ = x.shape
        qkv = self.Wqkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", query, key * scale)
        mask = torch.triu(
            torch.full((length, length), -10000.0, device=x.device), diagonal=1
        )
        scores = scores + mask.to(scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=value.dtype)
        attention = nn.functional.dropout(
            attention, self.dropout if self.training else 0.0
        )
        context = torch.einsum("bhts,bshd->bthd", attention, value)
        return self.out_proj(context.reshape(batch, length, -1))


class CausalConv(nn.Module):
    """Upstream's ``BaseConv``: ``conv(u) * projection(u) + u``.

    The convolution is depthwise, left-padded by ``kernel_size - 1`` and truncated to the
    input length, which is what makes it causal. The trailing ``+ u`` is internal to the
    mixer and sits inside the block's own residual, so this layer carries two skips.

    The projection is built before the convolution because upstream builds it first, and
    construction order is what decides which draw off the global generator each parameter
    gets. The distribution is the same either way; the weights are not.
    """

    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}; the long-convolution "
                "variant upstream selects at -1 is not ported"
            )
        self.kernel_size = kernel_size
        self.projection = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size, groups=d_model, padding=kernel_size - 1
        )

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d_model)`` in and out."""
        length = x.shape[1]
        convolved = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)
        return convolved * self.projection(x) + x


def _build_attention(d_model: int, **settings: Setting) -> nn.Module:
    return CausalAttention(
        d_model=d_model,
        num_heads=int(settings["num_heads"]),
        bias=bool(settings["bias"]),
        dropout=float(settings["dropout"]),
    )


def _build_conv(d_model: int, **settings: Setting) -> nn.Module:
    return CausalConv(d_model=d_model, kernel_size=int(settings["kernel_size"]))


def _build_slinoss(d_model: int, **settings: Setting) -> nn.Module:
    from slinoss.config import SLinOSSConfig
    from slinoss.mixer import SLinOSSMixer

    config = SLinOSSConfig(
        d_model=d_model,
        d_state=int(settings["d_state"]),
        expand=float(settings["expand"]),
        d_head=int(settings["d_head"]),
        n_groups=int(settings["n_groups"]),
        chunk_size=int(settings["chunk_size"]),
        d_conv=int(settings["d_conv"]),
        key_conv=bool(settings["key_conv"]),
        init_span=int(settings["init_span"]),
        w_max=float(settings["w_max"]),
        bias=bool(settings["bias"]),
        conv_bias=bool(settings["conv_bias"]),
    )
    return protect(SLinOSSMixer(config))


def _slinoss_defaults() -> dict[str, Setting]:
    """Read every slinoss setting's default off :class:`slinoss.config.SLinOSSConfig`.

    Only ``d_state`` is named here, because it has no default there. Everything else is
    read from the dataclass so this registry cannot drift from the contract it configures.
    """
    from slinoss.config import SLinOSSConfig

    declared = {field.name: field for field in fields(SLinOSSConfig)}
    defaults: dict[str, Setting] = {"d_state": 144}
    for name in (
        "expand",
        "d_head",
        "n_groups",
        "chunk_size",
        "d_conv",
        "key_conv",
        "init_span",
        "w_max",
        "bias",
        "conv_bias",
    ):
        default = declared[name].default
        assert isinstance(default, bool | int | float | str), name
        defaults[name] = default
    return defaults


def _coerce(key: str, default: Setting, text: str) -> Setting:
    """Coerce an override's text to the default's type.

    bool is tested before int because bool subclasses int, and ``bool("false")`` is True.

    Raises:
        ValueError: If the text does not read at the setting's type. The message names the
            setting and the type, because the alternative is int's own message with no
            mention of which flag produced it.
    """
    if isinstance(default, bool):
        lowered = text.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{key} is a flag; cannot read {text!r}")
    if isinstance(default, int):
        try:
            return int(text)
        except ValueError as error:
            raise ValueError(f"{key} is int; cannot read {text!r}") from error
    if isinstance(default, float):
        try:
            return float(text)
        except ValueError as error:
            raise ValueError(f"{key} is float; cannot read {text!r}") from error
    return text


def _scope_overrides(
    names: Sequence[str], overrides: Iterable[str]
) -> dict[str, list[str]]:
    """Split ``entry.key=value`` overrides by entry; pass unscoped ones through."""
    scoped: dict[str, list[str]] = {name: [] for name in names}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, text = item.split("=", 1)
        if "." in key:
            target, inner = key.split(".", 1)
            if target not in scoped:
                raise KeyError(
                    f"override {item!r} names {target}, which is not in the cycle "
                    f"{list(names)}"
                )
            scoped[target].append(f"{inner}={text}")
        elif len(scoped) > 1:
            raise ValueError(
                f"override {item!r} is ambiguous across {sorted(scoped)}; write "
                f"<mixer>.{item}"
            )
        else:
            scoped[names[0]].append(item)
    return scoped


def _register_builtins() -> None:
    # Attention's defaults are the figure-2 sweep's, not ``MHA``'s own: dropout 0.1
    # rather than 0.0, which both published configs pass, and one head. The modern
    # reproduction runs the same mixer at two heads; that is one override away.
    register(
        "attention",
        _build_attention,
        {"num_heads": 1, "bias": True, "dropout": 0.1},
        layer_index_policy="unused",
        max_length_policy="unused",
    )
    register(
        "conv",
        _build_conv,
        {"kernel_size": 3},
        layer_index_policy="unused",
        max_length_policy="unused",
    )
    register(
        "slinoss",
        _build_slinoss,
        _slinoss_defaults(),
        layer_index_policy="unused",
        max_length_policy="unused",
    )


_register_builtins()
