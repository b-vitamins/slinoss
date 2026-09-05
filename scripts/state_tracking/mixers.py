"""Sequence-mixer registry: what a state-tracking arm is comparing.

An arm names a mixer and the scaffold is identical, so a difference between two arms is a
difference between two sequence mixers. :func:`register` adds one, :func:`resolve` turns a
name and a list of ``key=value`` overrides into a factory
:func:`scripts.state_tracking.model.build_model` calls once per layer.

Three mixers ship here: the operator under test, and the two controls that bracket it.

    name       what it is                                       settings
    slinoss    the tree's mixer, at its own defaults            mixer config fields
    attention  causal multi-head attention with rotary          n_heads, rotary
    conv       causal depthwise convolution over d_conv taps    d_conv, expand

On this axis the two controls read differently than on a recall axis. ``conv`` has a
receptive field of ``d_conv`` positions per layer, so it is the star-free floor: a task it
solves is not measuring a carried state. ``attention`` is not a ceiling here. A fixed-depth
transformer's expressivity is uniform ``TC^0``, so it cannot track a non-solvable group at
growing length whatever its width, and on the group half of the suite it is expected to
fail past the trained length while a recurrence need not. It stays because it is the
strongest thing that succeeds on the automaton half, which localizes a failure there to
optimization rather than expressivity.

A baseline whose package is not a dependency of this tree lives in a module of its own that
calls :func:`register` at import, which ``--mixer-module`` imports before resolving.
Nothing in the registry may import an optional dependency at module scope. A diagonal
linear recurrence -- the control that separates solvable dynamics from the rest -- slots in
that way.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from scripts.harness import build_slinoss, slinoss_defaults
from scripts.state_tracking.model import MixerFactory


@dataclass
class Mixer:
    """A resolved mixer.

    Attributes:
        name: Registry key.
        factory: ``(d_model, max_length) -> module``, for the model builder.
        settings: The settings the factory closes over, defaults and overrides merged.
        max_length_policy: Whether the constructor consumes the scaffold's widest
            possible sequence. ``unused`` is explicit and means the value is recorded
            but never handed to the constructor.
        constructions: Effective configuration of every layer the factory built.
    """

    name: str
    factory: MixerFactory
    settings: dict[str, Any]
    max_length_policy: Literal["required", "unused"]
    constructions: list[dict[str, Any]]


@dataclass(frozen=True)
class MixerEntry:
    """One registry entry.

    Attributes:
        build: Constructor. Called as ``build(d_model, max_length, **settings)``
            only under a ``required`` length policy, otherwise as
            ``build(d_model, **settings)``.
        max_length_policy: ``required`` when ``build`` accepts the scaffold's
            ``max_length`` after ``d_model``; ``unused`` when it does not. There is no
            default because silently guessing this contract caused the two historical
            winner configurations to be mislabeled.
        defaults: Every setting ``build`` accepts, with its default. The set is closed:
            an override outside it is refused, and a default's type is what an override
            string is coerced to.
    """

    build: Callable[..., nn.Module]
    max_length_policy: Literal["required", "unused"]
    defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_length_policy not in {"required", "unused"}:
            raise ValueError(
                "max_length_policy must be 'required' or 'unused', got "
                f"{self.max_length_policy!r}"
            )


REGISTRY: dict[str, MixerEntry] = {}


def register(name: str, entry: MixerEntry) -> None:
    """Add a mixer under ``name``.

    Args:
        name: Registry key, as the command line spells it.
        entry: The constructor and its settings.

    Raises:
        ValueError: On a name already taken. Re-registration would silently change what
            an arm measured.
    """
    if name in REGISTRY:
        raise ValueError(f"mixer {name} is already registered")
    REGISTRY[name] = entry


def _coerce(name: str, key: str, default: Any, text: str) -> Any:
    """Read an override string at the type of the setting's default.

    Args:
        name: Mixer, for the message.
        key: Setting.
        default: Its default, whose type is the target.
        text: The override, as the command line gave it.

    Returns:
        The value.

    Raises:
        ValueError: On text the type does not read, or a default type with no rule.
    """
    if isinstance(default, bool):
        if text.lower() in {"true", "1", "yes"}:
            return True
        if text.lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"{name}: {key} is a flag, got {text!r}")
    for kind in (int, float, str):
        if isinstance(default, kind):
            try:
                return kind(text)
            except ValueError as exc:
                raise ValueError(
                    f"{name}: {key} is {kind.__name__}, got {text!r}"
                ) from exc
    raise ValueError(f"{name}: {key} has no rule for {type(default).__name__}")


def settings_from(name: str, overrides: Iterable[str]) -> dict[str, Any]:
    """Merge ``key=value`` overrides onto a mixer's defaults.

    Args:
        name: Registry key.
        overrides: Strings of the form ``key=value``.

    Returns:
        Every setting the mixer accepts.

    Raises:
        KeyError: On an unregistered name.
        ValueError: On a malformed override, a setting the mixer does not have, or a
            value its type does not read.
    """
    entry = REGISTRY[name]
    settings = dict(entry.defaults)
    for override in overrides:
        key, sep, text = override.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {override!r}")
        if key not in settings:
            raise ValueError(f"{name}: no setting {key}; has {sorted(settings)}")
        settings[key] = _coerce(name, key, settings[key], text)
    return settings


def resolve(name: str, overrides: Iterable[str] = ()) -> Mixer:
    """A mixer factory at its resolved settings.

    Args:
        name: Registry key.
        overrides: Strings of the form ``key=value``.

    Returns:
        The mixer.

    Raises:
        KeyError: On an unregistered name, naming what is registered.
        ValueError: From :func:`settings_from`.
    """
    if name not in REGISTRY:
        raise KeyError(f"no mixer {name}; registered: {sorted(REGISTRY)}")
    entry = REGISTRY[name]
    settings = settings_from(name, overrides)

    constructions: list[dict[str, Any]] = []

    def factory(d_model: int, max_length: int) -> nn.Module:
        if max_length < 1:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if entry.max_length_policy == "required":
            module = entry.build(d_model, max_length, **settings)
            consumed: int | None = max_length
        else:
            module = entry.build(d_model, **settings)
            consumed = None
        config = getattr(module, "config", None)
        effective = (
            asdict(cast(Any, config))
            if is_dataclass(config)
            else {"d_model": d_model, **settings}
        )
        for field_name in (
            "resolved_init_period_span",
            "resolved_init_decay_span",
        ):
            if hasattr(config, field_name):
                effective[field_name] = getattr(config, field_name)
        constructions.append(
            {
                "module": f"{type(module).__module__}.{type(module).__qualname__}",
                "effective_config": effective,
                "context": {
                    "max_length_supplied": max_length,
                    "max_length_policy": entry.max_length_policy,
                    "max_length_consumed": consumed,
                },
                "initialization": "mixer_constructor; no scaffold reinitialization",
            }
        )
        return module

    return Mixer(name, factory, settings, entry.max_length_policy, constructions)


def load_module(path: str) -> None:
    """Import a module so its :func:`register` calls run.

    Args:
        path: Importable module path. This is how a mixer whose package is not a
            dependency of this tree reaches the registry.

    Raises:
        ModuleNotFoundError: From the import.
    """
    importlib.import_module(path)


# controls:


class Rotary(nn.Module):
    """Rotary position code, the half-split convention.

    The tables are built once at ``max_length`` and sliced, so no arm pays a rebuild per
    step. A batch wider than ``max_length`` is a configuration error, not a resize: an
    evaluation past the trained length is the measurement, and silently extending the
    tables would hide which arm was asked to extrapolate.

    Args:
        d_head: Channels per head. Even.
        max_length: Longest sequence the tables cover.
        base: Frequency base.

    Raises:
        ValueError: On an odd ``d_head``, which the half split cannot pair.
    """

    def __init__(self, d_head: int, max_length: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {d_head}")
        freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        angle = torch.outer(torch.arange(max_length, dtype=torch.float32), freq)
        doubled = torch.cat([angle, angle], dim=-1)
        self.register_buffer("cos", doubled.cos(), persistent=False)
        self.register_buffer("sin", doubled.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Rotate every head's channels by its position's angle.

        Args:
            x: ``(B,H,T,P)``.

        Returns:
            The same shape and dtype.

        Raises:
            ValueError: When ``T`` is over the table's length.
        """
        length = x.shape[-2]
        table = cast(Tensor, self.cos)
        if length > table.shape[0]:
            raise ValueError(
                f"sequence of {length} is over the rotary table's {table.shape[0]}"
            )
        cos = table[:length].to(x.dtype)
        sin = cast(Tensor, self.sin)[:length].to(x.dtype)
        half = x.shape[-1] // 2
        flipped = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + flipped * sin


class CausalAttention(nn.Module):
    """Causal multi-head attention with rotary positions.

    Projections carry no bias and the attention is
    :func:`torch.nn.functional.scaled_dot_product_attention` rather than a fused kernel.

    Args:
        d_model: Stream width.
        max_length: Longest sequence, for the rotary tables.
        n_heads: Heads. Divides ``d_model``.
        rotary: Whether to rotate. Without it the mixer is position-blind and the
            scaffold carries no positional embedding, so a task whose answer depends on
            order collapses.

    Raises:
        ValueError: When ``n_heads`` does not divide ``d_model``.
    """

    def __init__(
        self, d_model: int, max_length: int, n_heads: int = 16, rotary: bool = True
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"n_heads {n_heads} does not divide d_model {d_model}")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = (
            Rotary(d_model // n_heads, max_length) if rotary else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Attend over the prefix.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``.
        """
        bsz, length, _ = x.shape
        qkv = self.qkv(x).unflatten(-1, (3, self.n_heads, -1)).permute(2, 0, 3, 1, 4)
        query, key, value = (self.rotary(qkv[0]), self.rotary(qkv[1]), qkv[2])
        out = nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        return self.out_proj(out.transpose(1, 2).reshape(bsz, length, -1))


class CausalConv(nn.Module):
    """Causal depthwise convolution over ``d_conv`` taps, gated.

    The star-free floor: its receptive field is ``d_conv`` positions per layer, so it can
    match a local pattern and can carry nothing across the sequence. A task it solves is
    not testing a state.

    Args:
        d_model: Stream width.
        d_conv: Taps.
        expand: Inner width multiplier.

    Raises:
        ValueError: On fewer than one tap.
    """

    def __init__(self, d_model: int, d_conv: int = 4, expand: float = 2.0) -> None:
        super().__init__()
        if d_conv < 1:
            raise ValueError(f"d_conv must be positive, got {d_conv}")
        inner = round(expand * d_model)
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner, bias=True)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Convolve over the prefix.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``.
        """
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        padded = nn.functional.pad(value.transpose(1, 2), (self.d_conv - 1, 0))
        mixed = self.conv(padded).transpose(1, 2)
        return self.out_proj(mixed * nn.functional.silu(gate))


def _register_builtins() -> None:
    """Register the three mixers this module defines.

    Called at import. The slinoss entry reads its defaults from
    :class:`slinoss.SLinOSSMixerConfig`, which imports torch and nothing optional.
    """
    register("slinoss", MixerEntry(build_slinoss, "unused", slinoss_defaults(144)))
    register(
        "attention",
        MixerEntry(
            lambda d_model, max_length, **kw: CausalAttention(
                d_model, max_length, **kw
            ),
            "required",
            {"n_heads": 16, "rotary": True},
        ),
    )
    register(
        "conv",
        MixerEntry(
            lambda d_model, **kw: CausalConv(d_model, **kw),
            "unused",
            {"d_conv": 4, "expand": 2.0},
        ),
    )


_register_builtins()
