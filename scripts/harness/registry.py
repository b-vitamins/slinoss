"""Sequence-mixer registry: what an arm is comparing.

An arm names a mixer and the scaffold is identical, so a difference between two arms is a
difference between two sequence mixers. :meth:`Registry.resolve` turns a name and a list of
``key=value`` overrides into a factory the model builder calls once per layer.

The registry is an object, not a module-level dict. Two axes register a mixer under the same
name with different defaults -- ``slinoss`` at ``d_state 144`` on a 128-wide state-tracking
stream is not ``slinoss`` at a language-modelling width -- and a shared dict makes importing
both axes in one process a re-registration error. One registry per axis, one machinery.

A baseline whose package is not a dependency of this tree lives in a module of its own that
registers at import, which ``--mixer-module`` imports before resolving. Nothing here may
import an optional dependency at module scope.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, cast

from torch import nn

__all__ = [
    "Mixer",
    "MixerEntry",
    "MixerFactory",
    "Registry",
    "load_module",
]

MixerFactory = Callable[[int, int], nn.Module]
"""``(d_model, max_length) -> module``. Maps ``(B,T,d_model)`` to ``(B,T,d_model)``.

The max length is passed because a mixer may need to size a positional term or a cache; it
is the widest sequence the arm can produce, not the width of any one batch."""


@dataclass
class Mixer:
    """A resolved mixer.

    Attributes:
        name: Registry key.
        factory: ``(d_model, max_length) -> module``, for the model builder.
        settings: The settings the factory closes over, defaults and overrides merged.
            Goes into the run record as it is.
        max_length_policy: Whether the constructor consumes the supplied maximum
            length. ``unused`` is explicit and means it is recorded but never passed.
        constructions: Effective configuration and context contract of every module
            built by :attr:`factory`.
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
        build: Constructor, called as ``build(d_model, max_length, **settings)``.
            The length is passed only under a ``required`` policy; under ``unused``
            it is called as ``build(d_model, **settings)``.
        max_length_policy: ``required`` when :attr:`build` consumes the length and
            ``unused`` when it does not. There is deliberately no default: guessing
            this contract is the silent-swallowing defect this type prevents.
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


class Registry:
    """The mixers one axis offers.

    Args:
        name: What this registry is for, used in the messages so a wrong-axis name
            reports the axis it was looked up in.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, MixerEntry] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        """Every registered name, sorted.

        Returns:
            The names.
        """
        return sorted(self._entries)

    def entry(self, name: str) -> MixerEntry:
        """One entry.

        Args:
            name: Registry key.

        Returns:
            The entry.

        Raises:
            KeyError: On an unregistered name, naming what is registered.
        """
        if name not in self._entries:
            raise KeyError(f"no {self.name} mixer {name}; registered: {self.names()}")
        return self._entries[name]

    def register(self, name: str, entry: MixerEntry) -> None:
        """Add a mixer under ``name``.

        Args:
            name: Registry key, as the command line spells it.
            entry: The constructor and its settings.

        Raises:
            ValueError: On a name already taken. Re-registration would silently change
                what an arm measured.
        """
        if name in self._entries:
            raise ValueError(f"{self.name} mixer {name} is already registered")
        self._entries[name] = entry

    def settings_from(self, name: str, overrides: Iterable[str]) -> dict[str, Any]:
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
        settings = dict(self.entry(name).defaults)
        for override in overrides:
            key, sep, text = override.partition("=")
            if not sep:
                raise ValueError(f"override must be key=value, got {override!r}")
            if key not in settings:
                raise ValueError(f"{name}: no setting {key}; has {sorted(settings)}")
            settings[key] = _coerce(name, key, settings[key], text)
        return settings

    def resolve(self, name: str, overrides: Iterable[str] = ()) -> Mixer:
        """A mixer factory at its resolved settings.

        Args:
            name: Registry key.
            overrides: Strings of the form ``key=value``.

        Returns:
            The mixer.

        Raises:
            KeyError: On an unregistered name, naming what is registered.
            ValueError: From :meth:`settings_from`.
        """
        entry = self.entry(name)
        settings = self.settings_from(name, overrides)
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

        return Mixer(
            name,
            factory,
            settings,
            entry.max_length_policy,
            constructions,
        )


def load_module(path: str) -> None:
    """Import a module so its registration calls run.

    Args:
        path: Importable module path. This is how a mixer whose package is not a
            dependency of this tree reaches a registry.

    Raises:
        ModuleNotFoundError: From the import.
    """
    importlib.import_module(path)
