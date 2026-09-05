"""One file per arm, holding everything needed to rebuild it and nothing that needs a class.

A checkpoint carries the scaffold config, the mixer name and its resolved settings, the corpus
manifest, the step, the transferred rates, and the state dict. Rebuilding therefore needs the
registry and no pickled module: every stored value is a scalar, a string, a plain dict, or a
tensor, so :func:`torch.load` reads it with ``weights_only=True``.

The mixer settings are stored resolved rather than as the overrides that produced them. An
override list read against a later registry could mean something else; the settings dict is
what was built.

The manifest is stored because a checkpoint that cannot name its corpus cannot be put in a
table. :func:`scripts.lm.run.table` reads the digest off here, so a zero-shot score and the
bits-per-byte it sits next to are known to come from one corpus.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import torch

from scripts.lm.corpus import CorpusManifest, from_dict, to_dict
from scripts.lm.model import MixerLM
from slinoss import SLinOSSConfig

__all__ = ["FORMAT", "Checkpoint", "load", "load_model", "save"]

FORMAT = 2
"""Layout version. A file from another version is refused, not guessed at."""


class Checkpoint(NamedTuple):
    """A read checkpoint, before the model is built.

    Attributes:
        config: The scaffold's config.
        manifest: The corpus the arm trained on.
        mixer: Registry name.
        mixer_settings: The settings the mixer was built at.
        hybrid_final: Registry name of the last layer's mixer when it differs, else None.
        hybrid_final_settings: That mixer's settings, or None.
        max_length: Longest sequence the arm was built for.
        step: Optimizer steps taken.
        lr: The transferred hidden rate.
        embedding_lr: The transferred token-table rate.
        seed: The run seed.
        state_dict: Parameter and buffer tensors.
    """

    config: SLinOSSConfig
    manifest: CorpusManifest | None
    mixer: str
    mixer_settings: dict[str, Any]
    hybrid_final: str | None
    hybrid_final_settings: dict[str, Any] | None
    max_length: int
    step: int
    lr: float
    embedding_lr: float
    seed: int
    state_dict: dict[str, torch.Tensor]


def save(
    path: Path,
    model: MixerLM,
    *,
    config: SLinOSSConfig,
    mixer: str,
    mixer_settings: dict[str, Any],
    max_length: int,
    step: int,
    lr: float,
    embedding_lr: float,
    seed: int,
    manifest: CorpusManifest | None = None,
    hybrid_final: str | None = None,
    hybrid_final_settings: dict[str, Any] | None = None,
) -> None:
    """Write one arm.

    Args:
        path: Destination ``.pt``. Its parent is created.
        model: The trained stack.
        config: The scaffold's config.
        mixer: Registry name.
        mixer_settings: Resolved settings, from :attr:`scripts.harness.Mixer.settings`.
        max_length: Longest sequence the arm was built for.
        step: Optimizer steps taken.
        lr: The transferred hidden rate.
        embedding_lr: The transferred token-table rate.
        seed: The run seed.
        manifest: The corpus manifest.
        hybrid_final: Registry name for the last layer, when it differs.
        hybrid_final_settings: That mixer's settings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "config": asdict(config),
        "manifest": None if manifest is None else to_dict(manifest),
        "mixer": mixer,
        "mixer_settings": dict(mixer_settings),
        "hybrid_final": hybrid_final,
        "hybrid_final_settings": (
            None if hybrid_final_settings is None else dict(hybrid_final_settings)
        ),
        "max_length": max_length,
        "step": step,
        "lr": lr,
        "embedding_lr": embedding_lr,
        "seed": seed,
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, path)


def load(path: Path) -> Checkpoint:
    """Read one arm without building it.

    Args:
        path: The ``.pt``.

    Returns:
        The checkpoint.

    Raises:
        FileNotFoundError: When the file is absent.
        ValueError: On another layout version, or a payload missing a field.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    found = payload.get("format")
    if found != FORMAT:
        raise ValueError(f"checkpoint format is {found} and this reader is {FORMAT}")
    raw_manifest = payload["manifest"]
    return Checkpoint(
        config=SLinOSSConfig(**payload["config"]),
        manifest=None if raw_manifest is None else from_dict(raw_manifest),
        mixer=payload["mixer"],
        mixer_settings=payload["mixer_settings"],
        hybrid_final=payload["hybrid_final"],
        hybrid_final_settings=payload["hybrid_final_settings"],
        max_length=payload["max_length"],
        step=payload["step"],
        lr=payload["lr"],
        embedding_lr=payload["embedding_lr"],
        seed=payload["seed"],
        state_dict=payload["state_dict"],
    )


def load_model(
    path: Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[MixerLM, Checkpoint]:
    """Read one arm and rebuild it.

    The mixers come from :data:`scripts.lm.mixers.REGISTRY` at the stored settings, so a
    baseline's checkpoint needs that baseline's package installed. Loading is strict: a
    state dict that does not match the rebuilt module is a different program, and a
    partially loaded model would report a number for it.

    Args:
        path: The ``.pt``.
        device: Destination device.
        dtype: Destination dtype.

    Returns:
        The model and the checkpoint it came from.

    Raises:
        FileNotFoundError: When the file is absent.
        ValueError: From :func:`load`, or on a state dict that does not match.
        KeyError: When the stored mixer is not registered.
        RuntimeError: From :meth:`torch.nn.Module.load_state_dict` on a shape mismatch.
    """
    from scripts.lm.mixers import REGISTRY
    from scripts.lm.model import build_model, layer_factories

    checkpoint = load(path)
    base_overrides = [
        f"{key}={value}" for key, value in checkpoint.mixer_settings.items()
    ]
    factory = REGISTRY.resolve(checkpoint.mixer, base_overrides).factory

    final = None
    if checkpoint.hybrid_final is not None:
        settings = checkpoint.hybrid_final_settings or {}
        overrides = [f"{key}={value}" for key, value in settings.items()]
        final = REGISTRY.resolve(checkpoint.hybrid_final, overrides).factory

    model = build_model(
        checkpoint.config,
        layer_factories(factory, checkpoint.config.n_layers, final),
        max_length=checkpoint.max_length,
    )
    model.load_state_dict(checkpoint.state_dict, strict=True)
    moved = model.to(device=device, dtype=dtype)
    return moved, checkpoint
