"""Fail-closed adapter for the first public KLA implementation.

This is a calibration instrument, not another SLinOSS arm.  The KLA paper does not
ship the source tree that produced its MAD table: the first public experiment driver
deliberately changes several paper settings, and later releases remove that driver.
When we ask whether a reconstructed paper harness can reproduce KLA's own bar, the
name ``kla-v001`` must therefore mean one exact public implementation rather than
whatever package happens to be importable.

Use an exact, clean checkout with its ``src`` directory on ``PYTHONPATH``::

    python -m scripts.mad.run \
        --mixer-module scripts.mad.baselines.kla_v001 \
        --mixer kla-v001 --profile kla-paper-v2 --task ficr

The adapter rejects an installed wheel, a descendant commit, or a dirty package tree.
Every construction records the verified external commit and package-tree object.
"""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from torch import Tensor, nn

from scripts.mad.mixers import MixerEntry, register

KLA_COMMIT = "3b1e4727f4cad23a2db5e341130a04337b8eb4e5"
"""Initial public release, tagged by its commit message as KLA v0.0.1."""

KLA_PACKAGE_TREE = "10019fee2a760bf851add7ff78cfb35b6a80f1b4"
"""Git tree object at ``KLA_COMMIT:src/kla``."""


@dataclass(frozen=True)
class ExternalSource:
    """Identity of the external implementation a constructed layer executed."""

    repository: str
    package_path: str
    version: str
    commit: str
    package_tree: str
    dirty: bool


@dataclass(frozen=True)
class ConstructionConfig:
    """Complete effective KLA config plus its external source identity."""

    d_model: int
    kla: dict[str, Any]
    external_source: dict[str, Any]


def _git(root: Path, *args: str) -> str:
    """Run a read-only Git query, preserving its useful failure message."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            f"cannot verify external KLA source: {detail.strip()}"
        ) from exc


def _source_identity(kla: ModuleType) -> ExternalSource:
    """Verify that ``kla`` came from the exact clean public-v0.0.1 checkout."""
    raw_file = getattr(kla, "__file__", None)
    if raw_file is None:
        raise RuntimeError(
            "external KLA module has no __file__; source is unverifiable"
        )
    package = Path(raw_file).resolve().parent
    root_text = _git(package, "rev-parse", "--show-toplevel")
    root = Path(root_text).resolve()
    try:
        relative = package.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"external KLA package {package} is outside Git root {root}"
        ) from exc
    if relative != "src/kla":
        raise RuntimeError(
            "external KLA must be imported from <checkout>/src/kla, got "
            f"{relative!r} under {root}"
        )

    version = str(getattr(kla, "__version__", ""))
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD:src/kla")
    dirty_text = _git(root, "status", "--porcelain=v1", "--", "src/kla")
    problems: list[str] = []
    if version != "0.0.1":
        problems.append(f"version is {version!r}, expected '0.0.1'")
    if commit != KLA_COMMIT:
        problems.append(f"commit is {commit}, expected {KLA_COMMIT}")
    if tree != KLA_PACKAGE_TREE:
        problems.append(f"package tree is {tree}, expected {KLA_PACKAGE_TREE}")
    if dirty_text:
        problems.append("src/kla has tracked or untracked modifications")
    if problems:
        raise RuntimeError("external KLA source mismatch: " + "; ".join(problems))
    return ExternalSource(
        repository=str(root),
        package_path=str(package),
        version=version,
        commit=commit,
        package_tree=tree,
        dirty=False,
    )


def _load_pinned_kla() -> tuple[ModuleType, ExternalSource]:
    """Import KLA lazily and verify it before any layer is constructed."""
    try:
        kla = importlib.import_module("kla")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "kla-v001 requires a clean checkout of kalman-linear-attention at "
            f"{KLA_COMMIT} with <checkout>/src on PYTHONPATH"
        ) from exc
    return kla, _source_identity(kla)


class PinnedKLA(nn.Module):
    """Thin wrapper that makes the external implementation/config recordable."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        kla, source = _load_pinned_kla()
        config = kla.KLAConfig(
            d_state=8,
            expand=1.0,
            process_noise_scale=0.01,
            dt_min=0.001,
            dt_max=0.1,
            conv_kernel_size=4,
            return_variance=False,
            backend="torch",
            scan_impl="associative",
        )
        self.config = ConstructionConfig(
            d_model=d_model,
            kla=asdict(config),
            external_source=asdict(source),
        )
        self.layer = cast(nn.Module, kla.KLALayer(d_model=d_model, config=config))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the pinned public KLA layer."""
        return cast(Tensor, self.layer(x))


def _build(d_model: int) -> nn.Module:
    """Build the fixed paper-sized plain-KLA calibration layer."""
    return PinnedKLA(d_model)


register(
    "kla-v001",
    MixerEntry(_build, "unused", {}, initialization_policy="scaffold"),
)
