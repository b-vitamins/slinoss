"""The optional KLA calibration adapter fails closed on source identity."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from scripts.mad.baselines import kla_v001
from scripts.mad.mixers import REGISTRY, resolve


def test_import_registers_without_importing_the_optional_package() -> None:
    """Registry discovery itself must not require KLA to be installed."""
    assert "kla-v001" in REGISTRY
    resolved = resolve("kla-v001")
    assert resolved.settings == {}
    assert resolved.initialization_policy == "scaffold"


def test_source_check_rejects_a_non_git_package(tmp_path: Path) -> None:
    """A wheel/version string cannot masquerade as the pinned source checkout."""
    package = tmp_path / "kla"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("__version__ = '0.0.1'\n")
    fake = ModuleType("kla")
    fake.__file__ = str(init)
    cast(Any, fake).__version__ = "0.0.1"
    with pytest.raises(RuntimeError, match="cannot verify external KLA source"):
        kla_v001._source_identity(fake)


def test_source_check_names_every_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No first failure hides a second way the external implementation drifted."""
    root = tmp_path / "checkout"
    package = root / "src" / "kla"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("__version__ = '9.9.9'\n")
    fake = ModuleType("kla")
    fake.__file__ = str(init)
    cast(Any, fake).__version__ = "9.9.9"

    answers = iter((str(root), "wrong-commit", "wrong-tree", " M src/kla/configs.py"))
    monkeypatch.setattr(kla_v001, "_git", lambda *_args: next(answers))
    with pytest.raises(RuntimeError) as caught:
        kla_v001._source_identity(fake)
    message = str(caught.value)
    assert "version" in message
    assert "commit" in message
    assert "package tree" in message
    assert "modifications" in message


def test_git_query_surfaces_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A verification subprocess failure is diagnostic rather than a fallback."""
    error = subprocess.CalledProcessError(128, ["git"], stderr="not a repository")
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )
    with pytest.raises(RuntimeError, match="not a repository"):
        kla_v001._git(tmp_path, "rev-parse", "HEAD")
