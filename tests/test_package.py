from __future__ import annotations

import tomllib
from pathlib import Path

import slinoss


def test_package_version_matches_pyproject() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    assert slinoss.__version__ == pyproject["project"]["version"]
