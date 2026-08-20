"""Profiler binary resolution: the probe, its order, and the named failure.

The subject is a host property, so every host fact is fabricated: PATH, the toolkit
root variables, and the install glob all point into ``tmp_path``. A test that read
the real host would pass on the fleet and fail on a laptop, which is the
environment dependence the probe exists to remove.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from slinoss.perf import tools
from slinoss.perf.tools import CUDA_ROOT_VARS, ToolNotFoundError, resolve_tool


def executable(directory: Path, name: str) -> Path:
    """Create ``name`` in ``directory`` with the execute bit set."""
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / name
    binary.write_text("#!/bin/sh\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


@pytest.fixture(autouse=True)
def bare_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A host with no PATH, no toolkit variable, and no default install."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for var in CUDA_ROOT_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        tools, "CUDA_BIN_GLOB", str(tmp_path / "none" / "cuda*" / "bin")
    )


def test_a_spec_naming_a_path_is_run_as_given(tmp_path: Path) -> None:
    # The caller named a binary on this host. A search would find a different one,
    # and resolving twice has to be resolving once: the driver resolves the value it
    # will hand the profiler, and every probe result carries a separator.
    spec = str(tmp_path / "nsight" / "ncu")
    assert resolve_tool(spec) == spec
    assert resolve_tool(resolve_tool(spec)) == spec


def test_path_wins_and_a_toolkit_directory_is_the_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PATH first, then the declared root. Both arcs, because only one is the bug.

    A venv's PATH does not carry the toolkit bin directory, so the second arc is
    what the default ``--ncu ncu`` needs; the first is what a host with the profiler
    installed already had.
    """
    on_path = executable(tmp_path / "bin", "nsys")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert resolve_tool("nsys") == str(on_path)

    in_toolkit = executable(tmp_path / "cuda-12.3" / "bin", "ncu")
    monkeypatch.setenv(CUDA_ROOT_VARS[0], str(tmp_path / "cuda-12.3"))
    assert resolve_tool("ncu") == str(in_toolkit)


def test_a_binary_found_nowhere_raises_naming_every_path_tried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The failure is loud and located. It is not a skip.

    A driver whose profiler is missing has nothing to judge, so degrading to a run
    without counters would report a clean audit over an empty capture. The error
    lists the candidates because a bare errno says only that something was missing,
    which is what cost a measurement.
    """
    root = tmp_path / "cuda-12.3"
    (root / "bin").mkdir(parents=True)
    monkeypatch.setenv(CUDA_ROOT_VARS[0], str(root))
    with pytest.raises(ToolNotFoundError) as caught:
        resolve_tool("ncu")
    message = str(caught.value)
    assert "'ncu' is not on PATH" in message
    assert str(root / "bin" / "ncu") in message
    assert CUDA_ROOT_VARS[0] in message
    # A caller already handling the errno subprocess would have raised handles this.
    assert isinstance(caught.value, FileNotFoundError)
    # Present but not executable is the same answer: the probe is for something to
    # run, and a data file at that name would fail later, inside the capture.
    plain = root / "bin" / "ncu"
    plain.write_text("not a program\n")
    plain.chmod(plain.stat().st_mode & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH)
    if os.access(plain, os.X_OK):  # pragma: no cover - root ignores the mode bits
        pytest.skip("this user executes anything")
    with pytest.raises(ToolNotFoundError):
        resolve_tool("ncu")


def test_the_search_order_is_declared_then_installed_then_deduplicated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Which toolkit answers when a host has two, and what a stale variable costs.

    A variable naming a root that does not exist is dropped rather than searched, so
    the list is what could hold a binary. The declared root is first because a host
    with two toolkits whose choice matters is a host that set the variable.
    """
    declared = tmp_path / "opt" / "cuda"
    (declared / "bin").mkdir(parents=True)
    for version in ("cuda-12.3", "cuda-11.8"):
        (tmp_path / "usr" / version / "bin").mkdir(parents=True)
    monkeypatch.setenv(CUDA_ROOT_VARS[0], str(declared))
    monkeypatch.setenv(CUDA_ROOT_VARS[1], str(tmp_path / "gone"))
    monkeypatch.setattr(tools, "CUDA_BIN_GLOB", str(tmp_path / "usr" / "cuda*" / "bin"))
    assert tools.cuda_bin_dirs() == (
        declared / "bin",
        tmp_path / "usr" / "cuda-12.3" / "bin",
        tmp_path / "usr" / "cuda-11.8" / "bin",
    )
    # The same directory named twice is searched once, so the error message lists
    # candidates rather than repetitions of one.
    monkeypatch.setenv(CUDA_ROOT_VARS[1], str(declared))
    assert tools.cuda_bin_dirs().count(declared / "bin") == 1
