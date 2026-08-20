"""Where the profiler binaries are, when PATH does not say.

``ncu`` and ``nsys`` ship in the CUDA toolkit's bin directory, which a virtual
environment's PATH does not carry. A driver that spawns the bare name then dies of
``FileNotFoundError`` raised out of :mod:`subprocess`, naming neither the binary nor
anywhere it was looked for, and the measurement it was in the middle of is lost.
Every driver in ``scripts/perf`` defaults to the bare name, so this is the default
path.

Resolution is a probe, not a fallback. A bare name is looked up on PATH and then in
each CUDA bin directory this host names; a name found nowhere raises
:class:`ToolNotFoundError` listing every path tried. Nothing here degrades to
running without the profiler: a pass that was never run must not read as clean, and
a driver whose profiler is missing has nothing to judge, so the error is the
outcome.

A spec that already carries a path separator is returned untouched. The caller named
a binary on this host, a search would run a different one, and the exec error names
the path it was given.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Final

__all__ = [
    "CUDA_BIN_GLOB",
    "CUDA_ROOT_VARS",
    "ToolNotFoundError",
    "cuda_bin_dirs",
    "resolve_tool",
]

CUDA_ROOT_VARS: Final[tuple[str, ...]] = ("CUDA_HOME", "CUDA_PATH")
"""Environment variables naming a toolkit root. Searched before anything guessed."""

CUDA_BIN_GLOB: Final = "/usr/local/cuda*/bin"
"""Where a toolkit installs by default, one directory per installed version.

Matches are searched in reverse name order, which is not version order:
``cuda-12.10`` sorts below ``cuda-12.3``. A host with two toolkits whose choice
matters therefore sets one of :data:`CUDA_ROOT_VARS` rather than relying on this.
"""


class ToolNotFoundError(FileNotFoundError):
    """A profiler binary is on neither PATH nor any CUDA bin directory.

    Subclasses :class:`FileNotFoundError`, so a caller already handling the errno
    :mod:`subprocess` would have raised handles this too.
    """


def cuda_bin_dirs() -> tuple[Path, ...]:
    """CUDA bin directories to search, most authoritative first, deduplicated.

    The declared roots come first, then the directory holding ``nvcc``, then the
    default install locations. A root that does not exist is dropped, so the list
    is what could hold a binary rather than what a variable claims.

    Returns:
        Existing directories, in search order.
    """
    found: list[Path] = []
    for var in CUDA_ROOT_VARS:
        root = os.environ.get(var)
        if root:
            found.append(Path(root) / "bin")
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        found.append(Path(nvcc).parent)
    found += [Path(one) for one in sorted(glob.glob(CUDA_BIN_GLOB), reverse=True)]
    out: list[Path] = []
    for one in found:
        if one.is_dir() and one not in out:
            out.append(one)
    return tuple(out)


def resolve_tool(spec: str) -> str:
    """Resolve a profiler binary name to a path this host can execute.

    Args:
        spec: Binary name, or a path to one. A spec containing a path separator is
            returned unchanged.

    Returns:
        The path to run. An absolute path when the probe found the binary, and
        ``spec`` itself when ``spec`` named a path.

    Raises:
        ToolNotFoundError: If a bare name is on neither PATH nor any directory
            :func:`cuda_bin_dirs` returned. The message lists every path tried,
            because a bare errno says only that something was missing.
    """
    if os.sep in spec or (os.altsep is not None and os.altsep in spec):
        return spec
    on_path = shutil.which(spec)
    if on_path is not None:
        return on_path
    tried: list[str] = []
    for directory in cuda_bin_dirs():
        candidate = directory / spec
        tried.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise ToolNotFoundError(
        f"{spec!r} is not on PATH and is not in any CUDA bin directory; tried "
        f"PATH={os.environ.get('PATH', '')!r} then {tried or ['no cuda bin directory']}"
        f"; pass an explicit path or set {CUDA_ROOT_VARS[0]}"
    )
