"""The payload builder: what one build covers, and whether an install carries it.

The tests in :mod:`tests.test_aot` prove that a capture compiles nothing when the
payload holds the key. This file covers the other half, which is who decides
whether it holds it. A builder narrower than the reachable set leaves a real process
compiling at every shape it did not build, and the failure is silent: dispatch falls
back, the graph records a trace it cannot replay, and nothing raises.

So the failure modes here are coverage and delivery. Coverage: a default width or
dtype list narrower than what the operator admits, a cell that produces no key, a
walk whose reach is set by a kernel other than the one it is building, and a
verification child that checks a cell other than the one it was given. Delivery: a
payload directory the package data does not name, which no installed distribution
carries however well the build ran.

The whole module is skipped without the DSL, including the assertions that are pure
Python, because importing the builder imports the DSL through :mod:`slinoss.aot`. The
wheel itself is not built here -- that needs the CUDA extension and a network-free
build backend -- so this file pins the declaration and the report carries the listing.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import io
import os
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import slinoss
from scripts.aot import payload as builder
from scripts.aot.payload import (
    DECODE_WIDTHS,
    DTYPES,
    Cell,
    _child_argv,
    _covers,
    cells,
    main,
    parse_args,
)
from slinoss import aot
from slinoss._precision import KERNEL_DTYPES
from slinoss.config import STATE_MULTIPLE

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

TINY = (
    "--layers",
    "1",
    "--d-model",
    "64",
    "--d-head",
    "16",
    "--batch",
    "1",
    "--prefill",
    "4",
    "--vocab",
    "128",
)
"""The smallest geometry a decode step runs at.

One layer of eight heads over one sequence. What is under test is which cells the
builder walks, and a cell's launch key carries no extent, so the geometry only has
to be legal.
"""

TWO = ("48", "384")
"""Two widths, one of each :func:`slinoss.ops.decode.cute.step.row_group` value.

``d_state`` 48 is ``N = 16`` and groups by half a warp; 384 is ``N = 128`` and groups
by a whole one. Two cells per dtype is the smallest list that distinguishes a builder
walking the list from one building the geometry it was given, and 384 is the rung a
walk that prefilled through the chunked scan cannot reach.
"""

SCRIPT = Path(builder.__file__).resolve()
"""The builder, as a path, for running it as a program."""


def _manifest(keys: tuple[str, ...]) -> aot.Manifest:
    """A manifest over textual keys alone.

    The coverage rule reads keys and nothing else, so a synthetic manifest can state
    a shortfall without a device, an export or an object file.

    Args:
        keys: The keys to hold.

    Returns:
        The manifest. Its prefixes and file names name nothing on disk.
    """
    entries = tuple(aot.Entry(key=key, prefix="p", file="p.o") for key in keys)
    return aot.Manifest(identity=aot.identity(), entries=entries)


class Built(NamedTuple):
    """One build of a two-width decode payload.

    Attributes:
        manifest: What the build wrote.
        printed: Its standard output.
        cells: The cells it was asked to cover.
        total: Bytes of object file on disk, summed over the entries.
    """

    manifest: aot.Manifest
    printed: str
    cells: tuple[Cell, ...]
    total: int


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Built]:
    """Build a decode payload over two widths and every kernel dtype.

    In a child process, not in this one. A manifest is written from the whole
    executor cache and a test session shares that cache, so an in-process build
    exports what earlier tests compiled alongside its own and the entry set stops
    being a measurement of one build -- in either direction, since a key already held
    is also a key this build does not add. A fresh process holds nothing, which is the
    case the counts read off the manifest are exact in.

    Module-scoped because it costs a process, and every fact read off it -- the entry
    set, the reported cost, the launchers it reached -- is that one run's.

    Yields:
        The :class:`Built`.
    """
    out = tmp_path_factory.mktemp("ladder") / "payload"
    argv = ["build", "--modes", "decode", "--widths", *TWO, "--no-verify"]
    argv += ["--out", str(out), *TINY]
    env = dict(os.environ)
    # The builder is run by path, so ``sys.path[0]`` is its own directory and the
    # tree root has to arrive some other way for ``import slinoss`` to find this
    # tree. Prepended, so it wins over an installed distribution of the same name.
    root = Path(slinoss.__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), *(p for p in [env.get("PYTHONPATH", "")] if p)]
    )
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = aot.read_manifest(out)
    total = sum((out / entry.file).stat().st_size for entry in manifest.entries)
    yield Built(manifest, done.stdout, cells("decode", parse_args(argv)), total)


def test_the_decode_ladder_defaults_to_the_whole_reachable_set() -> None:
    """The default must be the whole reachable set, never a useful subset.

    A payload is consulted by key, so a default that covers a fraction of the widths
    reinstates the compile at every other one, and reinstates it silently: the
    process runs, the fallback is a fallback, and only a strict load would have said
    so. The reachable set is what :class:`slinoss.config.SLinOSSConfig` admits for
    ``d_state`` up to the shared-memory bound and what the registry admits for an
    activation, so it is written here as those two rather than as a list.
    """
    assert tuple(DTYPES.values()) == KERNEL_DTYPES
    assert list(DECODE_WIDTHS) == [STATE_MULTIPLE * rung for rung in range(1, 9)]


def test_the_width_list_is_a_flag_and_defaults_to_the_whole_ladder() -> None:
    """A deployment that uses one width must be able to build one width.

    The full set is 27 entries and every cell is a compile, so a decode-only tree
    that runs at one ``d_state`` should not pay for the other seven. A flag rather
    than a narrower default: the flag costs a caller who knows their width nothing,
    and a narrower default costs everyone else correctness.
    """
    assert parse_args(["build", "--widths", "48", "96"]).widths == [48, 96]
    assert parse_args(["build"]).widths == list(DECODE_WIDTHS)


def test_a_decode_build_holds_one_forward_key_per_cell_and_one_carry_per_dtype(
    built: Built,
) -> None:
    """The entry set must be the cell set, exactly.

    ``decode_fwd`` is specialized on the activation dtype and on ``N``, through the
    ``(THREADS, row_group(N), N // row_group(N))`` it is compiled with, so it needs
    one entry per cell. ``decode_carry`` is compiled with ``(THREADS,)`` alone and
    reads only ``B`` and ``b_prev``, so it is specialized on the dtype and nothing
    else and one entry serves every width. Asserting both counts is what makes the
    claim a measurement: an equal total with the wrong split would mean a width
    reached no forward kernel.

    Equality rather than a bound, which the fixture's fresh process is what earns:
    the manifest of a process that built once holds that build's keys and no others.
    """
    assert len(built.cells) == len(TWO) * len(DTYPES)
    keys = [entry.key for entry in built.manifest.entries]
    forward = {key for key in keys if "decode_fwd" in key}
    carry = {key for key in keys if "decode_carry" in key}
    assert len(forward) == len(built.cells)
    assert len(carry) == len(DTYPES)


def test_the_build_reports_the_entry_count_and_the_payload_bytes(built: Built) -> None:
    """The cost of coverage must be on stdout, in bytes.

    Coverage is paid for in object files that ship inside the package, so the number
    a caller needs before choosing a width list is how large the payload is. KiB
    alone rounds; the byte total is what a wheel grows by.
    """
    assert f"entries        {len(built.manifest.entries)} " in built.printed
    assert f"{built.total:,} bytes" in built.printed


def test_a_build_short_a_decode_cell_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload short a decode cell must fail the build, not ship.

    Two halves, because either alone passes on a tree that does not close the
    failure: the rule refuses a manifest one forward key short, and the build submits
    the cells it walked to the rule. Without the second the rule is dead code and the
    shortfall surfaces later as a compile inside a capture.

    The manifests are synthetic. Nothing in the tree drops an entry on its own, and a
    real short export cannot be staged without also staging the export, while the
    rule reads keys alone. The spy counts rather than asserts inside the build, so a
    build that never calls the rule fails on an empty list rather than passing.
    """
    forward = tuple(f"m.decode_fwd|i128,i32,i4|d{name}" for name in DTYPES)
    carry = tuple(f"m.decode_carry|i128|d{name}" for name in DTYPES)
    short = len(DTYPES) - 1
    _covers(_manifest((*forward, *carry)), len(forward))
    with pytest.raises(RuntimeError, match=f"holds {short} decode_fwd"):
        _covers(_manifest((*forward[1:], *carry)), len(forward))
    with pytest.raises(RuntimeError, match=f"and {short} decode_carry"):
        _covers(_manifest((*forward, *carry[1:])), len(forward))

    submitted: list[int] = []
    monkeypatch.setattr(
        "scripts.aot.payload._covers",
        lambda _manifest, expected: submitted.append(expected),
    )
    argv = ["build", "--modes", "decode", "--widths", "48", "--no-verify"]
    argv += ["--out", str(tmp_path), *TINY]
    with redirect_stdout(io.StringIO()):
        assert main(argv) == 0
    assert submitted == [len(DTYPES)]


def test_a_decode_cell_never_runs_the_scan_so_the_ladder_reaches_384(
    built: Built, tmp_path: Path
) -> None:
    """The decode walk must not depend on the chunked scan, which is narrower.

    A decode step needs a state, and the obvious way to get a populated one is to
    prefill through the stack -- which runs the chunked scan, whose shared memory
    grows with ``d_state``. Measured on sm_89 at ``L 64 / P 64 / T 128``: 106,496 B
    asked against a 101,376 B capacity from ``d_state`` 336 up, so a prefilling walk
    raises two rungs below 384 and a decode payload loses the two widest widths its
    own kernel serves. A zero state reaches the same two launchers, because no launch
    key carries a value.

    Asserted on the launcher a key names rather than on the raise: the scan's shared
    budget is another lane's to move, and a walk that prefilled a narrow head would
    fit and still be wrong.
    """
    assert str(DECODE_WIDTHS[-1]) in TWO
    assert not [e for e in built.manifest.entries if "chunk_scan" in e.key]
    args = parse_args(["build", *TINY])
    cell = Cell(d_state=DECODE_WIDTHS[-1], dtype="bfloat16")
    assert "--no-prefill" in _child_argv("decode", args, tmp_path, cell)
    assert "--no-prefill" not in _child_argv("forward", args, tmp_path, cell)


def test_a_verification_child_carries_the_cell_it_verifies(tmp_path: Path) -> None:
    """A child must run the cell it was given, not the parent's geometry.

    Verification is the only thing that can tell a payload hit from a cache hit,
    since the parent holds every executor it compiled. A child that took the
    parent's ``--d-state`` and dtype would verify one cell as many times as there
    are cells and report the other twenty-three as covered.
    """
    args = parse_args(["build", *TINY])
    argv = _child_argv("decode", args, tmp_path, Cell(d_state=96, dtype="float16"))
    assert argv[argv.index("--d-state") + 1] == "96"
    assert argv[argv.index("--dtype") + 1] == "float16"
    assert argv[argv.index("--payload") + 1] == "strict"


def test_a_payload_is_declared_as_package_data_so_an_install_carries_it() -> None:
    """The build's output must be named by the package data, or no install has it.

    :data:`slinoss.aot.PAYLOAD_DIR` sits inside the package so that an installed
    distribution can carry a payload built before the wheel was made. That is a
    claim about ``pyproject.toml``, not about the path: a directory inside the
    package that no pattern names is copied by nothing, and the payload exists only
    in the tree it was built in. Both file kinds are checked, since a pattern that
    names the objects and not the manifest yields a payload that cannot be read.
    """
    root = Path(slinoss.__file__).resolve().parent.parent
    config = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["slinoss"]
    directory = aot.PAYLOAD_DIR.name
    for name in (aot.MANIFEST, "slinoss_0123456789abcdef.o"):
        candidate = f"{directory}/{name}"
        assert any(fnmatch.fnmatch(candidate, p) for p in patterns), candidate


def test_the_extension_sources_are_relative_so_a_wheel_can_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declaration no wheel build reaches delivers nothing.

    ``build_ext --inplace`` accepts an absolute source path and ``bdist_wheel``
    refuses one outright, so a tree can compile in place for years and still have no
    distribution to put a payload in. The paths must be relative to the directory
    ``setup.py`` sits in, which is where every build backend invokes it from.

    ``setup`` is stubbed and the module executed rather than run as a program: what is
    under test is the argument, and running it would compile. Only sources are
    checked, since that is the argument setuptools rejects; ``include_dirs`` may be
    absolute.
    """
    root = Path(slinoss.__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("slinoss_setup", root / "setup.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr("setuptools.setup", lambda **kwargs: None)
    spec.loader.exec_module(module)
    extensions: list[Any] = module.ext_modules
    # A CUDA host builds the extension, so an empty list here would mean the check
    # ran against nothing rather than that the tree is clean.
    assert extensions
    for extension in extensions:
        for source in extension.sources:
            assert not Path(source).is_absolute(), source
