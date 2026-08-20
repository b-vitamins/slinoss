"""Ahead-of-time compilation of the kernel tree.

A cold process compiles every executor a step needs before the step can run: the
trace, the MLIR passes and ptxas, once per launch key. ``cute.compile`` cannot
avoid it -- it sets ``no_cache`` unconditionally, so the DSL's own on-disk cache is
unreachable from here and every process pays the compile again.

This module is the payload that removes it. A build step runs one step of the
model, reads the executors that step compiled out of
:func:`slinoss._cute.compiled_launches`, and writes each one's relocatable object
alongside a manifest. A fresh process loads the manifest and hands
:func:`slinoss._cute.jit_launch` something to consult before it compiles.

    python3 scripts/aot/payload.py build      # build, into slinoss/_aot
    slinoss.aot.use()                         # load, at model start

The payload is identified by :func:`module_id`, a digest over everything that
changes the generated code: the source of every kernel module, the DSL version,
the CUDA version the DSL was built against, and the target architecture. A
payload whose id is not this tree's id is refused with the field that differs. It
is never silently ignored, because a stale payload runs the wrong kernel.

The torch version is deliberately not in the id. Torch supplies base pointers,
extents and strides through DLPack and the launch stream through its stream
handle; none of that reaches the trace, so no torch upgrade can change a
generated kernel.

Neither is the launch stream in a payload entry. Every launcher takes it as the
last of its runtime arguments and the exported object keeps it there, so one entry
serves every stream, which is what lets a captured graph replay a payload entry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any, Final, NamedTuple

import cutlass
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.base_dsl.version_info import CUDA_VERSION
from cutlass.cute.runtime import load_module

from slinoss._cute import Compiled, use_payload

__all__ = [
    "FORMAT",
    "KERNEL_ROOTS",
    "MANIFEST",
    "PAYLOAD_DIR",
    "Entry",
    "Identity",
    "Manifest",
    "Payload",
    "build",
    "current_arch",
    "identity",
    "launch_key",
    "load",
    "module_id",
    "read_manifest",
    "use",
]

PAYLOAD_DIR: Final = Path(__file__).resolve().parent / "_aot"
"""Where a payload lives by default. Inside the package, so an installed wheel
carries one if the build ran before the wheel was made, and gitignored, so a
built payload is never committed."""

MANIFEST: Final = "manifest.json"
"""The manifest's file name inside a payload directory."""

FORMAT: Final = 1
"""Manifest layout version. Bumped when a field's meaning changes, which is
separate from :func:`module_id`: the id says the kernels changed, this says the
reader changed."""

KERNEL_ROOTS: Final = ("_cute.py", "_reduce.py", "ops/*/cute")
"""What :func:`module_id` digests, relative to the package directory.

Every module that reaches a trace and nothing else. The two files are the shared
device-side helper set and the row reduction; the glob is every operator's kernel
package, which is where ``@cute.jit`` and ``@cute.kernel`` are allowed to live.
A host-side module -- a config, a guard, an autograd function -- shapes no
generated code and is left out, so editing one does not invalidate a payload.
"""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


_PACKAGE: Final = Path(__file__).resolve().parent
"""The package directory :data:`KERNEL_ROOTS` is relative to."""


def _kernel_files(root: Path = _PACKAGE) -> tuple[tuple[str, Path], ...]:
    """Every source file :func:`module_id` covers, as ``(relative, absolute)``.

    Args:
        root: Tree to walk. Not the package only so a test can build a digest over
            a tree it controls.

    Returns:
        The files, ordered by their relative path so the order does not depend on
        the filesystem's.
    """
    found: dict[str, Path] = {}
    for pattern in KERNEL_ROOTS:
        for path in root.glob(pattern):
            files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
            for file in files:
                found[file.relative_to(root).as_posix()] = file
    return tuple(sorted(found.items()))


def _source_digest(root: Path = _PACKAGE) -> str:
    """A digest over the kernel tree's source.

    The relative path goes in beside the bytes, so a module that moves changes the
    digest even when its text does not.

    Args:
        root: Tree to digest.

    Returns:
        A hex sha256.
    """
    digest = hashlib.sha256()
    for name, path in _kernel_files(root):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def current_arch() -> str:
    """The architecture the DSL will generate for.

    Read from the DSL rather than from the device, so ``CUTE_DSL_ARCH`` is
    honoured: a build for another architecture reports that architecture, and a
    host with no device reports the DSL's default rather than failing.

    Returns:
        An ``sm_*`` string.
    """
    return str(BaseDSL._get_dsl().envar.arch)


class Identity(NamedTuple):
    """What a payload was built under.

    Attributes:
        module_id: Digest over the four fields below. The one field a caller
            compares when it does not care which part moved.
        source_digest: Digest over the kernel tree's source.
        dsl_version: The CuTe DSL's version.
        cuda_version: The CUDA version the DSL was built against, ``major.minor``.
            It selects the PTX the DSL emits and the ptxas that assembles it.
        arch: The target architecture. An object file holds one architecture's
            cubin and nothing tells a wrong one apart at launch, so a mismatch is
            refused rather than attempted.
    """

    module_id: str
    source_digest: str
    dsl_version: str
    cuda_version: str
    arch: str


_CONTRIBUTING: Final = ("source_digest", "dsl_version", "cuda_version", "arch")
"""The fields :attr:`Identity.module_id` is a digest of, in message order.

Checked before ``module_id`` itself when a payload is refused: the id differs
whenever any of these does, so reporting it first would never name the cause.
"""


def identity(*, arch: str | None = None) -> Identity:
    """This tree's payload identity.

    Args:
        arch: Target architecture, or None for the DSL's own.

    Returns:
        The identity. Equal in two processes over the same source tree, DSL and
        architecture; different if one byte of one kernel module changes.
    """
    fields = (
        _source_digest(),
        str(cutlass.__version__),
        f"{CUDA_VERSION.major}.{CUDA_VERSION.minor}",
        current_arch() if arch is None else arch,
    )
    return Identity(hashlib.sha256("\0".join(fields).encode()).hexdigest(), *fields)


def module_id(*, arch: str | None = None) -> str:
    """:attr:`Identity.module_id` alone.

    Args:
        arch: Target architecture, or None for the DSL's own.

    Returns:
        A hex sha256.
    """
    return identity(arch=arch).module_id


# ---------------------------------------------------------------------------
# The textual launch key
# ---------------------------------------------------------------------------


def _term(value: object) -> str:
    """One key component as text.

    Each form carries a tag, so no two forms can collide: ``i1`` is the integer
    one and ``true`` is the boolean, and a payload built for one is not found by
    the other.

    Args:
        value: A compile-time argument, or one entry of a launch's declared
            signature.

    Returns:
        The text.

    Raises:
        TypeError: If the value has no text form. Not a miss: a launch key that
            cannot be written down cannot be in a payload at all, so the build
            would silently omit that kernel.
    """
    if value is None:
        return "none"
    # Before int: bool is a subclass of it, and a kernel specialized on a flag is
    # not the same kernel as one specialized on 0 or 1.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"i{value}"
    if isinstance(value, float):
        return f"f{value!r}"
    if isinstance(value, str):
        return f"s{value}"
    if isinstance(value, torch.dtype):
        return f"d{value}"
    if isinstance(value, type):
        return f"t{value.__module__}.{value.__qualname__}"
    if isinstance(value, tuple):
        return "(" + ",".join(_term(item) for item in value) + ")"
    raise TypeError(f"no text form for launch key component {value!r} ({type(value)})")


def _launcher_name(fn: Callable[..., None]) -> str:
    """The name a payload entry files a launcher under.

    Args:
        fn: The ``@cute.jit`` launcher.

    Returns:
        ``module.qualname``, which is unique: two launchers cannot share both.

    Raises:
        TypeError: If the callable carries neither, which would mean a launcher
            the payload cannot name.
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError(f"launcher {fn!r} has no module and qualified name to key on")
    return f"{module}.{qualname}"


def launch_key(
    fn: Callable[..., None],
    static: tuple[Hashable, ...],
    signature: tuple[Hashable, ...],
) -> str:
    """The textual form of a launch key.

    :func:`slinoss._cute.jit_launch` keys its executor cache on a function object,
    the compile-time arguments, the device and what the runtime arguments
    declared. A function object does not serialize and the device shapes no code,
    so this is the same key written down: the launcher's qualified name, the
    compile-time arguments, and the declared signature.

    The mapping is total. Every component of every key this tree builds has a text
    form, and one that does not raises rather than returning a value that would
    read as a payload miss.

    Args:
        fn: The ``@cute.jit`` launcher.
        static: Its compile-time arguments, in order.
        signature: What its runtime arguments declared.

    Returns:
        The key.

    Raises:
        TypeError: If any component has no text form.
    """
    return "|".join(
        (
            _launcher_name(fn),
            ",".join(_term(item) for item in static),
            ",".join(_term(item) for item in signature),
        )
    )


def _prefix(key: str) -> str:
    """The symbol prefix one entry's object file is exported under.

    A C identifier, since it prefixes every symbol in the object, and derived from
    the key so two entries of one launcher cannot collide.

    Args:
        key: The textual launch key.

    Returns:
        The prefix.
    """
    return "slinoss_" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


class Entry(NamedTuple):
    """One compiled launcher in a payload.

    Attributes:
        key: The textual launch key, which is what a launch is matched on.
        prefix: The symbol prefix the object was exported under. Also the name the
            loaded module is asked for.
        file: The object file, relative to the payload directory.
    """

    key: str
    prefix: str
    file: str


class Manifest(NamedTuple):
    """A payload's index.

    Attributes:
        identity: What the payload was built under.
        entries: Its entries, ordered by key.
    """

    identity: Identity
    entries: tuple[Entry, ...]


def _read_manifest(root: Path) -> Manifest:
    """Parse a payload's manifest.

    Args:
        root: The payload directory.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If the directory holds no manifest.
        ValueError: If the manifest is not this reader's format, or is missing a
            field. A truncated manifest is a failed build, not an empty payload.
    """
    path = root / MANIFEST
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"no payload manifest at {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload manifest {path} is not JSON: {exc}") from exc
    if raw.get("format") != FORMAT:
        raise ValueError(
            f"payload manifest {path} is format {raw.get('format')!r} and this "
            f"reader is format {FORMAT}"
        )
    try:
        return Manifest(
            identity=Identity(**raw["identity"]),
            entries=tuple(Entry(**entry) for entry in raw["entries"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"payload manifest {path} is missing a field: {exc}") from exc


def _write_manifest(root: Path, manifest: Manifest) -> None:
    """Write a payload's manifest.

    Args:
        root: The payload directory, which must exist.
        manifest: What to write.
    """
    body = {
        "format": FORMAT,
        "identity": manifest.identity._asdict(),
        "entries": [entry._asdict() for entry in manifest.entries],
    }
    (root / MANIFEST).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(
    launches: Iterable[Compiled],
    *,
    path: Path | str = PAYLOAD_DIR,
) -> Manifest:
    """Export every launch in ``launches`` and write the manifest.

    The compile has already happened: a ``Compiled`` holds the executor
    ``cute.compile`` returned, and exporting it writes that exact code rather
    than a second compile that might not agree with it.

    Two launches whose textual keys agree are one entry. The device index is not
    in a textual key, so a process that ran the same work on two devices produces
    one payload.

    Stale object files in the directory are removed, so a rebuild after a kernel
    loses a specialization does not leave the old one to be loaded by a manifest
    that no longer names it.

    Args:
        launches: What to export, from :func:`slinoss._cute.compiled_launches`.
        path: Payload directory. Created if absent.

    Returns:
        The manifest written.

    Raises:
        ValueError: If a launch's executor came from a payload rather than a
            compile. It holds no IR to export, and exporting one would write an
            object the next build could not reproduce.
        TypeError: If a launch's key has no text form.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Entry] = {}
    for launch in launches:
        key = launch_key(launch.fn, launch.static, launch.signature)
        if key in entries:
            continue
        if getattr(launch.executor, "load_from_binary", False):
            raise ValueError(
                f"the executor for {key} was loaded from a payload and holds no IR "
                f"to export; build from a process that compiled it"
            )
        prefix = _prefix(key)
        entry = Entry(key=key, prefix=prefix, file=f"{prefix}.o")
        (root / entry.file).write_bytes(launch.executor.dump_to_object(prefix))
        entries[key] = entry
    manifest = Manifest(
        identity=identity(), entries=tuple(entries[key] for key in sorted(entries))
    )
    kept = {entry.file for entry in manifest.entries}
    for stale in root.glob("*.o"):
        if stale.name not in kept:
            stale.unlink()
    _write_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class Payload:
    """A loaded payload, as :func:`slinoss._cute.jit_launch` consults it.

    An entry's object file is read on the first launch that wants it, not at load
    time: a decode-only process pays for the decode kernels alone. Reading one is
    a JIT link of a single relocatable object with no ptxas and no trace.

    Attributes:
        strict: Raise on a key this payload does not hold instead of compiling it.
            The mode a benchmark runs in, where a fallback compile would be a
            silent 100x on the first step.
    """

    __slots__ = ("_entries", "_modules", "_root", "identity", "strict")

    def __init__(self, root: Path, manifest: Manifest, *, strict: bool = False) -> None:
        """Hold a manifest's entries.

        Args:
            root: The payload directory.
            manifest: Its manifest, already checked against this tree.
            strict: See :attr:`strict`.
        """
        self._root = root
        self._entries = {entry.key: entry for entry in manifest.entries}
        self._modules: dict[str, Any] = {}
        self.identity = manifest.identity
        self.strict = strict

    def keys(self) -> tuple[str, ...]:
        """The textual launch keys this payload holds, ordered."""
        return tuple(sorted(self._entries))

    def __len__(self) -> int:
        """Entries held."""
        return len(self._entries)

    def lookup(
        self,
        fn: Callable[..., None],
        static: tuple[Hashable, ...],
        signature: tuple[Hashable, ...],
    ) -> Any | None:
        """The entry for one launch key, or None if this payload holds none.

        A fresh compiled function per call, from a module cached per file: the
        caller keys what it gets back on the device as well, and one function
        object binds a default executor to the first device it runs on.

        Args:
            fn: The ``@cute.jit`` launcher.
            static: Its compile-time arguments, in order.
            signature: What its runtime arguments declared.

        Returns:
            Something callable on the dynamic arguments alone, or None.

        Raises:
            TypeError: If the key has no text form.
        """
        entry = self._entries.get(launch_key(fn, static, signature))
        if entry is None:
            return None
        module = self._modules.get(entry.file)
        if module is None:
            module = self._modules[entry.file] = load_module(
                str(self._root / entry.file)
            )
        return getattr(module, entry.prefix)


def _refuse(built: Identity, live: Identity) -> str | None:
    """The first identity field that differs, as a message, or None.

    Args:
        built: What the payload was built under.
        live: What this tree is.

    Returns:
        The message, or None if the two agree.
    """
    for field in (*_CONTRIBUTING, "module_id"):
        was, now = getattr(built, field), getattr(live, field)
        if was != now:
            return f"{field} is {was} in the payload and {now} here"
    return None


def load(
    path: Path | str = PAYLOAD_DIR,
    *,
    strict: bool = False,
) -> Payload:
    """Read a payload and check it against this tree.

    Args:
        path: Payload directory.
        strict: See :attr:`Payload.strict`.

    Returns:
        The payload.

    Raises:
        FileNotFoundError: If the directory holds no manifest.
        ValueError: If the manifest is unreadable, or was built under a different
            identity. The message names the field that differs.
    """
    root = Path(path)
    manifest = _read_manifest(root)
    # Against this tree's own architecture, not the payload's: an object file holds
    # one architecture's cubin, so comparing a payload's arch against itself would
    # accept a payload that cannot launch here.
    difference = _refuse(manifest.identity, identity())
    if difference is not None:
        raise ValueError(
            f"the payload at {root} does not match this tree: {difference}"
        )
    return Payload(root, manifest, strict=strict)


def use(
    path: Path | str = PAYLOAD_DIR,
    *,
    strict: bool = False,
) -> Payload | None:
    """Load the payload at ``path`` and install it for this process.

    Args:
        path: Payload directory.
        strict: See :attr:`Payload.strict`.

    Returns:
        The payload, or None if the directory holds no manifest. A tree that ships
        no payload starts with no error and compiles as it always did; a payload
        that is present and does not match raises, because a stale payload is a
        wrong kernel rather than a missing one.

    Raises:
        ValueError: If a manifest is present and does not match this tree.
    """
    try:
        payload = load(path, strict=strict)
    except FileNotFoundError:
        return None
    use_payload(payload)
    return payload


def read_manifest(path: Path | str = PAYLOAD_DIR) -> Manifest:
    """Read a payload's manifest without checking it or loading an entry.

    What a build reports and a CI job inspects: the identity written and the keys
    held, off a host that cannot load the objects.

    Args:
        path: Payload directory.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If the directory holds no manifest.
        ValueError: If the manifest is unreadable.
    """
    return _read_manifest(Path(path))
