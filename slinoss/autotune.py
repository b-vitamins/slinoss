"""Launch-geometry selection from measurement, cached per user.

A kernel declares the launch geometries it can run as a :class:`Variants` and
resolves one per call site. With no cache the resolution returns the geometry the
kernel declared as its default, so an untuned tree launches exactly what it
launched before this module existed. Nothing here measures anything: the only
writer of a record is ``scripts/perf/tune.py``, which measures on the device in
front of it and stores what it measured.

Three concerns are separated:

- which geometries a kernel can run -- declared by the kernel, in code, next to
  the launch;
- which of them is fastest at one shape on one part -- measured, never inferred;
- where that measurement is kept -- a file the user owns, read once per process
  and ignored whenever it cannot be trusted.

Key. A record is addressed by the kernel name, a canonicalized shape, and the
device identity. Device identity is the marketing name, the compute capability
and the SM count. Those three fix the occupancy arithmetic a launch is chosen
against and nothing else about a part does, so one measurement covers every
identical part in a host and a second slot never re-measures it. A bus id or an
ordinal is deliberately absent, for the same reason.

Shape key. Three axes, because three are what move the answer:

- ``width``, exact. It is a compile-time argument of every block kernel and sets
  the per-thread segment count, hence the register footprint.
- ``rows``, rounded up to a power of two. A grid-strided launch takes
  ``min(rows, capacity)`` blocks, so only the ratio of rows to device capacity
  moves the choice; the exact count would give ``T = 2004`` and ``T = 2048``
  separate records holding one answer.
- ``itemsize``, exact. It sets the bytes one vector covers and the traffic per
  row. Two dtypes of one width are one entry, because the geometry question they
  ask is the same one.

Excluded from the key, and why. Batch and sequence separately: every block kernel
is launched over a flattened ``(rows, width)`` and neither extent is visible to
it. The epsilon and the cotangent-presence flags: they close halves of an
expression at compile time without changing the launch shape, and a variant they
do change is a different kernel name. The device ordinal: see above. The backend
name: a geometry is declared by one backend's kernel, so the kernel name already
carries it.

Trust. A stored geometry is used only if it is one of the variants the kernel
declares today. That single check subsumes every way a file can go stale or
wrong: a hand-edited value, an older schema, a geometry a rewritten kernel no
longer supports, and a record written against a different device all fail it and
fall back to the default. A missing file, an unreadable one, a truncated one and
one carrying another schema version are ignored rather than raised on, because a
tuning cache that can break a forward pass is worse than no tuning cache.

Cost. :meth:`Variants.select` is one tuple construction and one dict lookup on
the steady-state path, whatever the cache holds; the file is read at most once
per process, on the first miss, and never at import.

This module imports no part of :mod:`slinoss.perf`. The lookup runs on the launch
path and the measurement harness does not belong there; the tuning driver holds
both and converts between them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Final, Generic, NamedTuple, TypeVar, cast

import torch

__all__ = [
    "AGREEMENT_TOL",
    "CACHE_ENV",
    "CACHE_NAME",
    "SCHEMA",
    "Attempt",
    "DeviceKey",
    "Record",
    "ShapeKey",
    "Variants",
    "cache_path",
    "device_key",
    "install",
    "is_loaded",
    "load",
    "merge",
    "pinned",
    "records",
    "register",
    "registered",
    "reset",
    "save",
    "versions",
    "witnessed",
]

SCHEMA: Final = 1
"""Cache file format version. A file carrying any other value is ignored.

Bumped whenever a field changes meaning, and whenever a kernel body changes
enough that a duration measured against the old one no longer describes the new
one. The candidate check below catches a geometry a kernel dropped; it cannot
catch a kernel that got faster at the same geometry, and this is the lever for
that. Ignoring is the whole migration strategy: a record is a measurement that
can be taken again in seconds, so reading an old one wrongly costs more than
dropping it.
"""

CACHE_ENV: Final = "SLINOSS_TUNING_CACHE"
"""Environment variable naming the cache file. Overrides the default location."""

CACHE_NAME: Final = "tuning.json"
"""File name under the per-user cache directory."""

_CACHE_DIR_ENV: Final = "XDG_CACHE_HOME"

AGREEMENT_TOL: Final = {
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-10,
    torch.float32: 4e-6,
}
"""Largest relative disagreement a variant may show against its kernel's default.

A geometry says which thread reaches which element and, in a reducing kernel, the
summation order of the block reduction. So two variants of one kernel agree to the
tolerance of a float32 reduction and not bit for bit, and the bound at each operand
dtype is one narrowing step of it. At float32 the reduction order is the whole of
the difference and the bound is wider, which is what a 32-warp block's reduction
tree against a 4-warp block's needs.

Declared here and not in the driver: the tuner refuses a candidate that exceeds
this and ``tests/test_autotune.py`` holds every declared variant to it, and two
tables of one bound would drift.
"""


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _bucket(rows: int) -> int:
    """``rows`` rounded up to a power of two, at least one.

    Args:
        rows: Rows on the flattened axis.

    Returns:
        The bucket.
    """
    return 1 if rows <= 1 else 1 << (rows - 1).bit_length()


class ShapeKey(NamedTuple):
    """The canonicalized problem geometry a record answers for.

    Attributes:
        rows: Rows on the flattened axis, rounded up to a power of two.
        width: Trailing extent, exact.
        itemsize: Bytes of one operand element, exact.
    """

    rows: int
    width: int
    itemsize: int

    @classmethod
    def of(cls, rows: int, width: int, itemsize: int) -> ShapeKey:
        """Canonicalize a call's geometry.

        Args:
            rows: Rows on the flattened axis, as launched.
            width: Trailing extent.
            itemsize: Bytes of one operand element.

        Returns:
            The key.
        """
        return cls(rows=_bucket(rows), width=width, itemsize=itemsize)

    @property
    def text(self) -> str:
        """One field for a report line or a file."""
        return f"rows<={self.rows} width={self.width} itemsize={self.itemsize}"


class DeviceKey(NamedTuple):
    """The part a record was measured on.

    Attributes:
        name: Marketing name, as the driver reports it.
        capability: Compute capability, ``major.minor``.
        sm_count: Streaming multiprocessors.
    """

    name: str
    capability: str
    sm_count: int

    @property
    def text(self) -> str:
        """One field for a report line or a file."""
        return f"{self.name} sm_{self.capability.replace('.', '')} x{self.sm_count}"


@cache
def device_key(index: int) -> DeviceKey:
    """Identity of one CUDA device. Cached: the lookup path may not fork.

    The three properties are the ones :class:`slinoss.perf.device.DeviceInfo`
    reports, read from the same source. The clock and sharing probes it also runs
    are subprocesses, which is why this reads the properties directly rather than
    calling it: a resolution on the launch path cannot afford ``nvidia-smi``.

    Args:
        index: CUDA device ordinal.

    Returns:
        The identity.
    """
    props = torch.cuda.get_device_properties(index)
    return DeviceKey(
        name=props.name,
        capability=f"{props.major}.{props.minor}",
        sm_count=int(props.multi_processor_count),
    )


def versions() -> tuple[str, str]:
    """Torch and CuTe DSL versions, for a record's provenance.

    The DSL import is here rather than at module scope: this module is on the
    launch path of a tree that may have no DSL at all, and only the tuning driver
    needs the string.

    Returns:
        The torch version, and the ``cutlass`` version or ``unavailable``.
    """
    try:
        import cutlass
    except Exception:
        # Every way the DSL can fail to import is one fact here: it is absent.
        return str(torch.__version__), "unavailable"
    return str(torch.__version__), str(getattr(cutlass, "__version__", "unknown"))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """One geometry as the tuner found it.

    Durations are microseconds. No wall-clock time is stored anywhere in a
    record: on a host whose clock is not pinned a timestamp reads as authority it
    does not have, and the spread of the repeats is the fact that bounds a claim.

    Attributes:
        geometry: The geometry, in the field order of the kernel's geometry type.
        median_duration_us: Median of the repeats.
        min_duration_us: Fastest repeat.
        max_duration_us: Slowest repeat.
        samples_duration_us: The repeats themselves, in measurement order, so the
            three statistics above can be recomputed and checked.
        note: Why this geometry is not the winner, when the reason is not its
            duration. Empty for a geometry that ran.
    """

    geometry: tuple[int, ...]
    median_duration_us: float = 0.0
    min_duration_us: float = 0.0
    max_duration_us: float = 0.0
    samples_duration_us: tuple[float, ...] = ()
    note: str = ""

    @classmethod
    def of(cls, geometry: Sequence[int], samples: Sequence[float]) -> Attempt:
        """Build an attempt from the repeats that were run.

        Args:
            geometry: The geometry measured.
            samples: Per-repeat durations in microseconds, measurement order. At
                least one.

        Returns:
            The attempt.

        Raises:
            ValueError: If no sample was supplied. An attempt with no measurement
                behind it is not a measurement.
        """
        if not samples:
            raise ValueError(
                f"attempt at geometry {tuple(geometry)} has no samples; a record "
                f"with no measurement behind it is not a record"
            )
        ordered = sorted(samples)
        count = len(ordered)
        middle = count // 2
        median = (
            ordered[middle]
            if count % 2 == 1
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        return cls(
            geometry=tuple(geometry),
            median_duration_us=median,
            min_duration_us=ordered[0],
            max_duration_us=ordered[-1],
            samples_duration_us=tuple(samples),
        )

    @classmethod
    def refused(cls, geometry: Sequence[int], note: str) -> Attempt:
        """Build an attempt for a geometry that did not run.

        Args:
            geometry: The geometry.
            note: What refused it, verbatim.

        Returns:
            The attempt, with no durations.
        """
        return cls(geometry=tuple(geometry), note=note)

    @property
    def measured(self) -> bool:
        """Whether any repeat of this geometry ran."""
        return len(self.samples_duration_us) > 0


@dataclass(frozen=True)
class Record:
    """The measured winner at one key, and what it beat.

    Attributes:
        kernel: Kernel name, as registered.
        shape: The canonicalized shape.
        device: The part it was measured on.
        winner: The selected geometry. Must carry samples.
        runners_up: The next-fastest candidates, fastest first. A refused candidate
            carries its refusal instead of durations. A writer may keep fewer than
            it probed, which is why ``probe_count`` is separate: a truncated list
            must not read as the whole sweep.
        repeat_count: Repeats per candidate.
        probe_count: Candidates probed, including the winner and any that refused
            to launch.
        torch_version: Torch version at measurement.
        cutlass_version: CuTe DSL version at measurement.
        conditions: Clock and sharing state at measurement, verbatim. Both bound
            what the durations mean and neither is under the tuner's control.
    """

    kernel: str
    shape: ShapeKey
    device: DeviceKey
    winner: Attempt
    repeat_count: int
    probe_count: int = 0
    torch_version: str = ""
    cutlass_version: str = ""
    conditions: str = ""
    runners_up: tuple[Attempt, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a record with nothing measured behind it.

        Raises:
            ValueError: If the winner carries no samples, or if the repeat count
                is not positive.
        """
        if not self.winner.measured:
            raise ValueError(
                f"record for {self.kernel} at {self.shape.text} has no samples "
                f"behind its winner; a record with no measurement is not a record"
            )
        if self.repeat_count <= 0:
            raise ValueError(
                f"record for {self.kernel} at {self.shape.text} claims "
                f"{self.repeat_count} repeats"
            )

    @property
    def key(self) -> tuple[DeviceKey, str, ShapeKey]:
        """What this record is addressed by."""
        return (self.device, self.kernel, self.shape)


def _attempt_payload(attempt: Attempt) -> dict[str, Any]:
    return {
        "geometry": list(attempt.geometry),
        "median_duration_us": attempt.median_duration_us,
        "min_duration_us": attempt.min_duration_us,
        "max_duration_us": attempt.max_duration_us,
        "samples_duration_us": list(attempt.samples_duration_us),
        "note": attempt.note,
    }


def _record_payload(record: Record) -> dict[str, Any]:
    return {
        "kernel": record.kernel,
        "shape": {
            "rows": record.shape.rows,
            "width": record.shape.width,
            "itemsize": record.shape.itemsize,
        },
        "device": {
            "name": record.device.name,
            "capability": record.device.capability,
            "sm_count": record.device.sm_count,
        },
        "winner": _attempt_payload(record.winner),
        "runners_up": [_attempt_payload(a) for a in record.runners_up],
        "repeat_count": record.repeat_count,
        "probe_count": record.probe_count,
        "torch_version": record.torch_version,
        "cutlass_version": record.cutlass_version,
        "conditions": record.conditions,
    }


def _attempt_of(data: Any) -> Attempt:
    """Read one attempt, or raise. Callers treat any raise as an unusable file."""
    samples = tuple(float(v) for v in data["samples_duration_us"])
    attempt = Attempt(
        geometry=tuple(int(v) for v in data["geometry"]),
        median_duration_us=float(data["median_duration_us"]),
        min_duration_us=float(data["min_duration_us"]),
        max_duration_us=float(data["max_duration_us"]),
        samples_duration_us=samples,
        note=str(data.get("note", "")),
    )
    if not samples:
        return attempt
    # The three statistics are derived, so a file whose stored ones disagree with
    # its own samples was edited by hand and is not a measurement any more.
    rebuilt = Attempt.of(attempt.geometry, samples)
    if (
        rebuilt.median_duration_us != attempt.median_duration_us
        or rebuilt.min_duration_us != attempt.min_duration_us
        or rebuilt.max_duration_us != attempt.max_duration_us
    ):
        raise ValueError(
            f"attempt at geometry {attempt.geometry} states statistics its own "
            f"samples do not support"
        )
    return attempt


def _record_of(data: Any) -> Record:
    """Read one record, or raise. Callers treat any raise as one bad record."""
    shape = data["shape"]
    device = data["device"]
    return Record(
        kernel=str(data["kernel"]),
        shape=ShapeKey(
            rows=int(shape["rows"]),
            width=int(shape["width"]),
            itemsize=int(shape["itemsize"]),
        ),
        device=DeviceKey(
            name=str(device["name"]),
            capability=str(device["capability"]),
            sm_count=int(device["sm_count"]),
        ),
        winner=_attempt_of(data["winner"]),
        repeat_count=int(data["repeat_count"]),
        probe_count=int(data.get("probe_count", 0)),
        torch_version=str(data.get("torch_version", "")),
        cutlass_version=str(data.get("cutlass_version", "")),
        conditions=str(data.get("conditions", "")),
        runners_up=tuple(_attempt_of(a) for a in data.get("runners_up", ())),
    )


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def cache_path() -> Path:
    """Where the tuning cache lives.

    Never inside the package: a tuning run is the user's measurement of the
    user's device, and a package directory is read-only in a Guix profile and
    shared between projects besides.

    Read from the environment on every call rather than cached, so a test or a
    session can point at its own file without reloading the module.

    Returns:
        :data:`CACHE_ENV` if it is set and non-empty, else
        ``$XDG_CACHE_HOME/slinoss/tuning.json``, else
        ``~/.cache/slinoss/tuning.json``.
    """
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    root = os.environ.get(_CACHE_DIR_ENV)
    base = Path(root).expanduser() if root else Path.home() / ".cache"
    return base / "slinoss" / CACHE_NAME


def load(path: Path | None = None) -> tuple[Record, ...]:
    """Read a cache file. Never raises.

    Every failure mode is one outcome, an empty result: no file, no permission,
    a truncated write, a schema this build does not know, a record missing a
    field. The alternative is a forward pass that fails because of a cache, which
    is worse than no cache.

    Args:
        path: File to read, or None for :func:`cache_path`.

    Returns:
        The records it holds, in file order. Empty if it holds none or could not
        be read.
    """
    target = cache_path() if path is None else path
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return ()
    try:
        data = json.loads(text)
    except ValueError:
        return ()
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return ()
    raw = data.get("records")
    if not isinstance(raw, list):
        return ()
    out: list[Record] = []
    for item in raw:
        try:
            out.append(_record_of(item))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    return tuple(out)


def save(entries: Iterable[Record], path: Path | None = None) -> Path:
    """Write a cache file.

    Written to a neighbouring temporary file and renamed, so a reader never sees
    half a file and a failed write leaves the previous one in place.

    Args:
        entries: Records to write, in the order they should be read back.
        path: File to write, or None for :func:`cache_path`.

    Returns:
        The path written.
    """
    target = cache_path() if path is None else path
    payload = {
        "schema": SCHEMA,
        "records": [_record_payload(r) for r in entries],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(f"{target.name}.partial")
    scratch.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(scratch, target)
    return target


def merge(existing: Iterable[Record], fresh: Iterable[Record]) -> tuple[Record, ...]:
    """Combine a loaded cache with a fresh tuning run.

    A fresh record replaces an existing one at the same key and keeps its
    position, so re-tuning one shape does not reorder the file or drop the
    shapes that were not re-measured.

    Args:
        existing: Records already on disk.
        fresh: Records this run measured.

    Returns:
        The merged records.
    """
    replacement = {r.key: r for r in fresh}
    out: list[Record] = []
    for record in existing:
        out.append(replacement.pop(record.key, record))
    out.extend(replacement.values())
    return tuple(out)


# ---------------------------------------------------------------------------
# Process state
# ---------------------------------------------------------------------------

_MEMO: dict[tuple[str, int, int, int, int], tuple[int, ...]] = {}
"""Resolved geometry per call site and raw shape. The steady-state path.

Keyed on the raw arguments rather than the canonical key, so a hit costs one
tuple construction and one lookup and the canonicalization is paid once per
distinct call shape.
"""

_PINNED: dict[str, tuple[int, ...]] = {}
_TABLE: dict[tuple[DeviceKey, str, ShapeKey], Record] | None = None
_VARIANTS: dict[str, Variants[Any]] = {}
_WITNESS: list[tuple[str, ShapeKey]] | None = None
"""Where :func:`witnessed` collects. None outside it, which is the whole of its
cost on the steady-state path."""


def _index(entries: Iterable[Record]) -> dict[tuple[DeviceKey, str, ShapeKey], Record]:
    """Address records by key. A later record wins, so a file can be appended to."""
    return {record.key: record for record in entries}


def _table() -> dict[tuple[DeviceKey, str, ShapeKey], Record]:
    """The loaded cache, reading the file on first use.

    Not at import: a process that never launches a tuned kernel must not touch
    the filesystem for one, and an import that reads a file cannot be tested for
    not reading it.

    Returns:
        The index. Empty when there is no usable file.
    """
    global _TABLE
    if _TABLE is None:
        _TABLE = _index(load())
    return _TABLE


def is_loaded() -> bool:
    """Whether the cache file has been read in this process.

    False right after import, and after :func:`reset`. This is what makes the
    absence of import-time file IO testable.

    Returns:
        True once the file has been read or a table installed.
    """
    return _TABLE is not None


def records() -> tuple[Record, ...]:
    """The records in force, loading the file if it has not been read yet.

    Returns:
        The records, in key order of first appearance.
    """
    return tuple(_table().values())


def install(entries: Iterable[Record]) -> None:
    """Replace the in-force records without touching the file.

    Args:
        entries: The records to use.
    """
    global _TABLE
    _TABLE = _index(entries)
    _MEMO.clear()


def reset() -> None:
    """Drop every cached decision, the loaded table, and any pin.

    The next resolution reads the file again.
    """
    global _TABLE
    _TABLE = None
    _MEMO.clear()
    _PINNED.clear()


@contextmanager
def witnessed() -> Iterator[list[tuple[str, ShapeKey]]]:
    """Collect the keys resolved inside the block, in resolution order.

    Which shape a kernel resolves at is the kernel's own arithmetic, not the
    caller's: the parameter-gradient tail is launched over the row count of the
    grid that produced its partials, which is a function of the device and of the
    geometry the reducing kernel ran at. A driver that derived the key itself
    would write records at addresses no call ever looks up. So it runs the call
    once and reads the addresses back.

    Only the miss path records, and the memo is cleared on entry so every call
    site in the block misses once. A second call of the same shape inside the
    block adds nothing.

    Yields:
        The list, filled as the block runs. One entry per resolution, so a kernel
        launched at two shapes appears twice.
    """
    global _WITNESS
    previous = _WITNESS
    seen: list[tuple[str, ShapeKey]] = []
    _WITNESS = seen
    _MEMO.clear()
    try:
        yield seen
    finally:
        _WITNESS = previous
        _MEMO.clear()


@contextmanager
def pinned(choices: Mapping[str, Sequence[int]]) -> Iterator[None]:
    """Force a geometry per kernel for the duration of the block.

    How the tuner measures a candidate: the pin is read once, on the same miss
    path a cache hit takes, and the answer is memoized, so a loop inside the
    block resolves exactly as production does and the measurement is not the
    measurement of this context manager.

    A pinned geometry is still checked against the kernel's declared variants, so
    a typo in a driver is a fallback to the default and not an unheld launch.

    Args:
        choices: Geometry per kernel name.

    Yields:
        None.
    """
    previous = dict(_PINNED)
    _PINNED.update({name: tuple(int(v) for v in g) for name, g in choices.items()})
    _MEMO.clear()
    try:
        yield
    finally:
        _PINNED.clear()
        _PINNED.update(previous)
        _MEMO.clear()


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

G = TypeVar("G", bound=tuple[int, ...])


@dataclass(frozen=True)
class Variants(Generic[G]):
    """Every launch geometry one kernel can run, and which one it runs by default.

    A kernel constructs one of these next to its host wrapper and calls
    :meth:`select` where it used to read a geometry constant. Registering it is
    what makes it visible to ``scripts/perf/tune.py``.

    The default must be one of the candidates: a default outside the declared set
    is a set that does not describe the kernel, and it would also make the
    tuner's baseline unreachable from a cache.

    Attributes:
        kernel: Name, unique across the tree. The tuner's ``--kernel`` argument
            and the record key both use it.
        default: The geometry the kernel launches with no cache entry. Exactly
            what it launched before it was tunable.
        candidates: Every geometry the kernel is allowed to launch, the default
            among them.
    """

    kernel: str
    default: G
    candidates: tuple[G, ...]
    _allowed: dict[tuple[int, ...], G] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    """Declared candidate by its axis values. A stored geometry is looked up here
    rather than rebuilt, so a kernel's geometry type needs no constructor
    contract."""

    def __post_init__(self) -> None:
        """Check the declared set describes the kernel.

        Raises:
            ValueError: If there are no candidates, if the default is not one of
                them, or if a candidate has a different arity from the default,
                which would let a stored record fill the wrong axis.
        """
        if not self.candidates:
            raise ValueError(f"{self.kernel} declares no candidate geometries")
        width = len(self.default)
        odd = [c for c in self.candidates if len(c) != width]
        if odd:
            raise ValueError(
                f"{self.kernel} declares candidates of {sorted({len(c) for c in odd})} "
                f"axes against a default of {width}"
            )
        if self.default not in self.candidates:
            raise ValueError(
                f"{self.kernel} default {tuple(self.default)} is not one of its "
                f"candidates; the tuner's baseline would be unreachable"
            )
        object.__setattr__(
            self, "_allowed", {tuple(int(v) for v in c): c for c in self.candidates}
        )

    def select(self, rows: int, width: int, itemsize: int, index: int) -> G:
        """The geometry to launch this call with.

        One tuple construction and one dict lookup once a shape has been seen. On
        the first sight of a shape it consults the pin, then the cache, then falls
        back to the default; the file is read at most once per process.

        Args:
            rows: Rows on the flattened axis, as launched.
            width: Trailing extent.
            itemsize: Bytes of one operand element.
            index: CUDA device ordinal.

        Returns:
            The geometry, in the kernel's own geometry type.
        """
        key = (self.kernel, rows, width, itemsize, index)
        choice = _MEMO.get(key)
        if choice is None:
            choice = _MEMO.setdefault(key, self._resolve(rows, width, itemsize, index))
        return cast("G", choice)

    def admits(self, geometry: Sequence[int]) -> G | None:
        """The declared candidate ``geometry`` names, or None.

        Args:
            geometry: Axis values, in the field order of the geometry type.

        Returns:
            The candidate the kernel declared, as it declared it, or None if this
            kernel does not declare it. The declared object rather than a rebuilt
            one: a geometry read out of a file is a list of integers, and only the
            declaration knows what type carries them.
        """
        return self._allowed.get(tuple(int(v) for v in geometry))

    def _resolve(self, rows: int, width: int, itemsize: int, index: int) -> G:
        """Consult the pin, then the cache. Falls back to the default."""
        if _WITNESS is not None:
            _WITNESS.append((self.kernel, ShapeKey.of(rows, width, itemsize)))
        pin = _PINNED.get(self.kernel)
        if pin is not None:
            admitted = self.admits(pin)
            if admitted is not None:
                return admitted
        table = _table()
        if not table:
            return self.default
        found = table.get(
            (device_key(index), self.kernel, ShapeKey.of(rows, width, itemsize))
        )
        if found is None:
            return self.default
        admitted = self.admits(found.winner.geometry)
        return self.default if admitted is None else admitted


def register(variants: Variants[G]) -> Variants[G]:
    """Make a kernel's geometries visible to the tuning driver.

    Args:
        variants: The declaration. Returned, so a module can bind the result.

    Returns:
        ``variants``.

    Raises:
        ValueError: If the name is already registered to a different declaration.
            Two kernels sharing a name would share a record.
    """
    held = _VARIANTS.get(variants.kernel)
    if held is not None and held != variants:
        raise ValueError(
            f"{variants.kernel} is already registered with a different geometry set"
        )
    _VARIANTS[variants.kernel] = variants
    return variants


def registered() -> Mapping[str, Variants[Any]]:
    """Every registered kernel, by name.

    A kernel registers when its module is imported, so a caller that wants the
    whole set imports the operator packages first.

    Returns:
        The registry. Read-only view.
    """
    return dict(_VARIANTS)
