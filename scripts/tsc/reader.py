"""Read the UEA archive's own files, with no reader library between.

The archive ships each dataset twice: a relational ARFF pair and a ``.ts`` pair. The
preprocessing this harness reproduces reads the ARFF, so the ARFF is what this module reads,
and the ``.ts`` reader exists to cross-check it. Neither path imports ``sktime``,
``liac-arff`` or ``pandas``: a dependency here would put a third party's version between the
archive on disk and the array a run trains on, and the one property this module has to have
is that the same zip yields the same bytes forever.

The relational ARFF form, which is the only form the multivariate archive uses::

    @relation 'BasicMotions'
    @attribute relationalAtt relational
    @attribute att0 numeric
    ...                                  one inner attribute per timepoint
    @end relationalAtt
    @attribute activity {Standing,Running,Walking,Badminton}
    @data
    'd1t0,d1t1,...\\nd2t0,d2t1,...',Standing

One quoted field per instance, dimensions separated by a literal backslash-n, then the class
value. The class attribute's *name* varies across the archive, so it is identified by being
the one attribute declared outside the relational block, not by a name.

The ``.ts`` form carries its shape in the header and separates dimensions with ``:``::

    @dimensions 6
    @equalLength true
    @seriesLength 100
    @classLabel true Standing Running Walking Badminton
    @data
    d1t0,d1t1,...:d2t0,d2t1,...:Standing

The two do not always agree. The relational ARFF declares a fixed inner attribute count, so
a variable-length dataset is padded there with ``?`` and arrives here as trailing NaN, while
the ``.ts`` keeps the true length. That is a property of the archive, not of this reader, and
:func:`agree` reports it rather than reconciling it.

Missing values are NaN. Nothing is imputed, normalized, or reordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Split",
    "agree",
    "finite_prefix",
    "read_arff",
    "read_ts",
    "summarize",
]

_NAN = float("nan")


@dataclass(frozen=True)
class Split:
    """One file's instances, in file order.

    A list rather than one array because the archive holds ragged datasets and this reader
    does not decide what to do about them. :func:`scripts.tsc.corpus.stack` does, and it
    names the dataset it refused.

    Attributes:
        series: One ``(length, dimensions)`` float64 array per instance, in file order.
            Dimensions are in the order the file declares them.
        labels: The class value of each instance, as the text in the file. Not encoded: the
            encoding is over the training file's values and belongs to the next stage.
    """

    series: list[np.ndarray]
    labels: list[str]

    def __post_init__(self) -> None:
        if len(self.series) != len(self.labels):
            raise ValueError(
                f"{len(self.series)} instances and {len(self.labels)} labels"
            )

    @property
    def dimensions(self) -> int:
        """Channels per instance.

        Returns:
            The count, zero for an empty split.

        Raises:
            ValueError: When the instances disagree, which no archive file does.
        """
        widths = {int(item.shape[1]) for item in self.series}
        if len(widths) > 1:
            raise ValueError(f"instances carry {sorted(widths)} dimensions")
        return widths.pop() if widths else 0

    @property
    def lengths(self) -> list[int]:
        """Timepoints per instance, in file order.

        Returns:
            One length per instance.
        """
        return [int(item.shape[0]) for item in self.series]


def _text(path: Path) -> list[str]:
    """Read a file as lines.

    Latin-1, which never raises and is the identity on the ASCII the data section is written
    in. Some header comments in the archive are not UTF-8, and a decode error in a comment
    nothing reads would be a failure for the wrong reason.

    Args:
        path: The file.

    Returns:
        Its lines, without terminators.

    Raises:
        FileNotFoundError: When the file is absent.
    """
    return path.read_text(encoding="latin-1").splitlines()


def _values(text: str, *, expected: int | None = None) -> list[float]:
    """Parse one comma-separated channel.

    Args:
        text: The channel's values.
        expected: Count to insist on, or None to accept what is there.

    Returns:
        The values, with ``?`` and empty fields as NaN.

    Raises:
        ValueError: On a count other than ``expected``, or on a field that is not a number.
    """
    out = [_NAN if field in ("?", "") else float(field) for field in text.split(",")]
    if expected is not None and len(out) != expected:
        raise ValueError(f"channel of {len(out)} values, expected {expected}")
    return out


def _instance(rows: list[list[float]]) -> np.ndarray:
    """Assemble one instance from its channels.

    Args:
        rows: One list of values per dimension, all the same length.

    Returns:
        ``(length, dimensions)`` float64. The transpose of the channel stack, which is the
        layout the reference preprocessing produces.

    Raises:
        ValueError: On channels of unequal length, or on no channels at all.
    """
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"instance has channels of lengths {sorted(widths)}")
    return np.vstack([np.asarray(row, dtype=np.float64) for row in rows]).T


def _nominal(line: str) -> list[str]:
    """The values of a nominal attribute declaration.

    Args:
        line: An ``@attribute name {a,b,c}`` line.

    Returns:
        The values, unquoted and stripped.

    Raises:
        ValueError: When the line declares no value set.
    """
    if "{" not in line or "}" not in line:
        raise ValueError(f"not a nominal attribute: {line!r}")
    body = line[line.index("{") + 1 : line.rindex("}")]
    return [field.strip().strip("'\"") for field in body.split(",")]


def read_arff(path: Path) -> Split:
    """Read one relational ARFF file.

    Args:
        path: ``<Dataset>_TRAIN.arff`` or ``<Dataset>_TEST.arff``.

    Returns:
        The split.

    Raises:
        FileNotFoundError: When the file is absent.
        ValueError: On a file with no relational attribute, no class attribute or no
            ``@data``; on an instance that is not a quoted field; on a channel whose length
            is not the declared inner attribute count; or on a class value outside the
            declared set.
    """
    lines = _text(path)
    inner = 0
    in_relational = False
    classes: list[str] | None = None
    start: int | None = None
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        lowered = line.lower()
        if lowered.startswith("@data"):
            start = index + 1
            break
        if lowered.startswith("@end"):
            in_relational = False
        elif lowered.startswith("@attribute"):
            if in_relational:
                inner += 1
            elif lowered.endswith("relational"):
                in_relational = True
            else:
                classes = _nominal(line)
    if start is None:
        raise ValueError(f"{path} has no @data section")
    if inner == 0:
        raise ValueError(f"{path} declares no relational attribute")
    if classes is None:
        raise ValueError(f"{path} declares no class attribute")

    allowed = set(classes)
    series: list[np.ndarray] = []
    labels: list[str] = []
    for raw in lines[start:]:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        quote = line[0]
        if quote not in "'\"":
            raise ValueError(f"{path}: instance is not a quoted field: {line[:40]!r}")
        # The quoted field is numeric text with backslash-n separators and holds no quote of
        # its own, so the first repeat of the opening character closes it. Searching from the
        # right would find a quoted class value's closing quote instead.
        end = line.index(quote, 1)
        label = line[end + 1 :].lstrip(",").strip().strip("'\"")
        if label not in allowed:
            raise ValueError(f"{path}: class {label!r} is not one of {classes}")
        rows = [_values(part, expected=inner) for part in line[1:end].split("\\n")]
        series.append(_instance(rows))
        labels.append(label)
    return Split(series, labels)


def read_ts(path: Path) -> Split:
    """Read one ``.ts`` file.

    The cross-check on :func:`read_arff`, and the only place the archive states a dataset's
    intended shape. Timestamped files are refused rather than parsed: the multivariate
    archive carries none, so a parser for them would be untested code that silently accepts
    a format this harness has never seen.

    Args:
        path: ``<Dataset>_TRAIN.ts`` or ``<Dataset>_TEST.ts``.

    Returns:
        The split.

    Raises:
        FileNotFoundError: When the file is absent.
        ValueError: On ``@timeStamps true``; on a missing ``@data``, ``@dimensions`` or
            ``@classLabel``; on an instance whose channel count is not ``@dimensions``; on a
            class value outside ``@classLabel``; or on lengths that contradict a declared
            ``@equalLength``.
    """
    lines = _text(path)
    dimensions: int | None = None
    length: int | None = None
    equal = False
    classes: list[str] | None = None
    start: int | None = None
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("@data"):
            start = index + 1
            break
        if lowered.startswith("@timestamps"):
            if lowered.split()[-1] == "true":
                raise ValueError(
                    f"{path} is timestamped; this reader does not parse it"
                )
        elif lowered.startswith("@dimensions"):
            dimensions = int(line.split()[1])
        elif lowered.startswith("@serieslength"):
            length = int(line.split()[1])
        elif lowered.startswith("@equallength"):
            equal = line.split()[1].lower() == "true"
        elif lowered.startswith("@classlabel"):
            fields = line.split()
            classes = fields[2:] if fields[1].lower() == "true" else []
    if start is None:
        raise ValueError(f"{path} has no @data section")
    if dimensions is None or classes is None:
        raise ValueError(f"{path} declares no @dimensions or no @classLabel")

    allowed = set(classes)
    series: list[np.ndarray] = []
    labels: list[str] = []
    for raw in lines[start:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(":")
        if len(fields) != dimensions + 1:
            raise ValueError(
                f"{path}: instance has {len(fields) - 1} channels, "
                f"@dimensions says {dimensions}"
            )
        label = fields[-1].strip().strip("'\"")
        if label not in allowed:
            raise ValueError(f"{path}: class {label!r} is not one of {classes}")
        series.append(_instance([_values(field) for field in fields[:dimensions]]))
        labels.append(label)
    found = Split(series, labels)
    seen = set(found.lengths)
    if equal and length is not None and seen not in (set(), {length}):
        raise ValueError(
            f"{path} declares @equalLength at @seriesLength {length} and holds "
            f"{sorted(seen)}"
        )
    return found


def agree(left: Split, right: Split) -> bool:
    """Whether two readings of one dataset hold the same instances.

    NaN equals NaN here. The ARFF pads a ragged dataset with missing values, so an exact
    comparison would report every such dataset as a disagreement over padding rather than
    over data.

    Args:
        left: One reading.
        right: The other.

    Returns:
        True when the labels match in order and every instance matches elementwise.
    """
    if left.labels != right.labels or left.lengths != right.lengths:
        return False
    return all(
        np.array_equal(one, other, equal_nan=True)
        for one, other in zip(left.series, right.series, strict=True)
    )


def finite_prefix(item: np.ndarray) -> int:
    """Timepoints before the first all-missing one.

    The ARFF's fixed inner width pads a short instance with ``?`` at the end, so this is how
    long that instance really is.

    Args:
        item: ``(length, dimensions)``.

    Returns:
        The count. The full length when nothing is missing.
    """
    hits = np.flatnonzero(np.isnan(item).all(axis=1))
    return int(item.shape[0] if hits.size == 0 else hits[0])


def summarize(split: Split) -> dict[str, float]:
    """Counts a report can print without holding the data.

    Args:
        split: The split.

    Returns:
        Instance count, dimensions, the shortest and longest length, and the fraction of
        values that are missing.
    """
    total = sum(int(item.size) for item in split.series)
    missing = sum(int(np.isnan(item).sum()) for item in split.series)
    lengths = split.lengths
    return {
        "instances": float(len(split.series)),
        "dimensions": float(split.dimensions),
        "min_length": float(min(lengths)) if lengths else 0.0,
        "max_length": float(max(lengths)) if lengths else 0.0,
        "missing": 0.0 if total == 0 else missing / total,
    }
