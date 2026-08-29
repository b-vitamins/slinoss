"""Process the archive once, into arrays a run reads with numpy and a manifest that names them.

Prep time only. The output is two ``.npy`` files and a JSON manifest per dataset; training
imports nothing from here but :func:`load`. No pickle: a pickle is executable, it cannot be
read without the library that wrote it, and a corpus that outlives one JAX version is worth
more than one that does not.

The processing is the reference pipeline's, step for step, because the numbers this harness is
compared to were produced on its output:

    read both ARFF files          instances in file order, dimensions in declared order
    stack to ``(n, L, d)``        float64, then float32
    encode labels                 over the *training* file's values, sorted
    pool train then test          in that order
    ``np.unique(axis=0)``         deduplicate, and reorder by the returned first indices
    reorder the labels with it

Two of those steps are surprising and both are load-bearing. ``np.unique(axis=0)`` views the
rows as a structured dtype with one field per column, so the surviving order is a field-wise
numeric sort and not the file order -- the pooled set comes out sorted by its first timepoint's
first channel. And the reference's own ``original_idxs``, which would name the train/test
boundary, is computed *after* that reorder from the pre-reorder counts, so it points at
arbitrary rows; it is not written here. The train/test boundary does not survive processing,
which is why the protocol draws its own partition from a seed. See :mod:`scripts.tsc.split`.

A row holding a missing value never deduplicates against anything, itself included, because NaN
does not compare equal. That is the reference's behaviour and it is kept: changing it would
change the instance count, and the instance count sets the partition boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from scripts.tsc.reader import Split, read_arff

__all__ = [
    "DATA_NAME",
    "LABELS_NAME",
    "MANIFEST_NAME",
    "Corpus",
    "DatasetManifest",
    "arff_paths",
    "dataset_names",
    "encode",
    "load",
    "main",
    "process",
    "read_manifest",
    "sha256_of",
    "stack",
]

DATA_NAME = "data.npy"
"""The pooled, deduplicated, reordered instances."""

LABELS_NAME = "labels.npy"
"""The matching encoded labels."""

MANIFEST_NAME = "manifest.json"
"""What the two files are and which archive files they came from."""

_CHUNK = 1 << 20


class Corpus(NamedTuple):
    """One processed dataset, in memory.

    Attributes:
        data: ``(N, L, d)`` float32. Channel order is the archive's; no time channel and no
            normalization. Missing values are still NaN.
        labels: ``(N,)`` int32 class indices into ``manifest.classes``.
        manifest: What these arrays are.
    """

    data: np.ndarray
    labels: np.ndarray
    manifest: DatasetManifest


@dataclass(frozen=True)
class DatasetManifest:
    """What a processed dataset is, in the form a run record carries.

    Attributes:
        dataset: Archive folder name.
        classes: Label text in encoded order, so index ``i`` means ``classes[i]``. The
            encoding is over the training file alone, as the reference's is.
        instances: Rows in ``data.npy``, after deduplication. The partition boundaries are
            fractions of this, so it is the one number a split depends on.
        length: Timepoints.
        dimensions: Channels.
        train_instances: Rows the training ARFF held. Recorded because it identifies the
            archive's own split, and not usable as one here: deduplication reorders the pool,
            so the first ``train_instances`` rows of ``data.npy`` are not that split.
        removed: Rows deduplication dropped.
        missing: Fraction of values that are NaN.
        train_arff: Hex SHA-256 of the training ARFF.
        test_arff: Hex SHA-256 of the test ARFF.
        data_sha256: Hex SHA-256 of ``data.npy``.
        labels_sha256: Hex SHA-256 of ``labels.npy``.
    """

    dataset: str
    classes: list[str]
    instances: int
    length: int
    dimensions: int
    train_instances: int
    removed: int
    missing: float
    train_arff: str
    test_arff: str
    data_sha256: str
    labels_sha256: str


def sha256_of(path: Path) -> str:
    """Digest a file without holding it.

    Args:
        path: The file.

    Returns:
        Hex SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def arff_paths(archive: Path, dataset: str) -> tuple[Path, Path]:
    """Where one dataset's two ARFF files live.

    Args:
        archive: The extracted ``Multivariate_arff`` directory.
        dataset: Folder name, which is also the file prefix.

    Returns:
        The training and test paths, in that order.

    Raises:
        FileNotFoundError: When either is absent. Named individually, because a dataset with
            only one of the two is a partial extraction and not a dataset.
    """
    folder = archive / dataset
    paths = (folder / f"{dataset}_TRAIN.arff", folder / f"{dataset}_TEST.arff")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def dataset_names(archive: Path) -> list[str]:
    """Every folder of an archive that is a dataset, sorted.

    A folder counts when it holds at least one of its two ARFF files, which is what separates a
    partial extraction -- refused by :func:`arff_paths` -- from a folder that was never a dataset.
    The published archive ships two of the latter, ``Descriptions`` and ``Images``, so a default
    of every folder would stop the whole prep on the first one.

    Args:
        archive: The extracted ``Multivariate_arff`` directory.

    Returns:
        Folder names, each also a file prefix.
    """
    found = []
    for item in archive.iterdir():
        if item.is_dir() and any(
            (item / f"{item.name}_{half}.arff").is_file() for half in ("TRAIN", "TEST")
        ):
            found.append(item.name)
    return sorted(found)


def stack(split: Split, *, dataset: str) -> np.ndarray:
    """One array from a split's instances.

    Args:
        split: The instances.
        dataset: Name, for the message.

    Returns:
        ``(n, L, d)`` float32.

    Raises:
        ValueError: On an empty split, or on instances of different lengths. The reference
            pipeline raises here too, from ``np.stack``; this says which dataset and what the
            lengths were. The ARFF archive as distributed pads a variable-length dataset with
            ``?`` to its longest series, so nothing in it reaches this refusal -- a truncated
            or hand-built archive does, and then the name is what identifies it.
    """
    if not split.series:
        raise ValueError(f"{dataset} holds no instances")
    lengths = sorted(set(split.lengths))
    if len(lengths) != 1:
        raise ValueError(
            f"{dataset} holds instances of lengths {lengths}; the reference pipeline "
            f"cannot stack it"
        )
    return np.stack(split.series).astype(np.float32)


def encode(
    train: list[str], test: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map label text to indices, fitting on the training file only.

    ``sklearn.preprocessing.LabelEncoder`` fitted on the training labels, which is
    ``np.unique`` of them -- a lexicographic sort -- and then a lookup that raises on an
    unseen value. Transcribed rather than imported: the whole behaviour is a sort and a dict,
    and the class order decides what every reported per-class number means.

    Args:
        train: Training labels, in file order.
        test: Test labels, in file order.

    Returns:
        The two index arrays as int32, and the class text in encoded order.

    Raises:
        ValueError: When the test file carries a label the training file does not. The
            reference raises the same way, and a silently dropped class would move every
            index after it.
    """
    classes = sorted(set(train))
    index = {name: position for position, name in enumerate(classes)}
    unseen = sorted(set(test) - index.keys())
    if unseen:
        raise ValueError(f"test labels {unseen} are not in the training labels")
    return (
        np.asarray([index[name] for name in train], dtype=np.int32),
        np.asarray([index[name] for name in test], dtype=np.int32),
        classes,
    )


def process(archive: Path, dataset: str, out: Path | None = None) -> Corpus:
    """Run the pipeline on one dataset.

    Args:
        archive: The extracted ``Multivariate_arff`` directory.
        dataset: Folder name.
        out: Directory to write ``data.npy``, ``labels.npy`` and the manifest into, or None
            to process in memory. When None the manifest's digests are of the bytes that
            would have been written, so a run can check a store against an in-memory pass.

    Returns:
        The corpus.

    Raises:
        FileNotFoundError: From :func:`arff_paths`.
        ValueError: From :func:`stack` or :func:`encode`.
    """
    train_path, test_path = arff_paths(archive, dataset)
    train_split, test_split = read_arff(train_path), read_arff(test_path)
    train_data = stack(train_split, dataset=dataset)
    test_data = stack(test_split, dataset=dataset)
    train_labels, test_labels, classes = encode(train_split.labels, test_split.labels)

    pooled = np.concatenate([train_data, test_data])
    labels = np.concatenate([train_labels, test_labels])
    # The reference's own call, and the reorder is its return value's, not the file's. Rows
    # holding NaN never match, so they all survive.
    _, indices, inverse = np.unique(
        pooled, axis=0, return_index=True, return_inverse=True
    )
    data = pooled[indices]
    labels = labels[indices]

    manifest = DatasetManifest(
        dataset=dataset,
        classes=classes,
        instances=int(data.shape[0]),
        length=int(data.shape[1]),
        dimensions=int(data.shape[2]),
        train_instances=int(train_data.shape[0]),
        removed=int(inverse.size - indices.size),
        missing=float(np.isnan(data).mean()),
        train_arff=sha256_of(train_path),
        test_arff=sha256_of(test_path),
        data_sha256=_digest_of(data),
        labels_sha256=_digest_of(labels),
    )
    if out is not None:
        _write(out, data, labels, manifest)
    return Corpus(data, labels, manifest)


def _digest_of(array: np.ndarray) -> str:
    """Digest an array's ``.npy`` encoding.

    The file's digest and not the buffer's, so a manifest written from an in-memory pass
    matches one written from a store.

    Args:
        array: The array.

    Returns:
        Hex SHA-256.
    """
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _write(
    out: Path, data: np.ndarray, labels: np.ndarray, manifest: DatasetManifest
) -> None:
    """Write the two arrays and the manifest.

    Args:
        out: Destination directory. Created if absent.
        data: The instances.
        labels: The labels.
        manifest: What they are.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, array in ((DATA_NAME, data), (LABELS_NAME, labels)):
        with (out / name).open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
    text = json.dumps(asdict(manifest), indent=2, sort_keys=True)
    (out / MANIFEST_NAME).write_text(text + "\n", encoding="utf-8")


def read_manifest(root: Path) -> DatasetManifest:
    """Read a processed dataset's manifest.

    Args:
        root: The dataset's directory.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: When the directory holds no manifest.
        TypeError: On a manifest missing a field or carrying an unknown one, which is a
            manifest written by another version of this module.
    """
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {root}")
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return DatasetManifest(**raw)


def load(root: Path) -> Corpus:
    """Read a processed dataset, checking it is the one the manifest describes.

    The digests are re-verified on every read. A corpus that changed under a run is the
    failure this module exists to catch, and it is cheap next to the run.

    Args:
        root: The dataset's directory.

    Returns:
        The corpus.

    Raises:
        FileNotFoundError: When a file is absent.
        ValueError: When a digest or a shape does not match the manifest.
    """
    manifest = read_manifest(root)
    data = np.load(root / DATA_NAME, allow_pickle=False)
    labels = np.load(root / LABELS_NAME, allow_pickle=False)
    for name, found, stated in (
        (DATA_NAME, sha256_of(root / DATA_NAME), manifest.data_sha256),
        (LABELS_NAME, sha256_of(root / LABELS_NAME), manifest.labels_sha256),
    ):
        if found != stated:
            raise ValueError(f"{name} digests {found}, the manifest says {stated}")
    shape = (manifest.instances, manifest.length, manifest.dimensions)
    if data.shape != shape or data.dtype != np.float32:
        raise ValueError(
            f"{DATA_NAME} is {data.shape} {data.dtype}, expected {shape} f4"
        )
    if labels.shape != (manifest.instances,) or labels.dtype != np.int32:
        raise ValueError(f"{LABELS_NAME} is {labels.shape} {labels.dtype}, expected i4")
    return Corpus(data, labels, manifest)


def main(argv: list[str] | None = None) -> int:
    """Process datasets from an extracted archive into a corpus directory.

    Args:
        argv: Arguments, or None for the process's.

    Returns:
        A process exit status: zero unless a dataset failed.
    """
    parser = argparse.ArgumentParser(
        prog="scripts.tsc.corpus",
        description="Process UEA datasets into data.npy, labels.npy and a manifest.",
    )
    parser.add_argument("--archive", required=True, help="extracted Multivariate_arff")
    parser.add_argument(
        "--out", required=True, help="corpus root, one folder per dataset"
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        metavar="NAME",
        help="archive folder; repeatable, defaults to every dataset present",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="report and continue past a dataset the pipeline cannot stack",
    )
    options = parser.parse_args(argv)
    archive, out = Path(options.archive), Path(options.out)
    datasets = options.datasets or dataset_names(archive)
    failures = 0
    for name in datasets:
        try:
            corpus = process(archive, name, out / name)
        except (FileNotFoundError, ValueError) as exc:
            if not options.keep_going:
                raise
            failures += 1
            print(f"{name}: {type(exc).__name__}: {exc}")
            continue
        manifest = corpus.manifest
        print(
            f"{name}: {manifest.instances}x{manifest.length}x{manifest.dimensions}, "
            f"{len(manifest.classes)} classes, {manifest.removed} deduplicated, "
            f"{manifest.missing:.6f} missing"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
