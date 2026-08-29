"""The reference preprocessing, and the manifest that proves a corpus is the one it claims.

Two steps in that pipeline decide numbers and neither is obvious. ``np.unique(axis=0)`` returns
rows in a field-wise numeric sort, so the surviving pool is *reordered* and the archive's own
train/test boundary is gone -- which is why the protocol draws a partition from a seed instead.
And a row holding a missing value never deduplicates, itself included, because NaN compares
unequal, so the instance count depends on the NaN pattern and the instance count sets the
partition boundaries.

Both are reproduced rather than corrected, and both are pinned here. The digest tests cover the
other half of the module's job: a corpus that changed under a run is the failure it exists to
catch.

:func:`write_archive` is the synthetic archive the split and driver tests build on too. Its first
channel's first timepoint is a scrambled permutation of ``arange(instances)`` on purpose: a
reorder is only detectable when the sorted order differs from the file order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.tsc.corpus import (
    DATA_NAME,
    MANIFEST_NAME,
    Corpus,
    arff_paths,
    dataset_names,
    encode,
    load,
    main,
    process,
    read_manifest,
    stack,
)
from scripts.tsc.reader import Split

_HEADER = """@relation '{name}'
@attribute relationalAtt relational
{inner}@end relationalAtt
@attribute target {{{classes}}}
@data
"""

_SCRAMBLE = 7
"""Coprime with every instance count used here, so the sort key is a permutation of the index."""


def _arff(
    name: str, instances: list[tuple[list[list[float]], str]], classes: list[str]
) -> str:
    """One relational ARFF file's text.

    Args:
        name: Relation name.
        instances: ``(channels, label)`` per instance, each channel a list of values.
        classes: The declared class set.

    Returns:
        The file text.
    """
    width = len(instances[0][0][0])
    inner = "".join(f"@attribute att{i} numeric\n" for i in range(width))
    lines = []
    for channels, label in instances:
        body = "\\n".join(
            ",".join("?" if value != value else repr(value) for value in channel)
            for channel in channels
        )
        lines.append(f"'{body}',{label}\n")
    return _HEADER.format(name=name, inner=inner, classes=",".join(classes)) + "".join(
        lines
    )


def write_archive(root: Path, name: str = "Probe", *, instances: int = 24) -> Path:
    """Write a synthetic archive directory holding one dataset.

    Every instance is distinct and finite, three timepoints by two channels, labels cycling over
    three classes so a partition at any of the protocol's fractions can still see them all. The
    first channel carries ``(i * 7) % instances`` so file order and sorted order disagree.

    Args:
        root: Directory to create the archive under.
        name: Dataset folder and file prefix.
        instances: Instances, split as a 2:1 train/test ratio.

    Returns:
        The archive directory, ready for :func:`scripts.tsc.corpus.process`.
    """
    classes = ["a", "b", "c"]
    made = []
    for i in range(instances):
        key = float((i * _SCRAMBLE) % instances)
        made.append(
            (
                [
                    [key, key + 0.5, float(i) + 1.0],
                    [float(-i), float(-i) - 0.5, float(-i) - 1.0],
                ],
                classes[i % len(classes)],
            )
        )
    cut = (instances * 2) // 3
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}_TRAIN.arff").write_text(
        _arff(name, made[:cut], classes), "latin-1"
    )
    (folder / f"{name}_TEST.arff").write_text(
        _arff(name, made[cut:], classes), "latin-1"
    )
    return root


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A synthetic archive with one 24-instance dataset."""
    return write_archive(tmp_path / "archive")


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """Where the processed dataset is written."""
    return tmp_path / "corpus" / "Probe"


@pytest.fixture
def corpus(archive: Path, store: Path) -> Corpus:
    """That dataset, processed to disk."""
    return process(archive, "Probe", store)


def test_the_pool_is_deduplicated_and_reordered(archive: Path) -> None:
    """Duplicating a test instance loses one row, and the surviving order is a numeric sort.

    Both halves matter. The lost row moves every partition boundary, and the sort is why
    ``train_instances`` cannot be used as a split even though the manifest records it.
    """
    plain = process(archive, "Probe")
    assert plain.manifest.removed == 0
    assert plain.manifest.instances == 24
    # A field-wise sort of the rows, so channel 0 of timepoint 0 comes out ascending. The
    # archive wrote that column scrambled, so this is the reorder and not the file order.
    column = plain.data[:, 0, 0]
    assert np.array_equal(column, np.arange(24, dtype=np.float32))
    assert not np.array_equal(
        column, np.array([(i * _SCRAMBLE) % 24 for i in range(24)], dtype=np.float32)
    )
    # Labels travel with the rows: row i now holds the instance whose index solves i = 7k mod 24.
    inverse = {(i * _SCRAMBLE) % 24: i for i in range(24)}
    want = [inverse[position] % 3 for position in range(24)]
    assert plain.labels.tolist() == want

    duplicated = archive / "Probe" / "Probe_TEST.arff"
    text = duplicated.read_text("latin-1")
    duplicated.write_text(text + text.splitlines()[-1] + "\n", "latin-1")
    with_repeat = process(archive, "Probe")
    assert with_repeat.manifest.instances == plain.manifest.instances
    assert with_repeat.manifest.removed == 1


def test_a_missing_value_survives_deduplication(archive: Path) -> None:
    """A NaN row does not match itself, so an exact repeat of it keeps both copies.

    The reference behaves this way and the instance count sets the partition, so correcting it
    here would move every split away from the one the published bars used.
    """
    test_file = archive / "Probe" / "Probe_TEST.arff"
    holed = "'?,2.0,3.0\\n?,-2.0,-3.0',a\n"
    test_file.write_text(test_file.read_text("latin-1") + holed + holed, "latin-1")
    found = process(archive, "Probe")
    assert found.manifest.removed == 0
    assert found.manifest.instances == 26
    assert int(np.isnan(found.data).any(axis=(1, 2)).sum()) == 2


def test_labels_are_encoded_on_the_training_file_alone() -> None:
    """The class order is a sort of the training labels, and an unseen test label is refused.

    Dropping an unseen class silently would shift every index above it, so every reported
    per-class number would name the wrong class.
    """
    train, test, classes = encode(["b", "a", "b"], ["a", "b"])
    assert classes == ["a", "b"]
    assert train.tolist() == [1, 0, 1]
    assert test.tolist() == [0, 1]
    assert train.dtype == np.int32
    with pytest.raises(ValueError, match=r"\['c'\]"):
        encode(["a", "b"], ["c"])


def test_a_ragged_dataset_names_itself() -> None:
    """Instances of different lengths stop the pipeline with the dataset and the lengths.

    The ARFF archive pads a variable-length dataset to its longest series, so this refusal is for
    a truncated or hand-built one. The reference fails on it too, from ``np.stack``, with a
    message that does not say which dataset -- and an empty split, which the reference stacks
    into a zero-length array, is refused rather than carried into a partition.
    """
    ragged = Split(
        [np.zeros((3, 2), dtype=np.float64), np.zeros((4, 2), dtype=np.float64)],
        ["a", "b"],
    )
    with pytest.raises(ValueError, match=r"Probe holds instances of lengths \[3, 4\]"):
        stack(ragged, dataset="Probe")
    with pytest.raises(ValueError, match="Probe holds no instances"):
        stack(Split([], []), dataset="Probe")


def test_a_store_round_trips_and_a_tampered_one_is_refused(
    corpus: Corpus, store: Path
) -> None:
    """:func:`load` returns what :func:`process` wrote, and refuses a file that changed.

    The digest check is the whole reason the manifest exists: a corpus edited between two arms
    makes their numbers incomparable in a way nothing else would report.
    """
    reloaded = load(store)
    assert np.array_equal(reloaded.data, corpus.data)
    assert np.array_equal(reloaded.labels, corpus.labels)
    assert reloaded.manifest == corpus.manifest
    array = np.load(store / DATA_NAME)
    array[0, 0, 0] += 1.0
    with (store / DATA_NAME).open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    with pytest.raises(ValueError, match="digests"):
        load(store)


def test_an_in_memory_pass_digests_the_same_bytes(
    archive: Path, corpus: Corpus
) -> None:
    """Processing without writing produces the manifest a written store would have.

    That is what lets a run check a corpus directory against a fresh pass over the archive
    without a second copy of the data on disk.
    """
    assert process(archive, "Probe").manifest == corpus.manifest


def test_a_manifest_from_another_version_is_refused(
    corpus: Corpus, store: Path
) -> None:
    """An unknown or missing manifest field stops the read.

    A manifest silently accepted at the wrong shape would let a run train on arrays that are not
    what the record says they are.
    """
    del corpus
    raw = json.loads((store / MANIFEST_NAME).read_text("utf-8"))
    raw["unexpected"] = 1
    (store / MANIFEST_NAME).write_text(json.dumps(raw), "utf-8")
    with pytest.raises(TypeError):
        read_manifest(store)


def test_a_partial_extraction_names_the_missing_file(archive: Path) -> None:
    """One ARFF without the other is not a dataset."""
    (archive / "Probe" / "Probe_TEST.arff").unlink()
    with pytest.raises(FileNotFoundError, match=r"Probe_TEST\.arff"):
        arff_paths(archive, "Probe")


def test_the_default_enumeration_skips_a_folder_that_holds_no_arff_at_all(
    archive: Path,
) -> None:
    """The archive ships ``Descriptions`` and ``Images``, and neither is a dataset.

    Those two are the reason the default cannot simply be every folder: a bare
    ``--archive Multivariate_arff --out X`` would otherwise stop on the first of them and leave a
    corpus holding whatever sorted before it. A folder holding *one* of the two ARFF files is a
    different case and stays refused, because it is a partial extraction of a real dataset.
    """
    for name in ("Descriptions", "Images"):
        (archive / name).mkdir()
    (archive / "Descriptions" / "notes.txt").write_text("prose", "utf-8")
    assert dataset_names(archive) == ["Probe"]

    half = archive / "Half"
    half.mkdir()
    (half / "Half_TRAIN.arff").write_text(
        (archive / "Probe" / "Probe_TRAIN.arff").read_text("latin-1"), "latin-1"
    )
    assert dataset_names(archive) == ["Half", "Probe"]
    with pytest.raises(FileNotFoundError, match=r"Half_TEST\.arff"):
        arff_paths(archive, "Half")


def test_the_cli_writes_every_dataset_of_a_stock_archive_and_reports_each(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end on the default enumeration, with the archive's two non-datasets present.

    This is the invocation that prepares a corpus, so its failure mode is a corpus that is
    silently short of a dataset a sweep will later ask for.
    """
    (archive / "Images").mkdir()
    write_archive(archive, "Second", instances=12)
    out = tmp_path / "corpus"
    assert main(["--archive", str(archive), "--out", str(out)]) == 0
    assert sorted(item.name for item in out.iterdir()) == ["Probe", "Second"]
    assert read_manifest(out / "Second").instances == 12
    reported = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in reported] == ["Probe", "Second"]
