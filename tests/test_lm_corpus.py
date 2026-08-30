"""The corpus: content addressing, shard disjointness, and the byte count bits-per-byte uses.

The manifest is the only thing standing between a table and two hosts quietly reporting
numbers on different text, so the digest has to actually catch a changed file, the two shards
have to be disjoint, and the byte count has to be of the decoded text rather than of the token
file.

The tokenizer is not exercised here. A fake encoder gives every document a distinct token, so
shard membership is readable off the files and a disjointness failure names the document that
crossed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.lm.corpus import (
    DTYPE,
    MANIFEST_NAME,
    CorpusManifest,
    ShardCounts,
    from_dict,
    read_manifest,
    shard_path,
    to_dict,
    write_manifest,
    write_shards,
)

EOT = 0
PER_DOC = 8
TRAIN_TOKENS = 3 * (PER_DOC + 1)
VAL_TOKENS = 2 * (PER_DOC + 1)


def _documents(count: int) -> list[str]:
    """``count`` documents, each encoding to one distinct token repeated."""
    return [str(index) for index in range(count)]


def _encode(text: str) -> list[int]:
    """A document's ids: its index plus one, ``PER_DOC`` times.

    Distinct per document and never :data:`EOT`, so a shard's contents name the documents
    that landed in it.
    """
    return [int(text) + 1] * PER_DOC


def _write(
    root: Path,
    *,
    train_tokens: int = TRAIN_TOKENS,
    val_tokens: int = VAL_TOKENS,
    documents: int = 8,
) -> dict[str, ShardCounts]:
    """Write both shards from the fake stream."""
    return write_shards(
        root,
        _documents(documents),
        _encode,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        eot_token_id=EOT,
    )


def _manifest(counts: dict[str, ShardCounts]) -> CorpusManifest:
    """A manifest over those counts, with the fields prep would have filled."""
    return from_dict(
        {
            "tokenizer": "fake",
            "vocab_size": 16,
            "eot_token_id": EOT,
            "dataset": "fake",
            "dataset_config": "sample",
            "dataset_split": "train",
            "revision": "1.0.0",
            "text_field": "text",
            "dtype": DTYPE.__name__,
            "train": counts["train"]._asdict(),
            "val": counts["val"]._asdict(),
        }
    )


def _ids(root: Path, split: str) -> set[int]:
    """The non-separator token ids in a shard."""
    return {
        int(value) for value in np.fromfile(shard_path(root, split), dtype=DTYPE)
    } - {EOT}


def test_val_is_filled_first_and_the_shards_are_disjoint(tmp_path: Path) -> None:
    """A document lands whole in one shard, and validation takes the head of the stream.

    Filling validation first is what makes a 1.8B run and a 10.9B run scored on the same
    text. Disjointness is at the document level, not the token level: a document split
    across the two would put training text in the held-out set.
    """
    counts = _write(tmp_path)
    assert counts["val"].tokens == VAL_TOKENS
    assert counts["train"].tokens == TRAIN_TOKENS
    val, train = _ids(tmp_path, "val"), _ids(tmp_path, "train")
    assert val == {1, 2}
    assert train == {3, 4, 5}
    assert not val & train


def test_byte_count_is_of_the_text_not_the_token_file(tmp_path: Path) -> None:
    """Bits-per-byte is against the corpus, so the separator is not counted.

    Two documents of one character each are two bytes, while the shard holds eighteen
    tokens. Counting the token file instead would scale every bits-per-byte figure by the
    tokenizer's compression and make the column meaningless.
    """
    counts = _write(tmp_path)
    assert counts["val"].text_bytes == 2
    assert counts["val"].tokens == 18
    assert _manifest(counts).val_bytes_per_token == pytest.approx(2 / 18)


def test_a_stream_that_runs_out_is_an_error(tmp_path: Path) -> None:
    """A run at fewer tokens than were asked for is not the run that was asked for."""
    with pytest.raises(ValueError, match="ran out of documents"):
        _write(tmp_path, train_tokens=1000, documents=4)


def test_a_flipped_byte_is_caught_by_the_digest(tmp_path: Path) -> None:
    """The digest is the whole point: a corpus that changed under a run has to say so."""
    counts = _write(tmp_path)
    write_manifest(tmp_path, _manifest(counts))
    assert read_manifest(tmp_path).train.digest == counts["train"].digest

    path = shard_path(tmp_path, "train")
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="shard digest is"):
        read_manifest(tmp_path)


def test_the_manifest_round_trips_through_json(tmp_path: Path) -> None:
    """The two shard counts are written as mappings, not as arrays.

    :func:`dataclasses.asdict` leaves a namedtuple a namedtuple, which JSON writes as an
    array and no keyword reconstruction reads back. Left unfixed, every manifest would write
    fine and no manifest would load.
    """
    manifest = _manifest(_write(tmp_path))
    write_manifest(tmp_path, manifest)
    raw = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert isinstance(raw["train"], dict)
    assert read_manifest(tmp_path) == manifest
    assert from_dict(to_dict(manifest)) == manifest


def test_a_missing_manifest_is_not_a_default(tmp_path: Path) -> None:
    """A corpus directory with no manifest cannot be measured against."""
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        read_manifest(tmp_path)
