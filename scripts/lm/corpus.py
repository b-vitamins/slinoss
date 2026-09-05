"""Tokenize once, into a flat token file and a manifest that identifies it.

Prep time only. The output is two ``uint16`` files and a JSON manifest; training reads the
files with :mod:`numpy` and imports nothing from this module. Two consequences, both wanted:
a corpus outlives the ``datasets`` version that produced it, and a run's data is named by a
digest rather than by a path, so two hosts either agree or say so.

The manifest is what makes a table comparable. :func:`scripts.lm.run.table` refuses to
print rows whose ``train_sha256`` differ, which is the standing rule that no bits-per-byte
figure crosses hosts, enforced instead of remembered.

A document lands whole in exactly one shard, so the two shards are disjoint at the document
level and not merely at the token level. Validation is filled first: at a fixed
``val_tokens`` the held-out set is then the same prefix of the stream whatever the training
budget, so a 1.8B run and a 10.9B run are scored on the same text.

Byte counts are of the decoded text, not of the token file: bits-per-byte is against the
corpus, and the end-of-text token that separates documents is not corpus text. It is in the
token count because the model predicts it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "DTYPE",
    "MANIFEST_NAME",
    "CorpusManifest",
    "ShardCounts",
    "build",
    "from_dict",
    "read_manifest",
    "sha256_of",
    "shard_path",
    "to_dict",
    "write_manifest",
    "write_shards",
]

DTYPE = np.uint16
"""Token storage. Two bytes covers every tokenizer this harness uses and halves the file."""

DTYPE_MAX = 1 << 16
"""One past the largest id :data:`DTYPE` holds."""

MANIFEST_NAME = "manifest.json"
"""Manifest file name inside the corpus directory."""

SHARDS = ("train", "val")
"""The two splits, in the order they are written."""

_CHUNK = 1 << 20
"""Digest read size."""


class ShardCounts(NamedTuple):
    """What one shard ended up holding.

    Attributes:
        tokens: Token count, document separators included.
        text_bytes: UTF-8 bytes of the decoded text, separators excluded.
        digest: Hex SHA-256 of the token file.
    """

    tokens: int
    text_bytes: int
    digest: str


@dataclass(frozen=True)
class CorpusManifest:
    """What a corpus is, in the form a run record carries.

    Attributes:
        tokenizer: Tokenizer id, as :func:`transformers.AutoTokenizer.from_pretrained`
            takes it. The evaluation shim loads this exact string, so a corpus and a
            zero-shot score cannot disagree about the vocabulary.
        vocab_size: Ids the tokenizer emits. The head's ``vocab_size``.
        eot_token_id: The id written between documents.
        dataset: Dataset id.
        dataset_config: Dataset config name, or None.
        dataset_split: Split the documents were streamed from.
        revision: Dataset revision, or None when the loader reported none. A corpus
            without one is reproducible only by its digest.
        text_field: Field of a document that holds its text.
        dtype: Storage dtype name.
        train: Counts for the training shard.
        val: Counts for the validation shard.
    """

    tokenizer: str
    vocab_size: int
    eot_token_id: int
    dataset: str
    dataset_config: str | None
    dataset_split: str
    revision: str | None
    text_field: str
    dtype: str
    train: ShardCounts
    val: ShardCounts

    @property
    def val_bytes_per_token(self) -> float:
        """Decoded bytes per validation token.

        Returns:
            The ratio. Bits-per-byte is the mean loss in bits divided by this, so it is
            the one number that turns a token-level loss into a corpus-level one.

        Raises:
            ValueError: On an empty validation shard, which cannot be scored.
        """
        if self.val.tokens < 1:
            raise ValueError("validation shard holds no tokens")
        return self.val.text_bytes / self.val.tokens


def shard_path(root: Path, split: str) -> Path:
    """Where one shard's tokens live.

    Args:
        root: Corpus directory.
        split: ``train`` or ``val``.

    Returns:
        The path.

    Raises:
        ValueError: On a split that is not one of the two.
    """
    if split not in SHARDS:
        raise ValueError(f"split must be one of {SHARDS}, got {split!r}")
    return root / f"{split}.bin"


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


def write_shards(
    root: Path,
    documents: Iterable[str],
    encode: Callable[[str], list[int]],
    *,
    train_tokens: int,
    val_tokens: int,
    eot_token_id: int,
) -> dict[str, ShardCounts]:
    """Encode documents into the two token files.

    Validation is filled first and both shards stop at their budget, so a stream longer
    than ``train_tokens + val_tokens`` is truncated at a document boundary rather than
    sampled. A stream shorter than the budget is an error: a run at a token count nobody
    asked for is not comparable to one at the count they did.

    Args:
        root: Corpus directory. Created if absent.
        documents: The text, one document per item.
        encode: Text to ids. The separator is appended here, not by this callable.
        train_tokens: Training budget.
        val_tokens: Validation budget.
        eot_token_id: Separator id, written after every document.

    Returns:
        Counts per split.

    Raises:
        ValueError: On a non-positive budget, an id outside :data:`DTYPE`, or a stream
            that runs out before both budgets are met.
    """
    budgets = {"val": val_tokens, "train": train_tokens}
    for split, budget in budgets.items():
        if budget < 1:
            raise ValueError(f"{split}_tokens must be positive, got {budget}")
    if not eot_token_id < DTYPE_MAX:
        raise ValueError(f"eot_token_id {eot_token_id} does not fit {DTYPE.__name__}")

    root.mkdir(parents=True, exist_ok=True)
    order = ("val", "train")
    counts: dict[str, ShardCounts] = {}
    stream = iter(documents)
    for split in order:
        budget = budgets[split]
        path = shard_path(root, split)
        tokens = 0
        text_bytes = 0
        with path.open("wb") as handle:
            for text in stream:
                ids = [*encode(text), eot_token_id]
                block = np.asarray(ids, dtype=np.int64)
                if block.size and int(block.max()) >= DTYPE_MAX:
                    raise ValueError(
                        f"token id {int(block.max())} does not fit "
                        f"{DTYPE.__name__}; the tokenizer is too wide for this store"
                    )
                handle.write(block.astype(DTYPE).tobytes())
                tokens += int(block.size)
                text_bytes += len(text.encode("utf-8"))
                if tokens >= budget:
                    break
        if tokens < budget:
            raise ValueError(
                f"{split} shard ran out of documents at {tokens} of {budget} tokens"
            )
        counts[split] = ShardCounts(tokens, text_bytes, sha256_of(path))
    return counts


def build(
    root: Path,
    *,
    tokenizer: str,
    dataset: str,
    dataset_config: str | None,
    dataset_split: str,
    text_field: str,
    train_tokens: int,
    val_tokens: int,
) -> CorpusManifest:
    """Stream a dataset through a tokenizer into a corpus directory.

    The only function here that imports ``datasets`` or ``transformers``, and it imports
    them inside its body so training never pays for them.

    Args:
        root: Corpus directory.
        tokenizer: Tokenizer id.
        dataset: Dataset id.
        dataset_config: Config name, or None.
        dataset_split: Split to stream.
        text_field: Field holding a document's text.
        train_tokens: Training budget.
        val_tokens: Validation budget.

    Returns:
        The manifest, already written to ``root``.

    Raises:
        ValueError: From :func:`write_shards`, or on a tokenizer with no end-of-text id.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tok = AutoTokenizer.from_pretrained(tokenizer)
    eot = tok.eos_token_id
    if eot is None:
        raise ValueError(f"tokenizer {tokenizer} has no eos_token_id to separate with")
    stream = load_dataset(dataset, dataset_config, split=dataset_split, streaming=True)
    revision = getattr(getattr(stream, "info", None), "version", None)

    def documents() -> Iterable[str]:
        for record in stream:
            yield str(record[text_field])

    def encode(text: str) -> list[int]:
        return list(tok(text, add_special_tokens=False)["input_ids"])

    counts = write_shards(
        root,
        documents(),
        encode,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        eot_token_id=int(eot),
    )
    manifest = CorpusManifest(
        tokenizer=tokenizer,
        vocab_size=len(tok),
        eot_token_id=int(eot),
        dataset=dataset,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=None if revision is None else str(revision),
        text_field=text_field,
        dtype=DTYPE.__name__,
        train=counts["train"],
        val=counts["val"],
    )
    write_manifest(root, manifest)
    return manifest


def to_dict(manifest: CorpusManifest) -> dict[str, Any]:
    """The manifest as plain JSON-able data.

    Not :func:`dataclasses.asdict` alone: that leaves the two :class:`ShardCounts` as
    namedtuples, which JSON writes as arrays and no keyword reconstruction reads back. The
    counts are converted to mappings here so the file is self-describing and
    :func:`from_dict` is its inverse.

    Args:
        manifest: The manifest.

    Returns:
        A dict of scalars and two nested dicts.
    """
    raw = asdict(manifest)
    for split in SHARDS:
        raw[split] = dict(getattr(manifest, split)._asdict())
    return raw


def from_dict(raw: dict[str, Any]) -> CorpusManifest:
    """Rebuild a manifest from :func:`to_dict` output.

    Args:
        raw: The mapping.

    Returns:
        The manifest.

    Raises:
        TypeError: On a mapping missing a field or carrying an unknown one. A run record
            written by another version of this module is refused rather than partly read.
    """
    fields = dict(raw)
    train = ShardCounts(**fields.pop("train"))
    val = ShardCounts(**fields.pop("val"))
    return CorpusManifest(**fields, train=train, val=val)


def write_manifest(root: Path, manifest: CorpusManifest) -> None:
    """Write the manifest into a corpus directory.

    Args:
        root: Corpus directory.
        manifest: What to write.
    """
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_dict(manifest), indent=2, sort_keys=True)
    (root / MANIFEST_NAME).write_text(text + "\n", encoding="utf-8")


def read_manifest(root: Path) -> CorpusManifest:
    """Read a corpus directory's manifest.

    Args:
        root: Corpus directory.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: When the directory holds no manifest.
        ValueError: When a shard's digest no longer matches the manifest. A corpus that
            changed under a run is the failure this whole module exists to catch.
    """
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {root}")
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    manifest = from_dict(raw)
    for split in SHARDS:
        found = sha256_of(shard_path(root, split))
        stated: str = getattr(manifest, split).digest
        if found != stated:
            raise ValueError(
                f"{split} shard digest is {found} and the manifest says {stated}"
            )
    return manifest
