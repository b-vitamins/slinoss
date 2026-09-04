"""The driver, held to the two properties that let a sweep be run by several hosts at once.

A shard is launched, killed, restarted and harvested without anything coordinating the hosts at
runtime, so the driver has to be resumable and its records have to be self-describing. Both are
tested here against a synthetic corpus, end to end, at a four-step budget.

The refusal in :func:`scripts.tsc.run.lattice_from` is the one that earns its place. Every argument
that selects points is repeatable, and the defaults are the whole protocol, so a bare
``--corpus X`` without ``--shard`` names thirty points; running the first of them and writing one
record would look exactly like a finished single-point run. It is refused instead.

The determinism test is the other one. A record that cannot be reproduced from its own contents is
not evidence, and the seed does two separate jobs here -- the partition is JAX's stream and
initialization, dropout and batch order are torch's -- so this checks that both are pinned by the
one number the record carries.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.tsc.corpus import process, read_manifest
from scripts.tsc.protocol import DATASETS, REFERENCE, SEEDS
from scripts.tsc.run import build_parser, lattice_from, main, record_path
from scripts.tsc.sweep import Point, merge, plan_digest, points, shard
from tests.test_tsc_corpus import write_archive

DATASET = "Heartbeat"
"""The protocol dataset the synthetic corpus impersonates. Its setting includes the time channel
and a 16-wide stream, so the built model is the narrow one this axis actually runs."""

OTHER = "EthanolConcentration"
"""A second protocol dataset, left unprocessed, so a shard covering it has one failing point."""

SEED = SEEDS[0]

SMOKE = (
    "--num-steps",
    "4",
    "--print-steps",
    "2",
    "--device",
    "cpu",
    "--set",
    "blocks=1",
    "--set",
    "batch_size=4",
)
"""Two evaluations and one block. The batch is the published 32, which no split of a 24-instance
pool can serve, so it is overridden here rather than in the protocol."""


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    """A processed corpus holding one dataset under the protocol's name for it."""
    archive = write_archive(tmp_path / "archive", DATASET)
    root = tmp_path / "corpus"
    process(archive, DATASET, root / DATASET)
    return root


@pytest.fixture
def two_dataset_root(tmp_path: Path) -> Path:
    """A processed corpus holding two, so a plan can be sliced across datasets."""
    root = tmp_path / "corpus"
    for name in (DATASET, OTHER):
        process(write_archive(tmp_path / f"archive-{name}", name), name, root / name)
    return root


def argv_for(
    corpus: Path | str, out: Path, *extra: str, dataset: str = DATASET
) -> list[str]:
    """One point's command line.

    Args:
        corpus: Corpus root.
        out: Where records go.
        *extra: Further arguments.
        dataset: Which dataset.

    Returns:
        The argument list.
    """
    return [
        "--corpus",
        str(corpus),
        "--out",
        str(out),
        "--dataset",
        dataset,
        "--mixer",
        "linoss_im",
        "--seed",
        str(SEED),
        *SMOKE,
        *extra,
    ]


def only_point(argv: list[str]) -> Point:
    """The single point an invocation names.

    Args:
        argv: The argument list.

    Returns:
        The point, so a test can name its record before the run writes it.
    """
    selected = points(lattice_from(build_parser().parse_args(argv)))
    assert len(selected) == 1
    return selected[0]


def test_a_single_point_invocation_must_name_exactly_one_of_everything() -> None:
    """Without ``--shard``, more than one dataset, mixer or seed is refused.

    Every selector is repeatable and every default is the whole protocol, so the refusal is what
    separates "run this point" from "run this plan". Without it a bare invocation would run the
    first of thirty points and leave one record, which is indistinguishable from a completed run.
    """
    parse = build_parser().parse_args
    single = lattice_from(parse(argv_for("corpus", Path("out"))))
    assert len(points(single)) == 1
    with pytest.raises(
        ValueError, match="without --shard the invocation must name one"
    ):
        lattice_from(parse(["--corpus", "c"]))
    with pytest.raises(ValueError, match=r"got 1, 1 and 5"):
        lattice_from(parse(["--corpus", "c", "--dataset", DATASET]))
    # With a shard the same invocation is a plan, and the defaults are the protocol's own.
    whole = lattice_from(parse(["--corpus", "c", "--shard", "0/4"]))
    assert len(points(whole)) == len(DATASETS) * len(SEEDS)
    assert {point.mixer for point in points(whole)} == {"linoss_im"}


def test_the_listing_names_the_plan_and_every_point_and_runs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--list`` prints the plan digest and one line per point, and touches no path.

    This is how a fleet's shards are checked against each other before anything is launched, so
    the digest on the first line has to be the plan's own and the lines have to carry the mixer
    settings that distinguish two points of one dataset. The lines come in the order the shard
    will run them, which is not the lattice's order, and each still carries its own index.
    """
    out = tmp_path / "runs"
    argv = [
        "--corpus",
        "/nonexistent",
        "--out",
        str(out),
        "--dataset",
        DATASET,
        "--seed",
        str(SEED),
        "--shard",
        "0/1",
        "--list",
    ]
    lattice = lattice_from(build_parser().parse_args(argv))
    assert main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"plan {plan_digest(lattice)}"
    assert len(lines) == 2
    index, key, settings = lines[1].split("\t")
    assert (int(index), key) == (0, points(lattice)[0].key)
    assert settings == "-", (
        "no mixer setting, and an empty column would misalign the listing"
    )
    assert not out.exists(), "a listing must not create the output directory"

    swept = [*argv, "--sweep", "ssm_dim=16,32"]
    assert main(swept) == 0
    rows = capsys.readouterr().out.splitlines()[1:]
    # The shard's order, which is descending cost, so the wider state runs first.
    assert [row.split("\t")[2] for row in rows] == ["ssm_dim=32", "ssm_dim=16"]
    # The first column stays the point's place in the lattice, not its place in this listing.
    assert [int(row.split("\t")[0]) for row in rows] == [1, 0]
    assert len({row.split("\t")[1] for row in rows}) == 2


def test_a_record_is_named_for_the_key_alone_so_two_shards_share_a_directory(
    tmp_path: Path,
) -> None:
    """The path is the key and nothing else, and an override moves it.

    Two hosts write into one directory with no coordination, which works only if the name is a
    function of the point. The second half is what keeps a sweep honest: a run at an overridden
    ``ssm_dim`` must not land on the record of the run at the published one.
    """
    plain = only_point(argv_for("c", tmp_path))
    overridden = only_point(argv_for("c", tmp_path, "--set", "ssm_dim=32"))
    assert record_path(tmp_path, plain) == tmp_path / f"{plain.key}.json"
    assert record_path(tmp_path, plain) != record_path(tmp_path, overridden)
    assert plain.dataset == overridden.dataset and plain.seed == overridden.seed


def test_a_shard_resumes_by_leaving_the_records_it_already_wrote(
    tmp_path: Path,
) -> None:
    """``--skip-existing`` on a present record runs nothing at all.

    A killed host is restarted with the identical command line, so the skip has to be decided
    before the corpus is read. The corpus path here does not exist, which is what proves nothing
    ran: without the flag the same invocation raises from the corpus load.
    """
    out = tmp_path / "runs"
    argv = argv_for(tmp_path / "absent", out)
    point = only_point(argv)
    out.mkdir(parents=True)
    sentinel = {"plan": "whatever", "key": point.key, "test_accuracy": 0.125}
    record_path(out, point).write_text(json.dumps(sentinel), "utf-8")
    assert main([*argv, "--skip-existing"]) == 0
    assert json.loads(record_path(out, point).read_text("utf-8")) == sentinel
    with pytest.raises(FileNotFoundError):
        main(argv)


def test_a_failed_point_is_recorded_and_the_rest_of_the_shard_still_runs(
    corpus_root: Path, tmp_path: Path
) -> None:
    """``--keep-going`` writes an error record, continues, and returns a non-zero status.

    A shard of forty points must not lose thirty-nine to one bad configuration, and the failure
    has to be visible in the harvest rather than only in a log a host may not have kept. The
    status is what a fleet reads, so it is non-zero even though the shard completed.
    """
    out = tmp_path / "runs"
    argv = [
        "--corpus",
        str(corpus_root),
        "--out",
        str(out),
        "--dataset",
        DATASET,
        "--dataset",
        OTHER,
        "--mixer",
        "linoss_im",
        "--seed",
        str(SEED),
        "--shard",
        "0/1",
        *SMOKE,
    ]
    lattice = lattice_from(build_parser().parse_args(argv))
    assert main([*argv, "--keep-going"]) == 1
    by_dataset = {point.dataset: point for point in points(lattice)}
    failed = json.loads(record_path(out, by_dataset[OTHER]).read_text("utf-8"))
    assert failed["error"].startswith("FileNotFoundError")
    assert (failed["dataset"], failed["seed"]) == (OTHER, SEED)
    assert "test_accuracy" not in failed, "a failure must not read as a result"
    finished = json.loads(record_path(out, by_dataset[DATASET]).read_text("utf-8"))
    assert finished["test_accuracy"] is not None
    assert failed["plan"] == finished["plan"], (
        "both belong to the plan that was launched"
    )
    assert len(merge([failed, finished])) == 2
    with pytest.raises(FileNotFoundError):
        main(argv)


def test_one_point_end_to_end_writes_a_record_a_harvest_accepts(
    corpus_root: Path, tmp_path: Path
) -> None:
    """The record answers what produced the number without the process that produced it.

    Everything asserted here is something an arm is compared on later: the resolved setting after
    overrides, the mixer's settings after the published width is applied, the corpus manifest, the
    partition sizes, both parameter counts and the reference bar the number is measured against. A
    record missing any of them is a number whose meaning has to be reconstructed from a shell
    history.
    """
    out = tmp_path / "runs"
    argv = argv_for(corpus_root, out)
    point = only_point(argv)
    assert main(argv) == 0
    record = json.loads(record_path(out, point).read_text("utf-8"))

    assert record["plan"] == plan_digest(lattice_from(build_parser().parse_args(argv)))
    assert (record["dataset"], record["mixer"], record["seed"]) == (
        DATASET,
        "linoss_im",
        SEED,
    )
    assert record["setting"]["blocks"] == 1, "the fixed override reached the setting"
    assert record["setting"]["batch_size"] == 4
    assert record["setting"]["include_time"] is True
    # The published state width is applied first, so a swept ssm_dim would follow and win.
    assert record["overrides"] == [f"ssm_dim={DATASETS[DATASET].ssm_dim}"]
    assert record["mixer_settings"]["ssm_dim"] == DATASETS[DATASET].ssm_dim
    assert record["mixer_contract"] == {
        "max_length_policy": "unused",
        "initialization": "mixer_constructor; no scaffold reinitialization",
    }
    assert len(record["mixer_constructions"]) == 1
    construction = record["mixer_constructions"][0]
    assert construction["context"] == {
        "max_length_supplied": 3,
        "max_length_policy": "unused",
        "max_length_consumed": None,
    }
    assert record["corpus"] == asdict(read_manifest(corpus_root / DATASET))
    assert record["split_sizes"] == [16, 4, 4]
    assert (record["input_dim"], record["length"], record["classes"]) == (3, 3, 3)
    assert record["lengths"] == {
        "configured_task_length": 3,
        "training_ceiling": 3,
        "evaluation_ceiling": 3,
        "observed_tensor_width": {"train": 3, "validation": 3, "test": 3},
        "mixer_initialization_span": None,
    }
    assert record["seeds"] == {"model": SEED, "partition": SEED, "batch_order": SEED}
    assert len(record["data"]["identity"]) == 64
    assert record["precision"]["parameter_dtype"] == "torch.float32"
    assert record["precision"]["autocast"] is False
    provenance = record["provenance"]
    assert len(provenance["repository_commit"]) == 40
    assert len(provenance["source"]["tree"]) == 40
    assert len(provenance["harness"]["tree"]) == 40
    assert len(provenance["dirty_diff_sha256"]) == 64
    assert provenance["command_argv"]
    assert record["parameters"] > record["mixer_parameters"] > 0
    assert record["reference"] == REFERENCE[DATASET]._asdict()
    assert record["device"] == "cpu"
    assert record["steps"] == 4
    assert [ev["step"] for ev in record["evaluations"]] == [2, 4]
    assert 0.0 <= record["test_accuracy"] <= 1.0
    assert merge([record]) == [record]


def test_two_shards_tile_the_plan_into_one_directory_and_harvest_as_one(
    two_dataset_root: Path, tmp_path: Path
) -> None:
    """Each shard runs its own slice, into the shared directory, and the harvest takes both.

    This is the parallel story end to end rather than at the enumeration: a shard that ignored
    its slice and ran the whole plan would still leave a well-formed harvest, so what is asserted
    after the first shard is the slice it was given and nothing more. The two together are the
    plan, once, under one digest.
    """
    out = tmp_path / "runs"
    base = [
        "--corpus",
        str(two_dataset_root),
        "--out",
        str(out),
        "--dataset",
        DATASET,
        "--dataset",
        OTHER,
        "--mixer",
        "linoss_im",
        "--seed",
        str(SEED),
        "--sweep",
        "lr=1e-3,5e-4",
        *SMOKE,
    ]
    lattice = lattice_from(build_parser().parse_args([*base, "--shard", "0/2"]))
    every = {point.key for point in points(lattice)}
    assert len(every) == 4

    assert main([*base, "--shard", "0/2"]) == 0
    written = {path.stem for path in out.glob("*.json")}
    assert written == {point.key for point in shard(lattice, 0, 2).points}
    assert 0 < len(written) < len(every)

    assert main([*base, "--shard", "1/2"]) == 0
    records = [json.loads(path.read_text("utf-8")) for path in out.glob("*.json")]
    assert {record["key"] for record in records} == every
    assert {record["plan"] for record in records} == {plan_digest(lattice)}
    assert len(merge(records)) == 4


def test_two_runs_of_one_point_report_the_same_numbers(
    corpus_root: Path, tmp_path: Path
) -> None:
    """The record's seed fixes the partition, the initialization, the dropout and the batch order.

    The seed does two jobs on this axis and they run through different generators, so a record is
    only reproducible if both are pinned by it. A model built before the seed was set, or dropout
    drawing from a stream some earlier arm had advanced, would leave a record that cannot be
    re-run to its own number -- and the discrepancy would be small enough to read as noise.
    """
    numbers = []
    for name in ("first", "second"):
        out = tmp_path / name
        argv = argv_for(corpus_root, out)
        assert main(argv) == 0
        record = json.loads(record_path(out, only_point(argv)).read_text("utf-8"))
        numbers.append(
            (
                record["test_accuracy"],
                record["val_accuracy"],
                record["best_step"],
                [(ev["loss"], ev["train_accuracy"]) for ev in record["evaluations"]],
            )
        )
    assert numbers[0] == numbers[1]
