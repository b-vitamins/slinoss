"""The data layer: the six specs, and the pool one seed produces.

The digests pin the whole path -- spec settings, supplied draw sources, split order,
stacking -- against streams captured from `mad-lab`'s ``mad/data/instances.py`` at the
task configs in ``configs/tasks/*.yml``. A spec whose settings drift off its yml fails
here even though every generator is still correct.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from scripts.mad.instances import IGNORE_INDEX, Instance
from scripts.mad.tasks import (
    LEAKAGE_LIMIT,
    TASKS,
    Pool,
    TaskSpec,
    build_pool,
    leakage,
)

BASELINE_DIGESTS = {
    "icr": "7d8baf25a4231d1f867d07ea0654ed788c4b3d2bd592cbeb84a3fe3fe6c24086",
    "nicr": "38ebb175d1143db17e14f5af1d07ef4405c6e83eac66ceed082ddb0452a237d6",
    "ficr": "b85e1ae08d51c888e848cadca83f8bfcf4f2e4f7c02b375e746021cbba73bdb3",
    "mem": "31f6e4dceaf1024ba224c132877150af7334125cae9f365c986f357bdaac4a7d",
    "comp": "8d3d4c573e3b3ea307130ce234fb7b155b8b645c21db6e9b57e819e1d6d12c19",
    "sc": "5b648fa922e6ca9a6aa21860da0f0ea32aaaa727cb51eaefc6e035973c8813d4",
}
"""sha256 over four train then four test instances at each task's baseline settings,
inputs and targets in draw order, drawn from one generator seeded 12345."""

SMALL = 4
"""Examples per split in the tests. The baselines' own sizes are up to 12800 and the
settings under test are the ones the seed and the spec fix, not the count."""


def digest(pool: Pool) -> str:
    """Hash a pool in draw order.

    Args:
        pool: The pool.

    Returns:
        Hex sha256 over train inputs and targets row by row, then test.
    """
    running = hashlib.sha256()
    for inputs, targets in (
        (pool.train_inputs, pool.train_targets),
        (pool.test_inputs, pool.test_targets),
    ):
        for row_inputs, row_targets in zip(inputs, targets):
            running.update(row_inputs.tobytes())
            running.update(row_targets.tobytes())
    return running.hexdigest()


def small(name: str) -> TaskSpec:
    """A task's baseline with both splits cut to :data:`SMALL`."""
    return TASKS[name].override(num_train=SMALL, num_test=SMALL)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_pool_matches_upstream_stream(name: str) -> None:
    """A pool is the upstream stream: same settings, same order, same draws."""
    assert digest(build_pool(small(name), seed=12345)) == BASELINE_DIGESTS[name]


@pytest.mark.parametrize("name", sorted(TASKS))
def test_pool_is_rectangular_and_typed(name: str) -> None:
    """Both splits stack at one width, int64, and targets align with inputs."""
    pool = build_pool(small(name), seed=12345)
    for inputs, targets in (
        (pool.train_inputs, pool.train_targets),
        (pool.test_inputs, pool.test_targets),
    ):
        assert inputs.shape == (SMALL, pool.width)
        assert targets.shape == inputs.shape
        assert inputs.dtype == np.int64
        assert targets.dtype == np.int64
        assert inputs.min() >= 0
        assert inputs.max() < TASKS[name].vocab_size


@pytest.mark.parametrize("name", sorted(TASKS))
def test_pool_is_a_function_of_its_seed(name: str) -> None:
    """The same seed reproduces a pool; another seed does not."""
    spec = small(name)
    assert digest(build_pool(spec, seed=7)) == digest(build_pool(spec, seed=7))
    assert digest(build_pool(spec, seed=7)) != digest(build_pool(spec, seed=8))


@pytest.mark.parametrize("name", sorted(TASKS))
def test_test_targets_are_masked_where_train_targets_are_not(name: str) -> None:
    """Only the recall tasks mask their test split, and they mask nothing in train.

    Their train target is the shifted input, so every position is supervised;
    ``is_training`` selects the stream and mixing the two would train on the answer.
    Compression reconstructs the whole input and so masks nothing either way, while
    memorization and selective copying supervise the same positions in both splits.
    """
    pool = build_pool(small(name), seed=12345)
    masked = int((pool.test_targets == IGNORE_INDEX).sum())
    train_masked = int((pool.train_targets == IGNORE_INDEX).sum())
    if name == "comp":
        assert masked == 0 and train_masked == 0
    elif name in {"icr", "nicr", "ficr"}:
        assert train_masked == 0
        assert masked > 0
    else:
        assert masked == train_masked > 0


def test_leakage_is_reported_not_hidden() -> None:
    """A pool small enough to repeat itself reports leakage above the limit.

    `mad-lab` prints a warning and continues; here the number rides on the pool so a run
    record carries it and an arm on a leaked pool is identifiable after the fact.
    """
    tiny = TASKS["comp"].override(vocab_size=4, seq_len=4, num_train=64, num_test=64)
    pool = build_pool(tiny, seed=12345)
    assert pool.leakage > LEAKAGE_LIMIT
    assert build_pool(small("comp"), seed=12345).leakage == 0.0
    rows = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert leakage(rows, rows) == 1.0
    assert leakage(rows, rows + 10) == 0.0


def test_axes_are_refused_when_the_task_lacks_them() -> None:
    """Moving an axis a task does not have is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="no axis"):
        TASKS["comp"].override(frac_noise=0.4)
    with pytest.raises(ValueError, match="no axis"):
        TASKS["icr"].override(num_tokens_to_copy=8)
    assert TASKS["nicr"].override(frac_noise=0.4).extra["frac_noise"] == 0.4
    assert TASKS["icr"].override(seq_len=256).seq_len == 256


@pytest.mark.parametrize("name", sorted(TASKS))
def test_ladder_axes_are_real_axes(name: str) -> None:
    """Every difficulty rung a spec lists can actually be applied.

    A ladder naming an axis the task does not have would raise only once a sweep reached
    that rung, which is after the sweep has spent its time.
    """
    spec = TASKS[name]
    for axis, rungs in spec.ladder.items():
        assert rungs
        assert spec.override(**{axis: rungs[0]}) != spec


def test_ragged_instances_are_refused() -> None:
    """A generator whose width depends on its draws cannot stack, so it is an error.

    No MAD generator is ragged. The guard exists because a new one could be, and numpy
    would otherwise produce an object array and fail somewhere downstream instead.
    """

    def ragged(rng: np.random.Generator, *, vocab_size: int, seq_len: int) -> Instance:
        del vocab_size
        width = seq_len + int(rng.integers(0, 2))
        row = np.zeros(width, dtype=np.int64)
        return Instance(row, row)

    spec = TaskSpec(
        name="ragged",
        mad_name="ragged",
        generator=ragged,
        split_policy="invariant",
        vocab_size=16,
        seq_len=8,
        num_train=32,
        num_test=4,
    )
    with pytest.raises(ValueError, match="ragged instances"):
        build_pool(spec, seed=0)

    with pytest.raises(ValueError, match="split_policy"):
        TaskSpec(
            name="bad",
            mad_name="bad",
            generator=ragged,
            split_policy="ignored",  # type: ignore[arg-type]
            vocab_size=16,
            seq_len=8,
            num_train=4,
        )


def test_split_roles_are_explicit() -> None:
    """Only generators whose output depends on split role receive that role."""
    assert {name: spec.split_policy for name, spec in TASKS.items()} == {
        "icr": "required",
        "nicr": "required",
        "ficr": "required",
        "mem": "invariant",
        "comp": "invariant",
        "sc": "invariant",
    }


def test_specs_are_the_task_configs() -> None:
    """The settings that decide a published comparison, spelled out.

    `mad-lab`'s ``configs/tasks/*.yml``, less the training settings the protocol owns.
    Noisy recall's vocabulary is 32 rather than 16 because its noise symbols come out of
    the same range.
    """
    expected: dict[str, dict[str, Any]] = {
        "icr": {"vocab_size": 16, "seq_len": 128, "multi_query": True},
        "nicr": {
            "vocab_size": 32,
            "seq_len": 128,
            "multi_query": True,
            "noise_vocab_size": 16,
            "frac_noise": 0.2,
        },
        "ficr": {
            "vocab_size": 16,
            "seq_len": 128,
            "multi_query": True,
            "k_motif_size": 3,
            "v_motif_size": 3,
        },
        "mem": {"vocab_size": 256, "seq_len": 32},
        "comp": {"vocab_size": 16, "seq_len": 32},
        "sc": {"vocab_size": 16, "seq_len": 256, "num_tokens_to_copy": 16},
    }
    assert set(TASKS) == set(expected)
    for name, settings in expected.items():
        assert TASKS[name].kwargs == settings
        assert TASKS[name].num_test == 1280
    assert TASKS["mem"].num_train == 256
    assert all(TASKS[name].num_train == 12800 for name in expected if name != "mem")
    assert TASKS["comp"].bottleneck
    assert not any(TASKS[name].bottleneck for name in expected if name != "comp")
