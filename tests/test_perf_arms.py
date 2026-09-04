"""The arm every driver reaches an operator through, and the tables keyed by its name.

An operator is benchmarkable only if five tables agree on it: :data:`OPS` names it,
:func:`op_arm` builds it, :data:`OP_REGISTRIES` says what it dispatches through,
:data:`COVERAGE` says what it launches, and :data:`DECLARED` says what class each of
those is held to. A name in the first and missing from any other is a driver that
either crashes on a lookup or, worse, profiles something no rule reads. ``COVERAGE``
against ``DECLARED`` is checked in ``tests/test_perf_coverage.py``, which is import-free
and runs anywhere; the two that need the operator packages are checked here.

The arms are built and not run: building allocates, which is what a missing family
would fail at, and running needs the device the mixer's bands demand. The one arm whose
runner is exercised is ``decode``, on CUDA, because its state advances in place and
nothing else in the suite reads that from a perf arm.
"""

from __future__ import annotations

import pytest
import torch

from slinoss.perf.arms import op_arm
from slinoss.perf.coverage import COVERAGE, FORWARD_ONLY, MODES
from slinoss.perf.dispatch import OP_REGISTRIES
from slinoss.perf.timing import measure
from slinoss.perf.workload import (
    OPS,
    SHAPE_NAMES,
    decode_forward_only,
    decode_shape_by_name,
    make_decode_inputs,
)

CPU = torch.device("cpu")

SMALLEST = SHAPE_NAMES[0]
"""The cheapest shape every family holds, which is the one an allocation test wants."""


@pytest.fixture
def cuda() -> torch.device:
    """The first visible CUDA device, or a skip."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# the tables keyed by operator name
# ---------------------------------------------------------------------------


def test_every_operator_builds_an_arm_under_a_prefix_of_its_own() -> None:
    """A name in ``OPS`` with no family raises, and two names with one prefix merge.

    Region labels are the budget's keys, so two operators sharing a prefix would sum
    into one line and a comparison between them would read as one arm getting slower.
    """
    prefixes: dict[str, str] = {}
    for op in OPS:
        arm = op_arm(op, SMALLEST, CPU, dtype=torch.float32, grads=False)
        assert arm.shape.name == SMALLEST
        assert arm.shape.token_count > 0
        assert arm.shape.describe().startswith(f"{SMALLEST}: ")
        assert arm.prefix
        prefixes[arm.prefix] = op
    assert len(prefixes) == len(OPS)
    with pytest.raises(ValueError, match="unknown op 'attention'"):
        op_arm("attention", SMALLEST, CPU, dtype=torch.float32, grads=False)


def test_every_operator_names_the_registries_its_arm_dispatches_through() -> None:
    """The table with no other reader.

    ``dispatch_verdict`` is what stops a profile of a reference path, and it is keyed
    by operator name. An operator absent from ``OP_REGISTRIES`` raises there, which is
    a driver that cannot start; one present with the wrong list passes a verdict on a
    program it did not run. The first is what this closes.
    """
    assert set(OP_REGISTRIES) == set(OPS)
    for op, registries in OP_REGISTRIES.items():
        assert registries, f"{op} dispatches through nothing"
        labels = [one.label for one in registries]
        assert len(set(labels)) == len(labels), f"{op} asks one registry twice"


def test_an_operator_with_no_step_arm_refuses_the_mode_it_has_no_entry_for() -> None:
    """The other half of ``FORWARD_ONLY``, and the half that makes the absence safe.

    ``tests/test_perf_coverage.py`` asserts the entry is missing. What keeps that from
    being a hole is that the arm cannot be built either, and that the refusal lands
    before anything is allocated so it reaches a caller with no device.
    """
    for op in FORWARD_ONLY:
        assert (op, "forward") in COVERAGE
        with pytest.raises(ValueError, match="has no step arm"):
            op_arm(op, SMALLEST, CPU, dtype=torch.float32, grads=True)
    # Every other operator does build one, so the refusal is a property of the name
    # and not of the mode.
    for op in OPS:
        if op in FORWARD_ONLY:
            continue
        assert op_arm(op, SMALLEST, CPU, dtype=torch.float32, grads=True).differentiable
    assert "step" in MODES


# ---------------------------------------------------------------------------
# the decode arm
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_the_decode_arm_steps_one_state_in_place_under_one_region(
    cuda: torch.device,
) -> None:
    """One region, one state, and the state moves.

    A decode arm that reallocated its state per iteration would measure an allocation
    and a basis fill next to the step, and one that never advanced it would measure a
    first token forever. Both look like a working benchmark from the outside; the
    difference is whether the buffer the second call reads is the one the first call
    wrote.
    """
    arm = op_arm("decode", SMALLEST, cuda, dtype=torch.bfloat16, grads=False)
    run = arm.run(None, arm.prefix)
    timed = measure(run, label="decode", iters=3, warmup=1, device=cuda)
    assert [t.label for t in timed.regions] == ["decode.forward"]
    assert timed.region("decode.forward").spread.sample_count == 3
    # One token per sequence, so the throughput denominator is the batch.
    assert arm.shape.token_count == 1
    # No gradient anywhere: the step is a no_grad node whatever the caller's mode.
    assert arm.differentiable == ()


@pytest.mark.cuda
def test_the_decode_runner_advances_the_buffers_it_was_given(
    cuda: torch.device,
) -> None:
    """In place, in the caller's storage, and again on the next call.

    A captured replay records these addresses, so a step that rebound a field would
    leave the replay writing memory nobody reads. Two calls rather than one, because a
    single step against a zeroed carry moves it whether or not the second step reads
    what the first wrote.
    """
    shape = decode_shape_by_name(SMALLEST)
    inputs = make_decode_inputs(shape, cuda, dtype=torch.bfloat16)
    layers = inputs.state.layers
    # The conv and the key carries start zeroed, so a step that reaches them shows.
    assert not any(bool(layer.conv.any()) for layer in layers)
    run = decode_forward_only(inputs)
    run()
    assert all(bool(layer.conv.any()) for layer in layers)
    assert all(bool(layer.u_prev.any()) for layer in layers)
    first = tuple(layer.ssm.clone() for layer in layers)
    run()
    assert inputs.state.layers is layers
    assert all(
        not torch.equal(was, layer.ssm)
        for was, layer in zip(first, layers, strict=True)
    )
