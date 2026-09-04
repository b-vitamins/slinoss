"""The operator under test, inside the harness, on a device.

Every other test in this suite runs on the CPU against a control mixer, which cannot settle
the one thing an arm depends on: that the tree's own mixer builds from the registry at the
scaffold's width, is causal, reads a length it was never built for, and carries the protocol
at both supervision modes. Those four are the harness's own smoke test.

The third is the load-bearing one. The recurrence holds no length-dependent buffer, which is
why this axis can evaluate past the trained length at all, so one module is forwarded at four
widths rather than rebuilt per width.

No figure here is a measurement. Three steps on eight items settles that the arm runs and
nothing about what it scores.
"""

from __future__ import annotations

import math

import pytest
import torch

from slinoss import _C

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
if not _C.is_available():
    pytest.skip(
        f"{_C.EXTENSION} is not built; run {_C.BUILD_COMMAND}",
        allow_module_level=True,
    )

from scripts.state_tracking.instances import SplitConfig
from scripts.state_tracking.mixers import resolve
from scripts.state_tracking.run import run_arm
from scripts.state_tracking.tasks import AUTOMATA
from scripts.state_tracking.tasks import resolve as resolve_task
from scripts.state_tracking.train import TrainConfig

pytestmark = [pytest.mark.cuda]

D_MODEL = 128
"""The protocol's width. ``d_inner`` is ``2 * 128`` and ``d_head`` 64, so four heads."""

MAX_LENGTH = 40
TINY = TrainConfig(
    num_steps=3,
    batch_size=8,
    print_steps=2,
    early_stop_threshold=2.0,
    band_width=4,
    device="cuda",
)
TRAIN_SPLIT = SplitConfig(min_length=3, max_length=8, seed=0)
VAL_SPLIT = SplitConfig(min_length=8, max_length=12, seed=0, count=8)
MODEL_ARGS = {"d_model": D_MODEL, "n_layers": 1, "dropout": 0.0, "use_glu": False}


def _mixer(max_length: int = MAX_LENGTH) -> torch.nn.Module:
    """The registry's slinoss entry, built at its own defaults, on the device."""
    return resolve("slinoss").factory(D_MODEL, max_length).cuda()


def test_the_registry_builds_the_mixer_and_a_gradient_reaches_it() -> None:
    """The factory's module maps the stream to itself and every parameter takes a gradient.

    A parameter with no gradient is frozen in fact and not in the record, which would put
    the arm's reported parameter count over what it trained.
    """
    mixer = _mixer()
    x = torch.randn(2, 12, D_MODEL, device="cuda", requires_grad=True)
    out = mixer(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert bool(torch.isfinite(out).all())
    out.square().mean().backward()
    for name, param in mixer.named_parameters():
        assert param.grad is not None, name
        assert bool(torch.isfinite(param.grad).all()), name
    assert x.grad is not None


def test_the_mixer_is_causal() -> None:
    """Perturbing the last position moves no earlier output.

    The scan is chunked, so a chunk holds the perturbed token alongside earlier ones; the
    earlier outputs must still be a function of their own prefix alone. A leak here would
    void every number on the axis at once, since the label sits at the last position.
    """
    mixer = _mixer()
    mixer.eval()
    x = torch.randn(2, 12, D_MODEL, device="cuda")
    perturbed = x.clone()
    perturbed[:, -1] += 10.0
    with torch.no_grad():
        before = mixer(x)
        after = mixer(perturbed)
    assert torch.allclose(before[:, :-1], after[:, :-1], rtol=0.0, atol=1e-5)
    assert not torch.allclose(before[:, -1], after[:, -1])


@pytest.mark.parametrize("length", [3, 8, 40, 64])
def test_one_module_reads_a_length_it_was_not_built_for(length: int) -> None:
    """One module, four widths, including one over the ``max_length`` it was handed.

    The registry declares ``max_length`` unused because the recurrence carries no
    length-dependent buffer. That is what an evaluation past the trained length rests on,
    so a buffer introduced later has to fail here rather than at the first long batch.
    """
    mixer = _mixer(max_length=8)
    with torch.no_grad():
        out = mixer(torch.randn(2, length, D_MODEL, device="cuda"))
    assert out.shape == (2, length, D_MODEL)
    assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize("task_name", ["parity", "A5"])
def test_the_protocol_runs_at_both_supervision_modes(task_name: str) -> None:
    """A whole arm through the driver, at three steps, on both halves of the suite.

    ``parity`` supervises one position and ``A5`` supervises every one, and the two reach
    the loss by different masks. The record is asserted, not the accuracy: three steps
    measures nothing.
    """
    record = run_arm(
        resolve_task(task_name),
        "slinoss",
        [],
        MODEL_ARGS,
        TINY,
        TRAIN_SPLIT,
        VAL_SPLIT,
        quiet=True,
    )
    assert record["task"] == task_name
    assert record["steps_run"] == 3
    assert record["mixer"] == "slinoss"
    assert record["mixer_settings"]["d_state"] == 144
    assert record["parameters"] > record["mixer_parameters"] > 0
    assert [point[0] for point in record["points"]] == [0, 2]
    assert all(math.isfinite(point[2]) for point in record["points"])
    assert all(math.isfinite(point[3]) for point in record["points"])
    assert record["best"]["positions"] > 0
    assert record["best"]["bands"]
    assert record["solved"] is False


def test_the_two_halves_differ_only_in_the_head_and_the_mask() -> None:
    """A group arm and an automaton arm differ by the vocabulary alone.

    The scaffold and the mixer are identical across the suite, so a difference between two
    tasks is the task. A mixer whose parameter count moved with the vocabulary would put
    every cross-task comparison on a different model.
    """
    counts = {}
    for name in ("parity", "A5"):
        record = run_arm(
            resolve_task(name),
            "slinoss",
            [],
            MODEL_ARGS,
            TINY,
            TRAIN_SPLIT,
            VAL_SPLIT,
            quiet=True,
        )
        counts[name] = (record["parameters"], record["mixer_parameters"])
    assert counts["parity"][1] == counts["A5"][1]
    assert counts["parity"][0] < counts["A5"][0]
    assert AUTOMATA["parity"].vocab_size == 3
