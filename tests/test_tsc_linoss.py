"""LinOSS in torch, against a sequential complex-arithmetic transcription of the reference.

The layer is a baseline, so its job is to be the reference and nothing else. The strongest check
available without running JAX is a second implementation that shares no code with the first:
:func:`sequential_reference` walks the recurrence one timepoint at a time in python with complex
numbers, and the module has to agree with it to float64 precision at both discretizations.

That comparison covers the parts most likely to be wrong and hardest to see. The doubling scan is
an optimization over the reference's ``associative_scan`` and a wrong level count or a wrong
shift silently truncates the receptive field, which reads as a modelling result. The readout is
the *position* half only, ``Re(C x2) + D u``, and taking the velocity half or dropping the sign on
the imaginary product both leave a layer that trains.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from scripts.tsc.linoss import (
    DISCRETIZATIONS,
    LinOSSRecurrence,
    Transition,
    doubling_scan,
    transition,
)

# A power of two, one either side of it, one, and a length that needs four doubling levels.
LENGTHS = (1, 2, 3, 4, 5, 8, 9, 13)


def sequential_reference(layer: LinOSSRecurrence, x: Tensor) -> np.ndarray:
    """The reference recurrence, one timepoint at a time, in complex numpy.

    Shares no code with :class:`scripts.tsc.linoss.LinOSSRecurrence`: the state is a complex
    vector, the transition is rebuilt from the closed form, and the loop is sequential.

    Args:
        layer: The module whose parameters to read.
        x: ``(B,L,H)`` float64.

    Returns:
        ``(B,L,H)`` float64.
    """
    weights = {
        name: p.detach().numpy().astype(np.float64)
        for name, p in layer.named_parameters()
    }
    a = np.maximum(weights["a_diag"], 0.0)
    dt = 1.0 / (1.0 + np.exp(-weights["steps"]))
    b = weights["input_weight"][..., 0] + 1j * weights["input_weight"][..., 1]
    c = weights["output_weight"][..., 0] + 1j * weights["output_weight"][..., 1]
    if layer.discretization == "IM":
        schur = 1.0 / (1.0 + dt * dt * a)
        m11, m12, m21, m22 = (
            1.0 - dt * dt * a * schur,
            -dt * a * schur,
            dt * schur,
            schur,
        )
        gains = (m11 * dt, m21 * dt)
    else:
        m11, m12, m21, m22 = np.ones_like(a), -dt * a, dt, 1.0 - dt * dt * a
        gains = (dt, dt * dt)

    series = x.detach().numpy().astype(np.float64)
    out = np.zeros_like(series)
    for instance in range(series.shape[0]):
        velocity = np.zeros(a.shape, dtype=np.complex128)
        position = np.zeros(a.shape, dtype=np.complex128)
        for step in range(series.shape[1]):
            forcing = b @ series[instance, step]
            velocity, position = (
                m11 * velocity + m12 * position + gains[0] * forcing,
                m21 * velocity + m22 * position + gains[1] * forcing,
            )
            out[instance, step] = (
                np.real(c @ position) + weights["skip"] * series[instance, step]
            )
    return out


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_the_layer_matches_a_sequential_complex_transcription(
    discretization: str,
) -> None:
    """Every timepoint of the layer's output, against the loop, at both discretizations.

    A single comparison covering the transition, both forcing coefficients, the scan and the
    readout. In float64, so a mismatch is a formula and not an accumulation.
    """
    torch.manual_seed(11)
    layer = LinOSSRecurrence(6, ssm_dim=5, discretization=discretization).double()
    x = torch.randn(3, 9, 6, dtype=torch.float64)
    with torch.no_grad():
        found = layer(x).numpy()
    assert np.allclose(found, sequential_reference(layer, x), atol=1e-12)


def test_the_doubling_scan_matches_a_naive_prefix_sum() -> None:
    """``x_t = M x_{t-1} + F_t`` from ``x_{-1} = 0``, at lengths that exercise every level count.

    The scan squares ``M`` once per level and squares it only when another level follows. One
    extra squaring, or a shift by the wrong span, truncates the window at exactly the lengths a
    power-of-two test cannot see.
    """
    torch.manual_seed(3)
    step = Transition(*(torch.randn(4, dtype=torch.float64) * 0.3 for _ in range(4)))
    for length in LENGTHS:
        first = torch.randn(2, length, 4, dtype=torch.float64)
        second = torch.randn(2, length, 4, dtype=torch.float64)
        got_first, got_second = doubling_scan(step, first, second)
        state = (torch.zeros(2, 4, dtype=torch.float64),) * 2
        wanted_first, wanted_second = [], []
        for index in range(length):
            moved = step.apply(*state)
            state = (moved[0] + first[:, index], moved[1] + second[:, index])
            wanted_first.append(state[0])
            wanted_second.append(state[1])
        assert torch.allclose(
            got_first, torch.stack(wanted_first, dim=1), atol=1e-12
        ), length
        assert torch.allclose(
            got_second, torch.stack(wanted_second, dim=1), atol=1e-12
        ), length


def test_squaring_a_transition_is_matrix_multiplication() -> None:
    """``Transition.squared`` is ``M @ M`` entrywise, which is what every scan level relies on."""
    torch.manual_seed(5)
    entries = [torch.randn(3, dtype=torch.float64) for _ in range(4)]
    squared = Transition(*entries).squared()
    matrices = torch.stack(entries).reshape(2, 2, 3).permute(2, 0, 1)
    wanted = (matrices @ matrices).permute(1, 2, 0).reshape(4, 3)
    assert torch.allclose(torch.stack(tuple(squared)), wanted, atol=1e-12)


def test_the_two_discretizations_are_the_published_closed_forms() -> None:
    """IM carries the Schur factor into all four entries and both gains; IMEX carries none.

    Written out here from the paper's expressions rather than read back from the module, so a
    transposed entry or a gain applied to the wrong half fails.
    """
    a = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    dt = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)

    implicit, velocity_gain, position_gain = transition(a, dt, "IM")
    schur = 1.0 / (1.0 + dt * dt * a)
    for found, wanted in zip(
        implicit,
        (1.0 - dt * dt * a * schur, -dt * a * schur, dt * schur, schur),
        strict=True,
    ):
        assert torch.allclose(found, wanted, atol=1e-15)
    assert torch.allclose(velocity_gain, implicit[0] * dt, atol=1e-15)
    assert torch.allclose(position_gain, implicit[2] * dt, atol=1e-15)

    imex, velocity_gain, position_gain = transition(a, dt, "IMEX")
    for found, wanted in zip(
        imex, (torch.ones_like(a), -dt * a, dt, 1.0 - dt * dt * a), strict=True
    ):
        assert torch.allclose(found, wanted, atol=1e-15)
    assert torch.allclose(velocity_gain, dt, atol=1e-15)
    assert torch.allclose(position_gain, dt * dt, atol=1e-15)


def test_an_unknown_discretization_is_refused_rather_than_defaulted() -> None:
    """The reference prints a message and leaves the name unbound, so a typo trains on nothing.

    Refused at both entry points: the function and the module's constructor.
    """
    ones = torch.ones(2)
    with pytest.raises(ValueError, match="discretization must be one of"):
        transition(ones, ones, "Heun")
    with pytest.raises(ValueError, match="discretization must be one of"):
        LinOSSRecurrence(4, ssm_dim=2, discretization="Heun")


def test_bad_shapes_are_refused_at_the_boundary() -> None:
    """Mismatched forcings, an empty sequence axis, a wrong stream width, a non-positive size.

    All four otherwise surface as a broadcast that happens to work or a matmul error deep in a
    lane rather than at the call that was wrong.
    """
    step = Transition(*(torch.ones(3) for _ in range(4)))
    with pytest.raises(ValueError, match="forcings are"):
        doubling_scan(step, torch.zeros(1, 4, 3), torch.zeros(1, 5, 3))
    with pytest.raises(ValueError, match="no timepoints"):
        doubling_scan(step, torch.zeros(1, 0, 3), torch.zeros(1, 0, 3))
    layer = LinOSSRecurrence(4, ssm_dim=2)
    with pytest.raises(ValueError, match="expected last axis 4"):
        layer(torch.zeros(1, 3, 5))
    with pytest.raises(ValueError, match="must be positive"):
        LinOSSRecurrence(0, ssm_dim=2)


def test_the_parameter_shapes_and_widths_are_the_references() -> None:
    """``B`` is ``(P,H,2)`` and ``C`` is ``(H,P,2)``: complex as two real leaves, not one complex.

    An optimizer step on a complex parameter is not the step it takes on the two real parts, so
    the split is the reference's behaviour and not a storage choice. The trailing 2 also means the
    parameter count is ``2PH + 2HP + H + 2P``, which is what the ablation subtracts.
    """
    layer = LinOSSRecurrence(8, ssm_dim=5, discretization="IM")
    shapes = {name: tuple(p.shape) for name, p in layer.named_parameters()}
    assert shapes == {
        "a_diag": (5,),
        "input_weight": (5, 8, 2),
        "output_weight": (8, 5, 2),
        "skip": (8,),
        "steps": (5,),
    }
    assert (
        sum(p.numel() for p in layer.parameters()) == 2 * 5 * 8 + 2 * 8 * 5 + 8 + 2 * 5
    )
    assert layer(torch.zeros(2, 7, 8)).shape == (2, 7, 8)
    assert isinstance(layer, nn.Module)
