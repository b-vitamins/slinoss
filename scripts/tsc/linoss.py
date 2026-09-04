"""LinOSS, in torch: the reference baseline this harness has to reproduce.

The reference is the damped-oscillator state-space layer of the LinOSS paper, in JAX. It is
transcribed here rather than called, so one process runs every arm on one framework and a
difference between two arms is not a difference between two autodiff libraries.

The layer's state is a pair per oscillator, ``(velocity, position)``, advanced by a real 2x2
matrix that does not depend on the token::

    IM      schur = 1 / (1 + dt^2 A)
            M = [[1 - dt^2 A schur, -dt A schur], [dt schur, schur]]
            F = (M11 Bu dt, M21 Bu dt)

    IMEX    M = [[1, -dt A], [dt, 1 - dt^2 A]]
            F = (Bu dt, Bu dt^2)

with ``A = relu(A_diag)`` and ``dt = sigmoid(steps)``, both per oscillator. The readout is the
position half only, ``real(C x2)``, plus a per-channel skip ``D u``.

Because ``M`` is time invariant the scan needs no associative combine over matrices. Doubling
on the state alone is enough: :func:`doubling_scan` keeps a window and squares ``M`` once per
level, so it does ``ceil(log2 L)`` levels of ``(B,L,P)`` work against the reference's
``jax.lax.associative_scan``, which carries a four-word matrix alongside every element. Same
depth, a quarter of the state traffic, and the invariant is stated on the function.

``B`` and ``C`` are complex, held as a trailing pair of real components because an optimizer
step on a complex parameter is not the same step as on its two real parts. ``M`` is real, so
the recurrence runs on the real and imaginary halves at once with no complex arithmetic.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "DISCRETIZATIONS",
    "LinOSSRecurrence",
    "Transition",
    "doubling_scan",
    "transition",
]

DISCRETIZATIONS = ("IM", "IMEX")
"""The two discretizations the reference implements."""


class Transition(tuple[Tensor, Tensor, Tensor, Tensor]):
    """A real 2x2 transition, per oscillator, as four ``(P,)`` tensors in row order.

    A tuple subclass and not a dataclass because it is squared once per scan level and the
    squaring reads all four entries; naming them ``m11 m12 m21 m22`` at every call site is
    what makes the composition readable.
    """

    __slots__ = ()

    def __new__(cls, m11: Tensor, m12: Tensor, m21: Tensor, m22: Tensor) -> Transition:
        """Build a transition.

        Args:
            m11: Velocity from velocity.
            m12: Velocity from position.
            m21: Position from velocity.
            m22: Position from position.

        Returns:
            The transition.
        """
        return super().__new__(cls, (m11, m12, m21, m22))

    def squared(self) -> Transition:
        """The transition applied twice.

        Returns:
            ``M @ M``.
        """
        m11, m12, m21, m22 = self
        return Transition(
            m11 * m11 + m12 * m21,
            m11 * m12 + m12 * m22,
            m21 * m11 + m22 * m21,
            m21 * m12 + m22 * m22,
        )

    def apply(self, first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
        """Advance a state pair.

        Args:
            first: Velocity half, ``(..., P)``.
            second: Position half, ``(..., P)``.

        Returns:
            The advanced pair.
        """
        m11, m12, m21, m22 = self
        return (m11 * first + m12 * second, m21 * first + m22 * second)


def transition(
    a_diag: Tensor, dt: Tensor, discretization: str
) -> tuple[Transition, Tensor, Tensor]:
    """The transition and the two forcing coefficients.

    Args:
        a_diag: ``(P,)`` stiffness, already through ``relu``.
        dt: ``(P,)`` step, already through ``sigmoid``.
        discretization: ``IM`` or ``IMEX``.

    Returns:
        The transition, then the coefficient the velocity forcing multiplies ``Bu`` by and
        the coefficient the position forcing multiplies ``Bu`` by. Both ``(P,)``.

    Raises:
        ValueError: On an unknown discretization. The reference prints a message and returns
            an unbound name, so a typo there trains a model on nothing.
    """
    if discretization == "IM":
        schur = 1.0 / (1.0 + dt * dt * a_diag)
        found = Transition(
            1.0 - dt * dt * a_diag * schur,
            -dt * a_diag * schur,
            dt * schur,
            schur,
        )
        return found, found[0] * dt, found[2] * dt
    if discretization == "IMEX":
        found = Transition(
            torch.ones_like(a_diag),
            -dt * a_diag,
            dt,
            1.0 - dt * dt * a_diag,
        )
        return found, dt, dt * dt
    raise ValueError(
        f"discretization must be one of {DISCRETIZATIONS}, got {discretization!r}"
    )


def doubling_scan(
    step: Transition, first: Tensor, second: Tensor
) -> tuple[Tensor, Tensor]:
    """Run a time-invariant linear recurrence over the sequence axis.

    Computes ``x_t = M x_{t-1} + F_t`` with ``x_{-1} = 0``, which is
    ``x_t = sum_{j<=t} M^{t-j} F_j``.

    The invariant after ``k`` levels is that each output holds the window of the ``2**k`` most
    recent forcings, ``sum_{j > t - 2**k} M^{t-j} F_j``, and that ``step`` holds ``M**(2**k)``.
    Combining a window with the same window shifted by ``2**k`` and advanced by ``M**(2**k)``
    doubles it, so ``ceil(log2 L)`` levels cover the whole prefix.

    Args:
        step: The transition, real, ``(P,)`` per entry.
        first: Velocity forcing, ``(..., L, P)``.
        second: Position forcing, ``(..., L, P)``.

    Returns:
        The two state halves, each the shape of its forcing.

    Raises:
        ValueError: On forcings of different shapes, or on an empty sequence axis.
    """
    if first.shape != second.shape:
        raise ValueError(f"forcings are {tuple(first.shape)} and {tuple(second.shape)}")
    length = int(first.shape[-2])
    if length < 1:
        raise ValueError("forcing has no timepoints")
    span = 1
    while span < length:
        # Shift along the sequence axis, zero-filling the head: a window that runs off the
        # start of the sequence contributes nothing, which is the x_{-1} = 0 initial state.
        shifted_first = first.narrow(-2, 0, length - span)
        shifted_second = second.narrow(-2, 0, length - span)
        moved_first, moved_second = step.apply(shifted_first, shifted_second)
        pad = (0, 0, span, 0)
        first = first + nn.functional.pad(moved_first, pad)
        second = second + nn.functional.pad(moved_second, pad)
        span *= 2
        if span < length:
            step = step.squared()
    return (first, second)


class LinOSSRecurrence(nn.Module):
    """One LinOSS layer, mapping ``(B,L,H)`` to ``(B,L,H)``.

    Initialization is the reference's, distribution for distribution::

        A_diag  U(0, 1)                     (P,)
        B       U(-1/sqrt(H), 1/sqrt(H))    (P, H, 2)
        C       U(-1/sqrt(P), 1/sqrt(P))    (H, P, 2)
        D       N(0, 1)                     (H,)
        steps   U(0, 1)                     (P,)

    Not draw for draw: the reference's values come from a JAX key tree and this one's come from
    torch's generator. The seed fixes the data partition in this harness and the harness fixes
    initialization, which is stated in :mod:`scripts.tsc.split`. Matching the distributions is
    what parity means here; matching the bits would require running JAX.

    Args:
        d_model: Stream width, the reference's ``H``.
        ssm_dim: Oscillators, the reference's ``ssm_size`` and ``P``. The scan carries
            ``2 * ssm_dim`` reals per channel-free position, not ``ssm_dim``.
        discretization: ``IM`` or ``IMEX``.

    Raises:
        ValueError: On a non-positive width or state size, or an unknown discretization.
    """

    def __init__(
        self,
        d_model: int,
        ssm_dim: int = 64,
        discretization: str = "IM",
    ) -> None:
        super().__init__()
        if d_model < 1 or ssm_dim < 1:
            raise ValueError(
                f"d_model {d_model} and ssm_dim {ssm_dim} must be positive"
            )
        if discretization not in DISCRETIZATIONS:
            raise ValueError(
                f"discretization must be one of {DISCRETIZATIONS}, got {discretization!r}"
            )
        self.d_model = d_model
        self.ssm_dim = ssm_dim
        self.discretization = discretization
        self.a_diag = nn.Parameter(torch.rand(ssm_dim))
        self.input_weight = nn.Parameter(
            torch.empty(ssm_dim, d_model, 2).uniform_(
                -1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model)
            )
        )
        self.output_weight = nn.Parameter(
            torch.empty(d_model, ssm_dim, 2).uniform_(
                -1.0 / math.sqrt(ssm_dim), 1.0 / math.sqrt(ssm_dim)
            )
        )
        self.skip = nn.Parameter(torch.randn(d_model))
        self.steps = nn.Parameter(torch.rand(ssm_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Advance the oscillators over the sequence and read their positions.

        Args:
            x: ``(B,L,H)``.

        Returns:
            ``(B,L,H)``.

        Raises:
            ValueError: When the last axis is not ``d_model``.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"input is {tuple(x.shape)}, expected last axis {self.d_model}"
            )
        a_diag = nn.functional.relu(self.a_diag)
        dt = torch.sigmoid(self.steps)
        step, velocity_gain, position_gain = transition(a_diag, dt, self.discretization)

        # (2,B,L,P): index 0 real, 1 imaginary. M is real, so both halves run one scan.
        forcing = torch.stack(
            (
                x @ self.input_weight[..., 0].transpose(0, 1),
                x @ self.input_weight[..., 1].transpose(0, 1),
            )
        )
        _, position = doubling_scan(
            step, forcing * velocity_gain, forcing * position_gain
        )
        out = position[0] @ self.output_weight[..., 0].transpose(0, 1) - position[
            1
        ] @ self.output_weight[..., 1].transpose(0, 1)
        return out + self.skip * x
