"""Gradients of the reference.

Ground truth is float64 autograd through the reference, never a hand-derived
VJP: a hand derivation shares its algebra with the kernel, so an error in the
algebra passes silently. Every gradient is checked -- ``dU``, ``dtrans``, ``dK``,
``dB``, ``dC``, ``dz0``, ``db_prev``, ``du_prev`` -- and the two implementations
must produce the same analytic gradients, not merely the same forward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch
from torch import Tensor

from slinoss.ops.so3ssd import SO3SSDResult, so3ssd_ref, so3ssm
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

# Smallest legal geometry: N = 16 so 3N = 48, and P = 8. Full-mode gradcheck
# costs two forward passes per input element, so the shape stays minimal.
TINY: dict[str, Any] = {"bsz": 1, "heads": 1, "rows": 8, "lanes": 16}

# float64 autograd against float64 autograd. The gap is reordering roundoff.
# Worst measured over this file: 1.1e-14.
GRAD_REL = 1e-13

GRAD_NAMES: tuple[str, ...] = (
    "dU",
    "dtrans",
    "dK",
    "dB",
    "dC",
    "dz0",
    "db_prev",
    "du_prev",
)

Operator = Callable[..., SO3SSDResult]


def _chunked(chunk: int) -> Operator:
    def call(*operands: Tensor) -> SO3SSDResult:
        u, trans, k, b, c, z0, b_prev, u_prev = operands
        return so3ssd_ref(u, trans, k, b, c, chunk, z0=z0, b_prev=b_prev, u_prev=u_prev)

    return call


def _sequential(*operands: Tensor) -> SO3SSDResult:
    return so3ssm(*operands[:5], z0=operands[5], b_prev=operands[6], u_prev=operands[7])


def _all_outputs(fn: Operator) -> Callable[..., tuple[Tensor, ...]]:
    def call(*operands: Tensor) -> tuple[Tensor, ...]:
        out = fn(*operands)
        return (out.y, out.state, out.b_last, out.u_last)

    return call


def _leaves(inp: ScanInputs) -> tuple[Tensor, ...]:
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    return (*inp.args(), inp.z0, inp.b_prev, inp.u_prev)


def _tiny(**overrides: Any) -> ScanInputs:
    return make_inputs(requires_grad=True, **{**TINY, **overrides})


# ---------------------------------------------------------------------------
# gradcheck
# ---------------------------------------------------------------------------


def test_gradcheck_chunked_full_mode() -> None:
    """Full-mode float64 gradcheck on every operand and every output."""
    inp = _tiny(seqlen=3, seed=31)
    assert torch.autograd.gradcheck(
        _all_outputs(_chunked(16)), _leaves(inp), fast_mode=False, nondet_tol=0.0
    )


def test_gradcheck_sequential_full_mode() -> None:
    inp = _tiny(seqlen=3, seed=31)
    assert torch.autograd.gradcheck(
        _all_outputs(_sequential), _leaves(inp), fast_mode=False, nondet_tol=0.0
    )


@pytest.mark.parametrize(
    ("seqlen", "chunk"),
    [(1, 16), (7, 16), (16, 16), (17, 16), (32, 16), (48, 16), (20, 32), (33, 16)],
)
def test_gradcheck_chunked_shape_sweep(seqlen: int, chunk: int) -> None:
    inp = _tiny(seqlen=seqlen, seed=seqlen)
    assert torch.autograd.gradcheck(
        _all_outputs(_chunked(chunk)), _leaves(inp), fast_mode=True, nondet_tol=0.0
    )


@pytest.mark.parametrize("seqlen", [1, 7, 17])
def test_gradcheck_sequential_shape_sweep(seqlen: int) -> None:
    inp = _tiny(seqlen=seqlen, seed=seqlen)
    assert torch.autograd.gradcheck(
        _all_outputs(_sequential), _leaves(inp), fast_mode=True, nondet_tol=0.0
    )


def test_gradcheck_wider_lane_count() -> None:
    """A lane-indexed reduction correct at one N is not correct at another."""
    inp = _tiny(seqlen=9, heads=2, rows=16, lanes=32, seed=37)
    assert torch.autograd.gradcheck(
        _all_outputs(_chunked(16)), _leaves(inp), fast_mode=True, nondet_tol=0.0
    )


def test_gradcheck_without_initial_state() -> None:
    inp = _tiny(seqlen=17, seed=41, with_state=False, streaming=False)

    def call(*operands: Tensor) -> tuple[Tensor, ...]:
        u, trans, k, b, c = operands
        out = so3ssd_ref(u, trans, k, b, c, 16)
        return (out.y, out.state)

    assert torch.autograd.gradcheck(call, inp.args(), fast_mode=True, nondet_tol=0.0)


def test_gradcheck_state_output_alone() -> None:
    """``state`` is the only output a streaming caller keeps, so its gradient is
    checked on its own rather than only inside a sum with ``y``."""
    inp = _tiny(seqlen=20, seed=43)

    def call(*operands: Tensor) -> Tensor:
        return _chunked(16)(*operands).state

    assert torch.autograd.gradcheck(call, _leaves(inp), fast_mode=True, nondet_tol=0.0)


# ---------------------------------------------------------------------------
# The two implementations must agree on gradients, not only on outputs
# ---------------------------------------------------------------------------


def _cotangents(inp: ScanInputs, seed: int) -> tuple[Tensor, ...]:
    bsz, heads, seqlen, rows = (int(d) for d in inp.U.shape)
    state_dim = int(inp.B.shape[-1])
    gen = torch.Generator().manual_seed(seed)

    def like(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    return (
        like(bsz, heads, seqlen, rows),
        like(bsz, heads, rows, state_dim),
        like(bsz, heads, state_dim),
        like(bsz, heads, rows),
    )


def _grads(
    fn: Operator, inp: ScanInputs, cotangents: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    leaves = _leaves(inp)
    out = fn(*leaves)
    return tuple(
        torch.autograd.grad(
            (out.y, out.state, out.b_last, out.u_last), leaves, tuple(cotangents)
        )
    )


@pytest.mark.parametrize(("seqlen", "chunk"), [(8, 16), (33, 16), (40, 32), (64, 16)])
def test_analytic_gradients_agree(seqlen: int, chunk: int) -> None:
    ref = _tiny(seqlen=seqlen, seed=47 + seqlen)
    fast = _tiny(seqlen=seqlen, seed=47 + seqlen)
    cotangents = _cotangents(ref, 53)
    want = _grads(_sequential, ref, cotangents)
    got = _grads(_chunked(chunk), fast, cotangents)
    for name, a, b in zip(GRAD_NAMES, got, want):
        assert_max_rel(a, b, GRAD_REL, f"{name} at T={seqlen} L={chunk}")


def test_forward_and_backward_are_connected() -> None:
    """Compute the output with the chunked path, backpropagate through it, and
    compare end to end against the sequential path. A backward validated against
    a surrogate forward hides any gap between the surrogate and the real thing."""
    ref = _tiny(seqlen=40, seed=59)
    fast = _tiny(seqlen=40, seed=59)

    want_out = _sequential(*_leaves(ref))
    got_out = _chunked(16)(*_leaves(fast))
    assert_max_rel(got_out.y, want_out.y, GRAD_REL, "y")
    assert_max_rel(got_out.state, want_out.state, GRAD_REL, "state")

    want_out.y.square().sum().backward()
    got_out.y.square().sum().backward()
    for name, a, b in zip(GRAD_NAMES, _leaves(fast), _leaves(ref)):
        assert a.grad is not None and b.grad is not None
        assert_max_rel(a.grad, b.grad, GRAD_REL, name)


def test_gradients_are_finite_under_saturated_decay() -> None:
    inp = _tiny(seqlen=48, seed=61, ls_bias=25.0)
    out = _chunked(16)(*_leaves(inp))
    (out.y.sum() + out.state.sum()).backward()
    for name, leaf in zip(GRAD_NAMES, _leaves(inp)):
        assert leaf.grad is not None, name
        assert bool(torch.isfinite(leaf.grad).all()), name
