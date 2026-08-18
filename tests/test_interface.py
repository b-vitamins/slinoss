"""The public operator: autograd wiring, saved-storage discipline, and backends.

The Function is checked by ``gradcheck`` in float64 on every leaf, and its
forward and backward are checked together against the sequential reference: the
output is produced by the shipped path and then differentiated through, so a gap
between the path under test and the path the gradient was derived for cannot
hide.

The saved set is asserted, not assumed. Every tensor the graph holds must be one
of the operator's own inputs; anything else means an intermediate is being
stashed instead of recomputed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor

from slinoss.ops.so3ssd import (
    SO3SSDResult,
    get,
    names,
    register,
    resolve,
    so3ssd,
    so3ssd_ref,
    so3ssm,
)
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

TINY: dict[str, Any] = {"bsz": 1, "heads": 1, "rows": 8, "lanes": 16}

# float64 autograd through the Function against float64 autograd through the
# sequential reference. The gap is reordering roundoff.
# Worst measured over this file: 4.4e-15.
IFACE_REL = 1e-13

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


def _tiny(**overrides: Any) -> ScanInputs:
    return make_inputs(requires_grad=True, **{**TINY, **overrides})


def _leaves(inp: ScanInputs) -> tuple[Tensor, ...]:
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    return (*inp.args(), inp.z0, inp.b_prev, inp.u_prev)


def _outputs(
    chunk: int, backend: str | None = None
) -> Callable[..., tuple[Tensor, ...]]:
    def call(*operands: Tensor) -> tuple[Tensor, ...]:
        u, trans, k, b, c, z0, b_prev, u_prev = operands
        out = so3ssd(
            u,
            trans,
            k,
            b,
            c,
            chunk,
            z0=z0,
            b_prev=b_prev,
            u_prev=u_prev,
            backend=backend,
        )
        return (out.y, out.state, out.b_last, out.u_last)

    return call


# ---------------------------------------------------------------------------
# gradcheck through the Function
# ---------------------------------------------------------------------------


def test_gradcheck_full_mode() -> None:
    """Full-mode float64 gradcheck on every leaf and every output. Undefined
    output cotangents are part of what gradcheck exercises here."""
    inp = _tiny(seqlen=3, seed=211)
    assert torch.autograd.gradcheck(
        _outputs(16), _leaves(inp), fast_mode=False, nondet_tol=0.0
    )


@pytest.mark.parametrize(
    ("seqlen", "chunk"),
    [(1, 16), (7, 16), (16, 16), (17, 16), (33, 16), (20, 32), (48, 16)],
)
def test_gradcheck_shape_sweep(seqlen: int, chunk: int) -> None:
    inp = _tiny(seqlen=seqlen, seed=223 + seqlen)
    assert torch.autograd.gradcheck(
        _outputs(chunk), _leaves(inp), fast_mode=True, nondet_tol=0.0
    )


def test_gradcheck_wider_lane_count() -> None:
    inp = _tiny(seqlen=9, heads=2, rows=16, lanes=32, seed=227)
    assert torch.autograd.gradcheck(
        _outputs(16), _leaves(inp), fast_mode=True, nondet_tol=0.0
    )


def test_gradcheck_without_carry() -> None:
    inp = _tiny(seqlen=17, seed=229, with_state=False, streaming=False)

    def call(*operands: Tensor) -> tuple[Tensor, ...]:
        u, trans, k, b, c = operands
        out = so3ssd(u, trans, k, b, c, 16)
        return (out.y, out.state, out.b_last, out.u_last)

    assert torch.autograd.gradcheck(call, inp.args(), fast_mode=True, nondet_tol=0.0)


def test_gradcheck_state_output_alone() -> None:
    """``state`` is the only output a streaming caller keeps."""
    inp = _tiny(seqlen=20, seed=233)

    def call(*operands: Tensor) -> Tensor:
        return _outputs(16)(*operands)[1]

    assert torch.autograd.gradcheck(call, _leaves(inp), fast_mode=True, nondet_tol=0.0)


# ---------------------------------------------------------------------------
# The shipped path is the path the gradient was derived for
# ---------------------------------------------------------------------------


FIELDS: tuple[str, ...] = ("y", "state", "b_last", "u_last")


def test_forward_is_bitwise_the_reference() -> None:
    """The Function must not perturb the forward it wraps."""
    inp = make_inputs(**TINY, seqlen=40, seed=239)
    got = so3ssd(*inp.args(), 16, **inp.kw())
    want = so3ssd_ref(*inp.args(), 16, **inp.kw())
    for name in FIELDS:
        assert torch.equal(getattr(got, name), getattr(want, name)), name


def test_result_is_a_named_type_with_contiguous_fields() -> None:
    inp = make_inputs(**TINY, seqlen=33, seed=307)
    out = so3ssd(*inp.args(), 16, **inp.kw())
    assert isinstance(out, SO3SSDResult)
    for name in FIELDS:
        assert getattr(out, name).is_contiguous(), name


def test_forward_and_backward_are_connected() -> None:
    """Produce the output with the public path, backpropagate through it, and
    compare end to end against the sequential reference."""
    fast = _tiny(seqlen=40, seed=241)
    ref = _tiny(seqlen=40, seed=241)

    got = so3ssd(*fast.args(), 16, **fast.kw())
    want = so3ssm(*ref.args(), **ref.kw())
    assert_max_rel(got.y, want.y, IFACE_REL, "interface y")
    assert_max_rel(got.state, want.state, IFACE_REL, "interface state")

    (got.y.square().sum() + got.state.square().sum()).backward()
    (want.y.square().sum() + want.state.square().sum()).backward()
    for name, a, b in zip(GRAD_NAMES, _leaves(fast), _leaves(ref)):
        assert a.grad is not None and b.grad is not None, name
        assert_max_rel(a.grad, b.grad, IFACE_REL, f"interface {name}")


def test_streaming_split_matches_one_shot_through_autograd() -> None:
    """Split the sequence, feed the carry forward, and differentiate the whole
    two-call chain.

    The split point is not a multiple of the chunk, so the head ends on a ragged
    chunk and the tail starts mid-chunk. A split on a chunk boundary reduces to
    the same arithmetic and measures exactly zero.
    """
    whole = _tiny(seqlen=48, seed=251, with_state=False, streaming=False)
    split = _tiny(seqlen=48, seed=251, with_state=False, streaming=False)

    out = so3ssd(*whole.args(), 16)
    hu, htrans, hk, hb, hc = (t[:, :, :17].contiguous() for t in split.args())
    tu, ttrans, tk, tb, tc = (t[:, :, 17:].contiguous() for t in split.args())
    head = so3ssd(hu, htrans, hk, hb, hc, 16)
    tail = so3ssd(
        tu,
        ttrans,
        tk,
        tb,
        tc,
        16,
        z0=head.state,
        b_prev=head.b_last,
        u_prev=head.u_last,
    )
    joined = torch.cat([head.y, tail.y], dim=2)
    assert_max_rel(joined, out.y, IFACE_REL, "split y through autograd")
    assert_max_rel(tail.state, out.state, IFACE_REL, "split state through autograd")

    out.y.square().sum().backward()
    joined.square().sum().backward()
    for name, a, b in zip(GRAD_NAMES, split.args(), whole.args()):
        assert a.grad is not None and b.grad is not None, name
        assert_max_rel(a.grad, b.grad, IFACE_REL, f"split {name}")


def test_low_precision_gradients_reach_the_leaves() -> None:
    inp = _tiny(
        seqlen=40,
        seed=257,
        dtype=torch.float32,
        u_dtype=torch.bfloat16,
        bc_dtype=torch.bfloat16,
    )
    so3ssd(*inp.args(), 16, **inp.kw()).y.float().square().sum().backward()
    for name, leaf in zip(GRAD_NAMES, _leaves(inp)):
        assert leaf.grad is not None, name
        assert leaf.grad.dtype is leaf.dtype, name
        assert bool(torch.isfinite(leaf.grad).all()), name


def test_no_graph_when_nothing_requires_grad() -> None:
    inp = make_inputs(**TINY, seqlen=20, seed=263)
    out = so3ssd(*inp.args(), 16, **inp.kw())
    assert not out.y.requires_grad
    assert not out.state.requires_grad


def test_runs_on_device(device: torch.device) -> None:
    inp = _tiny(seqlen=33, seed=269, device=device)
    ref = _tiny(seqlen=33, seed=269, device=device)
    got = so3ssd(*inp.args(), 16, **inp.kw())
    want = so3ssm(*ref.args(), **ref.kw())
    got.y.square().sum().backward()
    want.y.square().sum().backward()
    for name, a, b in zip(GRAD_NAMES, _leaves(inp), _leaves(ref)):
        assert a.grad is not None and b.grad is not None, name
        assert_max_rel(a.grad, b.grad, IFACE_REL, f"interface {name} on {device.type}")


# ---------------------------------------------------------------------------
# Saved-storage discipline
# ---------------------------------------------------------------------------


def _saved_tensors(inp: ScanInputs, chunk: int) -> list[Tensor]:
    saved: list[Tensor] = []

    def pack(t: Tensor) -> Tensor:
        saved.append(t)
        return t

    def unpack(t: Tensor) -> Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = so3ssd(*inp.args(), chunk, **inp.kw())
    assert out.y.requires_grad
    return saved


def test_training_path_saves_five_tensors() -> None:
    """No initial state and no streaming carry is the training path. The budget is
    six storages per layer; this path uses five, and every one is an input."""
    inp = _tiny(seqlen=64, seed=271, with_state=False, streaming=False)
    saved = _saved_tensors(inp, 16)
    assert len(saved) == 5
    assert len(saved) <= 6
    assert {t.data_ptr() for t in saved} == {t.data_ptr() for t in inp.args()}


def test_streaming_path_saves_only_inputs() -> None:
    """Nothing derived is saved even when the carry is present: the log-scale and
    quaternion prefixes, the 3x3 table, the rotated ``B`` and ``C``, the score
    matrices, the decay mask, the increments, and the chunk-start states are all
    recomputed."""
    inp = _tiny(seqlen=64, seed=277)
    saved = _saved_tensors(inp, 16)
    assert len(saved) == 8
    assert {t.data_ptr() for t in saved} == {t.data_ptr() for t in _leaves(inp)}


def test_saved_count_is_independent_of_sequence_length() -> None:
    """A saved intermediate would scale with ``T``; an input does not."""
    counts = {
        seqlen: len(_saved_tensors(_tiny(seqlen=seqlen, seed=281), 16))
        for seqlen in (16, 64, 256)
    }
    assert set(counts.values()) == {8}


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def test_reference_backend_is_registered() -> None:
    assert "reference" in names()
    assert get("reference").name == "reference"


def test_explicit_backend_matches_automatic_selection() -> None:
    inp = make_inputs(**TINY, seqlen=33, seed=283)
    explicit = so3ssd(*inp.args(), 16, **inp.kw(), backend="reference")
    automatic = so3ssd(*inp.args(), 16, **inp.kw())
    assert torch.equal(explicit.y, automatic.y)


def test_resolve_picks_the_reference_on_cpu() -> None:
    assert resolve(None, "cpu").name == "reference"


def test_unknown_backend_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get("no-such-backend")


def test_named_backend_on_wrong_device_raises() -> None:
    with pytest.raises(ValueError, match="supports"):
        resolve("reference", "meta")


def test_no_backend_for_device_raises() -> None:
    with pytest.raises(ValueError, match="no backend supports"):
        resolve(None, "meta")


def test_duplicate_registration_raises() -> None:
    """Two implementations under one name is exactly what the registry prevents."""
    with pytest.raises(ValueError, match="already registered"):
        register(get("reference"))


def test_bad_input_raises_through_the_function() -> None:
    inp = make_inputs(**TINY, seqlen=20, seed=293)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        so3ssd(*inp.args(), 0, **inp.kw())
