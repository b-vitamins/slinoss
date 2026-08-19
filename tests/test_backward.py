"""The analytic backward and its adjoint building blocks.

Ground truth is float64 autograd, never a second hand derivation. Each adjoint
helper is checked against autograd through its own primal, and the assembled
backward is checked against autograd through :func:`so3ssm`, which is the
sequential definition of the operator rather than the chunked factorization the
backward is derived from.

Every output is exercised on its own as well as together: a cotangent that only
ever appears inside a sum with three others cannot be attributed.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from slinoss.ops.so3ssd import (
    ChunkedBackward,
    as_lanes,
    chunk_pad,
    chunked_backward,
    chunked_forward,
    quat_exp,
    quat_exp_vjp,
    quat_prefix_scan,
    quat_prefix_scan_vjp,
    rot_matrix,
    rot_matrix_vjp,
    so3ssd_bwd_ref,
    so3ssm,
    tap_matrix,
    tap_matrix_vjp,
)
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

TINY: dict[str, Any] = {"bsz": 1, "heads": 1, "rows": 16, "lanes": 16}

# float64 analytic VJP against float64 autograd. The gap is reordering roundoff.
# Worst measured over this file: 7.6e-15.
BWD_REL = 1e-13

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

Cotangents = tuple[Tensor, Tensor, Tensor, Tensor]
Optionals = tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]


# ---------------------------------------------------------------------------
# Adjoint helpers
# ---------------------------------------------------------------------------


def _rand(*shape: int, seed: int, scale: float = 1.0) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return scale * torch.randn(*shape, generator=gen, dtype=torch.float64)


@pytest.mark.parametrize("scale", [0.0, 1e-8, 1.0, 3.0])
def test_quat_exp_vjp_matches_autograd(scale: float) -> None:
    """Includes ``w = 0`` exactly, where the axis normal form has no derivative."""
    w = _rand(2, 3, 5, 3, seed=3, scale=scale).requires_grad_(True)
    q = quat_exp(w)
    dq = _rand(2, 3, 5, 4, seed=5)
    (want,) = torch.autograd.grad(q, w, dq)
    got = quat_exp_vjp(dq, w.detach())
    assert_max_rel(got, want, 1e-14, f"quat_exp_vjp at |w| ~ {scale}")


@pytest.mark.parametrize("length", [1, 2, 7, 16])
def test_quat_prefix_scan_vjp_matches_autograd(length: int) -> None:
    """The closed form replaces a sequential quaternion recurrence with a reverse
    cumulative sum, so it is checked at a power of two and away from one.

    The gap against autograd includes the O(eps) difference between the
    renormalized prefix the closed form uses and the raw product autograd
    differentiates, which is why this bound is looser than the other three.
    """
    w = _rand(2, 2, length, 3, seed=7 + length, scale=1.3)
    q = quat_exp(w).requires_grad_(True)
    prefix = quat_prefix_scan(q)
    dprefix = _rand(2, 2, length, 4, seed=11)
    (want,) = torch.autograd.grad(prefix, q, dprefix)
    got = quat_prefix_scan_vjp(dprefix, prefix.detach())
    assert_max_rel(got, want, 1e-12, f"quat_prefix_scan_vjp at L={length}")


def test_rot_matrix_vjp_matches_autograd() -> None:
    q = quat_exp(_rand(2, 3, 5, 3, seed=13, scale=1.7)).requires_grad_(True)
    rot = rot_matrix(q)
    drot = _rand(2, 3, 5, 3, 3, seed=17)
    (want,) = torch.autograd.grad(rot, q, drot)
    got = rot_matrix_vjp(drot, q.detach())
    assert_max_rel(got, want, 1e-14, "rot_matrix_vjp")


@pytest.mark.parametrize("scale", [0.0, 1.0, 3.0])
def test_tap_matrix_vjp_matches_autograd(scale: float) -> None:
    tap = _rand(2, 3, 5, 3, seed=19).requires_grad_(True)
    w = _rand(2, 3, 5, 3, seed=23, scale=scale).requires_grad_(True)
    kmat = tap_matrix(tap, w)
    dk = _rand(2, 3, 5, 3, 3, seed=29)
    want_tap, want_w = torch.autograd.grad(kmat, (tap, w), dk)
    got = tap_matrix_vjp(dk, tap.detach(), w.detach())
    assert_max_rel(got.tap, want_tap, 1e-14, f"tap_matrix_vjp tap at |w| ~ {scale}")
    assert_max_rel(got.w, want_w, 1e-14, f"tap_matrix_vjp w at |w| ~ {scale}")


# ---------------------------------------------------------------------------
# The assembled backward
# ---------------------------------------------------------------------------


def _pair(**overrides: Any) -> tuple[ScanInputs, ScanInputs]:
    """The same inputs twice: differentiable leaves, and detached operands."""
    kwargs = {**TINY, **overrides}
    return (
        make_inputs(requires_grad=True, **kwargs),
        make_inputs(requires_grad=False, **kwargs),
    )


def _cotangents(inp: ScanInputs, seed: int) -> Cotangents:
    bsz, heads, seqlen, rows = (int(d) for d in inp.U.shape)
    state_dim = int(inp.B.shape[-1])
    gen = torch.Generator(device=inp.U.device).manual_seed(seed)

    def like(*shape: int) -> Tensor:
        return torch.randn(
            *shape, generator=gen, dtype=inp.trans.dtype, device=inp.U.device
        )

    return (
        like(bsz, heads, seqlen, rows),
        like(bsz, heads, rows, state_dim),
        like(bsz, heads, state_dim),
        like(bsz, heads, rows),
    )


def _oracle(inp: ScanInputs, cotangents: Cotangents) -> tuple[Tensor | None, ...]:
    """Autograd through the sequential reference, in gradient-name order."""
    leaves = (*inp.args(), inp.z0, inp.b_prev, inp.u_prev)
    present = tuple(t for t in leaves if t is not None)
    out = so3ssm(*inp.args(), **inp.kw())
    grads = iter(
        torch.autograd.grad(
            (out.y, out.state, out.b_last, out.u_last), present, cotangents
        )
    )
    return tuple(None if leaf is None else next(grads) for leaf in leaves)


def _analytic(inp: ScanInputs, cot: Optionals, chunk: int) -> tuple[Tensor | None, ...]:
    return tuple(
        so3ssd_bwd_ref(cot[0], cot[1], cot[2], cot[3], *inp.args(), chunk, **inp.kw())
    )


def _compare(
    got: tuple[Tensor | None, ...],
    want: tuple[Tensor | None, ...],
    label: str,
) -> None:
    for name, a, b in zip(GRAD_NAMES, got, want):
        if b is None:
            assert a is None, f"{name} {label}: expected no gradient"
            continue
        assert a is not None, f"{name} {label}: missing gradient"
        assert_max_rel(a, b, BWD_REL, f"{name} {label}")


@pytest.mark.parametrize(
    ("seqlen", "chunk"),
    [(1, 16), (8, 16), (16, 16), (17, 16), (33, 16), (40, 32), (64, 16), (48, 64)],
)
def test_analytic_matches_autograd_oracle(seqlen: int, chunk: int) -> None:
    """Shape sweep: ragged tail, single chunk, three or more chunks, and a chunk
    longer than the sequence."""
    ref, fast = _pair(seqlen=seqlen, seed=71 + seqlen)
    cotangents = _cotangents(ref, 73)
    _compare(
        _analytic(fast, cotangents, chunk),
        _oracle(ref, cotangents),
        f"at T={seqlen} L={chunk}",
    )


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_analytic_isolates_each_output(index: int) -> None:
    """One nonzero output cotangent at a time."""
    ref, fast = _pair(seqlen=40, seed=79)
    full = _cotangents(ref, 83)
    isolated = tuple(
        c if i == index else torch.zeros_like(c) for i, c in enumerate(full)
    )
    picked: Cotangents = (isolated[0], isolated[1], isolated[2], isolated[3])
    _compare(_analytic(fast, picked, 16), _oracle(ref, picked), f"from output {index}")


def test_none_cotangent_equals_zero_cotangent() -> None:
    """``None`` is how autograd spells an unused output. It must be bit-identical
    to an explicit zero, not merely close."""
    _, fast = _pair(seqlen=33, seed=89)
    cot = _cotangents(fast, 97)
    zeros: Cotangents = (
        cot[0],
        torch.zeros_like(cot[1]),
        torch.zeros_like(cot[2]),
        cot[3],
    )
    got = _analytic(fast, (cot[0], None, None, cot[3]), 16)
    want = _analytic(fast, zeros, 16)
    for name, a, b in zip(GRAD_NAMES, got, want):
        assert a is not None and b is not None, name
        assert torch.equal(a, b), name


def test_all_cotangents_none_gives_zero_gradients() -> None:
    _, fast = _pair(seqlen=20, seed=101)
    for name, grad in zip(GRAD_NAMES, _analytic(fast, (None, None, None, None), 16)):
        assert grad is not None, name
        assert not bool(grad.any()), name


@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_gradients_are_chunk_size_independent(chunk: int) -> None:
    """The factorization is an identity, so the chunk length is a scheduling
    choice and never a numerical one."""
    _, fast = _pair(seqlen=48, seed=103)
    cotangents = _cotangents(fast, 107)
    _compare(
        _analytic(fast, cotangents, chunk),
        _analytic(fast, cotangents, 48),
        f"chunk {chunk} vs 48",
    )


def test_wider_lane_count() -> None:
    """A lane-indexed reduction correct at one N is not correct at another."""
    ref, fast = _pair(seqlen=33, heads=2, rows=16, lanes=32, seed=109)
    cotangents = _cotangents(ref, 113)
    _compare(_analytic(fast, cotangents, 16), _oracle(ref, cotangents), "at N=32")


def test_without_state_or_streaming() -> None:
    ref, fast = _pair(seqlen=17, seed=127, with_state=False, streaming=False)
    cotangents = _cotangents(ref, 131)
    _compare(_analytic(fast, cotangents, 16), _oracle(ref, cotangents), "no carry")


def test_state_only_streaming_split() -> None:
    """``z0`` without ``b_prev`` and ``u_prev`` is the decode carry."""
    ref, fast = _pair(seqlen=9, seed=137, streaming=False)
    cotangents = _cotangents(ref, 139)
    _compare(_analytic(fast, cotangents, 16), _oracle(ref, cotangents), "state only")


def test_dk_lane_three_is_exactly_zero() -> None:
    """Lane 3 of ``K`` exists for float4 alignment and is never read."""
    _, fast = _pair(seqlen=20, seed=149)
    cot = _cotangents(fast, 151)
    grads = so3ssd_bwd_ref(*cot, *fast.args(), 16, **fast.kw())
    assert not bool(grads.dK[..., 3].any())


def test_every_gradient_is_contiguous_and_matches_its_input() -> None:
    _, fast = _pair(seqlen=33, seed=157)
    cot = _cotangents(fast, 163)
    grads = so3ssd_bwd_ref(*cot, *fast.args(), 16, **fast.kw())
    leaves = (*fast.args(), fast.z0, fast.b_prev, fast.u_prev)
    for name, grad, leaf in zip(GRAD_NAMES, grads, leaves):
        assert grad is not None and leaf is not None, name
        assert grad.is_contiguous(), name
        assert grad.dtype is leaf.dtype, name
        assert grad.shape == leaf.shape, name


def test_low_precision_gradients_keep_input_dtypes() -> None:
    """Arithmetic runs in the pinned dtype; only the boundary is low precision."""
    _, fast = _pair(
        seqlen=40,
        seed=167,
        dtype=torch.float32,
        u_dtype=torch.bfloat16,
        bc_dtype=torch.bfloat16,
    )
    dy = torch.randn_like(fast.U)
    grads = so3ssd_bwd_ref(dy, None, None, None, *fast.args(), 16, **fast.kw())
    assert grads.dU.dtype is torch.bfloat16
    assert grads.dB.dtype is torch.bfloat16
    assert grads.dC.dtype is torch.bfloat16
    assert grads.dtrans.dtype is torch.float32
    assert grads.dK.dtype is torch.float32
    for name, grad in zip(GRAD_NAMES, grads):
        assert grad is not None and bool(torch.isfinite(grad).all()), name


def test_gradients_are_finite_under_saturated_decay() -> None:
    _, fast = _pair(seqlen=48, seed=173, ls_bias=25.0)
    cot = _cotangents(fast, 179)
    grads = so3ssd_bwd_ref(*cot, *fast.args(), 16, **fast.kw())
    for name, grad in zip(GRAD_NAMES, grads):
        assert grad is not None and bool(torch.isfinite(grad).all()), name


def test_matches_oracle_on_device(device: torch.device) -> None:
    """The reference backward runs on both devices and agrees on both."""
    ref, fast = _pair(seqlen=40, seed=181, device=device)
    cotangents = _cotangents(ref, 191)
    _compare(
        _analytic(fast, cotangents, 16),
        _oracle(ref, cotangents),
        f"analytic on {device.type}",
    )


# ---------------------------------------------------------------------------
# The named intermediates
# ---------------------------------------------------------------------------
#
# These are the quantities the backward's stages hand to each other, and each is
# the standard a separate kernel is held to. So each is checked against autograd
# through the primal that defines it, not against the assembled gradients: a term
# that is only ever seen inside a correct total can be wrong in a way that another
# term cancels, and the kernel that produces it alone would then have no standard
# at all.


def _parts(inp: ScanInputs, cot: Optionals, chunk: int) -> ChunkedBackward:
    return chunked_backward(*cot, *inp.args(), chunk, **inp.kw())


def _rebuild_starts(
    rot: Tensor, scale: Tensor, inc: Tensor, first: Tensor
) -> tuple[Tensor, Tensor]:
    """The inter-chunk recurrence, forward, as ``chunked_forward`` runs it.

    Args:
        rot: Each chunk's closing rotation, ``(B,H,C,3,3)``.
        scale: Each chunk's closing scale, ``(B,H,C)``.
        inc: Each chunk's increment in the outer frame, ``(B,H,C,P,N,3)``.
        first: The state entering chunk 0, ``(B,H,P,N,3)``.

    Returns:
        The per-chunk start states and the state after the last chunk.
    """
    starts = []
    state = first
    for c in range(int(inc.shape[2])):
        starts.append(state)
        state = (
            scale[:, :, c, None, None, None]
            * torch.einsum("bhij,bhpnj->bhpni", rot[:, :, c], state)
            + inc[:, :, c]
        )
    return torch.stack(starts, dim=2), state


@pytest.mark.parametrize(("seqlen", "chunk"), [(16, 16), (48, 16), (40, 16)])
def test_reverse_recurrence_cotangents_match_autograd(seqlen: int, chunk: int) -> None:
    """The four outputs of the reverse chunk recurrence, at one chunk, three, and
    a ragged tail.

    Autograd runs through the forward recurrence rebuilt from the forward's own
    operands, so the standard is the primal and not a second derivation of its
    adjoint. The rebuild is asserted to reproduce ``zstart`` and ``state`` before
    it is differentiated; otherwise an error in the rebuild would quietly redefine
    what the cotangents are compared against.

    ``dchunk_scale`` is a partial: the closing scale also reaches the output
    through the increment weight, and that path is part of ``dlogp`` instead. The
    rebuilt recurrence takes the scale as an independent operand, which is what
    isolates the one share this field carries.
    """
    ref, fast = _pair(seqlen=seqlen, seed=193 + seqlen)
    cot = _cotangents(ref, 197)
    parts = _parts(fast, cot, chunk)
    fw = chunked_forward(*fast.args(), chunk, **fast.kw())

    # The closing rotation and scale of each chunk, as the recurrence reads them.
    rot = fw.table.rot[..., -1, :, :].clone().requires_grad_(True)
    scale = torch.exp(2.0 * fw.lprefix[..., -1]).clone().requires_grad_(True)
    local = as_lanes(fw.inc_local).clone().requires_grad_(True)
    first = fw.zstart[:, :, 0].clone().requires_grad_(True)
    # inc is built inside the graph, not taken as a leaf: the closing rotation
    # reaches the output through the frame change as well as the recurrence, and
    # dchunk_rot carries both shares.
    inc = torch.einsum("bhcij,bhcpnj->bhcpni", rot, local)
    starts, state = _rebuild_starts(rot, scale, inc, first)
    assert_max_rel(starts, fw.zstart, 1e-15, "rebuilt zstart")
    assert_max_rel(state, fw.state, 1e-15, "rebuilt state")

    loss = (as_lanes(parts.dzstart) * starts).sum() + (as_lanes(cot[1]) * state).sum()
    g_rot, g_scale, g_inc, g_first = torch.autograd.grad(loss, (rot, scale, inc, first))
    assert_max_rel(parts.dinc, g_inc.flatten(-2, -1), BWD_REL, "dinc")
    assert_max_rel(parts.dz0, g_first.flatten(-2, -1), BWD_REL, "dz0")
    assert_max_rel(parts.dchunk_rot, g_rot, BWD_REL, "dchunk_rot")
    assert_max_rel(parts.dchunk_scale, g_scale, BWD_REL, "dchunk_scale")


def test_dzstart_carries_the_offset_term_only() -> None:
    """``dzstart`` is what the readout contributes to each chunk's start state.

    The recurrence's share arrives afterwards, so differentiating the whole output
    would accept the total here and leave the reverse scan with nothing to add.
    """
    ref, fast = _pair(seqlen=40, seed=199)
    cot = _cotangents(ref, 211)
    parts = _parts(fast, cot, 16)
    fw = chunked_forward(*ref.args(), 16, **ref.kw())
    dy_c = chunk_pad(cot[0], fw.length)
    (want,) = torch.autograd.grad((dy_c * fw.y_off).sum(), fw.zstart)
    assert_max_rel(parts.dzstart, want.flatten(-2, -1), BWD_REL, "dzstart")


def test_dlogp_matches_autograd_and_neither_half_is_empty() -> None:
    """The log-scale-prefix cotangent, and the split that two stages produce.

    The total is checked against autograd. The two halves sum to it by
    construction, so what needs checking is that each one is doing work: a split
    where one side is identically zero is a mislabelled stage, not a free one.
    """
    ref, fast = _pair(seqlen=40, seed=229)
    cot = _cotangents(ref, 233)
    parts = _parts(fast, cot, 16)
    fw = chunked_forward(*ref.args(), 16, **ref.kw())
    # b_last and u_last are slices of the inputs and reach no prefix, so the two
    # terms here are the whole dependence.
    loss = (cot[0] * fw.y).sum() + (as_lanes(cot[1]) * fw.state).sum()
    (want,) = torch.autograd.grad(loss, fw.lprefix)
    assert_max_rel(parts.dlogp, want, BWD_REL, "dlogp")
    assert bool(parts.dlogp_scan.any()), "the diagonal and increment half is zero"
    assert bool(parts.dlogp_off.any()), "the offset and transition half is zero"


@pytest.mark.parametrize("groups", [1, 2])
def test_chunk_zero_carry_is_the_streaming_feedback(groups: int) -> None:
    """``carry_u[0]`` and ``carry_b[0]`` are ``du_prev`` and ``db_prev``.

    The boundary stage reads one buffer for both jobs, scattering rows 1 onward
    onto chunk boundaries and handing row 0 back as the streaming gradient, so the
    two must be the same values and not merely close. Both group counts run
    because ``carry_b`` is summed over each group's heads and ``carry_u`` is not.
    """
    _, fast = _pair(seqlen=40, heads=2, groups=groups, seed=239)
    dy = torch.randn_like(fast.U)
    parts = _parts(fast, (dy, None, None, None), 16)
    assert parts.grads.du_prev is not None and parts.grads.db_prev is not None
    assert bool(parts.carry_u.any()) and bool(parts.carry_b.any())
    assert torch.equal(parts.carry_u[:, :, 0], parts.grads.du_prev)
    assert torch.equal(parts.carry_b[:, :, 0], parts.grads.db_prev)
