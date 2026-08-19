"""Reference parity and input contract.

:func:`so3ssm` steps the recurrence and defines correctness. :func:`so3ssd_ref`
evaluates the chunked factorization the kernels implement. They must agree to
float64 rounding at every supported shape, and three closed forms that share no
code with either pin the operator's semantics independently.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor

from slinoss.ops.so3ssd import (
    quat_exp,
    quat_prefix_scan,
    rot_matrix,
    so3ssd_ref,
    so3ssm,
    tap_matrix,
)
from tests.conftest import ScanInputs, assert_max_rel, make_inputs, max_err

LANES3 = (-1, 3)

# (bsz, heads, seqlen, rows, lanes, chunk).
#
# One case per distinct path through the factorization:
#
# - 40/16, a ragged tail over three chunks at the smallest legal N and the
#   smallest legal P, which is the generic case;
# - 16/16, one exact chunk, where no chunk reads a predecessor's state;
# - one token, which is also B = 1 and H = 1;
# - 64/16, four exact chunks, with P above the minimum and H neither 1 nor 2;
# - 33/16, a ragged tail of a single token, with N above the minimum;
# - 20/32, T below L, so the one chunk is 12 slots of padding, at H = 1 and the
#   widest P;
# - 128/64, the largest chunk length, two exact chunks, B = H = 1.
SHAPES: tuple[tuple[int, int, int, int, int, int], ...] = (
    (2, 2, 40, 16, 16, 16),
    (2, 2, 16, 16, 16, 16),
    (1, 1, 1, 16, 16, 16),
    (2, 3, 64, 32, 16, 16),
    (1, 2, 33, 16, 32, 16),
    (2, 1, 20, 48, 16, 32),
    (1, 1, 128, 16, 16, 64),
)

# (with_state, streaming), the carry-in combinations other than both present.
# Each names a pair of ``None`` branches, one per implementation, and nothing
# else: an absent ``z0`` is a zero state and an absent tap pair is a zero tap.
CARRY_IN: tuple[tuple[bool, bool], ...] = ((False, True), (True, False), (False, False))

# float64 end to end. The two implementations reassociate the same sums, so the
# gap is reordering roundoff over at most 128 tokens and 96 state components.
# Worst measured over this file: 1.6e-15 on cpu, 1.1e-14 on cuda at T = 128,
# where cuBLAS reassociates a longer reduction. Run with --tolerance-report to
# see every bound next to what it actually admitted.
PARITY_REL = 1e-13

Backend = Callable[[ScanInputs], Any]
BACKENDS: tuple[Backend, ...] = (
    lambda inp: so3ssm(*inp.args(), **inp.kw()),
    lambda inp: so3ssd_ref(*inp.args(), 16, **inp.kw()),
)
BACKEND_IDS: tuple[str, ...] = ("so3ssm", "so3ssd_ref")


def _shape_id(shape: tuple[int, ...]) -> str:
    return "b{}h{}t{}p{}n{}l{}".format(*shape)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def _parity(
    shape: tuple[int, int, int, int, int, int],
    *,
    device: torch.device | str,
    with_state: bool,
    streaming: bool,
) -> None:
    """One chunked call against the sequential one. The only copy of the body."""
    bsz, heads, seqlen, rows, lanes, chunk = shape
    inp = make_inputs(
        bsz=bsz,
        heads=heads,
        seqlen=seqlen,
        rows=rows,
        lanes=lanes,
        device=device,
        with_state=with_state,
        streaming=streaming,
        seed=seqlen,
    )
    want = so3ssm(*inp.args(), **inp.kw())
    got = so3ssd_ref(*inp.args(), chunk, **inp.kw())
    assert_max_rel(got.y, want.y, PARITY_REL, "y")
    assert_max_rel(got.state, want.state, PARITY_REL, "state")
    assert torch.equal(got.b_last, want.b_last)
    assert torch.equal(got.u_last, want.u_last)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_chunked_matches_sequential(
    shape: tuple[int, int, int, int, int, int], device: torch.device
) -> None:
    """Chunk geometry against the recurrence, with every carry-in present.

    The shape is what the factorization reassociates, so it carries the full
    sweep, crossed only with the device: the two implementations reduce over
    different extents and the BLAS behind each reduction is per device. The
    carry-in flags do not touch the geometry and are swept once, below.
    """
    _parity(shape, device=device, with_state=True, streaming=True)


@pytest.mark.parametrize(("with_state", "streaming"), CARRY_IN)
def test_chunked_matches_sequential_without_a_carry_in(
    with_state: bool, streaming: bool
) -> None:
    """The zero-fill branches, at the ragged three-chunk shape.

    An omitted operand is a zero tensor built inside each implementation, so a
    wrong shape or dtype there shows up as a parity failure at any geometry.
    """
    _parity(SHAPES[0], device="cpu", with_state=with_state, streaming=streaming)


# 16 is the base the others are compared against. 32 divides T as well and halves
# the chunk count; 64 leaves a ragged tail; 128 exceeds T, so the sequence is one
# padded chunk.
@pytest.mark.parametrize("chunk", [32, 64, 128])
def test_output_does_not_depend_on_chunk_size(chunk: int) -> None:
    inp = make_inputs(seqlen=96, seed=3)
    base = so3ssd_ref(*inp.args(), 16, **inp.kw())
    got = so3ssd_ref(*inp.args(), chunk, **inp.kw())
    assert_max_rel(got.y, base.y, PARITY_REL, f"y at L={chunk}")
    assert_max_rel(got.state, base.state, PARITY_REL, f"state at L={chunk}")


# (chunk, split) at T = 48. One case per distinct boundary:
#
# - a head of one token, so the tail starts from a state formed by a single step;
# - a split one token past a chunk boundary, where the head's last chunk is
#   ragged and its padding must not reach the carried state;
# - a split on a chunk boundary, the only case with no padding on either side;
# - a tail of one token, at a chunk length that leaves the whole-sequence call
#   ragged as well.
@pytest.mark.parametrize(
    ("chunk", "split"), [(16, 1), (16, 17), (16, 32), (32, 47)], ids=str
)
def test_streaming_split_matches_the_whole_sequence(chunk: int, split: int) -> None:
    """A split carries ``state``, ``b_last``, and ``u_last`` forward. Nothing
    else crosses the boundary."""
    seqlen = 48
    inp = make_inputs(seqlen=seqlen, seed=5, streaming=False)
    whole = so3ssd_ref(*inp.args(), chunk, **inp.kw())

    def cut(lo: int, hi: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        u, trans, k, b, c = (t[:, :, lo:hi].contiguous() for t in inp.args())
        return (u, trans, k, b, c)

    head = so3ssd_ref(*cut(0, split), chunk, z0=inp.z0)
    tail = so3ssd_ref(
        *cut(split, seqlen),
        chunk,
        z0=head.state,
        b_prev=head.b_last,
        u_prev=head.u_last,
    )
    joined = torch.cat([head.y, tail.y], dim=2)
    assert_max_rel(joined, whole.y, PARITY_REL, f"split y at {split}")
    assert_max_rel(tail.state, whole.state, PARITY_REL, f"split state at {split}")


def test_omitted_state_equals_explicit_zeros() -> None:
    inp = make_inputs(seqlen=24, seed=7, with_state=False, streaming=False)
    bsz, heads, _, rows = inp.U.shape
    kwargs: dict[str, Any] = {"dtype": inp.trans.dtype, "device": inp.U.device}
    zeros = inp._replace(
        z0=torch.zeros(bsz, heads, rows, inp.B.shape[-1], **kwargs),
        b_prev=torch.zeros(bsz, heads, inp.B.shape[-1], **kwargs),
        u_prev=torch.zeros(bsz, heads, rows, **kwargs),
    )
    for chunk in (16, 32):
        bare = so3ssd_ref(*inp.args(), chunk, **inp.kw())
        explicit = so3ssd_ref(*zeros.args(), chunk, **zeros.kw())
        assert max_err(explicit.y, bare.y) == 0.0
        assert max_err(explicit.state, bare.state) == 0.0


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_last_token_channels_are_the_last_token(backend: Backend) -> None:
    """The streaming channels feed straight back in, and the operator repacks
    nothing, so they must come out contiguous."""
    inp = make_inputs(seqlen=13, seed=11)
    out = backend(inp)
    assert torch.equal(out.b_last, inp.B[:, :, -1])
    assert torch.equal(out.u_last, inp.U[:, :, -1])
    assert out.b_last.is_contiguous()
    assert out.u_last.is_contiguous()
    assert out.state.is_contiguous()
    assert out.y.is_contiguous()


def test_result_shapes_and_dtypes() -> None:
    inp = make_inputs(
        bsz=2,
        heads=3,
        seqlen=20,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        u_dtype=torch.bfloat16,
        bc_dtype=torch.bfloat16,
        seed=13,
    )
    outs = (so3ssm(*inp.args(), **inp.kw()), so3ssd_ref(*inp.args(), 16, **inp.kw()))
    for out in outs:
        assert out.y.shape == (2, 3, 20, 16)
        assert out.y.dtype is torch.bfloat16
        assert out.state.shape == (2, 3, 16, 48)
        assert out.state.dtype is torch.float32
        assert out.b_last.shape == (2, 3, 48)
        assert out.u_last.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# Independent closed forms. Neither shares code with the implementations.
# ---------------------------------------------------------------------------


def test_decay_only_matches_the_global_prefix_form() -> None:
    """With both taps zero the state is a pure homogeneous transport of ``z0``,
    so the output follows from the global prefixes with no chunking at all."""
    inp = make_inputs(seqlen=37, rows=16, lanes=16, seed=17, streaming=False)
    assert inp.z0 is not None
    zero_k = torch.zeros_like(inp.K)

    lprefix = torch.cumsum(inp.trans[..., 3], dim=-1)
    rot = rot_matrix(quat_prefix_scan(quat_exp(inp.trans[..., :3])))
    crot = torch.einsum("bhtji,bhtnj->bhtni", rot, inp.C.unflatten(-1, LANES3))
    want = torch.exp(2.0 * lprefix)[..., None] * torch.einsum(
        "bhtni,bhpni->bhtp", crot, inp.z0.unflatten(-1, LANES3)
    )

    for out in (
        so3ssm(inp.U, inp.trans, zero_k, inp.B, inp.C, z0=inp.z0),
        so3ssd_ref(inp.U, inp.trans, zero_k, inp.B, inp.C, 16, z0=inp.z0),
    ):
        assert_max_rel(out.y, want, PARITY_REL, "decay-only y")


def test_zero_rotation_reduces_to_a_diagonal_state_space_model() -> None:
    """With ``w = 0`` every rotation is the identity and both taps collapse to
    ``kr``, so the operator is a scalar-decay SSM written out here in full."""
    seqlen = 24
    inp = make_inputs(seqlen=seqlen, rows=16, lanes=16, w_scale=0.0, seed=19)
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    assert float(inp.trans[..., :3].abs().max()) == 0.0

    decay = torch.exp(2.0 * inp.trans[..., 3])
    kr_prev, kr_curr = inp.K[..., 0, 0], inp.K[..., 1, 0]
    bshift = torch.cat([inp.b_prev[:, :, None], inp.B[:, :, :-1]], dim=2)
    ushift = torch.cat([inp.u_prev[:, :, None], inp.U[:, :, :-1]], dim=2)

    state = inp.z0
    steps = []
    for t in range(seqlen):
        forced_now = (kr_curr[:, :, t, None] * inp.B[:, :, t])[:, :, None, :]
        forced_prev = (kr_prev[:, :, t, None] * bshift[:, :, t])[:, :, None, :]
        state = (
            decay[:, :, t, None, None] * state
            + inp.U[:, :, t, :, None] * forced_now
            + ushift[:, :, t, :, None] * forced_prev
        )
        steps.append(torch.einsum("bhd,bhpd->bhp", inp.C[:, :, t], state))
    want = torch.stack(steps, dim=2)

    for out in (
        so3ssm(*inp.args(), **inp.kw()),
        so3ssd_ref(*inp.args(), 16, **inp.kw()),
    ):
        assert_max_rel(out.y, want, PARITY_REL, "diagonal y")
        assert_max_rel(out.state, state, PARITY_REL, "diagonal state")


def test_single_token_is_one_forced_step() -> None:
    inp = make_inputs(seqlen=1, rows=16, lanes=16, seed=23)
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    w = inp.trans[..., :3]
    rot = rot_matrix(quat_exp(w))[:, :, 0]
    scale = torch.exp(2.0 * inp.trans[..., 0, 3])
    kprev = tap_matrix(inp.K[..., 0, :3], w)[:, :, 0]
    kcurr = tap_matrix(inp.K[..., 1, :3], w)[:, :, 0]

    def tapped(matrix: Tensor, vec: Tensor) -> Tensor:
        return torch.einsum("bhij,bhnj->bhni", matrix, vec.unflatten(-1, LANES3))

    state = scale[..., None, None, None] * torch.einsum(
        "bhij,bhpnj->bhpni", rot, inp.z0.unflatten(-1, LANES3)
    )
    state = (
        state
        + inp.u_prev[..., None, None] * tapped(kprev, inp.b_prev)[..., None, :, :]
        + inp.U[:, :, 0][..., None, None]
        * tapped(kcurr, inp.B[:, :, 0])[..., None, :, :]
    )
    want = torch.einsum("bhni,bhpni->bhp", inp.C[:, :, 0].unflatten(-1, LANES3), state)[
        :, :, None, :
    ]

    for out in (
        so3ssm(*inp.args(), **inp.kw()),
        so3ssd_ref(*inp.args(), 16, **inp.kw()),
    ):
        assert_max_rel(out.y, want, PARITY_REL, "one-step y")
        assert_max_rel(out.state, state.flatten(-2, -1), PARITY_REL, "one-step state")


# ---------------------------------------------------------------------------
# Input contract. Every raise on a public path is triggered.
# ---------------------------------------------------------------------------


def _noncontig(t: Tensor) -> Tensor:
    """A view with the right shape and dtype and the wrong strides."""
    width = int(t.shape[-1])
    wide = torch.zeros(*t.shape[:-1], 2 * width, dtype=t.dtype, device=t.device)
    wide[..., :width] = t
    out = wide[..., :width]
    assert not out.is_contiguous()
    return out


def _base(**overrides: Any) -> ScanInputs:
    defaults: dict[str, Any] = {
        "bsz": 2,
        "heads": 2,
        "seqlen": 8,
        "rows": 16,
        "lanes": 16,
        "seed": 29,
    }
    return make_inputs(**{**defaults, **overrides})


def _state(inp: ScanInputs) -> Tensor:
    assert inp.z0 is not None
    return inp.z0


def _bprev(inp: ScanInputs) -> Tensor:
    assert inp.b_prev is not None
    return inp.b_prev


def _uprev(inp: ScanInputs) -> Tensor:
    assert inp.u_prev is not None
    return inp.u_prev


def _narrow_state(inp: ScanInputs, width: int) -> ScanInputs:
    return inp._replace(
        B=inp.B[..., :width].contiguous(),
        C=inp.C[..., :width].contiguous(),
        z0=_state(inp)[..., :width].contiguous(),
        b_prev=_bprev(inp)[..., :width].contiguous(),
    )


Mutate = Callable[[ScanInputs], ScanInputs]
Case = tuple[str, dict[str, Any], Mutate, type[Exception], str]

BAD_INPUTS: tuple[Case, ...] = (
    (
        "u_rank",
        {},
        lambda i: i._replace(U=i.U[:, :, :, 0]),
        ValueError,
        r"U must be \(B,H,T,P\)",
    ),
    (
        "b_rank",
        {},
        lambda i: i._replace(B=i.B[..., 0]),
        ValueError,
        r"B must be \(B,G,T,3N\)",
    ),
    ("empty_time", {"seqlen": 0}, lambda i: i, ValueError, "T must be at least 1"),
    (
        "trans_shape",
        {},
        lambda i: i._replace(trans=i.trans[..., :3].contiguous()),
        ValueError,
        "trans must have shape",
    ),
    (
        "k_shape",
        {},
        lambda i: i._replace(K=i.K[..., :1, :].contiguous()),
        ValueError,
        "K must have shape",
    ),
    (
        "b_time",
        {},
        lambda i: i._replace(B=i.B[:, :, :-1].contiguous()),
        ValueError,
        "B must have shape",
    ),
    (
        "c_time",
        {},
        lambda i: i._replace(C=i.C[:, :, :-1].contiguous()),
        ValueError,
        "C must have shape",
    ),
    (
        "z_shape",
        {},
        lambda i: i._replace(z0=_state(i)[..., :-3].contiguous()),
        ValueError,
        "z must have shape",
    ),
    (
        "b_prev_shape",
        {},
        lambda i: i._replace(b_prev=_bprev(i)[..., :-3].contiguous()),
        ValueError,
        "b_prev must have shape",
    ),
    (
        "u_prev_shape",
        {},
        lambda i: i._replace(u_prev=_uprev(i)[..., :-1].contiguous()),
        ValueError,
        "u_prev must have shape",
    ),
    (
        "lanes_not_multiple",
        {"lanes": 8},
        lambda i: i,
        ValueError,
        "3N must be 3 times a multiple of 16",
    ),
    (
        "state_not_triple",
        {"lanes": 32},
        lambda i: _narrow_state(i, 50),
        ValueError,
        "3N must be 3 times a multiple of 16",
    ),
    (
        "rows_not_multiple",
        {"rows": 12},
        lambda i: i,
        ValueError,
        "P must be a multiple of 16",
    ),
    (
        "unpaired_b",
        {},
        lambda i: i._replace(u_prev=None),
        ValueError,
        "passed together",
    ),
    (
        "unpaired_u",
        {},
        lambda i: i._replace(b_prev=None),
        ValueError,
        "passed together",
    ),
    (
        "u_strided",
        {},
        lambda i: i._replace(U=_noncontig(i.U)),
        ValueError,
        "U must be contiguous",
    ),
    (
        "trans_strided",
        {},
        lambda i: i._replace(trans=_noncontig(i.trans)),
        ValueError,
        "trans must be contiguous",
    ),
    (
        "k_strided",
        {},
        lambda i: i._replace(K=_noncontig(i.K)),
        ValueError,
        "K must be contiguous",
    ),
    (
        "c_strided",
        {},
        lambda i: i._replace(C=_noncontig(i.C)),
        ValueError,
        "C must be contiguous",
    ),
    (
        "z_strided",
        {},
        lambda i: i._replace(z0=_noncontig(_state(i))),
        ValueError,
        "z must be contiguous",
    ),
    (
        "u_dtype",
        {},
        lambda i: i._replace(U=i.U.to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "b_dtype",
        {},
        lambda i: i._replace(B=i.B.to(torch.int32)),
        TypeError,
        "supported",
    ),
    (
        "c_dtype",
        {},
        lambda i: i._replace(C=i.C.to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "b_prev_dtype",
        {},
        lambda i: i._replace(b_prev=_bprev(i).to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "u_prev_dtype",
        {},
        lambda i: i._replace(u_prev=_uprev(i).to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "trans_low",
        {},
        lambda i: i._replace(trans=i.trans.to(torch.bfloat16)),
        TypeError,
        "float32-pinned",
    ),
    (
        "k_low",
        {},
        lambda i: i._replace(K=i.K.to(torch.float16)),
        TypeError,
        "float32-pinned",
    ),
    (
        "z_low",
        {},
        lambda i: i._replace(z0=_state(i).to(torch.bfloat16)),
        TypeError,
        "float32-pinned",
    ),
)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize(
    ("base_kwargs", "mutate", "exc", "match"),
    [case[1:] for case in BAD_INPUTS],
    ids=[case[0] for case in BAD_INPUTS],
)
def test_rejects_bad_inputs(
    backend: Backend,
    base_kwargs: dict[str, Any],
    mutate: Mutate,
    exc: type[Exception],
    match: str,
) -> None:
    """Every message, on both entry points.

    The two share one checker today, so the backend axis is a duplicate of the
    table; it stays because each arm is a separate public function and a refusal
    that a rewrite moved behind a repack is what this table exists to catch.
    """
    inp = mutate(_base(**base_kwargs))
    with pytest.raises(exc, match=match):
        backend(inp)


def test_rejects_non_positive_chunk_size() -> None:
    # One comparison against 1 refuses zero and every negative alike.
    inp = _base()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        so3ssd_ref(*inp.args(), 0, **inp.kw())


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_rejects_mixed_devices(backend: Backend) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    inp = _base(device="cuda")
    with pytest.raises(ValueError, match="one device only"):
        backend(inp._replace(C=inp.C.cpu()))
