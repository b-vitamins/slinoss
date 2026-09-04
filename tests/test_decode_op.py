"""One-token boundary: parity, carry order, in-place semantics, input contract.

:func:`slinoss.ops.so3ssd.so3ssm` defines correctness and
:func:`slinoss.ops.so3ssd.so3ssd` is the path the mixer trains and prefills
through. The decode boundary is a third implementation of the same map at ``T = 1``,
so agreement with both is the gate: agreement with one admits a derivation error the
two share, and the map has four places where a single wrong line is a silent wrong
answer rather than an exception -- the readout taken before the forcing, the
rotation applied to the taps instead of to the incoming state, a carry written
before it is read, and a readout contracted per 3-vector rather than over all
``3N``.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple, TypedDict

import pytest
import torch
from torch import Tensor

from slinoss import mixer as mixer_module
from slinoss._precision import PINNED_TENSORS, pinned_dtype
from slinoss.config import SLinOSSConfig
from slinoss.graph import GraphedStep, capture
from slinoss.mixer import SLinOSSMixer
from slinoss.ops.decode import (
    TOKENS,
    DecodeResult,
    decode_no_backward,
    decode_ref,
    decode_step,
    names,
    resolve,
)
from slinoss.ops.decode.backends import CUTE, REFERENCE
from slinoss.ops.so3ssd import backends as scan_dispatch
from slinoss.ops.so3ssd import so3ssd, so3ssm
from slinoss.state import MixerState, oscillator_basis
from tests.conftest import (
    ScanInputs,
    assert_max_rel,
    make_inputs,
    max_err,
    projection_band,
)

CHUNK = 16
"""Chunk length the ``T``-token oracle runs at. The smallest legal one.

At ``T = 1`` every chunk length pads the same single slot, so the value reaches the
comparison only through the padded arithmetic the chunked path does either way.
"""

PARITY_REL = 1e-13
"""Bound on the boundary against both ``T``-token implementations, at float64.

float64 end to end. The three evaluate the same map with different reassociations:
one step against one padded chunk against a token loop. The tree's float64 parity
bound, unchanged: ``tests/test_reference.py``, ``tests/test_interface.py`` and
``tests/test_adversarial.py`` hold the same value. Measured worst over this file:
9.589e-16, on the state against the chunked path, 1.0% of the bound. Against the
token loop every figure here is 0.0, bitwise. Run with ``--tolerance-report`` to see
every bound next to what it admitted.
"""

# (bsz, heads, groups, rows, lanes). One case per distinct path through the
# grouping and the shape multiples:
#
# - 2/2/2, G = H, the smallest legal N and the smallest legal P, the generic case;
# - 1/1/1, one batch, one head, one group;
# - 4 heads over 1 group, the broadcast every head shares, at P above the minimum;
# - 4 heads over 2 groups, an intermediate G, at 3N above the minimum;
# - 2 heads at the widest P this file runs.
SHAPES: tuple[tuple[int, int, int, int, int], ...] = (
    (2, 2, 2, 16, 16),
    (1, 1, 1, 16, 16),
    (2, 4, 1, 32, 16),
    (2, 4, 2, 16, 32),
    (1, 2, 2, 48, 16),
)

# (label, w_scale, ls_bias). The transition regimes, swept rather than crossed:
# every case moves one of the two chart coordinates away from the middle of its
# range. ``w_scale`` reaches ``|w| -> 2*w_max``, six radians, so the rotation runs
# from the identity past a half turn; ``ls_bias`` reaches both ends of
# ``-LS_MAX_MAG*sigmoid``, the shortest and the longest horizon.
REGIMES: tuple[tuple[str, float, float], ...] = (
    ("no-drive", 0.0, 0.0),
    ("near-identity", 1e-3, 0.0),
    ("max-drive", 100.0, 0.0),
    ("fast-decay", 1.0, 8.0),
    ("no-decay", 1.0, -8.0),
)

STEPS = 256
"""Consecutive single-token calls in the long-run test.

Sixteen chunks of the oracle, so the carry crosses a chunk boundary fifteen times
and the state is rebuilt from its own output 255 times.
"""

MIXER_CONFIG = SLinOSSConfig(
    d_model=32, d_state=48, d_head=16, n_groups=2, chunk_size=CHUNK, bias=True
)
"""Four heads over two groups at the smallest legal state and head widths.

``d_conv`` and ``key_conv`` vary per case, so they are set where they are read.
"""


class CarryKwargs(TypedDict):
    """The three in-place carries, keyed by parameter name."""

    ssm: Tensor
    b_prev: Tensor
    u_prev: Tensor


class Carries(NamedTuple):
    """The state one call advances."""

    ssm: Tensor
    b_prev: Tensor
    u_prev: Tensor

    def kw(self) -> CarryKwargs:
        """Keyword operands, in signature order."""
        return {"ssm": self.ssm, "b_prev": self.b_prev, "u_prev": self.u_prev}


def _carries(inp: ScanInputs) -> Carries:
    """The fixture's carry-in as three buffers of its own.

    Cloned rather than aliased: the boundary writes these, so a call that shared
    them with the fixture would leave the oracle reading its own output.
    """
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    return Carries(
        ssm=inp.z0.clone(), b_prev=inp.b_prev.clone(), u_prev=inp.u_prev.clone()
    )


def _zeros(inp: ScanInputs) -> Carries:
    """Carries at a sequence start: what an omitted carry-in means."""
    bsz, heads, _, rows = inp.U.shape
    groups, dim = int(inp.B.shape[1]), int(inp.B.shape[-1])
    where: dict[str, Any] = {"device": inp.U.device}
    return Carries(
        ssm=torch.zeros(bsz, heads, rows, dim, dtype=inp.trans.dtype, **where),
        b_prev=torch.zeros(bsz, groups, dim, dtype=inp.B.dtype, **where),
        u_prev=torch.zeros(bsz, heads, rows, dtype=inp.U.dtype, **where),
    )


class Call(NamedTuple):
    """One valid call to the boundary: the operands and the state they advance."""

    inp: ScanInputs
    carry: Carries

    def run(self) -> Tensor:
        """Advance ``carry`` one token and return ``y``."""
        return decode_ref(*self.inp.args(), **self.carry.kw())

    def fresh(self) -> Call:
        """The same operands against a second copy of the same carry-in."""
        return Call(inp=self.inp, carry=_carries(self.inp))


def _token(inp: ScanInputs, t: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Token ``t`` of a longer call, packed as the mixer's producers hand it over.

    A token slice of a time-major buffer is strided, and every producer of a decode
    step emits one token contiguously, so the slice is packed here.
    """
    where = slice(t, t + 1)
    return (
        inp.U[:, :, where].contiguous(),
        inp.trans[:, :, where].contiguous(),
        inp.K[:, :, where].contiguous(),
        inp.B[:, :, where].contiguous(),
        inp.C[:, :, where].contiguous(),
    )


def _shape_id(shape: tuple[int, ...]) -> str:
    return "b{}h{}g{}p{}n{}".format(*shape)


def _base(**overrides: Any) -> Call:
    defaults: dict[str, Any] = {
        "seqlen": TOKENS,
        "bsz": 2,
        "heads": 2,
        "groups": 2,
        "rows": 16,
        "lanes": 16,
        "seed": 29,
    }
    inp = make_inputs(**{**defaults, **overrides})
    return Call(inp=inp, carry=_carries(inp))


# ---------------------------------------------------------------------------
# Parity. Three implementations of one map.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_agrees_with_both_reference_implementations(
    shape: tuple[int, int, int, int, int], device: torch.device
) -> None:
    """The boundary against the token loop and against the chunked path.

    Two-way agreement passes a derivation error the two sides share, so all three
    pairs are asserted. The carries come out bitwise: both are an operand at one
    token and neither implementation computes them.
    """
    bsz, heads, groups, rows, lanes = shape
    call = _base(
        bsz=bsz,
        heads=heads,
        groups=groups,
        rows=rows,
        lanes=lanes,
        device=device,
        seed=lanes + rows,
    )
    inp = call.inp
    got = call.run()
    seq = so3ssm(*inp.args(), **inp.kw())
    chunked = so3ssd(*inp.args(), CHUNK, **inp.kw())

    for label, want in (("so3ssm", seq), ("so3ssd", chunked)):
        assert_max_rel(got, want.y, PARITY_REL, f"y vs {label}")
        assert_max_rel(call.carry.ssm, want.state, PARITY_REL, f"state vs {label}")
        assert torch.equal(call.carry.b_prev, want.b_last), label
        assert torch.equal(call.carry.u_prev, want.u_last), label
    assert_max_rel(chunked.y, seq.y, PARITY_REL, "so3ssd vs so3ssm")


@pytest.mark.parametrize(
    ("w_scale", "ls_bias"), [case[1:] for case in REGIMES], ids=[c[0] for c in REGIMES]
)
def test_agrees_at_every_transition_regime(w_scale: float, ls_bias: float) -> None:
    """Both ends of the rotation chart and both ends of the decay range.

    The tap series switches branch inside :func:`slinoss.ops.scanprep.foh_taps` on
    the norm of the generator, so a boundary that read one tap column or one chart
    coordinate wrongly can still agree mid-range, where every term is the same
    order of magnitude.
    """
    call = _base(w_scale=w_scale, ls_bias=ls_bias, seed=31)
    got = call.run()
    want = so3ssm(*call.inp.args(), **call.inp.kw())
    assert_max_rel(got, want.y, PARITY_REL, "regime y")
    assert_max_rel(call.carry.ssm, want.state, PARITY_REL, "regime state")


def test_two_calls_carry_what_one_two_token_call_carries() -> None:
    """A ``T = 2`` call split in two. The only test that sees the carry order.

    ``b_prev`` and ``u_prev`` feed the previous tap of the next token, so a step
    that wrote them before forming that tap forces the current token twice and is
    exact at ``T = 1``. The carry-in of the second call is what the first one wrote,
    never a fabricated one.
    """
    inp = make_inputs(seqlen=2, bsz=2, heads=4, groups=2, rows=16, lanes=16, seed=37)
    carry = _carries(inp)
    ys = [
        decode_step(*_token(inp, t), **carry.kw()).y for t in range(int(inp.U.shape[2]))
    ]
    want = so3ssd(*inp.args(), CHUNK, **inp.kw())
    assert_max_rel(torch.cat(ys, dim=2), want.y, PARITY_REL, "split y")
    assert_max_rel(carry.ssm, want.state, PARITY_REL, "split state")
    assert torch.equal(carry.b_prev, want.b_last)
    assert torch.equal(carry.u_prev, want.u_last)


def test_a_long_run_of_single_calls_matches_one_whole_call() -> None:
    """:data:`STEPS` consecutive calls against one ``T``-token call. Both of them.

    A state written before it is read, or a carry that drifts by one token, is a
    fixed relative error per step: over 256 steps it leaves the trailing tokens
    wrong by a margin the first few do not show.

    Chained against the token loop as well as against the chunked path. A single
    step cannot separate the map from one that mishandles the carry, and the token
    loop is what defines the map, so the chained run is asserted against it
    directly rather than through the chunked path's agreement with it.
    """
    inp = make_inputs(
        seqlen=STEPS, bsz=1, heads=2, groups=1, rows=16, lanes=16, seed=41
    )
    carry = _carries(inp)
    ys = [decode_step(*_token(inp, t), **carry.kw()).y for t in range(STEPS)]
    got = torch.cat(ys, dim=2)
    for label, want in (
        ("so3ssm", so3ssm(*inp.args(), **inp.kw())),
        ("so3ssd", so3ssd(*inp.args(), CHUNK, **inp.kw())),
    ):
        assert_max_rel(got, want.y, PARITY_REL, f"long-run y vs {label}")
        assert_max_rel(carry.ssm, want.state, PARITY_REL, f"long-run state vs {label}")
        assert torch.equal(carry.b_prev, want.b_last), label
        assert torch.equal(carry.u_prev, want.u_last), label


@pytest.mark.parametrize(
    ("w_scale", "ls_bias"), [case[1:] for case in REGIMES], ids=[c[0] for c in REGIMES]
)
def test_a_sequence_start_drops_the_previous_tap_exactly(
    w_scale: float, ls_bias: float
) -> None:
    """Zero carries against the omitted carry-in, and the tap annihilated.

    An omitted carry-in is a zero one, so the previous tap must leave the numbers
    bit for bit alone rather than within a tolerance: at a sequence start every
    generated token would otherwise carry a term that is small only because its
    operand is.

    Over every transition regime, so ``w = 0`` is one of the cases: the previous
    tap must annihilate its operand at the chart origin too, where the tap series
    takes its analytic branch and ``kr``, ``g`` and ``h`` are not the mid-range
    values.
    """
    inp = make_inputs(
        seqlen=TOKENS,
        rows=16,
        lanes=16,
        seed=43,
        with_state=False,
        streaming=False,
        w_scale=w_scale,
        ls_bias=ls_bias,
    )
    start = _zeros(inp)
    got = decode_ref(*inp.args(), **start.kw())
    want = so3ssm(*inp.args(), **inp.kw())
    assert max_err(got, want.y) == 0.0
    assert max_err(start.ssm, want.state) == 0.0

    # The same numbers from an arbitrary b_prev: with u_prev zero the previous tap
    # is annihilated by the multiply, not merely small.
    other = _zeros(inp)._replace(b_prev=torch.randn_like(start.b_prev))
    assert torch.equal(decode_ref(*inp.args(), **other.kw()), got)
    assert torch.equal(other.ssm, start.ssm)


def test_the_oscillator_basis_start_matches_the_token_loop() -> None:
    """The shipped initial state, which is sparse and mostly exact zeros.

    :func:`slinoss.state.oscillator_basis` puts one unit coordinate in each row, so
    an index or a broadcast that a dense random state hides -- a rotation applied
    along the wrong lane axis, a state read head-major -- moves a single component
    here and is visible in the output.
    """
    cfg = MIXER_CONFIG
    inp = make_inputs(
        seqlen=TOKENS,
        bsz=2,
        heads=cfg.n_heads,
        groups=cfg.n_groups,
        rows=cfg.d_head,
        lanes=cfg.n_lanes,
        seed=47,
    )
    basis = oscillator_basis(cfg, device=inp.U.device, dtype=inp.trans.dtype)
    start = basis.unsqueeze(0).expand(2, -1, -1, -1).clone()
    want = so3ssm(*inp.args(), z0=start, b_prev=inp.b_prev, u_prev=inp.u_prev)
    carry = _carries(inp)._replace(ssm=start.clone())
    got = decode_ref(*inp.args(), **carry.kw())
    assert_max_rel(got, want.y, PARITY_REL, "basis y")
    assert_max_rel(carry.ssm, want.state, PARITY_REL, "basis state")


# ---------------------------------------------------------------------------
# In-place semantics and dtypes
# ---------------------------------------------------------------------------


def test_the_state_is_advanced_in_the_buffers_it_was_handed() -> None:
    """Three addresses unchanged, three buffers written.

    A returned state the caller copies doubles the only traffic the step has, and a
    rebound buffer leaves a captured graph writing memory no consumer reads, which
    replays as a state frozen at its first token. The carries are checked against
    the operands they must become, and the state against the value it must leave.
    """
    call = _base(dtype=torch.float32, u_dtype=torch.bfloat16, bc_dtype=torch.bfloat16)
    before = {name: buf.data_ptr() for name, buf in zip(call.carry._fields, call.carry)}
    was = call.carry.ssm.clone()

    out = decode_step(*call.inp.args(), **call.carry.kw())

    for name, buf in zip(call.carry._fields, call.carry):
        assert buf.data_ptr() == before[name], name
    assert not torch.equal(call.carry.ssm, was)
    assert torch.equal(call.carry.b_prev, call.inp.B[:, :, 0])
    assert torch.equal(call.carry.u_prev, call.inp.U[:, :, 0])
    assert out.y.data_ptr() != call.carry.ssm.data_ptr()


@pytest.mark.parametrize(
    ("activation", "state"),
    [(torch.bfloat16, torch.float32), (torch.float64, torch.float64)],
    ids=["bf16", "fp64"],
)
def test_the_state_carries_the_pinned_dtype_not_the_activation_dtype(
    activation: torch.dtype, state: torch.dtype
) -> None:
    """``z`` of :data:`slinoss._precision.PINNED_TENSORS`, through an in-place write.

    The state is the caller's buffer, so narrowing it to the activation dtype would
    downcast the recurrence at every step while posting the halved traffic as a
    kernel win. float64 activations widen it instead, which is what makes the
    reference an fp64 oracle.
    """
    assert "z" in PINNED_TENSORS
    call = _base(dtype=state, u_dtype=activation, bc_dtype=activation, seed=53)
    assert pinned_dtype(*call.inp.args()) is state

    out = decode_step(*call.inp.args(), **call.carry.kw())

    assert out.y.dtype is activation
    assert call.carry.ssm.dtype is state
    assert call.carry.b_prev.dtype is activation
    assert call.carry.u_prev.dtype is activation


def test_result_shapes_and_dtypes() -> None:
    """The token axis survives at extent one, and ``y`` is what the tail reads."""
    call = _base(bsz=2, heads=4, groups=2, rows=32, lanes=16, seed=59)
    out = decode_step(*call.inp.args(), **call.carry.kw())
    assert isinstance(out, DecodeResult)
    assert out.y.shape == (2, 4, TOKENS, 32)
    assert out.y.dtype is call.inp.U.dtype
    assert out.y.is_contiguous()
    assert call.carry.ssm.shape == (2, 4, 32, 48)


# ---------------------------------------------------------------------------
# No gradient, and which implementation ran
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "pick"),
    [("U", lambda c: c.inp.U), ("ssm", lambda c: c.carry.ssm)],
    ids=["operand", "carry"],
)
def test_refuses_an_operand_that_carries_a_gradient(
    label: str, pick: Callable[[Call], Tensor]
) -> None:
    """A gradient is refused at the boundary, not detached inside it.

    Detaching would hand back a tensor whose gradient is silently zero, which is a
    training run that reports a number. Both groups are covered: an operand and a
    carry.
    """
    call = _base(seed=61)
    pick(call).requires_grad_(True)
    with pytest.raises(ValueError, match=f"{label} requires a gradient"):
        decode_step(*call.inp.args(), **call.carry.kw())


def test_the_step_records_no_graph() -> None:
    """No node under an enabled grad mode. A recorded node is a leak per token."""
    call = _base(seed=67)
    with torch.enable_grad():
        out = decode_step(*call.inp.args(), **call.carry.kw())
    assert not out.y.requires_grad
    assert out.y.grad_fn is None


def test_the_backward_refuses_and_names_the_differentiable_operator() -> None:
    """The registry carries both directions; this operator's second one raises."""
    assert resolve(None, "cpu", torch.float32).backward is decode_no_backward
    with pytest.raises(NotImplementedError, match="takes no gradient"):
        decode_no_backward()


def test_the_selected_implementation_is_observable() -> None:
    """The name that ran comes back, and an absent backend raises.

    A registry whose kernel import failed holds the reference alone and answers
    every call, so a caller that cannot see which implementation answered cannot
    tell a kernel measurement from a torch one.
    """
    call = _base(seed=71)
    out = decode_step(*call.inp.args(), **call.carry.kw())
    assert out.backend == resolve(None, "cpu", call.inp.U.dtype).name
    assert out.backend in names()
    assert REFERENCE in names()
    again = call.fresh()
    with pytest.raises(ValueError, match=CUTE):
        decode_step(*again.inp.args(), **again.carry.kw(), backend=CUTE)


# ---------------------------------------------------------------------------
# The mixer's own operands, and which recurrence a token extent reaches
# ---------------------------------------------------------------------------


def _mixer(
    d_conv: int,
    key_conv: bool,
    dtype: torch.dtype,
    *,
    n_groups: int = MIXER_CONFIG.n_groups,
    seed: int = 0,
) -> SLinOSSMixer:
    """A seeded mixer on CUDA at ``dtype``, or a skip.

    CUDA only: a pitched band is refused off a CUDA device, and both vector
    operands of a mixer step are bands of the fused projection.

    Args:
        d_conv: Convolution width.
        key_conv: Convolve the key bands.
        dtype: Parameter dtype.
        n_groups: Groups sharing one ``B``/``C`` pair. Divides the four heads
            :data:`MIXER_CONFIG` carries.
        seed: Parameter seed. Two mixers built at different seeds hold different
            values in same-sized buffers, which is what a test of buffer lifetime
            needs.

    Returns:
        The mixer, with the output projection and the forcing band drawn.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = replace(MIXER_CONFIG, d_conv=d_conv, key_conv=key_conv, n_groups=n_groups)
    torch.manual_seed(seed)
    mixer = SLinOSSMixer(cfg, device=torch.device("cuda")).to(dtype)
    # Initialization zeroes the output projection and the projection's forcing band,
    # so an untouched mixer returns zeros and forces the recurrence with zeros: a
    # comparison of two step outputs would then hold on both sides whatever the two
    # paths did, and the two-tap forcing would never be evaluated on a value. Drawn
    # here rather than in the tests, since it is every mixer in this file that needs
    # it, and after construction so the lattice the parameter band states is intact.
    with torch.no_grad():
        mixer.out_proj.weight.normal_(std=0.05)
        mixer.in_proj.weight[mixer.layout.b_off : mixer.layout.c_off].normal_(std=0.05)
    return mixer


def _refuse_the_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make fetching the chunked scan's forward an error.

    The name is resolved either way, on device type and dtype; what a call selects
    is the backend it fetches, so this is where a routed step is caught reaching the
    wrong recurrence.
    """

    def refuse(name: str) -> Any:
        raise AssertionError(f"the step fetched the chunked scan backend {name!r}")

    monkeypatch.setattr(scan_dispatch, "get", refuse)


def _watch_the_boundary(monkeypatch: pytest.MonkeyPatch) -> list[Carries]:
    """Record the carries each call to the boundary is handed, and forward it.

    Returns:
        One entry per call, in call order, holding the operands themselves.
    """
    seen: list[Carries] = []

    def watch(
        *operands: Tensor, ssm: Tensor, b_prev: Tensor, u_prev: Tensor
    ) -> DecodeResult:
        seen.append(Carries(ssm=ssm, b_prev=b_prev, u_prev=u_prev))
        return decode_step(*operands, ssm=ssm, b_prev=b_prev, u_prev=u_prev)

    monkeypatch.setattr(mixer_module, "decode_step", watch)
    return seen


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("d_conv", "key_conv"),
    [(4, True), (4, False), (1, True)],
    ids=["keys", "no-keys", "width-1"],
)
def test_a_one_token_step_reaches_the_boundary_over_the_state_itself(
    monkeypatch: pytest.MonkeyPatch, d_conv: int, key_conv: bool
) -> None:
    """The routing, and the identity that makes three carry copies redundant.

    Two claims, and the second is why the first is worth anything. The step must
    reach the boundary and not the chunked scan at ``T = 1``, and the three buffers
    it hands over must be ``state``'s own, not copies: the boundary writes its state
    in place, so ``state.ssm.copy_(...)`` after it is a full read and a full write
    of a buffer onto itself. Identity is asserted by ``is`` and the addresses again
    after the call, since a rebound buffer would leave a captured graph writing
    memory no consumer reads.

    The operands are the real ones: ``U`` from the value convolution, ``B`` and
    ``C`` as pitched bands of the fused projection or of the key convolution's
    output, the parameters from the frontier, and the state from
    :meth:`slinoss.state.MixerState.allocate`. A fixture cannot produce that pitch,
    that dtype pair, and that sparse initial state together.
    """
    mixer = _mixer(d_conv, key_conv, torch.float64)
    cfg = mixer.config
    state = MixerState.allocate(
        cfg, 2, device=torch.device("cuda"), dtype=torch.float64
    )
    x = torch.randn(2, TOKENS, cfg.d_model, dtype=torch.float64, device=state.device)
    was = Carries(ssm=state.ssm, b_prev=state.b_prev, u_prev=state.u_prev)
    before = tuple(buf.data_ptr() for buf in was)

    seen = _watch_the_boundary(monkeypatch)
    _refuse_the_scan(monkeypatch)
    out = mixer.step(x, state)

    assert out.shape == (2, TOKENS, cfg.d_model)
    assert len(seen) == 1, "one token is one call to the boundary"
    for name, given, mine in zip(was._fields, seen[0], was, strict=True):
        assert given is mine, f"the step handed the boundary a copy of {name}"
    now = (state.ssm, state.b_prev, state.u_prev)
    assert tuple(buf.data_ptr() for buf in now) == before, "a carry was rebound"


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("d_conv", "key_conv"),
    [(4, True), (4, False), (1, True)],
    ids=["keys", "no-keys", "width-1"],
)
def test_a_routed_one_token_step_is_the_chunked_step_it_replaces(
    monkeypatch: pytest.MonkeyPatch, d_conv: int, key_conv: bool
) -> None:
    """The routed step against the same step through the chunked path, at float64.

    Ground truth is the mixer as it ran before the routing existed: one token
    through the chunked scan, at the same chunk length, over the same state. The
    T-token path is reached by moving the extent the branch tests off one, so
    nothing else about the call changes and the comparison is the two recurrences on
    one set of operands.

    All five carries, not only the three the recurrence advances: the two
    convolution windows are copied on both paths, and a routing that dropped one of
    those copies is a state that stops advancing while every number here still
    agrees on the step it was taken from.
    """
    mixer = _mixer(d_conv, key_conv, torch.float64)
    cfg = mixer.config
    cuda = torch.device("cuda")
    x = torch.randn(2, TOKENS, cfg.d_model, dtype=torch.float64, device=cuda)

    chunked = MixerState.allocate(cfg, 2, device=cuda, dtype=torch.float64)
    # No extent is zero, so the branch takes the T-token path over one token.
    monkeypatch.setattr(mixer_module, "TOKENS", 0)
    want = mixer.step(x, chunked)
    monkeypatch.undo()

    routed = MixerState.allocate(cfg, 2, device=cuda, dtype=torch.float64)
    got = mixer.step(x, routed)

    assert_max_rel(got, want, PARITY_REL, "mixer step out")
    assert_max_rel(routed.ssm, chunked.ssm, PARITY_REL, "mixer step ssm")
    for carry in ("conv", "keys", "b_prev", "u_prev"):
        assert torch.equal(getattr(routed, carry), getattr(chunked, carry)), carry


@pytest.mark.cuda
def test_more_than_one_token_still_reaches_the_chunked_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``T != 1`` is the path it was, and the boundary refuses that extent anyway.

    A prefill is not a decode step: the branch is on the extent alone, so this is
    the case that would silently take the one-token boundary if the test were
    written on anything else. All five carries advance, which is what the copies the
    T-token path ends with are for. ``b_prev`` reaches this comparison only because
    :func:`_mixer` draws the forcing band: at the initialized value that carry
    leaves the scan as the zero it entered with, and the copy of it is invisible.
    """
    mixer = _mixer(4, True, torch.float64)
    cfg = mixer.config
    cuda = torch.device("cuda")
    state = MixerState.allocate(cfg, 2, device=cuda, dtype=torch.float64)
    was = state.clone()
    x = torch.randn(2, CHUNK + 1, cfg.d_model, dtype=torch.float64, device=cuda)

    seen = _watch_the_boundary(monkeypatch)
    out = mixer.step(x, state)

    assert out.shape == (2, CHUNK + 1, cfg.d_model)
    assert seen == [], "a prefill reached the one-token boundary"
    for carry in ("conv", "keys", "ssm", "b_prev", "u_prev"):
        assert not torch.equal(getattr(state, carry), getattr(was, carry)), carry


@pytest.mark.cuda
@pytest.mark.parametrize("tokens", [TOKENS, CHUNK + 1], ids=["one-token", "prefill"])
def test_a_mixer_step_records_no_graph_at_either_extent(tokens: int) -> None:
    """Under an enabled grad mode, with parameters that carry gradients.

    The step calls its backends directly rather than through an autograd node, at
    both extents. A recorded node holds every intermediate of the step alive, which
    is a leak per token, and the boundary would refuse the call on the next token
    rather than at the one that recorded it.
    """
    mixer = _mixer(4, True, torch.float64)
    cfg = mixer.config
    cuda = torch.device("cuda")
    state = MixerState.allocate(cfg, 2, device=cuda, dtype=torch.float64)
    x = torch.randn(2, tokens, cfg.d_model, dtype=torch.float64, device=cuda)
    assert any(p.requires_grad for p in mixer.parameters())

    with torch.enable_grad():
        out = mixer.step(x, state)

    assert not out.requires_grad
    assert out.grad_fn is None


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("n_groups", "batch"),
    [(1, 1), (2, 3), (4, 2)],
    ids=["shared-pair", "two-per-group", "pair-per-head"],
)
def test_a_captured_one_token_step_replays_as_the_eager_step(
    n_groups: int, batch: int
) -> None:
    """One routed step in a CUDA graph, replayed against the eager step.

    bfloat16, because that is the dtype the kernel path runs; float32 resolves the
    reference and would say nothing about whether a decode launch reaches a graph.

    The graph records addresses, so this is the test that the routed step has none
    that move: the boundary writes ``state``'s buffers, and the eager step it is
    compared against is taken from a copy of the carries the capture left. Bitwise,
    on the output and on all five carries. One kernel sequence over one shape agrees
    to the bit, so a tolerance here would only admit a replay of something else.

    Every grouping rule, since the group axis is what decides how the ``B``/``C``
    bands are addressed: all four heads on one pair, two heads per pair, and a pair
    per head. The batch differs per case as well, so no case shares a shape with
    another and a launch geometry that only works at one of them fails here.

    :func:`slinoss.graph.capture` refuses a recording that compiled an executor, so
    the warmup it runs is also what proves the boundary traces before the capture
    rather than inside it.
    """
    mixer = _mixer(4, True, torch.bfloat16, n_groups=n_groups)
    cfg = mixer.config
    cuda = torch.device("cuda")
    state = MixerState.allocate(cfg, batch, device=cuda, dtype=torch.bfloat16)
    x = torch.randn(batch, TOKENS, cfg.d_model, dtype=torch.bfloat16, device=cuda)
    before = tuple(buf.data_ptr() for buf in (state.ssm, state.b_prev, state.u_prev))

    step = capture(lambda ids: mixer.step(ids, state), x)

    # From what the warmup and the recording left, not from the entry state: the
    # graph holds these buffers and the eager step has to start where a replay does.
    eager_state = state.clone()
    eager = mixer.step(x, eager_state).clone()

    replayed = step(x)

    assert torch.equal(replayed, eager), "the replay is not the step"
    for carry in ("conv", "keys", "ssm", "b_prev", "u_prev"):
        assert torch.equal(getattr(state, carry), getattr(eager_state, carry)), carry
    now = (state.ssm, state.b_prev, state.u_prev)
    assert tuple(buf.data_ptr() for buf in now) == before, "a carry was rebound"


@pytest.mark.cuda
def test_a_captured_step_replays_after_its_operands_are_dropped() -> None:
    """The step, alone, over memory the allocator has been asked to reuse.

    Found live, in the graph-boundary measurement: a driver that returns the graph
    arm alone drops the mixer and the state, since the only reference to either was
    the eager arm's closure. The recorded graph addresses those buffers by pointer
    and names no owner, so freeing them hands the blocks to the next allocation and
    the replay reads what that allocation wrote. At batch 8 through a stack it was a
    gather index of ``4575657221408423936`` against a vocabulary of 50257, at batch
    128 an illegal access, and with nothing else allocated no error at all -- just
    non-finite logits from a step that still returned.

    Two things are asserted, because either alone admits the defect: that the mixer
    and the state are still reachable after the caller's names for them are gone,
    and that the replay is still the step. Ground truth is a second mixer at the same
    seed, stepped from a copy of the carries the capture left, while the captured
    mixer is still alive. The churn between the drop and the replay is a third mixer
    of the same geometry at a different seed, so every freed buffer has a same-sized
    request to be handed to and holds different values once it is.
    """
    oracle = _mixer(4, True, torch.bfloat16)
    cfg = oracle.config
    cuda = torch.device("cuda")
    x = torch.randn(2, TOKENS, cfg.d_model, dtype=torch.bfloat16, device=cuda)

    def record() -> tuple[GraphedStep, Tensor, tuple[weakref.ref[Any], ...]]:
        """Capture a step over locals, and return only the step.

        The scope is the point: what a driver hands back is all that stays
        reachable, so the mixer and the state leave here as weak references.
        """
        mixer = _mixer(4, True, torch.bfloat16)
        state = MixerState.allocate(cfg, 2, device=cuda, dtype=torch.bfloat16)
        assert all(
            torch.equal(value, oracle.state_dict()[name])
            for name, value in mixer.state_dict().items()
        ), "one seed built two mixers that differ, so the oracle is not this step"
        step = capture(lambda ids: mixer.step(ids, state), x)
        eager = oracle.step(x, state.clone()).clone()
        return step, eager, (weakref.ref(mixer), weakref.ref(state))

    step, want, watch = record()
    gc.collect()

    assert watch[0]() is not None, "the step let the mixer it replays over be freed"
    assert watch[1]() is not None, "the step let the state it advances be freed"

    churn = _mixer(4, True, torch.bfloat16, seed=1)
    churn.step(x, MixerState.allocate(cfg, 2, device=cuda, dtype=torch.bfloat16))

    got = step(x)
    assert torch.isfinite(got).all(), "the replay read memory that was reallocated"
    assert torch.equal(got, want), "the replay is not the step that was recorded"


@pytest.mark.cuda
def test_accepts_the_vector_operands_as_projection_bands() -> None:
    """``B`` and ``C`` as column bands of a wider buffer.

    The mixer hands both out at the projection pitch, so a boundary that demanded
    contiguity would need a staging copy of that projection per step. A band holds
    the values of its packed copy, so every output must match bit for bit.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    packed = _base(device="cuda")
    banded = Call(
        inp=packed.inp._replace(
            B=projection_band(packed.inp.B), C=projection_band(packed.inp.C)
        ),
        carry=_carries(packed.inp),
    )
    assert not banded.inp.B.is_contiguous()
    assert not banded.inp.C.is_contiguous()

    want, got = packed.run(), banded.run()
    assert torch.equal(got, want)
    for field, mine, theirs in zip(
        banded.carry._fields, banded.carry, packed.carry, strict=True
    ):
        assert torch.equal(mine, theirs), field


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


def _lane_strided(t: Tensor) -> Tensor:
    """``t``'s values with a step along the trailing axis.

    A row pitch describes the gap between two rows, so no pitch expresses this one:
    the ``3N`` components of a token are not adjacent.
    """
    wide = torch.zeros(
        *t.shape[:-1], 2 * int(t.shape[-1]), dtype=t.dtype, device=t.device
    )
    out = wide[..., ::2]
    out.copy_(t)
    return out


def _ops(call: Call, **fields: Tensor) -> Call:
    """``call`` with operands replaced."""
    return call._replace(inp=call.inp._replace(**fields))


def _carry(call: Call, **fields: Tensor) -> Call:
    """``call`` with carries replaced."""
    return call._replace(carry=call.carry._replace(**fields))


def _narrow(call: Call, width: int) -> Call:
    """Every ``3N`` operand and carry cut to ``width``, so the shapes still agree."""
    return Call(
        inp=call.inp._replace(
            B=call.inp.B[..., :width].contiguous(),
            C=call.inp.C[..., :width].contiguous(),
        ),
        carry=call.carry._replace(
            ssm=call.carry.ssm[..., :width].contiguous(),
            b_prev=call.carry.b_prev[..., :width].contiguous(),
        ),
    )


Mutate = Callable[[Call], Call]
Case = tuple[str, dict[str, Any], Mutate, type[Exception], str]

BAD_INPUTS: tuple[Case, ...] = (
    (
        "u_rank",
        {},
        lambda c: _ops(c, U=c.inp.U[:, :, :, 0]),
        ValueError,
        r"U must be \(B,H,1,P\)",
    ),
    (
        "b_rank",
        {},
        lambda c: _ops(c, B=c.inp.B[..., 0]),
        ValueError,
        r"B must be \(B,G,1,3N\)",
    ),
    ("two_tokens", {"seqlen": 2}, lambda c: c, ValueError, "exactly 1 token"),
    (
        "groups_not_dividing",
        {"heads": 4, "groups": 3},
        lambda c: c,
        ValueError,
        "G dividing H",
    ),
    (
        "trans_shape",
        {},
        lambda c: _ops(c, trans=c.inp.trans[..., :3].contiguous()),
        ValueError,
        "trans must have shape",
    ),
    (
        "k_shape",
        {},
        lambda c: _ops(c, K=c.inp.K[..., :1, :].contiguous()),
        ValueError,
        "K must have shape",
    ),
    (
        "c_shape",
        {},
        lambda c: _ops(c, C=c.inp.C[..., :-3].contiguous()),
        ValueError,
        "C must have shape",
    ),
    (
        "ssm_shape",
        {},
        lambda c: _carry(c, ssm=c.carry.ssm[..., :-3].contiguous()),
        ValueError,
        "ssm must have shape",
    ),
    (
        "b_prev_shape",
        {},
        lambda c: _carry(c, b_prev=c.carry.b_prev[..., :-3].contiguous()),
        ValueError,
        "b_prev must have shape",
    ),
    (
        "u_prev_shape",
        {},
        lambda c: _carry(c, u_prev=c.carry.u_prev[..., :-1].contiguous()),
        ValueError,
        "u_prev must have shape",
    ),
    (
        "lanes_not_multiple",
        {"lanes": 8},
        lambda c: c,
        ValueError,
        "3N must be 3 times a multiple of 16",
    ),
    (
        "state_not_triple",
        {"lanes": 32},
        lambda c: _narrow(c, 50),
        ValueError,
        "3N must be 3 times a multiple of 16",
    ),
    (
        "rows_not_multiple",
        {"rows": 12},
        lambda c: c,
        ValueError,
        "P must be a multiple of 16",
    ),
    (
        "u_strided",
        {},
        lambda c: _ops(c, U=_noncontig(c.inp.U)),
        ValueError,
        "U must be contiguous",
    ),
    (
        "trans_strided",
        {},
        lambda c: _ops(c, trans=_noncontig(c.inp.trans)),
        ValueError,
        "trans must be contiguous",
    ),
    (
        "k_strided",
        {},
        lambda c: _ops(c, K=_noncontig(c.inp.K)),
        ValueError,
        "K must be contiguous",
    ),
    (
        "ssm_strided",
        {},
        lambda c: _carry(c, ssm=_noncontig(c.carry.ssm)),
        ValueError,
        "ssm must be contiguous",
    ),
    (
        "b_prev_strided",
        {},
        lambda c: _carry(c, b_prev=_noncontig(c.carry.b_prev)),
        ValueError,
        "b_prev must be contiguous",
    ),
    (
        "u_prev_strided",
        {},
        lambda c: _carry(c, u_prev=_noncontig(c.carry.u_prev)),
        ValueError,
        "u_prev must be contiguous",
    ),
    (
        "u_dtype",
        {},
        lambda c: _ops(c, U=c.inp.U.to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "b_dtype",
        {},
        lambda c: _ops(c, B=c.inp.B.to(torch.int32)),
        TypeError,
        "supported",
    ),
    (
        "c_dtype",
        {},
        lambda c: _ops(c, C=c.inp.C.to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "b_prev_dtype",
        {},
        lambda c: _carry(c, b_prev=c.carry.b_prev.to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "u_prev_dtype",
        {},
        lambda c: _carry(c, u_prev=c.carry.u_prev.to(torch.int64)),
        TypeError,
        "supported",
    ),
    (
        "trans_low",
        {},
        lambda c: _ops(c, trans=c.inp.trans.to(torch.bfloat16)),
        TypeError,
        "float32-pinned",
    ),
    (
        "k_low",
        {},
        lambda c: _ops(c, K=c.inp.K.to(torch.float16)),
        TypeError,
        "float32-pinned",
    ),
    (
        "ssm_low",
        {},
        lambda c: _carry(c, ssm=c.carry.ssm.to(torch.bfloat16)),
        TypeError,
        "float32-pinned",
    ),
    (
        "ssm_narrowed",
        {},
        lambda c: _carry(c, ssm=c.carry.ssm.to(torch.float32)),
        ValueError,
        "never narrowed",
    ),
    (
        "b_prev_activation_dtype",
        {"dtype": torch.float32, "bc_dtype": torch.bfloat16},
        lambda c: _carry(c, b_prev=c.carry.b_prev.to(torch.float32)),
        ValueError,
        "carries its dtype",
    ),
    (
        "u_prev_activation_dtype",
        {"dtype": torch.float32, "u_dtype": torch.bfloat16},
        lambda c: _carry(c, u_prev=c.carry.u_prev.to(torch.float32)),
        ValueError,
        "carries its dtype",
    ),
    (
        "aliased_carry",
        {},
        lambda c: _carry(c, b_prev=c.inp.B[:, :, 0]),
        ValueError,
        "shares storage with B",
    ),
)


@pytest.mark.parametrize(
    ("base_kwargs", "mutate", "exc", "match"),
    [case[1:] for case in BAD_INPUTS],
    ids=[case[0] for case in BAD_INPUTS],
)
def test_rejects_bad_operands(
    base_kwargs: dict[str, Any],
    mutate: Mutate,
    exc: type[Exception],
    match: str,
) -> None:
    """Every message the shared guard produces.

    Through :func:`slinoss.ops.decode.decode_ref` rather than the public callable:
    the public one resolves a backend from the activation dtype first, so an
    unsupported dtype is reported by the registry there and the guard's own message
    is only reachable here.
    """
    call = mutate(_base(**base_kwargs))
    with pytest.raises(exc, match=match):
        call.run()


def test_the_public_callable_does_not_swallow_the_guard() -> None:
    """A shape the guard refuses raises through the interface as well."""
    base = _base(seed=73)
    call = _ops(base, U=_noncontig(base.inp.U))
    with pytest.raises(ValueError, match="U must be contiguous"):
        decode_step(*call.inp.args(), **call.carry.kw())


@pytest.mark.cuda
def test_rejects_mixed_devices() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    call = _base(device="cuda")
    with pytest.raises(ValueError, match="one device only"):
        _ops(call, C=call.inp.C.cpu()).run()


@pytest.mark.cuda
def test_rejects_a_strided_state_axis_on_b() -> None:
    """The band rule frees ``B``'s row pitch and nothing else.

    A step along ``3N`` leaves a token's components non-adjacent, which no pitch
    argument expresses and no kernel reads.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    call = _base(device="cuda")
    with pytest.raises(ValueError, match="unit stride on its trailing axis"):
        _ops(call, B=_lane_strided(call.inp.B)).run()
