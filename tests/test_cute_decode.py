"""The CuTe one-token step, against the reference at float64.

:func:`slinoss.ops.decode.reference.decode_ref` is the oracle and
:func:`slinoss.ops.so3ssd.so3ssd_ref` is the whole-sequence authority behind it.
Operands are built in float32 and cast, and the oracle runs on an exact float64
upcast of those same cast tensors, so the residual is the kernel's own arithmetic
rather than the rounding of the inputs.

One case per failure mode the kernel can have on its own, which is not one case per
input value. The modes are: an index set that permutes the component transpose onto
the wrong offsets, a state width whose lane count does not divide the warp, a lane
walk of more than one 3-vector, an operand width the kernel widens from, a pitched
vector band, a grouping that broadcasts one ``B``/``C`` row over several heads, each
transition regime of the chart, a carry the first token of a sequence must not be able
to see, a carry written before it is read, a buffer rebound instead of advanced, a
dispatch that answered from the reference, and drift over a chained run.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss._cute import executor_count
from slinoss.ops.decode import decode_ref, decode_step
from slinoss.ops.decode.backends import CUTE, REFERENCE
from slinoss.ops.decode.cute import lane_exchange, lanes_per_thread, row_group
from slinoss.ops.decode.reference import TOKENS
from slinoss.ops.so3ssd import so3ssd_ref
from tests.conftest import (
    LS_BIAS,
    ScanInputs,
    assert_max_rel,
    make_inputs,
    projection_band,
)

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# LS_BIAS keeps the decay above float32 epsilon over a chained run; unbiased, the
# state contributes nothing after a few tokens and the rotation is tested on a zero.

# ``y`` is stored at the activation width and is the last operation on it, so the
# bound is one half-ulp of that width against the largest entry: 2^-9 for bfloat16
# and 2^-11 for float16. Everything before the store is float32. The float32 figure
# is the reduction bound, not a rounding: the readout sums 3N terms in an order the
# oracle does not share, and 3N reaches 384. Same reasoning and same values as
# ``tests/test_cute_forward.py`` and ``tests/test_cute_mma.py``.
Y_BOUNDS = {torch.bfloat16: 6e-3, torch.float16: 8e-4, torch.float32: 4e-6}

# The state is float32 at every activation width, and the activations it is formed
# from are the same cast tensors the oracle reads, so the bound is float32 rounding
# over about fifteen operations plus two approximate transcendentals: ``ex2.approx``
# in the scale and the ten-term float32 half-angle series in the rotation, each a few
# times 2^-22.
STATE_BOUND = 4e-6

# ``N``, one per distinct pair of a row group and a lane walk. Every legal state
# width up to the kernel cap: ``N`` a multiple of LANE_MULTIPLE, so the odd
# multiples take the half-warp group and the even ones a whole warp, and the walk
# runs from one 3-vector per thread to eight.
LANES = (16, 32, 48, 64, 80, 96, 112, 128)

DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# (label, w_scale, ls_bias). One coordinate of the chart moved off the middle of its
# range per case: ``w_scale`` zero is the exact origin of the tap chart, 1e-3 is the
# neighbourhood the series is truncated in, 100 drives ``|w|`` to the cap at
# ``2*w_max``; ``ls_bias`` reaches both ends of ``-LS_MAX_MAG*sigmoid``. Same set as
# ``tests/test_decode_op.py``, which is where the reference's own sweep lives.
REGIMES = (
    ("no-drive", 0.0, 0.0),
    ("near-identity", 1e-3, 0.0),
    ("max-drive", 100.0, 0.0),
    ("fast-decay", 1.0, 8.0),
    ("no-decay", 1.0, -8.0),
)

STEPS = 64
"""Consecutive single-token calls in the chained test.

Four chunks of the whole-sequence oracle at the chunk length below, so the carry
crosses a chunk boundary three times and the state is rebuilt from its own output 63
times.

Measured: 2.255e-06 on ``y`` and 1.607e-06 on the state at ``3N = 96``, against
1.940e-07 and 1.656e-07 for the same shape at one token. So the chain does cost about
an order of magnitude and the contraction by ``exp(2*ls)`` per token does not
cancel accumulation, which is 56% of the float32 bound rather than the 5% a single
step reads. Raising ``STEPS`` further needs the bound raised with it, and the figure
this reached is what says by how much.
"""

CHUNK = 16
"""Chunk length the whole-sequence oracle runs at. The smallest legal one."""


def _make(
    bsz: int,
    heads: int,
    groups: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
    *,
    seqlen: int = TOKENS,
    w_scale: float = 2.0,
    ls_bias: float = LS_BIAS,
    seed: int = 0,
) -> ScanInputs:
    """One operand set: float32 pinned tensors, ``dtype`` activations, on CUDA.

    A float64 ``dtype`` widens the pinned tensors too.
    :func:`slinoss._precision.pinned_dtype` reads float64 off any operand, so a
    float64 activation with a float32 state is not a legal call.
    """
    return make_inputs(
        bsz=bsz,
        heads=heads,
        groups=groups,
        seqlen=seqlen,
        rows=rows,
        lanes=lanes,
        dtype=torch.float64 if dtype is torch.float64 else torch.float32,
        device="cuda",
        seed=seed,
        w_scale=w_scale,
        ls_bias=ls_bias,
        u_dtype=dtype,
        bc_dtype=dtype,
    )


def _carries(inp: ScanInputs) -> tuple[Tensor, Tensor, Tensor]:
    """Fresh copies of the three in-place carries, in signature order."""
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    return inp.z0.clone(), inp.b_prev.clone(), inp.u_prev.clone()


def _oracle(
    inp: ScanInputs, carries: tuple[Tensor, Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    """``decode_ref`` on a float64 upcast of the same cast operands.

    Args:
        inp: The operand set. ``B`` and ``C`` are taken as given, band or not.
        carries: The three carries the kernel was handed, before the kernel ran.

    Returns:
        ``(y, state)`` at float64.
    """
    ssm, b_prev, u_prev = (one.double().clone().contiguous() for one in carries)
    y = decode_ref(
        inp.U.double().contiguous(),
        inp.trans.double().contiguous(),
        inp.K.double().contiguous(),
        inp.B.double().contiguous(),
        inp.C.double().contiguous(),
        ssm=ssm,
        b_prev=b_prev,
        u_prev=u_prev,
    )
    return y, ssm


def _tag(label: str, lanes: int, dtype: torch.dtype) -> str:
    width = str(dtype).removeprefix("torch.")
    return f"cute-decode[{label}/3N={3 * lanes}/{width}]"


def _run(inp: ScanInputs, carries: tuple[Tensor, Tensor, Tensor]) -> str:
    """One kernel call on ``carries``, advanced in place. Returns the backend name."""
    ssm, b_prev, u_prev = carries
    return decode_step(
        inp.U, inp.trans, inp.K, inp.B, inp.C, ssm=ssm, b_prev=b_prev, u_prev=u_prev
    ).backend


def _step(inp: ScanInputs, carries: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """One kernel call, asserting the kernel is what answered it."""
    ssm, b_prev, u_prev = carries
    out = decode_step(
        inp.U, inp.trans, inp.K, inp.B, inp.C, ssm=ssm, b_prev=b_prev, u_prev=u_prev
    )
    assert out.backend == CUTE, f"{out.backend} answered; the kernel did not register"
    return out.y


def _check(inp: ScanInputs, dtype: torch.dtype, label: str, lanes: int) -> None:
    """Run the kernel and the oracle on one operand set and bound the gap."""
    carries = _carries(inp)
    want_y, want_state = _oracle(inp, carries)
    got_y = _step(inp, carries)
    assert_max_rel(got_y, want_y, Y_BOUNDS[dtype], f"{_tag(label, lanes, dtype)}/y")
    assert_max_rel(
        carries[0], want_state, STATE_BOUND, f"{_tag(label, lanes, dtype)}/state"
    )


@pytest.mark.parametrize("group", (16, 32))
def test_the_component_transpose_is_the_permutation_it_claims(group: int) -> None:
    """Both layouts of the group's step, element for element, at both group widths.

    The exchange between plane layout and lane layout is a permutation of ``3*group``
    offsets over ``group`` threads, and a wrong index set is a permutation of the
    right length over the wrong offsets: the recurrence would still read finite
    numbers off three lanes it does not own, the reduction would still sum ``3N``
    terms, and the residual would be a plausible fraction rather than a failure. So
    the index set is checked as integer offsets rather than through the kernel, which
    also fixes the two group widths independently of which ``N`` reaches them.
    """
    swap = [lane_exchange(slot, group) for slot in range(group)]
    assert swap[0].segment == ((32 - group) << 8) | 31

    # Inbound. Plane ``k`` of thread ``t`` holds offset ``k*group + t``; thread ``t``
    # must end with ``3t``, ``3t+1``, ``3t+2``.
    for slot in range(group):
        for comp in range(3):
            source = swap[slot].inbound_lane[comp]
            plane = swap[source].inbound_plane[comp]
            assert plane * group + source == 3 * slot + comp, (
                f"inbound group={group} slot={slot} comp={comp}"
            )

    # Outbound, the inverse. Thread ``t`` holds ``3t``, ``3t+1``, ``3t+2`` and must
    # store ``k*group + t`` at plane ``k``.
    for slot in range(group):
        arrivals = [3 * swap[slot].outbound_lane[comp] + comp for comp in range(3)]
        for plane in range(3):
            assert arrivals[swap[slot].outbound_pick[plane]] == plane * group + slot, (
                f"outbound group={group} slot={slot} plane={plane}"
            )

    # Every lane read once per shuffle, in both directions: a permutation of the
    # group rather than a broadcast, which is what makes one shuffle per component
    # enough and two lanes reading one source a bug.
    for comp in range(3):
        assert sorted(one.inbound_lane[comp] for one in swap) == list(range(group))
        assert sorted(one.outbound_lane[comp] for one in swap) == list(range(group))


@pytest.mark.parametrize("lanes", LANES)
def test_every_state_width_matches_the_oracle(lanes: int) -> None:
    """Each row group and each lane-walk length, at one shape and one width.

    The lane-indexed reduction is what the sweep is for: a kernel correct at one
    ``N`` is not correct, and the two row-group widths take different butterflies
    over different lane sets.
    """
    inp = _make(1, 2, 1, 16, lanes, torch.bfloat16)
    _check(inp, torch.bfloat16, "width", lanes)
    # The launch geometry the sweep is covering, asserted rather than assumed: a
    # regression that made both branches equal would still pass the numerics.
    assert row_group(lanes) * lanes_per_thread(lanes) == lanes


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lanes", (16, 32))
def test_every_operand_width_matches_the_oracle(dtype: torch.dtype, lanes: int) -> None:
    """Each activation width the kernel widens from, at both row groups.

    Four heads over two groups, so a ``B``/``C`` row is broadcast over two heads and
    a head that read its own row would still be right at ``G == H``.
    """
    inp = _make(2, 4, 2, 32, lanes, dtype)
    _check(inp, dtype, "width-dtype", lanes)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_pitched_vector_band_matches_the_oracle(dtype: torch.dtype) -> None:
    """``B`` and ``C`` as column bands of a wider projection.

    The mixer hands the boundary a band, not a buffer, so the token stride is the
    projection width and the group axis strides less than the axis before it.
    """
    inp = _make(2, 4, 2, 16, 32, dtype)
    banded = inp._replace(B=projection_band(inp.B), C=projection_band(inp.C))
    assert banded.B.stride(-2) > banded.B.shape[-1]
    _check(banded, dtype, "band", 32)


@pytest.mark.parametrize(("label", "w_scale", "ls_bias"), REGIMES)
def test_every_transition_regime_matches_the_oracle(
    label: str, w_scale: float, ls_bias: float
) -> None:
    """Both ends of the rotation chart and both ends of the decay.

    ``w = 0`` exactly is the case the axis normal form cannot represent, and it is
    the one the polynomial tap chart exists for.
    """
    inp = _make(2, 2, 1, 16, 32, torch.float32, w_scale=w_scale, ls_bias=ls_bias)
    if w_scale == 0.0:
        assert bool((inp.trans[..., :3] == 0.0).all())
    _check(inp, torch.float32, label, 32)


def test_a_vanished_previous_tap_hides_the_carry_bitwise() -> None:
    """A sequence start cannot see the carry it was handed, to the last bit.

    Two ways the previous tap vanishes, both reachable at token zero: a zero
    ``u_prev``, which every fresh :class:`slinoss.state.MixerState` supplies, and a
    zero previous tap matrix. Either annihilates the term exactly rather than nearly,
    so a run from zeroed carries and a run from arbitrary ones are the same bits, and
    a kernel that folded the carry in through a division or a reciprocal would not be.
    """
    inp = _make(2, 4, 2, 16, 32, torch.float32)
    zero_tap = inp._replace(K=inp.K.clone())
    zero_tap.K[:, :, 0, 0, :3] = 0.0
    for label, prepared, drop_u in (
        ("zero-u-prev", inp, True),
        ("zero-tap", zero_tap, False),
    ):
        carried = _carries(prepared)
        zeroed = _carries(prepared)
        if drop_u:
            carried[2].zero_()
        zeroed[1].zero_()
        zeroed[2].zero_()
        want_y = _step(prepared, carried)
        got_y = _step(prepared, zeroed)
        assert torch.equal(got_y, want_y), f"{label}: y is not bitwise equal"
        assert torch.equal(zeroed[0], carried[0]), (
            f"{label}: state is not bitwise equal"
        )


def test_the_state_is_advanced_in_the_buffers_handed_in() -> None:
    """Three buffers written in place, at their own dtype, at their own addresses.

    A captured CUDA graph records buffer addresses, so a rebound field would leave a
    replay writing memory nobody reads. The two activation carries are exact copies
    of this token's operands, so they are checked bitwise rather than to a bound.
    """
    inp = _make(2, 4, 2, 32, 32, torch.bfloat16)
    carries = _carries(inp)
    ssm, b_prev, u_prev = carries
    before = (ssm.data_ptr(), b_prev.data_ptr(), u_prev.data_ptr())
    stale = ssm.clone()
    _step(inp, carries)
    assert (ssm.data_ptr(), b_prev.data_ptr(), u_prev.data_ptr()) == before
    assert ssm.dtype is torch.float32
    assert b_prev.dtype is inp.B.dtype and u_prev.dtype is inp.U.dtype
    assert not torch.equal(ssm, stale)
    assert torch.equal(b_prev, inp.B[:, :, 0])
    assert torch.equal(u_prev, inp.U[:, :, 0])


def test_the_kernel_is_the_backend_that_answers() -> None:
    """Dispatch reports which implementation ran, and float64 is not this one.

    A registry whose kernel import failed resolves to the reference and answers every
    call, so a measurement that cannot see the name is a measurement of whichever
    implementation was reachable.
    """
    inp = _make(1, 2, 1, 16, 32, torch.bfloat16)
    assert _run(inp, _carries(inp)) == CUTE
    ssm, b_prev, u_prev = _carries(inp)
    named = decode_step(
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        inp.C,
        ssm=ssm,
        b_prev=b_prev,
        u_prev=u_prev,
        backend=REFERENCE,
    )
    assert named.backend == REFERENCE
    # float64 is outside KERNEL_DTYPES, so the oracle width is a different
    # implementation rather than this one at another width.
    wide = _make(1, 2, 1, 16, 32, torch.float64)
    assert _run(wide, _carries(wide)) == REFERENCE


def test_one_compiled_variant_serves_repeated_calls() -> None:
    """A second call at one shape compiles nothing.

    A ``@cute.jit`` entry point called directly retraces on every call, which
    dominates a kernel that runs in microseconds. The launcher goes through the
    executor cache, so the count is flat after the first call of a shape.
    """
    inp = _make(1, 2, 1, 16, 32, torch.bfloat16)
    carries = _carries(inp)
    _step(inp, carries)
    held = executor_count()
    _step(inp, carries)
    assert executor_count() == held


@pytest.mark.parametrize("lanes", (16, 32))
def test_chained_steps_match_one_whole_sequence_call(lanes: int) -> None:
    """``STEPS`` single-token calls against the chunked path over the same tokens.

    The carry order is what this catches and a single step cannot: a carry written
    before it is read, or a readout taken before the forcing, is right at one token
    against a reference with the same error and wrong the moment the state is fed
    back in.
    """
    inp = _make(2, 4, 2, 16, lanes, torch.float32, seqlen=STEPS)
    carries = _carries(inp)
    ssm, b_prev, u_prev = carries
    outputs = []
    for step in range(STEPS):
        cut = tuple(one[:, :, step : step + 1].contiguous() for one in inp.args())
        token = inp._replace(U=cut[0], trans=cut[1], K=cut[2], B=cut[3], C=cut[4])
        outputs.append(_step(token, carries))

    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    want = so3ssd_ref(
        inp.U.double(),
        inp.trans.double(),
        inp.K.double(),
        inp.B.double(),
        inp.C.double(),
        CHUNK,
        z0=inp.z0.double(),
        b_prev=inp.b_prev.double(),
        u_prev=inp.u_prev.double(),
    )
    label = _tag("chained", lanes, torch.float32)
    assert_max_rel(torch.cat(outputs, dim=2), want.y, Y_BOUNDS[torch.float32], label)
    assert_max_rel(ssm, want.state, STATE_BOUND, f"{label}/state")
    assert torch.equal(b_prev, want.b_last.to(b_prev.dtype))
    assert torch.equal(u_prev, want.u_last.to(u_prev.dtype))
