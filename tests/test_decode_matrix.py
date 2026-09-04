"""The decode boundary's correctness matrix: the cells no other file reaches.

``tests/test_decode_op.py`` holds the reference boundary, the input contract and the
routing at float64. ``tests/test_cute_decode.py`` holds the kernel against a float64
oracle at one shape family and one chain length. This file holds what neither
reaches, one test per failure mode:

- which implementation answers each registry at each width, since the two
  recurrences do not fall back at the same one;
- float16 and float32 through the routed mixer step, where only float64 and
  bfloat16 had run;
- the fourth cell of the convolution chart, ``d_conv = 1`` with the key
  convolution off;
- launch geometries the kernel sweep skips: a batch that is not a power of two,
  ``G == H``, ``P`` above 32, and a state width past the one the sweep stops at;
- a rotation of exactly half a turn, where the quaternion's scalar part crosses
  zero;
- the shipped sparse oscillator basis as the incoming state, rather than a dense
  draw;
- a chain four times the kernel file's, so a bound covers the horizon it is
  asserted at;
- a whole sequence taken token by token, and taken as a prefill then a decode, at a
  kernel width rather than at float64;
- a graph replayed many times rather than once, two states advanced through one
  mixer, and two captures alive in one process;
- a state saved, reloaded and resumed, and a state rebuilt across a device and a
  dtype;
- activation dtypes that disagree, which the kernel refuses and the reference takes;
- the training path, which a step must not select at either extent.

Every bound below is a measured figure, taken on an RTX A6000 (sm_86) under both
torch 2.6.0 and torch 2.7.1. Every residual agreed to every printed digit across the
two, so the bounds are the arithmetic's and not a version's. Run with
``--tolerance-report`` to print each one beside the error measured under it; no bound
here sits above 2.5x its worst case, and the report is what says so.

The instrument is ``max |got - want|`` over ``max |want|``, so a figure below is a
relative bound and, times the operand scale, an absolute one. Absolute drift at
``(2,4,2,16,32)``, one token then :data:`STEPS` chained, against an oracle whose ``y``
reaches 3.43e+01 at one token and 8.37e+01 over the chain and whose state reaches
5.00e+00 and 1.02e+01:

- ``y``: 6.232e-02 -> 2.491e-01 bfloat16, 7.550e-03 -> 2.864e-02 float16,
  4.546e-06 -> 2.290e-04 float32;
- state: 9.770e-07 -> 1.585e-05, 1.032e-06 -> 1.699e-05, 1.105e-06 -> 1.638e-05.

So the horizon costs a factor of 21 in relative ``y`` error at float32 and a factor of
1.2 at float16, because a 16-bit store rounds away everything the chain accumulated.
That is why the single-step and chained bounds are separate constants: one pair sized
by the chain would leave float32 at a twentieth of its bound at one token, which is a
bound nothing can fail.
"""

from __future__ import annotations

import io
import math
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest
import torch
from torch import Tensor

import slinoss
from slinoss import mixer as mixer_module
from slinoss._precision import KERNEL_DTYPES, SUPPORTED_DTYPES
from slinoss.config import SLinOSSConfig
from slinoss.graph import capture
from slinoss.mixer import SLinOSSMixer
from slinoss.ops.conv import backends as conv_dispatch
from slinoss.ops.decode import backends as decode_dispatch
from slinoss.ops.decode import decode_ref, decode_step
from slinoss.ops.decode.backends import CUTE, REFERENCE
from slinoss.ops.mixer import backends as tail_dispatch
from slinoss.ops.scanprep import backends as prep_dispatch
from slinoss.ops.scanprep import foh_taps
from slinoss.ops.so3ssd import backends as scan_dispatch
from slinoss.ops.so3ssd import so3ssd_ref, so3ssm
from slinoss.state import MixerState, oscillator_basis
from tests.conftest import LS_BIAS, ScanInputs, assert_max_rel, make_inputs

TOKENS = mixer_module.TOKENS
"""The token extent the mixer routes to the one-token boundary.

Read off the module rather than imported from the operator, so a test that moves the
extent to force the other branch restores this value with it.
"""

CHUNK = 16
"""Chunk length every whole-sequence oracle here runs at. The smallest legal one."""

REGISTRIES = (
    ("conv", conv_dispatch),
    ("scanprep", prep_dispatch),
    ("so3ssd", scan_dispatch),
    ("mixer", tail_dispatch),
    ("decode", decode_dispatch),
)
"""Every registry a mixer step resolves against, in call order."""

STEP_Y = {torch.bfloat16: 6e-3, torch.float16: 6e-4, torch.float32: 4e-7}
"""One-token kernel ``y`` against a float64 oracle, per activation width.

``y`` is stored at the activation width and the store is the last operation on it, so
the 16-bit figures are set by that rounding: 2^-9 for bfloat16 and 2^-11 for float16,
against the largest entry rather than each own, which is why the bfloat16 measurement
sits above one half-ulp. The float32 figure is a reduction bound instead, since the
readout sums ``3N`` terms in an order the oracle does not share.

Worst measured over the 23 one-token cells here, at 55%, 47% and 45% of the bound:
3.318e-03 bfloat16, on the sparse oscillator-basis start, whose readout magnitude is
smaller than a dense draw's; 2.823e-04 float16 and 1.817e-07 float32, both at a half
turn. The bfloat16 figure is the tree's shared value; the other two are tightened
here, since the tree's are sized for a chain and these cells are one token.
"""

STEP_STATE = 6e-7
"""One-token kernel state against a float64 oracle, at every activation width.

The state is float32 whatever the activations are, and it is formed from the same cast
operands the oracle reads, so the bound is float32 rounding over about fifteen
operations plus two approximate transcendentals, and it does not vary with the
activation width. Worst measured 2.458e-07, at 41%.
"""

CHAIN_Y = {torch.bfloat16: 6e-3, torch.float16: 8e-4, torch.float32: 4e-6}
"""``y`` over :data:`STEPS` chained kernel steps, against a float64 oracle.

Ten times :data:`STEP_Y` at float32 and unchanged at bfloat16, because the 16-bit
store rounds away what the chain accumulated and float32 does not. Measured
2.978e-03, 3.422e-04 and 2.736e-06, at 50%, 43% and 68%.
"""

CHAIN_STATE = 4e-6
"""The state after :data:`STEPS` chained steps, against a float64 oracle.

Seven times :data:`STEP_STATE`, which is the accumulation measured rather than a
factor assumed: 1.670e-06 worst over the three widths, at 42%.
"""

ROUTED_OUT = {torch.bfloat16: 1.0e-2, torch.float16: 1.2e-3, torch.float32: 6.0e-7}
"""Routed one-token mixer step against the same step through the chunked scan.

Not a bound against an oracle. Every stage but the recurrence is the same code on the
same operands, so the residual is the two recurrences disagreeing at one width and
then passing through the tail. At bfloat16 and float16 the chunked side is itself a
16-bit tensor-core reduction over a padded chunk, which is why the figure is an order
of magnitude above :data:`STEP_Y`; at float32 the chunked side falls back to torch, so
that cell compares the kernel against torch.

Measured 4.673e-03, 5.834e-04, 2.848e-07, at 47%, 49% and 47%.
"""

ROUTED_SSM = {torch.bfloat16: 4.0e-4, torch.float16: 8.0e-5, torch.float32: 5.0e-7}
"""The recurrent state of the same comparison.

Measured 1.980e-04, 3.653e-05, 2.097e-07, at 50%, 46% and 42%.
"""

SEQ_BOUNDS = {torch.bfloat16: 1.2e-2, torch.float16: 1.6e-3, torch.float32: 1.2e-6}
"""A whole sequence taken one token at a time, against one call over all of it.

Wider than :data:`ROUTED_OUT` because the state is rebuilt from its own output at
every token rather than at one. Measured over :data:`TOTAL` tokens: 6.098e-03
bfloat16, 9.498e-04 float16, 6.260e-07 float32, at 51%, 59% and 52%.
"""

PARITY_REL = 1e-13
"""Bound on a float64 comparison. The tree's float64 parity bound, unchanged.

``tests/test_reference.py``, ``tests/test_interface.py`` and
``tests/test_decode_op.py`` all hold this value, and it is deliberately three orders
above what float64 measures: 3.020e-16 here. Tightened to the measurement it would
assert that two float64 reduction orders agree bit for bit, which is not the claim.
The claim is that the two paths evaluate one map, and 1e-13 is the width at which a
sign, an index or a tap that disagrees cannot hide.
"""

# (bsz, heads, groups, rows, lanes). One case per launch geometry the kernel sweep in
# ``tests/test_cute_decode.py`` does not reach, and each case carries at least two of
# them at once so the table stays one test rather than five:
#
# - 3/4/4: a batch that is not a power of two, and G == H, at N % 32 != 0;
# - 5/2/1: batch 5, and P = 48, an odd multiple of HEAD_MULTIPLE, at N % 32 == 0;
# - 2/6/3: six heads over three groups, and P = 64, at N = 48;
# - 1/8/8: G == H at eight heads, one batch, N = 64;
# - 3/2/1: N = 112, seven halves of a warp, at P = 32.
#
# Both row groups appear, which is the split the kernel branches on: N % 32 == 0 takes
# a whole warp per row and N % 32 != 0 takes half of one.
SKIPPED_SHAPES: tuple[tuple[int, int, int, int, int], ...] = (
    (3, 4, 4, 16, 16),
    (5, 2, 1, 48, 32),
    (2, 6, 3, 64, 48),
    (1, 8, 8, 16, 64),
    (3, 2, 1, 32, 112),
)

STEPS = 256
"""Consecutive single-token kernel calls in the chained test.

Sixteen chunks of the whole-sequence oracle: four times what
``tests/test_cute_decode.py`` runs, and the length the reference's own chain runs. The
state is rebuilt from its own output 255 times, and the accumulation is measured
rather than assumed at 1.975e-06 against 1.901e-07 at one token.
"""

TOTAL = 20
PREFILL = 12
"""Sequence length of the partition test, and where its one handover falls.

``PREFILL`` sits inside the first chunk and ``TOTAL - PREFILL`` carries the decode
past the chunk boundary at :data:`CHUNK`, so the handover is aligned to neither.
"""

MIXER_CONFIG = SLinOSSConfig(
    d_model=32, d_state=48, d_head=16, n_groups=2, chunk_size=CHUNK, bias=True
)
"""Four heads over two groups at the smallest legal state and head widths."""

BATCH = 2

TWO_CAPTURES = """
import sys
import torch
from slinoss.config import SLinOSSConfig
from slinoss.graph import capture_decode
from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

vocab, first_batch, second_batch, share = {vocab}, {first}, {second}, {share}
config = SLinOSSConfig(
    d_model=32, d_state=48, d_head=16, n_groups=2, chunk_size=16,
    n_layers=2, ffn_ratio=2.0, vocab_size=vocab,
)
device = torch.device("cuda")
torch.manual_seed(0)
stack = SLinOSSStack(config, device=device).to(torch.bfloat16)
one = StackState.allocate(config, first_batch, device=device, dtype=torch.bfloat16)
two = StackState.allocate(config, second_batch, device=device, dtype=torch.bfloat16)
stack(torch.randint(0, vocab, (first_batch, 5), device=device), one)
stack(torch.randint(0, vocab, (second_batch, 5), device=device), two)
first_step = capture_decode(stack, one)
second_step = capture_decode(stack, two, share=first_step if share else None)
logits = (
    first_step(torch.randint(0, vocab, (first_batch, 1), device=device)).clone(),
    second_step(torch.randint(0, vocab, (second_batch, 1), device=device)).clone(),
    first_step(torch.randint(0, vocab, (first_batch, 1), device=device)).clone(),
)
torch.cuda.synchronize()
if not all(bool(torch.isfinite(entry).all()) for entry in logits):
    sys.exit("a replay returned non-finite logits")
print("two captures replayed")
"""
"""A stack, two states, two captures, three replays, in a process of its own.

Text rather than a helper because the case under test is what a second capture does
to the process. A device-side assertion poisons the CUDA context, so every later test
in a process that hit one is untrustworthy whatever it reports; running this
elsewhere is what keeps the rest of the file meaningful.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _width(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` as ``bfloat16``, for a parameter id and a tolerance label."""
    return str(dtype).removeprefix("torch.")


def _cuda() -> torch.device:
    """The visible CUDA device, or a skip."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


def _kernel() -> torch.device:
    """The visible CUDA device with the decode kernel registered, or a skip.

    A tree whose CuTe DSL is missing registers the reference alone and answers every
    call from it. Skipping is the only honest outcome: the alternative is a matrix
    that passes while exercising a program nobody meant to certify.
    """
    device = _cuda()
    if CUTE not in decode_dispatch.names():
        pytest.skip("the decode CuTe backend did not register")
    return device


def _report() -> None:
    """Print which implementation answers every stage of a step, at every width.

    Not an assertion. Dispatch falls back silently, so a matrix that cannot name the
    program it ran certified whichever program happened to be reachable.
    """
    print(f"\ntorch {torch.__version__}")
    print(f"slinoss {slinoss.__file__}")
    print(f"cuda available {torch.cuda.is_available()}")
    for label, registry in REGISTRIES:
        print(f"{label:9s} names={registry.names()}")
        for device_type in ("cpu", "cuda"):
            for dtype in SUPPORTED_DTYPES:
                try:
                    answer = registry.resolve(None, device_type, dtype).name
                except ValueError as exc:
                    answer = f"raise: {exc}"
                print(f"    {device_type:4s} {dtype!s:16s} -> {answer}")


def _make(
    bsz: int,
    heads: int,
    groups: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
    *,
    seqlen: int = TOKENS,
    seed: int = 0,
) -> ScanInputs:
    """One kernel operand set: float32 pinned tensors, ``dtype`` activations, on CUDA.

    ``LS_BIAS`` keeps the decay above float32 epsilon over a chained run. Unbiased,
    the state contributes nothing after a few tokens and the rotation is then tested
    on a zero.
    """
    return make_inputs(
        bsz=bsz,
        heads=heads,
        groups=groups,
        seqlen=seqlen,
        rows=rows,
        lanes=lanes,
        dtype=torch.float32,
        device="cuda",
        seed=seed,
        w_scale=2.0,
        ls_bias=LS_BIAS,
        u_dtype=dtype,
        bc_dtype=dtype,
    )


def _carries(inp: ScanInputs) -> tuple[Tensor, Tensor, Tensor]:
    """Fresh copies of the three in-place carries, in signature order."""
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    return inp.z0.clone(), inp.b_prev.clone(), inp.u_prev.clone()


def _kw(inp: ScanInputs) -> dict[str, Tensor]:
    """The three carries as keyword arguments, cloned.

    Cloned so one operand set reaches two calls: the boundary advances all three in
    place, so a second call over the originals would start from what the first left.
    """
    ssm, b_prev, u_prev = _carries(inp)
    return {"ssm": ssm, "b_prev": b_prev, "u_prev": u_prev}


def _oracle(
    inp: ScanInputs, carried: tuple[Tensor, Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    """``decode_ref`` on a float64 upcast of the same cast operands.

    Args:
        inp: The operand set the kernel is given.
        carried: The three carries the kernel is given, before it advances them.

    Returns:
        ``(y, state)`` at float64. The state is the buffer the reference advanced.
    """
    ssm, b_prev, u_prev = (one.double().clone().contiguous() for one in carried)
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


def _step(inp: ScanInputs, carried: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """One call to the boundary, asserting the kernel is what answered it."""
    ssm, b_prev, u_prev = carried
    out = decode_step(
        inp.U, inp.trans, inp.K, inp.B, inp.C, ssm=ssm, b_prev=b_prev, u_prev=u_prev
    )
    assert out.backend == CUTE, f"{out.backend} answered, so this measures torch"
    return out.y


def _check(inp: ScanInputs, dtype: torch.dtype, label: str) -> None:
    """Run the kernel and the float64 oracle on one operand set and bound the gap."""
    carried = _carries(inp)
    want_y, want_state = _oracle(inp, carried)
    got_y = _step(inp, carried)
    tag = f"decode-matrix[{label}/{_width(dtype)}]"
    assert_max_rel(got_y, want_y, STEP_Y[dtype], f"{tag}/y")
    assert_max_rel(carried[0], want_state, STEP_STATE, f"{tag}/state")


def _rotate_by(inp: ScanInputs, angle: float) -> ScanInputs:
    """Replace the transition with a rotation of exactly ``angle`` radians per token.

    The chart is the rotation vector itself, bounded to ``|w| <= 2*w_max``, so an
    exact angle is a rescale of the axis rather than a raw value to solve for. The
    decay is the operand set's own, so only the rotation moves.

    Args:
        inp: The operand set, whose ``trans`` supplies the axis and the decay.
        angle: Rotation angle per token, radians.

    Returns:
        The same operand set with ``trans`` and ``K`` replaced.
    """
    rotvec = inp.trans[..., :3]
    axis = rotvec / rotvec.norm(dim=-1, keepdim=True)
    logscale = inp.trans[..., 3]
    turned = axis * angle
    tap = foh_taps(turned, logscale)
    return inp._replace(
        trans=torch.cat([turned, logscale[..., None]], dim=-1).contiguous(),
        K=torch.cat([tap, torch.zeros_like(tap[..., :1])], dim=-1).contiguous(),
    )


def _mixer(
    d_conv: int,
    key_conv: bool,
    dtype: torch.dtype,
    *,
    n_groups: int = MIXER_CONFIG.n_groups,
    seed: int = 0,
) -> SLinOSSMixer:
    """A seeded mixer on CUDA at ``dtype``, or a skip.

    CUDA only: both vector operands of a mixer step are pitched bands of the fused
    projection, and a pitched band is refused off a CUDA device.

    Initialization zeroes the output projection and the projection's forcing band, so
    an untouched mixer returns zeros and forces the recurrence with zeros. Both are
    drawn here, after construction, so the lattice the parameter band carries stays
    intact and no comparison of two step outputs holds on zeros.

    Args:
        d_conv: Convolution width.
        key_conv: Whether the key convolution exists.
        dtype: Activation dtype. The module is cast, not autocast.
        n_groups: ``G``.
        seed: Draw seed.

    Returns:
        The mixer, in evaluation-independent state: no mode is set here.
    """
    _cuda()
    config = replace(MIXER_CONFIG, d_conv=d_conv, key_conv=key_conv, n_groups=n_groups)
    torch.manual_seed(seed)
    mixer = SLinOSSMixer(config, device=torch.device("cuda")).to(dtype)
    with torch.no_grad():
        mixer.out_proj.weight.normal_(std=0.05)
        mixer.in_proj.weight[mixer.layout.b_off : mixer.layout.c_off].normal_(std=0.05)
    return mixer


def _allocate(config: SLinOSSConfig, batch: int, dtype: torch.dtype) -> MixerState:
    """A decode state on the visible CUDA device."""
    return MixerState.allocate(config, batch, device=torch.device("cuda"), dtype=dtype)


def _tokens(
    config: SLinOSSConfig, batch: int, tokens: int, dtype: torch.dtype
) -> Tensor:
    """``(batch, tokens, d_model)`` activations on the visible CUDA device."""
    return torch.randn(
        batch, tokens, config.d_model, dtype=dtype, device=torch.device("cuda")
    )


def _watch_the_boundary(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the backend that answered every call to the boundary, and forward it."""
    seen: list[str] = []
    forward = mixer_module.decode_step

    def watch(*operands: Tensor, ssm: Tensor, b_prev: Tensor, u_prev: Tensor) -> object:
        out = forward(*operands, ssm=ssm, b_prev=b_prev, u_prev=u_prev)
        seen.append(out.backend)
        return out

    monkeypatch.setattr(mixer_module, "decode_step", watch)
    return seen


def _buffers(state: MixerState) -> dict[str, Tensor]:
    """Every carry, keyed by field name, read off the dataclass.

    Never a written list. A hand-written list is free to drop the same buffer the code
    it checks drops, which is how a graph restore that never copied ``keys`` passed
    its own suite.
    """
    return {carry.name: getattr(state, carry.name) for carry in fields(state)}


# ---------------------------------------------------------------------------
# Which implementation answers
# ---------------------------------------------------------------------------


def test_the_reference_answers_every_width_on_a_cpu_device() -> None:
    """Catches a kernel backend that claims a device it cannot launch on.

    Every registry a step resolves against must answer with a torch implementation on
    CPU at every supported width, including the two the kernels are built for. A
    backend registered over ``("cpu", "cuda")`` by mistake would be selected here and
    fail at launch, and the reference boundary is what makes a CPU test of the map
    possible at all.

    Prints the whole dispatch table on the way through: which implementation answers
    every stage at every device and width, the torch version, and the tree
    :mod:`slinoss` was imported from. Dispatch falls back silently, so this is what
    states which program the rest of the file certified.
    """
    _report()
    for label, registry in REGISTRIES:
        for dtype in SUPPORTED_DTYPES:
            answer = registry.resolve(None, "cpu", dtype).name
            assert answer == REFERENCE, f"{label} answers {answer} on cpu at {dtype}"


@pytest.mark.cuda
@pytest.mark.cute
def test_the_two_recurrences_do_not_fall_back_at_the_same_width() -> None:
    """Catches a fallback boundary assumed to be shared by both recurrences.

    The one-token boundary registers over :data:`slinoss._precision.KERNEL_DTYPES`,
    which holds float32, and the chunked scan registers over a low-precision set that
    does not: its atom is a 16-bit tensor core. So at float32 on CUDA a decode step
    runs a kernel while a prefill of the same tokens runs torch, and at float64 both
    run torch.

    Asserted rather than described because the asymmetry decides what a measurement
    means. A float32 decode figure attributed to the reference is a kernel figure, and
    a float32 prefill figure attributed to the kernel is torch's. It also decides what
    an oracle is: the float64 oracle is a different implementation, not this one at a
    wider width.
    """
    _kernel()
    for dtype in KERNEL_DTYPES:
        assert decode_dispatch.resolve(None, "cuda", dtype).name == CUTE, dtype
    assert decode_dispatch.resolve(None, "cuda", torch.float64).name == REFERENCE
    assert scan_dispatch.resolve(None, "cuda", torch.float32).name == REFERENCE
    scan_kernel = scan_dispatch.resolve(None, "cuda", torch.bfloat16)
    assert scan_kernel.name == CUTE
    assert torch.float32 not in scan_kernel.dtypes
    assert torch.float32 in decode_dispatch.resolve(None, "cuda", torch.float32).dtypes


# ---------------------------------------------------------------------------
# The map at a half turn
# ---------------------------------------------------------------------------


def test_the_reference_is_the_sequential_map_at_a_half_turn() -> None:
    """Catches a rotation that is wrong only where the quaternion's scalar part is 0.

    ``|w| = pi`` is a half turn: ``cos(|w|/2)`` is zero, the quaternion is purely
    imaginary, and a conjugation written with a division by the scalar part, or with
    an axis recovered from it, is finite everywhere else and wrong here. The chart
    reaches a half turn as an interior point, since ``|w| <= 2*w_max`` is 6 radians at
    the shipped default, so this is a reachable transition rather than a limit.

    The regime sweeps elsewhere in the tree stop at 1e-3 on one side and run to the
    cap of 6 radians on the other, which straddles a half turn without landing on it.

    float64 on CPU against :func:`slinoss.ops.so3ssd.so3ssm`, which defines
    correctness, and bitwise: the two evaluate one token of the same map from the same
    state, so any difference at all is a difference in the map.
    """
    inp = make_inputs(
        bsz=BATCH,
        heads=4,
        groups=2,
        seqlen=TOKENS,
        rows=16,
        lanes=16,
        dtype=torch.float64,
        device="cpu",
        seed=3,
        ls_bias=LS_BIAS,
    )
    for label, angle in (("half-turn", math.pi), ("cap", 6.0), ("near-identity", 1e-3)):
        turned = _rotate_by(inp, angle)
        reached = float(turned.trans[..., :3].norm(dim=-1).min())
        assert reached == pytest.approx(angle, rel=1e-12), label
        carried = _kw(turned)
        got = decode_ref(*turned.args(), **carried)
        want = so3ssm(*turned.args(), **turned.kw())
        assert torch.equal(got, want.y), f"{label}: y is not the sequential map"
        assert torch.equal(carried["ssm"], want.state), f"{label}: state disagrees"


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("dtype", KERNEL_DTYPES, ids=_width)
def test_the_kernel_matches_the_oracle_at_a_half_turn(dtype: torch.dtype) -> None:
    """Catches a half-angle series evaluated where its argument is furthest from zero.

    The kernel builds the quaternion from a float32 series in ``|w|/2``, and a half
    turn is the largest argument a canonical rotation reaches. Both sides of it run as
    well, since a series that lost a term is smooth across the point while a branch on
    the sign of the scalar part is not.
    """
    _kernel()
    inp = _make(BATCH, 4, 2, 16, 32, dtype)
    for label, angle in (
        ("half-turn", math.pi),
        ("under-half-turn", math.pi - 1e-3),
        ("over-half-turn", math.pi + 1e-3),
    ):
        _check(_rotate_by(inp, angle), dtype, label)


# ---------------------------------------------------------------------------
# Launch geometries and state widths the kernel sweep skips
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("shape", SKIPPED_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32), ids=_width)
def test_the_kernel_matches_the_oracle_at_the_shapes_the_sweep_skips(
    shape: tuple[int, int, int, int, int], dtype: torch.dtype
) -> None:
    """Catches a launch geometry that works only at the shapes a sweep happened to use.

    The grid is ``(bsz, heads, rows / rows_per_block)`` over a row group of 16 or 32
    lanes, and the vector operands are addressed by ``heads // groups``. So a batch
    that is not a power of two, ``G == H`` where that divisor is one, and a ``P`` above
    the two the sweep runs are each a different index arithmetic, and each is a shape a
    decode driver reaches with no warning.
    """
    _kernel()
    _check(_make(*shape, dtype), dtype, "x".join(map(str, shape)))


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("lanes", (144, 256), ids=("3N=432", "3N=768"))
def test_the_kernel_matches_the_oracle_past_the_state_width_it_is_swept_to(
    lanes: int,
) -> None:
    """Catches a state width that is legal by the shape rules and unhandled anyway.

    ``d_state`` is legal at every positive multiple of 48, and neither
    :class:`slinoss.config.SLinOSSConfig`, the shared operand check, nor the kernel
    entry point caps it. The kernel sweep stops at ``3N = 384``, so every wider legal
    width is an unexercised lane walk: nine 3-vectors per thread over a half-warp group
    at ``3N = 432``, sixteen over a whole warp at ``3N = 768``.

    Correct out to ``3N = 2304`` when measured, so an unsupported width is not what is
    at stake here. An unexercised one is.
    """
    _kernel()
    _check(_make(1, 2, 1, 16, lanes, torch.bfloat16), torch.bfloat16, f"3N={3 * lanes}")


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32), ids=_width)
def test_the_shipped_initial_state_matches_the_oracle(dtype: torch.dtype) -> None:
    """Catches a state read that is wrong on the state a decode actually starts from.

    :meth:`slinoss.state.MixerState.allocate` starts the recurrence from
    :func:`slinoss.state.oscillator_basis`: one unit coordinate per row, at
    ``(h*P + p) mod 3N``, and zeros everywhere else. Every other kernel test hands it a
    dense draw, where a lane index off by a permutation still reads a plausible number.
    Against a sparse start it reads an exact zero where the carrier belongs and the
    readout of that row loses its whole homogeneous term.

    Run at the shape a mixer supplies, so the basis is the one a step would see rather
    than a sparse tensor of the same shape.
    """
    _kernel()
    config = SLinOSSConfig(
        d_model=64, d_state=48, d_head=16, n_groups=2, chunk_size=CHUNK
    )
    inp = _make(
        BATCH, config.n_heads, config.n_groups, config.d_head, config.n_lanes, dtype
    )
    basis = oscillator_basis(config, device=torch.device("cuda"))
    start = basis.unsqueeze(0).expand(BATCH, -1, -1, -1).clone()
    assert int(start.count_nonzero()) == BATCH * config.n_heads * config.d_head
    _check(inp._replace(z0=start), dtype, "oscillator-basis")


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("dtype", KERNEL_DTYPES, ids=_width)
def test_a_long_chain_of_kernel_steps_stays_inside_its_measured_drift(
    dtype: torch.dtype,
) -> None:
    """Catches a one-step tolerance quoted as a horizon tolerance.

    Chained residual is about ten times single-step residual on the state, so a bound
    justified at one token says nothing about a hundred. :data:`STEPS` is the horizon
    the bounds in this file are measured at, four times the chain the kernel file runs.

    Carry order is what a chain catches and a single step cannot: a carry written
    before it is read, or a readout taken before the forcing, is right at one token
    against a reference that shares the error and wrong the moment the state is fed
    back in. Ground truth is the whole-sequence path over the same tokens at float64,
    which reassociates the recurrence completely differently.
    """
    _kernel()
    inp = _make(BATCH, 4, 2, 16, 32, dtype, seqlen=STEPS)
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    carried = _carries(inp)
    outputs = []
    for index in range(STEPS):
        cut = tuple(one[:, :, index : index + 1].contiguous() for one in inp.args())
        token = inp._replace(U=cut[0], trans=cut[1], K=cut[2], B=cut[3], C=cut[4])
        outputs.append(_step(token, carried))

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
    tag = f"decode-matrix[chain-{STEPS}/{_width(dtype)}]"
    assert_max_rel(torch.cat(outputs, dim=2), want.y, CHAIN_Y[dtype], f"{tag}/y")
    assert_max_rel(carried[0], want.state, CHAIN_STATE, f"{tag}/state")
    # The two tap carries hold the last token's operands rather than a reduction, so a
    # downcast of the float64 oracle's own is exact and the comparison is bitwise.
    assert torch.equal(carried[1], want.b_last.to(carried[1].dtype)), "b_prev"
    assert torch.equal(carried[2], want.u_last.to(carried[2].dtype)), "u_prev"


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("dtype", KERNEL_DTYPES, ids=_width)
def test_every_partition_of_a_sequence_gives_the_whole_sequence(
    dtype: torch.dtype,
) -> None:
    """Catches a decode continuing a prefill from the wrong carry, at a kernel width.

    Three partitions of one sequence through one mixer: all of it in one call, all of
    it one token at a time, and a prefill followed by a decode that crosses a chunk
    boundary. The stepwise arm is the routed one-token boundary at every token; the
    split arm crosses from the chunked recurrence into the boundary exactly once,
    which is the handover a decode driver makes once per request and where a carry the
    chunked path left in the wrong place surfaces.

    At a kernel width rather than at float64. The float64 splits elsewhere in the tree
    run the reference on both sides of the handover; here the prefill runs the chunked
    kernel at a 16-bit width and the decode runs the one-token kernel, so the handover
    is between two implementations rather than inside one.
    """
    _kernel()
    mixer = _mixer(4, True, dtype)
    config = mixer.config
    x = _tokens(config, BATCH, TOTAL, dtype)

    with torch.no_grad():
        whole = mixer(x)

    stepwise_state = _allocate(config, BATCH, dtype)
    stepwise = torch.cat(
        [mixer.step(x[:, one : one + 1], stepwise_state) for one in range(TOTAL)], dim=1
    )

    split_state = _allocate(config, BATCH, dtype)
    head = mixer.step(x[:, :PREFILL], split_state)
    tail = [
        mixer.step(x[:, one : one + 1], split_state) for one in range(PREFILL, TOTAL)
    ]
    split = torch.cat([head, *tail], dim=1)

    tag = f"decode-matrix[partition/{_width(dtype)}]"
    assert_max_rel(stepwise, whole, SEQ_BOUNDS[dtype], f"{tag}/stepwise")
    assert_max_rel(split, whole, SEQ_BOUNDS[dtype], f"{tag}/prefill-then-decode")


# ---------------------------------------------------------------------------
# The routed step: widths, the convolution chart, and the carry asymmetry
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize("dtype", KERNEL_DTYPES, ids=_width)
def test_every_activation_width_routes_the_kernel_and_carries_what_the_scan_carries(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    """Catches a width that reaches the boundary and is answered by the reference.

    Every kernel width through the whole routed step, against the same step through
    the chunked scan over the same operands. The chunked arm is reached by moving the
    extent the branch tests off one, so nothing else about the call changes and the
    comparison is the two recurrences on one set of operands at one width.

    The backend name is asserted, not assumed. A registry whose kernel import failed
    answers every call from the reference and this comparison passes anyway: both arms
    would be torch, agreeing with each other about a program nobody measured.

    All five carries, not the three the recurrence advances. Both convolution windows
    are copied on either path, and a routing that dropped one of those copies is a
    state that stops advancing while every number here still agrees.
    """
    _kernel()
    mixer = _mixer(4, True, dtype)
    config = mixer.config
    x = _tokens(config, BATCH, TOKENS, dtype)

    chunked = _allocate(config, BATCH, dtype)
    monkeypatch.setattr(mixer_module, "TOKENS", 0)
    want = mixer.step(x, chunked)
    monkeypatch.undo()

    seen = _watch_the_boundary(monkeypatch)
    routed = _allocate(config, BATCH, dtype)
    got = mixer.step(x, routed)

    assert seen == [CUTE], f"the boundary answered from {seen}"
    tag = f"decode-matrix[routed/{_width(dtype)}]"
    assert_max_rel(got, want, ROUTED_OUT[dtype], f"{tag}/out")
    assert_max_rel(routed.ssm, chunked.ssm, ROUTED_SSM[dtype], f"{tag}/ssm")
    for name in ("conv", "keys", "b_prev", "u_prev"):
        assert torch.equal(getattr(routed, name), getattr(chunked, name)), name


@pytest.mark.cuda
def test_a_width_one_convolution_with_no_key_convolution_routes_and_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the one cell of the convolution chart the routing tests leave out.

    ``d_conv`` and ``key_conv`` are two independent switches and three of their four
    settings are exercised. The fourth, width one with the key convolution off, is the
    only one where both history buffers are zero length and neither is written: the
    value convolution degenerates to a pointwise multiply, and ``state.keys`` takes no
    copy because there is no key convolution to produce one. A routing that indexed a
    window relative to the end of a history would index an empty tensor here and
    nowhere else.

    float64 against the same step through the chunked scan, so the comparison holds to
    the tree's parity bound rather than to a width.
    """
    dtype = torch.float64
    mixer = _mixer(1, False, dtype)
    config = mixer.config
    x = _tokens(config, BATCH, TOKENS, dtype)

    chunked = _allocate(config, BATCH, dtype)
    assert chunked.conv.shape[1] == 0 and chunked.keys.shape[1] == 0
    empty = {name: buf.clone() for name, buf in _buffers(chunked).items()}
    monkeypatch.setattr(mixer_module, "TOKENS", 0)
    want = mixer.step(x, chunked)
    monkeypatch.undo()

    routed = _allocate(config, BATCH, dtype)
    got = mixer.step(x, routed)

    assert_max_rel(got, want, PARITY_REL, "decode-matrix[width-1-no-keys]/out")
    assert_max_rel(routed.ssm, chunked.ssm, PARITY_REL, "decode-matrix[width-1]/ssm")
    for name in ("conv", "keys", "b_prev", "u_prev"):
        assert torch.equal(getattr(routed, name), getattr(chunked, name)), name
    # A zero-length buffer is the one carry a step cannot advance, so a step that
    # resized one rather than writing into it would agree with a second state that
    # resized it the same way.
    assert torch.equal(routed.conv, empty["conv"])
    assert torch.equal(routed.keys, empty["keys"])


@pytest.mark.cuda
@pytest.mark.parametrize("tokens", (TOKENS, CHUNK + 1), ids=("one-token", "prefill"))
def test_a_step_rebinds_none_of_the_five_carries(tokens: int) -> None:
    """Catches a rebound carry at either extent, over the whole buffer set.

    A captured graph records addresses, so a field rebound rather than written leaves
    replay reading and writing memory no consumer sees. The three the recurrence
    advances are checked elsewhere; the set enumerated from the dataclass is what
    covers the two convolution windows as well, and what cannot drop a buffer the
    container later grows.
    """
    dtype = torch.float64
    mixer = _mixer(4, True, dtype)
    config = mixer.config
    state = _allocate(config, BATCH, dtype)
    before = {name: buf.data_ptr() for name, buf in _buffers(state).items()}

    mixer.step(_tokens(config, BATCH, tokens, dtype), state)

    after = {name: buf.data_ptr() for name, buf in _buffers(state).items()}
    assert after == before
    assert set(before) == {"conv", "keys", "ssm", "b_prev", "u_prev"}


@pytest.mark.cuda
@torch.no_grad()
def test_the_two_kept_carry_copies_are_the_two_that_are_not_redundant() -> None:
    """Catches a carry copy deleted on the wrong side of the routed step's asymmetry.

    Three of the five copies the chunked path ends with are redundant at one token and
    two are not, and which is which follows from who owns the buffer each stage writes.
    The boundary writes ``ssm``, ``b_prev`` and ``u_prev`` in the caller's own storage,
    so a copy after it reads and writes a buffer onto itself. Both convolutions return
    a fresh window instead, so their copies are the only way that history advances.

    Both halves, by address. Asserting only that the three are written in place invites
    the symmetric deletion of the other two, which is a state whose convolution
    history silently stops moving while every recurrence number stays right.

    Under ``no_grad``, because this rebuilds the stages the mixer's own step runs and
    that step is itself a ``no_grad`` region: the parameters carry gradients, so the
    projection and the convolutions would hand the boundary operands it refuses. The
    two convolution backends do not refuse identically. The kernel is a custom op and
    hands back a window carrying no gradient, so the boundary's refusal is reached
    only through the reference path, which is to say only in a tree whose extension
    did not load; without the ``no_grad`` this test reads the build, not the copies.
    """
    dtype = torch.bfloat16
    mixer = _mixer(4, True, dtype)
    config = mixer.config
    state = _allocate(config, BATCH, dtype)
    proj = torch.nn.functional.linear(
        _tokens(config, BATCH, TOKENS, dtype), mixer.in_proj.weight, mixer.in_proj.bias
    )
    conv_backend = conv_dispatch.resolve(None, "cuda", dtype)
    conv = conv_backend.forward(
        mixer.layout.value(proj),
        mixer.conv_weight.to(dtype),
        None if mixer.conv_bias is None else mixer.conv_bias.to(dtype),
        activation=True,
        initial_state=state.conv,
        d_head=config.d_head,
    )
    assert mixer.key_weight is not None
    keys = conv_backend.forward(
        mixer.layout.keys(proj),
        mixer.key_weight.to(dtype),
        None,
        activation=False,
        initial_state=state.keys,
    )
    assert conv.state.data_ptr() != state.conv.data_ptr(), (
        "the value convolution advanced state.conv in place, so its copy is redundant"
    )
    assert keys.state.data_ptr() != state.keys.data_ptr(), (
        "the key convolution advanced state.keys in place, so its copy is redundant"
    )

    params = prep_dispatch.resolve(None, "cuda", dtype).forward(
        mixer.layout.params(proj),
        mixer.transition_bias,
        heads=config.n_heads,
        w_max=config.w_max,
    )
    pointers = (state.ssm.data_ptr(), state.b_prev.data_ptr(), state.u_prev.data_ptr())
    stale = state.ssm.clone()
    out = decode_step(
        conv.y,
        params.trans,
        params.K,
        mixer.layout.key_b(keys.y),
        mixer.layout.key_c(keys.y),
        ssm=state.ssm,
        b_prev=state.b_prev,
        u_prev=state.u_prev,
    )
    assert out.y.shape == (BATCH, config.n_heads, TOKENS, config.d_head)
    now = (state.ssm.data_ptr(), state.b_prev.data_ptr(), state.u_prev.data_ptr())
    assert now == pointers, "the boundary rebound a carry, so a copy is required"
    assert not torch.equal(state.ssm, stale), "the boundary did not advance the state"


# ---------------------------------------------------------------------------
# Graphs, and more than one state
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.cute
def test_a_captured_step_replayed_many_times_is_the_step_taken_many_times() -> None:
    """Catches a replay that is right once and wrong from the second call on.

    Every graph test in the tree replays exactly once, and one replay is the count that
    cannot see a carry the graph read as a constant folded in at capture: the first
    replay starts from what the capture left, so it agrees, and the second starts from
    what the first replay left, so it does not.

    Eight rounds, bitwise, against eight eager steps from a copy of the carries the
    capture left, and all five carries after them.
    """
    _kernel()
    dtype = torch.bfloat16
    mixer = _mixer(4, True, dtype)
    config = mixer.config
    state = _allocate(config, BATCH, dtype)
    x = _tokens(config, BATCH, TOKENS, dtype)

    step = capture(lambda ids: mixer.step(ids, state), x)

    eager_state = state.clone()
    rounds = 8
    eager = [mixer.step(x, eager_state).clone() for _ in range(rounds)]
    replayed = [step(x).clone() for _ in range(rounds)]

    for index, (got, want) in enumerate(zip(replayed, eager, strict=True)):
        assert torch.equal(got, want), f"replay {index} is not the step"
    for name, buffer in _buffers(state).items():
        assert torch.equal(buffer, _buffers(eager_state)[name]), name


@pytest.mark.cuda
def test_two_states_advance_through_one_mixer_without_touching_each_other() -> None:
    """Catches per-step state held anywhere but in the state the caller passed in.

    A server steps many requests through one module, at different batches and
    different positions. Any buffer cached on the module -- a convolution window, a
    previous forcing vector, a resolved shape -- makes the second request's step depend
    on the first's, and a single-state test cannot see it because there is nothing to
    interfere with.

    Two states at two batches, stepped alternately, against each stepped alone.
    Bitwise: one kernel sequence over one shape agrees to the bit, so a tolerance here
    would only admit interference.
    """
    dtype = torch.bfloat16
    mixer = _mixer(4, True, dtype)
    config = mixer.config
    rounds = 4
    small = _tokens(config, 2, TOKENS, dtype)
    large = _tokens(config, 3, TOKENS, dtype)

    def alone(x: Tensor) -> list[Tensor]:
        state = _allocate(config, int(x.shape[0]), dtype)
        return [mixer.step(x, state).clone() for _ in range(rounds)]

    solo_small, solo_large = alone(small), alone(large)

    state_small = _allocate(config, 2, dtype)
    state_large = _allocate(config, 3, dtype)
    for index in range(rounds):
        got_small = mixer.step(small, state_small)
        got_large = mixer.step(large, state_large)
        assert torch.equal(got_small, solo_small[index]), f"round {index}, batch 2"
        assert torch.equal(got_large, solo_large[index]), f"round {index}, batch 3"


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize(
    ("vocab", "first", "second", "share"),
    [(17, 2, 2, False), (17, 2, 3, True), (50257, 8, 8, False)],
    ids=["same-batch", "shared-pool", "wide-vocab"],
)
def test_two_decode_captures_coexist_in_one_process(
    vocab: int, first: int, second: int, share: bool
) -> None:
    """Catches a second capture that invalidates the first, or the device.

    Two graphs is what a driver with two request slots records, and a second capture
    has been reported to fire a device-side ``indexSelectSmallIndex: srcIndex <
    srcSelectDimSize`` at some shapes. Both replays run, and the first replays again
    after the second capture, since a pool the second capture took over is visible only
    in the first step's next replay.

    In a subprocess. A device-side assertion poisons the CUDA context, so a failure
    here would otherwise make every later test in this process report on a device that
    is already gone; the subprocess also puts the device's own text in the failure.
    """
    _kernel()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [root, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            TWO_CAPTURES.format(vocab=vocab, first=first, second=second, share=share),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        check=False,
    )
    assert done.returncode == 0, (
        f"two captures at vocab={vocab} batches=({first},{second}) share={share} "
        f"exited {done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    assert "two captures replayed" in done.stdout, done.stdout


# ---------------------------------------------------------------------------
# The state container across a save, a device and a dtype
# ---------------------------------------------------------------------------


def test_a_saved_state_reloads_as_the_state_it_was() -> None:
    """Catches a checkpoint of a decode state that drops or reorders a buffer.

    A paused request is a state written out and read back, and the buffer set is
    enumerated from the dataclass on both sides so neither list can drift from the
    container. A payload missing one buffer must not construct: a container that
    defaulted a missing carry would resume from a zero window and produce a
    continuation that is silently not the sequence it was cut from.

    On CPU. This is the container's invariant, not a device's.
    """
    config = replace(MIXER_CONFIG, d_conv=4, key_conv=True)
    state = MixerState.allocate(config, BATCH, device="cpu", dtype=torch.float32)
    for index, buffer in enumerate(_buffers(state).values(), start=1):
        buffer.fill_(float(index))

    payload = {name: buffer.clone() for name, buffer in _buffers(state).items()}
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    revived = MixerState(**torch.load(stream, weights_only=True))

    for name, buffer in _buffers(revived).items():
        original = _buffers(state)[name]
        assert torch.equal(buffer, original), name
        assert buffer.dtype is original.dtype, name
        assert buffer.shape == original.shape, name
    for dropped in payload:
        with pytest.raises(TypeError, match=dropped):
            MixerState(**{k: v for k, v in payload.items() if k != dropped})


@pytest.mark.cuda
def test_a_state_moved_across_a_device_and_a_dtype_keeps_the_pinning_rule() -> None:
    """Catches a move that carries ``ssm`` along with the activation dtype.

    Neither state container has a ``to``, so a move is a rebuild from the dataclass
    fields, and the one rule a blanket cast over those fields breaks is the float32
    pinning of ``ssm``: rotation error enters the transform squared, so a 16-bit
    recurrent state is a correctness defect rather than a trade. The container refuses
    it, and this is the test that says so at the place a move gets written.

    A half-moved container is refused as well, since the alternative is a step reading
    four buffers off one device and one off another.
    """
    device = _cuda()
    config = replace(MIXER_CONFIG, d_conv=4, key_conv=True)
    host = MixerState.allocate(config, BATCH, device="cpu", dtype=torch.bfloat16)
    for buffer in _buffers(host).values():
        buffer.normal_()

    moved = MixerState(**{name: buf.to(device) for name, buf in _buffers(host).items()})
    assert moved.device == torch.device("cuda", torch.cuda.current_device())
    assert moved.ssm.dtype is torch.float32
    for name, buffer in _buffers(moved).items():
        assert torch.equal(buffer.cpu(), _buffers(host)[name]), name

    recast = MixerState(
        **{
            name: buf if name == "ssm" else buf.to(torch.float16)
            for name, buf in _buffers(moved).items()
        }
    )
    assert recast.conv.dtype is torch.float16
    assert recast.ssm.dtype is torch.float32

    with pytest.raises(TypeError, match="float32-pinned"):
        MixerState(
            **{name: buf.to(torch.bfloat16) for name, buf in _buffers(moved).items()}
        )
    with pytest.raises(ValueError, match="one device only"):
        MixerState(
            **{
                name: buf if name == "conv" else buf.cpu()
                for name, buf in _buffers(moved).items()
            }
        )


# ---------------------------------------------------------------------------
# What a step refuses, and what it must not select
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.cute
def test_activation_widths_that_disagree_are_refused_by_the_kernel_alone() -> None:
    """Catches a kernel that widens two element types from one load path.

    The shared operand check pairs each carry with its own operand and leaves ``U`` and
    ``B`` free to disagree, so the reference takes a bfloat16 ``U`` beside a float16
    ``B``. The kernel cannot: it widens on load from one element type per tensor, and
    the pair is not worth two widening types. So the refusal is the kernel's own, and
    the fallback is not silent -- an explicitly named reference still runs the call.

    Both halves are asserted. A kernel that accepted the pair would read one operand's
    bits at the other's width and return finite, plausible, wrong numbers.
    """
    _kernel()
    inp = _make(BATCH, 4, 2, 16, 32, torch.bfloat16)
    assert inp.b_prev is not None
    mixed = inp._replace(
        B=inp.B.to(torch.float16),
        C=inp.C.to(torch.float16),
        b_prev=inp.b_prev.to(torch.float16),
    )
    ssm, b_prev, u_prev = _carries(mixed)
    with pytest.raises(TypeError, match="one dtype per call"):
        decode_step(*mixed.args(), ssm=ssm, b_prev=b_prev, u_prev=u_prev)
    taken = decode_step(*mixed.args(), **_kw(mixed), backend=REFERENCE)
    assert taken.backend == REFERENCE
    assert taken.y.dtype is torch.bfloat16


@pytest.mark.cuda
@pytest.mark.parametrize("tokens", (TOKENS, CHUNK + 1), ids=("one-token", "prefill"))
def test_a_step_selects_no_training_path(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    """Catches a step that runs the module's training composition.

    Three ways the training path gets selected by accident, each silent. The autograd
    node the forward is built from holds every intermediate of the step alive, a leak
    per token, and is refused by the boundary on some later token rather than at the
    one that recorded it. A module left in training mode takes whatever branch trains.
    And an enabled grad mode around the call is the normal condition rather than an
    unusual one, since a decode loop runs in whatever mode its caller had.

    The node is trapped rather than inferred from ``grad_fn``, because a node whose
    output was detached leaves no ``grad_fn`` and every intermediate alive anyway. That
    the trap can fire at all is asserted through the forward, so a renamed node does
    not read as a pass.
    """
    dtype = torch.bfloat16
    mixer = _mixer(4, True, dtype)
    config = mixer.config
    x = _tokens(config, BATCH, tokens, dtype)
    assert any(one.requires_grad for one in mixer.parameters())

    trained = _allocate(config, BATCH, dtype)
    evaluated = _allocate(config, BATCH, dtype)
    mixer.train()
    with torch.enable_grad():
        got = mixer.step(x, trained).clone()
    assert got.grad_fn is None and not got.requires_grad
    mixer.eval()
    want = mixer.step(x, evaluated).clone()
    assert torch.equal(got, want), "training mode changed the step"
    for name, buffer in _buffers(trained).items():
        assert torch.equal(buffer, _buffers(evaluated)[name]), name

    reached: list[int] = []
    node = mixer_module._SLinOSSMixerFunction
    apply = node.apply

    def trap(*args: object, **kwargs: object) -> object:
        reached.append(1)
        return apply(*args, **kwargs)

    monkeypatch.setattr(node, "apply", staticmethod(trap))
    fresh = _allocate(config, BATCH, dtype)
    with torch.enable_grad():
        mixer.step(x, fresh)
    assert reached == [], "the step selected the training path"
    with torch.enable_grad():
        mixer(x)
    assert reached == [1], "the trap never fires, so its silence proves nothing"
