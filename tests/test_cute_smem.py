"""The shared-memory residency bar, its inverse, and the state widths the bar
leaves the two scan directions.

The first three tests are pure arithmetic and need no device. They exist because
four kernels size their launch bounds on this bar and the arithmetic was wrong in
the unsafe direction: it admitted an arena that runs one block short, and a
residency step is worth 20% to 43% of a kernel's cycles, so the error is not a
rounding detail.

The last four sweep the legal ``(chunk_size, d_head)`` family and read the widest
``d_state`` each direction of the scan can run off the layouts. They exist because
the footprint grows with ``d_state`` while :class:`slinoss.config.SLinOSSConfig`
bounds it only below, so the accepted family and the runnable family are different
families and the difference was being carried as a remembered constant instead of a
measurement. A constant cannot express it: the ceiling moves with ``chunk_size`` and
``d_head`` together, and the two directions do not share it.

Nothing is compiled. Every budget is checked by
:func:`slinoss._cute.assert_smem_fits` in the launcher, before ``jit_launch``, so
stubbing the launch keeps every check a caller would meet and deletes the trace, the
compile and the kernel behind it. The sweep therefore runs the real selectors --
whichever lane block, fold, span or warp count each launcher picks for the geometry --
rather than a second copy of their arithmetic that could drift from them.

The numbers are derived from :func:`smem_capacity`, never written down, so the
tests hold on any carveout rather than pinning one architecture.
"""

from __future__ import annotations

import pkgutil
import re
from collections.abc import Iterator
from importlib import import_module
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")

from slinoss._cute import (
    SMEM_GRANULE,
    SMEM_RESERVED,
    smem_budget,
    smem_capacity,
    smem_residency,
)
from slinoss.config import (
    HEAD_MULTIPLE,
    MAX_CHUNK,
    MIN_CHUNK,
    STATE_MULTIPLE,
    SLinOSSConfig,
)
from slinoss.ops.so3ssd import so3ssd
from tests.conftest import LS_BIAS, make_inputs

BLOCKS = range(1, 7)


def _powers(low: int, high: int) -> tuple[int, ...]:
    """Powers of two in ``[low, high]``, low inclusive."""
    out: list[int] = []
    size = low
    while size <= high:
        out.append(size)
        size *= 2
    return tuple(out)


CHUNKS = _powers(MIN_CHUNK, MAX_CHUNK)
"""Every legal ``chunk_size``."""

HEADS = tuple(range(HEAD_MULTIPLE, 8 * HEAD_MULTIPLE + 1, HEAD_MULTIPLE))
"""Legal ``d_head`` up to the widest the faceoff registers, 128."""

WIDTHS = tuple(range(STATE_MULTIPLE, 17 * STATE_MULTIPLE, STATE_MULTIPLE))
"""Legal ``d_state`` from 48 to 768, past every ceiling any geometry has."""

SWEEP_DTYPE = torch.bfloat16
"""Activation dtype of the sweep. The footprints are dtype-dependent, and this is
the one the mixer ships; float32 activations do not reach the cute scan at all."""

BUDGET = ("shared memory", "carveout")
"""Discriminant of a residency refusal against every other ``ValueError``.

The first is :func:`slinoss._cute.assert_smem_fits`, raised when one launcher's
arena is over capacity. The second is a launcher whose whole tile ladder is over
capacity, so it has no block left to try. Anything else is a real fault and the
sweep re-raises it."""

DEAD: dict[tuple[int, int], tuple[str, int]] = {
    (64, 112): ("chunk_vector_bwd", 105_232),
    (64, 128): ("chunk_vector_bwd", 111_152),
    (128, 16): ("chunk_vector_bwd", 135_088),
    (128, 32): ("chunk_vector_bwd", 135_088),
    (128, 48): ("chunk_vector_bwd", 142_736),
    (128, 64): ("chunk_vector_bwd", 150_704),
    (128, 80): ("chunk_vector_bwd", 158_672),
    (128, 96): ("chunk_input_bwd", 110_896),
    (128, 112): ("chunk_input_bwd", 122_768),
    (128, 128): ("chunk_input_bwd", 135_152),
}
"""Every ``(chunk_size, d_head)`` whose backward admits no state width at all.

Keyed by the pair, valued by the launcher that binds it and the bytes that launcher
needs at the narrowest legal state, ``3N = 48``: if the narrowest does not fit,
nothing does. Both values are properties of the layouts and the activation dtype
rather than of the part, so they are pinned here; whether they exceed what a block
may hold is asked of :func:`smem_capacity` instead. Measured at bf16, identical on
two hosts and two torch versions.

`chunk_vector_bwd` binds seven of the ten and `chunk_input_bwd` the other three, and
the split is the chunk length: at ``chunk_size = 128`` from ``d_head = 96`` no
`chunk_input_bwd` lane block fits either, so those three have two blockers."""

NEEDS = re.compile(r"needs (\d+) B")
"""Byte count inside either refusal message. Both spell it ``needs N B``."""

DEFERRED = (
    "the forward accepts a geometry whose backward cannot fit, so the refusal "
    "arrives at the first .backward(): ten pairs, chunk_size 64 at d_head 112 and "
    "128 and chunk_size 128 at every d_head, bound by chunk_vector_bwd through "
    "d_head 80 and by chunk_input_bwd from d_head 96"
)
"""Why the forward-refusal test does not pass yet.

Strict, so the guard that makes it pass also fails the marker and forces the
marker's removal in the same change."""


def test_the_bar_rounds_up_to_the_allocation_granule() -> None:
    # The hardware rounds one block's total, tiles plus reservation, up to the
    # allocation granule before dividing the carveout by it. Dividing the
    # carveout first and subtracting the reservation afterwards skips that
    # rounding and reads high: on sm_86 it puts the three-block bar 85 B above
    # the truth, so an arena sized to it computes as three blocks and runs as
    # two. The bar is therefore granule-aligned once the reservation is added,
    # and one byte past it costs a block.
    carveout = smem_capacity() + SMEM_RESERVED
    for blocks in BLOCKS:
        bar = smem_budget(blocks)
        assert (bar + SMEM_RESERVED) % SMEM_GRANULE == 0, blocks
        assert smem_residency(bar) == blocks, blocks
        if blocks > 1:
            # Not at one block: the floor is one by contract, so an arena past
            # capacity still reads as one and assert_smem_fits is what refuses it.
            assert smem_residency(bar + 1) < blocks, blocks

        naive = carveout // blocks - SMEM_RESERVED
        assert naive >= bar, blocks
        if naive > bar:
            assert smem_residency(naive) < blocks, blocks


def test_the_budget_is_the_exact_inverse_of_the_residency() -> None:
    # One function answers "how many blocks do these bytes allow" and the other
    # "how many bytes does that many blocks allow". A caller that sizes tiles
    # with one and is bounded by the other needs them to agree at the boundary,
    # which is the only place either is interesting.
    assert smem_budget(1) == smem_capacity()
    for blocks in BLOCKS:
        assert smem_residency(smem_budget(blocks)) == blocks, blocks

    for nbytes in (smem_budget(2), smem_budget(3), smem_capacity() // 2):
        assert smem_budget(smem_residency(nbytes)) >= nbytes, nbytes


def test_a_budget_below_one_block_is_refused() -> None:
    # Zero blocks has no largest budget, and the caller asking for it has a bug
    # a silently huge return value would hide.
    with pytest.raises(ValueError, match="at least one"):
        smem_budget(0)


class _Widths(NamedTuple):
    """State widths one ``(chunk_size, d_head)`` geometry admits.

    Attributes:
        forward: Every swept ``d_state`` whose forward launchers all fit.
        backward: Every swept ``d_state`` whose forward and backward launchers all
            fit. Not a subset by construction of the budget: a wider state can
            select a narrower tile, so the two sets are read independently.
        refusal: First refusal message per direction, keyed by ``backward``, so a
            failure names the kernel and the byte count that bound it.
    """

    forward: tuple[int, ...]
    backward: tuple[int, ...]
    refusal: dict[bool, str]


def _no_launch(*_args: object, **_kwargs: object) -> None:
    """Drop one kernel launch. The budget was already checked to reach here."""
    return None


def _stub_launches(patch: pytest.MonkeyPatch) -> int:
    """Replace ``jit_launch`` in every scan launcher with :func:`_no_launch`.

    Imports the whole ``cute`` subtree first: an unimported launcher cannot be
    stubbed, and would compile on first dispatch rather than report its budget.

    Args:
        patch: Context whose exit restores every launcher.

    Returns:
        Number of launchers stubbed.
    """
    package = import_module("slinoss.ops.so3ssd.cute")
    modules = [package]
    for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        modules.append(import_module(info.name))

    stubbed = 0
    for module in modules:
        if hasattr(module, "jit_launch"):
            patch.setattr(module, "jit_launch", _no_launch)
            stubbed += 1
    return stubbed


def _admits(
    chunk: int, rows: int, dim: int, *, backward: bool, take: bool = True
) -> tuple[bool, str]:
    """Run one geometry through the launchers and report whether the arenas fit.

    Args:
        chunk: ``chunk_size``, the scan chunk length ``L``.
        rows: ``d_head``, the rows per head ``P``.
        dim: ``d_state``, the state width ``3N``.
        backward: Make the input bands gradient leaves.
        take: Take the gradient, which reaches six more launchers. ``False`` with
            ``backward`` true stops after the forward, so a refusal is the
            forward's own and not the backward's.

    Returns:
        Whether every launcher the call reaches fits, and the refusal message of
        the first that did not.

    Raises:
        ValueError: Any failure that is not a shared-memory refusal.
    """
    inp = make_inputs(
        bsz=1,
        heads=2,
        groups=1,
        seqlen=2 * chunk,
        rows=rows,
        lanes=dim // 3,
        dtype=torch.float32,
        device="cuda",
        seed=0,
        w_scale=2.0,
        ls_bias=LS_BIAS,
        u_dtype=SWEEP_DTYPE,
        bc_dtype=SWEEP_DTYPE,
    )
    # The two taps are the carry into the first chunk; make_inputs supplies both.
    assert inp.b_prev is not None and inp.u_prev is not None

    def band(tensor: torch.Tensor) -> torch.Tensor:
        """Make one input band a gradient leaf, for the backward only.

        The dtypes are the ones make_inputs produced: the transition and the taps
        stay float32, the three activation bands are the sweep dtype.
        """
        if not backward:
            return tensor
        return tensor.detach().clone().requires_grad_()

    try:
        out = so3ssd(
            band(inp.U.to(SWEEP_DTYPE)),
            band(inp.trans),
            band(inp.K),
            band(inp.B.to(SWEEP_DTYPE)),
            band(inp.C.to(SWEEP_DTYPE)),
            chunk,
            z0=inp.z0,
            b_prev=inp.b_prev.to(SWEEP_DTYPE),
            u_prev=inp.u_prev.to(SWEEP_DTYPE),
            backend="cute",
        )
        if backward and take:
            # The launches are stubbed, so y holds whatever the buffer held.
            out.y.float().nan_to_num().sum().backward()
    except ValueError as exc:
        if any(mark in str(exc) for mark in BUDGET):
            return False, str(exc)
        raise
    return True, ""


@pytest.fixture(scope="module")
def widths() -> Iterator[dict[tuple[int, int], _Widths]]:
    """Read both ceilings off the layouts for every legal geometry.

    The stub is held for the lifetime of the fixture, not just the sweep, so a test
    that asks for this table may also call :func:`_admits` itself without compiling.

    Yields:
        The admitted widths keyed by ``(chunk_size, d_head)``.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    with pytest.MonkeyPatch.context() as patch:
        stubbed = _stub_launches(patch)
        assert stubbed, "no scan launcher exposes jit_launch, so nothing was stubbed"

        table: dict[tuple[int, int], _Widths] = {}
        for chunk in CHUNKS:
            for rows in HEADS:
                found: dict[bool, list[int]] = {False: [], True: []}
                refusal: dict[bool, str] = {}
                for backward in (False, True):
                    for dim in WIDTHS:
                        fits, message = _admits(chunk, rows, dim, backward=backward)
                        if fits:
                            found[backward].append(dim)
                        elif backward not in refusal:
                            refusal[backward] = message
                table[chunk, rows] = _Widths(
                    forward=tuple(found[False]),
                    backward=tuple(found[True]),
                    refusal=refusal,
                )
        yield table


@pytest.mark.cuda
@pytest.mark.cute
def test_the_stub_leaves_the_residency_bar_in_force(
    widths: dict[tuple[int, int], _Widths],
) -> None:
    """Catch a sweep that measured nothing.

    A stub that also swallowed the budget check, or a launcher that checks its
    arena after the launch rather than before, both read as "every width fits"
    and would turn the three tests below into decoration. The widest swept width is
    over capacity at every geometry by a wide margin, in both directions, so a
    sweep that admits it is not measuring the bar.
    """
    top = WIDTHS[-1]
    for key, admitted in sorted(widths.items()):
        assert top not in admitted.forward, f"L={key[0]} P={key[1]} forward"
        assert top not in admitted.backward, f"L={key[0]} P={key[1]} backward"
        assert admitted.refusal, f"L={key[0]} P={key[1]} refused nothing"


@pytest.mark.cuda
@pytest.mark.cute
def test_the_admitted_widths_are_a_prefix_of_the_legal_ladder(
    widths: dict[tuple[int, int], _Widths],
) -> None:
    """Catch a hole, which makes every stated ceiling wrong.

    Each launcher's footprint grows with ``d_state``, but several pick a tile from
    a ladder, and a selector that switches ladder rungs with the width could admit
    a wide state while refusing a narrow one. Then the admitted family has a hole,
    "the widest that fits" stops being a ceiling, and docs, harnesses and any
    future config check all read a bound that skips runnable configurations.
    """
    for key, admitted in sorted(widths.items()):
        for direction, found in (
            ("forward", admitted.forward),
            ("backward", admitted.backward),
        ):
            want = WIDTHS[: len(found)]
            assert found == want, (
                f"L={key[0]} P={key[1]} {direction} admits {found}, "
                f"not the prefix {want}"
            )


@pytest.mark.cuda
@pytest.mark.cute
def test_the_geometries_that_cannot_train_are_the_measured_ten(
    widths: dict[tuple[int, int], _Widths],
) -> None:
    """Catch a layout change that moves the untrainable set, in either direction.

    Ten of the thirty-two legal ``(chunk_size, d_head)`` pairs have an empty
    backward set: the whole ``3N`` extents the backward holds are over a block's
    capacity at the narrowest state the config allows, so no ``d_state`` trains
    there. Pinning the set and its two binding launchers makes a layout change that
    frees a pair, or that costs one, arrive here rather than in a run: the ten are
    what the shipped `chunk_size 64` and the whole `chunk_size 128` column of a
    faceoff table have to be labelled against.
    """
    dead = {key for key, admitted in widths.items() if not admitted.backward}
    assert dead == set(DEAD), (
        f"untrainable geometries are {sorted(dead)}, measured {sorted(DEAD)}"
    )

    for key in sorted(dead):
        kernel, nbytes = DEAD[key]
        tag = f"L={key[0]} P={key[1]}"
        message = widths[key].refusal[True]
        assert message.startswith(kernel), f"{tag}: {kernel} no longer binds, {message}"
        found = NEEDS.search(message)
        assert found is not None, f"{tag}: no byte count in {message}"
        assert int(found[1]) == nbytes, f"{tag}: {message}"
        # The bytes are the layout's; only the comparison is the part's.
        assert nbytes > smem_capacity(), tag


@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.xfail(strict=True, reason=DEFERRED)
def test_a_geometry_whose_backward_cannot_fit_is_refused_at_the_forward(
    widths: dict[tuple[int, int], _Widths],
) -> None:
    """Asserts where the refusal belongs, which is not where it arrives.

    That the ten pairs cannot train is a capacity, not a defect. That the forward
    takes them is: ``SLinOSSConfig`` accepts the pair, the forward records a node,
    and the failure lands at the first ``.backward()``, which is a training step
    that fails after the forward rather than at it -- the same deferral
    ``ops/decode/interface.py`` refuses by not being an autograd node at all.
    ``requires_grad`` is known at the forward, so the refusal is available there,
    one construction earlier than a committed harness, data and schedule.

    Strict xfail: this is the missing guard, not a passing contract, and the guard
    is a training-path change that does not belong in this branch.
    """
    del widths  # Requested for the stub it holds, not for the table.
    for chunk, rows in sorted(DEAD):
        # Proof the shape family is accepted: expand 2.0 puts d_inner at 8 * rows,
        # so d_head divides it and n_groups 1 divides n_heads.
        SLinOSSConfig(
            d_model=4 * rows,
            d_state=STATE_MULTIPLE,
            d_head=rows,
            chunk_size=chunk,
        )
        fits, _ = _admits(chunk, rows, STATE_MULTIPLE, backward=True, take=False)
        assert not fits, (
            f"L={chunk} P={rows}: the forward took operands carrying a gradient "
            f"whose backward has no state width that fits"
        )
