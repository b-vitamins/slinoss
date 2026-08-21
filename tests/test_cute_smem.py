"""The shared-memory residency bar and its inverse.

Pure arithmetic, so no device is needed. Both tests exist because four kernels
size their launch bounds on this bar and the arithmetic was wrong in the unsafe
direction: it admitted an arena that runs one block short, and a residency step
is worth 20% to 43% of a kernel's cycles, so the error is not a rounding detail.

The numbers are derived from :func:`smem_capacity`, never written down, so the
tests hold on any carveout rather than pinning one architecture.
"""

import pytest

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")

from slinoss._cute import (
    SMEM_GRANULE,
    SMEM_RESERVED,
    smem_budget,
    smem_capacity,
    smem_residency,
)

BLOCKS = range(1, 7)


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
