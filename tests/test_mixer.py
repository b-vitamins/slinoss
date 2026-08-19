"""Band geometry of the mixer's fused projection.

CPU only and no kernel: the layout is arithmetic over strides, and the property
under test is that every consumer's operand is a view of the one projection
output rather than a copy of part of it.
"""

from __future__ import annotations

import pytest
import torch

from slinoss._guard import PROJ_ALIGN, SECTOR_BYTES
from slinoss.config import SLinOSSConfig
from slinoss.mixer import ProjectionLayout
from slinoss.ops.scanprep import PARAM_COLS

# Two configurations, chosen for the parameter band's width against the padding
# multiple: 12 heads is 120 columns and pads, 16 heads is 160 and does not. The
# second also carries more than one group, which is the case where the two state
# bands are wider than one column block each.
CONFIGS = [
    pytest.param(
        SLinOSSConfig(d_model=288, d_state=48, d_head=48, n_groups=1), id="padded"
    ),
    pytest.param(
        SLinOSSConfig(d_model=128, d_state=48, d_head=16, n_groups=4), id="exact"
    ),
]


def _proj(layout: ProjectionLayout, seqlen: int = 6) -> torch.Tensor:
    """A projection output to cut bands from."""
    gen = torch.Generator().manual_seed(0)
    return torch.randn(2, seqlen, layout.width, generator=gen)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_bands_tile_the_projection_up_to_the_padding(cfg: SLinOSSConfig) -> None:
    """Every column belongs to exactly one band, or to the padding past them all.

    An overlap would have two consumers writing one cotangent column and one of
    them losing; a gap would be a column the projection GEMM computes and nothing
    reads. Both are invisible in a parity test, because either one still produces
    operands of the right shape.
    """
    layout = ProjectionLayout.from_config(cfg)
    widths = (
        layout.d_inner,
        layout.d_inner,
        layout.groups * layout.state_dim,
        layout.groups * layout.state_dim,
        PARAM_COLS * layout.heads,
    )
    offsets = (0, layout.gate_off, layout.b_off, layout.c_off, layout.params_off)
    for offset, width, following in zip(offsets, widths, offsets[1:], strict=False):
        assert offset + width == following
    assert offsets[-1] + widths[-1] + layout.pad_width == layout.width


@pytest.mark.parametrize("cfg", CONFIGS)
def test_every_band_row_starts_and_steps_on_a_sector(cfg: SLinOSSConfig) -> None:
    """The offsets and the pitch are sector multiples, so no row spans a spare one.

    Only the parameter band has a width the configuration does not already make a
    multiple of :data:`PROJ_ALIGN`, so this is the check that the band order keeps
    the padding past every offset rather than between two of them.
    """
    layout = ProjectionLayout.from_config(cfg)
    for offset in (layout.gate_off, layout.b_off, layout.c_off, layout.params_off):
        assert offset % PROJ_ALIGN == 0
    assert layout.width % PROJ_ALIGN == 0
    # The rule is a byte rule, and the padding is in elements at the narrowest
    # element size any kernel takes.
    assert PROJ_ALIGN * 2 % SECTOR_BYTES == 0


@pytest.mark.parametrize("cfg", CONFIGS)
def test_bands_are_views_at_the_projection_pitch(cfg: SLinOSSConfig) -> None:
    """No band is a copy, and each carries the pitch its consumer's guard expects.

    The whole reason for one projection is that its consumers index it where it
    lies. A ``reshape`` or a ``contiguous`` anywhere in the cutting would still
    return the right values, and would cost a pass over the activations per band.
    """
    layout = ProjectionLayout.from_config(cfg)
    proj = _proj(layout)
    base = proj.untyped_storage().data_ptr()
    token_major = (layout.value(proj), layout.gate(proj), layout.params(proj))
    group_major = (layout.b(proj), layout.c(proj))
    for band in (*token_major, *group_major):
        assert band.untyped_storage().data_ptr() == base
        assert band.stride(-1) == 1
        assert band.stride(-2) == layout.width
    for band in group_major:
        assert band.shape == (2, layout.groups, 6, layout.state_dim)
        # The group axis strides by one vector width, which is less than the token
        # axis strides. A band cut from a group-major buffer has it the other way.
        assert band.stride(1) == layout.state_dim


@pytest.mark.parametrize("cfg", CONFIGS)
def test_state_bands_are_the_two_halves_of_their_columns(cfg: SLinOSSConfig) -> None:
    """``B`` and ``C`` hold the projected values, group-major, in that order.

    The permute is what the scan reads through, so a transposed group axis or the
    two bands swapped would be a silent relabelling of the state.
    """
    layout = ProjectionLayout.from_config(cfg)
    proj = _proj(layout)
    span = layout.groups * layout.state_dim
    for band, offset in (
        (layout.b(proj), layout.b_off),
        (layout.c(proj), layout.c_off),
    ):
        want = proj[..., offset : offset + span]
        want = want.unflatten(-1, (layout.groups, layout.state_dim))
        assert torch.equal(band, want.permute(0, 2, 1, 3))


def test_rejects_a_layout_whose_band_starts_mid_sector() -> None:
    """A width the padding rule does not cover is refused where it is stated.

    Unreachable through :meth:`ProjectionLayout.from_config`, because the
    configuration validates every multiple it derives an offset from. It stays a
    raise so that widening one of those multiples fails here rather than in a
    bandwidth counter nobody is reading.
    """
    with pytest.raises(ValueError, match="mid-sector"):
        ProjectionLayout(d_inner=24, heads=1, groups=1, state_dim=48, width=192)
