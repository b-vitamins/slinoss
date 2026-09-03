"""Band geometry of the mixer's fused projection, and the mixer over it.

The layout half is CPU-only arithmetic over strides, and the property under test
is that every consumer's operand is a view of the one projection output rather
than a copy of part of it.

The mixer half needs a device for anything that runs a forward: every operand
between the two projections is a column band, and the pitched-layout rule of
:func:`slinoss._guard.check_pitched` is a CUDA rule. Initialization and the
padding of the projection stay on the CPU, where both are arithmetic over
parameters.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.func import functional_call
from torch.nn.functional import linear

from slinoss._guard import PROJ_ALIGN, SECTOR_BYTES
from slinoss.config import SLinOSSConfig
from slinoss.mixer import (
    FALLBACK_SPAN,
    ProjectionLayout,
    SLinOSSMixer,
    fibonacci_axes,
    head_band,
    head_grid,
    head_lattice,
)
from slinoss.ops.conv import backends as conv_dispatch
from slinoss.ops.conv import causal_conv1d
from slinoss.ops.mixer import backends as tail_dispatch
from slinoss.ops.mixer import mixer_tail
from slinoss.ops.scanprep import (
    LS_COLUMN,
    LS_MAX_MAG,
    PARAM_COLS,
    ROTVEC_COLUMNS,
    anchored_rotvec,
    bounded_logscale,
    bounded_rotvec,
    scanprep,
)
from slinoss.ops.scanprep import backends as prep_dispatch
from slinoss.ops.so3ssd import backends as scan_dispatch
from slinoss.ops.so3ssd import quat_exp, rot_matrix, so3ssd
from tests.conftest import assert_max_rel

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


# Small enough for a float64 reference over three chunks, and both widths pad: the
# grouped one carries 4 heads over 2 groups, and the ungrouped one carries one group
# per head, which is the wider state band and the other B/C pullback.
MIXER_CONFIG = SLinOSSConfig(
    d_model=32, d_state=48, d_head=16, n_groups=2, chunk_size=16, bias=True
)
UNGROUPED_CONFIG = SLinOSSConfig(
    d_model=32, d_state=48, d_head=16, n_groups=4, chunk_size=16
)
PAD_CONFIG = SLinOSSConfig(
    d_model=32, d_state=48, d_head=32, n_groups=2, chunk_size=16, bias=True
)
"""A head count whose parameter band leaves the projection width off the sector.

Every other band is a sector multiple by construction, so pad columns exist only
when ``PARAM_COLS * H`` is not one. :data:`MIXER_CONFIG` carries four heads, whose
sixteen parameter columns are a multiple, and the two pad tests would then run
against an empty view. Two heads leave eight columns and the width rounds up by
eight.
"""
BATCH = 2
SEQLEN = 40

PARITY_TOL = 1e-15
"""Bound on the fused node against the same operators under autograd, at float64.

Measured: 0.0 for the output and for all ten gradients. Both paths issue the same
backends on the same operands in the same order, and every band's cotangent is
written once, so the results agree bitwise. The bound is a few float64 ulp, left
as a tolerance rather than an equality so that a reordered reduction inside a
backend reports a number.
"""

BAND = (4.0, 4096.0)
"""A lattice band the grid assertions read, independent of any configuration.

:func:`slinoss.mixer.head_band` derives both ends, so a test that asserts the grid
spans what it is given must be given a band rather than read one.
"""

INIT_TOL = 1e-6
"""Bound on recovering an initialization lattice through its own bounded map.

The raw values are float32 and both inverses amplify that rounding by the raw
magnitude they produce. For the decay row, ``d(log|ls|)/draw = 1 - sigmoid(raw)``,
so a horizon comes back with relative error up to ``|raw| * eps``: 4.6e-7 at the
4096-token rung, where ``raw = -7.62``. The rotation row carries the axis rounding
on top of the radius rounding. This is a rounding bound, not a tolerance on the
arithmetic; the arithmetic itself is exact in float64.
"""

_DPROJ_BAND = {
    "value": "dx",
    "gate": "dgate",
    "B": "dB",
    "C": "dC",
    "params": "dparams",
}
_FORWARD_BANDS = {
    "conv": (("value", 0),),
    "prep": (("params", 0),),
    "scan": (("B", 3), ("C", 4)),
    "tail": (("gate", 2),),
}


@pytest.fixture
def cuda() -> torch.device:
    """The first visible CUDA device, or a skip."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


def _activations(
    cfg: SLinOSSConfig, device: torch.device, dtype: torch.dtype, *, seed: int = 0
) -> Tensor:
    """One batch of activations, ``(BATCH, SEQLEN, d_model)``, in ``dtype``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        BATCH, SEQLEN, cfg.d_model, generator=gen, device=device, dtype=dtype
    )


def _public_composition(mixer: SLinOSSMixer, x: Tensor) -> Tensor:
    """The mixer's six steps through the public per-operator autograd nodes.

    Ground truth for the fused node: the same operands and the same backends, with
    autograd composing them and allocating a cotangent per band.

    Args:
        mixer: Supplies the parameters, the layout, and the configuration.
        x: ``(B,T,d_model)``.

    Returns:
        ``(B,T,d_model)``, in ``x``'s dtype.
    """
    cfg = mixer.config
    layout = mixer.layout
    proj = linear(x, mixer.in_proj.weight, mixer.in_proj.bias)
    step = causal_conv1d(
        layout.value(proj),
        mixer.conv_weight,
        mixer.conv_bias,
        activation=True,
        d_head=cfg.d_head,
    )
    params = scanprep(
        layout.params(proj), mixer.transition_bias, heads=cfg.n_heads, w_max=cfg.w_max
    )
    keys = (
        None
        if mixer.key_weight is None
        else causal_conv1d(layout.keys(proj), mixer.key_weight, activation=False)
    )
    b_band = layout.b(proj) if keys is None else layout.key_b(keys.y)
    c_band = layout.c(proj) if keys is None else layout.key_c(keys.y)
    z0 = mixer.initial_state[None].expand(x.shape[0], -1, -1, -1).contiguous()
    scan = so3ssd(step.y, params.trans, params.K, b_band, c_band, cfg.chunk_size, z0=z0)
    tail = mixer_tail(
        scan.y,
        step.y,
        layout.gate(proj),
        mixer.skip_gain,
        mixer.norm_weight,
        eps=cfg.norm_eps,
    )
    return linear(tail, mixer.out_proj.weight, mixer.out_proj.bias)


def _spy_bands(
    monkeypatch: pytest.MonkeyPatch, seen: dict[str, Tensor], dseen: dict[str, Tensor]
) -> None:
    """Record the band each stage reads and the destination each pullback writes.

    Patched at the dispatch module the mixer looks its backend up in, so the record
    is what the composition passes and the call itself is unchanged.

    Args:
        monkeypatch: Undoes the four patches at teardown.
        seen: Filled with the forward's band operands, by band name.
        dseen: Filled with the backward's destination keywords, by keyword name.
    """
    for dispatch, stage in (
        (conv_dispatch, "conv"),
        (prep_dispatch, "prep"),
        (scan_dispatch, "scan"),
        (tail_dispatch, "tail"),
    ):
        real_get = dispatch.get

        def get(name: str, real_get: Any = real_get, stage: str = stage) -> Any:
            backend = real_get(name)
            real_forward = backend.forward
            real_backward = backend.backward
            bands = _FORWARD_BANDS[stage]

            # The key convolution is the one call that carries activation=False,
            # and it reads the state bands rather than the value band. Its
            # provenance is covered by the gradient parity against the public
            # composition; recording it here would overwrite the value band's.
            def forward(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get("activation", True):
                    for label, index in bands:
                        seen[label] = args[index]
                return real_forward(*args, **kwargs)

            def backward(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get("activation", True):
                    for label, _ in bands:
                        dseen[_DPROJ_BAND[label]] = kwargs[_DPROJ_BAND[label]]
                return real_backward(*args, **kwargs)

            return backend._replace(
                forward=cast("Any", forward), backward=cast("Any", backward)
            )

        monkeypatch.setattr(dispatch, "get", get)


@pytest.mark.cuda
def test_gradients_match_the_public_composition(cuda: torch.device) -> None:
    """The fused node's pullback is the composition's, gradient by gradient.

    The one buffer, the handover of the tail's ``du`` as the scan's addend, and the
    order the bands are written in are all invisible in the forward. Any of them
    wrong gives a gradient that is still finite, still the right shape, and wrong by
    a term, on one parameter out of ten.
    """
    cfg = MIXER_CONFIG
    mixer = SLinOSSMixer(cfg, device=cuda).to(torch.float64)
    names = [name for name, _ in mixer.named_parameters()]
    params = list(mixer.parameters())
    x = _activations(cfg, cuda, torch.float64)
    dout = _activations(cfg, cuda, torch.float64, seed=1)

    fused_x = x.clone().requires_grad_(True)
    ref_x = x.clone().requires_grad_(True)
    fused = mixer(fused_x)
    ref = _public_composition(mixer, ref_x)
    assert_max_rel(fused, ref, PARITY_TOL, "mixer forward")

    fused_grads = torch.autograd.grad(fused, [fused_x, *params], dout)
    ref_grads = torch.autograd.grad(ref, [ref_x, *params], dout)
    for name, got, want in zip(["x", *names], fused_grads, ref_grads, strict=True):
        assert_max_rel(got, want, PARITY_TOL, f"mixer grad {name}")


@pytest.mark.cuda
def test_gradcheck_over_every_input(cuda: torch.device) -> None:
    """Numerical ground truth for the pullback, over a sequence that ends mid-chunk.

    Parity is against a composition of the same backends, so a wrong VJP inside one
    of them satisfies both paths. Fast mode projects the jacobian onto one random
    direction per input, which covers every element of it up to that projection.
    """
    cfg = SLinOSSConfig(
        d_model=16, d_state=48, d_head=16, n_groups=1, chunk_size=16, bias=True
    )
    mixer = SLinOSSMixer(cfg, device=cuda).to(torch.float64)
    names = [name for name, _ in mixer.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_(True) for p in mixer.parameters())
    x = _activations(cfg, cuda, torch.float64)[:1, :24].clone().requires_grad_(True)

    def run(x: Tensor, *params: Tensor) -> Tensor:
        return functional_call(mixer, dict(zip(names, params, strict=True)), (x,))

    assert torch.autograd.gradcheck(run, (x, *params), fast_mode=True)


def test_pad_columns_of_the_projection_are_zero() -> None:
    """The columns no band owns carry zeros for every input.

    The projection computes them because its width is padded to the sector rule.
    Left as numbers, they are what a band addressed one sector wide reads, and they
    are plausible values rather than an obvious fault.
    """
    cfg = PAD_CONFIG
    mixer = SLinOSSMixer(cfg)
    assert mixer.layout.pad_width > 0
    x = _activations(cfg, torch.device("cpu"), torch.float32)
    proj = linear(x, mixer.in_proj.weight, mixer.in_proj.bias)
    assert not mixer.layout.pad(proj).any()


@pytest.mark.cuda
def test_pad_columns_of_the_gradient_buffer_are_zero(cuda: torch.device) -> None:
    """The one band of the cotangent buffer no consumer writes is zeroed.

    ``dproj`` is allocated uninitialized, so a missing zero leaves the input
    projection's pullback reducing over whatever the caching allocator last held
    there. The pad rows of ``in_proj.weight.grad`` are where that shows, and a NaN
    in them poisons a global gradient norm rather than one row.
    """
    cfg = PAD_CONFIG
    mixer = SLinOSSMixer(cfg, device=cuda, dtype=torch.float32)
    assert mixer.layout.pad_width > 0
    x = _activations(cfg, cuda, torch.float32)
    # Free blocks of exactly the buffer's size, filled with NaN: the allocator
    # serves the backward's torch.empty from them.
    poison = [
        torch.full((BATCH, SEQLEN, mixer.layout.width), float("nan"), device=cuda)
        for _ in range(4)
    ]
    del poison
    mixer(x).square().sum().backward()

    grad = mixer.in_proj.weight.grad
    assert grad is not None
    stop = mixer.layout.params_off + PARAM_COLS * cfg.n_heads
    assert not grad[stop:].any()
    assert bool(grad.isfinite().all())


@pytest.mark.cuda
def test_every_operand_and_every_destination_is_a_band(
    cuda: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One projection in, one cotangent buffer out, and no copy of either.

    The composition is fused for exactly this: a ``contiguous`` or a ``cat``
    anywhere in it returns the same numbers at the cost of a pass over the
    activations per band, and a per-band cotangent buffer costs another five. Both
    are silent in every other test here.

    Without the key convolution, which is the case where every operand is a band of
    the projection itself. With it, ``B`` and ``C`` are bands of its output instead,
    which :func:`test_the_key_convolution_keeps_b_and_c_one_buffer` asserts and the
    gradient parity above covers.
    """
    cfg = replace(MIXER_CONFIG, key_conv=False)
    seen: dict[str, Tensor] = {}
    dseen: dict[str, Tensor] = {}
    _spy_bands(monkeypatch, seen, dseen)
    mixer = SLinOSSMixer(cfg, device=cuda, dtype=torch.float32)
    mixer(_activations(cfg, cuda, torch.float32)).square().sum().backward()

    layout = mixer.layout
    offsets = {
        "value": 0,
        "gate": layout.gate_off,
        "B": layout.b_off,
        "C": layout.c_off,
        "params": layout.params_off,
    }
    assert set(seen) == set(offsets)
    assert set(dseen) == set(_DPROJ_BAND.values())
    proj_base = seen["value"].untyped_storage().data_ptr()
    grad_base = dseen["dx"].untyped_storage().data_ptr()
    assert grad_base != proj_base
    for label, offset in offsets.items():
        for band, base in (
            (seen[label], proj_base),
            (dseen[_DPROJ_BAND[label]], grad_base),
        ):
            assert band.untyped_storage().data_ptr() == base, label
            assert band.stride(-1) == 1, label
            assert band.stride(-2) == layout.width, label
            assert (band.data_ptr() - base) // band.element_size() == offset, label


@pytest.mark.cuda
def test_the_key_convolution_keeps_b_and_c_one_buffer(cuda: torch.device) -> None:
    """The key convolution writes one buffer, and the scan reads both bands of it.

    Its output cannot be a band of the projection -- the projection holds the
    operand it convolves -- so the fusion invariant moves rather than holding: one
    extra buffer, both state bands cut from it, and the pullback landing back in the
    projection's own cotangent band. Two calls over the halves, or a ``cat`` of two
    outputs, would return the same numbers and cost a second pass.
    """
    cfg = MIXER_CONFIG
    assert cfg.key_conv
    mixer = SLinOSSMixer(cfg, device=cuda, dtype=torch.float32)
    key_weight = mixer.key_weight
    assert key_weight is not None
    layout = mixer.layout
    proj = torch.randn(BATCH, SEQLEN, layout.width, device=cuda, dtype=torch.float32)
    keys = causal_conv1d(layout.keys(proj), key_weight, activation=False).y
    span = layout.groups * layout.state_dim
    assert keys.shape == (BATCH, SEQLEN, 2 * span)
    base = keys.untyped_storage().data_ptr()
    for band, offset in ((layout.key_b(keys), 0), (layout.key_c(keys), span)):
        assert band.untyped_storage().data_ptr() == base
        assert band.stride(-1) == 1
        assert band.stride(-2) == 2 * span
        assert (band.data_ptr() - base) // band.element_size() == offset
    # The delta taps make the convolution the identity at initialization, so the two
    # bands are the projection's own until a tap moves.
    assert torch.equal(layout.key_b(keys), layout.b(proj))
    assert torch.equal(layout.key_c(keys), layout.c(proj))


def test_initialization_inverts_the_bounded_maps() -> None:
    """The effective transition bias holds raw values, not mapped lattice values.

    Both scale maps are bounded and neither is the identity, so a lattice written
    straight into the rows lands at a decay and a period the map has moved, and both
    stay inside their invariant. The result is a mixer that trains and never covers
    the timescales it reports.
    """
    cfg = MIXER_CONFIG
    rows = SLinOSSMixer(cfg).transition_bias.detach().double()
    horizon, period = (t.double() for t in head_lattice(cfg.n_heads, head_band(cfg)))
    ls = bounded_logscale(rows[:, LS_COLUMN])
    assert_max_rel(-0.5 / ls, horizon, INIT_TOL, "mixer init decay")
    w = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max)
    turn = w.norm(dim=-1)
    assert_max_rel(2.0 * math.pi / turn, period, INIT_TOL, "mixer init period")
    # I1 and I2 at step zero. The lattice stays inside the chart scale, leaving
    # another whole scale before the rotation asymptote.
    assert bool((ls <= 0.0).all())
    assert bool((ls >= -LS_MAX_MAG).all())
    assert float(turn.max()) < cfg.w_max
    assert rows.shape[1] == PARAM_COLS


def test_turns_per_lifetime_sweeps_rather_than_holding_at_one() -> None:
    """The two rows come from two ranges, so their ratio is free.

    Turns per amplitude lifetime is ``h/p``. Driving both rows from one grid pins it
    to one at every head -- every head turning exactly once before it forgets, a
    schedule nothing asks for -- and the per-row assertions above cannot see that,
    because each row is still correct against its own grid. The ratio is the
    invariant, and what it has to do is sweep: the corner ``h/p >> 1`` is a
    narrowband resonator and the corner ``h/p << 1`` is a head that decays before it
    turns, which is the scalar transition.
    """
    cfg = MIXER_CONFIG
    rows = SLinOSSMixer(cfg).transition_bias.detach().double()
    tau = -0.5 / bounded_logscale(rows[:, LS_COLUMN])
    turn = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max).norm(dim=-1)
    quality = tau * turn / (2.0 * math.pi)
    horizon, period = (t.double() for t in head_lattice(cfg.n_heads, head_band(cfg)))
    assert_max_rel(quality, horizon / period, INIT_TOL, "turns per lifetime")
    band = head_band(cfg)
    span = band[1] / band[0]
    assert float(quality.max() / quality.min()) == pytest.approx(span**2, rel=1e-5)


def test_the_lattice_snakes_and_repeats_no_pair() -> None:
    """The two axes as a lattice, not as a product written in either order.

    Consecutive heads differ in one coordinate, which is what makes the head axis a
    usable grouping axis: a plain product jumps the whole horizon range at every
    period boundary. ``H`` need not factor, so the tail of the last period is
    dropped rather than wrapped, and no pair may repeat -- a repeat is a head that
    costs parameters and adds no timescale.
    """
    for heads in (1, 2, 4, 7, 12, 16):
        horizon, period = head_lattice(heads, BAND)
        assert horizon.shape == period.shape == (heads,)
        pairs = {(float(h), float(p)) for h, p in zip(horizon, period, strict=True)}
        assert len(pairs) == heads
        moved = (horizon.diff() != 0.0).long() + (period.diff() != 0.0).long()
        assert bool((moved <= 1).all())
    # Both endpoints of both ranges, at a head count that fills the lattice exactly.
    horizon, period = head_lattice(16, BAND)
    assert (float(horizon.min()), float(horizon.max())) == BAND
    assert (float(period.min()), float(period.max())) == BAND


def test_the_head_grid_is_log_spaced() -> None:
    """A scale grid, not a linear one.

    A linear grid over a ratio of 1024 puts fourteen of sixteen heads past the
    hundred-token mark and leaves the short timescales to two of them. Log spacing is
    what makes the rungs a bank; both endpoints are included so the ranges mean what
    they say.
    """
    grid = head_grid(BAND, 5).double()
    step = (BAND[1] / BAND[0]) ** 0.25
    assert_max_rel(grid[1:] / grid[:-1], torch.full((4,), step).double(), 1e-6, "step")
    assert float(grid[0]) == BAND[0]
    assert float(grid[-1]) == BAND[1]


def test_the_axes_are_equidistributed_and_seedless() -> None:
    """One unit axis per head, a function of ``H`` alone.

    A pseudo-random draw needs a seed to reproduce and still clusters: two axes a few
    degrees apart are two heads carrying one rotation plane. The Fibonacci set has a
    minimum separation that falls like ``1/sqrt(H)``, so it is asserted against that
    rather than against a constant.
    """
    for heads in (1, 4, 16, 64):
        axes = fibonacci_axes(heads).double()
        assert axes.shape == (heads, 3)
        assert_max_rel(axes.norm(dim=-1), torch.ones(heads).double(), 1e-6, "unit")
        if heads > 1:
            gram = (axes @ axes.T).fill_diagonal_(-1.0)
            assert float(gram.max()) < 1.0 - 1.0 / heads
    assert torch.equal(fibonacci_axes(8), fibonacci_axes(8))


def test_conjugation_turns_by_the_rotation_vector_norm() -> None:
    """The period axis is a period of the rotation, not of half of it.

    ``quat_exp`` builds a half-angle quaternion and conjugation doubles the angle
    back, so a token advances the phase by ``|w|``. Read as ``2|w|``, every
    initialized period is halved, and a halved period is still a plausible one, so
    nothing else here fails.
    """
    cfg = MIXER_CONFIG
    rows = SLinOSSMixer(cfg).transition_bias.detach().double()
    w = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max)
    trace = rot_matrix(quat_exp(w)).diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    period = head_lattice(cfg.n_heads, head_band(cfg))[1].double()
    assert_max_rel(2.0 * math.pi / angle, period, INIT_TOL, "mixer rotation period")


def test_the_fastest_rung_asks_for_half_of_any_chart_scale() -> None:
    """The fast end of the initialized band is half the chart scale.

    A band whose fast end were an absolute token count would cross the canonical
    SO(3) range for a small enough ``w_max``. Deriving it from the scale puts the
    rung at exactly :data:`slinoss.mixer._MAP_HEADROOM`, independent of that scale.
    """
    for w_max in (0.5, 1.0, 3.1415925):
        cfg = SLinOSSConfig(
            d_model=32, d_state=48, d_head=16, n_groups=2, w_max=w_max, seq_len=4096
        )
        assert 2.0 * math.pi / head_band(cfg)[0] == pytest.approx(0.5 * w_max)
        rows = SLinOSSMixer(cfg).transition_bias.detach().double()
        assert bool(rows.isfinite().all())
        # I2 still holds, with the initialization inside the named scale.
        turn = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max).norm(dim=-1)
        assert float(turn.max()) < cfg.w_max


def test_the_band_narrows_to_the_trained_sequence() -> None:
    """The slow end is one turn across ``seq_len``, not a harness's longest run.

    A head whose period exceeds the sequence never completes a turn and a head whose
    horizon does never decays within it: both are constants, and the absolute slow
    end put six of eight rungs there at 32 tokens. The band is the mechanism, so it
    is asserted on the rungs rather than on the constant.
    """
    base = dict(d_model=32, d_state=48, d_head=16, n_groups=2)
    for seq_len in (32, 128, 1024):
        cfg = SLinOSSConfig(**base, seq_len=seq_len)  # type: ignore[arg-type]
        horizon, period = head_lattice(cfg.n_heads, head_band(cfg))
        assert float(period.max()) == pytest.approx(float(seq_len))
        assert float(horizon.max()) == pytest.approx(float(seq_len))
    # Absent the sequence there is no band to derive, so the widest one stands.
    assert head_band(SLinOSSConfig(**base))[1] == FALLBACK_SPAN  # type: ignore[arg-type]


def test_the_default_chart_reaches_a_half_turn_at_finite_radius() -> None:
    """Every order-2 SO(3) element is an interior point, not an asymptote."""
    cfg = SLinOSSConfig(d_model=32, d_state=48)
    raw = torch.zeros(1, 3, dtype=torch.float64)
    fraction = math.pi / cfg.w_max
    raw[0, 0] = fraction / math.sqrt(1.0 - 0.25 * fraction * fraction)
    turn = float(bounded_rotvec(raw, cfg.w_max).norm(dim=-1))
    assert turn == pytest.approx(math.pi, abs=1e-14)
    assert float(raw.norm()) < 2.0


def test_the_projection_starts_the_transition_at_the_lattice() -> None:
    """The parameter band's rows are zero, so every token applies the lattice itself.

    Under ``nn.Linear``'s default those rows carry a Kaiming-uniform draw, and on
    unit-RMS input a column of one fluctuates by ``1/sqrt(3)`` whatever ``d_model``
    is. That random transition obscures the lattice before training starts. Zeroed,
    the lattice is what step zero applies.
    """
    cfg = MIXER_CONFIG
    mixer = SLinOSSMixer(cfg)
    x = _activations(cfg, torch.device("cpu"), torch.float32)
    proj = linear(x, mixer.in_proj.weight, mixer.in_proj.bias)
    assert cfg.bias
    assert not mixer.layout.params(proj).any()


def test_the_homogeneous_orbit_starts_from_a_cyclic_basis() -> None:
    """Zero cannot carry a rotation; every row instead gets one unit coordinate."""
    mixer = SLinOSSMixer(MIXER_CONFIG)
    state = mixer.initial_state
    assert state.shape == (
        MIXER_CONFIG.n_heads,
        MIXER_CONFIG.d_head,
        MIXER_CONFIG.d_state,
    )
    assert torch.equal(state.square().sum(-1), torch.ones_like(state[..., 0]))
    column = state.argmax(-1)
    want = torch.arange(MIXER_CONFIG.d_inner).reshape(
        MIXER_CONFIG.n_heads, MIXER_CONFIG.d_head
    )
    assert torch.equal(column, want.remainder(MIXER_CONFIG.d_state))


def test_forcing_and_residual_output_start_as_trainable_no_ops() -> None:
    """The seeded orbit is not drowned by random forcing or a random residual."""
    mixer = SLinOSSMixer(MIXER_CONFIG)
    layout = mixer.layout
    assert not mixer.in_proj.weight[layout.b_off : layout.c_off].any()
    assert mixer.in_proj.bias is not None
    assert not mixer.in_proj.bias[layout.b_off : layout.c_off].any()
    assert not mixer.out_proj.weight.any()
    assert mixer.out_proj.bias is not None
    assert not mixer.out_proj.bias.any()


def test_the_zeroed_parameter_band_still_takes_gradient() -> None:
    """Zero rows are a starting point, not a stop.

    The pullback to a projection row is ``sum_t dL/dband_t x_t`` and carries no
    factor of the row itself. Addition gives every head a unit pullback, so the
    slowest initialized period does not suppress its token gradient relative to the
    fastest one.
    """
    cfg = MIXER_CONFIG
    mixer = SLinOSSMixer(cfg)
    x = _activations(cfg, torch.device("cpu"), torch.float32)
    proj = linear(x, mixer.in_proj.weight, mixer.in_proj.bias)
    rows = mixer.layout.params(proj).unflatten(-1, (cfg.n_heads, PARAM_COLS))
    bias = mixer.transition_bias[:, ROTVEC_COLUMNS]
    band = anchored_rotvec(rows[..., ROTVEC_COLUMNS], bias)
    bounded_rotvec(band, cfg.w_max).sum().backward()

    grad = mixer.in_proj.weight.grad
    assert grad is not None
    stop = mixer.layout.params_off + PARAM_COLS * cfg.n_heads
    band = grad[mixer.layout.params_off : stop]
    assert bool(band.abs().amax() > 0.0)
    # Every head's rotation rows, not just the first: a band addressed one head wide
    # would leave the rest at zero and never separate the timescales.
    reached = band.unflatten(0, (cfg.n_heads, PARAM_COLS))[:, ROTVEC_COLUMNS]
    assert bool((reached.abs().amax(dim=(-2, -1)) > 0.0).all())


def test_transition_rows_are_a_no_decay_embedding() -> None:
    """An operating point is not a weight magnitude for AdamW to shrink."""
    mixer = SLinOSSMixer(MIXER_CONFIG)
    named = dict(mixer.named_parameters())
    assert named["transition_embedding.weight"] is mixer.param_bias
    assert getattr(mixer.param_bias, "_no_weight_decay", False)
    assert getattr(mixer.d_skip, "_no_weight_decay", False)


def test_a_legacy_checkpoint_loads_in_the_new_coordinates() -> None:
    """The parameterization repair does not turn old checkpoints into bad states."""
    cfg = MIXER_CONFIG
    source = SLinOSSMixer(cfg)
    state = source.state_dict()
    del state["transition_embedding.weight"]
    old_bias = torch.randn_like(source.param_bias)
    old_skip = torch.linspace(0.25, 1.75, cfg.n_heads)
    state["param_bias"] = old_bias
    state["d_skip"] = old_skip
    state._metadata[""]["version"] = 1  # type: ignore[attr-defined]

    loaded = SLinOSSMixer(cfg)
    loaded.load_state_dict(state, strict=True)
    old_rotation = old_bias[:, ROTVEC_COLUMNS]
    old_w = old_rotation * (
        cfg.w_max
        * torch.rsqrt(1.0 + (old_rotation * old_rotation).sum(-1, keepdim=True))
    )
    new_w = bounded_rotvec(loaded.transition_bias[:, ROTVEC_COLUMNS], cfg.w_max)
    torch.testing.assert_close(new_w, old_w)
    assert torch.equal(loaded.transition_bias[:, LS_COLUMN], old_bias[:, LS_COLUMN])
    assert torch.equal(loaded.skip_gain, old_skip)
    assert "initial_state" not in loaded.state_dict()


def test_the_skip_gain_starts_at_one() -> None:
    """The direct path is the identity at init.

    A standard-normal gain gives half the heads a sign-flipped skip around a path that
    carries the token itself, so the stream starts by subtracting part of its own
    input, and the spread is a per-head scale nothing asked for. One is what the same
    parameter is upstream.
    """
    mixer = SLinOSSMixer(MIXER_CONFIG)
    assert torch.equal(mixer.d_skip, torch.ones_like(mixer.d_skip))
    assert torch.equal(mixer.skip_gain, torch.ones_like(mixer.skip_gain))


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(torch.bfloat16, id="bf16", marks=pytest.mark.cute),
    ],
)
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(MIXER_CONFIG, id="grouped"),
        pytest.param(UNGROUPED_CONFIG, id="ungrouped"),
    ],
)
def test_output_keeps_the_shape_and_the_dtype(
    cuda: torch.device, cfg: SLinOSSConfig, dtype: torch.dtype
) -> None:
    """Every reachable combination of backend, dtype, and grouping runs both ways.

    float32 and bfloat16 resolve to different backends for three of the four
    stages, and ``n_groups < n_heads`` takes the other ``B``/``C`` pullback. A
    combination whose operands a backend refuses raises at the guard, and a
    gradient the fused node drops is a ``None`` here rather than a wrong number.
    """
    mixer = SLinOSSMixer(cfg, device=cuda, dtype=dtype)
    x = _activations(cfg, cuda, dtype)
    out = mixer(x)
    assert out.shape == x.shape
    assert out.dtype is dtype
    assert bool(out.isfinite().all())
    out.square().sum().backward()
    for name, param in mixer.named_parameters():
        assert param.grad is not None, name
        assert param.grad.shape == param.shape, name
        assert bool(param.grad.isfinite().all()), name


@pytest.mark.cuda
@pytest.mark.cute
def test_the_documented_shape_fits_the_backward_arena(cuda: torch.device) -> None:
    """One training step at the configuration the README documents.

    The backward's shared-memory live set is set by ``chunk_size``, ``d_head`` and
    ``3N``, and it is wider than the forward's, so a configuration that forwards is
    not evidence that it trains. Nothing here depends on the sequence length, which
    is why the shared one is enough. The other configurations in this module run at
    ``chunk_size = 16``, where the arena is at its narrowest and this bound is not
    the one that binds.
    """
    cfg = SLinOSSConfig(d_model=576, d_state=48, expand=2.0, d_head=48, chunk_size=64)
    mixer = SLinOSSMixer(cfg, device=cuda, dtype=torch.bfloat16)
    out = mixer(_activations(cfg, cuda, torch.bfloat16))
    out.float().square().sum().backward()
    for name, param in mixer.named_parameters():
        assert param.grad is not None, name
        assert bool(param.grad.isfinite().all()), name


def test_the_pinned_parameters_stay_float32_through_a_module_cast() -> None:
    """A module-wide demotion must not take the pinned parameters with it.

    ``mixer.to(torch.bfloat16)`` is how the module reaches a kernel dtype, and
    scanprep refuses a low-precision ``param_bias`` (I4). Demoted, the cast succeeds
    and the next forward raises from an operator that has nothing to do with it.
    ``d_skip`` is the second, and it demotes silently rather than raising, so the
    only thing that catches it is this.

    ``norm_weight`` is the control: it follows the cast, which is what makes the two
    parameters of the tail differ in dtype on the shipped call.
    """
    mixer = SLinOSSMixer(MIXER_CONFIG).to(torch.bfloat16)
    for name in SLinOSSMixer.CRITICAL_FP32_TENSORS:
        assert getattr(mixer, name).dtype is torch.float32, name
    assert mixer.norm_weight.dtype is torch.bfloat16
    assert mixer.in_proj.weight.dtype is torch.bfloat16
    assert mixer.conv_weight.dtype is torch.bfloat16
    # A widening cast is left alone, so a float64 oracle stays float64 end to end.
    wide = SLinOSSMixer(MIXER_CONFIG).to(torch.float64)
    for name in SLinOSSMixer.CRITICAL_FP32_TENSORS:
        assert getattr(wide, name).dtype is torch.float64, name


def test_a_host_call_raises_from_a_band_guard() -> None:
    """No CPU path. The layout rule every band is held to is a device rule.

    The operators accept a host tensor as far as the scan, which refuses one. A
    fallback that ran the whole composition on the host would be a second
    composition, held to no alignment rule and covered by no kernel test.
    """
    cfg = MIXER_CONFIG
    mixer = SLinOSSMixer(cfg)
    x = _activations(cfg, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="CUDA device"):
        mixer(x)
