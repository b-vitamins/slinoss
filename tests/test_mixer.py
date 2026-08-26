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
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.func import functional_call
from torch.nn.functional import linear

from slinoss._guard import PROJ_ALIGN, SECTOR_BYTES
from slinoss.config import SLinOSSConfig
from slinoss.mixer import (
    HORIZON_RANGE,
    ProjectionLayout,
    SLinOSSMixer,
    head_grid,
)
from slinoss.ops.conv import backends as conv_dispatch
from slinoss.ops.conv import causal_conv1d
from slinoss.ops.mixer import backends as tail_dispatch
from slinoss.ops.mixer import mixer_tail
from slinoss.ops.scanprep import (
    LS_COLUMN,
    PARAM_COLS,
    ROTVEC_COLUMNS,
    TAP_COLUMNS,
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

INIT_TOL = 2e-7
"""Bound on recovering an initialization grid through its own bounded map.

The raw values are float32, so the round trip through ``log(expm1(.))`` and back
carries that rounding and nothing else. Measured: 6.250e-08 for the decay row,
2.988e-08 for the rotation row, and 3.262e-08 for their ratio.
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
        layout.params(proj), mixer.param_bias, heads=cfg.n_heads, w_max=cfg.w_max
    )
    scan = so3ssd(
        step.y, params.trans, params.K, layout.b(proj), layout.c(proj), cfg.chunk_size
    )
    tail = mixer_tail(
        scan.y,
        step.y,
        layout.gate(proj),
        mixer.d_skip,
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

            def forward(*args: Any, **kwargs: Any) -> Any:
                for label, index in bands:
                    seen[label] = args[index]
                return real_forward(*args, **kwargs)

            def backward(*args: Any, **kwargs: Any) -> Any:
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
    cfg = MIXER_CONFIG
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
    cfg = MIXER_CONFIG
    mixer = SLinOSSMixer(cfg, device=cuda, dtype=torch.float32)
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
    """
    cfg = MIXER_CONFIG
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


def test_initialization_inverts_the_bounded_maps() -> None:
    """``param_bias`` holds the raw values the grids ask for, not the grids.

    Both scale maps are bounded and neither is the identity, so a grid written
    straight into the rows lands at a decay and a period the map has moved, and both
    stay inside their invariant. The result is a mixer that trains and never covers
    the timescales it reports.
    """
    cfg = MIXER_CONFIG
    rows = SLinOSSMixer(cfg).param_bias.detach().double()
    ls = bounded_logscale(rows[:, LS_COLUMN])
    horizon = head_grid(HORIZON_RANGE, cfg.n_heads).double()
    assert_max_rel(-0.5 / ls, horizon, INIT_TOL, "mixer init decay")
    w = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max)
    turn = w.norm(dim=-1)
    assert_max_rel(2.0 * math.pi / turn, horizon, INIT_TOL, "mixer init period")
    # I1 and I2 at step zero. The grids are what leaves a margin in the second.
    assert bool((ls <= 0.0).all())
    assert float(turn.max()) < cfg.w_max
    taps = rows[:, TAP_COLUMNS].unflatten(-1, (2, 3))
    assert torch.equal(taps[..., 0], torch.full_like(taps[..., 0], 0.5))
    assert not taps[..., 1:].any()


def test_every_head_turns_once_per_lifetime() -> None:
    """Both scale rows come from one grid, read back through both bounded maps.

    The assertions above hold a row to a constant, so a grid split back into one
    range per row passes them while the ratio between the rows sweeps: the shortest
    heads lose their oscillation inside the amplitude lifetime, the longest carry
    several, and nothing states the schedule. The ratio is the invariant.
    """
    cfg = MIXER_CONFIG
    rows = SLinOSSMixer(cfg).param_bias.detach().double()
    tau = -0.5 / bounded_logscale(rows[:, LS_COLUMN])
    turn = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max).norm(dim=-1)
    assert_max_rel(2.0 * math.pi / turn, tau, INIT_TOL, "turns per lifetime")


def test_conjugation_turns_by_the_rotation_vector_norm() -> None:
    """The period grid is a period of the rotation, not of half of it.

    ``quat_exp`` builds a half-angle quaternion and conjugation doubles the angle
    back, so a token advances the phase by ``|w|``. Read as ``2|w|``, every
    initialized period is halved, and a halved period is still a plausible one, so
    nothing else here fails.
    """
    cfg = MIXER_CONFIG
    rows = SLinOSSMixer(cfg).param_bias.detach().double()
    w = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max)
    trace = rot_matrix(quat_exp(w)).diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    horizon = head_grid(HORIZON_RANGE, cfg.n_heads).double()
    assert_max_rel(2.0 * math.pi / angle, horizon, INIT_TOL, "mixer rotation period")


def test_a_period_the_bound_cannot_reach_stays_finite() -> None:
    """The angle cap on the inverse of ``bounded_rotvec``.

    The inverse is ``s * rsqrt(1 - s*s)`` at ``s = |w| / w_max``, so a ``w_max``
    below what the shortest period asks for takes the root of a negative number and
    every row is NaN. Inactive at the default ``w_max``, so no other test here
    reaches it, and a NaN row trains to nothing rather than raising.
    """
    cfg = SLinOSSConfig(d_model=32, d_state=48, d_head=16, n_groups=2, w_max=1.0)
    assert 2.0 * math.pi / HORIZON_RANGE[0] > cfg.w_max
    rows = SLinOSSMixer(cfg).param_bias.detach().double()
    assert bool(rows.isfinite().all())
    # I2 still holds, which is what the cap is for.
    turn = bounded_rotvec(rows[:, ROTVEC_COLUMNS], cfg.w_max).norm(dim=-1)
    assert float(turn.max()) < cfg.w_max


def test_the_default_bound_reaches_a_half_turn() -> None:
    """What the default ``w_max`` leaves outside the ball.

    ``bounded_rotvec`` approaches the bound without reaching it, so the reachable
    rotations turn by strictly less than ``w_max``, and a half turn is exactly pi. A
    bound short of pi therefore deletes every order-2 element of every finite
    rotation group the recurrence could carry, at any weight, and I2 forbids closing
    the gap by raising the bound to pi itself.
    """
    cfg = SLinOSSConfig(d_model=32, d_state=48)
    raw = torch.zeros(1, 3, dtype=torch.float64)
    raw[0, 0] = 1e6
    turn = float(bounded_rotvec(raw, cfg.w_max).norm(dim=-1))
    assert turn < cfg.w_max < math.pi
    # Half the float32 spacing at pi, the dtype every kernel carries the bound in.
    assert math.pi - turn < 1.2e-7


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


def test_param_bias_stays_float32_through_a_module_cast() -> None:
    """A module-wide demotion must not take the one pinned parameter with it.

    ``mixer.to(torch.bfloat16)`` is how the module reaches a kernel dtype, and
    scanprep refuses a low-precision ``param_bias`` (I4). Demoted, the cast succeeds
    and the next forward raises from an operator that has nothing to do with it.
    """
    mixer = SLinOSSMixer(MIXER_CONFIG).to(torch.bfloat16)
    assert mixer.param_bias.dtype is torch.float32
    assert mixer.in_proj.weight.dtype is torch.bfloat16
    assert mixer.conv_weight.dtype is torch.bfloat16
    # A widening cast is left alone, so a float64 oracle stays float64 end to end.
    assert (
        SLinOSSMixer(MIXER_CONFIG).to(torch.float64).param_bias.dtype is torch.float64
    )


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
