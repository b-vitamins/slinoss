"""Fused SLinOSS sequence mixer and its input-projection layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import linear

from slinoss._guard import PROJ_ALIGN
from slinoss._linear import linear_backward
from slinoss._precision import Float32Module, cast_opt, cast_to
from slinoss.config import ROTATION_CHART_SCALE_MAX, SLinOSSMixerConfig
from slinoss.ops.conv import backends as conv_dispatch
from slinoss.ops.mixer import backends as tail_dispatch
from slinoss.ops.scanprep import LS_COLUMN, LS_MAX_MAG, PARAM_COLS, ROTVEC_COLUMNS
from slinoss.ops.scanprep import backends as prep_dispatch
from slinoss.ops.so3ssd import backends as scan_dispatch
from slinoss.ops.so3ssd.reference import ScanPrologue
from slinoss.state import MixerState, _cyclic_state

__all__ = ["ProjectionLayout", "SLinOSSMixer"]


def _align_up(width: int) -> int:
    """``width`` rounded up to a multiple of :data:`slinoss._guard.PROJ_ALIGN`."""
    return -(-width // PROJ_ALIGN) * PROJ_ALIGN


@dataclass(frozen=True)
class ProjectionLayout:
    """Aligned value, gate, B, C, transition, and padding bands."""

    d_inner: int
    heads: int
    groups: int
    state_dim: int
    width: int

    @classmethod
    def from_config(cls, cfg: SLinOSSMixerConfig) -> ProjectionLayout:
        """Build the layout implied by ``cfg``."""
        bands = 2 * cfg.d_inner + 2 * cfg.n_groups * cfg.d_state
        return cls(
            d_inner=cfg.d_inner,
            heads=cfg.n_heads,
            groups=cfg.n_groups,
            state_dim=cfg.d_state,
            width=bands + _align_up(PARAM_COLS * cfg.n_heads),
        )

    def __post_init__(self) -> None:
        for name, offset in (
            ("gate", self.gate_off),
            ("B", self.b_off),
            ("C", self.c_off),
            ("params", self.params_off),
            ("width", self.width),
        ):
            if offset % PROJ_ALIGN != 0:
                raise ValueError(
                    f"{name} lands on column {offset}, which is not a multiple of "
                    f"{PROJ_ALIGN}: a band row would start mid-sector"
                )
        if self.width < self.params_off + PARAM_COLS * self.heads:
            raise ValueError(
                f"width {self.width} is below the {self.params_off} columns of "
                f"bands plus {PARAM_COLS * self.heads} of parameters"
            )

    @property
    def gate_off(self) -> int:
        """First column of the gate band."""
        return self.d_inner

    @property
    def b_off(self) -> int:
        """First column of the ``B`` band."""
        return 2 * self.d_inner

    @property
    def c_off(self) -> int:
        """First column of the ``C`` band."""
        return self.b_off + self.groups * self.state_dim

    @property
    def params_off(self) -> int:
        """First column of the parameter band."""
        return self.c_off + self.groups * self.state_dim

    @property
    def pad_width(self) -> int:
        """Columns past the last band. They belong to no consumer.

        A cotangent buffer must still zero them: the projection's own pullback
        reads the whole width, so a column no band wrote is a column of garbage
        rather than a column of zero.
        """
        return self.width - self.params_off - PARAM_COLS * self.heads

    def value(self, proj: Tensor) -> Tensor:
        """Value view ``[B,T,E]``."""
        return proj[..., : self.d_inner]

    def gate(self, proj: Tensor) -> Tensor:
        """Gate view ``[B,T,E]``."""
        return proj[..., self.gate_off : self.b_off]

    def b(self, proj: Tensor) -> Tensor:
        """B view ``[B,G,T,S]``."""
        return self._vectors(proj, self.b_off)

    def c(self, proj: Tensor) -> Tensor:
        """C view ``[B,G,T,S]``."""
        return self._vectors(proj, self.c_off)

    def keys(self, proj: Tensor) -> Tensor:
        """Adjacent B/C view ``[B,T,2GS]`` for one convolution."""
        return proj[..., self.b_off : self.params_off]

    def params(self, proj: Tensor) -> Tensor:
        """Token-transition view ``[B,T,H*PARAM_COLS]``."""
        stop = self.params_off + PARAM_COLS * self.heads
        return proj[..., self.params_off : stop]

    def pad(self, proj: Tensor) -> Tensor:
        """Alignment-padding view, empty when ``pad_width == 0``."""
        return proj[..., self.params_off + PARAM_COLS * self.heads :]

    def key_b(self, keys: Tensor) -> Tensor:
        """B view inside a convolved B/C buffer."""
        return self._vectors(keys, 0)

    def key_c(self, keys: Tensor) -> Tensor:
        """C view inside a convolved B/C buffer."""
        return self._vectors(keys, self.groups * self.state_dim)

    def _vectors(self, proj: Tensor, offset: int) -> Tensor:
        """One state band as the scan reads it.

        ``unflatten`` of a unit-stride trailing axis and ``permute`` are both views,
        so the group-major shape costs no copy.
        """
        band = proj[..., offset : offset + self.groups * self.state_dim]
        return band.unflatten(-1, (self.groups, self.state_dim)).permute(0, 2, 1, 3)


_PERIOD_BAND = (4.0, 256.0)
_HORIZON_BAND = (4.0, 4096.0)
_L2_EPS = 1e-12


def _head_grid(bounds: tuple[float, float], heads: int) -> Tensor:
    """Log-spaced ``float32`` grid with exact endpoints."""
    low, high = bounds
    ramp = torch.arange(heads, dtype=torch.float64) / (heads - 1)
    return (low * (high / low) ** ramp).float()


def _head_lattice(heads: int) -> tuple[Tensor, Tensor]:
    """Independent decay-horizon and rotation-period spectra."""
    if heads == 1:
        horizon = math.sqrt(_HORIZON_BAND[0] * _HORIZON_BAND[1])
        period = math.sqrt(_PERIOD_BAND[0] * _PERIOD_BAND[1])
        return (
            torch.tensor([horizon], dtype=torch.float32),
            torch.tensor([period], dtype=torch.float32),
        )
    if heads < 4:
        horizon = _head_grid(_HORIZON_BAND, heads)
        period = _head_grid(_PERIOD_BAND, heads).roll(1)
        return horizon, period
    n_h = max(1, math.isqrt(heads))
    n_p = -(-heads // n_h)
    rows = _head_grid(_HORIZON_BAND, n_h).expand(n_p, n_h).clone()
    rows[1::2] = rows[1::2].flip(-1)
    period = _head_grid(_PERIOD_BAND, n_p).repeat_interleave(n_h)
    return rows.reshape(-1)[:heads], period[:heads]


def _fibonacci_axes(heads: int) -> Tensor:
    """Deterministic spherical-Fibonacci unit axes, shape ``[H,3]``."""
    index = torch.arange(heads, dtype=torch.float64) + 0.5
    height = 1.0 - 2.0 * index / heads
    radius = (1.0 - height * height).clamp(min=0.0).sqrt()
    azimuth = index * (math.pi * (3.0 - math.sqrt(5.0)))
    plane = torch.stack((azimuth.cos(), azimuth.sin()), dim=-1) * radius[:, None]
    return torch.cat((plane, height[:, None]), dim=-1).float()


def _transition_bias_init(config: SLinOSSMixerConfig) -> Tensor:
    """Invert the physical period/horizon lattice into raw transition rows."""
    heads = config.n_heads
    rows = torch.zeros(heads, PARAM_COLS, dtype=torch.float32)
    horizon, period = _head_lattice(heads)
    decay = 0.5 / (horizon * LS_MAX_MAG)
    rows[:, LS_COLUMN] = torch.logit(decay)
    angle = 2.0 * math.pi / (period * ROTATION_CHART_SCALE_MAX)
    radius = angle * torch.rsqrt(1.0 - 0.25 * angle * angle)
    rows[:, ROTVEC_COLUMNS] = radius[:, None] * _fibonacci_axes(heads)
    return rows


def _l2_normalize_(value: Tensor) -> Tensor:
    """Normalize the trailing state axis in place and return its inverse norm."""
    dtype = torch.float64 if value.dtype is torch.float64 else torch.float32
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True, dtype=dtype)
    scale = norm.clamp_min(_L2_EPS).reciprocal()
    value.mul_(scale.to(value.dtype))
    return scale


def _l2_normalize_vjp_(grad: Tensor, unit: Tensor, scale: Tensor) -> None:
    """Apply the L2-normalization pullback to ``grad`` in place."""
    radial = torch.linalg.vecdot(grad, unit, dim=-1).unsqueeze(-1)
    radial.masked_fill_(scale >= 1.0 / _L2_EPS, 0.0)
    grad.addcmul_(unit, radial, value=-1.0)
    grad.mul_(scale.to(grad.dtype))


class _Backends(NamedTuple):
    """Resolved backend names in execution order."""

    conv: str
    prep: str
    scan: str
    tail: str


def _resolve(proj: Tensor) -> _Backends:
    """Resolve once from the post-autocast projection dtype and device."""
    device, dtype = proj.device.type, proj.dtype
    return _Backends(
        conv=conv_dispatch.resolve(None, device, dtype).name,
        prep=prep_dispatch.resolve(None, device, dtype).name,
        scan=scan_dispatch.resolve(None, device, dtype).name,
        tail=tail_dispatch.resolve(None, device, dtype).name,
    )


_Grads = tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    None,
    None,
    None,
]


class _SLinOSSMixerFunction(torch.autograd.Function):
    """One autograd node so every consumer writes one shared ``dproj`` buffer."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: Tensor,
        in_weight: Tensor,
        in_bias: Tensor | None,
        conv_weight: Tensor,
        conv_bias: Tensor | None,
        key_weight: Tensor | None,
        transition_bias: Tensor,
        d_skip: Tensor,
        norm_weight: Tensor,
        out_weight: Tensor,
        out_bias: Tensor | None,
        initial_state: Tensor,
        layout: ProjectionLayout,
        config: SLinOSSMixerConfig,
    ) -> Tensor:
        proj = linear(x, in_weight, in_bias)
        picks = _resolve(proj)
        # The taps carry the activation dtype: a kernel backend holds every operand
        # of the convolution to one dtype, and the reference widens either way.
        step = conv_dispatch.get(picks.conv).forward(
            layout.value(proj),
            cast_to(conv_weight, proj.dtype),
            cast_opt(conv_bias, proj.dtype),
            activation=True,
            d_head=config.d_head,
        )
        # No activation on the keys and no bias: a state vector's direction is what
        # the scan contracts against, and a rectifier on its components confines
        # every key to one octant. Token-major, because the scan reads B and C
        # group-major from a flat band either way.
        keys = (
            None
            if key_weight is None
            else conv_dispatch.get(picks.conv)
            .forward(
                layout.keys(proj),
                cast_to(key_weight, proj.dtype),
                None,
                activation=False,
            )
            .y
        )
        b_band = layout.b(proj) if keys is None else layout.key_b(keys)
        c_band = layout.c(proj) if keys is None else layout.key_c(keys)
        b_scale = _l2_normalize_(b_band)
        c_scale = _l2_normalize_(c_band)
        params = prep_dispatch.get(picks.prep).forward(
            layout.params(proj),
            transition_bias,
            heads=config.n_heads,
            w_max=ROTATION_CHART_SCALE_MAX,
        )
        z0 = initial_state.unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()
        scan = scan_dispatch.get(picks.scan).forward(
            step.y,
            params.trans,
            params.K,
            b_band,
            c_band,
            config.chunk_size,
            z0=z0,
        )
        tail = tail_dispatch.get(picks.tail).forward(
            scan.y, step.y, layout.gate(proj), d_skip, norm_weight, eps=config.norm_eps
        )
        ctx.save_for_backward(
            x,
            proj,
            step.y,
            params.trans,
            params.K,
            scan.y,
            tail,
            in_weight,
            in_bias,
            conv_weight,
            conv_bias,
            transition_bias,
            d_skip,
            norm_weight,
            out_weight,
            out_bias,
            key_weight,
            keys,
            z0,
            b_scale,
            c_scale,
            # Last, so the parameter slices above stay fixed. Three Nones from a
            # backend whose backward rebuilds the boundary instead.
            *((None, None, None) if scan.prologue is None else scan.prologue),
        )
        ctx.layout = layout
        ctx.config = config
        ctx.picks = picks
        return linear(tail, out_weight, out_bias)

    @staticmethod
    def backward(ctx: Any, dout: Tensor) -> _Grads:  # type: ignore[override]
        saved = ctx.saved_tensors
        x, proj, conv_y, trans, K, scan_y, tail = saved[:7]
        in_weight, in_bias, conv_weight, conv_bias, transition_bias = saved[7:12]
        d_skip, norm_weight, out_weight, out_bias = saved[12:16]
        key_weight, keys, z0 = saved[16:19]
        b_scale, c_scale = saved[19:21]
        zstart, cquat, cscale = saved[21:]
        layout: ProjectionLayout = ctx.layout
        config: SLinOSSMixerConfig = ctx.config
        picks: _Backends = ctx.picks

        out_grads = linear_backward(
            dout, tail, out_weight, has_bias=out_bias is not None
        )
        # One buffer for every band's cotangent, uninitialized: each consumer writes
        # its own band in full. Only the columns no consumer owns are zeroed, and
        # the projection's pullback then reads the whole width.
        dproj = torch.empty(proj.shape, dtype=proj.dtype, device=proj.device)
        layout.pad(dproj).zero_()

        tail_grads = tail_dispatch.get(picks.tail).backward(
            out_grads.dinput,
            scan_y,
            conv_y,
            layout.gate(proj),
            d_skip,
            norm_weight,
            eps=config.norm_eps,
            dgate=layout.gate(dproj),
        )
        # With a key convolution the scan's B and C cotangents belong to its output,
        # not to the projection's band, so they land in their own buffer and the
        # convolution's own pullback carries them the rest of the way. Without one
        # the buffer is dproj itself and the band is written in place.
        dkeys = dproj if keys is None else torch.empty_like(keys)
        b_band = layout.b(proj) if keys is None else layout.key_b(keys)
        c_band = layout.c(proj) if keys is None else layout.key_c(keys)
        db_band = layout.b(dproj) if keys is None else layout.key_b(dkeys)
        dc_band = layout.c(dproj) if keys is None else layout.key_c(dkeys)
        # The tail's du is the skip path's share of the scan's dU, handed over as the
        # scan's addend. The returned dU is the sum, so nothing adds it afterwards.
        scan_grads = scan_dispatch.get(picks.scan).backward(
            tail_grads.dy,
            None,
            None,
            None,
            conv_y,
            trans,
            K,
            b_band,
            c_band,
            config.chunk_size,
            z0=z0,
            dB=db_band,
            dC=dc_band,
            dU_init=tail_grads.du,
            prologue=(None if zstart is None else ScanPrologue(zstart, cquat, cscale)),
        )
        _l2_normalize_vjp_(db_band, b_band, b_scale)
        _l2_normalize_vjp_(dc_band, c_band, c_scale)
        key_dweight: Tensor | None = None
        if key_weight is not None:
            key_dweight = cast_to(
                conv_dispatch.get(picks.conv)
                .backward(
                    dkeys,
                    None,
                    layout.keys(proj),
                    cast_to(key_weight, proj.dtype),
                    None,
                    activation=False,
                    dx=layout.keys(dproj),
                )
                .dweight,
                key_weight.dtype,
            )
        prep_grads = prep_dispatch.get(picks.prep).backward(
            scan_grads.dtrans,
            scan_grads.dK,
            layout.params(proj),
            transition_bias,
            heads=config.n_heads,
            w_max=ROTATION_CHART_SCALE_MAX,
            dparams=layout.params(dproj),
        )
        conv_grads = conv_dispatch.get(picks.conv).backward(
            scan_grads.dU,
            None,
            layout.value(proj),
            cast_to(conv_weight, proj.dtype),
            cast_opt(conv_bias, proj.dtype),
            activation=True,
            dx=layout.value(dproj),
        )
        in_grads = linear_backward(dproj, x, in_weight, has_bias=in_bias is not None)
        return (
            cast_to(in_grads.dinput, x.dtype),
            cast_to(in_grads.dweight, in_weight.dtype),
            cast_opt(in_grads.dbias, in_weight.dtype),
            cast_to(conv_grads.dweight, conv_weight.dtype),
            cast_opt(conv_grads.dbias, conv_weight.dtype),
            key_dweight,
            prep_grads.dtransition_bias,
            tail_grads.dd_skip,
            tail_grads.dweight,
            cast_to(out_grads.dweight, out_weight.dtype),
            cast_opt(out_grads.dbias, out_weight.dtype),
            None,
            None,
            None,
        )


class SLinOSSMixer(Float32Module):
    """Fused SO(3) state-space mixer with whole-sequence and decode paths."""

    _float32_names = (
        "transition_bias",
        "d_skip",
        "initial_state",
    )

    def __init__(
        self,
        config: SLinOSSMixerConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layout = ProjectionLayout.from_config(config)
        self.in_proj = nn.Linear(
            config.d_model,
            self.layout.width,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        self.conv_weight = nn.Parameter(
            torch.empty(config.d_inner, config.d_conv, device=device, dtype=dtype)
        )
        self.conv_bias: Tensor | None = (
            nn.Parameter(torch.empty(config.d_inner, device=device, dtype=dtype))
            if config.conv_bias
            else None
        )
        self.key_weight: Tensor | None = (
            nn.Parameter(
                torch.empty(
                    2 * config.n_groups * config.d_state,
                    config.d_conv,
                    device=device,
                    dtype=dtype,
                )
            )
            if config.key_conv
            else None
        )
        init_device = device if device is not None else "cpu"
        self.transition_bias = nn.Parameter(
            _transition_bias_init(config).to(device=init_device)
        )
        self.d_skip = nn.Parameter(
            torch.ones(config.n_heads, device=device, dtype=torch.float32)
        )
        cast(Any, self.transition_bias)._no_weight_decay = True
        cast(Any, self.d_skip)._no_weight_decay = True
        self.register_buffer(
            "initial_state",
            _cyclic_state(
                config.n_heads,
                config.d_head,
                config.d_state,
                device=init_device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.norm_weight = nn.Parameter(
            torch.ones(config.n_heads, config.d_head, device=device, dtype=dtype)
        )
        self.out_proj = nn.Linear(
            config.d_inner,
            config.d_model,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            out_bias = cast(Tensor | None, self.out_proj.bias)
            if out_bias is not None:
                out_bias.zero_()
            bc_weight = self.in_proj.weight[self.layout.b_off : self.layout.params_off]
            row_scale = bc_weight.float().square().sum(-1, keepdim=True).rsqrt()
            bc_weight.mul_(
                (row_scale / math.sqrt(config.d_state)).to(bc_weight.dtype)
            )
            self.in_proj.weight[self.layout.params_off :].zero_()
            in_bias = cast(Tensor | None, self.in_proj.bias)
            if in_bias is not None:
                in_bias[self.layout.b_off : self.layout.params_off].zero_()
                in_bias[self.layout.params_off :].zero_()
            bound = 1.0 / math.sqrt(config.d_conv)
            self.conv_weight.uniform_(-bound, bound)
            if self.conv_bias is not None:
                self.conv_bias.zero_()
            if self.key_weight is not None:
                self.key_weight.zero_()
                self.key_weight[:, -1] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        """Mix ``x[B,T,D]``."""
        return cast(
            "Tensor",
            _SLinOSSMixerFunction.apply(
                x,
                self.in_proj.weight,
                self.in_proj.bias,
                self.conv_weight,
                self.conv_bias,
                self.key_weight,
                self.transition_bias,
                self.d_skip,
                self.norm_weight,
                self.out_proj.weight,
                self.out_proj.bias,
                self.initial_state,
                self.layout,
                self.config,
            ),
        )

    @torch.no_grad()
    def step(self, x: Tensor, state: MixerState) -> Tensor:
        """Mix a prefill/decode segment and advance ``state`` in place."""
        cfg, layout = self.config, self.layout
        if x.ndim != 3 or x.shape[2] != cfg.d_model:
            raise ValueError(f"expected (B,T,{cfg.d_model}), got {tuple(x.shape)}")
        if x.shape[0] != state.batch:
            raise ValueError(
                f"x holds batch {int(x.shape[0])} and state holds {state.batch}"
            )
        proj = linear(x, self.in_proj.weight, self.in_proj.bias)
        if proj.dtype is not state.conv.dtype:
            raise ValueError(
                f"the projection is {proj.dtype} and the state is "
                f"{state.conv.dtype}; cast the module, not the state"
            )
        picks = _resolve(proj)
        if (self.key_weight is None) != (state.keys is None):
            raise ValueError("state key history does not match key_conv")
        conv = conv_dispatch.get(picks.conv).forward(
            layout.value(proj),
            cast_to(self.conv_weight, proj.dtype),
            cast_opt(self.conv_bias, proj.dtype),
            activation=True,
            initial_state=state.conv,
            d_head=cfg.d_head,
        )
        keys = (
            None
            if self.key_weight is None
            else conv_dispatch.get(picks.conv).forward(
                layout.keys(proj),
                cast_to(self.key_weight, proj.dtype),
                None,
                activation=False,
                initial_state=cast(Tensor, state.keys),
            )
        )
        b_band = layout.b(proj) if keys is None else layout.key_b(keys.y)
        c_band = layout.c(proj) if keys is None else layout.key_c(keys.y)
        _l2_normalize_(b_band)
        _l2_normalize_(c_band)
        params = prep_dispatch.get(picks.prep).forward(
            layout.params(proj),
            self.transition_bias,
            heads=cfg.n_heads,
            w_max=ROTATION_CHART_SCALE_MAX,
        )
        scan = scan_dispatch.get(picks.scan).forward(
            conv.y,
            params.trans,
            params.K,
            b_band,
            c_band,
            cfg.chunk_size,
            z0=state.ssm,
            b_prev=state.b_prev,
            u_prev=state.u_prev,
        )
        tail = tail_dispatch.get(picks.tail).forward(
            scan.y,
            conv.y,
            layout.gate(proj),
            self.d_skip,
            self.norm_weight,
            eps=cfg.norm_eps,
        )
        out = linear(tail, self.out_proj.weight, self.out_proj.bias)
        # After every read: the scan starts from state.ssm itself, and the
        # convolution's incoming window is state.conv.
        state.conv.copy_(conv.state)
        if keys is not None:
            cast(Tensor, state.keys).copy_(keys.state)
        state.ssm.copy_(scan.state)
        state.b_prev.copy_(scan.b_last)
        state.u_prev.copy_(scan.u_last)
        return out
