"""Residual block built around the SLinOSS mixer."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from slinoss.layers import RMSNorm, SLinOSSMixer
from slinoss.layers.mlp import SLinOSSMLP
from slinoss.layers.state import SLinOSSMixerState
from slinoss.ops.block import block_ffn_residual

from .config import NormKind, SLinOSSBlockConfig
from .state import SLinOSSBlockState


def _make_norm(
    kind: NormKind,
    d_model: int,
    *,
    eps: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    factory_kwargs = {"device": device, "dtype": dtype}
    if kind == "rmsnorm":
        return RMSNorm(d_model, eps=eps, **factory_kwargs)
    return nn.LayerNorm(d_model, eps=eps, **factory_kwargs)


def _apply_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight = getattr(norm, "weight", None)
    if isinstance(weight, torch.Tensor):
        return norm(x.to(dtype=weight.dtype))
    return norm(x)


class SLinOSSBlock(nn.Module):
    """Serial pre-norm residual block with mixer then FFN."""

    def __init__(
        self,
        config: SLinOSSBlockConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.residual_in_fp32 = bool(config.residual_in_fp32)
        self.residual_dropout = float(config.residual_dropout)
        self.mixer_norm = _make_norm(
            config.norm_kind,
            config.d_model,
            eps=config.norm_eps,
            device=device,
            dtype=dtype,
        )
        self.mixer = SLinOSSMixer(
            config.d_model,
            d_state=config.mixer.d_state,
            expand=config.mixer.expand,
            d_head=config.mixer.d_head,
            d_conv=config.mixer.d_conv,
            chunk_size=config.mixer.chunk_size,
            bc_groups=config.mixer.bc_groups,
            dt_min=config.mixer.dt_min,
            dt_max=config.mixer.dt_max,
            dt_init_floor=config.mixer.dt_init_floor,
            gamma_min=config.mixer.gamma_min,
            gamma_max=config.mixer.gamma_max,
            theta_init_min=config.mixer.theta_init_min,
            theta_init_max=config.mixer.theta_init_max,
            r_min=config.mixer.r_min,
            r_max=config.mixer.r_max,
            eps=config.mixer.eps,
            device=device,
            dtype=dtype,
        )
        if config.ffn is None:
            self.ffn_norm: nn.Module | None = None
            self.ffn: SLinOSSMLP | None = None
        else:
            self.ffn_norm = _make_norm(
                config.norm_kind,
                config.d_model,
                eps=config.norm_eps,
                device=device,
                dtype=dtype,
            )
            self.ffn = SLinOSSMLP.from_config(
                config.d_model,
                config.ffn,
                device=device,
                dtype=dtype,
            )

    def mixer_inputs(
        self,
        x: torch.Tensor,
        *,
        context: object | None = None,
    ) -> torch.Tensor:
        """Return the tensor consumed by the mixer branch.

        Downstream block variants can override this to inject structured
        transforms after the pre-mixer normalization without replacing the
        whole residual block. `context` is intentionally opaque so stack
        specializations can thread auxiliary per-call metadata without
        hardwiring those details into the core block API.
        """

        del context
        return _apply_norm(self.mixer_norm, x)

    def ffn_inputs(
        self,
        x: torch.Tensor,
        *,
        context: object | None = None,
    ) -> torch.Tensor:
        """Return the tensor consumed by the FFN branch."""

        if self.ffn_norm is None:
            raise RuntimeError("FFN inputs requested for a mixer-only block.")
        del context
        return _apply_norm(self.ffn_norm, x)

    def forward_mixer_branch(
        self,
        x: torch.Tensor,
        *,
        state: SLinOSSBlockState | None = None,
        return_state: bool = False,
        context: object | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, SLinOSSBlockState]:
        mixed = self.mixer(
            self.mixer_inputs(x, context=context),
            state=None if state is None else state.mixer,
            return_state=return_state,
        )
        if not return_state:
            return cast(torch.Tensor, mixed)
        mixed_out, mixer_state = cast(tuple[torch.Tensor, SLinOSSMixerState], mixed)
        return mixed_out, SLinOSSBlockState(mixer=mixer_state)

    def forward_ffn_branch(
        self,
        x: torch.Tensor,
        *,
        context: object | None = None,
    ) -> torch.Tensor:
        if self.ffn is None:
            raise RuntimeError("FFN branch requested for a mixer-only block.")
        return self.ffn(self.ffn_inputs(x, context=context))

    def step_mixer_branch(
        self,
        x: torch.Tensor,
        state: SLinOSSBlockState | None = None,
        *,
        inplace: bool | None = None,
        context: object | None = None,
    ) -> tuple[torch.Tensor, SLinOSSBlockState]:
        mixed, mixer_state = self.mixer.step(
            self.mixer_inputs(x, context=context),
            None if state is None else state.mixer,
            inplace=inplace,
        )
        return mixed, SLinOSSBlockState(mixer=mixer_state)

    def step_ffn_branch(
        self,
        x: torch.Tensor,
        *,
        context: object | None = None,
    ) -> torch.Tensor:
        if self.ffn is None:
            raise RuntimeError("FFN branch requested for a mixer-only block.")
        return self.ffn.decode_one(self.ffn_inputs(x, context=context))

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSBlockState:
        return SLinOSSBlockState(
            mixer=self.mixer.init_state(batch_size, device=device, dtype=dtype)
        )

    def init_decode_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSBlockState:
        return SLinOSSBlockState(
            mixer=self.mixer.init_decode_state(batch_size, device=device, dtype=dtype)
        )

    def _drop_branch(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_dropout <= 0.0:
            return x
        return F.dropout(x, p=self.residual_dropout, training=self.training)

    def _residual_add(
        self, residual: torch.Tensor, branch: torch.Tensor
    ) -> torch.Tensor:
        compute_dtype = torch.float32 if self.residual_in_fp32 else residual.dtype
        return (
            residual.to(compute_dtype)
            .add(branch.to(compute_dtype))
            .to(dtype=residual.dtype)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: SLinOSSBlockState | None = None,
        return_state: bool = False,
        context: object | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, SLinOSSBlockState]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x shape (batch, T, {self.d_model}), got {tuple(x.shape)}."
            )

        next_state: SLinOSSBlockState | None = None
        if return_state:
            mixed, next_state = cast(
                tuple[torch.Tensor, SLinOSSBlockState],
                self.forward_mixer_branch(
                    x,
                    state=state,
                    return_state=True,
                    context=context,
                ),
            )
        else:
            mixed = cast(
                torch.Tensor,
                self.forward_mixer_branch(
                    x,
                    state=state,
                    return_state=False,
                    context=context,
                ),
            )
        out = self._residual_add(x, self._drop_branch(mixed))
        if self.ffn is not None and self.ffn_norm is not None:
            out = block_ffn_residual(self, out, context=context)
        if not return_state:
            return out
        if next_state is None:
            raise RuntimeError("Expected a next state from the mixer.")
        return out, next_state

    def step(
        self,
        x: torch.Tensor,
        state: SLinOSSBlockState | None = None,
        *,
        inplace: bool | None = None,
        context: object | None = None,
    ) -> tuple[torch.Tensor, SLinOSSBlockState]:
        return_token = x.ndim == 2
        token = x
        if token.ndim == 3:
            if token.shape[1] != 1:
                raise ValueError(
                    f"Expected time dim 1 for block.step, got {tuple(token.shape)}."
                )
            token = token[:, 0, :]
        if token.ndim != 2 or token.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x shape (batch, {self.d_model}) or (batch, 1, {self.d_model}), "
                f"got {tuple(x.shape)}."
            )

        mixed, next_state = self.step_mixer_branch(
            token,
            state,
            inplace=inplace,
            context=context,
        )
        out = self._residual_add(token, self._drop_branch(mixed.to(dtype=token.dtype)))
        if self.ffn is not None and self.ffn_norm is not None:
            out = self._residual_add(
                out,
                self._drop_branch(self.step_ffn_branch(out, context=context)),
            )
        if not return_token:
            out = out.unsqueeze(1)
        return out, next_state


__all__ = ["SLinOSSBlock"]
