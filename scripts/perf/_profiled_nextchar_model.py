"""Perf-only instrumented nextchar model."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from _nextchar_model import FeedForward
from slinoss.layers import SLinOSSMixer
from slinoss.layers.backend import ReferenceScanPrepBackend, ScanInputs
from slinoss.layers.state import SLinOSSMixerState, ScanState
from slinoss.perf import call_region


class ProfiledSLinOSSMixer(SLinOSSMixer):
    def _build_reference_scanprep_profiled(
        self,
        *,
        value: torch.Tensor,
        params: torch.Tensor,
        bc: torch.Tensor,
    ) -> ScanInputs:
        batch, T, _ = map(int, value.shape)
        U = call_region(
            "mixer.scanprep.pack_u",
            self.scanprep._pack_scan_u,
            value,
            batch,
            T,
        )
        bc = call_region(
            "mixer.scanprep.bc_norm", self.scanprep._normalize_scan_bc_rows, bc
        )
        coeffs = call_region(
            "mixer.scanprep.coefficients",
            self.scanprep._scan_coeffs_from_flat_params,
            params,
            batch,
            T,
        )
        B, C = call_region(
            "mixer.scanprep.pack_bc",
            self.scanprep._pack_scan_bc,
            bc,
            batch,
            T,
        )
        return ScanInputs(U=U, M=coeffs[0], K=coeffs[1], B=B, C=C)

    def _build_scan_inputs(
        self, *, value: torch.Tensor, params: torch.Tensor, bc: torch.Tensor
    ) -> ScanInputs:
        if isinstance(self.scanprep.backend, ReferenceScanPrepBackend):
            return cast(
                ScanInputs,
                call_region(
                    "mixer.scanprep.total",
                    self._build_reference_scanprep_profiled,
                    value=value,
                    params=params,
                    bc=bc,
                ),
            )
        return cast(
            ScanInputs,
            call_region("mixer.scanprep.total", self.scanprep, value, params, bc),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: SLinOSSMixerState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SLinOSSMixerState]:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, T, d_model); got {tuple(x.shape)}.")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected input last dim {self.d_model}; got {x.shape[-1]}."
            )

        batch, T, _ = map(int, x.shape)
        if T == 0:
            empty = x.new_empty((batch, 0, self.d_model))
            next_state = SLinOSSMixerState() if state is None else state
            return (empty, next_state) if return_state else empty

        proj = call_region("mixer.in_proj", self.in_proj, x)
        gate, value_raw, params, bc_flat = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.param_proj_dim, self.bc_proj_dim],
            dim=-1,
        )
        conv_state_in = None if state is None else state.conv
        conv_out, conv_state = call_region(
            "mixer.dw_conv",
            self._apply_causal_depthwise_conv_with_state,
            value_raw,
            conv_state_in,
        )
        value = call_region("mixer.dw_conv_activation", F.silu, conv_out)
        bc = call_region("mixer.bc_emit", self._reshape_bc, bc_flat, batch, T)

        scan_inputs = self._build_scan_inputs(value=value, params=params, bc=bc)
        scan_state_in = None if state is None else state.scan
        scan_result = call_region(
            "v2x2ssd.total",
            self._run_scan_backend,
            scan_inputs,
            scan_state_in,
            return_state,
        )
        if return_state:
            scan_y, scan_state = cast(tuple[torch.Tensor, ScanState], scan_result)
        else:
            scan_y = cast(torch.Tensor, scan_result)
            scan_state = None

        gated = call_region(
            "mixer.gate_skip",
            self._apply_gate_skip,
            scan_y,
            scan_inputs.U,
            gate,
            batch,
            T,
        )
        out = call_region("mixer.out_proj", self.out_proj, gated)

        if not return_state:
            return out

        next_state = SLinOSSMixerState(
            conv=conv_state, scan=cast(ScanState, scan_state)
        )
        return out, next_state


class ProfiledNextCharBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = ProfiledSLinOSSMixer(
            d_model,
            d_state=d_state,
            expand=expand,
            d_head=d_head,
            d_conv=d_conv,
            chunk_size=chunk_size,
            normalize_bc=True,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm1 = call_region("norms.pre_mixer", self.norm1, x)
        x = call_region("residual.mixer", torch.add, x, self.mixer(norm1))
        norm2 = call_region("norms.pre_ffn", self.norm2, x)
        x = call_region(
            "residual.ffn",
            torch.add,
            x,
            call_region("ffn", self.ff, norm2),
        )
        return x


class ProfiledNextCharLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.block_size = int(block_size)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, self.block_size, d_model))
        self.blocks = nn.ModuleList(
            [
                ProfiledNextCharBlock(
                    d_model,
                    d_state=d_state,
                    expand=expand,
                    d_head=d_head,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.perf_trainable_params: tuple[torch.nn.Parameter, ...] = ()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _add_pos_embed(self, x: torch.Tensor, T: int) -> torch.Tensor:
        return x + self.pos_embed[:, :T, :]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (batch, T), got {tuple(idx.shape)}.")
        if idx.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {idx.shape[1]} exceeds block_size {self.block_size}."
            )
        x = call_region("embed.token", self.token_embed, idx)
        x = call_region("embed.pos", self._add_pos_embed, x, int(idx.shape[1]))
        for block in self.blocks:
            x = block(x)
        x = call_region("norms.final", self.norm_f, x)
        return call_region("head.logits", self.lm_head, x)
