"""Perf-only instrumented nextchar model."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from _nextchar_model import FeedForward
from slinoss.layers import SLinOSSMixer
from slinoss.layers.backend import (
    AutoScanBackend,
    CuteScanBackend,
    ReferenceScanPrepBackend,
    ScanInputs,
)
from slinoss.layers.state import SLinOSSMixerState, ScanState
from slinoss.ops.mixer import mixer_tail, split_mixer_projection
from slinoss.perf import call_region


class ProfiledSLinOSSMixer(SLinOSSMixer):
    def _emit_mixer_branches(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return split_mixer_projection(
            x,
            self.in_proj.weight,
            d_inner=self.d_inner,
            param_proj_dim=self.param_proj_dim,
        )

    def _emit_scan_bc(
        self,
        bc_flat: torch.Tensor,
        *,
        batch_size: int,
        time_steps: int,
    ) -> torch.Tensor:
        return bc_flat.view(
            batch_size,
            time_steps,
            self.bc_groups,
            self.scanprep.bc_param_rows,
            self.d_state,
        )

    def _run_scan_backend(
        self,
        scan_inputs: ScanInputs,
        scan_state: ScanState | None,
        return_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        use_cute_scan = scan_inputs.U.device.type == "cuda" and isinstance(
            self.scan_backend,
            (AutoScanBackend, CuteScanBackend),
        )
        if use_cute_scan and self.d_state % 8 != 0:
            raise ValueError(
                "The current CuTe scan backend requires d_state to be a multiple "
                f"of 8 (got d_state={self.d_state})."
            )
        return self.scan_backend(
            scan_inputs,
            chunk_size=self.chunk_size,
            state=scan_state,
            return_state=return_state,
        )

    def _profile_reference_scanprep(
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
            batch=batch,
            T=T,
        )
        B_pairs, C_pairs = cast(
            tuple[torch.Tensor, torch.Tensor],
            call_region(
                "mixer.scanprep.bc_parameterize",
                self.scanprep._parameterize_scan_bc_pairs,
                bc,
            ),
        )
        M, K = cast(
            tuple[torch.Tensor, torch.Tensor],
            call_region(
                "mixer.scanprep.coefficients",
                self.scanprep._make_scan_coefficients_from_flat_params,
                params,
                batch=batch,
                T=T,
            ),
        )
        B, C = call_region(
            "mixer.scanprep.pack_bc",
            self.scanprep._pack_scan_bc,
            B_pairs,
            C_pairs,
            batch=batch,
            T=T,
        )
        return ScanInputs(U=U, M=M, K=K, B=B, C=C)

    def _profile_scanprep(
        self, *, value: torch.Tensor, params: torch.Tensor, bc: torch.Tensor
    ) -> ScanInputs:
        if isinstance(self.scanprep.backend, ReferenceScanPrepBackend):
            return cast(
                ScanInputs,
                call_region(
                    "mixer.scanprep.total",
                    self._profile_reference_scanprep,
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

        gate, value_raw, params, bc_flat = call_region(
            "mixer.in_proj",
            self._emit_mixer_branches,
            x,
        )
        conv_state_in = None if state is None else state.conv
        conv_out, conv_state = call_region(
            "mixer.dw_conv",
            self.cconv_backend,
            self,
            value_raw,
            conv_state_in,
        )
        value = call_region(
            "mixer.dw_conv_activation", torch.nn.functional.silu, conv_out
        )
        bc = call_region(
            "mixer.bc_emit",
            self._emit_scan_bc,
            bc_flat,
            batch_size=batch,
            time_steps=T,
        )

        scan_inputs = self._profile_scanprep(value=value, params=params, bc=bc)
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

        out = call_region(
            "mixer.tail",
            mixer_tail,
            scan_y,
            gate,
            self.out_norm,
            self.out_proj,
        )

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
        expand: float,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        bc_groups: int,
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
            bc_groups=bc_groups,
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
        expand: float,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        bc_groups: int,
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
                    bc_groups=bc_groups,
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
