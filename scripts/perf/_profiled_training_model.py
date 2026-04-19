"""Perf-only instrumented LM shells built from SLinOSS blocks and stacks."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from slinoss.blocks import (
    SLinOSSBlock,
    SLinOSSBlockConfig,
    SLinOSSStack,
    SLinOSSStackConfig,
)
from slinoss.layers import SLinOSSMixer
from slinoss.layers.backend import (
    AutoScanBackend,
    CuteScanBackend,
    ScanInputs,
)
from slinoss.layers.state import SLinOSSMixerState, ScanState
from slinoss.ops.block import block_ffn_residual
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

    def _profile_scanprep(
        self,
        *,
        value: torch.Tensor,
        params: torch.Tensor,
        bc: torch.Tensor,
    ) -> ScanInputs:
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

        batch, time_steps, _ = map(int, x.shape)
        if time_steps == 0:
            empty = x.new_empty((batch, 0, self.d_model))
            next_state = SLinOSSMixerState() if state is None else state
            return (empty, next_state) if return_state else empty

        gate, value_proj, params_flat, bc_flat = call_region(
            "mixer.in_proj",
            self._emit_mixer_branches,
            x,
        )
        conv_state_in = None if state is None else state.conv
        conv_out, conv_state = call_region(
            "mixer.dw_conv",
            self.cconv_backend,
            self,
            value_proj,
            conv_state_in,
        )
        bc = call_region(
            "mixer.bc_emit",
            self._emit_scan_bc,
            bc_flat,
            batch_size=batch,
            time_steps=time_steps,
        )

        scan_inputs = self._profile_scanprep(value=conv_out, params=params_flat, bc=bc)
        scan_state_in = None if state is None else state.scan
        scan_result = call_region(
            "v2x2ssd.total",
            self._run_scan_backend,
            scan_inputs,
            scan_state_in,
            return_state,
        )
        if return_state:
            scan_output, scan_state = cast(tuple[torch.Tensor, ScanState], scan_result)
        else:
            scan_output = cast(torch.Tensor, scan_result)
            scan_state = None

        out = call_region(
            "mixer.tail",
            mixer_tail,
            scan_output,
            gate,
            self.out_norm,
            self.out_proj,
        )

        if not return_state:
            return out
        return out, SLinOSSMixerState(conv=conv_state, scan=cast(ScanState, scan_state))


class ProfiledSLinOSSBlock(SLinOSSBlock):
    def __init__(
        self,
        config: SLinOSSBlockConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        self.mixer = ProfiledSLinOSSMixer(
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: object | None = None,
        return_state: bool = False,
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        if state is not None or return_state:
            raise ValueError(
                "ProfiledSLinOSSBlock only supports the stateless training path."
            )
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x shape (batch, T, {self.d_model}), got {tuple(x.shape)}."
            )
        norm1 = call_region("norms.pre_mixer", self.mixer_inputs, x)
        mixed = cast(torch.Tensor, self.mixer(norm1))
        out = call_region(
            "residual.mixer", self._residual_add, x, self._drop_branch(mixed)
        )
        if self.ffn is None or self.ffn_norm is None:
            return out
        # Keep profiling on the same FFN execution path as production block.forward.
        # This avoids harness drift for memory-forensics work.
        return cast(
            torch.Tensor,
            call_region("ffn", block_ffn_residual, self, out),
        )


class ProfiledSLinOSSStack(SLinOSSStack):
    def __init__(
        self,
        config: SLinOSSStackConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                ProfiledSLinOSSBlock(block, device=device, dtype=dtype)
                for block in config.blocks
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: object | None = None,
        return_state: bool = False,
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        if state is not None or return_state:
            raise ValueError(
                "ProfiledSLinOSSStack only supports the stateless training path."
            )
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x shape (batch, T, {self.d_model}), got {tuple(x.shape)}."
            )
        hidden = x
        for block in cast(list[ProfiledSLinOSSBlock], list(self.blocks)):
            hidden = block(hidden)
        if self.final_norm is not None:
            hidden = call_region("norms.final", self._apply_final_norm, hidden)
        return hidden.to(dtype=x.dtype)


class ProfiledTrainingLM(nn.Module):
    """Perf-only instrumented LM shell layered on the public stack surface."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        stack_config: SLinOSSStackConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(stack_config.d_model)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.token_embed = nn.Embedding(vocab_size, self.d_model, **factory_kwargs)
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.seq_len, self.d_model, **factory_kwargs)
        )
        self.backbone = ProfiledSLinOSSStack(
            stack_config,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False, **factory_kwargs)
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

    def _add_pos_embed(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x + self.pos_embed[:, :seq_len, :]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (batch, T), got {tuple(idx.shape)}.")
        if idx.shape[1] > self.seq_len:
            raise ValueError(
                f"Sequence length {idx.shape[1]} exceeds seq_len {self.seq_len}."
            )
        hidden = call_region("embed.token", self.token_embed, idx)
        hidden = call_region(
            "embed.pos", self._add_pos_embed, hidden, int(idx.shape[1])
        )
        hidden = self.backbone(hidden)
        return call_region("head.logits", self.lm_head, hidden)


__all__ = [
    "ProfiledSLinOSSMixer",
    "ProfiledTrainingLM",
]
