"""Backbone stack built from SLinOSS blocks."""

from __future__ import annotations

from typing import cast

import torch
import torch.utils.checkpoint
from torch import nn

from .block import SLinOSSBlock, _apply_norm, _make_norm
from .config import SLinOSSBlockConfig, SLinOSSStackConfig
from .state import SLinOSSBlockState, SLinOSSStackState


class SLinOSSStack(nn.Module):
    """A compositional SLinOSS backbone without embeddings or heads."""

    def __init__(
        self,
        config: SLinOSSStackConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.blocks = nn.ModuleList(
            [
                self.build_block(index, block, device=device, dtype=dtype)
                for index, block in enumerate(config.blocks)
            ]
        )
        if config.final_norm_kind == "none":
            self.final_norm: nn.Module | None = None
        else:
            self.final_norm = _make_norm(
                config.final_norm_kind,
                config.d_model,
                eps=config.final_norm_eps,
                device=device,
                dtype=dtype,
            )

    def build_block(
        self,
        index: int,
        config: SLinOSSBlockConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSBlock:
        """Construct a block for this stack layer."""

        del index
        return SLinOSSBlock(config, device=device, dtype=dtype)

    def forward_block_context(
        self,
        index: int,
        hidden: torch.Tensor,
        *,
        state: SLinOSSStackState | None,
        context: object | None = None,
    ) -> object | None:
        """Resolve the opaque per-call context passed to a block in forward."""

        del index, hidden, state
        return context

    def step_block_context(
        self,
        index: int,
        hidden: torch.Tensor,
        *,
        state: SLinOSSStackState | None,
        context: object | None = None,
    ) -> object | None:
        """Resolve the opaque per-call context passed to a block in decode."""

        del index, hidden, state
        return context

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSStackState:
        return SLinOSSStackState(
            layers=[
                block.init_state(batch_size, device=device, dtype=dtype)
                for block in cast(list[SLinOSSBlock], list(self.blocks))
            ]
        )

    def init_decode_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSStackState:
        return SLinOSSStackState(
            layers=[
                block.init_decode_state(batch_size, device=device, dtype=dtype)
                for block in cast(list[SLinOSSBlock], list(self.blocks))
            ]
        )

    def _apply_final_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_norm is None:
            return x
        return _apply_norm(self.final_norm, x)

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: SLinOSSStackState | None = None,
        return_state: bool = False,
        context: object | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, SLinOSSStackState]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x shape (batch, T, {self.d_model}), got {tuple(x.shape)}."
            )

        if self.config.gradient_checkpointing and self.training:
            if state is not None or return_state:
                raise ValueError(
                    "gradient_checkpointing in SLinOSSStack requires "
                    "state=None and return_state=False."
                )
            hidden = x
            for index, block in enumerate(cast(list[SLinOSSBlock], list(self.blocks))):
                block_context = self.forward_block_context(
                    index,
                    hidden,
                    state=None,
                    context=context,
                )

                def custom_forward(
                    tensor: torch.Tensor,
                    module: SLinOSSBlock = block,
                    block_context: object | None = block_context,
                ) -> torch.Tensor:
                    return module(tensor, context=block_context)

                hidden = cast(
                    torch.Tensor,
                    torch.utils.checkpoint.checkpoint(
                        custom_forward,
                        hidden,
                        use_reentrant=False,
                    ),
                )
            return self._apply_final_norm(hidden).to(dtype=x.dtype)

        hidden = x
        next_layers: list[SLinOSSBlockState] = []
        if state is not None and len(state.layers) != len(self.blocks):
            raise ValueError(
                "State layer count must match stack depth. "
                f"Got {len(state.layers)} state layers for {len(self.blocks)} blocks."
            )
        for index, block in enumerate(cast(list[SLinOSSBlock], list(self.blocks))):
            layer_state = None if state is None else state.layers[index]
            block_context = self.forward_block_context(
                index,
                hidden,
                state=state,
                context=context,
            )
            if return_state:
                hidden, next_state = cast(
                    tuple[torch.Tensor, SLinOSSBlockState],
                    block(
                        hidden,
                        state=layer_state,
                        return_state=True,
                        context=block_context,
                    ),
                )
                next_layers.append(next_state)
            else:
                hidden = cast(
                    torch.Tensor,
                    block(
                        hidden,
                        state=layer_state,
                        return_state=False,
                        context=block_context,
                    ),
                )
        out = self._apply_final_norm(hidden).to(dtype=x.dtype)
        if not return_state:
            return out
        return out, SLinOSSStackState(layers=next_layers)

    def step(
        self,
        x: torch.Tensor,
        state: SLinOSSStackState | None = None,
        *,
        inplace: bool | None = None,
        context: object | None = None,
    ) -> tuple[torch.Tensor, SLinOSSStackState]:
        return_token = x.ndim == 2
        hidden = x
        if hidden.ndim == 3:
            if hidden.shape[1] != 1:
                raise ValueError(
                    f"Expected time dim 1 for stack.step, got {tuple(hidden.shape)}."
                )
            hidden = hidden[:, 0, :]
        if hidden.ndim != 2 or hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected hidden shape (batch, {self.d_model}), got {tuple(hidden.shape)}."
            )

        next_layers: list[SLinOSSBlockState] = []
        if state is not None and len(state.layers) != len(self.blocks):
            raise ValueError(
                "State layer count must match stack depth. "
                f"Got {len(state.layers)} state layers for {len(self.blocks)} blocks."
            )
        for index, block in enumerate(cast(list[SLinOSSBlock], list(self.blocks))):
            layer_state = None if state is None else state.layers[index]
            block_context = self.step_block_context(
                index,
                hidden,
                state=state,
                context=context,
            )
            hidden, next_state = block.step(
                hidden,
                layer_state,
                inplace=inplace,
                context=block_context,
            )
            next_layers.append(next_state)

        hidden = self._apply_final_norm(hidden).to(dtype=x.dtype)
        if not return_token:
            hidden = hidden.unsqueeze(1)
        return hidden, SLinOSSStackState(layers=next_layers)


__all__ = ["SLinOSSStack"]
