from __future__ import annotations

import torch

from slinoss.blocks import (
    SLinOSSBlock,
    SLinOSSBlockConfig,
    SLinOSSMixerConfig,
    SLinOSSStack,
    SLinOSSStackConfig,
    sandwich_block_schedule,
    scaled_budget_schedule,
    uniform_block_schedule,
)
from slinoss.layers import SLinOSSMLPConfig


def _small_block_config(*, norm_kind: str = "rmsnorm") -> SLinOSSBlockConfig:
    return SLinOSSBlockConfig(
        d_model=32,
        mixer=SLinOSSMixerConfig(
            d_state=8,
            expand=1.0,
            d_head=8,
            d_conv=2,
            chunk_size=4,
            dt_min=1.0e-3,
            dt_max=1.0e-2,
            dt_init_floor=1.0e-3,
            r_min=0.8,
        ),
        ffn=SLinOSSMLPConfig(hidden_dim=48, multiple_of=16, bias=False),
        norm_kind=norm_kind,  # type: ignore[arg-type]
        residual_in_fp32=True,
    )


def test_uniform_block_schedule_repeats_block_config() -> None:
    block = _small_block_config()

    schedule = uniform_block_schedule(block, n_layers=3)

    assert schedule == (block, block, block)


def test_sandwich_block_schedule_places_edge_blocks() -> None:
    stem = _small_block_config()
    middle = _small_block_config()
    tail = _small_block_config(norm_kind="layernorm")

    schedule = sandwich_block_schedule(
        stem=stem,
        middle=middle,
        tail=tail,
        n_layers=4,
    )

    assert schedule[0] == stem
    assert schedule[-1] == tail
    assert schedule[1] == middle
    assert schedule[2] == middle


def test_scaled_budget_schedule_scales_requested_fields() -> None:
    base = SLinOSSBlockConfig(
        d_model=32,
        mixer=SLinOSSMixerConfig(
            d_state=8,
            expand=1.0,
            d_head=8,
            d_conv=2,
            chunk_size=4,
        ),
        ffn=SLinOSSMLPConfig(expand=2.0, multiple_of=16, bias=False),
    )

    schedule = scaled_budget_schedule(
        base,
        n_layers=3,
        mixer_expand_range=(1.0, 2.0),
        ffn_expand_range=(2.0, 3.0),
        residual_dropout_range=(0.0, 0.2),
    )

    assert schedule[0].mixer.expand == 1.0
    assert schedule[-1].mixer.expand == 2.0
    assert schedule[0].ffn is not None
    assert schedule[-1].ffn is not None
    assert schedule[0].ffn.expand == 2.0
    assert schedule[-1].ffn.expand == 3.0
    assert schedule[0].residual_dropout == 0.0
    assert schedule[-1].residual_dropout == 0.2


def test_block_forward_and_return_state_preserve_shape() -> None:
    torch.manual_seed(0)
    block = SLinOSSBlock(_small_block_config())
    x = torch.randn(2, 6, 32)

    out, state = block(x, return_state=True)

    assert out.shape == x.shape
    assert state.mixer.scan.state is not None


def test_block_step_matches_forward_tokenwise() -> None:
    torch.manual_seed(0)
    block = SLinOSSBlock(_small_block_config())
    x = torch.randn(2, 5, 32)

    expected = block(x)
    state = block.init_decode_state(2)
    actual_tokens: list[torch.Tensor] = []
    for t in range(x.shape[1]):
        token_out, state = block.step(x[:, t, :], state)
        actual_tokens.append(token_out.unsqueeze(1))
    actual = torch.cat(actual_tokens, dim=1)

    assert torch.allclose(actual, expected, atol=1.0e-4, rtol=1.0e-4)


def test_stack_forward_step_and_uniform_config() -> None:
    torch.manual_seed(0)
    block = _small_block_config()
    stack = SLinOSSStack(SLinOSSStackConfig.uniform(block, n_layers=2))
    x = torch.randn(2, 4, 32)

    expected = stack(x)
    state = stack.init_decode_state(2)
    actual_tokens: list[torch.Tensor] = []
    for t in range(x.shape[1]):
        token_out, state = stack.step(x[:, t, :], state)
        actual_tokens.append(token_out.unsqueeze(1))
    actual = torch.cat(actual_tokens, dim=1)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1.0e-4, rtol=1.0e-4)


def test_stack_supports_gradient_checkpointing_without_state() -> None:
    torch.manual_seed(0)
    block = _small_block_config()
    stack = SLinOSSStack(
        SLinOSSStackConfig.uniform(
            block,
            n_layers=2,
            gradient_checkpointing=True,
        )
    )
    stack.train()
    x = torch.randn(2, 4, 32, requires_grad=True)

    out = stack(x)
    loss = out.square().mean()
    loss.backward()

    assert x.grad is not None


def test_block_exposes_mixer_input_hook_for_downstream_specialization() -> None:
    class HookedBlock(SLinOSSBlock):
        def __init__(self, config: SLinOSSBlockConfig) -> None:
            super().__init__(config)
            self.forward_calls = 0
            self.step_calls = 0

        def mixer_inputs(
            self,
            x: torch.Tensor,
            *,
            context: object | None = None,
        ) -> torch.Tensor:
            del context
            if x.ndim == 3:
                self.forward_calls += 1
            else:
                self.step_calls += 1
            return super().mixer_inputs(x)

    torch.manual_seed(0)
    block = HookedBlock(_small_block_config())
    x = torch.randn(2, 3, 32)

    _ = block(x)
    state = block.init_decode_state(2)
    _ = block.step(x[:, 0, :], state)

    assert block.forward_calls == 1
    assert block.step_calls == 1


def test_stack_can_build_custom_blocks_and_thread_context() -> None:
    class ContextBlock(SLinOSSBlock):
        def __init__(
            self,
            config: SLinOSSBlockConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(config, device=device, dtype=dtype)
            self.forward_contexts: list[torch.Tensor] = []
            self.step_contexts: list[torch.Tensor] = []

        def mixer_inputs(
            self,
            x: torch.Tensor,
            *,
            context: object | None = None,
        ) -> torch.Tensor:
            if isinstance(context, torch.Tensor):
                if x.ndim == 3:
                    self.forward_contexts.append(context)
                else:
                    self.step_contexts.append(context)
            return super().mixer_inputs(x, context=context)

    class ContextStack(SLinOSSStack):
        def build_block(
            self,
            index: int,
            config: SLinOSSBlockConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
        ) -> SLinOSSBlock:
            del index
            return ContextBlock(config, device=device, dtype=dtype)

        def forward_block_context(
            self,
            index: int,
            hidden: torch.Tensor,
            *,
            state: object | None,
            context: object | None = None,
        ) -> object | None:
            del hidden, state
            if not isinstance(context, torch.Tensor):
                return context
            return context + index

        def step_block_context(
            self,
            index: int,
            hidden: torch.Tensor,
            *,
            state: object | None,
            context: object | None = None,
        ) -> object | None:
            del hidden, state
            if not isinstance(context, torch.Tensor):
                return context
            return context + index

    torch.manual_seed(0)
    block = _small_block_config()
    stack = ContextStack(SLinOSSStackConfig.uniform(block, n_layers=2))
    blocks = list(stack.blocks)
    assert all(isinstance(layer, ContextBlock) for layer in blocks)

    x = torch.randn(2, 3, 32)
    context = torch.randn(2, 3, 32)
    _ = stack(x, context=context)

    first = blocks[0]
    second = blocks[1]
    assert isinstance(first, ContextBlock)
    assert isinstance(second, ContextBlock)
    assert torch.equal(first.forward_contexts[0], context)
    assert torch.equal(second.forward_contexts[0], context + 1)

    step_context = torch.randn(2, 32)
    state = stack.init_decode_state(2)
    _ = stack.step(x[:, 0, :], state, context=step_context)

    assert torch.equal(first.step_contexts[0], step_context)
    assert torch.equal(second.step_contexts[0], step_context + 1)
