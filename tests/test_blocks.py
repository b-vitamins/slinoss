from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
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
from slinoss.ops.block import block_ffn_residual


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


def test_block_ffn_residual_matches_eager_value_and_grads() -> None:
    torch.manual_seed(0)
    block = SLinOSSBlock(_small_block_config())
    block.train()
    x = torch.randn(2, 4, 32, requires_grad=True)

    ffn_params: list[torch.Tensor] = []
    if block.ffn_norm is not None:
        weight = getattr(block.ffn_norm, "weight", None)
        if isinstance(weight, torch.Tensor):
            ffn_params.append(weight)
        bias = getattr(block.ffn_norm, "bias", None)
        if isinstance(bias, torch.Tensor):
            ffn_params.append(bias)
    if block.ffn is not None:
        ffn_params.extend(
            [
                block.ffn.in_proj.weight,
                block.ffn.out_proj.weight,
            ]
        )
        if block.ffn.in_proj.bias is not None:
            ffn_params.append(block.ffn.in_proj.bias)
        if block.ffn.out_proj.bias is not None:
            ffn_params.append(block.ffn.out_proj.bias)

    eager = block._residual_add(
        x,
        block._drop_branch(block.forward_ffn_branch(x)),
    )
    eager_loss = eager.square().mean()
    eager_grads = torch.autograd.grad(
        eager_loss,
        (x, *ffn_params),
        retain_graph=True,
    )

    remat = block_ffn_residual(block, x)
    remat_loss = remat.square().mean()
    remat_grads = torch.autograd.grad(remat_loss, (x, *ffn_params))

    assert torch.allclose(remat, eager, atol=1.0e-5, rtol=1.0e-5)
    for ref, got in zip(eager_grads, remat_grads, strict=True):
        assert ref is not None
        assert got is not None
        assert torch.allclose(got, ref, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.parametrize("kind", ["gelu", "swiglu"])
def test_block_ffn_residual_cuda_cute_matches_eager_value_and_grads(
    kind: str,
) -> None:
    pytest.importorskip("cutlass")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CuTe FFN path.")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 support is required for the CuTe FFN path.")

    torch.manual_seed(0)
    cfg = replace(
        _small_block_config(),
        ffn=SLinOSSMLPConfig(
            kind=kind,  # type: ignore[arg-type]
            hidden_dim=48,
            multiple_of=16,
            bias=True,
        ),
    )
    block = SLinOSSBlock(cfg, device="cuda", dtype=torch.bfloat16).train()
    x = torch.randn((2, 4, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    ffn_params: list[torch.Tensor] = []
    if block.ffn_norm is not None:
        weight = getattr(block.ffn_norm, "weight", None)
        if isinstance(weight, torch.Tensor):
            ffn_params.append(weight)
    if block.ffn is not None:
        ffn_params.extend(
            [
                block.ffn.in_proj.weight,
                block.ffn.out_proj.weight,
            ]
        )
        if block.ffn.in_proj.bias is not None:
            ffn_params.append(block.ffn.in_proj.bias)
        if block.ffn.out_proj.bias is not None:
            ffn_params.append(block.ffn.out_proj.bias)

    eager = block._residual_add(
        x_ref,
        block._drop_branch(block.forward_ffn_branch(x_ref)),
    )
    eager_loss = eager.square().mean()
    eager_grads = torch.autograd.grad(
        eager_loss,
        (x_ref, *ffn_params),
        retain_graph=True,
    )

    remat = block_ffn_residual(block, x)
    remat_loss = remat.square().mean()
    remat_grads = torch.autograd.grad(remat_loss, (x, *ffn_params))

    torch.testing.assert_close(eager, remat, atol=6e-2, rtol=6e-2)
    for ref, got in zip(eager_grads, remat_grads, strict=True):
        assert ref is not None
        assert got is not None
        torch.testing.assert_close(ref, got, atol=8e-2, rtol=8e-2)


def test_block_ffn_residual_cuda_cute_saves_only_residual_and_parameters() -> None:
    pytest.importorskip("cutlass")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CuTe FFN path.")

    torch.manual_seed(0)
    cfg = replace(
        _small_block_config(),
        ffn=SLinOSSMLPConfig(kind="gelu", hidden_dim=48, multiple_of=16, bias=True),
    )
    block = SLinOSSBlock(cfg, device="cuda", dtype=torch.float16).train()
    assert block.ffn_norm is not None
    assert block.ffn_norm.weight is not None
    assert block.ffn is not None
    assert block.ffn.in_proj.bias is not None
    assert block.ffn.out_proj.bias is not None
    norm_weight = cast(torch.Tensor, block.ffn_norm.weight)
    in_proj_weight = cast(torch.Tensor, block.ffn.in_proj.weight)
    in_proj_bias = cast(torch.Tensor, block.ffn.in_proj.bias)
    out_proj_weight = cast(torch.Tensor, block.ffn.out_proj.weight)
    out_proj_bias = cast(torch.Tensor, block.ffn.out_proj.bias)
    x = torch.randn((2, 4, 32), device="cuda", dtype=torch.float16, requires_grad=True)

    saved: list[tuple[tuple[int, ...], torch.dtype]] = []

    def pack_hook(t: torch.Tensor) -> torch.Tensor:
        saved.append((tuple(map(int, t.shape)), t.dtype))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        block_ffn_residual(block, x)

    expected = [
        (tuple(map(int, x.shape)), x.dtype),
        (tuple(map(int, norm_weight.shape)), norm_weight.dtype),
        (tuple(map(int, in_proj_weight.shape)), in_proj_weight.dtype),
        (tuple(map(int, in_proj_bias.shape)), in_proj_bias.dtype),
        (tuple(map(int, out_proj_weight.shape)), out_proj_weight.dtype),
        (tuple(map(int, out_proj_bias.shape)), out_proj_bias.dtype),
    ]
    assert saved == expected


def test_block_ffn_residual_compile_boundary_is_explicit() -> None:
    pytest.importorskip("cutlass")
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    cfg = replace(
        _small_block_config(),
        ffn=SLinOSSMLPConfig(kind="gelu", hidden_dim=48, multiple_of=16, bias=True),
    )
    block = SLinOSSBlock(cfg, device="cuda", dtype=torch.float16).eval()
    x = torch.randn((2, 4, 32), device="cuda", dtype=torch.float16)

    explain = torch._dynamo.explain(lambda t: block_ffn_residual(block, t))
    result = explain(x)

    if len(result.break_reasons) == 0:
        assert result.graph_count == 0
        assert result.graph_break_count == -1
    else:
        assert result.graph_break_count >= len(result.break_reasons)
        assert all(
            "torch.compiler.disable" in break_reason.reason
            for break_reason in result.break_reasons
        )
