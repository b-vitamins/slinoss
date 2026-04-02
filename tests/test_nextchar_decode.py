from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
import torch

from slinoss.layers import (
    ReferenceMixerDecodeBackend,
    ReferenceCConv1dBackend,
    ReferenceScanBackend,
    ReferenceScanPrepBackend,
    SLinOSSMixer,
)
from slinoss.models import NextCharBlock, NextCharLM
from slinoss.ops.v2x2ssd.cute.decode import mixer_decode_step_cute


def _cuda_decode_dtypes() -> list[torch.dtype]:
    dtypes = [torch.float16]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def _make_noncontiguous_clone(x: torch.Tensor) -> torch.Tensor:
    padded = torch.empty((*x.shape, 2), device=x.device, dtype=x.dtype)
    padded[..., 0].copy_(x)
    padded[..., 1].zero_()
    out = padded[..., 0]
    assert tuple(out.shape) == tuple(x.shape)
    assert not out.is_contiguous()
    return out


def _force_reference_sequence_backends(model: NextCharLM) -> None:
    for block in cast(list[NextCharBlock], list(model.blocks)):
        block.mixer.backend = ReferenceScanBackend(compute_dtype=torch.float32)
        block.mixer.scanprep.backend = ReferenceScanPrepBackend()
        block.mixer.cconv_backend = ReferenceCConv1dBackend()
        block.mixer.decode_backend = ReferenceMixerDecodeBackend()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_mixer_step_supported_cuda_matches_forward_without_calling_forward(
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    x = torch.randn((batch_size, 7, 128), device="cuda", dtype=dtype)

    with torch.no_grad():
        y_full, state_full = mixer(x, return_state=True)

        def _raise_forward(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError(
                "supported decode step should not route through forward"
            )

        mixer.forward = _raise_forward  # type: ignore[method-assign]

        state_step = mixer.init_state(x.shape[0], device="cuda", dtype=dtype)
        pieces: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            y_t, state_step = mixer.step(x[:, t, :], state_step)
            pieces.append(y_t.unsqueeze(1))
        y_step = torch.cat(pieces, dim=1)

    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    torch.testing.assert_close(y_step, y_full, atol=atol, rtol=0.0)
    assert state_full.conv is not None and state_step.conv is not None
    assert state_full.scan.state is not None and state_step.scan.state is not None
    assert state_full.scan.b_prev is not None and state_step.scan.b_prev is not None
    assert state_full.scan.u_prev is not None and state_step.scan.u_prev is not None
    assert state_step._engine is not None
    assert state_step.scan.state.stride()[-2:] == (1, mixer.d_head)
    torch.testing.assert_close(state_step.conv, state_full.conv, atol=atol, rtol=0.0)
    torch.testing.assert_close(
        state_step.scan.state, state_full.scan.state, atol=atol, rtol=0.0
    )
    torch.testing.assert_close(
        state_step.scan.b_prev, state_full.scan.b_prev, atol=atol, rtol=0.0
    )
    torch.testing.assert_close(
        state_step.scan.u_prev, state_full.scan.u_prev, atol=atol, rtol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_init_decode_state_uses_decode_layout_only_for_cute_path(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    fast_state = mixer.init_decode_state(1, device="cuda", dtype=dtype)
    assert fast_state.scan.state is not None
    assert tuple(fast_state.scan.state.shape) == (1, mixer.n_heads, mixer.d_head, 128)
    assert not fast_state.scan.state.is_contiguous()
    assert fast_state.scan.state.stride()[-2:] == (1, mixer.d_head)

    mixer.decode_backend = ReferenceMixerDecodeBackend()
    ref_state = mixer.init_decode_state(1, device="cuda", dtype=dtype)
    assert ref_state.scan.state is not None
    assert ref_state.scan.state.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_nextchar_decode_one_matches_full_forward_on_supported_cuda(
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    model = (
        NextCharLM(
            vocab_size=64,
            block_size=16,
            d_model=128,
            n_layers=2,
            d_state=64,
            expand=2,
            d_head=64,
            d_conv=4,
            chunk_size=32,
        )
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    idx = torch.randint(0, 64, (batch_size, 6), device="cuda", dtype=torch.long)

    with torch.no_grad():
        logits_full = model(idx)
        decode_state = model.init_decode_state(batch_size, device="cuda", dtype=dtype)
        pieces: list[torch.Tensor] = []
        for t in range(idx.shape[1]):
            logits_t, decode_state = model.decode_one(idx[:, t], decode_state)
            pieces.append(logits_t.unsqueeze(1))
        logits_decode = torch.cat(pieces, dim=1)

    assert decode_state.position == idx.shape[1]
    assert decode_state._engine is not None
    atol = 8e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    torch.testing.assert_close(logits_decode, logits_full, atol=atol, rtol=0.0)


def test_nextchar_decode_one_falls_back_on_cpu() -> None:
    torch.manual_seed(0)

    model = NextCharLM(
        vocab_size=32,
        block_size=8,
        d_model=32,
        n_layers=2,
        d_state=16,
        expand=2,
        d_head=32,
        d_conv=4,
        chunk_size=8,
    ).eval()
    idx = torch.randint(0, 32, (2, 5), dtype=torch.long)

    with torch.no_grad():
        logits_full = model(idx)
        decode_state = model.init_decode_state(2)
        pieces: list[torch.Tensor] = []
        for t in range(idx.shape[1]):
            logits_t, decode_state = model.decode_one(idx[:, t], decode_state)
            pieces.append(logits_t.unsqueeze(1))
        logits_decode = torch.cat(pieces, dim=1)

    assert decode_state._engine is None
    torch.testing.assert_close(logits_decode, logits_full, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nextchar_decode_one_falls_back_for_unsupported_cuda_shape() -> None:
    torch.manual_seed(0)

    model = (
        NextCharLM(
            vocab_size=32,
            block_size=8,
            d_model=64,
            n_layers=2,
            d_state=16,
            expand=2,
            d_head=32,
            d_conv=4,
            chunk_size=8,
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    _force_reference_sequence_backends(model)
    idx = torch.randint(0, 32, (2, 5), device="cuda", dtype=torch.long)

    with torch.no_grad():
        logits_full = model(idx)
        decode_state = model.init_decode_state(2, device="cuda", dtype=torch.float16)
        pieces: list[torch.Tensor] = []
        for t in range(idx.shape[1]):
            logits_t, decode_state = model.decode_one(idx[:, t], decode_state)
            pieces.append(logits_t.unsqueeze(1))
        logits_decode = torch.cat(pieces, dim=1)

    assert decode_state._engine is None
    torch.testing.assert_close(logits_decode, logits_full, atol=5e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_step_supported_cuda_matches_reference_backends(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    mixer_ref = deepcopy(mixer).eval()
    mixer_ref.backend = ReferenceScanBackend(compute_dtype=torch.float32)
    mixer_ref.scanprep.backend = ReferenceScanPrepBackend()
    mixer_ref.cconv_backend = ReferenceCConv1dBackend()
    mixer_ref.decode_backend = ReferenceMixerDecodeBackend()
    x = torch.randn((2, 7, 128), device="cuda", dtype=dtype)

    with torch.no_grad():
        state_fast = mixer.init_state(x.shape[0], device="cuda", dtype=dtype)
        fast_pieces: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            y_t, state_fast = mixer.step(x[:, t, :], state_fast)
            fast_pieces.append(y_t.unsqueeze(1))
        y_fast = torch.cat(fast_pieces, dim=1)

        state_ref = mixer_ref.init_state(x.shape[0], device="cuda", dtype=dtype)
        ref_pieces: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            y_t, state_ref = mixer_ref.step(x[:, t, :], state_ref)
            ref_pieces.append(y_t.unsqueeze(1))
        y_ref = torch.cat(ref_pieces, dim=1)

    atol = 6e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    torch.testing.assert_close(y_fast, y_ref, atol=atol, rtol=0.0)
    assert state_fast.conv is not None and state_ref.conv is not None
    assert state_fast.scan.state is not None and state_ref.scan.state is not None
    assert state_fast.scan.b_prev is not None and state_ref.scan.b_prev is not None
    assert state_fast.scan.u_prev is not None and state_ref.scan.u_prev is not None
    torch.testing.assert_close(state_fast.conv, state_ref.conv, atol=atol, rtol=0.0)
    torch.testing.assert_close(
        state_fast.scan.state, state_ref.scan.state, atol=atol, rtol=0.0
    )
    torch.testing.assert_close(
        state_fast.scan.b_prev, state_ref.scan.b_prev, atol=atol, rtol=0.0
    )
    torch.testing.assert_close(
        state_fast.scan.u_prev, state_ref.scan.u_prev, atol=atol, rtol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_step_supported_cuda_reuses_scan_state_buffers(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    x = torch.randn((2, 128), device="cuda", dtype=dtype)

    with torch.no_grad():
        state = mixer.init_state(x.shape[0], device="cuda", dtype=dtype)
        assert state.scan.state is not None
        assert state.scan.b_prev is not None
        assert state.scan.u_prev is not None

        state_ptr = state.scan.state.data_ptr()
        b_prev_ptr = state.scan.b_prev.data_ptr()
        u_prev_ptr = state.scan.u_prev.data_ptr()

        y, returned_state = mixer.step(x, state)
        assert state.scan.state is not None
        promoted_state_ptr = state.scan.state.data_ptr()
        promoted_b_prev_ptr = state.scan.b_prev.data_ptr()
        promoted_u_prev_ptr = state.scan.u_prev.data_ptr()
        y2, returned_state2 = mixer.step(x, state)

    assert y.shape == (2, 128)
    assert y2.shape == (2, 128)
    assert returned_state is state
    assert returned_state2 is state
    assert state.scan.state is not None
    assert state.scan.b_prev is not None
    assert state.scan.u_prev is not None
    assert state_ptr != promoted_state_ptr
    assert state.scan.state.stride()[-2:] == (1, mixer.d_head)
    assert promoted_b_prev_ptr == b_prev_ptr
    assert promoted_u_prev_ptr == u_prev_ptr
    assert state.scan.state.data_ptr() == promoted_state_ptr
    assert state.scan.b_prev.data_ptr() == promoted_b_prev_ptr
    assert state.scan.u_prev.data_ptr() == promoted_u_prev_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_step_supported_cuda_inplace_false_preserves_input_state(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    x = torch.randn((2, 128), device="cuda", dtype=dtype)

    with torch.no_grad():
        state = mixer.init_state(x.shape[0], device="cuda", dtype=dtype)
        assert state.scan.state is not None
        assert state.scan.b_prev is not None
        assert state.scan.u_prev is not None
        state_before = state.clone()

        _, next_state = mixer.step(x, state, inplace=False)

    assert next_state is not state
    assert state.scan.state is not None and state_before.scan.state is not None
    assert state.scan.b_prev is not None and state_before.scan.b_prev is not None
    assert state.scan.u_prev is not None and state_before.scan.u_prev is not None
    torch.testing.assert_close(state.scan.state, state_before.scan.state)
    torch.testing.assert_close(state.scan.b_prev, state_before.scan.b_prev)
    torch.testing.assert_close(state.scan.u_prev, state_before.scan.u_prev)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_decode_step_cute_matches_noncontiguous_prev_inputs(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    batch = 2
    x = torch.randn((batch, 128), device="cuda", dtype=dtype)

    with torch.no_grad():
        state0 = mixer.init_state(batch, device="cuda", dtype=dtype)
        proj = mixer.in_proj(x)
        gate, value_raw, params_flat, bc_flat = torch.split(
            proj,
            [mixer.d_inner, mixer.d_inner, mixer.param_proj_dim, mixer.bc_proj_dim],
            dim=-1,
        )
        value, _ = mixer._apply_causal_depthwise_conv_step(value_raw, state0.conv)
        value_h = value.view(batch, mixer.n_heads, mixer.d_head).contiguous()
        params_h = params_flat.view(
            batch, mixer.n_heads, mixer.scanprep.param_dim
        ).contiguous()
        bc_h = bc_flat.view(batch, mixer.n_heads, 4, mixer.d_state).contiguous()
        gate_h = gate.view(batch, mixer.n_heads, mixer.d_head).contiguous()
        skip = mixer.skip.view(mixer.n_heads, mixer.d_head)

        initial_states = torch.randn(
            (batch, mixer.n_heads, mixer.d_head, 128),
            device="cuda",
            dtype=dtype,
        )
        b_prev = torch.randn(
            (batch, mixer.n_heads, 128),
            device="cuda",
            dtype=dtype,
        )
        u_prev = torch.randn(
            (batch, mixer.n_heads, mixer.d_head),
            device="cuda",
            dtype=dtype,
        )

        y_ref, final_ref, b_last_ref, u_last_ref = mixer_decode_step_cute(
            value_h,
            params_h,
            bc_h,
            gate_h,
            skip,
            initial_states=initial_states,
            B_prev=b_prev,
            U_prev=u_prev,
            dt_min=mixer.scanprep.dt_min,
            dt_max=mixer.scanprep.dt_max,
            r_min=mixer.scanprep.r_min,
            r_max=mixer.scanprep.r_max,
            theta_bound=mixer.scanprep.theta_bound,
            k_max=mixer.scanprep.k_max,
            eps=mixer.scanprep.eps,
            dt_bias=mixer.scanprep.dt_bias,
            gamma_bias=mixer.scanprep.gamma_bias,
            omega_bias=mixer.scanprep.omega_bias,
            mix_r_bias=mixer.scanprep.mix_r_bias,
            mix_theta_bias=mixer.scanprep.mix_theta_bias,
            mix_k_prev_bias=mixer.scanprep.mix_k_prev_bias,
            mix_k_curr_bias=mixer.scanprep.mix_k_curr_bias,
            b_scale=mixer.scanprep.b_scale,
            c_scale=mixer.scanprep.c_scale,
            output_dtype=dtype,
        )
        y_nc, final_nc, b_last_nc, u_last_nc = mixer_decode_step_cute(
            value_h,
            params_h,
            bc_h,
            gate_h,
            skip,
            initial_states=initial_states,
            B_prev=_make_noncontiguous_clone(b_prev),
            U_prev=_make_noncontiguous_clone(u_prev),
            dt_min=mixer.scanprep.dt_min,
            dt_max=mixer.scanprep.dt_max,
            r_min=mixer.scanprep.r_min,
            r_max=mixer.scanprep.r_max,
            theta_bound=mixer.scanprep.theta_bound,
            k_max=mixer.scanprep.k_max,
            eps=mixer.scanprep.eps,
            dt_bias=mixer.scanprep.dt_bias,
            gamma_bias=mixer.scanprep.gamma_bias,
            omega_bias=mixer.scanprep.omega_bias,
            mix_r_bias=mixer.scanprep.mix_r_bias,
            mix_theta_bias=mixer.scanprep.mix_theta_bias,
            mix_k_prev_bias=mixer.scanprep.mix_k_prev_bias,
            mix_k_curr_bias=mixer.scanprep.mix_k_curr_bias,
            b_scale=mixer.scanprep.b_scale,
            c_scale=mixer.scanprep.c_scale,
            output_dtype=dtype,
        )

    torch.testing.assert_close(y_nc, y_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(final_nc, final_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_last_nc, b_last_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(u_last_nc, u_last_ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_decode_step_cute_matches_noncontiguous_output_buffers(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    batch = 2
    x = torch.randn((batch, 128), device="cuda", dtype=dtype)

    with torch.no_grad():
        state0 = mixer.init_state(batch, device="cuda", dtype=dtype)
        proj = mixer.in_proj(x)
        gate, value_raw, params_flat, bc_flat = torch.split(
            proj,
            [mixer.d_inner, mixer.d_inner, mixer.param_proj_dim, mixer.bc_proj_dim],
            dim=-1,
        )
        value, _ = mixer._apply_causal_depthwise_conv_step(value_raw, state0.conv)
        value_h = value.view(batch, mixer.n_heads, mixer.d_head).contiguous()
        params_h = params_flat.view(
            batch, mixer.n_heads, mixer.scanprep.param_dim
        ).contiguous()
        bc_h = bc_flat.view(batch, mixer.n_heads, 4, mixer.d_state).contiguous()
        gate_h = gate.view(batch, mixer.n_heads, mixer.d_head).contiguous()
        skip = mixer.skip.view(mixer.n_heads, mixer.d_head)

        initial_states = torch.randn(
            (batch, mixer.n_heads, mixer.d_head, 128),
            device="cuda",
            dtype=dtype,
        )
        b_prev = torch.randn(
            (batch, mixer.n_heads, 128),
            device="cuda",
            dtype=dtype,
        )
        u_prev = torch.randn(
            (batch, mixer.n_heads, mixer.d_head),
            device="cuda",
            dtype=dtype,
        )

        y_ref, final_ref, b_last_ref, u_last_ref = mixer_decode_step_cute(
            value_h,
            params_h,
            bc_h,
            gate_h,
            skip,
            initial_states=initial_states,
            B_prev=b_prev,
            U_prev=u_prev,
            dt_min=mixer.scanprep.dt_min,
            dt_max=mixer.scanprep.dt_max,
            r_min=mixer.scanprep.r_min,
            r_max=mixer.scanprep.r_max,
            theta_bound=mixer.scanprep.theta_bound,
            k_max=mixer.scanprep.k_max,
            eps=mixer.scanprep.eps,
            dt_bias=mixer.scanprep.dt_bias,
            gamma_bias=mixer.scanprep.gamma_bias,
            omega_bias=mixer.scanprep.omega_bias,
            mix_r_bias=mixer.scanprep.mix_r_bias,
            mix_theta_bias=mixer.scanprep.mix_theta_bias,
            mix_k_prev_bias=mixer.scanprep.mix_k_prev_bias,
            mix_k_curr_bias=mixer.scanprep.mix_k_curr_bias,
            b_scale=mixer.scanprep.b_scale,
            c_scale=mixer.scanprep.c_scale,
            output_dtype=dtype,
        )

        final_state_out = _make_noncontiguous_clone(final_ref)
        b_last_out = _make_noncontiguous_clone(b_prev)
        u_last_out = _make_noncontiguous_clone(u_prev)
        y_out, final_out, b_last_out_got, u_last_out_got = mixer_decode_step_cute(
            value_h,
            params_h,
            bc_h,
            gate_h,
            skip,
            initial_states=initial_states,
            B_prev=b_prev,
            U_prev=u_prev,
            dt_min=mixer.scanprep.dt_min,
            dt_max=mixer.scanprep.dt_max,
            r_min=mixer.scanprep.r_min,
            r_max=mixer.scanprep.r_max,
            theta_bound=mixer.scanprep.theta_bound,
            k_max=mixer.scanprep.k_max,
            eps=mixer.scanprep.eps,
            dt_bias=mixer.scanprep.dt_bias,
            gamma_bias=mixer.scanprep.gamma_bias,
            omega_bias=mixer.scanprep.omega_bias,
            mix_r_bias=mixer.scanprep.mix_r_bias,
            mix_theta_bias=mixer.scanprep.mix_theta_bias,
            mix_k_prev_bias=mixer.scanprep.mix_k_prev_bias,
            mix_k_curr_bias=mixer.scanprep.mix_k_curr_bias,
            b_scale=mixer.scanprep.b_scale,
            c_scale=mixer.scanprep.c_scale,
            output_dtype=dtype,
            final_state_out=final_state_out,
            b_last_out=b_last_out,
            u_last_out=u_last_out,
        )

    assert final_out.data_ptr() == final_state_out.data_ptr()
    assert b_last_out_got.data_ptr() == b_last_out.data_ptr()
    assert u_last_out_got.data_ptr() == u_last_out.data_ptr()
    torch.testing.assert_close(y_out, y_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(final_out, final_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_last_out_got, b_last_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(u_last_out_got, u_last_ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_decode_step_cute_allows_aliasing_state_and_b_prev_outputs(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        device="cuda",
        dtype=dtype,
    ).eval()
    x = torch.randn((2, 128), device="cuda", dtype=dtype)

    with torch.no_grad():
        state = mixer.init_state(x.shape[0], device="cuda", dtype=dtype)
        assert state.scan.state is not None
        assert state.scan.b_prev is not None
        assert state.scan.u_prev is not None
        proj = mixer.in_proj(x)
        gate, value_raw, params_flat, bc_flat = torch.split(
            proj,
            [mixer.d_inner, mixer.d_inner, mixer.param_proj_dim, mixer.bc_proj_dim],
            dim=-1,
        )
        value, _ = mixer._apply_causal_depthwise_conv_step(value_raw, state.conv)
        value_h = value.view(x.shape[0], mixer.n_heads, mixer.d_head).contiguous()
        params_h = params_flat.view(
            x.shape[0], mixer.n_heads, mixer.scanprep.param_dim
        ).contiguous()
        bc_h = bc_flat.view(x.shape[0], mixer.n_heads, 4, mixer.d_state).contiguous()
        gate_h = gate.view(x.shape[0], mixer.n_heads, mixer.d_head).contiguous()
        skip = mixer.skip.view(mixer.n_heads, mixer.d_head)

        state_ref = state.scan.state.clone()
        b_prev_ref = state.scan.b_prev.clone()
        u_prev_ref = state.scan.u_prev.clone()
        y_ref, final_ref, b_last_ref, u_last_ref = mixer_decode_step_cute(
            value_h,
            params_h,
            bc_h,
            gate_h,
            skip,
            initial_states=state_ref,
            B_prev=b_prev_ref,
            U_prev=u_prev_ref,
            dt_min=mixer.scanprep.dt_min,
            dt_max=mixer.scanprep.dt_max,
            r_min=mixer.scanprep.r_min,
            r_max=mixer.scanprep.r_max,
            theta_bound=mixer.scanprep.theta_bound,
            k_max=mixer.scanprep.k_max,
            eps=mixer.scanprep.eps,
            dt_bias=mixer.scanprep.dt_bias,
            gamma_bias=mixer.scanprep.gamma_bias,
            omega_bias=mixer.scanprep.omega_bias,
            mix_r_bias=mixer.scanprep.mix_r_bias,
            mix_theta_bias=mixer.scanprep.mix_theta_bias,
            mix_k_prev_bias=mixer.scanprep.mix_k_prev_bias,
            mix_k_curr_bias=mixer.scanprep.mix_k_curr_bias,
            b_scale=mixer.scanprep.b_scale,
            c_scale=mixer.scanprep.c_scale,
            output_dtype=dtype,
        )

        state_alias = state.scan.state.clone()
        b_prev_alias = state.scan.b_prev.clone()
        u_prev_alias = state.scan.u_prev.clone()
        y_alias, final_alias, b_last_alias, u_last_alias = mixer_decode_step_cute(
            value_h,
            params_h,
            bc_h,
            gate_h,
            skip,
            initial_states=state_alias,
            B_prev=b_prev_alias,
            U_prev=u_prev_alias,
            dt_min=mixer.scanprep.dt_min,
            dt_max=mixer.scanprep.dt_max,
            r_min=mixer.scanprep.r_min,
            r_max=mixer.scanprep.r_max,
            theta_bound=mixer.scanprep.theta_bound,
            k_max=mixer.scanprep.k_max,
            eps=mixer.scanprep.eps,
            dt_bias=mixer.scanprep.dt_bias,
            gamma_bias=mixer.scanprep.gamma_bias,
            omega_bias=mixer.scanprep.omega_bias,
            mix_r_bias=mixer.scanprep.mix_r_bias,
            mix_theta_bias=mixer.scanprep.mix_theta_bias,
            mix_k_prev_bias=mixer.scanprep.mix_k_prev_bias,
            mix_k_curr_bias=mixer.scanprep.mix_k_curr_bias,
            b_scale=mixer.scanprep.b_scale,
            c_scale=mixer.scanprep.c_scale,
            output_dtype=dtype,
            final_state_out=state_alias,
            b_last_out=b_prev_alias,
            u_last_out=u_prev_alias,
        )

    atol = 6e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    assert final_alias.data_ptr() == state_alias.data_ptr()
    assert b_last_alias.data_ptr() == b_prev_alias.data_ptr()
    assert u_last_alias.data_ptr() == u_prev_alias.data_ptr()
    torch.testing.assert_close(y_alias, y_ref, atol=atol, rtol=0.0)
    torch.testing.assert_close(final_alias, final_ref, atol=atol, rtol=0.0)
    torch.testing.assert_close(b_last_alias, b_last_ref, atol=atol, rtol=0.0)
    torch.testing.assert_close(u_last_alias, u_last_ref, atol=atol, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
@pytest.mark.parametrize("batch_size", [1, 2])
def test_nextchar_decode_one_supported_cuda_matches_reference_path_and_state(
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    model = (
        NextCharLM(
            vocab_size=64,
            block_size=16,
            d_model=128,
            n_layers=2,
            d_state=64,
            expand=2,
            d_head=64,
            d_conv=4,
            chunk_size=32,
        )
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    model_ref = deepcopy(model).eval()
    _force_reference_sequence_backends(model_ref)
    idx = torch.randint(0, 64, (batch_size, 6), device="cuda", dtype=torch.long)

    with torch.no_grad():
        state_fast = model.init_decode_state(batch_size, device="cuda", dtype=dtype)
        fast_pieces: list[torch.Tensor] = []
        for t in range(idx.shape[1]):
            logits_t, state_fast = model.decode_one(idx[:, t], state_fast)
            fast_pieces.append(logits_t.unsqueeze(1))
        logits_fast = torch.cat(fast_pieces, dim=1)

        state_ref = model_ref.init_decode_state(batch_size, device="cuda", dtype=dtype)
        ref_pieces: list[torch.Tensor] = []
        for t in range(idx.shape[1]):
            logits_t, state_ref = model_ref._decode_one_eager_inplace(
                idx[:, t], state_ref
            )
            ref_pieces.append(logits_t.unsqueeze(1))
        logits_ref = torch.cat(ref_pieces, dim=1)

    assert state_fast.position == state_ref.position == idx.shape[1]
    assert state_fast._engine is not None
    assert state_ref._engine is None
    atol = 8e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    torch.testing.assert_close(logits_fast, logits_ref, atol=atol, rtol=0.0)
    for fast_layer, ref_layer in zip(state_fast.layers, state_ref.layers, strict=True):
        assert fast_layer.conv is not None and ref_layer.conv is not None
        assert fast_layer.scan.state is not None and ref_layer.scan.state is not None
        assert fast_layer.scan.b_prev is not None and ref_layer.scan.b_prev is not None
        assert fast_layer.scan.u_prev is not None and ref_layer.scan.u_prev is not None
        torch.testing.assert_close(fast_layer.conv, ref_layer.conv, atol=atol, rtol=0.0)
        torch.testing.assert_close(
            fast_layer.scan.state,
            ref_layer.scan.state,
            atol=atol,
            rtol=0.0,
        )
        torch.testing.assert_close(
            fast_layer.scan.b_prev,
            ref_layer.scan.b_prev,
            atol=atol,
            rtol=0.0,
        )
        torch.testing.assert_close(
            fast_layer.scan.u_prev,
            ref_layer.scan.u_prev,
            atol=atol,
            rtol=0.0,
        )
