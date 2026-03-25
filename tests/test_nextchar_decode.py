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


def _cuda_decode_dtypes() -> list[torch.dtype]:
    dtypes = [torch.float16]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def _force_reference_sequence_backends(model: NextCharLM) -> None:
    for block in cast(list[NextCharBlock], list(model.blocks)):
        block.mixer.backend = ReferenceScanBackend(compute_dtype=torch.float32)
        block.mixer.scanprep.backend = ReferenceScanPrepBackend()
        block.mixer.cconv_backend = ReferenceCConv1dBackend()
        block.mixer.decode_backend = ReferenceMixerDecodeBackend()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _cuda_decode_dtypes())
def test_mixer_step_supported_cuda_matches_forward_without_calling_forward(
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
    x = torch.randn((2, 7, 128), device="cuda", dtype=dtype)

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
def test_nextchar_decode_one_matches_full_forward_on_supported_cuda(
    dtype: torch.dtype,
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
    idx = torch.randint(0, 64, (2, 6), device="cuda", dtype=torch.long)

    with torch.no_grad():
        logits_full = model(idx)
        decode_state = model.init_decode_state(2, device="cuda", dtype=dtype)
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
def test_nextchar_decode_one_supported_cuda_matches_reference_path_and_state(
    dtype: torch.dtype,
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
    idx = torch.randint(0, 64, (2, 6), device="cuda", dtype=torch.long)

    with torch.no_grad():
        state_fast = model.init_decode_state(2, device="cuda", dtype=dtype)
        fast_pieces: list[torch.Tensor] = []
        for t in range(idx.shape[1]):
            logits_t, state_fast = model.decode_one(idx[:, t], state_fast)
            fast_pieces.append(logits_t.unsqueeze(1))
        logits_fast = torch.cat(fast_pieces, dim=1)

        state_ref = model_ref.init_decode_state(2, device="cuda", dtype=dtype)
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
