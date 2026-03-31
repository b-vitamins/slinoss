from __future__ import annotations

import math
import pytest
import torch

import slinoss.ops.v2x2ssd.cute.kernels.bwd as v2x2ssd_bwd_mod
from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd import (
    _resolve_chunk_scan_launch_cfg,
    compile_chunk_scan_kernel,
    chunk_scan_cute,
    state_passing_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.common import _make_ptr_arg
from slinoss.ops.v2x2ssd.reference import chunk_increment, chunk_scan, state_passing


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_scan_inputs(
    *,
    batch: int,
    heads: int,
    T: int,
    N: int,
    P: int,
    device: torch.device,
    value_dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=value_dtype)
    B = torch.randn((batch, heads, T, 2 * N), device=device, dtype=value_dtype) * 0.1
    C = torch.randn((batch, heads, T, 2 * N), device=device, dtype=value_dtype) * 0.1
    initial_states = torch.randn(
        (batch, heads, P, 2 * N), device=device, dtype=torch.float32
    )

    b_prev = (
        torch.randn((batch, heads, N), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, N), device=device, dtype=torch.float32)
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=torch.float32)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, C, initial_states, B_prev, U_prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_launch_cfg_falls_back_for_issue_3_shape() -> None:
    pytest.importorskip("cutlass")

    device_index = torch.cuda.current_device()
    cfg_issue3 = _resolve_chunk_scan_launch_cfg(
        D=512,
        P=64,
        L=64,
        tc_dtype=torch.float16,
        output_dtype=torch.float16,
        device_index=device_index,
        requested_m_block_size=None,
        requested_n_block_size=64,
        requested_num_threads=128,
    )
    # With D-staged shared-memory residency, the issue-3 shape now fits the
    # primary 64x64x128 family directly (no 64-thread fallback required).
    assert cfg_issue3 == (64, 64, 128)

    assert _resolve_chunk_scan_launch_cfg(
        D=256,
        P=64,
        L=64,
        tc_dtype=torch.float16,
        output_dtype=torch.float16,
        device_index=device_index,
        requested_m_block_size=None,
        requested_n_block_size=64,
        requested_num_threads=128,
    ) == (64, 64, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_compile_entrypoint_reuses_cache() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    inc_ref, m_ref = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    starts_ref, _ = state_passing(
        inc_ref,
        m_ref,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )

    compiled_a, out_chunk_a, out_view_a = compile_chunk_scan_kernel(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    compiled_b, out_chunk_b, out_view_b = compile_chunk_scan_kernel(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    assert compiled_a is compiled_b
    assert out_chunk_a.shape == out_chunk_b.shape == (8, 32, 1, 16)
    assert out_view_a.shape == out_view_b.shape == (2, 2, 64, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_fwd_ptr_cache_keeps_same_base_views_distinct() -> None:
    pytest.importorskip("cutlass")

    x = torch.empty((1, 1, 1, 2, 2), device="cuda", dtype=torch.float32)
    base_ptr, _ = _make_ptr_arg(x)
    slice0_ptr, _ = _make_ptr_arg(x[:, :, :, 0, :])
    slice1_ptr, _ = _make_ptr_arg(x[:, :, :, 1, :])

    assert base_ptr is not slice0_ptr
    assert base_ptr is not slice1_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_state_passing_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, chunks, N, P = 2, 2, 16, 8, 16
    D = 2 * N
    inc = torch.randn((batch, heads, chunks, P, D), device="cuda", dtype=torch.float32)
    radius = 0.6 + 0.35 * torch.rand((batch, heads, chunks), device="cuda")
    angle = (2.0 * math.pi) * torch.rand(
        (batch, heads, chunks), device="cuda"
    ) - math.pi
    m_chunk = (
        torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()
    )
    initial_states = torch.randn(
        (batch, heads, P, D), device="cuda", dtype=torch.float32
    )

    starts_ref, final_ref = state_passing(
        inc,
        m_chunk,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )
    starts_cute, final_cute = state_passing_cute(
        inc,
        m_chunk,
        initial_states=initial_states,
    )

    assert starts_cute.dtype == torch.float32
    assert final_cute.dtype == torch.float32
    torch.testing.assert_close(starts_cute, starts_ref, atol=4e-3, rtol=0.0)
    torch.testing.assert_close(final_cute, final_ref, atol=2e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    inc_ref, m_ref = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    starts_ref, _ = state_passing(
        inc_ref,
        m_ref,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )

    y_ref = chunk_scan(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    y_cute = chunk_scan_cute(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(y_cute, y_ref, atol=2e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_cute_is_repeatable_for_issue_9_shape() -> None:
    pytest.importorskip("cutlass")

    for output_dtype, atol in (
        (torch.bfloat16, 5e-2),
        (torch.float32, 5e-3),
    ):
        for _ in range(8):
            torch.manual_seed(0)
            U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
                batch=1,
                heads=16,
                T=128,
                N=128,
                P=64,
                device=torch.device("cuda"),
                value_dtype=torch.bfloat16,
            )

            inc_ref, m_ref = chunk_increment(
                U,
                M,
                K,
                B,
                B_prev=B_prev,
                U_prev=U_prev,
                T=U.shape[2],
                chunk_size=128,
                compute_dtype=torch.float32,
            )
            starts_ref, _ = state_passing(
                inc_ref,
                m_ref,
                initial_states=initial_states,
                compute_dtype=torch.float32,
            )

            y_ref = chunk_scan(
                U,
                M,
                K,
                B,
                C,
                starts_ref,
                B_prev=B_prev,
                U_prev=U_prev,
                T=U.shape[2],
                chunk_size=128,
                output_dtype=output_dtype,
                compute_dtype=torch.float32,
            )
            y_cute = chunk_scan_cute(
                U,
                M,
                K,
                B,
                C,
                starts_ref,
                B_prev=B_prev,
                U_prev=U_prev,
                chunk_size=128,
                output_dtype=output_dtype,
                compute_dtype=torch.float32,
            )

            torch.testing.assert_close(y_cute, y_ref, atol=atol, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_stateless_forward_is_mode_invariant() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, _initial_states, _B_prev, _U_prev = _make_scan_inputs(
        batch=2,
        heads=4,
        T=65,
        N=32,
        P=64,
        device=torch.device("cuda"),
        value_dtype=torch.float32,
    )
    U.requires_grad_()
    M.requires_grad_()
    K.requires_grad_()
    B.requires_grad_()
    C.requires_grad_()

    with torch.enable_grad():
        y_grad = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
    with torch.no_grad():
        y_no_grad = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
    with torch.inference_mode():
        y_infer = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )

    for y in (y_grad, y_no_grad, y_infer):
        assert torch.isfinite(y).all()

    torch.testing.assert_close(y_no_grad, y_grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(y_infer, y_grad, atol=0.0, rtol=0.0)

    with torch.no_grad():
        y_repeat = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
    torch.testing.assert_close(y_repeat, y_no_grad, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_stateful_forward_is_mode_invariant() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=4,
        T=65,
        N=32,
        P=64,
        device=torch.device("cuda"),
        value_dtype=torch.float32,
    )
    for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev):
        tensor.requires_grad_()

    with torch.enable_grad():
        out_grad = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            return_state=True,
        )
    with torch.no_grad():
        out_no_grad = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            return_state=True,
        )
    with torch.inference_mode():
        out_infer = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            return_state=True,
        )

    for output_tuple in (out_grad, out_no_grad, out_infer):
        for tensor in output_tuple:
            assert torch.isfinite(tensor).all()

    for got, want in zip(out_no_grad, out_grad, strict=True):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    for got, want in zip(out_infer, out_grad, strict=True):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)

    with torch.no_grad():
        out_repeat = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            return_state=True,
        )
    for got, want in zip(out_repeat, out_no_grad, strict=True):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_stateful_forward_is_repeatable_for_issue_9_shape() -> None:
    pytest.importorskip("cutlass")

    for _ in range(12):
        torch.manual_seed(0)
        U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
            batch=1,
            heads=16,
            T=128,
            N=128,
            P=64,
            device=torch.device("cuda"),
            value_dtype=torch.bfloat16,
        )

        out_ref = v2x2ssd(
            U,
            M,
            K,
            B,
            C,
            chunk_size=128,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        out_cute = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=128,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            return_state=True,
        )

        torch.testing.assert_close(out_cute[0], out_ref[0], atol=1e-1, rtol=0.0)
        torch.testing.assert_close(out_cute[1], out_ref[1], atol=1e-1, rtol=0.0)
        torch.testing.assert_close(out_cute[2], out_ref[2], atol=1e-1, rtol=0.0)
        torch.testing.assert_close(out_cute[3], out_ref[3], atol=1e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_training_backward_is_repeatable() -> None:
    pytest.importorskip("cutlass")

    def run_once() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)
        U, M, K, B, C, _initial_states, _B_prev, _U_prev = _make_scan_inputs(
            batch=2,
            heads=4,
            T=65,
            N=32,
            P=64,
            device=torch.device("cuda"),
            value_dtype=torch.float32,
        )
        U.requires_grad_()
        M.requires_grad_()
        K.requires_grad_()
        B.requires_grad_()
        C.requires_grad_()

        y = v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=64,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        loss = y.square().mean()
        loss.backward()

        assert U.grad is not None
        assert M.grad is not None
        assert K.grad is not None
        assert B.grad is not None
        assert C.grad is not None
        grads = (
            U.grad.detach().clone(),
            M.grad.detach().clone(),
            K.grad.detach().clone(),
            B.grad.detach().clone(),
            C.grad.detach().clone(),
        )
        return y.detach().clone(), grads

    y1, grads1 = run_once()
    y2, grads2 = run_once()

    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()
    torch.testing.assert_close(y1, y2, atol=0.0, rtol=0.0)

    for grad1, grad2 in zip(grads1, grads2, strict=True):
        assert torch.isfinite(grad1).all()
        assert torch.isfinite(grad2).all()
        torch.testing.assert_close(grad1, grad2, atol=1e-7, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_training_backward_supports_issue_5_shape() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, _initial_states, _B_prev, _U_prev = _make_scan_inputs(
        batch=1,
        heads=4,
        T=65,
        N=256,
        P=64,
        device=torch.device("cuda"),
        value_dtype=torch.float16,
    )
    U.requires_grad_()
    M.requires_grad_()
    K.requires_grad_()
    B.requires_grad_()
    C.requires_grad_()

    y = v2x2ssd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=64,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    loss = y.square().mean()
    loss.backward()

    assert torch.isfinite(y).all()
    for tensor in (U, M, K, B, C):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_cute_matches_reference_stage_issue_3_shape() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, _initial_states, _B_prev, _U_prev = _make_scan_inputs(
        batch=2,
        heads=4,
        T=65,
        N=256,
        P=64,
        device=torch.device("cuda"),
        value_dtype=torch.float16,
    )
    chunk_size = 64

    inc_ref, m_ref = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=None,
        U_prev=None,
        T=U.shape[2],
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    starts_ref, _ = state_passing(
        inc_ref,
        m_ref,
        initial_states=None,
        compute_dtype=torch.float32,
    )

    y_ref = chunk_scan(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=None,
        U_prev=None,
        T=U.shape[2],
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    y_cute = chunk_scan_cute(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=None,
        U_prev=None,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(y_cute, y_ref, atol=1e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_matches_reference_forward_training_only() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    y_ref, _final_ref, _b_last_ref, _u_last_ref = v2x2ssd(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        initial_states=None,
        B_prev=None,
        U_prev=None,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    y_cute = v2x2ssd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(y_cute, y_ref, atol=2e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_matches_reference_autograd_training_only() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=65,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32
    weight = torch.randn((2, 2, 65, 16), device="cuda", dtype=torch.float32)

    def run(
        op, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        U_i, M_i, K_i, B_i, C_i = tensors
        if op is v2x2ssd:
            Y, _final_state, _b_last, _u_last = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                initial_states=None,
                B_prev=None,
                U_prev=None,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        else:
            Y = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        loss = (Y * weight).sum()
        grads = torch.autograd.grad(
            loss,
            (U_i, M_i, K_i, B_i, C_i),
            retain_graph=False,
        )
        return (Y,), grads

    ref_tensors = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in (U, M, K, B, C)
    )
    cute_tensors = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in (U, M, K, B, C)
    )

    ref_out, ref_grads = run(v2x2ssd, ref_tensors)
    cute_out, cute_grads = run(v2x2ssd_cute, cute_tensors)

    for got, want in zip(cute_out, ref_out, strict=True):
        torch.testing.assert_close(got, want, atol=1e-3, rtol=0.0)
    grad_names = ("U", "M", "K", "B", "C")
    atol_by_grad = {
        "U": 1e-1,
        "M": 3e-1,
        "K": 5e-1,
        "B": 4e-1,
        "C": 3e-1,
    }
    for name, got, want in zip(grad_names, cute_grads, ref_grads, strict=True):
        # The integrated chunk-scan backward now reuses the promoted packed
        # tensor-core dU/dC/dQ/dK slices. Public gradients that depend on those
        # packed intermediates inherit the same principled low-precision
        # contract: fp16/bf16 transport, fp32 dense accumulation, and exact fp32
        # scatter/reduction back to public layout. The remaining slices stay on
        # the tighter exact budget.
        torch.testing.assert_close(got, want, atol=atol_by_grad[name], rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_matches_reference_autograd_with_state() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=65,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32
    y_weight = torch.randn((2, 2, 65, 16), device="cuda", dtype=torch.float32)
    state_weight = torch.randn((2, 2, 16, 16), device="cuda", dtype=torch.float32)
    b_last_weight = torch.randn((2, 2, 16), device="cuda", dtype=torch.float32)
    u_last_weight = torch.randn((2, 2, 16), device="cuda", dtype=torch.float32)

    def run(
        op, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        U_i, M_i, K_i, B_i, C_i, initial_i, B_prev_i, U_prev_i = tensors
        if op is v2x2ssd:
            outputs = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                initial_states=initial_i,
                B_prev=B_prev_i,
                U_prev=U_prev_i,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        else:
            outputs = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                initial_states=initial_i,
                B_prev=B_prev_i,
                U_prev=U_prev_i,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
                return_state=True,
            )
        Y, final_state, B_last, U_last = outputs
        loss = (
            (Y * y_weight).sum()
            + (final_state * state_weight).sum()
            + (B_last * b_last_weight).sum()
            + (U_last * u_last_weight).sum()
        )
        grads = torch.autograd.grad(
            loss,
            (U_i, M_i, K_i, B_i, C_i, initial_i, B_prev_i, U_prev_i),
            retain_graph=False,
        )
        return outputs, grads

    ref_tensors = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev)
    )
    cute_tensors = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev)
    )

    ref_out, ref_grads = run(v2x2ssd, ref_tensors)
    cute_out, cute_grads = run(v2x2ssd_cute, cute_tensors)

    for got, want in zip(cute_out, ref_out, strict=True):
        torch.testing.assert_close(got, want, atol=1e-3, rtol=0.0)

    grad_names = ("U", "M", "K", "B", "C", "initial_states", "B_prev", "U_prev")
    atol_by_grad = {
        "U": 1e-1,
        "M": 3e-1,
        "K": 5e-1,
        "B": 4e-1,
        "C": 3e-1,
        "initial_states": 5e-3,
        "B_prev": 5e-3,
        "U_prev": 5e-3,
    }
    for name, got, want in zip(grad_names, cute_grads, ref_grads, strict=True):
        torch.testing.assert_close(got, want, atol=atol_by_grad[name], rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_stateful_backward_clears_fused_dm_chunk_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    orig_get_bwd_workspace = v2x2ssd_bwd_mod._get_bwd_workspace

    def _poisoned_get_bwd_workspace(*args, **kwargs):
        workspace = orig_get_bwd_workspace(*args, **kwargs)
        workspace[6].fill_(123.0)
        return workspace

    monkeypatch.setattr(
        v2x2ssd_bwd_mod, "_get_bwd_workspace", _poisoned_get_bwd_workspace
    )

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1,
        heads=1,
        T=33,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32
    y_weight = torch.randn((1, 1, 33, 16), device="cuda", dtype=torch.float32)
    state_weight = torch.randn((1, 1, 16, 16), device="cuda", dtype=torch.float32)
    b_last_weight = torch.randn((1, 1, 16), device="cuda", dtype=torch.float32)
    u_last_weight = torch.randn((1, 1, 16), device="cuda", dtype=torch.float32)

    def run(
        op, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        U_i, M_i, K_i, B_i, C_i, initial_i, B_prev_i, U_prev_i = tensors
        if op is v2x2ssd:
            outputs = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                initial_states=initial_i,
                B_prev=B_prev_i,
                U_prev=U_prev_i,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        else:
            outputs = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                initial_states=initial_i,
                B_prev=B_prev_i,
                U_prev=U_prev_i,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
                return_state=True,
            )
        Y, final_state, B_last, U_last = outputs
        loss = (
            (Y * y_weight).sum()
            + (final_state * state_weight).sum()
            + (B_last * b_last_weight).sum()
            + (U_last * u_last_weight).sum()
        )
        grads = torch.autograd.grad(
            loss,
            (U_i, M_i, K_i, B_i, C_i, initial_i, B_prev_i, U_prev_i),
            retain_graph=False,
        )
        return outputs, grads

    ref_tensors = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev)
    )
    cute_tensors = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev)
    )

    ref_out, ref_grads = run(v2x2ssd, ref_tensors)
    cute_out, cute_grads = run(v2x2ssd_cute, cute_tensors)

    for got, want in zip(cute_out, ref_out, strict=True):
        torch.testing.assert_close(got, want, atol=1e-3, rtol=0.0)

    grad_names = ("U", "M", "K", "B", "C", "initial_states", "B_prev", "U_prev")
    atol_by_grad = {
        "U": 1e-1,
        "M": 3e-1,
        "K": 5e-1,
        "B": 4e-1,
        "C": 3e-1,
        "initial_states": 5e-3,
        "B_prev": 5e-3,
        "U_prev": 5e-3,
    }
    for name, got, want in zip(grad_names, cute_grads, ref_grads, strict=True):
        torch.testing.assert_close(got, want, atol=atol_by_grad[name], rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_segmented_training_matches_single_pass_for_realistic_shape() -> (
    None
):
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1,
        heads=2,
        T=33,
        N=64,
        P=64,
        device=torch.device("cuda"),
    )
    chunk_size = 32
    weight = torch.randn((1, 2, 33, 64), device="cuda", dtype=torch.float32)

    full_tensors = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev)
    )

    seg1_tensors = (
        U[:, :, :17, :].contiguous().detach().clone().requires_grad_(True),
        M[:, :, :17, :].contiguous().detach().clone().requires_grad_(True),
        K[:, :, :17, :, :].contiguous().detach().clone().requires_grad_(True),
        B[:, :, :17, :].contiguous().detach().clone().requires_grad_(True),
        C[:, :, :17, :].contiguous().detach().clone().requires_grad_(True),
    )
    seg2_tensors = (
        U[:, :, 17:, :].contiguous().detach().clone().requires_grad_(True),
        M[:, :, 17:, :].contiguous().detach().clone().requires_grad_(True),
        K[:, :, 17:, :, :].contiguous().detach().clone().requires_grad_(True),
        B[:, :, 17:, :].contiguous().detach().clone().requires_grad_(True),
        C[:, :, 17:, :].contiguous().detach().clone().requires_grad_(True),
    )
    initial_seg = initial_states.detach().clone().requires_grad_(True)
    B_prev_seg = B_prev.detach().clone().requires_grad_(True)
    U_prev_seg = U_prev.detach().clone().requires_grad_(True)

    Y_full, _final_full, _B_last_full, _U_last_full = v2x2ssd_cute(
        full_tensors[0],
        full_tensors[1],
        full_tensors[2],
        full_tensors[3],
        full_tensors[4],
        chunk_size=chunk_size,
        initial_states=full_tensors[5],
        B_prev=full_tensors[6],
        U_prev=full_tensors[7],
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
        return_state=True,
    )
    loss_full = (Y_full * weight).sum()
    full_grads = torch.autograd.grad(loss_full, full_tensors)

    Y_a, final_state_a, B_last_a, U_last_a = v2x2ssd_cute(
        seg1_tensors[0],
        seg1_tensors[1],
        seg1_tensors[2],
        seg1_tensors[3],
        seg1_tensors[4],
        chunk_size=chunk_size,
        initial_states=initial_seg,
        B_prev=B_prev_seg,
        U_prev=U_prev_seg,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
        return_state=True,
    )
    Y_b, _final_state_b, _B_last_b, _U_last_b = v2x2ssd_cute(
        seg2_tensors[0],
        seg2_tensors[1],
        seg2_tensors[2],
        seg2_tensors[3],
        seg2_tensors[4],
        chunk_size=chunk_size,
        initial_states=final_state_a,
        B_prev=B_last_a,
        U_prev=U_last_a,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
        return_state=True,
    )
    Y_seg = torch.cat((Y_a, Y_b), dim=2)
    loss_seg = (Y_seg * weight).sum()
    seg_grads = torch.autograd.grad(
        loss_seg,
        (
            *seg1_tensors,
            *seg2_tensors,
            initial_seg,
            B_prev_seg,
            U_prev_seg,
        ),
    )

    torch.testing.assert_close(Y_seg, Y_full, atol=2e-4, rtol=0.0)

    gradient_pairs = (
        ("U", torch.cat((seg_grads[0], seg_grads[5]), dim=2), full_grads[0], 5e-4),
        ("M", torch.cat((seg_grads[1], seg_grads[6]), dim=2), full_grads[1], 5e-3),
        ("K", torch.cat((seg_grads[2], seg_grads[7]), dim=2), full_grads[2], 5e-3),
        ("B", torch.cat((seg_grads[3], seg_grads[8]), dim=2), full_grads[3], 5e-3),
        ("C", torch.cat((seg_grads[4], seg_grads[9]), dim=2), full_grads[4], 2e-3),
        ("initial", seg_grads[10], full_grads[5], 5e-5),
        ("B_prev", seg_grads[11], full_grads[6], 2e-3),
        ("U_prev", seg_grads[12], full_grads[7], 5e-4),
    )
    for _name, got, want, atol in gradient_pairs:
        torch.testing.assert_close(got, want, atol=atol, rtol=0.0)
