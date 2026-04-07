from __future__ import annotations

from typing import cast

import pytest
import torch

from slinoss.layers import (
    AutoScanBackend,
    CuteScanBackend,
    ReferenceCConv1dBackend,
    ReferenceScanBackend,
    ReferenceScanPrepBackend,
    SLinOSSMixer,
    ScanInputs,
    ScanPrepInputs,
    ScanState,
)
from slinoss.layers.mixer import _SplitMixerProjectionFn


class SpyBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.last_inputs: ScanInputs | None = None
        self.last_state: ScanState | None = None
        self.last_return_state = False

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        del chunk_size
        self.calls += 1
        self.last_inputs = inputs
        self.last_state = state
        self.last_return_state = return_state

        batch, heads, _, P = map(int, inputs.U.shape)
        D = int(inputs.B.shape[-1])
        next_state = ScanState(
            state=torch.zeros(
                (batch, heads, P, D), device=inputs.U.device, dtype=inputs.U.dtype
            ),
            b_prev=inputs.B[:, :, -1, :].contiguous(),
            u_prev=inputs.U[:, :, -1, :].contiguous(),
        )
        if not return_state:
            return torch.zeros_like(inputs.U)
        return torch.zeros_like(inputs.U), next_state


class CaptureScanPrepBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.last_inputs: ScanPrepInputs | None = None

    def __call__(self, owner: object, inputs: ScanPrepInputs) -> ScanInputs:
        self.calls += 1
        self.last_inputs = inputs
        return owner._prepare_inputs_reference(inputs)  # type: ignore[attr-defined]


class ZeroConvBackend:
    def __call__(
        self,
        owner: SLinOSSMixer,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del conv_state
        state_len = max(owner.d_conv - 1, 0)
        next_state = x.new_zeros((x.shape[0], owner.d_inner, state_len))
        return torch.zeros_like(x), next_state


def _cuda_amp_dtype_supported(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    if dtype == torch.bfloat16:
        return torch.cuda.is_bf16_supported()
    return dtype == torch.float16


def _make_mixer(*, backend: object | None = None) -> SLinOSSMixer:
    return SLinOSSMixer(
        12,
        d_state=3,
        expand=2,
        d_head=6,
        d_conv=3,
        chunk_size=4,
        backend=backend,  # type: ignore[arg-type]
    )


def test_mixer_calls_backend_with_canonical_scan_shapes() -> None:
    torch.manual_seed(0)
    spy = SpyBackend()
    mixer = _make_mixer(backend=spy)
    x = torch.randn((2, 5, 12), dtype=torch.float32)

    y, state = mixer(x, return_state=True)

    assert spy.calls == 1
    assert spy.last_inputs is not None
    assert spy.last_state is None
    assert spy.last_return_state is True
    assert y.shape == (2, 5, 12)
    assert state.conv is not None
    assert state.scan.state is not None
    assert state.scan.b_prev is not None
    assert state.scan.u_prev is not None

    assert spy.last_inputs.U.shape == (2, 4, 5, 6)
    assert spy.last_inputs.M.shape == (2, 4, 5, 2)
    assert spy.last_inputs.K.shape == (2, 4, 5, 2, 2)
    assert spy.last_inputs.B.shape == (2, 4, 5, 6)
    assert spy.last_inputs.C.shape == (2, 4, 5, 6)
    assert state.conv.shape == (2, 24, 2)
    assert state.scan.state.shape == (2, 4, 6, 6)
    assert state.scan.b_prev.shape == (2, 4, 6)
    assert state.scan.u_prev.shape == (2, 4, 6)


def test_mixer_defaults_to_auto_scan_backend() -> None:
    mixer = _make_mixer()
    assert isinstance(mixer.backend, AutoScanBackend)


def test_mixer_emits_bc_from_in_proj() -> None:
    mixer = _make_mixer()
    assert not hasattr(mixer, "bc_proj")
    assert mixer.in_proj.out_features == (
        2 * mixer.d_inner + mixer.param_proj_dim + mixer.bc_proj_dim
    )


def test_mixer_issue_1_bc_emission_is_decoupled_from_value_path() -> None:
    torch.manual_seed(0)
    ref_scanprep = CaptureScanPrepBackend()
    zero_scanprep = CaptureScanPrepBackend()
    ref_backend = SpyBackend()
    zero_backend = SpyBackend()
    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=64,
        scanprep_backend=ref_scanprep,  # type: ignore[arg-type]
        backend=ref_backend,  # type: ignore[arg-type]
    )
    zero_mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=64,
        scanprep_backend=zero_scanprep,  # type: ignore[arg-type]
        cconv_backend=ZeroConvBackend(),  # type: ignore[arg-type]
        backend=zero_backend,  # type: ignore[arg-type]
    )
    zero_mixer.load_state_dict(mixer.state_dict())

    x = torch.randn((2, 65, 128), dtype=torch.float32)
    expected_bc = mixer.in_proj(x)[..., -mixer.bc_proj_dim :].view(
        2, 65, mixer.n_heads, mixer.scanprep.bc_param_rows, mixer.d_state
    )

    mixer(x)
    zero_mixer(x)

    assert ref_scanprep.last_inputs is not None
    assert zero_scanprep.last_inputs is not None
    assert torch.count_nonzero(ref_scanprep.last_inputs.value).item() > 0
    assert torch.count_nonzero(zero_scanprep.last_inputs.value).item() == 0
    assert not torch.allclose(
        ref_scanprep.last_inputs.value, zero_scanprep.last_inputs.value
    )
    assert torch.allclose(ref_scanprep.last_inputs.bc, expected_bc, atol=1e-6)
    assert torch.allclose(zero_scanprep.last_inputs.bc, expected_bc, atol=1e-6)
    assert torch.allclose(
        ref_scanprep.last_inputs.bc, zero_scanprep.last_inputs.bc, atol=1e-6
    )


def test_mixer_cute_scan_matches_reference_for_issue_9_bf16_shape() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    cute_mixer = SLinOSSMixer(
        512,
        d_state=128,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=128,
        scanprep_backend=ReferenceScanPrepBackend(),
        backend=AutoScanBackend(),
        cconv_backend=ReferenceCConv1dBackend(),
        device="cuda",
        dtype=torch.bfloat16,
    )
    ref_mixer = SLinOSSMixer(
        512,
        d_state=128,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=128,
        scanprep_backend=ReferenceScanPrepBackend(),
        backend=ReferenceScanBackend(compute_dtype=torch.float32),
        cconv_backend=ReferenceCConv1dBackend(),
        device="cuda",
        dtype=torch.bfloat16,
    )
    ref_mixer.load_state_dict(cute_mixer.state_dict())

    x = torch.randn((1, 128, 512), device="cuda", dtype=torch.bfloat16)
    y_cute = cute_mixer(x)
    y_ref = ref_mixer(x)

    assert torch.isfinite(y_cute).all()
    torch.testing.assert_close(y_cute, y_ref, atol=1e-1, rtol=0.0)

    state_cute = cute_mixer.init_state(1, device="cuda", dtype=torch.bfloat16)
    state_ref = ref_mixer.init_state(1, device="cuda", dtype=torch.bfloat16)
    y_cute_state, next_cute = cute_mixer(x, state=state_cute, return_state=True)
    y_ref_state, next_ref = ref_mixer(x, state=state_ref, return_state=True)

    assert torch.isfinite(y_cute_state).all()
    torch.testing.assert_close(y_cute_state, y_ref_state, atol=1e-1, rtol=0.0)
    assert next_cute.conv is not None
    assert next_ref.conv is not None
    assert next_cute.scan.state is not None
    assert next_ref.scan.state is not None
    assert next_cute.scan.b_prev is not None
    assert next_ref.scan.b_prev is not None
    assert next_cute.scan.u_prev is not None
    assert next_ref.scan.u_prev is not None
    torch.testing.assert_close(next_cute.conv, next_ref.conv, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(
        next_cute.scan.state, next_ref.scan.state, atol=1e-1, rtol=0.0
    )
    torch.testing.assert_close(
        next_cute.scan.b_prev, next_ref.scan.b_prev, atol=1e-1, rtol=0.0
    )
    torch.testing.assert_close(
        next_cute.scan.u_prev, next_ref.scan.u_prev, atol=1e-2, rtol=0.0
    )


def test_mixer_rejects_incompatible_d_state_for_cute_scan_backend() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        12,
        d_state=3,
        expand=2,
        d_head=6,
        d_conv=3,
        chunk_size=4,
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn((2, 5, 12), device="cuda", dtype=torch.float16)

    with pytest.raises(
        ValueError,
        match="CuTe scan backend requires d_state to be a multiple of 8",
    ):
        mixer(x)


def test_mixer_backward_supports_issue_2_shape() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        128,
        d_state=16,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=64,
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn(
        (2, 65, 128), device="cuda", dtype=torch.float16, requires_grad=True
    )

    y = mixer(x)
    loss = y.to(dtype=torch.float32).square().mean()
    loss.backward()

    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in mixer.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_mixer_forward_supports_issue_3_shape() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        256,
        d_state=256,
        expand=1,
        d_head=64,
        d_conv=4,
        chunk_size=64,
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn((1, 65, 256), device="cuda", dtype=torch.float16)

    y = mixer(x)

    assert y.shape == (1, 65, 256)
    assert torch.isfinite(y).all()


def test_mixer_backward_supports_issue_5_shape() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        256,
        d_state=256,
        expand=1,
        d_head=64,
        d_conv=4,
        chunk_size=64,
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn(
        (1, 65, 256), device="cuda", dtype=torch.float16, requires_grad=True
    )

    y = mixer(x)
    loss = y.to(dtype=torch.float32).square().mean()
    loss.backward()

    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in mixer.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_mixer_cute_stateless_forward_is_mode_invariant_for_issue_6() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=128,
        d_conv=4,
        chunk_size=64,
        device="cuda",
        dtype=torch.float32,
    ).eval()
    x = torch.randn((2, 65, 128), device="cuda", dtype=torch.float32)

    with torch.enable_grad():
        y_grad = mixer(x)
    with torch.no_grad():
        y_no_grad = mixer(x)
    with torch.inference_mode():
        y_infer = mixer(x)

    for y in (y_grad, y_no_grad, y_infer):
        assert torch.isfinite(y).all()

    torch.testing.assert_close(y_no_grad, y_grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(y_infer, y_grad, atol=0.0, rtol=0.0)


def test_mixer_torch_compile_contains_only_intentional_compiler_boundaries() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=64,
        device="cuda",
        dtype=torch.float16,
    ).eval()
    x = torch.randn((2, 64, 128), device="cuda", dtype=torch.float16)

    explain = torch._dynamo.explain(mixer)
    result = explain(x)

    assert result.graph_break_count >= len(result.break_reasons)
    assert len(result.break_reasons) >= 1
    assert all(
        "torch.compiler.disable" in break_reason.reason
        for break_reason in result.break_reasons
    )


def test_mixer_torch_compile_runs_training_with_cute_boundaries() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=64,
        device="cuda",
        dtype=torch.float16,
    ).train()
    compiled = torch.compile(mixer, backend="eager")

    x0 = torch.randn(
        (2, 64, 128), device="cuda", dtype=torch.float16, requires_grad=True
    )
    y0 = compiled(x0)
    loss0 = y0.to(dtype=torch.float32).square().mean()
    loss0.backward()

    assert torch.isfinite(y0).all()
    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()
    for param in mixer.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()

    mixer.zero_grad(set_to_none=True)
    x1 = torch.randn(
        (2, 64, 128), device="cuda", dtype=torch.float16, requires_grad=True
    )
    y1 = compiled(x1)
    loss1 = y1.to(dtype=torch.float32).square().mean()
    loss1.backward()

    assert torch.isfinite(y1).all()
    assert x1.grad is not None
    assert torch.isfinite(x1.grad).all()
    for param in mixer.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_mixer_cute_segmented_forward_matches_single_pass() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        backend=CuteScanBackend(),
        scanprep_backend=ReferenceScanPrepBackend(),
        cconv_backend=ReferenceCConv1dBackend(),
        device="cuda",
        dtype=torch.float32,
    ).eval()
    x = torch.randn((2, 49, 128), device="cuda", dtype=torch.float32)

    y_full, state_full = mixer(x, return_state=True)
    y_a, state = mixer(x[:, :17, :], return_state=True)
    y_b, state = mixer(x[:, 17:, :], state=state, return_state=True)

    assert state_full.scan.state is not None
    assert state_full.scan.b_prev is not None
    assert state_full.scan.u_prev is not None
    assert state.scan.state is not None
    assert state.scan.b_prev is not None
    assert state.scan.u_prev is not None
    for tensor in (
        y_full,
        y_a,
        y_b,
        state_full.scan.state,
        state_full.scan.b_prev,
        state_full.scan.u_prev,
        state.scan.state,
        state.scan.b_prev,
        state.scan.u_prev,
    ):
        assert torch.isfinite(tensor).all()

    y_segmented = torch.cat([y_a, y_b], dim=1)
    torch.testing.assert_close(y_segmented, y_full, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(
        state.scan.state, state_full.scan.state, atol=1e-3, rtol=0.0
    )
    torch.testing.assert_close(
        state.scan.b_prev, state_full.scan.b_prev, atol=1e-3, rtol=0.0
    )
    torch.testing.assert_close(
        state.scan.u_prev, state_full.scan.u_prev, atol=1e-3, rtol=0.0
    )


def test_mixer_cute_segmented_training_matches_single_pass() -> None:
    if not torch.cuda.is_available():
        return

    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    mixer_full = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        backend=CuteScanBackend(),
        scanprep_backend=ReferenceScanPrepBackend(),
        cconv_backend=ReferenceCConv1dBackend(),
        device="cuda",
        dtype=torch.float32,
    ).train()
    mixer_seg = SLinOSSMixer(
        128,
        d_state=64,
        expand=2,
        d_head=64,
        d_conv=4,
        chunk_size=32,
        backend=CuteScanBackend(),
        scanprep_backend=ReferenceScanPrepBackend(),
        cconv_backend=ReferenceCConv1dBackend(),
        device="cuda",
        dtype=torch.float32,
    ).train()
    mixer_seg.load_state_dict(mixer_full.state_dict())

    x_full = torch.randn(
        (2, 33, 128), device="cuda", dtype=torch.float32, requires_grad=True
    )
    x_seg = x_full.detach().clone().requires_grad_(True)
    weight = torch.randn((2, 33, 128), device="cuda", dtype=torch.float32)

    y_full = mixer_full(x_full)
    loss_full = (y_full * weight).sum()
    loss_full.backward()

    y_a, state = mixer_seg(x_seg[:, :17, :], return_state=True)
    y_b, state = mixer_seg(x_seg[:, 17:, :], state=state, return_state=True)
    y_seg = torch.cat([y_a, y_b], dim=1)
    loss_seg = (y_seg * weight).sum()
    loss_seg.backward()

    assert x_full.grad is not None
    assert x_seg.grad is not None
    assert torch.isfinite(y_full).all()
    assert torch.isfinite(y_seg).all()
    assert torch.isfinite(x_full.grad).all()
    assert torch.isfinite(x_seg.grad).all()

    torch.testing.assert_close(y_seg, y_full, atol=2e-3, rtol=0.0)
    torch.testing.assert_close(x_seg.grad, x_full.grad, atol=2e-2, rtol=0.0)
    for (name_full, param_full), (name_seg, param_seg) in zip(
        mixer_full.named_parameters(),
        mixer_seg.named_parameters(),
        strict=True,
    ):
        assert name_full == name_seg
        assert param_full.grad is not None
        assert param_seg.grad is not None
        assert torch.isfinite(param_full.grad).all()
        assert torch.isfinite(param_seg.grad).all()
        torch.testing.assert_close(param_seg.grad, param_full.grad, atol=2e-2, rtol=0.0)


def test_mixer_step_matches_full_forward() -> None:
    torch.manual_seed(1)
    mixer = _make_mixer()
    x = torch.randn((2, 7, 12), dtype=torch.float32)

    y_full, state_full = mixer(x, return_state=True)
    state_step = mixer.init_state(x.shape[0], dtype=x.dtype)
    pieces: list[torch.Tensor] = []
    for t in range(x.shape[1]):
        y_t, state_step = mixer.step(x[:, t, :], state_step)
        pieces.append(y_t.unsqueeze(1))
    y_step = torch.cat(pieces, dim=1)

    assert state_full.conv is not None
    assert state_step.conv is not None
    assert state_full.scan.state is not None
    assert state_step.scan.state is not None
    assert state_full.scan.b_prev is not None
    assert state_step.scan.b_prev is not None
    assert state_full.scan.u_prev is not None
    assert state_step.scan.u_prev is not None

    assert torch.allclose(y_full, y_step, atol=1e-6, rtol=1e-6)
    assert torch.allclose(state_full.conv, state_step.conv, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        state_full.scan.state, state_step.scan.state, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        state_full.scan.b_prev, state_step.scan.b_prev, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        state_full.scan.u_prev, state_step.scan.u_prev, atol=1e-6, rtol=1e-6
    )


def test_mixer_segmented_forward_matches_single_pass() -> None:
    torch.manual_seed(2)
    mixer = _make_mixer()
    x = torch.randn((2, 9, 12), dtype=torch.float32)

    y_full = mixer(x)
    y_a, state = mixer(x[:, :4, :], return_state=True)
    y_b, state = mixer(x[:, 4:, :], state=state, return_state=True)

    assert state.conv is not None
    assert state.scan.state is not None
    assert state.scan.b_prev is not None
    assert state.scan.u_prev is not None
    assert torch.allclose(y_full, torch.cat([y_a, y_b], dim=1), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_split_mixer_projection_backward_supports_cuda_autocast(
    amp_dtype: torch.dtype,
) -> None:
    if not _cuda_amp_dtype_supported(amp_dtype):
        pytest.skip("CUDA autocast dtype is unavailable")

    d_model = 16
    d_inner = 8
    param_proj_dim = 6
    out_rows = 2 * d_inner + param_proj_dim + 10
    x = torch.randn(
        (2, 3, d_model),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.randn(
        (out_rows, d_model),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    with torch.autocast("cuda", dtype=amp_dtype):
        gate, value, params, bc = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _SplitMixerProjectionFn.apply(
                x,
                weight,
                d_inner,
                param_proj_dim,
            ),
        )
        loss = (
            gate.square().mean()
            + value.square().mean()
            + params.square().mean()
            + bc.square().mean()
        )

    loss.backward()

    assert x.grad is not None
    assert weight.grad is not None
    assert x.grad.dtype == x.dtype
    assert weight.grad.dtype == weight.dtype
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_mixer_backward_supports_cuda_autocast(amp_dtype: torch.dtype) -> None:
    if not _cuda_amp_dtype_supported(amp_dtype):
        pytest.skip("CUDA autocast dtype is unavailable")

    mixer = SLinOSSMixer(
        32,
        d_state=8,
        expand=2,
        d_head=16,
        d_conv=3,
        chunk_size=4,
        scanprep_backend=ReferenceScanPrepBackend(),
        backend=ReferenceScanBackend(compute_dtype=torch.float32),
        cconv_backend=ReferenceCConv1dBackend(),
        device="cuda",
        dtype=torch.float32,
    ).train()
    x = torch.randn((2, 8, 32), device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.autocast("cuda", dtype=amp_dtype):
        loss = mixer(x).square().mean()

    loss.backward()

    assert x.grad is not None
    assert mixer.in_proj.weight.grad is not None
    assert x.grad.dtype == x.dtype
    assert mixer.in_proj.weight.grad.dtype == mixer.in_proj.weight.dtype
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(mixer.in_proj.weight.grad).all()
