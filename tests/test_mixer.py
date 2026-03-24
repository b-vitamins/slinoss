from __future__ import annotations

import pytest
import torch

from slinoss.layers import (
    AutoScanBackend,
    SLinOSSMixer,
    ScanInputs,
    ScanPrepInputs,
    ScanState,
)


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


def _make_mixer(*, backend: object | None = None) -> SLinOSSMixer:
    return SLinOSSMixer(
        12,
        d_state=3,
        expand=2,
        d_head=6,
        d_conv=3,
        chunk_size=4,
        normalize_bc=True,
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
        normalize_bc=True,
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
        normalize_bc=True,
        scanprep_backend=zero_scanprep,  # type: ignore[arg-type]
        cconv_backend=ZeroConvBackend(),  # type: ignore[arg-type]
        backend=zero_backend,  # type: ignore[arg-type]
    )
    zero_mixer.load_state_dict(mixer.state_dict())

    x = torch.randn((2, 65, 128), dtype=torch.float32)
    expected_bc = mixer.in_proj(x)[..., -mixer.bc_proj_dim :].view(
        2, 65, mixer.n_heads, 4, mixer.d_state
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
    assert torch.allclose(ref_scanprep.last_inputs.bc, expected_bc)
    assert torch.allclose(zero_scanprep.last_inputs.bc, expected_bc)
    assert torch.allclose(ref_scanprep.last_inputs.bc, zero_scanprep.last_inputs.bc)


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
        normalize_bc=True,
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
        normalize_bc=True,
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
        normalize_bc=True,
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
        normalize_bc=True,
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
        normalize_bc=True,
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
        normalize_bc=True,
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
        normalize_bc=True,
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
