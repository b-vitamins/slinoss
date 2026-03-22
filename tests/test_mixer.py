from __future__ import annotations

import pytest
import torch

from slinoss.layers import AutoScanBackend, SLinOSSMixer, ScanInputs, ScanState


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
