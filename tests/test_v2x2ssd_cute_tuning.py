from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd import (
    _resolve_forward_autotune_bundle,
    tune_chunk_increment_cute,
    tune_chunk_scan_cute,
    tune_state_passing_cute,
    v2x2ssd_fwd_cute,
)
import slinoss.ops.v2x2ssd.cute.kernels.fwd as v2x2ssd_fwd_mod
import slinoss.ops.v2x2ssd.cute.tuning.fwd as tuning_fwd_mod
from slinoss.ops.v2x2ssd.cute.tuning import (
    ChunkIncrementConfig,
    ChunkScanConfig,
    ForwardConfigBundle,
    StatePassingConfig,
    load_cute_tuning_db,
    save_cute_tuning_db,
    store_tuning_record,
)
from slinoss.ops.v2x2ssd.cute.tuning.hardware import current_hardware_fingerprint


def test_autotune_mode_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLINOSS_CUTE_AUTOTUNE", raising=False)
    assert tuning_fwd_mod.autotune_mode() == "0"
    assert not tuning_fwd_mod.autotune_enabled()


def test_chunk_increment_candidate_configs_prune_unsupported_device_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_config = ChunkIncrementConfig(cta_tiler=(64, 128, 64), num_stages=3)
    good_config = ChunkIncrementConfig(cta_tiler=(64, 128, 64), num_stages=2)

    def _fake_can_implement(self, dtype, *, device_index=None):
        return (
            self.cta_tiler != bad_config.cta_tiler
            or self.num_stages != bad_config.num_stages
        )

    monkeypatch.setattr(
        tuning_fwd_mod.ChunkIncrementFwdAmpere,
        "can_implement",
        _fake_can_implement,
    )
    configs = tuning_fwd_mod.chunk_increment_candidate_configs(
        P=64,
        D=256,
        chunk_size=128,
        tc_dtype=torch.float16,
        device_index=0,
    )
    assert bad_config not in configs
    assert good_config in configs


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
    value_dtype: torch.dtype = torch.float16,
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


def test_tuning_db_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE_CACHE_DIR", str(tmp_path))
    payload = {"version": 1, "records": {"example": {"config": {"foo": 1}}}}
    save_cute_tuning_db(payload)
    loaded = load_cute_tuning_db()
    assert loaded["records"]["example"]["config"]["foo"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tune_chunk_increment_cute_reuses_cached_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    torch.manual_seed(0)
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE_CACHE_DIR", str(tmp_path))
    U, M, K, B, _C, _initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1, heads=1, T=64, N=8, P=64, device=torch.device("cuda")
    )
    hardware = current_hardware_fingerprint(device_index=torch.cuda.current_device())
    cached_config = ChunkIncrementConfig(cta_tiler=(64, 64, 32), num_stages=2)
    problem_key = v2x2ssd_fwd_mod.chunk_increment_problem_key(
        tc_dtype=torch.float16,
        P=64,
        D=16,
        chunk_size=32,
        has_prev=True,
    )
    store_tuning_record(
        scope="chunk_increment",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=cached_config.to_record(),
    )
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_matching_packaged_chunk_increment_specs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not inspect packaged specs on DB hit")
        ),
    )
    assert (
        tune_chunk_increment_cute(
            U,
            M,
            K,
            B,
            U_prev=U_prev,
            B_prev=B_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
        )
        == cached_config
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tune_chunk_increment_cute_skips_invalid_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    torch.manual_seed(0)
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE", "force")
    U, M, K, B, _C, _initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1, heads=1, T=64, N=8, P=64, device=torch.device("cuda")
    )
    bad_config = ChunkIncrementConfig(cta_tiler=(64, 128, 64), num_stages=3)
    good_config = ChunkIncrementConfig(cta_tiler=(64, 64, 32), num_stages=2)
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_matching_packaged_chunk_increment_specs",
        lambda **kwargs: (),
    )
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "chunk_increment_candidate_configs",
        lambda **kwargs: (bad_config, good_config),
    )

    def _fake_make_chunk_increment_prepared_launch(*args, **kwargs):
        if (
            kwargs["cta_tiler"] == bad_config.cta_tiler
            and kwargs["num_stages"] == bad_config.num_stages
        ):
            raise RuntimeError("CUDA Error: cudaErrorInvalidValue")
        return SimpleNamespace(
            compiled=lambda *runtime_args: None,
            runtime_args=(),
            outputs=SimpleNamespace(
                increment_chunk=torch.empty(
                    (1, 64, 16), device="cuda", dtype=torch.float32
                ),
                chunk_multiplier_storage=torch.empty(
                    (1, 2), device="cuda", dtype=torch.float32
                ),
            ),
        )

    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_make_chunk_increment_prepared_launch",
        _fake_make_chunk_increment_prepared_launch,
    )
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "benchmark_cuda_callable",
        lambda *args, **kwargs: 1.0,
    )

    assert (
        tune_chunk_increment_cute(
            U,
            M,
            K,
            B,
            U_prev=U_prev,
            B_prev=B_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
        )
        == good_config
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tune_state_passing_cute_reuses_cached_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE_CACHE_DIR", str(tmp_path))
    increment = torch.randn((1, 1, 2, 64, 16), device="cuda", dtype=torch.float32)
    radius = 0.6 + 0.35 * torch.rand((1, 1, 2), device="cuda")
    angle = (2.0 * math.pi) * torch.rand((1, 1, 2), device="cuda") - math.pi
    chunk_multiplier = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32)
    hardware = current_hardware_fingerprint(device_index=torch.cuda.current_device())
    cached_config = StatePassingConfig(num_threads=64, vecs_per_thread=8)
    problem_key = v2x2ssd_fwd_mod.state_passing_problem_key(
        P=64,
        D=16,
        has_init=False,
    )
    store_tuning_record(
        scope="state_passing",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=cached_config.to_record(),
    )
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_matching_packaged_state_passing_specs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not inspect packaged specs on DB hit")
        ),
    )
    assert (
        tune_state_passing_cute(
            increment,
            chunk_multiplier,
            initial_states=None,
        )
        == cached_config
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tune_chunk_scan_cute_reuses_cached_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    torch.manual_seed(0)
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE_CACHE_DIR", str(tmp_path))
    U, M, K, B, C, _initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1, heads=1, T=64, N=8, P=64, device=torch.device("cuda")
    )
    chunk_starts = torch.zeros((1, 1, 2, 64, 16), device="cuda", dtype=torch.float32)
    hardware = current_hardware_fingerprint(device_index=torch.cuda.current_device())
    cached_config = ChunkScanConfig(m_block_size=32, n_block_size=32, num_threads=64)
    problem_key = v2x2ssd_fwd_mod.chunk_scan_problem_key(
        tc_dtype=torch.float16,
        output_dtype=torch.float32,
        P=64,
        D=16,
        chunk_size=32,
        has_prev=True,
    )
    store_tuning_record(
        scope="chunk_scan",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=cached_config.to_record(),
    )
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_matching_packaged_chunk_scan_specs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not inspect packaged specs on DB hit")
        ),
    )
    assert (
        tune_chunk_scan_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_starts,
            B_prev=B_prev,
            U_prev=U_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        == cached_config
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resolve_forward_autotune_bundle_reuses_cached_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    torch.manual_seed(0)
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLINOSS_CUTE_AUTOTUNE", "1")
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1, heads=1, T=64, N=8, P=64, device=torch.device("cuda")
    )
    hardware = current_hardware_fingerprint(device_index=torch.cuda.current_device())
    cached_bundle = ForwardConfigBundle(
        chunk_increment=ChunkIncrementConfig(cta_tiler=(64, 64, 32), num_stages=2),
        state_passing=StatePassingConfig(num_threads=64, vecs_per_thread=8),
        chunk_scan=ChunkScanConfig(m_block_size=32, n_block_size=32, num_threads=64),
    )
    problem_key = v2x2ssd_fwd_mod.forward_problem_key(
        tc_dtype=torch.float16,
        output_dtype=torch.float32,
        P=64,
        D=16,
        chunk_size=32,
        has_prev=True,
        has_init=True,
        n_chunks=2,
    )
    store_tuning_record(
        scope="forward",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=cached_bundle.to_record(),
    )
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_matching_packaged_forward_aot_specs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not inspect packaged specs on DB hit")
        ),
    )
    assert (
        _resolve_forward_autotune_bundle(
            U=U,
            M=M,
            K=K,
            B=B,
            C=C,
            chunk_size=32,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
        )
        == cached_bundle
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_fwd_cute_threads_cached_tuned_bundle_into_compile_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1, heads=1, T=64, N=8, P=64, device=torch.device("cuda")
    )
    tuned_bundle = ForwardConfigBundle(
        chunk_increment=ChunkIncrementConfig(cta_tiler=(64, 64, 32), num_stages=1),
        state_passing=StatePassingConfig(num_threads=64, vecs_per_thread=8),
        chunk_scan=ChunkScanConfig(m_block_size=32, n_block_size=32, num_threads=64),
    )
    seen_config_bundle: ForwardConfigBundle | None = None

    def _fake_get_compiled_v2x2ssd_fwd_kernel(**kwargs):
        nonlocal seen_config_bundle
        compile_artifacts = kwargs["compile_artifacts"]
        seen_config_bundle = compile_artifacts.config_bundle

        def _noop(*runtime_args):
            return None

        return _noop

    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_resolve_forward_autotune_bundle",
        lambda **kwargs: tuned_bundle,
    )
    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_get_compiled_v2x2ssd_fwd_kernel",
        _fake_get_compiled_v2x2ssd_fwd_kernel,
    )
    _ = v2x2ssd_fwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=32,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_intermediates=False,
    )
    assert seen_config_bundle == tuned_bundle
