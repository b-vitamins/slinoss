from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

pytest.importorskip("cutlass")

import slinoss.ops.v2x2ssd.cute.aot as cute_aot_mod
import slinoss.ops.v2x2ssd.cute.kernels.fwd as v2x2ssd_fwd_mod
from slinoss.ops.v2x2ssd.cute.kernels.fwd import v2x2ssd_fwd_cute


@pytest.mark.parametrize(
    ("raw_arch", "normalized"),
    [
        ("8.0", "sm_80"),
        ("8.6+PTX", "sm_86"),
        ("sm_89", "sm_89"),
        ("compute_90", "sm_90"),
        ("90", "sm_90"),
        ("", ""),
    ],
)
def test_normalize_arch_tag(raw_arch: str, normalized: str) -> None:
    assert cute_aot_mod._normalize_arch_tag(raw_arch) == normalized


def test_arch_tags_from_env_prefers_explicit_tags(monkeypatch) -> None:
    monkeypatch.setenv("SLINOSS_CUTE_FORWARD_AOT_ARCH_TAGS", "sm_89,9.0,sm_89")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.6")
    assert cute_aot_mod._arch_tags_from_env() == ("sm_89", "sm_90")


def test_arch_tags_from_env_falls_back_to_torch_arch_list(monkeypatch) -> None:
    monkeypatch.delenv("SLINOSS_CUTE_FORWARD_AOT_ARCH_TAGS", raising=False)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX 8.0")
    assert cute_aot_mod._arch_tags_from_env() == ("sm_80", "sm_86")


def test_default_forward_aot_specs_expand_requested_arch_tags() -> None:
    specs = cute_aot_mod.default_forward_aot_specs(("sm_80", "sm_86"))
    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_80", "sm_86"}
    assert len(specs) == (
        len(cute_aot_mod._search_space_forward_specs(arch_tag="sm_80"))
        + len(cute_aot_mod._search_space_forward_specs(arch_tag="sm_86"))
    )


@pytest.mark.parametrize(
    ("helper_name", "spec"),
    [
        (
            "_compile_chunk_increment_aot",
            cute_aot_mod._search_space_chunk_increment_specs(arch_tag="sm_80")[0],
        ),
        (
            "_compile_state_passing_aot",
            cute_aot_mod._search_space_state_passing_specs(arch_tag="sm_80")[0],
        ),
        (
            "_compile_chunk_scan_aot",
            cute_aot_mod._search_space_chunk_scan_specs(arch_tag="sm_80")[0],
        ),
        (
            "_compile_forward_aot",
            cute_aot_mod._search_space_forward_specs(arch_tag="sm_80")[0],
        ),
    ],
)
def test_aot_compile_helpers_use_compile_only(
    monkeypatch,
    helper_name: str,
    spec: object,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_compile(*args, **kwargs):
        captured["kwargs"] = kwargs
        return "compiled"

    monkeypatch.setattr(cute_aot_mod.cute, "compile", _fake_compile)
    helper = getattr(cute_aot_mod, helper_name)
    compiled = helper(spec)
    assert compiled == "compiled"
    assert captured["kwargs"]["no_jit_engine"] is True
    assert captured["kwargs"]["options"] == "--enable-tvm-ffi --gpu-arch=sm_80"


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_export_v2x2ssd_fwd_cute_aot_roundtrips_through_loaded_module(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    device = torch.device("cuda")
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1,
        heads=1,
        T=32,
        N=8,
        P=16,
        device=device,
    )

    exported = cute_aot_mod.export_v2x2ssd_fwd_cute_aot(
        U,
        M,
        K,
        B,
        C,
        chunk_size=32,
        compute_dtype=torch.float32,
        output_dtype=torch.float16,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        package_root=tmp_path,
    )
    assert exported.object_file is not None
    assert exported.object_file.is_file()
    assert exported.shared_library.is_file()
    assert exported.metadata_file.is_file()

    runtime_artifacts = v2x2ssd_fwd_mod._make_forward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        chunk_size=32,
        compute_dtype=torch.float32,
        output_dtype=torch.float16,
        m_block_size=None,
        n_block_size=64,
        scan_num_threads=128,
        state_num_threads=128,
        state_vecs_per_thread=8,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_final_state=True,
        return_intermediates=False,
        prepared_inputs=None,
        validate_runtime_contract=True,
    )
    loaded = cute_aot_mod.load_tvm_ffi_function(
        exported.shared_library,
        function_name=exported.function_name,
    )
    loaded(*runtime_artifacts.runtime_args)
    torch.cuda.synchronize(device)

    reference = v2x2ssd_fwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=32,
        compute_dtype=torch.float32,
        output_dtype=torch.float16,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_final_state=True,
        return_intermediates=False,
    )
    reference = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], reference
    )
    reference_output, reference_final_state, _m_chunk, _chunk_starts = reference
    assert runtime_artifacts.outputs.final_state is not None
    torch.testing.assert_close(
        runtime_artifacts.outputs.output,
        reference_output,
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        runtime_artifacts.outputs.final_state,
        reference_final_state,
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_fwd_cute_prefers_packaged_aot_over_jit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    device = torch.device("cuda")
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1,
        heads=1,
        T=32,
        N=8,
        P=16,
        device=device,
    )

    cute_aot_mod.export_v2x2ssd_fwd_cute_aot(
        U,
        M,
        K,
        B,
        C,
        chunk_size=32,
        compute_dtype=torch.float32,
        output_dtype=torch.float16,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        package_root=tmp_path,
    )

    monkeypatch.setattr(cute_aot_mod, "_PACKAGED_AOT_ROOT", tmp_path)
    cute_aot_mod.clear_packaged_forward_aot_cache()
    v2x2ssd_fwd_mod._FWD_HOST_CACHE.clear()

    def _compile_should_not_run(*args, **kwargs):
        raise AssertionError("expected packaged AOT forward to bypass cute.compile")

    monkeypatch.setattr(v2x2ssd_fwd_mod.cute, "compile", _compile_should_not_run)
    out = v2x2ssd_fwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=32,
        compute_dtype=torch.float32,
        output_dtype=torch.float16,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_final_state=True,
        return_intermediates=False,
    )
    assert len(out) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stage_aot_export_helpers_emit_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    device = torch.device("cuda")
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1,
        heads=1,
        T=32,
        N=8,
        P=16,
        device=device,
    )

    stage_root = tmp_path / "stages"
    exported_increment = cute_aot_mod.export_chunk_increment_cute_aot(
        U,
        M,
        K,
        B,
        U_prev=U_prev,
        B_prev=B_prev,
        chunk_size=32,
        compute_dtype=torch.float32,
        package_root=stage_root,
    )
    increment, chunk_multiplier = v2x2ssd_fwd_mod.chunk_increment_cute(
        U,
        M,
        K,
        B,
        U_prev=U_prev,
        B_prev=B_prev,
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    exported_state = cute_aot_mod.export_state_passing_cute_aot(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
        package_root=stage_root,
    )
    chunk_starts, _final_state = v2x2ssd_fwd_mod.state_passing_cute(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
    )
    exported_scan = cute_aot_mod.export_chunk_scan_cute_aot(
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
        output_dtype=torch.float16,
        package_root=stage_root,
    )

    for exported in (exported_increment, exported_state, exported_scan):
        assert exported.object_file is not None
        assert exported.object_file.is_file()
        assert exported.shared_library.is_file()
        assert exported.metadata_file.is_file()
