from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

pytest.importorskip("cutlass")

import slinoss.ops.v2x2ssd.cute.aot as cute_aot_mod
import slinoss.ops.v2x2ssd.cute.kernels.bwd as v2x2ssd_bwd_mod
import slinoss.ops.v2x2ssd.cute.kernels.fwd as v2x2ssd_fwd_mod
from slinoss.ops.v2x2ssd.cute.tuning.hardware import current_hardware_fingerprint
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
    monkeypatch.setenv("SLINOSS_CUTE_AOT_ARCH_TAGS", "sm_70,sm_89,9.0,sm_89")
    monkeypatch.setenv("SLINOSS_CUTE_FORWARD_AOT_ARCH_TAGS", "sm_80")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.6")
    assert cute_aot_mod._arch_tags_from_env() == ("sm_89", "sm_90")


def test_arch_tags_from_env_falls_back_to_torch_arch_list(monkeypatch) -> None:
    monkeypatch.delenv("SLINOSS_CUTE_AOT_ARCH_TAGS", raising=False)
    monkeypatch.delenv("SLINOSS_CUTE_FORWARD_AOT_ARCH_TAGS", raising=False)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX 8.0")
    assert cute_aot_mod._arch_tags_from_env() == ("sm_80", "sm_86")


def test_arch_tags_from_env_drops_unsupported_forward_arches(monkeypatch) -> None:
    monkeypatch.delenv("SLINOSS_CUTE_AOT_ARCH_TAGS", raising=False)
    monkeypatch.delenv("SLINOSS_CUTE_FORWARD_AOT_ARCH_TAGS", raising=False)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0;8.0;8.6")
    assert cute_aot_mod._arch_tags_from_env() == ("sm_80", "sm_86")


def test_default_forward_aot_specs_expand_requested_arch_tags() -> None:
    specs = cute_aot_mod.default_forward_aot_specs(("sm_80", "sm_86"))
    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_80", "sm_86"}
    assert all(spec.bc_groups is None for spec in specs)
    assert {spec.chunk_size for spec in specs} == set(
        cute_aot_mod._DEFAULT_FORWARD_AOT_CHUNK_SIZES
    )
    assert {spec.output_dtype_name for spec in specs} == {"float16", "bfloat16"}
    assert len(specs) == (
        2
        * len(cute_aot_mod._DEFAULT_FORWARD_AOT_CHUNK_SIZES)
        * len(cute_aot_mod._AOT_SEARCH_TC_DTYPES)
        * 2
    )


def test_default_forward_aot_specs_drop_unsupported_requested_arch_tags() -> None:
    specs = cute_aot_mod.default_forward_aot_specs(("sm_70", "sm_80"))
    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_80"}


def test_default_backward_aot_specs_expand_requested_arch_tags() -> None:
    specs = cute_aot_mod.default_backward_aot_specs(("sm_80", "sm_86"))
    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_80", "sm_86"}
    assert all(spec.bc_groups is None for spec in specs)
    assert {spec.chunk_size for spec in specs} == set(
        cute_aot_mod._DEFAULT_FORWARD_AOT_CHUNK_SIZES
    )
    assert {spec.tc_dtype_name for spec in specs} == {"float16", "bfloat16"}
    assert len(specs) == (
        2
        * len(cute_aot_mod._DEFAULT_FORWARD_AOT_CHUNK_SIZES)
        * len(cute_aot_mod._AOT_SEARCH_TC_DTYPES)
    )


def test_search_space_forward_aot_specs_expand_beyond_default_specs() -> None:
    default_specs = cute_aot_mod.default_forward_aot_specs(("sm_80",))
    search_specs = cute_aot_mod.search_space_forward_aot_specs(("sm_80",))
    assert len(search_specs) > len(default_specs)
    assert set(cute_aot_mod._DEFAULT_FORWARD_AOT_CHUNK_SIZES) <= {
        spec.chunk_size for spec in search_specs
    }
    assert {spec.chunk_size for spec in search_specs} <= set(
        cute_aot_mod._AOT_SEARCH_CHUNK_SIZES
    )
    assert {spec.output_dtype_name for spec in search_specs} == {
        "float16",
        "bfloat16",
        "float32",
    }


def test_v2x2ssd_aot_record_load_defaults_bc_groups_to_head_matched_identity() -> None:
    increment_base = cute_aot_mod._search_space_chunk_increment_specs(arch_tag="sm_80")[
        0
    ]
    increment = cute_aot_mod._chunk_increment_spec_from_record(
        {
            key: value
            for key, value in asdict(increment_base).items()
            if key != "bc_groups"
        }
    )
    assert increment.bc_groups is None
    assert increment.module_id == increment_base.module_id

    scan_base = cute_aot_mod._search_space_chunk_scan_specs(arch_tag="sm_80")[0]
    scan = cute_aot_mod._chunk_scan_spec_from_record(
        {key: value for key, value in asdict(scan_base).items() if key != "bc_groups"}
    )
    assert scan.bc_groups is None
    assert scan.module_id == scan_base.module_id

    forward_base = cute_aot_mod._search_space_forward_specs(arch_tag="sm_80")[0]
    forward = cute_aot_mod._forward_spec_from_record(
        {
            key: value
            for key, value in asdict(forward_base).items()
            if key != "bc_groups"
        }
    )
    assert forward.bc_groups is None
    assert forward.module_id == forward_base.module_id

    backward_base = cute_aot_mod.default_backward_aot_specs(("sm_80",))[0]
    backward = cute_aot_mod._backward_spec_from_record(
        {
            key: value
            for key, value in asdict(backward_base).items()
            if key != "bc_groups"
        }
    )
    assert backward.bc_groups is None
    assert backward.module_id == backward_base.module_id


def test_grouped_v2x2ssd_aot_module_ids_include_bc_groups_identity() -> None:
    increment_base = cute_aot_mod._search_space_chunk_increment_specs(arch_tag="sm_80")[
        0
    ]
    increment_grouped = replace(increment_base, bc_groups=1)
    assert increment_grouped.module_id != increment_base.module_id
    assert "g1" in increment_grouped.module_id

    scan_base = cute_aot_mod._search_space_chunk_scan_specs(arch_tag="sm_80")[0]
    scan_grouped = replace(scan_base, bc_groups=1)
    assert scan_grouped.module_id != scan_base.module_id
    assert "g1" in scan_grouped.module_id

    forward_base = cute_aot_mod._search_space_forward_specs(arch_tag="sm_80")[0]
    forward_grouped = replace(forward_base, bc_groups=1)
    assert forward_grouped.module_id != forward_base.module_id
    assert "g1" in forward_grouped.module_id

    backward_base = cute_aot_mod.default_backward_aot_specs(("sm_80",))[0]
    backward_grouped = replace(backward_base, bc_groups=1)
    assert backward_grouped.module_id != backward_base.module_id
    assert "g1" in backward_grouped.module_id


def test_build_default_forward_aot_package_only_exports_forward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = cute_aot_mod.default_forward_aot_specs(("sm_80",))[:2]
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    def _stage_compile_should_not_run(spec: object):
        raise AssertionError(f"unexpected stage compile for {spec!r}")

    monkeypatch.setattr(
        cute_aot_mod,
        "_compile_chunk_increment_aot",
        _stage_compile_should_not_run,
    )
    monkeypatch.setattr(
        cute_aot_mod,
        "_compile_state_passing_aot",
        _stage_compile_should_not_run,
    )
    monkeypatch.setattr(
        cute_aot_mod,
        "_compile_chunk_scan_aot",
        _stage_compile_should_not_run,
    )
    monkeypatch.setattr(
        cute_aot_mod,
        "_compile_forward_aot",
        lambda spec: compiled_ids.append(spec.module_id) or "compiled",
    )

    def _fake_export(
        compiled,
        *,
        kind: str,
        module_id: str,
        function_name: str,
        package_root,
        keep_object_file: bool = True,
    ):
        exported_kinds.append(kind)
        package_root = Path(package_root)
        return cute_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(cute_aot_mod, "export_tvm_ffi_compiled_module", _fake_export)
    monkeypatch.setattr(
        cute_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = cute_aot_mod.build_default_forward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["v2x2ssd_fwd", "v2x2ssd_fwd"]
    assert registered_kinds == ["v2x2ssd_fwd", "v2x2ssd_fwd"]


def test_build_default_backward_aot_package_only_exports_backward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = cute_aot_mod.default_backward_aot_specs(("sm_80",))[:2]
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    monkeypatch.setattr(
        cute_aot_mod,
        "_compile_backward_aot",
        lambda spec: compiled_ids.append(spec.module_id) or "compiled",
    )

    def _fake_export(
        compiled,
        *,
        kind: str,
        module_id: str,
        function_name: str,
        package_root,
        keep_object_file: bool = True,
    ):
        exported_kinds.append(kind)
        package_root = Path(package_root)
        return cute_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(cute_aot_mod, "export_tvm_ffi_compiled_module", _fake_export)
    monkeypatch.setattr(
        cute_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = cute_aot_mod.build_default_backward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["v2x2ssd_bwd", "v2x2ssd_bwd"]
    assert registered_kinds == ["v2x2ssd_bwd", "v2x2ssd_bwd"]


def test_build_default_cute_aot_package_builds_forward_then_backward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def _fake_forward(*, package_root, specs=None, arch_tags=None, clean=True):
        calls.append(("forward", clean))
        return (
            cute_aot_mod.ExportedTVMFFIModule(
                kind="v2x2ssd_fwd",
                module_id="fwd",
                function_name="fwd",
                object_file=None,
                shared_library=Path(package_root) / "artifacts" / "fwd.so",
                metadata_file=Path(package_root) / "artifacts" / "fwd.json",
            ),
        )

    def _fake_backward(*, package_root, specs=None, arch_tags=None, clean=True):
        calls.append(("backward", clean))
        return (
            cute_aot_mod.ExportedTVMFFIModule(
                kind="v2x2ssd_bwd",
                module_id="bwd",
                function_name="bwd",
                object_file=None,
                shared_library=Path(package_root) / "artifacts" / "bwd.so",
                metadata_file=Path(package_root) / "artifacts" / "bwd.json",
            ),
        )

    monkeypatch.setattr(
        cute_aot_mod,
        "build_default_forward_aot_package",
        _fake_forward,
    )
    monkeypatch.setattr(
        cute_aot_mod,
        "build_default_backward_aot_package",
        _fake_backward,
    )

    exported = cute_aot_mod.build_default_cute_aot_package(package_root=tmp_path)

    assert calls == [("forward", True), ("backward", False)]
    assert [module.kind for module in exported] == ["v2x2ssd_fwd", "v2x2ssd_bwd"]


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
        (
            "_compile_backward_aot",
            cute_aot_mod.default_backward_aot_specs(("sm_80",))[0],
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
    bc_groups: int | None = None,
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
    resolved_bc_groups = heads if bc_groups is None else int(bc_groups)
    if resolved_bc_groups <= 0:
        raise ValueError("bc_groups must be positive.")
    if heads % resolved_bc_groups != 0:
        raise ValueError(
            f"bc_groups must divide heads. Got heads={heads}, bc_groups={resolved_bc_groups}."
        )
    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=value_dtype)
    B = (
        torch.randn(
            (batch, resolved_bc_groups, T, 2 * N),
            device=device,
            dtype=value_dtype,
        )
        * 0.1
    )
    C = (
        torch.randn(
            (batch, resolved_bc_groups, T, 2 * N),
            device=device,
            dtype=value_dtype,
        )
        * 0.1
    )
    initial_states = torch.randn(
        (batch, heads, P, 2 * N), device=device, dtype=torch.float32
    )

    b_prev = (
        torch.randn((batch, resolved_bc_groups, N), device=device, dtype=torch.float32)
        + 1j
        * torch.randn(
            (batch, resolved_bc_groups, N), device=device, dtype=torch.float32
        )
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=torch.float32)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, C, initial_states, B_prev, U_prev


def _make_backward_inputs(
    *,
    batch: int,
    heads: int,
    T: int,
    N: int,
    P: int,
    device: torch.device,
    value_dtype: torch.dtype = torch.float16,
    bc_groups: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=batch,
        heads=heads,
        bc_groups=bc_groups,
        T=T,
        N=N,
        P=P,
        device=device,
        value_dtype=value_dtype,
    )
    _output, _final_state, m_chunk, chunk_starts = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        v2x2ssd_fwd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=32,
            compute_dtype=torch.float32,
            output_dtype=value_dtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            return_final_state=True,
            return_intermediates=True,
        ),
    )
    d_out = torch.randn_like(U)
    return (
        U,
        M,
        K,
        B,
        C,
        initial_states,
        B_prev,
        U_prev,
        m_chunk,
        chunk_starts,
        d_out,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_v2x2ssd_fwd_aot_spec_uses_fp16_tc_dtype_for_float32_inputs() -> None:
    torch.manual_seed(0)
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1,
        heads=1,
        T=32,
        N=8,
        P=16,
        device=torch.device("cuda"),
        value_dtype=torch.float32,
    )

    spec = cute_aot_mod.infer_v2x2ssd_fwd_aot_spec(
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
    )

    assert spec.tc_dtype == torch.float16


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
    wrapped_loaded = v2x2ssd_fwd_mod._make_packaged_v2x2ssd_fwd_callable(loaded)
    wrapped_loaded(*runtime_artifacts.runtime_args)
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
    arch_tag = current_hardware_fingerprint(
        device_index=torch.cuda.current_device()
    ).arch_tag

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
        arch_tag=arch_tag,
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
def test_v2x2ssd_fwd_grouped_packaged_aot_prefers_packaged_over_jit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    device = torch.device("cuda")
    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=1,
        heads=4,
        bc_groups=1,
        T=32,
        N=8,
        P=16,
        device=device,
    )
    arch_tag = current_hardware_fingerprint(
        device_index=torch.cuda.current_device()
    ).arch_tag

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
        arch_tag=arch_tag,
        package_root=tmp_path,
    )

    monkeypatch.setattr(cute_aot_mod, "_PACKAGED_AOT_ROOT", tmp_path)
    cute_aot_mod.clear_packaged_forward_aot_cache()
    v2x2ssd_fwd_mod._FWD_HOST_CACHE.clear()

    def _compile_should_not_run(*args, **kwargs):
        raise AssertionError(
            "expected grouped packaged AOT forward to bypass cute.compile"
        )

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
def test_export_v2x2ssd_bwd_cute_aot_roundtrips_through_loaded_module(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    device = torch.device("cuda")
    (
        U,
        M,
        K,
        B,
        C,
        initial_states,
        B_prev,
        U_prev,
        m_chunk,
        chunk_starts,
        d_out,
    ) = _make_backward_inputs(
        batch=1,
        heads=1,
        T=32,
        N=8,
        P=16,
        device=device,
    )
    d_final_state = torch.randn_like(initial_states)

    exported = cute_aot_mod.export_v2x2ssd_bwd_cute_aot(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=32,
        compute_dtype=torch.float32,
        package_root=tmp_path,
    )
    assert exported.object_file is not None
    assert exported.object_file.is_file()
    assert exported.shared_library.is_file()
    assert exported.metadata_file.is_file()

    runtime_artifacts = v2x2ssd_bwd_mod._make_backward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=32,
        compute_dtype=torch.float32,
        scan_num_threads_du=128,
        scan_num_threads_db=128,
        scan_num_threads_dcdr=128,
        scan_num_threads_param=32,
        state_num_threads=128,
        state_pairs_per_thread=8,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
        prepared_inputs=None,
        validate_runtime_contract=True,
    )
    loaded = cute_aot_mod.load_tvm_ffi_function(
        exported.shared_library,
        function_name=exported.function_name,
    )
    wrapped_loaded = v2x2ssd_bwd_mod._make_packaged_v2x2ssd_bwd_callable(
        loaded,
        problem_shape=runtime_artifacts.problem_shape,
    )
    wrapped_loaded(*runtime_artifacts.runtime_args)
    torch.cuda.synchronize(device)

    runtime_public = v2x2ssd_bwd_mod._materialize_backward_public_outputs(
        runtime_artifacts,
        time_steps=U.shape[2],
        U_dtype=U.dtype,
        B_dtype=B.dtype,
        C_dtype=C.dtype,
        initial_state_dtype=initial_states.dtype,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    reference = v2x2ssd_bwd_mod._v2x2ssd_bwd_cute_prevalidated(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=32,
        compute_dtype=torch.float32,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
    )
    for actual, expected in zip(runtime_public, reference, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_bwd_cute_prefers_packaged_aot_over_jit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    device = torch.device("cuda")
    (
        U,
        M,
        K,
        B,
        C,
        initial_states,
        B_prev,
        U_prev,
        m_chunk,
        chunk_starts,
        d_out,
    ) = _make_backward_inputs(
        batch=1,
        heads=1,
        T=32,
        N=8,
        P=16,
        device=device,
    )
    d_final_state = torch.randn_like(initial_states)
    arch_tag = current_hardware_fingerprint(
        device_index=torch.cuda.current_device()
    ).arch_tag

    cute_aot_mod.export_v2x2ssd_bwd_cute_aot(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=32,
        compute_dtype=torch.float32,
        arch_tag=arch_tag,
        package_root=tmp_path,
    )

    monkeypatch.setattr(cute_aot_mod, "_PACKAGED_AOT_ROOT", tmp_path)
    cute_aot_mod.clear_packaged_aot_cache()
    v2x2ssd_bwd_mod._BWD_HOST_CACHE.clear()

    def _compile_should_not_run(*args, **kwargs):
        raise AssertionError("expected packaged AOT backward to bypass cute.compile")

    monkeypatch.setattr(v2x2ssd_bwd_mod.cute, "compile", _compile_should_not_run)
    grads = v2x2ssd_bwd_mod._v2x2ssd_bwd_cute_prevalidated(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=32,
        compute_dtype=torch.float32,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
    )
    assert len(grads) == 8


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
