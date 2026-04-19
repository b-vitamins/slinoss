from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")

import slinoss._cute_aot as cute_aot_common
from slinoss.layers.scanprep import SLinOSSScanPrep
import slinoss.ops.scanprep.cute.aot as scanprep_aot_mod
import slinoss.ops.scanprep.cute.kernels as scanprep_kernels_mod


def test_scanprep_aot_defaults_match_scanprep_layer_defaults() -> None:
    init_sig = inspect.signature(SLinOSSScanPrep.__init__)
    expected = scanprep_aot_mod._DEFAULT_SCANPREP_CONFIG_KWARGS
    for key in expected:
        assert key in init_sig.parameters
        assert expected[key] == init_sig.parameters[key].default


def test_default_forward_aot_specs_expand_requested_arch_tags() -> None:
    specs = scanprep_aot_mod.default_forward_aot_specs(("sm_86", "sm_89"))

    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_86", "sm_89"}
    assert {
        (spec.heads, spec.bc_groups, spec.d_head, spec.d_state) for spec in specs
    } == {
        (24, 1, 64, 128),
        (32, 1, 64, 128),
        (48, 1, 64, 128),
        (64, 1, 64, 128),
    }
    assert {spec.store_coeff_aux for spec in specs} == {False, True}
    assert {spec.value_dtype_name for spec in specs} == {"bfloat16"}
    assert len(specs) == 2 * 4 * 1 * 2


def test_default_backward_aot_specs_expand_requested_arch_tags() -> None:
    specs = scanprep_aot_mod.default_backward_aot_specs(("sm_86", "sm_89"))

    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_86", "sm_89"}
    assert {
        (spec.heads, spec.bc_groups, spec.d_head, spec.d_state) for spec in specs
    } == {
        (24, 1, 64, 128),
        (32, 1, 64, 128),
        (48, 1, 64, 128),
        (64, 1, 64, 128),
    }
    assert {spec.bc_dtype_name for spec in specs} == {"bfloat16"}
    assert len(specs) == 2 * 4 * 1


def test_scanprep_aot_record_load_defaults_bc_groups_to_heads() -> None:
    record = {
        "arch_tag": "sm_80",
        "heads": 8,
        "d_head": 64,
        "d_state": 128,
        "value_dtype_name": "float16",
        "params_dtype_name": "float16",
        "bc_dtype_name": "float16",
        "bias_dtype_name": "float16",
        "store_coeff_aux": False,
        "config": scanprep_aot_mod._DEFAULT_SCANPREP_CONFIG_KWARGS,
    }

    fwd = scanprep_aot_mod._forward_spec_from_record(record)
    assert fwd.bc_groups == fwd.heads == 8
    assert "_g8_" in fwd.module_id

    bwd = scanprep_aot_mod._backward_spec_from_record(
        {
            **record,
            "value_grad_dtype_name": "float16",
            "params_grad_dtype_name": "float16",
        }
    )
    assert bwd.bc_groups == bwd.heads == 8
    assert "_g8_" in bwd.module_id


def test_build_default_forward_aot_package_only_exports_forward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = scanprep_aot_mod.default_forward_aot_specs(("sm_86",))[:2]
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    monkeypatch.setattr(
        scanprep_aot_mod,
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
        return scanprep_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(
        scanprep_aot_mod,
        "export_tvm_ffi_compiled_module",
        _fake_export,
    )
    monkeypatch.setattr(
        scanprep_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = scanprep_aot_mod.build_default_forward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["scanprep_fwd", "scanprep_fwd"]
    assert registered_kinds == ["scanprep_fwd", "scanprep_fwd"]


def test_build_default_backward_aot_package_only_exports_backward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = scanprep_aot_mod.default_backward_aot_specs(("sm_86",))[:2]
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    monkeypatch.setattr(
        scanprep_aot_mod,
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
        return scanprep_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(
        scanprep_aot_mod,
        "export_tvm_ffi_compiled_module",
        _fake_export,
    )
    monkeypatch.setattr(
        scanprep_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = scanprep_aot_mod.build_default_backward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["scanprep_bwd", "scanprep_bwd"]
    assert registered_kinds == ["scanprep_bwd", "scanprep_bwd"]


def test_build_default_cute_aot_package_builds_forward_then_backward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def _fake_forward(*, package_root, specs=None, arch_tags=None, clean=True):
        calls.append(("forward", clean))
        return (
            scanprep_aot_mod.ExportedTVMFFIModule(
                kind="scanprep_fwd",
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
            scanprep_aot_mod.ExportedTVMFFIModule(
                kind="scanprep_bwd",
                module_id="bwd",
                function_name="bwd",
                object_file=None,
                shared_library=Path(package_root) / "artifacts" / "bwd.so",
                metadata_file=Path(package_root) / "artifacts" / "bwd.json",
            ),
        )

    monkeypatch.setattr(
        scanprep_aot_mod,
        "build_default_forward_aot_package",
        _fake_forward,
    )
    monkeypatch.setattr(
        scanprep_aot_mod,
        "build_default_backward_aot_package",
        _fake_backward,
    )

    exported = scanprep_aot_mod.build_default_cute_aot_package(package_root=tmp_path)

    assert [module.module_id for module in exported] == ["fwd", "bwd"]
    assert calls == [("forward", True), ("backward", False)]


def test_get_compiled_scanprep_fwd_kernel_prefers_packaged_aot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = scanprep_kernels_mod._make_scanprep_config(
        **scanprep_aot_mod._DEFAULT_SCANPREP_CONFIG_KWARGS
    )
    runtime_artifacts = scanprep_kernels_mod._make_forward_runtime_artifacts(
        torch.empty((2, 5, 8), dtype=torch.float16),
        torch.empty((2, 5, 6), dtype=torch.float16),
        torch.empty((2, 5, 2, 4, 3), dtype=torch.float16),
        config=config,
        n_heads=2,
        d_state=3,
        d_head=4,
        dt_bias=torch.empty((2,), dtype=torch.float32),
        alpha_bias=torch.empty((2,), dtype=torch.float32),
        theta_mod_bias=torch.empty((2,), dtype=torch.float32),
        theta_bias=torch.empty((2,), dtype=torch.float32),
        theta_sign=torch.empty((2,), dtype=torch.float32),
        store_coeff_aux=False,
    )
    compile_artifacts = (
        scanprep_kernels_mod._make_forward_compile_artifacts_from_runtime_artifacts(
            runtime_artifacts
        )
    )
    scanprep_kernels_mod._SCANPREP_FWD_CACHE.clear()
    captured_specs: list[scanprep_aot_mod.ForwardAOTSpec] = []
    packaged = object()

    monkeypatch.setattr(
        cute_aot_common, "current_cuda_arch_tag", lambda device: "sm_80"
    )
    monkeypatch.setattr(
        scanprep_aot_mod,
        "try_load_packaged_scanprep_fwd_function",
        lambda spec: captured_specs.append(spec) or packaged,
    )
    monkeypatch.setattr(
        scanprep_kernels_mod,
        "_compile_scanprep_fwd_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected JIT compile")
        ),
    )

    compiled = scanprep_kernels_mod._get_compiled_scanprep_fwd_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=torch.device("cuda", 0),
        config=config,
    )

    assert compiled is packaged
    assert captured_specs and captured_specs[0].arch_tag == "sm_80"
    assert captured_specs[0].store_coeff_aux is False


def test_get_compiled_scanprep_bwd_kernel_prefers_packaged_aot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = scanprep_kernels_mod._make_scanprep_config(
        **scanprep_aot_mod._DEFAULT_SCANPREP_CONFIG_KWARGS
    )
    runtime_artifacts = scanprep_kernels_mod._make_backward_runtime_artifacts(
        torch.empty((2, 5, 2, 4, 3), dtype=torch.float16),
        torch.empty((2, 2, 14, 5), dtype=torch.float32),
        config=config,
        dU=torch.empty((2, 2, 5, 4), dtype=torch.float16),
        dM=torch.empty((2, 2, 5, 2), dtype=torch.float32),
        dK=torch.empty((2, 2, 5, 2, 2), dtype=torch.float32),
        dB=torch.empty((2, 2, 5, 6), dtype=torch.float16),
        dC=torch.empty((2, 2, 5, 6), dtype=torch.float16),
        n_heads=2,
        d_head=4,
        d_state=3,
        value_dtype=torch.float16,
        params_dtype=torch.float16,
        dt_bias=torch.empty((2,), dtype=torch.float32),
        theta_bias=torch.empty((2,), dtype=torch.float32),
        theta_sign=torch.empty((2,), dtype=torch.float32),
    )
    compile_artifacts = (
        scanprep_kernels_mod._make_backward_compile_artifacts_from_runtime_artifacts(
            runtime_artifacts
        )
    )
    scanprep_kernels_mod._SCANPREP_BWD_CACHE.clear()
    captured_specs: list[scanprep_aot_mod.BackwardAOTSpec] = []
    packaged = object()

    monkeypatch.setattr(
        cute_aot_common, "current_cuda_arch_tag", lambda device: "sm_80"
    )
    monkeypatch.setattr(
        scanprep_aot_mod,
        "try_load_packaged_scanprep_bwd_function",
        lambda spec: captured_specs.append(spec) or packaged,
    )
    monkeypatch.setattr(
        scanprep_kernels_mod,
        "_compile_scanprep_bwd_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected JIT compile")
        ),
    )

    compiled = scanprep_kernels_mod._get_compiled_scanprep_bwd_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=torch.device("cuda", 0),
        config=config,
    )

    assert compiled is packaged
    assert captured_specs and captured_specs[0].arch_tag == "sm_80"
    assert captured_specs[0].bc_dtype_name == "float16"
