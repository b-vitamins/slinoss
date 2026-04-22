from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")

import slinoss._cute_aot as cute_aot_common
import slinoss.ops.block.cute.activation as block_activation_mod
import slinoss.ops.block.cute.aot as block_aot_mod
import slinoss.ops.block.cute.common as block_common_mod
import slinoss.ops.block.cute.norm as block_norm_mod


def test_default_forward_aot_specs_expand_requested_arch_tags() -> None:
    specs = block_aot_mod.default_forward_aot_specs(("sm_86", "sm_89"))
    norm_specs = tuple(
        spec for spec in specs if isinstance(spec, block_aot_mod.NormForwardAOTSpec)
    )
    activation_specs = tuple(
        spec
        for spec in specs
        if isinstance(spec, block_aot_mod.ActivationForwardAOTSpec)
    )

    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_86", "sm_89"}
    assert {spec.d_model for spec in norm_specs} == {576, 768, 1024}
    assert {(spec.hidden_dim, spec.kind) for spec in activation_specs} == {
        (2304, "gelu"),
        (2304, "swiglu"),
        (3072, "gelu"),
        (3072, "swiglu"),
        (4096, "gelu"),
        (4096, "swiglu"),
    }


def test_default_backward_aot_specs_expand_requested_arch_tags() -> None:
    specs = block_aot_mod.default_backward_aot_specs(("sm_86", "sm_89"))
    norm_specs = tuple(
        spec for spec in specs if isinstance(spec, block_aot_mod.NormBackwardAOTSpec)
    )
    activation_specs = tuple(
        spec
        for spec in specs
        if isinstance(spec, block_aot_mod.ActivationBackwardAOTSpec)
    )

    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_86", "sm_89"}
    assert {spec.d_model for spec in norm_specs} == {576, 768, 1024}
    assert {(spec.hidden_dim, spec.kind) for spec in activation_specs} == {
        (2304, "gelu"),
        (2304, "swiglu"),
        (3072, "gelu"),
        (3072, "swiglu"),
        (4096, "gelu"),
        (4096, "swiglu"),
    }


def test_build_default_forward_aot_package_only_exports_forward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    all_specs = block_aot_mod.default_forward_aot_specs(("sm_86",))
    specs = (
        next(
            spec
            for spec in all_specs
            if isinstance(spec, block_aot_mod.NormForwardAOTSpec)
        ),
        next(
            spec
            for spec in all_specs
            if isinstance(spec, block_aot_mod.ActivationForwardAOTSpec)
        ),
    )
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    monkeypatch.setattr(
        block_aot_mod,
        "_compile_norm_forward_aot",
        lambda spec: compiled_ids.append(spec.module_id) or "compiled",
    )
    monkeypatch.setattr(
        block_aot_mod,
        "_compile_activation_forward_aot",
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
        del compiled, keep_object_file
        exported_kinds.append(kind)
        package_root = Path(package_root)
        return block_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(
        block_aot_mod,
        "export_tvm_ffi_compiled_module",
        _fake_export,
    )
    monkeypatch.setattr(
        block_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = block_aot_mod.build_default_forward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["ffn_norm_fwd", "ffn_activation_fwd"]
    assert registered_kinds == ["ffn_norm_fwd", "ffn_activation_fwd"]


def test_build_default_backward_aot_package_only_exports_backward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    all_specs = block_aot_mod.default_backward_aot_specs(("sm_86",))
    specs = (
        next(
            spec
            for spec in all_specs
            if isinstance(spec, block_aot_mod.NormBackwardAOTSpec)
        ),
        next(
            spec
            for spec in all_specs
            if isinstance(spec, block_aot_mod.ActivationBackwardAOTSpec)
        ),
    )
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    monkeypatch.setattr(
        block_aot_mod,
        "_compile_norm_backward_aot",
        lambda spec: compiled_ids.append(spec.module_id) or "compiled",
    )
    monkeypatch.setattr(
        block_aot_mod,
        "_compile_activation_backward_aot",
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
        del compiled, keep_object_file
        exported_kinds.append(kind)
        package_root = Path(package_root)
        return block_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(
        block_aot_mod,
        "export_tvm_ffi_compiled_module",
        _fake_export,
    )
    monkeypatch.setattr(
        block_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = block_aot_mod.build_default_backward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["ffn_norm_bwd", "ffn_activation_bwd"]
    assert registered_kinds == ["ffn_norm_bwd", "ffn_activation_bwd"]


def test_build_default_cute_aot_package_builds_forward_then_backward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def _fake_forward(*, package_root, specs=None, arch_tags=None, clean=True):
        del specs, arch_tags
        calls.append(("forward", clean))
        return (
            block_aot_mod.ExportedTVMFFIModule(
                kind="ffn_norm_fwd",
                module_id="fwd",
                function_name="fwd",
                object_file=None,
                shared_library=Path(package_root) / "artifacts" / "fwd.so",
                metadata_file=Path(package_root) / "artifacts" / "fwd.json",
            ),
        )

    def _fake_backward(*, package_root, specs=None, arch_tags=None, clean=True):
        del specs, arch_tags
        calls.append(("backward", clean))
        return (
            block_aot_mod.ExportedTVMFFIModule(
                kind="ffn_norm_bwd",
                module_id="bwd",
                function_name="bwd",
                object_file=None,
                shared_library=Path(package_root) / "artifacts" / "bwd.so",
                metadata_file=Path(package_root) / "artifacts" / "bwd.json",
            ),
        )

    monkeypatch.setattr(
        block_aot_mod,
        "build_default_forward_aot_package",
        _fake_forward,
    )
    monkeypatch.setattr(
        block_aot_mod,
        "build_default_backward_aot_package",
        _fake_backward,
    )

    exported = block_aot_mod.build_default_cute_aot_package(package_root=tmp_path)

    assert [module.module_id for module in exported] == ["fwd", "bwd"]
    assert calls == [("forward", True), ("backward", False)]


def _make_norm_forward_runtime_artifacts() -> block_norm_mod.ForwardRuntimeArtifacts:
    input_info = block_common_mod.FfnNormInputInfo(
        batch_size=2,
        time_steps=5,
        hidden_dim=12,
        device_index=0,
    )
    residual = torch.empty((2, 5, 12), dtype=torch.float16)
    norm_weight = torch.empty((12,), dtype=torch.float16)
    output = torch.empty_like(residual)
    runtime_args = (residual, norm_weight, output)
    alignments = block_common_mod._runtime_alignments(runtime_args)
    return block_norm_mod.ForwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        output=output,
        alignments=alignments,
        cache_key=(
            input_info,
            1.0e-5,
            alignments,
            block_common_mod._runtime_signature_key(runtime_args),
        ),
    )


def _make_activation_backward_runtime_artifacts() -> (
    block_activation_mod.BackwardRuntimeArtifacts
):
    input_info = block_common_mod.FfnActivationInputInfo(
        batch_size=2,
        time_steps=5,
        hidden_dim=16,
        kind="swiglu",
        device_index=0,
    )
    projected = torch.empty((2, 5, 32), dtype=torch.float16)
    d_hidden = torch.empty((2, 5, 16), dtype=torch.float16)
    output = torch.empty_like(projected)
    runtime_args = (projected, d_hidden, output)
    alignments = block_common_mod._runtime_alignments(runtime_args)
    return block_activation_mod.BackwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        output=output,
        alignments=alignments,
        cache_key=(
            input_info,
            alignments,
            block_common_mod._runtime_signature_key(runtime_args),
        ),
    )


def test_get_compiled_ffn_norm_fwd_kernel_prefers_packaged_aot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_artifacts = _make_norm_forward_runtime_artifacts()
    compile_artifacts = block_norm_mod._make_forward_compile_artifacts(
        runtime_artifacts,
        eps=1.0e-5,
    )
    captured_specs: list[block_aot_mod.NormForwardAOTSpec] = []
    packaged = object()

    block_norm_mod._NORM_FWD_CACHE.clear()
    block_aot_mod.clear_packaged_aot_cache()

    monkeypatch.setattr(
        cute_aot_common, "current_cuda_arch_tag", lambda device: "sm_80"
    )
    monkeypatch.setattr(
        block_aot_mod,
        "try_load_packaged_ffn_norm_fwd_function",
        lambda spec: captured_specs.append(spec) or packaged,
    )
    monkeypatch.setattr(
        block_norm_mod,
        "_compile_forward_kernel",
        lambda *args, **kwargs: pytest.fail("unexpected JIT compile"),
    )

    compiled = block_norm_mod._get_compiled_forward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=torch.device("cuda"),
        eps=1.0e-5,
    )

    assert compiled is packaged
    assert len(captured_specs) == 1
    assert captured_specs[0].arch_tag == "sm_80"
    assert captured_specs[0].d_model == 12
    assert captured_specs[0].residual_dtype_name == "float16"
    assert captured_specs[0].norm_weight_dtype_name == "float16"


def test_get_compiled_ffn_activation_bwd_kernel_prefers_packaged_aot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_artifacts = _make_activation_backward_runtime_artifacts()
    compile_artifacts = block_activation_mod._make_backward_compile_artifacts(
        runtime_artifacts
    )
    captured_specs: list[block_aot_mod.ActivationBackwardAOTSpec] = []
    packaged = object()

    block_activation_mod._ACT_BWD_CACHE.clear()
    block_aot_mod.clear_packaged_aot_cache()

    monkeypatch.setattr(
        cute_aot_common, "current_cuda_arch_tag", lambda device: "sm_80"
    )
    monkeypatch.setattr(
        block_aot_mod,
        "try_load_packaged_ffn_activation_bwd_function",
        lambda spec: captured_specs.append(spec) or packaged,
    )
    monkeypatch.setattr(
        block_activation_mod,
        "_compile_backward_kernel",
        lambda *args, **kwargs: pytest.fail("unexpected JIT compile"),
    )

    compiled = block_activation_mod._get_compiled_backward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=torch.device("cuda"),
    )

    assert compiled is packaged
    assert len(captured_specs) == 1
    assert captured_specs[0].arch_tag == "sm_80"
    assert captured_specs[0].hidden_dim == 16
    assert captured_specs[0].kind == "swiglu"
    assert captured_specs[0].projected_dtype_name == "float16"
    assert captured_specs[0].d_hidden_dtype_name == "float16"
