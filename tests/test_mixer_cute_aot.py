from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")

import slinoss._cute_aot as cute_aot_common
import slinoss.ops.mixer.cute.aot as mixer_aot_mod
import slinoss.ops.mixer.cute.bwd as mixer_bwd_mod
import slinoss.ops.mixer.cute.common as mixer_common_mod
import slinoss.ops.mixer.cute.fwd as mixer_fwd_mod


def test_default_forward_aot_specs_expand_requested_arch_tags() -> None:
    specs = mixer_aot_mod.default_forward_aot_specs(("sm_86", "sm_89"))

    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_86", "sm_89"}
    assert {(spec.heads, spec.d_head, spec.has_skip) for spec in specs} == {
        (18, 64, True),
        (24, 64, True),
        (32, 64, True),
        (48, 64, True),
        (64, 64, True),
    }
    assert {spec.scan_dtype_name for spec in specs} == {"bfloat16"}
    assert {spec.d_skip_dtype_name for spec in specs} == {"float32"}


def test_default_backward_aot_specs_expand_requested_arch_tags() -> None:
    specs = mixer_aot_mod.default_backward_aot_specs(("sm_86", "sm_89"))

    assert specs
    assert {spec.arch_tag for spec in specs} == {"sm_86", "sm_89"}
    assert {(spec.heads, spec.d_head, spec.has_skip) for spec in specs} == {
        (18, 64, True),
        (24, 64, True),
        (32, 64, True),
        (48, 64, True),
        (64, 64, True),
    }
    assert {spec.scan_dtype_name for spec in specs} == {"bfloat16"}
    assert {spec.d_skip_dtype_name for spec in specs} == {"float32"}


def test_build_default_forward_aot_package_only_exports_forward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = mixer_aot_mod.default_forward_aot_specs(("sm_86",))[:2]
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    monkeypatch.setattr(
        mixer_aot_mod,
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
        del compiled, keep_object_file
        exported_kinds.append(kind)
        package_root = Path(package_root)
        return mixer_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(
        mixer_aot_mod,
        "export_tvm_ffi_compiled_module",
        _fake_export,
    )
    monkeypatch.setattr(
        mixer_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = mixer_aot_mod.build_default_forward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["mixer_tail_rowwise_fwd", "mixer_tail_rowwise_fwd"]
    assert registered_kinds == ["mixer_tail_rowwise_fwd", "mixer_tail_rowwise_fwd"]


def test_build_default_backward_aot_package_only_exports_backward_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = mixer_aot_mod.default_backward_aot_specs(("sm_86",))[:2]
    compiled_ids: list[str] = []
    exported_kinds: list[str] = []
    registered_kinds: list[str] = []

    monkeypatch.setattr(
        mixer_aot_mod,
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
        del compiled, keep_object_file
        exported_kinds.append(kind)
        package_root = Path(package_root)
        return mixer_aot_mod.ExportedTVMFFIModule(
            kind=kind,
            module_id=module_id,
            function_name=function_name,
            object_file=None,
            shared_library=package_root / "artifacts" / f"{module_id}.so",
            metadata_file=package_root / "artifacts" / f"{module_id}.json",
        )

    monkeypatch.setattr(
        mixer_aot_mod,
        "export_tvm_ffi_compiled_module",
        _fake_export,
    )
    monkeypatch.setattr(
        mixer_aot_mod,
        "register_aot_artifact",
        lambda **kwargs: registered_kinds.append(kwargs["kind"]),
    )

    exported = mixer_aot_mod.build_default_backward_aot_package(
        package_root=tmp_path,
        specs=specs,
        clean=False,
    )

    assert [module.module_id for module in exported] == [
        spec.module_id for spec in specs
    ]
    assert compiled_ids == [spec.module_id for spec in specs]
    assert exported_kinds == ["mixer_tail_rowwise_bwd", "mixer_tail_rowwise_bwd"]
    assert registered_kinds == ["mixer_tail_rowwise_bwd", "mixer_tail_rowwise_bwd"]


def test_build_default_cute_aot_package_builds_forward_then_backward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def _fake_forward(*, package_root, specs=None, arch_tags=None, clean=True):
        del specs, arch_tags
        calls.append(("forward", clean))
        return (
            mixer_aot_mod.ExportedTVMFFIModule(
                kind="mixer_tail_rowwise_fwd",
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
            mixer_aot_mod.ExportedTVMFFIModule(
                kind="mixer_tail_rowwise_bwd",
                module_id="bwd",
                function_name="bwd",
                object_file=None,
                shared_library=Path(package_root) / "artifacts" / "bwd.so",
                metadata_file=Path(package_root) / "artifacts" / "bwd.json",
            ),
        )

    monkeypatch.setattr(
        mixer_aot_mod,
        "build_default_forward_aot_package",
        _fake_forward,
    )
    monkeypatch.setattr(
        mixer_aot_mod,
        "build_default_backward_aot_package",
        _fake_backward,
    )

    exported = mixer_aot_mod.build_default_cute_aot_package(package_root=tmp_path)

    assert [module.module_id for module in exported] == ["fwd", "bwd"]
    assert calls == [("forward", True), ("backward", False)]


def _make_forward_runtime_artifacts() -> mixer_fwd_mod.ForwardRuntimeArtifacts:
    input_info = mixer_common_mod.TailRowwiseInputInfo(
        batch_size=2,
        time_steps=5,
        heads=3,
        d_head=4,
        has_skip=True,
        device_index=0,
    )
    hidden_dim = input_info.hidden_dim
    scan_output = torch.empty((2, 3, 5, 4), dtype=torch.float16)
    gate = torch.empty((2, 5, hidden_dim), dtype=torch.float16)
    out_norm_weight = torch.empty((hidden_dim,), dtype=torch.float16)
    skip_input = torch.empty((2, 3, 5, 4), dtype=torch.float16)
    d_skip = torch.empty((3,), dtype=torch.float32)
    output = torch.empty((2, 5, hidden_dim), dtype=torch.float16)
    runtime_args = (
        scan_output,
        gate,
        out_norm_weight,
        skip_input,
        d_skip,
        output,
    )
    alignments = mixer_common_mod._runtime_alignments(runtime_args)
    return mixer_fwd_mod.ForwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        output=output,
        storage_dtype=scan_output.dtype,
        alignments=alignments,
        cache_key=(
            input_info,
            1.0e-5,
            mixer_common_mod._runtime_signature_key(runtime_args),
        ),
    )


def _make_backward_runtime_artifacts() -> mixer_bwd_mod.BackwardRuntimeArtifacts:
    input_info = mixer_common_mod.TailRowwiseInputInfo(
        batch_size=2,
        time_steps=5,
        heads=3,
        d_head=4,
        has_skip=True,
        device_index=0,
    )
    hidden_dim = input_info.hidden_dim
    scan_output = torch.empty((2, 3, 5, 4), dtype=torch.float16)
    gate = torch.empty((2, 5, hidden_dim), dtype=torch.float16)
    out_norm_weight = torch.empty((hidden_dim,), dtype=torch.float16)
    skip_input = torch.empty((2, 3, 5, 4), dtype=torch.float16)
    d_skip = torch.empty((3,), dtype=torch.float32)
    d_normed = torch.empty((2, 5, hidden_dim), dtype=torch.float16)
    d_scan_output = torch.empty_like(scan_output)
    d_gate = torch.empty_like(gate)
    d_skip_input = torch.empty_like(scan_output)
    d_d_skip = torch.zeros((3,), dtype=torch.float32)
    d_norm_weight_accum = torch.zeros((hidden_dim,), dtype=torch.float32)
    outputs = mixer_bwd_mod.BackwardOutputs(
        d_scan_output=d_scan_output,
        d_gate=d_gate,
        d_skip_input=d_skip_input,
        d_d_skip=d_d_skip,
        d_norm_weight_accum=d_norm_weight_accum,
    )
    runtime_args = (
        scan_output,
        gate,
        out_norm_weight,
        skip_input,
        d_skip,
        d_normed,
        d_scan_output,
        d_gate,
        d_skip_input,
        d_d_skip,
        d_norm_weight_accum,
    )
    alignments = mixer_common_mod._runtime_alignments(runtime_args)
    return mixer_bwd_mod.BackwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        outputs=outputs,
        alignments=alignments,
        cache_key=(
            input_info,
            1.0e-5,
            mixer_common_mod._runtime_signature_key(runtime_args),
        ),
    )


def test_get_compiled_mixer_tail_rowwise_fwd_kernel_prefers_packaged_aot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_artifacts = _make_forward_runtime_artifacts()
    compile_artifacts = mixer_fwd_mod._make_forward_compile_artifacts(runtime_artifacts)
    captured_specs: list[mixer_aot_mod.ForwardAOTSpec] = []
    packaged = object()

    mixer_fwd_mod._FWD_CACHE.clear()
    mixer_aot_mod.clear_packaged_aot_cache()

    monkeypatch.setattr(
        cute_aot_common, "current_cuda_arch_tag", lambda device: "sm_80"
    )
    monkeypatch.setattr(
        mixer_aot_mod,
        "try_load_packaged_mixer_tail_rowwise_fwd_function",
        lambda spec: captured_specs.append(spec) or packaged,
    )
    monkeypatch.setattr(
        mixer_fwd_mod,
        "_compile_forward_kernel",
        lambda *args, **kwargs: pytest.fail("unexpected JIT compile"),
    )

    compiled = mixer_fwd_mod._get_compiled_forward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=torch.device("cuda"),
        eps=1.0e-5,
    )

    assert compiled is packaged
    assert len(captured_specs) == 1
    assert captured_specs[0].arch_tag == "sm_80"
    assert captured_specs[0].heads == 3
    assert captured_specs[0].d_head == 4
    assert captured_specs[0].has_skip is True
    assert captured_specs[0].d_skip_dtype_name == "float32"


def test_get_compiled_mixer_tail_rowwise_bwd_kernel_prefers_packaged_aot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_artifacts = _make_backward_runtime_artifacts()
    compile_artifacts = mixer_bwd_mod._make_backward_compile_artifacts(
        runtime_artifacts
    )
    captured_specs: list[mixer_aot_mod.BackwardAOTSpec] = []
    packaged = object()

    mixer_bwd_mod._BWD_CACHE.clear()
    mixer_aot_mod.clear_packaged_aot_cache()

    monkeypatch.setattr(
        cute_aot_common, "current_cuda_arch_tag", lambda device: "sm_80"
    )
    monkeypatch.setattr(
        mixer_aot_mod,
        "try_load_packaged_mixer_tail_rowwise_bwd_function",
        lambda spec: captured_specs.append(spec) or packaged,
    )
    monkeypatch.setattr(
        mixer_bwd_mod,
        "_compile_backward_kernel",
        lambda *args, **kwargs: pytest.fail("unexpected JIT compile"),
    )

    compiled = mixer_bwd_mod._get_compiled_backward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=torch.device("cuda"),
        eps=1.0e-5,
    )

    assert compiled is packaged
    assert len(captured_specs) == 1
    assert captured_specs[0].arch_tag == "sm_80"
    assert captured_specs[0].heads == 3
    assert captured_specs[0].d_head == 4
    assert captured_specs[0].has_skip is True
    assert captured_specs[0].d_skip_dtype_name == "float32"
    assert captured_specs[0].d_normed_dtype_name == "float16"
