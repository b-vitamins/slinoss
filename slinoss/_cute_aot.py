"""Shared TVM-FFI CuTe AOT helpers."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, cast

import cutlass.cute as cute
import torch

from slinoss._cute_runtime import ensure_cute_runtime_env

_SUPPORTED_CUTE_AOT_ARCH_TAGS = frozenset(
    {
        "sm_80",
        "sm_86",
        "sm_87",
        "sm_89",
        "sm_90",
        "sm_90a",
    }
)


def _dtype_tag(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "f32"
    raise TypeError(f"Unsupported dtype: {dtype}")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"Unsupported dtype: {dtype}")


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise TypeError(f"Unsupported dtype name: {name}")


def _sanitize_stem(text: str) -> str:
    return text.replace(".", "_").replace("-", "_")


def _normalize_arch_tag(raw_arch: str) -> str:
    raw_arch = raw_arch.strip()
    if not raw_arch:
        return ""
    if raw_arch.startswith("sm_"):
        return raw_arch
    if raw_arch.startswith("compute_"):
        return "sm_" + raw_arch.removeprefix("compute_")
    raw_arch = raw_arch.removesuffix("+PTX").strip()
    if "." in raw_arch:
        major, minor = raw_arch.split(".", 1)
        return f"sm_{int(major)}{int(minor)}"
    if raw_arch.isdigit():
        return f"sm_{raw_arch}"
    return raw_arch


def current_cuda_arch_tag(
    device: torch.device | int | None = None,
) -> str:
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"Expected a CUDA device. Got {device}.")
        device_index = (
            int(device.index)
            if device.index is not None
            else torch.cuda.current_device()
        )
    elif device is None:
        device_index = torch.cuda.current_device()
    else:
        device_index = int(device)
    major, minor = torch.cuda.get_device_capability(device_index)
    arch_tag = _normalize_arch_tag(f"{major}.{minor}")
    if not arch_tag:
        raise RuntimeError(
            f"Failed to resolve a CUDA arch tag for device index {device_index}."
        )
    return arch_tag


def _filter_supported_cute_aot_arch_tags(
    arch_tags: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            arch_tag
            for arch_tag in arch_tags
            if arch_tag == "any" or arch_tag in _SUPPORTED_CUTE_AOT_ARCH_TAGS
        )
    )


def _arch_tags_from_env() -> tuple[str, ...]:
    explicit_arch_tags = os.environ.get("SLINOSS_CUTE_AOT_ARCH_TAGS", "").strip()
    if not explicit_arch_tags:
        explicit_arch_tags = os.environ.get(
            "SLINOSS_CUTE_FORWARD_AOT_ARCH_TAGS", ""
        ).strip()
    if explicit_arch_tags:
        return _filter_supported_cute_aot_arch_tags(
            tuple(
                dict.fromkeys(
                    normalized
                    for normalized in (
                        _normalize_arch_tag(value)
                        for value in explicit_arch_tags.replace(",", ";").split(";")
                    )
                    if normalized
                )
            )
        )

    torch_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if torch_arch_list:
        return _filter_supported_cute_aot_arch_tags(
            tuple(
                dict.fromkeys(
                    normalized
                    for normalized in (
                        _normalize_arch_tag(value)
                        for value in torch_arch_list.replace(" ", ";").split(";")
                    )
                    if normalized
                )
            )
        )

    return ()


def _resolve_cute_aot_arch_tags(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if arch_tags is None:
        resolved_arch_tags = _arch_tags_from_env()
        return resolved_arch_tags if resolved_arch_tags else ("any",)

    resolved_arch_tags = _filter_supported_cute_aot_arch_tags(arch_tags)
    if resolved_arch_tags:
        return resolved_arch_tags
    raise ValueError(f"Unsupported CuTe AOT arch tags: {arch_tags}")


def _aot_compile_options(arch_tag: str) -> str:
    options = ["--enable-tvm-ffi"]
    if arch_tag and arch_tag != "any":
        options.append(f"--gpu-arch={arch_tag}")
    return " ".join(options)


def _link_shared_library(
    *,
    object_file_path: Path,
    shared_library_path: Path,
    runtime_libraries: tuple[Path, ...],
    runtime_rpath: str,
) -> None:
    shared_library_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-shared",
        "-o",
        str(shared_library_path),
        str(object_file_path),
        *map(str, runtime_libraries),
        # CuTe TVM-FFI modules currently carry one text relocation in generated
        # host glue. Make the linker policy explicit to avoid noisy build logs.
        "-Wl,-z,notext",
        f"-Wl,-rpath,{runtime_rpath}",
        "-Wl,--enable-new-dtags",
    ]
    subprocess.run(cmd, check=True)


def find_tvm_ffi_runtime_libraries() -> tuple[Path, ...]:
    ensure_cute_runtime_env()
    return tuple(
        Path(path) for path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
    )


def _copy_runtime_libraries(runtime_dir: Path) -> tuple[Path, ...]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    libraries = tuple(find_tvm_ffi_runtime_libraries())
    copied: list[Path] = []
    for library in libraries:
        target = runtime_dir / library.name
        if not target.exists():
            shutil.copy2(library, target)
        copied.append(target)
    return tuple(copied)


@dataclass(frozen=True)
class ExportedTVMFFIModule:
    kind: str
    module_id: str
    function_name: str
    object_file: Path | None
    shared_library: Path
    metadata_file: Path


def export_tvm_ffi_compiled_module(
    compiled,
    *,
    kind: str,
    module_id: str,
    function_name: str,
    package_root: str | os.PathLike[str] | Path,
    keep_object_file: bool = True,
) -> ExportedTVMFFIModule:
    ensure_cute_runtime_env()
    package_root = Path(package_root)
    artifact_dir = package_root / "artifacts"
    runtime_dir = package_root / "runtime"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runtime_paths = _copy_runtime_libraries(runtime_dir)
    object_file = artifact_dir / f"{module_id}.o"
    shared_library = artifact_dir / f"{module_id}.so"
    metadata_file = artifact_dir / f"{module_id}.json"

    compiled.export_to_c(
        str(object_file),
        function_name=function_name,
        enable_pic=True,
        export_only_tvm_ffi_symbols=True,
    )
    _link_shared_library(
        object_file_path=object_file,
        shared_library_path=shared_library,
        runtime_libraries=runtime_paths,
        runtime_rpath="$ORIGIN/../runtime",
    )

    object_file_result: Path | None = object_file
    if not keep_object_file:
        object_file.unlink(missing_ok=True)
        object_file_result = None

    exported = ExportedTVMFFIModule(
        kind=kind,
        module_id=module_id,
        function_name=function_name,
        object_file=object_file_result,
        shared_library=shared_library,
        metadata_file=metadata_file,
    )
    metadata = {
        "kind": kind,
        "module_id": module_id,
        "function_name": function_name,
        "object_file": (
            str(object_file.relative_to(package_root))
            if object_file_result is not None
            else None
        ),
        "shared_library": str(shared_library.relative_to(package_root)),
        "runtime_libraries": [
            str(path.relative_to(package_root)) for path in runtime_paths
        ],
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return exported


def load_tvm_ffi_module(
    module_path: str | os.PathLike[str] | Path,
):
    ensure_cute_runtime_env()
    return cute.runtime.load_module(str(Path(module_path)), enable_tvm_ffi=True)


def load_tvm_ffi_function(
    module_path: str | os.PathLike[str] | Path,
    *,
    function_name: str,
):
    module = load_tvm_ffi_module(module_path)
    return getattr(module, function_name)


def _manifest_record(
    *,
    kind: str,
    spec: Any,
    exported: ExportedTVMFFIModule,
    package_root: Path,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": exported.module_id,
        "function_name": exported.function_name,
        "object_file": (
            str(exported.object_file.relative_to(package_root))
            if exported.object_file is not None
            else None
        ),
        "shared_library": str(exported.shared_library.relative_to(package_root)),
        "metadata_file": str(exported.metadata_file.relative_to(package_root)),
        "spec": asdict(spec),
    }


def _read_manifest(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "manifest.json"
    if not manifest_path.exists():
        return {"artifacts": []}
    return json.loads(manifest_path.read_text())


def _write_manifest(package_root: Path, manifest: dict[str, Any]) -> None:
    manifest_path = package_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def register_aot_artifact(
    *,
    kind: str,
    spec: Any,
    exported: ExportedTVMFFIModule,
    package_root: str | os.PathLike[str] | Path,
) -> None:
    package_root = Path(package_root)
    manifest = _read_manifest(package_root)
    record = _manifest_record(
        kind=kind,
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    artifacts = [
        existing
        for existing in manifest.get("artifacts", [])
        if existing["id"] != record["id"]
    ]
    artifacts.append(record)
    manifest["artifacts"] = sorted(artifacts, key=lambda item: item["id"])
    _write_manifest(package_root, manifest)


def _arch_matches(candidate_arch_tag: str, requested_arch_tag: str | None) -> bool:
    if requested_arch_tag is None:
        return True
    return candidate_arch_tag == "any" or candidate_arch_tag == requested_arch_tag


def _list_packaged_aot_records(
    *,
    kind: str,
    package_root: str | os.PathLike[str] | Path,
    arch_tag: str | None = None,
) -> tuple[dict[str, Any], ...]:
    manifest = _read_manifest(Path(package_root))
    return tuple(
        candidate
        for candidate in manifest.get("artifacts", [])
        if candidate.get("kind") == kind
        and _arch_matches(
            str(cast(dict[str, Any], candidate["spec"]).get("arch_tag", "any")),
            arch_tag,
        )
    )


def _try_load_packaged_compiled_function(
    *,
    kind: str,
    module_id: str,
    arch_tag: str,
    cache: dict[str, object],
    package_root: str | os.PathLike[str] | Path,
):
    if os.environ.get("SLINOSS_DISABLE_CUTE_AOT") == "1":
        return None

    package_root = Path(package_root)
    cached = cache.get(module_id)
    if cached is not None:
        return cached

    record = next(
        (
            candidate
            for candidate in _list_packaged_aot_records(
                kind=kind,
                package_root=package_root,
                arch_tag=arch_tag,
            )
            if candidate.get("id") == module_id
        ),
        None,
    )
    if record is None:
        return None

    function_name = str(record["function_name"])
    shared_library = package_root / str(record["shared_library"])
    object_file_rel = record.get("object_file")
    candidate_paths = [shared_library]
    if object_file_rel:
        candidate_paths.append(package_root / str(object_file_rel))
    for module_path in candidate_paths:
        if not module_path.exists():
            continue
        try:
            loaded = load_tvm_ffi_function(module_path, function_name=function_name)
        except Exception:
            continue
        cache[module_id] = loaded
        return loaded
    return None


__all__ = [
    "ExportedTVMFFIModule",
    "_aot_compile_options",
    "_arch_tags_from_env",
    "_dtype_from_name",
    "_dtype_name",
    "_dtype_tag",
    "_filter_supported_cute_aot_arch_tags",
    "_list_packaged_aot_records",
    "_normalize_arch_tag",
    "_resolve_cute_aot_arch_tags",
    "_sanitize_stem",
    "_try_load_packaged_compiled_function",
    "current_cuda_arch_tag",
    "export_tvm_ffi_compiled_module",
    "find_tvm_ffi_runtime_libraries",
    "load_tvm_ffi_function",
    "load_tvm_ffi_module",
    "register_aot_artifact",
]
