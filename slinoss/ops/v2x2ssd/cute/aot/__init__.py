"""Ahead-of-time helpers for the CuTe ``v2x2ssd`` forward stack.

This module keeps the AOT surface explicit and separate from the eager/JIT
runtime path:

- compile/export helpers for the forward stages and combined forward
- a small TVM-FFI export/load wrapper around CuTe runtime APIs
- packaged-artifact discovery for release-built forward modules

The packaged forward AOT path is intentionally specialized to the forward
kernel/model configuration (for example ``P``, ``D``, chunk size, dtypes, and
launch policy) while remaining dynamic in batch/time via runtime tensor views.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, cast

import torch
import cuda.bindings.driver as cuda
import cutlass.cute as cute

from slinoss._cute_runtime import ensure_cute_runtime_env
from slinoss.ops.v2x2ssd.cute.tuning.types import (
    ChunkIncrementConfig,
    ChunkScanConfig,
    ForwardConfigBundle,
    StatePassingConfig,
)

from ..kernels.fwd.chunk_increment import ChunkIncrementFwdAmpere
from ..kernels.fwd.chunk_scan import ChunkScanFwdAmpere
from ..kernels.fwd.state_passing import StatePassingFwdAmpere


_PACKAGED_AOT_ROOT = Path(__file__).resolve().parent
_PACKAGED_AOT_MANIFEST = _PACKAGED_AOT_ROOT / "manifest.json"
_PACKAGED_AOT_RUNTIME_DIR = _PACKAGED_AOT_ROOT / "runtime"
_PACKAGED_AOT_ARTIFACT_DIR = _PACKAGED_AOT_ROOT / "artifacts"
_PACKAGED_FORWARD_CACHE: dict[str, object] = {}
_PACKAGED_STAGE_CACHE: dict[str, object] = {}


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


def _bool_tag(flag: bool) -> str:
    return "1" if bool(flag) else "0"


def _sanitize_stem(text: str) -> str:
    return text.replace(".", "_").replace("-", "_")


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
        f"-Wl,-rpath,{runtime_rpath}",
        "-Wl,--enable-new-dtags",
    ]
    subprocess.run(cmd, check=True)


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


def find_tvm_ffi_runtime_libraries() -> tuple[Path, ...]:
    """Return the CuTe/TVM runtime libraries required by TVM-FFI modules."""
    ensure_cute_runtime_env()
    return tuple(
        Path(path) for path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
    )


@dataclass(frozen=True)
class ExportedTVMFFIModule:
    """Filesystem payload for one exported TVM-FFI CuTe module."""

    kind: str
    module_id: str
    function_name: str
    object_file: Path | None
    shared_library: Path
    metadata_file: Path


@dataclass(frozen=True)
class ChunkIncrementAOTSpec:
    arch_tag: str
    P: int
    D: int
    chunk_size: int
    tc_dtype_name: str
    config: ChunkIncrementConfig

    @property
    def tc_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.tc_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "chunk_increment"
            f"__arch{self.arch_tag}"
            f"__p{self.P}_d{self.D}_l{self.chunk_size}"
            f"__tc{_dtype_tag(self.tc_dtype)}"
            f"__cta{'x'.join(str(v) for v in self.config.cta_tiler)}"
            f"__stg{self.config.num_stages}"
        )


@dataclass(frozen=True)
class StatePassingAOTSpec:
    arch_tag: str
    P: int
    D: int
    config: StatePassingConfig
    has_init: bool

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "state_passing"
            f"__arch{self.arch_tag}"
            f"__p{self.P}_d{self.D}"
            f"__thr{self.config.num_threads}"
            f"__vec{self.config.vecs_per_thread}"
            f"__init{_bool_tag(self.has_init)}"
        )


@dataclass(frozen=True)
class ChunkScanAOTSpec:
    arch_tag: str
    P: int
    D: int
    chunk_size: int
    tc_dtype_name: str
    output_dtype_name: str
    config: ChunkScanConfig

    @property
    def tc_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.tc_dtype_name)

    @property
    def output_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.output_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "chunk_scan"
            f"__arch{self.arch_tag}"
            f"__p{self.P}_d{self.D}_l{self.chunk_size}"
            f"__tc{_dtype_tag(self.tc_dtype)}"
            f"__out{_dtype_tag(self.output_dtype)}"
            f"__cfg{self.config.m_block_size}x{self.config.n_block_size}x{self.config.num_threads}"
        )


@dataclass(frozen=True)
class ForwardAOTSpec:
    arch_tag: str
    P: int
    D: int
    chunk_size: int
    tc_dtype_name: str
    output_dtype_name: str
    config_bundle: ForwardConfigBundle
    has_init: bool

    @property
    def tc_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.tc_dtype_name)

    @property
    def output_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.output_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "v2x2ssd_fwd"
            f"__arch{self.arch_tag}"
            f"__p{self.P}_d{self.D}_l{self.chunk_size}"
            f"__tc{_dtype_tag(self.tc_dtype)}"
            f"__out{_dtype_tag(self.output_dtype)}"
            f"__inc{'x'.join(str(v) for v in self.config_bundle.chunk_increment.cta_tiler)}"
            f"__stg{self.config_bundle.chunk_increment.num_stages}"
            f"__state{self.config_bundle.state_passing.num_threads}x{self.config_bundle.state_passing.vecs_per_thread}"
            f"__scan{self.config_bundle.chunk_scan.m_block_size}x{self.config_bundle.chunk_scan.n_block_size}x{self.config_bundle.chunk_scan.num_threads}"
            f"__init{_bool_tag(self.has_init)}"
        )


_AOT_SEARCH_P = 64
_AOT_SEARCH_D = 256
_AOT_SEARCH_CHUNK_SIZES = (32, 64, 128, 256)
_AOT_SEARCH_TC_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_FORWARD_AOT_ARCH_TAGS = frozenset(
    {
        "sm_80",
        "sm_86",
        "sm_87",
        "sm_89",
        "sm_90",
        "sm_90a",
    }
)


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


def _filter_supported_forward_aot_arch_tags(
    arch_tags: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            arch_tag
            for arch_tag in arch_tags
            if arch_tag == "any" or arch_tag in _SUPPORTED_FORWARD_AOT_ARCH_TAGS
        )
    )


def _arch_tags_from_env() -> tuple[str, ...]:
    explicit_arch_tags = os.environ.get(
        "SLINOSS_CUTE_FORWARD_AOT_ARCH_TAGS", ""
    ).strip()
    if explicit_arch_tags:
        return _filter_supported_forward_aot_arch_tags(
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
        return _filter_supported_forward_aot_arch_tags(
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


def _resolve_forward_aot_arch_tags(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if arch_tags is None:
        resolved_arch_tags = _arch_tags_from_env()
        return resolved_arch_tags if resolved_arch_tags else ("any",)

    resolved_arch_tags = _filter_supported_forward_aot_arch_tags(arch_tags)
    if resolved_arch_tags:
        return resolved_arch_tags
    raise ValueError(f"Unsupported forward AOT arch tags: {arch_tags}")


def _candidate_output_dtypes(tc_dtype: torch.dtype) -> tuple[torch.dtype, ...]:
    output_dtypes = [torch.float32]
    if tc_dtype != torch.float32:
        output_dtypes.append(tc_dtype)
    return tuple(output_dtypes)


def _search_space_chunk_increment_specs(
    *, arch_tag: str = "any"
) -> tuple[ChunkIncrementAOTSpec, ...]:
    from ..tuning.fwd import chunk_increment_candidate_configs

    specs: list[ChunkIncrementAOTSpec] = []
    for chunk_size in _AOT_SEARCH_CHUNK_SIZES:
        for tc_dtype in _AOT_SEARCH_TC_DTYPES:
            for config in chunk_increment_candidate_configs(
                P=_AOT_SEARCH_P,
                D=_AOT_SEARCH_D,
                chunk_size=chunk_size,
            ):
                specs.append(
                    ChunkIncrementAOTSpec(
                        arch_tag=arch_tag,
                        P=_AOT_SEARCH_P,
                        D=_AOT_SEARCH_D,
                        chunk_size=chunk_size,
                        tc_dtype_name=_dtype_name(tc_dtype),
                        config=config,
                    )
                )
    return tuple(specs)


def _search_space_state_passing_specs(
    *, arch_tag: str = "any"
) -> tuple[StatePassingAOTSpec, ...]:
    from ..tuning.fwd import state_passing_candidate_configs

    specs: list[StatePassingAOTSpec] = []
    for has_init in (False, True):
        for config in state_passing_candidate_configs(
            P=_AOT_SEARCH_P,
            D=_AOT_SEARCH_D,
            has_init=has_init,
        ):
            specs.append(
                StatePassingAOTSpec(
                    arch_tag=arch_tag,
                    P=_AOT_SEARCH_P,
                    D=_AOT_SEARCH_D,
                    config=config,
                    has_init=has_init,
                )
            )
    return tuple(specs)


def _search_space_chunk_scan_specs(
    *, arch_tag: str = "any"
) -> tuple[ChunkScanAOTSpec, ...]:
    from ..tuning.fwd import chunk_scan_candidate_configs

    specs: list[ChunkScanAOTSpec] = []
    for chunk_size in _AOT_SEARCH_CHUNK_SIZES:
        for tc_dtype in _AOT_SEARCH_TC_DTYPES:
            for output_dtype in _candidate_output_dtypes(tc_dtype):
                for config in chunk_scan_candidate_configs(
                    P=_AOT_SEARCH_P,
                    D=_AOT_SEARCH_D,
                    chunk_size=chunk_size,
                    tc_dtype=tc_dtype,
                    output_dtype=output_dtype,
                    device_index=None,
                ):
                    specs.append(
                        ChunkScanAOTSpec(
                            arch_tag=arch_tag,
                            P=_AOT_SEARCH_P,
                            D=_AOT_SEARCH_D,
                            chunk_size=chunk_size,
                            tc_dtype_name=_dtype_name(tc_dtype),
                            output_dtype_name=_dtype_name(output_dtype),
                            config=config,
                        )
                    )
    return tuple(specs)


def _search_space_forward_specs(*, arch_tag: str = "any") -> tuple[ForwardAOTSpec, ...]:
    from ..tuning.fwd import forward_bundle_candidates

    specs: list[ForwardAOTSpec] = []
    for chunk_size in _AOT_SEARCH_CHUNK_SIZES:
        for tc_dtype in _AOT_SEARCH_TC_DTYPES:
            for output_dtype in _candidate_output_dtypes(tc_dtype):
                for has_init in (False, True):
                    for config_bundle in forward_bundle_candidates(
                        P=_AOT_SEARCH_P,
                        D=_AOT_SEARCH_D,
                        chunk_size=chunk_size,
                        tc_dtype=tc_dtype,
                        output_dtype=output_dtype,
                        device_index=None,
                        has_init=has_init,
                    ):
                        specs.append(
                            ForwardAOTSpec(
                                arch_tag=arch_tag,
                                P=_AOT_SEARCH_P,
                                D=_AOT_SEARCH_D,
                                chunk_size=chunk_size,
                                tc_dtype_name=_dtype_name(tc_dtype),
                                output_dtype_name=_dtype_name(output_dtype),
                                config_bundle=config_bundle,
                                has_init=has_init,
                            )
                        )
    return tuple(specs)


@lru_cache(maxsize=8)
def default_forward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[ForwardAOTSpec, ...]:
    resolved_arch_tags = _resolve_forward_aot_arch_tags(arch_tags)
    return tuple(
        spec
        for arch_tag in resolved_arch_tags
        for spec in _search_space_forward_specs(arch_tag=arch_tag)
    )


def _chunk_increment_spec_from_record(record: dict[str, Any]) -> ChunkIncrementAOTSpec:
    return ChunkIncrementAOTSpec(
        arch_tag=str(record["arch_tag"]),
        P=int(record["P"]),
        D=int(record["D"]),
        chunk_size=int(record["chunk_size"]),
        tc_dtype_name=str(record["tc_dtype_name"]),
        config=ChunkIncrementConfig.from_record(cast(dict[str, Any], record["config"])),
    )


def _state_passing_spec_from_record(record: dict[str, Any]) -> StatePassingAOTSpec:
    return StatePassingAOTSpec(
        arch_tag=str(record["arch_tag"]),
        P=int(record["P"]),
        D=int(record["D"]),
        config=StatePassingConfig.from_record(cast(dict[str, Any], record["config"])),
        has_init=bool(record["has_init"]),
    )


def _chunk_scan_spec_from_record(record: dict[str, Any]) -> ChunkScanAOTSpec:
    return ChunkScanAOTSpec(
        arch_tag=str(record["arch_tag"]),
        P=int(record["P"]),
        D=int(record["D"]),
        chunk_size=int(record["chunk_size"]),
        tc_dtype_name=str(record["tc_dtype_name"]),
        output_dtype_name=str(record["output_dtype_name"]),
        config=ChunkScanConfig.from_record(cast(dict[str, Any], record["config"])),
    )


def _forward_spec_from_record(record: dict[str, Any]) -> ForwardAOTSpec:
    return ForwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        P=int(record["P"]),
        D=int(record["D"]),
        chunk_size=int(record["chunk_size"]),
        tc_dtype_name=str(record["tc_dtype_name"]),
        output_dtype_name=str(record["output_dtype_name"]),
        config_bundle=ForwardConfigBundle.from_record(
            cast(dict[str, Any], record["config_bundle"])
        ),
        has_init=bool(record["has_init"]),
    )


def _make_masked_fake_tensor_spec_arg(
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    align: int,
    dynamic_shape_mask: tuple[bool, ...],
    dynamic_stride_mask: tuple[bool, ...],
):
    from ..kernels.fwd.common import _torch_to_cutlass_dtype

    fake_shape = tuple(
        cute.sym_int32() if is_dynamic else int(dim)
        for dim, is_dynamic in zip(shape, dynamic_shape_mask, strict=True)
    )
    fake_stride = tuple(
        0 if int(step) == 0 else (cute.sym_int32() if is_dynamic else int(step))
        for step, is_dynamic in zip(stride, dynamic_stride_mask, strict=True)
    )
    return cute.runtime.make_fake_tensor(
        _torch_to_cutlass_dtype(dtype),
        fake_shape,
        stride=fake_stride,
        assumed_align=int(align),
    )


def _masked_compile_args_from_specs(
    *tensor_specs: tuple[
        torch.dtype,
        tuple[tuple[int, ...], tuple[int, ...]],
        int,
        tuple[bool, ...],
        tuple[bool, ...],
    ],
) -> tuple[tuple[int, ...], tuple[object, ...]]:
    from ..kernels.fwd.common import _compile_env_stream_placeholder

    ensure_cute_runtime_env()
    alignments = tuple(int(align) for _, _, align, _, _ in tensor_specs)
    compile_args = tuple(
        _make_masked_fake_tensor_spec_arg(
            dtype=dtype,
            shape=spec[0],
            stride=spec[1],
            align=align,
            dynamic_shape_mask=dynamic_shape_mask,
            dynamic_stride_mask=dynamic_stride_mask,
        )
        for dtype, spec, align, dynamic_shape_mask, dynamic_stride_mask in tensor_specs
    ) + (_compile_env_stream_placeholder(),)
    return alignments, compile_args


def _aot_compile_options(arch_tag: str) -> str:
    options = ["--enable-tvm-ffi"]
    if arch_tag and arch_tag != "any":
        options.append(f"--gpu-arch={arch_tag}")
    return " ".join(options)


def _representative_chunk_increment_problem_shape(
    spec: ChunkIncrementAOTSpec,
) -> tuple[int, ...]:
    return (2, 2, 2 * spec.chunk_size, spec.P, spec.D, 2, spec.chunk_size)


def _representative_state_passing_problem_shape(
    spec: StatePassingAOTSpec,
) -> tuple[int, ...]:
    return (2, 2, 2, spec.P, spec.D)


def _representative_chunk_scan_problem_shape(
    spec: ChunkScanAOTSpec,
) -> tuple[int, ...]:
    return (2, 2, 2 * spec.chunk_size, spec.P, spec.D, 2, spec.chunk_size)


def _representative_forward_problem_shape(spec: ForwardAOTSpec) -> tuple[int, ...]:
    return (2, 2, 2 * spec.chunk_size, spec.P, spec.D, 2, spec.chunk_size)


def _infer_chunk_increment_aot_spec(
    U: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    arch_tag: str = "any",
) -> ChunkIncrementAOTSpec:
    from ..kernels.fwd import _make_chunk_increment_compile_artifacts, _tc_input_dtype

    compile_artifacts = _make_chunk_increment_compile_artifacts(
        U,
        B,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        has_prev=False,
    )
    _batch_size, _heads, _padded_time, P, D, _n_chunks, resolved_chunk_size = (
        compile_artifacts.problem_shape
    )
    return ChunkIncrementAOTSpec(
        arch_tag=arch_tag,
        P=int(P),
        D=int(D),
        chunk_size=int(resolved_chunk_size),
        tc_dtype_name=_dtype_name(_tc_input_dtype(U.dtype, compute_dtype)),
        config=compile_artifacts.config,
    )


def _infer_state_passing_aot_spec(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int,
    vecs_per_thread: int,
    arch_tag: str = "any",
) -> StatePassingAOTSpec:
    from ..kernels.fwd import _make_state_passing_compile_artifacts

    compile_artifacts = _make_state_passing_compile_artifacts(
        increment,
        chunk_multiplier,
        has_init=initial_states is not None,
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
    )
    _batch_size, _heads, _chunks, P, D = compile_artifacts.problem_shape
    return StatePassingAOTSpec(
        arch_tag=arch_tag,
        P=int(P),
        D=int(D),
        config=compile_artifacts.config,
        has_init=initial_states is not None,
    )


def _infer_chunk_scan_aot_spec(
    U: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    m_block_size: int | None,
    n_block_size: int,
    num_threads: int,
    arch_tag: str = "any",
) -> ChunkScanAOTSpec:
    from ..kernels.fwd import _make_chunk_scan_compile_artifacts, _tc_input_dtype

    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
    )
    compile_artifacts = _make_chunk_scan_compile_artifacts(
        U,
        B,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        device_index=device_index,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        has_prev=False,
    )
    _batch_size, _heads, _padded_time, P, D, _n_chunks, resolved_chunk_size = (
        compile_artifacts.problem_shape
    )
    return ChunkScanAOTSpec(
        arch_tag=arch_tag,
        P=int(P),
        D=int(D),
        chunk_size=int(resolved_chunk_size),
        tc_dtype_name=_dtype_name(_tc_input_dtype(U.dtype, compute_dtype)),
        output_dtype_name=_dtype_name(output_dtype),
        config=compile_artifacts.config,
    )


def infer_v2x2ssd_fwd_aot_spec(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    scan_num_threads: int = 128,
    state_num_threads: int = 128,
    state_vecs_per_thread: int = 8,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    chunk_increment_cta_tiler: tuple[int, int, int] | None = None,
    chunk_increment_num_stages: int | None = None,
    arch_tag: str = "any",
) -> ForwardAOTSpec:
    from ..kernels.fwd import _make_forward_compile_artifacts, _tc_input_dtype
    from ..tuning.types import (
        ChunkIncrementConfig,
        ChunkScanConfig,
        ForwardConfigBundle,
        StatePassingConfig,
    )

    compile_artifacts = _make_forward_compile_artifacts(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        scan_num_threads=scan_num_threads,
        state_num_threads=state_num_threads,
        state_vecs_per_thread=state_vecs_per_thread,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        config_bundle=(
            None
            if chunk_increment_cta_tiler is None and chunk_increment_num_stages is None
            else ForwardConfigBundle(
                chunk_increment=ChunkIncrementConfig(
                    cta_tiler=(
                        chunk_increment_cta_tiler
                        if chunk_increment_cta_tiler is not None
                        else (64, 96, 32)
                    ),
                    num_stages=3
                    if chunk_increment_num_stages is None
                    else int(chunk_increment_num_stages),
                ),
                state_passing=StatePassingConfig(
                    num_threads=int(state_num_threads),
                    vecs_per_thread=int(state_vecs_per_thread),
                ),
                chunk_scan=ChunkScanConfig(
                    m_block_size=64 if m_block_size is None else int(m_block_size),
                    n_block_size=int(n_block_size),
                    num_threads=int(scan_num_threads),
                ),
            )
        ),
    )
    _batch_size, _heads, _padded_time, P, D, _n_chunks, resolved_chunk_size = (
        compile_artifacts.problem_shape
    )
    return ForwardAOTSpec(
        arch_tag=arch_tag,
        P=int(P),
        D=int(D),
        chunk_size=int(resolved_chunk_size),
        tc_dtype_name=_dtype_name(_tc_input_dtype(U.dtype, compute_dtype)),
        output_dtype_name=_dtype_name(output_dtype),
        config_bundle=compile_artifacts.config_bundle,
        has_init=initial_states is not None,
    )


def _compile_chunk_increment_aot(spec: ChunkIncrementAOTSpec):
    from ..kernels.fwd import (
        _chunk_increment_tensor_specs,
        _compile_min_align_for_dtype,
    )

    representative_shape = _representative_chunk_increment_problem_shape(spec)
    (
        u_spec,
        b_spec,
        m_spec,
        k_spec,
        u_prev_spec,
        b_prev_spec,
        increment_spec,
        chunk_multiplier_spec,
    ) = _chunk_increment_tensor_specs(representative_shape)
    tc_align = _compile_min_align_for_dtype(spec.tc_dtype)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    _alignments, compile_args = _masked_compile_args_from_specs(
        (spec.tc_dtype, u_spec, tc_align, (False, True, True), (False, False, True)),
        (spec.tc_dtype, b_spec, tc_align, (False, True, True), (False, False, True)),
        (torch.float32, m_spec, fp32_align, (False, True, True), (False, False, True)),
        (torch.float32, k_spec, fp32_align, (False, True, True), (False, False, True)),
        (spec.tc_dtype, u_prev_spec, tc_align, (False, True), (False, False)),
        (spec.tc_dtype, b_prev_spec, tc_align, (False, True), (False, False)),
        (
            torch.float32,
            increment_spec,
            fp32_align,
            (False, False, True),
            (False, False, False),
        ),
        (
            torch.float32,
            chunk_multiplier_spec,
            fp32_align,
            (False, True),
            (False, False),
        ),
    )

    @cute.jit
    def _chunk_increment_aot_host_wrapper(
        U_view: cute.Tensor,
        B_view: cute.Tensor,
        M_view: cute.Tensor,
        K_prev_view: cute.Tensor,
        U_prev_view: cute.Tensor,
        B_prev_view: cute.Tensor,
        increment_view: cute.Tensor,
        chunk_multiplier_view: cute.Tensor,
        stream: cuda.CUstream,
    ):
        kernel = ChunkIncrementFwdAmpere(
            cast(Any, U_view.element_type),
            chunk_size=spec.chunk_size,
            cta_tiler=spec.config.cta_tiler,
            num_stages=spec.config.num_stages,
        )
        K_curr_view = cute.make_tensor(
            cast(Any, K_prev_view.iterator) + 2, K_prev_view.layout
        )
        kernel._launch_kernel(
            U_view,
            B_view,
            M_view,
            K_prev_view,
            K_curr_view,
            U_prev_view,
            B_prev_view,
            increment_view,
            chunk_multiplier_view,
            stream=stream,
        )

    return cute.compile(
        _chunk_increment_aot_host_wrapper,
        *compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_state_passing_aot(spec: StatePassingAOTSpec):
    from ..kernels.fwd import (
        _choose_copy_bits_for_linear_tiles_from_properties,
        _compile_min_align_for_dtype,
        _make_state_passing_cfg,
        _state_passing_tensor_specs,
    )

    representative_shape = _representative_state_passing_problem_shape(spec)
    (
        increment_spec,
        chunk_multiplier_spec,
        chunk_starts_spec,
        final_state_spec,
    ) = _state_passing_tensor_specs(representative_shape)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    _alignments, compile_args = _masked_compile_args_from_specs(
        (
            torch.float32,
            increment_spec,
            fp32_align,
            (True, True, True, False, False),
            (True, True, False, False, False),
        ),
        (
            torch.float32,
            chunk_multiplier_spec,
            fp32_align,
            (True, True, True, False),
            (True, True, False, False),
        ),
        (
            torch.float32,
            chunk_starts_spec,
            fp32_align,
            (True, True, True, False, False),
            (True, True, False, False, False),
        ),
        (
            torch.float32,
            final_state_spec,
            fp32_align,
            (True, True, False, False),
            (True, False, False, False),
        ),
        (
            torch.float32,
            final_state_spec,
            fp32_align,
            (True, True, False, False),
            (True, False, False, False),
        ),
    )
    state_elem_count = spec.P * spec.D
    elems_per_thread = 2 * int(spec.config.vecs_per_thread)
    copy_bits_in = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=fp32_align,
        tile_stride_elems=state_elem_count,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=fp32_align,
        tile_stride_elems=state_elem_count,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_state_in = 32
    copy_bits_state_out = 32
    launch_cfg = _make_state_passing_cfg(
        num_threads=spec.config.num_threads,
        vecs_per_thread=spec.config.vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        has_init=spec.has_init,
    )
    (
        num_threads,
        vecs_per_thread,
        _copy_bits_in,
        _copy_bits_out,
        _copy_bits_state_in,
        _copy_bits_state_out,
        has_init,
    ) = launch_cfg
    kernel = StatePassingFwdAmpere(
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        has_init=has_init,
    )

    @cute.jit
    def _state_passing_aot_host_wrapper(
        increment_view: cute.Tensor,
        chunk_multiplier_view: cute.Tensor,
        chunk_starts_view: cute.Tensor,
        final_state_view: cute.Tensor,
        initial_state_view: cute.Tensor,
        stream: cuda.CUstream,
    ):
        kernel.call_on_stream(
            increment_view,
            chunk_multiplier_view,
            chunk_starts_view,
            final_state_view,
            initial_state_view,
            stream,
        )

    return cute.compile(
        _state_passing_aot_host_wrapper,
        *compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_chunk_scan_aot(spec: ChunkScanAOTSpec):
    from ..kernels.fwd import _chunk_scan_tensor_specs, _compile_min_align_for_dtype

    representative_shape = _representative_chunk_scan_problem_shape(spec)
    (
        u_spec,
        b_spec,
        m_spec,
        k_spec,
        z0_spec,
        u_prev_spec,
        b_prev_spec,
        output_spec,
    ) = _chunk_scan_tensor_specs(representative_shape)
    tc_align = _compile_min_align_for_dtype(spec.tc_dtype)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    output_align = _compile_min_align_for_dtype(spec.output_dtype)
    _alignments, compile_args = _masked_compile_args_from_specs(
        (
            spec.tc_dtype,
            u_spec,
            tc_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            spec.tc_dtype,
            b_spec,
            tc_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            spec.tc_dtype,
            b_spec,
            tc_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            torch.float32,
            m_spec,
            fp32_align,
            (True, False, False),
            (False, False, False),
        ),
        (
            torch.float32,
            k_spec,
            fp32_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            torch.float32,
            z0_spec,
            fp32_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (spec.tc_dtype, u_prev_spec, tc_align, (True, False), (False, False)),
        (spec.tc_dtype, b_prev_spec, tc_align, (True, False), (False, False)),
        (
            spec.output_dtype,
            output_spec,
            output_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
    )
    m_block_size = spec.config.m_block_size
    n_block_size = spec.config.n_block_size
    num_threads = spec.config.num_threads

    @cute.jit
    def _chunk_scan_aot_host_wrapper(
        U_view: cute.Tensor,
        B_view: cute.Tensor,
        C_view: cute.Tensor,
        M_view: cute.Tensor,
        K_view: cute.Tensor,
        Z0_view: cute.Tensor,
        U_prev_view: cute.Tensor,
        B_prev_view: cute.Tensor,
        output_view: cute.Tensor,
        stream: cuda.CUstream,
    ):
        kernel = ChunkScanFwdAmpere(
            D=spec.D,
            P=spec.P,
            L=spec.chunk_size,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
        )
        kernel._launch_main_kernel(
            U_view,
            B_view,
            C_view,
            M_view,
            K_view,
            Z0_view,
            U_prev_view,
            B_prev_view,
            output_view,
            stream=stream,
        )

    return cute.compile(
        _chunk_scan_aot_host_wrapper,
        *compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_forward_aot(spec: ForwardAOTSpec):
    from ..kernels.fwd import (
        _choose_copy_bits_for_linear_tiles_from_properties,
        _chunk_increment_tensor_specs,
        _chunk_scan_tensor_specs,
        _compile_min_align_for_dtype,
        _make_state_passing_cfg,
        _state_passing_tensor_specs,
    )

    representative_shape = _representative_forward_problem_shape(spec)
    chunk_increment_specs = _chunk_increment_tensor_specs(representative_shape)
    state_passing_specs = _state_passing_tensor_specs((2, 2, 2, spec.P, spec.D))
    chunk_scan_specs = _chunk_scan_tensor_specs(representative_shape)
    tc_align = _compile_min_align_for_dtype(spec.tc_dtype)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    output_align = _compile_min_align_for_dtype(spec.output_dtype)
    (
        u_increment_spec,
        b_increment_spec,
        m_increment_spec,
        k_increment_spec,
        u_prev_increment_spec,
        b_prev_increment_spec,
        increment_spec,
        chunk_multiplier_spec,
    ) = chunk_increment_specs
    (
        increment_state_spec,
        chunk_multiplier_state_spec,
        chunk_starts_spec,
        final_state_spec,
    ) = state_passing_specs
    (
        u_scan_spec,
        b_scan_spec,
        m_scan_spec,
        k_scan_spec,
        z0_scan_spec,
        u_prev_scan_spec,
        b_prev_scan_spec,
        output_spec,
    ) = chunk_scan_specs
    _alignments, compile_args = _masked_compile_args_from_specs(
        (
            spec.tc_dtype,
            u_increment_spec,
            tc_align,
            (False, True, True),
            (False, False, True),
        ),
        (
            spec.tc_dtype,
            b_increment_spec,
            tc_align,
            (False, True, True),
            (False, False, True),
        ),
        (
            torch.float32,
            m_increment_spec,
            fp32_align,
            (False, True, True),
            (False, False, True),
        ),
        (
            torch.float32,
            k_increment_spec,
            fp32_align,
            (False, True, True),
            (False, False, True),
        ),
        (spec.tc_dtype, u_prev_increment_spec, tc_align, (False, True), (False, False)),
        (spec.tc_dtype, b_prev_increment_spec, tc_align, (False, True), (False, False)),
        (
            torch.float32,
            increment_spec,
            fp32_align,
            (False, False, True),
            (False, False, False),
        ),
        (
            torch.float32,
            chunk_multiplier_spec,
            fp32_align,
            (False, True),
            (False, False),
        ),
        (
            torch.float32,
            increment_state_spec,
            fp32_align,
            (True, True, True, False, False),
            (True, True, False, False, False),
        ),
        (
            torch.float32,
            chunk_multiplier_state_spec,
            fp32_align,
            (True, True, True, False),
            (True, True, False, False),
        ),
        (
            torch.float32,
            chunk_starts_spec,
            fp32_align,
            (True, True, True, False, False),
            (True, True, False, False, False),
        ),
        (
            torch.float32,
            final_state_spec,
            fp32_align,
            (True, True, False, False),
            (True, False, False, False),
        ),
        (
            torch.float32,
            final_state_spec,
            fp32_align,
            (True, True, False, False),
            (True, False, False, False),
        ),
        (
            spec.tc_dtype,
            u_scan_spec,
            tc_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            spec.tc_dtype,
            b_scan_spec,
            tc_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            spec.tc_dtype,
            b_scan_spec,
            tc_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            torch.float32,
            m_scan_spec,
            fp32_align,
            (True, False, False),
            (False, False, False),
        ),
        (
            torch.float32,
            k_scan_spec,
            fp32_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (
            torch.float32,
            z0_scan_spec,
            fp32_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
        (spec.tc_dtype, u_prev_scan_spec, tc_align, (True, False), (False, False)),
        (spec.tc_dtype, b_prev_scan_spec, tc_align, (True, False), (False, False)),
        (
            spec.output_dtype,
            output_spec,
            output_align,
            (True, False, False, False),
            (False, False, False, False),
        ),
    )

    state_elem_count = spec.P * spec.D
    state_elems_per_thread = 2 * int(spec.config_bundle.state_passing.vecs_per_thread)
    state_copy_bits_in = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=fp32_align,
        tile_stride_elems=state_elem_count,
        elems_per_thread=state_elems_per_thread,
    )
    state_copy_bits_out = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=fp32_align,
        tile_stride_elems=state_elem_count,
        elems_per_thread=state_elems_per_thread,
    )
    state_copy_bits_state_in = 32
    state_copy_bits_state_out = 32
    (
        _state_num_threads,
        _state_vecs_per_thread,
        _copy_bits_in,
        _copy_bits_out,
        _copy_bits_state_in,
        _copy_bits_state_out,
        _has_init,
    ) = _make_state_passing_cfg(
        num_threads=spec.config_bundle.state_passing.num_threads,
        vecs_per_thread=spec.config_bundle.state_passing.vecs_per_thread,
        copy_bits_in=state_copy_bits_in,
        copy_bits_out=state_copy_bits_out,
        copy_bits_state_in=state_copy_bits_state_in,
        copy_bits_state_out=state_copy_bits_state_out,
        has_init=spec.has_init,
    )
    m_block_size = spec.config_bundle.chunk_scan.m_block_size
    n_block_size = spec.config_bundle.chunk_scan.n_block_size
    scan_num_threads = spec.config_bundle.chunk_scan.num_threads
    state_num_threads = spec.config_bundle.state_passing.num_threads
    state_vecs_per_thread = spec.config_bundle.state_passing.vecs_per_thread
    has_init = spec.has_init
    state_passing_kernel = StatePassingFwdAmpere(
        num_threads=state_num_threads,
        vecs_per_thread=state_vecs_per_thread,
        copy_bits_in=state_copy_bits_in,
        copy_bits_out=state_copy_bits_out,
        copy_bits_state_in=state_copy_bits_state_in,
        copy_bits_state_out=state_copy_bits_state_out,
        has_init=has_init,
    )

    @cute.jit
    def _forward_aot_host_wrapper(
        U_increment_view: cute.Tensor,
        B_increment_view: cute.Tensor,
        M_increment_view: cute.Tensor,
        K_increment_view: cute.Tensor,
        U_prev_increment_view: cute.Tensor,
        B_prev_increment_view: cute.Tensor,
        increment_view: cute.Tensor,
        chunk_multiplier_view: cute.Tensor,
        increment_state_view: cute.Tensor,
        chunk_multiplier_state_view: cute.Tensor,
        chunk_starts_view: cute.Tensor,
        final_state_view: cute.Tensor,
        initial_state_view: cute.Tensor,
        U_scan_view: cute.Tensor,
        B_scan_view: cute.Tensor,
        C_scan_view: cute.Tensor,
        M_scan_view: cute.Tensor,
        K_scan_view: cute.Tensor,
        Z0_scan_view: cute.Tensor,
        U_prev_scan_view: cute.Tensor,
        B_prev_scan_view: cute.Tensor,
        output_view: cute.Tensor,
        stream: cuda.CUstream,
    ):
        chunk_increment_kernel = ChunkIncrementFwdAmpere(
            cast(Any, U_increment_view.element_type),
            chunk_size=spec.chunk_size,
            cta_tiler=spec.config_bundle.chunk_increment.cta_tiler,
            num_stages=spec.config_bundle.chunk_increment.num_stages,
        )
        K_current_view = cute.make_tensor(
            cast(Any, K_increment_view.iterator) + 2, K_increment_view.layout
        )
        chunk_increment_kernel._launch_kernel(
            U_increment_view,
            B_increment_view,
            M_increment_view,
            K_increment_view,
            K_current_view,
            U_prev_increment_view,
            B_prev_increment_view,
            increment_view,
            chunk_multiplier_view,
            stream=stream,
        )
        state_passing_kernel.call_on_stream(
            increment_state_view,
            chunk_multiplier_state_view,
            chunk_starts_view,
            final_state_view,
            initial_state_view,
            stream,
        )
        chunk_scan_kernel = ChunkScanFwdAmpere(
            D=spec.D,
            P=spec.P,
            L=spec.chunk_size,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=scan_num_threads,
        )
        chunk_scan_kernel._launch_main_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            Z0_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            output_view,
            stream=stream,
        )

    return cute.compile(
        _forward_aot_host_wrapper,
        *compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def export_tvm_ffi_compiled_module(
    compiled,
    *,
    kind: str,
    module_id: str,
    function_name: str,
    package_root: str | os.PathLike[str] | Path,
    keep_object_file: bool = True,
) -> ExportedTVMFFIModule:
    """Export, link, and describe a compiled TVM-FFI module."""
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
    """Load a TVM-FFI CuTe module from an object or shared library file."""
    ensure_cute_runtime_env()
    return cute.runtime.load_module(str(Path(module_path)), enable_tvm_ffi=True)


def load_tvm_ffi_function(
    module_path: str | os.PathLike[str] | Path,
    *,
    function_name: str,
):
    """Load one callable TVM-FFI function from a module artifact."""
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


def _read_manifest(
    package_root: Path,
) -> dict[str, Any]:
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
    package_root: str | os.PathLike[str] | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[dict[str, Any], ...]:
    package_root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    manifest = _read_manifest(package_root)
    return tuple(
        candidate
        for candidate in manifest.get("artifacts", [])
        if candidate.get("kind") == kind
        and _arch_matches(
            str(cast(dict[str, Any], candidate["spec"]).get("arch_tag", "any")),
            arch_tag,
        )
    )


def list_packaged_chunk_increment_aot_specs(
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[ChunkIncrementAOTSpec, ...]:
    return tuple(
        _chunk_increment_spec_from_record(cast(dict[str, Any], record["spec"]))
        for record in _list_packaged_aot_records(
            kind="chunk_increment_fwd",
            package_root=package_root,
            arch_tag=arch_tag,
        )
    )


def list_packaged_state_passing_aot_specs(
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[StatePassingAOTSpec, ...]:
    return tuple(
        _state_passing_spec_from_record(cast(dict[str, Any], record["spec"]))
        for record in _list_packaged_aot_records(
            kind="state_passing_fwd",
            package_root=package_root,
            arch_tag=arch_tag,
        )
    )


def list_packaged_chunk_scan_aot_specs(
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[ChunkScanAOTSpec, ...]:
    return tuple(
        _chunk_scan_spec_from_record(cast(dict[str, Any], record["spec"]))
        for record in _list_packaged_aot_records(
            kind="chunk_scan_fwd",
            package_root=package_root,
            arch_tag=arch_tag,
        )
    )


def list_packaged_forward_aot_specs(
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[ForwardAOTSpec, ...]:
    return tuple(
        _forward_spec_from_record(cast(dict[str, Any], record["spec"]))
        for record in _list_packaged_aot_records(
            kind="v2x2ssd_fwd",
            package_root=package_root,
            arch_tag=arch_tag,
        )
    )


def try_load_packaged_v2x2ssd_fwd_function(
    spec: ForwardAOTSpec,
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
):
    """Return a packaged forward AOT function if one matches ``spec``."""
    if os.environ.get("SLINOSS_DISABLE_CUTE_AOT") == "1":
        return None

    package_root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    module_id = spec.module_id
    cached = _PACKAGED_FORWARD_CACHE.get(module_id)
    if cached is not None:
        return cached

    record = next(
        (
            candidate
            for candidate in _list_packaged_aot_records(
                kind="v2x2ssd_fwd",
                package_root=package_root,
                arch_tag=spec.arch_tag,
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
        _PACKAGED_FORWARD_CACHE[module_id] = loaded
        return loaded
    return None


def _try_load_packaged_stage_function(
    *,
    kind: str,
    module_id: str,
    arch_tag: str,
    package_root: str | os.PathLike[str] | Path | None = None,
):
    if os.environ.get("SLINOSS_DISABLE_CUTE_AOT") == "1":
        return None
    package_root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    cached = _PACKAGED_STAGE_CACHE.get(module_id)
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
        _PACKAGED_STAGE_CACHE[module_id] = loaded
        return loaded
    return None


def try_load_packaged_chunk_increment_function(
    spec: ChunkIncrementAOTSpec,
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
):
    return _try_load_packaged_stage_function(
        kind="chunk_increment_fwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
        package_root=package_root,
    )


def try_load_packaged_state_passing_function(
    spec: StatePassingAOTSpec,
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
):
    return _try_load_packaged_stage_function(
        kind="state_passing_fwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
        package_root=package_root,
    )


def try_load_packaged_chunk_scan_function(
    spec: ChunkScanAOTSpec,
    *,
    package_root: str | os.PathLike[str] | Path | None = None,
):
    return _try_load_packaged_stage_function(
        kind="chunk_scan_fwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
        package_root=package_root,
    )


def clear_packaged_forward_aot_cache() -> None:
    _PACKAGED_FORWARD_CACHE.clear()
    _PACKAGED_STAGE_CACHE.clear()


def export_chunk_increment_cute_aot(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    cta_tiler: tuple[int, int, int] | None = None,
    num_stages: int | None = None,
    arch_tag: str = "any",
    package_root: str | os.PathLike[str] | Path,
) -> ExportedTVMFFIModule:
    spec = _infer_chunk_increment_aot_spec(
        U,
        B,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        arch_tag=arch_tag,
    )
    if cta_tiler is not None or num_stages is not None:
        spec = ChunkIncrementAOTSpec(
            arch_tag=spec.arch_tag,
            P=spec.P,
            D=spec.D,
            chunk_size=spec.chunk_size,
            tc_dtype_name=spec.tc_dtype_name,
            config=ChunkIncrementConfig(
                cta_tiler=(
                    spec.config.cta_tiler
                    if cta_tiler is None
                    else cast(tuple[int, int, int], tuple(int(v) for v in cta_tiler))
                ),
                num_stages=spec.config.num_stages
                if num_stages is None
                else int(num_stages),
            ),
        )
    compiled = _compile_chunk_increment_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="chunk_increment_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="chunk_increment_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_state_passing_cute_aot(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int = 128,
    vecs_per_thread: int = 8,
    arch_tag: str = "any",
    package_root: str | os.PathLike[str] | Path,
) -> ExportedTVMFFIModule:
    spec = _infer_state_passing_aot_spec(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        arch_tag=arch_tag,
    )
    compiled = _compile_state_passing_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="state_passing_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="state_passing_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_chunk_scan_cute_aot(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    chunk_size: int = 64,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    num_threads: int = 128,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    arch_tag: str = "any",
    package_root: str | os.PathLike[str] | Path,
) -> ExportedTVMFFIModule:
    spec = _infer_chunk_scan_aot_spec(
        U,
        B,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        arch_tag=arch_tag,
    )
    compiled = _compile_chunk_scan_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="chunk_scan_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="chunk_scan_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_v2x2ssd_fwd_cute_aot(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    scan_num_threads: int = 128,
    state_num_threads: int = 128,
    state_vecs_per_thread: int = 8,
    chunk_increment_cta_tiler: tuple[int, int, int] | None = None,
    chunk_increment_num_stages: int | None = None,
    arch_tag: str = "any",
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    package_root: str | os.PathLike[str] | Path,
) -> ExportedTVMFFIModule:
    spec = infer_v2x2ssd_fwd_aot_spec(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        scan_num_threads=scan_num_threads,
        state_num_threads=state_num_threads,
        state_vecs_per_thread=state_vecs_per_thread,
        chunk_increment_cta_tiler=chunk_increment_cta_tiler,
        chunk_increment_num_stages=chunk_increment_num_stages,
        arch_tag=arch_tag,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    compiled = _compile_forward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="v2x2ssd_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="v2x2ssd_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_tuned_v2x2ssd_fwd_cute_aot(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    config_bundle: ForwardConfigBundle,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    arch_tag: str = "any",
    package_root: str | os.PathLike[str] | Path,
) -> ExportedTVMFFIModule:
    return export_v2x2ssd_fwd_cute_aot(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=config_bundle.chunk_scan.m_block_size,
        n_block_size=config_bundle.chunk_scan.n_block_size,
        scan_num_threads=config_bundle.chunk_scan.num_threads,
        state_num_threads=config_bundle.state_passing.num_threads,
        state_vecs_per_thread=config_bundle.state_passing.vecs_per_thread,
        chunk_increment_cta_tiler=config_bundle.chunk_increment.cta_tiler,
        chunk_increment_num_stages=config_bundle.chunk_increment.num_stages,
        arch_tag=arch_tag,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        package_root=package_root,
    )


def build_forward_aot_search_space_package(
    *,
    package_root: str | os.PathLike[str] | Path = _PACKAGED_AOT_ROOT,
    specs: tuple[ForwardAOTSpec, ...] | None = None,
    arch_tags: tuple[str, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    package_root = Path(package_root)
    resolved_arch_tags = _resolve_forward_aot_arch_tags(arch_tags)
    resolved_specs = (
        default_forward_aot_specs(resolved_arch_tags) if specs is None else specs
    )
    if clean:
        shutil.rmtree(package_root / "artifacts", ignore_errors=True)
        shutil.rmtree(package_root / "runtime", ignore_errors=True)
        (package_root / "manifest.json").unlink(missing_ok=True)
    exported_modules: list[ExportedTVMFFIModule] = []
    for arch_tag in resolved_arch_tags:
        for spec in _search_space_chunk_increment_specs(arch_tag=arch_tag):
            compiled = _compile_chunk_increment_aot(spec)
            exported = export_tvm_ffi_compiled_module(
                compiled,
                kind="chunk_increment_fwd",
                module_id=spec.module_id,
                function_name=spec.module_id,
                package_root=package_root,
                keep_object_file=False,
            )
            register_aot_artifact(
                kind="chunk_increment_fwd",
                spec=spec,
                exported=exported,
                package_root=package_root,
            )
            exported_modules.append(exported)
        for spec in _search_space_state_passing_specs(arch_tag=arch_tag):
            compiled = _compile_state_passing_aot(spec)
            exported = export_tvm_ffi_compiled_module(
                compiled,
                kind="state_passing_fwd",
                module_id=spec.module_id,
                function_name=spec.module_id,
                package_root=package_root,
                keep_object_file=False,
            )
            register_aot_artifact(
                kind="state_passing_fwd",
                spec=spec,
                exported=exported,
                package_root=package_root,
            )
            exported_modules.append(exported)
        for spec in _search_space_chunk_scan_specs(arch_tag=arch_tag):
            compiled = _compile_chunk_scan_aot(spec)
            exported = export_tvm_ffi_compiled_module(
                compiled,
                kind="chunk_scan_fwd",
                module_id=spec.module_id,
                function_name=spec.module_id,
                package_root=package_root,
                keep_object_file=False,
            )
            register_aot_artifact(
                kind="chunk_scan_fwd",
                spec=spec,
                exported=exported,
                package_root=package_root,
            )
            exported_modules.append(exported)
    for spec in resolved_specs:
        compiled = _compile_forward_aot(spec)
        exported = export_tvm_ffi_compiled_module(
            compiled,
            kind="v2x2ssd_fwd",
            module_id=spec.module_id,
            function_name=spec.module_id,
            package_root=package_root,
            keep_object_file=False,
        )
        register_aot_artifact(
            kind="v2x2ssd_fwd",
            spec=spec,
            exported=exported,
            package_root=package_root,
        )
        exported_modules.append(exported)
    return tuple(exported_modules)


def build_default_forward_aot_package(
    *,
    package_root: str | os.PathLike[str] | Path = _PACKAGED_AOT_ROOT,
    specs: tuple[ForwardAOTSpec, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    return build_forward_aot_search_space_package(
        package_root=package_root,
        specs=specs,
        clean=clean,
    )


__all__ = [
    "ChunkIncrementAOTSpec",
    "ChunkScanAOTSpec",
    "ExportedTVMFFIModule",
    "ForwardAOTSpec",
    "StatePassingAOTSpec",
    "build_default_forward_aot_package",
    "build_forward_aot_search_space_package",
    "clear_packaged_forward_aot_cache",
    "default_forward_aot_specs",
    "export_chunk_increment_cute_aot",
    "export_chunk_scan_cute_aot",
    "export_state_passing_cute_aot",
    "export_tvm_ffi_compiled_module",
    "export_tuned_v2x2ssd_fwd_cute_aot",
    "export_v2x2ssd_fwd_cute_aot",
    "find_tvm_ffi_runtime_libraries",
    "infer_v2x2ssd_fwd_aot_spec",
    "list_packaged_chunk_increment_aot_specs",
    "list_packaged_chunk_scan_aot_specs",
    "list_packaged_forward_aot_specs",
    "list_packaged_state_passing_aot_specs",
    "load_tvm_ffi_function",
    "load_tvm_ffi_module",
    "register_aot_artifact",
    "try_load_packaged_chunk_increment_function",
    "try_load_packaged_chunk_scan_function",
    "try_load_packaged_state_passing_function",
    "try_load_packaged_v2x2ssd_fwd_function",
]
