"""Ahead-of-time helpers for the CuTe FFN rowwise kernels.

This packaged surface owns only the non-GEMM FFN rowwise stages:

- RMSNorm forward/backward
- GELU/SwiGLU activation forward/backward

The input/output projection GEMMs remain on the eager/vendor path and are not
part of this AOT payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Any, cast

import cutlass.cute as cute
import torch

from slinoss._cute_aot import (
    ExportedTVMFFIModule,
    _aot_compile_options,
    _dtype_from_name,
    _dtype_name,
    _dtype_tag,
    _list_packaged_aot_records,
    _resolve_cute_aot_arch_tags,
    _sanitize_stem,
    _try_load_packaged_compiled_function,
    export_tvm_ffi_compiled_module,
    register_aot_artifact,
)

from ..activation import FfnActivationBwdFused, FfnActivationFwdFused
from ..common import ActivationKind, make_fake_tensor_arg, validate_activation_operands
from ..norm import FfnRmsNormBwdFused, FfnRmsNormFwdFused
from ..common import validate_norm_operands

_PACKAGED_AOT_ROOT = Path(__file__).resolve().parent
_PACKAGED_NORM_FORWARD_CACHE: dict[str, object] = {}
_PACKAGED_NORM_BACKWARD_CACHE: dict[str, object] = {}
_PACKAGED_ACT_FORWARD_CACHE: dict[str, object] = {}
_PACKAGED_ACT_BACKWARD_CACHE: dict[str, object] = {}

_DEFAULT_AOT_MODEL_HIDDEN_DIMS: tuple[tuple[int, int], ...] = (
    (576, 2304),
    (768, 3072),
    (1024, 4096),
)
_DEFAULT_AOT_ACTIVATION_KINDS: tuple[ActivationKind, ...] = ("gelu", "swiglu")
_DEFAULT_ACTIVATION_DTYPE_NAME = "bfloat16"
_DEFAULT_NORM_DTYPE_NAME = "bfloat16"
_DEFAULT_NORM_WEIGHT_DTYPE_NAME = "bfloat16"
_DEFAULT_EPS = 1.0e-5
_REPRESENTATIVE_BATCH_SIZE = 2
_REPRESENTATIVE_TIME_STEPS = 64


def _float_tag(value: float) -> str:
    text = f"{float(value):.8g}"
    text = text.replace("+", "")
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


@dataclass(frozen=True)
class NormForwardAOTSpec:
    arch_tag: str
    d_model: int
    residual_dtype_name: str
    norm_weight_dtype_name: str
    eps: float

    @property
    def residual_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.residual_dtype_name)

    @property
    def norm_weight_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.norm_weight_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "ffn_norm_fwd"
            f"__arch{self.arch_tag}"
            f"__d{self.d_model}"
            f"__x{_dtype_tag(self.residual_dtype)}"
            f"__w{_dtype_tag(self.norm_weight_dtype)}"
            f"__eps{_float_tag(self.eps)}"
        )


@dataclass(frozen=True)
class NormBackwardAOTSpec:
    arch_tag: str
    d_model: int
    residual_dtype_name: str
    norm_weight_dtype_name: str
    d_output_dtype_name: str
    eps: float

    @property
    def residual_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.residual_dtype_name)

    @property
    def norm_weight_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.norm_weight_dtype_name)

    @property
    def d_output_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.d_output_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "ffn_norm_bwd"
            f"__arch{self.arch_tag}"
            f"__d{self.d_model}"
            f"__x{_dtype_tag(self.residual_dtype)}"
            f"__w{_dtype_tag(self.norm_weight_dtype)}"
            f"__g{_dtype_tag(self.d_output_dtype)}"
            f"__eps{_float_tag(self.eps)}"
        )


@dataclass(frozen=True)
class ActivationForwardAOTSpec:
    arch_tag: str
    hidden_dim: int
    kind: ActivationKind
    projected_dtype_name: str

    @property
    def projected_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.projected_dtype_name)

    @property
    def projected_dim(self) -> int:
        return int(self.hidden_dim if self.kind == "gelu" else 2 * self.hidden_dim)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "ffn_activation_fwd"
            f"__arch{self.arch_tag}"
            f"__kind{self.kind}"
            f"__h{self.hidden_dim}"
            f"__x{_dtype_tag(self.projected_dtype)}"
        )


@dataclass(frozen=True)
class ActivationBackwardAOTSpec:
    arch_tag: str
    hidden_dim: int
    kind: ActivationKind
    projected_dtype_name: str
    d_hidden_dtype_name: str

    @property
    def projected_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.projected_dtype_name)

    @property
    def d_hidden_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.d_hidden_dtype_name)

    @property
    def projected_dim(self) -> int:
        return int(self.hidden_dim if self.kind == "gelu" else 2 * self.hidden_dim)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "ffn_activation_bwd"
            f"__arch{self.arch_tag}"
            f"__kind{self.kind}"
            f"__h{self.hidden_dim}"
            f"__x{_dtype_tag(self.projected_dtype)}"
            f"__g{_dtype_tag(self.d_hidden_dtype)}"
        )


def _default_norm_forward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[NormForwardAOTSpec, ...]:
    return tuple(
        NormForwardAOTSpec(
            arch_tag=arch_tag,
            d_model=d_model,
            residual_dtype_name=_DEFAULT_NORM_DTYPE_NAME,
            norm_weight_dtype_name=_DEFAULT_NORM_WEIGHT_DTYPE_NAME,
            eps=_DEFAULT_EPS,
        )
        for arch_tag in _resolve_cute_aot_arch_tags(arch_tags)
        for d_model, _ in _DEFAULT_AOT_MODEL_HIDDEN_DIMS
    )


def _default_norm_backward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[NormBackwardAOTSpec, ...]:
    return tuple(
        NormBackwardAOTSpec(
            arch_tag=arch_tag,
            d_model=d_model,
            residual_dtype_name=_DEFAULT_NORM_DTYPE_NAME,
            norm_weight_dtype_name=_DEFAULT_NORM_WEIGHT_DTYPE_NAME,
            d_output_dtype_name=_DEFAULT_NORM_DTYPE_NAME,
            eps=_DEFAULT_EPS,
        )
        for arch_tag in _resolve_cute_aot_arch_tags(arch_tags)
        for d_model, _ in _DEFAULT_AOT_MODEL_HIDDEN_DIMS
    )


def _default_activation_forward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[ActivationForwardAOTSpec, ...]:
    return tuple(
        ActivationForwardAOTSpec(
            arch_tag=arch_tag,
            hidden_dim=hidden_dim,
            kind=kind,
            projected_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
        )
        for arch_tag in _resolve_cute_aot_arch_tags(arch_tags)
        for _, hidden_dim in _DEFAULT_AOT_MODEL_HIDDEN_DIMS
        for kind in _DEFAULT_AOT_ACTIVATION_KINDS
    )


def _default_activation_backward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[ActivationBackwardAOTSpec, ...]:
    return tuple(
        ActivationBackwardAOTSpec(
            arch_tag=arch_tag,
            hidden_dim=hidden_dim,
            kind=kind,
            projected_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            d_hidden_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
        )
        for arch_tag in _resolve_cute_aot_arch_tags(arch_tags)
        for _, hidden_dim in _DEFAULT_AOT_MODEL_HIDDEN_DIMS
        for kind in _DEFAULT_AOT_ACTIVATION_KINDS
    )


@lru_cache(maxsize=8)
def default_forward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[NormForwardAOTSpec | ActivationForwardAOTSpec, ...]:
    return (
        *_default_norm_forward_aot_specs(arch_tags),
        *_default_activation_forward_aot_specs(arch_tags),
    )


@lru_cache(maxsize=8)
def default_backward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[NormBackwardAOTSpec | ActivationBackwardAOTSpec, ...]:
    return (
        *_default_norm_backward_aot_specs(arch_tags),
        *_default_activation_backward_aot_specs(arch_tags),
    )


def _norm_forward_spec_from_record(record: dict[str, Any]) -> NormForwardAOTSpec:
    return NormForwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        d_model=int(record["d_model"]),
        residual_dtype_name=str(record["residual_dtype_name"]),
        norm_weight_dtype_name=str(record["norm_weight_dtype_name"]),
        eps=float(record.get("eps", _DEFAULT_EPS)),
    )


def _norm_backward_spec_from_record(record: dict[str, Any]) -> NormBackwardAOTSpec:
    return NormBackwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        d_model=int(record["d_model"]),
        residual_dtype_name=str(record["residual_dtype_name"]),
        norm_weight_dtype_name=str(record["norm_weight_dtype_name"]),
        d_output_dtype_name=str(
            record.get("d_output_dtype_name", record["residual_dtype_name"])
        ),
        eps=float(record.get("eps", _DEFAULT_EPS)),
    )


def _activation_forward_spec_from_record(
    record: dict[str, Any],
) -> ActivationForwardAOTSpec:
    return ActivationForwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        hidden_dim=int(record["hidden_dim"]),
        kind=cast(ActivationKind, str(record["kind"])),
        projected_dtype_name=str(record["projected_dtype_name"]),
    )


def _activation_backward_spec_from_record(
    record: dict[str, Any],
) -> ActivationBackwardAOTSpec:
    return ActivationBackwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        hidden_dim=int(record["hidden_dim"]),
        kind=cast(ActivationKind, str(record["kind"])),
        projected_dtype_name=str(record["projected_dtype_name"]),
        d_hidden_dtype_name=str(
            record.get("d_hidden_dtype_name", record["projected_dtype_name"])
        ),
    )


def list_packaged_forward_aot_specs(
    *,
    package_root: str | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[NormForwardAOTSpec | ActivationForwardAOTSpec, ...]:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return (
        *tuple(
            _norm_forward_spec_from_record(cast(dict[str, Any], record["spec"]))
            for record in _list_packaged_aot_records(
                kind="ffn_norm_fwd",
                package_root=root,
                arch_tag=arch_tag,
            )
        ),
        *tuple(
            _activation_forward_spec_from_record(cast(dict[str, Any], record["spec"]))
            for record in _list_packaged_aot_records(
                kind="ffn_activation_fwd",
                package_root=root,
                arch_tag=arch_tag,
            )
        ),
    )


def list_packaged_backward_aot_specs(
    *,
    package_root: str | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[NormBackwardAOTSpec | ActivationBackwardAOTSpec, ...]:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return (
        *tuple(
            _norm_backward_spec_from_record(cast(dict[str, Any], record["spec"]))
            for record in _list_packaged_aot_records(
                kind="ffn_norm_bwd",
                package_root=root,
                arch_tag=arch_tag,
            )
        ),
        *tuple(
            _activation_backward_spec_from_record(cast(dict[str, Any], record["spec"]))
            for record in _list_packaged_aot_records(
                kind="ffn_activation_bwd",
                package_root=root,
                arch_tag=arch_tag,
            )
        ),
    )


def try_load_packaged_ffn_norm_fwd_function(
    spec: NormForwardAOTSpec,
    *,
    package_root: str | Path | None = None,
) -> object | None:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        cache=_PACKAGED_NORM_FORWARD_CACHE,
        package_root=root,
        kind="ffn_norm_fwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
    )


def try_load_packaged_ffn_norm_bwd_function(
    spec: NormBackwardAOTSpec,
    *,
    package_root: str | Path | None = None,
) -> object | None:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        cache=_PACKAGED_NORM_BACKWARD_CACHE,
        package_root=root,
        kind="ffn_norm_bwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
    )


def try_load_packaged_ffn_activation_fwd_function(
    spec: ActivationForwardAOTSpec,
    *,
    package_root: str | Path | None = None,
) -> object | None:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        cache=_PACKAGED_ACT_FORWARD_CACHE,
        package_root=root,
        kind="ffn_activation_fwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
    )


def try_load_packaged_ffn_activation_bwd_function(
    spec: ActivationBackwardAOTSpec,
    *,
    package_root: str | Path | None = None,
) -> object | None:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        cache=_PACKAGED_ACT_BACKWARD_CACHE,
        package_root=root,
        kind="ffn_activation_bwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
    )


def clear_packaged_aot_cache() -> None:
    _PACKAGED_NORM_FORWARD_CACHE.clear()
    _PACKAGED_NORM_BACKWARD_CACHE.clear()
    _PACKAGED_ACT_FORWARD_CACHE.clear()
    _PACKAGED_ACT_BACKWARD_CACHE.clear()


def infer_ffn_norm_fwd_aot_spec(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    arch_tag: str = "any",
) -> NormForwardAOTSpec:
    input_info = validate_norm_operands(residual, norm_weight)
    return NormForwardAOTSpec(
        arch_tag=arch_tag,
        d_model=input_info.hidden_dim,
        residual_dtype_name=_dtype_name(residual.dtype),
        norm_weight_dtype_name=_dtype_name(norm_weight.dtype),
        eps=float(eps),
    )


def infer_ffn_norm_bwd_aot_spec(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    d_output: torch.Tensor,
    *,
    eps: float,
    arch_tag: str = "any",
) -> NormBackwardAOTSpec:
    input_info = validate_norm_operands(residual, norm_weight)
    if tuple(map(int, d_output.shape)) != tuple(map(int, residual.shape)):
        raise ValueError(
            "d_output must match residual shape; got "
            f"{tuple(map(int, d_output.shape))} expected {tuple(map(int, residual.shape))}."
        )
    return NormBackwardAOTSpec(
        arch_tag=arch_tag,
        d_model=input_info.hidden_dim,
        residual_dtype_name=_dtype_name(residual.dtype),
        norm_weight_dtype_name=_dtype_name(norm_weight.dtype),
        d_output_dtype_name=_dtype_name(d_output.dtype),
        eps=float(eps),
    )


def infer_ffn_activation_fwd_aot_spec(
    projected: torch.Tensor,
    *,
    kind: ActivationKind,
    arch_tag: str = "any",
) -> ActivationForwardAOTSpec:
    input_info = validate_activation_operands(projected, None, kind=kind)
    return ActivationForwardAOTSpec(
        arch_tag=arch_tag,
        hidden_dim=input_info.hidden_dim,
        kind=input_info.kind,
        projected_dtype_name=_dtype_name(projected.dtype),
    )


def infer_ffn_activation_bwd_aot_spec(
    projected: torch.Tensor,
    d_hidden: torch.Tensor,
    *,
    kind: ActivationKind,
    arch_tag: str = "any",
) -> ActivationBackwardAOTSpec:
    input_info = validate_activation_operands(projected, d_hidden, kind=kind)
    return ActivationBackwardAOTSpec(
        arch_tag=arch_tag,
        hidden_dim=input_info.hidden_dim,
        kind=input_info.kind,
        projected_dtype_name=_dtype_name(projected.dtype),
        d_hidden_dtype_name=_dtype_name(d_hidden.dtype),
    )


def _compile_norm_forward_aot(spec: NormForwardAOTSpec):
    residual = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, spec.d_model),
        dtype=spec.residual_dtype,
    )
    norm_weight = torch.empty((spec.d_model,), dtype=spec.norm_weight_dtype)
    output = torch.empty_like(residual)
    return cute.compile(
        FfnRmsNormFwdFused(hidden_dim=spec.d_model, eps=spec.eps),
        make_fake_tensor_arg(residual),
        make_fake_tensor_arg(norm_weight),
        make_fake_tensor_arg(output),
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_norm_backward_aot(spec: NormBackwardAOTSpec):
    residual = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, spec.d_model),
        dtype=spec.residual_dtype,
    )
    norm_weight = torch.empty((spec.d_model,), dtype=spec.norm_weight_dtype)
    d_output = torch.empty_like(residual, dtype=spec.d_output_dtype)
    d_input = torch.empty_like(residual)
    d_weight_accum = torch.empty((spec.d_model,), dtype=torch.float32)
    return cute.compile(
        FfnRmsNormBwdFused(hidden_dim=spec.d_model, eps=spec.eps),
        make_fake_tensor_arg(residual),
        make_fake_tensor_arg(norm_weight),
        make_fake_tensor_arg(d_output),
        make_fake_tensor_arg(d_input),
        make_fake_tensor_arg(d_weight_accum),
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_activation_forward_aot(spec: ActivationForwardAOTSpec):
    projected = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            _REPRESENTATIVE_TIME_STEPS,
            spec.projected_dim,
        ),
        dtype=spec.projected_dtype,
    )
    hidden = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, spec.hidden_dim),
        dtype=spec.projected_dtype,
    )
    return cute.compile(
        FfnActivationFwdFused(hidden_dim=spec.hidden_dim, kind=spec.kind),
        make_fake_tensor_arg(projected),
        make_fake_tensor_arg(hidden),
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_activation_backward_aot(spec: ActivationBackwardAOTSpec):
    projected = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            _REPRESENTATIVE_TIME_STEPS,
            spec.projected_dim,
        ),
        dtype=spec.projected_dtype,
    )
    d_hidden = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, spec.hidden_dim),
        dtype=spec.d_hidden_dtype,
    )
    d_projected = torch.empty_like(projected)
    return cute.compile(
        FfnActivationBwdFused(hidden_dim=spec.hidden_dim, kind=spec.kind),
        make_fake_tensor_arg(projected),
        make_fake_tensor_arg(d_hidden),
        make_fake_tensor_arg(d_projected),
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def export_ffn_norm_fwd_cute_aot(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_ffn_norm_fwd_aot_spec(
        residual,
        norm_weight,
        eps=eps,
        arch_tag=arch_tag,
    )
    compiled = _compile_norm_forward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="ffn_norm_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="ffn_norm_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_ffn_norm_bwd_cute_aot(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    d_output: torch.Tensor,
    *,
    eps: float,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_ffn_norm_bwd_aot_spec(
        residual,
        norm_weight,
        d_output,
        eps=eps,
        arch_tag=arch_tag,
    )
    compiled = _compile_norm_backward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="ffn_norm_bwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="ffn_norm_bwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_ffn_activation_fwd_cute_aot(
    projected: torch.Tensor,
    *,
    kind: ActivationKind,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_ffn_activation_fwd_aot_spec(
        projected,
        kind=kind,
        arch_tag=arch_tag,
    )
    compiled = _compile_activation_forward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="ffn_activation_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="ffn_activation_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_ffn_activation_bwd_cute_aot(
    projected: torch.Tensor,
    d_hidden: torch.Tensor,
    *,
    kind: ActivationKind,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_ffn_activation_bwd_aot_spec(
        projected,
        d_hidden,
        kind=kind,
        arch_tag=arch_tag,
    )
    compiled = _compile_activation_backward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="ffn_activation_bwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="ffn_activation_bwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def build_default_forward_aot_package(
    *,
    package_root: str | Path = _PACKAGED_AOT_ROOT,
    specs: tuple[NormForwardAOTSpec | ActivationForwardAOTSpec, ...] | None = None,
    arch_tags: tuple[str, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    package_root = Path(package_root)
    if clean:
        shutil.rmtree(package_root / "artifacts", ignore_errors=True)
        shutil.rmtree(package_root / "runtime", ignore_errors=True)
        (package_root / "manifest.json").unlink(missing_ok=True)
    exported: list[ExportedTVMFFIModule] = []
    selected_specs = default_forward_aot_specs(arch_tags) if specs is None else specs
    for spec in selected_specs:
        if isinstance(spec, NormForwardAOTSpec):
            compiled = _compile_norm_forward_aot(spec)
            module = export_tvm_ffi_compiled_module(
                compiled,
                kind="ffn_norm_fwd",
                module_id=spec.module_id,
                function_name=spec.module_id,
                package_root=package_root,
                keep_object_file=False,
            )
            register_aot_artifact(
                kind="ffn_norm_fwd",
                spec=spec,
                exported=module,
                package_root=package_root,
            )
            exported.append(module)
            continue
        if isinstance(spec, ActivationForwardAOTSpec):
            compiled = _compile_activation_forward_aot(spec)
            module = export_tvm_ffi_compiled_module(
                compiled,
                kind="ffn_activation_fwd",
                module_id=spec.module_id,
                function_name=spec.module_id,
                package_root=package_root,
                keep_object_file=False,
            )
            register_aot_artifact(
                kind="ffn_activation_fwd",
                spec=spec,
                exported=module,
                package_root=package_root,
            )
            exported.append(module)
            continue
        raise TypeError(f"Unsupported FFN forward AOT spec: {type(spec)!r}.")
    return tuple(exported)


def build_default_backward_aot_package(
    *,
    package_root: str | Path = _PACKAGED_AOT_ROOT,
    specs: tuple[NormBackwardAOTSpec | ActivationBackwardAOTSpec, ...] | None = None,
    arch_tags: tuple[str, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    package_root = Path(package_root)
    if clean:
        shutil.rmtree(package_root / "artifacts", ignore_errors=True)
        shutil.rmtree(package_root / "runtime", ignore_errors=True)
        (package_root / "manifest.json").unlink(missing_ok=True)
    exported: list[ExportedTVMFFIModule] = []
    selected_specs = default_backward_aot_specs(arch_tags) if specs is None else specs
    for spec in selected_specs:
        if isinstance(spec, NormBackwardAOTSpec):
            compiled = _compile_norm_backward_aot(spec)
            module = export_tvm_ffi_compiled_module(
                compiled,
                kind="ffn_norm_bwd",
                module_id=spec.module_id,
                function_name=spec.module_id,
                package_root=package_root,
                keep_object_file=False,
            )
            register_aot_artifact(
                kind="ffn_norm_bwd",
                spec=spec,
                exported=module,
                package_root=package_root,
            )
            exported.append(module)
            continue
        if isinstance(spec, ActivationBackwardAOTSpec):
            compiled = _compile_activation_backward_aot(spec)
            module = export_tvm_ffi_compiled_module(
                compiled,
                kind="ffn_activation_bwd",
                module_id=spec.module_id,
                function_name=spec.module_id,
                package_root=package_root,
                keep_object_file=False,
            )
            register_aot_artifact(
                kind="ffn_activation_bwd",
                spec=spec,
                exported=module,
                package_root=package_root,
            )
            exported.append(module)
            continue
        raise TypeError(f"Unsupported FFN backward AOT spec: {type(spec)!r}.")
    return tuple(exported)


def build_default_cute_aot_package(
    *,
    package_root: str | Path = _PACKAGED_AOT_ROOT,
    arch_tags: tuple[str, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    package_root = Path(package_root)
    exported_forward = build_default_forward_aot_package(
        package_root=package_root,
        arch_tags=arch_tags,
        clean=clean,
    )
    exported_backward = build_default_backward_aot_package(
        package_root=package_root,
        arch_tags=arch_tags,
        clean=False,
    )
    return (*exported_forward, *exported_backward)


__all__ = [
    "ActivationBackwardAOTSpec",
    "ActivationForwardAOTSpec",
    "ExportedTVMFFIModule",
    "NormBackwardAOTSpec",
    "NormForwardAOTSpec",
    "build_default_backward_aot_package",
    "build_default_cute_aot_package",
    "build_default_forward_aot_package",
    "clear_packaged_aot_cache",
    "default_backward_aot_specs",
    "default_forward_aot_specs",
    "export_ffn_activation_bwd_cute_aot",
    "export_ffn_activation_fwd_cute_aot",
    "export_ffn_norm_bwd_cute_aot",
    "export_ffn_norm_fwd_cute_aot",
    "infer_ffn_activation_bwd_aot_spec",
    "infer_ffn_activation_fwd_aot_spec",
    "infer_ffn_norm_bwd_aot_spec",
    "infer_ffn_norm_fwd_aot_spec",
    "list_packaged_backward_aot_specs",
    "list_packaged_forward_aot_specs",
    "try_load_packaged_ffn_activation_bwd_function",
    "try_load_packaged_ffn_activation_fwd_function",
    "try_load_packaged_ffn_norm_bwd_function",
    "try_load_packaged_ffn_norm_fwd_function",
]
