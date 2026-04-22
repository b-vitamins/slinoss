"""Ahead-of-time helpers for the CuTe mixer-tail rowwise kernels.

This packaged surface owns only the CuTe rowwise stage:

- forward: skip/add + SiLU gate + RMSNorm
- backward: rowwise gradients for scan output, gate, skip, and norm weight

The output projection remains on the eager/vendor GEMM path and is not part of
this AOT payload.
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

from ..bwd import (
    BackwardCompileArtifacts,
    BackwardOutputs,
    BackwardRuntimeArtifacts,
    _make_backward_compile_artifacts,
    _make_backward_host_wrapper,
)
from ..common import (
    TailRowwiseInputInfo,
    _runtime_alignments,
    _runtime_signature_key,
    dummy_d_skip,
    dummy_skip_input,
)
from ..fwd import (
    ForwardCompileArtifacts,
    ForwardRuntimeArtifacts,
    _make_forward_compile_artifacts,
    _make_forward_host_wrapper,
)

_PACKAGED_AOT_ROOT = Path(__file__).resolve().parent
_PACKAGED_FORWARD_CACHE: dict[str, object] = {}
_PACKAGED_BACKWARD_CACHE: dict[str, object] = {}

_DEFAULT_AOT_GEOMETRIES: tuple[tuple[int, int, bool], ...] = (
    # Default training workload: d_inner=1152, d_head=64 => heads=18.
    (18, 64, True),
    # Common larger production geometries with d_head=64.
    (24, 64, True),
    (32, 64, True),
    (48, 64, True),
    (64, 64, True),
)
_DEFAULT_ACTIVATION_DTYPE_NAME = "bfloat16"
_DEFAULT_D_SKIP_DTYPE_NAME = "float32"
_DEFAULT_TAIL_EPS = 1.0e-5
_REPRESENTATIVE_BATCH_SIZE = 2
_REPRESENTATIVE_TIME_STEPS = 64


def _float_tag(value: float) -> str:
    text = f"{float(value):.8g}"
    text = text.replace("+", "")
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


def _validate_supported_dtype(dtype: torch.dtype, *, name: str) -> None:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported {name} dtype: {dtype}.")


def _infer_input_info(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    *,
    has_skip: bool,
) -> TailRowwiseInputInfo:
    if scan_output.ndim != 4:
        raise ValueError(
            "scan_output must be 4-D; got "
            f"{scan_output.ndim}-D with shape {tuple(map(int, scan_output.shape))}."
        )
    batch_size, heads, time_steps, d_head = map(int, scan_output.shape)
    expected_gate = (batch_size, time_steps, heads * d_head)
    if gate.ndim != 3 or tuple(map(int, gate.shape)) != expected_gate:
        raise ValueError(
            f"gate must be {expected_gate}; got {tuple(map(int, gate.shape))}."
        )
    return TailRowwiseInputInfo(
        batch_size=batch_size,
        time_steps=time_steps,
        heads=heads,
        d_head=d_head,
        has_skip=has_skip,
        device_index=0,
    )


def _validate_rowwise_operands(
    *,
    input_info: TailRowwiseInputInfo,
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    skip_input: torch.Tensor | None,
    d_skip: torch.Tensor | None,
    d_normed: torch.Tensor | None = None,
) -> None:
    _validate_supported_dtype(scan_output.dtype, name="scan_output")
    _validate_supported_dtype(gate.dtype, name="gate")
    _validate_supported_dtype(out_norm_weight.dtype, name="out_norm_weight")
    if (
        out_norm_weight.ndim != 1
        or int(out_norm_weight.numel()) != input_info.hidden_dim
    ):
        raise ValueError(
            "out_norm_weight must be 1-D with length "
            f"{input_info.hidden_dim}; got {tuple(map(int, out_norm_weight.shape))}."
        )
    if (skip_input is None) != (d_skip is None):
        raise ValueError(
            "skip_input and d_skip must either both be provided or both be None."
        )
    if input_info.has_skip:
        assert skip_input is not None
        assert d_skip is not None
        expected_skip = (
            input_info.batch_size,
            input_info.heads,
            input_info.time_steps,
            input_info.d_head,
        )
        if tuple(map(int, skip_input.shape)) != expected_skip:
            raise ValueError(
                f"skip_input must be {expected_skip}; got {tuple(map(int, skip_input.shape))}."
            )
        if d_skip.ndim != 1 or int(d_skip.numel()) != input_info.heads:
            raise ValueError(
                f"d_skip must be 1-D with length {input_info.heads}; got {tuple(map(int, d_skip.shape))}."
            )
        _validate_supported_dtype(skip_input.dtype, name="skip_input")
        _validate_supported_dtype(d_skip.dtype, name="d_skip")
    if d_normed is not None:
        expected_d_normed = (
            input_info.batch_size,
            input_info.time_steps,
            input_info.hidden_dim,
        )
        if tuple(map(int, d_normed.shape)) != expected_d_normed:
            raise ValueError(
                f"d_normed must be {expected_d_normed}; got {tuple(map(int, d_normed.shape))}."
            )
        _validate_supported_dtype(d_normed.dtype, name="d_normed")


@dataclass(frozen=True)
class ForwardAOTSpec:
    arch_tag: str
    heads: int
    d_head: int
    has_skip: bool
    scan_dtype_name: str
    gate_dtype_name: str
    norm_dtype_name: str
    skip_dtype_name: str
    d_skip_dtype_name: str
    eps: float

    @property
    def scan_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.scan_dtype_name)

    @property
    def gate_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.gate_dtype_name)

    @property
    def norm_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.norm_dtype_name)

    @property
    def skip_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.skip_dtype_name)

    @property
    def d_skip_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.d_skip_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "mixer_tail_rowwise_fwd"
            f"__arch{self.arch_tag}"
            f"__h{self.heads}_p{self.d_head}_skip{int(self.has_skip)}"
            f"__scan{_dtype_tag(self.scan_dtype)}"
            f"__gate{_dtype_tag(self.gate_dtype)}"
            f"__norm{_dtype_tag(self.norm_dtype)}"
            f"__skip{_dtype_tag(self.skip_dtype)}"
            f"__dskip{_dtype_tag(self.d_skip_dtype)}"
            f"__eps{_float_tag(self.eps)}"
        )


@dataclass(frozen=True)
class BackwardAOTSpec:
    arch_tag: str
    heads: int
    d_head: int
    has_skip: bool
    scan_dtype_name: str
    gate_dtype_name: str
    norm_dtype_name: str
    skip_dtype_name: str
    d_skip_dtype_name: str
    d_normed_dtype_name: str
    eps: float

    @property
    def scan_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.scan_dtype_name)

    @property
    def gate_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.gate_dtype_name)

    @property
    def norm_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.norm_dtype_name)

    @property
    def skip_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.skip_dtype_name)

    @property
    def d_skip_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.d_skip_dtype_name)

    @property
    def d_normed_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.d_normed_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "mixer_tail_rowwise_bwd"
            f"__arch{self.arch_tag}"
            f"__h{self.heads}_p{self.d_head}_skip{int(self.has_skip)}"
            f"__scan{_dtype_tag(self.scan_dtype)}"
            f"__gate{_dtype_tag(self.gate_dtype)}"
            f"__norm{_dtype_tag(self.norm_dtype)}"
            f"__skip{_dtype_tag(self.skip_dtype)}"
            f"__dskip{_dtype_tag(self.d_skip_dtype)}"
            f"__dnormed{_dtype_tag(self.d_normed_dtype)}"
            f"__eps{_float_tag(self.eps)}"
        )


@lru_cache(maxsize=8)
def default_forward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[ForwardAOTSpec, ...]:
    resolved_arch_tags = _resolve_cute_aot_arch_tags(arch_tags)
    return tuple(
        ForwardAOTSpec(
            arch_tag=arch_tag,
            heads=heads,
            d_head=d_head,
            has_skip=has_skip,
            scan_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            gate_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            norm_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            skip_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            d_skip_dtype_name=_DEFAULT_D_SKIP_DTYPE_NAME,
            eps=_DEFAULT_TAIL_EPS,
        )
        for arch_tag in resolved_arch_tags
        for heads, d_head, has_skip in _DEFAULT_AOT_GEOMETRIES
    )


@lru_cache(maxsize=8)
def default_backward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[BackwardAOTSpec, ...]:
    resolved_arch_tags = _resolve_cute_aot_arch_tags(arch_tags)
    return tuple(
        BackwardAOTSpec(
            arch_tag=arch_tag,
            heads=heads,
            d_head=d_head,
            has_skip=has_skip,
            scan_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            gate_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            norm_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            skip_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            d_skip_dtype_name=_DEFAULT_D_SKIP_DTYPE_NAME,
            d_normed_dtype_name=_DEFAULT_ACTIVATION_DTYPE_NAME,
            eps=_DEFAULT_TAIL_EPS,
        )
        for arch_tag in resolved_arch_tags
        for heads, d_head, has_skip in _DEFAULT_AOT_GEOMETRIES
    )


def _forward_spec_from_record(record: dict[str, Any]) -> ForwardAOTSpec:
    return ForwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        heads=int(record["heads"]),
        d_head=int(record["d_head"]),
        has_skip=bool(record.get("has_skip", True)),
        scan_dtype_name=str(record["scan_dtype_name"]),
        gate_dtype_name=str(record["gate_dtype_name"]),
        norm_dtype_name=str(record["norm_dtype_name"]),
        skip_dtype_name=str(record.get("skip_dtype_name", record["scan_dtype_name"])),
        d_skip_dtype_name=str(record.get("d_skip_dtype_name", "float32")),
        eps=float(record.get("eps", _DEFAULT_TAIL_EPS)),
    )


def _backward_spec_from_record(record: dict[str, Any]) -> BackwardAOTSpec:
    return BackwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        heads=int(record["heads"]),
        d_head=int(record["d_head"]),
        has_skip=bool(record.get("has_skip", True)),
        scan_dtype_name=str(record["scan_dtype_name"]),
        gate_dtype_name=str(record["gate_dtype_name"]),
        norm_dtype_name=str(record["norm_dtype_name"]),
        skip_dtype_name=str(record.get("skip_dtype_name", record["scan_dtype_name"])),
        d_skip_dtype_name=str(record.get("d_skip_dtype_name", "float32")),
        d_normed_dtype_name=str(
            record.get("d_normed_dtype_name", record["scan_dtype_name"])
        ),
        eps=float(record.get("eps", _DEFAULT_TAIL_EPS)),
    )


def list_packaged_forward_aot_specs(
    *,
    package_root: str | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[ForwardAOTSpec, ...]:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return tuple(
        _forward_spec_from_record(cast(dict[str, Any], record["spec"]))
        for record in _list_packaged_aot_records(
            kind="mixer_tail_rowwise_fwd",
            package_root=root,
            arch_tag=arch_tag,
        )
    )


def list_packaged_backward_aot_specs(
    *,
    package_root: str | Path | None = None,
    arch_tag: str | None = None,
) -> tuple[BackwardAOTSpec, ...]:
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return tuple(
        _backward_spec_from_record(cast(dict[str, Any], record["spec"]))
        for record in _list_packaged_aot_records(
            kind="mixer_tail_rowwise_bwd",
            package_root=root,
            arch_tag=arch_tag,
        )
    )


def try_load_packaged_mixer_tail_rowwise_fwd_function(
    spec: ForwardAOTSpec,
    *,
    package_root: str | Path | None = None,
):
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        kind="mixer_tail_rowwise_fwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
        cache=_PACKAGED_FORWARD_CACHE,
        package_root=root,
    )


def try_load_packaged_mixer_tail_rowwise_bwd_function(
    spec: BackwardAOTSpec,
    *,
    package_root: str | Path | None = None,
):
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        kind="mixer_tail_rowwise_bwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
        cache=_PACKAGED_BACKWARD_CACHE,
        package_root=root,
    )


def clear_packaged_aot_cache() -> None:
    _PACKAGED_FORWARD_CACHE.clear()
    _PACKAGED_BACKWARD_CACHE.clear()


def _make_forward_runtime_artifacts_from_spec(
    spec: ForwardAOTSpec,
) -> ForwardRuntimeArtifacts:
    hidden_dim = int(spec.heads * spec.d_head)
    input_info = TailRowwiseInputInfo(
        batch_size=_REPRESENTATIVE_BATCH_SIZE,
        time_steps=_REPRESENTATIVE_TIME_STEPS,
        heads=spec.heads,
        d_head=spec.d_head,
        has_skip=spec.has_skip,
        device_index=0,
    )
    scan_output = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            spec.heads,
            _REPRESENTATIVE_TIME_STEPS,
            spec.d_head,
        ),
        dtype=spec.scan_dtype,
    )
    gate = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, hidden_dim),
        dtype=spec.gate_dtype,
    )
    out_norm_weight = torch.empty((hidden_dim,), dtype=spec.norm_dtype)
    skip_input = (
        torch.empty_like(scan_output, dtype=spec.skip_dtype)
        if spec.has_skip
        else dummy_skip_input(scan_output.device, spec.scan_dtype)
    )
    d_skip = (
        torch.empty((spec.heads,), dtype=spec.d_skip_dtype)
        if spec.has_skip
        else dummy_d_skip(scan_output.device)
    )
    output = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, hidden_dim),
        dtype=spec.scan_dtype,
    )
    runtime_args = (
        scan_output,
        gate,
        out_norm_weight,
        skip_input,
        d_skip,
        output,
    )
    alignments = _runtime_alignments(runtime_args)
    return ForwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        output=output,
        storage_dtype=spec.scan_dtype,
        alignments=alignments,
        cache_key=(input_info, float(spec.eps), _runtime_signature_key(runtime_args)),
    )


def _make_backward_runtime_artifacts_from_spec(
    spec: BackwardAOTSpec,
) -> BackwardRuntimeArtifacts:
    hidden_dim = int(spec.heads * spec.d_head)
    input_info = TailRowwiseInputInfo(
        batch_size=_REPRESENTATIVE_BATCH_SIZE,
        time_steps=_REPRESENTATIVE_TIME_STEPS,
        heads=spec.heads,
        d_head=spec.d_head,
        has_skip=spec.has_skip,
        device_index=0,
    )
    scan_output = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            spec.heads,
            _REPRESENTATIVE_TIME_STEPS,
            spec.d_head,
        ),
        dtype=spec.scan_dtype,
    )
    gate = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, hidden_dim),
        dtype=spec.gate_dtype,
    )
    out_norm_weight = torch.empty((hidden_dim,), dtype=spec.norm_dtype)
    skip_input = (
        torch.empty_like(scan_output, dtype=spec.skip_dtype)
        if spec.has_skip
        else dummy_skip_input(scan_output.device, spec.scan_dtype)
    )
    d_skip = (
        torch.empty((spec.heads,), dtype=spec.d_skip_dtype)
        if spec.has_skip
        else dummy_d_skip(scan_output.device)
    )
    d_normed = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, _REPRESENTATIVE_TIME_STEPS, hidden_dim),
        dtype=spec.d_normed_dtype,
    )
    d_scan_output = torch.empty_like(scan_output)
    d_gate = torch.empty_like(gate)
    d_skip_input = (
        torch.empty_like(scan_output, dtype=spec.scan_dtype)
        if spec.has_skip
        else dummy_skip_input(scan_output.device, spec.scan_dtype)
    )
    d_d_skip = torch.zeros(
        (spec.heads if spec.has_skip else 1,),
        dtype=torch.float32,
    )
    d_norm_weight_accum = torch.zeros((hidden_dim,), dtype=torch.float32)
    outputs = BackwardOutputs(
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
        outputs.d_scan_output,
        outputs.d_gate,
        outputs.d_skip_input,
        outputs.d_d_skip,
        outputs.d_norm_weight_accum,
    )
    alignments = _runtime_alignments(runtime_args)
    return BackwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        outputs=outputs,
        alignments=alignments,
        cache_key=(input_info, float(spec.eps), _runtime_signature_key(runtime_args)),
    )


def infer_mixer_tail_rowwise_fwd_aot_spec(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    *,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
    eps: float,
    arch_tag: str = "any",
) -> ForwardAOTSpec:
    has_skip = skip_input is not None or d_skip is not None
    input_info = _infer_input_info(scan_output, gate, has_skip=has_skip)
    _validate_rowwise_operands(
        input_info=input_info,
        scan_output=scan_output,
        gate=gate,
        out_norm_weight=out_norm_weight,
        skip_input=skip_input,
        d_skip=d_skip,
    )
    return ForwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        d_head=input_info.d_head,
        has_skip=input_info.has_skip,
        scan_dtype_name=_dtype_name(scan_output.dtype),
        gate_dtype_name=_dtype_name(gate.dtype),
        norm_dtype_name=_dtype_name(out_norm_weight.dtype),
        skip_dtype_name=(
            _dtype_name(skip_input.dtype)
            if skip_input is not None
            else _dtype_name(scan_output.dtype)
        ),
        d_skip_dtype_name=(
            _dtype_name(d_skip.dtype)
            if d_skip is not None
            else _DEFAULT_D_SKIP_DTYPE_NAME
        ),
        eps=float(eps),
    )


def infer_mixer_tail_rowwise_bwd_aot_spec(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    d_normed: torch.Tensor,
    *,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
    eps: float,
    arch_tag: str = "any",
) -> BackwardAOTSpec:
    has_skip = skip_input is not None or d_skip is not None
    input_info = _infer_input_info(scan_output, gate, has_skip=has_skip)
    _validate_rowwise_operands(
        input_info=input_info,
        scan_output=scan_output,
        gate=gate,
        out_norm_weight=out_norm_weight,
        skip_input=skip_input,
        d_skip=d_skip,
        d_normed=d_normed,
    )
    return BackwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        d_head=input_info.d_head,
        has_skip=input_info.has_skip,
        scan_dtype_name=_dtype_name(scan_output.dtype),
        gate_dtype_name=_dtype_name(gate.dtype),
        norm_dtype_name=_dtype_name(out_norm_weight.dtype),
        skip_dtype_name=(
            _dtype_name(skip_input.dtype)
            if skip_input is not None
            else _dtype_name(scan_output.dtype)
        ),
        d_skip_dtype_name=(
            _dtype_name(d_skip.dtype)
            if d_skip is not None
            else _DEFAULT_D_SKIP_DTYPE_NAME
        ),
        d_normed_dtype_name=_dtype_name(d_normed.dtype),
        eps=float(eps),
    )


def _compile_forward_aot(spec: ForwardAOTSpec):
    runtime_artifacts = _make_forward_runtime_artifacts_from_spec(spec)
    compile_artifacts = cast(
        ForwardCompileArtifacts,
        _make_forward_compile_artifacts(runtime_artifacts),
    )
    host_wrapper = _make_forward_host_wrapper(
        compile_artifacts,
        eps=spec.eps,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_backward_aot(spec: BackwardAOTSpec):
    runtime_artifacts = _make_backward_runtime_artifacts_from_spec(spec)
    compile_artifacts = cast(
        BackwardCompileArtifacts,
        _make_backward_compile_artifacts(runtime_artifacts),
    )
    host_wrapper = _make_backward_host_wrapper(
        compile_artifacts,
        eps=spec.eps,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def export_mixer_tail_rowwise_fwd_cute_aot(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    *,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
    eps: float,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_mixer_tail_rowwise_fwd_aot_spec(
        scan_output,
        gate,
        out_norm_weight,
        skip_input=skip_input,
        d_skip=d_skip,
        eps=eps,
        arch_tag=arch_tag,
    )
    compiled = _compile_forward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="mixer_tail_rowwise_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="mixer_tail_rowwise_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_mixer_tail_rowwise_bwd_cute_aot(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    d_normed: torch.Tensor,
    *,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
    eps: float,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_mixer_tail_rowwise_bwd_aot_spec(
        scan_output,
        gate,
        out_norm_weight,
        d_normed,
        skip_input=skip_input,
        d_skip=d_skip,
        eps=eps,
        arch_tag=arch_tag,
    )
    compiled = _compile_backward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="mixer_tail_rowwise_bwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="mixer_tail_rowwise_bwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def build_default_forward_aot_package(
    *,
    package_root: str | Path = _PACKAGED_AOT_ROOT,
    specs: tuple[ForwardAOTSpec, ...] | None = None,
    arch_tags: tuple[str, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    package_root = Path(package_root)
    resolved_specs = default_forward_aot_specs(arch_tags) if specs is None else specs
    if clean:
        shutil.rmtree(package_root / "artifacts", ignore_errors=True)
        shutil.rmtree(package_root / "runtime", ignore_errors=True)
        (package_root / "manifest.json").unlink(missing_ok=True)
    exported_modules: list[ExportedTVMFFIModule] = []
    for spec in resolved_specs:
        compiled = _compile_forward_aot(spec)
        exported = export_tvm_ffi_compiled_module(
            compiled,
            kind="mixer_tail_rowwise_fwd",
            module_id=spec.module_id,
            function_name=spec.module_id,
            package_root=package_root,
            keep_object_file=False,
        )
        register_aot_artifact(
            kind="mixer_tail_rowwise_fwd",
            spec=spec,
            exported=exported,
            package_root=package_root,
        )
        exported_modules.append(exported)
    return tuple(exported_modules)


def build_default_backward_aot_package(
    *,
    package_root: str | Path = _PACKAGED_AOT_ROOT,
    specs: tuple[BackwardAOTSpec, ...] | None = None,
    arch_tags: tuple[str, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    package_root = Path(package_root)
    resolved_specs = default_backward_aot_specs(arch_tags) if specs is None else specs
    if clean:
        shutil.rmtree(package_root / "artifacts", ignore_errors=True)
        shutil.rmtree(package_root / "runtime", ignore_errors=True)
        (package_root / "manifest.json").unlink(missing_ok=True)
    exported_modules: list[ExportedTVMFFIModule] = []
    for spec in resolved_specs:
        compiled = _compile_backward_aot(spec)
        exported = export_tvm_ffi_compiled_module(
            compiled,
            kind="mixer_tail_rowwise_bwd",
            module_id=spec.module_id,
            function_name=spec.module_id,
            package_root=package_root,
            keep_object_file=False,
        )
        register_aot_artifact(
            kind="mixer_tail_rowwise_bwd",
            spec=spec,
            exported=exported,
            package_root=package_root,
        )
        exported_modules.append(exported)
    return tuple(exported_modules)


def build_default_cute_aot_package(
    *,
    package_root: str | Path = _PACKAGED_AOT_ROOT,
    forward_specs: tuple[ForwardAOTSpec, ...] | None = None,
    backward_specs: tuple[BackwardAOTSpec, ...] | None = None,
    arch_tags: tuple[str, ...] | None = None,
    clean: bool = True,
) -> tuple[ExportedTVMFFIModule, ...]:
    package_root = Path(package_root)
    exported_forward = build_default_forward_aot_package(
        package_root=package_root,
        specs=forward_specs,
        arch_tags=arch_tags,
        clean=clean,
    )
    exported_backward = build_default_backward_aot_package(
        package_root=package_root,
        specs=backward_specs,
        arch_tags=arch_tags,
        clean=False,
    )
    return (*exported_forward, *exported_backward)


__all__ = [
    "BackwardAOTSpec",
    "ExportedTVMFFIModule",
    "ForwardAOTSpec",
    "build_default_backward_aot_package",
    "build_default_cute_aot_package",
    "build_default_forward_aot_package",
    "clear_packaged_aot_cache",
    "default_backward_aot_specs",
    "default_forward_aot_specs",
    "export_mixer_tail_rowwise_bwd_cute_aot",
    "export_mixer_tail_rowwise_fwd_cute_aot",
    "infer_mixer_tail_rowwise_bwd_aot_spec",
    "infer_mixer_tail_rowwise_fwd_aot_spec",
    "list_packaged_backward_aot_specs",
    "list_packaged_forward_aot_specs",
    "try_load_packaged_mixer_tail_rowwise_bwd_function",
    "try_load_packaged_mixer_tail_rowwise_fwd_function",
]
