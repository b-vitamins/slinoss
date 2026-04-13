"""Ahead-of-time helpers for the CuTe ``scanprep`` stack.

This module keeps the packaged TVM-FFI surface explicit and separate from the
eager/JIT runtime path:

- compile/export helpers for the public forward and backward wrappers
- packaged-artifact discovery and load helpers for release-built modules
- curated default payload specs that match the repo's default training workload

The packaged path stays specialized to the scanprep model configuration
(``heads``, ``d_head``, ``d_state``, dtypes, and scalar parameter ranges) while
remaining dynamic in batch/time via runtime tensor views.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, cast

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

from ..common import COEFF_AUX_FIELDS, SCANPREP_PARAM_DIM

if TYPE_CHECKING:
    from ..kernels import ScanPrepConfig


_PACKAGED_AOT_ROOT = Path(__file__).resolve().parent
_PACKAGED_FORWARD_CACHE: dict[str, object] = {}
_PACKAGED_BACKWARD_CACHE: dict[str, object] = {}

_DEFAULT_AOT_HEADS = 23
_DEFAULT_AOT_D_HEAD = 64
_DEFAULT_AOT_D_STATE = 128
_DEFAULT_AOT_DTYPE_NAMES = ("float16", "bfloat16")
# These defaults intentionally mirror the hot ``SLinOSSMixer`` constructor
# rather than the standalone ``SLinOSSScanPrep`` defaults, so the curated AOT
# payload covers the default nextchar training path without extra JIT churn.
_DEFAULT_SCANPREP_CONFIG_KWARGS = {
    "dt_min": 1.0e-3,
    "dt_max": 1.0e-1,
    "theta_init_min": 0.2,
    "theta_init_max": 1.0,
    "gamma_min": 2.0,
    "gamma_max": 8.0,
    "r_min": 0.9,
    "r_max": 0.9999,
    "eps": 1.0e-8,
}
_REPRESENTATIVE_BATCH_SIZE = 2
_REPRESENTATIVE_TIME_STEPS = 64


def _bool_tag(flag: bool) -> str:
    return "1" if bool(flag) else "0"


def _float_tag(value: float) -> str:
    text = f"{float(value):.8g}"
    text = text.replace("+", "")
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


def _config_tag(config: "ScanPrepConfig") -> str:
    return (
        f"dt{_float_tag(config.dt_min)}_{_float_tag(config.dt_max)}"
        f"__theta{_float_tag(config.theta_init_min)}_{_float_tag(config.theta_init_max)}"
        f"__gamma{_float_tag(config.gamma_min)}_{_float_tag(config.gamma_max)}"
        f"__r{_float_tag(config.r_min)}_{_float_tag(config.r_max)}"
        f"__eps{_float_tag(config.eps)}"
    )


@dataclass(frozen=True)
class ForwardAOTSpec:
    arch_tag: str
    heads: int
    d_head: int
    d_state: int
    value_dtype_name: str
    params_dtype_name: str
    bc_dtype_name: str
    bias_dtype_name: str
    store_coeff_aux: bool
    config: "ScanPrepConfig"

    @property
    def value_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.value_dtype_name)

    @property
    def params_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.params_dtype_name)

    @property
    def bc_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.bc_dtype_name)

    @property
    def bias_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.bias_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "scanprep_fwd"
            f"__arch{self.arch_tag}"
            f"__h{self.heads}_p{self.d_head}_n{self.d_state}"
            f"__val{_dtype_tag(self.value_dtype)}"
            f"__param{_dtype_tag(self.params_dtype)}"
            f"__bc{_dtype_tag(self.bc_dtype)}"
            f"__bias{_dtype_tag(self.bias_dtype)}"
            f"__aux{_bool_tag(self.store_coeff_aux)}"
            f"__{_config_tag(self.config)}"
        )


@dataclass(frozen=True)
class BackwardAOTSpec:
    arch_tag: str
    heads: int
    d_head: int
    d_state: int
    bc_dtype_name: str
    value_grad_dtype_name: str
    params_grad_dtype_name: str
    bias_dtype_name: str
    config: "ScanPrepConfig"

    @property
    def bc_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.bc_dtype_name)

    @property
    def value_grad_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.value_grad_dtype_name)

    @property
    def params_grad_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.params_grad_dtype_name)

    @property
    def bias_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.bias_dtype_name)

    @property
    def module_id(self) -> str:
        return _sanitize_stem(
            "scanprep_bwd"
            f"__arch{self.arch_tag}"
            f"__h{self.heads}_p{self.d_head}_n{self.d_state}"
            f"__bc{_dtype_tag(self.bc_dtype)}"
            f"__dval{_dtype_tag(self.value_grad_dtype)}"
            f"__dparam{_dtype_tag(self.params_grad_dtype)}"
            f"__bias{_dtype_tag(self.bias_dtype)}"
            f"__{_config_tag(self.config)}"
        )


def _shared_bias_dtype_name(*tensors: torch.Tensor) -> str:
    if not tensors:
        raise ValueError("Expected at least one bias tensor.")
    dtype_names = {_dtype_name(tensor.dtype) for tensor in tensors}
    if len(dtype_names) != 1:
        raise ValueError(f"Expected one shared bias dtype. Got {sorted(dtype_names)}.")
    return next(iter(dtype_names))


def _make_scanprep_config() -> "ScanPrepConfig":
    from ..kernels import _make_scanprep_config

    return _make_scanprep_config(**_DEFAULT_SCANPREP_CONFIG_KWARGS)


@lru_cache(maxsize=8)
def default_forward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[ForwardAOTSpec, ...]:
    resolved_arch_tags = _resolve_cute_aot_arch_tags(arch_tags)
    config = _make_scanprep_config()
    return tuple(
        ForwardAOTSpec(
            arch_tag=arch_tag,
            heads=_DEFAULT_AOT_HEADS,
            d_head=_DEFAULT_AOT_D_HEAD,
            d_state=_DEFAULT_AOT_D_STATE,
            value_dtype_name=dtype_name,
            params_dtype_name=dtype_name,
            bc_dtype_name=dtype_name,
            bias_dtype_name=dtype_name,
            store_coeff_aux=store_coeff_aux,
            config=config,
        )
        for arch_tag in resolved_arch_tags
        for dtype_name in _DEFAULT_AOT_DTYPE_NAMES
        for store_coeff_aux in (False, True)
    )


@lru_cache(maxsize=8)
def default_backward_aot_specs(
    arch_tags: tuple[str, ...] | None = None,
) -> tuple[BackwardAOTSpec, ...]:
    resolved_arch_tags = _resolve_cute_aot_arch_tags(arch_tags)
    config = _make_scanprep_config()
    return tuple(
        BackwardAOTSpec(
            arch_tag=arch_tag,
            heads=_DEFAULT_AOT_HEADS,
            d_head=_DEFAULT_AOT_D_HEAD,
            d_state=_DEFAULT_AOT_D_STATE,
            bc_dtype_name=dtype_name,
            value_grad_dtype_name=dtype_name,
            params_grad_dtype_name=dtype_name,
            bias_dtype_name=dtype_name,
            config=config,
        )
        for arch_tag in resolved_arch_tags
        for dtype_name in _DEFAULT_AOT_DTYPE_NAMES
    )


def _forward_spec_from_record(record: dict[str, Any]) -> ForwardAOTSpec:
    from ..kernels import ScanPrepConfig

    return ForwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        heads=int(record["heads"]),
        d_head=int(record["d_head"]),
        d_state=int(record["d_state"]),
        value_dtype_name=str(record["value_dtype_name"]),
        params_dtype_name=str(record["params_dtype_name"]),
        bc_dtype_name=str(record["bc_dtype_name"]),
        bias_dtype_name=str(record["bias_dtype_name"]),
        store_coeff_aux=bool(record["store_coeff_aux"]),
        config=ScanPrepConfig(**cast(dict[str, float], record["config"])),
    )


def _backward_spec_from_record(record: dict[str, Any]) -> BackwardAOTSpec:
    from ..kernels import ScanPrepConfig

    return BackwardAOTSpec(
        arch_tag=str(record["arch_tag"]),
        heads=int(record["heads"]),
        d_head=int(record["d_head"]),
        d_state=int(record["d_state"]),
        bc_dtype_name=str(record["bc_dtype_name"]),
        value_grad_dtype_name=str(record["value_grad_dtype_name"]),
        params_grad_dtype_name=str(record["params_grad_dtype_name"]),
        bias_dtype_name=str(record["bias_dtype_name"]),
        config=ScanPrepConfig(**cast(dict[str, float], record["config"])),
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
            kind="scanprep_fwd",
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
            kind="scanprep_bwd",
            package_root=root,
            arch_tag=arch_tag,
        )
    )


def try_load_packaged_scanprep_fwd_function(
    spec: ForwardAOTSpec,
    *,
    package_root: str | Path | None = None,
):
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        kind="scanprep_fwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
        cache=_PACKAGED_FORWARD_CACHE,
        package_root=root,
    )


def try_load_packaged_scanprep_bwd_function(
    spec: BackwardAOTSpec,
    *,
    package_root: str | Path | None = None,
):
    root = _PACKAGED_AOT_ROOT if package_root is None else Path(package_root)
    return _try_load_packaged_compiled_function(
        kind="scanprep_bwd",
        module_id=spec.module_id,
        arch_tag=spec.arch_tag,
        cache=_PACKAGED_BACKWARD_CACHE,
        package_root=root,
    )


def clear_packaged_aot_cache() -> None:
    _PACKAGED_FORWARD_CACHE.clear()
    _PACKAGED_BACKWARD_CACHE.clear()


def _make_forward_runtime_artifacts_from_spec(spec: ForwardAOTSpec):
    from ..kernels import _make_forward_runtime_artifacts

    value = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            _REPRESENTATIVE_TIME_STEPS,
            spec.heads * spec.d_head,
        ),
        dtype=spec.value_dtype,
    )
    params = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            _REPRESENTATIVE_TIME_STEPS,
            spec.heads * SCANPREP_PARAM_DIM,
        ),
        dtype=spec.params_dtype,
    )
    bc = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            _REPRESENTATIVE_TIME_STEPS,
            spec.heads,
            4,
            spec.d_state,
        ),
        dtype=spec.bc_dtype,
    )
    bias = torch.empty((spec.heads,), dtype=spec.bias_dtype)
    return _make_forward_runtime_artifacts(
        value,
        params,
        bc,
        config=spec.config,
        n_heads=spec.heads,
        d_state=spec.d_state,
        d_head=spec.d_head,
        dt_bias=bias,
        gamma_bias=bias.clone(),
        theta_mod_bias=bias.clone(),
        theta_bias=bias.clone(),
        theta_sign=bias.clone(),
        store_coeff_aux=spec.store_coeff_aux,
    )


def _make_backward_runtime_artifacts_from_spec(spec: BackwardAOTSpec):
    from ..kernels import _make_backward_runtime_artifacts

    bc = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            _REPRESENTATIVE_TIME_STEPS,
            spec.heads,
            4,
            spec.d_state,
        ),
        dtype=spec.bc_dtype,
    )
    coeff_aux = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            spec.heads,
            COEFF_AUX_FIELDS,
            _REPRESENTATIVE_TIME_STEPS,
        ),
        dtype=torch.float32,
    )
    dU = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            spec.heads,
            _REPRESENTATIVE_TIME_STEPS,
            spec.d_head,
        ),
        dtype=spec.value_grad_dtype,
    )
    dB = torch.empty(
        (
            _REPRESENTATIVE_BATCH_SIZE,
            spec.heads,
            _REPRESENTATIVE_TIME_STEPS,
            2 * spec.d_state,
        ),
        dtype=spec.bc_dtype,
    )
    dC = torch.empty_like(dB)
    dM = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, spec.heads, _REPRESENTATIVE_TIME_STEPS, 2),
        dtype=torch.float32,
    )
    dK = torch.empty(
        (_REPRESENTATIVE_BATCH_SIZE, spec.heads, _REPRESENTATIVE_TIME_STEPS, 2, 2),
        dtype=torch.float32,
    )
    bias = torch.empty((spec.heads,), dtype=spec.bias_dtype)
    return _make_backward_runtime_artifacts(
        bc,
        coeff_aux,
        config=spec.config,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        n_heads=spec.heads,
        d_head=spec.d_head,
        d_state=spec.d_state,
        value_dtype=spec.value_grad_dtype,
        params_dtype=spec.params_grad_dtype,
        dt_bias=bias,
        theta_bias=bias.clone(),
        theta_sign=bias.clone(),
    )


def infer_scanprep_fwd_aot_spec(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    n_heads: int,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    store_coeff_aux: bool = False,
    arch_tag: str = "any",
) -> ForwardAOTSpec:
    from ..kernels import (
        _make_forward_input_info,
        _make_scanprep_config,
        _validate_forward_operands,
    )

    input_info = _make_forward_input_info(
        value,
        n_heads=n_heads,
        d_head=d_head,
        d_state=d_state,
    )
    _validate_forward_operands(params, bc, input_info=input_info)
    return ForwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        d_head=input_info.d_head,
        d_state=input_info.d_state,
        value_dtype_name=_dtype_name(value.dtype),
        params_dtype_name=_dtype_name(params.dtype),
        bc_dtype_name=_dtype_name(bc.dtype),
        bias_dtype_name=_shared_bias_dtype_name(
            dt_bias,
            gamma_bias,
            theta_mod_bias,
            theta_bias,
            theta_sign,
        ),
        store_coeff_aux=bool(store_coeff_aux),
        config=_make_scanprep_config(
            dt_min=dt_min,
            dt_max=dt_max,
            theta_init_min=theta_init_min,
            theta_init_max=theta_init_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
        ),
    )


def infer_scanprep_bwd_aot_spec(
    *,
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
    n_heads: int,
    d_head: int,
    d_state: int,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    arch_tag: str = "any",
) -> BackwardAOTSpec:
    from ..kernels import (
        _make_backward_input_info,
        _make_scanprep_config,
        _validate_backward_operands,
    )

    input_info = _make_backward_input_info(
        bc,
        n_heads=n_heads,
        d_head=d_head,
        d_state=d_state,
    )
    _validate_backward_operands(
        coeff_aux,
        input_info=input_info,
        dU=None,
        dM=None,
        dK=None,
        dB=None,
        dC=None,
        dt_bias=dt_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    return BackwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        d_head=input_info.d_head,
        d_state=input_info.d_state,
        bc_dtype_name=_dtype_name(bc.dtype),
        value_grad_dtype_name=_dtype_name(value_dtype),
        params_grad_dtype_name=_dtype_name(params_dtype),
        bias_dtype_name=_shared_bias_dtype_name(
            dt_bias,
            theta_bias,
            theta_sign,
        ),
        config=_make_scanprep_config(
            dt_min=dt_min,
            dt_max=dt_max,
            theta_init_min=theta_init_min,
            theta_init_max=theta_init_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
        ),
    )


def _compile_forward_aot(spec: ForwardAOTSpec):
    from ..kernels import (
        _make_forward_compile_artifacts_from_runtime_artifacts,
        _make_scanprep_fwd_host_wrapper,
    )

    runtime_artifacts = _make_forward_runtime_artifacts_from_spec(spec)
    compile_artifacts = _make_forward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    host_wrapper = _make_scanprep_fwd_host_wrapper(
        compile_artifacts=compile_artifacts,
        config=spec.config,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def _compile_backward_aot(spec: BackwardAOTSpec):
    from ..kernels import (
        _make_backward_compile_artifacts_from_runtime_artifacts,
        _make_scanprep_bwd_host_wrapper,
    )

    runtime_artifacts = _make_backward_runtime_artifacts_from_spec(spec)
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    host_wrapper = _make_scanprep_bwd_host_wrapper(
        compile_artifacts=compile_artifacts,
        config=spec.config,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_aot_compile_options(spec.arch_tag),
        no_jit_engine=True,
    )


def export_scanprep_fwd_cute_aot(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    n_heads: int,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    store_coeff_aux: bool = False,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_scanprep_fwd_aot_spec(
        value,
        params,
        bc,
        dt_bias=dt_bias,
        gamma_bias=gamma_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
        n_heads=n_heads,
        d_state=d_state,
        d_head=d_head,
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        store_coeff_aux=store_coeff_aux,
        arch_tag=arch_tag,
    )
    compiled = _compile_forward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="scanprep_fwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="scanprep_fwd",
        spec=spec,
        exported=exported,
        package_root=package_root,
    )
    return exported


def export_scanprep_bwd_cute_aot(
    *,
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
    n_heads: int,
    d_head: int,
    d_state: int,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    arch_tag: str = "any",
    package_root: str | Path,
) -> ExportedTVMFFIModule:
    spec = infer_scanprep_bwd_aot_spec(
        bc=bc,
        coeff_aux=coeff_aux,
        n_heads=n_heads,
        d_head=d_head,
        d_state=d_state,
        value_dtype=value_dtype,
        params_dtype=params_dtype,
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
        arch_tag=arch_tag,
    )
    compiled = _compile_backward_aot(spec)
    exported = export_tvm_ffi_compiled_module(
        compiled,
        kind="scanprep_bwd",
        module_id=spec.module_id,
        function_name=spec.module_id,
        package_root=package_root,
    )
    register_aot_artifact(
        kind="scanprep_bwd",
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
            kind="scanprep_fwd",
            module_id=spec.module_id,
            function_name=spec.module_id,
            package_root=package_root,
            keep_object_file=False,
        )
        register_aot_artifact(
            kind="scanprep_fwd",
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
            kind="scanprep_bwd",
            module_id=spec.module_id,
            function_name=spec.module_id,
            package_root=package_root,
            keep_object_file=False,
        )
        register_aot_artifact(
            kind="scanprep_bwd",
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
    "export_scanprep_bwd_cute_aot",
    "export_scanprep_fwd_cute_aot",
    "infer_scanprep_bwd_aot_spec",
    "infer_scanprep_fwd_aot_spec",
    "list_packaged_backward_aot_specs",
    "list_packaged_forward_aot_specs",
    "try_load_packaged_scanprep_bwd_function",
    "try_load_packaged_scanprep_fwd_function",
]
