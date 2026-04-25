"""CuTe scanprep host stack and kernel surface."""

from dataclasses import astuple, dataclass
from typing import Callable, cast

import cutlass.cute as cute
import torch

from slinoss._cute_runtime import launch_tvm_ffi_on_current_stream
from slinoss.ops._cute_common import (
    _device_cache_key,
    _is_cuda_graph_capturing,
    _make_compile_args,
    _runtime_alignments,
    _runtime_signature_key,
)
from slinoss.perf import note_cache_event

from ..common import (
    SCANPREP_PARAM_DIM,
    contiguous_tensor,
)
from .bwd import ScanPrepBwdFused
from .fwd import ScanPrepFwdFused


_SCANPREP_FWD_CACHE: dict[tuple, object] = {}
_SCANPREP_BWD_CACHE: dict[tuple, object] = {}
_TVM_FFI_COMPILE_OPTIONS = "--enable-tvm-ffi"


@dataclass(frozen=True)
class ScanPrepConfig:
    dt_min: float
    dt_max: float
    theta_init_min: float
    theta_init_max: float
    theta_mod_scale: float
    alpha_min: float
    alpha_max: float
    r_min: float
    r_max: float
    eps: float


@dataclass(frozen=True)
class InputInfo:
    batch_size: int
    time_steps: int
    heads: int
    groups: int
    d_head: int
    d_state: int
    device_index: int


@dataclass(frozen=True)
class ForwardOutputs:
    U: torch.Tensor
    M: torch.Tensor
    K: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor


@dataclass(frozen=True)
class ForwardRuntimeArtifacts:
    input_info: InputInfo
    runtime_args: tuple[torch.Tensor, ...]
    outputs: ForwardOutputs
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class ForwardCompileArtifacts:
    input_info: InputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class BackwardOutputs:
    value_grad: torch.Tensor
    dparams: torch.Tensor
    bc_grad: torch.Tensor
    bias_grad: torch.Tensor


@dataclass(frozen=True)
class BackwardRuntimeArtifacts:
    input_info: InputInfo
    runtime_args: tuple[torch.Tensor, ...]
    outputs: BackwardOutputs
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class BackwardCompileArtifacts:
    input_info: InputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


def _cache_set(cache: dict, key: tuple, value, *, limit: int) -> None:
    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= int(limit):
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def _raise_cold_capture_error(direction: str, resource: str) -> None:
    raise RuntimeError(
        f"CuTe scanprep {direction} {resource} is cold during CUDA graph capture. "
        f"Warm the same scanprep {direction} spec once outside capture before graph capture."
    )


def _make_scanprep_config(
    *,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    theta_mod_scale: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
) -> ScanPrepConfig:
    return ScanPrepConfig(
        dt_min=float(dt_min),
        dt_max=float(dt_max),
        theta_init_min=float(theta_init_min),
        theta_init_max=float(theta_init_max),
        theta_mod_scale=float(theta_mod_scale),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        r_min=float(r_min),
        r_max=float(r_max),
        eps=float(eps),
    )


def _make_forward_input_info(
    value: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int | None = None,
    d_head: int,
    d_state: int,
) -> InputInfo:
    batch_size, time_steps, width = map(int, value.shape)
    if width != int(n_heads * d_head):
        raise ValueError(f"value width must be {n_heads * d_head}. Got {width}.")
    resolved_bc_groups = int(n_heads if bc_groups is None else bc_groups)
    if resolved_bc_groups <= 0 or n_heads % resolved_bc_groups != 0:
        raise ValueError(
            "n_heads must be divisible by bc_groups. "
            f"Got {n_heads}, {resolved_bc_groups}."
        )
    return InputInfo(
        batch_size=batch_size,
        time_steps=time_steps,
        heads=int(n_heads),
        groups=resolved_bc_groups,
        d_head=int(d_head),
        d_state=int(d_state),
        device_index=_device_cache_key(value.device),
    )


def _make_backward_input_info(
    bc: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int | None = None,
    d_head: int,
    d_state: int,
) -> InputInfo:
    batch_size, time_steps, _, _, _ = map(int, bc.shape)
    resolved_bc_groups = int(n_heads if bc_groups is None else bc_groups)
    if resolved_bc_groups <= 0 or n_heads % resolved_bc_groups != 0:
        raise ValueError(
            "n_heads must be divisible by bc_groups. "
            f"Got {n_heads}, {resolved_bc_groups}."
        )
    return InputInfo(
        batch_size=batch_size,
        time_steps=time_steps,
        heads=int(n_heads),
        groups=resolved_bc_groups,
        d_head=int(d_head),
        d_state=int(d_state),
        device_index=_device_cache_key(bc.device),
    )


def _validate_forward_operands(
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    input_info: InputInfo,
) -> None:
    expected_param_shape = (
        input_info.batch_size,
        input_info.time_steps,
        input_info.heads * SCANPREP_PARAM_DIM,
    )
    if tuple(map(int, params.shape)) != expected_param_shape:
        raise ValueError(
            f"params must be {expected_param_shape}. Got {tuple(params.shape)}."
        )
    expected_bc_shape = (
        input_info.batch_size,
        input_info.time_steps,
        input_info.groups,
        4,
        input_info.d_state,
    )
    if tuple(map(int, bc.shape)) != expected_bc_shape:
        raise ValueError(f"bc must be {expected_bc_shape}. Got {tuple(bc.shape)}.")


def _validate_optional_grad(
    grad: torch.Tensor | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    if grad is not None and tuple(map(int, grad.shape)) != expected_shape:
        raise ValueError(f"{name} must be {expected_shape}. Got {tuple(grad.shape)}.")


def _validate_backward_operands(
    params: torch.Tensor,
    *,
    input_info: InputInfo,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> None:
    expected_param_shape = (
        input_info.batch_size,
        input_info.time_steps,
        input_info.heads * SCANPREP_PARAM_DIM,
    )
    if tuple(map(int, params.shape)) != expected_param_shape:
        raise ValueError(
            f"params must be {expected_param_shape}. Got {tuple(params.shape)}."
        )
    _validate_optional_grad(
        dU,
        name="dU",
        expected_shape=(
            input_info.batch_size,
            input_info.heads,
            input_info.time_steps,
            input_info.d_head,
        ),
    )
    _validate_optional_grad(
        dM,
        name="dM",
        expected_shape=(
            input_info.batch_size,
            input_info.heads,
            input_info.time_steps,
            2,
        ),
    )
    _validate_optional_grad(
        dK,
        name="dK",
        expected_shape=(
            input_info.batch_size,
            input_info.heads,
            input_info.time_steps,
            2,
            2,
        ),
    )
    _validate_optional_grad(
        dB,
        name="dB",
        expected_shape=(
            input_info.batch_size,
            input_info.groups,
            input_info.time_steps,
            2 * input_info.d_state,
        ),
    )
    _validate_optional_grad(
        dC,
        name="dC",
        expected_shape=(
            input_info.batch_size,
            input_info.groups,
            input_info.time_steps,
            2 * input_info.d_state,
        ),
    )
    for name, tensor in (
        ("dt_bias", dt_bias),
        ("alpha_bias", alpha_bias),
        ("theta_mod_bias", theta_mod_bias),
        ("theta_bias", theta_bias),
        ("theta_sign", theta_sign),
    ):
        expected_bias_shape = (input_info.heads,)
        if tuple(map(int, tensor.shape)) != expected_bias_shape:
            raise ValueError(
                f"{name} must be {expected_bias_shape}. Got {tuple(tensor.shape)}."
            )


def _make_forward_outputs(
    *,
    input_info: InputInfo,
    device: torch.device,
    value_dtype: torch.dtype,
    bc_dtype: torch.dtype,
) -> ForwardOutputs:
    return ForwardOutputs(
        U=torch.empty(
            (
                input_info.batch_size,
                input_info.heads,
                input_info.time_steps,
                input_info.d_head,
            ),
            device=device,
            dtype=value_dtype,
        ),
        M=torch.empty(
            (input_info.batch_size, input_info.heads, input_info.time_steps, 2),
            device=device,
            dtype=torch.float32,
        ),
        K=torch.empty(
            (input_info.batch_size, input_info.heads, input_info.time_steps, 2, 2),
            device=device,
            dtype=torch.float32,
        ),
        B=torch.empty(
            (
                input_info.batch_size,
                input_info.groups,
                input_info.time_steps,
                2 * input_info.d_state,
            ),
            device=device,
            dtype=bc_dtype,
        ),
        C=torch.empty(
            (
                input_info.batch_size,
                input_info.groups,
                input_info.time_steps,
                2 * input_info.d_state,
            ),
            device=device,
            dtype=bc_dtype,
        ),
    )


def _make_forward_runtime_args(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    outputs: ForwardOutputs,
) -> tuple[torch.Tensor, ...]:
    return (
        contiguous_tensor(value),
        contiguous_tensor(bc),
        contiguous_tensor(params),
        dt_bias,
        alpha_bias,
        theta_mod_bias,
        theta_bias,
        theta_sign,
        outputs.U,
        outputs.B,
        outputs.C,
        outputs.M,
        outputs.K,
    )


def _make_forward_cache_key(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    input_info: InputInfo,
    config: ScanPrepConfig,
) -> tuple:
    return (
        (
            input_info.heads,
            input_info.groups,
            input_info.d_head,
            input_info.d_state,
        ),
        input_info.device_index,
        *_runtime_signature_key(runtime_args),
        *astuple(config),
    )


def _make_forward_runtime_artifacts(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    config: ScanPrepConfig,
    n_heads: int,
    bc_groups: int | None = None,
    d_state: int,
    d_head: int,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> ForwardRuntimeArtifacts:
    input_info = _make_forward_input_info(
        value,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_head=d_head,
        d_state=d_state,
    )
    _validate_forward_operands(params, bc, input_info=input_info)
    outputs = _make_forward_outputs(
        input_info=input_info,
        device=value.device,
        value_dtype=value.dtype,
        bc_dtype=bc.dtype,
    )
    runtime_args = _make_forward_runtime_args(
        value,
        params,
        bc,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
        outputs=outputs,
    )
    alignments = _runtime_alignments(runtime_args)
    return ForwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        outputs=outputs,
        alignments=alignments,
        cache_key=_make_forward_cache_key(
            runtime_args,
            input_info=input_info,
            config=config,
        ),
    )


def _make_forward_compile_artifacts_from_runtime_artifacts(
    runtime_artifacts: ForwardRuntimeArtifacts,
) -> ForwardCompileArtifacts:
    return ForwardCompileArtifacts(
        input_info=runtime_artifacts.input_info,
        compile_args=_make_compile_args(
            runtime_artifacts.runtime_args,
            alignments=runtime_artifacts.alignments,
        ),
        alignments=runtime_artifacts.alignments,
        cache_key=runtime_artifacts.cache_key,
    )


def _make_scanprep_fwd_host_wrapper(
    *,
    compile_artifacts: ForwardCompileArtifacts,
    config: ScanPrepConfig,
) -> ScanPrepFwdFused:
    input_info = compile_artifacts.input_info
    return ScanPrepFwdFused(
        h_size=input_info.heads,
        g_size=input_info.groups,
        p_size=input_info.d_head,
        n_size=input_info.d_state,
        dt_min=config.dt_min,
        dt_max=config.dt_max,
        theta_init_min=config.theta_init_min,
        theta_init_max=config.theta_init_max,
        theta_mod_scale=config.theta_mod_scale,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        r_min=config.r_min,
        r_max=config.r_max,
        eps=config.eps,
    )


def _compile_scanprep_fwd_kernel(
    compile_artifacts: ForwardCompileArtifacts,
    *,
    config: ScanPrepConfig,
) -> object:
    host_wrapper = _make_scanprep_fwd_host_wrapper(
        compile_artifacts=compile_artifacts,
        config=config,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _make_scanprep_fwd_aot_spec(
    runtime_artifacts: ForwardRuntimeArtifacts,
    *,
    config: ScanPrepConfig,
    arch_tag: str,
):
    from slinoss._cute_aot import _dtype_name
    from ..aot import ForwardAOTSpec

    value_arg, bc_arg, params_arg = runtime_artifacts.runtime_args[:3]
    bias_args = runtime_artifacts.runtime_args[3:8]
    bias_dtype_names = {_dtype_name(tensor.dtype) for tensor in bias_args}
    if len(bias_dtype_names) != 1:
        raise ValueError(
            f"Expected one shared scanprep forward bias dtype. Got {sorted(bias_dtype_names)}."
        )
    input_info = runtime_artifacts.input_info
    return ForwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        bc_groups=input_info.groups,
        d_head=input_info.d_head,
        d_state=input_info.d_state,
        value_dtype_name=_dtype_name(value_arg.dtype),
        params_dtype_name=_dtype_name(params_arg.dtype),
        bc_dtype_name=_dtype_name(bc_arg.dtype),
        bias_dtype_name=next(iter(bias_dtype_names)),
        config=config,
    )


def _get_compiled_scanprep_fwd_kernel(
    runtime_artifacts: ForwardRuntimeArtifacts,
    compile_artifacts: ForwardCompileArtifacts,
    *,
    device: torch.device,
    config: ScanPrepConfig,
) -> object:
    from slinoss._cute_aot import current_cuda_arch_tag
    from ..aot import try_load_packaged_scanprep_fwd_function

    cache_key = compile_artifacts.cache_key
    compiled = _SCANPREP_FWD_CACHE.get(cache_key)
    if compiled is not None:
        note_cache_event("cute.scanprep.fwd.host_compile", hit=True)
        return compiled
    packaged = try_load_packaged_scanprep_fwd_function(
        _make_scanprep_fwd_aot_spec(
            runtime_artifacts,
            config=config,
            arch_tag=current_cuda_arch_tag(device),
        )
    )
    if packaged is not None:
        note_cache_event("cute.scanprep.fwd.host_aot", hit=True)
        _SCANPREP_FWD_CACHE[cache_key] = packaged
        return packaged
    note_cache_event("cute.scanprep.fwd.host_aot", hit=False)
    note_cache_event("cute.scanprep.fwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("forward", "launcher cache")
    compiled = _compile_scanprep_fwd_kernel(
        compile_artifacts,
        config=config,
    )
    _SCANPREP_FWD_CACHE[cache_key] = compiled
    return compiled


def _run_scanprep_fwd_cute(
    runtime_artifacts: ForwardRuntimeArtifacts,
    *,
    config: ScanPrepConfig,
) -> ForwardOutputs:
    compile_artifacts = _make_forward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    compiled = cast(
        Callable[..., object],
        _get_compiled_scanprep_fwd_kernel(
            runtime_artifacts,
            compile_artifacts,
            device=runtime_artifacts.outputs.U.device,
            config=config,
        ),
    )
    launch_tvm_ffi_on_current_stream(compiled, *runtime_artifacts.runtime_args)
    return runtime_artifacts.outputs


def _scanprep_fwd_cute_prevalidated(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int | None,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    theta_mod_scale: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> ForwardOutputs:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        theta_mod_scale=theta_mod_scale,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
    )
    runtime_artifacts = _make_forward_runtime_artifacts(
        value,
        params,
        bc,
        config=config,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_state=d_state,
        d_head=d_head,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    return _run_scanprep_fwd_cute(runtime_artifacts, config=config)


def compile_scanprep_fwd_cute(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int | None = None,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    theta_mod_scale: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> object:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        theta_mod_scale=theta_mod_scale,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
    )
    runtime_artifacts = _make_forward_runtime_artifacts(
        value,
        params,
        bc,
        config=config,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_state=d_state,
        d_head=d_head,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    compile_artifacts = _make_forward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    return _get_compiled_scanprep_fwd_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=value.device,
        config=config,
    )


def scanprep_fwd_cute(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int | None = None,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    theta_mod_scale: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = _scanprep_fwd_cute_prevalidated(
        value,
        params,
        bc,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_state=d_state,
        d_head=d_head,
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        theta_mod_scale=theta_mod_scale,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    return outputs.U, outputs.M, outputs.K, outputs.B, outputs.C


def _materialize_optional_grad(
    grad: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if grad is None:
        return torch.zeros(shape, device=device, dtype=dtype)
    if grad.dtype != dtype:
        grad = grad.to(dtype=dtype)
    return contiguous_tensor(grad)


def _make_backward_outputs(
    *,
    input_info: InputInfo,
    device: torch.device,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    bc_dtype: torch.dtype,
) -> BackwardOutputs:
    bias_grad = torch.zeros((input_info.heads, 4), device=device, dtype=torch.float32)
    return BackwardOutputs(
        value_grad=torch.empty(
            (
                input_info.batch_size,
                input_info.time_steps,
                input_info.heads * input_info.d_head,
            ),
            device=device,
            dtype=value_dtype,
        ),
        dparams=torch.empty(
            (
                input_info.batch_size,
                input_info.time_steps,
                input_info.heads * SCANPREP_PARAM_DIM,
            ),
            device=device,
            dtype=params_dtype,
        ),
        bc_grad=torch.empty(
            (
                input_info.batch_size,
                input_info.time_steps,
                input_info.groups,
                4,
                input_info.d_state,
            ),
            device=device,
            dtype=bc_dtype,
        ),
        bias_grad=bias_grad,
    )


def _make_backward_runtime_args(
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    input_info: InputInfo,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    value_dtype: torch.dtype,
    outputs: BackwardOutputs,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        _materialize_optional_grad(
            dU,
            shape=(
                input_info.batch_size,
                input_info.heads,
                input_info.time_steps,
                input_info.d_head,
            ),
            device=bc.device,
            dtype=value_dtype,
        ),
        contiguous_tensor(bc),
        _materialize_optional_grad(
            dB,
            shape=(
                input_info.batch_size,
                input_info.groups,
                input_info.time_steps,
                2 * input_info.d_state,
            ),
            device=bc.device,
            dtype=bc.dtype,
        ),
        _materialize_optional_grad(
            dC,
            shape=(
                input_info.batch_size,
                input_info.groups,
                input_info.time_steps,
                2 * input_info.d_state,
            ),
            device=bc.device,
            dtype=bc.dtype,
        ),
        contiguous_tensor(params),
        dt_bias,
        alpha_bias,
        theta_mod_bias,
        theta_bias,
        theta_sign,
        _materialize_optional_grad(
            dM,
            shape=(
                input_info.batch_size,
                input_info.heads,
                input_info.time_steps,
                2,
            ),
            device=bc.device,
            dtype=torch.float32,
        ),
        _materialize_optional_grad(
            dK,
            shape=(
                input_info.batch_size,
                input_info.heads,
                input_info.time_steps,
                2,
                2,
            ),
            device=bc.device,
            dtype=torch.float32,
        ),
        outputs.value_grad,
        outputs.bc_grad,
        outputs.dparams,
        outputs.bias_grad,
    )


def _make_backward_cache_key(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    input_info: InputInfo,
    config: ScanPrepConfig,
) -> tuple:
    return (
        (
            input_info.heads,
            input_info.groups,
            input_info.d_head,
            input_info.d_state,
            SCANPREP_PARAM_DIM,
        ),
        input_info.device_index,
        *_runtime_signature_key(runtime_args),
        *astuple(config),
    )


def _make_backward_runtime_artifacts(
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    config: ScanPrepConfig,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    n_heads: int,
    bc_groups: int | None = None,
    d_head: int,
    d_state: int,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> BackwardRuntimeArtifacts:
    input_info = _make_backward_input_info(
        bc,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_head=d_head,
        d_state=d_state,
    )
    _validate_backward_operands(
        params,
        input_info=input_info,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    outputs = _make_backward_outputs(
        input_info=input_info,
        device=bc.device,
        value_dtype=value_dtype,
        params_dtype=params_dtype,
        bc_dtype=bc.dtype,
    )
    runtime_args = _make_backward_runtime_args(
        params,
        bc,
        input_info=input_info,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        value_dtype=value_dtype,
        outputs=outputs,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    alignments = _runtime_alignments(runtime_args)
    return BackwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        outputs=outputs,
        alignments=alignments,
        cache_key=_make_backward_cache_key(
            runtime_args,
            input_info=input_info,
            config=config,
        ),
    )


def _make_backward_compile_artifacts_from_runtime_artifacts(
    runtime_artifacts: BackwardRuntimeArtifacts,
) -> BackwardCompileArtifacts:
    return BackwardCompileArtifacts(
        input_info=runtime_artifacts.input_info,
        compile_args=_make_compile_args(
            runtime_artifacts.runtime_args,
            alignments=runtime_artifacts.alignments,
        ),
        alignments=runtime_artifacts.alignments,
        cache_key=runtime_artifacts.cache_key,
    )


def _make_scanprep_bwd_host_wrapper(
    *,
    compile_artifacts: BackwardCompileArtifacts,
    config: ScanPrepConfig,
) -> ScanPrepBwdFused:
    input_info = compile_artifacts.input_info
    return ScanPrepBwdFused(
        h_size=input_info.heads,
        g_size=input_info.groups,
        p_size=input_info.d_head,
        n_size=input_info.d_state,
        param_dim=SCANPREP_PARAM_DIM,
        dt_min=config.dt_min,
        dt_max=config.dt_max,
        theta_init_min=config.theta_init_min,
        theta_init_max=config.theta_init_max,
        theta_mod_scale=config.theta_mod_scale,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        r_min=config.r_min,
        r_max=config.r_max,
        eps=config.eps,
    )


def _compile_scanprep_bwd_kernel(
    compile_artifacts: BackwardCompileArtifacts,
    *,
    config: ScanPrepConfig,
) -> object:
    host_wrapper = _make_scanprep_bwd_host_wrapper(
        compile_artifacts=compile_artifacts,
        config=config,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _make_scanprep_bwd_aot_spec(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    config: ScanPrepConfig,
    arch_tag: str,
):
    from slinoss._cute_aot import _dtype_name
    from ..aot import BackwardAOTSpec

    input_info = runtime_artifacts.input_info
    bias_args = runtime_artifacts.runtime_args[5:10]
    bias_dtype_names = {_dtype_name(tensor.dtype) for tensor in bias_args}
    if len(bias_dtype_names) != 1:
        raise ValueError(
            f"Expected one shared scanprep backward bias dtype. Got {sorted(bias_dtype_names)}."
        )
    return BackwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        bc_groups=input_info.groups,
        d_head=input_info.d_head,
        d_state=input_info.d_state,
        bc_dtype_name=_dtype_name(runtime_artifacts.outputs.bc_grad.dtype),
        value_grad_dtype_name=_dtype_name(runtime_artifacts.outputs.value_grad.dtype),
        params_grad_dtype_name=_dtype_name(runtime_artifacts.outputs.dparams.dtype),
        bias_dtype_name=next(iter(bias_dtype_names)),
        config=config,
    )


def _get_compiled_scanprep_bwd_kernel(
    runtime_artifacts: BackwardRuntimeArtifacts,
    compile_artifacts: BackwardCompileArtifacts,
    *,
    device: torch.device,
    config: ScanPrepConfig,
) -> object:
    from slinoss._cute_aot import current_cuda_arch_tag
    from ..aot import try_load_packaged_scanprep_bwd_function

    cache_key = compile_artifacts.cache_key
    compiled = _SCANPREP_BWD_CACHE.get(cache_key)
    if compiled is not None:
        note_cache_event("cute.scanprep.bwd.host_compile", hit=True)
        return compiled
    packaged = try_load_packaged_scanprep_bwd_function(
        _make_scanprep_bwd_aot_spec(
            runtime_artifacts,
            config=config,
            arch_tag=current_cuda_arch_tag(device),
        )
    )
    if packaged is not None:
        note_cache_event("cute.scanprep.bwd.host_aot", hit=True)
        _SCANPREP_BWD_CACHE[cache_key] = packaged
        return packaged
    note_cache_event("cute.scanprep.bwd.host_aot", hit=False)
    note_cache_event("cute.scanprep.bwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("backward", "launcher cache")
    compiled = _compile_scanprep_bwd_kernel(
        compile_artifacts,
        config=config,
    )
    _SCANPREP_BWD_CACHE[cache_key] = compiled
    return compiled


def _materialize_backward_public_outputs(
    outputs: BackwardOutputs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bias_grad = outputs.bias_grad
    return (
        outputs.value_grad,
        outputs.dparams,
        outputs.bc_grad,
        bias_grad[:, 0].contiguous(),
        bias_grad[:, 1].contiguous(),
        bias_grad[:, 2].contiguous(),
        bias_grad[:, 3].contiguous(),
    )


def _run_scanprep_bwd_cute(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    config: ScanPrepConfig,
) -> BackwardOutputs:
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    compiled = cast(
        Callable[..., object],
        _get_compiled_scanprep_bwd_kernel(
            runtime_artifacts,
            compile_artifacts,
            device=runtime_artifacts.outputs.value_grad.device,
            config=config,
        ),
    )
    launch_tvm_ffi_on_current_stream(compiled, *runtime_artifacts.runtime_args)
    return runtime_artifacts.outputs


def _scanprep_bwd_cute_prevalidated(
    *,
    params: torch.Tensor,
    bc: torch.Tensor,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    n_heads: int,
    bc_groups: int | None,
    d_head: int,
    d_state: int,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    theta_mod_scale: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> BackwardOutputs:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        theta_mod_scale=theta_mod_scale,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
    )
    runtime_artifacts = _make_backward_runtime_artifacts(
        params,
        bc,
        config=config,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_head=d_head,
        d_state=d_state,
        value_dtype=value_dtype,
        params_dtype=params_dtype,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    return _run_scanprep_bwd_cute(
        runtime_artifacts,
        config=config,
    )


def compile_scanprep_bwd_cute(
    *,
    params: torch.Tensor,
    bc: torch.Tensor,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    n_heads: int,
    bc_groups: int | None = None,
    d_head: int,
    d_state: int,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    theta_mod_scale: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> object:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        theta_mod_scale=theta_mod_scale,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
    )
    runtime_artifacts = _make_backward_runtime_artifacts(
        params,
        bc,
        config=config,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_head=d_head,
        d_state=d_state,
        value_dtype=value_dtype,
        params_dtype=params_dtype,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    return _get_compiled_scanprep_bwd_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=bc.device,
        config=config,
    )


def scanprep_bwd_cute(
    *,
    params: torch.Tensor,
    bc: torch.Tensor,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    n_heads: int,
    bc_groups: int | None = None,
    d_head: int,
    d_state: int,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    theta_mod_scale: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    outputs = _scanprep_bwd_cute_prevalidated(
        params=params,
        bc=bc,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        n_heads=n_heads,
        bc_groups=bc_groups,
        d_head=d_head,
        d_state=d_state,
        value_dtype=value_dtype,
        params_dtype=params_dtype,
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        theta_mod_scale=theta_mod_scale,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    return _materialize_backward_public_outputs(outputs)


__all__ = [
    "ScanPrepBwdFused",
    "ScanPrepFwdFused",
    "compile_scanprep_bwd_cute",
    "compile_scanprep_fwd_cute",
    "scanprep_bwd_cute",
    "scanprep_fwd_cute",
]
