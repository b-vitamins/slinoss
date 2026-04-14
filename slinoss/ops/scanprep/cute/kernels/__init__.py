"""CuTe scanprep host stack and kernel surface."""

from dataclasses import astuple, dataclass
from typing import Callable, cast

import cutlass.cute as cute
import torch

from slinoss.perf import note_cache_event

from ..common import (
    COEFF_AUX_FIELDS,
    SCANPREP_PARAM_DIM,
    contiguous_tensor,
    make_fake_tensor_arg,
    tensor_compile_signature,
)
from .bwd import ScanPrepBwdFused
from .fwd import ScanPrepFwdFused


_SCANPREP_FWD_CACHE: dict[tuple, object] = {}
_SCANPREP_BWD_CACHE: dict[tuple, object] = {}
_SCANPREP_DUMMY_COEFF_AUX_CACHE: dict[tuple, torch.Tensor] = {}
_SCANPREP_DUMMY_CACHE_LIMIT = 8
_TVM_FFI_COMPILE_OPTIONS = "--enable-tvm-ffi"


@dataclass(frozen=True)
class ScanPrepConfig:
    dt_min: float
    dt_max: float
    theta_init_min: float
    theta_init_max: float
    gamma_min: float
    gamma_max: float
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
    coeff_aux: torch.Tensor


@dataclass(frozen=True)
class ForwardRuntimeArtifacts:
    input_info: InputInfo
    runtime_args: tuple[torch.Tensor, ...]
    outputs: ForwardOutputs
    alignments: tuple[int, ...]
    cache_key: tuple
    store_coeff_aux: bool


@dataclass(frozen=True)
class ForwardCompileArtifacts:
    input_info: InputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple
    store_coeff_aux: bool


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


def _record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
    if not torch.cuda.is_available():
        return
    stream = torch.cuda.current_stream()
    for tensor in tensors:
        if tensor is not None and tensor.device.type == "cuda":
            tensor.record_stream(stream)


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _raise_cold_capture_error(direction: str, resource: str) -> None:
    raise RuntimeError(
        f"CuTe scanprep {direction} {resource} is cold during CUDA graph capture. "
        f"Warm the same scanprep {direction} spec once outside capture before graph capture."
    )


def _device_cache_key(device: torch.device) -> int:
    return 0 if device.index is None else int(device.index)


def _make_scanprep_config(
    *,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
) -> ScanPrepConfig:
    return ScanPrepConfig(
        dt_min=float(dt_min),
        dt_max=float(dt_max),
        theta_init_min=float(theta_init_min),
        theta_init_max=float(theta_init_max),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        r_min=float(r_min),
        r_max=float(r_max),
        eps=float(eps),
    )


def _runtime_alignments(runtime_args: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    return tuple(tensor_compile_signature(tensor)[1] for tensor in runtime_args)


def _runtime_signature_key(
    runtime_args: tuple[torch.Tensor, ...],
) -> tuple[object, ...]:
    return tuple(
        component
        for tensor in runtime_args
        for component in tensor_compile_signature(tensor)
    )


def _make_compile_args(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    alignments: tuple[int, ...],
) -> tuple[object, ...]:
    return tuple(
        make_fake_tensor_arg(tensor, align=align)
        for tensor, align in zip(runtime_args, alignments, strict=True)
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
    coeff_aux: torch.Tensor,
    *,
    input_info: InputInfo,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    dt_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> None:
    expected_coeff_aux_shape = (
        input_info.batch_size,
        input_info.heads,
        COEFF_AUX_FIELDS,
        input_info.time_steps,
    )
    if tuple(map(int, coeff_aux.shape)) != expected_coeff_aux_shape:
        raise ValueError(
            f"coeff_aux must be {expected_coeff_aux_shape}. Got {tuple(coeff_aux.shape)}."
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
        ("theta_bias", theta_bias),
        ("theta_sign", theta_sign),
    ):
        expected_bias_shape = (input_info.heads,)
        if tuple(map(int, tensor.shape)) != expected_bias_shape:
            raise ValueError(
                f"{name} must be {expected_bias_shape}. Got {tuple(tensor.shape)}."
            )


def _get_dummy_coeff_aux(
    *, input_info: InputInfo, device: torch.device
) -> torch.Tensor:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        input_info.batch_size,
        input_info.heads,
        input_info.time_steps,
    )
    cached = _SCANPREP_DUMMY_COEFF_AUX_CACHE.get(key)
    if cached is not None:
        note_cache_event("cute.scanprep.fwd.dummy_coeff_aux", hit=True)
        return cached
    note_cache_event("cute.scanprep.fwd.dummy_coeff_aux", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("forward", "dummy_coeff_aux cache")
    cached = torch.empty(
        (
            input_info.batch_size,
            input_info.heads,
            COEFF_AUX_FIELDS,
            input_info.time_steps,
        ),
        device=device,
        dtype=torch.float32,
    )
    _cache_set(
        _SCANPREP_DUMMY_COEFF_AUX_CACHE,
        key,
        cached,
        limit=_SCANPREP_DUMMY_CACHE_LIMIT,
    )
    return cached


def _make_forward_outputs(
    *,
    input_info: InputInfo,
    device: torch.device,
    value_dtype: torch.dtype,
    bc_dtype: torch.dtype,
    store_coeff_aux: bool,
) -> ForwardOutputs:
    coeff_aux = (
        torch.empty(
            (
                input_info.batch_size,
                input_info.heads,
                COEFF_AUX_FIELDS,
                input_info.time_steps,
            ),
            device=device,
            dtype=torch.float32,
        )
        if store_coeff_aux
        else _get_dummy_coeff_aux(input_info=input_info, device=device)
    )
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
        coeff_aux=coeff_aux,
    )


def _make_forward_runtime_args(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
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
        gamma_bias,
        theta_mod_bias,
        theta_bias,
        theta_sign,
        outputs.U,
        outputs.B,
        outputs.C,
        outputs.M,
        outputs.K,
        outputs.coeff_aux,
    )


def _make_forward_cache_key(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    input_info: InputInfo,
    config: ScanPrepConfig,
    store_coeff_aux: bool,
) -> tuple:
    return (
        (
            input_info.heads,
            input_info.groups,
            input_info.d_head,
            input_info.d_state,
            bool(store_coeff_aux),
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
    gamma_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    store_coeff_aux: bool,
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
        store_coeff_aux=bool(store_coeff_aux),
    )
    runtime_args = _make_forward_runtime_args(
        value,
        params,
        bc,
        dt_bias=dt_bias,
        gamma_bias=gamma_bias,
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
            store_coeff_aux=bool(store_coeff_aux),
        ),
        store_coeff_aux=bool(store_coeff_aux),
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
        store_coeff_aux=runtime_artifacts.store_coeff_aux,
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
        store_coeff_aux=compile_artifacts.store_coeff_aux,
        dt_min=config.dt_min,
        dt_max=config.dt_max,
        theta_init_min=config.theta_init_min,
        theta_init_max=config.theta_init_max,
        gamma_min=config.gamma_min,
        gamma_max=config.gamma_max,
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
        store_coeff_aux=runtime_artifacts.store_coeff_aux,
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
    compiled(*runtime_artifacts.runtime_args)
    _record_tensors_on_current_stream(*runtime_artifacts.runtime_args)
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    store_coeff_aux: bool,
) -> ForwardOutputs:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
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
        gamma_bias=gamma_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
        store_coeff_aux=bool(store_coeff_aux),
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
    store_coeff_aux: bool = False,
) -> object:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
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
        gamma_bias=gamma_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
        store_coeff_aux=bool(store_coeff_aux),
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
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
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
        gamma_bias=gamma_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
        store_coeff_aux=False,
    )
    return outputs.U, outputs.M, outputs.K, outputs.B, outputs.C


def scanprep_fwd_cute_with_aux(
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
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
]:
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
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
        gamma_bias=gamma_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
        store_coeff_aux=True,
    )
    return outputs.U, outputs.M, outputs.K, outputs.B, outputs.C, outputs.coeff_aux


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
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
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
        contiguous_tensor(coeff_aux),
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
        dt_bias,
        theta_bias,
        theta_sign,
        outputs.value_grad,
        outputs.bc_grad,
        outputs.dparams,
        outputs.bias_grad,
    )


def _make_backward_cache_key(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    input_info: InputInfo,
    bc: torch.Tensor,
    config: ScanPrepConfig,
) -> tuple:
    bc_signature = tensor_compile_signature(contiguous_tensor(bc))
    return (
        (
            input_info.heads,
            input_info.groups,
            input_info.d_head,
            input_info.d_state,
            SCANPREP_PARAM_DIM,
        ),
        input_info.device_index,
        *bc_signature,
        *_runtime_signature_key(runtime_args),
        *astuple(config),
    )


def _make_backward_runtime_artifacts(
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
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
        coeff_aux,
        input_info=input_info,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        dt_bias=dt_bias,
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
        bc,
        coeff_aux,
        input_info=input_info,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        value_dtype=value_dtype,
        outputs=outputs,
        dt_bias=dt_bias,
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
            bc=bc,
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
        gamma_min=config.gamma_min,
        gamma_max=config.gamma_max,
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
    bias_args = runtime_artifacts.runtime_args[6:9]
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
    compiled(*runtime_artifacts.runtime_args)
    _record_tensors_on_current_stream(*runtime_artifacts.runtime_args)
    return runtime_artifacts.outputs


def _scanprep_bwd_cute_prevalidated(
    *,
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> BackwardOutputs:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
    )
    runtime_artifacts = _make_backward_runtime_artifacts(
        bc,
        coeff_aux,
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
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    return _run_scanprep_bwd_cute(
        runtime_artifacts,
        config=config,
    )


def compile_scanprep_bwd_cute(
    *,
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> object:
    config = _make_scanprep_config(
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
    )
    runtime_artifacts = _make_backward_runtime_artifacts(
        bc,
        coeff_aux,
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
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
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
        bc=bc,
        coeff_aux=coeff_aux,
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
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
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
    "scanprep_fwd_cute_with_aux",
]
