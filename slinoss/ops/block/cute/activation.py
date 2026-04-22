"""CuTe activation kernels for the block FFN training path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import torch

import cutlass
import cutlass.cute as cute

from slinoss.perf import note_cache_event

from .common import (
    ActivationKind,
    FfnActivationInputInfo,
    _is_cuda_graph_capturing,
    _launchable,
    _make_compile_args,
    _raise_cold_capture_error,
    _record_tensors_on_current_stream,
    _runtime_alignments,
    _runtime_signature_key,
    _size,
    contiguous_tensor,
    gelu_tanh,
    gelu_tanh_grad,
    safe_cast_to_dtype,
    silu,
    silu_grad,
    validate_activation_operands,
)

_ACT_FWD_CACHE: dict[tuple, object] = {}
_ACT_BWD_CACHE: dict[tuple, object] = {}
_TVM_FFI_COMPILE_OPTIONS = "--enable-tvm-ffi"


@dataclass(frozen=True)
class ForwardRuntimeArtifacts:
    input_info: FfnActivationInputInfo
    runtime_args: tuple[torch.Tensor, ...]
    output: torch.Tensor
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class ForwardCompileArtifacts:
    input_info: FfnActivationInputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class BackwardRuntimeArtifacts:
    input_info: FfnActivationInputInfo
    runtime_args: tuple[torch.Tensor, ...]
    output: torch.Tensor
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class BackwardCompileArtifacts:
    input_info: FfnActivationInputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


def _grid_shape(*, total_rows: int, warps_per_block: int) -> tuple[int, int, int]:
    return ((total_rows + warps_per_block - 1) // warps_per_block, 1, 1)


class FfnActivationFwdFused:
    """Host wrapper for the live FFN activation forward kernel."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        kind: ActivationKind,
        warps_per_block: int = 8,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.kind = kind
        self.warps_per_block = int(warps_per_block)
        if self.hidden_dim <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid FFN activation forward shape.")
        self.block_size = self.warps_per_block * 32

    @cute.kernel
    def _rowwise_fwd(
        self,
        mProjected: cute.Tensor,
        mHidden: cute.Tensor,
        total_rows_,
        time_steps_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        row = block_x * self.warps_per_block + warp
        row_valid = row < total_rows_
        num_iters = (self.hidden_dim + 31) // 32
        if row_valid:
            b = row // time_steps_
            t = row - b * time_steps_
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    hidden = cutlass.Float32(0.0)
                    if self.kind == "swiglu":
                        value = cutlass.Float32(mProjected[b, t, d])
                        gate = cutlass.Float32(mProjected[b, t, d + self.hidden_dim])
                        hidden = value * silu(gate)
                    else:
                        hidden = gelu_tanh(cutlass.Float32(mProjected[b, t, d]))
                    mHidden[b, t, d] = safe_cast_to_dtype(
                        hidden,
                        mHidden.element_type,
                    )

    @cute.jit
    def __call__(
        self,
        projected: cute.Tensor,
        hidden: cute.Tensor,
    ):
        batch = _size(projected, mode=[0])
        time_steps = _size(projected, mode=[1])
        total_rows = batch * time_steps
        _launchable(
            self._rowwise_fwd(
                projected,
                hidden,
                total_rows,
                time_steps,
            )
        ).launch(
            grid=_grid_shape(
                total_rows=total_rows,
                warps_per_block=self.warps_per_block,
            ),
            block=(self.block_size, 1, 1),
        )


class FfnActivationBwdFused:
    """Host wrapper for the live FFN activation backward kernel."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        kind: ActivationKind,
        warps_per_block: int = 8,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.kind = kind
        self.warps_per_block = int(warps_per_block)
        if self.hidden_dim <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid FFN activation backward shape.")
        self.block_size = self.warps_per_block * 32

    @cute.kernel
    def _rowwise_bwd(
        self,
        mProjected: cute.Tensor,
        mDHidden: cute.Tensor,
        mDProjected: cute.Tensor,
        total_rows_,
        time_steps_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        row = block_x * self.warps_per_block + warp
        row_valid = row < total_rows_
        num_iters = (self.hidden_dim + 31) // 32
        if row_valid:
            b = row // time_steps_
            t = row - b * time_steps_
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    d_hidden = cutlass.Float32(mDHidden[b, t, d])
                    if self.kind == "swiglu":
                        value = cutlass.Float32(mProjected[b, t, d])
                        gate = cutlass.Float32(mProjected[b, t, d + self.hidden_dim])
                        mDProjected[b, t, d] = safe_cast_to_dtype(
                            d_hidden * silu(gate),
                            mDProjected.element_type,
                        )
                        mDProjected[b, t, d + self.hidden_dim] = safe_cast_to_dtype(
                            d_hidden * value * silu_grad(gate),
                            mDProjected.element_type,
                        )
                    else:
                        projected = cutlass.Float32(mProjected[b, t, d])
                        mDProjected[b, t, d] = safe_cast_to_dtype(
                            d_hidden * gelu_tanh_grad(projected),
                            mDProjected.element_type,
                        )

    @cute.jit
    def __call__(
        self,
        projected: cute.Tensor,
        d_hidden: cute.Tensor,
        d_projected: cute.Tensor,
    ):
        batch = _size(projected, mode=[0])
        time_steps = _size(projected, mode=[1])
        total_rows = batch * time_steps
        _launchable(
            self._rowwise_bwd(
                projected,
                d_hidden,
                d_projected,
                total_rows,
                time_steps,
            )
        ).launch(
            grid=_grid_shape(
                total_rows=total_rows,
                warps_per_block=self.warps_per_block,
            ),
            block=(self.block_size, 1, 1),
        )


def _make_forward_runtime_artifacts(
    projected: torch.Tensor,
    *,
    kind: ActivationKind,
) -> ForwardRuntimeArtifacts:
    input_info = validate_activation_operands(projected, None, kind=kind)
    projected_c = contiguous_tensor(projected)
    output = torch.empty(
        (input_info.batch_size, input_info.time_steps, input_info.hidden_dim),
        device=projected_c.device,
        dtype=projected_c.dtype,
    )
    runtime_args = (projected_c, output)
    alignments = _runtime_alignments(runtime_args)
    cache_key = (
        input_info,
        alignments,
        _runtime_signature_key(runtime_args),
    )
    return ForwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        output=output,
        alignments=alignments,
        cache_key=cache_key,
    )


def _make_forward_compile_artifacts(
    runtime_artifacts: ForwardRuntimeArtifacts,
) -> ForwardCompileArtifacts:
    compile_args = _make_compile_args(
        runtime_artifacts.runtime_args,
        alignments=runtime_artifacts.alignments,
    )
    return ForwardCompileArtifacts(
        input_info=runtime_artifacts.input_info,
        compile_args=compile_args,
        alignments=runtime_artifacts.alignments,
        cache_key=runtime_artifacts.cache_key,
    )


def _make_backward_runtime_artifacts(
    projected: torch.Tensor,
    d_hidden: torch.Tensor,
    *,
    kind: ActivationKind,
) -> BackwardRuntimeArtifacts:
    input_info = validate_activation_operands(projected, d_hidden, kind=kind)
    projected_c = contiguous_tensor(projected)
    d_hidden_c = contiguous_tensor(d_hidden)
    output = torch.empty_like(projected_c)
    runtime_args = (projected_c, d_hidden_c, output)
    alignments = _runtime_alignments(runtime_args)
    cache_key = (
        input_info,
        alignments,
        _runtime_signature_key(runtime_args),
    )
    return BackwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        output=output,
        alignments=alignments,
        cache_key=cache_key,
    )


def _make_backward_compile_artifacts(
    runtime_artifacts: BackwardRuntimeArtifacts,
) -> BackwardCompileArtifacts:
    compile_args = _make_compile_args(
        runtime_artifacts.runtime_args,
        alignments=runtime_artifacts.alignments,
    )
    return BackwardCompileArtifacts(
        input_info=runtime_artifacts.input_info,
        compile_args=compile_args,
        alignments=runtime_artifacts.alignments,
        cache_key=runtime_artifacts.cache_key,
    )


def _compile_forward_kernel(
    compile_artifacts: ForwardCompileArtifacts,
) -> object:
    launcher = FfnActivationFwdFused(
        hidden_dim=compile_artifacts.input_info.hidden_dim,
        kind=compile_artifacts.input_info.kind,
    )
    return cute.compile(
        launcher,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _compile_backward_kernel(
    compile_artifacts: BackwardCompileArtifacts,
) -> object:
    launcher = FfnActivationBwdFused(
        hidden_dim=compile_artifacts.input_info.hidden_dim,
        kind=compile_artifacts.input_info.kind,
    )
    return cute.compile(
        launcher,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _make_ffn_activation_fwd_aot_spec(
    runtime_artifacts: ForwardRuntimeArtifacts,
    *,
    arch_tag: str,
) -> Any:
    from slinoss._cute_aot import _dtype_name
    from .aot import ActivationForwardAOTSpec

    projected, _ = runtime_artifacts.runtime_args
    return ActivationForwardAOTSpec(
        arch_tag=arch_tag,
        hidden_dim=runtime_artifacts.input_info.hidden_dim,
        kind=runtime_artifacts.input_info.kind,
        projected_dtype_name=_dtype_name(projected.dtype),
    )


def _make_ffn_activation_bwd_aot_spec(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    arch_tag: str,
) -> Any:
    from slinoss._cute_aot import _dtype_name
    from .aot import ActivationBackwardAOTSpec

    projected, d_hidden, _ = runtime_artifacts.runtime_args
    return ActivationBackwardAOTSpec(
        arch_tag=arch_tag,
        hidden_dim=runtime_artifacts.input_info.hidden_dim,
        kind=runtime_artifacts.input_info.kind,
        projected_dtype_name=_dtype_name(projected.dtype),
        d_hidden_dtype_name=_dtype_name(d_hidden.dtype),
    )


def _get_compiled_forward_kernel(
    runtime_artifacts: ForwardRuntimeArtifacts,
    compile_artifacts: ForwardCompileArtifacts,
    *,
    device: torch.device,
) -> object:
    from slinoss._cute_aot import current_cuda_arch_tag
    from .aot import try_load_packaged_ffn_activation_fwd_function

    compiled = _ACT_FWD_CACHE.get(runtime_artifacts.cache_key)
    if compiled is not None:
        note_cache_event("cute.block.ffn.activation.fwd.host_compile", hit=True)
        return compiled

    packaged = try_load_packaged_ffn_activation_fwd_function(
        _make_ffn_activation_fwd_aot_spec(
            runtime_artifacts,
            arch_tag=current_cuda_arch_tag(device),
        )
    )
    if packaged is not None:
        note_cache_event("cute.block.ffn.activation.fwd.host_aot", hit=True)
        _ACT_FWD_CACHE[runtime_artifacts.cache_key] = packaged
        return packaged

    note_cache_event("cute.block.ffn.activation.fwd.host_aot", hit=False)
    note_cache_event("cute.block.ffn.activation.fwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("activation.fwd")
    compiled = _compile_forward_kernel(compile_artifacts)
    _ACT_FWD_CACHE[runtime_artifacts.cache_key] = compiled
    return compiled


def _get_compiled_backward_kernel(
    runtime_artifacts: BackwardRuntimeArtifacts,
    compile_artifacts: BackwardCompileArtifacts,
    *,
    device: torch.device,
) -> object:
    from slinoss._cute_aot import current_cuda_arch_tag
    from .aot import try_load_packaged_ffn_activation_bwd_function

    compiled = _ACT_BWD_CACHE.get(runtime_artifacts.cache_key)
    if compiled is not None:
        note_cache_event("cute.block.ffn.activation.bwd.host_compile", hit=True)
        return compiled

    packaged = try_load_packaged_ffn_activation_bwd_function(
        _make_ffn_activation_bwd_aot_spec(
            runtime_artifacts,
            arch_tag=current_cuda_arch_tag(device),
        )
    )
    if packaged is not None:
        note_cache_event("cute.block.ffn.activation.bwd.host_aot", hit=True)
        _ACT_BWD_CACHE[runtime_artifacts.cache_key] = packaged
        return packaged

    note_cache_event("cute.block.ffn.activation.bwd.host_aot", hit=False)
    note_cache_event("cute.block.ffn.activation.bwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("activation.bwd")
    compiled = _compile_backward_kernel(compile_artifacts)
    _ACT_BWD_CACHE[runtime_artifacts.cache_key] = compiled
    return compiled


def _ffn_activation_fwd_cute_prevalidated(
    projected: torch.Tensor,
    *,
    kind: ActivationKind,
) -> torch.Tensor:
    runtime_artifacts = _make_forward_runtime_artifacts(projected, kind=kind)
    compile_artifacts = _make_forward_compile_artifacts(runtime_artifacts)
    compiled = _get_compiled_forward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=projected.device,
    )
    _record_tensors_on_current_stream(*runtime_artifacts.runtime_args)
    cast(Callable[..., None], compiled)(*runtime_artifacts.runtime_args)
    return runtime_artifacts.output


def _ffn_activation_bwd_cute_prevalidated(
    projected: torch.Tensor,
    d_hidden: torch.Tensor,
    *,
    kind: ActivationKind,
) -> torch.Tensor:
    runtime_artifacts = _make_backward_runtime_artifacts(
        projected,
        d_hidden,
        kind=kind,
    )
    compile_artifacts = _make_backward_compile_artifacts(runtime_artifacts)
    compiled = _get_compiled_backward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=projected.device,
    )
    _record_tensors_on_current_stream(*runtime_artifacts.runtime_args)
    cast(Callable[..., None], compiled)(*runtime_artifacts.runtime_args)
    return runtime_artifacts.output


__all__ = [
    "_ffn_activation_bwd_cute_prevalidated",
    "_ffn_activation_fwd_cute_prevalidated",
]
