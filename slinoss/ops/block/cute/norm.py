"""CuTe RMSNorm kernels for the block FFN training path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import cutlass.pipeline as pipeline

from slinoss.perf import note_cache_event

from .common import (
    FfnNormInputInfo,
    _is_cuda_graph_capturing,
    _launchable,
    _llvm_ptr,
    _make_compile_args,
    _make_layout,
    _raise_cold_capture_error,
    _runtime_alignments,
    _runtime_signature_key,
    _size,
    contiguous_tensor,
    launch_tvm_ffi_on_current_stream,
    safe_cast_to_dtype,
    validate_norm_operands,
    warp_reduce_sum,
)

_NORM_FWD_CACHE: dict[tuple, object] = {}
_NORM_BWD_CACHE: dict[tuple, object] = {}
_TVM_FFI_COMPILE_OPTIONS = "--enable-tvm-ffi"


@dataclass(frozen=True)
class ForwardRuntimeArtifacts:
    input_info: FfnNormInputInfo
    runtime_args: tuple[torch.Tensor, ...]
    output: torch.Tensor
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class ForwardCompileArtifacts:
    input_info: FfnNormInputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class BackwardOutputs:
    d_input: torch.Tensor
    d_weight: torch.Tensor


@dataclass(frozen=True)
class BackwardRuntimeArtifacts:
    input_info: FfnNormInputInfo
    runtime_args: tuple[torch.Tensor, ...]
    outputs: BackwardOutputs
    d_weight_accum: torch.Tensor
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class BackwardCompileArtifacts:
    input_info: FfnNormInputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


def _grid_shape(*, total_rows: int, warps_per_block: int) -> tuple[int, int, int]:
    return ((total_rows + warps_per_block - 1) // warps_per_block, 1, 1)


class FfnRmsNormFwdFused:
    """Host wrapper for the FFN RMSNorm forward kernel."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        eps: float,
        warps_per_block: int = 8,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.warps_per_block = int(warps_per_block)
        if self.hidden_dim <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid FFN RMSNorm forward shape.")
        self.block_size = self.warps_per_block * 32

    @cute.kernel
    def _rowwise_fwd(
        self,
        mInput: cute.Tensor,
        mWeight: cute.Tensor,
        mOutput: cute.Tensor,
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
        b = cutlass.Int32(0)
        t = cutlass.Int32(0)
        sum_sq = cutlass.Float32(0.0)
        inv_hidden_dim = cutlass.Float32(1.0 / float(self.hidden_dim))
        inv_rms = cutlass.Float32(0.0)

        if row_valid:
            b = row // time_steps_
            t = row - b * time_steps_
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    x = cutlass.Float32(mInput[b, t, d])
                    sum_sq = sum_sq + x * x
            sum_sq = warp_reduce_sum(sum_sq)
            inv_rms = cutlass.Float32(1.0) / cute_math.sqrt(
                sum_sq * inv_hidden_dim + cutlass.Float32(self.eps)
            )
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    x = cutlass.Float32(mInput[b, t, d])
                    w = cutlass.Float32(mWeight[d])
                    mOutput[b, t, d] = safe_cast_to_dtype(
                        x * inv_rms * w,
                        mOutput.element_type,
                    )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        norm_weight: cute.Tensor,
        out: cute.Tensor,
    ):
        batch = _size(x, mode=[0])
        time_steps = _size(x, mode=[1])
        total_rows = batch * time_steps
        _launchable(
            self._rowwise_fwd(
                x,
                norm_weight,
                out,
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


class FfnRmsNormBwdFused:
    """Host wrapper for the FFN RMSNorm backward kernel."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        eps: float,
        warps_per_block: int = 8,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.warps_per_block = int(warps_per_block)
        if self.hidden_dim <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid FFN RMSNorm backward shape.")
        self.block_size = self.warps_per_block * 32
        self.d_weight_smem_stride = ((self.hidden_dim + 3) // 4) * 4
        self.d_weight_smem_bytes = self.warps_per_block * self.d_weight_smem_stride * 4

    @cute.kernel
    def _rowwise_bwd(
        self,
        mInput: cute.Tensor,
        mWeight: cute.Tensor,
        mDOutput: cute.Tensor,
        mDInput: cute.Tensor,
        mDWeightAccum: cute.Tensor,
        total_rows_,
        time_steps_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        row = block_x * self.warps_per_block + warp
        row_valid = row < total_rows_
        smem = cutlass.utils.SmemAllocator()
        sDWeight = smem.allocate_tensor(
            cutlass.Float32,
            _make_layout(
                (self.warps_per_block, self.hidden_dim),
                stride=(self.d_weight_smem_stride, 1),
            ),
            16,
        )
        num_iters = (self.hidden_dim + 31) // 32
        b = cutlass.Int32(0)
        t = cutlass.Int32(0)
        sum_sq = cutlass.Float32(0.0)
        dot = cutlass.Float32(0.0)
        inv_hidden_dim = cutlass.Float32(1.0 / float(self.hidden_dim))
        inv_rms = cutlass.Float32(0.0)
        norm_coeff = cutlass.Float32(0.0)

        if row_valid:
            b = row // time_steps_
            t = row - b * time_steps_
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    x = cutlass.Float32(mInput[b, t, d])
                    g = cutlass.Float32(mDOutput[b, t, d])
                    w = cutlass.Float32(mWeight[d])
                    sum_sq = sum_sq + x * x
                    dot = dot + g * w * x
            sum_sq = warp_reduce_sum(sum_sq)
            dot = warp_reduce_sum(dot)
            inv_rms = cutlass.Float32(1.0) / cute_math.sqrt(
                sum_sq * inv_hidden_dim + cutlass.Float32(self.eps)
            )
            norm_coeff = dot * inv_rms * inv_rms * inv_rms * inv_hidden_dim

        if row_valid:
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    x = cutlass.Float32(mInput[b, t, d])
                    g = cutlass.Float32(mDOutput[b, t, d])
                    w = cutlass.Float32(mWeight[d])
                    d_input = g * w * inv_rms - x * norm_coeff
                    mDInput[b, t, d] = safe_cast_to_dtype(
                        d_input,
                        mDInput.element_type,
                    )
                    sDWeight[warp, d] = g * x * inv_rms
        else:
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    sDWeight[warp, d] = cutlass.Float32(0.0)
        pipeline.sync()

        if warp == 0:
            for d_iter in cutlass.range_constexpr(num_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                d = lane + d_iter * 32
                if d < self.hidden_dim:
                    block_sum = cutlass.Float32(0.0)
                    for source_warp in cutlass.range_constexpr(self.warps_per_block):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                        block_sum = block_sum + sDWeight[source_warp, d]
                    cute.arch.atomic_add(
                        _llvm_ptr(mDWeightAccum.iterator + d),
                        block_sum,
                    )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        norm_weight: cute.Tensor,
        d_out: cute.Tensor,
        d_x: cute.Tensor,
        d_norm_weight: cute.Tensor,
    ):
        batch = _size(x, mode=[0])
        time_steps = _size(x, mode=[1])
        total_rows = batch * time_steps
        _launchable(
            self._rowwise_bwd(
                x,
                norm_weight,
                d_out,
                d_x,
                d_norm_weight,
                total_rows,
                time_steps,
            )
        ).launch(
            grid=_grid_shape(
                total_rows=total_rows,
                warps_per_block=self.warps_per_block,
            ),
            block=(self.block_size, 1, 1),
            smem=self.d_weight_smem_bytes,
        )


def _make_forward_runtime_artifacts(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
) -> ForwardRuntimeArtifacts:
    input_info = validate_norm_operands(residual, norm_weight)
    residual_c = contiguous_tensor(residual)
    norm_weight_c = contiguous_tensor(norm_weight)
    output = torch.empty_like(residual_c)
    runtime_args = (residual_c, norm_weight_c, output)
    alignments = _runtime_alignments(runtime_args)
    cache_key = (
        input_info,
        float(eps),
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
    *,
    eps: float,
) -> ForwardCompileArtifacts:
    input_info = runtime_artifacts.input_info
    compile_args = _make_compile_args(
        runtime_artifacts.runtime_args,
        alignments=runtime_artifacts.alignments,
    )
    return ForwardCompileArtifacts(
        input_info=input_info,
        compile_args=compile_args,
        alignments=runtime_artifacts.alignments,
        cache_key=runtime_artifacts.cache_key,
    )


def _make_backward_runtime_artifacts(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    d_output: torch.Tensor,
    *,
    eps: float,
) -> BackwardRuntimeArtifacts:
    input_info = validate_norm_operands(residual, norm_weight)
    if tuple(map(int, d_output.shape)) != tuple(map(int, residual.shape)):
        raise ValueError(
            "d_output must match residual shape; got "
            f"{tuple(d_output.shape)} expected {tuple(residual.shape)}."
        )
    residual_c = contiguous_tensor(residual)
    norm_weight_c = contiguous_tensor(norm_weight)
    d_output_c = contiguous_tensor(d_output)
    d_input = torch.empty_like(residual_c)
    d_weight = torch.empty_like(norm_weight_c)
    d_weight_accum = torch.zeros(
        (input_info.hidden_dim,),
        device=norm_weight_c.device,
        dtype=torch.float32,
    )
    outputs = BackwardOutputs(
        d_input=d_input,
        d_weight=d_weight,
    )
    runtime_args = (
        residual_c,
        norm_weight_c,
        d_output_c,
        outputs.d_input,
        d_weight_accum,
    )
    alignments = _runtime_alignments(runtime_args)
    cache_key = (
        input_info,
        float(eps),
        alignments,
        _runtime_signature_key(runtime_args),
    )
    return BackwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        outputs=outputs,
        d_weight_accum=d_weight_accum,
        alignments=alignments,
        cache_key=cache_key,
    )


def _make_backward_compile_artifacts(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    eps: float,
) -> BackwardCompileArtifacts:
    input_info = runtime_artifacts.input_info
    compile_args = _make_compile_args(
        runtime_artifacts.runtime_args,
        alignments=runtime_artifacts.alignments,
    )
    return BackwardCompileArtifacts(
        input_info=input_info,
        compile_args=compile_args,
        alignments=runtime_artifacts.alignments,
        cache_key=runtime_artifacts.cache_key,
    )


def _compile_forward_kernel(
    compile_artifacts: ForwardCompileArtifacts,
    *,
    eps: float,
) -> object:
    launcher = FfnRmsNormFwdFused(
        hidden_dim=compile_artifacts.input_info.hidden_dim,
        eps=eps,
    )
    return cute.compile(
        launcher,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _compile_backward_kernel(
    compile_artifacts: BackwardCompileArtifacts,
    *,
    eps: float,
) -> object:
    launcher = FfnRmsNormBwdFused(
        hidden_dim=compile_artifacts.input_info.hidden_dim,
        eps=eps,
    )
    return cute.compile(
        launcher,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _make_ffn_norm_fwd_aot_spec(
    runtime_artifacts: ForwardRuntimeArtifacts,
    *,
    eps: float,
    arch_tag: str,
) -> Any:
    from slinoss._cute_aot import _dtype_name
    from .aot import NormForwardAOTSpec

    residual, norm_weight, _ = runtime_artifacts.runtime_args
    return NormForwardAOTSpec(
        arch_tag=arch_tag,
        d_model=runtime_artifacts.input_info.hidden_dim,
        residual_dtype_name=_dtype_name(residual.dtype),
        norm_weight_dtype_name=_dtype_name(norm_weight.dtype),
        eps=float(eps),
    )


def _make_ffn_norm_bwd_aot_spec(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    eps: float,
    arch_tag: str,
) -> Any:
    from slinoss._cute_aot import _dtype_name
    from .aot import NormBackwardAOTSpec

    residual, norm_weight, d_output, _, _ = runtime_artifacts.runtime_args
    return NormBackwardAOTSpec(
        arch_tag=arch_tag,
        d_model=runtime_artifacts.input_info.hidden_dim,
        residual_dtype_name=_dtype_name(residual.dtype),
        norm_weight_dtype_name=_dtype_name(norm_weight.dtype),
        d_output_dtype_name=_dtype_name(d_output.dtype),
        eps=float(eps),
    )


def _get_compiled_forward_kernel(
    runtime_artifacts: ForwardRuntimeArtifacts,
    compile_artifacts: ForwardCompileArtifacts,
    *,
    device: torch.device,
    eps: float,
) -> object:
    from slinoss._cute_aot import current_cuda_arch_tag
    from .aot import try_load_packaged_ffn_norm_fwd_function

    compiled = _NORM_FWD_CACHE.get(runtime_artifacts.cache_key)
    if compiled is not None:
        note_cache_event("cute.block.ffn.norm.fwd.host_compile", hit=True)
        return compiled

    packaged = try_load_packaged_ffn_norm_fwd_function(
        _make_ffn_norm_fwd_aot_spec(
            runtime_artifacts,
            eps=eps,
            arch_tag=current_cuda_arch_tag(device),
        )
    )
    if packaged is not None:
        note_cache_event("cute.block.ffn.norm.fwd.host_aot", hit=True)
        _NORM_FWD_CACHE[runtime_artifacts.cache_key] = packaged
        return packaged

    note_cache_event("cute.block.ffn.norm.fwd.host_aot", hit=False)
    note_cache_event("cute.block.ffn.norm.fwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("norm.fwd")
    compiled = _compile_forward_kernel(compile_artifacts, eps=eps)
    _NORM_FWD_CACHE[runtime_artifacts.cache_key] = compiled
    return compiled


def _get_compiled_backward_kernel(
    runtime_artifacts: BackwardRuntimeArtifacts,
    compile_artifacts: BackwardCompileArtifacts,
    *,
    device: torch.device,
    eps: float,
) -> object:
    from slinoss._cute_aot import current_cuda_arch_tag
    from .aot import try_load_packaged_ffn_norm_bwd_function

    compiled = _NORM_BWD_CACHE.get(runtime_artifacts.cache_key)
    if compiled is not None:
        note_cache_event("cute.block.ffn.norm.bwd.host_compile", hit=True)
        return compiled

    packaged = try_load_packaged_ffn_norm_bwd_function(
        _make_ffn_norm_bwd_aot_spec(
            runtime_artifacts,
            eps=eps,
            arch_tag=current_cuda_arch_tag(device),
        )
    )
    if packaged is not None:
        note_cache_event("cute.block.ffn.norm.bwd.host_aot", hit=True)
        _NORM_BWD_CACHE[runtime_artifacts.cache_key] = packaged
        return packaged

    note_cache_event("cute.block.ffn.norm.bwd.host_aot", hit=False)
    note_cache_event("cute.block.ffn.norm.bwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("norm.bwd")
    compiled = _compile_backward_kernel(compile_artifacts, eps=eps)
    _NORM_BWD_CACHE[runtime_artifacts.cache_key] = compiled
    return compiled


def _ffn_rmsnorm_fwd_cute_prevalidated(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    runtime_artifacts = _make_forward_runtime_artifacts(
        residual,
        norm_weight,
        eps=eps,
    )
    compile_artifacts = _make_forward_compile_artifacts(runtime_artifacts, eps=eps)
    compiled = _get_compiled_forward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=residual.device,
        eps=eps,
    )
    launch_tvm_ffi_on_current_stream(
        cast(Callable[..., None], compiled),
        *runtime_artifacts.runtime_args,
    )
    return runtime_artifacts.output


def _ffn_rmsnorm_bwd_cute_prevalidated(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    d_output: torch.Tensor,
    *,
    eps: float,
) -> BackwardOutputs:
    runtime_artifacts = _make_backward_runtime_artifacts(
        residual,
        norm_weight,
        d_output,
        eps=eps,
    )
    compile_artifacts = _make_backward_compile_artifacts(runtime_artifacts, eps=eps)
    compiled = _get_compiled_backward_kernel(
        runtime_artifacts,
        compile_artifacts,
        device=residual.device,
        eps=eps,
    )
    launch_tvm_ffi_on_current_stream(
        cast(Callable[..., None], compiled),
        *runtime_artifacts.runtime_args,
    )
    if runtime_artifacts.outputs.d_weight.dtype == torch.float32:
        runtime_artifacts.outputs.d_weight.copy_(runtime_artifacts.d_weight_accum)
    else:
        runtime_artifacts.outputs.d_weight.copy_(
            runtime_artifacts.d_weight_accum.to(
                dtype=runtime_artifacts.outputs.d_weight.dtype
            )
        )
    return runtime_artifacts.outputs


__all__ = [
    "BackwardOutputs",
    "_ffn_rmsnorm_bwd_cute_prevalidated",
    "_ffn_rmsnorm_fwd_cute_prevalidated",
]
