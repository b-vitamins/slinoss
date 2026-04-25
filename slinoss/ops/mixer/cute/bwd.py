"""CuTe backward launcher for the fused mixer-tail rowwise path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import cutlass.pipeline as pipeline

from slinoss.perf import note_cache_event

from .common import (
    TailRowwiseInputInfo,
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
    dummy_d_skip,
    dummy_skip_input,
    launch_tvm_ffi_on_current_stream,
    safe_cast_to_dtype,
    silu,
    silu_grad,
    validate_common_operands,
    warp_reduce_sum,
)

_TVM_FFI_COMPILE_OPTIONS = "--enable-tvm-ffi"
_BWD_CACHE: dict[tuple, object] = {}


@dataclass(frozen=True)
class BackwardOutputs:
    d_scan_output: torch.Tensor
    d_gate: torch.Tensor
    d_skip_input: torch.Tensor
    d_d_skip: torch.Tensor
    d_norm_weight_accum: torch.Tensor


@dataclass(frozen=True)
class BackwardRuntimeArtifacts:
    input_info: TailRowwiseInputInfo
    runtime_args: tuple[torch.Tensor, ...]
    outputs: BackwardOutputs
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class BackwardCompileArtifacts:
    input_info: TailRowwiseInputInfo
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


def _grid_shape(*, total_rows: int, warps_per_block: int) -> tuple[int, int, int]:
    return ((total_rows + warps_per_block - 1) // warps_per_block, 1, 1)


class MixerTailRowwiseBwdFused:
    """Host wrapper launching the fused tail rowwise backward kernel."""

    def __init__(
        self,
        *,
        h_size: int,
        p_size: int,
        eps: float,
        has_skip: bool,
        warps_per_block: int = 8,
    ) -> None:
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.hidden_dim = int(self.h_size * self.p_size)
        self.eps = float(eps)
        self.has_skip = bool(has_skip)
        self.warps_per_block = int(warps_per_block)
        if self.h_size <= 0 or self.p_size <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid mixer-tail backward shape.")
        self.block_size = self.warps_per_block * 32
        self.d_norm_head_smem_stride = ((self.p_size + 3) // 4) * 4
        self.d_norm_head_smem_bytes = (
            self.warps_per_block * self.d_norm_head_smem_stride * 4
        )

    @cute.kernel
    def _rowwise_bwd(
        self,
        mScanOutput: cute.Tensor,
        mGate: cute.Tensor,
        mNormWeight: cute.Tensor,
        mSkipInput: cute.Tensor,
        mDSkip: cute.Tensor,
        mDNormed: cute.Tensor,
        mDScanOutput: cute.Tensor,
        mDGate: cute.Tensor,
        mDSkipInput: cute.Tensor,
        mDDSkip: cute.Tensor,
        mDNormWeight: cute.Tensor,
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
        sDNormWeightHead = smem.allocate_tensor(
            cutlass.Float32,
            _make_layout(
                (self.warps_per_block, self.p_size),
                stride=(self.d_norm_head_smem_stride, 1),
            ),
            16,
        )
        num_p_iters = (self.p_size + 31) // 32
        h_limit = cutlass.Int32(self.h_size)
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
            h = cutlass.Int32(0)
            while cute.elem_less(h, h_limit):  # pyright: ignore[reportArgumentType]
                skip_scale = (
                    cutlass.Float32(mDSkip[h])
                    if self.has_skip
                    else cutlass.Float32(0.0)
                )
                for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                    p = lane + p_iter * 32
                    if p < self.p_size:
                        d = h * self.p_size + p
                        pre = cutlass.Float32(mScanOutput[b, h, t, p])
                        if self.has_skip:
                            pre = (
                                pre
                                + cutlass.Float32(mSkipInput[b, h, t, p]) * skip_scale
                            )
                        hidden = pre * silu(mGate[b, t, d])
                        sum_sq = sum_sq + hidden * hidden
                        g = cutlass.Float32(mDNormed[b, t, d])
                        dot = dot + g * cutlass.Float32(mNormWeight[d]) * hidden
                h = h + 1

            sum_sq = warp_reduce_sum(sum_sq)
            dot = warp_reduce_sum(dot)
            inv_rms = cutlass.Float32(1.0) / cute_math.sqrt(
                sum_sq * inv_hidden_dim + cutlass.Float32(self.eps)
            )
            inv_rms_cubed = inv_rms * inv_rms * inv_rms
            norm_coeff = dot * inv_rms_cubed * inv_hidden_dim

        h = cutlass.Int32(0)
        while cute.elem_less(h, h_limit):  # pyright: ignore[reportArgumentType]
            if row_valid:
                skip_scale = (
                    cutlass.Float32(mDSkip[h])
                    if self.has_skip
                    else cutlass.Float32(0.0)
                )
                head_sum = cutlass.Float32(0.0)
                for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                    p = lane + p_iter * 32
                    if p < self.p_size:
                        d = h * self.p_size + p
                        gate = cutlass.Float32(mGate[b, t, d])
                        skip_value = (
                            cutlass.Float32(mSkipInput[b, h, t, p])
                            if self.has_skip
                            else cutlass.Float32(0.0)
                        )
                        pre = cutlass.Float32(mScanOutput[b, h, t, p])
                        if self.has_skip:
                            pre = pre + skip_value * skip_scale
                        act = silu(gate)
                        hidden = pre * act
                        g = cutlass.Float32(mDNormed[b, t, d])
                        sDNormWeightHead[warp, p] = g * hidden * inv_rms
                        d_hidden = (
                            g * cutlass.Float32(mNormWeight[d]) * inv_rms
                            - hidden * norm_coeff
                        )
                        d_pre = d_hidden * act
                        d_gate = d_hidden * pre * silu_grad(gate)
                        mDScanOutput[b, h, t, p] = safe_cast_to_dtype(
                            d_pre,
                            mDScanOutput.element_type,
                        )
                        mDGate[b, t, d] = safe_cast_to_dtype(
                            d_gate,
                            mDGate.element_type,
                        )
                        if self.has_skip:
                            mDSkipInput[b, h, t, p] = safe_cast_to_dtype(
                                d_pre * skip_scale,
                                mDSkipInput.element_type,
                            )
                            head_sum = head_sum + d_pre * skip_value
                if self.has_skip:
                    head_sum = warp_reduce_sum(head_sum)
                    if lane == 0:
                        cute.arch.atomic_add(_llvm_ptr(mDDSkip.iterator + h), head_sum)
            else:
                for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                    p = lane + p_iter * 32
                    if p < self.p_size:
                        sDNormWeightHead[warp, p] = cutlass.Float32(0.0)
            pipeline.sync()
            if warp == 0:
                for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                    p = lane + p_iter * 32
                    if p < self.p_size:
                        d = h * self.p_size + p
                        block_sum = cutlass.Float32(0.0)
                        for source_warp in cutlass.range_constexpr(  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                            self.warps_per_block
                        ):
                            block_sum = block_sum + sDNormWeightHead[source_warp, p]
                        cute.arch.atomic_add(
                            _llvm_ptr(mDNormWeight.iterator + d),
                            block_sum,
                        )
            pipeline.sync()
            h = h + 1

    @cute.jit
    def __call__(
        self,
        scan_output: cute.Tensor,
        gate: cute.Tensor,
        norm_weight: cute.Tensor,
        skip_input: cute.Tensor,
        d_skip: cute.Tensor,
        d_normed: cute.Tensor,
        d_scan_output: cute.Tensor,
        d_gate: cute.Tensor,
        d_skip_input: cute.Tensor,
        d_d_skip: cute.Tensor,
        d_norm_weight: cute.Tensor,
    ):
        batch = _size(scan_output, mode=[0])
        time_steps = _size(scan_output, mode=[2])
        total_rows = batch * time_steps
        _launchable(
            self._rowwise_bwd(
                scan_output,
                gate,
                norm_weight,
                skip_input,
                d_skip,
                d_normed,
                d_scan_output,
                d_gate,
                d_skip_input,
                d_d_skip,
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
            smem=self.d_norm_head_smem_bytes,
        )


def _make_backward_outputs(
    input_info: TailRowwiseInputInfo,
    *,
    device: torch.device,
    scan_dtype: torch.dtype,
    gate_dtype: torch.dtype,
    skip_dtype: torch.dtype,
) -> BackwardOutputs:
    d_scan_output = torch.empty(
        (
            input_info.batch_size,
            input_info.heads,
            input_info.time_steps,
            input_info.d_head,
        ),
        device=device,
        dtype=scan_dtype,
    )
    d_gate = torch.empty(
        (input_info.batch_size, input_info.time_steps, input_info.hidden_dim),
        device=device,
        dtype=gate_dtype,
    )
    if input_info.has_skip:
        d_skip_input = torch.empty(
            (
                input_info.batch_size,
                input_info.heads,
                input_info.time_steps,
                input_info.d_head,
            ),
            device=device,
            dtype=skip_dtype,
        )
        d_d_skip = torch.zeros((input_info.heads,), device=device, dtype=torch.float32)
    else:
        d_skip_input = dummy_skip_input(device, skip_dtype)
        d_d_skip = torch.zeros((1,), device=device, dtype=torch.float32)
    d_norm_weight_accum = torch.zeros(
        (input_info.hidden_dim,),
        device=device,
        dtype=torch.float32,
    )
    return BackwardOutputs(
        d_scan_output=d_scan_output,
        d_gate=d_gate,
        d_skip_input=d_skip_input,
        d_d_skip=d_d_skip,
        d_norm_weight_accum=d_norm_weight_accum,
    )


def _make_backward_runtime_artifacts(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    d_normed: torch.Tensor,
    *,
    skip_input: torch.Tensor | None,
    d_skip: torch.Tensor | None,
    eps: float,
) -> BackwardRuntimeArtifacts:
    input_info = validate_common_operands(
        scan_output,
        gate,
        out_norm_weight,
        skip_input=skip_input,
        d_skip=d_skip,
        d_normed=d_normed,
    )
    scan_output_c = contiguous_tensor(scan_output)
    gate_c = contiguous_tensor(gate)
    out_norm_weight_c = contiguous_tensor(out_norm_weight)
    d_normed_c = contiguous_tensor(d_normed)
    skip_input_c = (
        dummy_skip_input(scan_output.device, scan_output.dtype)
        if skip_input is None
        else contiguous_tensor(skip_input)
    )
    d_skip_c = (
        dummy_d_skip(scan_output.device)
        if d_skip is None
        else contiguous_tensor(d_skip)
    )
    outputs = _make_backward_outputs(
        input_info,
        device=scan_output.device,
        scan_dtype=scan_output.dtype,
        gate_dtype=gate.dtype,
        skip_dtype=scan_output.dtype,
    )
    runtime_args = (
        scan_output_c,
        gate_c,
        out_norm_weight_c,
        skip_input_c,
        d_skip_c,
        d_normed_c,
        outputs.d_scan_output,
        outputs.d_gate,
        outputs.d_skip_input,
        outputs.d_d_skip,
        outputs.d_norm_weight_accum,
    )
    alignments = _runtime_alignments(runtime_args)
    cache_key = (
        input_info,
        float(eps),
        _runtime_signature_key(runtime_args),
    )
    return BackwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        outputs=outputs,
        alignments=alignments,
        cache_key=cache_key,
    )


def _make_backward_compile_artifacts(
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


def _make_backward_host_wrapper(
    compile_artifacts: BackwardCompileArtifacts,
    *,
    eps: float,
) -> MixerTailRowwiseBwdFused:
    input_info = compile_artifacts.input_info
    return MixerTailRowwiseBwdFused(
        h_size=input_info.heads,
        p_size=input_info.d_head,
        eps=eps,
        has_skip=input_info.has_skip,
    )


def _compile_backward_kernel(
    compile_artifacts: BackwardCompileArtifacts,
    *,
    eps: float,
) -> object:
    host_wrapper = _make_backward_host_wrapper(
        compile_artifacts,
        eps=eps,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _make_mixer_tail_rowwise_bwd_aot_spec(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    arch_tag: str,
    eps: float,
):
    from slinoss._cute_aot import _dtype_name
    from .aot import BackwardAOTSpec

    (
        scan_output,
        gate,
        out_norm_weight,
        skip_input,
        d_skip,
        d_normed,
        *_,
    ) = runtime_artifacts.runtime_args
    input_info = runtime_artifacts.input_info
    return BackwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        d_head=input_info.d_head,
        has_skip=input_info.has_skip,
        scan_dtype_name=_dtype_name(scan_output.dtype),
        gate_dtype_name=_dtype_name(gate.dtype),
        norm_dtype_name=_dtype_name(out_norm_weight.dtype),
        skip_dtype_name=_dtype_name(skip_input.dtype),
        d_skip_dtype_name=_dtype_name(d_skip.dtype),
        d_normed_dtype_name=_dtype_name(d_normed.dtype),
        eps=float(eps),
    )


def _get_compiled_backward_kernel(
    runtime_artifacts: BackwardRuntimeArtifacts,
    compile_artifacts: BackwardCompileArtifacts,
    *,
    device: torch.device,
    eps: float,
) -> object:
    from slinoss._cute_aot import current_cuda_arch_tag
    from .aot import try_load_packaged_mixer_tail_rowwise_bwd_function

    cache_key = compile_artifacts.cache_key
    compiled = _BWD_CACHE.get(cache_key)
    if compiled is not None:
        note_cache_event("cute.mixer.tail.rowwise.bwd.host_compile", hit=True)
        return compiled
    packaged = try_load_packaged_mixer_tail_rowwise_bwd_function(
        _make_mixer_tail_rowwise_bwd_aot_spec(
            runtime_artifacts,
            arch_tag=current_cuda_arch_tag(device),
            eps=eps,
        )
    )
    if packaged is not None:
        note_cache_event("cute.mixer.tail.rowwise.bwd.host_aot", hit=True)
        _BWD_CACHE[cache_key] = packaged
        return packaged
    note_cache_event("cute.mixer.tail.rowwise.bwd.host_aot", hit=False)
    note_cache_event("cute.mixer.tail.rowwise.bwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("backward")
    compiled = _compile_backward_kernel(compile_artifacts, eps=eps)
    _BWD_CACHE[cache_key] = compiled
    return compiled


def _run_backward_kernel(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    eps: float,
) -> BackwardOutputs:
    compile_artifacts = _make_backward_compile_artifacts(runtime_artifacts)
    compiled = cast(
        Callable[..., object],
        _get_compiled_backward_kernel(
            runtime_artifacts,
            compile_artifacts,
            device=runtime_artifacts.outputs.d_scan_output.device,
            eps=eps,
        ),
    )
    launch_tvm_ffi_on_current_stream(compiled, *runtime_artifacts.runtime_args)
    return runtime_artifacts.outputs


def _mixer_tail_rowwise_bwd_cute_prevalidated(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    d_normed: torch.Tensor,
    *,
    skip_input: torch.Tensor | None,
    d_skip: torch.Tensor | None,
    eps: float,
) -> BackwardOutputs:
    runtime_artifacts = _make_backward_runtime_artifacts(
        scan_output,
        gate,
        out_norm_weight,
        d_normed,
        skip_input=skip_input,
        d_skip=d_skip,
        eps=eps,
    )
    return _run_backward_kernel(runtime_artifacts, eps=eps)


__all__ = [
    "BackwardOutputs",
    "MixerTailRowwiseBwdFused",
    "_mixer_tail_rowwise_bwd_cute_prevalidated",
]
