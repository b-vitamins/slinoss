"""CuTe forward launcher for the fused mixer-tail rowwise path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from slinoss.perf import note_cache_event

from .common import (
    TailRowwiseInputInfo,
    _is_cuda_graph_capturing,
    _launchable,
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
    torch_to_cutlass_dtype,
    validate_common_operands,
    warp_reduce_sum,
)

_TVM_FFI_COMPILE_OPTIONS = "--enable-tvm-ffi"
_FWD_CACHE: dict[tuple, object] = {}


@dataclass(frozen=True)
class ForwardRuntimeArtifacts:
    input_info: TailRowwiseInputInfo
    runtime_args: tuple[torch.Tensor, ...]
    output: torch.Tensor
    storage_dtype: torch.dtype
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class ForwardCompileArtifacts:
    input_info: TailRowwiseInputInfo
    compile_args: tuple[object, ...]
    storage_dtype: torch.dtype
    alignments: tuple[int, ...]
    cache_key: tuple


def _grid_shape(*, total_rows: int, warps_per_block: int) -> tuple[int, int, int]:
    return ((total_rows + warps_per_block - 1) // warps_per_block, 1, 1)


class MixerTailRowwiseFwdFused:
    """Host wrapper launching the fused tail rowwise forward kernel."""

    def __init__(
        self,
        *,
        h_size: int,
        p_size: int,
        eps: float,
        has_skip: bool,
        storage_dtype: torch.dtype,
        warps_per_block: int = 8,
    ) -> None:
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.hidden_dim = int(self.h_size * self.p_size)
        self.eps = float(eps)
        self.has_skip = bool(has_skip)
        self.warps_per_block = int(warps_per_block)
        if self.h_size <= 0 or self.p_size <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid mixer-tail forward shape.")
        self.block_size = self.warps_per_block * 32
        self.hidden_storage_type = torch_to_cutlass_dtype(storage_dtype)
        self.hidden_smem_stride = self.hidden_dim
        self.hidden_smem_bytes = (
            self.warps_per_block
            * self.hidden_smem_stride
            * int(torch.empty((), dtype=storage_dtype).element_size())
        )

    @cute.kernel
    def _rowwise_fwd(
        self,
        mScanOutput: cute.Tensor,
        mGate: cute.Tensor,
        mNormWeight: cute.Tensor,
        mSkipInput: cute.Tensor,
        mDSkip: cute.Tensor,
        mNormed: cute.Tensor,
        total_rows_,
        time_steps_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        row = block_x * self.warps_per_block + warp
        smem = cutlass.utils.SmemAllocator()
        sHidden = smem.allocate_tensor(
            self.hidden_storage_type,
            _make_layout(
                (self.warps_per_block, self.hidden_dim),
                stride=(self.hidden_smem_stride, 1),
            ),
            16,
        )
        if row < total_rows_:
            b = row // time_steps_
            t = row - b * time_steps_
            num_p_iters = (self.p_size + 31) // 32
            inv_hidden_dim = cutlass.Float32(1.0 / float(self.hidden_dim))
            h_limit = cutlass.Int32(self.h_size)

            sum_sq = cutlass.Float32(0.0)
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
                        sHidden[warp, d] = safe_cast_to_dtype(
                            hidden,
                            self.hidden_storage_type,
                        )
                        sum_sq = sum_sq + hidden * hidden
                h = h + 1

            sum_sq = warp_reduce_sum(sum_sq)
            inv_rms = cutlass.Float32(1.0) / cute_math.sqrt(
                sum_sq * inv_hidden_dim + cutlass.Float32(self.eps)
            )

            h = cutlass.Int32(0)
            while cute.elem_less(h, h_limit):  # pyright: ignore[reportArgumentType]
                for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                    p = lane + p_iter * 32
                    if p < self.p_size:
                        d = h * self.p_size + p
                        hidden = cutlass.Float32(sHidden[warp, d])
                        out = hidden * inv_rms * cutlass.Float32(mNormWeight[d])
                        mNormed[b, t, d] = safe_cast_to_dtype(out, mNormed.element_type)
                h = h + 1

    @cute.jit
    def __call__(
        self,
        scan_output: cute.Tensor,
        gate: cute.Tensor,
        norm_weight: cute.Tensor,
        skip_input: cute.Tensor,
        d_skip: cute.Tensor,
        normed: cute.Tensor,
    ):
        batch = _size(scan_output, mode=[0])
        time_steps = _size(scan_output, mode=[2])
        total_rows = batch * time_steps
        _launchable(
            self._rowwise_fwd(
                scan_output,
                gate,
                norm_weight,
                skip_input,
                d_skip,
                normed,
                total_rows,
                time_steps,
            )
        ).launch(
            grid=_grid_shape(
                total_rows=total_rows,
                warps_per_block=self.warps_per_block,
            ),
            block=(self.block_size, 1, 1),
            smem=self.hidden_smem_bytes,
        )


def _make_forward_output(
    input_info: TailRowwiseInputInfo, *, device, dtype
) -> torch.Tensor:
    return torch.empty(
        (input_info.batch_size, input_info.time_steps, input_info.hidden_dim),
        device=device,
        dtype=dtype,
    )


def _make_forward_runtime_artifacts(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    *,
    skip_input: torch.Tensor | None,
    d_skip: torch.Tensor | None,
    eps: float,
) -> ForwardRuntimeArtifacts:
    input_info = validate_common_operands(
        scan_output,
        gate,
        out_norm_weight,
        skip_input=skip_input,
        d_skip=d_skip,
    )
    scan_output_c = contiguous_tensor(scan_output)
    gate_c = contiguous_tensor(gate)
    out_norm_weight_c = contiguous_tensor(out_norm_weight)
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
    output = _make_forward_output(
        input_info,
        device=scan_output.device,
        dtype=scan_output.dtype,
    )
    runtime_args = (
        scan_output_c,
        gate_c,
        out_norm_weight_c,
        skip_input_c,
        d_skip_c,
        output,
    )
    alignments = _runtime_alignments(runtime_args)
    cache_key = (
        input_info,
        float(eps),
        _runtime_signature_key(runtime_args),
    )
    return ForwardRuntimeArtifacts(
        input_info=input_info,
        runtime_args=runtime_args,
        output=output,
        storage_dtype=scan_output.dtype,
        alignments=alignments,
        cache_key=cache_key,
    )


def _make_forward_compile_artifacts(
    runtime_artifacts: ForwardRuntimeArtifacts,
) -> ForwardCompileArtifacts:
    return ForwardCompileArtifacts(
        input_info=runtime_artifacts.input_info,
        compile_args=_make_compile_args(
            runtime_artifacts.runtime_args,
            alignments=runtime_artifacts.alignments,
        ),
        storage_dtype=runtime_artifacts.storage_dtype,
        alignments=runtime_artifacts.alignments,
        cache_key=runtime_artifacts.cache_key,
    )


def _make_forward_host_wrapper(
    compile_artifacts: ForwardCompileArtifacts,
    *,
    eps: float,
) -> MixerTailRowwiseFwdFused:
    input_info = compile_artifacts.input_info
    return MixerTailRowwiseFwdFused(
        h_size=input_info.heads,
        p_size=input_info.d_head,
        eps=eps,
        has_skip=input_info.has_skip,
        storage_dtype=compile_artifacts.storage_dtype,
    )


def _compile_forward_kernel(
    compile_artifacts: ForwardCompileArtifacts,
    *,
    eps: float,
) -> object:
    host_wrapper = _make_forward_host_wrapper(
        compile_artifacts,
        eps=eps,
    )
    return cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options=_TVM_FFI_COMPILE_OPTIONS,
    )


def _make_mixer_tail_rowwise_fwd_aot_spec(
    runtime_artifacts: ForwardRuntimeArtifacts,
    *,
    arch_tag: str,
    eps: float,
):
    from slinoss._cute_aot import _dtype_name
    from .aot import ForwardAOTSpec

    scan_output, gate, out_norm_weight, skip_input, d_skip, _ = (
        runtime_artifacts.runtime_args
    )
    input_info = runtime_artifacts.input_info
    return ForwardAOTSpec(
        arch_tag=arch_tag,
        heads=input_info.heads,
        d_head=input_info.d_head,
        has_skip=input_info.has_skip,
        scan_dtype_name=_dtype_name(scan_output.dtype),
        gate_dtype_name=_dtype_name(gate.dtype),
        norm_dtype_name=_dtype_name(out_norm_weight.dtype),
        skip_dtype_name=_dtype_name(skip_input.dtype),
        d_skip_dtype_name=_dtype_name(d_skip.dtype),
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
    from .aot import try_load_packaged_mixer_tail_rowwise_fwd_function

    cache_key = compile_artifacts.cache_key
    compiled = _FWD_CACHE.get(cache_key)
    if compiled is not None:
        note_cache_event("cute.mixer.tail.rowwise.fwd.host_compile", hit=True)
        return compiled
    packaged = try_load_packaged_mixer_tail_rowwise_fwd_function(
        _make_mixer_tail_rowwise_fwd_aot_spec(
            runtime_artifacts,
            arch_tag=current_cuda_arch_tag(device),
            eps=eps,
        )
    )
    if packaged is not None:
        note_cache_event("cute.mixer.tail.rowwise.fwd.host_aot", hit=True)
        _FWD_CACHE[cache_key] = packaged
        return packaged
    note_cache_event("cute.mixer.tail.rowwise.fwd.host_aot", hit=False)
    note_cache_event("cute.mixer.tail.rowwise.fwd.host_compile", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("forward")
    compiled = _compile_forward_kernel(compile_artifacts, eps=eps)
    _FWD_CACHE[cache_key] = compiled
    return compiled


def _run_forward_kernel(
    runtime_artifacts: ForwardRuntimeArtifacts,
    *,
    eps: float,
) -> torch.Tensor:
    compile_artifacts = _make_forward_compile_artifacts(runtime_artifacts)
    compiled = cast(
        Callable[..., object],
        _get_compiled_forward_kernel(
            runtime_artifacts,
            compile_artifacts,
            device=runtime_artifacts.output.device,
            eps=eps,
        ),
    )
    launch_tvm_ffi_on_current_stream(compiled, *runtime_artifacts.runtime_args)
    return runtime_artifacts.output


def _mixer_tail_rowwise_fwd_cute_prevalidated(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    *,
    skip_input: torch.Tensor | None,
    d_skip: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    runtime_artifacts = _make_forward_runtime_artifacts(
        scan_output,
        gate,
        out_norm_weight,
        skip_input=skip_input,
        d_skip=d_skip,
        eps=eps,
    )
    return _run_forward_kernel(runtime_artifacts, eps=eps)


__all__ = ["MixerTailRowwiseFwdFused", "_mixer_tail_rowwise_fwd_cute_prevalidated"]
