"""CuTe backward kernels for the ``v2x2ssd`` chunk-scan stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import cutlass
import cutlass.cute as cute

from ....common import _tc_input_dtype
from ...fwd.common import _make_fake_tensor_arg

from .db import ChunkScanBwdDBAmpere
from .dcdr import ChunkScanBwdDCDRAmpere
from .dlp import ChunkScanBwdDLPAmpere
from .du import ChunkScanBwdDUAmpere
from .dz0 import ChunkScanBwdDZ0Ampere
from .param_scan import ChunkScanBwdParamScanAmpere


_COMPILED_CACHE: dict[tuple, ChunkScanBwdCompiledKernels] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_PREV_CACHE_LIMIT = 8
TensorSpec = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class ChunkScanBwdCompiledKernels:
    dz0: object
    du: object
    db: object
    dcdr: object
    dlp: object
    param_scan: object


@dataclass(frozen=True)
class ChunkScanBwdStageLaunchers:
    dz0: Callable[[], None]
    du: Callable[[], None]
    db: Callable[[], None]
    dcdr: Callable[[], None]
    dlp: Callable[[], None]
    param_scan: Callable[[], None]


@dataclass(frozen=True)
class ChunkScanBwdOutputs:
    bc_groups: int
    chunk_start_grad: torch.Tensor
    value_grad_chunk: torch.Tensor
    key_grad_chunk: torch.Tensor
    value_boundary_grad: torch.Tensor
    key_boundary_grad: torch.Tensor
    logprefix_grad: torch.Tensor
    query_grad_chunk: torch.Tensor
    rotation_grad: torch.Tensor
    transition_grad: torch.Tensor
    tap_prev_grad: torch.Tensor
    tap_curr_grad: torch.Tensor

    def public_outputs(
        self,
        *,
        T: int,
        value_dtype: torch.dtype,
        key_dtype: torch.dtype,
        query_dtype: torch.dtype,
        return_prev_grads: bool,
    ) -> tuple[torch.Tensor, ...]:
        key_grad_chunk = _reduce_heads_to_bc_groups(
            self.key_grad_chunk,
            bc_groups=self.bc_groups,
        )
        key_boundary_grad = _reduce_heads_to_bc_groups(
            self.key_boundary_grad,
            bc_groups=self.bc_groups,
        )
        query_grad_chunk = _reduce_heads_to_bc_groups(
            self.query_grad_chunk,
            bc_groups=self.bc_groups,
        )
        value_grad = _public_from_chunked_with_boundary_carries(
            self.value_grad_chunk,
            self.value_boundary_grad,
            T=T,
            dtype=value_dtype,
        )
        key_grad = _public_from_chunked_with_boundary_carries(
            key_grad_chunk,
            key_boundary_grad,
            T=T,
            dtype=key_dtype,
        )
        public_outputs = (
            value_grad,
            _public_from_param_scan(self.transition_grad, T=T),
            _public_dk_from_parts(self.tap_prev_grad, self.tap_curr_grad, T=T),
            key_grad,
            _public_from_chunked(query_grad_chunk, T=T, dtype=query_dtype),
            self.chunk_start_grad.to(dtype=torch.float32).contiguous(),
        )
        if not return_prev_grads:
            return public_outputs
        return public_outputs + (
            key_boundary_grad[:, :, 0, :].to(dtype=key_dtype).contiguous(),
            self.value_boundary_grad[:, :, 0, :].to(dtype=value_dtype).contiguous(),
        )


@dataclass(frozen=True)
class PreparedChunkScanBwdLaunch:
    compiled: ChunkScanBwdCompiledKernels
    outputs: ChunkScanBwdOutputs
    launchers: ChunkScanBwdStageLaunchers
    launch: Callable[[], None]


@dataclass(frozen=True)
class ChunkScanBwdDUWorkspace:
    value_grad_chunk: torch.Tensor
    key_grad_scratch: torch.Tensor
    value_boundary_grad: torch.Tensor
    key_boundary_scratch: torch.Tensor
    logprefix_scratch: torch.Tensor
    transition_prev_scratch: torch.Tensor
    transition_curr_scratch: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.value_grad_chunk,
            self.key_grad_scratch,
            self.value_boundary_grad,
            self.key_boundary_scratch,
            self.logprefix_scratch,
            self.transition_prev_scratch,
            self.transition_curr_scratch,
        )


@dataclass(frozen=True)
class ChunkScanBwdDURuntimeArtifacts:
    workspace: ChunkScanBwdDUWorkspace
    runtime_args: tuple[torch.Tensor, ...]
    alignments: tuple[int, ...]


@dataclass(frozen=True)
class ChunkScanBwdParamScanRuntimeArtifacts:
    logprefix_input: torch.Tensor
    transition_prev_input: torch.Tensor
    transition_curr_input: torch.Tensor
    rotation_input: torch.Tensor
    transition_output: torch.Tensor
    tap_prev_output: torch.Tensor
    tap_curr_output: torch.Tensor
    transition_view: torch.Tensor
    tap_prev_view: torch.Tensor
    tap_curr_view: torch.Tensor

    def runtime_args_for(
        self,
        mM: torch.Tensor,
        mK: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return (
            mM,
            mK,
            self.logprefix_input,
            self.transition_prev_input,
            self.transition_curr_input,
            self.rotation_input,
            self.transition_output,
            self.tap_prev_output,
            self.tap_curr_output,
        )

    def tensor_specs_for(
        self,
        mM: torch.Tensor,
        mK: torch.Tensor,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return tuple(
            _make_tensor_spec_from_tensor(tensor)
            for tensor in self.runtime_args_for(mM, mK)
        )

    def alignments_for(
        self,
        mM: torch.Tensor,
        mK: torch.Tensor,
    ) -> tuple[int, ...]:
        return tuple(_assumed_align(tensor) for tensor in self.runtime_args_for(mM, mK))

    def compile_args_for(
        self,
        mM: torch.Tensor,
        mK: torch.Tensor,
    ) -> tuple[object, ...]:
        runtime_args = self.runtime_args_for(mM, mK)
        alignments = self.alignments_for(mM, mK)
        return tuple(
            _make_fake_tensor_arg(tensor, align=align)
            for tensor, align in zip(runtime_args, alignments, strict=True)
        )

    @property
    def keepalive(self) -> tuple[torch.Tensor, ...]:
        return (
            self.logprefix_input,
            self.transition_prev_input,
            self.transition_curr_input,
            self.rotation_input,
            self.transition_output,
            self.tap_prev_output,
            self.tap_curr_output,
        )


def _record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
    stream = None
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is None or tensor.device.type != "cuda":
            continue
        ident = id(tensor)
        if ident in seen:
            continue
        if stream is None:
            stream = torch.cuda.current_stream(device=tensor.device)
        tensor.record_stream(stream)
        seen.add(ident)


def _cache_set(cache: dict, key: object, value, *, limit: int) -> None:
    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= int(limit):
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def _torch_to_cutlass_dtype(dt: torch.dtype) -> type[cutlass.Numeric]:
    if dt == torch.float16:
        return cutlass.Float16
    if dt == torch.bfloat16:
        return cutlass.BFloat16
    if dt == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported dtype: {dt}")


def _pad_zero_time(
    tensor: torch.Tensor,
    *,
    T_pad: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.to(dtype=dtype).contiguous()
    T = int(tensor.shape[2])
    if T == T_pad:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[2] = T_pad - T
    pad = torch.zeros(pad_shape, device=tensor.device, dtype=dtype)
    return torch.cat((tensor, pad), dim=2).contiguous()


def _pad_m_identity(M: torch.Tensor, *, T_pad: int) -> torch.Tensor:
    M = M.to(dtype=torch.float32).contiguous()
    T = int(M.shape[2])
    if T == T_pad:
        return M
    pad_shape = list(M.shape)
    pad_shape[2] = T_pad - T
    pad = torch.zeros(pad_shape, device=M.device, dtype=torch.float32)
    pad[..., 0] = 1.0
    return torch.cat((M, pad), dim=2).contiguous()


def _get_zero_prev_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    heads: int,
    bc_groups: int,
    P: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        dtype,
        int(batch_size),
        int(heads),
        int(bc_groups),
        int(P),
        int(D),
    )
    cached = _ZERO_PREV_CACHE.get(key)
    if cached is None:
        cached = (
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
            torch.zeros((batch_size, bc_groups, D), device=device, dtype=dtype),
        )
        _cache_set(_ZERO_PREV_CACHE, key, cached, limit=_ZERO_PREV_CACHE_LIMIT)
    return cached


def _reduce_heads_to_bc_groups(
    x: torch.Tensor,
    *,
    bc_groups: int,
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("Expected tensor with a head axis at dim=1.")
    batch_size, heads = map(int, x.shape[:2])
    if heads % bc_groups != 0:
        raise ValueError(
            f"bc_groups must divide heads. Got heads={heads}, bc_groups={bc_groups}."
        )
    if heads == bc_groups:
        return x
    heads_per_group = heads // bc_groups
    grouped_shape = (batch_size, bc_groups, heads_per_group, *x.shape[2:])
    return x.reshape(grouped_shape).sum(dim=2)


def _chunk_scan_device_label(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return f"{props.name} (sm_{props.major}{props.minor})"


def _validate_dz0_support(
    *,
    tc_dtype: torch.dtype,
    chunk_size: int,
    dz0_cta_tiler: tuple[int, int, int],
    device_index: int,
) -> None:
    kernel = ChunkScanBwdDZ0Ampere(
        _torch_to_cutlass_dtype(tc_dtype),
        chunk_size=chunk_size,
        cta_tiler=dz0_cta_tiler,
    )
    info = kernel.support_info(device_index=device_index)
    if info.supported:
        return

    device_label = _chunk_scan_device_label(device_index)
    raise ValueError(
        f"No supported chunk_scan backward dz0 kernel fits {device_label} for "
        f"(chunk_size={chunk_size}, cta_tiler={dz0_cta_tiler}). "
        f"The current variant needs {info.required_smem_bytes}B > "
        f"{info.smem_capacity_bytes}B shared memory."
    )


def _validate_dcdr_support(
    *,
    tc_dtype: torch.dtype,
    chunk_size: int,
    D: int,
    P: int,
    num_threads: int,
    device_index: int,
) -> None:
    kernel = ChunkScanBwdDCDRAmpere(
        _torch_to_cutlass_dtype(tc_dtype),
        chunk_size=chunk_size,
        D=D,
        P=P,
        num_threads=num_threads,
    )
    info = kernel.support_info(
        _torch_to_cutlass_dtype(tc_dtype),
        device_index=device_index,
    )
    if info.supported:
        return

    device_label = _chunk_scan_device_label(device_index)
    raise ValueError(
        f"No supported chunk_scan backward dcdr kernel fits {device_label} for "
        f"(chunk_size={chunk_size}, D={D}, P={P}, num_threads={num_threads}). "
        f"The current low-SMEM variant needs {info.required_smem_bytes}B > "
        f"{info.smem_capacity_bytes}B shared memory."
    )


def _validate_dlp_support(
    *,
    tc_dtype: torch.dtype,
    chunk_size: int,
    D: int,
    P: int,
    num_threads: int,
    device_index: int,
) -> None:
    kernel = ChunkScanBwdDLPAmpere(
        _torch_to_cutlass_dtype(tc_dtype),
        chunk_size=chunk_size,
        D=D,
        P=P,
        num_threads=num_threads,
    )
    info = kernel.support_info(
        _torch_to_cutlass_dtype(tc_dtype),
        device_index=device_index,
    )
    if info.supported:
        return

    device_label = _chunk_scan_device_label(device_index)
    raise ValueError(
        f"No supported chunk_scan backward dlp kernel fits {device_label} for "
        f"(chunk_size={chunk_size}, D={D}, P={P}, num_threads={num_threads}). "
        f"The current low-SMEM variant needs {info.required_smem_bytes}B > "
        f"{info.smem_capacity_bytes}B shared memory."
    )


def _validate_db_support(
    *,
    tc_dtype: torch.dtype,
    chunk_size: int,
    D: int,
    P: int,
    num_threads: int,
    device_index: int,
) -> None:
    kernel = ChunkScanBwdDBAmpere(
        _torch_to_cutlass_dtype(tc_dtype),
        chunk_size=chunk_size,
        D=D,
        P=P,
        num_threads=num_threads,
    )
    info = kernel.support_info(
        _torch_to_cutlass_dtype(tc_dtype),
        device_index=device_index,
    )
    if info.supported:
        return

    device_label = _chunk_scan_device_label(device_index)
    raise ValueError(
        f"No supported chunk_scan backward db kernel fits {device_label} for "
        f"(chunk_size={chunk_size}, D={D}, P={P}, num_threads={num_threads}). "
        f"The current low-SMEM variant needs {info.required_smem_bytes}B > "
        f"{info.smem_capacity_bytes}B shared memory."
    )


def _validate_du_support(
    *,
    tc_dtype: torch.dtype,
    chunk_size: int,
    D: int,
    P: int,
    num_threads: int,
    device_index: int,
) -> None:
    kernel = ChunkScanBwdDUAmpere(
        _torch_to_cutlass_dtype(tc_dtype),
        chunk_size=chunk_size,
        D=D,
        P=P,
        num_threads=num_threads,
    )
    info = kernel.support_info(
        _torch_to_cutlass_dtype(tc_dtype),
        device_index=device_index,
    )
    if info.supported:
        return

    device_label = _chunk_scan_device_label(device_index)
    raise ValueError(
        f"No supported chunk_scan backward du kernel fits {device_label} for "
        f"(chunk_size={chunk_size}, D={D}, P={P}, num_threads={num_threads}). "
        f"The current DU variant needs {info.required_smem_bytes}B > "
        f"{info.smem_capacity_bytes}B shared memory."
    )


def _assumed_align(
    t: torch.Tensor,
    candidates_bytes: tuple[int, ...] = (16, 8, 4),
) -> int:
    elem_align = max(1, t.element_size())
    ptr = int(t.data_ptr())
    for align in candidates_bytes:
        if align < elem_align:
            continue
        if (ptr % align) == 0:
            return align
    return elem_align


def _make_row_major_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = running
        running *= int(shape[i])
    return tuple(stride)


def _make_tensor_spec(
    shape: tuple[int, ...],
    *,
    stride: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(dim) for dim in shape)
    if stride is None:
        stride = _make_row_major_stride(shape)
    else:
        stride = tuple(int(step) for step in stride)
    return shape, stride


def _make_tensor_spec_from_tensor(
    t: torch.Tensor,
) -> TensorSpec:
    return _make_tensor_spec(
        tuple(map(int, t.shape)), stride=tuple(map(int, t.stride()))
    )


def _make_tensor_from_spec(
    tensor: cute.Tensor,
    spec: TensorSpec,
):
    shape, stride = spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


@dataclass(frozen=True)
class ChunkScanBwdDBWorkspace:
    value_grad_scratch: torch.Tensor
    key_grad_chunk: torch.Tensor
    value_boundary_scratch: torch.Tensor
    key_boundary_grad: torch.Tensor
    logprefix_scratch: torch.Tensor
    transition_prev_scratch: torch.Tensor
    transition_curr_scratch: torch.Tensor


@dataclass(frozen=True)
class ChunkScanBwdDBRuntimeArtifacts:
    workspace: ChunkScanBwdDBWorkspace
    runtime_args: tuple[torch.Tensor, ...]
    alignments: tuple[int, ...]
    tensor_specs: tuple[TensorSpec, ...]


def _make_db_workspace(
    *,
    BHC: int,
    L: int,
    D: int,
    P: int,
    device: torch.device,
    tc_dtype: torch.dtype,
    u_block_like: torch.Tensor,
    b_block_like: torch.Tensor,
) -> ChunkScanBwdDBWorkspace:
    return ChunkScanBwdDBWorkspace(
        value_grad_scratch=torch.empty_like(u_block_like),
        key_grad_chunk=torch.empty_like(b_block_like),
        value_boundary_scratch=torch.empty((BHC, P), device=device, dtype=tc_dtype),
        key_boundary_grad=torch.empty((BHC, D), device=device, dtype=tc_dtype),
        logprefix_scratch=torch.empty((BHC, L), device=device, dtype=torch.float32),
        transition_prev_scratch=torch.empty(
            (BHC, L, 2),
            device=device,
            dtype=torch.float32,
        ),
        transition_curr_scratch=torch.empty(
            (BHC, L, 2),
            device=device,
            dtype=torch.float32,
        ),
    )


def _make_db_runtime_artifacts(
    *,
    U_blk: torch.Tensor,
    B_blk: torch.Tensor,
    C_blk: torch.Tensor,
    M_blk: torch.Tensor,
    K_blk: torch.Tensor,
    dOut_blk: torch.Tensor,
    U_prev0_flat: torch.Tensor,
    B_prev0_flat: torch.Tensor,
    workspace: ChunkScanBwdDBWorkspace,
) -> ChunkScanBwdDBRuntimeArtifacts:
    runtime_args = (
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        U_prev0_flat,
        B_prev0_flat,
        workspace.value_grad_scratch,
        workspace.key_grad_chunk,
        workspace.value_boundary_scratch,
        workspace.key_boundary_grad,
        workspace.logprefix_scratch,
        workspace.transition_prev_scratch,
        workspace.transition_curr_scratch,
    )
    return ChunkScanBwdDBRuntimeArtifacts(
        workspace=workspace,
        runtime_args=runtime_args,
        alignments=tuple(_assumed_align(tensor) for tensor in runtime_args),
        tensor_specs=_make_tensor_specs_from_tensors(*runtime_args),
    )


def _compile_db_wrapper(
    *,
    runtime_artifacts: ChunkScanBwdDBRuntimeArtifacts,
    cfg: tuple[int, int, int, int],
    cutlass_dtype,
):
    wrapper = _make_db_host_wrapper(
        spec=runtime_artifacts.tensor_specs,
        cfg=cfg,
        cutlass_dtype=cutlass_dtype,
    )
    compile_args = tuple(
        _make_fake_tensor_arg(tensor, align=align)
        for tensor, align in zip(
            runtime_artifacts.runtime_args,
            runtime_artifacts.alignments,
            strict=True,
        )
    )
    return cute.compile(
        wrapper,
        *compile_args,
        options="--enable-tvm-ffi",
    )


def _make_tensor_specs_from_tensors(
    *tensors: torch.Tensor,
) -> tuple[TensorSpec, ...]:
    return tuple(_make_tensor_spec_from_tensor(tensor) for tensor in tensors)


def _make_compile_args(
    runtime_args: tuple[torch.Tensor, ...],
    alignments: tuple[int, ...],
) -> tuple[object, ...]:
    return tuple(
        _make_fake_tensor_arg(tensor, align=align)
        for tensor, align in zip(runtime_args, alignments, strict=True)
    )


@dataclass(frozen=True)
class ChunkScanBwdDCDRRuntimeArtifacts:
    runtime_args: tuple[torch.Tensor, ...]
    alignments: tuple[int, ...]
    tensor_specs: tuple[TensorSpec, ...]
    compile_args: tuple[object, ...]


def _make_dcdr_runtime_artifacts(
    *,
    U_blk: torch.Tensor,
    B_blk: torch.Tensor,
    C_blk: torch.Tensor,
    M_blk: torch.Tensor,
    K_blk: torch.Tensor,
    dOut_blk: torch.Tensor,
    U_prev0_flat: torch.Tensor,
    B_prev0_flat: torch.Tensor,
    Z0_blk: torch.Tensor,
    dC: torch.Tensor,
    d_logprefix: torch.Tensor,
    d_r: torch.Tensor,
) -> ChunkScanBwdDCDRRuntimeArtifacts:
    runtime_args = (
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        U_prev0_flat,
        B_prev0_flat,
        Z0_blk,
        dC,
        d_logprefix,
        d_r,
    )
    alignments = tuple(_assumed_align(tensor) for tensor in runtime_args)
    return ChunkScanBwdDCDRRuntimeArtifacts(
        runtime_args=runtime_args,
        alignments=alignments,
        tensor_specs=_make_tensor_specs_from_tensors(*runtime_args),
        compile_args=_make_compile_args(runtime_args, alignments),
    )


def _allocate_du_workspace(
    *,
    value_template: torch.Tensor,
    key_template: torch.Tensor,
    device: torch.device,
    tc_dtype: torch.dtype,
    BHC: int,
    chunk_size: int,
    D: int,
) -> ChunkScanBwdDUWorkspace:
    return ChunkScanBwdDUWorkspace(
        value_grad_chunk=torch.empty_like(value_template),
        key_grad_scratch=torch.empty_like(key_template),
        value_boundary_grad=torch.empty(
            (BHC, value_template.shape[-1]), device=device, dtype=tc_dtype
        ),
        key_boundary_scratch=torch.empty((BHC, D), device=device, dtype=tc_dtype),
        logprefix_scratch=torch.empty(
            (BHC, chunk_size), device=device, dtype=torch.float32
        ),
        transition_prev_scratch=torch.empty(
            (BHC, chunk_size, 2), device=device, dtype=torch.float32
        ),
        transition_curr_scratch=torch.empty(
            (BHC, chunk_size, 2), device=device, dtype=torch.float32
        ),
    )


def _make_du_runtime_args(
    *,
    U_blk: torch.Tensor,
    B_blk: torch.Tensor,
    C_blk: torch.Tensor,
    M_blk: torch.Tensor,
    K_blk: torch.Tensor,
    dOut_blk: torch.Tensor,
    U_prev0_flat: torch.Tensor,
    B_prev0_flat: torch.Tensor,
    workspace: ChunkScanBwdDUWorkspace,
) -> tuple[torch.Tensor, ...]:
    return (
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        U_prev0_flat,
        B_prev0_flat,
        *workspace.tensors(),
    )


def _make_du_runtime_artifacts(
    *,
    U_blk: torch.Tensor,
    B_blk: torch.Tensor,
    C_blk: torch.Tensor,
    M_blk: torch.Tensor,
    K_blk: torch.Tensor,
    dOut_blk: torch.Tensor,
    U_prev0_flat: torch.Tensor,
    B_prev0_flat: torch.Tensor,
    device: torch.device,
    tc_dtype: torch.dtype,
    BHC: int,
    chunk_size: int,
    D: int,
) -> ChunkScanBwdDURuntimeArtifacts:
    workspace = _allocate_du_workspace(
        value_template=U_blk,
        key_template=B_blk,
        device=device,
        tc_dtype=tc_dtype,
        BHC=BHC,
        chunk_size=chunk_size,
        D=D,
    )
    runtime_args = _make_du_runtime_args(
        U_blk=U_blk,
        B_blk=B_blk,
        C_blk=C_blk,
        M_blk=M_blk,
        K_blk=K_blk,
        dOut_blk=dOut_blk,
        U_prev0_flat=U_prev0_flat,
        B_prev0_flat=B_prev0_flat,
        workspace=workspace,
    )
    return ChunkScanBwdDURuntimeArtifacts(
        workspace=workspace,
        runtime_args=runtime_args,
        alignments=tuple(_assumed_align(tensor) for tensor in runtime_args),
    )


def _public_from_chunked(
    x: torch.Tensor,
    *,
    T: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    out = x.reshape(B, H, C * L, F)[:, :, :T, :]
    return _materialize_public_output(x, out, dtype=dtype)


def _public_from_chunked_with_boundary_carries(
    x: torch.Tensor,
    x_prev: torch.Tensor,
    *,
    T: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # `_fold_chunk_boundary_carries` is intentionally in-place for the raw
    # staged buffers, so clone here to keep repeated public materializations
    # stable when callers inspect a prepared launch more than once.
    return _public_from_chunked(
        _fold_chunk_boundary_carries(x.clone(), x_prev),
        T=T,
        dtype=dtype,
    )


def _fold_chunk_boundary_carries(
    x: torch.Tensor,
    x_prev: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 5 or x_prev.ndim != 4:
        raise ValueError("Expected chunked main grads and per-chunk boundary carries.")
    if x.shape[:3] != x_prev.shape[:3] or x.shape[-1] != x_prev.shape[-1]:
        raise ValueError("Chunked grads and boundary carries must agree on (B,H,C,F).")
    n_chunks = int(x.shape[2])
    if n_chunks <= 1:
        return x
    x[:, :, :-1, -1, :].add_(x_prev[:, :, 1:, :])
    return x


def _public_from_param_scan(
    x: torch.Tensor,
    *,
    T: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    B, H, C, S, L, F = map(int, x.shape)
    if S != 1:
        raise ValueError("Only n_splits=1 is supported by the public wrapper.")
    out = x[:, :, :, 0, :, :].reshape(B, H, C * L, F)[:, :, :T, :]
    return _materialize_public_output(x, out, dtype=dtype)


def _public_dk_from_parts(
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    *,
    T: int,
) -> torch.Tensor:
    if dKprev.shape != dKcurr.shape:
        raise ValueError("dKprev and dKcurr must have identical shapes.")
    B, H, C, S, L, F = map(int, dKprev.shape)
    if S != 1:
        raise ValueError("Only n_splits=1 is supported by the public wrapper.")
    dK = torch.stack((dKprev[:, :, :, 0, :, :], dKcurr[:, :, :, 0, :, :]), dim=4)
    return (
        dK.reshape(B, H, C * L, 2, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()
    )


def _make_param_scan_runtime_artifacts(
    *,
    d_logprefix: torch.Tensor,
    d_m_prev: torch.Tensor,
    d_m_curr: torch.Tensor,
    d_r: torch.Tensor,
    batch_size: int,
    heads: int,
    n_chunks: int,
    chunk_size: int,
) -> ChunkScanBwdParamScanRuntimeArtifacts:
    if d_logprefix.ndim != 2:
        raise ValueError("dlogprefix scratch must be shaped as (BHC, L).")
    if d_m_prev.shape != d_m_curr.shape:
        raise ValueError("dMprev/dMcurr scratch must have identical shapes.")
    if (
        d_m_prev.ndim != 3
        or d_m_prev.shape[0] != d_logprefix.shape[0]
        or d_m_prev.shape[1] != d_logprefix.shape[1]
        or d_m_prev.shape[2] != 2
    ):
        raise ValueError(
            "dMprev scratch must be shaped as (BHC, L, 2) matching dlogprefix."
        )
    if (
        d_r.ndim != 3
        or d_r.shape[0] != d_logprefix.shape[0]
        or d_r.shape[1] != d_logprefix.shape[1]
        or d_r.shape[2] != 4
    ):
        raise ValueError(
            "dR scratch must be shaped as (BHC, L, 4) matching dlogprefix."
        )

    logprefix_input = d_logprefix.unsqueeze(1).contiguous()
    transition_prev_input = d_m_prev.unsqueeze(1).contiguous()
    transition_curr_input = d_m_curr.unsqueeze(1).contiguous()
    rotation_input = d_r.unsqueeze(1).contiguous()
    transition_output = torch.empty_like(transition_prev_input)
    tap_prev_output = torch.empty_like(transition_prev_input)
    tap_curr_output = torch.empty_like(transition_prev_input)

    param_split_count = int(logprefix_input.shape[1])
    transition_view = transition_output.reshape(
        batch_size, heads, n_chunks, param_split_count, chunk_size, 2
    )
    tap_prev_view = tap_prev_output.reshape(
        batch_size, heads, n_chunks, param_split_count, chunk_size, 2
    )
    tap_curr_view = tap_curr_output.reshape(
        batch_size, heads, n_chunks, param_split_count, chunk_size, 2
    )
    return ChunkScanBwdParamScanRuntimeArtifacts(
        logprefix_input=logprefix_input,
        transition_prev_input=transition_prev_input,
        transition_curr_input=transition_curr_input,
        rotation_input=rotation_input,
        transition_output=transition_output,
        tap_prev_output=tap_prev_output,
        tap_curr_output=tap_curr_output,
        transition_view=transition_view,
        tap_prev_view=tap_prev_view,
        tap_curr_view=tap_curr_view,
    )


def _materialize_public_output(
    source: torch.Tensor,
    out: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = out.dtype if dtype is None else dtype
    if out.dtype != target_dtype:
        out = out.to(dtype=target_dtype)
    if not out.is_contiguous():
        out = out.contiguous()
    # Public returns must not alias cached workspace storage when the view
    # already has the public layout and dtype.
    if out.data_ptr() == source.data_ptr():
        out = out.clone()
    return out


def _resolve_dz0_cta_tiler(*, D: int) -> tuple[int, int, int]:
    # The 96-wide D tile is fine when it covers the state width cleanly, but
    # mixed full+tail tiling perturbs the current dz0 epilogue on realistic
    # D=2N mixer shapes. Use the same tail-safe family selection as the forward
    # chunk-increment path.
    if D <= 96 or D % 96 == 0:
        return (64, 96, 32)
    return (64, 128, 32)


def _compiled_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    C_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    chunk_starts_shape: tuple[int, ...],
    d_out_shape: tuple[int, ...],
    chunk_size: int,
    has_prev: bool,
    dz0_cta_tiler: tuple[int, int, int],
    alignments: tuple[int, ...],
    num_threads_du: int,
    num_threads_db: int,
    num_threads_dcdr: int,
    num_threads_param: int,
) -> tuple:
    return (
        "chunk_scan_bwd",
        device_index,
        tc_dtype,
        U_shape,
        B_shape,
        C_shape,
        M_shape,
        K_shape,
        chunk_starts_shape,
        d_out_shape,
        int(chunk_size),
        has_prev,
        dz0_cta_tiler,
        alignments,
        int(num_threads_du),
        int(num_threads_db),
        int(num_threads_dcdr),
        int(num_threads_param),
    )


def _make_dz0_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, tuple[int, int, int], int, int],
    cutlass_dtype,
):
    chunk_size, dz0_cta_tiler, heads, bc_groups = cfg
    d_out_spec, c_spec, m_spec, dz0_spec = spec

    @cute.jit
    def _dz0_host_wrapper(
        DOut_ptr: cute.Tensor,
        C_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        DZ0_ptr: cute.Tensor,
    ):
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mDZ0 = _make_tensor_from_spec(DZ0_ptr, dz0_spec)

        kernel = ChunkScanBwdDZ0Ampere(
            cutlass_dtype,
            chunk_size=chunk_size,
            cta_tiler=dz0_cta_tiler,
            heads=heads,
            bc_groups=bc_groups,
        )
        kernel(mDOut, mC, mM, mDZ0)

    return _dz0_host_wrapper


def _make_du_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
    cutlass_dtype,
):
    chunk_size, D, P, num_threads, heads, bc_groups = cfg
    (
        u_spec,
        b_spec,
        c_spec,
        m_spec,
        k_spec,
        d_out_spec,
        u_prev0_spec,
        b_prev0_spec,
        d_u_spec,
        d_b_scratch_spec,
        d_u_prev_spec,
        d_b_prev_scratch_spec,
        d_logprefix_spec,
        d_m_prev_spec,
        d_m_curr_spec,
    ) = spec

    @cute.jit
    def _du_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        C_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        K_ptr: cute.Tensor,
        DOut_ptr: cute.Tensor,
        UPrev0_ptr: cute.Tensor,
        BPrev0_ptr: cute.Tensor,
        DU_ptr: cute.Tensor,
        DBScratch_ptr: cute.Tensor,
        DUPrev_ptr: cute.Tensor,
        DBPrevScratch_ptr: cute.Tensor,
        DLogPrefix_ptr: cute.Tensor,
        DMp_ptr: cute.Tensor,
        DMc_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mUPrev0 = _make_tensor_from_spec(UPrev0_ptr, u_prev0_spec)
        mBPrev0 = _make_tensor_from_spec(BPrev0_ptr, b_prev0_spec)
        mDU = _make_tensor_from_spec(DU_ptr, d_u_spec)
        mDBScratch = _make_tensor_from_spec(DBScratch_ptr, d_b_scratch_spec)
        mDUPrev = _make_tensor_from_spec(DUPrev_ptr, d_u_prev_spec)
        mDBPrevScratch = _make_tensor_from_spec(
            DBPrevScratch_ptr, d_b_prev_scratch_spec
        )
        mDLogPrefix = _make_tensor_from_spec(DLogPrefix_ptr, d_logprefix_spec)
        mDMPrev = _make_tensor_from_spec(DMp_ptr, d_m_prev_spec)
        mDMCurr = _make_tensor_from_spec(DMc_ptr, d_m_curr_spec)

        kernel = ChunkScanBwdDUAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            num_threads=num_threads,
            heads=heads,
            bc_groups=bc_groups,
        )
        kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mUPrev0,
            mBPrev0,
            mDU,
            mDBScratch,
            mDUPrev,
            mDBPrevScratch,
            mDLogPrefix,
            mDMPrev,
            mDMCurr,
        )

    return _du_host_wrapper


def _make_db_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
    cutlass_dtype,
):
    chunk_size, D, P, num_threads, heads, bc_groups = cfg
    (
        u_spec,
        b_spec,
        c_spec,
        m_spec,
        k_spec,
        d_out_spec,
        u_prev0_spec,
        b_prev0_spec,
        d_u_spec,
        d_b_spec,
        d_u_prev_spec,
        d_b_prev_spec,
        d_logprefix_spec,
        d_m_prev_spec,
        d_m_curr_spec,
    ) = spec

    @cute.jit
    def _db_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        C_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        K_ptr: cute.Tensor,
        DOut_ptr: cute.Tensor,
        UPrev0_ptr: cute.Tensor,
        BPrev0_ptr: cute.Tensor,
        DU_ptr: cute.Tensor,
        DB_ptr: cute.Tensor,
        DUPrev_ptr: cute.Tensor,
        DBPrev_ptr: cute.Tensor,
        DLogPrefix_ptr: cute.Tensor,
        DMPrev_ptr: cute.Tensor,
        DMCurr_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mUPrev0 = _make_tensor_from_spec(UPrev0_ptr, u_prev0_spec)
        mBPrev0 = _make_tensor_from_spec(BPrev0_ptr, b_prev0_spec)
        mDU = _make_tensor_from_spec(DU_ptr, d_u_spec)
        mDB = _make_tensor_from_spec(DB_ptr, d_b_spec)
        mDUPrev = _make_tensor_from_spec(DUPrev_ptr, d_u_prev_spec)
        mDBPrev = _make_tensor_from_spec(DBPrev_ptr, d_b_prev_spec)
        mDLogPrefix = _make_tensor_from_spec(DLogPrefix_ptr, d_logprefix_spec)
        mDMPrev = _make_tensor_from_spec(DMPrev_ptr, d_m_prev_spec)
        mDMCurr = _make_tensor_from_spec(DMCurr_ptr, d_m_curr_spec)

        kernel = ChunkScanBwdDBAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            num_threads=num_threads,
            heads=heads,
            bc_groups=bc_groups,
        )
        kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mUPrev0,
            mBPrev0,
            mDU,
            mDB,
            mDUPrev,
            mDBPrev,
            mDLogPrefix,
            mDMPrev,
            mDMCurr,
        )

    return _db_host_wrapper


def _make_dcdr_host_wrapper(
    *,
    tensor_specs: tuple[TensorSpec, ...],
    kernel_cfg: tuple[int, ...],
    cutlass_dtype,
):
    chunk_size, D, P, num_threads, heads, bc_groups = kernel_cfg
    (
        u_spec,
        b_spec,
        c_spec,
        m_spec,
        k_spec,
        d_out_spec,
        u_prev0_spec,
        b_prev0_spec,
        z0_spec,
        d_c_spec,
        d_logprefix_spec,
        d_r_spec,
    ) = tensor_specs

    @cute.jit
    def _dcdr_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        C_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        K_ptr: cute.Tensor,
        DOut_ptr: cute.Tensor,
        UPrev0_ptr: cute.Tensor,
        BPrev0_ptr: cute.Tensor,
        Z0_ptr: cute.Tensor,
        DC_ptr: cute.Tensor,
        DLogPrefix_ptr: cute.Tensor,
        DR_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mUPrev0 = _make_tensor_from_spec(UPrev0_ptr, u_prev0_spec)
        mBPrev0 = _make_tensor_from_spec(BPrev0_ptr, b_prev0_spec)
        mZ0 = _make_tensor_from_spec(Z0_ptr, z0_spec)
        mDC = _make_tensor_from_spec(DC_ptr, d_c_spec)
        mDLogPrefix = _make_tensor_from_spec(DLogPrefix_ptr, d_logprefix_spec)
        mDR = _make_tensor_from_spec(DR_ptr, d_r_spec)

        kernel = ChunkScanBwdDCDRAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            num_threads=num_threads,
            heads=heads,
            bc_groups=bc_groups,
        )
        kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mUPrev0,
            mBPrev0,
            mZ0,
            mDC,
            mDLogPrefix,
            mDR,
        )

    return _dcdr_host_wrapper


def _make_dlp_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
    cutlass_dtype,
):
    chunk_size, D, P, num_threads, heads, bc_groups = cfg
    (
        u_spec,
        b_spec,
        c_spec,
        m_spec,
        k_spec,
        d_out_spec,
        u_prev0_spec,
        b_prev0_spec,
        d_logprefix_spec,
    ) = spec

    @cute.jit
    def _dlp_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        C_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        K_ptr: cute.Tensor,
        DOut_ptr: cute.Tensor,
        UPrev0_ptr: cute.Tensor,
        BPrev0_ptr: cute.Tensor,
        DLogPrefix_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mUPrev0 = _make_tensor_from_spec(UPrev0_ptr, u_prev0_spec)
        mBPrev0 = _make_tensor_from_spec(BPrev0_ptr, b_prev0_spec)
        mDLogPrefix = _make_tensor_from_spec(DLogPrefix_ptr, d_logprefix_spec)

        kernel = ChunkScanBwdDLPAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            num_threads=num_threads,
            heads=heads,
            bc_groups=bc_groups,
        )
        kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mUPrev0,
            mBPrev0,
            mDLogPrefix,
        )

    return _dlp_host_wrapper


def _make_param_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
):
    chunk_size, num_threads = cfg
    (
        m_spec,
        k_spec,
        d_logprefix_spec,
        d_m_prev_spec,
        d_m_curr_spec,
        d_r_spec,
        dm_output_spec,
        dkprev_output_spec,
        dkcurr_output_spec,
    ) = spec

    @cute.jit
    def _param_host_wrapper(
        M_ptr: cute.Tensor,
        K_ptr: cute.Tensor,
        DLogPrefix_ptr: cute.Tensor,
        DMprev_ptr: cute.Tensor,
        DMcurr_ptr: cute.Tensor,
        DR_ptr: cute.Tensor,
        DMout_ptr: cute.Tensor,
        DKprev_ptr: cute.Tensor,
        DKcurr_ptr: cute.Tensor,
    ):
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDLogPrefix = _make_tensor_from_spec(DLogPrefix_ptr, d_logprefix_spec)
        mDMPrev = _make_tensor_from_spec(DMprev_ptr, d_m_prev_spec)
        mDMCurr = _make_tensor_from_spec(DMcurr_ptr, d_m_curr_spec)
        mDR = _make_tensor_from_spec(DR_ptr, d_r_spec)
        mDMout = _make_tensor_from_spec(DMout_ptr, dm_output_spec)
        mDKprev = _make_tensor_from_spec(DKprev_ptr, dkprev_output_spec)
        mDKcurr = _make_tensor_from_spec(DKcurr_ptr, dkcurr_output_spec)

        kernel = ChunkScanBwdParamScanAmpere(
            chunk_size=chunk_size,
            num_threads=num_threads,
        )
        kernel(mM, mK, mDLogPrefix, mDMPrev, mDMCurr, mDR, mDMout, mDKprev, mDKcurr)

    return _param_host_wrapper


def compile_chunk_scan_bwd_kernels(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    num_threads_du: int = 128,
    num_threads_db: int = 128,
    num_threads_dcdr: int = 128,
    num_threads_param: int = 32,
) -> PreparedChunkScanBwdLaunch:
    """Prepare the standalone chunk-scan backward stage launches and outputs."""
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")

    Bsz, H, T, P = map(int, U.shape)
    G = int(B.shape[1])
    D = int(B.shape[-1])
    if H % G != 0:
        raise ValueError(f"bc_groups must divide heads. Got heads={H}, bc_groups={G}.")
    if B.shape != (Bsz, G, T, D) or C.shape != (Bsz, G, T, D):
        raise ValueError("B/C must be (B,G,T,D) with contiguous head-to-group mapping.")
    if M.shape != (Bsz, H, T, 2):
        raise ValueError(f"M must be (B,H,T,2)={(Bsz, H, T, 2)}.")
    if K.shape != (Bsz, H, T, 2, 2):
        raise ValueError(f"K must be (B,H,T,2,2)={(Bsz, H, T, 2, 2)}.")
    if d_out.shape != (Bsz, H, T, P):
        raise ValueError("d_out must be (B,H,T,P) matching U.")
    if D % 2 != 0:
        raise ValueError("D must be divisible by 2 (flattened 2N).")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L
    if chunk_starts.shape != (Bsz, H, n_chunks, P, D):
        raise ValueError(
            "chunk_starts must be (B,H,C,P,D) "
            f"={(Bsz, H, n_chunks, P, D)}. Got {tuple(chunk_starts.shape)}."
        )
    if num_threads_dcdr != 128:
        raise ValueError("num_threads_dcdr must be 128 for dcdr/dlp kernels.")
    num_threads_dlp = num_threads_dcdr

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    device_index = (
        U.device.index if U.device.index is not None else torch.cuda.current_device()
    )
    dz0_cta_tiler = _resolve_dz0_cta_tiler(D=D)
    _validate_dz0_support(
        tc_dtype=tc_dtype,
        chunk_size=L,
        dz0_cta_tiler=dz0_cta_tiler,
        device_index=int(device_index),
    )
    _validate_dcdr_support(
        tc_dtype=tc_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_dcdr,
        device_index=int(device_index),
    )
    _validate_dlp_support(
        tc_dtype=tc_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_dlp,
        device_index=int(device_index),
    )
    _validate_db_support(
        tc_dtype=tc_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_db,
        device_index=int(device_index),
    )
    _validate_du_support(
        tc_dtype=tc_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_du,
        device_index=int(device_index),
    )

    U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
    C_tc = _pad_zero_time(C, T_pad=T_pad, dtype=tc_dtype)
    d_out_tc = _pad_zero_time(d_out, T_pad=T_pad, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=T_pad)
    K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
    chunk_starts_f = chunk_starts.to(dtype=torch.float32).contiguous()

    if B_prev is None:
        U_prev0, B_prev0 = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=Bsz,
            heads=H,
            bc_groups=G,
            P=P,
            D=D,
        )
    else:
        if B_prev.shape != (Bsz, G, D) or U_prev.shape != (Bsz, H, P):
            raise ValueError("B_prev/U_prev must be (B,G,D)/(B,H,P).")
        B_prev0 = B_prev.to(dtype=tc_dtype).contiguous()
        U_prev0 = U_prev.to(dtype=tc_dtype).contiguous()

    BH = Bsz * H
    BG = Bsz * G
    BHC = BH * n_chunks
    BGC = BG * n_chunks

    d_out_operand = d_out_tc.reshape(BH, T_pad, P).permute(2, 1, 0)
    c_operand = C_tc.reshape(BG, T_pad, D).permute(2, 1, 0)
    m_operand = M_f.reshape(BH, T_pad, 2).permute(2, 1, 0)

    d_z0 = torch.empty((BHC, P, D), device=U.device, dtype=torch.float32)
    d_z0_perm = d_z0.permute(1, 2, 0)
    d_z0_view = d_z0.reshape(Bsz, H, n_chunks, P, D)

    U_blk = U_tc.reshape(BH, n_chunks, L, P).reshape(BHC, L, 1, P).contiguous()
    B_blk = B_tc.reshape(BG, n_chunks, L, D).reshape(BGC, L, 1, D).contiguous()
    C_blk = C_tc.reshape(BG, n_chunks, L, D).reshape(BGC, L, 1, D).contiguous()
    M_blk = M_f.reshape(BH, n_chunks, L, 2).reshape(BHC, L, 2).contiguous()
    K_blk = K_f.reshape(BH, n_chunks, L, 2, 2).reshape(BHC, L, 2, 2).contiguous()
    dOut_blk = d_out_tc.reshape(BH, n_chunks, L, P).reshape(BHC, L, 1, P).contiguous()
    Z0_blk = chunk_starts_f.reshape(BH, n_chunks, P, D).reshape(BHC, P, D).contiguous()

    U_prev0_flat = U_prev0.reshape(BH, P).contiguous()
    B_prev0_flat = B_prev0.reshape(BG, D).contiguous()

    du_runtime_artifacts = _make_du_runtime_artifacts(
        U_blk=U_blk,
        B_blk=B_blk,
        C_blk=C_blk,
        M_blk=M_blk,
        K_blk=K_blk,
        dOut_blk=dOut_blk,
        U_prev0_flat=U_prev0_flat,
        B_prev0_flat=B_prev0_flat,
        device=U.device,
        tc_dtype=tc_dtype,
        BHC=BHC,
        chunk_size=L,
        D=D,
    )
    db_workspace = _make_db_workspace(
        BHC=BHC,
        L=L,
        D=D,
        P=P,
        device=U.device,
        tc_dtype=tc_dtype,
        u_block_like=U_blk,
        b_block_like=B_blk,
    )
    db_runtime_artifacts = _make_db_runtime_artifacts(
        U_blk=U_blk,
        B_blk=B_blk,
        C_blk=C_blk,
        M_blk=M_blk,
        K_blk=K_blk,
        dOut_blk=dOut_blk,
        U_prev0_flat=U_prev0_flat,
        B_prev0_flat=B_prev0_flat,
        workspace=db_workspace,
    )

    d_logprefix = torch.empty((BHC, L), device=U.device, dtype=torch.float32)
    d_c = torch.empty_like(C_blk)
    d_r = torch.empty((BHC, L, 4), device=U.device, dtype=torch.float32)

    param_scan_runtime_artifacts = _make_param_scan_runtime_artifacts(
        d_logprefix=d_logprefix,
        d_m_prev=db_workspace.transition_prev_scratch,
        d_m_curr=db_workspace.transition_curr_scratch,
        d_r=d_r,
        batch_size=Bsz,
        heads=H,
        n_chunks=n_chunks,
        chunk_size=L,
    )

    cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)

    dz0_runtime_args = (d_out_operand, c_operand, m_operand, d_z0_perm)
    du_runtime_args = du_runtime_artifacts.runtime_args
    db_runtime_args = db_runtime_artifacts.runtime_args
    dcdr_runtime_artifacts = _make_dcdr_runtime_artifacts(
        U_blk=U_blk,
        B_blk=B_blk,
        C_blk=C_blk,
        M_blk=M_blk,
        K_blk=K_blk,
        dOut_blk=dOut_blk,
        U_prev0_flat=U_prev0_flat,
        B_prev0_flat=B_prev0_flat,
        Z0_blk=Z0_blk,
        dC=d_c,
        d_logprefix=d_logprefix,
        d_r=d_r,
    )
    dcdr_runtime_args = dcdr_runtime_artifacts.runtime_args
    dlp_runtime_args = (
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        U_prev0_flat,
        B_prev0_flat,
        d_logprefix,
    )
    dlp_tensor_specs = _make_tensor_specs_from_tensors(*dlp_runtime_args)
    param_scan_runtime_args = param_scan_runtime_artifacts.runtime_args_for(
        M_blk, K_blk
    )
    dz0_alignments = tuple(_assumed_align(tensor) for tensor in dz0_runtime_args)
    du_alignments = du_runtime_artifacts.alignments
    db_alignments = db_runtime_artifacts.alignments
    dcdr_alignments = dcdr_runtime_artifacts.alignments
    dlp_alignments = tuple(_assumed_align(tensor) for tensor in dlp_runtime_args)
    param_scan_alignments = param_scan_runtime_artifacts.alignments_for(M_blk, K_blk)
    alignments = (
        dz0_alignments
        + du_alignments
        + db_alignments
        + dcdr_alignments
        + dlp_alignments
        + param_scan_alignments
    )
    keepalive = (
        U_tc,
        B_tc,
        C_tc,
        d_out_tc,
        M_f,
        K_f,
        chunk_starts_f,
        U_prev0,
        B_prev0,
        d_out_operand,
        c_operand,
        m_operand,
        d_z0_perm,
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        Z0_blk,
        U_prev0_flat,
        B_prev0_flat,
        *param_scan_runtime_artifacts.keepalive,
        *du_runtime_artifacts.workspace.tensors()[1:],
        db_workspace,
    )
    cache_key = _compiled_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        U_shape=tuple(U.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        d_out_shape=tuple(d_out.shape),
        chunk_size=L,
        has_prev=B_prev is not None,
        dz0_cta_tiler=dz0_cta_tiler,
        alignments=alignments,
        num_threads_du=num_threads_du,
        num_threads_db=num_threads_db,
        num_threads_dcdr=num_threads_dcdr,
        num_threads_param=num_threads_param,
    )

    compiled_kernels = _COMPILED_CACHE.get(cache_key)
    if compiled_kernels is None:
        dz0_wrapper = _make_dz0_host_wrapper(
            spec=(
                _make_tensor_spec_from_tensor(d_out_operand),
                _make_tensor_spec_from_tensor(c_operand),
                _make_tensor_spec_from_tensor(m_operand),
                _make_tensor_spec_from_tensor(d_z0_perm),
            ),
            cfg=(L, dz0_cta_tiler, H, G),
            cutlass_dtype=cutlass_dtype,
        )
        du_wrapper = _make_du_host_wrapper(
            spec=_make_tensor_specs_from_tensors(*du_runtime_args),
            cfg=(L, D, P, num_threads_du, H, G),
            cutlass_dtype=cutlass_dtype,
        )
        dcdr_wrapper = _make_dcdr_host_wrapper(
            tensor_specs=dcdr_runtime_artifacts.tensor_specs,
            kernel_cfg=(L, D, P, num_threads_dcdr, H, G),
            cutlass_dtype=cutlass_dtype,
        )
        dlp_wrapper = _make_dlp_host_wrapper(
            spec=dlp_tensor_specs,
            cfg=(L, D, P, num_threads_dlp, H, G),
            cutlass_dtype=cutlass_dtype,
        )
        param_scan_wrapper = _make_param_host_wrapper(
            spec=param_scan_runtime_artifacts.tensor_specs_for(M_blk, K_blk),
            cfg=(L, num_threads_param),
        )
        dz0_compile_args = _make_compile_args(dz0_runtime_args, dz0_alignments)
        du_compile_args = _make_compile_args(du_runtime_args, du_alignments)
        dlp_compile_args = _make_compile_args(dlp_runtime_args, dlp_alignments)
        param_scan_compile_args = param_scan_runtime_artifacts.compile_args_for(
            M_blk, K_blk
        )
        compiled_kernels = ChunkScanBwdCompiledKernels(
            dz0=cute.compile(
                dz0_wrapper,
                *dz0_compile_args,
                options="--enable-tvm-ffi",
            ),
            du=cute.compile(
                du_wrapper,
                *du_compile_args,
                options="--enable-tvm-ffi",
            ),
            db=_compile_db_wrapper(
                runtime_artifacts=db_runtime_artifacts,
                cfg=(L, D, P, num_threads_db, H, G),
                cutlass_dtype=cutlass_dtype,
            ),
            dcdr=cute.compile(
                dcdr_wrapper,
                *dcdr_runtime_artifacts.compile_args,
                options="--enable-tvm-ffi",
            ),
            dlp=cute.compile(
                dlp_wrapper,
                *dlp_compile_args,
                options="--enable-tvm-ffi",
            ),
            param_scan=cute.compile(
                param_scan_wrapper,
                *param_scan_compile_args,
                options="--enable-tvm-ffi",
            ),
        )
        _COMPILED_CACHE[cache_key] = compiled_kernels

    def _launch_dz0() -> None:
        _ = keepalive
        compiled_kernels.dz0(*dz0_runtime_args)

    def _launch_du() -> None:
        _ = keepalive
        compiled_kernels.du(*du_runtime_args)

    def _launch_db() -> None:
        _ = keepalive
        compiled_kernels.db(*db_runtime_args)

    def _launch_dcdr() -> None:
        _ = keepalive
        compiled_kernels.dcdr(*dcdr_runtime_args)

    def _launch_dlp() -> None:
        _ = keepalive
        compiled_kernels.dlp(*dlp_runtime_args)

    def _launch_param_scan() -> None:
        _ = keepalive
        compiled_kernels.param_scan(*param_scan_runtime_args)

    def launch() -> None:
        _launch_dz0()
        _launch_du()
        _launch_db()
        _launch_dcdr()
        _launch_dlp()
        _launch_param_scan()
        _record_tensors_on_current_stream(
            *(dz0_runtime_args + du_runtime_args + db_runtime_args),
            *(dcdr_runtime_args + dlp_runtime_args + param_scan_runtime_args),
        )

    outputs = ChunkScanBwdOutputs(
        bc_groups=G,
        chunk_start_grad=d_z0_view,
        value_grad_chunk=du_runtime_artifacts.workspace.value_grad_chunk.reshape(
            Bsz, H, n_chunks, L, P
        ),
        key_grad_chunk=db_runtime_artifacts.workspace.key_grad_chunk.reshape(
            Bsz, G, n_chunks, L, D
        ),
        value_boundary_grad=du_runtime_artifacts.workspace.value_boundary_grad.reshape(
            Bsz, H, n_chunks, P
        ),
        key_boundary_grad=db_runtime_artifacts.workspace.key_boundary_grad.reshape(
            Bsz, H, n_chunks, D
        ),
        logprefix_grad=d_logprefix.reshape(Bsz, H, n_chunks, L),
        query_grad_chunk=d_c.reshape(Bsz, G, n_chunks, L, D),
        rotation_grad=d_r.reshape(Bsz, H, n_chunks, L, 4),
        transition_grad=param_scan_runtime_artifacts.transition_view,
        tap_prev_grad=param_scan_runtime_artifacts.tap_prev_view,
        tap_curr_grad=param_scan_runtime_artifacts.tap_curr_view,
    )
    launchers = ChunkScanBwdStageLaunchers(
        dz0=_launch_dz0,
        du=_launch_du,
        db=_launch_db,
        dcdr=_launch_dcdr,
        dlp=_launch_dlp,
        param_scan=_launch_param_scan,
    )
    return PreparedChunkScanBwdLaunch(
        compiled=compiled_kernels,
        outputs=outputs,
        launchers=launchers,
        launch=launch,
    )


def chunk_scan_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_prev_grads: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Thin public wrapper over the compiled chunk-scan backward kernel bundle."""
    prepared = compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    prepared.launch()
    return prepared.outputs.public_outputs(
        T=U.shape[2],
        value_dtype=U.dtype,
        key_dtype=B.dtype,
        query_dtype=C.dtype,
        return_prev_grads=return_prev_grads,
    )


__all__ = [
    "chunk_scan_bwd_cute",
    "compile_chunk_scan_bwd_kernels",
]
