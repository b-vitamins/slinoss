"""CuTe backward kernels for the ``v2x2ssd`` chunk-increment stage.

This module owns the host-side CuTe wrappers for the four chunk-increment
backward kernels:

- interior ``dB``
- interior ``dU``
- chunk-boundary carry gradients
- parameter scan

The public entrypoint ``chunk_increment_bwd_cute`` compiles or reuses those
wrappers, launches the fused stage path, and then folds the per-chunk boundary
carries back into the public ``(B, H, T, F)`` gradient views.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import cutlass.cute as cute

from ....common import _tc_input_dtype
from ...fwd.common import _make_fake_tensor_arg
from .boundary import ChunkIncrementBwdBoundaryAmpere
from .common import _assumed_align, _torch_to_cutlass_dtype
from .db import ChunkIncrementBwdDBAmpere
from .du import ChunkIncrementBwdDUAmpere
from .param_scan import ChunkIncrementBwdParamScanAmpere


TensorSpec = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class ChunkIncrementBwdBoundaryTensorSpecs:
    d_inc_boundary: TensorSpec
    prev_b: TensorSpec
    prev_u: TensorSpec
    transition: TensorSpec
    d_mp0: TensorSpec


@dataclass(frozen=True)
class ChunkIncrementBwdBoundaryOutputs:
    db_prev: torch.Tensor
    du_prev: torch.Tensor
    d_mp0: torch.Tensor


@dataclass(frozen=True)
class ChunkIncrementBwdRuntimeArtifacts:
    runtime_args: tuple[torch.Tensor, ...]
    alignments: tuple[int, ...]
    compile_args: tuple[object, ...]


@dataclass(frozen=True)
class ChunkIncrementBwdCompiledBundle:
    db: object
    du: object
    boundary: object
    param_scan: object
    stage: object


@dataclass(frozen=True)
class ChunkIncrementBwdInputInfo:
    batch_size: int
    heads: int
    time_steps: int
    P: int
    D: int
    chunk_size: int
    n_chunks: int
    padded_time: int
    batch_head_count: int
    batch_head_chunk_count: int
    device_index: int
    tc_dtype: torch.dtype
    cutlass_dtype: object


@dataclass(frozen=True)
class ChunkIncrementBwdPreparedInputs:
    u_padded: torch.Tensor
    b_padded: torch.Tensor
    m_padded: torch.Tensor
    kprev_chunk: torch.Tensor
    kcurr_chunk: torch.Tensor
    d_inc_chunk: torch.Tensor
    d_m_chunk: torch.Tensor
    b_prev_chunks: torch.Tensor
    u_prev_chunks: torch.Tensor


@dataclass(frozen=True)
class ChunkIncrementBwdWorkspace:
    d_b_chunk: torch.Tensor
    d_u_chunk: torch.Tensor
    boundary: ChunkIncrementBwdBoundaryOutputs
    d_msum_part: torch.Tensor
    d_m: torch.Tensor
    d_kprev: torch.Tensor
    d_kcurr: torch.Tensor


@dataclass(frozen=True)
class ChunkIncrementBwdOutputs:
    d_b_chunk: torch.Tensor
    d_u_chunk: torch.Tensor
    d_b_prev: torch.Tensor
    d_u_prev: torch.Tensor
    d_msum_part: torch.Tensor
    d_mp0: torch.Tensor
    d_m: torch.Tensor
    d_kprev: torch.Tensor
    d_kcurr: torch.Tensor


@dataclass(frozen=True)
class ChunkIncrementBwdRuntimeBundle:
    db: ChunkIncrementBwdRuntimeArtifacts
    du: ChunkIncrementBwdRuntimeArtifacts
    boundary: ChunkIncrementBwdRuntimeArtifacts
    param_scan: ChunkIncrementBwdRuntimeArtifacts
    stage: ChunkIncrementBwdRuntimeArtifacts


@dataclass(frozen=True)
class PreparedChunkIncrementBwdLaunch:
    compiled: ChunkIncrementBwdCompiledBundle
    outputs: ChunkIncrementBwdOutputs
    launch: Callable[[], None]


_COMPILED_CACHE: dict[tuple, ChunkIncrementBwdCompiledBundle] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_PREV_CACHE_LIMIT = 8


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


def _cache_set(cache: dict, key: tuple, value, *, limit: int) -> None:
    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= int(limit):
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def _get_zero_prev_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    heads: int,
    P: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        dtype,
        int(batch_size),
        int(heads),
        int(P),
        int(D),
    )
    cached = _ZERO_PREV_CACHE.get(key)
    if cached is None:
        cached = (
            torch.zeros((batch_size, heads, D), device=device, dtype=dtype),
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
        )
        _cache_set(_ZERO_PREV_CACHE, key, cached, limit=_ZERO_PREV_CACHE_LIMIT)
    return cached


def _chunk_increment_device_label(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return f"{props.name} (sm_{props.major}{props.minor})"


def _pad_zero_time(
    tensor: torch.Tensor,
    *,
    padded_time: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.to(dtype=dtype).contiguous()
    time_steps = int(tensor.shape[2])
    if time_steps == padded_time:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[2] = padded_time - time_steps
    pad = torch.zeros(pad_shape, device=tensor.device, dtype=dtype)
    return torch.cat((tensor, pad), dim=2).contiguous()


def _pad_m_identity(
    transition: torch.Tensor,
    *,
    padded_time: int,
) -> torch.Tensor:
    transition = transition.to(dtype=torch.float32).contiguous()
    time_steps = int(transition.shape[2])
    if time_steps == padded_time:
        return transition
    pad_shape = list(transition.shape)
    pad_shape[2] = padded_time - time_steps
    pad = torch.zeros(pad_shape, device=transition.device, dtype=torch.float32)
    pad[..., 0] = 1.0
    return torch.cat((transition, pad), dim=2).contiguous()


def _public_from_chunked_output(
    x: torch.Tensor,
    *,
    time_steps: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    batch_size, heads, n_chunks, chunk_size, feature_size = map(int, x.shape)
    out = x.reshape(batch_size, heads, n_chunks * chunk_size, feature_size)[
        :, :, :time_steps, :
    ]
    target_dtype = out.dtype if dtype is None else dtype
    if out.dtype != target_dtype:
        out = out.to(dtype=target_dtype)
    return out.contiguous()


def _public_dk_from_parts(
    d_kprev: torch.Tensor,
    d_kcurr: torch.Tensor,
    *,
    time_steps: int,
) -> torch.Tensor:
    if d_kprev.shape != d_kcurr.shape:
        raise ValueError("d_kprev and d_kcurr must have identical shapes.")
    d_k = torch.stack((d_kprev, d_kcurr), dim=4)
    batch_size, heads, n_chunks, chunk_size, _, feature_size = map(int, d_k.shape)
    return (
        d_k.reshape(batch_size, heads, n_chunks * chunk_size, 2, feature_size)[
            :, :, :time_steps, :, :
        ]
        .to(dtype=torch.float32)
        .contiguous()
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


def _chunk_increment_bwd_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    d_inc_shape: tuple[int, ...],
    d_m_chunk_shape: tuple[int, ...],
    chunk_size: int,
    has_prev: bool,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "chunk_increment_bwd",
        device_index,
        tc_dtype,
        U_shape,
        B_shape,
        M_shape,
        K_shape,
        d_inc_shape,
        d_m_chunk_shape,
        int(chunk_size),
        has_prev,
        alignments,
    )


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
) -> TensorSpec:
    shape = tuple(int(dim) for dim in shape)
    if stride is None:
        stride = _make_row_major_stride(shape)
    else:
        stride = tuple(int(step) for step in stride)
    return shape, stride


def _make_tensor_from_spec(
    tensor: cute.Tensor,
    spec: TensorSpec,
):
    shape, stride = spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _make_boundary_tensor_specs(
    problem_shape: tuple[int, ...],
) -> ChunkIncrementBwdBoundaryTensorSpecs:
    L, P, D, BHC = problem_shape
    return ChunkIncrementBwdBoundaryTensorSpecs(
        d_inc_boundary=_make_tensor_spec((BHC, P, D), stride=(P * D, D, 1)),
        prev_b=_make_tensor_spec((D, BHC), stride=(1, D)),
        prev_u=_make_tensor_spec((P, BHC), stride=(1, P)),
        transition=_make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2)),
        d_mp0=_make_tensor_spec((2, BHC), stride=(BHC, 1)),
    )


def _prepare_boundary_prev_chunks(
    *,
    U_tc: torch.Tensor,
    B_tc: torch.Tensor,
    U_prev0: torch.Tensor,
    B_prev0: torch.Tensor,
    batch_head_count: int,
    n_chunks: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    P = int(U_tc.shape[-1])
    D = int(B_tc.shape[-1])
    U_chunks = U_tc.reshape(batch_head_count, n_chunks, chunk_size, P)
    B_chunks = B_tc.reshape(batch_head_count, n_chunks, chunk_size, D)

    U_prev_chunks = torch.empty(
        (batch_head_count, n_chunks, P), device=U_tc.device, dtype=U_tc.dtype
    )
    B_prev_chunks = torch.empty(
        (batch_head_count, n_chunks, D), device=B_tc.device, dtype=B_tc.dtype
    )
    U_prev_chunks[:, 0, :] = U_prev0.reshape(batch_head_count, P)
    B_prev_chunks[:, 0, :] = B_prev0.reshape(batch_head_count, D)
    if n_chunks > 1:
        U_prev_chunks[:, 1:, :] = U_chunks[:, :-1, -1, :]
        B_prev_chunks[:, 1:, :] = B_chunks[:, :-1, -1, :]
    return B_prev_chunks, U_prev_chunks


def _allocate_boundary_outputs(
    *,
    batch_head_chunk_count: int,
    D: int,
    P: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ChunkIncrementBwdBoundaryOutputs:
    return ChunkIncrementBwdBoundaryOutputs(
        db_prev=torch.empty((batch_head_chunk_count, D), device=device, dtype=dtype),
        du_prev=torch.empty((batch_head_chunk_count, P), device=device, dtype=dtype),
        d_mp0=torch.empty(
            (2, batch_head_chunk_count), device=device, dtype=torch.float32
        ),
    )


def _make_runtime_artifacts(
    *runtime_args: torch.Tensor,
) -> ChunkIncrementBwdRuntimeArtifacts:
    alignments = tuple(_assumed_align(tensor) for tensor in runtime_args)
    compile_args = tuple(
        _make_fake_tensor_arg(tensor, align=align)
        for tensor, align in zip(runtime_args, alignments, strict=True)
    )
    return ChunkIncrementBwdRuntimeArtifacts(
        runtime_args=runtime_args,
        alignments=alignments,
        compile_args=compile_args,
    )


def _make_boundary_runtime_artifacts(
    *,
    d_inc_tc: torch.Tensor,
    b_prev_chunks: torch.Tensor,
    u_prev_chunks: torch.Tensor,
    transition_chunk: torch.Tensor,
    kprev_chunk: torch.Tensor,
    d_u_prev: torch.Tensor,
    d_b_prev: torch.Tensor,
    d_mp0: torch.Tensor,
) -> ChunkIncrementBwdRuntimeArtifacts:
    return _make_runtime_artifacts(
        d_inc_tc,
        b_prev_chunks,
        u_prev_chunks,
        transition_chunk,
        kprev_chunk,
        d_u_prev,
        d_b_prev,
        d_mp0,
    )


def _make_db_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, ...],
    cutlass_dtype,
):
    L, P, D, BHC = problem_shape
    chunk_size, n_d_tiles = launch_cfg

    u_spec = _make_tensor_spec((L, P, BHC), stride=(P, 1, L * P))
    b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_inc_dp_spec = _make_tensor_spec((D, P, BHC), stride=(1, D, P * D))
    d_b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    d_msum_part_spec = _make_tensor_spec(
        (2, L, n_d_tiles, BHC),
        stride=(L * n_d_tiles * BHC, n_d_tiles * BHC, BHC, 1),
    )

    @cute.jit
    def _db_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DIncDP_ptr: cute.Tensor,
        DB_ptr: cute.Tensor,
        DMsumPart_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDIncDP = _make_tensor_from_spec(DIncDP_ptr, d_inc_dp_spec)
        mDB = _make_tensor_from_spec(DB_ptr, d_b_spec)
        mDMsumPart = _make_tensor_from_spec(DMsumPart_ptr, d_msum_part_spec)

        kernel = ChunkIncrementBwdDBAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        kernel(mU, mB, mM, mKprev, mKcurr, mDIncDP, mDB, mDMsumPart)

    return _db_host_wrapper


def _make_du_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, ...],
    cutlass_dtype,
):
    L, P, D, BHC = problem_shape
    (chunk_size,) = launch_cfg

    d_inc_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))
    b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_u_spec = _make_tensor_spec((P, L, BHC), stride=(1, P, L * P))

    @cute.jit
    def _du_host_wrapper(
        DInc_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DU_ptr: cute.Tensor,
    ):
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDU = _make_tensor_from_spec(DU_ptr, d_u_spec)

        kernel = ChunkIncrementBwdDUAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        kernel(mDInc, mB, mM, mKprev, mKcurr, mDU)

    return _du_host_wrapper


def _make_boundary_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, ...],
    cutlass_dtype,
):
    _L, P, D, _BHC = problem_shape
    (chunk_size,) = launch_cfg

    boundary_specs = _make_boundary_tensor_specs(problem_shape)

    @cute.jit
    def _boundary_host_wrapper(
        DInc_ptr: cute.Tensor,
        BPrev_ptr: cute.Tensor,
        UPrev_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        DUPrev_ptr: cute.Tensor,
        DBPrev_ptr: cute.Tensor,
        DMp0_ptr: cute.Tensor,
    ):
        mDInc = _make_tensor_from_spec(DInc_ptr, boundary_specs.d_inc_boundary)
        mBPrev = _make_tensor_from_spec(BPrev_ptr, boundary_specs.prev_b)
        mUPrev = _make_tensor_from_spec(UPrev_ptr, boundary_specs.prev_u)
        mM = _make_tensor_from_spec(M_ptr, boundary_specs.transition)
        mKprev = _make_tensor_from_spec(Kprev_ptr, boundary_specs.transition)
        mDUPrev = _make_tensor_from_spec(DUPrev_ptr, boundary_specs.prev_u)
        mDBPrev = _make_tensor_from_spec(DBPrev_ptr, boundary_specs.prev_b)
        mDMp0 = _make_tensor_from_spec(DMp0_ptr, boundary_specs.d_mp0)

        kernel = ChunkIncrementBwdBoundaryAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        kernel(mDInc, mBPrev, mUPrev, mM, mKprev, mDUPrev, mDBPrev, mDMp0)

    return _boundary_host_wrapper


def _make_param_scan_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, ...],
):
    L, BHC = problem_shape
    (chunk_size, n_d_tiles) = launch_cfg

    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_msum_part_spec = _make_tensor_spec(
        (2, L, n_d_tiles, BHC),
        stride=(L * n_d_tiles * BHC, n_d_tiles * BHC, BHC, 1),
    )
    d_mp0_spec = _make_tensor_spec((2, BHC), stride=(BHC, 1))
    d_mchunk_spec = _make_tensor_spec((2, BHC), stride=(1, 2))
    d_param_spec = _make_tensor_spec((2, L, BHC), stride=(L * BHC, BHC, 1))

    @cute.jit
    def _param_host_wrapper(
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DMsumPart_ptr: cute.Tensor,
        DMp0_ptr: cute.Tensor,
        DMchunk_ptr: cute.Tensor,
        DM_ptr: cute.Tensor,
        DKprev_ptr: cute.Tensor,
        DKcurr_ptr: cute.Tensor,
    ):
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDMsumPart = _make_tensor_from_spec(DMsumPart_ptr, d_msum_part_spec)
        mDMp0 = _make_tensor_from_spec(DMp0_ptr, d_mp0_spec)
        mDMchunk = _make_tensor_from_spec(DMchunk_ptr, d_mchunk_spec)
        mDM = _make_tensor_from_spec(DM_ptr, d_param_spec)
        mDKprev = _make_tensor_from_spec(DKprev_ptr, d_param_spec)
        mDKcurr = _make_tensor_from_spec(DKcurr_ptr, d_param_spec)

        kernel = ChunkIncrementBwdParamScanAmpere(
            chunk_size=chunk_size,
            n_d_tiles=n_d_tiles,
        )
        kernel(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
        )

    return _param_host_wrapper


def _make_stage_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, ...],
    cutlass_dtype,
):
    L, P, D, BHC = problem_shape
    chunk_size, n_d_tiles = launch_cfg

    u_spec = _make_tensor_spec((L, P, BHC), stride=(P, 1, L * P))
    b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_inc_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))
    d_inc_dp_spec = _make_tensor_spec((D, P, BHC), stride=(1, D, P * D))
    boundary_specs = _make_boundary_tensor_specs(problem_shape)
    d_b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    d_u_spec = _make_tensor_spec((P, L, BHC), stride=(1, P, L * P))
    d_msum_part_spec = _make_tensor_spec(
        (2, L, n_d_tiles, BHC),
        stride=(L * n_d_tiles * BHC, n_d_tiles * BHC, BHC, 1),
    )
    d_mchunk_spec = _make_tensor_spec((2, BHC), stride=(1, 2))
    d_param_spec = _make_tensor_spec((2, L, BHC), stride=(L * BHC, BHC, 1))

    @cute.jit
    def _stage_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DInc_ptr: cute.Tensor,
        BPrev_ptr: cute.Tensor,
        UPrev_ptr: cute.Tensor,
        DB_ptr: cute.Tensor,
        DU_ptr: cute.Tensor,
        DBPrev_ptr: cute.Tensor,
        DUPrev_ptr: cute.Tensor,
        DMsumPart_ptr: cute.Tensor,
        DMp0_ptr: cute.Tensor,
        DMchunk_ptr: cute.Tensor,
        DM_ptr: cute.Tensor,
        DKprev_ptr: cute.Tensor,
        DKcurr_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_spec)
        mDIncDP = _make_tensor_from_spec(DInc_ptr, d_inc_dp_spec)
        mDIncBoundary = _make_tensor_from_spec(DInc_ptr, boundary_specs.d_inc_boundary)
        mBPrev = _make_tensor_from_spec(BPrev_ptr, boundary_specs.prev_b)
        mUPrev = _make_tensor_from_spec(UPrev_ptr, boundary_specs.prev_u)
        mDB = _make_tensor_from_spec(DB_ptr, d_b_spec)
        mDU = _make_tensor_from_spec(DU_ptr, d_u_spec)
        mDBPrev = _make_tensor_from_spec(DBPrev_ptr, boundary_specs.prev_b)
        mDUPrev = _make_tensor_from_spec(DUPrev_ptr, boundary_specs.prev_u)
        mDMsumPart = _make_tensor_from_spec(DMsumPart_ptr, d_msum_part_spec)
        mDMp0 = _make_tensor_from_spec(DMp0_ptr, boundary_specs.d_mp0)
        mDMchunk = _make_tensor_from_spec(DMchunk_ptr, d_mchunk_spec)
        mDM = _make_tensor_from_spec(DM_ptr, d_param_spec)
        mDKprev = _make_tensor_from_spec(DKprev_ptr, d_param_spec)
        mDKcurr = _make_tensor_from_spec(DKcurr_ptr, d_param_spec)

        k_db = ChunkIncrementBwdDBAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        k_du = ChunkIncrementBwdDUAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        k_boundary = ChunkIncrementBwdBoundaryAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        k_param_scan = ChunkIncrementBwdParamScanAmpere(
            chunk_size=chunk_size,
            n_d_tiles=n_d_tiles,
        )

        k_db(mU, mB, mM, mKprev, mKcurr, mDIncDP, mDB, mDMsumPart)
        k_du(mDInc, mB, mM, mKprev, mKcurr, mDU)
        k_boundary(mDIncBoundary, mBPrev, mUPrev, mM, mKprev, mDUPrev, mDBPrev, mDMp0)
        k_param_scan(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
        )

    return _stage_host_wrapper


def _validate_chunk_increment_bwd_prev_inputs(
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> None:
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")


def _make_chunk_increment_bwd_input_info(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
) -> ChunkIncrementBwdInputInfo:
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")

    batch_size, heads, time_steps, P = map(int, U.shape)
    D = int(B.shape[-1])
    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
    )
    if B.shape != (batch_size, heads, time_steps, D):
        raise ValueError("B must be (B,H,T,D) matching U.")
    if M.shape != (batch_size, heads, time_steps, 2):
        raise ValueError(f"M must be (B,H,T,2)={(batch_size, heads, time_steps, 2)}.")
    if K.shape != (batch_size, heads, time_steps, 2, 2):
        raise ValueError(
            f"K must be (B,H,T,2,2)={(batch_size, heads, time_steps, 2, 2)}."
        )
    if D % 2 != 0:
        raise ValueError("D must be divisible by 2 (flattened 2N).")

    resolved_chunk_size = int(chunk_size)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
    padded_time = n_chunks * resolved_chunk_size

    if tuple(d_inc.shape) != (batch_size, heads, n_chunks, P, D):
        raise ValueError(
            f"d_inc must be (B,H,C,P,D)={(batch_size, heads, n_chunks, P, D)}. "
            f"Got {tuple(d_inc.shape)}."
        )
    if tuple(d_m_chunk.shape) != (batch_size, heads, n_chunks, 2):
        raise ValueError(
            f"d_m_chunk must be (B,H,C,2)={(batch_size, heads, n_chunks, 2)}. "
            f"Got {tuple(d_m_chunk.shape)}."
        )

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    return ChunkIncrementBwdInputInfo(
        batch_size=batch_size,
        heads=heads,
        time_steps=time_steps,
        P=P,
        D=D,
        chunk_size=resolved_chunk_size,
        n_chunks=n_chunks,
        padded_time=padded_time,
        batch_head_count=batch_size * heads,
        batch_head_chunk_count=batch_size * heads * n_chunks,
        device_index=device_index,
        tc_dtype=tc_dtype,
        cutlass_dtype=_torch_to_cutlass_dtype(tc_dtype),
    )


def _make_chunk_increment_bwd_prev_state(
    *,
    input_info: ChunkIncrementBwdInputInfo,
    device: torch.device,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if B_prev is None:
        return _get_zero_prev_tensors(
            device=device,
            dtype=input_info.tc_dtype,
            batch_size=input_info.batch_size,
            heads=input_info.heads,
            P=input_info.P,
            D=input_info.D,
        )

    if B_prev.shape != (input_info.batch_size, input_info.heads, input_info.D):
        raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
    if U_prev is None or U_prev.shape != (
        input_info.batch_size,
        input_info.heads,
        input_info.P,
    ):
        raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
    return (
        B_prev.to(dtype=input_info.tc_dtype).contiguous(),
        U_prev.to(dtype=input_info.tc_dtype).contiguous(),
    )


def _make_chunk_increment_bwd_prepared_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    input_info: ChunkIncrementBwdInputInfo,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> ChunkIncrementBwdPreparedInputs:
    u_padded = _pad_zero_time(
        U, padded_time=input_info.padded_time, dtype=input_info.tc_dtype
    )
    b_padded = _pad_zero_time(
        B, padded_time=input_info.padded_time, dtype=input_info.tc_dtype
    )
    m_padded = _pad_m_identity(M, padded_time=input_info.padded_time)
    k_padded = _pad_zero_time(
        K,
        padded_time=input_info.padded_time,
        dtype=torch.float32,
    )
    d_inc_chunk = d_inc.to(dtype=input_info.tc_dtype).contiguous()
    d_m_chunk_f32 = d_m_chunk.to(dtype=torch.float32).contiguous()

    b_prev0, u_prev0 = _make_chunk_increment_bwd_prev_state(
        input_info=input_info,
        device=U.device,
        B_prev=B_prev,
        U_prev=U_prev,
    )

    k_chunks = k_padded.reshape(
        input_info.batch_head_count,
        input_info.n_chunks,
        input_info.chunk_size,
        2,
        2,
    )
    kprev_chunk = (
        k_chunks[..., 0, :]
        .reshape(input_info.batch_head_chunk_count, input_info.chunk_size, 2)
        .contiguous()
    )
    kcurr_chunk = (
        k_chunks[..., 1, :]
        .reshape(input_info.batch_head_chunk_count, input_info.chunk_size, 2)
        .contiguous()
    )

    b_prev_chunks, u_prev_chunks = _prepare_boundary_prev_chunks(
        U_tc=u_padded,
        B_tc=b_padded,
        U_prev0=u_prev0,
        B_prev0=b_prev0,
        batch_head_count=input_info.batch_head_count,
        n_chunks=input_info.n_chunks,
        chunk_size=input_info.chunk_size,
    )

    return ChunkIncrementBwdPreparedInputs(
        u_padded=u_padded,
        b_padded=b_padded,
        m_padded=m_padded,
        kprev_chunk=kprev_chunk,
        kcurr_chunk=kcurr_chunk,
        d_inc_chunk=d_inc_chunk,
        d_m_chunk=d_m_chunk_f32,
        b_prev_chunks=b_prev_chunks,
        u_prev_chunks=u_prev_chunks,
    )


def _validate_chunk_increment_bwd_support(
    input_info: ChunkIncrementBwdInputInfo,
) -> int:
    db_kernel = ChunkIncrementBwdDBAmpere(
        input_info.cutlass_dtype,
        chunk_size=input_info.chunk_size,
        D=input_info.D,
        P=input_info.P,
    )
    db_info = db_kernel.support_info(
        input_info.cutlass_dtype,
        device_index=input_info.device_index,
    )
    if not db_info.supported:
        device_label = _chunk_increment_device_label(input_info.device_index)
        raise ValueError(
            "ChunkIncrementBwdDBAmpere requires "
            f"{db_info.required_smem_bytes} bytes of shared memory, but "
            f"{device_label} only exposes {db_info.smem_capacity_bytes} bytes."
        )

    n_d_tiles = (input_info.D + db_kernel.bN - 1) // db_kernel.bN

    du_kernel = ChunkIncrementBwdDUAmpere(
        input_info.cutlass_dtype,
        chunk_size=input_info.chunk_size,
        D=input_info.D,
        P=input_info.P,
    )
    du_info = du_kernel.support_info(
        input_info.cutlass_dtype,
        device_index=input_info.device_index,
    )
    if not du_info.supported:
        device_label = _chunk_increment_device_label(input_info.device_index)
        raise ValueError(
            "ChunkIncrementBwdDUAmpere requires "
            f"{du_info.required_smem_bytes} bytes of shared memory, but "
            f"{device_label} only exposes {du_info.smem_capacity_bytes} bytes."
        )

    boundary_kernel = ChunkIncrementBwdBoundaryAmpere(
        input_info.cutlass_dtype,
        chunk_size=input_info.chunk_size,
        D=input_info.D,
        P=input_info.P,
    )
    boundary_info = boundary_kernel.support_info(device_index=input_info.device_index)
    if not boundary_info.supported:
        device_label = _chunk_increment_device_label(input_info.device_index)
        raise ValueError(
            "ChunkIncrementBwdBoundaryAmpere requires "
            f"{boundary_info.required_smem_bytes} bytes of shared memory, but "
            f"{device_label} only exposes {boundary_info.smem_capacity_bytes} bytes."
        )

    param_kernel = ChunkIncrementBwdParamScanAmpere(
        chunk_size=input_info.chunk_size,
        n_d_tiles=n_d_tiles,
    )
    param_info = param_kernel.support_info(device_index=input_info.device_index)
    if not param_info.supported:
        device_label = _chunk_increment_device_label(input_info.device_index)
        raise ValueError(
            "ChunkIncrementBwdParamScanAmpere requires "
            f"{param_info.required_smem_bytes} bytes of shared memory, but "
            f"{device_label} only exposes {param_info.smem_capacity_bytes} bytes."
        )

    return n_d_tiles


def _make_chunk_increment_bwd_workspace(
    *,
    input_info: ChunkIncrementBwdInputInfo,
    device: torch.device,
    n_d_tiles: int,
) -> ChunkIncrementBwdWorkspace:
    return ChunkIncrementBwdWorkspace(
        d_b_chunk=torch.empty(
            (input_info.batch_head_chunk_count, input_info.chunk_size, input_info.D),
            device=device,
            dtype=input_info.tc_dtype,
        ),
        d_u_chunk=torch.empty(
            (input_info.batch_head_chunk_count, input_info.chunk_size, input_info.P),
            device=device,
            dtype=input_info.tc_dtype,
        ),
        boundary=_allocate_boundary_outputs(
            batch_head_chunk_count=input_info.batch_head_chunk_count,
            D=input_info.D,
            P=input_info.P,
            device=device,
            dtype=input_info.tc_dtype,
        ),
        d_msum_part=torch.empty(
            (2, input_info.chunk_size, n_d_tiles, input_info.batch_head_chunk_count),
            device=device,
            dtype=torch.float32,
        ),
        d_m=torch.empty(
            (2, input_info.chunk_size, input_info.batch_head_chunk_count),
            device=device,
            dtype=torch.float32,
        ),
        d_kprev=torch.empty(
            (2, input_info.chunk_size, input_info.batch_head_chunk_count),
            device=device,
            dtype=torch.float32,
        ),
        d_kcurr=torch.empty(
            (2, input_info.chunk_size, input_info.batch_head_chunk_count),
            device=device,
            dtype=torch.float32,
        ),
    )


def _make_chunk_increment_bwd_runtime_bundle(
    prepared_inputs: ChunkIncrementBwdPreparedInputs,
    workspace: ChunkIncrementBwdWorkspace,
) -> ChunkIncrementBwdRuntimeBundle:
    return ChunkIncrementBwdRuntimeBundle(
        db=_make_runtime_artifacts(
            prepared_inputs.u_padded,
            prepared_inputs.b_padded,
            prepared_inputs.m_padded,
            prepared_inputs.kprev_chunk,
            prepared_inputs.kcurr_chunk,
            prepared_inputs.d_inc_chunk,
            workspace.d_b_chunk,
            workspace.d_msum_part,
        ),
        du=_make_runtime_artifacts(
            prepared_inputs.d_inc_chunk,
            prepared_inputs.b_padded,
            prepared_inputs.m_padded,
            prepared_inputs.kprev_chunk,
            prepared_inputs.kcurr_chunk,
            workspace.d_u_chunk,
        ),
        boundary=_make_boundary_runtime_artifacts(
            d_inc_tc=prepared_inputs.d_inc_chunk,
            b_prev_chunks=prepared_inputs.b_prev_chunks,
            u_prev_chunks=prepared_inputs.u_prev_chunks,
            transition_chunk=prepared_inputs.m_padded,
            kprev_chunk=prepared_inputs.kprev_chunk,
            d_u_prev=workspace.boundary.du_prev,
            d_b_prev=workspace.boundary.db_prev,
            d_mp0=workspace.boundary.d_mp0,
        ),
        param_scan=_make_runtime_artifacts(
            prepared_inputs.m_padded,
            prepared_inputs.kprev_chunk,
            prepared_inputs.kcurr_chunk,
            workspace.d_msum_part,
            workspace.boundary.d_mp0,
            prepared_inputs.d_m_chunk,
            workspace.d_m,
            workspace.d_kprev,
            workspace.d_kcurr,
        ),
        stage=_make_runtime_artifacts(
            prepared_inputs.u_padded,
            prepared_inputs.b_padded,
            prepared_inputs.m_padded,
            prepared_inputs.kprev_chunk,
            prepared_inputs.kcurr_chunk,
            prepared_inputs.d_inc_chunk,
            prepared_inputs.b_prev_chunks,
            prepared_inputs.u_prev_chunks,
            workspace.d_b_chunk,
            workspace.d_u_chunk,
            workspace.boundary.db_prev,
            workspace.boundary.du_prev,
            workspace.d_msum_part,
            workspace.boundary.d_mp0,
            prepared_inputs.d_m_chunk,
            workspace.d_m,
            workspace.d_kprev,
            workspace.d_kcurr,
        ),
    )


def _get_compiled_chunk_increment_bwd_bundle(
    *,
    input_info: ChunkIncrementBwdInputInfo,
    cache_key: tuple,
    n_d_tiles: int,
    runtime_bundle: ChunkIncrementBwdRuntimeBundle,
) -> ChunkIncrementBwdCompiledBundle:
    cached = _COMPILED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    problem_shape = (
        input_info.chunk_size,
        input_info.P,
        input_info.D,
        input_info.batch_head_chunk_count,
    )
    db_wrapper = _make_db_host_wrapper(
        problem_shape=problem_shape,
        launch_cfg=(input_info.chunk_size, n_d_tiles),
        cutlass_dtype=input_info.cutlass_dtype,
    )
    du_wrapper = _make_du_host_wrapper(
        problem_shape=problem_shape,
        launch_cfg=(input_info.chunk_size,),
        cutlass_dtype=input_info.cutlass_dtype,
    )
    boundary_wrapper = _make_boundary_host_wrapper(
        problem_shape=problem_shape,
        launch_cfg=(input_info.chunk_size,),
        cutlass_dtype=input_info.cutlass_dtype,
    )
    param_scan_wrapper = _make_param_scan_host_wrapper(
        problem_shape=(input_info.chunk_size, input_info.batch_head_chunk_count),
        launch_cfg=(input_info.chunk_size, n_d_tiles),
    )
    stage_wrapper = _make_stage_host_wrapper(
        problem_shape=problem_shape,
        launch_cfg=(input_info.chunk_size, n_d_tiles),
        cutlass_dtype=input_info.cutlass_dtype,
    )
    cached = ChunkIncrementBwdCompiledBundle(
        db=cute.compile(
            db_wrapper,
            *runtime_bundle.db.compile_args,
            options="--enable-tvm-ffi",
        ),
        du=cute.compile(
            du_wrapper,
            *runtime_bundle.du.compile_args,
            options="--enable-tvm-ffi",
        ),
        boundary=cute.compile(
            boundary_wrapper,
            *runtime_bundle.boundary.compile_args,
            options="--enable-tvm-ffi",
        ),
        param_scan=cute.compile(
            param_scan_wrapper,
            *runtime_bundle.param_scan.compile_args,
            options="--enable-tvm-ffi",
        ),
        stage=cute.compile(
            stage_wrapper,
            *runtime_bundle.stage.compile_args,
            options="--enable-tvm-ffi",
        ),
    )
    _COMPILED_CACHE[cache_key] = cached
    return cached


def _make_chunk_increment_bwd_outputs(
    workspace: ChunkIncrementBwdWorkspace,
    *,
    input_info: ChunkIncrementBwdInputInfo,
    n_d_tiles: int,
) -> ChunkIncrementBwdOutputs:
    return ChunkIncrementBwdOutputs(
        d_b_chunk=workspace.d_b_chunk.reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.chunk_size,
            input_info.D,
        ),
        d_u_chunk=workspace.d_u_chunk.reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.chunk_size,
            input_info.P,
        ),
        d_b_prev=workspace.boundary.db_prev.reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.D,
        ),
        d_u_prev=workspace.boundary.du_prev.reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.P,
        ),
        d_msum_part=workspace.d_msum_part.permute(3, 1, 2, 0).reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.chunk_size,
            n_d_tiles,
            2,
        ),
        d_mp0=workspace.boundary.d_mp0.permute(1, 0).reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            2,
        ),
        d_m=workspace.d_m.permute(2, 1, 0).reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.chunk_size,
            2,
        ),
        d_kprev=workspace.d_kprev.permute(2, 1, 0).reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.chunk_size,
            2,
        ),
        d_kcurr=workspace.d_kcurr.permute(2, 1, 0).reshape(
            input_info.batch_size,
            input_info.heads,
            input_info.n_chunks,
            input_info.chunk_size,
            2,
        ),
    )


def _make_chunk_increment_bwd_prepared_launch(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
) -> PreparedChunkIncrementBwdLaunch:
    _validate_chunk_increment_bwd_prev_inputs(B_prev, U_prev)
    input_info = _make_chunk_increment_bwd_input_info(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
    )
    prepared_inputs = _make_chunk_increment_bwd_prepared_inputs(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        input_info=input_info,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    n_d_tiles = _validate_chunk_increment_bwd_support(input_info)
    workspace = _make_chunk_increment_bwd_workspace(
        input_info=input_info,
        device=U.device,
        n_d_tiles=n_d_tiles,
    )
    runtime_bundle = _make_chunk_increment_bwd_runtime_bundle(
        prepared_inputs,
        workspace,
    )
    cache_key = _chunk_increment_bwd_key(
        device_index=input_info.device_index,
        tc_dtype=input_info.tc_dtype,
        U_shape=tuple(U.shape),
        B_shape=tuple(B.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        d_inc_shape=tuple(d_inc.shape),
        d_m_chunk_shape=tuple(d_m_chunk.shape),
        chunk_size=input_info.chunk_size,
        has_prev=B_prev is not None,
        alignments=runtime_bundle.stage.alignments,
    )
    compiled = _get_compiled_chunk_increment_bwd_bundle(
        input_info=input_info,
        cache_key=cache_key,
        n_d_tiles=n_d_tiles,
        runtime_bundle=runtime_bundle,
    )
    outputs = _make_chunk_increment_bwd_outputs(
        workspace,
        input_info=input_info,
        n_d_tiles=n_d_tiles,
    )

    compiled_stage = compiled.stage
    stage_runtime_args = runtime_bundle.stage.runtime_args

    def launch() -> None:
        compiled_stage(*stage_runtime_args)
        _record_tensors_on_current_stream(*stage_runtime_args)

    return PreparedChunkIncrementBwdLaunch(
        compiled=compiled,
        outputs=outputs,
        launch=launch,
    )


def _prepared_chunk_increment_bwd_tuple(
    prepared_launch: PreparedChunkIncrementBwdLaunch,
    *,
    include_launch: bool,
) -> tuple:
    outputs = prepared_launch.outputs
    base = (
        prepared_launch.compiled.db,
        prepared_launch.compiled.du,
        prepared_launch.compiled.boundary,
        prepared_launch.compiled.param_scan,
        outputs.d_b_chunk,
        outputs.d_u_chunk,
        outputs.d_b_prev,
        outputs.d_u_prev,
        outputs.d_msum_part,
        outputs.d_mp0,
        outputs.d_m,
        outputs.d_kprev,
        outputs.d_kcurr,
    )
    if include_launch:
        return (*base, prepared_launch.launch)
    return base


def _make_chunk_increment_bwd_public_outputs(
    *,
    U: torch.Tensor,
    B: torch.Tensor,
    outputs: ChunkIncrementBwdOutputs,
    return_prev_grads: bool,
) -> tuple[torch.Tensor, ...]:
    d_u_public = _fold_chunk_boundary_carries(outputs.d_u_chunk, outputs.d_u_prev)
    d_b_public = _fold_chunk_boundary_carries(outputs.d_b_chunk, outputs.d_b_prev)

    if not return_prev_grads:
        return (
            _public_from_chunked_output(
                d_u_public,
                time_steps=U.shape[2],
                dtype=U.dtype,
            ),
            _public_from_chunked_output(outputs.d_m, time_steps=U.shape[2]),
            _public_dk_from_parts(
                outputs.d_kprev,
                outputs.d_kcurr,
                time_steps=U.shape[2],
            ),
            _public_from_chunked_output(
                d_b_public,
                time_steps=U.shape[2],
                dtype=B.dtype,
            ),
        )
    return (
        _public_from_chunked_output(
            d_u_public,
            time_steps=U.shape[2],
            dtype=U.dtype,
        ),
        _public_from_chunked_output(outputs.d_m, time_steps=U.shape[2]),
        _public_dk_from_parts(
            outputs.d_kprev,
            outputs.d_kcurr,
            time_steps=U.shape[2],
        ),
        _public_from_chunked_output(
            d_b_public,
            time_steps=U.shape[2],
            dtype=B.dtype,
        ),
        outputs.d_b_prev[:, :, 0, :].to(dtype=B.dtype).contiguous(),
        outputs.d_u_prev[:, :, 0, :].to(dtype=U.dtype).contiguous(),
    )


def compile_chunk_increment_bwd_kernels(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_launchers: bool = False,
) -> tuple:
    """Compile the standalone chunk-increment backward kernels and allocate outputs."""
    prepared_launch = _make_chunk_increment_bwd_prepared_launch(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    return _prepared_chunk_increment_bwd_tuple(
        prepared_launch,
        include_launch=return_launchers,
    )


def chunk_increment_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_prev_grads: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Thin public wrapper over the compiled chunk-increment backward kernel bundle."""
    prepared_launch = _make_chunk_increment_bwd_prepared_launch(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    prepared_launch.launch()
    return _make_chunk_increment_bwd_public_outputs(
        U=U,
        B=B,
        outputs=prepared_launch.outputs,
        return_prev_grads=return_prev_grads,
    )


__all__ = [
    "chunk_increment_bwd_cute",
    "compile_chunk_increment_bwd_kernels",
]
