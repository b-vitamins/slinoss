"""CuTe host wrapper for the fused ``v2x2ssd`` backward state-passing stage.

Inputs:
- ``chunk_starts``: ``(B, H, C, P, D)`` float32 state before each forward chunk
- ``m_chunk``: ``(B, H, C, 2)`` float32 packed-complex chunk multipliers
- ``d_chunk_starts``: ``(B, H, C, P, D)`` float32 upstream gradient for chunk starts
- ``d_final``: ``(B, H, P, D)`` float32 upstream gradient for the final state

Outputs:
- ``d_inc``: ``(B, H, C, P, D)`` float32 gradient for chunk-local increments
- ``d_m_chunk``: ``(B, H, C, 2)`` float32 gradient for chunk multipliers
- ``d_initial``: ``(B, H, P, D)`` float32 gradient for the initial state

This module owns the Python-side host JIT boundary:
- validate and normalize public operands
- allocate public gradient outputs
- build TVM FFI runtime and compile arguments
- compile and cache the host wrapper
- prepare relaunchable kernel invocations
"""

from dataclasses import dataclass

import torch
from cuda.bindings import driver as cuda
import cutlass.cute as cute

from slinoss._cute_runtime import TensorSpec

from .common import (
    _TileConfig,
    _choose_copy_bits_for_linear_tiles,
    _make_static_tensor_spec_view,
    _make_tensor_spec,
    _make_tvm_ffi_runtime_and_compile_args_from_specs,
    _record_tensors_on_current_stream,
)
from .state_passing import StatePassingBwdAmpere


StatePassingBwdProblemShape = tuple[int, int, int, int, int]
StatePassingBwdLaunchConfig = tuple[int, int, int, int, int, int, int]

_STATE_PASSING_BWD_CACHE: dict[tuple, object] = {}


@dataclass(frozen=True)
class StatePassingBwdTensorSpecs:
    chunk_starts: TensorSpec
    d_chunk_starts: TensorSpec
    d_final: TensorSpec
    chunk_multiplier: TensorSpec
    d_increment: TensorSpec
    d_chunk_multiplier: TensorSpec
    d_initial: TensorSpec


@dataclass(frozen=True)
class StatePassingBwdOperands:
    chunk_starts: torch.Tensor
    chunk_multiplier: torch.Tensor
    d_chunk_starts: torch.Tensor
    d_final: torch.Tensor


@dataclass(frozen=True)
class StatePassingBwdOutputs:
    d_increment: torch.Tensor
    d_chunk_multiplier: torch.Tensor
    d_initial: torch.Tensor


@dataclass(frozen=True)
class StatePassingBwdCompileArtifacts:
    problem_shape: StatePassingBwdProblemShape
    launch_cfg: StatePassingBwdLaunchConfig
    config: _TileConfig
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class StatePassingBwdRuntimeArtifacts:
    operands: StatePassingBwdOperands
    runtime_args: tuple[torch.Tensor, ...]
    compile_artifacts: StatePassingBwdCompileArtifacts
    outputs: StatePassingBwdOutputs


@dataclass(frozen=True)
class PreparedStatePassingBwdLaunch:
    compiled: object
    runtime_args: tuple[torch.Tensor, ...]
    outputs: StatePassingBwdOutputs


def _make_state_passing_bwd_tensor_specs(
    problem_shape: StatePassingBwdProblemShape,
) -> StatePassingBwdTensorSpecs:
    B, H, C, P, D = problem_shape
    return StatePassingBwdTensorSpecs(
        chunk_starts=_make_tensor_spec((B, H, C, P, D)),
        d_chunk_starts=_make_tensor_spec((B, H, C, P, D)),
        d_final=_make_tensor_spec((B, H, P, D)),
        chunk_multiplier=_make_tensor_spec((B, H, C, 2)),
        d_increment=_make_tensor_spec((B, H, C, P, D)),
        d_chunk_multiplier=_make_tensor_spec((B, H, C, 2)),
        d_initial=_make_tensor_spec((B, H, P, D)),
    )


def _make_state_passing_bwd_cache_key(
    *,
    device_index: int,
    problem_shape: StatePassingBwdProblemShape,
    launch_cfg: StatePassingBwdLaunchConfig,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "state_passing_bwd",
        int(device_index),
        tuple(int(dim) for dim in problem_shape),
        tuple(int(value) for value in launch_cfg),
        tuple(int(align) for align in alignments),
    )


def _make_state_passing_bwd_launch_cfg(
    *,
    config: _TileConfig,
    operands: StatePassingBwdOperands,
    outputs: StatePassingBwdOutputs,
    state_elem_count: int,
) -> StatePassingBwdLaunchConfig:
    return (
        int(config.num_threads),
        int(config.pairs_per_thread),
        _choose_copy_bits_for_linear_tiles(
            operands.chunk_starts,
            tile_stride_elems=state_elem_count,
            elems_per_thread=config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles(
            operands.d_chunk_starts,
            tile_stride_elems=state_elem_count,
            elems_per_thread=config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles(
            outputs.d_increment,
            tile_stride_elems=state_elem_count,
            elems_per_thread=config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles(
            outputs.d_initial,
            tile_stride_elems=state_elem_count,
            elems_per_thread=config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles(
            operands.d_final,
            tile_stride_elems=state_elem_count,
            elems_per_thread=config.elems_per_thread,
        ),
    )


def _validate_state_passing_bwd_inputs(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
) -> StatePassingBwdProblemShape:
    if chunk_starts.ndim != 5:
        raise ValueError(
            f"chunk_starts must be (B,H,C,P,D). Got {tuple(chunk_starts.shape)}."
        )
    if d_chunk_starts.shape != chunk_starts.shape:
        raise ValueError(
            "d_chunk_starts must match chunk_starts. "
            f"Got {tuple(d_chunk_starts.shape)} vs {tuple(chunk_starts.shape)}."
        )
    if m_chunk.ndim != 4 or m_chunk.shape[-1] != 2:
        raise ValueError(f"m_chunk must be (B,H,C,2). Got {tuple(m_chunk.shape)}.")
    if chunk_starts.dtype != torch.float32:
        raise TypeError("chunk_starts must be float32.")
    if d_chunk_starts.dtype != torch.float32 or d_final.dtype != torch.float32:
        raise TypeError("Upstream gradients must be float32.")
    if m_chunk.dtype != torch.float32:
        raise TypeError("m_chunk must be float32.")
    if chunk_starts.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if (
        d_chunk_starts.device != chunk_starts.device
        or d_final.device != chunk_starts.device
        or m_chunk.device != chunk_starts.device
    ):
        raise ValueError("All tensors must live on the same CUDA device.")

    B, H, C, P, D = map(int, chunk_starts.shape)
    if tuple(m_chunk.shape[:3]) != (B, H, C):
        raise ValueError(
            "m_chunk leading dims must match chunk_starts. "
            f"Got {tuple(m_chunk.shape[:3])} vs {(B, H, C)}."
        )
    if tuple(d_final.shape) != (B, H, P, D):
        raise ValueError(f"d_final must be (B,H,P,D)={(B, H, P, D)}.")
    return B, H, C, P, D


def _normalize_state_passing_bwd_config(
    *,
    num_threads: int,
    pairs_per_thread: int,
) -> _TileConfig:
    config = _TileConfig(
        num_threads=int(num_threads),
        pairs_per_thread=int(pairs_per_thread),
    )
    if config.num_threads <= 0:
        raise ValueError("num_threads must be positive.")
    if config.num_threads % 32 != 0:
        raise ValueError("num_threads must be a multiple of 32.")
    if config.pairs_per_thread <= 0:
        raise ValueError("pairs_per_thread must be positive.")
    return config


def _make_state_passing_bwd_operands(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
) -> StatePassingBwdOperands:
    return StatePassingBwdOperands(
        chunk_starts=chunk_starts.contiguous(),
        chunk_multiplier=m_chunk.contiguous(),
        d_chunk_starts=d_chunk_starts.contiguous(),
        d_final=d_final.contiguous(),
    )


def _allocate_state_passing_bwd_outputs(
    *,
    problem_shape: StatePassingBwdProblemShape,
    device: torch.device,
) -> StatePassingBwdOutputs:
    B, H, C, P, D = problem_shape
    return StatePassingBwdOutputs(
        d_increment=torch.empty((B, H, C, P, D), device=device, dtype=torch.float32),
        d_chunk_multiplier=torch.zeros(
            (B, H, C, 2), device=device, dtype=torch.float32
        ),
        d_initial=torch.empty((B, H, P, D), device=device, dtype=torch.float32),
    )


def _make_state_passing_bwd_runtime_args(
    *,
    operands: StatePassingBwdOperands,
    outputs: StatePassingBwdOutputs,
    tensor_specs: StatePassingBwdTensorSpecs,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[object, ...]]:
    return _make_tvm_ffi_runtime_and_compile_args_from_specs(
        (operands.chunk_starts, torch.float32, tensor_specs.chunk_starts),
        (operands.d_chunk_starts, torch.float32, tensor_specs.d_chunk_starts),
        (operands.d_final, torch.float32, tensor_specs.d_final),
        (operands.chunk_multiplier, torch.float32, tensor_specs.chunk_multiplier),
        (outputs.d_increment, torch.float32, tensor_specs.d_increment),
        (
            outputs.d_chunk_multiplier,
            torch.float32,
            tensor_specs.d_chunk_multiplier,
        ),
        (outputs.d_initial, torch.float32, tensor_specs.d_initial),
    )


def _make_state_passing_bwd_compile_artifacts(
    *,
    device_index: int,
    problem_shape: StatePassingBwdProblemShape,
    launch_cfg: StatePassingBwdLaunchConfig,
    config: _TileConfig,
    compile_args: tuple[object, ...],
    alignments: tuple[int, ...],
) -> StatePassingBwdCompileArtifacts:
    return StatePassingBwdCompileArtifacts(
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        config=config,
        compile_args=compile_args,
        alignments=alignments,
        cache_key=_make_state_passing_bwd_cache_key(
            device_index=device_index,
            problem_shape=problem_shape,
            launch_cfg=launch_cfg,
            alignments=alignments,
        ),
    )


def _make_state_passing_bwd_host_wrapper(
    *,
    problem_shape: StatePassingBwdProblemShape,
    launch_cfg: StatePassingBwdLaunchConfig,
):
    tensor_specs = _make_state_passing_bwd_tensor_specs(problem_shape)
    (
        num_threads,
        pairs_per_thread,
        copy_bits_chunk_starts,
        copy_bits_d_chunk_starts,
        copy_bits_d_increment,
        copy_bits_d_initial,
        copy_bits_d_final,
    ) = launch_cfg
    state_passing_kernel = StatePassingBwdAmpere(
        _TileConfig(
            num_threads=num_threads,
            pairs_per_thread=pairs_per_thread,
        ),
        copy_bits_starts=copy_bits_chunk_starts,
        copy_bits_dstarts=copy_bits_d_chunk_starts,
        copy_bits_dinc=copy_bits_d_increment,
        copy_bits_initial=copy_bits_d_initial,
        copy_bits_final=copy_bits_d_final,
    )

    @cute.jit
    def _state_passing_bwd_host_wrapper(
        chunk_starts_t: cute.Tensor,
        d_chunk_starts_t: cute.Tensor,
        d_final_t: cute.Tensor,
        chunk_multiplier_t: cute.Tensor,
        d_increment_t: cute.Tensor,
        d_chunk_multiplier_t: cute.Tensor,
        d_initial_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        chunk_starts_view = _make_static_tensor_spec_view(
            chunk_starts_t, tensor_specs.chunk_starts
        )
        d_chunk_starts_view = _make_static_tensor_spec_view(
            d_chunk_starts_t, tensor_specs.d_chunk_starts
        )
        d_final_view = _make_static_tensor_spec_view(d_final_t, tensor_specs.d_final)
        chunk_multiplier_view = _make_static_tensor_spec_view(
            chunk_multiplier_t, tensor_specs.chunk_multiplier
        )
        d_increment_view = _make_static_tensor_spec_view(
            d_increment_t, tensor_specs.d_increment
        )
        d_chunk_multiplier_view = _make_static_tensor_spec_view(
            d_chunk_multiplier_t, tensor_specs.d_chunk_multiplier
        )
        d_initial_view = _make_static_tensor_spec_view(
            d_initial_t, tensor_specs.d_initial
        )

        state_passing_kernel.call_on_stream(
            chunk_starts_view,
            d_chunk_starts_view,
            d_final_view,
            chunk_multiplier_view,
            d_increment_view,
            d_chunk_multiplier_view,
            d_initial_view,
            stream,
        )

    return _state_passing_bwd_host_wrapper


def _make_state_passing_bwd_runtime_artifacts(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    num_threads: int,
    pairs_per_thread: int,
) -> StatePassingBwdRuntimeArtifacts:
    problem_shape = _validate_state_passing_bwd_inputs(
        chunk_starts,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
    )
    config = _normalize_state_passing_bwd_config(
        num_threads=num_threads,
        pairs_per_thread=pairs_per_thread,
    )
    operands = _make_state_passing_bwd_operands(
        chunk_starts,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
    )
    outputs = _allocate_state_passing_bwd_outputs(
        problem_shape=problem_shape,
        device=operands.chunk_starts.device,
    )
    tensor_specs = _make_state_passing_bwd_tensor_specs(problem_shape)
    launch_cfg = _make_state_passing_bwd_launch_cfg(
        config=config,
        operands=operands,
        outputs=outputs,
        state_elem_count=int(problem_shape[3]) * int(problem_shape[4]),
    )
    runtime_args, alignments, compile_args = _make_state_passing_bwd_runtime_args(
        operands=operands,
        outputs=outputs,
        tensor_specs=tensor_specs,
    )
    compile_artifacts = _make_state_passing_bwd_compile_artifacts(
        device_index=(
            operands.chunk_starts.device.index
            if operands.chunk_starts.device.index is not None
            else -1
        ),
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        config=config,
        compile_args=compile_args,
        alignments=alignments,
    )
    return StatePassingBwdRuntimeArtifacts(
        operands=operands,
        runtime_args=runtime_args,
        compile_artifacts=compile_artifacts,
        outputs=outputs,
    )


def _get_compiled_state_passing_bwd_kernel(
    compile_artifacts: StatePassingBwdCompileArtifacts,
):
    compiled = _STATE_PASSING_BWD_CACHE.get(compile_artifacts.cache_key)
    if compiled is None:
        host_wrapper = _make_state_passing_bwd_host_wrapper(
            problem_shape=compile_artifacts.problem_shape,
            launch_cfg=compile_artifacts.launch_cfg,
        )
        compiled = cute.compile(
            host_wrapper,
            *compile_artifacts.compile_args,
            options="--enable-tvm-ffi",
        )
        _STATE_PASSING_BWD_CACHE[compile_artifacts.cache_key] = compiled
    return compiled


def _reset_state_passing_bwd_outputs(outputs: StatePassingBwdOutputs) -> None:
    # The chunk-multiplier gradient is accumulated atomically across state tiles.
    # The increment and initial-state gradients are fully overwritten per launch.
    outputs.d_chunk_multiplier.zero_()


def _make_state_passing_bwd_prepared_launch(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    num_threads: int,
    pairs_per_thread: int,
) -> PreparedStatePassingBwdLaunch:
    runtime_artifacts = _make_state_passing_bwd_runtime_artifacts(
        chunk_starts,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
        num_threads=num_threads,
        pairs_per_thread=pairs_per_thread,
    )
    compiled = _get_compiled_state_passing_bwd_kernel(
        runtime_artifacts.compile_artifacts
    )
    return PreparedStatePassingBwdLaunch(
        compiled=compiled,
        runtime_args=runtime_artifacts.runtime_args,
        outputs=runtime_artifacts.outputs,
    )


def compile_state_passing_bwd_kernel(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    num_threads: int = 128,
    pairs_per_thread: int = 8,
    return_launcher: bool = False,
) -> tuple:
    """Compile the fused backward state-passing kernel and allocate outputs."""
    prepared = _make_state_passing_bwd_prepared_launch(
        chunk_starts,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
        num_threads=num_threads,
        pairs_per_thread=pairs_per_thread,
    )
    compiled = prepared.compiled
    runtime_args = prepared.runtime_args
    outputs = prepared.outputs

    def launch() -> None:
        _reset_state_passing_bwd_outputs(outputs)
        compiled(*runtime_args)
        _record_tensors_on_current_stream(*runtime_args)

    base = (
        compiled,
        outputs.d_increment,
        outputs.d_chunk_multiplier,
        outputs.d_initial,
    )
    if return_launcher:
        return (*base, launch)
    return base


def state_passing_bwd_cute(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    num_threads: int = 128,
    pairs_per_thread: int = 8,
    return_d_initial: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Run the fused backward state-passing kernel and return public gradients."""

    _compiled, d_increment, d_chunk_multiplier, d_initial, launch = (
        compile_state_passing_bwd_kernel(
            chunk_starts,
            m_chunk,
            d_chunk_starts=d_chunk_starts,
            d_final=d_final,
            num_threads=num_threads,
            pairs_per_thread=pairs_per_thread,
            return_launcher=True,
        )
    )
    launch()
    if not return_d_initial:
        return (
            d_increment.to(dtype=torch.float32).contiguous(),
            d_chunk_multiplier.to(dtype=torch.float32).contiguous(),
        )
    return (
        d_increment.to(dtype=torch.float32).contiguous(),
        d_chunk_multiplier.to(dtype=torch.float32).contiguous(),
        d_initial.to(dtype=torch.float32).contiguous(),
    )


__all__ = [
    "compile_state_passing_bwd_kernel",
    "state_passing_bwd_cute",
]
