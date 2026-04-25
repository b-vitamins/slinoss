"""Common CuTe helpers for backward ``v2x2ssd`` kernels."""

from dataclasses import dataclass

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from slinoss._cute_runtime import (
    TensorSpec,
    make_runtime_tensor_spec_view,
)
from slinoss.ops._cute_common import (
    _compile_env_stream_placeholder,
    assumed_align,
    make_fake_tensor_spec_arg,
    make_row_major_stride,
)


LOG2_E = 1.4426950408889634074
TWO_LOG2_E = 2.0 * LOG2_E
MIN_STABLE_PREFIX_LOG = -40.0
FLOAT16_FINITE_MAX = 65504.0
BFLOAT16_FINITE_MAX = 3.3895313892515355e38
FLOAT32_FINITE_MAX = 3.4028234663852886e38


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _default_tc_k_tile(k_extent: int) -> int:
    if k_extent >= 32 and k_extent % 32 == 0:
        return 32
    if k_extent >= 16 and k_extent % 16 == 0:
        return 16
    raise ValueError("Tensor-core K extent must be divisible by 16.")


def _default_async_copy_bits(
    *,
    dtype_width: int,
    major_mode: utils.LayoutEnum,
    tile_m: int,
    tile_k: int,
    num_threads: int,
) -> int:
    """Pick the widest async-copy width whose thread tiling is legal."""

    for copy_bits in (128, 64, 32, 16):
        if copy_bits < dtype_width or copy_bits % dtype_width != 0:
            continue
        copy_elems = copy_bits // dtype_width

        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            for tm in range(1, num_threads + 1):
                if num_threads % tm != 0:
                    continue
                tn = num_threads // tm
                tile_k_seg = tn * copy_elems
                if int(tile_k) % tile_k_seg != 0:
                    continue
                return copy_bits
            continue

        shape_dim_0 = (int(tile_m) + int(copy_elems) - 1) // int(copy_elems)
        if shape_dim_0 <= num_threads:
            for cand in range(shape_dim_0, num_threads + 1):
                if num_threads % cand == 0:
                    return copy_bits

    raise ValueError("Failed to find a legal async-copy width for the given tile.")


def _make_tensor_spec(
    shape: tuple[int, ...],
    *,
    stride: tuple[int, ...] | None = None,
) -> TensorSpec:
    resolved_shape = tuple(int(dim) for dim in shape)
    resolved_stride = (
        make_row_major_stride(resolved_shape)
        if stride is None
        else tuple(int(step) for step in stride)
    )
    return resolved_shape, resolved_stride


def _make_static_tensor_spec_view(
    tensor: cute.Tensor,
    tensor_spec: TensorSpec,
) -> cute.Tensor:
    shape, stride = tensor_spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _make_runtime_tensor_views_from_specs(
    *tensor_specs: tuple[torch.Tensor, TensorSpec],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        make_runtime_tensor_spec_view(tensor, spec) for tensor, spec in tensor_specs
    )


def _make_tvm_ffi_runtime_and_compile_args_from_specs(
    *tensor_specs: tuple[torch.Tensor, torch.dtype, TensorSpec],
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[object, ...]]:
    runtime_args = _make_runtime_tensor_views_from_specs(
        *((tensor, spec) for tensor, _dtype, spec in tensor_specs)
    )
    alignments = tuple(assumed_align(tensor) for tensor in runtime_args)
    compile_args = tuple(
        make_fake_tensor_spec_arg(
            dtype=dtype,
            shape=spec[0],
            stride=spec[1],
            align=align,
        )
        for (_tensor, dtype, spec), align in zip(tensor_specs, alignments, strict=True)
    ) + (_compile_env_stream_placeholder(),)
    return runtime_args, alignments, compile_args


@dataclass(frozen=True)
class _TileConfig:
    num_threads: int = 128
    pairs_per_thread: int = 8

    @property
    def elems_per_thread(self) -> int:
        return 2 * int(self.pairs_per_thread)

    @property
    def tile(self) -> int:
        return int(self.num_threads) * self.elems_per_thread


@cute.jit
def clamp_nonpositive_prefix_log(logp):
    logp = cutlass.Float32(logp)
    zero = cutlass.Float32(0.0)
    min_log = cutlass.Float32(MIN_STABLE_PREFIX_LOG)
    pos_inf = cutlass.Float32(float("inf"))
    neg_inf = cutlass.Float32(float("-inf"))
    clamped = cutlass.select_(logp > zero, zero, logp)
    clamped = cutlass.select_(clamped < min_log, min_log, clamped)
    clamped = cutlass.select_(logp == pos_inf, logp, clamped)
    clamped = cutlass.select_(logp == neg_inf, logp, clamped)
    clamped = cutlass.select_(logp != logp, logp, clamped)
    return clamped


@cute.jit
def clamp_finite_for_dtype(x, dtype):
    x = cutlass.Float32(x)
    pos_inf = cutlass.Float32(float("inf"))
    neg_inf = cutlass.Float32(float("-inf"))
    max_abs = cutlass.Float32(FLOAT32_FINITE_MAX)
    if cutlass.const_expr(dtype == cutlass.Float16):
        max_abs = cutlass.Float32(FLOAT16_FINITE_MAX)
    elif cutlass.const_expr(dtype == cutlass.BFloat16):
        max_abs = cutlass.Float32(BFLOAT16_FINITE_MAX)
    clamped = cutlass.select_(x > max_abs, max_abs, x)
    clamped = cutlass.select_(clamped < -max_abs, -max_abs, clamped)
    clamped = cutlass.select_(x == pos_inf, x, clamped)
    clamped = cutlass.select_(x == neg_inf, x, clamped)
    clamped = cutlass.select_(x != x, x, clamped)
    return clamped


@cute.jit
def safe_cast_to_dtype(x, dtype):
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        return cutlass.BFloat16(x)
    x = clamp_finite_for_dtype(x, dtype)
    if cutlass.const_expr(dtype == cutlass.Float16):
        return cutlass.Float16(x)
    return cutlass.Float32(x)


@cute.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@cute.jit
def conj_mul_phase(xr, xi, pr, pi):
    """Return ``conj(x) * p`` for packed complex scalars."""
    return xr * pr + xi * pi, xr * pi - xi * pr


@cute.jit
def mul_conj_phase(xr, xi, pr, pi):
    """Return ``x * conj(p)`` for packed complex scalars."""
    return xr * pr + xi * pi, xi * pr - xr * pi


@cute.jit
def apply_complex_tap(xr, xi, kr, ki):
    """Apply packed complex tap ``k`` to packed complex value ``x``."""
    return kr * xr - ki * xi, kr * xi + ki * xr


@cute.jit
def apply_complex_tap_adjoint(gr, gi, kr, ki):
    """Adjoint of :func:`apply_complex_tap` for packed real-imag pairs."""
    return kr * gr + ki * gi, -ki * gr + kr * gi
