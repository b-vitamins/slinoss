"""Shared CuTe helpers for the fused ``scanprep`` backend."""

from typing import Any, cast

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

_cutlass_max = cast(Any, getattr(cutlass, "max"))
_cutlass_select = cast(Any, getattr(cutlass, "select_"))
FLOAT16_FINITE_MAX = 65504.0
BFLOAT16_FINITE_MAX = 3.3895313892515355e38
FLOAT32_FINITE_MAX = 3.4028234663852886e38
COMPLEX_DIV_DENOM_FLOOR = 1.0e-30

SCANPREP_PARAM_DIM = 3

COEFF_AUX_ALPHA = 0
COEFF_AUX_THETA_TANH = 1
COEFF_AUX_THETA_DRIVE = 2
COEFF_AUX_DT = 3
COEFF_AUX_EXP_TERM = 4
COEFF_AUX_R = 5
COEFF_AUX_THETA = 6
COEFF_AUX_RHO_RE = 7
COEFF_AUX_RHO_IM = 8
COEFF_AUX_LOG_R = 9
COEFF_AUX_KAPPA1_RE = 10
COEFF_AUX_KAPPA1_IM = 11
COEFF_AUX_KAPPA2_RE = 12
COEFF_AUX_KAPPA2_IM = 13


def make_row_major_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = running
        running *= int(shape[i])
    return tuple(stride)


def torch_to_cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported CuTe scanprep dtype: {dtype}.")


def assumed_align(tensor: torch.Tensor) -> int:
    elem_align = max(1, tensor.element_size())
    ptr = int(tensor.data_ptr())
    for align in (16, 8, 4):
        if align < elem_align:
            continue
        if (ptr % align) == 0:
            return align
    return elem_align


def contiguous_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def tensor_compile_signature(tensor: torch.Tensor) -> tuple[torch.dtype, int]:
    return tensor.dtype, assumed_align(tensor)


def make_fake_tensor_arg(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    align: int | None = None,
    dynamic_stride: bool = False,
):
    fake_shape = tuple(
        int(dim) for dim in (shape if shape is not None else tensor.shape)
    )
    fake_stride = tuple(
        int(step) for step in (stride if stride is not None else tensor.stride())
    )
    assumed = int(align if align is not None else assumed_align(tensor))
    if not dynamic_stride and fake_stride == make_row_major_stride(fake_shape):
        dynamic_shape = tuple(cute.sym_int32() for _ in fake_shape)
        return cute.runtime.make_fake_compact_tensor(
            torch_to_cutlass_dtype(tensor.dtype),
            dynamic_shape,
            stride_order=tuple(reversed(range(len(fake_shape)))),
            assumed_align=assumed,
        )
    if dynamic_stride:
        dynamic_shape = tuple(cute.sym_int32() for _ in fake_shape)
        dynamic_fake_stride = tuple(
            0 if step == 0 else cute.sym_int32() for step in fake_stride
        )
        return cute.runtime.make_fake_tensor(
            torch_to_cutlass_dtype(tensor.dtype),
            dynamic_shape,
            stride=dynamic_fake_stride,
            assumed_align=assumed,
        )
    return cute.runtime.make_fake_tensor(
        torch_to_cutlass_dtype(tensor.dtype),
        fake_shape,
        stride=fake_stride,
        assumed_align=assumed,
    )


def sigmoid(x):
    one = cutlass.Float32(1.0)
    x_f = cutlass.Float32(x)
    return cutlass.Float32(one / (one + cute_math.exp(-x_f)))


def softplus(x):
    zero = cutlass.Float32(0.0)
    one = cutlass.Float32(1.0)
    x_f = cutlass.Float32(x)
    abs_x = _cutlass_max(x_f, -x_f)
    return cutlass.Float32(
        _cutlass_max(x_f, zero) + cute_math.log(one + cute_math.exp(-abs_x))
    )


def principal_angle(theta):
    theta_f = cutlass.Float32(theta)
    sin_theta = cutlass.Float32(cute_math.sin(theta_f))
    cos_theta = cutlass.Float32(cute_math.cos(theta_f))
    return cutlass.Float32(cute_math.atan2(sin_theta, cos_theta))


def complex_div(num_re, num_im, den_re, den_im):
    num_re_f = cutlass.Float32(num_re)
    num_im_f = cutlass.Float32(num_im)
    den_re_f = cutlass.Float32(den_re)
    den_im_f = cutlass.Float32(den_im)
    denom = den_re_f * den_re_f + den_im_f * den_im_f
    denom = _cutlass_max(denom, cutlass.Float32(COMPLEX_DIV_DENOM_FLOOR))
    out_re = (num_re_f * den_re_f + num_im_f * den_im_f) / denom
    out_im = (num_im_f * den_re_f - num_re_f * den_im_f) / denom
    return cutlass.Float32(out_re), cutlass.Float32(out_im)


def clamp_finite_for_dtype(x, dtype):
    x = cutlass.Float32(x)
    pos_inf = cutlass.Float32(float("inf"))
    neg_inf = cutlass.Float32(float("-inf"))
    max_abs = cutlass.Float32(FLOAT32_FINITE_MAX)
    if dtype == cutlass.Float16:
        max_abs = cutlass.Float32(FLOAT16_FINITE_MAX)
    elif dtype == cutlass.BFloat16:
        max_abs = cutlass.Float32(BFLOAT16_FINITE_MAX)
    clamped = _cutlass_select(x > max_abs, max_abs, x)
    clamped = _cutlass_select(clamped < -max_abs, -max_abs, clamped)
    clamped = _cutlass_select(x == pos_inf, x, clamped)
    clamped = _cutlass_select(x == neg_inf, x, clamped)
    clamped = _cutlass_select(x != x, x, clamped)
    return clamped


def safe_cast_to_dtype(x, dtype):
    x = clamp_finite_for_dtype(x, dtype)
    if dtype == cutlass.Float16:
        return cutlass.Float16(x)
    if dtype == cutlass.BFloat16:
        return cutlass.BFloat16(x)
    return cutlass.Float32(x)


def complex_mul(a_re, a_im, b_re, b_im):
    a_re_f = cutlass.Float32(a_re)
    a_im_f = cutlass.Float32(a_im)
    b_re_f = cutlass.Float32(b_re)
    b_im_f = cutlass.Float32(b_im)
    out_re = a_re_f * b_re_f - a_im_f * b_im_f
    out_im = a_re_f * b_im_f + a_im_f * b_re_f
    return cutlass.Float32(out_re), cutlass.Float32(out_im)


def complex_mul_conj(a_re, a_im, b_re, b_im):
    return complex_mul(a_re, a_im, b_re, -cutlass.Float32(b_im))


def real_mul_conj(a_re, a_im, b_re, b_im):
    a_re_f = cutlass.Float32(a_re)
    a_im_f = cutlass.Float32(a_im)
    b_re_f = cutlass.Float32(b_re)
    b_im_f = cutlass.Float32(b_im)
    return cutlass.Float32(a_re_f * b_re_f + a_im_f * b_im_f)


__all__ = [
    "SCANPREP_PARAM_DIM",
    "COEFF_AUX_DT",
    "COEFF_AUX_EXP_TERM",
    "COEFF_AUX_KAPPA1_IM",
    "COEFF_AUX_KAPPA1_RE",
    "COEFF_AUX_KAPPA2_IM",
    "COEFF_AUX_KAPPA2_RE",
    "COEFF_AUX_LOG_R",
    "COEFF_AUX_R",
    "COEFF_AUX_RHO_IM",
    "COEFF_AUX_RHO_RE",
    "COEFF_AUX_THETA",
    "COEFF_AUX_THETA_DRIVE",
    "COEFF_AUX_THETA_TANH",
    "COEFF_AUX_ALPHA",
    "assumed_align",
    "contiguous_tensor",
    "complex_div",
    "complex_mul",
    "complex_mul_conj",
    "make_fake_tensor_arg",
    "make_row_major_stride",
    "principal_angle",
    "real_mul_conj",
    "safe_cast_to_dtype",
    "sigmoid",
    "softplus",
    "tensor_compile_signature",
    "torch_to_cutlass_dtype",
]
