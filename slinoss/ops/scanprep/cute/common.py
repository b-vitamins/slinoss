"""Shared CuTe helpers for the fused scanprep backend."""

from __future__ import annotations

from typing import Any, cast

import torch

import cutlass
import cutlass.cute.math as cute_math

_cutlass_max = cast(Any, getattr(cutlass, "max"))

COEFF_AUX_DT_U = 0
COEFF_AUX_GAMMA_SIGMOID = 1
COEFF_AUX_OMEGA = 2
COEFF_AUX_R_DIRECT_U = 3
COEFF_AUX_THETA_DIRECT_TANH = 4
COEFF_AUX_MIX_R = 5
COEFF_AUX_MIX_THETA = 6
COEFF_AUX_MIX_K_PREV = 7
COEFF_AUX_MIX_K_CURR = 8
COEFF_AUX_K_PREV_TANH_RE = 9
COEFF_AUX_K_PREV_TANH_IM = 10
COEFF_AUX_K_CURR_TANH_RE = 11
COEFF_AUX_K_CURR_TANH_IM = 12
COEFF_AUX_DT = 13
COEFF_AUX_GAMMA = 14
COEFF_AUX_EXP_TERM = 15
COEFF_AUX_DELTA_R = 16
COEFF_AUX_DELTA_THETA = 17
COEFF_AUX_R = 18
COEFF_AUX_THETA = 19
COEFF_AUX_RHO_RE = 20
COEFF_AUX_RHO_IM = 21
COEFF_AUX_LOG_R = 22
COEFF_AUX_KAPPA1_RE = 23
COEFF_AUX_KAPPA1_IM = 24
COEFF_AUX_KAPPA2_RE = 25
COEFF_AUX_KAPPA2_IM = 26
COEFF_AUX_FIELDS = 27

BIAS_GRAD_PARAM_INDICES = (0, 1, 2, 5, 6, 7, 8)


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


def lerp(a, b, w):
    a_f = cutlass.Float32(a)
    b_f = cutlass.Float32(b)
    w_f = cutlass.Float32(w)
    return cutlass.Float32(a_f + w_f * (b_f - a_f))


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
    out_re = (num_re_f * den_re_f + num_im_f * den_im_f) / denom
    out_im = (num_im_f * den_re_f - num_re_f * den_im_f) / denom
    return cutlass.Float32(out_re), cutlass.Float32(out_im)


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
    "BIAS_GRAD_PARAM_INDICES",
    "COEFF_AUX_DELTA_R",
    "COEFF_AUX_DELTA_THETA",
    "COEFF_AUX_DT",
    "COEFF_AUX_DT_U",
    "COEFF_AUX_EXP_TERM",
    "COEFF_AUX_FIELDS",
    "COEFF_AUX_GAMMA",
    "COEFF_AUX_GAMMA_SIGMOID",
    "COEFF_AUX_KAPPA1_IM",
    "COEFF_AUX_KAPPA1_RE",
    "COEFF_AUX_KAPPA2_IM",
    "COEFF_AUX_KAPPA2_RE",
    "COEFF_AUX_K_CURR_TANH_IM",
    "COEFF_AUX_K_CURR_TANH_RE",
    "COEFF_AUX_K_PREV_TANH_IM",
    "COEFF_AUX_K_PREV_TANH_RE",
    "COEFF_AUX_LOG_R",
    "COEFF_AUX_MIX_K_CURR",
    "COEFF_AUX_MIX_K_PREV",
    "COEFF_AUX_MIX_R",
    "COEFF_AUX_MIX_THETA",
    "COEFF_AUX_OMEGA",
    "COEFF_AUX_R",
    "COEFF_AUX_RHO_IM",
    "COEFF_AUX_RHO_RE",
    "COEFF_AUX_R_DIRECT_U",
    "COEFF_AUX_THETA",
    "COEFF_AUX_THETA_DIRECT_TANH",
    "assumed_align",
    "complex_div",
    "complex_mul",
    "complex_mul_conj",
    "lerp",
    "make_row_major_stride",
    "principal_angle",
    "real_mul_conj",
    "sigmoid",
    "softplus",
    "torch_to_cutlass_dtype",
]
