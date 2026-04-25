"""Shared CuTe helpers for the fused ``scanprep`` backend."""

from typing import Any, cast

import cutlass
import cutlass.cute.math as cute_math

from slinoss.ops._cute_common import (
    assumed_align,
    contiguous_tensor,
    make_fake_tensor_arg,
    make_row_major_stride,
    safe_cast_to_dtype,
    tensor_compile_signature,
    torch_to_cutlass_dtype,
)

_cutlass_max = cast(Any, getattr(cutlass, "max"))
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
