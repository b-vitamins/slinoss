"""Common CuTe helpers for the standalone ``v2x2ssd`` chunk-scan bwd kernels."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


LOG2_E = 1.4426950408889634074
TWO_LOG2_E = 2.0 * LOG2_E
# Valid scanprep-generated chunk prefixes stay close to zero; this floor only
# activates when a broken reconstruction would otherwise drive exp/reciprocal
# into 0/inf territory and poison the rest of the kernel.
MIN_STABLE_PREFIX_LOG = -40.0
FLOAT16_FINITE_MAX = 65504.0
BFLOAT16_FINITE_MAX = 3.3895313892515355e38
FLOAT32_FINITE_MAX = 3.4028234663852886e38


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
    x = clamp_finite_for_dtype(x, dtype)
    if cutlass.const_expr(dtype == cutlass.Float16):
        return cutlass.Float16(x)
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        return cutlass.BFloat16(x)
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
