"""CuTe decode entry point for the one-token SLinOSS recurrent middle."""

from __future__ import annotations

import os
from typing import Callable, cast

import torch
import cutlass.cute as cute

from slinoss.ops.v2x2ssd.cute.kernels.fwd.common import (
    _assumed_align,
    _compile_env_stream_placeholder,
    _ensure_min_alignment,
    _make_fake_tensor_arg,
)

from .kernels.decode import MixerDecodeStepFwd

_DECODE_CACHE: dict[tuple[object, ...], object] = {}
_DECODE_MIN_ALIGN = 16


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _raise_cold_capture_error(resource: str) -> None:
    raise RuntimeError(
        "CuTe mixer decode "
        f"{resource} is cold during CUDA graph capture. "
        "Warm the same mixer decode spec once outside capture before graph capture."
    )


def _aligned_empty(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _ensure_min_alignment(
        torch.empty(shape, device=device, dtype=dtype),
        min_align=_DECODE_MIN_ALIGN,
    )


def _aligned_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    return _ensure_min_alignment(
        torch.empty_like(tensor),
        min_align=_DECODE_MIN_ALIGN,
    )


def _parse_env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer when set. Got {raw!r}.") from exc


def _select_decode_tuning(
    *,
    batch: int,
    heads: int,
    p_size: int,
) -> tuple[int, int, int]:
    del batch, heads
    tile_p = 32
    num_warps = 4
    vec_n = 4
    if tile_p > p_size:
        tile_p = p_size
    while tile_p > 1 and p_size % tile_p != 0:
        tile_p //= 2
    if p_size % tile_p != 0:
        tile_p = 1
    tile_p = _parse_env_int("SLINOSS_MIXER_DECODE_TILE_P") or tile_p
    num_warps = _parse_env_int("SLINOSS_MIXER_DECODE_NUM_WARPS") or num_warps
    vec_n = _parse_env_int("SLINOSS_MIXER_DECODE_VEC_N") or vec_n
    return tile_p, num_warps, vec_n


def _select_fused_decode_tuning(
    *,
    batch: int,
    heads: int,
    p_size: int,
    d_model: int,
) -> tuple[int, int, int]:
    del batch, heads, d_model
    tile_p = 64 if p_size >= 64 else p_size
    num_warps = 4
    vec_n = 2
    tile_p = _parse_env_int("SLINOSS_MIXER_DECODE_FUSED_TILE_P") or tile_p
    num_warps = _parse_env_int("SLINOSS_MIXER_DECODE_FUSED_NUM_WARPS") or num_warps
    vec_n = _parse_env_int("SLINOSS_MIXER_DECODE_FUSED_VEC_N") or vec_n
    return tile_p, num_warps, vec_n


def _validate_decode_inputs(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    gate: torch.Tensor,
    skip: torch.Tensor,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if value.ndim != 3:
        raise ValueError(f"value must be (B,H,P). Got {tuple(value.shape)}.")
    if params.ndim != 3 or params.shape[-1] != 13:
        raise ValueError(f"params must be (B,H,13). Got {tuple(params.shape)}.")
    if bc.ndim != 4 or bc.shape[-2] != 4:
        raise ValueError(f"bc must be (B,H,4,N). Got {tuple(bc.shape)}.")
    if gate.shape != value.shape:
        raise ValueError(f"gate must match value exactly. Got {tuple(gate.shape)}.")
    batch, heads, P = map(int, value.shape)
    if tuple(map(int, params.shape[:2])) != (batch, heads):
        raise ValueError("params leading dims must match value.")
    if tuple(map(int, bc.shape[:2])) != (batch, heads):
        raise ValueError("bc leading dims must match value.")
    N = int(bc.shape[-1])
    if tuple(map(int, skip.shape)) != (heads, P):
        raise ValueError(f"skip must be {(heads, P)}. Got {tuple(skip.shape)}.")
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")
    if initial_states is not None and tuple(map(int, initial_states.shape)) != (
        batch,
        heads,
        P,
        2 * N,
    ):
        raise ValueError(
            "initial_states must be "
            f"{(batch, heads, P, 2 * N)}. Got {tuple(initial_states.shape)}."
        )
    if B_prev is not None and tuple(map(int, B_prev.shape)) != (batch, heads, 2 * N):
        raise ValueError(
            f"B_prev must be {(batch, heads, 2 * N)}. Got {tuple(B_prev.shape)}."
        )
    if U_prev is not None and tuple(map(int, U_prev.shape)) != (batch, heads, P):
        raise ValueError(
            f"U_prev must be {(batch, heads, P)}. Got {tuple(U_prev.shape)}."
        )
    return batch, heads, P, N


def mixer_decode_step_cute(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    gate: torch.Tensor,
    skip: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    dt_min: float,
    dt_max: float,
    r_min: float,
    r_max: float,
    theta_bound: float,
    k_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
    omega_bias: torch.Tensor,
    mix_r_bias: torch.Tensor,
    mix_theta_bias: torch.Tensor,
    mix_k_prev_bias: torch.Tensor,
    mix_k_curr_bias: torch.Tensor,
    b_scale: torch.Tensor | None,
    c_scale: torch.Tensor | None,
    output_dtype: torch.dtype,
    final_state_out: torch.Tensor | None = None,
    b_last_out: torch.Tensor | None = None,
    u_last_out: torch.Tensor | None = None,
    out_proj_weight: torch.Tensor | None = None,
    projected_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, P, N = _validate_decode_inputs(
        value,
        params,
        bc,
        gate,
        skip,
        initial_states,
        B_prev,
        U_prev,
    )
    if value.device.type != "cuda":
        raise ValueError("mixer_decode_step_cute requires CUDA tensors.")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "mixer_decode_step_cute supports only float16 and bfloat16 activations."
        )
    if output_dtype != value.dtype:
        raise ValueError("output_dtype must match the decode activation dtype.")
    if P != 64 or N != 64:
        raise ValueError(
            "mixer_decode_step_cute currently supports only P=64 and N=64."
        )
    if b_scale is None or c_scale is None:
        raise ValueError("mixer_decode_step_cute requires normalized BC scales.")
    fuse_outproj = out_proj_weight is not None or projected_out is not None
    if fuse_outproj and (out_proj_weight is None or projected_out is None):
        raise ValueError(
            "out_proj_weight and projected_out must be provided together for fused projection."
        )

    value_c = value if value.is_contiguous() else value.contiguous()
    params_c = params if params.is_contiguous() else params.contiguous()
    bc_c = bc if bc.is_contiguous() else bc.contiguous()
    gate_c = gate if gate.is_contiguous() else gate.contiguous()
    skip_c = skip if skip.is_contiguous() else skip.contiguous()
    state_c = (
        initial_states
        if initial_states is not None
        else _ensure_min_alignment(
            torch.zeros(
                (batch, heads, P, 2 * N), device=value.device, dtype=value.dtype
            ),
            min_align=_DECODE_MIN_ALIGN,
        )
    )
    b_prev_c = (
        B_prev
        if B_prev is not None
        else _ensure_min_alignment(
            torch.zeros((batch, heads, 2 * N), device=value.device, dtype=value.dtype),
            min_align=_DECODE_MIN_ALIGN,
        )
    )
    u_prev_c = (
        U_prev
        if U_prev is not None
        else _ensure_min_alignment(
            torch.zeros((batch, heads, P), device=value.device, dtype=value.dtype),
            min_align=_DECODE_MIN_ALIGN,
        )
    )

    y = _aligned_empty((batch, heads, P), device=value.device, dtype=output_dtype)
    final_state = (
        final_state_out if final_state_out is not None else _aligned_empty_like(state_c)
    )
    b_last = b_last_out if b_last_out is not None else _aligned_empty_like(b_prev_c)
    u_last = u_last_out if u_last_out is not None else _aligned_empty_like(u_prev_c)
    if tuple(map(int, final_state.shape)) != tuple(map(int, state_c.shape)):
        raise ValueError(
            f"final_state_out must be {tuple(state_c.shape)}. Got {tuple(final_state.shape)}."
        )
    if tuple(map(int, b_last.shape)) != tuple(map(int, b_prev_c.shape)):
        raise ValueError(
            f"b_last_out must be {tuple(b_prev_c.shape)}. Got {tuple(b_last.shape)}."
        )
    if tuple(map(int, u_last.shape)) != tuple(map(int, u_prev_c.shape)):
        raise ValueError(
            f"u_last_out must be {tuple(u_prev_c.shape)}. Got {tuple(u_last.shape)}."
        )
    if out_proj_weight is None:
        out_proj = _aligned_empty((1, heads, P), device=value.device, dtype=value.dtype)
        projected = _aligned_empty(
            (batch, 1, 1), device=value.device, dtype=torch.float32
        )
        d_model = 1
    else:
        d_model = int(out_proj_weight.shape[0])
        if out_proj_weight.ndim != 2 or tuple(map(int, out_proj_weight.shape)) != (
            d_model,
            heads * P,
        ):
            raise ValueError(
                "out_proj_weight must be (d_model, H*P). "
                f"Got {tuple(out_proj_weight.shape)} for H*P={heads * P}."
            )
        if projected_out is None:
            raise ValueError(
                "projected_out must be provided when out projection is fused."
            )
        if tuple(map(int, projected_out.shape)) != (batch, d_model):
            raise ValueError(
                f"projected_out must be {(batch, d_model)}. Got {tuple(projected_out.shape)}."
            )
        if projected_out.dtype != torch.float32:
            raise ValueError(
                "projected_out must use float32 for stable fused atomic accumulation."
            )
        if projected_out.device != value.device:
            raise ValueError(
                "projected_out must live on the same device as decode inputs."
            )
        out_proj = (
            out_proj_weight
            if out_proj_weight.is_contiguous()
            else out_proj_weight.contiguous()
        )
        projected = _aligned_empty(
            (batch, heads, d_model), device=value.device, dtype=torch.float32
        )
        projected.zero_()
    state_stride = cast(
        tuple[int, int, int, int],
        tuple(int(v) for v in state_c.stride()),
    )
    final_state_stride = cast(
        tuple[int, int, int, int],
        tuple(int(v) for v in final_state.stride()),
    )
    prev_b_stride = cast(
        tuple[int, int, int],
        tuple(int(v) for v in b_prev_c.stride()),
    )
    prev_u_stride = cast(
        tuple[int, int, int],
        tuple(int(v) for v in u_prev_c.stride()),
    )
    u_last_stride = cast(
        tuple[int, int, int],
        tuple(int(v) for v in u_last.stride()),
    )

    value_align = _assumed_align(value_c)
    params_align = _assumed_align(params_c)
    bc_align = _assumed_align(bc_c)
    gate_align = _assumed_align(gate_c)
    skip_align = _assumed_align(skip_c)
    state_align = _assumed_align(state_c)
    b_prev_align = _assumed_align(b_prev_c)
    u_prev_align = _assumed_align(u_prev_c)
    dt_bias_align = _assumed_align(dt_bias)
    gamma_bias_align = _assumed_align(gamma_bias)
    omega_bias_align = _assumed_align(omega_bias)
    mix_r_bias_align = _assumed_align(mix_r_bias)
    mix_theta_bias_align = _assumed_align(mix_theta_bias)
    mix_k_prev_bias_align = _assumed_align(mix_k_prev_bias)
    mix_k_curr_bias_align = _assumed_align(mix_k_curr_bias)
    b_scale_align = _assumed_align(b_scale)
    c_scale_align = _assumed_align(c_scale)
    y_align = _assumed_align(y)
    final_state_align = _assumed_align(final_state)
    u_last_align = _assumed_align(u_last)
    out_proj_align = _assumed_align(out_proj)
    projected_align = _assumed_align(projected)

    spec = (batch, heads, P, N)
    if fuse_outproj:
        tile_p, num_warps, vec_n = _select_fused_decode_tuning(
            batch=batch,
            heads=heads,
            p_size=P,
            d_model=d_model,
        )
    else:
        tile_p, num_warps, vec_n = _select_decode_tuning(
            batch=batch,
            heads=heads,
            p_size=P,
        )
    p_tiles = (P + tile_p - 1) // tile_p
    b_prev_aliases_output = (
        b_last_out is not None and b_last_out.data_ptr() == b_prev_c.data_ptr()
    )
    b_last_kernel = (
        _aligned_empty_like(b_prev_c)
        if b_prev_aliases_output and p_tiles > 1
        else b_last
    )
    b_last_kernel_align = _assumed_align(b_last_kernel)
    b_last_kernel_stride = cast(
        tuple[int, int, int],
        tuple(int(v) for v in b_last_kernel.stride()),
    )

    cache_key = (
        spec,
        tile_p,
        num_warps,
        vec_n,
        bool(fuse_outproj),
        int(d_model),
        int(value.device.index or 0),
        value.dtype,
        params_c.dtype,
        bc_c.dtype,
        gate_c.dtype,
        skip_c.dtype,
        state_c.dtype,
        b_prev_c.dtype,
        u_prev_c.dtype,
        y.dtype,
        final_state.dtype,
        b_last.dtype,
        u_last.dtype,
        out_proj.dtype,
        projected.dtype,
        value_align,
        params_align,
        bc_align,
        gate_align,
        skip_align,
        state_align,
        b_prev_align,
        u_prev_align,
        dt_bias_align,
        gamma_bias_align,
        omega_bias_align,
        mix_r_bias_align,
        mix_theta_bias_align,
        mix_k_prev_bias_align,
        mix_k_curr_bias_align,
        b_scale_align,
        c_scale_align,
        y_align,
        final_state_align,
        b_last_kernel_align,
        u_last_align,
        out_proj_align,
        projected_align,
        state_stride,
        final_state_stride,
        prev_b_stride,
        prev_u_stride,
        b_last_kernel_stride,
        u_last_stride,
        float(dt_min),
        float(dt_max),
        float(r_min),
        float(r_max),
        float(theta_bound),
        float(k_max),
        float(eps),
    )
    compiled = _DECODE_CACHE.get(cache_key)
    if compiled is None:
        if _is_cuda_graph_capturing(value.device):
            _raise_cold_capture_error("compile cache")
        compiled = cute.compile(
            MixerDecodeStepFwd(
                spec=spec,
                d_model=d_model,
                fuse_outproj=bool(fuse_outproj),
                state_stride=state_stride,
                final_state_stride=final_state_stride,
                prev_b_stride=prev_b_stride,
                prev_u_stride=prev_u_stride,
                b_last_stride=b_last_kernel_stride,
                u_last_stride=u_last_stride,
                state_align_bytes=state_align,
                tile_p=tile_p,
                num_warps=num_warps,
                vec_n=vec_n,
                normalize_bc=True,
                dt_min=dt_min,
                dt_max=dt_max,
                r_min=r_min,
                r_max=r_max,
                theta_bound=theta_bound,
                k_max=k_max,
                eps=eps,
            ),
            _make_fake_tensor_arg(value_c, align=value_align),
            _make_fake_tensor_arg(params_c, align=params_align),
            _make_fake_tensor_arg(bc_c, align=bc_align),
            _make_fake_tensor_arg(gate_c, align=gate_align),
            _make_fake_tensor_arg(skip_c, align=skip_align),
            _make_fake_tensor_arg(
                state_c,
                shape=tuple(int(v) for v in state_c.shape),
                stride=state_stride,
                align=state_align,
            ),
            _make_fake_tensor_arg(
                b_prev_c,
                shape=tuple(int(v) for v in b_prev_c.shape),
                stride=prev_b_stride,
                align=b_prev_align,
            ),
            _make_fake_tensor_arg(
                u_prev_c,
                shape=tuple(int(v) for v in u_prev_c.shape),
                stride=prev_u_stride,
                align=u_prev_align,
            ),
            _make_fake_tensor_arg(dt_bias, align=dt_bias_align),
            _make_fake_tensor_arg(gamma_bias, align=gamma_bias_align),
            _make_fake_tensor_arg(omega_bias, align=omega_bias_align),
            _make_fake_tensor_arg(mix_r_bias, align=mix_r_bias_align),
            _make_fake_tensor_arg(mix_theta_bias, align=mix_theta_bias_align),
            _make_fake_tensor_arg(mix_k_prev_bias, align=mix_k_prev_bias_align),
            _make_fake_tensor_arg(mix_k_curr_bias, align=mix_k_curr_bias_align),
            _make_fake_tensor_arg(b_scale, align=b_scale_align),
            _make_fake_tensor_arg(c_scale, align=c_scale_align),
            _make_fake_tensor_arg(y, align=y_align),
            _make_fake_tensor_arg(
                final_state,
                shape=tuple(int(v) for v in final_state.shape),
                stride=final_state_stride,
                align=final_state_align,
            ),
            _make_fake_tensor_arg(
                b_last_kernel,
                shape=tuple(int(v) for v in b_last_kernel.shape),
                stride=b_last_kernel_stride,
                align=b_last_kernel_align,
            ),
            _make_fake_tensor_arg(
                u_last,
                shape=tuple(int(v) for v in u_last.shape),
                stride=u_last_stride,
                align=u_last_align,
            ),
            _make_fake_tensor_arg(out_proj, align=out_proj_align),
            _make_fake_tensor_arg(projected, align=projected_align),
            _compile_env_stream_placeholder(),
            options="--enable-tvm-ffi",
        )
        _DECODE_CACHE[cache_key] = compiled
    cast(Callable[..., None], compiled)(
        value_c,
        params_c,
        bc_c,
        gate_c,
        skip_c,
        state_c,
        b_prev_c,
        u_prev_c,
        dt_bias,
        gamma_bias,
        omega_bias,
        mix_r_bias,
        mix_theta_bias,
        mix_k_prev_bias,
        mix_k_curr_bias,
        b_scale,
        c_scale,
        y,
        final_state,
        b_last_kernel,
        u_last,
        out_proj,
        projected,
    )
    if b_last_kernel is not b_last:
        b_last.copy_(b_last_kernel)
    if out_proj_weight is not None:
        torch.sum(projected, dim=1, out=projected_out)
    y_flat = cast(torch.Tensor, y.reshape(batch, heads * P).contiguous())
    return y_flat, final_state, b_last, u_last


__all__ = ["mixer_decode_step_cute"]
