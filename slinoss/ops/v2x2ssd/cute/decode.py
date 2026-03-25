"""CuTe decode entry point for the one-token SLinOSS recurrent middle."""

from __future__ import annotations

from typing import Callable, cast

import cuda.bindings.driver as cuda
import torch
import cutlass.cute as cute

from slinoss.ops.scanprep.cute.common import make_ptr_arg

from .kernels.decode import MixerDecodeStepFwd

_DECODE_CACHE: dict[tuple[object, ...], object] = {}


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

    value_c = value if value.is_contiguous() else value.contiguous()
    params_c = params if params.is_contiguous() else params.contiguous()
    bc_c = bc if bc.is_contiguous() else bc.contiguous()
    gate_c = gate if gate.is_contiguous() else gate.contiguous()
    skip_c = skip if skip.is_contiguous() else skip.contiguous()
    state_c = (
        initial_states
        if initial_states is not None
        else torch.zeros(
            (batch, heads, P, 2 * N), device=value.device, dtype=value.dtype
        )
    )
    b_prev_c = (
        B_prev
        if B_prev is not None
        else torch.zeros((batch, heads, 2 * N), device=value.device, dtype=value.dtype)
    )
    u_prev_c = (
        U_prev
        if U_prev is not None
        else torch.zeros((batch, heads, P), device=value.device, dtype=value.dtype)
    )

    y = torch.empty((batch, heads, P), device=value.device, dtype=output_dtype)
    final_state = torch.empty_like(state_c)
    b_last = torch.empty_like(b_prev_c)
    u_last = torch.empty_like(u_prev_c)

    value_ptr, value_align = make_ptr_arg(value_c)
    params_ptr, params_align = make_ptr_arg(params_c)
    bc_ptr, bc_align = make_ptr_arg(bc_c)
    gate_ptr, gate_align = make_ptr_arg(gate_c)
    skip_ptr, skip_align = make_ptr_arg(skip_c)
    state_ptr, state_align = make_ptr_arg(state_c)
    b_prev_ptr, b_prev_align = make_ptr_arg(b_prev_c)
    u_prev_ptr, u_prev_align = make_ptr_arg(u_prev_c)
    dt_bias_ptr, dt_bias_align = make_ptr_arg(dt_bias)
    gamma_bias_ptr, gamma_bias_align = make_ptr_arg(gamma_bias)
    omega_bias_ptr, omega_bias_align = make_ptr_arg(omega_bias)
    mix_r_bias_ptr, mix_r_bias_align = make_ptr_arg(mix_r_bias)
    mix_theta_bias_ptr, mix_theta_bias_align = make_ptr_arg(mix_theta_bias)
    mix_k_prev_bias_ptr, mix_k_prev_bias_align = make_ptr_arg(mix_k_prev_bias)
    mix_k_curr_bias_ptr, mix_k_curr_bias_align = make_ptr_arg(mix_k_curr_bias)
    b_scale_ptr, b_scale_align = make_ptr_arg(b_scale)
    c_scale_ptr, c_scale_align = make_ptr_arg(c_scale)
    y_ptr, y_align = make_ptr_arg(y)
    final_state_ptr, final_state_align = make_ptr_arg(final_state)
    b_last_ptr, b_last_align = make_ptr_arg(b_last)
    u_last_ptr, u_last_align = make_ptr_arg(u_last)

    spec = (batch, heads, P, N)
    cache_key = (
        spec,
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
        b_last_align,
        u_last_align,
        float(dt_min),
        float(dt_max),
        float(r_min),
        float(r_max),
        float(theta_bound),
        float(k_max),
        float(eps),
    )
    compiled = _DECODE_CACHE.get(cache_key)
    current_stream = cuda.CUstream(
        torch.cuda.current_stream(device=value.device).cuda_stream
    )
    if compiled is None:
        compiled = cute.compile(
            MixerDecodeStepFwd(
                spec=spec,
                normalize_bc=True,
                dt_min=dt_min,
                dt_max=dt_max,
                r_min=r_min,
                r_max=r_max,
                theta_bound=theta_bound,
                k_max=k_max,
                eps=eps,
            ),
            value_ptr,
            params_ptr,
            bc_ptr,
            gate_ptr,
            skip_ptr,
            state_ptr,
            b_prev_ptr,
            u_prev_ptr,
            dt_bias_ptr,
            gamma_bias_ptr,
            omega_bias_ptr,
            mix_r_bias_ptr,
            mix_theta_bias_ptr,
            mix_k_prev_bias_ptr,
            mix_k_curr_bias_ptr,
            b_scale_ptr,
            c_scale_ptr,
            y_ptr,
            final_state_ptr,
            b_last_ptr,
            u_last_ptr,
            current_stream,
        )
        _DECODE_CACHE[cache_key] = compiled
    cast(Callable[..., None], compiled)(
        value_ptr,
        params_ptr,
        bc_ptr,
        gate_ptr,
        skip_ptr,
        state_ptr,
        b_prev_ptr,
        u_prev_ptr,
        dt_bias_ptr,
        gamma_bias_ptr,
        omega_bias_ptr,
        mix_r_bias_ptr,
        mix_theta_bias_ptr,
        mix_k_prev_bias_ptr,
        mix_k_curr_bias_ptr,
        b_scale_ptr,
        c_scale_ptr,
        y_ptr,
        final_state_ptr,
        b_last_ptr,
        u_last_ptr,
        current_stream,
    )
    y_flat = cast(torch.Tensor, y.reshape(batch, heads * P).contiguous())
    return y_flat, final_state.contiguous(), b_last.contiguous(), u_last.contiguous()


__all__ = ["mixer_decode_step_cute"]
