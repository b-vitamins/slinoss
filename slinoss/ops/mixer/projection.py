"""Low-level projection split op used by ``SLinOSSMixer``."""

from typing import cast

import torch


def _promote_dtypes(*dtypes: torch.dtype) -> torch.dtype:
    result = dtypes[0]
    for dtype in dtypes[1:]:
        result = torch.promote_types(result, dtype)
    return result


def _to_dtype_if_needed(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype == dtype:
        return x
    return x.to(dtype=dtype)


def _projection_row_bounds(
    *,
    d_inner: int,
    param_proj_dim: int,
) -> tuple[int, int]:
    gate_value_end = 2 * int(d_inner)
    params_end = gate_value_end + int(param_proj_dim)
    return gate_value_end, params_end


class _SplitMixerProjectionFn(torch.autograd.Function):
    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        d_inner: int,
        param_proj_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_inner = int(d_inner)
        param_proj_dim = int(param_proj_dim)
        gate_value_end, params_end = _projection_row_bounds(
            d_inner=d_inner,
            param_proj_dim=param_proj_dim,
        )
        x_flat = x.reshape(-1, x.shape[-1])
        gate_value = torch.mm(x_flat, weight[:gate_value_end, :].t()).view(
            *x.shape[:-1],
            gate_value_end,
        )
        params = torch.mm(x_flat, weight[gate_value_end:params_end, :].t()).view(
            *x.shape[:-1],
            param_proj_dim,
        )
        bc_flat = torch.mm(x_flat, weight[params_end:, :].t()).view(
            *x.shape[:-1],
            weight.shape[0] - params_end,
        )
        gate, value_raw = torch.split(gate_value, [d_inner, d_inner], dim=-1)
        ctx.save_for_backward(x, weight)
        ctx.d_inner = d_inner
        ctx.param_proj_dim = param_proj_dim
        return gate, value_raw, params, bc_flat

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx,
        grad_gate: torch.Tensor,
        grad_value: torch.Tensor,
        grad_params: torch.Tensor,
        grad_bc_flat: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        x, weight = ctx.saved_tensors
        d_inner = int(ctx.d_inner)
        param_proj_dim = int(ctx.param_proj_dim)
        gate_value_end, params_end = _projection_row_bounds(
            d_inner=d_inner,
            param_proj_dim=param_proj_dim,
        )
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None, None, None

        grad_x: torch.Tensor | None = None
        grad_weight: torch.Tensor | None = None

        if ctx.needs_input_grad[0]:
            grad_input_dtype = _promote_dtypes(
                weight.dtype,
                grad_gate.dtype,
                grad_value.dtype,
                grad_params.dtype,
                grad_bc_flat.dtype,
            )
            grad_gate_flat = _to_dtype_if_needed(
                grad_gate.reshape(-1, d_inner),
                grad_input_dtype,
            )
            grad_value_flat = _to_dtype_if_needed(
                grad_value.reshape(-1, d_inner),
                grad_input_dtype,
            )
            grad_params_flat = _to_dtype_if_needed(
                grad_params.reshape(-1, param_proj_dim),
                grad_input_dtype,
            )
            grad_bc_flat_2d = _to_dtype_if_needed(
                grad_bc_flat.reshape(-1, weight.shape[0] - params_end),
                grad_input_dtype,
            )
            grad_x_flat = torch.mm(
                grad_bc_flat_2d,
                _to_dtype_if_needed(weight[params_end:, :], grad_input_dtype),
            )
            grad_x_flat.addmm_(
                grad_gate_flat,
                _to_dtype_if_needed(weight[:d_inner, :], grad_input_dtype),
            )
            grad_x_flat.addmm_(
                grad_value_flat,
                _to_dtype_if_needed(
                    weight[d_inner:gate_value_end, :],
                    grad_input_dtype,
                ),
            )
            grad_x_flat.addmm_(
                grad_params_flat,
                _to_dtype_if_needed(
                    weight[gate_value_end:params_end, :],
                    grad_input_dtype,
                ),
            )
            grad_x = _to_dtype_if_needed(grad_x_flat, x.dtype).view_as(x)

        if ctx.needs_input_grad[1]:
            grad_weight_dtype = _promote_dtypes(
                x.dtype,
                grad_gate.dtype,
                grad_value.dtype,
                grad_params.dtype,
                grad_bc_flat.dtype,
            )
            x_flat = _to_dtype_if_needed(x.reshape(-1, x.shape[-1]), grad_weight_dtype)
            grad_gate_flat = _to_dtype_if_needed(
                grad_gate.reshape(-1, d_inner),
                grad_weight_dtype,
            )
            grad_value_flat = _to_dtype_if_needed(
                grad_value.reshape(-1, d_inner),
                grad_weight_dtype,
            )
            grad_params_flat = _to_dtype_if_needed(
                grad_params.reshape(-1, param_proj_dim),
                grad_weight_dtype,
            )
            grad_bc_flat_2d = _to_dtype_if_needed(
                grad_bc_flat.reshape(-1, weight.shape[0] - params_end),
                grad_weight_dtype,
            )
            grad_weight = torch.empty_like(weight, dtype=grad_weight_dtype)
            grad_weight[:d_inner, :] = torch.mm(grad_gate_flat.t(), x_flat)
            grad_weight[d_inner:gate_value_end, :] = torch.mm(
                grad_value_flat.t(),
                x_flat,
            )
            grad_weight[gate_value_end:params_end, :] = torch.mm(
                grad_params_flat.t(),
                x_flat,
            )
            grad_weight[params_end:, :] = torch.mm(grad_bc_flat_2d.t(), x_flat)
            grad_weight = _to_dtype_if_needed(grad_weight, weight.dtype)

        return grad_x, grad_weight, None, None


def split_mixer_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    d_inner: int,
    param_proj_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _SplitMixerProjectionFn.apply(
            x,
            weight,
            int(d_inner),
            int(param_proj_dim),
        ),
    )


__all__ = ["_SplitMixerProjectionFn", "split_mixer_projection"]
