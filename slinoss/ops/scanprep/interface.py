"""Autograd entry point for the bounded parameter maps.

Saves ``w_raw`` and ``ls_raw``. The tap map is the identity, so ``tap_raw`` carries
nothing the backward needs, and neither packed output is read back: the backward
sees the two cotangents and the two raw operands. That is two saved tensors against
five operands and two outputs.

No ``torch.amp.custom_fwd``. It casts every input to the autocast dtype, which is
the opposite of I4: the maps produce float32 whatever the input width, and the
backend decides the promotion.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from slinoss.ops.scanprep.backends import get, resolve
from slinoss.ops.scanprep.reference import ScanParams

__all__ = ["ScanPrepFunction", "scanprep"]

_Packed = tuple[Tensor, Tensor]
_Grads = tuple[Tensor, Tensor, Tensor, None, None]


class ScanPrepFunction(torch.autograd.Function):
    """Differentiable bounded maps.

    Returns a positional tuple because :class:`torch.autograd.Function` requires
    one. :func:`scanprep` names the fields.
    """

    @staticmethod
    def forward(
        ctx: Any,
        w_raw: Tensor,
        ls_raw: Tensor,
        tap_raw: Tensor,
        w_max: float,
        backend_name: str,
    ) -> _Packed:
        out = get(backend_name).forward(w_raw, ls_raw, tap_raw, w_max=w_max)
        ctx.save_for_backward(w_raw, ls_raw)
        ctx.w_max = w_max
        ctx.backend_name = backend_name
        return out.trans, out.K

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dtrans: Tensor,
        dK: Tensor,
    ) -> _Grads:
        w_raw, ls_raw = ctx.saved_tensors
        grads = get(ctx.backend_name).backward(
            dtrans, dK, w_raw, ls_raw, w_max=ctx.w_max
        )
        return grads.dw_raw, grads.dls_raw, grads.dtap_raw, None, None


def scanprep(
    w_raw: Tensor,
    ls_raw: Tensor,
    tap_raw: Tensor,
    *,
    w_max: float,
    backend: str | None = None,
) -> ScanParams:
    """Bounded parameter maps. The public operator.

    Args:
        w_raw: Unconstrained rotation vectors, ``(B,H,T,3)``.
        ls_raw: Unconstrained log-scales, ``(B,H,T)``.
        tap_raw: Unconstrained taps ``(kr, g, h)``, ``(B,H,T,2,3)``.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.
        backend: Backend name, or ``None`` to select the fastest registered
            backend for the device and dtype.

    Returns:
        A :class:`slinoss.ops.scanprep.ScanParams`, float32 (I4).

    Raises:
        ValueError: On a shape, layout, device, or bound violation, or an unusable
            backend.
        TypeError: On an unsupported or mixed dtype.
    """
    impl = resolve(backend, w_raw.device.type, w_raw.dtype)
    trans, packed = cast(
        "_Packed",
        ScanPrepFunction.apply(w_raw, ls_raw, tap_raw, w_max, impl.name),
    )
    return ScanParams(trans=trans, K=packed)
