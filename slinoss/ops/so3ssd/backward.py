"""Analytic backward of the SO(3) chunked scan. Reference implementation.

Every chunk-local intermediate is rematerialized by calling
:func:`slinoss.ops.so3ssd.reference.chunked_forward` again. The recompute is
therefore the forward by construction and cannot drift from it. Only the operator
inputs cross the boundary between the two passes; nothing derived is saved.

Every gradient is a dense GEMM or an elementwise map under the transposed decay
mask. The one sequential piece is the reverse chunk recurrence, which carries a
``(B,H,P,3N)`` accumulator and runs once per chunk.

Correctness ground truth is float64 autograd through
:func:`slinoss.ops.so3ssd.reference.so3ssm`, never this derivation. A hand-derived
VJP shares its algebra with the code that implements it, so an algebra error
passes silently unless something independent disagrees.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import pad as _pad

from slinoss._precision import autocast_disabled, pinned_dtype
from slinoss.ops.so3ssd.reference import (
    as_lanes,
    chunk_pad,
    chunked_forward,
    from_heads,
    quat_exp_vjp,
    quat_prefix_scan_vjp,
    rot_matrix_vjp,
    tap_matrix,
    tap_matrix_vjp,
)

__all__ = ["SO3SSDGrads", "so3ssd_bwd_ref"]


class SO3SSDGrads(NamedTuple):
    """Cotangents of the operator inputs. Every field is contiguous.

    A field is ``None`` exactly when the corresponding input was ``None``, which
    is what :class:`torch.autograd.Function` expects for an absent optional
    argument.

    Attributes:
        dU: Shape ``(B,H,T,P)``, dtype of ``U``.
        dtrans: Shape ``(B,H,T,4)``, dtype of ``trans``.
        dK: Shape ``(B,H,T,2,4)``, dtype of ``K``. Lane 3 is exactly zero: the
            forward never reads it.
        dB: Shape ``(B,G,T,3N)``, dtype of ``B``. Summed over the heads of each
            group.
        dC: Shape ``(B,G,T,3N)``, dtype of ``C``. Summed like ``dB``.
        dz0: Shape ``(B,H,P,3N)`` or ``None``.
        db_prev: Shape ``(B,G,3N)`` or ``None``.
        du_prev: Shape ``(B,H,P)`` or ``None``.
    """

    dU: Tensor
    dtrans: Tensor
    dK: Tensor
    dB: Tensor
    dC: Tensor
    dz0: Tensor | None
    db_prev: Tensor | None
    du_prev: Tensor | None


def _unchunk(t: Tensor, seqlen: int) -> Tensor:
    """``(B,H,C,L,...) -> (B,H,T,...)``, dropping the zero-padded tail."""
    return t.flatten(2, 3)[:, :, :seqlen]


def _scatter_last(t: Tensor, length: int) -> Tensor:
    """``(B,H,C) -> (B,H,C,L)`` with ``t`` at ``l = L-1`` and zeros before it."""
    return _pad(t[..., None], (length - 1, 0))


def so3ssd_bwd_ref(
    dy: Tensor | None,
    dstate: Tensor | None,
    db_last: Tensor | None,
    du_last: Tensor | None,
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    *,
    z0: Tensor | None = None,
    b_prev: Tensor | None = None,
    u_prev: Tensor | None = None,
) -> SO3SSDGrads:
    """Cotangents of every input of :func:`so3ssd_ref`.

    Args:
        dy: Cotangent of ``y``, shape ``(B,H,T,P)``. ``None`` is zero.
        dstate: Cotangent of ``state``, shape ``(B,H,P,3N)``. ``None`` is zero.
        db_last: Cotangent of ``b_last``, shape ``(B,G,3N)``. ``None`` is zero.
        du_last: Cotangent of ``u_last``, shape ``(B,H,P)``. ``None`` is zero.
        U: Input weights, shape ``(B,H,T,P)``.
        trans: ``(w_x, w_y, w_z, ls)`` per token, shape ``(B,H,T,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned.
        B: Input vectors, shape ``(B,G,T,3N)``.
        C: Output vectors, shape ``(B,G,T,3N)``.
        chunk_size: Chunk length ``L``. Must match the forward.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned.
        b_prev: ``b_{-1}``, shape ``(B,G,3N)``.
        u_prev: ``u_{-1}``, shape ``(B,H,P)``.

    Returns:
        A :class:`SO3SSDGrads`.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation, or a
            non-positive ``chunk_size``. Raised by the rematerializing forward.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    fw = chunked_forward(
        U, trans, K, B, C, chunk_size, z0=z0, b_prev=b_prev, u_prev=u_prev
    )
    dtype = pinned_dtype(U, trans, K, B, C)
    length = fw.length
    seqlen = fw.seqlen
    n_chunks = int(fw.u.shape[2])

    with autocast_disabled(U.device.type):
        dy_c = (
            torch.zeros_like(fw.u)
            if dy is None
            else chunk_pad(dy.to(dtype).contiguous(), length)
        )
        acc = (
            torch.zeros_like(fw.state)
            if dstate is None
            else as_lanes(dstate.to(dtype).contiguous())
        )

        # y_off = exp(2*lp) * <crot, zstart>. The scalar is per (chunk, token),
        # so its cotangent is a row sum over P.
        expl = torch.exp(2.0 * fw.lprefix)[..., None]
        gram = torch.einsum("bhclni,bhcpni->bhclp", fw.crot, fw.zstart)
        dgram = dy_c * expl
        dlp = 2.0 * expl[..., 0] * (dy_c * gram).sum(-1)
        dcrot = torch.einsum("bhclp,bhcpni->bhclni", dgram, fw.zstart)
        dzstart = torch.einsum("bhclp,bhclni->bhcpni", dgram, fw.crot)

        # y_diag = (score_now * dmask) @ u + (score_prv * dmask) @ ushift.
        dm_now = torch.einsum("bhclp,bhcrp->bhclr", dy_c, fw.u)
        dm_prv = torch.einsum("bhclp,bhcrp->bhclr", dy_c, fw.ushift)
        du = torch.einsum("bhclr,bhclp->bhcrp", fw.score_now * fw.dmask, dy_c)
        dushift = torch.einsum("bhclr,bhclp->bhcrp", fw.score_prv * fw.dmask, dy_c)
        dscore_now = dm_now * fw.dmask
        dscore_prv = dm_prv * fw.dmask
        # The strictly upper triangle of dmask is exactly zero, so the cotangent
        # of the exponent is too: the causal mask needs no separate handling.
        dexpo = (dm_now * fw.score_now + dm_prv * fw.score_prv) * fw.dmask
        dlp = dlp + 2.0 * (dexpo.sum(-1) - dexpo.sum(-2))

        crot_f = fw.crot.flatten(-2, -1)
        bnow_f = fw.bnow.flatten(-2, -1)
        bprv_f = fw.bprv.flatten(-2, -1)
        dcrot_f = dcrot.flatten(-2, -1) + dscore_now @ bnow_f + dscore_prv @ bprv_f
        dbnow_f = dscore_now.transpose(-1, -2) @ crot_f
        dbprv_f = dscore_prv.transpose(-1, -2) @ crot_f

        # Reverse chunk recurrence. zstart[c] feeds both the transition into
        # chunk c+1 and y_off of chunk c, so its two cotangents meet here.
        chunk_rot = fw.table.rot[..., -1, :, :]
        chunk_scale = torch.exp(2.0 * fw.lprefix[..., -1])
        dinc_rev: list[Tensor] = []
        drot_rev: list[Tensor] = []
        dscale_rev: list[Tensor] = []
        for c in reversed(range(n_chunks)):
            rot_c = chunk_rot[:, :, c]
            scale_c = chunk_scale[:, :, c]
            start_c = fw.zstart[:, :, c]
            dinc_rev.append(acc)
            dscale_rev.append(
                (acc * torch.einsum("bhij,bhpnj->bhpni", rot_c, start_c)).sum(
                    (-3, -2, -1)
                )
            )
            drot_rev.append(
                scale_c[..., None, None]
                * torch.einsum("bhpni,bhpnj->bhij", acc, start_c)
            )
            acc = (
                scale_c[..., None, None, None]
                * torch.einsum("bhij,bhpni->bhpnj", rot_c, acc)
                + dzstart[:, :, c]
            )
        dinc = torch.stack(dinc_rev[::-1], dim=2)
        dchunk_rot = torch.stack(drot_rev[::-1], dim=2)
        dchunk_scale = torch.stack(dscale_rev[::-1], dim=2)

        # inc = R(Q_{L-1}) inc_local, one frame change per chunk.
        dinc_local = torch.einsum("bhcij,bhcpni->bhcpnj", chunk_rot, dinc)
        dchunk_rot = dchunk_rot + torch.einsum(
            "bhcpni,bhcpnj->bhcij", dinc, as_lanes(fw.inc_local)
        )
        dinc_local_f = dinc_local.flatten(-2, -1)

        # I6 again in reverse: the increment weight rides u, size P, not brot.
        duw = torch.einsum("bhcpd,bhcrd->bhcrp", dinc_local_f, bnow_f)
        dupw = torch.einsum("bhcpd,bhcrd->bhcrp", dinc_local_f, bprv_f)
        dbnow_f = dbnow_f + torch.einsum(
            "bhcpd,bhcrp->bhcrd", dinc_local_f, fw.u * fw.wgt
        )
        dbprv_f = dbprv_f + torch.einsum(
            "bhcpd,bhcrp->bhcrd", dinc_local_f, fw.ushift * fw.wgt
        )
        du = du + duw * fw.wgt
        dushift = dushift + dupw * fw.wgt
        dexpw = ((duw * fw.u).sum(-1) + (dupw * fw.ushift).sum(-1)) * fw.wgt[..., 0]

        # lp reaches the output through dmask, wgt, chunk_scale, and exp(2*lp).
        # wgt and chunk_scale both differentiate the last token of the chunk.
        dlp = dlp - 2.0 * dexpw
        dlp = dlp + _scatter_last(
            2.0 * (dexpw.sum(-1) + chunk_scale * dchunk_scale), length
        )

        # Rowwise change of basis. The 3x3 operand is shared by all N lanes, so
        # its cotangent is a lane reduction and the vector cotangent is a matvec.
        dcrot_n = as_lanes(dcrot_f)
        dbnow_n = as_lanes(dbnow_f)
        dbprv_n = as_lanes(dbprv_f)
        dac = torch.einsum("bhclni,bhclnj->bhclij", dcrot_n, as_lanes(fw.c))
        dan = torch.einsum("bhclni,bhclnj->bhclij", dbnow_n, as_lanes(fw.b))
        dap = torch.einsum("bhclni,bhclnj->bhclij", dbprv_n, as_lanes(fw.bshift))
        dc_n = torch.einsum("bhclij,bhclni->bhclnj", fw.table.ac, dcrot_n)
        db_n = torch.einsum("bhclij,bhclni->bhclnj", fw.table.an, dbnow_n)
        dbs_n = torch.einsum("bhclij,bhclni->bhclnj", fw.table.ap, dbprv_n)

        # Table composition: ac = R^T, ap = ac Kprev, an = ac Kcurr.
        kprev = tap_matrix(fw.tap[..., 0, :], fw.w)
        kcurr = tap_matrix(fw.tap[..., 1, :], fw.w)
        act = fw.table.ac.transpose(-1, -2)
        dac = dac + dap @ kprev.transpose(-1, -2) + dan @ kcurr.transpose(-1, -2)
        gprev = tap_matrix_vjp(act @ dap, fw.tap[..., 0, :], fw.w)
        gcurr = tap_matrix_vjp(act @ dan, fw.tap[..., 1, :], fw.w)
        dtap = torch.stack([gprev.tap, gcurr.tap], dim=-2)

        drot = dac.transpose(-1, -2) + _pad(
            dchunk_rot[..., None, :, :], (0, 0, 0, 0, length - 1, 0)
        )
        dquat = quat_prefix_scan_vjp(rot_matrix_vjp(drot, fw.qprefix), fw.qprefix)
        dw = gprev.w + gcurr.w + quat_exp_vjp(dquat, fw.w)
        dls = dlp.flip(-1).cumsum(-1).flip(-1)

        dU_t = _unchunk(du, seqlen)
        dB_t = _unchunk(db_n.flatten(-2, -1), seqlen)
        dushift_t = _unchunk(dushift, seqlen)
        dbshift_t = _unchunk(dbs_n.flatten(-2, -1), seqlen)

        # b_{t-1} and u_{t-1} are the same tensors read one token earlier, so
        # their cotangents shift forward by one and the head falls out as the
        # streaming feedback gradient.
        dU_t = dU_t + _pad(dushift_t[:, :, 1:], (0, 0, 0, 1))
        dB_t = dB_t + _pad(dbshift_t[:, :, 1:], (0, 0, 0, 1))
        if du_last is not None:
            dU_t = dU_t + _pad(du_last.to(dtype)[:, :, None], (0, 0, seqlen - 1, 0))

        dtrans_t = torch.cat(
            [_unchunk(dw, seqlen), _unchunk(dls, seqlen)[..., None]], dim=-1
        )
        # Lane 3 of K is a hard zero in the forward, so it is a hard zero here.
        dK_t = _pad(_unchunk(dtap, seqlen), (0, 1))

        # B and C are read by every head of their group, so their cotangents are
        # summed back over those heads. Identity when G == H.
        groups = int(B.shape[1])
        dB_g = from_heads(dB_t, groups)
        # b_last is a slice of the grouped B, not a per-head read of it, so its
        # cotangent lands after the group reduction. Added before it, one group's
        # cotangent would be counted once per head of that group.
        if db_last is not None:
            dB_g = dB_g + _pad(db_last.to(dtype)[:, :, None], (0, 0, seqlen - 1, 0))
        return SO3SSDGrads(
            dU=dU_t.to(U.dtype).contiguous(),
            dtrans=dtrans_t.to(trans.dtype).contiguous(),
            dK=dK_t.to(K.dtype).contiguous(),
            dB=dB_g.to(B.dtype).contiguous(),
            dC=from_heads(_unchunk(dc_n.flatten(-2, -1), seqlen), groups)
            .to(C.dtype)
            .contiguous(),
            dz0=None if z0 is None else acc.flatten(-2, -1).to(z0.dtype).contiguous(),
            db_prev=(
                None
                if b_prev is None
                else from_heads(dbshift_t[:, :, 0], groups)
                .to(b_prev.dtype)
                .contiguous()
            ),
            du_prev=(
                None
                if u_prev is None
                else dushift_t[:, :, 0].to(u_prev.dtype).contiguous()
            ),
        )
