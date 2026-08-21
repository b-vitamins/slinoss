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
    ScanPrologue,
    as_lanes,
    check_grad_band,
    chunk_pad,
    chunked_forward,
    from_heads,
    quat_exp_vjp,
    quat_prefix_scan_vjp,
    rot_matrix_vjp,
    tap_matrix,
    tap_matrix_vjp,
)

__all__ = [
    "ChunkTransition",
    "ChunkedBackward",
    "SO3SSDGrads",
    "chunk_transition_cotangents",
    "chunked_backward",
    "so3ssd_bwd_ref",
]


class SO3SSDGrads(NamedTuple):
    """Cotangents of the operator inputs.

    Every field is contiguous except a ``dB`` or ``dC`` the caller supplied, which
    carries the layout it arrived with.

    A field is ``None`` exactly when the corresponding input was ``None``, which
    is what :class:`torch.autograd.Function` expects for an absent optional
    argument.

    Attributes:
        dU: Shape ``(B,H,T,P)``, dtype of ``U``. Carries a ``dU_init`` seed as an
            addend, never as a destination.
        dtrans: Shape ``(B,H,T,4)``, dtype of ``trans``.
        dK: Shape ``(B,H,T,2,4)``, dtype of ``K``. Lane 3 is exactly zero: the
            forward never reads it.
        dB: Shape ``(B,G,T,3N)``, dtype of ``B``. Summed over the heads of each
            group. The caller's destination itself when the call named one.
        dC: Shape ``(B,G,T,3N)``, dtype of ``C``. Summed like ``dB``, and a named
            destination like ``dB``.
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


class ChunkedBackward(NamedTuple):
    """The cotangents, and every backward quantity that spans two kernels.

    Produced by :func:`chunked_backward`. The counterpart of
    :class:`slinoss.ops.so3ssd.reference.ChunkedForward`: the backward does not
    factor into one pass, so the quantities its stages hand to each other are
    named here rather than left anonymous inside one function. That makes each
    stage's parity testable against float64 autograd through the reference
    instead of against a second hand-derivation of the same algebra.

    Time is chunked: an axis pair ``(C,L)`` replaces ``T``, with the ragged tail
    zero-padded. Every field is in the pinned working dtype and follows the
    trailing-``3N`` lane-major layout of the tensor contract, so nothing here
    needs reshaping to compare against a kernel's buffer.

    ``partial_bc`` has no entry: it is a split-sequence partial sum of ``dB``,
    an artifact of how wide a reduction one launch covers, and it has no
    counterpart in the factorization.

    Attributes:
        grads: The operator's cotangents, the public projection of the rest.
        dzstart: Cotangent of ``zstart``, ``(B,H,C,P,3N)``. The readout half of
            each chunk's start state, before the reverse chunk recurrence.
        dinc: Cotangent of each chunk's increment, ``(B,H,C,P,3N)``. The reverse
            chunk recurrence carries this.
        dz0: Cotangent of the initial state, ``(B,H,P,3N)``. The exit value of
            that recurrence, present whether or not ``z0`` was.
        dlogp_scan: The part of the log-scale-prefix cotangent that the diagonal
            and increment terms produce, ``(B,H,C,L)``.
        dlogp_off: The rest of it: the offset term and the chunk-transition
            scale, ``(B,H,C,L)``. Where the split falls is a consequence of which
            stage holds which operand, so both halves are named and neither is
            derivable from the other.
        dlogp: The sum of the two, ``(B,H,C,L)``. The reverse cumulative sum of
            this over the chunk is the log-scale cotangent.
        dchunk_rot: Cotangent of each chunk's closing rotation matrix,
            ``(B,H,C,3,3)``, row-major.
        dchunk_scale: Cotangent of each chunk's closing scale, ``(B,H,C)``.
        chunk_rot: Each chunk's closing rotation matrix, ``(B,H,C,3,3)``,
            row-major. A forward quantity, named here because it is the third
            operand of :func:`chunk_transition_cotangents`: without it a stage
            oracle cannot evaluate that contraction on the buffers a kernel is
            handed, only on the ones this function chose.
        chunk_scale: Each chunk's closing scale, ``(B,H,C)``. The other operand of
            the same contraction, under the contract of ``chunk_rot``.
        carry_u: The ``u_{t-1}`` cotangent of each chunk's first token,
            ``(B,H,C,P)``, which belongs to the previous chunk's last token.
            Index 0 is ``grads.du_prev``.
        carry_b: The ``b_{t-1}`` cotangent of each chunk's first token,
            ``(B,G,C,3N)``, summed over each group's heads. Index 0 is
            ``grads.db_prev``.
    """

    grads: SO3SSDGrads
    dzstart: Tensor
    dinc: Tensor
    dz0: Tensor
    dlogp_scan: Tensor
    dlogp_off: Tensor
    dlogp: Tensor
    dchunk_rot: Tensor
    dchunk_scale: Tensor
    chunk_rot: Tensor
    chunk_scale: Tensor
    carry_u: Tensor
    carry_b: Tensor


def _unchunk(t: Tensor, seqlen: int) -> Tensor:
    """``(B,H,C,L,...) -> (B,H,T,...)``, dropping the zero-padded tail."""
    return t.flatten(2, 3)[:, :, :seqlen]


def _scatter_last(t: Tensor, length: int) -> Tensor:
    """``(B,H,C) -> (B,H,C,L)`` with ``t`` at ``l = L-1`` and zeros before it."""
    return _pad(t[..., None], (length - 1, 0))


class ChunkTransition(NamedTuple):
    """One chunk transition's two cotangents.

    Produced by :func:`chunk_transition_cotangents`.

    Attributes:
        dchunk_rot: ``(B,H,3,3)``, row-major. The transition's share of the closing
            rotation's cotangent. A partial: the increment's frame change reaches
            the same matrix and its term is added by the caller.
        dchunk_scale: ``(B,H)``. The whole cotangent of the closing scale, which
            nothing else produces.
    """

    dchunk_rot: Tensor
    dchunk_scale: Tensor


def chunk_transition_cotangents(
    dinc: Tensor, zstart: Tensor, chunk_rot: Tensor, chunk_scale: Tensor
) -> ChunkTransition:
    """Differentiate ``chunk_scale * R(chunk_rot) zstart`` in its two parameters.

    One chunk, no ``C`` axis: the reverse chunk recurrence calls this per chunk and
    the reduction order follows from that. A stage oracle calls it on the state
    buffers a kernel was handed, so that the kernel is not charged for the rounding
    of its own inputs. Both outputs contract the same operand pair and nothing else,
    so a low-precision ``dinc`` or ``zstart`` moves them by that dtype's epsilon.

    Args:
        dinc: Cotangent of the chunk's increment, ``(B,H,P,N,3)``.
        zstart: The state entering the chunk, ``(B,H,P,N,3)``.
        chunk_rot: The chunk's closing rotation matrix, ``(B,H,3,3)``, row-major.
        chunk_scale: The chunk's closing scale, ``(B,H)``.

    Returns:
        A :class:`ChunkTransition`. Both fields take the promoted dtype of the
        operands.
    """
    return ChunkTransition(
        dchunk_rot=chunk_scale[..., None, None]
        * torch.einsum("bhpni,bhpnj->bhij", dinc, zstart),
        dchunk_scale=(dinc * torch.einsum("bhij,bhpnj->bhpni", chunk_rot, zstart)).sum(
            (-3, -2, -1)
        ),
    )


def _store(dest: Tensor | None, grad: Tensor) -> Tensor:
    """Put ``grad`` in ``dest``, or give it a buffer of its own.

    Copy, never accumulate: a destination is one column band of a buffer whose other
    columns belong to other operators, and no phase zeroed this band.

    Args:
        dest: The caller's destination, or ``None``.
        grad: The gradient, already in the destination's dtype.

    Returns:
        ``dest`` itself when the call named one, else a contiguous buffer.
    """
    if dest is None:
        return grad.contiguous()
    dest.copy_(grad)
    return dest


def chunked_backward(
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
    dB: Tensor | None = None,
    dC: Tensor | None = None,
    dU_init: Tensor | None = None,
) -> ChunkedBackward:
    """Differentiate the chunked factorization and keep every shared quantity.

    Vectorized over ``T``. The only Python loop is over chunks, in reverse, which
    is the recurrence the backward ``state_passing`` kernel owns.

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
        dB: Destination for the ``B`` cotangent, shape ``(B,G,T,3N)``, dtype and
            device of ``B``, possibly pitched. Written in full, never accumulated
            into and never zeroed first, and returned as this same object. ``None``
            allocates a contiguous buffer. See
            :func:`slinoss.ops.so3ssd.reference.check_grad_band`.
        dC: Destination for the ``C`` cotangent, shape ``(B,G,T,3N)``, under the
            contract of ``dB``.
        dU_init: Addend for the ``U`` cotangent, shape ``(B,H,T,P)``, dtype and
            device of ``U``, possibly pitched. Read and never written: the returned
            ``dU`` is this plus the cotangent of ``U``, and ``dU`` stays a buffer
            this function allocates. It joins the accumulation in the pinned working
            dtype, so the sum narrows to ``U``'s dtype once instead of twice.

    Returns:
        A :class:`ChunkedBackward`.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation, or a
            non-positive ``chunk_size``. Raised by the rematerializing forward.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    fw = chunked_forward(
        U, trans, K, B, C, chunk_size, z0=z0, b_prev=b_prev, u_prev=u_prev
    )
    # After the rematerializing forward, which validates the operands a caller's
    # buffer is measured against.
    if dB is not None:
        check_grad_band(dB, B, "dB")
    if dC is not None:
        check_grad_band(dC, C, "dC")
    if dU_init is not None:
        check_grad_band(dU_init, U, "dU_init")
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
        dlp_off = 2.0 * expl[..., 0] * (dy_c * gram).sum(-1)
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
        dlp_scan = 2.0 * (dexpo.sum(-1) - dexpo.sum(-2))

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
            trans_c = chunk_transition_cotangents(acc, start_c, rot_c, scale_c)
            dscale_rev.append(trans_c.dchunk_scale)
            drot_rev.append(trans_c.dchunk_rot)
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
        # Split by producer: the diagonal and increment terms are one stage's, the
        # offset and chunk-transition terms another's, and the two stages meet
        # across a launch. Summing them here in one expression would leave the
        # first stage's output unnamed and so untestable on its own.
        dlp_scan = dlp_scan - 2.0 * dexpw
        dlp_scan = dlp_scan + _scatter_last(2.0 * dexpw.sum(-1), length)
        dlp_off = dlp_off + _scatter_last(2.0 * chunk_scale * dchunk_scale, length)
        dlp = dlp_scan + dlp_off

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
        # The seed joins the accumulation rather than its result: one narrowing store
        # of the total, instead of narrowing both addends and adding afterwards.
        if dU_init is not None:
            dU_t = dU_t + dU_init.to(dtype)

        dtrans_t = torch.cat(
            [_unchunk(dw, seqlen), _unchunk(dls, seqlen)[..., None]], dim=-1
        )
        # Lane 3 of K is a hard zero in the forward, so it is a hard zero here.
        dK_t = _pad(_unchunk(dtap, seqlen), (0, 1))

        # B and C are read by every head of their group, so their cotangents are
        # summed back over those heads. Identity when G == H.
        groups = int(B.shape[1])
        dB_g = from_heads(dB_t, groups)
        dC_g = from_heads(_unchunk(dc_n.flatten(-2, -1), seqlen), groups)
        # b_last is a slice of the grouped B, not a per-head read of it, so its
        # cotangent lands after the group reduction. Added before it, one group's
        # cotangent would be counted once per head of that group.
        if db_last is not None:
            dB_g = dB_g + _pad(db_last.to(dtype)[:, :, None], (0, 0, seqlen - 1, 0))

        # Token 0 of each chunk carries into the previous chunk's last token, so
        # its two shift cotangents are the rows that cross a chunk boundary. Chunk
        # 0's rows have no previous chunk and are the streaming feedback instead.
        carry_u = dushift[:, :, :, 0, :]
        carry_b = from_heads(dbs_n[:, :, :, 0].flatten(-2, -1), groups)
        grads = SO3SSDGrads(
            dU=dU_t.to(U.dtype).contiguous(),
            dtrans=dtrans_t.to(trans.dtype).contiguous(),
            dK=dK_t.to(K.dtype).contiguous(),
            dB=_store(dB, dB_g.to(B.dtype)),
            dC=_store(dC, dC_g.to(C.dtype)),
            dz0=None if z0 is None else acc.flatten(-2, -1).to(z0.dtype).contiguous(),
            db_prev=(
                None
                if b_prev is None
                else carry_b[:, :, 0].to(b_prev.dtype).contiguous()
            ),
            du_prev=(
                None
                if u_prev is None
                else carry_u[:, :, 0].to(u_prev.dtype).contiguous()
            ),
        )
        return ChunkedBackward(
            grads=grads,
            dzstart=dzstart.flatten(-2, -1),
            dinc=dinc.flatten(-2, -1),
            dz0=acc.flatten(-2, -1),
            dlogp_scan=dlp_scan,
            dlogp_off=dlp_off,
            dlogp=dlp,
            dchunk_rot=dchunk_rot,
            dchunk_scale=dchunk_scale,
            chunk_rot=chunk_rot,
            chunk_scale=chunk_scale,
            carry_u=carry_u,
            carry_b=carry_b,
        )


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
    dB: Tensor | None = None,
    dC: Tensor | None = None,
    dU_init: Tensor | None = None,
    prologue: ScanPrologue | None = None,
) -> SO3SSDGrads:
    """Cotangents of every input of :func:`slinoss.ops.so3ssd.reference.so3ssd_ref`.

    A thin projection of :func:`chunked_backward` onto the operator's gradient
    contract.

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
        dB: Destination for the ``B`` cotangent, shape ``(B,G,T,3N)``, dtype and
            device of ``B``, possibly pitched. Written in full, never accumulated
            into and never zeroed first, and returned as this same object. ``None``
            allocates a contiguous buffer. See
            :func:`slinoss.ops.so3ssd.reference.check_grad_band`.
        dC: Destination for the ``C`` cotangent, shape ``(B,G,T,3N)``, under the
            contract of ``dB``.
        dU_init: Addend for the ``U`` cotangent, shape ``(B,H,T,P)``, dtype and
            device of ``U``, possibly pitched. Read and never written: the returned
            ``dU`` is this plus the cotangent of ``U``. Not a destination, and never
            returned by identity.
        prologue: Ignored. This backward rematerializes the whole forward, in the
            reference's own chunk-major representation, so the chunked kernels'
            chunk boundary buys it nothing. Present because
            :class:`slinoss.ops.so3ssd.backends.ScanBackward` carries it.

    Returns:
        A :class:`SO3SSDGrads`.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation, a
            non-positive ``chunk_size``, or a caller's buffer off the pitched-layout
            contract.
        TypeError: On an unsupported dtype, a low-precision pinned tensor, or a
            caller's buffer whose dtype is not that of the operand it belongs to.
    """
    return chunked_backward(
        dy,
        dstate,
        db_last,
        du_last,
        U,
        trans,
        K,
        B,
        C,
        chunk_size,
        z0=z0,
        b_prev=b_prev,
        u_prev=u_prev,
        dB=dB,
        dC=dC,
        dU_init=dU_init,
    ).grads
