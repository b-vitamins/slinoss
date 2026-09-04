"""One-token step of the SO(3) scan. Pure-PyTorch reference.

The chunked operator is the T-token path; this is the ``T = 1`` one. Both evaluate
the same map, so nothing is derived here. Every transition quantity comes from
:mod:`slinoss.ops.so3ssd.reference`, and the body below is the loop body of
:func:`slinoss.ops.so3ssd.so3ssm` with the token axis at extent one and the state
read from and written to the caller's own buffers.

The correspondence, term by term, against that body at token ``t``:

    here                                  so3ssm
    ------------------------------------  -------------------------------------
    w = trans[:, :, 0, :3]                w = trans[..., :3]
    scale = exp(2*trans[:, :, 0, 3])      scale = exp(2*trans[..., 3])
    rot = rot_matrix(quat_exp(w))         same
    kprev = tap_matrix(K[..., 0, :3], w)  same
    kcurr = tap_matrix(K[..., 1, :3], w)  same
    blane = as_lanes(to_heads(B[:,:,0]))  blane[:, :, t]
    clane = as_lanes(to_heads(C[:,:,0]))  clane[:, :, t]
    bp = as_lanes(to_heads(b_prev))       bp, the loop's carry
    up = u_prev                           up, the loop's carry
    scale * rot @ as_lanes(ssm)           scale[:,:,t] * einsum(rot[:,:,t], state)
    up * kprev @ bp                       up * vprev
    u * kcurr @ blane                     u[:,:,t] * vcurr
    y = <clane, state>                    outputs.append(...)
    ssm <- state, b_prev <- B, u_prev <- U  bp = blane[:,:,t]; up = u[:,:,t]

``tests/test_decode_op.py`` asserts the two agree in float64 at the tree's parity
bound rather than trusting the table.

Only :func:`slinoss.ops.so3ssd.so3ssm` is mirrored. The chunked factorization is
not: its change of basis into the chunk-local frame applies ``R(Q_t)^T`` to ``c_t``
(:func:`slinoss.ops.so3ssd.reference.transform_table`), which one token of scan
never reaches, so the transpose ``docs/operator.md`` disagrees with is outside this
path.

The state is advanced in place, in the caller's storage. See
:mod:`slinoss.ops.decode.interface` for why that is the signature and what it
costs.
"""

from __future__ import annotations

from typing import NamedTuple, NoReturn

import torch
from torch import Tensor

from slinoss._guard import check_pitched
from slinoss._precision import (
    autocast_disabled,
    cast_to,
    check_pinned,
    check_supported,
    pinned_dtype,
)
from slinoss.config import HEAD_MULTIPLE, LANE_MULTIPLE
from slinoss.ops.so3ssd.reference import (
    as_lanes,
    quat_exp,
    rot_matrix,
    tap_matrix,
    to_heads,
)

__all__ = [
    "TOKENS",
    "DecodeShapes",
    "check_operands",
    "decode_no_backward",
    "decode_ref",
]

TOKENS = 1
"""Token extent the boundary accepts.

The axis is kept rather than squeezed. Every operand is then byte-for-byte what
:meth:`slinoss.mixer.SLinOSSMixer.step` already hands the scan, including the
pitched ``B`` and ``C`` bands, and ``y`` is what the mixer tail already reads. A
squeezed boundary would need a reshape on both sides of itself.
"""


class DecodeShapes(NamedTuple):
    """Extents :func:`check_operands` read off the operands.

    Attributes:
        bsz: ``B``.
        heads: ``H``.
        groups: ``G``, dividing ``H``.
        rows: ``P``.
        state_dim: ``3N``.
        lanes: ``N``.
    """

    bsz: int
    heads: int
    groups: int
    rows: int
    state_dim: int
    lanes: int


def check_operands(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    ssm: Tensor,
    b_prev: Tensor,
    u_prev: Tensor,
) -> DecodeShapes:
    """Validate the operand set. The contract every backend shares.

    The order is rank, then the token extent, then shapes, then the shape
    multiples, then layout, then dtypes, then the two state rules: a wrong-shaped
    operand reports its shape rather than an alignment its offset also violates.
    The rules through layout are the scan's own, restated over a one-token
    boundary; the two after them are this operator's, because the scan allocates
    its state and this one writes the caller's.

    Args:
        U: Input weights, ``(B,H,TOKENS,P)``.
        trans: Packed transition, ``(B,H,TOKENS,4)``.
        K: Packed taps, ``(B,H,TOKENS,2,4)``.
        B: Input vectors, ``(B,G,TOKENS,3N)``. Contiguous or one pitched band of a
            wider tensor.
        C: Output vectors, ``(B,G,TOKENS,3N)``. Like ``B``.
        ssm: Recurrent state, ``(B,H,P,3N)``, advanced in place.
        b_prev: Previous token's vector, ``(B,G,3N)``, advanced in place.
        u_prev: Previous token's input, ``(B,H,P)``, advanced in place.

    Returns:
        A :class:`DecodeShapes`.

    Raises:
        ValueError: On a rank, token-extent, shape, shape-multiple, contiguity,
            pitch, device, state-dtype, or storage-sharing violation.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    if U.ndim != 4:
        raise ValueError(f"U must be (B,H,{TOKENS},P), got shape {tuple(U.shape)}")
    if B.ndim != 4:
        raise ValueError(f"B must be (B,G,{TOKENS},3N), got shape {tuple(B.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in U.shape)
    groups = int(B.shape[1])
    state_dim = int(B.shape[-1])
    # Not a loop over T. A boundary that accepted more would be a second scan
    # implementation, and the extent is what a kernel specializes on.
    if seqlen != TOKENS:
        raise ValueError(
            f"the decode boundary takes exactly {TOKENS} token, got T={seqlen}; "
            f"slinoss.ops.so3ssd.so3ssd is the T-token path"
        )
    if groups < 1 or heads % groups != 0:
        raise ValueError(
            f"B and C carry G groups with G dividing H; got G={groups}, H={heads}"
        )

    named: list[tuple[str, Tensor, tuple[int, ...]]] = [
        ("trans", trans, (bsz, heads, TOKENS, 4)),
        ("K", K, (bsz, heads, TOKENS, 2, 4)),
        ("B", B, (bsz, groups, TOKENS, state_dim)),
        ("C", C, (bsz, groups, TOKENS, state_dim)),
        ("ssm", ssm, (bsz, heads, rows, state_dim)),
        ("b_prev", b_prev, (bsz, groups, state_dim)),
        ("u_prev", u_prev, (bsz, heads, rows)),
    ]
    for name, tensor, shape in named:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )

    if state_dim % 3 != 0 or (state_dim // 3) % LANE_MULTIPLE != 0:
        raise ValueError(
            f"3N must be 3 times a multiple of {LANE_MULTIPLE}, got 3N={state_dim}"
        )
    if rows % HEAD_MULTIPLE != 0:
        raise ValueError(f"P must be a multiple of {HEAD_MULTIPLE}, got P={rows}")

    # ``B`` and ``C`` arrive as column bands of the mixer's fused projection, so
    # their token stride is the projection width and demanding contiguity of them
    # would demand a copy of that projection. Every other operand owns its buffer.
    # The three carries own theirs twice over: they are written in place, and a
    # kernel that stores into a pitched carry stores at a pitch no allocation of
    # this state has.
    banded = {"B", "C"}
    for name, tensor in [("U", U), *((n, t) for n, t, _ in named)]:
        if name not in banded and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; no repacking is done")
        if tensor.device != U.device:
            raise ValueError(
                f"{name} is on {tensor.device}, U is on {U.device}; one device only"
            )
    # A contiguous band meets the pitched rule at a pitch equal to its row width,
    # so only a strided one is handed over. That rule's alignment clause is a
    # device rule, and this reference is the CPU oracle as well.
    bands = ((B, "B"), (C, "C"))
    check_pitched(tuple(one for one in bands if not one[0].is_contiguous()))

    check_supported(U, "U")
    check_supported(B, "B")
    check_supported(C, "C")
    check_supported(b_prev, "b_prev")
    check_supported(u_prev, "u_prev")
    check_pinned(trans, "trans")
    check_pinned(K, "K")
    check_pinned(ssm, "ssm")

    # ``ssm`` is ``z`` of :data:`slinoss._precision.PINNED_TENSORS`: float32, and
    # float64 only where the activations are. The scan allocates its own state and
    # can promote into it; this boundary writes the caller's buffer, so a state
    # narrower than the pinned dtype would downcast the recurrence at every step
    # and report the traffic it saved as a kernel win.
    dtype = pinned_dtype(U, trans, K, B, C)
    if ssm.dtype is not dtype:
        raise ValueError(
            f"ssm is {ssm.dtype} and the pinned dtype of this call is {dtype}; the "
            f"state is written in place and is never narrowed"
        )
    # The two carries are the operands B and U at one token, not accumulators, so
    # they join the activation group. Same rule as slinoss.state.MixerState.
    for name, carry, operand, source in (
        ("b_prev", b_prev, B, "B"),
        ("u_prev", u_prev, U, "U"),
    ):
        if carry.dtype is not operand.dtype:
            raise ValueError(
                f"{name} is {carry.dtype} and {source} is {operand.dtype}; the carry "
                f"is that operand at one token and carries its dtype"
            )

    # Every carry is read before it is written, so an operand in the same
    # allocation is read after it has been replaced. Storage identity rather than
    # an interval test: a carry is a persistent buffer and an operand is a
    # projection band, so no legitimate call puts the two in one allocation.
    operands = (("U", U), ("trans", trans), ("K", K), ("B", B), ("C", C))
    for name, carry in (("ssm", ssm), ("b_prev", b_prev), ("u_prev", u_prev)):
        base = carry.untyped_storage().data_ptr()
        for other, tensor in operands:
            if tensor.untyped_storage().data_ptr() == base:
                raise ValueError(
                    f"{name} shares storage with {other}; the state is advanced in "
                    f"place, so an operand in the same allocation is read after it "
                    f"has been overwritten"
                )

    return DecodeShapes(bsz, heads, groups, rows, state_dim, state_dim // 3)


def decode_ref(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    /,
    *,
    ssm: Tensor,
    b_prev: Tensor,
    u_prev: Tensor,
) -> Tensor:
    """Step the recurrence one token and advance the state in place.

    Args:
        U: Input weights, ``(B,H,TOKENS,P)``, activation dtype.
        trans: ``(w_x, w_y, w_z, ls)``, ``(B,H,TOKENS,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, ``(B,H,TOKENS,2,4)``, pinned. Tap index 0 is
            previous and 1 is current; lane 3 is ignored.
        B: Input vectors, ``(B,G,TOKENS,3N)``, activation dtype. Grouped: head
            ``h`` reads group ``h // (H // G)``.
        C: Output vectors, ``(B,G,TOKENS,3N)``, activation dtype. Grouped like
            ``B``.
        ssm: Recurrent state, ``(B,H,P,3N)``, pinned dtype, contiguous. Read, then
            overwritten with the state after this token.
        b_prev: ``b`` at the previous token, ``(B,G,3N)``, dtype of ``B``,
            contiguous. Read, then overwritten with ``B[:, :, 0]``.
        u_prev: ``u`` at the previous token, ``(B,H,P)``, dtype of ``U``,
            contiguous. Read, then overwritten with ``U[:, :, 0]``.

    Returns:
        ``y``, shape ``(B,H,TOKENS,P)``, dtype of ``U``, contiguous.

    Raises:
        ValueError: On a rank, token-extent, shape, shape-multiple, contiguity,
            pitch, device, state-dtype, or storage-sharing violation.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    check_operands(U, trans, K, B, C, ssm, b_prev, u_prev)
    heads = int(U.shape[1])
    dtype = ssm.dtype

    with autocast_disabled(U.device.type):
        w = cast_to(trans[:, :, 0, :3], dtype)
        scale = torch.exp(2.0 * cast_to(trans[:, :, 0, 3], dtype))
        rot = rot_matrix(quat_exp(w))
        kprev = tap_matrix(cast_to(K[:, :, 0, 0, :3], dtype), w)
        kcurr = tap_matrix(cast_to(K[:, :, 0, 1, :3], dtype), w)
        # The promotion is the oracle's, not a kernel's: a kernel reads the
        # activation band at its own width and accumulates in the pinned dtype.
        u = cast_to(U[:, :, 0], dtype)
        up = cast_to(u_prev, dtype)
        # B and C are grouped, everything else is per head. Broadcast once so the
        # step below is written per head throughout.
        blane = as_lanes(to_heads(cast_to(B[:, :, 0], dtype), heads))
        clane = as_lanes(to_heads(cast_to(C[:, :, 0], dtype), heads))
        bp = as_lanes(to_heads(cast_to(b_prev, dtype), heads))

        state = scale[..., None, None, None] * torch.einsum(
            "bhij,bhpnj->bhpni", rot, as_lanes(ssm)
        )
        vprev = torch.einsum("bhij,bhnj->bhni", kprev, bp)
        vcurr = torch.einsum("bhij,bhnj->bhni", kcurr, blane)
        state = (
            state
            + up[..., None, None] * vprev[..., None, :, :]
            + u[..., None, None] * vcurr[..., None, :, :]
        )
        y = torch.einsum("bhni,bhpni->bhp", clane, state)

    # After every read. The carries feed the next step's two-tap forcing, and the
    # state is the caller's buffer rather than a fresh tensor: see the interface.
    ssm.copy_(state.flatten(-2, -1))
    b_prev.copy_(B[:, :, 0])
    u_prev.copy_(U[:, :, 0])
    return y.to(U.dtype).unsqueeze(2).contiguous()


def decode_no_backward(*args: object, **kwargs: object) -> NoReturn:
    """Refuse a gradient.

    Registered as every decode backend's backward, and never called by a public
    path: :func:`slinoss.ops.decode.decode_step` builds no graph, so nothing
    reaches this except a caller who looked the backend up and called it. It is
    registered rather than left absent because :class:`slinoss._registry.Backend`
    carries both directions, and a backward that raises here is a backward that
    cannot be reached from a training step by accident.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "the decode boundary takes no gradient; slinoss.ops.so3ssd.so3ssd is the "
        "differentiable path and slinoss.ops.decode advances an inference state"
    )
