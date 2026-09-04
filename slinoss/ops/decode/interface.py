"""Public entry point for the one-token scan step. The decode operator boundary.

No :class:`torch.autograd.Function`. The step advances an inference state, and the
state it advances is the caller's own buffer, so there is no forward whose inputs a
graph could hold: :func:`slinoss.ops.so3ssd.so3ssd` is the differentiable path over
``T`` tokens and this one is not differentiable at any ``T``. A function whose
backward raised would still record a node and defer the failure to ``.backward()``,
which is a training step that fails after the forward rather than at it.

The state is mutated in place, in the caller's storage, and the mutation is the
signature rather than a returned tensor the caller copies. Two reasons, both
measured rather than stylistic.

Traffic. Per call, with ``s_a`` the activation element size and ``s_z`` the state's:

    read   U        B*H*P*s_a          write  y         B*H*P*s_a
    read   trans    16*B*H             write  ssm       B*H*P*3N*s_z
    read   K        32*B*H             write  b_prev    B*G*3N*s_a
    read   B        B*G*3N*s_a         write  u_prev    B*H*P*s_a
    read   C        B*G*3N*s_a
    read   ssm      B*H*P*3N*s_z

At ``B 1, H 16, P 64, 3N 96, G 1`` with bfloat16 activations and a float32 state
that is 793,920 B, of which the two state passes are 786,432 B: 99.06%. The step's
cost is one read and one write of ``ssm`` and nothing else, so a boundary that
returned a fresh state and left the caller to copy it into the carry would run four
passes over that 99% instead of two. ``y`` is a reduction of the new state and falls
out of the same pass.

Class. 12 flop per state element against those 8 bytes is 1.5 flop/byte against a
machine balance of 163 on this part, so the arithmetic is under 1% of the roofline
and the kernel that replaces the reference here is ``DRAM-bound``: at least 85% of
the bandwidth measured at its own footprint, per ``docs/kernels.md``. That footprint
is under L2 at batch 1, where the same document gives no verdict and the kernel is
named unjudged, so the judged shape is a batch whose state exceeds the cache.

Stability. CUDA-graph capture records buffer addresses. Every write here goes
through the tensor the caller passed, so a captured replay writes the buffers its
consumers read; a rebound field would leave replay writing memory nobody reads.
``y`` is allocated per call because it does not cross a replay.

The update is legal in place because at ``T = 1`` it is lane-local: a lane reads its
three components, rotates and scales within them, and writes the same three
addresses, so no term crosses a lane or a row. That holds only at one token, and only
for inference. :func:`slinoss.ops.so3ssd.so3ssd` keeps ``z0`` for its backward and so
cannot take this signature. :func:`slinoss.ops.decode.reference.decode_ref` forms the
new state out of place and copies it in at the end, so the oracle cannot hide a
lane-crossing error behind the aliasing a kernel is permitted.

The fusion boundary of this version is the ``T = 1`` scan recurrence and nothing
else. Outside it, in producer order: the fused input projection, the value
convolution, the key convolution, the parameter maps of
:mod:`slinoss.ops.scanprep`, and the mixer tail with its output projection. So the
call reads ``U``, ``trans``, ``K``, ``B`` and ``C`` out of global memory and writes
``y`` back to it.

Every later fusion moves that boundary outward without touching the state
semantics. Folding the value convolution in adds ``conv`` as a fourth in-place carry
and drops ``U`` from the arguments; folding the key convolution in adds ``keys`` and
drops ``B`` and ``C``; folding the parameter maps in drops ``trans`` and ``K`` for
the projection's parameter band and the head bias; folding the tail in replaces
``y`` with the mixed token. None of them changes ``ssm``, ``b_prev`` or ``u_prev``,
which are the three buffers a captured graph holds addresses for and the whole of
what one step of this operator carries.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from slinoss.ops.decode.backends import resolve

__all__ = ["DecodeResult", "decode_step"]


class DecodeResult(NamedTuple):
    """Return type of :func:`decode_step`.

    Attributes:
        y: Mixed token, shape ``(B,H,TOKENS,P)``, dtype of ``U``, contiguous. The
            token axis is kept at extent one, which is what the mixer tail reads.
        backend: Name of the implementation that ran. Reported rather than assumed:
            a backend registry whose kernel import failed resolves to the reference
            and answers every call, so a caller who cannot see which implementation
            answered cannot tell a kernel measurement from a torch one. See
            :mod:`slinoss.ops.decode.backends`.

    The advanced state is not a field. It is the caller's own three buffers, written
    in place; see the module docstring.
    """

    y: Tensor
    backend: str


@torch.no_grad()
def decode_step(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    *,
    ssm: Tensor,
    b_prev: Tensor,
    u_prev: Tensor,
    backend: str | None = None,
) -> DecodeResult:
    """Mix one token and advance the state in place. The public operator.

    Continuing from a state this function last wrote reproduces the whole-sequence
    result of :func:`slinoss.ops.so3ssd.so3ssd` over the same tokens, so stepping a
    sequence in any partition from a freshly allocated
    :class:`slinoss.state.MixerState` is the same operator.

    Not an autograd node, and no operand may carry a gradient: see the module
    docstring.

    A sequence start is spelled as zero carries, not as an omitted one. The
    ``T``-token path accepts ``b_prev=None`` because it allocates the state it
    returns; this one writes the caller's three buffers, and a carry that is
    ``None`` has no storage to write. Zeros are not a branch in the arithmetic: the
    previous tap is linear in ``b_prev`` and scaled by ``u_prev``, so a zero carry
    annihilates the term exactly, at every tap value including ``w = 0``.
    :meth:`slinoss.state.MixerState.allocate` already zeroes both.

    Args:
        U: Input weights, ``(B,H,TOKENS,P)``, activation dtype, contiguous.
        trans: ``(w_x, w_y, w_z, ls)``, ``(B,H,TOKENS,4)``, pinned, contiguous.
        K: Per-tap ``(kr, g, h, 0)``, ``(B,H,TOKENS,2,4)``, pinned, contiguous. Tap
            index 0 is previous and 1 is current; lane 3 is ignored.
        B: Input vectors, ``(B,G,TOKENS,3N)``, activation dtype. Contiguous or one
            pitched band of a wider tensor. Grouped: ``G`` divides ``H`` and head
            ``h`` reads group ``h // (H // G)``.
        C: Output vectors, ``(B,G,TOKENS,3N)``, activation dtype. Like ``B``.
        ssm: Recurrent state, ``(B,H,P,3N)``, contiguous, in the call's pinned
            dtype: float32, or float64 where the activations are float64.
            Overwritten with the state after this token.
        b_prev: ``b`` at the previous token, ``(B,G,3N)``, dtype of ``B``,
            contiguous. Overwritten with ``B[:, :, 0]``.
        u_prev: ``u`` at the previous token, ``(B,H,P)``, dtype of ``U``,
            contiguous. Overwritten with ``U[:, :, 0]``.
        backend: Backend name, or ``None`` to select the fastest registered backend
            for the device and the activation dtype.

    Returns:
        A :class:`DecodeResult`.

    Raises:
        ValueError: On a rank, token-extent, shape, shape-multiple, contiguity,
            pitch, device, state-dtype, or storage-sharing violation, on an operand
            that requires a gradient, or on an unusable backend.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    # Refused rather than detached. Detaching would return a tensor whose gradient
    # is silently zero, which is a training run that reports a number.
    for name, tensor in (
        ("U", U),
        ("trans", trans),
        ("K", K),
        ("B", B),
        ("C", C),
        ("ssm", ssm),
        ("b_prev", b_prev),
        ("u_prev", u_prev),
    ):
        if tensor.requires_grad:
            raise ValueError(
                f"{name} requires a gradient; the decode boundary takes none, and "
                f"slinoss.ops.so3ssd.so3ssd is the differentiable path"
            )
    impl = resolve(backend, U.device.type, U.dtype)
    y = impl.forward(U, trans, K, B, C, ssm=ssm, b_prev=b_prev, u_prev=u_prev)
    return DecodeResult(y=y, backend=impl.name)
