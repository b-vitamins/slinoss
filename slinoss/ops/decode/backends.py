"""Backend registry for the one-token scan step.

The lookup itself is :class:`slinoss._registry.Registry`, which every operator
shares. This module holds only what is the step's own: the two call signatures, and
which implementations exist.

Resolution is on device type and activation dtype. Shape is not a resolution axis,
so nothing here can be selected by a token extent: the scan's registry cannot hand
a ``T == 1`` call to this operator, and this registry cannot hand a longer one back.
Which of the two boundaries a caller wants is a call-site decision, and
:meth:`slinoss.mixer.SLinOSSMixer.step` makes it by calling one of them.

The CuTe backend registers at priority 10 over ``("cuda",)`` with
:data:`slinoss._precision.KERNEL_DTYPES`, in a ``_register_cute`` guarded by
:func:`torch.cuda.is_available` and then by ``except ImportError: return``, exactly as
:mod:`slinoss.ops.so3ssd.backends` and :mod:`slinoss.ops.scanprep.backends` do. It
carries no backward: the operator has none, so both backends register the same raise.
float64 is outside ``KERNEL_DTYPES`` and stays on the reference, which is what makes
the float64 oracle a separate implementation rather than the same one at another
width.

That guard is also why :class:`slinoss.ops.decode.interface.DecodeResult` names the
backend that ran. An ``ImportError`` inside ``_register_cute`` -- a DSL that is not
installed, a kernel module that fails to import, an extension that was never built
-- leaves the registry holding the reference alone and every call silently taking
it. A measurement that cannot see which implementation answered is a measurement of
whichever one happened to be reachable.
"""

from __future__ import annotations

from typing import NoReturn, Protocol

import torch
from torch import Tensor

from slinoss._precision import KERNEL_DTYPES, SUPPORTED_DTYPES
from slinoss._registry import Backend, Registry
from slinoss.ops.decode.reference import decode_no_backward, decode_ref

__all__ = [
    "Backend",
    "DecodeBackend",
    "DecodeBackward",
    "DecodeForward",
    "get",
    "names",
    "register",
    "resolve",
]

REFERENCE = "reference"
CUTE = "cute"


class DecodeForward(Protocol):
    """Forward signature every backend implements.

    The three carries are keyword-only and are advanced in place. They are operands
    rather than options: a backend that ignored one would return a token computed
    from a state nobody advanced.
    """

    def __call__(
        self,
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
    ) -> Tensor: ...


class DecodeBackward(Protocol):
    """Backward signature every backend implements. Every one refuses.

    :class:`slinoss._registry.Backend` carries both directions, and the direction
    that does not exist is registered as a raise rather than left out, so a caller
    who reaches for it is told which operator to train through. See
    :func:`slinoss.ops.decode.reference.decode_no_backward`.
    """

    def __call__(self, *args: object, **kwargs: object) -> NoReturn: ...


DecodeBackend = Backend[DecodeForward, DecodeBackward]

_REGISTRY: Registry[DecodeForward, DecodeBackward] = Registry("decode")

register = _REGISTRY.register
names = _REGISTRY.names
get = _REGISTRY.get
resolve = _REGISTRY.resolve


register(
    Backend(
        name=REFERENCE,
        forward=decode_ref,
        backward=decode_no_backward,
        device_types=("cpu", "cuda"),
        dtypes=SUPPORTED_DTYPES,
        priority=0,
    )
)


def _register_cute() -> None:
    """Register the CuTe backend if this host can run it."""
    if not torch.cuda.is_available():
        return
    try:
        from slinoss.ops.decode.cute.step import decode_forward
    except ImportError:
        return
    register(
        Backend(
            name=CUTE,
            forward=decode_forward,
            backward=decode_no_backward,
            device_types=("cuda",),
            dtypes=KERNEL_DTYPES,
            priority=10,
        )
    )


_register_cute()
