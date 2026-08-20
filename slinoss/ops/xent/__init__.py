"""Fused cross entropy over a padded class axis."""

from slinoss.ops.xent.backends import (
    Backend,
    XentBackend,
    XentBackward,
    XentForward,
    get,
    names,
    register,
    resolve,
)
from slinoss.ops.xent.interface import CrossEntropyFunction, cross_entropy
from slinoss.ops.xent.reference import (
    LABEL_DTYPES,
    XentGrads,
    XentState,
    xent_bwd_ref,
    xent_ref,
    xent_shape,
)

__all__ = [
    "LABEL_DTYPES",
    "Backend",
    "CrossEntropyFunction",
    "XentBackend",
    "XentBackward",
    "XentForward",
    "XentGrads",
    "XentState",
    "cross_entropy",
    "get",
    "names",
    "register",
    "resolve",
    "xent_bwd_ref",
    "xent_ref",
    "xent_shape",
]
