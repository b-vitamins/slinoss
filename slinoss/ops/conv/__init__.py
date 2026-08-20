"""Causal depthwise conv1d operator."""

from slinoss.ops.conv.backends import (
    Backend,
    ConvBackward,
    ConvForward,
    causal_conv1d_bwd_native,
    causal_conv1d_fwd_native,
    get,
    names,
    register,
    resolve,
)
from slinoss.ops.conv.interface import CausalConv1dFunction, causal_conv1d
from slinoss.ops.conv.reference import (
    ConvDims,
    ConvGrads,
    ConvStep,
    causal_conv1d_bwd_ref,
    causal_conv1d_ref,
    causal_conv1d_update_ref,
    check_cotangents,
    check_dx_out,
    check_operands,
    conv_output_shape,
    conv_state_shape,
)

__all__ = [
    "Backend",
    "CausalConv1dFunction",
    "ConvBackward",
    "ConvDims",
    "ConvForward",
    "ConvGrads",
    "ConvStep",
    "causal_conv1d",
    "causal_conv1d_bwd_native",
    "causal_conv1d_bwd_ref",
    "causal_conv1d_fwd_native",
    "causal_conv1d_ref",
    "causal_conv1d_update_ref",
    "check_cotangents",
    "check_dx_out",
    "check_operands",
    "conv_output_shape",
    "conv_state_shape",
    "get",
    "names",
    "register",
    "resolve",
]
