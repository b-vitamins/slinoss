"""CuTe kernels for the one-token scan step.

Importing this package imports the CuTe DSL. The reference path is
:mod:`slinoss.ops.decode.reference` and needs neither the DSL nor a GPU. The public
entry point is :func:`slinoss.ops.decode.decode_step`, which dispatches here through
the registry.
"""

from slinoss.ops.decode.cute.step import (
    decode_forward,
    lane_exchange,
    lanes_per_thread,
    row_group,
    rows_per_block,
)

__all__ = [
    "decode_forward",
    "lane_exchange",
    "lanes_per_thread",
    "row_group",
    "rows_per_block",
]
