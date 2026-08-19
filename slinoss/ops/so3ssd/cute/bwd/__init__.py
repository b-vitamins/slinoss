"""Backward kernels for the SO(3) chunked scan.

Five kernels, run in this order after the forward's chunk increment and state
passing are rematerialized:

- ``chunk_start``: the cotangent of each chunk's start state.
- ``state_passing``: the reverse inter-chunk scan, in place over that buffer.
- ``chunk_input``: ``dU``, the streaming input carry, and the log-scale and
  chunk-transition cotangents.
- ``chunk_vector``: ``dB``, ``dC``, ``dtrans``, ``dK``, and the streaming vector
  carry.
- ``boundary``: the per-chunk-boundary row and the streaming terms.

One module per kernel, as in :mod:`slinoss.ops.so3ssd.cute.fwd`. The chunk-local
prefixes are recomputed inside each of them from ``trans``; they never cross a
kernel boundary.
"""
