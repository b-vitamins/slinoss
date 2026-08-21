"""Forward kernels: increment passing, chunk scan.

``chunk_increment`` and ``state_passing`` are the unfused pair ``increment_passing``
replaces. The forward launches neither; both stay as the arm the fusion is measured
against.
"""
