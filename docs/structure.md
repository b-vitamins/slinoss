# Tree structure and dispatch

## Per-operator shape

Every operator lives in `slinoss/ops/<name>/` and has the same five parts:

- `reference.py` -- pure PyTorch. The mathematical authority. Holds the reference
  forward, the reference backward, and the gradient named types. An analytic
  backward large enough to stand alone takes its own `backward.py` beside it,
  still pure PyTorch, and carries the gradient named types with it. `so3ssd` is
  the one that does.
- `cute/` -- CuTe DSL kernels and their host wrappers, and nothing else: no
  autograd function, no public differentiable callable, no named type. An
  operator whose fast path is the C++ extension has no `cute/` at all; `conv` is
  the one.
- `backends.py` -- the forward and backward `Protocol`s, the `Registry`, the
  registrations.
- `interface.py` -- the one `torch.autograd.Function` and the one public
  callable.
- `__init__.py` -- the re-export surface.

A kernel is never the specification. If a kernel and the reference disagree, the
kernel is wrong until proven otherwise in float64.

## Dispatch

One entry point per operator, not one per implementation.

- `interface.py` holds the operator's single `torch.autograd.Function` and its
  single public callable, which takes `backend: str | None` and resolves before
  applying. Nothing else constructs an autograd function for that operator.
- `backends.py` registers the reference at priority 0 over `("cpu", "cuda")` and
  CuTe at priority 10 over `("cuda",)`, behind a CUDA-availability check and then
  an `ImportError` guard. A tree with no CUDA and no DSL still imports and
  resolves to the reference.
- The gradient named type lives in `reference.py`. In the kernel module it would
  force the reference backend to import a kernel to name its own return type.
- Resolution is on device type and activation dtype. Shape is not a resolution
  axis, so a shape a kernel cannot hold raises rather than resolving to another
  backend. A shared-memory bound is a bound on the configuration, not a fallback
  trigger.
- A direction implemented as several launches has one driver module in `cute/`
  that sequences them, and `backends.py` registers the driver, never a kernel.
  The backward driver rematerializes what it needs from the saved inputs; a
  quantity the forward saved only for the backward's convenience is activation
  memory the operator does not need.
- No `torch.amp.custom_fwd`. It casts every input to the autocast dtype, which is
  the opposite of I4. The backend decides the promotion.

## Layout

```
slinoss/
  mixer.py        SLinOSSMixer
  blocks.py       SLinOSSBlock
  stack.py        SLinOSSStack
  config.py       SLinOSSConfig, and the shape multiples every operator asserts
  state.py        inference state containers
  decode.py       single-token decode path
  graph.py        CUDA-graph capture and replay
  _precision.py   float32-pinning policy, dtype sets
  _registry.py    Backend and Registry, shared by every operator
  _cute.py        the one device-side helper set and the executor cache
  _guard.py       host-side layout, device, and dtype checks
  _reduce.py      the row reduction of a per-block partial buffer
  ops/
    so3ssd/       the scan operator
      cute/
        common.py   device math, tile arithmetic, table slots
        mma.py      the four GEMM forms, MMA constants, shared-memory pitches
        table.py    the 3x3 transform table and every staging helper
        prefix.py   both chunk-local prefixes and their adjoints
        guard.py    this operator's dtype sets and shape checks
        forward.py  the driver that sequences the three forward launches
        backward.py the driver that sequences the seven backward launches
        fwd/        chunk_increment, state_passing, chunk_scan
        bwd/        chunk_start, state_passing, chunk_input, chunk_vector, boundary
    scanprep/     parameter maps: rotation vector, log-scale, taps
    mixer/        fused rowwise mixer tail
    block/        fused norm and activation kernels
    conv/         causal conv1d; the fast path is the C++ extension, not CuTe
  _C/             causal conv1d extension bindings
  perf/           budget taxonomy, region timers, memory forensics, workload table
csrc/             causal conv1d CUDA and C++
tests/
scripts/{bench,perf,aot}/
examples/
assets/           plots only, 300 DPI minimum, vector where possible
```

`slinoss/_cute.py`, `_guard.py`, `_registry.py`, `_reduce.py`, and every shared
module under `ops/so3ssd/cute/` are single-implementation modules. A helper that
belongs in one of them is added there, never copied into a kernel module. A rule
shared by several kernels lives in the shared module even when only one kernel
reaches it today.

Every package directory has an `__init__.py`. A namespace package that works
in-tree and breaks in a wheel is a packaging bug. A subpackage `__init__.py`
under `cute/` carries a docstring and no re-exports; consumers import kernel
modules by full path.
