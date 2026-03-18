# SLinOSS

SLinOSS is a selective oscillatory state-space model built around a token-dependent
parametric oscillator. It follows the SSD/Mamba-style project-conv-scan layout,
but replaces scalar decays with a damped rotation and exact two-endpoint forcing:

`h_t = M_t h_{t-1} + K_{t-1} d_{t-1} + K_t d_t`

In this repo, the core pieces are:

- [`SLinOSSMixer`](./slinoss/layers/mixer.py): the paper-faithful mixer layer
- [`SLinOSSScanPrep`](./slinoss/layers/scanprep.py): reference scan-preparation boundary for bounded per-token oscillator parameterization and exact FOH taps
- [`v2x2ssd`](./slinoss/ops/v2x2ssd/reference.py): the current reference scan backend

## Install

For a local source install, run:

```bash
pip install .
```

For local development with CUDA/CuTe extras and repo tooling, run:

```bash
pip install -e .[cuda,dev]
```

The repo also ships a Guix environment in [`manifest.scm`](./manifest.scm), which
remains the reproducible path for CuTe/CUTLASS development.

For downstream repos that should install SLinOSS without building from source,
depend on a GitHub Releases wheel.

Reference / CPU installs can use the universal wheel:

```txt
slinoss @ https://github.com/b-vitamins/slinoss/releases/download/v0.1.0/slinoss-0.1.0-py3-none-any.whl
```

CUDA installs that need both the CuTe backend and the compiled
`setup.py` causal-conv extension should use the matching platform wheel plus the
`cuda` extra. For Linux x86_64 on Python 3.11, that looks like:

```txt
slinoss[cuda] @ https://github.com/b-vitamins/slinoss/releases/download/v0.1.0/slinoss-0.1.0-cp311-cp311-linux_x86_64.whl
```

In `pyproject.toml`, the equivalent is:

```toml
dependencies = [
  "slinoss[cuda] @ https://github.com/b-vitamins/slinoss/releases/download/v0.1.0/slinoss-0.1.0-cp311-cp311-linux_x86_64.whl",
]
```

Replace `v0.1.0` and `0.1.0` with the release tag and package version you want
to consume, and pick the wheel asset whose Python and platform tags match your
environment.

## Example

For a minimal end-to-end run, see [`examples/nextchar.py`](./examples/nextchar.py).
It trains a small next-character model on `enwik8` using the default scan
backend and writes all artifacts to `/tmp/nextchar` by default.

Run it with:

```bash
python3 examples/nextchar.py
```
