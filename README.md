# SLinOSS

SLinOSS is a selective oscillatory state-space model built around a token-dependent
parametric oscillator. It follows the SSD/Mamba-style project-conv-scan layout,
but replaces scalar decays with a damped rotation and exact two-endpoint forcing:

`h_t = M_t h_{t-1} + K_{t-1} d_{t-1} + K_t d_t`

In this repo, the core pieces are:

- [`SLinOSSMixer`](./slinoss/layers/mixer.py): the paper-faithful mixer layer
- [`SLinOSSDiscretizer`](./slinoss/layers/discretization.py): bounded per-token oscillator parameterization and exact FOH taps
- [`v2x2ssd`](./slinoss/ops/v2x2ssd/reference.py): the current reference scan backend

## Install

For a pip-based local install, run:

```bash
pip install -r requirements.txt
```

For local development tools as well, run:

```bash
pip install -r requirements-dev.txt
```

The repo also ships a Guix environment in [`manifest.scm`](./manifest.scm), which
remains the reproducible path for CuTe/CUTLASS development.

## Example

For a minimal end-to-end run, see [`examples/nextchar.py`](./examples/nextchar.py).
It trains a small next-character model on `enwik8` using the default scan
backend and writes all artifacts to `/tmp/nextchar` by default.

Run it with:

```bash
python3 examples/nextchar.py
```
