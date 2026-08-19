"""Shared fixtures.

Every operator input comes from the real parameter maps. A fabricated ``trans``
or ``K`` does not exercise the invariants the kernels rely on, and a fabricated
chunk-start state does not exercise chunk composition, so neither is built here.
"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import pytest
import torch
from torch import Tensor

from slinoss.ops.scanprep import scanprep_ref

W_MAX = 3.0


class ScanKwargs(TypedDict):
    """The optional operands, keyed by parameter name.

    Typed rather than ``dict[str, Tensor | None]`` so that ``**inp.kw()`` checks
    against the callee's own parameters instead of every keyword it accepts.
    """

    z0: Tensor | None
    b_prev: Tensor | None
    u_prev: Tensor | None


class ScanInputs(NamedTuple):
    """A valid call to either reference implementation."""

    U: Tensor
    trans: Tensor
    K: Tensor
    B: Tensor
    C: Tensor
    z0: Tensor | None
    b_prev: Tensor | None
    u_prev: Tensor | None

    def args(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Positional operands, in signature order."""
        return (self.U, self.trans, self.K, self.B, self.C)

    def kw(self) -> ScanKwargs:
        """Keyword operands, in signature order."""
        return {"z0": self.z0, "b_prev": self.b_prev, "u_prev": self.u_prev}

    def leaves(self) -> tuple[Tensor, ...]:
        """Every operand that carries a gradient, in gradient-name order:
        ``dU, dtrans, dK, dB, dC, dz0, db_prev, du_prev``."""
        candidates = (*self.args(), self.z0, self.b_prev, self.u_prev)
        return tuple(t for t in candidates if t is not None and t.requires_grad)


def make_inputs(
    *,
    bsz: int = 2,
    heads: int = 2,
    groups: int | None = None,
    seqlen: int = 40,
    rows: int = 16,
    lanes: int = 16,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    seed: int = 0,
    with_state: bool = True,
    streaming: bool = True,
    w_scale: float = 1.0,
    ls_bias: float = 0.0,
    w_max: float = W_MAX,
    requires_grad: bool = False,
    u_dtype: torch.dtype | None = None,
    bc_dtype: torch.dtype | None = None,
) -> ScanInputs:
    """Build a valid operator call.

    Args:
        bsz: Batch.
        heads: Heads.
        groups: Groups sharing one ``B``/``C`` pair. Defaults to ``heads``, which
            is the ungrouped case. Must divide ``heads``.
        seqlen: Tokens.
        rows: ``P``.
        lanes: ``N``. ``3N`` is the state width.
        dtype: Pinned dtype. float64 for oracle work.
        device: Device for every tensor.
        seed: Generator seed.
        with_state: Pass ``z0`` rather than defaulting to zero.
        streaming: Pass ``b_prev`` and ``u_prev``.
        w_scale: Multiplies the raw rotation vector. Large values drive ``|w|``
            to ``w_max``; zero gives ``w = 0`` exactly.
        ls_bias: Added to the raw log-scale. Positive means stronger decay.
        w_max: Rotation-vector bound handed to the parameter map.
        requires_grad: Mark every operand a differentiable leaf.
        u_dtype: Cast ``U`` and ``u_prev`` after construction. Defaults to
            ``dtype``. The streaming tail is part of ``U``, so it carries the
            same dtype.
        bc_dtype: Cast ``B``, ``C``, and ``b_prev`` after construction. Defaults
            to ``dtype``.

    Returns:
        A :class:`ScanInputs`.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=dtype, device=device)

    # At the default the vector operands keep the shapes they had before there was
    # a group axis, so the generator draws the same element counts in the same
    # order and no ungrouped case moves.
    bc_heads = heads if groups is None else groups
    state_dim = 3 * lanes
    params = scanprep_ref(
        rnd(bsz, heads, seqlen, 3) * w_scale,
        rnd(bsz, heads, seqlen) + ls_bias,
        rnd(bsz, heads, seqlen, 2, 3),
        w_max=w_max,
    )

    def leaf(t: Tensor, cast: torch.dtype | None = None) -> Tensor:
        out = t.detach().clone()
        if cast is not None:
            out = out.to(cast)
        return out.contiguous().requires_grad_(requires_grad)

    return ScanInputs(
        U=leaf(rnd(bsz, heads, seqlen, rows), u_dtype),
        trans=leaf(params.trans),
        K=leaf(params.K),
        B=leaf(rnd(bsz, bc_heads, seqlen, state_dim), bc_dtype),
        C=leaf(rnd(bsz, bc_heads, seqlen, state_dim), bc_dtype),
        z0=leaf(rnd(bsz, heads, rows, state_dim)) if with_state else None,
        b_prev=leaf(rnd(bsz, bc_heads, state_dim), bc_dtype) if streaming else None,
        u_prev=leaf(rnd(bsz, heads, rows), u_dtype) if streaming else None,
    )


def max_err(a: Tensor, b: Tensor) -> float:
    """Maximum absolute difference, computed in float64."""
    return float((a.detach().double() - b.detach().double()).abs().max())


def rel_err(a: Tensor, b: Tensor) -> float:
    """Maximum absolute difference over the reference magnitude."""
    scale = float(b.detach().double().abs().max())
    return max_err(a, b) / max(scale, torch.finfo(torch.float64).tiny)


_MEASURED: dict[str, tuple[float, float]] = {}


def assert_max_rel(got: Tensor, want: Tensor, bound: float, label: str) -> float:
    """Assert a maximum relative error and record the measured value.

    Recording the measured figure is what keeps a tolerance honest: a bound that
    never approaches its limit is a bound nobody has checked. Run with
    ``--tolerance-report`` to print every bound next to the worst error observed
    under it.
    """
    err = rel_err(got, want)
    seen = _MEASURED.get(label)
    _MEASURED[label] = (max(err, seen[0]) if seen else err, bound)
    assert err < bound, f"{label}: max relative error {err:.3e} exceeds {bound:.1e}"
    return err


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tolerance-report",
        action="store_true",
        help="print the worst relative error measured under every tolerance",
    )


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    if not config.getoption("--tolerance-report") or not _MEASURED:
        return
    rows = sorted(_MEASURED.items(), key=lambda kv: kv[1][0] / kv[1][1])
    terminalreporter.section("tolerance report")
    terminalreporter.write_line(f"{'measured':>10}  {'bound':>8}  {'used':>6}  label")
    for label, (err, bound) in rows:
        terminalreporter.write_line(
            f"{err:10.3e}  {bound:8.1e}  {err / bound:5.1%}  {label}"
        )


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Every device the reference must run on."""
    name = str(request.param)
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device(name)
