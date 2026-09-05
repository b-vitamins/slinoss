"""Shared fixtures.

Every operator input comes from the real parameter maps. A fabricated ``trans``
or ``K`` does not exercise the invariants the kernels rely on, and a fabricated
chunk-start state does not exercise chunk composition, so neither is built here.

The maps and not the frontier: the operator's contract is the packed ``(w, ls)``
and its taps, so that is what is built. The additive token/bias frontier has its own
coverage in ``tests/test_scanprep.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, TypedDict

import pytest
import torch
from torch import Tensor

from slinoss.ops.scanprep import LS_MAX_MAG, bounded_logscale, bounded_rotvec, foh_taps

W_MAX = 3.0

LS_DECAY = 0.0181
"""Per-token log-scale the cute fixtures run at, and the figure their bounds hold.

A 64-token chunk keeps a tenth of its amplitude at this rate and a 128-token chunk a
hundredth, so both ends of the ``K`` extent carry weight. Unbiased the same chunks
reach 1e-7 and 1e-14, under float32 epsilon, and a chunk-length GEMM is then tested
on a band next to the diagonal; undecayed it is tested on a constant.
"""

LS_BIAS = -math.log(LS_MAX_MAG / LS_DECAY - 1.0)
"""Raw bias reaching :data:`LS_DECAY` through ``-LS_MAX_MAG*sigmoid``, about -2.55.

Inverted rather than written down. The bias fixes the whole fixture's decay rate and
every cute tolerance was derived at one, so a change to ``LS_MAX_MAG`` that left a
literal behind would move every bound's regime without moving a bound.
"""


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
            to ``2*w_max``; zero gives ``w = 0`` exactly.
        ls_bias: Added to the raw log-scale. Positive means stronger decay.
        w_max: Rotation-vector chart scale handed to the parameter map.
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
    w = bounded_rotvec(rnd(bsz, heads, seqlen, 3) * w_scale, w_max)
    ls = bounded_logscale(rnd(bsz, heads, seqlen) + ls_bias)
    tap = foh_taps(w, ls)
    trans = torch.cat([w, ls[..., None]], dim=-1)
    packed = torch.cat([tap, torch.zeros_like(tap[..., :1])], dim=-1)

    def leaf(t: Tensor, cast: torch.dtype | None = None) -> Tensor:
        out = t.detach().clone()
        if cast is not None:
            out = out.to(cast)
        return out.contiguous().requires_grad_(requires_grad)

    return ScanInputs(
        U=leaf(rnd(bsz, heads, seqlen, rows), u_dtype),
        trans=leaf(trans),
        K=leaf(packed),
        B=leaf(rnd(bsz, bc_heads, seqlen, state_dim), bc_dtype),
        C=leaf(rnd(bsz, bc_heads, seqlen, state_dim), bc_dtype),
        z0=leaf(rnd(bsz, heads, rows, state_dim)) if with_state else None,
        b_prev=leaf(rnd(bsz, bc_heads, state_dim), bc_dtype) if streaming else None,
        u_prev=leaf(rnd(bsz, heads, rows), u_dtype) if streaming else None,
    )


def projection_band(bc: Tensor) -> Tensor:
    """A ``(B,G,T,3N)`` vector operand laid out as a band of a wider tensor.

    The mixer projects value, gate, ``B``, ``C`` and the parameters in one GEMM, so a
    vector operand reaches a kernel as a column band of that output: unit stride
    along ``3N``, and a token stride equal to the projection width. The group axis
    therefore strides less than the axis before it, which a band cut out of a
    head-major buffer would not reproduce. Two groups of padding sit ahead of the band
    and one behind, so neither the offset nor the pitch is the one a dedicated buffer
    would have.

    Args:
        bc: ``(B,G,T,3N)``, any dtype, on any device.

    Returns:
        A view holding the same values, same shape and dtype, with
        ``stride(-2) > shape[-1]``.
    """
    bsz, groups, seqlen, dim = bc.shape
    wide = torch.empty(bsz, seqlen, groups + 3, dim, dtype=bc.dtype, device=bc.device)
    band = wide[:, :, 2 : 2 + groups]
    band.copy_(bc.permute(0, 2, 1, 3))
    return band.permute(0, 2, 1, 3)


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


@pytest.fixture
def reference_smem_capacity(monkeypatch: pytest.MonkeyPatch) -> int:
    """Fix dispatch-policy tests to a device-independent shared-memory budget."""
    from slinoss import _cute

    capacity = 101_376
    monkeypatch.setattr(_cute, "get_smem_capacity_in_bytes", lambda: capacity)
    return capacity


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
