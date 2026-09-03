"""Device-side SO(3) math: the quaternion exponential, the prefix product, the
rotation matrix, and the tap chart.

Every kernel shares these definitions, so an error here is an error everywhere.
Bounds are absolute and stated in float64 unless a test says otherwise.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from slinoss.ops.so3ssd import (
    quat_conj,
    quat_exp,
    quat_mul,
    quat_prefix_scan,
    rot_matrix,
    skew,
    tap_matrix,
    transform_table,
)

# Below the chart's asymptote at 2*w_max < 2*pi.
ANGLES: tuple[float, ...] = (
    0.0,
    1e-14,
    1e-10,
    1e-6,
    1e-3,
    0.1,
    0.5,
    1.0,
    math.pi / 2,
    2.0,
    3.0,
    3.14159,
    4.0,
    5.0,
    6.0,
    2.0 * math.pi - 1e-6,
)
AXES_PER_ANGLE = 8


def _axes(count: int, dtype: torch.dtype = torch.float64) -> Tensor:
    gen = torch.Generator().manual_seed(11)
    raw = torch.randn(count, 3, generator=gen, dtype=torch.float64)
    return (raw / raw.norm(dim=-1, keepdim=True)).to(dtype)


def _sweep(dtype: torch.dtype = torch.float64) -> tuple[Tensor, Tensor, Tensor]:
    """Rotation vectors covering the reachable domain.

    Returns:
        ``(w, theta, axis)`` with shapes ``(A,K,3)``, ``(A,K)``, ``(A,K,3)``.
    """
    theta = torch.tensor(ANGLES, dtype=dtype)[:, None].expand(
        len(ANGLES), AXES_PER_ANGLE
    )
    axis = _axes(AXES_PER_ANGLE, dtype).expand(len(ANGLES), AXES_PER_ANGLE, 3)
    return theta[..., None] * axis, theta, axis


def _transcendental(theta: Tensor, axis: Tensor) -> Tensor:
    half = 0.5 * theta
    return torch.cat(
        [torch.cos(half)[..., None], torch.sin(half)[..., None] * axis], -1
    )


def _transcendental_of(w: Tensor) -> Tensor:
    """Closed form from ``w`` alone, safe at ``w = 0`` because ``sin(0) == 0``."""
    theta = w.norm(dim=-1, keepdim=True)
    axis = w / torch.where(theta > 0, theta, torch.ones_like(theta))
    return torch.cat([torch.cos(0.5 * theta), torch.sin(0.5 * theta) * axis], dim=-1)


def test_quat_exp_matches_transcendental_float64() -> None:
    w, theta, axis = _sweep()
    err = float((quat_exp(w) - _transcendental(theta, axis)).abs().max())
    # Truncation remains below float64 rounding over |w| < 2*pi.
    assert err < 1e-15, err


def test_quat_exp_matches_transcendental_float32() -> None:
    # Compare against the closed form at the float32-rounded input so the gap is
    # the series' own error, not the rounding of w.
    w32 = _sweep()[0].float()
    err = float((quat_exp(w32).double() - _transcendental_of(w32.double())).abs().max())
    assert err < 1e-6, err


def test_quat_exp_is_unit_norm() -> None:
    w, _, _ = _sweep()
    err = float((quat_exp(w).norm(dim=-1) - 1.0).abs().max())
    assert err < 1e-15, err


def test_quat_exp_at_zero_is_the_identity_exactly() -> None:
    q = quat_exp(torch.zeros(4, 3, dtype=torch.float64))
    assert torch.equal(q, torch.tensor([1.0, 0.0, 0.0, 0.0]).double().expand(4, 4))


def test_quat_exp_gradcheck() -> None:
    w = _sweep()[0].reshape(-1, 3)[::5].detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(quat_exp, (w,))


def test_quat_exp_gradcheck_at_zero() -> None:
    # The series is evaluated in s = |w|^2, so there is no sqrt and the map is
    # differentiable at w = 0. An implementation in |w| fails here.
    zero = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(quat_exp, (zero,))


def test_quat_exp_grad_is_finite_at_zero() -> None:
    zero = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
    quat_exp(zero).sum().backward()
    assert zero.grad is not None
    assert torch.isfinite(zero.grad).all()


def test_quat_mul_identity_and_conjugate() -> None:
    w, _, _ = _sweep()
    q = quat_exp(w)
    ident = torch.zeros_like(q)
    ident[..., 0] = 1.0
    assert float((quat_mul(q, ident) - q).abs().max()) < 1e-15
    assert float((quat_mul(ident, q) - q).abs().max()) < 1e-15
    assert float((quat_mul(q, quat_conj(q)) - ident).abs().max()) < 2e-15


def test_quat_mul_is_associative() -> None:
    q = quat_exp(_sweep()[0]).reshape(-1, 4)
    a, b, c = q[:32], q.flip(0)[:32], q[16:48]
    left = quat_mul(quat_mul(a, b), c)
    right = quat_mul(a, quat_mul(b, c))
    assert float((left - right).abs().max()) < 1e-15


def test_skew_is_the_cross_product() -> None:
    gen = torch.Generator().manual_seed(3)
    w = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    v = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    got = (skew(w) @ v[..., None]).squeeze(-1)
    assert float((got - torch.linalg.cross(w, v)).abs().max()) < 1e-15


def test_rot_matrix_is_a_rotation() -> None:
    w, _, _ = _sweep()
    rot = rot_matrix(quat_exp(w))
    eye = torch.eye(3, dtype=rot.dtype).expand_as(rot)
    # R is quadratic in q and R^T R is quartic, so orthogonality carries a few
    # ulp. Measured worst case 1.33e-15 over the sweep.
    assert float((rot.transpose(-1, -2) @ rot - eye).abs().max()) < 5e-15
    assert float((torch.linalg.det(rot) - 1.0).abs().max()) < 1e-14


def test_rot_matrix_equals_quaternion_conjugation() -> None:
    gen = torch.Generator().manual_seed(5)
    w, _, _ = _sweep()
    q = quat_exp(w).reshape(-1, 4)
    v = torch.randn(q.shape[0], 3, generator=gen, dtype=torch.float64)
    pure = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    conj = quat_mul(quat_mul(q, pure), quat_conj(q))[..., 1:]
    got = (rot_matrix(q) @ v[..., None]).squeeze(-1)
    assert float((got - conj).abs().max()) < 1e-14


def test_rot_matrix_is_a_homomorphism() -> None:
    q = quat_exp(_sweep()[0]).reshape(-1, 4)
    a, b = q[:48], q.flip(0)[:48]
    err = float(
        (rot_matrix(quat_mul(a, b)) - rot_matrix(a) @ rot_matrix(b)).abs().max()
    )
    assert err < 1e-14, err


def test_rot_matrix_matches_rodrigues() -> None:
    w, theta, axis = _sweep()
    sin, cos = torch.sin(theta)[..., None, None], torch.cos(theta)[..., None, None]
    outer = axis[..., :, None] * axis[..., None, :]
    eye = torch.eye(3, dtype=w.dtype)
    rodrigues = cos * eye + sin * skew(axis) + (1.0 - cos) * outer
    err = float((rot_matrix(quat_exp(w)) - rodrigues).abs().max())
    assert err < 1e-14, err


def _serial_prefix(q: Tensor) -> Tensor:
    out = [q[..., 0, :]]
    for t in range(1, q.shape[-2]):
        out.append(quat_mul(q[..., t, :], out[-1]))
    return torch.stack(out, dim=-2)


@pytest.mark.parametrize("length", [1, 2, 3, 5, 16, 17, 64, 65, 128])
def test_quat_prefix_scan_matches_the_serial_fold(length: int) -> None:
    gen = torch.Generator().manual_seed(length)
    w = torch.randn(2, length, 3, generator=gen, dtype=torch.float64)
    q = quat_exp(w)
    got = quat_prefix_scan(q)
    want = _serial_prefix(q)
    want = want / want.norm(dim=-1, keepdim=True)
    # Hillis-Steele reassociates the product, so the gap is reordering roundoff
    # and grows like length * eps.
    assert float((got - want).abs().max()) < 1e-13


@pytest.mark.parametrize("length", [1, 7, 64])
def test_quat_prefix_scan_is_renormalized(length: int) -> None:
    gen = torch.Generator().manual_seed(length)
    q = quat_exp(torch.randn(3, length, 3, generator=gen, dtype=torch.float64))
    assert float((quat_prefix_scan(q).norm(dim=-1) - 1.0).abs().max()) < 1e-15


def test_quat_prefix_scan_gradcheck() -> None:
    gen = torch.Generator().manual_seed(9)
    w = torch.randn(1, 6, 3, generator=gen, dtype=torch.float64, requires_grad=True)

    def fn(x: Tensor) -> Tensor:
        return quat_prefix_scan(quat_exp(x))

    assert torch.autograd.gradcheck(fn, (w,))


def test_tap_matrix_matches_the_polynomial() -> None:
    gen = torch.Generator().manual_seed(13)
    w = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    tap = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    v = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    kr, par, imag = tap.unbind(-1)
    want = (
        kr[:, None] * v
        + par[:, None] * (w * v).sum(-1, keepdim=True) * w
        + imag[:, None] * torch.linalg.cross(w, v)
    )
    got = (tap_matrix(tap, w) @ v[..., None]).squeeze(-1)
    assert float((got - want).abs().max()) < 1e-14


def test_tap_matrix_matches_the_axis_normal_form() -> None:
    # k_re = kr, k_par = kr + g|w|^2, k_im = h|w|. The normal form is singular at
    # w = 0, so it is only checked away from the origin.
    gen = torch.Generator().manual_seed(17)
    w = torch.randn(96, 3, generator=gen, dtype=torch.float64)
    tap = torch.randn(96, 3, generator=gen, dtype=torch.float64)
    v = torch.randn(96, 3, generator=gen, dtype=torch.float64)
    norm = w.norm(dim=-1, keepdim=True)
    axis = w / norm
    kr, par, imag = tap.unbind(-1)
    k_re, k_par, k_im = kr, kr + par * norm[:, 0] ** 2, imag * norm[:, 0]
    v_par = (axis * v).sum(-1, keepdim=True) * axis
    want = (
        k_par[:, None] * v_par
        + k_re[:, None] * (v - v_par)
        + k_im[:, None] * torch.linalg.cross(axis, v)
    )
    got = (tap_matrix(tap, w) @ v[..., None]).squeeze(-1)
    assert float((got - want).abs().max()) < 1e-13


def test_tap_matrix_at_zero_is_a_scaled_identity() -> None:
    tap = torch.tensor([[2.0, -3.0, 5.0]], dtype=torch.float64)
    got = tap_matrix(tap, torch.zeros(1, 3, dtype=torch.float64))
    assert torch.equal(got, 2.0 * torch.eye(3, dtype=torch.float64)[None])


def test_tap_matrix_gradcheck_at_zero() -> None:
    tap = torch.tensor([[0.5, -1.5, 2.5]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(tap_matrix, (tap, zero))


def test_transform_table_is_the_composition() -> None:
    gen = torch.Generator().manual_seed(19)
    w = torch.randn(2, 8, 3, generator=gen, dtype=torch.float64)
    tap = torch.randn(2, 8, 2, 3, generator=gen, dtype=torch.float64)
    qprefix = quat_prefix_scan(quat_exp(w))
    table = transform_table(w, tap, qprefix)
    eye = torch.eye(3, dtype=w.dtype).expand_as(table.rot)
    assert float((table.ac @ table.rot - eye).abs().max()) < 1e-15
    assert float((table.rot @ table.ac - eye).abs().max()) < 1e-15
    for index, matrix in ((0, table.ap), (1, table.an)):
        want = table.ac @ tap_matrix(tap[..., index, :], w)
        assert float((matrix - want).abs().max()) < 1e-15


def test_transform_table_rot_is_the_prefix_rotation() -> None:
    gen = torch.Generator().manual_seed(23)
    w = torch.randn(1, 5, 3, generator=gen, dtype=torch.float64)
    tap = torch.zeros(1, 5, 2, 3, dtype=torch.float64)
    qprefix = quat_prefix_scan(quat_exp(w))
    table = transform_table(w, tap, qprefix)
    step = rot_matrix(quat_exp(w))
    running = step[:, 0]
    for t in range(w.shape[1]):
        if t:
            running = step[:, t] @ running
        assert float((table.rot[:, t] - running).abs().max()) < 1e-14
