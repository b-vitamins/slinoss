"""Float32-pinning policy."""

from __future__ import annotations

import pytest
import torch

from slinoss._precision import (
    LOW_PRECISION_DTYPES,
    SUPPORTED_DTYPES,
    WIDE_DTYPES,
    autocast_disabled,
    check_pinned,
    check_supported,
    device_type_of,
    pinned_dtype,
)


def test_dtype_sets_partition() -> None:
    assert set(SUPPORTED_DTYPES) == set(LOW_PRECISION_DTYPES) | set(WIDE_DTYPES)
    assert not set(LOW_PRECISION_DTYPES) & set(WIDE_DTYPES)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_check_supported_accepts(dtype: torch.dtype) -> None:
    check_supported(torch.zeros(2, dtype=dtype), "U")


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.int32, torch.bool, torch.complex64]
)
def test_check_supported_rejects(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="supported"):
        check_supported(torch.zeros(2, dtype=dtype), "U")


@pytest.mark.parametrize("dtype", WIDE_DTYPES)
def test_check_pinned_accepts_wide(dtype: torch.dtype) -> None:
    check_pinned(torch.zeros(2, dtype=dtype), "trans")


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_check_pinned_rejects_low(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="pinned"):
        check_pinned(torch.zeros(2, dtype=dtype), "trans")


def test_check_pinned_rejects_unsupported() -> None:
    with pytest.raises(TypeError, match="supported"):
        check_pinned(torch.zeros(2, dtype=torch.int64), "trans")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_pinned_dtype_is_float32_without_float64(dtype: torch.dtype) -> None:
    assert pinned_dtype(torch.zeros(1, dtype=dtype)) is torch.float32


def test_pinned_dtype_promotes_to_float64() -> None:
    low = torch.zeros(1, dtype=torch.bfloat16)
    wide = torch.zeros(1, dtype=torch.float64)
    assert pinned_dtype(low, wide) is torch.float64
    assert pinned_dtype(wide, low) is torch.float64


def test_pinned_dtype_needs_an_operand() -> None:
    with pytest.raises(ValueError, match="at least one tensor"):
        pinned_dtype()


def test_device_type_of() -> None:
    assert device_type_of(torch.zeros(1)) == "cpu"


def test_autocast_disabled_keeps_matmul_wide() -> None:
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert (a @ b).dtype is torch.bfloat16
        with autocast_disabled("cpu"):
            assert (a @ b).dtype is torch.float32
        assert (a @ b).dtype is torch.bfloat16


def test_autocast_disabled_is_a_noop_outside_autocast() -> None:
    a = torch.randn(4, 4)
    with autocast_disabled("cpu"):
        assert (a @ a).dtype is torch.float32
