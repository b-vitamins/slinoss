"""The shared backend registry.

Every operator's ``register``, ``names``, ``get``, and ``resolve`` are bound methods
of one :class:`slinoss._registry.Registry`, so the resolution rule is tested here on
a registry the test owns rather than four times against four operator globals. An
operator's own test file checks only which backends it registers and what that makes
``resolve`` return -- that is operator-specific data, not the rule.

The backends below are labels. Resolution never calls a backend, so a sentinel
string is a truthful stand-in and a real kernel would only make the test slower.
"""

from __future__ import annotations

import pytest
import torch

from slinoss._registry import Backend, Registry

CPU_ONLY = ("cpu",)
CUDA_ONLY = ("cuda",)
BOTH = ("cpu", "cuda")

NARROW = (torch.bfloat16, torch.float16)
ALL_DTYPES = (*NARROW, torch.float32, torch.float64)


def entry(
    name: str,
    *,
    device_types: tuple[str, ...] = BOTH,
    dtypes: tuple[torch.dtype, ...] = ALL_DTYPES,
    priority: int = 0,
) -> Backend[str, str]:
    """A backend whose entry points are labels."""
    return Backend(
        name=name,
        forward=f"{name}-forward",
        backward=f"{name}-backward",
        device_types=device_types,
        dtypes=dtypes,
        priority=priority,
    )


def registry(*backends: Backend[str, str]) -> Registry[str, str]:
    """A fresh registry holding ``backends``."""
    reg: Registry[str, str] = Registry("test-operator")
    for backend in backends:
        reg.register(backend)
    return reg


def test_register_returns_the_backend_so_a_module_can_bind_in_one_statement() -> None:
    reg: Registry[str, str] = Registry("test-operator")
    backend = entry("reference")
    assert reg.register(backend) is backend


def test_names_are_sorted_regardless_of_registration_order() -> None:
    reg = registry(entry("native"), entry("reference"), entry("cute"))
    assert reg.names() == ("cute", "native", "reference")


def test_get_returns_the_named_backend() -> None:
    reg = registry(entry("reference"))
    assert reg.get("reference").forward == "reference-forward"


def test_duplicate_registration_is_rejected() -> None:
    """Two implementations under one name is exactly what the registry prevents."""
    reg = registry(entry("reference"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(entry("reference"))


def test_unknown_name_is_rejected() -> None:
    reg = registry(entry("reference"))
    with pytest.raises(ValueError, match="unknown backend"):
        reg.get("no-such-backend")


def test_highest_priority_wins() -> None:
    reg = registry(entry("reference"), entry("fast", priority=10))
    assert reg.resolve(None, "cpu", torch.float32).name == "fast"


def test_an_explicit_name_overrides_priority() -> None:
    reg = registry(entry("reference"), entry("fast", priority=10))
    assert reg.resolve("reference", "cpu", torch.float32).name == "reference"


def test_a_backend_that_does_not_support_the_device_is_skipped() -> None:
    reg = registry(
        entry("reference"), entry("fast", device_types=CUDA_ONLY, priority=10)
    )
    assert reg.resolve(None, "cpu", torch.float32).name == "reference"


def test_a_backend_that_does_not_support_the_dtype_is_skipped() -> None:
    # The reason resolution takes a dtype at all: the scan's tensor-core atom is
    # 16-bit, so a float32 activation has no fast path and the caller wants the
    # reference rather than an exception from inside a kernel.
    reg = registry(entry("reference"), entry("fast", dtypes=NARROW, priority=10))
    assert reg.resolve(None, "cuda", torch.float32).name == "reference"
    assert reg.resolve(None, "cuda", torch.bfloat16).name == "fast"


def test_a_named_backend_on_an_unsupported_device_is_rejected() -> None:
    reg = registry(entry("reference"))
    with pytest.raises(ValueError, match="supports"):
        reg.resolve("reference", "meta", torch.float32)


def test_a_named_backend_on_an_unsupported_dtype_is_rejected() -> None:
    reg = registry(entry("fast", dtypes=NARROW))
    with pytest.raises(ValueError, match="supports"):
        reg.resolve("fast", "cpu", torch.float32)


def test_a_named_backend_reports_the_device_before_the_dtype() -> None:
    # Order is part of the contract: a call that violates both is reported under
    # the device rule, so one fix at a time is possible.
    reg = registry(entry("fast", device_types=CUDA_ONLY, dtypes=NARROW))
    with pytest.raises(ValueError, match="not 'cpu'"):
        reg.resolve("fast", "cpu", torch.float32)


def test_a_device_with_no_backend_is_rejected() -> None:
    reg = registry(entry("reference", device_types=CPU_ONLY))
    with pytest.raises(ValueError, match="no test-operator backend supports device"):
        reg.resolve(None, "meta", torch.float32)


def test_a_dtype_no_backend_supports_is_rejected() -> None:
    reg = registry(entry("fast", dtypes=NARROW))
    with pytest.raises(ValueError, match="no test-operator backend supports"):
        reg.resolve(None, "cpu", torch.float64)


def test_the_message_names_the_operator_so_it_says_which_registry_refused() -> None:
    reg: Registry[str, str] = Registry("so3ssd")
    reg.register(entry("reference", device_types=CPU_ONLY))
    with pytest.raises(ValueError, match="no so3ssd backend"):
        reg.resolve(None, "meta", torch.float32)


def test_two_registries_do_not_share_entries() -> None:
    """One registry per operator.

    A shared dict would let one operator resolve to another operator's kernel, which
    is a wrong answer rather than an error.
    """
    first = registry(entry("reference"))
    second: Registry[str, str] = Registry("other")
    assert second.names() == ()
    assert first.names() == ("reference",)
