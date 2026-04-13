"""State containers for streaming SLinOSS execution."""

from dataclasses import dataclass, field

import torch


def _maybe_detach(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.detach()


def _maybe_clone(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.clone()


def _maybe_to(
    x: torch.Tensor | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor | None:
    if x is None:
        return None
    if device is not None and dtype is not None:
        return x.to(device=device, dtype=dtype)
    if device is not None:
        return x.to(device=device)
    if dtype is not None:
        return x.to(dtype=dtype)
    return x


def _copy_if_present_(dst: torch.Tensor | None, src: torch.Tensor | None) -> None:
    if dst is not None and src is not None:
        dst.copy_(src)


def _adopt_tensor(
    current: torch.Tensor | None,
    updated: torch.Tensor | None,
) -> torch.Tensor | None:
    if current is None or updated is None:
        return updated
    if (
        tuple(current.shape) != tuple(updated.shape)
        or current.device != updated.device
        or current.dtype != updated.dtype
    ):
        return updated
    if current is not updated:
        current.copy_(updated)
    return current


@dataclass
class ScanState:
    """Recurrent state for scan backends.

    All tensors use the canonical packed layout expected by ``v2x2ssd``:

    - ``state``: ``(batch, heads, P, 2N)``
    - ``b_prev``: ``(batch, groups, 2N)``
    - ``u_prev``: ``(batch, heads, P)``
    """

    state: torch.Tensor | None = None
    b_prev: torch.Tensor | None = None
    u_prev: torch.Tensor | None = None

    def copy_(self, other: "ScanState") -> "ScanState":
        _copy_if_present_(self.state, other.state)
        _copy_if_present_(self.b_prev, other.b_prev)
        _copy_if_present_(self.u_prev, other.u_prev)
        return self

    def adopt_(self, other: "ScanState") -> "ScanState":
        self.state = _adopt_tensor(self.state, other.state)
        self.b_prev = _adopt_tensor(self.b_prev, other.b_prev)
        self.u_prev = _adopt_tensor(self.u_prev, other.u_prev)
        return self

    def detach(self) -> "ScanState":
        return ScanState(
            state=_maybe_detach(self.state),
            b_prev=_maybe_detach(self.b_prev),
            u_prev=_maybe_detach(self.u_prev),
        )

    def clone(self) -> "ScanState":
        return ScanState(
            state=_maybe_clone(self.state),
            b_prev=_maybe_clone(self.b_prev),
            u_prev=_maybe_clone(self.u_prev),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ScanState":
        return ScanState(
            state=_maybe_to(self.state, device=device, dtype=dtype),
            b_prev=_maybe_to(self.b_prev, device=device, dtype=dtype),
            u_prev=_maybe_to(self.u_prev, device=device, dtype=dtype),
        )


@dataclass
class SLinOSSMixerState:
    """Streaming state for ``SLinOSSMixer``."""

    conv: torch.Tensor | None = None
    scan: ScanState = field(default_factory=ScanState)
    _engine: object | None = field(default=None, repr=False, compare=False)

    def copy_(self, other: "SLinOSSMixerState") -> "SLinOSSMixerState":
        _copy_if_present_(self.conv, other.conv)
        self.scan.copy_(other.scan)
        return self

    def adopt_(self, other: "SLinOSSMixerState") -> "SLinOSSMixerState":
        self.conv = _adopt_tensor(self.conv, other.conv)
        self.scan.adopt_(other.scan)
        return self

    def detach(self) -> "SLinOSSMixerState":
        return SLinOSSMixerState(
            conv=_maybe_detach(self.conv),
            scan=self.scan.detach(),
        )

    def clone(self) -> "SLinOSSMixerState":
        return SLinOSSMixerState(
            conv=_maybe_clone(self.conv),
            scan=self.scan.clone(),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SLinOSSMixerState":
        return SLinOSSMixerState(
            conv=_maybe_to(self.conv, device=device, dtype=dtype),
            scan=self.scan.to(device=device, dtype=dtype),
        )


__all__ = ["ScanState", "SLinOSSMixerState"]
