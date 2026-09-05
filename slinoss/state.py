"""Mutable-buffer state for prefill and token decode."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from slinoss._precision import check_pinned, check_supported
from slinoss.config import SLinOSSConfig, SLinOSSMixerConfig

__all__ = ["MixerState", "StackState", "oscillator_basis"]


def _state_dtype(dtype: torch.dtype) -> torch.dtype:
    """Dtype of ``ssm`` under an activation dtype of ``dtype``."""
    return torch.float64 if dtype is torch.float64 else torch.float32


def _cyclic_basis(
    heads: int,
    rows: int,
    state_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """One deterministic unit carrier per recurrent row."""
    basis = torch.zeros(heads, rows, state_dim, device=device, dtype=dtype)
    column = torch.arange(heads * rows, device=device).reshape(heads, rows)
    column = column.remainder(state_dim)
    return basis.scatter_(-1, column[..., None], 1.0)


def oscillator_basis(
    config: SLinOSSMixerConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build the fixed homogeneous carrier ``[H,P,S]``."""
    return _cyclic_basis(
        config.n_heads,
        config.d_head,
        config.d_state,
        device=device,
        dtype=dtype,
    )


@dataclass(frozen=True)
class MixerState:
    """One layer's convolution, scan, and FOH decode carries."""

    conv: Tensor
    keys: Tensor | None
    ssm: Tensor
    b_prev: Tensor
    u_prev: Tensor

    def __post_init__(self) -> None:
        if self.conv.ndim != 3:
            raise ValueError(
                f"conv must be (B, d_conv - 1, d_inner), got {tuple(self.conv.shape)}"
            )
        if self.keys is not None and (
            self.keys.ndim != 3
            or int(self.keys.shape[1]) != int(self.conv.shape[1])
        ):
            raise ValueError(
                f"keys must be (B, d_conv - 1, 2*G*3N) over conv's own history "
                f"length, got {tuple(self.keys.shape)} against "
                f"{tuple(self.conv.shape)}"
            )
        if self.ssm.ndim != 4:
            raise ValueError(f"ssm must be (B,H,P,3N), got {tuple(self.ssm.shape)}")
        if self.b_prev.ndim != 3:
            raise ValueError(f"b_prev must be (B,G,3N), got {tuple(self.b_prev.shape)}")
        if self.u_prev.ndim != 3:
            raise ValueError(f"u_prev must be (B,H,P), got {tuple(self.u_prev.shape)}")
        check_supported(self.conv, "conv")
        check_pinned(self.ssm, "ssm")
        if self.conv.dtype is torch.float64 and self.ssm.dtype is not torch.float64:
            raise ValueError(
                f"conv is float64, so ssm must be float64 rather than "
                f"{self.ssm.dtype}; a narrower state downcasts the recurrence"
            )
        # The two carries join the activation group rather than the pinned one: they
        # are the operands B and U themselves at one token, not an accumulator. After
        # the pinning rule above, so a float64 activation dtype reports the state it
        # needs rather than the carry that already followed it.
        carries = (("b_prev", self.b_prev), ("u_prev", self.u_prev))
        if self.keys is not None:
            carries = (("keys", self.keys), *carries)
        for name, carry in carries:
            if carry.dtype is not self.conv.dtype:
                raise ValueError(
                    f"{name} is {carry.dtype} and conv is {self.conv.dtype}; "
                    f"one activation dtype only"
                )
        channels = int(self.conv.shape[2])
        heads, rows = (int(d) for d in self.ssm.shape[1:3])
        if channels != heads * rows:
            raise ValueError(
                f"conv holds {channels} channels and ssm holds {heads} heads of "
                f"{rows} rows; both are d_inner"
            )
        dim = int(self.ssm.shape[3])
        if tuple(self.u_prev.shape[1:]) != (heads, rows):
            raise ValueError(
                f"u_prev holds {tuple(self.u_prev.shape[1:])} and ssm holds "
                f"{(heads, rows)}; both are (H, P)"
            )
        groups = int(self.b_prev.shape[1])
        if int(self.b_prev.shape[2]) != dim:
            raise ValueError(
                f"b_prev holds {int(self.b_prev.shape[2])} lanes and ssm holds "
                f"{dim}; both are 3N"
            )
        if groups < 1 or heads % groups:
            raise ValueError(
                f"b_prev holds {groups} groups, which does not divide the {heads} "
                f"heads ssm holds"
            )
        if self.keys is not None and int(self.keys.shape[2]) != 2 * groups * dim:
            raise ValueError(
                f"keys holds {int(self.keys.shape[2])} channels and B and C hold "
                f"{2 * groups * dim} between them"
            )
        batches = {
            "conv": int(self.conv.shape[0]),
            "ssm": int(self.ssm.shape[0]),
            "b_prev": int(self.b_prev.shape[0]),
            "u_prev": int(self.u_prev.shape[0]),
        }
        if self.keys is not None:
            batches["keys"] = int(self.keys.shape[0])
        if len(set(batches.values())) != 1:
            raise ValueError(f"one batch only, got {batches}")
        devices = {
            "conv": self.conv.device,
            "ssm": self.ssm.device,
            "b_prev": self.b_prev.device,
            "u_prev": self.u_prev.device,
        }
        if self.keys is not None:
            devices["keys"] = self.keys.device
        if len(set(devices.values())) != 1:
            raise ValueError(f"one device only, got {devices}")

    @classmethod
    def allocate(
        cls,
        config: SLinOSSMixerConfig,
        batch: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> MixerState:
        """Allocate one state for a fixed positive ``batch``."""
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        basis = oscillator_basis(
            config, device=device, dtype=_state_dtype(dtype)
        ).unsqueeze(0)
        return cls(
            conv=torch.zeros(
                batch, config.d_conv - 1, config.d_inner, dtype=dtype, device=device
            ),
            keys=(
                torch.zeros(
                    batch,
                    config.d_conv - 1,
                    2 * config.n_groups * config.d_state,
                    dtype=dtype,
                    device=device,
                )
                if config.key_conv
                else None
            ),
            ssm=basis.expand(batch, -1, -1, -1).clone(),
            b_prev=torch.zeros(
                batch, config.n_groups, config.d_state, dtype=dtype, device=device
            ),
            u_prev=torch.zeros(
                batch, config.n_heads, config.d_head, dtype=dtype, device=device
            ),
        )

    def reset(self) -> None:
        """Restore every buffer to its initial value in place.

        In place because a captured graph holds these addresses; a fresh
        allocation is not the buffer the graph writes.
        """
        self.conv.zero_()
        if self.keys is not None:
            self.keys.zero_()
        basis = _cyclic_basis(
            int(self.ssm.shape[1]),
            int(self.ssm.shape[2]),
            int(self.ssm.shape[3]),
            device=self.ssm.device,
            dtype=self.ssm.dtype,
        )
        self.ssm.copy_(basis.unsqueeze(0).expand_as(self.ssm))
        self.b_prev.zero_()
        self.u_prev.zero_()

    def clone(self) -> MixerState:
        """Copy every buffer.

        Returns:
            A state sharing no storage with this one.
        """
        return MixerState(
            conv=self.conv.clone(),
            keys=None if self.keys is None else self.keys.clone(),
            ssm=self.ssm.clone(),
            b_prev=self.b_prev.clone(),
            u_prev=self.u_prev.clone(),
        )

    @property
    def batch(self) -> int:
        """Batch ``B``."""
        return int(self.ssm.shape[0])

    @property
    def device(self) -> torch.device:
        """Device every buffer lives on."""
        return self.ssm.device


@dataclass(frozen=True)
class StackState:
    """One :class:`MixerState` per layer, sharing batch and device."""

    layers: tuple[MixerState, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("layers must hold at least one MixerState")
        head = self.layers[0]
        for index, layer in enumerate(self.layers[1:], start=1):
            if layer.batch != head.batch:
                raise ValueError(
                    f"layer {index} has batch {layer.batch} and layer 0 has "
                    f"{head.batch}; one batch only"
                )
            if layer.device != head.device:
                raise ValueError(
                    f"layer {index} is on {layer.device} and layer 0 is on "
                    f"{head.device}; one device only"
                )

    @classmethod
    def allocate(
        cls,
        config: SLinOSSConfig,
        batch: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> StackState:
        """Allocate ``config.n_layers`` independent states."""
        # One allocation per layer. A repeated entry would alias one buffer
        # across the stack, so every layer would read its neighbour's state.
        return cls(
            layers=tuple(
                MixerState.allocate(config, batch, device=device, dtype=dtype)
                for _ in range(config.n_layers)
            )
        )

    def reset(self) -> None:
        """Restore every layer in place. See :meth:`MixerState.reset`."""
        for layer in self.layers:
            layer.reset()

    def clone(self) -> StackState:
        """Copy every layer.

        Returns:
            A state sharing no storage with this one.
        """
        return StackState(layers=tuple(layer.clone() for layer in self.layers))

    @property
    def batch(self) -> int:
        """Batch ``B``."""
        return self.layers[0].batch

    @property
    def device(self) -> torch.device:
        """Device every buffer lives on."""
        return self.layers[0].device
