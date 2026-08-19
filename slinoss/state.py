"""Inference state containers for the single-token decode path.

:class:`MixerState` holds one layer, :class:`StackState` holds a stack. Both are
frozen: the buffers are allocated once and every later write goes through them
in place. CUDA-graph capture records buffer addresses, so a container that
rebinds a field leaves replay writing memory no consumer reads.

``ssm`` is float32 under every low-precision activation dtype, as I4 requires.
The one exception is a float64 activation dtype, which widens ``ssm`` to float64
so a float64 path stays an oracle end to end instead of meeting a narrower
state mid-recurrence.

No step counter. The decode path reads none, and a counter is state that can
disagree with the buffers it claims to describe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from slinoss._precision import check_pinned, check_supported
from slinoss.config import SLinOSSConfig

__all__ = ["MixerState", "StackState"]


def _state_dtype(dtype: torch.dtype) -> torch.dtype:
    """Dtype of ``ssm`` under an activation dtype of ``dtype``."""
    return torch.float64 if dtype is torch.float64 else torch.float32


@dataclass(frozen=True)
class MixerState:
    """Decode state of one mixer layer.

    Attributes:
        conv: Causal-convolution history, shape ``(B, d_conv - 1, d_inner)``,
            activation dtype. Time-major, so index ``-1`` is the newest token.
        ssm: Recurrent scan state, shape ``(B, H, P, 3N)``. float32, or float64
            when the activation dtype is float64. ``H * P`` is ``d_inner``.
    """

    conv: Tensor
    ssm: Tensor

    def __post_init__(self) -> None:
        if self.conv.ndim != 3:
            raise ValueError(
                f"conv must be (B, d_conv - 1, d_inner), got {tuple(self.conv.shape)}"
            )
        if self.ssm.ndim != 4:
            raise ValueError(f"ssm must be (B,H,P,3N), got {tuple(self.ssm.shape)}")
        check_supported(self.conv, "conv")
        check_pinned(self.ssm, "ssm")
        if self.conv.dtype is torch.float64 and self.ssm.dtype is not torch.float64:
            raise ValueError(
                f"conv is float64, so ssm must be float64 rather than "
                f"{self.ssm.dtype}; a narrower state downcasts the recurrence"
            )
        channels = int(self.conv.shape[2])
        heads, rows = (int(d) for d in self.ssm.shape[1:3])
        if channels != heads * rows:
            raise ValueError(
                f"conv holds {channels} channels and ssm holds {heads} heads of "
                f"{rows} rows; both are d_inner"
            )
        batch, ssm_batch = int(self.conv.shape[0]), int(self.ssm.shape[0])
        if batch != ssm_batch:
            raise ValueError(
                f"conv has batch {batch} and ssm has batch {ssm_batch}; one batch only"
            )
        if self.conv.device != self.ssm.device:
            raise ValueError(
                f"conv is on {self.conv.device} and ssm is on {self.ssm.device}; "
                f"one device only"
            )

    @classmethod
    def allocate(
        cls,
        config: SLinOSSConfig,
        batch: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> MixerState:
        """Allocate zeroed buffers for one layer.

        Args:
            config: Shape contract. ``d_conv`` and ``d_inner`` size ``conv``;
                ``n_heads``, ``d_head``, and ``d_state`` size ``ssm``.
            batch: Batch ``B``, fixed for the lifetime of the buffers.
            device: Where both buffers live.
            dtype: Activation dtype, carried by ``conv``. ``ssm`` is float32,
                or float64 when ``dtype`` is float64.

        Returns:
            The state.

        Raises:
            ValueError: If ``batch`` is not positive.
            TypeError: If ``dtype`` is not supported.
        """
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        return cls(
            conv=torch.zeros(
                batch, config.d_conv - 1, config.d_inner, dtype=dtype, device=device
            ),
            ssm=torch.zeros(
                batch,
                config.n_heads,
                config.d_head,
                config.d_state,
                dtype=_state_dtype(dtype),
                device=device,
            ),
        )

    def reset(self) -> None:
        """Zero both buffers in place.

        In place because a captured graph holds these addresses; a fresh
        allocation is not the buffer the graph writes.
        """
        self.conv.zero_()
        self.ssm.zero_()

    def clone(self) -> MixerState:
        """Copy both buffers.

        Returns:
            A state sharing no storage with this one.
        """
        return MixerState(conv=self.conv.clone(), ssm=self.ssm.clone())

    @property
    def batch(self) -> int:
        """Batch ``B``."""
        return int(self.ssm.shape[0])

    @property
    def device(self) -> torch.device:
        """Device both buffers live on."""
        return self.ssm.device


@dataclass(frozen=True)
class StackState:
    """Decode state of a stack.

    Attributes:
        layers: One :class:`MixerState` per layer, in stack order. All share a
            batch and a device.
    """

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
        """Allocate zeroed buffers for every layer.

        Args:
            config: Shape contract. ``n_layers`` fixes the entry count.
            batch: Batch ``B``, fixed for the lifetime of the buffers.
            device: Where every buffer lives.
            dtype: Activation dtype. See :meth:`MixerState.allocate`.

        Returns:
            The state, with ``config.n_layers`` entries.

        Raises:
            ValueError: If ``batch`` is not positive.
            TypeError: If ``dtype`` is not supported.
        """
        # One allocation per layer. A repeated entry would alias one buffer
        # across the stack, so every layer would read its neighbour's state.
        return cls(
            layers=tuple(
                MixerState.allocate(config, batch, device=device, dtype=dtype)
                for _ in range(config.n_layers)
            )
        )

    def reset(self) -> None:
        """Zero every layer in place. See :meth:`MixerState.reset`."""
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
