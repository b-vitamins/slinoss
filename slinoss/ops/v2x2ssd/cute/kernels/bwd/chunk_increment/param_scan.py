"""CuTe backward parameter-scan kernel for the ``v2x2ssd`` chunk-increment stage.

``ChunkIncrementBwdParamScanAmpere`` owns the chunk-local metadata and tap scan
that turns:

- raw packed-complex transitions ``M``
- raw packed-complex taps ``Kprev`` / ``Kcurr``
- ``dMsumPart`` partial reductions from the ``db`` workhorse
- ``DMp0`` boundary carry from the boundary workhorse
- ``dMchunk`` chunk-end reverse-scan carry from the public gradient

into the direct stage-native outputs ``dM``, ``dKprev``, and ``dKcurr``.

Tensor contracts:

- ``M``: ``(2, L, BHC)`` fp32 packed-complex transitions
- ``Kprev`` / ``Kcurr``: ``(2, L, BHC)`` fp32 packed-complex taps
- ``DMsumPart``: ``(2, L, n_d_tiles, BHC)`` fp32 per-``D``-tile reductions
- ``DMp0``: ``(2, BHC)`` fp32 boundary entry gradient for the previous tap path
- ``DMchunk``: ``(2, BHC)`` fp32 chunk-end reverse-scan carry
- ``DM`` / ``DKprev`` / ``DKcurr``: ``(2, L, BHC)`` fp32 outputs
"""

from dataclasses import dataclass
from typing import ClassVar

from cuda.bindings import driver as cuda
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


@dataclass(frozen=True)
class ChunkIncrementBwdParamScanLayoutBundle:
    strict_suffix_state_layout: object


@dataclass(frozen=True)
class ChunkIncrementBwdParamScanKernelBundle:
    layouts: ChunkIncrementBwdParamScanLayoutBundle
    shared_storage_cls: object
    smem_bytes: int


@dataclass(frozen=True)
class ChunkIncrementBwdParamScanSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkIncrementBwdParamScanAmpere:
    """Ampere one-warp backward parameter-scan kernel for the chunk-increment stage.

    One thread owns one ``BHC`` lane. The kernel first materializes the strict
    suffix transition product ``suffix_{>t}`` for every time step into shared
    memory, then walks the chunk in forward time to recover:

    - ``dKprev`` from the boundary-seeded previous-tap path
    - ``dKcurr`` from the per-``D``-tile reduced current-tap path
    - ``dM`` from the reverse suffix-state carry entering each transition
    """

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkIncrementBwdParamScanSupportInfo]
    ] = {}

    def __init__(
        self,
        *,
        chunk_size: int,
        n_d_tiles: int,
        num_threads: int = 32,
    ):
        self.L = int(chunk_size)
        self.n_d_tiles = int(n_d_tiles)
        self.num_threads = int(num_threads)

        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.n_d_tiles <= 0:
            raise ValueError("n_d_tiles must be positive.")
        if self.num_threads != 32:
            raise ValueError("This kernel assumes one warp per CTA.")

    def _strict_suffix_state_smem_layout(self):
        return cute.make_layout(
            (self.L, 2, self.num_threads),
            stride=(2 * self.num_threads, self.num_threads, 1),
        )

    def _smem_capacity_bytes(self, device_index: int | None = None) -> int:
        if torch.cuda.is_available():
            if device_index is None:
                device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(int(device_index))
            capacity = int(getattr(props, "shared_memory_per_block_optin", 0))
            if capacity > 0:
                return capacity
            cc = f"sm_{props.major}{props.minor}"
            return int(utils.get_smem_capacity_in_bytes(cc))
        return int(utils.get_smem_capacity_in_bytes("sm_80"))

    @staticmethod
    def _align_up(offset: int, align: int) -> int:
        return ((offset + align - 1) // align) * align

    @classmethod
    def _struct_size_bytes(cls, fields: list[tuple[int, int]]) -> int:
        offset = 0
        max_align = 1
        for size, align in fields:
            offset = cls._align_up(offset, align)
            offset += size
            max_align = max(max_align, align)
        return cls._align_up(offset, max_align)

    def _required_smem_bytes(self) -> int:
        return self._struct_size_bytes([(self.L * 2 * self.num_threads * 4, 4)])

    def support_info(
        self,
        *,
        device_index: int | None = None,
    ) -> ChunkIncrementBwdParamScanSupportInfo:
        if device_index is None:
            device_key = (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
            )
        else:
            device_key = int(device_index)
        cache_key = (
            type(self),
            self.L,
            self.n_d_tiles,
            self.num_threads,
            device_key,
        )
        cached = self._SUPPORT_INFO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        info = ChunkIncrementBwdParamScanSupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            required_smem_bytes=self._required_smem_bytes(),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def can_implement(self, *, device_index: int | None = None) -> bool:
        return self.support_info(device_index=device_index).supported

    def _make_layout_bundle(self) -> ChunkIncrementBwdParamScanLayoutBundle:
        return ChunkIncrementBwdParamScanLayoutBundle(
            strict_suffix_state_layout=self._strict_suffix_state_smem_layout(),
        )

    def _make_shared_storage(
        self,
        layouts: ChunkIncrementBwdParamScanLayoutBundle,
    ):
        @cute.struct
        class SharedStorage:
            strict_suffix_state: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.strict_suffix_state_layout)
                ],
                4,
            ]

        return SharedStorage

    def _make_kernel_bundle(self) -> ChunkIncrementBwdParamScanKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage_cls = self._make_shared_storage(layouts)
        return ChunkIncrementBwdParamScanKernelBundle(
            layouts=layouts,
            shared_storage_cls=shared_storage_cls,
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
        )

    @cute.jit
    def _validate_operands(
        self,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDMsumPart: cute.Tensor,
        mDMp0: cute.Tensor,
        mDMchunk: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
    ):
        if cutlass.const_expr(
            mM.element_type != cutlass.Float32
            or mKprev.element_type != cutlass.Float32
            or mKcurr.element_type != cutlass.Float32
            or mDMsumPart.element_type != cutlass.Float32
            or mDMp0.element_type != cutlass.Float32
            or mDMchunk.element_type != cutlass.Float32
            or mDM.element_type != cutlass.Float32
            or mDKprev.element_type != cutlass.Float32
            or mDKcurr.element_type != cutlass.Float32
        ):
            raise TypeError("param_scan operands and outputs must all be Float32.")
        if cutlass.const_expr(mM.shape[0] != 2 or mM.shape[1] != self.L):
            raise ValueError("M must be (2, L, BHC).")
        if cutlass.const_expr(mKprev.shape[0] != 2 or mKprev.shape[1] != self.L):
            raise ValueError("Kprev must be (2, L, BHC).")
        if cutlass.const_expr(mKcurr.shape[0] != 2 or mKcurr.shape[1] != self.L):
            raise ValueError("Kcurr must be (2, L, BHC).")
        if cutlass.const_expr(
            mDMsumPart.shape[0] != 2
            or mDMsumPart.shape[1] != self.L
            or mDMsumPart.shape[2] != self.n_d_tiles
        ):
            raise ValueError("DMsumPart must be (2, L, n_d_tiles, BHC).")
        if cutlass.const_expr(mDMp0.shape[0] != 2):
            raise ValueError("DMp0 must be (2, BHC).")
        if cutlass.const_expr(mDMchunk.shape[0] != 2):
            raise ValueError("DMchunk must be (2, BHC).")
        if cutlass.const_expr(mDM.shape[0] != 2 or mDM.shape[1] != self.L):
            raise ValueError("DM output must be (2, L, BHC).")
        if cutlass.const_expr(mDKprev.shape[0] != 2 or mDKprev.shape[1] != self.L):
            raise ValueError("DKprev output must be (2, L, BHC).")
        if cutlass.const_expr(mDKcurr.shape[0] != 2 or mDKcurr.shape[1] != self.L):
            raise ValueError("DKcurr output must be (2, L, BHC).")

    def _launch_kernel(
        self,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDMsumPart: cute.Tensor,
        mDMp0: cute.Tensor,
        mDMchunk: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle()
        batch_head_chunk_count = cute.size(mM.shape[2])
        grid_x = cute.ceil_div(batch_head_chunk_count, self.num_threads)
        launch_kwargs = {
            "grid": [cute.size(grid_x), 1, 1],
            "block": [self.num_threads, 1, 1],
            "smem": bundle.smem_bytes,
        }
        if stream is not None:
            launch_kwargs["stream"] = stream

        self.kernel(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
            bundle.layouts.strict_suffix_state_layout,
            bundle.shared_storage_cls,
        ).launch(**launch_kwargs)

    @cute.jit
    def _validate_and_launch(
        self,
        mM: cute.Tensor,  # (2, L, BHC) fp32
        mKprev: cute.Tensor,  # (2, L, BHC) fp32
        mKcurr: cute.Tensor,  # (2, L, BHC) fp32
        mDMsumPart: cute.Tensor,  # (2, L, n_d_tiles, BHC) fp32
        mDMp0: cute.Tensor,  # (2, BHC) fp32
        mDMchunk: cute.Tensor,  # (2, BHC) fp32
        mDM: cute.Tensor,  # (2, L, BHC) fp32
        mDKprev: cute.Tensor,  # (2, L, BHC) fp32
        mDKcurr: cute.Tensor,  # (2, L, BHC) fp32
        stream: cuda.CUstream | None = None,
    ):
        self._validate_operands(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
        )
        self._launch_kernel(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
            stream=stream,
        )

    @cute.jit
    def __call__(
        self,
        mM: cute.Tensor,  # (2, L, BHC) fp32
        mKprev: cute.Tensor,  # (2, L, BHC) fp32
        mKcurr: cute.Tensor,  # (2, L, BHC) fp32
        mDMsumPart: cute.Tensor,  # (2, L, n_d_tiles, BHC) fp32
        mDMp0: cute.Tensor,  # (2, BHC) fp32
        mDMchunk: cute.Tensor,  # (2, BHC) fp32
        mDM: cute.Tensor,  # (2, L, BHC) fp32
        mDKprev: cute.Tensor,  # (2, L, BHC) fp32
        mDKcurr: cute.Tensor,  # (2, L, BHC) fp32
    ):
        self._validate_and_launch(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
        )

    @cute.jit
    def call_on_stream(
        self,
        mM: cute.Tensor,  # (2, L, BHC) fp32
        mKprev: cute.Tensor,  # (2, L, BHC) fp32
        mKcurr: cute.Tensor,  # (2, L, BHC) fp32
        mDMsumPart: cute.Tensor,  # (2, L, n_d_tiles, BHC) fp32
        mDMp0: cute.Tensor,  # (2, BHC) fp32
        mDMchunk: cute.Tensor,  # (2, BHC) fp32
        mDM: cute.Tensor,  # (2, L, BHC) fp32
        mDKprev: cute.Tensor,  # (2, L, BHC) fp32
        mDKcurr: cute.Tensor,  # (2, L, BHC) fp32
        stream: cuda.CUstream,
    ):
        self._validate_and_launch(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDMsumPart: cute.Tensor,
        mDMp0: cute.Tensor,
        mDMchunk: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        strict_suffix_state_layout: cute.Layout,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()

        batch_head_chunk_count = cute.size(mM.shape[2])
        batch_head_chunk = block_x * self.num_threads + tidx
        batch_head_chunk_valid = cute.elem_less(
            batch_head_chunk, batch_head_chunk_count
        )
        batch_head_chunk_safe = cutlass.min(
            batch_head_chunk, batch_head_chunk_count - cutlass.Int32(1)
        )
        lane_idx = tidx

        # Shared storage holds the strict suffix product after each time step for
        # one warp's worth of ``BHC`` lanes.
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_strict_suffix_after_step = storage.strict_suffix_state.get_tensor(
            strict_suffix_state_layout
        )

        # Reverse-time prepass: materialize suffix_{>t} for every step. Invalid
        # lanes still write the identity so later loads never observe garbage.
        suffix_after_re = cutlass.Float32(1.0)
        suffix_after_im = cutlass.Float32(0.0)
        for reverse_time_index in cutlass.range(self.L, unroll=1):
            time_step = (self.L - 1) - reverse_time_index
            s_strict_suffix_after_step[time_step, 0, lane_idx] = cutlass.select_(
                batch_head_chunk_valid, suffix_after_re, cutlass.Float32(1.0)
            )
            s_strict_suffix_after_step[time_step, 1, lane_idx] = cutlass.select_(
                batch_head_chunk_valid, suffix_after_im, cutlass.Float32(0.0)
            )

            transition_re = cutlass.Float32(mM[0, time_step, batch_head_chunk_safe])
            transition_im = cutlass.Float32(mM[1, time_step, batch_head_chunk_safe])
            next_re = suffix_after_re * transition_re - suffix_after_im * transition_im
            next_im = suffix_after_re * transition_im + suffix_after_im * transition_re
            suffix_after_re = next_re
            suffix_after_im = next_im

        # Forward-time parameter scan-backward over the cached strict suffix
        # products. ``DMp0`` seeds the previous-tap path at step 0 and
        # ``DMchunk`` seeds the reverse metadata carry entering the chunk tail.
        prev_tap_grad_re = cutlass.Float32(mDMp0[0, batch_head_chunk_safe])
        prev_tap_grad_im = cutlass.Float32(mDMp0[1, batch_head_chunk_safe])
        suffix_carry_re = cutlass.Float32(mDMchunk[0, batch_head_chunk_safe])
        suffix_carry_im = cutlass.Float32(mDMchunk[1, batch_head_chunk_safe])

        for time_step in cutlass.range(self.L, unroll=1):
            suffix_after_re = cutlass.Float32(
                s_strict_suffix_after_step[time_step, 0, lane_idx]
            )
            suffix_after_im = cutlass.Float32(
                s_strict_suffix_after_step[time_step, 1, lane_idx]
            )

            curr_tap_grad_re = cutlass.Float32(0.0)
            curr_tap_grad_im = cutlass.Float32(0.0)
            for d_tile in cutlass.range(self.n_d_tiles, unroll=1):
                curr_tap_grad_re = curr_tap_grad_re + cutlass.Float32(
                    mDMsumPart[0, time_step, d_tile, batch_head_chunk_safe]
                )
                curr_tap_grad_im = curr_tap_grad_im + cutlass.Float32(
                    mDMsumPart[1, time_step, d_tile, batch_head_chunk_safe]
                )

            d_kprev_re = (
                suffix_after_re * prev_tap_grad_re + suffix_after_im * prev_tap_grad_im
            )
            d_kprev_im = (
                suffix_after_re * prev_tap_grad_im - suffix_after_im * prev_tap_grad_re
            )
            d_kcurr_re = (
                suffix_after_re * curr_tap_grad_re + suffix_after_im * curr_tap_grad_im
            )
            d_kcurr_im = (
                suffix_after_re * curr_tap_grad_im - suffix_after_im * curr_tap_grad_re
            )

            prev_tap_re = cutlass.Float32(mKprev[0, time_step, batch_head_chunk_safe])
            prev_tap_im = cutlass.Float32(mKprev[1, time_step, batch_head_chunk_safe])
            curr_tap_re = cutlass.Float32(mKcurr[0, time_step, batch_head_chunk_safe])
            curr_tap_im = cutlass.Float32(mKcurr[1, time_step, batch_head_chunk_safe])

            d_suffix_re = (
                prev_tap_re * prev_tap_grad_re + prev_tap_im * prev_tap_grad_im
            ) + (curr_tap_re * curr_tap_grad_re + curr_tap_im * curr_tap_grad_im)
            d_suffix_im = (
                prev_tap_re * prev_tap_grad_im - prev_tap_im * prev_tap_grad_re
            ) + (curr_tap_re * curr_tap_grad_im - curr_tap_im * curr_tap_grad_re)

            d_transition_re = (
                suffix_carry_re * suffix_after_re + suffix_carry_im * suffix_after_im
            )
            d_transition_im = (
                suffix_carry_im * suffix_after_re - suffix_carry_re * suffix_after_im
            )

            if batch_head_chunk_valid:
                mDKprev[0, time_step, batch_head_chunk_safe] = d_kprev_re
                mDKprev[1, time_step, batch_head_chunk_safe] = d_kprev_im
                mDKcurr[0, time_step, batch_head_chunk_safe] = d_kcurr_re
                mDKcurr[1, time_step, batch_head_chunk_safe] = d_kcurr_im
                mDM[0, time_step, batch_head_chunk_safe] = d_transition_re
                mDM[1, time_step, batch_head_chunk_safe] = d_transition_im

            if time_step + 1 < self.L:
                transition_re = cutlass.Float32(mM[0, time_step, batch_head_chunk_safe])
                transition_im = cutlass.Float32(mM[1, time_step, batch_head_chunk_safe])
                next_carry_re = (
                    suffix_carry_re * transition_re + suffix_carry_im * transition_im
                )
                next_carry_im = (
                    suffix_carry_im * transition_re - suffix_carry_re * transition_im
                )
                suffix_carry_re = next_carry_re + d_suffix_re
                suffix_carry_im = next_carry_im + d_suffix_im

            prev_tap_grad_re = curr_tap_grad_re
            prev_tap_grad_im = curr_tap_grad_im


__all__ = ["ChunkIncrementBwdParamScanAmpere"]
