"""CuTe backward boundary kernel for the ``v2x2ssd`` chunk-increment stage.

``ChunkIncrementBwdBoundaryAmpere`` computes the per-chunk boundary gradients
for the rank-1 boundary correction:

``d_inc += U_prev outer (Mp0 * B_prev)``

Tensor contracts:

- ``DIncBoundary``: ``(BHC, P, D)`` fp16/bf16 boundary slice of ``dInc``
- ``BPrev``: ``(D, BHC)`` fp16/bf16 boundary key rows
- ``UPrev``: ``(P, BHC)`` fp16/bf16 boundary value rows
- ``M``: ``(2, L, BHC)`` fp32 packed-complex transitions
- ``Kprev``: ``(2, L, BHC)`` fp32 packed-complex previous-pass taps
- ``DUPrev``: ``(P, BHC)`` fp16/bf16 output boundary-value gradients
- ``DBPrev``: ``(D, BHC)`` fp16/bf16 output boundary-key gradients
- ``DMp0``: ``(2, BHC)`` fp32 packed-complex output chunk-boundary summaries

The trailing ``D`` dimension stores packed complex pairs, so ``D`` must be
even and conceptually corresponds to ``2 * N``.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


@dataclass(frozen=True)
class ChunkIncrementBwdBoundaryKernelBundle:
    async_copy_atom: object
    smem_bytes: int


@dataclass(frozen=True)
class ChunkIncrementBwdBoundarySupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkIncrementBwdBoundaryAmpere:
    """Ampere backward boundary kernel for the ``v2x2ssd`` chunk-increment stage."""

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkIncrementBwdBoundarySupportInfo]
    ] = {}

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        D: int,
        P: int,
        heads: int | None = None,
        bc_groups: int | None = None,
        n_chunks: int | None = None,
        num_threads: int = 192,
    ):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.mp_dtype = cutlass.Float32

        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        self.num_threads = int(num_threads)
        self.has_group_geometry = heads is not None
        if self.has_group_geometry:
            self.heads = int(heads)
            self.bc_groups = self.heads if bc_groups is None else int(bc_groups)
            if n_chunks is None:
                raise ValueError(
                    "n_chunks must be specified when grouped BC is active."
                )
            self.n_chunks = int(n_chunks)
            if self.heads <= 0 or self.bc_groups <= 0:
                raise ValueError("heads and bc_groups must be positive.")
            if self.heads % self.bc_groups != 0:
                raise ValueError("bc_groups must divide heads.")
            if self.n_chunks <= 0:
                raise ValueError("n_chunks must be positive.")
            self.heads_per_bc_group = self.heads // self.bc_groups
        else:
            if bc_groups is not None:
                raise ValueError("bc_groups requires heads to be specified.")
            self.heads = 0
            self.bc_groups = 0
            self.n_chunks = 0
            self.heads_per_bc_group = 0

        self.du_prev_threads = 64
        self.db_prev_threads = self.num_threads - self.du_prev_threads
        self.async_copy_bits = 128
        self.async_copy_elems = self.async_copy_bits // self.ab_dtype.width

        self.scan_threads = self._resolve_scan_threads(self.L)
        self.scan_warp_count = self.scan_threads // 32
        self.db_prev_warp_count = self.db_prev_threads // 32

        if self.D % 2 != 0:
            raise ValueError("D must be divisible by 2 (flattened 2N).")
        if self.num_threads <= self.du_prev_threads:
            raise ValueError("num_threads must be > 64.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.db_prev_threads <= 0:
            raise ValueError("num_threads must leave room for the dB_prev workers.")
        if self.db_prev_threads % 32 != 0:
            raise ValueError("dB_prev worker count must be a multiple of 32.")
        if self.scan_threads > self.num_threads:
            raise ValueError("num_threads must cover the configured suffix scan.")
        if self.async_copy_bits % self.ab_dtype.width != 0:
            raise ValueError("async_copy_bits must be divisible by the input width.")

        self.fast_tiled = (
            self.P <= self.du_prev_threads
            and self.D >= 128
            and self.D % self.async_copy_elems == 0
        )
        self.d_stage = 128 if self.fast_tiled else self.D
        if self.fast_tiled:
            self.d_smem_stride = self.d_stage + self.async_copy_elems
        else:
            self.d_smem_stride = (
                self._align_up(self.D, self.async_copy_elems) + self.async_copy_elems
            )

    @cute.jit
    def _batch_group_chunk_index(self, batch_head_chunk_idx: int):
        if cutlass.const_expr(not self.has_group_geometry):
            return batch_head_chunk_idx
        batch_head = batch_head_chunk_idx // cutlass.Int32(self.n_chunks)
        chunk_index = batch_head_chunk_idx - batch_head * cutlass.Int32(self.n_chunks)
        batch_idx = batch_head // cutlass.Int32(self.heads)
        head_idx = batch_head - batch_idx * cutlass.Int32(self.heads)
        group_idx = head_idx // cutlass.Int32(self.heads_per_bc_group)
        batch_group = batch_idx * cutlass.Int32(self.bc_groups) + group_idx
        return batch_group * cutlass.Int32(self.n_chunks) + chunk_index

    @staticmethod
    def _resolve_scan_threads(chunk_size: int) -> int:
        if chunk_size <= 64:
            return 64
        if chunk_size <= 128:
            return 128
        return 0

    def _mp0_layout(self):
        return cute.make_layout((2,), stride=(1,))

    def _scan_warp_transition_layout(self):
        return cute.make_layout((max(1, self.scan_warp_count), 2), stride=(2, 1))

    def _db_prev_partial_sum_layout(self):
        return cute.make_layout((self.db_prev_warp_count, 2), stride=(2, 1))

    def _boundary_value_smem_layout(self):
        return cute.make_layout((self.P,), stride=(1,))

    def _decayed_boundary_key_smem_layout(self):
        return cute.make_layout((self.d_stage,), stride=(1,))

    def _staged_dinc_smem_layout(self):
        return cute.make_layout(
            (self.P, self.d_smem_stride), stride=(self.d_smem_stride, 1)
        )

    def _staged_dinc_vector_smem_layout(self):
        return cute.make_layout(
            (
                self.P,
                self.d_smem_stride // self.async_copy_elems,
                self.async_copy_elems,
            ),
            stride=(self.d_smem_stride, self.async_copy_elems, 1),
        )

    def _make_dinc_vector_view(self, mDInc: cute.Tensor) -> cute.Tensor:
        return cute.make_tensor(
            mDInc.iterator,
            cute.make_layout(
                (
                    mDInc.shape[0],
                    mDInc.shape[1],
                    self.D // self.async_copy_elems,
                    self.async_copy_elems,
                ),
                stride=(self.P * self.D, self.D, self.async_copy_elems, 1),
            ),
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

    def _required_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        in_bytes = in_dtype.width // 8
        fields = [
            (2 * 4, 8),
            (max(1, self.scan_warp_count) * 2 * 4, 8),
            (self.db_prev_warp_count * 2 * 4, 8),
            (self.P * 4, 4),
            (self.d_stage * 4, 4),
            (self.P * self.d_smem_stride * in_bytes, 16),
        ]
        return self._struct_size_bytes(fields)

    def support_info(
        self,
        *,
        device_index: int | None = None,
    ) -> ChunkIncrementBwdBoundarySupportInfo:
        device_key = (
            int(torch.cuda.current_device())
            if device_index is None and torch.cuda.is_available()
            else (-1 if device_index is None else int(device_index))
        )
        cache_key = (
            type(self),
            self.ab_dtype,
            self.L,
            self.D,
            self.P,
            self.num_threads,
            device_key,
        )
        cached = self._SUPPORT_INFO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        info = ChunkIncrementBwdBoundarySupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            required_smem_bytes=self._required_smem_bytes(self.ab_dtype),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def can_implement(self, *, device_index: int | None = None) -> bool:
        return self.support_info(device_index=device_index).supported

    def _make_async_dinc_copy_atom(self, dinc_dtype: type[cutlass.Numeric]):
        return cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            dinc_dtype,
            num_bits_per_copy=self.async_copy_bits,
        )

    def _make_shared_storage(self, dinc_dtype: type[cutlass.Numeric]):
        mp0_layout = self._mp0_layout()
        scan_warp_transition_layout = self._scan_warp_transition_layout()
        db_prev_partial_sum_layout = self._db_prev_partial_sum_layout()
        boundary_value_layout = self._boundary_value_smem_layout()
        decayed_boundary_key_layout = self._decayed_boundary_key_smem_layout()
        staged_dinc_layout = self._staged_dinc_smem_layout()

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mp0": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(mp0_layout)],
                8,
            ],
            "scan_warp_transition": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(scan_warp_transition_layout)
                ],
                8,
            ],
            "db_prev_partial_sum": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(db_prev_partial_sum_layout)
                ],
                8,
            ],
            "boundary_value": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(boundary_value_layout)
                ],
                4,
            ],
            "decayed_boundary_key": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(decayed_boundary_key_layout)
                ],
                4,
            ],
            "staged_dinc": cute.struct.Align[
                cute.struct.MemRange[dinc_dtype, cute.cosize(staged_dinc_layout)],
                16,
            ],
        }
        return cute.struct(SharedStorage)

    def _make_kernel_bundle(
        self,
        mDInc: cute.Tensor,
    ) -> ChunkIncrementBwdBoundaryKernelBundle:
        shared_storage_cls = self._make_shared_storage(mDInc.element_type)
        return ChunkIncrementBwdBoundaryKernelBundle(
            async_copy_atom=self._make_async_dinc_copy_atom(mDInc.element_type),
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
        )

    @cute.jit
    def _validate_operands(
        self,
        mDInc: cute.Tensor,
        mBPrev: cute.Tensor,
        mUPrev: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mDUPrev: cute.Tensor,
        mDBPrev: cute.Tensor,
        mDMp0: cute.Tensor,
    ):
        value_stream_dtype_ok = (
            mDInc.element_type
            == mBPrev.element_type
            == mUPrev.element_type
            == mDUPrev.element_type
            == mDBPrev.element_type
        )
        if cutlass.const_expr(not value_stream_dtype_ok):
            raise TypeError(
                "Boundary value streams and outputs must share element type."
            )
        if cutlass.const_expr(
            mDInc.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("Boundary value streams must be Float16 or BFloat16.")
        if cutlass.const_expr(
            not (
                mM.element_type
                == mKprev.element_type
                == mDMp0.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("M/Kprev/DMp0 must be Float32.")
        if cutlass.const_expr(mDInc.shape[1] != mUPrev.shape[0]):
            raise ValueError("DIncBoundary and UPrev must agree on the P dimension.")
        if cutlass.const_expr(mDInc.shape[2] != mBPrev.shape[0]):
            raise ValueError("DIncBoundary and BPrev must agree on the D dimension.")
        if cutlass.const_expr(mDUPrev.shape[0] != mUPrev.shape[0]):
            raise ValueError("DUPrev must match UPrev.")
        if cutlass.const_expr(mDBPrev.shape[0] != mBPrev.shape[0]):
            raise ValueError("DBPrev must match BPrev.")
        if cutlass.const_expr(mM.shape[0] != 2 or mKprev.shape[0] != 2):
            raise ValueError("M and Kprev must have leading packed-complex extent 2.")
        if cutlass.const_expr(mDMp0.shape[0] != 2):
            raise ValueError("DMp0 must have leading packed-complex extent 2.")
        if cutlass.const_expr(mM.shape[1] != mKprev.shape[1]):
            raise ValueError("M and Kprev must share the chunk time dimension.")
        if cutlass.const_expr(mBPrev.shape[0] % 2 != 0):
            raise ValueError("BPrev D dimension must be even because D stores pairs.")

    def _launch_kernel(
        self,
        mDInc: cute.Tensor,
        mBPrev: cute.Tensor,
        mUPrev: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mDUPrev: cute.Tensor,
        mDBPrev: cute.Tensor,
        mDMp0: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle(mDInc)
        launch_kwargs = {
            "grid": (cute.size(mDInc.shape[0]), 1, 1),
            "block": [self.num_threads, 1, 1],
            "smem": bundle.smem_bytes,
        }
        if stream is not None:
            launch_kwargs["stream"] = stream

        self.kernel(
            mDInc,
            mBPrev,
            mUPrev,
            mM,
            mKprev,
            mDUPrev,
            mDBPrev,
            mDMp0,
            bundle.async_copy_atom,
        ).launch(**launch_kwargs)

    @cute.jit
    def _validate_and_launch(
        self,
        mDInc: cute.Tensor,
        mBPrev: cute.Tensor,
        mUPrev: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mDUPrev: cute.Tensor,
        mDBPrev: cute.Tensor,
        mDMp0: cute.Tensor,
        stream=None,
    ):
        self._validate_operands(
            mDInc,
            mBPrev,
            mUPrev,
            mM,
            mKprev,
            mDUPrev,
            mDBPrev,
            mDMp0,
        )
        self._launch_kernel(
            mDInc,
            mBPrev,
            mUPrev,
            mM,
            mKprev,
            mDUPrev,
            mDBPrev,
            mDMp0,
            stream=stream,
        )

    @cute.jit
    def __call__(
        self,
        mDInc: cute.Tensor,  # (BHC, P, D) fp16/bf16
        mBPrev: cute.Tensor,  # (D, BHC) fp16/bf16
        mUPrev: cute.Tensor,  # (P, BHC) fp16/bf16
        mM: cute.Tensor,  # (2, L, BHC) fp32
        mKprev: cute.Tensor,  # (2, L, BHC) fp32
        mDUPrev: cute.Tensor,  # (P, BHC) fp16/bf16
        mDBPrev: cute.Tensor,  # (D, BHC) fp16/bf16
        mDMp0: cute.Tensor,  # (2, BHC) fp32
    ):
        self._validate_and_launch(
            mDInc,
            mBPrev,
            mUPrev,
            mM,
            mKprev,
            mDUPrev,
            mDBPrev,
            mDMp0,
        )

    @cute.jit
    def call_on_stream(
        self,
        mDInc: cute.Tensor,  # (BHC, P, D) fp16/bf16
        mBPrev: cute.Tensor,  # (D, BHC) fp16/bf16
        mUPrev: cute.Tensor,  # (P, BHC) fp16/bf16
        mM: cute.Tensor,  # (2, L, BHC) fp32
        mKprev: cute.Tensor,  # (2, L, BHC) fp32
        mDUPrev: cute.Tensor,  # (P, BHC) fp16/bf16
        mDBPrev: cute.Tensor,  # (D, BHC) fp16/bf16
        mDMp0: cute.Tensor,  # (2, BHC) fp32
        stream: cuda.CUstream,
    ):
        self._validate_and_launch(
            mDInc,
            mBPrev,
            mUPrev,
            mM,
            mKprev,
            mDUPrev,
            mDBPrev,
            mDMp0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDInc: cute.Tensor,  # (BHC, P, D)
        mBPrev: cute.Tensor,  # (D, BHC)
        mUPrev: cute.Tensor,  # (P, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mDUPrev: cute.Tensor,  # (P, BHC)
        mDBPrev: cute.Tensor,  # (D, BHC)
        mDMp0: cute.Tensor,  # (2, BHC)
        async_copy_atom: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        batch_head_chunk, _, _ = cute.arch.block_idx()
        batch_group_chunk = self._batch_group_chunk_index(batch_head_chunk)
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        # Shared-memory setup.
        shared_storage_cls = self._make_shared_storage(mDInc.element_type)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_mp0 = storage.mp0.get_tensor(self._mp0_layout())
        s_scan_warp_transition = storage.scan_warp_transition.get_tensor(
            self._scan_warp_transition_layout()
        )
        s_db_prev_partial_sum = storage.db_prev_partial_sum.get_tensor(
            self._db_prev_partial_sum_layout()
        )
        s_boundary_value = storage.boundary_value.get_tensor(
            self._boundary_value_smem_layout()
        )
        s_decayed_boundary_key = storage.decayed_boundary_key.get_tensor(
            self._decayed_boundary_key_smem_layout()
        )
        s_staged_dinc = storage.staged_dinc.get_tensor(self._staged_dinc_smem_layout())
        s_staged_dinc_vec = cute.make_tensor(
            s_staged_dinc.iterator, self._staged_dinc_vector_smem_layout()
        )
        g_dinc_vec = self._make_dinc_vector_view(mDInc)

        # Cache the boundary value row in shared memory.
        boundary_row = tidx
        while cute.elem_less(boundary_row, self.P):
            s_boundary_value[boundary_row] = mUPrev[boundary_row, batch_head_chunk].to(
                cutlass.Float32
            )
            boundary_row = boundary_row + self.num_threads

        # Reverse-time suffix scan to recover Mp0 = suffix_after[0] * Kprev[0].
        if self.scan_threads != 0 and cute.elem_less(
            tidx, cutlass.Int32(self.scan_threads)
        ):
            reverse_time_index = cutlass.Int32(self.L - 1) - tidx
            suffix_transition_re = cutlass.Float32(1.0)
            suffix_transition_im = cutlass.Float32(0.0)
            if cute.elem_less(cutlass.Int32(0), reverse_time_index):
                suffix_transition_re = cutlass.Float32(
                    mM[0, reverse_time_index, batch_head_chunk]
                )
                suffix_transition_im = cutlass.Float32(
                    mM[1, reverse_time_index, batch_head_chunk]
                )

            for offset in (1, 2, 4, 8, 16):
                prior_re = cute.arch.shuffle_sync_up(
                    suffix_transition_re, offset=offset, mask=-1, mask_and_clamp=0
                )
                prior_im = cute.arch.shuffle_sync_up(
                    suffix_transition_im, offset=offset, mask=-1, mask_and_clamp=0
                )
                should_update = lane_idx >= cutlass.Int32(offset)
                next_re = (
                    prior_re * suffix_transition_re - prior_im * suffix_transition_im
                )
                next_im = (
                    prior_re * suffix_transition_im + prior_im * suffix_transition_re
                )
                suffix_transition_re = cutlass.select_(
                    should_update, next_re, suffix_transition_re
                )
                suffix_transition_im = cutlass.select_(
                    should_update, next_im, suffix_transition_im
                )

            if lane_idx == cutlass.Int32(31) and cute.elem_less(
                warp_idx, cutlass.Int32(self.scan_warp_count)
            ):
                s_scan_warp_transition[warp_idx, 0] = suffix_transition_re
                s_scan_warp_transition[warp_idx, 1] = suffix_transition_im

        if cutlass.const_expr(not self.fast_tiled):
            boundary_row = tidx
            dinc_vector_count = self.D // self.async_copy_elems
            while cute.elem_less(boundary_row, self.P):
                dinc_vector = cutlass.Int32(0)
                while cute.elem_less(dinc_vector, dinc_vector_count):
                    cute.copy(
                        async_copy_atom,
                        g_dinc_vec[batch_head_chunk, boundary_row, dinc_vector, None],
                        s_staged_dinc_vec[boundary_row, dinc_vector, None],
                    )
                    dinc_vector = dinc_vector + 1
                boundary_col = dinc_vector_count * self.async_copy_elems
                while cute.elem_less(boundary_col, self.D):
                    s_staged_dinc[boundary_row, boundary_col] = mDInc[
                        batch_head_chunk, boundary_row, boundary_col
                    ]
                    boundary_col = boundary_col + 1
                boundary_row = boundary_row + self.num_threads
            cute.arch.cp_async_commit_group()

        cute.arch.sync_threads()

        if tidx == 0:
            suffix_after_step0_re = cutlass.Float32(1.0)
            suffix_after_step0_im = cutlass.Float32(0.0)

            if self.scan_threads != 0:
                for scan_warp in cutlass.range(self.scan_warp_count, unroll=1):
                    warp_re = s_scan_warp_transition[scan_warp, 0]
                    warp_im = s_scan_warp_transition[scan_warp, 1]
                    next_re = (
                        suffix_after_step0_re * warp_re
                        - suffix_after_step0_im * warp_im
                    )
                    next_im = (
                        suffix_after_step0_re * warp_im
                        + suffix_after_step0_im * warp_re
                    )
                    suffix_after_step0_re = next_re
                    suffix_after_step0_im = next_im
            else:
                for t_it in cutlass.range(self.L - 1, unroll=1):
                    reverse_time_index = cutlass.Int32(self.L - 1) - t_it
                    transition_re = cutlass.Float32(
                        mM[0, reverse_time_index, batch_head_chunk]
                    )
                    transition_im = cutlass.Float32(
                        mM[1, reverse_time_index, batch_head_chunk]
                    )
                    next_re = (
                        suffix_after_step0_re * transition_re
                        - suffix_after_step0_im * transition_im
                    )
                    next_im = (
                        suffix_after_step0_re * transition_im
                        + suffix_after_step0_im * transition_re
                    )
                    suffix_after_step0_re = next_re
                    suffix_after_step0_im = next_im

            prev_tap_re = cutlass.Float32(mKprev[0, 0, batch_head_chunk])
            prev_tap_im = cutlass.Float32(mKprev[1, 0, batch_head_chunk])
            s_mp0[0] = (
                suffix_after_step0_re * prev_tap_re
                - suffix_after_step0_im * prev_tap_im
            )
            s_mp0[1] = (
                suffix_after_step0_re * prev_tap_im
                + suffix_after_step0_im * prev_tap_re
            )

        cute.arch.sync_threads()

        boundary_coeff_re = s_mp0[0]
        boundary_coeff_im = s_mp0[1]
        du_worker_warp_count = cutlass.Int32(self.du_prev_threads // 32)

        if cutlass.const_expr(self.fast_tiled):
            du_prev_accum = cutlass.Float32(0.0)
            du_boundary_row = tidx
            d_mp0_partial_re = cutlass.Float32(0.0)
            d_mp0_partial_im = cutlass.Float32(0.0)

            stage_tile_count = (self.D + self.d_stage - 1) // self.d_stage
            stage_vector_count = self.d_stage // self.async_copy_elems

            for d_tile in cutlass.range_constexpr(stage_tile_count):
                stage_col_start = cutlass.Int32(d_tile * self.d_stage)
                stage_width = min(self.d_stage, self.D - d_tile * self.d_stage)
                stage_vector_width = stage_width // self.async_copy_elems

                # Stage the current dInc slab into shared memory.
                copy_task = tidx
                total_copy_tasks = cutlass.Int32(self.P * stage_vector_width)
                while cute.elem_less(copy_task, total_copy_tasks):
                    copy_boundary_row = copy_task // cutlass.Int32(stage_vector_width)
                    stage_vector = copy_task - copy_boundary_row * cutlass.Int32(
                        stage_vector_width
                    )
                    cute.copy(
                        async_copy_atom,
                        g_dinc_vec[
                            batch_head_chunk,
                            copy_boundary_row,
                            d_tile * stage_vector_count + stage_vector,
                            None,
                        ],
                        s_staged_dinc_vec[copy_boundary_row, stage_vector, None],
                    )
                    copy_task = copy_task + self.num_threads
                cute.arch.cp_async_commit_group()

                # Rotate the boundary key row by Mp0 for this D tile.
                complex_pair = tidx
                while cute.elem_less(complex_pair, cutlass.Int32(stage_width // 2)):
                    d_pair_start = complex_pair * 2
                    b_prev_re = mBPrev[
                        stage_col_start + d_pair_start + 0, batch_group_chunk
                    ].to(cutlass.Float32)
                    b_prev_im = mBPrev[
                        stage_col_start + d_pair_start + 1, batch_group_chunk
                    ].to(cutlass.Float32)
                    s_decayed_boundary_key[d_pair_start + 0] = (
                        boundary_coeff_re * b_prev_re - boundary_coeff_im * b_prev_im
                    )
                    s_decayed_boundary_key[d_pair_start + 1] = (
                        boundary_coeff_re * b_prev_im + boundary_coeff_im * b_prev_re
                    )
                    complex_pair = complex_pair + self.num_threads

                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                # dU_prev workers: each thread owns at most one boundary row.
                if cute.elem_less(
                    tidx, cutlass.Int32(self.du_prev_threads)
                ) and cute.elem_less(du_boundary_row, self.P):
                    boundary_col = cutlass.Int32(0)
                    while cute.elem_less(boundary_col, cutlass.Int32(stage_width)):
                        du_prev_accum = (
                            du_prev_accum
                            + s_staged_dinc[du_boundary_row, boundary_col].to(
                                cutlass.Float32
                            )
                            * s_decayed_boundary_key[boundary_col]
                        )
                        boundary_col = boundary_col + 1

                # dB_prev workers: two half-warps cooperate on each complex pair.
                if not cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                    db_local = tidx - self.du_prev_threads
                    complex_pair = db_local // 2
                    half_selector = db_local - complex_pair * 2
                    if cute.elem_less(complex_pair, cutlass.Int32(stage_width // 2)):
                        d_pair_start = complex_pair * 2
                        g0 = cutlass.Float32(0.0)
                        g1 = cutlass.Float32(0.0)
                        p_base = half_selector * 32
                        for p_off in cutlass.range(32, unroll=1):
                            boundary_row = cutlass.Int32(p_base + p_off)
                            if cute.elem_less(boundary_row, self.P):
                                boundary_value = s_boundary_value[boundary_row]
                                g0 = g0 + boundary_value * s_staged_dinc[
                                    boundary_row, d_pair_start + 0
                                ].to(cutlass.Float32)
                                g1 = g1 + boundary_value * s_staged_dinc[
                                    boundary_row, d_pair_start + 1
                                ].to(cutlass.Float32)

                        g0 = g0 + cute.arch.shuffle_sync_bfly(
                            g0, offset=1, mask=-1, mask_and_clamp=31
                        )
                        g1 = g1 + cute.arch.shuffle_sync_bfly(
                            g1, offset=1, mask=-1, mask_and_clamp=31
                        )

                        if half_selector == cutlass.Int32(0):
                            global_pair_start = stage_col_start + d_pair_start
                            mDBPrev[global_pair_start + 0, batch_head_chunk] = (
                                boundary_coeff_re * g0 + boundary_coeff_im * g1
                            ).to(mDBPrev.element_type)
                            mDBPrev[global_pair_start + 1, batch_head_chunk] = (
                                boundary_coeff_re * g1 - boundary_coeff_im * g0
                            ).to(mDBPrev.element_type)

                            b_prev_re = mBPrev[
                                global_pair_start + 0, batch_group_chunk
                            ].to(cutlass.Float32)
                            b_prev_im = mBPrev[
                                global_pair_start + 1, batch_group_chunk
                            ].to(cutlass.Float32)
                            d_mp0_partial_re = d_mp0_partial_re + (
                                g0 * b_prev_re + g1 * b_prev_im
                            )
                            d_mp0_partial_im = d_mp0_partial_im + (
                                b_prev_re * g1 - b_prev_im * g0
                            )

                cute.arch.sync_threads()

            if cute.elem_less(
                tidx, cutlass.Int32(self.du_prev_threads)
            ) and cute.elem_less(du_boundary_row, self.P):
                mDUPrev[du_boundary_row, batch_head_chunk] = du_prev_accum.to(
                    mDUPrev.element_type
                )

            if not cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                for offset in (16, 8, 4, 2, 1):
                    d_mp0_partial_re = d_mp0_partial_re + cute.arch.shuffle_sync_bfly(
                        d_mp0_partial_re, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    d_mp0_partial_im = d_mp0_partial_im + cute.arch.shuffle_sync_bfly(
                        d_mp0_partial_im, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if lane_idx == cutlass.Int32(0):
                    db_prev_warp = warp_idx - du_worker_warp_count
                    s_db_prev_partial_sum[db_prev_warp, 0] = d_mp0_partial_re
                    s_db_prev_partial_sum[db_prev_warp, 1] = d_mp0_partial_im
        else:
            # Full-D fallback when the boundary slab is too wide for the tiled path.
            complex_pair = tidx
            complex_pair_count = self.D // 2
            while cute.elem_less(complex_pair, complex_pair_count):
                d_pair_start = complex_pair * 2
                b_prev_re = mBPrev[d_pair_start + 0, batch_group_chunk].to(
                    cutlass.Float32
                )
                b_prev_im = mBPrev[d_pair_start + 1, batch_group_chunk].to(
                    cutlass.Float32
                )
                s_decayed_boundary_key[d_pair_start + 0] = (
                    boundary_coeff_re * b_prev_re - boundary_coeff_im * b_prev_im
                )
                s_decayed_boundary_key[d_pair_start + 1] = (
                    boundary_coeff_re * b_prev_im + boundary_coeff_im * b_prev_re
                )
                complex_pair = complex_pair + self.num_threads

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            du_prev_accum = cutlass.Float32(0.0)
            if cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                boundary_row = tidx
                while cute.elem_less(boundary_row, self.P):
                    du_prev_accum = cutlass.Float32(0.0)
                    boundary_col = cutlass.Int32(0)
                    while cute.elem_less(boundary_col, self.D):
                        du_prev_accum = (
                            du_prev_accum
                            + s_staged_dinc[boundary_row, boundary_col].to(
                                cutlass.Float32
                            )
                            * s_decayed_boundary_key[boundary_col]
                        )
                        boundary_col = boundary_col + 1
                    mDUPrev[boundary_row, batch_head_chunk] = du_prev_accum.to(
                        mDUPrev.element_type
                    )
                    boundary_row = boundary_row + self.du_prev_threads

            d_mp0_partial_re = cutlass.Float32(0.0)
            d_mp0_partial_im = cutlass.Float32(0.0)
            if not cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                complex_pair = tidx - self.du_prev_threads
                complex_pair_stride = cutlass.Int32(self.db_prev_threads)

                while cute.elem_less(complex_pair, complex_pair_count):
                    d_pair_start = complex_pair * 2
                    g0 = cutlass.Float32(0.0)
                    g1 = cutlass.Float32(0.0)
                    for boundary_row in cutlass.range(self.P, unroll=1):
                        boundary_value = s_boundary_value[boundary_row]
                        g0 = g0 + boundary_value * s_staged_dinc[
                            boundary_row, d_pair_start + 0
                        ].to(cutlass.Float32)
                        g1 = g1 + boundary_value * s_staged_dinc[
                            boundary_row, d_pair_start + 1
                        ].to(cutlass.Float32)

                    mDBPrev[d_pair_start + 0, batch_head_chunk] = (
                        boundary_coeff_re * g0 + boundary_coeff_im * g1
                    ).to(mDBPrev.element_type)
                    mDBPrev[d_pair_start + 1, batch_head_chunk] = (
                        boundary_coeff_re * g1 - boundary_coeff_im * g0
                    ).to(mDBPrev.element_type)

                    b_prev_re = mBPrev[d_pair_start + 0, batch_group_chunk].to(
                        cutlass.Float32
                    )
                    b_prev_im = mBPrev[d_pair_start + 1, batch_group_chunk].to(
                        cutlass.Float32
                    )
                    d_mp0_partial_re = d_mp0_partial_re + (
                        g0 * b_prev_re + g1 * b_prev_im
                    )
                    d_mp0_partial_im = d_mp0_partial_im + (
                        b_prev_re * g1 - b_prev_im * g0
                    )
                    complex_pair = complex_pair + complex_pair_stride

                for offset in (16, 8, 4, 2, 1):
                    d_mp0_partial_re = d_mp0_partial_re + cute.arch.shuffle_sync_bfly(
                        d_mp0_partial_re, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    d_mp0_partial_im = d_mp0_partial_im + cute.arch.shuffle_sync_bfly(
                        d_mp0_partial_im, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if lane_idx == cutlass.Int32(0):
                    db_prev_warp = warp_idx - du_worker_warp_count
                    s_db_prev_partial_sum[db_prev_warp, 0] = d_mp0_partial_re
                    s_db_prev_partial_sum[db_prev_warp, 1] = d_mp0_partial_im

        cute.arch.sync_threads()

        if tidx == 0:
            d_mp0_re = cutlass.Float32(0.0)
            d_mp0_im = cutlass.Float32(0.0)
            for db_prev_warp in cutlass.range(self.db_prev_warp_count, unroll=1):
                d_mp0_re = d_mp0_re + s_db_prev_partial_sum[db_prev_warp, 0]
                d_mp0_im = d_mp0_im + s_db_prev_partial_sum[db_prev_warp, 1]
            mDMp0[0, batch_head_chunk] = d_mp0_re
            mDMp0[1, batch_head_chunk] = d_mp0_im
