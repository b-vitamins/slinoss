"""CuTe backward ``dc`` workhorse for the ``v2x2ssd`` chunk-scan stage.

This file is intentionally written in the same overall shape as the
``v3x3ssd`` ``chunk_scan`` ``dc`` workhorse:

- one monolithic stage-native kernel class
- direct public-stage inputs and outputs
- full shared-memory metadata prepass from raw packed ``M``
- row-tile ``dQ`` accumulation over causal ``n_tile`` blocks
- off-term accumulation against ``Z0``
- direct epilogue ownership of ``dC``

The adaptation is only in the scan algebra:

- quaternion transport becomes unit-complex transport
- 3-vectors become interleaved complex pairs
- FOH vector taps become complex-scalar taps
- ``trans`` becomes raw packed ``M``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import (
    LOG2_E,
    TWO_LOG2_E,
    apply_complex_tap,
    complex_mul,
    conj_mul_phase,
)


@dataclass(frozen=True)
class ChunkScanBwdDCLayoutBundle:
    s_u_prev_layout: object
    s_b_prev_layout: object
    sDY_layout: object
    sV_layout: object
    sK_layout: object
    sDS_layout: object
    sZ0_layout: object
    s_scale_full_layout: object
    s_inv_scale_full_layout: object
    s_phase_full_layout: object
    tile_prefix_log_layout: object
    tile_prefix_phase_layout: object
    s_row_layout: object
    s_inv_row_layout: object
    s_phase_row_layout: object
    s_phase_col_layout: object
    s_tap_layout: object
    s_row_dlp_layout: object
    s_row_dR_layout: object


@dataclass(frozen=True)
class ChunkScanBwdDCCopyBundle:
    gmem_tiled_copy_D: object
    gmem_tiled_copy_D_async: object
    gmem_tiled_copy_P: object
    gmem_tiled_copy_D_f32: object
    gmem_tiled_store_D: object


@dataclass(frozen=True)
class ChunkScanBwdDCKernelBundle:
    layouts: ChunkScanBwdDCLayoutBundle
    copies: ChunkScanBwdDCCopyBundle
    tiled_mma: object
    smem_bytes: int


@dataclass(frozen=True)
class ChunkScanBwdDCSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkScanBwdDCAmpere:
    """Ampere tensor-core kernel for ``dC``.

    Computes, per chunk (BHC):
      - ``dC`` (queries)

    Notes:
      - This kernel owns the contraction-heavy ``dc`` slice directly.
      - Metadata is reconstructed inside the CTA from raw packed ``M``.
      - The heavy contractions follow the same overall pattern as ``v3``:
        ``dS = dY @ V^T``, score reconstruction, ``dQ += dS_scaled @ K_rot``,
        then an off-term against ``Z0``.
    """

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkScanBwdDCSupportInfo]
    ] = {}

    def __init__(self, dtype, *, chunk_size, D, P, num_threads=128):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        self.num_threads = int(num_threads)
        self.kv_tile = 32
        self.p_tile = 32

        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.L % self.kv_tile != 0:
            raise ValueError("chunk_size must be a multiple of 32.")
        if self.D % 2 != 0:
            raise ValueError("D must be divisible by 2 (flattened 2N).")
        if self.num_threads != 128:
            raise ValueError("num_threads must be 128 for the 32x32 tensorop tile.")

        self.mma_inst_shape = (16, 8, 16)
        self.warp_layout_mnk = (2, 2, 1)
        self.atom_layout_mnk = self.warp_layout_mnk

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32

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

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _tail_pad_layout(self):
        return cute.make_layout((128,), stride=(1,))

    def _smem_block_size_D(self) -> int:
        return 64 if self.D_padded % 64 == 0 else 32

    @staticmethod
    def _swizzle_bits(smem_block_size: int) -> int:
        return 3 if smem_block_size == 64 else 2

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

    def _make_layout_bundle(self) -> ChunkScanBwdDCLayoutBundle:
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        n_tiles = self.L // kv_tile

        smem_k_block_size_D = self._smem_block_size_D()
        swizzle_bits_D = self._swizzle_bits(smem_k_block_size_D)
        sD_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_D, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_D), stride=(smem_k_block_size_D, 1)),
        )
        sK_layout = cute.tile_to_shape(
            sD_layout_atom, (kv_tile, smem_k_block_size_D), (0, 1)
        )
        sZ0_layout = cute.tile_to_shape(
            sD_layout_atom, (p_tile, smem_k_block_size_D), (0, 1)
        )

        smem_k_block_size_P = p_tile
        swizzle_bits_P = self._swizzle_bits(smem_k_block_size_P)
        sP_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_P, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_P), stride=(smem_k_block_size_P, 1)),
        )
        sDY_layout = cute.tile_to_shape(sP_layout_atom, (kv_tile, p_tile), (0, 1))
        sV_layout = cute.tile_to_shape(sP_layout_atom, (kv_tile, p_tile), (0, 1))

        smem_k_block_size_blk = kv_tile
        swizzle_bits_blk = self._swizzle_bits(smem_k_block_size_blk)
        sBlk_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_blk, 3, 3),
            0,
            cute.make_layout(
                (8, smem_k_block_size_blk), stride=(smem_k_block_size_blk, 1)
            ),
        )
        sDS_layout = cute.tile_to_shape(sBlk_layout_atom, (kv_tile, kv_tile), (0, 1))

        return ChunkScanBwdDCLayoutBundle(
            s_u_prev_layout=cute.make_layout((self.P,), stride=(1,)),
            s_b_prev_layout=cute.make_layout((self.D,), stride=(1,)),
            sDY_layout=sDY_layout,
            sV_layout=sV_layout,
            sK_layout=sK_layout,
            sDS_layout=sDS_layout,
            sZ0_layout=sZ0_layout,
            s_scale_full_layout=cute.make_layout((self.L,), stride=(1,)),
            s_inv_scale_full_layout=cute.make_layout((self.L,), stride=(1,)),
            s_phase_full_layout=cute.make_layout((self.L, 2), stride=(2, 1)),
            tile_prefix_log_layout=cute.make_layout((n_tiles,), stride=(1,)),
            tile_prefix_phase_layout=cute.make_layout((n_tiles, 2), stride=(2, 1)),
            s_row_layout=cute.make_layout((kv_tile,), stride=(1,)),
            s_inv_row_layout=cute.make_layout((kv_tile,), stride=(1,)),
            s_phase_row_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
            s_phase_col_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
            s_tap_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
            s_row_dlp_layout=cute.make_layout((kv_tile,), stride=(1,)),
            s_row_dR_layout=cute.make_layout((kv_tile, 4), stride=(4, 1)),
        )

    def _required_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        in_bytes = in_dtype.width // 8
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        d_block = self._smem_block_size_D()
        n_tiles = self.L // kv_tile
        return self._struct_size_bytes(
            [
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * d_block * in_bytes, 16),
                (kv_tile * kv_tile * in_bytes, 16),
                (self.P * in_bytes, 16),
                (self.D * in_bytes, 16),
                (self.L * 4, 4),
                (self.L * 4, 4),
                (self.L * 2 * 4, 16),
                (n_tiles * 4, 4),
                (n_tiles * 2 * 4, 16),
                (n_tiles * 4, 4),
                (n_tiles * 2 * 4, 16),
                (kv_tile * 4, 4),
                (kv_tile * 4, 4),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 4, 4),
                (kv_tile * 4 * 4, 16),
                (128 * 4, 16),
            ]
        )

    def support_info(
        self,
        in_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> ChunkScanBwdDCSupportInfo:
        if in_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return ChunkScanBwdDCSupportInfo(0, 1)

        if device_index is None:
            device_key = (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
            )
        else:
            device_key = int(device_index)
        cache_key = (
            type(self),
            self.L,
            self.D,
            self.P,
            self.num_threads,
            in_dtype,
            device_key,
        )
        cached = self._SUPPORT_INFO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        info = ChunkScanBwdDCSupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            required_smem_bytes=self._required_smem_bytes(in_dtype),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def _make_copy_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDCCopyBundle:
        universal_copy_bits = 128
        async_elems_in = universal_copy_bits // in_dtype.width
        smem_k_block_size_D = self._smem_block_size_D()
        tD_shape_dim_1 = smem_k_block_size_D // async_elems_in
        tD_layout = cute.make_layout(
            (self.num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        tD_shape_dim_1_f32 = smem_k_block_size_D // (
            universal_copy_bits // cutlass.Float32.width
        )
        tD_layout_f32 = cute.make_layout(
            (self.num_threads // tD_shape_dim_1_f32, tD_shape_dim_1_f32),
            stride=(tD_shape_dim_1_f32, 1),
        )
        tP_shape_dim_1 = self.p_tile // async_elems_in
        tP_layout = cute.make_layout(
            (self.num_threads // tP_shape_dim_1, tP_shape_dim_1),
            stride=(tP_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))
        v_f32_layout = cute.make_layout(
            (1, universal_copy_bits // cutlass.Float32.width)
        )

        atom_universal_copy_in = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy_f32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        return ChunkScanBwdDCCopyBundle(
            gmem_tiled_copy_D=cute.make_tiled_copy_tv(
                atom_universal_copy_in, tD_layout, v_in_layout
            ),
            gmem_tiled_copy_D_async=cute.make_tiled_copy_tv(
                atom_async_copy_in, tD_layout, v_in_layout
            ),
            gmem_tiled_copy_P=cute.make_tiled_copy_tv(
                atom_async_copy_in, tP_layout, v_in_layout
            ),
            gmem_tiled_copy_D_f32=cute.make_tiled_copy_tv(
                atom_universal_copy_f32, tD_layout_f32, v_f32_layout
            ),
            gmem_tiled_store_D=cute.make_tiled_copy_tv(
                atom_universal_copy_out, tD_layout, v_in_layout
            ),
        )

    def _make_tiled_mma(self, in_dtype: type[cutlass.Numeric]):
        op = cute.nvgpu.warp.MmaF16BF16Op(in_dtype, self.acc_dtype, (16, 8, 16))
        permutation_mnk = (
            self.atom_layout_mnk[0] * 16,
            self.atom_layout_mnk[1] * 8 * 2,
            self.atom_layout_mnk[2] * 16,
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        return cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

    def _make_kernel_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDCKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage = self._make_shared_storage(in_dtype, layouts)
        return ChunkScanBwdDCKernelBundle(
            layouts=layouts,
            copies=self._make_copy_bundle(in_dtype),
            tiled_mma=self._make_tiled_mma(in_dtype),
            smem_bytes=int(shared_storage.size_in_bytes()),
        )

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkScanBwdDCLayoutBundle,
    ):
        tail_pad_layout = self._tail_pad_layout()

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sDY": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sDY_layout)], 16
            ],
            "sV_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sV_layout)], 16
            ],
            "sK_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sK_layout)], 16
            ],
            "sDS_blk": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sDS_layout)], 16
            ],
            "s_u_prev": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.s_u_prev_layout)], 16
            ],
            "s_b_prev": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.s_b_prev_layout)], 16
            ],
            "s_scale_full": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_scale_full_layout)
                ],
                4,
            ],
            "s_inv_scale_full": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_inv_scale_full_layout)
                ],
                4,
            ],
            "s_phase_full": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_phase_full_layout)
                ],
                16,
            ],
            "s_tile_end_log": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_log_layout)
                ],
                4,
            ],
            "s_tile_end_phase": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_phase_layout)
                ],
                16,
            ],
            "s_tile_off_log": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_log_layout)
                ],
                4,
            ],
            "s_tile_off_phase": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_phase_layout)
                ],
                16,
            ],
            "s_row_scale": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_row_layout)
                ],
                4,
            ],
            "s_inv_row_scale": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_inv_row_layout)
                ],
                4,
            ],
            "s_phase_row": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_phase_row_layout)
                ],
                16,
            ],
            "s_phase_col": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_phase_col_layout)
                ],
                16,
            ],
            "s_tap_prev": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_tap_layout)
                ],
                16,
            ],
            "s_tap_curr": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_tap_layout)
                ],
                16,
            ],
            "s_row_dlp": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_row_dlp_layout)
                ],
                4,
            ],
            "s_row_dR": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.s_row_dR_layout)
                ],
                16,
            ],
            "tail_pad": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tail_pad_layout)],
                16,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDOut: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mZ0: cute.Tensor,
        mDC: cute.Tensor,
        mDLogp: cute.Tensor,
        mDR: cute.Tensor,
    ):
        if cutlass.const_expr(
            mU.element_type != mB.element_type
            or mU.element_type != mC.element_type
            or mU.element_type != mDOut.element_type
        ):
            raise TypeError("U/B/C/dOut must share dtype.")
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("U/B/C/dOut must be Float16/BFloat16.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mK.element_type != cutlass.Float32):
            raise TypeError("K must be Float32.")
        if cutlass.const_expr(
            mZ0.element_type != cutlass.Float32 and mZ0.element_type != mU.element_type
        ):
            raise TypeError("Z0 must be Float32 or match U/B dtype.")
        if cutlass.const_expr(mDC.element_type != mU.element_type):
            raise TypeError("dC must share dtype with U/B.")
        if cutlass.const_expr(mDLogp.element_type != cutlass.Float32):
            raise TypeError("dlogprefix must be Float32.")
        if cutlass.const_expr(mDR.element_type != cutlass.Float32):
            raise TypeError("dR must be Float32.")

        if cutlass.const_expr(
            mU.shape[1] != self.L or mB.shape[1] != self.L or mC.shape[1] != self.L
        ):
            raise ValueError("U/B/C must have shape (BHC, L, 1, ...).")
        if cutlass.const_expr(mU.shape[2] != 1 or mB.shape[2] != 1 or mC.shape[2] != 1):
            raise ValueError("U/B/C must have a singleton dim2 (BHC, L, 1, ...).")
        if cutlass.const_expr(mM.shape[1] != self.L or mM.shape[2] != 2):
            raise ValueError("M must be (BHC, L, 2).")
        if cutlass.const_expr(
            mK.shape[1] != self.L or mK.shape[2] != 2 or mK.shape[3] != 2
        ):
            raise ValueError("K must be (BHC, L, 2, 2).")
        if cutlass.const_expr(mDOut.shape[1] != self.L or mDOut.shape[2] != 1):
            raise ValueError("dOut must be (BHC, L, 1, P).")
        if cutlass.const_expr(mZ0.shape[1] != self.P or mZ0.shape[2] != self.D):
            raise ValueError("Z0 must be (BHC, P, D).")
        if cutlass.const_expr(mDC.shape[1] != self.L or mDC.shape[2] != 1):
            raise ValueError("dC must be (BHC, L, 1, D).")
        if cutlass.const_expr(mDLogp.shape[1] != self.L):
            raise ValueError("dlogprefix must be (BHC, L).")
        if cutlass.const_expr(mDR.shape[1] != self.L or mDR.shape[2] != 4):
            raise ValueError("dR must be (BHC, L, 4).")

        Pp = self.P_padded
        in_dtype = mU.element_type
        p_tile = self.p_tile
        if cutlass.const_expr(Pp % p_tile != 0):
            raise ValueError("P_padded must be a multiple of 32.")
        kernel_bundle = self._make_kernel_bundle(in_dtype)

        grid_z = cute.size(mU.shape[0])
        self.kernel_dc(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mZ0,
            mDC,
            mDLogp,
            mDR,
            kernel_bundle.layouts.s_u_prev_layout,
            kernel_bundle.layouts.s_b_prev_layout,
            kernel_bundle.layouts.sDY_layout,
            kernel_bundle.layouts.sV_layout,
            kernel_bundle.layouts.sK_layout,
            kernel_bundle.layouts.sDS_layout,
            kernel_bundle.layouts.sZ0_layout,
            kernel_bundle.layouts.s_scale_full_layout,
            kernel_bundle.layouts.s_inv_scale_full_layout,
            kernel_bundle.layouts.s_phase_full_layout,
            kernel_bundle.layouts.tile_prefix_log_layout,
            kernel_bundle.layouts.tile_prefix_phase_layout,
            kernel_bundle.layouts.s_row_layout,
            kernel_bundle.layouts.s_inv_row_layout,
            kernel_bundle.layouts.s_phase_row_layout,
            kernel_bundle.layouts.s_phase_col_layout,
            kernel_bundle.layouts.s_tap_layout,
            kernel_bundle.layouts.s_row_dlp_layout,
            kernel_bundle.layouts.s_row_dR_layout,
            kernel_bundle.copies.gmem_tiled_copy_D,
            kernel_bundle.copies.gmem_tiled_copy_D_async,
            kernel_bundle.copies.gmem_tiled_copy_P,
            kernel_bundle.copies.gmem_tiled_copy_D_f32,
            kernel_bundle.copies.gmem_tiled_store_D,
            kernel_bundle.tiled_mma,
        ).launch(
            grid=(1, 1, grid_z),
            block=[self.num_threads, 1, 1],
            smem=kernel_bundle.smem_bytes,
        )

    @cute.kernel
    def kernel_dc(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDOut: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mZ0: cute.Tensor,
        mDC: cute.Tensor,
        mDLogp: cute.Tensor,
        mDR: cute.Tensor,
        s_u_prev_layout: cute.Layout,
        s_b_prev_layout: cute.Layout,
        sDY_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sDS_layout: cute.Layout,
        sZ0_layout: cute.Layout,
        s_scale_full_layout: cute.Layout,
        s_inv_scale_full_layout: cute.Layout,
        s_phase_full_layout: cute.Layout,
        tile_prefix_log_layout: cute.Layout,
        tile_prefix_phase_layout: cute.Layout,
        s_row_layout: cute.Layout,
        s_inv_row_layout: cute.Layout,
        s_phase_row_layout: cute.Layout,
        s_phase_col_layout: cute.Layout,
        s_tap_layout: cute.Layout,
        s_row_dlp_layout: cute.Layout,
        s_row_dR_layout: cute.Layout,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_tiled_copy_D_async: cute.TiledCopy,
        gmem_tiled_copy_P: cute.TiledCopy,
        gmem_tiled_copy_D_f32: cute.TiledCopy,
        gmem_tiled_store_D: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()

        BH = mU_prev0.shape[0]
        BHC = mU.shape[0]
        n_chunks = BHC // BH
        bh = bidz // n_chunks
        chunk = bidz - bh * n_chunks

        Dp = self.D_padded
        Pp = self.P_padded
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        n_p_tiles = Pp // p_tile
        n_tiles = self.L // kv_tile

        layouts = ChunkScanBwdDCLayoutBundle(
            s_u_prev_layout=s_u_prev_layout,
            s_b_prev_layout=s_b_prev_layout,
            sDY_layout=sDY_layout,
            sV_layout=sV_layout,
            sK_layout=sK_layout,
            sDS_layout=sDS_layout,
            sZ0_layout=sZ0_layout,
            s_scale_full_layout=s_scale_full_layout,
            s_inv_scale_full_layout=s_inv_scale_full_layout,
            s_phase_full_layout=s_phase_full_layout,
            tile_prefix_log_layout=tile_prefix_log_layout,
            tile_prefix_phase_layout=tile_prefix_phase_layout,
            s_row_layout=s_row_layout,
            s_inv_row_layout=s_inv_row_layout,
            s_phase_row_layout=s_phase_row_layout,
            s_phase_col_layout=s_phase_col_layout,
            s_tap_layout=s_tap_layout,
            s_row_dlp_layout=s_row_dlp_layout,
            s_row_dR_layout=s_row_dR_layout,
        )
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._make_shared_storage(mU.element_type, layouts)
        storage = smem.allocate(SharedStorage)
        sDY = storage.sDY.get_tensor(sDY_layout)
        sV_tile = storage.sV_tile.get_tensor(sV_layout)
        sK_tile = storage.sK_tile.get_tensor(sK_layout)
        sDS_blk = storage.sDS_blk.get_tensor(sDS_layout)
        # `Z0` is only consumed after the causal `n_tile` sweep, so it can reuse
        # the same 32xDp slab as `sK_tile` instead of reserving a third full-D tile.
        sZ0 = cute.make_tensor(sK_tile.iterator, sZ0_layout)
        s_u_prev = storage.s_u_prev.get_tensor(s_u_prev_layout)
        s_b_prev = storage.s_b_prev.get_tensor(s_b_prev_layout)

        s_scale_full = storage.s_scale_full.get_tensor(s_scale_full_layout)
        s_inv_scale_full = storage.s_inv_scale_full.get_tensor(s_inv_scale_full_layout)
        s_phase_full = storage.s_phase_full.get_tensor(s_phase_full_layout)
        s_tile_end_log = storage.s_tile_end_log.get_tensor(tile_prefix_log_layout)
        s_tile_end_phase = storage.s_tile_end_phase.get_tensor(tile_prefix_phase_layout)
        s_tile_off_log = storage.s_tile_off_log.get_tensor(tile_prefix_log_layout)
        s_tile_off_phase = storage.s_tile_off_phase.get_tensor(tile_prefix_phase_layout)

        s_row_scale = storage.s_row_scale.get_tensor(s_row_layout)
        s_inv_row_scale = storage.s_inv_row_scale.get_tensor(s_inv_row_layout)
        s_phase_row = storage.s_phase_row.get_tensor(s_phase_row_layout)
        s_phase_col = storage.s_phase_col.get_tensor(s_phase_col_layout)
        s_tap_prev = storage.s_tap_prev.get_tensor(s_tap_layout)
        s_tap_curr = storage.s_tap_curr.get_tensor(s_tap_layout)
        s_row_dlp = storage.s_row_dlp.get_tensor(s_row_dlp_layout)
        s_row_dR = storage.s_row_dR.get_tensor(s_row_dR_layout)

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        num_warps = self.num_threads // 32
        eps = cutlass.Float32(1.0e-20)
        one = cutlass.Float32(1.0)

        for tile in range(n_tiles):
            if warp == cutlass.Int32(tile % num_warps) and lane < cutlass.Int32(
                kv_tile
            ):
                t = cutlass.Int32(tile * kv_tile) + lane
                mr = cutlass.Float32(mM[bidz, t, 0])
                mi = cutlass.Float32(mM[bidz, t, 1])
                mag2 = cutlass.Float32(mr * mr + mi * mi + eps)
                inv_mag = cutlass.Float32(cute.math.rsqrt(mag2, fastmath=True))
                pr = mr * inv_mag
                pi = mi * inv_mag
                logp = cutlass.Float32(
                    cute.math.log2(mag2, fastmath=False)
                    * cutlass.Float32(0.25 / LOG2_E)
                )
                for offset in (1, 2, 4, 8, 16):
                    other_log = cute.arch.shuffle_sync_up(
                        logp, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    opr = cute.arch.shuffle_sync_up(
                        pr, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    opi = cute.arch.shuffle_sync_up(
                        pi, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    pred = lane >= cutlass.Int32(offset)
                    logp = cutlass.select_(pred, logp + other_log, logp)
                    nr, ni = complex_mul(pr, pi, opr, opi)
                    pr = cutlass.select_(pred, nr, pr)
                    pi = cutlass.select_(pred, ni, pi)
                s_scale_full[t] = logp
                s_phase_full[t, 0] = pr
                s_phase_full[t, 1] = pi
                if lane == cutlass.Int32(kv_tile - 1):
                    s_tile_end_log[tile] = logp
                    s_tile_end_phase[tile, 0] = pr
                    s_tile_end_phase[tile, 1] = pi
        cute.arch.barrier()

        if warp == cutlass.Int32(0) and lane == cutlass.Int32(0):
            off_log = cutlass.Float32(0.0)
            off_r = cutlass.Float32(1.0)
            off_i = cutlass.Float32(0.0)
            for tile in range(n_tiles):
                s_tile_off_log[tile] = off_log
                s_tile_off_phase[tile, 0] = off_r
                s_tile_off_phase[tile, 1] = off_i
                last_log = cutlass.Float32(s_tile_end_log[tile])
                last_r = cutlass.Float32(s_tile_end_phase[tile, 0])
                last_i = cutlass.Float32(s_tile_end_phase[tile, 1])
                off_log = off_log + last_log
                off_r, off_i = complex_mul(last_r, last_i, off_r, off_i)
        cute.arch.barrier()

        for tile in range(n_tiles):
            if warp == cutlass.Int32(tile % num_warps) and lane < cutlass.Int32(
                kv_tile
            ):
                t = cutlass.Int32(tile * kv_tile) + lane
                logp = cutlass.Float32(s_scale_full[t]) + cutlass.Float32(
                    s_tile_off_log[tile]
                )
                pr = cutlass.Float32(s_phase_full[t, 0])
                pi = cutlass.Float32(s_phase_full[t, 1])
                off_r = cutlass.Float32(s_tile_off_phase[tile, 0])
                off_i = cutlass.Float32(s_tile_off_phase[tile, 1])
                pr, pi = complex_mul(pr, pi, off_r, off_i)
                mag2 = cutlass.Float32(pr * pr + pi * pi + eps)
                inv_mag = cutlass.Float32(cute.math.rsqrt(mag2, fastmath=True))
                pr = pr * inv_mag
                pi = pi * inv_mag
                scale = cute.math.exp2(
                    logp * cutlass.Float32(TWO_LOG2_E), fastmath=True
                )
                s_scale_full[t] = scale
                s_inv_scale_full[t] = one / scale
                s_phase_full[t, 0] = pr
                s_phase_full[t, 1] = pi
        cute.arch.barrier()

        prev_bidz = cutlass.select_(
            chunk > cutlass.Int32(0), bidz - cutlass.Int32(1), bidz
        )
        iters_p = (self.P + self.num_threads - 1) // self.num_threads
        for it in range(iters_p):
            p = tidx + cutlass.Int32(it * self.num_threads)
            if p < cutlass.Int32(self.P):
                u_prev0 = mU_prev0[bh, p]
                u_prev_chunk = mU[prev_bidz, self.L - 1, 0, p]
                s_u_prev[p] = cutlass.select_(
                    chunk == cutlass.Int32(0), u_prev0, u_prev_chunk
                )
        iters_d = (self.D + self.num_threads - 1) // self.num_threads
        for it in range(iters_d):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(self.D):
                b_prev0 = mB_prev0[bh, d]
                b_prev_chunk = mB[prev_bidz, self.L - 1, 0, d]
                s_b_prev[d] = cutlass.select_(
                    chunk == cutlass.Int32(0), b_prev0, b_prev_chunk
                )
        cute.arch.barrier()

        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_BT = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mU.element_type,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_tiled_copy_BT = cute.make_tiled_copy_B(smem_copy_atom_BT, tiled_mma)
        thr_mma = tiled_mma.get_slice(tidx)

        mcS = cute.make_identity_tensor((mU.shape[0], self.L, mU.shape[2], self.L))
        mcKD = cute.make_identity_tensor((mU.shape[0], self.L, mU.shape[2], Dp))
        mcS_full = mcS[bidz, None, 0, None]
        mcKD_full = mcKD[bidz, None, 0, None]

        d_block = self._smem_block_size_D()
        n_d_tiles = Dp // d_block

        acc_shape_blk = thr_mma.partition_shape_C((kv_tile, kv_tile))
        acc_shape_tileD_blk = thr_mma.partition_shape_C((kv_tile, d_block))

        thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        thr_copy_BT = smem_tiled_copy_BT.get_slice(tidx)
        gmem_thr_copy_D_async = gmem_tiled_copy_D_async.get_slice(tidx)
        gmem_thr_copy_P = gmem_tiled_copy_P.get_slice(tidx)

        tSrDS_blk = thr_mma.make_fragment_A(thr_mma.partition_A(sDS_blk))
        tSsDS_blk = thr_copy_A.partition_S(sDS_blk)
        tSrDS_blk_view = thr_copy_A.retile(tSrDS_blk)
        tSrDY_tile = thr_mma.make_fragment_A(thr_mma.partition_A(sDY))
        tSsDY_tile = thr_copy_A.partition_S(sDY)
        tSrDY_tile_view = thr_copy_A.retile(tSrDY_tile)
        tSrV_tile = thr_mma.make_fragment_B(thr_mma.partition_B(sV_tile))
        tSsV_tile = thr_copy_B.partition_S(sV_tile)
        tSrV_tile_view = thr_copy_B.retile(tSrV_tile)
        tDYs = gmem_thr_copy_P.partition_D(sDY)
        mcDY = cute.make_identity_tensor(mDOut.layout.shape)
        mcDY_full = mcDY[bidz, None, 0, None]

        total_v_tile = cutlass.Int32(kv_tile * p_tile)
        iters_v_tile = (total_v_tile + self.num_threads - 1) // self.num_threads
        total_pairs_tile = cutlass.Int32(kv_tile * (d_block // 2))
        iters_pairs_tile = (total_pairs_tile + self.num_threads - 1) // self.num_threads
        total_k_tile = cutlass.Int32(kv_tile * d_block)
        iters_k_tile = (total_k_tile + self.num_threads - 1) // self.num_threads
        total_z0_tile = cutlass.Int32(p_tile * d_block)
        iters_z0_tile = (total_z0_tile + self.num_threads - 1) // self.num_threads
        total_blk_tile = cutlass.Int32(kv_tile * kv_tile)
        iters_blk_tile = (total_blk_tile + self.num_threads - 1) // self.num_threads
        tDYp = None
        if cutlass.const_expr(self.P != Pp):
            tDYp = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tDYs.shape[0][1],
                        cute.size(tDYs, mode=[1]),
                        cute.size(tDYs, mode=[2]),
                    ),
                    stride=(cute.size(tDYs, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )

        for m_tile in range(n_tiles):
            m0 = cutlass.Int32(m_tile * kv_tile)

            if warp == cutlass.Int32(0) and lane < cutlass.Int32(kv_tile):
                t = m0 + lane
                s_row_scale[lane] = cutlass.Float32(s_scale_full[t])
                s_phase_row[lane, 0] = cutlass.Float32(s_phase_full[t, 0])
                s_phase_row[lane, 1] = cutlass.Float32(s_phase_full[t, 1])
            if tidx < cutlass.Int32(kv_tile):
                s_row_dlp[tidx] = cutlass.Float32(0.0)
            if tidx < cutlass.Int32(kv_tile * 4):
                row = tidx // cutlass.Int32(4)
                col = tidx - row * cutlass.Int32(4)
                s_row_dR[row, col] = cutlass.Float32(0.0)
            cute.arch.barrier()

            for d_tile in range(n_d_tiles):
                d_base = cutlass.Int32(d_tile * d_block)
                pairs_per_row = cutlass.Int32(d_block // 2)

                sK_blk = sK_tile
                sKt_blk_layout = cute.make_layout(
                    (d_block, kv_tile), stride=(kv_tile, 1)
                )
                sKt_blk = cute.composition(sK_blk, sKt_blk_layout)
                tKsK_blk = gmem_thr_copy_D_async.partition_D(sK_blk)
                sZ0_blk = sZ0
                sZ0t_blk_layout = cute.make_layout(
                    (d_block, p_tile), stride=(p_tile, 1)
                )
                sZ0t_blk = cute.composition(sZ0_blk, sZ0t_blk_layout)

                tSrK_blk = thr_mma.make_fragment_B(thr_mma.partition_B(sKt_blk))
                tSsK_blk = thr_copy_BT.partition_S(sKt_blk)
                tSrK_blk_view = thr_copy_BT.retile(tSrK_blk)
                tSrZ0_blk = thr_mma.make_fragment_B(thr_mma.partition_B(sZ0t_blk))
                tSsZ0_blk = thr_copy_BT.partition_S(sZ0t_blk)
                tSrZ0_blk_view = thr_copy_BT.retile(tSrZ0_blk)

                acc_dQ_total = cute.make_rmem_tensor(
                    acc_shape_tileD_blk, cutlass.Float32
                )
                acc_dQ_total.fill(0.0)

                for n_tile in range(m_tile + 1):
                    n0 = cutlass.Int32(n_tile * kv_tile)

                    if warp == cutlass.Int32(0) and lane < cutlass.Int32(kv_tile):
                        t = n0 + lane
                        s_inv_row_scale[lane] = cutlass.Float32(s_inv_scale_full[t])
                        s_phase_col[lane, 0] = cutlass.Float32(s_phase_full[t, 0])
                        s_phase_col[lane, 1] = cutlass.Float32(s_phase_full[t, 1])
                        s_tap_prev[lane, 0] = cutlass.Float32(mK[bidz, t, 0, 0])
                        s_tap_prev[lane, 1] = cutlass.Float32(mK[bidz, t, 0, 1])
                        s_tap_curr[lane, 0] = cutlass.Float32(mK[bidz, t, 1, 0])
                        s_tap_curr[lane, 1] = cutlass.Float32(mK[bidz, t, 1, 1])
                    cute.arch.barrier()

                    for pass_id in range(2):
                        if pass_id == 1:
                            gB = cute.local_tile(
                                mB[bidz, None, 0, None],
                                (kv_tile, d_block),
                                (n_tile, d_tile),
                            )
                            tBg = gmem_thr_copy_D_async.partition_S(gB)
                            if cutlass.const_expr(self.D == Dp):
                                cute.copy(gmem_tiled_copy_D_async, tBg, tKsK_blk)
                            else:
                                cB = cute.local_tile(
                                    mcKD_full, (kv_tile, d_block), (n_tile, d_tile)
                                )
                                tBc = gmem_thr_copy_D_async.partition_S(cB)
                                tKp = cute.make_rmem_tensor(
                                    cute.make_layout(
                                        (
                                            tKsK_blk.shape[0][1],
                                            cute.size(tKsK_blk, mode=[1]),
                                            cute.size(tKsK_blk, mode=[2]),
                                        ),
                                        stride=(cute.size(tKsK_blk, mode=[2]), 0, 1),
                                    ),
                                    cutlass.Boolean,
                                )
                                for rest_v in cutlass.range_constexpr(tKp.shape[0]):
                                    for rest_k in cutlass.range_constexpr(tKp.shape[2]):
                                        tKp[rest_v, 0, rest_k] = cute.elem_less(
                                            tBc[(0, rest_v), 0, rest_k][3],
                                            mB.layout.shape[3],
                                        )
                                for vi in cutlass.range_constexpr(
                                    cute.size(tKsK_blk.shape[1])
                                ):
                                    if cute.elem_less(
                                        tBc[0, vi, 0][1], mB.layout.shape[1]
                                    ):
                                        cute.copy(
                                            gmem_tiled_copy_D_async,
                                            tBg[None, vi, None],
                                            tKsK_blk[None, vi, None],
                                            pred=tKp[None, vi, None],
                                        )
                                    else:
                                        tKsK_blk[None, vi, None].fill(0)
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                        else:
                            for it in range(iters_k_tile):
                                idx = tidx + cutlass.Int32(it * self.num_threads)
                                if idx < total_k_tile:
                                    t_local = idx // cutlass.Int32(d_block)
                                    d_local = idx - t_local * cutlass.Int32(d_block)
                                    d = d_base + d_local
                                    val = cutlass.Float32(0.0).to(mU.element_type)
                                    if d < cutlass.Int32(self.D):
                                        t = n0 + t_local
                                        t_src = t - cutlass.Int32(1)
                                        if t_src >= cutlass.Int32(0):
                                            val = mB[bidz, t_src, 0, d]
                                        else:
                                            val = s_b_prev[d]
                                    sK_blk[t_local, d_local] = val
                        cute.arch.barrier()

                        for it in range(iters_pairs_tile):
                            idx = tidx + cutlass.Int32(it * self.num_threads)
                            if idx < total_pairs_tile:
                                t_local = idx // pairs_per_row
                                vv = idx - t_local * pairs_per_row
                                t = n0 + t_local
                                d0_local = vv * cutlass.Int32(2)
                                d0 = d_base + d0_local
                                out0 = cutlass.Float32(0.0).to(mU.element_type)
                                out1 = cutlass.Float32(0.0).to(mU.element_type)
                                if t < cutlass.Int32(self.L) and cute.elem_less(
                                    d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                                ):
                                    bx = cutlass.Float32(
                                        sK_blk[t_local, d0_local + 0].to(
                                            cutlass.Float32
                                        )
                                    )
                                    by = cutlass.Float32(
                                        sK_blk[t_local, d0_local + 1].to(
                                            cutlass.Float32
                                        )
                                    )
                                    kr = cutlass.Float32(0.0)
                                    ki = cutlass.Float32(0.0)
                                    if pass_id == 1:
                                        kr = cutlass.Float32(s_tap_curr[t_local, 0])
                                        ki = cutlass.Float32(s_tap_curr[t_local, 1])
                                    else:
                                        kr = cutlass.Float32(s_tap_prev[t_local, 0])
                                        ki = cutlass.Float32(s_tap_prev[t_local, 1])
                                    tr, ti = apply_complex_tap(bx, by, kr, ki)
                                    pr = cutlass.Float32(s_phase_col[t_local, 0])
                                    pi = cutlass.Float32(s_phase_col[t_local, 1])
                                    outx, outy = conj_mul_phase(tr, ti, pr, pi)
                                    out0 = outx.to(mU.element_type)
                                    out1 = outy.to(mU.element_type)
                                sK_blk[t_local, d0_local + 0] = out0
                                sK_blk[t_local, d0_local + 1] = out1
                        cute.arch.barrier()

                        acc_dS_blk = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_dS_blk.fill(0.0)

                        for p_tile_idx in cutlass.range_constexpr(n_p_tiles):
                            p0 = cutlass.Int32(p_tile_idx * p_tile)
                            gDY = cute.local_tile(
                                mDOut[bidz, None, 0, None],
                                (kv_tile, p_tile),
                                (m_tile, p_tile_idx),
                            )
                            tDYg = gmem_thr_copy_P.partition_S(gDY)
                            if cutlass.const_expr(self.P == Pp):
                                cute.copy(gmem_tiled_copy_P, tDYg, tDYs)
                            else:
                                cDY = cute.local_tile(
                                    mcDY_full, (kv_tile, p_tile), (m_tile, p_tile_idx)
                                )
                                tDYc = gmem_thr_copy_P.partition_S(cDY)
                                for rest_v in cutlass.range_constexpr(tDYp.shape[0]):
                                    for rest_k in cutlass.range_constexpr(
                                        tDYp.shape[2]
                                    ):
                                        tDYp[rest_v, 0, rest_k] = cute.elem_less(
                                            tDYc[(0, rest_v), 0, rest_k][3],
                                            mDOut.layout.shape[3],
                                        )
                                for vi in cutlass.range_constexpr(
                                    cute.size(tDYs.shape[1])
                                ):
                                    if cute.elem_less(
                                        tDYc[0, vi, 0][1], mDOut.layout.shape[1]
                                    ):
                                        cute.copy(
                                            gmem_tiled_copy_P,
                                            tDYg[None, vi, None],
                                            tDYs[None, vi, None],
                                            pred=tDYp[None, vi, None],
                                        )
                                    else:
                                        tDYs[None, vi, None].fill(0)
                            cute.arch.cp_async_commit_group()
                            for it in range(iters_v_tile):
                                idx = tidx + cutlass.Int32(it * self.num_threads)
                                if idx < total_v_tile:
                                    t_local = idx // cutlass.Int32(p_tile)
                                    p_local = idx - t_local * cutlass.Int32(p_tile)
                                    p = p0 + p_local
                                    vv = cutlass.Float32(0.0).to(mU.element_type)
                                    if p < cutlass.Int32(self.P):
                                        if pass_id == 1:
                                            src_row = n0 + t_local
                                            if src_row < cutlass.Int32(self.L):
                                                vv = mU[bidz, src_row, 0, p]
                                        else:
                                            src_row = n0 + t_local - cutlass.Int32(1)
                                            if src_row >= cutlass.Int32(0):
                                                vv = mU[bidz, src_row, 0, p]
                                            else:
                                                vv = s_u_prev[p]
                                    sV_tile[t_local, p_local] = vv
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.barrier()

                            for k in cutlass.range_constexpr(
                                cute.size(tSsDY_tile.shape[2])
                            ):
                                cute.copy(
                                    smem_tiled_copy_A,
                                    tSsDY_tile[None, None, k],
                                    tSrDY_tile_view[None, None, k],
                                )
                                cute.copy(
                                    smem_tiled_copy_B,
                                    tSsV_tile[None, None, k],
                                    tSrV_tile_view[None, None, k],
                                )
                            for k in cutlass.range_constexpr(
                                cute.size(tSsDY_tile.shape[2])
                            ):
                                cute.gemm(
                                    tiled_mma,
                                    acc_dS_blk,
                                    tSrDY_tile[None, None, k],
                                    tSrV_tile[None, None, k],
                                    acc_dS_blk,
                                )
                            cute.arch.barrier()

                        cS_blk = cute.local_tile(
                            mcS_full, (kv_tile, kv_tile), (m_tile, n_tile)
                        )
                        tScS_blk = thr_mma.partition_C(cS_blk)
                        tScS_blk_mn = self._make_acc_tensor_mn_view(tScS_blk)
                        acc_dS_blk_mn = self._make_acc_tensor_mn_view(acc_dS_blk)
                        tCsDS_blk = thr_mma.partition_C(sDS_blk)
                        tCrDS_blk = cute.make_fragment_like(tCsDS_blk, mU.element_type)
                        tCrDS_blk_mn = self._make_acc_tensor_mn_view(tCrDS_blk)
                        for it in range(iters_blk_tile):
                            idx = tidx + cutlass.Int32(it * self.num_threads)
                            if idx < total_blk_tile:
                                row = idx // cutlass.Int32(kv_tile)
                                col = idx - row * cutlass.Int32(kv_tile)
                                sDS_blk[row, col] = cutlass.Float32(0.0).to(
                                    mU.element_type
                                )
                        cute.arch.barrier()
                        diag_tile = m_tile == n_tile
                        for r in cutlass.range_constexpr(
                            cute.size(acc_dS_blk_mn.shape[0])
                        ):
                            row_idx = cutlass.Int32(tScS_blk_mn[r, 0][1])
                            row_local = row_idx - m0
                            rs = cutlass.Float32(0.0)
                            if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                                rs = cutlass.Float32(s_row_scale[row_local])
                            for c in cutlass.range_constexpr(
                                cute.size(acc_dS_blk_mn.shape[1])
                            ):
                                col_idx = cutlass.Int32(tScS_blk_mn[0, c][3])
                                col_local = col_idx - n0
                                ds_scaled_q0 = cutlass.Float32(0.0).to(mU.element_type)
                                if cute.elem_less(
                                    row_idx, cutlass.Int32(self.L)
                                ) and cute.elem_less(col_idx, cutlass.Int32(self.L)):
                                    if (not diag_tile) or cute.elem_less(
                                        col_idx, row_idx + cutlass.Int32(1)
                                    ):
                                        inv_rs = cutlass.Float32(
                                            s_inv_row_scale[col_local]
                                        )
                                        scale = rs * inv_rs
                                        ds_unscaled_f32 = acc_dS_blk_mn[r, c]
                                        ds_scaled_q0 = (ds_unscaled_f32 * scale).to(
                                            mU.element_type
                                        )
                                tCrDS_blk_mn[r, c] = ds_scaled_q0
                        cute.autovec_copy(tCrDS_blk, tCsDS_blk)
                        cute.arch.barrier()

                        cute.copy(
                            smem_tiled_copy_A,
                            tSsDS_blk[None, None, 0],
                            tSrDS_blk_view[None, None, 0],
                        )
                        cute.copy(
                            smem_tiled_copy_BT,
                            tSsK_blk[None, None, 0],
                            tSrK_blk_view[None, None, 0],
                        )
                        for k in cutlass.range_constexpr(cute.size(tSsDS_blk.shape[2])):
                            k_next = (k + 1) % cute.size(tSsDS_blk.shape[2])
                            cute.copy(
                                smem_tiled_copy_A,
                                tSsDS_blk[None, None, k_next],
                                tSrDS_blk_view[None, None, k_next],
                            )
                            cute.copy(
                                smem_tiled_copy_BT,
                                tSsK_blk[None, None, k_next],
                                tSrK_blk_view[None, None, k_next],
                            )
                            cute.gemm(
                                tiled_mma,
                                acc_dQ_total,
                                tSrDS_blk[None, None, k],
                                tSrK_blk[None, None, k],
                                acc_dQ_total,
                            )
                        cute.arch.barrier()

                acc_dQ_off = cute.make_rmem_tensor(acc_shape_tileD_blk, cutlass.Float32)
                acc_dQ_off.fill(0.0)
                tSrDY_off = thr_mma.make_fragment_A(thr_mma.partition_A(sDY))
                tSsDY_off = thr_copy_A.partition_S(sDY)
                tSrDY_off_view = thr_copy_A.retile(tSrDY_off)
                for p_tile_idx in cutlass.range_constexpr(n_p_tiles):
                    p0 = cutlass.Int32(p_tile_idx * p_tile)
                    gDY = cute.local_tile(
                        mDOut[bidz, None, 0, None],
                        (kv_tile, p_tile),
                        (m_tile, p_tile_idx),
                    )
                    tDYg = gmem_thr_copy_P.partition_S(gDY)
                    if cutlass.const_expr(self.P == Pp):
                        cute.copy(gmem_tiled_copy_P, tDYg, tDYs)
                    else:
                        cDY = cute.local_tile(
                            mcDY_full, (kv_tile, p_tile), (m_tile, p_tile_idx)
                        )
                        tDYc = gmem_thr_copy_P.partition_S(cDY)
                        for rest_v in cutlass.range_constexpr(tDYp.shape[0]):
                            for rest_k in cutlass.range_constexpr(tDYp.shape[2]):
                                tDYp[rest_v, 0, rest_k] = cute.elem_less(
                                    tDYc[(0, rest_v), 0, rest_k][3],
                                    mDOut.layout.shape[3],
                                )
                        for vi in cutlass.range_constexpr(cute.size(tDYs.shape[1])):
                            if cute.elem_less(tDYc[0, vi, 0][1], mDOut.layout.shape[1]):
                                cute.copy(
                                    gmem_tiled_copy_P,
                                    tDYg[None, vi, None],
                                    tDYs[None, vi, None],
                                    pred=tDYp[None, vi, None],
                                )
                            else:
                                tDYs[None, vi, None].fill(0)
                    cute.arch.cp_async_commit_group()
                    for it in range(iters_z0_tile):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < total_z0_tile:
                            p_local = idx // cutlass.Int32(d_block)
                            d_local = idx - p_local * cutlass.Int32(d_block)
                            d = d_base + d_local
                            p = p0 + p_local
                            val = cutlass.Float32(0.0).to(mU.element_type)
                            if p < cutlass.Int32(self.P) and d < cutlass.Int32(self.D):
                                if cutlass.const_expr(
                                    mZ0.element_type == cutlass.Float32
                                ):
                                    val = mZ0[bidz, p, d].to(mU.element_type)
                                else:
                                    val = mZ0[bidz, p, d]
                                if (d & 1) == 1:
                                    val = -val
                            sZ0_blk[p_local, d_local] = val
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsDY_off[None, None, 0],
                        tSrDY_off_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_BT,
                        tSsZ0_blk[None, None, 0],
                        tSrZ0_blk_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsDY_off.shape[2])):
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsDY_off[None, None, k],
                            tSrDY_off_view[None, None, k],
                        )
                        cute.copy(
                            smem_tiled_copy_BT,
                            tSsZ0_blk[None, None, k],
                            tSrZ0_blk_view[None, None, k],
                        )
                    for k in cutlass.range_constexpr(cute.size(tSsDY_off.shape[2])):
                        cute.gemm(
                            tiled_mma,
                            acc_dQ_off,
                            tSrDY_off[None, None, k],
                            tSrZ0_blk[None, None, k],
                            acc_dQ_off,
                        )
                    cute.arch.barrier()

                acc_dQ_total_mn = self._make_acc_tensor_mn_view(acc_dQ_total)
                acc_dQ_off_mn = self._make_acc_tensor_mn_view(acc_dQ_off)
                cKD_tile = cute.local_tile(
                    mcKD_full, (kv_tile, d_block), (m_tile, d_tile)
                )
                tOcKD_tile = thr_mma.partition_C(cKD_tile)
                tOcKD_tile_mn = self._make_acc_tensor_mn_view(tOcKD_tile)
                for r in cutlass.range_constexpr(cute.size(acc_dQ_total_mn.shape[0])):
                    row_idx = cutlass.Int32(tOcKD_tile_mn[r, 0][1])
                    row_local = row_idx - m0
                    rs = cutlass.Float32(0.0)
                    if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                        rs = cutlass.Float32(s_row_scale[row_local])
                    for c in cutlass.range(
                        cute.size(acc_dQ_total_mn.shape[1]), unroll_full=True
                    ):
                        off_scaled = acc_dQ_off_mn[r, c] * rs
                        acc_dQ_total_mn[r, c] = acc_dQ_total_mn[r, c] + off_scaled
                        acc_dQ_off_mn[r, c] = off_scaled

                tCsDQ = thr_mma.partition_C(sK_blk)
                tCrDQ = cute.make_fragment_like(tCsDQ, mU.element_type)
                tCrDQ[None] = acc_dQ_total.load().to(mU.element_type)
                cute.autovec_copy(tCrDQ, tCsDQ)
                cute.arch.barrier()

                nvec = cutlass.Int32(d_block // 2)
                row_local = warp
                while cute.elem_less(row_local, cutlass.Int32(kv_tile)):
                    t = m0 + row_local
                    row_dlp_sum = cutlass.Float32(0.0)
                    dR00_sum = cutlass.Float32(0.0)
                    dR01_sum = cutlass.Float32(0.0)
                    dR10_sum = cutlass.Float32(0.0)
                    dR11_sum = cutlass.Float32(0.0)
                    vv = lane
                    while cute.elem_less(vv, nvec):
                        d0_local = vv * cutlass.Int32(2)
                        d0 = d_base + d0_local
                        out0 = cutlass.Float32(0.0).to(mU.element_type)
                        out1 = cutlass.Float32(0.0).to(mU.element_type)
                        if cute.elem_less(t, cutlass.Int32(self.L)) and cute.elem_less(
                            d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                        ):
                            dq0 = cutlass.Float32(
                                sK_blk[row_local, d0_local + 0].to(cutlass.Float32)
                            )
                            dq1 = cutlass.Float32(
                                sK_blk[row_local, d0_local + 1].to(cutlass.Float32)
                            )
                            pr = cutlass.Float32(s_phase_row[row_local, 0])
                            pi = cutlass.Float32(s_phase_row[row_local, 1])
                            dc0, dc1 = conj_mul_phase(dq0, dq1, pr, pi)
                            c0 = cutlass.Float32(mC[bidz, t, 0, d0 + 0])
                            c1 = cutlass.Float32(mC[bidz, t, 0, d0 + 1])
                            row_dlp_sum = row_dlp_sum + dc0 * c0 + dc1 * c1
                            dR00_sum = dR00_sum + dq0 * c0
                            dR01_sum = dR01_sum + dq0 * c1
                            dR10_sum = dR10_sum + dq1 * c0
                            dR11_sum = dR11_sum + dq1 * c1
                            out0 = dc0.to(mU.element_type)
                            out1 = dc1.to(mU.element_type)
                        sK_blk[row_local, d0_local + 0] = out0
                        sK_blk[row_local, d0_local + 1] = out1
                        vv = vv + cutlass.Int32(32)
                    for off in (16, 8, 4, 2, 1):
                        row_dlp_sum = row_dlp_sum + cute.arch.shuffle_sync_bfly(
                            row_dlp_sum, offset=off, mask=-1, mask_and_clamp=31
                        )
                        dR00_sum = dR00_sum + cute.arch.shuffle_sync_bfly(
                            dR00_sum, offset=off, mask=-1, mask_and_clamp=31
                        )
                        dR01_sum = dR01_sum + cute.arch.shuffle_sync_bfly(
                            dR01_sum, offset=off, mask=-1, mask_and_clamp=31
                        )
                        dR10_sum = dR10_sum + cute.arch.shuffle_sync_bfly(
                            dR10_sum, offset=off, mask=-1, mask_and_clamp=31
                        )
                        dR11_sum = dR11_sum + cute.arch.shuffle_sync_bfly(
                            dR11_sum, offset=off, mask=-1, mask_and_clamp=31
                        )
                    if lane == cutlass.Int32(0):
                        s_row_dlp[row_local] = s_row_dlp[row_local] + row_dlp_sum
                        s_row_dR[row_local, 0] = s_row_dR[row_local, 0] + dR00_sum
                        s_row_dR[row_local, 1] = s_row_dR[row_local, 1] + dR01_sum
                        s_row_dR[row_local, 2] = s_row_dR[row_local, 2] + dR10_sum
                        s_row_dR[row_local, 3] = s_row_dR[row_local, 3] + dR11_sum
                    row_local = row_local + cutlass.Int32(num_warps)
                cute.arch.barrier()

                gDC = cute.local_tile(
                    mDC[bidz, None, 0, None], (kv_tile, d_block), (m_tile, d_tile)
                )
                gmem_thr_store_D = gmem_tiled_store_D.get_slice(tidx)
                tDsC = gmem_thr_store_D.partition_S(sK_blk)
                tDgC = gmem_thr_store_D.partition_D(gDC)
                if cutlass.const_expr(self.D == Dp):
                    cute.copy(gmem_tiled_store_D, tDsC, tDgC)
                else:
                    tDrC = cute.make_rmem_tensor_like(tDgC, mU.element_type)
                    cute.copy(gmem_tiled_store_D, tDsC, tDrC)
                    mcDC = cute.make_identity_tensor(mDC.layout.shape)
                    cDC = cute.local_tile(
                        mcDC[bidz, None, 0, None], (kv_tile, d_block), (m_tile, d_tile)
                    )
                    tDcDC = gmem_thr_store_D.partition_D(cDC)
                    tDpDC = cute.make_rmem_tensor(
                        cute.make_layout(
                            (tDgC.shape[0][1], tDgC.shape[1], tDgC.shape[2]),
                            stride=(tDgC.shape[2], 0, 1),
                        ),
                        cutlass.Boolean,
                    )
                    for rest_v in cutlass.range_constexpr(tDpDC.shape[0]):
                        for rest_n in cutlass.range_constexpr(
                            cute.size(tDpDC.shape[2])
                        ):
                            tDpDC[rest_v, 0, rest_n] = cute.elem_less(
                                tDcDC[(0, rest_v), 0, rest_n][3], mDC.layout.shape[3]
                            )
                    for rest_m in cutlass.range_constexpr(cute.size(tDpDC.shape[1])):
                        if cute.elem_less(tDcDC[0, rest_m, 0][1], mDC.layout.shape[1]):
                            cute.copy(
                                gmem_tiled_store_D,
                                tDrC[None, rest_m, None],
                                tDgC[None, rest_m, None],
                                pred=tDpDC[None, rest_m, None],
                            )

            cute.arch.barrier()
            if tidx < cutlass.Int32(kv_tile):
                t = m0 + tidx
                if cute.elem_less(t, cutlass.Int32(self.L)):
                    mDLogp[bidz, t] = cutlass.Float32(2.0) * s_row_dlp[tidx]
                    mDR[bidz, t, 0] = s_row_dR[tidx, 0]
                    mDR[bidz, t, 1] = s_row_dR[tidx, 1]
                    mDR[bidz, t, 2] = s_row_dR[tidx, 2]
                    mDR[bidz, t, 3] = s_row_dR[tidx, 3]


__all__ = ["ChunkScanBwdDCAmpere"]
