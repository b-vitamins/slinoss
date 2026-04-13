"""CuTe backward kernel for the ``v2x2ssd`` chunk-scan ``dZ0`` stage.

``ChunkScanBwdDZ0Ampere`` is the live Ampere tensor-core implementation used by
the backward path to accumulate the public chunk-start state gradient ``dZ0``.
It reconstructs prefix magnitude/phase metadata from the raw packed-complex
transitions ``M``, scales ``dOut`` tile-by-tile, rotates ``C`` by the conjugate
prefix phase, and runs one tensor-core GEMM that writes ``dZ0`` directly.

Tensor contracts:

- ``dOut``: ``(P, T, BH)`` fp16/bf16 packed row/column-major operand
- ``C``: ``(D, T, BG)`` fp16/bf16 packed complex query-side operand
- ``M``: ``(2, T, BH)`` fp32 packed complex transitions
- ``dZ0``: ``(P, D, BHC)`` fp32 public chunk-start state gradient

The trailing ``D`` dimension stores packed complex pairs, so ``D`` must be even
and conceptually corresponds to ``2 * N``.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import (
    LOG2_E,
    TWO_LOG2_E,
    clamp_nonpositive_prefix_log,
    complex_mul,
    mul_conj_phase,
    safe_cast_to_dtype,
)


@dataclass(frozen=True)
class ChunkScanBwdDZ0SupportInfo:
    smem_capacity_bytes: int
    prefix_smem_bytes: int
    operand_smem_bytes: int
    output_smem_bytes: int

    @property
    def required_smem_bytes(self) -> int:
        return self.prefix_smem_bytes + max(
            self.operand_smem_bytes,
            self.output_smem_bytes,
        )

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


@dataclass(frozen=True)
class ChunkScanBwdDZ0LayoutBundle:
    dout_major_mode: object
    c_major_mode: object
    dz0_major_mode: object
    dout_layout: object
    c_layout: object
    output_layout: object
    output_alias_pad_layout: object


@dataclass(frozen=True)
class ChunkScanBwdDZ0CopyBundle:
    gmem_tiled_copy_dout: object
    gmem_tiled_copy_c: object
    gmem_tiled_copy_output: object
    smem_tiled_copy_dout: object
    smem_tiled_copy_c: object


@dataclass(frozen=True)
class ChunkScanBwdDZ0KernelBundle:
    layouts: ChunkScanBwdDZ0LayoutBundle
    copies: ChunkScanBwdDZ0CopyBundle
    tiled_mma: object
    shared_storage_cls: object
    prefix_smem_bytes: int
    operand_smem_bytes: int
    output_smem_bytes: int

    @property
    def smem_size(self) -> int:
        return self.prefix_smem_bytes + max(
            self.operand_smem_bytes,
            self.output_smem_bytes,
        )


class ChunkScanBwdDZ0Ampere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``dZ0`` slice.

    This kernel owns the public chunk-start state gradient. It reconstructs the
    prefix magnitude and phase metadata from raw packed ``M``, scales the
    staged ``dOut`` tiles, rotates ``C`` by the conjugate prefix phase, and
    runs one tensor-core GEMM that writes ``dZ0`` directly.
    """

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkScanBwdDZ0SupportInfo]
    ] = {}

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        cta_tiler: tuple[int, int, int] = (64, 96, 32),  # (bM=P, bN=D, bK=time)
        atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
        num_stages: int = 2,
        heads: int | None = None,
        bc_groups: int | None = None,
    ):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float32

        self.L = int(chunk_size)
        self.cta_tiler = tuple(int(dim) for dim in cta_tiler)
        self.atom_layout_mnk = tuple(int(dim) for dim in atom_layout_mnk)
        self.num_stages = int(num_stages)
        self.has_group_geometry = heads is not None
        if self.has_group_geometry:
            self.heads = int(heads)
            self.bc_groups = self.heads if bc_groups is None else int(bc_groups)
            if self.heads <= 0 or self.bc_groups <= 0:
                raise ValueError("heads and bc_groups must be positive.")
            if self.heads % self.bc_groups != 0:
                raise ValueError("bc_groups must divide heads.")
            self.heads_per_bc_group = self.heads // self.bc_groups
        else:
            if bc_groups is not None:
                raise ValueError("bc_groups requires heads to be specified.")
            self.heads = 0
            self.bc_groups = 0
            self.heads_per_bc_group = 0

        if self.ab_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("dtype must be Float16/BFloat16 for the tensor-core path.")

        self.bM, self.bN, self.bK = self.cta_tiler
        if self.L % self.bK != 0:
            raise ValueError("chunk_size must be divisible by bK for this kernel.")
        if self.bN % 2 != 0:
            raise ValueError("bN (D tile) must be divisible by 2 because D = 2N.")
        if self.num_stages < 2:
            raise ValueError("num_stages must be >= 2.")
        if (self.num_stages - 1) > self.k_tile_count:
            raise ValueError(
                "num_stages too large for chunk_size/bK (insufficient K tiles)."
            )

        self.mma_inst_shape = (16, 8, 16)
        mma_m, mma_n, mma_k = self.mma_inst_shape
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        self.num_threads = atom_m * atom_n * atom_k * 32

        if self.L > self.num_threads:
            raise ValueError("chunk_size too large for this CTA thread count.")
        if self.bM % (atom_m * mma_m) != 0:
            raise ValueError("bM must be divisible by the MMA instruction shape.")
        if self.bN % (atom_n * mma_n * 2) != 0:
            raise ValueError("bN must be divisible by the MMA instruction shape.")
        if atom_k != 1:
            raise ValueError("atom_layout_mnk K must be 1.")
        if self.bK % mma_k != 0:
            raise ValueError("bK must be divisible by the MMA instruction shape.")

    @cute.jit
    def _batch_group(self, batch_head: int):
        if cutlass.const_expr(not self.has_group_geometry):
            return batch_head
        batch_idx = batch_head // cutlass.Int32(self.heads)
        head_idx = batch_head - batch_idx * cutlass.Int32(self.heads)
        group_idx = head_idx // cutlass.Int32(self.heads_per_bc_group)
        return batch_idx * cutlass.Int32(self.bc_groups) + group_idx

    @property
    def k_tile_count(self) -> int:
        return self.L // self.bK

    @property
    def num_warps(self) -> int:
        return self.num_threads // 32

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

    def _dout_smem_elems(self) -> int:
        return self.bM * self.bK * self.num_stages

    def _c_smem_elems(self) -> int:
        return self.bN * self.bK * self.num_stages

    def _output_smem_elems(self) -> int:
        return self.bM * self.bN

    def _prefix_smem_bytes(self) -> int:
        prefix_fields = [
            (self.L * 4, 16),
            (self.L * 4, 16),
            (self.L * 4, 16),
            (self.num_warps * 4, 16),
            (self.num_warps * 4, 16),
            (self.num_warps * 2 * 4, 16),
            (self.num_warps * 2 * 4, 16),
        ]
        return self._struct_size_bytes(prefix_fields)

    def _operand_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        input_bytes = in_dtype.width // 8
        operand_fields = [
            (self._dout_smem_elems() * input_bytes, 16),
            (self._c_smem_elems() * input_bytes, 16),
        ]
        return self._struct_size_bytes(operand_fields)

    def _output_smem_bytes(self) -> int:
        output_bytes = self.c_dtype.width // 8
        return self._struct_size_bytes([(self._output_smem_elems() * output_bytes, 16)])

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

    def support_info(
        self,
        *,
        device_index: int | None = None,
    ) -> ChunkScanBwdDZ0SupportInfo:
        if device_index is None:
            device_key = (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
            )
        else:
            device_key = int(device_index)

        cache_key = (
            type(self),
            self.ab_dtype,
            self.L,
            self.cta_tiler,
            self.atom_layout_mnk,
            self.num_stages,
            device_key,
        )
        cached = self._SUPPORT_INFO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        info = ChunkScanBwdDZ0SupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_index=device_key),
            prefix_smem_bytes=self._prefix_smem_bytes(),
            operand_smem_bytes=self._operand_smem_bytes(self.ab_dtype),
            output_smem_bytes=self._output_smem_bytes(),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def can_implement(self, *, device_index: int | None = None) -> bool:
        return self.support_info(device_index=device_index).supported

    # Host-side bundle builders
    def _prefix_row_layout(self) -> cute.Layout:
        return cute.make_layout((self.L,), stride=(1,))

    def _warp_prefix_row_layout(self) -> cute.Layout:
        return cute.make_layout((self.num_warps,), stride=(1,))

    def _warp_prefix_phase_layout(self) -> cute.Layout:
        return cute.make_layout((self.num_warps, 2), stride=(2, 1))

    def _make_operand_smem_layout(
        self,
        dtype: type[cutlass.Numeric],
        major_mode,
        copy_bits: int,
        smem_tiler: tuple[int, int, int],
    ):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        if major_mode_size >= 64:
            if major_mode_size % 64 == 0:
                major_mode_size = 64
            elif major_mode_size % 32 == 0:
                major_mode_size = 32
            else:
                major_mode_size = 64

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_output_layout(self) -> cute.Layout:
        return cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))

    def _operand_smem_bytes_from_layouts(
        self,
        dout_layout: cute.Layout,
        c_layout: cute.Layout,
    ) -> int:
        operand_fields = [
            (int(cute.size_in_bytes(self.ab_dtype, dout_layout)), 16),
            (int(cute.size_in_bytes(self.ab_dtype, c_layout)), 16),
        ]
        return self._struct_size_bytes(operand_fields)

    def _output_smem_bytes_from_layout(
        self,
        output_layout: cute.Layout,
    ) -> int:
        return self._struct_size_bytes(
            [(int(cute.size_in_bytes(self.c_dtype, output_layout)), 16)]
        )

    def _make_output_alias_pad_layout(
        self,
        dout_layout: cute.Layout,
        c_layout: cute.Layout,
        output_layout: cute.Layout,
    ) -> cute.Layout:
        operand_smem_bytes = self._operand_smem_bytes_from_layouts(
            dout_layout,
            c_layout,
        )
        output_smem_bytes = self._output_smem_bytes_from_layout(output_layout)
        pad_bytes = max(0, output_smem_bytes - operand_smem_bytes)
        return cute.make_layout((((pad_bytes + 3) // 4),), stride=(1,))

    def _make_gmem_tiled_copy_input(
        self,
        atom_copy,
        dtype: type[cutlass.Numeric],
        major_mode,
        copy_bits: int,
        *,
        tile_m: int,
    ):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = self.bK // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1),
            stride=(shape_dim_1, 1),
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = (int(tile_m) + int(copy_elems) - 1) // int(copy_elems)
            if shape_dim_0 > self.num_threads:
                raise ValueError("tile_m too large for vectorized col-major copy.")
            thread_m = None
            for candidate in range(shape_dim_0, self.num_threads + 1):
                if self.num_threads % candidate == 0:
                    thread_m = candidate
                    break
            if thread_m is None:
                raise ValueError(
                    "Internal error: failed to find divisor for col-major copy."
                )
            thread_layout = cute.make_layout(
                (thread_m, self.num_threads // thread_m), stride=(1, thread_m)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_gmem_tiled_copy_output(
        self,
        atom_copy,
        dtype: type[cutlass.Numeric],
        major_mode,
        copy_bits: int,
    ):
        copy_elems = copy_bits // dtype.width
        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            value_layout = cute.make_layout((1, copy_elems))
            best_tm = None
            best_tn = None
            for tm in range(1, self.num_threads + 1):
                if self.num_threads % tm != 0:
                    continue
                tn = self.num_threads // tm
                tile_m = tm
                tile_n = tn * copy_elems
                if (self.bM % tile_m) != 0:
                    continue
                if (self.bN % tile_n) != 0:
                    continue
                if best_tm is None or tile_n > (best_tn * copy_elems):
                    best_tm = tm
                    best_tn = tn
            if best_tm is None:
                shape_dim_1 = self.bN // copy_elems
                thread_layout = cute.make_layout(
                    (self.num_threads // shape_dim_1, shape_dim_1),
                    stride=(shape_dim_1, 1),
                )
            else:
                thread_layout = cute.make_layout(
                    (best_tm, best_tn), stride=(best_tn, 1)
                )
            return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

        value_layout = cute.make_layout((copy_elems, 1))
        shape_dim_0 = (int(self.bM) + int(copy_elems) - 1) // int(copy_elems)
        if shape_dim_0 > self.num_threads:
            raise ValueError("bM too large for vectorized col-major store.")
        thread_m = None
        for candidate in range(shape_dim_0, self.num_threads + 1):
            if self.num_threads % candidate == 0:
                thread_m = candidate
                break
        if thread_m is None:
            raise ValueError(
                "Internal error: failed to find divisor for col-major store."
            )
        thread_layout = cute.make_layout(
            (thread_m, self.num_threads // thread_m), stride=(1, thread_m)
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_layout_bundle(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mDZ0: cute.Tensor,
    ) -> ChunkScanBwdDZ0LayoutBundle:
        copy_bits = 128
        dout_major_mode = utils.LayoutEnum.from_tensor(mDOut)
        c_major_mode = utils.LayoutEnum.from_tensor(mC)
        dz0_major_mode = utils.LayoutEnum.from_tensor(mDZ0)
        dout_layout = self._make_operand_smem_layout(
            self.ab_dtype,
            dout_major_mode,
            copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        c_layout = self._make_operand_smem_layout(
            self.ab_dtype,
            c_major_mode,
            copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        output_layout = self._make_output_layout()
        return ChunkScanBwdDZ0LayoutBundle(
            dout_major_mode=dout_major_mode,
            c_major_mode=c_major_mode,
            dz0_major_mode=dz0_major_mode,
            dout_layout=dout_layout,
            c_layout=c_layout,
            output_layout=output_layout,
            output_alias_pad_layout=self._make_output_alias_pad_layout(
                dout_layout,
                c_layout,
                output_layout,
            ),
        )

    def _make_tiled_mma(self):
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype,
            self.acc_dtype,
            self.mma_inst_shape,
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        return cute.make_tiled_mma(
            op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

    def _make_copy_bundle(
        self,
        layouts: ChunkScanBwdDZ0LayoutBundle,
        tiled_mma: cute.TiledMma,
    ) -> ChunkScanBwdDZ0CopyBundle:
        copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            self.ab_dtype,
            num_bits_per_copy=copy_bits,
        )
        atom_output_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.c_dtype,
            num_bits_per_copy=copy_bits,
        )
        atom_smem_to_reg_dout = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                layouts.dout_major_mode != utils.LayoutEnum.ROW_MAJOR,
                4,
            ),
            self.ab_dtype,
        )
        atom_smem_to_reg_c = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                layouts.c_major_mode != utils.LayoutEnum.ROW_MAJOR,
                4,
            ),
            self.ab_dtype,
        )
        return ChunkScanBwdDZ0CopyBundle(
            gmem_tiled_copy_dout=self._make_gmem_tiled_copy_input(
                atom_async_copy,
                self.ab_dtype,
                layouts.dout_major_mode,
                copy_bits,
                tile_m=self.bM,
            ),
            gmem_tiled_copy_c=self._make_gmem_tiled_copy_input(
                atom_async_copy,
                self.ab_dtype,
                layouts.c_major_mode,
                copy_bits,
                tile_m=self.bN,
            ),
            gmem_tiled_copy_output=self._make_gmem_tiled_copy_output(
                atom_output_copy,
                self.c_dtype,
                layouts.dz0_major_mode,
                copy_bits,
            ),
            smem_tiled_copy_dout=cute.make_tiled_copy_A(
                atom_smem_to_reg_dout,
                tiled_mma,
            ),
            smem_tiled_copy_c=cute.make_tiled_copy_B(atom_smem_to_reg_c, tiled_mma),
        )

    def _make_shared_storage(
        self,
        layouts: ChunkScanBwdDZ0LayoutBundle,
    ):
        @cute.struct
        class SharedStorage:
            prefix_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.L],
                16,
            ]
            prefix_phase_re: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.L],
                16,
            ]
            prefix_phase_im: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.L],
                16,
            ]
            warp_prefix_log_total: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps],
                16,
            ]
            warp_prefix_log_offset: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps],
                16,
            ]
            warp_prefix_phase_total: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps * 2],
                16,
            ]
            warp_prefix_phase_offset: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps * 2],
                16,
            ]
            dout_tile: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(layouts.dout_layout)],
                16,
            ]
            c_tile: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(layouts.c_layout)],
                16,
            ]
            # Extend the staged operand slab when the aliased fp32 output tile
            # is larger than the staged fp16/bf16 input footprint.
            output_alias_pad: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(layouts.output_alias_pad_layout),
                ],
                16,
            ]

        return SharedStorage

    def _make_kernel_bundle(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mDZ0: cute.Tensor,
    ) -> ChunkScanBwdDZ0KernelBundle:
        layouts = self._make_layout_bundle(mDOut, mC, mDZ0)
        tiled_mma = self._make_tiled_mma()
        copies = self._make_copy_bundle(layouts, tiled_mma)
        shared_storage_cls = self._make_shared_storage(layouts)
        bundle = ChunkScanBwdDZ0KernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=tiled_mma,
            shared_storage_cls=shared_storage_cls,
            prefix_smem_bytes=self._prefix_smem_bytes(),
            operand_smem_bytes=self._operand_smem_bytes_from_layouts(
                layouts.dout_layout,
                layouts.c_layout,
            ),
            output_smem_bytes=self._output_smem_bytes_from_layout(
                layouts.output_layout
            ),
        )
        actual_smem_bytes = int(shared_storage_cls.size_in_bytes())
        if actual_smem_bytes != bundle.smem_size:
            raise ValueError(
                "Shared storage accounting drifted from the declared layout bundle: "
                f"{actual_smem_bytes}B actual vs {bundle.smem_size}B expected."
            )
        return bundle

    # Kernel state builders
    def _decode_batch_head_chunk(
        self,
        mDOut: cute.Tensor,
        mDZ0: cute.Tensor,
        batch_head_chunk: int,
    ) -> SimpleNamespace:
        batch_head_count = mDOut.shape[2]
        batch_head_chunk_count = mDZ0.shape[2]
        n_chunks = batch_head_chunk_count // batch_head_count
        batch_head = batch_head_chunk // n_chunks
        return SimpleNamespace(
            batch_head=batch_head,
            batch_group=self._batch_group(batch_head),
            chunk_start=(batch_head_chunk - batch_head * n_chunks) * self.L,
        )

    def _make_output_gmem_tile(
        self,
        mDZ0: cute.Tensor,
        batch_head_chunk: int,
        block_m: int,
        block_n: int,
    ):
        return cute.local_tile(
            mDZ0[None, None, batch_head_chunk],
            tiler=self.cta_tiler,
            coord=(block_m, block_n, None),
            proj=(1, 1, None),
        )

    def _make_output_shared_tensor(
        self,
        s_dout: cute.Tensor,
        output_layout: cute.Layout,
    ):
        return cute.make_tensor(
            cute.recast_ptr(s_dout.iterator, dtype=self.c_dtype),
            output_layout,
        )

    def _make_shared_tensor_bundle(
        self,
        storage,
        dout_layout: cute.ComposedLayout,
        c_layout: cute.ComposedLayout,
        output_layout: cute.Layout,
    ) -> SimpleNamespace:
        s_dout = storage.dout_tile.get_tensor(dout_layout)
        return SimpleNamespace(
            prefix_scale=storage.prefix_scale.get_tensor(self._prefix_row_layout()),
            prefix_phase_re=storage.prefix_phase_re.get_tensor(
                self._prefix_row_layout()
            ),
            prefix_phase_im=storage.prefix_phase_im.get_tensor(
                self._prefix_row_layout()
            ),
            warp_prefix_log_total=storage.warp_prefix_log_total.get_tensor(
                self._warp_prefix_row_layout()
            ),
            warp_prefix_log_offset=storage.warp_prefix_log_offset.get_tensor(
                self._warp_prefix_row_layout()
            ),
            warp_prefix_phase_total=storage.warp_prefix_phase_total.get_tensor(
                self._warp_prefix_phase_layout()
            ),
            warp_prefix_phase_offset=storage.warp_prefix_phase_offset.get_tensor(
                self._warp_prefix_phase_layout()
            ),
            s_dout=s_dout,
            s_c=storage.c_tile.get_tensor(c_layout),
            s_output=self._make_output_shared_tensor(s_dout, output_layout),
        )

    def _make_input_coord_tiles(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        tile_info: SimpleNamespace,
        block_m: int,
        block_n: int,
    ) -> SimpleNamespace:
        coord_dout_base = cute.domain_offset(
            (0, tile_info.chunk_start, 0),
            cute.make_identity_tensor(mDOut.layout.shape),
        )
        coord_c_base = cute.domain_offset(
            (0, tile_info.chunk_start, 0),
            cute.make_identity_tensor(mC.layout.shape),
        )
        return SimpleNamespace(
            dout_coord=cute.local_tile(
                coord_dout_base[None, None, tile_info.batch_head],
                tiler=self.cta_tiler,
                coord=(block_m, block_n, None),
                proj=(1, None, 1),
            ),
            c_coord=cute.local_tile(
                coord_c_base[None, None, tile_info.batch_group],
                tiler=self.cta_tiler,
                coord=(block_m, block_n, None),
                proj=(None, 1, 1),
            ),
        )

    def _make_input_gmem_tiles(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        tile_info: SimpleNamespace,
        block_m: int,
        block_n: int,
    ) -> SimpleNamespace:
        g_dout_base = cute.domain_offset((0, tile_info.chunk_start, 0), mDOut)
        g_c_base = cute.domain_offset((0, tile_info.chunk_start, 0), mC)
        g_dout = cute.local_tile(
            g_dout_base[None, None, tile_info.batch_head],
            tiler=self.cta_tiler,
            coord=(block_m, block_n, None),
            proj=(1, None, 1),
        )
        g_c = cute.local_tile(
            g_c_base[None, None, tile_info.batch_group],
            tiler=self.cta_tiler,
            coord=(block_m, block_n, None),
            proj=(None, 1, 1),
        )
        return SimpleNamespace(
            g_dout=cute.make_tensor(g_dout.iterator.align(16), g_dout.layout),
            g_c=cute.make_tensor(g_c.iterator.align(16), g_c.layout),
        )

    def _make_input_stage_state(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        tile_info: SimpleNamespace,
        shared_tensors: SimpleNamespace,
        gmem_tiled_copy_dout: cute.TiledCopy,
        gmem_tiled_copy_c: cute.TiledCopy,
        *,
        tidx: int,
        block_m: int,
        block_n: int,
    ) -> SimpleNamespace:
        gmem_thr_copy_dout = gmem_tiled_copy_dout.get_slice(tidx)
        gmem_thr_copy_c = gmem_tiled_copy_c.get_slice(tidx)
        t_dout_smem = gmem_thr_copy_dout.partition_D(shared_tensors.s_dout)
        t_c_smem = gmem_thr_copy_c.partition_D(shared_tensors.s_c)

        coord_tiles = self._make_input_coord_tiles(
            mDOut,
            mC,
            tile_info,
            block_m,
            block_n,
        )
        gmem_tiles = self._make_input_gmem_tiles(
            mDOut,
            mC,
            tile_info,
            block_m,
            block_n,
        )

        return SimpleNamespace(
            gmem_tiled_copy_dout=gmem_tiled_copy_dout,
            gmem_tiled_copy_c=gmem_tiled_copy_c,
            s_dout=shared_tensors.s_dout,
            s_c=shared_tensors.s_c,
            t_dout_smem=t_dout_smem,
            t_c_smem=t_c_smem,
            t_dout_gmem=gmem_thr_copy_dout.partition_S(gmem_tiles.g_dout),
            t_c_gmem=gmem_thr_copy_c.partition_S(gmem_tiles.g_c),
            t_dout_pred=self._make_input_copy_row_predicate(
                t_dout_smem,
                gmem_thr_copy_dout.partition_S(coord_tiles.dout_coord),
                mDOut.shape[0],
            ),
            t_c_pred=self._make_input_copy_row_predicate(
                t_c_smem,
                gmem_thr_copy_c.partition_S(coord_tiles.c_coord),
                mC.shape[0],
            ),
        )

    def _make_mma_state(
        self,
        tiled_mma: cute.TiledMma,
        smem_tiled_copy_dout: cute.TiledCopy,
        smem_tiled_copy_c: cute.TiledCopy,
        shared_tensors: SimpleNamespace,
        g_output: cute.Tensor,
        *,
        tidx: int,
    ) -> SimpleNamespace:
        thr_mma = tiled_mma.get_slice(tidx)
        t_smem_dout_mma = thr_mma.partition_A(shared_tensors.s_dout)
        t_smem_c_mma = thr_mma.partition_B(shared_tensors.s_c)
        t_output_smem_mma = thr_mma.partition_C(shared_tensors.s_output)
        t_output_gmem_mma = thr_mma.partition_C(g_output)

        t_reg_dout = tiled_mma.make_fragment_A(t_smem_dout_mma[None, None, None, 0])
        t_reg_c = tiled_mma.make_fragment_B(t_smem_c_mma[None, None, None, 0])
        acc_output = tiled_mma.make_fragment_C(t_output_gmem_mma)
        acc_output.fill(0.0)

        smem_thr_copy_dout = smem_tiled_copy_dout.get_slice(tidx)
        smem_thr_copy_c = smem_tiled_copy_c.get_slice(tidx)
        return SimpleNamespace(
            tiled_mma=tiled_mma,
            smem_tiled_copy_dout=smem_tiled_copy_dout,
            smem_tiled_copy_c=smem_tiled_copy_c,
            t_output_smem_mma=t_output_smem_mma,
            acc_output=acc_output,
            t_smem_dout_copy=smem_thr_copy_dout.partition_S(shared_tensors.s_dout),
            t_reg_dout_copy=smem_thr_copy_dout.retile(t_reg_dout),
            t_smem_c_copy=smem_thr_copy_c.partition_S(shared_tensors.s_c),
            t_reg_c_copy=smem_thr_copy_c.retile(t_reg_c),
            t_reg_dout=t_reg_dout,
            t_reg_c=t_reg_c,
        )

    def _make_prefix_state(
        self,
        mM: cute.Tensor,
        tile_info: SimpleNamespace,
        shared_tensors: SimpleNamespace,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            batch_head=tile_info.batch_head,
            chunk_start=tile_info.chunk_start,
            m_transition=mM,
            prefix_scale=shared_tensors.prefix_scale,
            prefix_phase_re=shared_tensors.prefix_phase_re,
            prefix_phase_im=shared_tensors.prefix_phase_im,
            warp_prefix_log_total=shared_tensors.warp_prefix_log_total,
            warp_prefix_log_offset=shared_tensors.warp_prefix_log_offset,
            warp_prefix_phase_total=shared_tensors.warp_prefix_phase_total,
            warp_prefix_phase_offset=shared_tensors.warp_prefix_phase_offset,
        )

    def _make_output_coord_tile(
        self,
        mDZ0: cute.Tensor,
        batch_head_chunk: int,
        block_m: int,
        block_n: int,
    ):
        coord_output = cute.make_identity_tensor(mDZ0.layout.shape)
        return cute.local_tile(
            coord_output[None, None, batch_head_chunk],
            tiler=self.cta_tiler,
            coord=(block_m, block_n, None),
            proj=(1, 1, None),
        )

    def _make_output_state(
        self,
        mDZ0: cute.Tensor,
        g_output: cute.Tensor,
        shared_tensors: SimpleNamespace,
        mma_state: SimpleNamespace,
        gmem_tiled_copy_output: cute.TiledCopy,
        *,
        tidx: int,
        batch_head_chunk: int,
        block_m: int,
        block_n: int,
    ) -> SimpleNamespace:
        gmem_thr_copy_output = gmem_tiled_copy_output.get_slice(tidx)
        t_output_smem_epilogue = gmem_thr_copy_output.partition_S(
            shared_tensors.s_output
        )
        t_output_gmem_epilogue = gmem_thr_copy_output.partition_D(g_output)
        output_coord_tile = self._make_output_coord_tile(
            mDZ0,
            batch_head_chunk,
            block_m,
            block_n,
        )
        t_output_coord = gmem_thr_copy_output.partition_S(output_coord_tile)
        return SimpleNamespace(
            gmem_tiled_copy_output=gmem_tiled_copy_output,
            t_output_smem_mma=mma_state.t_output_smem_mma,
            t_output_smem_epilogue=t_output_smem_epilogue,
            t_output_gmem_epilogue=t_output_gmem_epilogue,
            t_output_pred=self._make_output_copy_predicate(
                t_output_gmem_epilogue,
                t_output_coord,
                row_limit=mDZ0.shape[0],
                col_limit=mDZ0.shape[1],
            ),
        )

    # Device copy/MMA helpers
    @cute.jit
    def _accumulate_from_staged_tiles(
        self,
        tiled_mma: cute.TiledMma,
        acc_output: cute.Tensor,
        smem_tiled_copy_dout: cute.TiledCopy,
        smem_tiled_copy_c: cute.TiledCopy,
        t_smem_dout: cute.Tensor,
        t_reg_dout_copy: cute.Tensor,
        t_smem_c: cute.Tensor,
        t_reg_c_copy: cute.Tensor,
        t_reg_dout: cute.Tensor,
        t_reg_c: cute.Tensor,
    ):
        cute.copy(
            smem_tiled_copy_dout,
            t_smem_dout[None, None, 0],
            t_reg_dout_copy[None, None, 0],
        )
        cute.copy(
            smem_tiled_copy_c,
            t_smem_c[None, None, 0],
            t_reg_c_copy[None, None, 0],
        )
        for k_block in cutlass.range_constexpr(cute.size(t_smem_dout.shape[2])):
            k_block_next = (k_block + 1) % cute.size(t_smem_dout.shape[2])
            cute.copy(
                smem_tiled_copy_dout,
                t_smem_dout[None, None, k_block_next],
                t_reg_dout_copy[None, None, k_block_next],
            )
            cute.copy(
                smem_tiled_copy_c,
                t_smem_c[None, None, k_block_next],
                t_reg_c_copy[None, None, k_block_next],
            )
            cute.gemm(
                tiled_mma,
                acc_output,
                t_reg_dout[None, None, k_block],
                t_reg_c[None, None, k_block],
                acc_output,
            )
        cute.arch.barrier()

    @cute.jit
    def _make_input_copy_row_predicate(
        self,
        partitioned_tensor: cute.Tensor,
        partitioned_coord: cute.Tensor,
        row_limit: int,
    ):
        pred = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    partitioned_tensor.shape[0][1],
                    cute.size(partitioned_tensor, mode=[1]),
                    cute.size(partitioned_tensor, mode=[2]),
                ),
                stride=(cute.size(partitioned_tensor, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(pred.shape[0]):
            for row in range(pred.shape[1]):
                pred[rest_v, row, 0] = cute.elem_less(
                    partitioned_coord[(0, rest_v), row, 0, 0][0],
                    row_limit,
                )
        return pred

    @cute.jit
    def _copy_input_stage_tile(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        t_copy_src: cute.Tensor,
        t_copy_dst: cute.Tensor,
        copy_pred: cute.Tensor,
        *,
        tile_index: int,
        smem_stage: int,
    ):
        cute.copy(
            gmem_tiled_copy,
            t_copy_src[None, None, None, tile_index],
            t_copy_dst[None, None, None, smem_stage],
            pred=copy_pred,
        )

    @cute.jit
    def _zero_staged_input_tiles(self, input_state: SimpleNamespace):
        input_state.t_dout_smem.fill(0)
        input_state.t_c_smem.fill(0)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_input_tile_pair(
        self,
        input_state: SimpleNamespace,
        *,
        tile_index: int,
        smem_stage: int,
    ):
        self._copy_input_stage_tile(
            input_state.gmem_tiled_copy_dout,
            input_state.t_dout_gmem,
            input_state.t_dout_smem,
            input_state.t_dout_pred,
            tile_index=tile_index,
            smem_stage=smem_stage,
        )
        self._copy_input_stage_tile(
            input_state.gmem_tiled_copy_c,
            input_state.t_c_gmem,
            input_state.t_c_smem,
            input_state.t_c_pred,
            tile_index=tile_index,
            smem_stage=smem_stage,
        )

    @cute.jit
    def _prefetch_initial_input_tiles(self, input_state: SimpleNamespace):
        next_k_tile = cutlass.Int32(0)
        for smem_stage in cutlass.range_constexpr(self.num_stages - 1):
            self._prefetch_input_tile_pair(
                input_state,
                tile_index=next_k_tile,
                smem_stage=smem_stage,
            )
            next_k_tile = next_k_tile + 1
            cute.arch.cp_async_commit_group()
        return next_k_tile

    # Prefix helpers
    @cute.jit
    def _load_local_prefix_transition(
        self,
        prefix_state: SimpleNamespace,
        tidx,
    ):
        logp = cutlass.Float32(0.0)
        phase_re = cutlass.Float32(1.0)
        phase_im = cutlass.Float32(0.0)
        if tidx < self.L:
            time_idx = prefix_state.chunk_start + tidx
            mr = cutlass.Float32(
                prefix_state.m_transition[0, time_idx, prefix_state.batch_head]
            )
            mi = cutlass.Float32(
                prefix_state.m_transition[1, time_idx, prefix_state.batch_head]
            )
            mag2 = mr * mr + mi * mi + cutlass.Float32(1.0e-20)
            inv_mag = cutlass.Float32(cute.math.rsqrt(mag2))
            phase_re = mr * inv_mag
            phase_im = mi * inv_mag
            logp = cute.math.log2(mag2, fastmath=False) * cutlass.Float32(0.25 / LOG2_E)
        return logp, phase_re, phase_im

    @cute.jit
    def _scan_prefix_within_warp(self, logp, phase_re, phase_im, lane):
        for offset in (1, 2, 4, 8, 16):
            other_log = cute.arch.shuffle_sync_up(
                logp,
                offset=offset,
                mask=-1,
                mask_and_clamp=0,
            )
            other_phase_re = cute.arch.shuffle_sync_up(
                phase_re,
                offset=offset,
                mask=-1,
                mask_and_clamp=0,
            )
            other_phase_im = cute.arch.shuffle_sync_up(
                phase_im,
                offset=offset,
                mask=-1,
                mask_and_clamp=0,
            )
            has_prefix = lane >= cutlass.Int32(offset)
            logp = cutlass.select_(has_prefix, logp + other_log, logp)
            next_phase_re, next_phase_im = complex_mul(
                phase_re,
                phase_im,
                other_phase_re,
                other_phase_im,
            )
            phase_re = cutlass.select_(has_prefix, next_phase_re, phase_re)
            phase_im = cutlass.select_(has_prefix, next_phase_im, phase_im)
        return logp, phase_re, phase_im

    @cute.jit
    def _populate_warp_prefix_offsets(
        self,
        prefix_state: SimpleNamespace,
        warp,
        lane,
        logp,
        phase_re,
        phase_im,
    ):
        if lane == cutlass.Int32(31):
            prefix_state.warp_prefix_log_total[warp] = logp
            prefix_state.warp_prefix_phase_total[warp, 0] = phase_re
            prefix_state.warp_prefix_phase_total[warp, 1] = phase_im
        cute.arch.barrier()

        if warp == cutlass.Int32(0) and lane == cutlass.Int32(0):
            running_log = cutlass.Float32(0.0)
            running_phase_re = cutlass.Float32(1.0)
            running_phase_im = cutlass.Float32(0.0)
            for warp_idx in cutlass.range_constexpr(self.num_warps):
                prefix_state.warp_prefix_log_offset[warp_idx] = running_log
                prefix_state.warp_prefix_phase_offset[warp_idx, 0] = running_phase_re
                prefix_state.warp_prefix_phase_offset[warp_idx, 1] = running_phase_im

                total_log = prefix_state.warp_prefix_log_total[warp_idx]
                total_phase_re = prefix_state.warp_prefix_phase_total[warp_idx, 0]
                total_phase_im = prefix_state.warp_prefix_phase_total[warp_idx, 1]
                next_running_phase_re = (
                    running_phase_re * total_phase_re
                    - running_phase_im * total_phase_im
                )
                next_running_phase_im = (
                    running_phase_re * total_phase_im
                    + running_phase_im * total_phase_re
                )
                running_log = running_log + total_log
                running_phase_re = next_running_phase_re
                running_phase_im = next_running_phase_im
        cute.arch.barrier()

    @cute.jit
    def _apply_warp_prefix_offset(
        self,
        prefix_state: SimpleNamespace,
        warp,
        logp,
        phase_re,
        phase_im,
    ):
        warp_log_offset = prefix_state.warp_prefix_log_offset[warp]
        warp_phase_re_offset = prefix_state.warp_prefix_phase_offset[warp, 0]
        warp_phase_im_offset = prefix_state.warp_prefix_phase_offset[warp, 1]
        logp = logp + warp_log_offset
        phase_re, phase_im = complex_mul(
            phase_re,
            phase_im,
            warp_phase_re_offset,
            warp_phase_im_offset,
        )

        phase_norm2 = phase_re * phase_re + phase_im * phase_im
        phase_inv = cutlass.Float32(
            cute.math.rsqrt(phase_norm2 + cutlass.Float32(1.0e-20))
        )
        phase_re = phase_re * phase_inv
        phase_im = phase_im * phase_inv
        return logp, phase_re, phase_im

    @cute.jit
    def _store_prefix_metadata(
        self,
        prefix_state: SimpleNamespace,
        tidx,
        logp,
        phase_re,
        phase_im,
    ):
        if tidx < self.L:
            stable_logp = clamp_nonpositive_prefix_log(logp)
            prefix_state.prefix_scale[tidx] = cute.math.exp2(
                stable_logp * cutlass.Float32(TWO_LOG2_E),
                fastmath=True,
            )
            prefix_state.prefix_phase_re[tidx] = phase_re
            prefix_state.prefix_phase_im[tidx] = phase_im
        cute.arch.barrier()

    @cute.jit
    def _compute_phase_prefix_metadata(self, prefix_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        logp, phase_re, phase_im = self._load_local_prefix_transition(
            prefix_state, tidx
        )
        logp, phase_re, phase_im = self._scan_prefix_within_warp(
            logp,
            phase_re,
            phase_im,
            lane,
        )
        self._populate_warp_prefix_offsets(
            prefix_state,
            warp,
            lane,
            logp,
            phase_re,
            phase_im,
        )
        logp, phase_re, phase_im = self._apply_warp_prefix_offset(
            prefix_state,
            warp,
            logp,
            phase_re,
            phase_im,
        )
        self._store_prefix_metadata(
            prefix_state,
            tidx,
            logp,
            phase_re,
            phase_im,
        )

    # Mainloop helpers
    @cute.jit
    def _scale_staged_dout_tile_from_prefix(
        self,
        s_dout: cute.Tensor,
        prefix_scale: cute.Tensor,
        *,
        k_tile_offset: int,
        smem_pipe_read: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        idx = tidx
        while idx < cutlass.Int32(self.bM * self.bK):
            row_idx = idx // self.bK
            k_idx = idx - row_idx * self.bK
            time_idx = k_tile_offset + k_idx
            value = cutlass.Float32(s_dout[row_idx, k_idx, smem_pipe_read])
            value = value * cutlass.Float32(prefix_scale[time_idx])
            s_dout[row_idx, k_idx, smem_pipe_read] = safe_cast_to_dtype(
                value, out_dtype
            )
            idx = idx + self.num_threads

    @cute.jit
    def _rotate_staged_c_tile_from_prefix(
        self,
        s_c: cute.Tensor,
        prefix_phase_re: cute.Tensor,
        prefix_phase_im: cute.Tensor,
        *,
        k_tile_offset: int,
        smem_pipe_read: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        idx = tidx
        while idx < cutlass.Int32(self.bK * (self.bN // 2)):
            k_idx = idx // (self.bN // 2)
            vec_idx = idx - k_idx * (self.bN // 2)
            time_idx = k_tile_offset + k_idx
            d_col = vec_idx * 2
            xr = cutlass.Float32(s_c[d_col + 0, k_idx, smem_pipe_read])
            xi = cutlass.Float32(s_c[d_col + 1, k_idx, smem_pipe_read])
            rr, ri = mul_conj_phase(
                xr,
                xi,
                cutlass.Float32(prefix_phase_re[time_idx]),
                cutlass.Float32(prefix_phase_im[time_idx]),
            )
            s_c[d_col + 0, k_idx, smem_pipe_read] = safe_cast_to_dtype(rr, out_dtype)
            s_c[d_col + 1, k_idx, smem_pipe_read] = safe_cast_to_dtype(ri, out_dtype)
            idx = idx + self.num_threads

    @cute.jit
    def _run_mainloop(
        self,
        input_state: SimpleNamespace,
        mma_state: SimpleNamespace,
        prefix_state: SimpleNamespace,
        *,
        dout_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
    ):
        next_k_tile = self._prefetch_initial_input_tiles(input_state)
        for k_tile in cutlass.range_constexpr(self.k_tile_count):
            smem_pipe_read = k_tile % self.num_stages
            smem_pipe_write = (k_tile + (self.num_stages - 1)) % self.num_stages

            cute.arch.cp_async_wait_group(self.num_stages - 2)
            cute.arch.sync_threads()

            k_tile_offset = k_tile * self.bK
            self._scale_staged_dout_tile_from_prefix(
                input_state.s_dout,
                prefix_state.prefix_scale,
                k_tile_offset=k_tile_offset,
                smem_pipe_read=smem_pipe_read,
                out_dtype=dout_dtype,
            )
            self._rotate_staged_c_tile_from_prefix(
                input_state.s_c,
                prefix_state.prefix_phase_re,
                prefix_state.prefix_phase_im,
                k_tile_offset=k_tile_offset,
                smem_pipe_read=smem_pipe_read,
                out_dtype=c_dtype,
            )
            cute.arch.sync_threads()

            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                mma_state.acc_output,
                mma_state.smem_tiled_copy_dout,
                mma_state.smem_tiled_copy_c,
                mma_state.t_smem_dout_copy[None, None, None, smem_pipe_read],
                mma_state.t_reg_dout_copy,
                mma_state.t_smem_c_copy[None, None, None, smem_pipe_read],
                mma_state.t_reg_c_copy,
                mma_state.t_reg_dout,
                mma_state.t_reg_c,
            )

            if k_tile + (self.num_stages - 1) < self.k_tile_count:
                self._prefetch_input_tile_pair(
                    input_state,
                    tile_index=next_k_tile,
                    smem_stage=smem_pipe_write,
                )
                next_k_tile = next_k_tile + 1
                cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    # Epilogue helpers
    @cute.jit
    def _make_output_copy_predicate(
        self,
        t_output_gmem: cute.Tensor,
        t_output_coord: cute.Tensor,
        *,
        row_limit: int,
        col_limit: int,
    ):
        pred = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    t_output_gmem.shape[0][1],
                    cute.size(t_output_gmem, mode=[1]),
                    cute.size(t_output_gmem, mode=[2]),
                ),
                stride=(
                    cute.size(t_output_gmem, mode=[1])
                    * cute.size(t_output_gmem, mode=[2]),
                    cute.size(t_output_gmem, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(pred.shape[0]):
            for col in range(pred.shape[2]):
                col_ok = cute.elem_less(
                    t_output_coord[(0, rest_v), 0, col][1],
                    col_limit,
                )
                for row in range(pred.shape[1]):
                    row_ok = cute.elem_less(
                        t_output_coord[(0, rest_v), row, 0][0],
                        row_limit,
                    )
                    pred[rest_v, row, col] = col_ok & row_ok
        return pred

    @cute.jit
    def _store_output(
        self,
        acc_output: cute.Tensor,
        t_output_smem_mma: cute.Tensor,
        t_output_smem_epilogue: cute.Tensor,
        t_output_gmem_epilogue: cute.Tensor,
        t_output_pred: cute.Tensor,
        gmem_tiled_copy_output: cute.TiledCopy,
    ):
        cute.autovec_copy(acc_output, t_output_smem_mma)
        cute.arch.sync_threads()

        r_output = cute.make_rmem_tensor_like(t_output_smem_epilogue, self.c_dtype)
        cute.autovec_copy(t_output_smem_epilogue, r_output)
        for col in range(t_output_pred.shape[2]):
            cute.copy(
                gmem_tiled_copy_output,
                r_output[None, None, col],
                t_output_gmem_epilogue[None, None, col],
                pred=t_output_pred[None, None, col],
            )

    # Host launch API
    @cute.jit(preprocess=True)
    def _validate_main_operands(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mDZ0: cute.Tensor,
    ):
        if cutlass.const_expr(
            mDOut.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("dOut/C must be Float16/BFloat16 for the tensor-core path.")
        if cutlass.const_expr(mDOut.element_type != mC.element_type):
            raise TypeError("dOut and C must share element type.")
        if cutlass.const_expr(mDOut.element_type != self.ab_dtype):
            raise TypeError("dOut/C element type must match the kernel input dtype.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mDZ0.element_type != cutlass.Float32):
            raise TypeError("dZ0 must be Float32.")

    def _launch_main_kernel(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mDZ0: cute.Tensor,
        *,
        stream=None,
    ):
        bundle = self._make_kernel_bundle(mDOut, mC, mDZ0)
        grid_dim = cute.ceil_div(mDZ0.shape, (self.bM, self.bN, 1))
        launch_kwargs = {
            "grid": (
                cute.size(grid_dim[0]),
                cute.size(grid_dim[1]),
                cute.size(mDZ0.shape[2]),
            ),
            "block": [self.num_threads, 1, 1],
            "smem": bundle.smem_size,
        }
        if stream is not None:
            launch_kwargs["stream"] = stream
        self.kernel(
            mDOut,
            mC,
            mM,
            mDZ0,
            bundle.layouts.dout_layout,
            bundle.layouts.c_layout,
            bundle.layouts.output_layout,
            bundle.copies.gmem_tiled_copy_dout,
            bundle.copies.gmem_tiled_copy_c,
            bundle.copies.gmem_tiled_copy_output,
            bundle.copies.smem_tiled_copy_dout,
            bundle.copies.smem_tiled_copy_c,
            bundle.tiled_mma,
            bundle.shared_storage_cls,
        ).launch(**launch_kwargs)

    @cute.jit(preprocess=True)
    def __call__(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mDZ0: cute.Tensor,
    ):
        self._validate_main_operands(mDOut, mC, mM, mDZ0)
        self._launch_main_kernel(mDOut, mC, mM, mDZ0)

    @cute.jit(preprocess=True)
    def call_on_stream(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mDZ0: cute.Tensor,
        stream,
    ):
        self._validate_main_operands(mDOut, mC, mM, mDZ0)
        self._launch_main_kernel(mDOut, mC, mM, mDZ0, stream=stream)

    # Kernel entrypoint
    @cute.kernel(preprocess=True)
    def kernel(
        self,
        mDOut: cute.Tensor,  # (P, T, BH)
        mC: cute.Tensor,  # (D, T, BG)
        mM: cute.Tensor,  # (2, T, BH)
        mDZ0: cute.Tensor,  # (P, D, BHC)
        dout_layout: cute.ComposedLayout,
        c_layout: cute.ComposedLayout,
        output_layout: cute.Layout,
        gmem_tiled_copy_dout: cute.TiledCopy,
        gmem_tiled_copy_c: cute.TiledCopy,
        gmem_tiled_copy_output: cute.TiledCopy,
        smem_tiled_copy_dout: cute.TiledCopy,
        smem_tiled_copy_c: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_m, block_n, batch_head_chunk = cute.arch.block_idx()

        tile_info = self._decode_batch_head_chunk(mDOut, mDZ0, batch_head_chunk)
        g_output = self._make_output_gmem_tile(
            mDZ0,
            batch_head_chunk,
            block_m,
            block_n,
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        shared_tensors = self._make_shared_tensor_bundle(
            storage,
            dout_layout,
            c_layout,
            output_layout,
        )

        input_state = self._make_input_stage_state(
            mDOut,
            mC,
            tile_info,
            shared_tensors,
            gmem_tiled_copy_dout,
            gmem_tiled_copy_c,
            tidx=tidx,
            block_m=block_m,
            block_n=block_n,
        )
        mma_state = self._make_mma_state(
            tiled_mma,
            smem_tiled_copy_dout,
            smem_tiled_copy_c,
            shared_tensors,
            g_output,
            tidx=tidx,
        )
        prefix_state = self._make_prefix_state(
            mM,
            tile_info,
            shared_tensors,
        )
        self._compute_phase_prefix_metadata(prefix_state)

        self._zero_staged_input_tiles(input_state)
        self._run_mainloop(
            input_state,
            mma_state,
            prefix_state,
            dout_dtype=mDOut.element_type,
            c_dtype=mC.element_type,
        )

        output_state = self._make_output_state(
            mDZ0,
            g_output,
            shared_tensors,
            mma_state,
            gmem_tiled_copy_output,
            tidx=tidx,
            batch_head_chunk=batch_head_chunk,
            block_m=block_m,
            block_n=block_n,
        )
        self._store_output(
            mma_state.acc_output,
            output_state.t_output_smem_mma,
            output_state.t_output_smem_epilogue,
            output_state.t_output_gmem_epilogue,
            output_state.t_output_pred,
            output_state.gmem_tiled_copy_output,
        )


__all__ = ["ChunkScanBwdDZ0Ampere"]
