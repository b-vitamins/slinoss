"""CuTe forward kernel for the ``v2x2ssd`` chunk-increment stage.

Logical tensors
---------------
- ``U``: ``(BHC, L, P)``
- ``M``: ``(BHC, L, 2)``
- ``K``: ``(BHC, L, 2, 2)``
- ``B``: ``(BHC, L, D)`` with ``D = 2N`` interleaved complex lanes

Layout / launch contract
------------------------
- Host prep produces:
  - ``A_main``: ``(P, L, BHC)``, feature-contiguous
  - ``B_main``: ``(D, L, BHC)``, feature-contiguous
  - ``inc``: ``(P, D, BHC)``, row-major in ``(P, D)``
- CTA owns one ``(P_tile, D_tile, bhc)`` tile.
- Grid is ``(ceil_div(P, bM), ceil_div(D, bN), BHC)``.
- Thread layout is the standard SIMT GEMM layout from the CuTe ``sgemm``
  example. Each thread owns a contiguous fragment in the flattened
  accumulator tile.
- Predication is only for tile tails in ``P``, ``D``, and the initial
  residue ``L`` tile.

Numerical contract
------------------
This stage stays entirely in direct multiplicative form. Unlike the chunk-scan
stage, it never needs reciprocal prefix factors or log-domain segment ratios,
so there is no ``0 * inf`` hazard to guard against here.
"""

from __future__ import annotations

from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _complex_dtype_from_real,
    _pad_time_partial,
    _resolve_dtypes,
    _resolve_prev0,
    _to_complex_scalar,
    _to_complex_taps,
    _validate_chunk_increment_inputs,
)


_CompiledKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]
_COMPILED_GEMM: dict[_CompiledKey, Callable[..., object]] = {}
_BatchBmmKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    int,
    int,
    tuple[int, int, int],
    int,
    int,
]
_COMPILED_BATCH_BMM: dict[_BatchBmmKey, Callable[..., object]] = {}


class _BatchedSgemmFp32Ampere:
    """Ampere SIMT GEMM over ``(M, K, batch) x (N, K, batch) -> (M, N, batch)``.

    Logical shapes:
    - ``mA``: ``(M, K, B)``, column-major in ``M`` for the wrapper's prepared
      ``A_main = (P, L, BHC)``
    - ``mB``: ``(N, K, B)``, column-major in ``N`` for the wrapper's prepared
      ``B_main = (D, L, BHC)``
    - ``mC``: ``(M, N, B)``, row-major in ``(M, N)``

    This is the minimal serious kernel for ``chunk_increment``: it leaves the
    recurrence-specific algebra on the host and turns the dense work into one
    batched GEMM.
    """

    def __init__(
        self,
        *,
        cta_tiler: tuple[int, int, int] = (64, 64, 16),
        num_stages: int = 3,
        num_threads: int = 256,
    ) -> None:
        self._cta_tiler = cta_tiler
        self._num_stages = int(num_stages)
        self._num_threads = int(num_threads)

        self._bM, self._bN, self._bK = map(int, self._cta_tiler)
        if self._bM % 16 != 0 or self._bN % 16 != 0:
            raise ValueError("bM and bN must be multiples of 16.")
        if self._num_threads % 16 != 0:
            raise ValueError("num_threads must be a multiple of 16.")
        if self._num_stages < 3:
            raise ValueError("num_stages must be at least 3.")

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor) -> None:
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(1, (self._bM + padding_a), self._bK * (self._bM + padding_a)),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(1, (self._bN + padding_b), self._bK * (self._bN + padding_b)),
        )

        # A/B are prepared with the feature axis contiguous. Use vectorized
        # copies only along that guaranteed contiguous major mode.
        tA = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        tB = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        vA = cute.make_layout((1, 1))
        vB = cute.make_layout((1, 1))
        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width,
        )

        if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized = 4 if (mA.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width * num_vectorized,
            )
            major_mode_size = self._bM // num_vectorized
            tA = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vA = cute.make_layout((num_vectorized, 1))

        if cutlass.const_expr(self.b_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized = 4 if (mB.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mB.element_type,
                num_bits_per_copy=mB.element_type.width * num_vectorized,
            )
            major_mode_size = self._bN // num_vectorized
            tB = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vB = cute.make_layout((num_vectorized, 1))

        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)

        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
            atoms_layout = cute.make_layout(
                (16, self._num_threads // 16, 1), stride=(1, 16, 0)
            )

        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        perm_m = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
        perm_n = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
        tiled_mma = cute.make_tiled_mma(
            op, atoms_layout, permutation_mnk=(perm_m, perm_n, None)
        )

        grid_dim = cute.ceil_div(mC.shape, (self._bM, self._bN, 1))
        grid_z = cute.size(mC.shape[2])

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=(cute.size(grid_dim[0]), cute.size(grid_dim[1]), grid_z),
            block=[cute.size(atoms_layout), 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)

        thr_mma = tiled_mma.get_slice(tidx)

        gA = cute.local_tile(
            mA[None, None, bidz],
            tiler=self._cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB[None, None, bidz],
            tiler=self._cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gC = cute.local_tile(
            mC[None, None, bidz],
            tiler=self._cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        # Like the CuTe SGEMM example, move to the first irregular K tile so the
        # mainloop can assume regular K tiles after the prologue.
        residue_k = mA.shape[1] - cutlass.Int32(self._bK) * gA.shape[2]
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gB = cute.domain_offset((0, residue_k, 0), gB)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mB.shape)
        cA = cute.local_tile(
            mcA[None, None, bidz],
            tiler=self._cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB[None, None, bidz],
            tiler=self._cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        cA = cute.domain_offset((0, residue_k, 0), cA)
        cB = cute.domain_offset((0, residue_k, 0), cB)
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tApA_residue = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(
                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                    cute.size(tAsA, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        tBpB_residue = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(
                    cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                    cute.size(tBsB, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )

        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )

        for rest_v in range(tApA_residue.shape[0]):
            for m in range(tApA_residue.shape[1]):
                for k in range(tApA_residue.shape[2]):
                    coord_a = tAcA[(0, rest_v), m, k, 0]
                    tApA_residue[rest_v, m, k] = cute.elem_less(
                        (coord_a[0], cutlass.Int32(-1)), (mA.shape[0], coord_a[1])
                    )
        for rest_v in range(tBpB_residue.shape[0]):
            for n in range(tBpB_residue.shape[1]):
                for k in range(tBpB_residue.shape[2]):
                    coord_b = tBcB[(0, rest_v), n, k, 0]
                    tBpB_residue[rest_v, n, k] = cute.elem_less(
                        (coord_b[0], cutlass.Int32(-1)), (mB.shape[0], coord_b[1])
                    )

        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)

        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA_residue,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
            pred=tBpB_residue,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )

        for k_tile in range(1, k_pipe_max - 1):
            if k_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < k_tile_count
                else cutlass.Int32(0)
            )
            cute.arch.cp_async_commit_group()

        if k_tile_count < k_pipe_max:
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cutlass.Boolean(0)
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cutlass.Boolean(0)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])

        if k_block_max > 1:
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            cute.arch.barrier()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(k_pipe_max - 2)
                    cute.arch.barrier()

                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_p[None, None, k_block_next],
                    tCrB[None, None, k_block_next],
                )

                if k_block == 0:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

                if k_block == 0:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = cutlass.Int32(0)
                    # After the final real tile we recycle tile 1. Tile 0 is the
                    # residue tile and may be irregular, so treating it as the
                    # harmless steady-state dummy tile is not safe.
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(1)
                    )

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        pred_c = cute.make_fragment(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx
        residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy
        for i in range(cute.size(tCrC.shape)):
            pred_c[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=pred_c)


def _mark_prepared_input(t: torch.Tensor) -> cute.Tensor:
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=0)
        .mark_compact_shape_dynamic(mode=0, stride_order=(2, 1, 0), divisibility=4)
    )


def _mark_prepared_input_rowmajor(t: torch.Tensor) -> cute.Tensor:
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=4)
    )


def _mark_output(t: torch.Tensor) -> cute.Tensor:
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=4)
    )


def _prepared_suffix_products(m_blk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(suffix_after, m_chunk)`` for ``m_blk`` with shape ``(..., L)``.

    ``suffix_after[..., t]`` is the product of transitions strictly after step
    ``t`` inside the chunk. This stays in direct multiplicative form, so exact
    zero transitions remain well-defined here.
    """
    suffix_after = torch.ones_like(m_blk)
    L = int(m_blk.shape[-1])
    if L > 1:
        suffix_after[..., :-1] = torch.flip(
            torch.cumprod(torch.flip(m_blk[..., 1:], dims=(-1,)), dim=-1),
            dims=(-1,),
        )
    m_chunk = torch.prod(m_blk, dim=-1)
    return suffix_after, m_chunk


def _prepare_chunk_increment_operands(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
]:
    batch_size, n_heads, _, N, P = _validate_chunk_increment_inputs(
        U, M, K, B, B_prev, U_prev, int(U.shape[2])
    )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")

    D = 2 * N
    rdtype, _ = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_increment kernel supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )

    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = U.device

    U_f, M_f, K_f, B_f, _, n_chunks = _pad_time_partial(
        U, M, K, B, chunk_size=chunk_size, real_dtype=rdtype
    )
    L = int(chunk_size)

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_f, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]
    b_t = _as_complex_pairs(B_f, name="B").to(dtype=cplx_dtype)

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    k_prev_blk = k_prev.reshape(batch_size, n_heads, n_chunks, L)
    k_curr_blk = k_curr.reshape(batch_size, n_heads, n_chunks, L)
    b_blk = b_t.reshape(batch_size, n_heads, n_chunks, L, N)
    u_curr_blk = U_f.reshape(batch_size, n_heads, n_chunks, L, P)

    suffix_after, m_chunk = _prepared_suffix_products(m_blk)
    alpha_sum = suffix_after.unsqueeze(-1) * k_curr_blk.unsqueeze(-1) * b_blk
    if L > 1:
        alpha_sum[..., :-1, :] = alpha_sum[..., :-1, :] + (
            suffix_after[..., 1:].unsqueeze(-1)
            * k_prev_blk[..., 1:].unsqueeze(-1)
            * b_blk[..., :-1, :]
        )

    b_prev_chunk0 = torch.empty(
        (batch_size, n_heads, n_chunks, N), device=device, dtype=cplx_dtype
    )
    b_prev_chunk0[:, :, 0, :] = b_prev0
    if n_chunks > 1:
        b_prev_chunk0[:, :, 1:, :] = b_blk[:, :, :-1, -1, :]
    boundary = (
        suffix_after[..., 0].unsqueeze(-1)
        * k_prev_blk[..., 0].unsqueeze(-1)
        * b_prev_chunk0
    )

    u_head = torch.empty(
        (batch_size, n_heads, n_chunks, P), device=device, dtype=rdtype
    )
    u_head[:, :, 0, :] = u_prev0
    if n_chunks > 1:
        u_head[:, :, 1:, :] = u_curr_blk[:, :, :-1, -1, :]

    BHC = batch_size * n_heads * n_chunks

    A_main = u_curr_blk.reshape(BHC, L, P)
    B_main = torch.view_as_real(alpha_sum).reshape(BHC, L, D)
    u_head = u_head.reshape(BHC, P)
    b_head = torch.view_as_real(boundary).reshape(BHC, D)

    return A_main, B_main, u_head, b_head, m_chunk, batch_size, n_heads, n_chunks, P


def _compiled_key(
    A3: torch.Tensor, B3: torch.Tensor, C3: torch.Tensor, *, device_index: int
) -> _CompiledKey:
    return (
        device_index,
        A3.dtype,
        tuple(int(x) for x in A3.shape),
        tuple(int(x) for x in A3.stride()),
        tuple(int(x) for x in B3.shape),
        tuple(int(x) for x in B3.stride()),
        tuple(int(x) for x in C3.shape),
        tuple(int(x) for x in C3.stride()),
    )


def _get_compiled_gemm(
    A3: torch.Tensor, B3: torch.Tensor, C3: torch.Tensor
) -> Callable[..., object]:
    if A3.device.type != "cuda":
        raise ValueError("CuTe chunk_increment requires CUDA tensors.")

    device_index = 0 if A3.device.index is None else int(A3.device.index)
    key = _compiled_key(A3, B3, C3, device_index=device_index)
    compiled = _COMPILED_GEMM.get(key)
    if compiled is not None:
        return compiled

    kernel = _BatchedSgemmFp32Ampere(cta_tiler=(64, 64, 16))
    mA = _mark_prepared_input(A3)
    mB = _mark_prepared_input(B3)
    mC = _mark_output(C3)
    compiled = cute.compile(kernel, mA, mB, mC)
    _COMPILED_GEMM[key] = compiled
    return compiled


def _view_mode(view: torch.Tensor) -> int:
    if int(view.stride()[0]) == 1:
        return 0
    if int(view.stride()[1]) == 1:
        return 1
    raise ValueError(
        "Expected a view with a unit-stride leading dimension in mode 0 or 1. "
        f"Got shape={tuple(view.shape)} stride={tuple(view.stride())}."
    )


def _mark_batched_view(view: torch.Tensor, *, mode: int) -> cute.Tensor:
    if mode == 0:
        return _mark_prepared_input(view)
    if mode == 1:
        return _mark_prepared_input_rowmajor(view)
    raise ValueError(f"Unsupported batched GEMM mode {mode}.")


def _batched_sgemm_config(M: int, N: int, K: int) -> tuple[tuple[int, int, int], int, int]:
    tuned: dict[tuple[int, int, int], tuple[tuple[int, int, int], int, int]] = {
        (32, 32, 32): ((32, 32, 16), 3, 128),
        # Exact chunk_scan backward hot shapes.
        (64, 64, 96): ((64, 64, 16), 3, 128),
        (64, 64, 64): ((64, 64, 32), 3, 256),
        (64, 96, 64): ((64, 64, 16), 3, 128),
    }
    return tuned.get((M, N, K), ((64, 64, 16), 3, 256))


def _dummy_a_view(
    *,
    batch_size: int,
    M: int,
    K: int,
    mode: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mode == 0:
        return torch.empty((batch_size, K, M), device=device, dtype=dtype).permute(2, 1, 0)
    if mode == 1:
        return torch.empty((batch_size, M, K), device=device, dtype=dtype).permute(1, 2, 0)
    raise ValueError(f"Unsupported batched GEMM mode {mode}.")


def _dummy_b_view(
    *,
    batch_size: int,
    N: int,
    K: int,
    mode: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mode == 0:
        return torch.empty((batch_size, K, N), device=device, dtype=dtype).permute(2, 1, 0)
    if mode == 1:
        return torch.empty((batch_size, N, K), device=device, dtype=dtype).permute(1, 2, 0)
    raise ValueError(f"Unsupported batched GEMM mode {mode}.")


def _get_compiled_batch_bmm(
    A_view: torch.Tensor,
    B_view: torch.Tensor,
    C_view: torch.Tensor,
    *,
    a_mode: int,
    b_mode: int,
    cta_tiler: tuple[int, int, int],
    num_stages: int,
    num_threads: int,
) -> Callable[..., object]:
    if A_view.device.type != "cuda":
        raise ValueError("CuTe batched GEMM requires CUDA tensors.")

    device_index = 0 if A_view.device.index is None else int(A_view.device.index)
    key: _BatchBmmKey = (
        device_index,
        A_view.dtype,
        tuple(int(x) for x in A_view.shape),
        tuple(int(x) for x in A_view.stride()),
        tuple(int(x) for x in B_view.shape),
        tuple(int(x) for x in B_view.stride()),
        a_mode,
        b_mode,
        cta_tiler,
        int(num_stages),
        int(num_threads),
    )
    compiled = _COMPILED_BATCH_BMM.get(key)
    if compiled is not None:
        return compiled

    M, K, batch_size = map(int, A_view.shape)
    N = int(B_view.shape[0])
    kernel = _BatchedSgemmFp32Ampere(
        cta_tiler=cta_tiler,
        num_stages=num_stages,
        num_threads=num_threads,
    )
    dummy_A = _dummy_a_view(
        batch_size=batch_size,
        M=M,
        K=K,
        mode=a_mode,
        device=A_view.device,
        dtype=A_view.dtype,
    )
    dummy_B = _dummy_b_view(
        batch_size=batch_size,
        N=N,
        K=K,
        mode=b_mode,
        device=B_view.device,
        dtype=B_view.dtype,
    )
    dummy_C = torch.empty((batch_size, M, N), device=A_view.device, dtype=A_view.dtype).permute(1, 2, 0)
    compiled = cute.compile(
        kernel,
        _mark_batched_view(dummy_A, mode=a_mode),
        _mark_batched_view(dummy_B, mode=b_mode),
        _mark_output(dummy_C),
    )
    _COMPILED_BATCH_BMM[key] = compiled
    return compiled


def batched_sgemm_fp32_cute(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Exact fp32 batched GEMM for ``(B, M, K) @ (B, K, N)`` on CUDA.

    This reuses the CuTe fp32 SIMT GEMM from ``chunk_increment`` but consumes
    batch-first backward tensors directly via layout marking instead of forcing
    a repacking copy into feature-major buffers. That is what keeps the exact
    ``chunk_scan`` backward hot path from giving back the raw GEMM win to host
    layout churn.
    """
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("CuTe batched GEMM requires CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise ValueError("CuTe batched GEMM expects float32 inputs.")
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("CuTe batched GEMM expects rank-3 tensors.")
    if int(A.shape[0]) != int(B.shape[0]) or int(A.shape[2]) != int(B.shape[1]):
        raise ValueError(
            "Expected A=(B,M,K) and B=(B,K,N). Got "
            f"{tuple(A.shape)} and {tuple(B.shape)}."
        )

    batch_size, M, K = map(int, A.shape)
    _, _, N = map(int, B.shape)
    A_view = A.permute(1, 2, 0)
    B_view = B.permute(2, 1, 0)
    a_mode = _view_mode(A_view)
    b_mode = _view_mode(B_view)
    cta_tiler, num_stages, num_threads = _batched_sgemm_config(M, N, K)
    C = torch.empty((batch_size, M, N), device=A.device, dtype=torch.float32)
    C_view = C.permute(1, 2, 0)
    compiled = _get_compiled_batch_bmm(
        A_view,
        B_view,
        C_view,
        a_mode=a_mode,
        b_mode=b_mode,
        cta_tiler=cta_tiler,
        num_stages=num_stages,
        num_threads=num_threads,
    )
    compiled(
        _mark_batched_view(A_view, mode=a_mode),
        _mark_batched_view(B_view, mode=b_mode),
        _mark_output(C_view),
    )
    return C


def chunk_increment_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes per-chunk affine increments and chunk transitions with CuTe.

    Returns:
    - ``inc``: ``(batch, heads, chunks, P, 2N)``
    - ``m_chunk``: ``(batch, heads, chunks, 2)``
    """
    if not (U.is_cuda and M.is_cuda and K.is_cuda and B.is_cuda):
        raise ValueError("CuTe chunk_increment requires CUDA tensors.")

    A_main, B_main, u_head, b_head, m_chunk, batch_size, n_heads, n_chunks, P = (
        _prepare_chunk_increment_operands(
            U,
            M,
            K,
            B,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=compute_dtype,
        )
    )

    BHC = int(A_main.shape[0])
    D = int(B_main.shape[-1])

    inc_out = torch.empty((BHC, P, D), device=U.device, dtype=torch.float32)

    # Keep the prepared operands in the same compact layout that worked well for
    # the v3 forward kernels: feature-major, then time, then batch.
    A3 = A_main.permute(2, 1, 0)
    B3 = B_main.permute(2, 1, 0)
    C3 = inc_out.permute(1, 2, 0)

    compiled = _get_compiled_gemm(A3, B3, C3)
    mA = _mark_prepared_input(A3)
    mB = _mark_prepared_input(B3)
    mC = _mark_output(C3)
    compiled(mA, mB, mC)

    # The boundary contribution is a single rank-1 update per chunk summary.
    inc_out.addcmul_(u_head.unsqueeze(-1), b_head.unsqueeze(1), value=1.0)

    inc = inc_out.reshape(batch_size, n_heads, n_chunks, P, D).contiguous()
    m_chunk_packed = torch.view_as_real(m_chunk).reshape(
        batch_size, n_heads, n_chunks, 2
    )
    m_chunk_packed = m_chunk_packed.contiguous()
    return inc, m_chunk_packed


__all__ = ["batched_sgemm_fp32_cute", "chunk_increment_cute"]
