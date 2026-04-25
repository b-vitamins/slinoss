"""CuTe backward state-passing kernel for ``v2x2ssd``.

Inputs:
- ``chunk_starts``: ``(B, H, C, P, D)`` float32 state before each forward chunk
- ``d_chunk_starts``: ``(B, H, C, P, D)`` float32 upstream gradient for chunk starts
- ``d_final``: ``(B, H, P, D)`` float32 upstream gradient for the final state
- ``m_chunk``: ``(B, H, C, 2)`` float32 packed-complex chunk multipliers

Outputs:
- ``d_inc``: ``(B, H, C, P, D)`` float32 gradient for chunk-local increments
- ``d_chunk_multiplier``: ``(B, H, C, 2)`` float32 gradient for chunk multipliers
- ``d_initial``: ``(B, H, P, D)`` float32 gradient for the initial state

The kernel walks chunks in reverse while parallelizing over the flattened
state axis ``S = P * D``. ``D = 2N`` stores packed complex values as
interleaved real and imaginary lanes.
"""

from dataclasses import dataclass

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

from .common import _TileConfig


@dataclass(frozen=True)
class BwdStatePassingLayoutBundle:
    chunk_state_layout: object
    chunk_multiplier_layout: object
    state_layout: object
    state_tile_layout: object
    thread_value_layout: object


@dataclass(frozen=True)
class BwdStatePassingCopyBundle:
    chunk_start_vector_copy: object
    chunk_start_scalar_copy: object
    d_chunk_start_vector_copy: object
    d_chunk_start_scalar_copy: object
    d_increment_vector_copy: object
    d_increment_scalar_copy: object
    d_initial_vector_copy: object
    d_initial_scalar_copy: object
    d_final_vector_copy: object
    d_final_scalar_copy: object
    multiplier_copy: object


@dataclass(frozen=True)
class BwdStatePassingLaunchBundle:
    chunk_starts_flat: object
    d_chunk_starts_flat: object
    d_final_flat: object
    chunk_multiplier_flat: object
    d_increment_flat: object
    d_chunk_multiplier_flat: object
    d_initial_flat: object
    state_tile_coord_tensor: object
    layouts: BwdStatePassingLayoutBundle
    copies: BwdStatePassingCopyBundle
    grid_x: object
    grid_y: object


class BwdStatePassingAmpere:
    """Ampere backward state-passing kernel with fp32 state math."""

    def __init__(
        self,
        cfg: _TileConfig,
        *,
        copy_bits_starts: int,
        copy_bits_dstarts: int,
        copy_bits_dinc: int,
        copy_bits_initial: int,
        copy_bits_final: int,
    ):
        self.cfg = cfg
        self.num_threads = int(cfg.num_threads)
        self.pairs_per_thread = int(cfg.pairs_per_thread)
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.pairs_per_thread <= 0:
            raise ValueError("pairs_per_thread must be positive.")

        self.state_elems_per_thread = 2 * self.pairs_per_thread
        self.state_tile_elems = self.num_threads * self.state_elems_per_thread

        self.copy_bits_starts = int(copy_bits_starts)
        self.copy_bits_dstarts = int(copy_bits_dstarts)
        self.copy_bits_dinc = int(copy_bits_dinc)
        self.copy_bits_initial = int(copy_bits_initial)
        self.copy_bits_final = int(copy_bits_final)

    def _warp_partial_layout(self):
        return cute.make_layout((self.num_threads // 32, 2), stride=(2, 1))

    def _make_shared_storage(self):
        warp_partial_layout = self._warp_partial_layout()

        @cute.struct
        class SharedStorage:
            warp_partial: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(warp_partial_layout)],
                8,
            ]

        return SharedStorage

    def _make_layout_bundle(
        self,
        *,
        batch_head_count: int,
        chunk_count: int,
        state_elem_count: int,
    ) -> BwdStatePassingLayoutBundle:
        chunk_state_layout = cute.make_layout(
            (batch_head_count, chunk_count, state_elem_count),
            stride=(chunk_count * state_elem_count, state_elem_count, 1),
        )
        chunk_multiplier_layout = cute.make_layout(
            (batch_head_count, chunk_count, 2),
            stride=(chunk_count * 2, 2, 1),
        )
        state_layout = cute.make_layout(
            (batch_head_count, state_elem_count), stride=(state_elem_count, 1)
        )
        state_tile_layout = cute.make_layout(self.state_tile_elems)
        thread_value_layout = cute.make_layout(
            (self.num_threads, self.state_elems_per_thread),
            stride=(self.state_elems_per_thread, 1),
        )
        return BwdStatePassingLayoutBundle(
            chunk_state_layout=chunk_state_layout,
            chunk_multiplier_layout=chunk_multiplier_layout,
            state_layout=state_layout,
            state_tile_layout=state_tile_layout,
            thread_value_layout=thread_value_layout,
        )

    @staticmethod
    def _make_copy_atom(dtype: type[cutlass.Numeric], num_bits: int):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=int(num_bits),
        )

    def _make_copy_bundle(
        self,
        *,
        chunk_start_dtype: type[cutlass.Numeric],
        d_chunk_start_dtype: type[cutlass.Numeric],
        d_increment_dtype: type[cutlass.Numeric],
        d_initial_dtype: type[cutlass.Numeric],
        d_final_dtype: type[cutlass.Numeric],
        multiplier_dtype: type[cutlass.Numeric],
    ) -> BwdStatePassingCopyBundle:
        return BwdStatePassingCopyBundle(
            chunk_start_vector_copy=self._make_copy_atom(
                chunk_start_dtype, self.copy_bits_starts
            ),
            chunk_start_scalar_copy=self._make_copy_atom(
                chunk_start_dtype, chunk_start_dtype.width
            ),
            d_chunk_start_vector_copy=self._make_copy_atom(
                d_chunk_start_dtype, self.copy_bits_dstarts
            ),
            d_chunk_start_scalar_copy=self._make_copy_atom(
                d_chunk_start_dtype, d_chunk_start_dtype.width
            ),
            d_increment_vector_copy=self._make_copy_atom(
                d_increment_dtype, self.copy_bits_dinc
            ),
            d_increment_scalar_copy=self._make_copy_atom(
                d_increment_dtype, d_increment_dtype.width
            ),
            d_initial_vector_copy=self._make_copy_atom(
                d_initial_dtype, self.copy_bits_initial
            ),
            d_initial_scalar_copy=self._make_copy_atom(
                d_initial_dtype, d_initial_dtype.width
            ),
            d_final_vector_copy=self._make_copy_atom(
                d_final_dtype, self.copy_bits_final
            ),
            d_final_scalar_copy=self._make_copy_atom(
                d_final_dtype, d_final_dtype.width
            ),
            multiplier_copy=self._make_copy_atom(
                multiplier_dtype, multiplier_dtype.width * 2
            ),
        )

    def _make_launch_bundle(
        self,
        *,
        chunk_starts: cute.Tensor,
        d_chunk_starts: cute.Tensor,
        d_final: cute.Tensor,
        chunk_multiplier: cute.Tensor,
        d_increment: cute.Tensor,
        d_chunk_multiplier: cute.Tensor,
        d_initial: cute.Tensor,
    ) -> BwdStatePassingLaunchBundle:
        batch_size, head_count, chunk_count, state_rows, state_width = d_increment.shape
        batch_head_count = batch_size * head_count
        state_elem_count = state_rows * state_width

        layouts = self._make_layout_bundle(
            batch_head_count=batch_head_count,
            chunk_count=chunk_count,
            state_elem_count=state_elem_count,
        )
        copies = self._make_copy_bundle(
            chunk_start_dtype=chunk_starts.element_type,
            d_chunk_start_dtype=d_chunk_starts.element_type,
            d_increment_dtype=d_increment.element_type,
            d_initial_dtype=d_initial.element_type,
            d_final_dtype=d_final.element_type,
            multiplier_dtype=chunk_multiplier.element_type,
        )

        chunk_starts_flat = cute.make_tensor(
            chunk_starts.iterator, layouts.chunk_state_layout
        )
        d_chunk_starts_flat = cute.make_tensor(
            d_chunk_starts.iterator, layouts.chunk_state_layout
        )
        d_final_flat = cute.make_tensor(d_final.iterator, layouts.state_layout)
        chunk_multiplier_flat = cute.make_tensor(
            chunk_multiplier.iterator, layouts.chunk_multiplier_layout
        )
        d_increment_flat = cute.make_tensor(
            d_increment.iterator, layouts.chunk_state_layout
        )
        d_chunk_multiplier_flat = cute.make_tensor(
            d_chunk_multiplier.iterator, layouts.chunk_multiplier_layout
        )
        d_initial_flat = cute.make_tensor(d_initial.iterator, layouts.state_layout)
        state_identity = cute.make_identity_tensor(state_elem_count)
        state_tile_coord_tensor = cute.zipped_divide(
            state_identity, tiler=layouts.state_tile_layout
        )

        return BwdStatePassingLaunchBundle(
            chunk_starts_flat=chunk_starts_flat,
            d_chunk_starts_flat=d_chunk_starts_flat,
            d_final_flat=d_final_flat,
            chunk_multiplier_flat=chunk_multiplier_flat,
            d_increment_flat=d_increment_flat,
            d_chunk_multiplier_flat=d_chunk_multiplier_flat,
            d_initial_flat=d_initial_flat,
            state_tile_coord_tensor=state_tile_coord_tensor,
            layouts=layouts,
            copies=copies,
            grid_x=cute.ceil_div(state_elem_count, self.state_tile_elems),
            grid_y=batch_head_count,
        )

    def _make_kernel_invocation(
        self,
        launch_bundle: BwdStatePassingLaunchBundle,
        shared_storage_cls: cutlass.Constexpr,
    ):
        return self.kernel(
            launch_bundle.chunk_starts_flat,
            launch_bundle.d_chunk_starts_flat,
            launch_bundle.d_final_flat,
            launch_bundle.chunk_multiplier_flat,
            launch_bundle.d_increment_flat,
            launch_bundle.d_chunk_multiplier_flat,
            launch_bundle.d_initial_flat,
            launch_bundle.state_tile_coord_tensor,
            launch_bundle.layouts.state_tile_layout,
            launch_bundle.layouts.thread_value_layout,
            launch_bundle.copies.chunk_start_vector_copy,
            launch_bundle.copies.chunk_start_scalar_copy,
            launch_bundle.copies.d_chunk_start_vector_copy,
            launch_bundle.copies.d_chunk_start_scalar_copy,
            launch_bundle.copies.d_increment_vector_copy,
            launch_bundle.copies.d_increment_scalar_copy,
            launch_bundle.copies.d_initial_vector_copy,
            launch_bundle.copies.d_initial_scalar_copy,
            launch_bundle.copies.d_final_vector_copy,
            launch_bundle.copies.d_final_scalar_copy,
            launch_bundle.copies.multiplier_copy,
            shared_storage_cls,
        )

    @cute.jit
    def _make_thread_value_view(
        self,
        global_tensor: cute.Tensor,
        state_tile_layout: cute.Layout,
        cta_coord,
        thread_value_layout: cute.Layout,
        thread_idx: cutlass.Int32,
    ):
        tiled_tensor = cute.zipped_divide(global_tensor, tiler=state_tile_layout)
        cta_tensor = tiled_tensor[cta_coord]
        thread_tensor = cute.composition(cta_tensor, thread_value_layout)
        return thread_tensor[thread_idx, None]

    @cute.jit
    def _make_thread_predicate(
        self,
        state_tile_coord_tensor: cute.Tensor,
        cta_coord,
        thread_value_layout: cute.Layout,
        thread_idx: cutlass.Int32,
        total_state_elems,
        is_partial_tile,
    ):
        thread_state_coords = cute.composition(
            state_tile_coord_tensor[cta_coord], thread_value_layout
        )[thread_idx, None]
        thread_predicate = cute.make_rmem_tensor(
            thread_state_coords.shape, cutlass.Boolean
        )
        thread_predicate.fill(cutlass.Boolean(True))
        if is_partial_tile:
            for value_idx in cutlass.range_constexpr(cute.size(thread_predicate)):
                thread_predicate[value_idx] = cute.elem_less(
                    thread_state_coords[value_idx], total_state_elems
                )
        return thread_predicate

    @cute.jit
    def _copy_fragment_from_global(
        self,
        vector_copy,
        scalar_copy,
        source_view: cute.Tensor,
        destination_fragment: cute.Tensor,
        thread_predicate: cute.Tensor,
        is_partial_tile,
    ):
        destination_fragment.fill(0)
        if is_partial_tile:
            cute.copy(
                scalar_copy,
                source_view,
                destination_fragment,
                pred=thread_predicate,
            )
        else:
            cute.copy(vector_copy, source_view, destination_fragment)

    @cute.jit
    def _copy_fragment_to_global(
        self,
        vector_copy,
        scalar_copy,
        source_fragment: cute.Tensor,
        destination_view: cute.Tensor,
        thread_predicate: cute.Tensor,
        is_partial_tile,
    ):
        if is_partial_tile:
            cute.copy(
                scalar_copy,
                source_fragment,
                destination_view,
                pred=thread_predicate,
            )
        else:
            cute.copy(vector_copy, source_fragment, destination_view)

    @cute.jit
    def _store_state_adjoint_to_global(
        self,
        vector_copy,
        scalar_copy,
        state_adjoint: cute.Tensor,
        destination_view: cute.Tensor,
        thread_predicate: cute.Tensor,
        is_partial_tile,
    ):
        destination_fragment = cute.make_rmem_tensor_like(destination_view)
        destination_fragment.store(
            state_adjoint.load().to(destination_view.element_type)
        )
        self._copy_fragment_to_global(
            vector_copy,
            scalar_copy,
            destination_fragment,
            destination_view,
            thread_predicate,
            is_partial_tile,
        )

    @cute.jit
    def _accumulate_chunk_multiplier_gradient(
        self,
        state_adjoint: cute.Tensor,
        chunk_start_fragment: cute.Tensor,
        d_chunk_multiplier_partial: cute.Tensor,
    ):
        d_chunk_multiplier_partial.fill(0.0)
        state_pairs_per_thread = cute.size(state_adjoint) // 2
        for pair_idx in cutlass.range_constexpr(state_pairs_per_thread):
            base = pair_idx * 2
            start_real = chunk_start_fragment[base + 0]
            start_imag = chunk_start_fragment[base + 1]
            grad_real = state_adjoint[base + 0]
            grad_imag = state_adjoint[base + 1]
            d_chunk_multiplier_partial[0] += (
                grad_real * start_real + grad_imag * start_imag
            )
            d_chunk_multiplier_partial[1] += (
                grad_imag * start_real - grad_real * start_imag
            )
        for offset in (16, 8, 4, 2, 1):
            d_chunk_multiplier_partial[0] += cute.arch.shuffle_sync_bfly(
                d_chunk_multiplier_partial[0],
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )
            d_chunk_multiplier_partial[1] += cute.arch.shuffle_sync_bfly(
                d_chunk_multiplier_partial[1],
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )

    @cute.jit
    def _load_chunk_multiplier(
        self,
        multiplier_copy,
        global_multiplier: cute.Tensor,
        chunk_multiplier: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        chunk_multiplier.fill(0.0)
        if lane == cutlass.Int32(0):
            multiplier_fragment = cute.make_rmem_tensor_like(global_multiplier)
            cute.copy(multiplier_copy, global_multiplier, multiplier_fragment)
            chunk_multiplier_value = multiplier_fragment.load().to(cutlass.Float32)
            chunk_multiplier[0] = chunk_multiplier_value[0]
            chunk_multiplier[1] = chunk_multiplier_value[1]
        for offset in (1, 2, 4, 8, 16):
            chunk_multiplier[0] += cute.arch.shuffle_sync_bfly(
                chunk_multiplier[0],
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )
            chunk_multiplier[1] += cute.arch.shuffle_sync_bfly(
                chunk_multiplier[1],
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )

    @cute.jit
    def _apply_chunk_backprop(
        self,
        state_adjoint: cute.Tensor,
        d_chunk_start_fragment: cute.Tensor,
        chunk_multiplier: cute.Tensor,
    ):
        multiplier_real, multiplier_imag = chunk_multiplier[0], chunk_multiplier[1]
        state_pairs_per_thread = cute.size(state_adjoint) // 2
        for pair_idx in cutlass.range_constexpr(state_pairs_per_thread):
            base = pair_idx * 2
            grad_real = state_adjoint[base + 0]
            grad_imag = state_adjoint[base + 1]

            propagated_real = multiplier_real * grad_real + multiplier_imag * grad_imag
            propagated_imag = multiplier_real * grad_imag - multiplier_imag * grad_real

            state_adjoint[base + 0] = propagated_real + d_chunk_start_fragment[base + 0]
            state_adjoint[base + 1] = propagated_imag + d_chunk_start_fragment[base + 1]

    @cute.jit
    def _atomic_add_chunk_multiplier_gradient(
        self,
        d_chunk_multiplier_flat: cute.Tensor,
        batch_head_idx,
        chunk_idx,
        d_chunk_multiplier_partial: cute.Tensor,
        s_warp_partial: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        if lane == cutlass.Int32(0):
            s_warp_partial[warp, 0] = d_chunk_multiplier_partial[0]
            s_warp_partial[warp, 1] = d_chunk_multiplier_partial[1]
        pipeline.sync()
        if warp == cutlass.Int32(0) and lane == cutlass.Int32(0):
            total_re = cutlass.Float32(0.0)
            total_im = cutlass.Float32(0.0)
            for warp_idx in cutlass.range_constexpr(self.num_threads // 32):
                total_re = total_re + s_warp_partial[warp_idx, 0]
                total_im = total_im + s_warp_partial[warp_idx, 1]
            global_d_chunk_multiplier = d_chunk_multiplier_flat[
                batch_head_idx, chunk_idx, None
            ]
            cute.arch.atomic_add(
                (global_d_chunk_multiplier.iterator + 0).llvm_ptr,
                total_re,
            )
            cute.arch.atomic_add(
                (global_d_chunk_multiplier.iterator + 1).llvm_ptr,
                total_im,
            )
        pipeline.sync()

    @cute.jit
    def __call__(
        self,
        chunk_starts: cute.Tensor,  # (B,H,C,P,D)
        d_chunk_starts: cute.Tensor,  # (B,H,C,P,D)
        d_final: cute.Tensor,  # (B,H,P,D)
        chunk_multiplier: cute.Tensor,  # (B,H,C,2)
        d_increment: cute.Tensor,  # (B,H,C,P,D)
        d_chunk_multiplier: cute.Tensor,  # (B,H,C,2)
        d_initial: cute.Tensor,  # (B,H,P,D)
    ):
        launch_bundle = self._make_launch_bundle(
            chunk_starts=chunk_starts,
            d_chunk_starts=d_chunk_starts,
            d_final=d_final,
            chunk_multiplier=chunk_multiplier,
            d_increment=d_increment,
            d_chunk_multiplier=d_chunk_multiplier,
            d_initial=d_initial,
        )
        shared_storage_cls = self._make_shared_storage()
        self._make_kernel_invocation(launch_bundle, shared_storage_cls).launch(
            grid=[launch_bundle.grid_x, launch_bundle.grid_y, 1],
            block=[self.num_threads, 1, 1],
            smem=int(shared_storage_cls.size_in_bytes()),
        )

    @cute.jit
    def call_on_stream(
        self,
        chunk_starts: cute.Tensor,  # (B,H,C,P,D)
        d_chunk_starts: cute.Tensor,  # (B,H,C,P,D)
        d_final: cute.Tensor,  # (B,H,P,D)
        chunk_multiplier: cute.Tensor,  # (B,H,C,2)
        d_increment: cute.Tensor,  # (B,H,C,P,D)
        d_chunk_multiplier: cute.Tensor,  # (B,H,C,2)
        d_initial: cute.Tensor,  # (B,H,P,D)
        stream: cuda.CUstream,
    ):
        launch_bundle = self._make_launch_bundle(
            chunk_starts=chunk_starts,
            d_chunk_starts=d_chunk_starts,
            d_final=d_final,
            chunk_multiplier=chunk_multiplier,
            d_increment=d_increment,
            d_chunk_multiplier=d_chunk_multiplier,
            d_initial=d_initial,
        )
        shared_storage_cls = self._make_shared_storage()
        self._make_kernel_invocation(launch_bundle, shared_storage_cls).launch(
            grid=[launch_bundle.grid_x, launch_bundle.grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            smem=int(shared_storage_cls.size_in_bytes()),
        )

    @cute.kernel
    def kernel(
        self,
        chunk_starts_flat: cute.Tensor,  # (BH, C, S)
        d_chunk_starts_flat: cute.Tensor,  # (BH, C, S)
        d_final_flat: cute.Tensor,  # (BH, S)
        chunk_multiplier_flat: cute.Tensor,  # (BH, C, 2)
        d_increment_flat: cute.Tensor,  # (BH, C, S)
        d_chunk_multiplier_flat: cute.Tensor,  # (BH, C, 2)
        d_initial_flat: cute.Tensor,  # (BH, S)
        state_tile_coord_tensor: cute.Tensor,  # (tile, ntiles)
        state_tile_layout: cute.Layout,
        thread_value_layout: cute.Layout,  # (tid, vid) -> linear coord in [0, tile)
        chunk_start_vector_copy,
        chunk_start_scalar_copy,
        d_chunk_start_vector_copy,
        d_chunk_start_scalar_copy,
        d_increment_vector_copy,
        d_increment_scalar_copy,
        d_initial_vector_copy,
        d_initial_scalar_copy,
        d_final_vector_copy,
        d_final_scalar_copy,
        multiplier_copy,
        shared_storage_cls: cutlass.Constexpr,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_x, batch_head_idx, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_warp_partial = storage.warp_partial.get_tensor(self._warp_partial_layout())

        total_state_elems = d_increment_flat.shape[2]
        chunk_count = d_increment_flat.shape[1]

        state_tile_start = cutlass.Int32(self.state_tile_elems) * block_x
        residue = total_state_elems - state_tile_start
        is_partial_tile = cute.elem_less(residue, cutlass.Int32(self.state_tile_elems))

        cta_coord = (None, block_x)
        thread_predicate = self._make_thread_predicate(
            state_tile_coord_tensor=state_tile_coord_tensor,
            cta_coord=cta_coord,
            thread_value_layout=thread_value_layout,
            thread_idx=thread_idx,
            total_state_elems=total_state_elems,
            is_partial_tile=is_partial_tile,
        )

        global_final_state = d_final_flat[batch_head_idx, None]
        thread_final_state = self._make_thread_value_view(
            global_final_state,
            state_tile_layout,
            cta_coord,
            thread_value_layout,
            thread_idx,
        )

        state_adjoint = cute.make_rmem_tensor(thread_final_state.shape, cutlass.Float32)
        state_adjoint.fill(0.0)
        d_final_fragment = cute.make_rmem_tensor_like(thread_final_state)
        self._copy_fragment_from_global(
            d_final_vector_copy,
            d_final_scalar_copy,
            thread_final_state,
            d_final_fragment,
            thread_predicate,
            is_partial_tile,
        )
        state_adjoint.store(d_final_fragment.load().to(cutlass.Float32))

        chunk_start_fragment = cute.make_rmem_tensor_like(thread_final_state)
        d_chunk_start_fragment = cute.make_rmem_tensor_like(thread_final_state)
        chunk_multiplier_value = cute.make_rmem_tensor((2,), cutlass.Float32)
        d_chunk_multiplier_partial = cute.make_rmem_tensor((2,), cutlass.Float32)

        for chunk_offset in cutlass.range(chunk_count, unroll=1):
            chunk_idx = chunk_count - 1 - chunk_offset

            global_d_increment = d_increment_flat[batch_head_idx, chunk_idx, None]
            thread_d_increment = self._make_thread_value_view(
                global_d_increment,
                state_tile_layout,
                cta_coord,
                thread_value_layout,
                thread_idx,
            )
            self._store_state_adjoint_to_global(
                d_increment_vector_copy,
                d_increment_scalar_copy,
                state_adjoint,
                thread_d_increment,
                thread_predicate,
                is_partial_tile,
            )

            global_chunk_start = chunk_starts_flat[batch_head_idx, chunk_idx, None]
            thread_chunk_start = self._make_thread_value_view(
                global_chunk_start,
                state_tile_layout,
                cta_coord,
                thread_value_layout,
                thread_idx,
            )
            self._copy_fragment_from_global(
                chunk_start_vector_copy,
                chunk_start_scalar_copy,
                thread_chunk_start,
                chunk_start_fragment,
                thread_predicate,
                is_partial_tile,
            )
            chunk_start_fp32 = chunk_start_fragment.load().to(cutlass.Float32)

            self._accumulate_chunk_multiplier_gradient(
                state_adjoint,
                chunk_start_fp32,
                d_chunk_multiplier_partial,
            )
            self._atomic_add_chunk_multiplier_gradient(
                d_chunk_multiplier_flat,
                batch_head_idx,
                chunk_idx,
                d_chunk_multiplier_partial,
                s_warp_partial,
            )

            global_d_chunk_start = d_chunk_starts_flat[batch_head_idx, chunk_idx, None]
            thread_d_chunk_start = self._make_thread_value_view(
                global_d_chunk_start,
                state_tile_layout,
                cta_coord,
                thread_value_layout,
                thread_idx,
            )
            self._copy_fragment_from_global(
                d_chunk_start_vector_copy,
                d_chunk_start_scalar_copy,
                thread_d_chunk_start,
                d_chunk_start_fragment,
                thread_predicate,
                is_partial_tile,
            )
            d_chunk_start_fp32 = d_chunk_start_fragment.load().to(cutlass.Float32)

            global_chunk_multiplier = chunk_multiplier_flat[
                batch_head_idx, chunk_idx, None
            ]
            self._load_chunk_multiplier(
                multiplier_copy,
                global_chunk_multiplier,
                chunk_multiplier_value,
            )
            self._apply_chunk_backprop(
                state_adjoint,
                d_chunk_start_fp32,
                chunk_multiplier_value,
            )

        global_d_initial = d_initial_flat[batch_head_idx, None]
        thread_d_initial = self._make_thread_value_view(
            global_d_initial,
            state_tile_layout,
            cta_coord,
            thread_value_layout,
            thread_idx,
        )

        self._store_state_adjoint_to_global(
            d_initial_vector_copy,
            d_initial_scalar_copy,
            state_adjoint,
            thread_d_initial,
            thread_predicate,
            is_partial_tile,
        )

        return
