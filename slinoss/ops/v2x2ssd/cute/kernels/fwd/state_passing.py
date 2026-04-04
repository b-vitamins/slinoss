"""CuTe forward kernel for the v2x2ssd state-passing stage.

Inputs:
- ``inc``: ``(B, H, C, P, D)`` float32 chunk-local state increments
- ``m_chunk``: ``(B, H, C, 2)`` float32 packed-complex chunk multipliers
- ``initial_states``: optional ``(B, H, P, D)`` float32 initial state

Outputs:
- ``chunk_starts``: ``(B, H, C, P, D)`` float32 state before each chunk update
- ``final_state``: ``(B, H, P, D)`` float32 state after the last chunk update

The kernel is sequential over chunks and parallel over the flattened state axis
``S = P * D`` where ``D = 2N`` stores packed complex values as interleaved
real/imag lanes.
"""

from dataclasses import dataclass

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute


@dataclass(frozen=True)
class StatePassingLayoutBundle:
    chunk_state_layout: object
    chunk_multiplier_layout: object
    state_layout: object
    state_tile_layout: object
    thread_value_layout: object


@dataclass(frozen=True)
class StatePassingCopyBundle:
    increment_vector_copy: object
    increment_scalar_copy: object
    chunk_start_vector_copy: object
    chunk_start_scalar_copy: object
    initial_state_vector_copy: object
    initial_state_scalar_copy: object
    final_state_vector_copy: object
    final_state_scalar_copy: object
    multiplier_copy: object


@dataclass(frozen=True)
class StatePassingLaunchBundle:
    increment_flat: object
    chunk_multiplier_flat: object
    chunk_starts_flat: object
    final_state_flat: object
    initial_state_flat: object
    state_tile_coord_tensor: object
    layouts: StatePassingLayoutBundle
    copies: StatePassingCopyBundle
    grid_x: object
    grid_y: object


class StatePassingFwdAmpere:
    """Ampere forward state-passing kernel with fp32 state math."""

    def __init__(
        self,
        *,
        num_threads: int = 128,
        vecs_per_thread: int = 8,
        copy_bits_in: int,
        copy_bits_out: int,
        copy_bits_state_in: int,
        copy_bits_state_out: int,
        has_init: bool,
    ):
        self.num_threads = int(num_threads)
        self.vecs_per_thread = int(vecs_per_thread)
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.vecs_per_thread <= 0:
            raise ValueError("vecs_per_thread must be positive.")

        self.state_elems_per_thread = 2 * self.vecs_per_thread
        self.state_tile_elems = self.num_threads * self.state_elems_per_thread

        self.copy_bits_in = int(copy_bits_in)
        self.copy_bits_out = int(copy_bits_out)
        self.copy_bits_state_in = int(copy_bits_state_in)
        self.copy_bits_state_out = int(copy_bits_state_out)
        self.has_init = bool(has_init)

    def _make_layout_bundle(
        self,
        *,
        batch_head_count: int,
        chunk_count: int,
        state_elem_count: int,
    ) -> StatePassingLayoutBundle:
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
        return StatePassingLayoutBundle(
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
        increment_dtype: type[cutlass.Numeric],
        chunk_start_dtype: type[cutlass.Numeric],
        initial_state_dtype: type[cutlass.Numeric],
        final_state_dtype: type[cutlass.Numeric],
        multiplier_dtype: type[cutlass.Numeric],
    ) -> StatePassingCopyBundle:
        return StatePassingCopyBundle(
            increment_vector_copy=self._make_copy_atom(
                increment_dtype, self.copy_bits_in
            ),
            increment_scalar_copy=self._make_copy_atom(
                increment_dtype, increment_dtype.width
            ),
            chunk_start_vector_copy=self._make_copy_atom(
                chunk_start_dtype, self.copy_bits_out
            ),
            chunk_start_scalar_copy=self._make_copy_atom(
                chunk_start_dtype, chunk_start_dtype.width
            ),
            initial_state_vector_copy=self._make_copy_atom(
                initial_state_dtype, self.copy_bits_state_in
            ),
            initial_state_scalar_copy=self._make_copy_atom(
                initial_state_dtype, initial_state_dtype.width
            ),
            final_state_vector_copy=self._make_copy_atom(
                final_state_dtype, self.copy_bits_state_out
            ),
            final_state_scalar_copy=self._make_copy_atom(
                final_state_dtype, final_state_dtype.width
            ),
            multiplier_copy=self._make_copy_atom(
                multiplier_dtype, multiplier_dtype.width * 2
            ),
        )

    def _make_launch_bundle(
        self,
        *,
        increment: cute.Tensor,
        chunk_multiplier: cute.Tensor,
        chunk_starts: cute.Tensor,
        final_state: cute.Tensor,
        initial_state_or_dummy: cute.Tensor,
    ) -> StatePassingLaunchBundle:
        batch_size, head_count, chunk_count, state_rows, state_width = increment.shape
        batch_head_count = batch_size * head_count
        state_elem_count = state_rows * state_width

        layouts = self._make_layout_bundle(
            batch_head_count=batch_head_count,
            chunk_count=chunk_count,
            state_elem_count=state_elem_count,
        )
        copies = self._make_copy_bundle(
            increment_dtype=increment.element_type,
            chunk_start_dtype=chunk_starts.element_type,
            initial_state_dtype=initial_state_or_dummy.element_type,
            final_state_dtype=final_state.element_type,
            multiplier_dtype=chunk_multiplier.element_type,
        )

        increment_flat = cute.make_tensor(
            increment.iterator, layouts.chunk_state_layout
        )
        chunk_multiplier_flat = cute.make_tensor(
            chunk_multiplier.iterator, layouts.chunk_multiplier_layout
        )
        chunk_starts_flat = cute.make_tensor(
            chunk_starts.iterator, layouts.chunk_state_layout
        )
        final_state_flat = cute.make_tensor(final_state.iterator, layouts.state_layout)
        initial_state_flat = cute.make_tensor(
            initial_state_or_dummy.iterator, layouts.state_layout
        )
        state_identity = cute.make_identity_tensor(state_elem_count)
        state_tile_coord_tensor = cute.zipped_divide(
            state_identity, tiler=layouts.state_tile_layout
        )

        return StatePassingLaunchBundle(
            increment_flat=increment_flat,
            chunk_multiplier_flat=chunk_multiplier_flat,
            chunk_starts_flat=chunk_starts_flat,
            final_state_flat=final_state_flat,
            initial_state_flat=initial_state_flat,
            state_tile_coord_tensor=state_tile_coord_tensor,
            layouts=layouts,
            copies=copies,
            grid_x=cute.ceil_div(state_elem_count, self.state_tile_elems),
            grid_y=batch_head_count,
        )

    def _make_kernel_invocation(self, launch_bundle: StatePassingLaunchBundle):
        return self.kernel(
            launch_bundle.increment_flat,
            launch_bundle.chunk_multiplier_flat,
            launch_bundle.chunk_starts_flat,
            launch_bundle.final_state_flat,
            launch_bundle.initial_state_flat,
            launch_bundle.state_tile_coord_tensor,
            launch_bundle.layouts.state_tile_layout,
            launch_bundle.layouts.thread_value_layout,
            launch_bundle.copies.increment_vector_copy,
            launch_bundle.copies.increment_scalar_copy,
            launch_bundle.copies.chunk_start_vector_copy,
            launch_bundle.copies.chunk_start_scalar_copy,
            launch_bundle.copies.initial_state_vector_copy,
            launch_bundle.copies.initial_state_scalar_copy,
            launch_bundle.copies.final_state_vector_copy,
            launch_bundle.copies.final_state_scalar_copy,
            launch_bundle.copies.multiplier_copy,
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
    def _apply_chunk_update(
        self,
        state_accumulator: cute.Tensor,
        increment_fragment: cute.Tensor,
        chunk_multiplier,
    ):
        multiplier_real, multiplier_imag = chunk_multiplier[0], chunk_multiplier[1]
        state_pairs_per_thread = cute.size(state_accumulator) // 2
        for pair_idx in cutlass.range_constexpr(state_pairs_per_thread):
            base = pair_idx * 2
            state_real = state_accumulator[base + 0]
            state_imag = state_accumulator[base + 1]

            rotated_real = multiplier_real * state_real - multiplier_imag * state_imag
            rotated_imag = multiplier_real * state_imag + multiplier_imag * state_real

            state_accumulator[base + 0] = rotated_real + increment_fragment[base + 0]
            state_accumulator[base + 1] = rotated_imag + increment_fragment[base + 1]

    @cute.jit
    def __call__(
        self,
        increment: cute.Tensor,  # (B,H,C,P,D)
        chunk_multiplier: cute.Tensor,  # (B,H,C,2)
        chunk_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        final_state: cute.Tensor,  # (B,H,P,D) fp32
        initial_state_or_dummy: cute.Tensor,  # (B,H,P,D) or ignored when has_init=False
    ):
        launch_bundle = self._make_launch_bundle(
            increment=increment,
            chunk_multiplier=chunk_multiplier,
            chunk_starts=chunk_starts,
            final_state=final_state,
            initial_state_or_dummy=initial_state_or_dummy,
        )

        self._make_kernel_invocation(launch_bundle).launch(
            grid=[launch_bundle.grid_x, launch_bundle.grid_y, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.jit
    def call_on_stream(
        self,
        increment: cute.Tensor,  # (B,H,C,P,D)
        chunk_multiplier: cute.Tensor,  # (B,H,C,2)
        chunk_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        final_state: cute.Tensor,  # (B,H,P,D) fp32
        initial_state_or_dummy: cute.Tensor,  # (B,H,P,D) or ignored when has_init=False
        stream: cuda.CUstream,
    ):
        launch_bundle = self._make_launch_bundle(
            increment=increment,
            chunk_multiplier=chunk_multiplier,
            chunk_starts=chunk_starts,
            final_state=final_state,
            initial_state_or_dummy=initial_state_or_dummy,
        )

        self._make_kernel_invocation(launch_bundle).launch(
            grid=[launch_bundle.grid_x, launch_bundle.grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        increment_flat: cute.Tensor,  # (BH, C, S)
        chunk_multiplier_flat: cute.Tensor,  # (BH, C, 2)
        chunk_starts_flat: cute.Tensor,  # (BH, C, S) fp32
        final_state_flat: cute.Tensor,  # (BH, S) fp32
        initial_state_flat: cute.Tensor,  # (BH, S)
        state_tile_coord_tensor: cute.Tensor,  # (tile, ntiles)
        state_tile_layout: cute.Layout,
        thread_value_layout: cute.Layout,  # (tid, vid) -> linear coord in [0, tile)
        increment_vector_copy,
        increment_scalar_copy,
        chunk_start_vector_copy,
        chunk_start_scalar_copy,
        initial_state_vector_copy,
        initial_state_scalar_copy,
        final_state_vector_copy,
        final_state_scalar_copy,
        multiplier_copy,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_x, batch_head_idx, _ = cute.arch.block_idx()

        total_state_elems = increment_flat.shape[2]
        chunk_count = increment_flat.shape[1]

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

        global_initial_state = initial_state_flat[batch_head_idx, None]
        thread_initial_state = self._make_thread_value_view(
            global_initial_state,
            state_tile_layout,
            cta_coord,
            thread_value_layout,
            thread_idx,
        )

        state_accumulator = cute.make_rmem_tensor(
            thread_initial_state.shape, cutlass.Float32
        )
        state_accumulator.fill(0.0)
        if cutlass.const_expr(self.has_init):
            initial_state_fragment = cute.make_rmem_tensor_like(thread_initial_state)
            self._copy_fragment_from_global(
                initial_state_vector_copy,
                initial_state_scalar_copy,
                thread_initial_state,
                initial_state_fragment,
                thread_predicate,
                is_partial_tile,
            )
            state_accumulator.store(initial_state_fragment.load().to(cutlass.Float32))

        increment_fragment = cute.make_rmem_tensor_like(thread_initial_state)
        output_fragment = cute.make_rmem_tensor(
            thread_initial_state.shape, cutlass.Float32
        )

        for chunk_idx in cutlass.range(chunk_count, unroll=1):
            global_chunk_starts = chunk_starts_flat[batch_head_idx, chunk_idx, None]
            thread_chunk_starts = self._make_thread_value_view(
                global_chunk_starts,
                state_tile_layout,
                cta_coord,
                thread_value_layout,
                thread_idx,
            )

            output_fragment.store(state_accumulator.load())
            self._copy_fragment_to_global(
                chunk_start_vector_copy,
                chunk_start_scalar_copy,
                output_fragment,
                thread_chunk_starts,
                thread_predicate,
                is_partial_tile,
            )

            global_increment = increment_flat[batch_head_idx, chunk_idx, None]
            thread_increment = self._make_thread_value_view(
                global_increment,
                state_tile_layout,
                cta_coord,
                thread_value_layout,
                thread_idx,
            )
            self._copy_fragment_from_global(
                increment_vector_copy,
                increment_scalar_copy,
                thread_increment,
                increment_fragment,
                thread_predicate,
                is_partial_tile,
            )
            increment_fp32 = increment_fragment.load().to(cutlass.Float32)

            global_multiplier = chunk_multiplier_flat[batch_head_idx, chunk_idx, None]
            multiplier_fragment = cute.make_rmem_tensor_like(global_multiplier)
            cute.copy(multiplier_copy, global_multiplier, multiplier_fragment)
            chunk_multiplier_value = multiplier_fragment.load().to(cutlass.Float32)

            self._apply_chunk_update(
                state_accumulator=state_accumulator,
                increment_fragment=increment_fp32,
                chunk_multiplier=chunk_multiplier_value,
            )

        global_final_state = final_state_flat[batch_head_idx, None]
        thread_final_state = self._make_thread_value_view(
            global_final_state,
            state_tile_layout,
            cta_coord,
            thread_value_layout,
            thread_idx,
        )

        output_fragment.store(state_accumulator.load())
        self._copy_fragment_to_global(
            final_state_vector_copy,
            final_state_scalar_copy,
            output_fragment,
            thread_final_state,
            thread_predicate,
            is_partial_tile,
        )

        return
