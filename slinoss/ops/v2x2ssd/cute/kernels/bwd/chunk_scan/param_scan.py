"""CuTe backward kernel for the ``v2x2ssd`` chunk-scan ``param_scan`` stage.

``ChunkScanBwdParamScanAmpere`` computes the reverse metadata scan for ``dM``
and the raw complex tap gradients ``dKprev`` / ``dKcurr``. One thread handles
one ``(BHC, split)`` lane and walks the chunk in reverse time.

Tensor contracts:

- ``M``: ``(BHC, L, 2)`` fp32 packed complex transitions
- ``K``: ``(BHC, L, 2, 2)`` fp32 packed complex taps for the previous/current
  diagonal passes
- ``dlogprefix``: ``(BHC, n_splits, L)`` fp32 reverse log-prefix partials
- ``dMprev`` / ``dMcurr``: ``(BHC, n_splits, L, 2)`` fp32 transformed tap
  partials before the reverse metadata scan
- ``dR``: ``(BHC, n_splits, L, 4)`` fp32 partials of the real phase matrix
- ``dM``: ``(BHC, n_splits, L, 2)`` fp32 reverse metadata output
- ``dKprev`` / ``dKcurr``: ``(BHC, n_splits, L, 2)`` fp32 raw tap gradients
"""

from __future__ import annotations

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute

from .common import complex_mul

# Keep phase normalization numerically well behaved on zero-padded tail steps
# without materially perturbing valid chunk-local transitions.
EPS_NORM = 1.0e-20


class ChunkScanBwdParamScanAmpere:
    """Reverse metadata scan + raw tap grads for the chunk-scan backward path."""

    def __init__(
        self,
        *,
        chunk_size: int,
        num_threads: int = 32,
        assume_dmprev_zero: bool = False,
    ):
        self.L = int(chunk_size)
        self.num_threads = int(num_threads)
        self.assume_dmprev_zero = bool(assume_dmprev_zero)

        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")

    def _grid_dim(
        self,
        mM: cute.Tensor,
        mDLogPrefix: cute.Tensor,
    ) -> tuple[int, int, int]:
        batch_head_chunks = cute.size(mM.shape[0])
        n_splits = cute.size(mDLogPrefix.shape[1])
        return (
            cute.ceil_div(batch_head_chunks, self.num_threads),
            n_splits,
            1,
        )

    # Host launch API
    @cute.jit
    def _validate_operands(
        self,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        mDR: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
    ):
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mK.element_type != cutlass.Float32):
            raise TypeError("K must be Float32.")
        if cutlass.const_expr(mDLogPrefix.element_type != cutlass.Float32):
            raise TypeError("dlogprefix must be Float32.")
        if cutlass.const_expr(
            mDMprev.element_type != cutlass.Float32
            or mDMcurr.element_type != cutlass.Float32
            or mDR.element_type != cutlass.Float32
        ):
            raise TypeError("dMprev/dMcurr/dR must be Float32.")
        if cutlass.const_expr(
            mDM.element_type != cutlass.Float32
            or mDKprev.element_type != cutlass.Float32
            or mDKcurr.element_type != cutlass.Float32
        ):
            raise TypeError("dM/dK outputs must be Float32.")

        if cutlass.const_expr(mM.shape[1] != self.L or mM.shape[2] != 2):
            raise ValueError("M must be shaped as (BHC, L, 2).")
        if cutlass.const_expr(
            mK.shape[1] != self.L or mK.shape[2] != 2 or mK.shape[3] != 2
        ):
            raise ValueError("K must be shaped as (BHC, L, 2, 2) matching M.")
        if cutlass.const_expr(
            mDLogPrefix.shape[1] != 1 or mDLogPrefix.shape[2] != self.L
        ):
            raise ValueError(
                "dlogprefix must be shaped as (BHC, n_splits, L) matching M."
            )
        if cutlass.const_expr(
            mDMprev.shape[1] != 1
            or mDMprev.shape[2] != self.L
            or mDMprev.shape[3] != 2
            or mDMcurr.shape[1] != 1
            or mDMcurr.shape[2] != self.L
            or mDMcurr.shape[3] != 2
        ):
            raise ValueError(
                "dMprev/dMcurr must be shaped as (BHC, n_splits, L, 2) matching dlogprefix."
            )
        if cutlass.const_expr(
            mDR.shape[1] != 1 or mDR.shape[2] != self.L or mDR.shape[3] != 4
        ):
            raise ValueError(
                "dR must be shaped as (BHC, n_splits, L, 4) matching dlogprefix."
            )
        if cutlass.const_expr(
            mDM.shape[1] != 1 or mDM.shape[2] != self.L or mDM.shape[3] != 2
        ):
            raise ValueError("dM output must be shaped as (BHC, n_splits, L, 2).")
        if cutlass.const_expr(
            mDKprev.shape[1] != 1 or mDKprev.shape[2] != self.L or mDKprev.shape[3] != 2
        ):
            raise ValueError("dKprev output must be shaped as (BHC, n_splits, L, 2).")
        if cutlass.const_expr(
            mDKcurr.shape[1] != 1 or mDKcurr.shape[2] != self.L or mDKcurr.shape[3] != 2
        ):
            raise ValueError("dKcurr output must be shaped as (BHC, n_splits, L, 2).")

    def _launch_main_kernel(
        self,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        mDR: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        launch_kwargs = {
            "grid": self._grid_dim(mM, mDLogPrefix),
            "block": [self.num_threads, 1, 1],
        }
        if stream is not None:
            launch_kwargs["stream"] = stream

        self.kernel(
            mM,
            mK,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
            mDR,
            mDM,
            mDKprev,
            mDKcurr,
        ).launch(**launch_kwargs)

    @cute.jit
    def __call__(
        self,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        mDR: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
    ):
        self._validate_operands(
            mM, mK, mDLogPrefix, mDMprev, mDMcurr, mDR, mDM, mDKprev, mDKcurr
        )
        self._launch_main_kernel(
            mM, mK, mDLogPrefix, mDMprev, mDMcurr, mDR, mDM, mDKprev, mDKcurr
        )

    @cute.jit
    def call_on_stream(
        self,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        mDR: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self._validate_operands(
            mM, mK, mDLogPrefix, mDMprev, mDMcurr, mDR, mDM, mDKprev, mDKcurr
        )
        self._launch_main_kernel(
            mM,
            mK,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
            mDR,
            mDM,
            mDKprev,
            mDKcurr,
            stream=stream,
        )

    # Reverse scan helpers
    @cute.jit
    def _lane_info(self, batch_head_chunks: int):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, split, _ = cute.arch.block_idx()
        bhc = block_x * self.num_threads + tidx
        bhc_valid = cute.elem_less(bhc, batch_head_chunks)
        bhc_safe = cutlass.min(bhc, batch_head_chunks - cutlass.Int32(1))
        return split, bhc_valid, bhc_safe

    @cute.jit
    def _load_transition_state(
        self,
        mM: cute.Tensor,
        *,
        bhc_safe: int,
        t: int,
    ):
        transition_re = cutlass.Float32(mM[bhc_safe, t, 0])
        transition_im = cutlass.Float32(mM[bhc_safe, t, 1])
        magnitude_sq = transition_re * transition_re + transition_im * transition_im
        inv_magnitude = cutlass.Float32(
            cute.math.rsqrt(magnitude_sq + cutlass.Float32(EPS_NORM))
        )
        magnitude = magnitude_sq * inv_magnitude
        unit_re = transition_re * inv_magnitude
        unit_im = transition_im * inv_magnitude
        return transition_re, transition_im, magnitude_sq, magnitude, unit_re, unit_im

    @cute.jit
    def _normalize_phase_pair(self, phase_re, phase_im):
        phase_norm_sq = phase_re * phase_re + phase_im * phase_im
        inv_phase_norm = cutlass.Float32(
            cute.math.rsqrt(phase_norm_sq + cutlass.Float32(EPS_NORM))
        )
        return phase_re * inv_phase_norm, phase_im * inv_phase_norm

    @cute.jit
    def _compute_total_chunk_phase_product(
        self,
        mM: cute.Tensor,
        *,
        bhc_safe: int,
    ):
        total_phase_re = cutlass.Float32(1.0)
        total_phase_im = cutlass.Float32(0.0)
        for t_it in cutlass.range(self.L, unroll=1):
            t = (self.L - 1) - t_it
            _, _, _, _, unit_re, unit_im = self._load_transition_state(
                mM,
                bhc_safe=bhc_safe,
                t=t,
            )
            total_phase_re, total_phase_im = complex_mul(
                total_phase_re, total_phase_im, unit_re, unit_im
            )
            total_phase_re, total_phase_im = self._normalize_phase_pair(
                total_phase_re, total_phase_im
            )
        return total_phase_re, total_phase_im

    @cute.jit
    def _load_tap_coefficients(
        self,
        mK: cute.Tensor,
        *,
        bhc_safe: int,
        t: int,
    ):
        key_prev_re = cutlass.Float32(mK[bhc_safe, t, 0, 0])
        key_prev_im = cutlass.Float32(mK[bhc_safe, t, 0, 1])
        key_curr_re = cutlass.Float32(mK[bhc_safe, t, 1, 0])
        key_curr_im = cutlass.Float32(mK[bhc_safe, t, 1, 1])
        return key_prev_re, key_prev_im, key_curr_re, key_curr_im

    @cute.jit
    def _load_tap_partials(
        self,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        *,
        bhc_safe: int,
        split: int,
        t: int,
    ):
        if cutlass.const_expr(self.assume_dmprev_zero):
            dmprev_re = cutlass.Float32(0.0)
            dmprev_im = cutlass.Float32(0.0)
        else:
            dmprev_re = cutlass.Float32(mDMprev[bhc_safe, split, t, 0])
            dmprev_im = cutlass.Float32(mDMprev[bhc_safe, split, t, 1])
        dmcurr_re = cutlass.Float32(mDMcurr[bhc_safe, split, t, 0])
        dmcurr_im = cutlass.Float32(mDMcurr[bhc_safe, split, t, 1])
        return dmprev_re, dmprev_im, dmcurr_re, dmcurr_im

    @cute.jit
    def _reconstruct_prefix_phase(
        self,
        total_phase_re,
        total_phase_im,
        suffix_phase_re,
        suffix_phase_im,
    ):
        prefix_phase_re, prefix_phase_im = complex_mul(
            total_phase_re, total_phase_im, suffix_phase_re, -suffix_phase_im
        )
        return self._normalize_phase_pair(prefix_phase_re, prefix_phase_im)

    @cute.jit
    def _accumulate_phase_matrix_partials(
        self,
        mDR: cute.Tensor,
        *,
        bhc_safe: int,
        split: int,
        t: int,
        dmprev_re,
        dmprev_im,
        dmcurr_re,
        dmcurr_im,
        key_prev_re,
        key_prev_im,
        key_curr_re,
        key_curr_im,
    ):
        phase_matrix_vjp_00 = dmprev_re * key_prev_re + dmcurr_re * key_curr_re
        phase_matrix_vjp_01 = dmprev_re * key_prev_im + dmcurr_re * key_curr_im
        phase_matrix_vjp_10 = dmprev_im * key_prev_re + dmcurr_im * key_curr_re
        phase_matrix_vjp_11 = dmprev_im * key_prev_im + dmcurr_im * key_curr_im

        phase_matrix_vjp_00 = phase_matrix_vjp_00 + cutlass.Float32(
            mDR[bhc_safe, split, t, 0]
        )
        phase_matrix_vjp_01 = phase_matrix_vjp_01 + cutlass.Float32(
            mDR[bhc_safe, split, t, 1]
        )
        phase_matrix_vjp_10 = phase_matrix_vjp_10 + cutlass.Float32(
            mDR[bhc_safe, split, t, 2]
        )
        phase_matrix_vjp_11 = phase_matrix_vjp_11 + cutlass.Float32(
            mDR[bhc_safe, split, t, 3]
        )
        return (
            phase_matrix_vjp_00,
            phase_matrix_vjp_01,
            phase_matrix_vjp_10,
            phase_matrix_vjp_11,
        )

    @cute.jit
    def _compute_raw_tap_grads(
        self,
        prefix_phase_re,
        prefix_phase_im,
        dmprev_re,
        dmprev_im,
        dmcurr_re,
        dmcurr_im,
    ):
        dkprev_re = prefix_phase_re * dmprev_re + prefix_phase_im * dmprev_im
        dkprev_im = prefix_phase_im * dmprev_re - prefix_phase_re * dmprev_im
        dkcurr_re = prefix_phase_re * dmcurr_re + prefix_phase_im * dmcurr_im
        dkcurr_im = prefix_phase_im * dmcurr_re - prefix_phase_re * dmcurr_im
        return dkprev_re, dkprev_im, dkcurr_re, dkcurr_im

    @cute.jit
    def _accumulate_phase_scan_input(
        self,
        phase_matrix_vjp_00,
        phase_matrix_vjp_01,
        phase_matrix_vjp_10,
        phase_matrix_vjp_11,
        phase_scan_carry_re,
        phase_scan_carry_im,
    ):
        phase_vjp_local_re = phase_matrix_vjp_00 - phase_matrix_vjp_11
        phase_vjp_local_im = phase_matrix_vjp_01 + phase_matrix_vjp_10
        phase_vjp_re = phase_vjp_local_re + phase_scan_carry_re
        phase_vjp_im = phase_vjp_local_im + phase_scan_carry_im
        return phase_vjp_re, phase_vjp_im

    @cute.jit
    def _advance_phase_scan_state(
        self,
        total_phase_re,
        total_phase_im,
        suffix_phase_re,
        suffix_phase_im,
        unit_re,
        unit_im,
        phase_vjp_re,
        phase_vjp_im,
    ):
        suffix_prev_re, suffix_prev_im = complex_mul(
            suffix_phase_re, suffix_phase_im, unit_re, unit_im
        )
        suffix_prev_re, suffix_prev_im = self._normalize_phase_pair(
            suffix_prev_re, suffix_prev_im
        )
        prefix_prev_re, prefix_prev_im = complex_mul(
            total_phase_re, total_phase_im, suffix_prev_re, -suffix_prev_im
        )

        unit_phase_vjp_re = (
            phase_vjp_re * prefix_prev_re + phase_vjp_im * prefix_prev_im
        )
        unit_phase_vjp_im = (
            -phase_vjp_re * prefix_prev_im + phase_vjp_im * prefix_prev_re
        )
        next_phase_scan_carry_re = phase_vjp_re * unit_re + phase_vjp_im * unit_im
        next_phase_scan_carry_im = -phase_vjp_re * unit_im + phase_vjp_im * unit_re
        return (
            suffix_prev_re,
            suffix_prev_im,
            unit_phase_vjp_re,
            unit_phase_vjp_im,
            next_phase_scan_carry_re,
            next_phase_scan_carry_im,
        )

    @cute.jit
    def _project_phase_vjp_to_transition(
        self,
        unit_phase_vjp_re,
        unit_phase_vjp_im,
        unit_re,
        unit_im,
        magnitude,
    ):
        unit_dot_vjp = unit_re * unit_phase_vjp_re + unit_im * unit_phase_vjp_im
        dphase_transition_re = (unit_phase_vjp_re - unit_re * unit_dot_vjp) / magnitude
        dphase_transition_im = (unit_phase_vjp_im - unit_im * unit_dot_vjp) / magnitude
        return dphase_transition_re, dphase_transition_im

    @cute.jit
    def _accumulate_log_magnitude_grad(
        self,
        mDLogPrefix: cute.Tensor,
        *,
        bhc_safe: int,
        split: int,
        t: int,
        dlogmag_running,
        transition_re,
        transition_im,
        magnitude_sq,
    ):
        next_dlogmag_running = dlogmag_running + cutlass.Float32(
            mDLogPrefix[bhc_safe, split, t]
        )
        scale = (
            cutlass.Float32(0.5)
            * next_dlogmag_running
            / (magnitude_sq + cutlass.Float32(EPS_NORM))
        )
        return next_dlogmag_running, scale * transition_re, scale * transition_im

    @cute.jit
    def _store_step_outputs(
        self,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        *,
        bhc_safe: int,
        split: int,
        t: int,
        bhc_valid,
        dkprev_re,
        dkprev_im,
        dkcurr_re,
        dkcurr_im,
        dphase_transition_re,
        dphase_transition_im,
        dlogmag_transition_re,
        dlogmag_transition_im,
    ):
        if bhc_valid:
            mDKprev[bhc_safe, split, t, 0] = dkprev_re
            mDKprev[bhc_safe, split, t, 1] = dkprev_im
            mDKcurr[bhc_safe, split, t, 0] = dkcurr_re
            mDKcurr[bhc_safe, split, t, 1] = dkcurr_im
            mDM[bhc_safe, split, t, 0] = dphase_transition_re + dlogmag_transition_re
            mDM[bhc_safe, split, t, 1] = dphase_transition_im + dlogmag_transition_im

    # Kernel entrypoint
    @cute.kernel
    def kernel(
        self,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        mDR: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
    ):
        batch_head_chunks = cute.size(mM.shape[0])
        split, bhc_valid, bhc_safe = self._lane_info(batch_head_chunks)

        # Reconstruct the chunk-wide unit-phase product once for this lane.
        total_phase_re, total_phase_im = self._compute_total_chunk_phase_product(
            mM,
            bhc_safe=bhc_safe,
        )

        # Reverse scan state for the phase suffix, the carried phase VJP, and
        # the reverse inclusive cumsum over the log-magnitude channel.
        suffix_phase_re = cutlass.Float32(1.0)
        suffix_phase_im = cutlass.Float32(0.0)
        phase_scan_carry_re = cutlass.Float32(0.0)
        phase_scan_carry_im = cutlass.Float32(0.0)
        dlogmag_running = cutlass.Float32(0.0)

        for t_it in cutlass.range(self.L, unroll=1):
            t = (self.L - 1) - t_it

            # Load the raw transition, tap coefficients, and transformed tap
            # partials for this reverse-time step.
            (
                transition_re,
                transition_im,
                magnitude_sq,
                magnitude,
                unit_re,
                unit_im,
            ) = self._load_transition_state(
                mM,
                bhc_safe=bhc_safe,
                t=t,
            )
            key_prev_re, key_prev_im, key_curr_re, key_curr_im = (
                self._load_tap_coefficients(
                    mK,
                    bhc_safe=bhc_safe,
                    t=t,
                )
            )
            dmprev_re, dmprev_im, dmcurr_re, dmcurr_im = self._load_tap_partials(
                mDMprev,
                mDMcurr,
                bhc_safe=bhc_safe,
                split=split,
                t=t,
            )

            # Reconstruct the prefix phase seen by the tap matrix at time t.
            prefix_phase_re, prefix_phase_im = self._reconstruct_prefix_phase(
                total_phase_re,
                total_phase_im,
                suffix_phase_re,
                suffix_phase_im,
            )

            # Accumulate the local phase-matrix VJP sources and map the
            # transformed partials back to raw tap gradients.
            (
                phase_matrix_vjp_00,
                phase_matrix_vjp_01,
                phase_matrix_vjp_10,
                phase_matrix_vjp_11,
            ) = self._accumulate_phase_matrix_partials(
                mDR,
                bhc_safe=bhc_safe,
                split=split,
                t=t,
                dmprev_re=dmprev_re,
                dmprev_im=dmprev_im,
                dmcurr_re=dmcurr_re,
                dmcurr_im=dmcurr_im,
                key_prev_re=key_prev_re,
                key_prev_im=key_prev_im,
                key_curr_re=key_curr_re,
                key_curr_im=key_curr_im,
            )
            dkprev_re, dkprev_im, dkcurr_re, dkcurr_im = self._compute_raw_tap_grads(
                prefix_phase_re,
                prefix_phase_im,
                dmprev_re,
                dmprev_im,
                dmcurr_re,
                dmcurr_im,
            )

            # Push the phase VJP through the reverse unit-phase scan.
            phase_vjp_re, phase_vjp_im = self._accumulate_phase_scan_input(
                phase_matrix_vjp_00,
                phase_matrix_vjp_01,
                phase_matrix_vjp_10,
                phase_matrix_vjp_11,
                phase_scan_carry_re,
                phase_scan_carry_im,
            )
            (
                suffix_prev_re,
                suffix_prev_im,
                unit_phase_vjp_re,
                unit_phase_vjp_im,
                phase_scan_carry_re,
                phase_scan_carry_im,
            ) = self._advance_phase_scan_state(
                total_phase_re,
                total_phase_im,
                suffix_phase_re,
                suffix_phase_im,
                unit_re,
                unit_im,
                phase_vjp_re,
                phase_vjp_im,
            )
            (
                dphase_transition_re,
                dphase_transition_im,
            ) = self._project_phase_vjp_to_transition(
                unit_phase_vjp_re,
                unit_phase_vjp_im,
                unit_re,
                unit_im,
                magnitude,
            )

            # Reverse inclusive cumsum of dlogprefix contributes the
            # log-magnitude
            # channel of dM at the same time step.
            (
                dlogmag_running,
                dlogmag_transition_re,
                dlogmag_transition_im,
            ) = self._accumulate_log_magnitude_grad(
                mDLogPrefix,
                bhc_safe=bhc_safe,
                split=split,
                t=t,
                dlogmag_running=dlogmag_running,
                transition_re=transition_re,
                transition_im=transition_im,
                magnitude_sq=magnitude_sq,
            )
            self._store_step_outputs(
                mDM,
                mDKprev,
                mDKcurr,
                bhc_safe=bhc_safe,
                split=split,
                t=t,
                bhc_valid=bhc_valid,
                dkprev_re=dkprev_re,
                dkprev_im=dkprev_im,
                dkcurr_re=dkcurr_re,
                dkcurr_im=dkcurr_im,
                dphase_transition_re=dphase_transition_re,
                dphase_transition_im=dphase_transition_im,
                dlogmag_transition_re=dlogmag_transition_re,
                dlogmag_transition_im=dlogmag_transition_im,
            )

            suffix_phase_re = suffix_prev_re
            suffix_phase_im = suffix_prev_im


__all__ = ["ChunkScanBwdParamScanAmpere"]
