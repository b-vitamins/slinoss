"""CuTe backward parameter kernels for ``v2x2ssd``.

``BwdParamScanAmpere`` writes the main reverse metadata and tap gradients.
``BwdParamIncrementAccumulatorAmpere`` consumes the state-gradient reductions
and adds its contribution into the same public parameter-gradient storage
through the matching alias layout.

Tensor contracts:

- ``M``: ``(BHC, L, 2)`` fp32 packed complex transitions
- ``K``: ``(BHC, L, 2, 2)`` fp32 packed complex taps for the previous/current
  diagonal passes
- ``dlogprefix``: ``(BHC, 1, L)`` fp32 reverse log-prefix partials
- ``dMprev`` / ``dMcurr``: ``(BHC, 1, L, 2)`` fp32 transformed tap
  partials before the reverse metadata scan
- ``dR``: ``(BHC, 1, L, 4)`` fp32 partials of the real phase matrix
- ``dM``: ``(BHC, 1, L, 2)`` fp32 reverse metadata output
- ``dKprev`` / ``dKcurr``: ``(BHC, 1, L, 2)`` fp32 raw tap gradients
"""

from dataclasses import dataclass
from typing import ClassVar

from cuda.bindings import driver as cuda
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import complex_mul

# Keep phase normalization numerically well behaved on zero-padded tail steps
# without materially perturbing valid chunk-local transitions.
EPS_NORM = 1.0e-20


class BwdParamScanAmpere:
    """Reverse metadata scan and raw tap gradients."""

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
    ) -> tuple[int, int, int]:
        batch_head_chunks = cute.size(mM.shape[0])
        return (
            cute.ceil_div(batch_head_chunks, self.num_threads),
            1,
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
            raise ValueError("dlogprefix must be shaped as (BHC, 1, L) matching M.")
        if cutlass.const_expr(
            mDMprev.shape[1] != 1
            or mDMprev.shape[2] != self.L
            or mDMprev.shape[3] != 2
            or mDMcurr.shape[1] != 1
            or mDMcurr.shape[2] != self.L
            or mDMcurr.shape[3] != 2
        ):
            raise ValueError(
                "dMprev/dMcurr must be shaped as (BHC, 1, L, 2) matching dlogprefix."
            )
        if cutlass.const_expr(
            mDR.shape[1] != 1 or mDR.shape[2] != self.L or mDR.shape[3] != 4
        ):
            raise ValueError("dR must be shaped as (BHC, 1, L, 4) matching dlogprefix.")
        if cutlass.const_expr(
            mDM.shape[1] != 1 or mDM.shape[2] != self.L or mDM.shape[3] != 2
        ):
            raise ValueError("dM output must be shaped as (BHC, 1, L, 2).")
        if cutlass.const_expr(
            mDKprev.shape[1] != 1 or mDKprev.shape[2] != self.L or mDKprev.shape[3] != 2
        ):
            raise ValueError("dKprev output must be shaped as (BHC, 1, L, 2).")
        if cutlass.const_expr(
            mDKcurr.shape[1] != 1 or mDKcurr.shape[2] != self.L or mDKcurr.shape[3] != 2
        ):
            raise ValueError("dKcurr output must be shaped as (BHC, 1, L, 2).")

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
            "grid": self._grid_dim(mM),
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

    # State-gradient accumulator.
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


@dataclass(frozen=True)
class BwdParamIncrementAccumulatorLayoutBundle:
    strict_suffix_state_layout: object


@dataclass(frozen=True)
class BwdParamIncrementAccumulatorKernelBundle:
    layouts: BwdParamIncrementAccumulatorLayoutBundle
    shared_storage_cls: object
    smem_bytes: int


@dataclass(frozen=True)
class BwdParamIncrementAccumulatorSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class BwdParamIncrementAccumulatorAmpere:
    """Ampere one-warp backward state-gradient parameter accumulator.

    One thread owns one ``BHC`` lane. The kernel first materializes the strict
    suffix transition product ``suffix_{>t}`` for every time step into shared
    memory, then walks the chunk in forward time to recover:

    - ``dKprev`` from the boundary-seeded previous-tap path
    - ``dKcurr`` from the per-``D``-tile reduced current-tap path
    - ``dM`` from the reverse suffix-state carry entering each transition
    """

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], BwdParamIncrementAccumulatorSupportInfo]
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
    ) -> BwdParamIncrementAccumulatorSupportInfo:
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

        info = BwdParamIncrementAccumulatorSupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            required_smem_bytes=self._required_smem_bytes(),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def can_implement(self, *, device_index: int | None = None) -> bool:
        return self.support_info(device_index=device_index).supported

    def _make_layout_bundle(self) -> BwdParamIncrementAccumulatorLayoutBundle:
        return BwdParamIncrementAccumulatorLayoutBundle(
            strict_suffix_state_layout=self._strict_suffix_state_smem_layout(),
        )

    def _make_shared_storage(
        self,
        layouts: BwdParamIncrementAccumulatorLayoutBundle,
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

    def _make_kernel_bundle(self) -> BwdParamIncrementAccumulatorKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage_cls = self._make_shared_storage(layouts)
        return BwdParamIncrementAccumulatorKernelBundle(
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
                mDKprev[0, time_step, batch_head_chunk_safe] = (
                    mDKprev[0, time_step, batch_head_chunk_safe].to(cutlass.Float32)
                    + d_kprev_re
                )
                mDKprev[1, time_step, batch_head_chunk_safe] = (
                    mDKprev[1, time_step, batch_head_chunk_safe].to(cutlass.Float32)
                    + d_kprev_im
                )
                mDKcurr[0, time_step, batch_head_chunk_safe] = (
                    mDKcurr[0, time_step, batch_head_chunk_safe].to(cutlass.Float32)
                    + d_kcurr_re
                )
                mDKcurr[1, time_step, batch_head_chunk_safe] = (
                    mDKcurr[1, time_step, batch_head_chunk_safe].to(cutlass.Float32)
                    + d_kcurr_im
                )
                mDM[0, time_step, batch_head_chunk_safe] = (
                    mDM[0, time_step, batch_head_chunk_safe].to(cutlass.Float32)
                    + d_transition_re
                )
                mDM[1, time_step, batch_head_chunk_safe] = (
                    mDM[1, time_step, batch_head_chunk_safe].to(cutlass.Float32)
                    + d_transition_im
                )

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


__all__ = ["BwdParamIncrementAccumulatorAmpere", "BwdParamScanAmpere"]
