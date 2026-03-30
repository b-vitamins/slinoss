#!/usr/bin/env python3
"""Run isolated CuTe scanprep forward launches for Nsight Compute."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import cutlass.cute as cute

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import dtype_from_str, ensure_cuda, seed_all  # noqa: E402
from slinoss.layers import SLinOSSScanPrep  # noqa: E402
from slinoss.ops.scanprep.cute.common import (  # noqa: E402
    COEFF_AUX_FIELDS,
    make_ptr_arg,
)
from slinoss.ops.scanprep.cute.kernels.fwd import ScanPrepFwdFused  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=8, help="Batch size.")
    parser.add_argument("--H", type=int, default=12, help="Number of heads.")
    parser.add_argument("--T", type=int, default=2048, help="Sequence length.")
    parser.add_argument("--P", type=int, default=64, help="Head width.")
    parser.add_argument("--N", type=int, default=128, help="State width.")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm launches to run before starting the CUDA profiler.",
    )
    parser.add_argument("--pack-warps-per-block", type=int, default=8)
    parser.add_argument("--coeff-block-size", type=int, default=256)
    parser.add_argument(
        "--no-normalize-bc",
        action="store_false",
        dest="normalize_bc",
        help="Disable BC normalization to isolate the non-normalized path.",
    )
    parser.set_defaults(normalize_bc=True)
    return parser.parse_args()


def _profile_once(fn, *, warmup: int) -> None:
    for _ in range(max(0, int(warmup))):
        fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    seed_all(args.seed)

    device = torch.device(args.device)
    dtype = dtype_from_str(args.dtype)
    batch = int(args.B)
    heads = int(args.H)
    t_size = int(args.T)
    p_size = int(args.P)
    d_state = int(args.N)

    prep = SLinOSSScanPrep(
        n_heads=heads,
        d_state=d_state,
        d_head=p_size,
        normalize_bc=args.normalize_bc,
        device=device,
    ).to(dtype=dtype)

    value = torch.randn((batch, t_size, heads * p_size), device=device, dtype=dtype)
    params = torch.randn((batch, t_size, heads * 13), device=device, dtype=dtype)
    bc = torch.randn((batch, t_size, heads, 4, d_state), device=device, dtype=dtype)

    U = torch.empty((batch, heads, t_size, p_size), device=device, dtype=dtype)
    M = torch.empty((batch, heads, t_size, 2), device=device, dtype=torch.float32)
    K = torch.empty((batch, heads, t_size, 2, 2), device=device, dtype=torch.float32)
    B = torch.empty((batch, heads, t_size, 2 * d_state), device=device, dtype=dtype)
    C = torch.empty_like(B)
    rms_inv = torch.empty((batch, heads, t_size, 4), device=device, dtype=torch.float32)
    coeff_aux = torch.empty(
        (batch, heads, COEFF_AUX_FIELDS, t_size),
        device=device,
        dtype=torch.float32,
    )

    if args.normalize_bc:
        assert prep.b_scale is not None
        assert prep.c_scale is not None
        b_scale = prep.b_scale.detach().contiguous()
        c_scale = prep.c_scale.detach().contiguous()
    else:
        b_scale = torch.empty((heads, 2, d_state), device=device, dtype=dtype)
        c_scale = torch.empty((heads, 2, d_state), device=device, dtype=dtype)

    value_ptr, _ = make_ptr_arg(value)
    bc_ptr, _ = make_ptr_arg(bc)
    b_scale_ptr, _ = make_ptr_arg(b_scale)
    c_scale_ptr, _ = make_ptr_arg(c_scale)
    params_ptr, _ = make_ptr_arg(params)
    dt_bias_ptr, _ = make_ptr_arg(prep.dt_bias.detach())
    gamma_bias_ptr, _ = make_ptr_arg(prep.gamma_bias.detach())
    omega_bias_ptr, _ = make_ptr_arg(prep.omega_bias.detach())
    mix_r_bias_ptr, _ = make_ptr_arg(prep.mix_r_bias.detach())
    mix_theta_bias_ptr, _ = make_ptr_arg(prep.mix_theta_bias.detach())
    mix_k_prev_bias_ptr, _ = make_ptr_arg(prep.mix_k_prev_bias.detach())
    mix_k_curr_bias_ptr, _ = make_ptr_arg(prep.mix_k_curr_bias.detach())
    u_ptr, _ = make_ptr_arg(U)
    b_ptr, _ = make_ptr_arg(B)
    c_ptr, _ = make_ptr_arg(C)
    m_ptr, _ = make_ptr_arg(M)
    k_ptr, _ = make_ptr_arg(K)
    rms_inv_ptr, _ = make_ptr_arg(rms_inv)
    coeff_aux_ptr, _ = make_ptr_arg(coeff_aux)

    compiled = cute.compile(
        ScanPrepFwdFused(
            spec=(batch, t_size, heads, p_size, d_state),
            params_in_stride=tuple(int(s) for s in params.stride()),
            normalize_bc=args.normalize_bc,
            store_rms_inv=bool(args.normalize_bc),
            store_coeff_aux=True,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            theta_bound=prep.theta_bound,
            k_max=prep.k_max,
            eps=prep.eps,
            pack_warps_per_block=args.pack_warps_per_block,
            coeff_block_size=args.coeff_block_size,
        ),
        value_ptr,
        bc_ptr,
        b_scale_ptr,
        c_scale_ptr,
        params_ptr,
        dt_bias_ptr,
        gamma_bias_ptr,
        omega_bias_ptr,
        mix_r_bias_ptr,
        mix_theta_bias_ptr,
        mix_k_prev_bias_ptr,
        mix_k_curr_bias_ptr,
        u_ptr,
        b_ptr,
        c_ptr,
        m_ptr,
        k_ptr,
        rms_inv_ptr,
        coeff_aux_ptr,
    )

    def run() -> None:
        compiled(
            value_ptr,
            bc_ptr,
            b_scale_ptr,
            c_scale_ptr,
            params_ptr,
            dt_bias_ptr,
            gamma_bias_ptr,
            omega_bias_ptr,
            mix_r_bias_ptr,
            mix_theta_bias_ptr,
            mix_k_prev_bias_ptr,
            mix_k_curr_bias_ptr,
            u_ptr,
            b_ptr,
            c_ptr,
            m_ptr,
            k_ptr,
            rms_inv_ptr,
            coeff_aux_ptr,
        )

    _profile_once(run, warmup=args.warmup)
    checksum = (
        U.sum()
        + M.sum()
        + K.sum()
        + B.sum()
        + C.sum()
        + coeff_aux.sum()
        + rms_inv.sum()
    )
    print(
        f"B={batch} H={heads} T={t_size} P={p_size} N={d_state} "
        f"dtype={args.dtype} normalize_bc={args.normalize_bc}"
    )
    print(f"checksum={float(checksum):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
