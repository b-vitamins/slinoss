#!/usr/bin/env python3
"""Run isolated CuTe scanprep forward launches for Nsight Compute."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast

import cutlass.cute as cute
import torch

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
    make_fake_tensor_arg,
)
from slinoss.ops.scanprep.cute.kernels.fwd import ScanPrepFwdFused  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=8, help="Batch size.")
    parser.add_argument("--H", type=int, default=12, help="Number of heads.")
    parser.add_argument("--T", type=int, default=2048, help="Sequence length.")
    parser.add_argument("--P", type=int, default=64, help="Head width.")
    parser.add_argument("--N", type=int, default=128, help="State width.")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
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
        device=device,
    ).to(dtype=dtype)

    value = torch.randn((batch, t_size, heads * p_size), device=device, dtype=dtype)
    params = torch.randn(
        (batch, t_size, heads * prep.param_dim), device=device, dtype=dtype
    )
    bc_amp = torch.randn(
        (batch, t_size, heads, prep.bc_param_rows, d_state),
        device=device,
        dtype=dtype,
    )
    bc = prep._parameterize_scan_bc_rows(bc_amp)

    U = torch.empty((batch, heads, t_size, p_size), device=device, dtype=dtype)
    M = torch.empty((batch, heads, t_size, 2), device=device, dtype=torch.float32)
    K = torch.empty((batch, heads, t_size, 2, 2), device=device, dtype=torch.float32)
    B = torch.empty((batch, heads, t_size, 2 * d_state), device=device, dtype=dtype)
    C = torch.empty_like(B)
    coeff_aux = torch.empty(
        (batch, heads, COEFF_AUX_FIELDS, t_size),
        device=device,
        dtype=torch.float32,
    )

    compiled = cute.compile(
        ScanPrepFwdFused(
            h_size=heads,
            g_size=heads,
            p_size=p_size,
            n_size=d_state,
            store_coeff_aux=True,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            theta_init_min=prep.theta_init_min,
            theta_init_max=prep.theta_init_max,
            alpha_min=prep.alpha_min,
            alpha_max=prep.alpha_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            pack_warps_per_block=args.pack_warps_per_block,
            coeff_block_size=args.coeff_block_size,
        ),
        make_fake_tensor_arg(value),
        make_fake_tensor_arg(bc),
        make_fake_tensor_arg(params),
        make_fake_tensor_arg(prep.dt_bias.detach()),
        make_fake_tensor_arg(prep.alpha_bias.detach()),
        make_fake_tensor_arg(prep.theta_mod_bias.detach()),
        make_fake_tensor_arg(prep.theta_bias.detach()),
        make_fake_tensor_arg(cast(torch.Tensor, prep.theta_sign).detach()),
        make_fake_tensor_arg(U),
        make_fake_tensor_arg(B),
        make_fake_tensor_arg(C),
        make_fake_tensor_arg(M),
        make_fake_tensor_arg(K),
        make_fake_tensor_arg(coeff_aux),
        options="--enable-tvm-ffi",
    )

    def run() -> None:
        compiled(
            value,
            bc,
            params,
            prep.dt_bias.detach(),
            prep.alpha_bias.detach(),
            prep.theta_mod_bias.detach(),
            prep.theta_bias.detach(),
            cast(torch.Tensor, prep.theta_sign).detach(),
            U,
            B,
            C,
            M,
            K,
            coeff_aux,
        )

    _profile_once(run, warmup=args.warmup)
    checksum = U.sum() + M.sum() + K.sum() + B.sum() + C.sum() + coeff_aux.sum()
    print(f"B={batch} H={heads} T={t_size} P={p_size} N={d_state} dtype={args.dtype}")
    print(f"checksum={float(checksum):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
