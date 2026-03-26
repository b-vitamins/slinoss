#!/usr/bin/env python3
"""Run isolated CuTe scanprep backward launches for Nsight Compute."""

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
from slinoss.ops.scanprep.cute.common import make_ptr_arg  # noqa: E402
from slinoss.ops.scanprep.cute.fwd import scanprep_fwd_cute_with_aux  # noqa: E402
from slinoss.ops.scanprep.cute.kernels.bwd import ScanPrepBwdFused  # noqa: E402


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
    parser.add_argument("--pack-warps-per-block", type=int, default=12)
    parser.add_argument("--coeff-block-size", type=int, default=512)
    parser.add_argument("--scale-block-size", type=int, default=256)
    parser.add_argument("--bias-block-size", type=int, default=224)
    parser.add_argument(
        "--no-normalize-bc",
        action="store_false",
        dest="normalize_bc",
        help="Disable BC normalization to isolate the non-normalized backward path.",
    )
    parser.set_defaults(normalize_bc=True)
    return parser.parse_args()


def _profile_once(fn, *, warmup: int, prepare) -> None:
    for _ in range(max(0, int(warmup))):
        prepare()
        fn()
    torch.cuda.synchronize()
    prepare()
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
    params_flat = torch.randn((batch, t_size, heads * 13), device=device, dtype=dtype)
    bc = torch.randn((batch, t_size, heads, 4, d_state), device=device, dtype=dtype)
    dU = torch.randn((batch, heads, t_size, p_size), device=device, dtype=dtype)
    dM = torch.randn((batch, heads, t_size, 2), device=device, dtype=torch.float32)
    dK = torch.randn((batch, heads, t_size, 2, 2), device=device, dtype=torch.float32)
    dB = torch.randn((batch, heads, t_size, 2 * d_state), device=device, dtype=dtype)
    dC = torch.randn((batch, heads, t_size, 2 * d_state), device=device, dtype=dtype)
    du_stride = (
        int(dU.stride()[0]),
        int(dU.stride()[1]),
        int(dU.stride()[2]),
        int(dU.stride()[3]),
    )

    if args.normalize_bc:
        assert prep.b_scale is not None
        assert prep.c_scale is not None
        b_scale = prep.b_scale.detach().contiguous()
        c_scale = prep.c_scale.detach().contiguous()
    else:
        b_scale = torch.empty((heads, 2, d_state), device=device, dtype=dtype)
        c_scale = torch.empty((heads, 2, d_state), device=device, dtype=dtype)

    with torch.no_grad():
        _, _, _, _, _, rms_inv, coeff_aux = scanprep_fwd_cute_with_aux(
            value,
            params_flat,
            bc,
            n_heads=heads,
            d_state=d_state,
            d_head=p_size,
            normalize_bc=args.normalize_bc,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            theta_bound=prep.theta_bound,
            k_max=prep.k_max,
            eps=prep.eps,
            dt_bias=prep.dt_bias.detach(),
            gamma_bias=prep.gamma_bias.detach(),
            omega_bias=prep.omega_bias.detach(),
            mix_r_bias=prep.mix_r_bias.detach(),
            mix_theta_bias=prep.mix_theta_bias.detach(),
            mix_k_prev_bias=prep.mix_k_prev_bias.detach(),
            mix_k_curr_bias=prep.mix_k_curr_bias.detach(),
            b_scale=b_scale if args.normalize_bc else None,
            c_scale=c_scale if args.normalize_bc else None,
        )

    value_grad = torch.empty(
        (batch, t_size, heads * p_size), device=device, dtype=dtype
    )
    bc_grad = torch.empty(
        (batch, t_size, heads, 4, d_state), device=device, dtype=dtype
    )
    dparams = torch.empty((batch, t_size, heads * 13), device=device, dtype=dtype)
    scale_tile_count = (batch * t_size + (256 - 1)) // 256
    scale_partial = torch.empty(
        (scale_tile_count, heads, 4, d_state),
        device=device,
        dtype=torch.float32,
    )
    scale_grad = torch.zeros((heads, 4, d_state), device=device, dtype=torch.float32)
    bias_tile_count = (batch * t_size + (256 - 1)) // 256
    bias_partial = torch.empty(
        (bias_tile_count, heads, 7),
        device=device,
        dtype=torch.float32,
    )
    bias_grad = torch.empty((heads, 7), device=device, dtype=torch.float32)

    du_ptr, _ = make_ptr_arg(dU)
    bc_ptr, _ = make_ptr_arg(bc)
    db_ptr, _ = make_ptr_arg(dB)
    dc_ptr, _ = make_ptr_arg(dC)
    b_scale_ptr, _ = make_ptr_arg(b_scale)
    c_scale_ptr, _ = make_ptr_arg(c_scale)
    rms_inv_ptr, _ = make_ptr_arg(rms_inv)
    coeff_aux_ptr, _ = make_ptr_arg(coeff_aux)
    dm_ptr, _ = make_ptr_arg(dM)
    dk_ptr, _ = make_ptr_arg(dK)
    value_grad_ptr, _ = make_ptr_arg(value_grad)
    bc_grad_ptr, _ = make_ptr_arg(bc_grad)
    dparams_ptr, _ = make_ptr_arg(dparams)
    scale_partial_ptr, _ = make_ptr_arg(scale_partial)
    scale_grad_ptr, _ = make_ptr_arg(scale_grad)
    bias_partial_ptr, _ = make_ptr_arg(bias_partial)
    bias_grad_ptr, _ = make_ptr_arg(bias_grad)

    compiled = cute.compile(
        ScanPrepBwdFused(
            spec=(batch, t_size, heads, p_size, d_state, 13),
            du_stride=du_stride,
            normalize_bc=args.normalize_bc,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            theta_bound=prep.theta_bound,
            k_max=prep.k_max,
            eps=prep.eps,
            pack_warps_per_block=args.pack_warps_per_block,
            coeff_block_size=args.coeff_block_size,
            scale_block_size=args.scale_block_size,
            bias_block_size=args.bias_block_size,
        ),
        du_ptr,
        bc_ptr,
        db_ptr,
        dc_ptr,
        b_scale_ptr,
        c_scale_ptr,
        rms_inv_ptr,
        coeff_aux_ptr,
        dm_ptr,
        dk_ptr,
        value_grad_ptr,
        bc_grad_ptr,
        dparams_ptr,
        scale_partial_ptr,
        scale_grad_ptr,
        bias_partial_ptr,
        bias_grad_ptr,
    )

    def prepare() -> None:
        scale_grad.zero_()
        bias_grad.zero_()

    def run() -> None:
        compiled(
            du_ptr,
            bc_ptr,
            db_ptr,
            dc_ptr,
            b_scale_ptr,
            c_scale_ptr,
            rms_inv_ptr,
            coeff_aux_ptr,
            dm_ptr,
            dk_ptr,
            value_grad_ptr,
            bc_grad_ptr,
            dparams_ptr,
            scale_partial_ptr,
            scale_grad_ptr,
            bias_partial_ptr,
            bias_grad_ptr,
        )

    _profile_once(run, warmup=args.warmup, prepare=prepare)
    checksum = (
        value_grad.sum()
        + bc_grad.sum()
        + dparams.sum()
        + bias_grad.sum()
        + scale_grad.sum()
    )
    print(
        f"B={batch} H={heads} T={t_size} P={p_size} N={d_state} "
        f"dtype={args.dtype} normalize_bc={args.normalize_bc}"
    )
    print(f"checksum={float(checksum):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
