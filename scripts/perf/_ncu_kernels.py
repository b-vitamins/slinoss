from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import sys
from typing import Callable, cast

import cutlass.cute as cute
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import (  # noqa: E402
    PerfConfig,
    dtype_from_str,
    make_inputs,
)
from _nextchar import DEFAULT_NEXTCHAR_PERF_CONFIG  # noqa: E402
from slinoss.layers import SLinOSSScanPrep  # noqa: E402
from slinoss.ops.scanprep.cute.common import (  # noqa: E402
    COEFF_AUX_FIELDS,
    make_ptr_arg,
)
from slinoss.ops.scanprep.cute.fwd import scanprep_fwd_cute_with_aux  # noqa: E402
from slinoss.ops.scanprep.cute.kernels.bwd import ScanPrepBwdFused  # noqa: E402
from slinoss.ops.scanprep.cute.kernels.fwd import ScanPrepFwdFused  # noqa: E402
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment import (  # noqa: E402
    compile_chunk_increment_bwd_kernels,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import (  # noqa: E402
    compile_chunk_scan_bwd_kernels,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import (  # noqa: E402
    compile_state_passing_bwd_kernel,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd import (  # noqa: E402
    _compile_chunk_increment_kernel_impl,
    _compile_chunk_scan_kernel_impl,
    _compile_state_passing_kernel_impl,
    chunk_increment_cute,
)


DEFAULT_V2_BATCH = int(DEFAULT_NEXTCHAR_PERF_CONFIG.batch_size)
DEFAULT_V2_HEADS = int(DEFAULT_NEXTCHAR_PERF_CONFIG.n_heads)
DEFAULT_V2_T = int(DEFAULT_NEXTCHAR_PERF_CONFIG.block_size)
DEFAULT_V2_N = int(DEFAULT_NEXTCHAR_PERF_CONFIG.d_state)
DEFAULT_V2_P = int(DEFAULT_NEXTCHAR_PERF_CONFIG.d_head)
DEFAULT_V2_CHUNK = int(DEFAULT_NEXTCHAR_PERF_CONFIG.chunk_size)
DEFAULT_V2_DTYPE = DEFAULT_NEXTCHAR_PERF_CONFIG.dtype


@dataclass(frozen=True)
class ScanPrepPerfConfig:
    batch: int = DEFAULT_V2_BATCH
    heads: int = DEFAULT_V2_HEADS
    T: int = DEFAULT_V2_T
    P: int = DEFAULT_V2_P
    N: int = DEFAULT_V2_N
    dtype: torch.dtype = DEFAULT_NEXTCHAR_PERF_CONFIG.dtype
    device: str = "cuda"
    seed: int = 0
    normalize_bc: bool = True
    pack_warps_per_block: int = 8
    coeff_block_size_fwd: int = 256
    coeff_block_size_bwd: int = 512
    scale_block_size: int = 256
    bias_block_size: int = 224

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass
class KernelRunner:
    name: str
    effective_bytes: int
    launch: Callable[[], None]
    prepare: Callable[[], None]
    note: str | None = None


KERNEL_ORDER = (
    "scanprep_fwd",
    "scanprep_bwd",
    "chunk_increment_fwd",
    "state_passing_fwd",
    "chunk_scan_fwd",
    "chunk_increment_bwd_boundary",
    "chunk_increment_bwd_db",
    "chunk_increment_bwd_du",
    "chunk_increment_bwd_param",
    "chunk_scan_bwd_dz0",
    "chunk_scan_bwd_du",
    "chunk_scan_bwd_db",
    "chunk_scan_bwd_dcdr",
    "chunk_scan_bwd_dlp",
    "chunk_scan_bwd_param",
    "state_passing_bwd",
)


def list_kernel_names() -> tuple[str, ...]:
    return KERNEL_ORDER


def _noop() -> None:
    return None


def _closure_vars(fn: Callable[..., object]) -> dict[str, object]:
    names = fn.__code__.co_freevars
    cells = fn.__closure__ or ()
    return {name: cell.cell_contents for name, cell in zip(names, cells, strict=False)}


def _closure_var(fn: Callable[..., object], name: str) -> object:
    vars_map = _closure_vars(fn)
    if name not in vars_map:
        raise RuntimeError(f"Missing closure variable '{name}' for function {fn}.")
    return vars_map[name]


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _shape_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return int(prod(int(dim) for dim in shape)) * int(
        torch.empty((), dtype=dtype).element_size()
    )


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _tc_input_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    if dtype == torch.float32:
        return torch.float16
    raise TypeError(f"Unsupported dtype: {dtype}")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return dtype_from_str(dtype_name)


def _n_chunks(T: int, chunk_size: int) -> int:
    return (int(T) + int(chunk_size) - 1) // int(chunk_size)


def _t_pad(T: int, chunk_size: int) -> int:
    return _n_chunks(T, chunk_size) * int(chunk_size)


def _build_scanprep_fwd_runner(cfg: ScanPrepPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    prep = SLinOSSScanPrep(
        n_heads=cfg.heads,
        d_state=cfg.N,
        d_head=cfg.P,
        normalize_bc=cfg.normalize_bc,
        device=device,
    ).to(dtype=dtype)

    value = torch.randn(
        (cfg.batch, cfg.T, cfg.heads * cfg.P), device=device, dtype=dtype
    )
    params = torch.randn((cfg.batch, cfg.T, cfg.heads * 13), device=device, dtype=dtype)
    bc = torch.randn(
        (cfg.batch, cfg.T, cfg.heads, 4, cfg.N), device=device, dtype=dtype
    )

    U = torch.empty((cfg.batch, cfg.heads, cfg.T, cfg.P), device=device, dtype=dtype)
    M = torch.empty(
        (cfg.batch, cfg.heads, cfg.T, 2), device=device, dtype=torch.float32
    )
    K = torch.empty(
        (cfg.batch, cfg.heads, cfg.T, 2, 2), device=device, dtype=torch.float32
    )
    B = torch.empty(
        (cfg.batch, cfg.heads, cfg.T, 2 * cfg.N), device=device, dtype=dtype
    )
    C = torch.empty_like(B)
    rms_inv = torch.empty(
        (cfg.batch, cfg.heads, cfg.T, 4), device=device, dtype=torch.float32
    )
    coeff_aux = torch.empty(
        (cfg.batch, cfg.heads, COEFF_AUX_FIELDS, cfg.T),
        device=device,
        dtype=torch.float32,
    )

    if cfg.normalize_bc:
        assert prep.b_scale is not None
        assert prep.c_scale is not None
        b_scale = prep.b_scale.detach().contiguous()
        c_scale = prep.c_scale.detach().contiguous()
    else:
        b_scale = torch.empty((cfg.heads, 2, cfg.N), device=device, dtype=dtype)
        c_scale = torch.empty((cfg.heads, 2, cfg.N), device=device, dtype=dtype)

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

    params_in_stride = (
        int(params.stride()[0]),
        int(params.stride()[1]),
        int(params.stride()[2]),
    )

    compiled = cute.compile(
        ScanPrepFwdFused(
            spec=(cfg.batch, cfg.T, cfg.heads, cfg.P, cfg.N),
            params_in_stride=params_in_stride,
            normalize_bc=cfg.normalize_bc,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            theta_bound=prep.theta_bound,
            k_max=prep.k_max,
            eps=prep.eps,
            pack_warps_per_block=cfg.pack_warps_per_block,
            coeff_block_size=cfg.coeff_block_size_fwd,
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

    def launch() -> None:
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

    effective_bytes = _tensor_bytes(
        value,
        bc,
        b_scale,
        c_scale,
        params,
        prep.dt_bias.detach(),
        prep.gamma_bias.detach(),
        prep.omega_bias.detach(),
        prep.mix_r_bias.detach(),
        prep.mix_theta_bias.detach(),
        prep.mix_k_prev_bias.detach(),
        prep.mix_k_curr_bias.detach(),
        U,
        B,
        C,
        M,
        K,
        rms_inv,
        coeff_aux,
    )
    return KernelRunner(
        name="scanprep_fwd",
        effective_bytes=effective_bytes,
        launch=launch,
        prepare=_noop,
    )


def _build_scanprep_bwd_runner(cfg: ScanPrepPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    prep = SLinOSSScanPrep(
        n_heads=cfg.heads,
        d_state=cfg.N,
        d_head=cfg.P,
        normalize_bc=cfg.normalize_bc,
        device=device,
    ).to(dtype=dtype)

    value = torch.randn(
        (cfg.batch, cfg.T, cfg.heads * cfg.P), device=device, dtype=dtype
    )
    params_flat = torch.randn(
        (cfg.batch, cfg.T, cfg.heads * 13), device=device, dtype=dtype
    )
    bc = torch.randn(
        (cfg.batch, cfg.T, cfg.heads, 4, cfg.N), device=device, dtype=dtype
    )
    dU = torch.randn((cfg.batch, cfg.heads, cfg.T, cfg.P), device=device, dtype=dtype)
    dM = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, 2), device=device, dtype=torch.float32
    )
    dK = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, 2, 2), device=device, dtype=torch.float32
    )
    dB = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, 2 * cfg.N), device=device, dtype=dtype
    )
    dC = torch.randn_like(dB)
    du_stride = (
        int(dU.stride()[0]),
        int(dU.stride()[1]),
        int(dU.stride()[2]),
        int(dU.stride()[3]),
    )

    if cfg.normalize_bc:
        assert prep.b_scale is not None
        assert prep.c_scale is not None
        b_scale = prep.b_scale.detach().contiguous()
        c_scale = prep.c_scale.detach().contiguous()
    else:
        b_scale = torch.empty((cfg.heads, 2, cfg.N), device=device, dtype=dtype)
        c_scale = torch.empty((cfg.heads, 2, cfg.N), device=device, dtype=dtype)

    with torch.no_grad():
        _, _, _, _, _, rms_inv, coeff_aux = scanprep_fwd_cute_with_aux(
            value,
            params_flat,
            bc,
            n_heads=cfg.heads,
            d_state=cfg.N,
            d_head=cfg.P,
            normalize_bc=cfg.normalize_bc,
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
            b_scale=b_scale if cfg.normalize_bc else None,
            c_scale=c_scale if cfg.normalize_bc else None,
        )

    value_grad = torch.empty(
        (cfg.batch, cfg.T, cfg.heads * cfg.P), device=device, dtype=dtype
    )
    bc_grad = torch.empty(
        (cfg.batch, cfg.T, cfg.heads, 4, cfg.N), device=device, dtype=dtype
    )
    dparams = torch.empty(
        (cfg.batch, cfg.T, cfg.heads * 13), device=device, dtype=dtype
    )
    scale_tile_count = (cfg.batch * cfg.T + (256 - 1)) // 256
    scale_partial = torch.empty(
        (scale_tile_count, cfg.heads, 4, cfg.N), device=device, dtype=torch.float32
    )
    scale_grad = torch.zeros((cfg.heads, 4, cfg.N), device=device, dtype=torch.float32)
    bias_tile_count = (cfg.batch * cfg.T + (256 - 1)) // 256
    bias_partial = torch.empty(
        (bias_tile_count, cfg.heads, 7), device=device, dtype=torch.float32
    )
    bias_grad = torch.empty((cfg.heads, 7), device=device, dtype=torch.float32)

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
            spec=(cfg.batch, cfg.T, cfg.heads, cfg.P, cfg.N, 13),
            du_stride=du_stride,
            normalize_bc=cfg.normalize_bc,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            theta_bound=prep.theta_bound,
            k_max=prep.k_max,
            eps=prep.eps,
            pack_warps_per_block=cfg.pack_warps_per_block,
            coeff_block_size=cfg.coeff_block_size_bwd,
            scale_block_size=cfg.scale_block_size,
            bias_block_size=cfg.bias_block_size,
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

    def launch() -> None:
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

    effective_bytes = _tensor_bytes(
        dU,
        bc,
        dB,
        dC,
        b_scale,
        c_scale,
        rms_inv,
        coeff_aux,
        dM,
        dK,
        value_grad,
        bc_grad,
        dparams,
        scale_grad,
        bias_grad,
    )
    return KernelRunner(
        name="scanprep_bwd",
        effective_bytes=effective_bytes,
        launch=launch,
        prepare=prepare,
    )


def _build_v2x2ssd_forward_runners(cfg: PerfConfig) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    tensors = make_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    C = tensors["C"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    n_chunks = _n_chunks(cfg.T, cfg.chunk_size)
    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    tc_dtype = _tc_input_dtype(cfg.dtype)

    _compiled_inc, inc_chunk, m_chunk_chunk, launch_inc = (
        _compile_chunk_increment_kernel_impl(
            U,
            M,
            K,
            B,
            U_prev0=U_prev,
            B_prev0=B_prev,
            chunk_size=cfg.chunk_size,
            compute_dtype=torch.float32,
        )
    )
    launch_inc()
    inc = inc_chunk.reshape(cfg.batch, cfg.heads, n_chunks, cfg.P, D)
    m_chunk = m_chunk_chunk.reshape(cfg.batch, cfg.heads, n_chunks, 2)

    _compiled_state, chunk_starts, final_state, launch_state = (
        _compile_state_passing_kernel_impl(
            inc,
            m_chunk,
            initial_states=initial_states,
        )
    )
    launch_state()

    _compiled_scan, out_chunk, _out_view, launch_scan = _compile_chunk_scan_kernel_impl(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    bytes_u = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_bc = _shape_bytes((cfg.batch, cfg.heads, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)

    return {
        "chunk_increment_fwd": KernelRunner(
            name="chunk_increment_fwd",
            effective_bytes=(
                bytes_u
                + bytes_bc
                + bytes_m
                + bytes_k
                + _tensor_bytes(U_prev, B_prev, inc_chunk, m_chunk_chunk)
            ),
            launch=launch_inc,
            prepare=_noop,
        ),
        "state_passing_fwd": KernelRunner(
            name="state_passing_fwd",
            effective_bytes=_tensor_bytes(
                inc, m_chunk, initial_states, chunk_starts, final_state
            ),
            launch=launch_state,
            prepare=_noop,
        ),
        "chunk_scan_fwd": KernelRunner(
            name="chunk_scan_fwd",
            effective_bytes=(
                bytes_u
                + bytes_bc
                + bytes_bc
                + bytes_m
                + bytes_k
                + _tensor_bytes(chunk_starts, U_prev, B_prev, out_chunk)
            ),
            launch=launch_scan,
            prepare=_noop,
        ),
    }


def _build_v2x2ssd_chunk_increment_bwd_runners(
    cfg: PerfConfig,
) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    tensors = make_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    inc, m_chunk = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev0=B_prev,
        U_prev0=U_prev,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)
    (
        _compiled_db,
        _compiled_du,
        _compiled_boundary,
        _compiled_param,
        dB,
        dU,
        dB_prev,
        dU_prev,
        dMsum_part,
        dMp0,
        dM,
        dKprev,
        dKcurr,
        _launch_sequential,
        _launch_overlapped,
    ) = compile_chunk_increment_bwd_kernels(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
        return_launchers=True,
    )
    stage_args = _closure_var(_launch_sequential, "stage_args")
    if not isinstance(stage_args, tuple):
        raise RuntimeError(
            "chunk_increment_bwd launcher closure did not expose stage_args tuple."
        )

    db_args = (
        stage_args[0],
        stage_args[1],
        stage_args[2],
        stage_args[3],
        stage_args[4],
        stage_args[5],
        stage_args[8],
        stage_args[12],
    )
    du_args = (
        stage_args[5],
        stage_args[1],
        stage_args[2],
        stage_args[3],
        stage_args[4],
        stage_args[9],
    )
    boundary_args = (
        stage_args[5],
        stage_args[6],
        stage_args[7],
        stage_args[2],
        stage_args[3],
        stage_args[11],
        stage_args[10],
        stage_args[13],
    )
    param_args = (
        stage_args[2],
        stage_args[3],
        stage_args[4],
        stage_args[12],
        stage_args[13],
        stage_args[14],
        stage_args[15],
        stage_args[16],
        stage_args[17],
    )

    launchers: dict[str, Callable[[], None]] = {
        "chunk_increment_bwd_db": lambda compiled=_compiled_db, args=db_args: compiled(
            *args
        ),
        "chunk_increment_bwd_du": lambda compiled=_compiled_du, args=du_args: compiled(
            *args
        ),
        "chunk_increment_bwd_boundary": lambda compiled=_compiled_boundary,
        args=boundary_args: compiled(*args),
        "chunk_increment_bwd_param": lambda compiled=_compiled_param,
        args=param_args: compiled(*args),
    }
    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    n_chunks = _n_chunks(cfg.T, cfg.chunk_size)
    tc_dtype = _tc_input_dtype(cfg.dtype)
    bytes_u = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_b = _shape_bytes((cfg.batch, cfg.heads, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_d_inc = _shape_bytes(tuple(int(dim) for dim in d_inc.shape), tc_dtype)
    bytes_prev_b = _shape_bytes((cfg.batch, cfg.heads, n_chunks, D), tc_dtype)
    bytes_prev_u = _shape_bytes((cfg.batch, cfg.heads, n_chunks, cfg.P), tc_dtype)
    bytes_kprev = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)

    def prepare_param() -> None:
        launchers["chunk_increment_bwd_db"]()
        launchers["chunk_increment_bwd_boundary"]()

    return {
        "chunk_increment_bwd_db": KernelRunner(
            name="chunk_increment_bwd_db",
            effective_bytes=bytes_u
            + bytes_b
            + bytes_m
            + bytes_k
            + bytes_d_inc
            + _tensor_bytes(dB, dMsum_part),
            launch=launchers["chunk_increment_bwd_db"],
            prepare=_noop,
        ),
        "chunk_increment_bwd_du": KernelRunner(
            name="chunk_increment_bwd_du",
            effective_bytes=bytes_b
            + bytes_m
            + bytes_k
            + bytes_d_inc
            + _tensor_bytes(dU),
            launch=launchers["chunk_increment_bwd_du"],
            prepare=_noop,
        ),
        "chunk_increment_bwd_boundary": KernelRunner(
            name="chunk_increment_bwd_boundary",
            effective_bytes=bytes_d_inc
            + bytes_m
            + bytes_prev_b
            + bytes_prev_u
            + bytes_kprev
            + _tensor_bytes(dU_prev, dB_prev, dMp0),
            launch=launchers["chunk_increment_bwd_boundary"],
            prepare=_noop,
        ),
        "chunk_increment_bwd_param": KernelRunner(
            name="chunk_increment_bwd_param",
            effective_bytes=bytes_m
            + bytes_k
            + _tensor_bytes(dMsum_part, dMp0, d_m_chunk, dM, dKprev, dKcurr),
            launch=launchers["chunk_increment_bwd_param"],
            prepare=prepare_param,
        ),
    }


def _build_v2x2ssd_state_passing_bwd_runners(
    cfg: PerfConfig,
) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    tensors = make_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    inc, m_chunk = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev0=B_prev,
        U_prev0=U_prev,
        compute_dtype=torch.float32,
    )
    _compiled_state, chunk_starts, _final_state, launch_state = (
        _compile_state_passing_kernel_impl(
            inc,
            m_chunk,
            initial_states=initial_states,
        )
    )
    launch_state()

    d_chunk_starts = torch.randn_like(chunk_starts)
    d_final = torch.randn_like(initial_states)
    (
        _compiled,
        d_inc,
        d_m_chunk,
        d_initial,
        launch,
    ) = compile_state_passing_bwd_kernel(
        chunk_starts,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
        return_launcher=True,
    )
    compiled = _closure_var(launch, "compiled")
    dynamic_args = _closure_var(launch, "dynamic_args")
    if not isinstance(dynamic_args, tuple):
        raise RuntimeError(
            "state_passing_bwd launcher closure did not expose dynamic_args tuple."
        )

    def prepare() -> None:
        d_m_chunk.zero_()

    return {
        "state_passing_bwd": KernelRunner(
            name="state_passing_bwd",
            effective_bytes=_tensor_bytes(
                chunk_starts,
                d_chunk_starts,
                d_final,
                m_chunk,
                d_inc,
                d_m_chunk,
                d_initial,
            ),
            launch=lambda compiled=compiled, args=dynamic_args: compiled(*args),
            prepare=prepare,
        ),
    }


def _build_v2x2ssd_chunk_scan_bwd_runners(
    cfg: PerfConfig,
) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    tensors = make_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B"]
    C = tensors["C"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev"]
    U_prev = tensors["U_prev"]

    inc, m_chunk = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev0=B_prev,
        U_prev0=U_prev,
        compute_dtype=torch.float32,
    )
    _compiled_state, chunk_starts, _final_state, launch_state = (
        _compile_state_passing_kernel_impl(
            inc,
            m_chunk,
            initial_states=initial_states,
        )
    )
    launch_state()
    d_out = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    (
        _compiled_dz0,
        _compiled_du,
        _compiled_db,
        _compiled_dcdr,
        _compiled_param,
        dZ0,
        dU,
        dB,
        dU_prev,
        dB_prev,
        dlogp,
        dC,
        dR,
        dM,
        dKprev,
        dKcurr,
        _launch_sequential,
        _launch_overlapped,
    ) = compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
        return_launchers=True,
    )
    raw_launchers = {
        "chunk_scan_bwd_dz0": _closure_var(_launch_sequential, "_launch_dz0"),
        "chunk_scan_bwd_du": _closure_var(_launch_sequential, "_launch_du"),
        "chunk_scan_bwd_db": _closure_var(_launch_sequential, "_launch_db"),
        "chunk_scan_bwd_dcdr": _closure_var(_launch_sequential, "_launch_dcdr"),
        "chunk_scan_bwd_dlp": _closure_var(_launch_sequential, "_launch_dlp"),
        "chunk_scan_bwd_param": _closure_var(_launch_sequential, "_launch_param"),
    }
    launchers: dict[str, Callable[[], None]] = {}
    for key, value in raw_launchers.items():
        if not callable(value):
            raise RuntimeError(
                f"Expected callable closure for {key}, got {type(value)!r}."
            )
        launchers[key] = cast(Callable[[], None], value)

    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    tc_dtype = _tc_input_dtype(cfg.dtype)
    bytes_u = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_b = _shape_bytes((cfg.batch, cfg.heads, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_d_out = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_dphase = _shape_bytes(
        (2, cfg.chunk_size, cfg.batch * cfg.heads * _n_chunks(cfg.T, cfg.chunk_size)),
        torch.float32,
    )

    def prepare_param() -> None:
        launchers["chunk_scan_bwd_db"]()
        launchers["chunk_scan_bwd_dcdr"]()
        launchers["chunk_scan_bwd_dlp"]()

    return {
        "chunk_scan_bwd_dz0": KernelRunner(
            name="chunk_scan_bwd_dz0",
            effective_bytes=bytes_d_out + bytes_b + bytes_m + _tensor_bytes(dZ0),
            launch=launchers["chunk_scan_bwd_dz0"],
            prepare=_noop,
        ),
        "chunk_scan_bwd_du": KernelRunner(
            name="chunk_scan_bwd_du",
            effective_bytes=bytes_u
            + bytes_b
            + bytes_b
            + bytes_m
            + bytes_k
            + bytes_d_out
            + _tensor_bytes(U_prev, B_prev, dU, dB, dU_prev, dB_prev, dlogp, dM, dM),
            launch=launchers["chunk_scan_bwd_du"],
            prepare=_noop,
        ),
        "chunk_scan_bwd_db": KernelRunner(
            name="chunk_scan_bwd_db",
            effective_bytes=bytes_u
            + bytes_b
            + bytes_b
            + bytes_m
            + bytes_k
            + bytes_d_out
            + _tensor_bytes(U_prev, B_prev, dU, dB, dU_prev, dB_prev, dlogp, dM, dM),
            launch=launchers["chunk_scan_bwd_db"],
            prepare=_noop,
        ),
        "chunk_scan_bwd_dcdr": KernelRunner(
            name="chunk_scan_bwd_dcdr",
            effective_bytes=bytes_u
            + bytes_b
            + bytes_b
            + bytes_m
            + bytes_k
            + bytes_d_out
            + _tensor_bytes(U_prev, B_prev, chunk_starts, dC, dlogp, dR)
            + bytes_dphase,
            launch=launchers["chunk_scan_bwd_dcdr"],
            prepare=_noop,
        ),
        "chunk_scan_bwd_dlp": KernelRunner(
            name="chunk_scan_bwd_dlp",
            effective_bytes=bytes_u
            + bytes_b
            + bytes_b
            + bytes_m
            + bytes_k
            + bytes_d_out
            + _tensor_bytes(U_prev, B_prev, dlogp),
            launch=launchers["chunk_scan_bwd_dlp"],
            prepare=_noop,
        ),
        "chunk_scan_bwd_param": KernelRunner(
            name="chunk_scan_bwd_param",
            effective_bytes=bytes_m
            + bytes_k
            + bytes_dphase
            + _tensor_bytes(dlogp, dR, dM, dM, dM, dKprev, dKcurr),
            launch=launchers["chunk_scan_bwd_param"],
            prepare=prepare_param,
            note="integrated stage in chunk_scan_bwd",
        ),
    }


def build_kernel_runners(
    *,
    v2_cfg: PerfConfig | None = None,
    scanprep_cfg: ScanPrepPerfConfig | None = None,
) -> list[KernelRunner]:
    if v2_cfg is None:
        v2_cfg = PerfConfig(
            batch=DEFAULT_V2_BATCH,
            heads=DEFAULT_V2_HEADS,
            T=DEFAULT_V2_T,
            N=DEFAULT_V2_N,
            P=DEFAULT_V2_P,
            chunk_size=DEFAULT_V2_CHUNK,
            dtype=DEFAULT_V2_DTYPE,
            device="cuda",
            seed=0,
        )
    if scanprep_cfg is None:
        scanprep_cfg = ScanPrepPerfConfig()

    runners: dict[str, KernelRunner] = {}
    runners.update(
        {
            "scanprep_fwd": _build_scanprep_fwd_runner(scanprep_cfg),
            "scanprep_bwd": _build_scanprep_bwd_runner(scanprep_cfg),
        }
    )
    runners.update(_build_v2x2ssd_forward_runners(v2_cfg))
    runners.update(_build_v2x2ssd_chunk_increment_bwd_runners(v2_cfg))
    runners.update(_build_v2x2ssd_chunk_scan_bwd_runners(v2_cfg))
    runners.update(_build_v2x2ssd_state_passing_bwd_runners(v2_cfg))
    return [runners[name] for name in KERNEL_ORDER]


def build_kernel_runner(
    name: str,
    *,
    v2_cfg: PerfConfig | None = None,
    scanprep_cfg: ScanPrepPerfConfig | None = None,
) -> KernelRunner:
    if v2_cfg is None:
        v2_cfg = PerfConfig(
            batch=DEFAULT_V2_BATCH,
            heads=DEFAULT_V2_HEADS,
            T=DEFAULT_V2_T,
            N=DEFAULT_V2_N,
            P=DEFAULT_V2_P,
            chunk_size=DEFAULT_V2_CHUNK,
            dtype=DEFAULT_V2_DTYPE,
            device="cuda",
            seed=0,
        )
    if scanprep_cfg is None:
        scanprep_cfg = ScanPrepPerfConfig()

    if name == "scanprep_fwd":
        return _build_scanprep_fwd_runner(scanprep_cfg)
    if name == "scanprep_bwd":
        return _build_scanprep_bwd_runner(scanprep_cfg)
    if name in {
        "chunk_increment_fwd",
        "state_passing_fwd",
        "chunk_scan_fwd",
    }:
        return _build_v2x2ssd_forward_runners(v2_cfg)[name]
    if name in {
        "chunk_increment_bwd_boundary",
        "chunk_increment_bwd_db",
        "chunk_increment_bwd_du",
        "chunk_increment_bwd_param",
    }:
        return _build_v2x2ssd_chunk_increment_bwd_runners(v2_cfg)[name]
    if name in {
        "chunk_scan_bwd_dz0",
        "chunk_scan_bwd_du",
        "chunk_scan_bwd_db",
        "chunk_scan_bwd_dcdr",
        "chunk_scan_bwd_dlp",
        "chunk_scan_bwd_param",
    }:
        return _build_v2x2ssd_chunk_scan_bwd_runners(v2_cfg)[name]
    if name in {"state_passing_bwd"}:
        return _build_v2x2ssd_state_passing_bwd_runners(v2_cfg)[name]
    raise KeyError(f"Unknown kernel runner: {name}")
