from __future__ import annotations

from dataclasses import dataclass
import inspect
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

from _common import dtype_from_str  # noqa: E402
from _nextchar import DEFAULT_NEXTCHAR_PERF_CONFIG  # noqa: E402
from slinoss.layers import SLinOSSScanPrep  # noqa: E402
from slinoss.ops.scanprep.cute.common import (  # noqa: E402
    COEFF_AUX_FIELDS,
    SCANPREP_PARAM_DIM,
    make_fake_tensor_arg,
)
from slinoss.ops.scanprep.cute.kernels import scanprep_fwd_cute_with_aux  # noqa: E402
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
    _make_chunk_increment_prepared_launch,
    _make_chunk_scan_prepared_launch,
    _make_state_passing_prepared_launch,
    chunk_increment_cute,
)


DEFAULT_V2_BATCH = int(DEFAULT_NEXTCHAR_PERF_CONFIG.batch_size)
DEFAULT_V2_HEADS = int(DEFAULT_NEXTCHAR_PERF_CONFIG.n_heads)
DEFAULT_V2_T = int(DEFAULT_NEXTCHAR_PERF_CONFIG.block_size)
DEFAULT_V2_N = int(DEFAULT_NEXTCHAR_PERF_CONFIG.d_state)
DEFAULT_V2_P = int(DEFAULT_NEXTCHAR_PERF_CONFIG.d_head)
DEFAULT_V2_CHUNK = int(DEFAULT_NEXTCHAR_PERF_CONFIG.chunk_size)
DEFAULT_V2_DTYPE = DEFAULT_NEXTCHAR_PERF_CONFIG.dtype
DEFAULT_V2_BC_GROUPS = int(
    getattr(DEFAULT_NEXTCHAR_PERF_CONFIG, "resolved_bc_groups", DEFAULT_V2_HEADS)
)


@dataclass(frozen=True)
class V2KernelPerfConfig:
    batch: int = DEFAULT_V2_BATCH
    heads: int = DEFAULT_V2_HEADS
    T: int = DEFAULT_V2_T
    N: int = DEFAULT_V2_N
    P: int = DEFAULT_V2_P
    chunk_size: int = DEFAULT_V2_CHUNK
    bc_groups: int | None = None
    dtype: torch.dtype = DEFAULT_NEXTCHAR_PERF_CONFIG.dtype
    device: str = "cuda"
    seed: int = 0

    @property
    def D(self) -> int:
        return 2 * int(self.N)

    @property
    def resolved_bc_groups(self) -> int:
        return int(self.heads if self.bc_groups is None else self.bc_groups)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass(frozen=True)
class ScanPrepPerfConfig:
    batch: int = DEFAULT_V2_BATCH
    heads: int = DEFAULT_V2_HEADS
    T: int = DEFAULT_V2_T
    P: int = DEFAULT_V2_P
    N: int = DEFAULT_V2_N
    bc_groups: int | None = None
    dtype: torch.dtype = DEFAULT_NEXTCHAR_PERF_CONFIG.dtype
    device: str = "cuda"
    seed: int = 0
    pack_warps_per_block: int = 8
    coeff_block_size_fwd: int = 256
    coeff_block_size_bwd: int = 512

    @property
    def resolved_bc_groups(self) -> int:
        return int(self.heads if self.bc_groups is None else self.bc_groups)

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


def _validate_bc_groups(heads: int, bc_groups: int) -> tuple[int, int]:
    resolved_heads = int(heads)
    resolved_groups = int(bc_groups)
    if resolved_groups < 1:
        raise ValueError(f"bc_groups must be positive. Got {resolved_groups}.")
    if resolved_groups > resolved_heads:
        raise ValueError(
            "bc_groups must not exceed the head count. "
            f"Got bc_groups={resolved_groups}, heads={resolved_heads}."
        )
    if resolved_heads % resolved_groups != 0:
        raise ValueError(
            "bc_groups must divide the head count so contiguous group mapping is well-defined. "
            f"Got bc_groups={resolved_groups}, heads={resolved_heads}."
        )
    return resolved_groups, resolved_heads // resolved_groups


def _materialize_grouped_rows(
    tensor: torch.Tensor,
    *,
    heads_per_group: int,
) -> torch.Tensor:
    if heads_per_group == 1:
        return tensor.contiguous()
    batch, groups = map(int, tensor.shape[:2])
    tail = tuple(int(dim) for dim in tensor.shape[2:])
    return (
        tensor.unsqueeze(2)
        .expand(batch, groups, heads_per_group, *tail)
        .reshape(batch, groups * heads_per_group, *tail)
        .contiguous()
    )


def _scaled_tensor_bytes(
    tensor: torch.Tensor,
    *,
    bc_groups: int,
    heads: int,
) -> int:
    groups = int(bc_groups)
    resolved_heads = int(heads)
    if groups == resolved_heads:
        return _tensor_bytes(tensor)
    numer = _tensor_bytes(tensor) * groups
    if numer % resolved_heads != 0:
        raise ValueError(
            "Grouped byte scaling must divide evenly. "
            f"Got bytes={_tensor_bytes(tensor)}, bc_groups={groups}, heads={resolved_heads}."
        )
    return numer // resolved_heads


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return torch.view_as_real(z).reshape(*z.shape[:-1], 2 * z.shape[-1]).to(real_dtype)


def _build_grouped_v2_inputs(cfg: V2KernelPerfConfig) -> dict[str, torch.Tensor]:
    device = cfg.torch_device
    batch, heads, T, N, P = cfg.batch, cfg.heads, cfg.T, cfg.N, cfg.P
    bc_groups, _ = _validate_bc_groups(heads, cfg.resolved_bc_groups)
    D = 2 * N

    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * torch.pi) * torch.rand((batch, heads, T), device=device) - torch.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=cfg.dtype)
    B_grouped = (
        torch.randn((batch, bc_groups, T, D), device=device, dtype=cfg.dtype) * 0.1
    )
    C_grouped = (
        torch.randn((batch, bc_groups, T, D), device=device, dtype=cfg.dtype) * 0.1
    )
    initial_states = torch.randn((batch, heads, P, D), device=device, dtype=cfg.dtype)

    b_prev = (
        torch.randn((batch, bc_groups, N), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, bc_groups, N), device=device, dtype=torch.float32)
    ) * 0.1
    B_prev_grouped = _pack_complex_pairs(b_prev, real_dtype=cfg.dtype)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=cfg.dtype)
    return {
        "U": U.contiguous(),
        "M": M,
        "K": K,
        "B_grouped": B_grouped.contiguous(),
        "C_grouped": C_grouped.contiguous(),
        "initial_states": initial_states.contiguous(),
        "B_prev_grouped": B_prev_grouped.contiguous(),
        "U_prev": U_prev.contiguous(),
    }


def _make_scanprep_module(cfg: ScanPrepPerfConfig) -> SLinOSSScanPrep:
    kwargs = {
        "n_heads": cfg.heads,
        "d_state": cfg.N,
        "d_head": cfg.P,
        "device": cfg.torch_device,
    }
    if "bc_groups" in inspect.signature(SLinOSSScanPrep).parameters:
        kwargs["bc_groups"] = cfg.resolved_bc_groups
    return SLinOSSScanPrep(**kwargs).to(dtype=cfg.dtype)


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
    bc_groups, heads_per_group = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
    prep = _make_scanprep_module(cfg)

    value = torch.randn(
        (cfg.batch, cfg.T, cfg.heads * cfg.P), device=device, dtype=dtype
    )
    params = torch.randn(
        (cfg.batch, cfg.T, cfg.heads * SCANPREP_PARAM_DIM), device=device, dtype=dtype
    )
    bc_grouped = torch.randn(
        (cfg.batch, cfg.T, bc_groups, prep.bc_param_rows, cfg.N),
        device=device,
        dtype=dtype,
    )
    prep_input_groups = int(getattr(prep, "bc_groups", cfg.heads))
    bc_for_prep = (
        bc_grouped
        if prep_input_groups == bc_groups
        else _materialize_grouped_rows(bc_grouped, heads_per_group=heads_per_group)
    )
    bc_rows = prep._parameterize_scan_bc_rows(bc_for_prep)
    if int(bc_rows.shape[2]) != bc_groups:
        raise ValueError(
            "scanprep forward NCU runner requires grouped BC rows. "
            f"Got bc rows with group dim {int(bc_rows.shape[2])}, expected {bc_groups}."
        )

    U = torch.empty((cfg.batch, cfg.heads, cfg.T, cfg.P), device=device, dtype=dtype)
    M = torch.empty(
        (cfg.batch, cfg.heads, cfg.T, 2), device=device, dtype=torch.float32
    )
    K = torch.empty(
        (cfg.batch, cfg.heads, cfg.T, 2, 2), device=device, dtype=torch.float32
    )
    B = torch.empty(
        (cfg.batch, bc_groups, cfg.T, 2 * cfg.N), device=device, dtype=dtype
    )
    C = torch.empty_like(B)
    coeff_aux = torch.empty(
        (cfg.batch, cfg.heads, COEFF_AUX_FIELDS, cfg.T),
        device=device,
        dtype=torch.float32,
    )

    compiled = cute.compile(
        ScanPrepFwdFused(
            h_size=cfg.heads,
            g_size=bc_groups,
            p_size=cfg.P,
            n_size=cfg.N,
            store_coeff_aux=True,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            theta_init_min=prep.theta_init_min,
            theta_init_max=prep.theta_init_max,
            gamma_min=prep.gamma_min,
            gamma_max=prep.gamma_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            pack_warps_per_block=cfg.pack_warps_per_block,
            coeff_block_size=cfg.coeff_block_size_fwd,
        ),
        make_fake_tensor_arg(value),
        make_fake_tensor_arg(bc_rows),
        make_fake_tensor_arg(params),
        make_fake_tensor_arg(prep.dt_bias.detach()),
        make_fake_tensor_arg(prep.gamma_bias.detach()),
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

    def launch() -> None:
        compiled(
            value,
            bc_rows,
            params,
            prep.dt_bias.detach(),
            prep.gamma_bias.detach(),
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

    effective_bytes = (
        _shape_bytes((cfg.batch, cfg.T, cfg.heads * cfg.P), dtype)
        + _shape_bytes(
            (cfg.batch, cfg.T, cfg.heads * SCANPREP_PARAM_DIM),
            dtype,
        )
        + _shape_bytes(
            (cfg.batch, cfg.T, bc_groups, prep.bc_param_rows, cfg.N),
            dtype,
        )
        + _tensor_bytes(
            prep.dt_bias.detach(),
            prep.gamma_bias.detach(),
            prep.theta_mod_bias.detach(),
            prep.theta_bias.detach(),
            cast(torch.Tensor, prep.theta_sign).detach(),
        )
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, cfg.P), dtype)
        + 2 * _shape_bytes((cfg.batch, bc_groups, cfg.T, 2 * cfg.N), dtype)
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2), torch.float32)
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2, 2), torch.float32)
        + _shape_bytes(
            (cfg.batch, cfg.heads, COEFF_AUX_FIELDS, cfg.T),
            torch.float32,
        )
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
    bc_groups, heads_per_group = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
    prep = _make_scanprep_module(cfg)

    value = torch.randn(
        (cfg.batch, cfg.T, cfg.heads * cfg.P), device=device, dtype=dtype
    )
    params_flat = torch.randn(
        (cfg.batch, cfg.T, cfg.heads * SCANPREP_PARAM_DIM),
        device=device,
        dtype=dtype,
    )
    bc_grouped = torch.randn(
        (cfg.batch, cfg.T, bc_groups, prep.bc_param_rows, cfg.N),
        device=device,
        dtype=dtype,
    )
    prep_input_groups = int(getattr(prep, "bc_groups", cfg.heads))
    bc_for_prep = (
        bc_grouped
        if prep_input_groups == bc_groups
        else _materialize_grouped_rows(bc_grouped, heads_per_group=heads_per_group)
    )
    bc_rows = prep._parameterize_scan_bc_rows(bc_for_prep)
    if int(bc_rows.shape[2]) != bc_groups:
        raise ValueError(
            "scanprep backward NCU runner requires grouped BC rows. "
            f"Got bc rows with group dim {int(bc_rows.shape[2])}, expected {bc_groups}."
        )
    dU = torch.randn((cfg.batch, cfg.heads, cfg.T, cfg.P), device=device, dtype=dtype)
    dM = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, 2), device=device, dtype=torch.float32
    )
    dK = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, 2, 2), device=device, dtype=torch.float32
    )
    dB = torch.randn(
        (cfg.batch, bc_groups, cfg.T, 2 * cfg.N), device=device, dtype=dtype
    )
    dC = torch.randn_like(dB)
    with torch.no_grad():
        _, _, _, _, _, coeff_aux = scanprep_fwd_cute_with_aux(
            value,
            params_flat,
            bc_rows,
            n_heads=cfg.heads,
            bc_groups=bc_groups,
            d_state=cfg.N,
            d_head=cfg.P,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            theta_init_min=prep.theta_init_min,
            theta_init_max=prep.theta_init_max,
            gamma_min=prep.gamma_min,
            gamma_max=prep.gamma_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            dt_bias=prep.dt_bias.detach(),
            gamma_bias=prep.gamma_bias.detach(),
            theta_mod_bias=prep.theta_mod_bias.detach(),
            theta_bias=prep.theta_bias.detach(),
            theta_sign=cast(torch.Tensor, prep.theta_sign).detach(),
        )

    value_grad = torch.empty(
        (cfg.batch, cfg.T, cfg.heads * cfg.P), device=device, dtype=dtype
    )
    bc_grad = torch.empty(
        (cfg.batch, cfg.T, bc_groups, 4, cfg.N), device=device, dtype=dtype
    )
    dparams = torch.empty(
        (cfg.batch, cfg.T, cfg.heads * SCANPREP_PARAM_DIM),
        device=device,
        dtype=dtype,
    )
    bias_grad = torch.zeros((cfg.heads, 4), device=device, dtype=torch.float32)

    compiled = cute.compile(
        ScanPrepBwdFused(
            h_size=cfg.heads,
            g_size=bc_groups,
            p_size=cfg.P,
            n_size=cfg.N,
            param_dim=prep.param_dim,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            theta_init_min=prep.theta_init_min,
            theta_init_max=prep.theta_init_max,
            gamma_min=prep.gamma_min,
            gamma_max=prep.gamma_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            pack_warps_per_block=cfg.pack_warps_per_block,
            coeff_block_size=cfg.coeff_block_size_bwd,
        ),
        make_fake_tensor_arg(dU),
        make_fake_tensor_arg(dB),
        make_fake_tensor_arg(dC),
        make_fake_tensor_arg(coeff_aux),
        make_fake_tensor_arg(dM),
        make_fake_tensor_arg(dK),
        make_fake_tensor_arg(prep.dt_bias.detach()),
        make_fake_tensor_arg(prep.theta_bias.detach()),
        make_fake_tensor_arg(cast(torch.Tensor, prep.theta_sign).detach()),
        make_fake_tensor_arg(value_grad),
        make_fake_tensor_arg(bc_grad),
        make_fake_tensor_arg(dparams),
        make_fake_tensor_arg(bias_grad),
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        bias_grad.zero_()

    def launch() -> None:
        compiled(
            dU,
            dB,
            dC,
            coeff_aux,
            dM,
            dK,
            prep.dt_bias.detach(),
            prep.theta_bias.detach(),
            cast(torch.Tensor, prep.theta_sign).detach(),
            value_grad,
            bc_grad,
            dparams,
            bias_grad,
        )

    effective_bytes = (
        _shape_bytes((cfg.batch, cfg.heads, cfg.T, cfg.P), dtype)
        + 2 * _shape_bytes((cfg.batch, bc_groups, cfg.T, 2 * cfg.N), dtype)
        + _shape_bytes(
            (cfg.batch, cfg.heads, COEFF_AUX_FIELDS, cfg.T),
            torch.float32,
        )
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2), torch.float32)
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2, 2), torch.float32)
        + _tensor_bytes(
            prep.dt_bias.detach(),
            prep.theta_bias.detach(),
            cast(torch.Tensor, prep.theta_sign).detach(),
        )
        + _shape_bytes((cfg.batch, cfg.T, cfg.heads * cfg.P), dtype)
        + _shape_bytes((cfg.batch, cfg.T, bc_groups, 4, cfg.N), dtype)
        + _shape_bytes(
            (cfg.batch, cfg.T, cfg.heads * SCANPREP_PARAM_DIM),
            dtype,
        )
        + _shape_bytes((cfg.heads, 4), torch.float32)
    )
    return KernelRunner(
        name="scanprep_bwd",
        effective_bytes=effective_bytes,
        launch=launch,
        prepare=prepare,
    )


def _build_v2x2ssd_forward_runners(
    cfg: V2KernelPerfConfig,
) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    bc_groups, _ = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B_grouped"]
    C = tensors["C_grouped"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev_grouped"]
    U_prev = tensors["U_prev"]

    n_chunks = _n_chunks(cfg.T, cfg.chunk_size)
    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    tc_dtype = _tc_input_dtype(cfg.dtype)

    prepared_increment = _make_chunk_increment_prepared_launch(
        U,
        M,
        K,
        B,
        U_prev=U_prev,
        B_prev=B_prev,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
    )
    cast(Callable[..., None], prepared_increment.compiled)(
        *prepared_increment.runtime_args
    )
    inc_chunk = prepared_increment.outputs.increment_chunk
    chunk_multiplier_storage = prepared_increment.outputs.chunk_multiplier_storage
    increment = inc_chunk.reshape(cfg.batch, cfg.heads, n_chunks, cfg.P, D)
    chunk_multiplier = chunk_multiplier_storage.reshape(
        cfg.batch, cfg.heads, n_chunks, 2
    )

    prepared_state = _make_state_passing_prepared_launch(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
    )
    cast(Callable[..., None], prepared_state.compiled)(*prepared_state.runtime_args)
    chunk_starts = prepared_state.outputs.chunk_starts
    final_state = prepared_state.outputs.final_state

    prepared_scan = _make_chunk_scan_prepared_launch(
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
    out_chunk = prepared_scan.outputs.output_chunk

    bytes_u = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_bc = _shape_bytes((cfg.batch, bc_groups, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_u_prev = _shape_bytes((cfg.batch, cfg.heads, cfg.P), tc_dtype)
    bytes_b_prev = _shape_bytes((cfg.batch, bc_groups, D), tc_dtype)

    return {
        "chunk_increment_fwd": KernelRunner(
            name="chunk_increment_fwd",
            effective_bytes=(
                bytes_u
                + bytes_bc
                + bytes_m
                + bytes_k
                + bytes_u_prev
                + bytes_b_prev
                + _tensor_bytes(inc_chunk, chunk_multiplier_storage)
            ),
            launch=lambda prepared=prepared_increment: cast(
                Callable[..., None], prepared.compiled
            )(*prepared.runtime_args),
            prepare=_noop,
        ),
        "state_passing_fwd": KernelRunner(
            name="state_passing_fwd",
            effective_bytes=_tensor_bytes(
                increment,
                chunk_multiplier,
                initial_states,
                chunk_starts,
                final_state,
            ),
            launch=lambda prepared=prepared_state: cast(
                Callable[..., None], prepared.compiled
            )(*prepared.runtime_args),
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
                + bytes_u_prev
                + bytes_b_prev
                + _tensor_bytes(chunk_starts, out_chunk)
            ),
            launch=lambda prepared=prepared_scan: cast(
                Callable[..., None], prepared.compiled
            )(*prepared.runtime_args),
            prepare=_noop,
        ),
    }


def _build_v2x2ssd_chunk_increment_bwd_runners(
    cfg: V2KernelPerfConfig,
) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    bc_groups, heads_per_group = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B_grouped = tensors["B_grouped"]
    B = _materialize_grouped_rows(B_grouped, heads_per_group=heads_per_group)
    B_prev_grouped = tensors["B_prev_grouped"]
    B_prev = _materialize_grouped_rows(
        B_prev_grouped,
        heads_per_group=heads_per_group,
    )
    U_prev = tensors["U_prev"]

    inc, m_chunk = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
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
        launch,
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
    launch_vars = _closure_vars(launch)
    stage_args = launch_vars.get("stage_runtime_args", launch_vars.get("stage_args"))
    if not isinstance(stage_args, tuple):
        raise RuntimeError(
            "chunk_increment_bwd launcher closure did not expose stage runtime args."
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
    bytes_b = _shape_bytes((cfg.batch, bc_groups, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_d_inc = _shape_bytes(tuple(int(dim) for dim in d_inc.shape), tc_dtype)
    bytes_prev_b = _shape_bytes((cfg.batch, bc_groups, n_chunks, D), tc_dtype)
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
            + bytes_b
            + _tensor_bytes(dMsum_part),
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
            + _tensor_bytes(dU_prev, dMp0)
            + _shape_bytes((cfg.batch, bc_groups, D), tc_dtype),
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
    cfg: V2KernelPerfConfig,
) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    _, heads_per_group = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = _materialize_grouped_rows(
        tensors["B_grouped"],
        heads_per_group=heads_per_group,
    )
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = _materialize_grouped_rows(
        tensors["B_prev_grouped"],
        heads_per_group=heads_per_group,
    )
    U_prev = tensors["U_prev"]

    inc, m_chunk = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )
    prepared_state = _make_state_passing_prepared_launch(
        inc,
        m_chunk,
        initial_states=initial_states,
    )
    cast(Callable[..., None], prepared_state.compiled)(*prepared_state.runtime_args)
    chunk_starts = prepared_state.outputs.chunk_starts

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
    launch_vars = _closure_vars(launch)
    compiled = cast(Callable[..., None], launch_vars.get("compiled"))
    dynamic_args = launch_vars.get("runtime_args", launch_vars.get("dynamic_args"))
    if not isinstance(dynamic_args, tuple):
        raise RuntimeError(
            "state_passing_bwd launcher closure did not expose runtime args."
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
    cfg: V2KernelPerfConfig,
) -> dict[str, KernelRunner]:
    _seed_all(cfg.seed)
    bc_groups, heads_per_group = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = _materialize_grouped_rows(
        tensors["B_grouped"],
        heads_per_group=heads_per_group,
    )
    C = _materialize_grouped_rows(
        tensors["C_grouped"],
        heads_per_group=heads_per_group,
    )
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = _materialize_grouped_rows(
        tensors["B_prev_grouped"],
        heads_per_group=heads_per_group,
    )
    U_prev = tensors["U_prev"]

    inc, m_chunk = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=cfg.chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )
    prepared_state = _make_state_passing_prepared_launch(
        inc,
        m_chunk,
        initial_states=initial_states,
    )
    cast(Callable[..., None], prepared_state.compiled)(*prepared_state.runtime_args)
    chunk_starts = prepared_state.outputs.chunk_starts
    d_out = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    prepared = compile_chunk_scan_bwd_kernels(
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
    )
    outputs = prepared.outputs
    launchers: dict[str, Callable[[], None]] = {
        "chunk_scan_bwd_dz0": cast(Callable[[], None], prepared.launchers.dz0),
        "chunk_scan_bwd_du": cast(Callable[[], None], prepared.launchers.du),
        "chunk_scan_bwd_db": cast(Callable[[], None], prepared.launchers.db),
        "chunk_scan_bwd_dcdr": cast(Callable[[], None], prepared.launchers.dcdr),
        "chunk_scan_bwd_dlp": cast(Callable[[], None], prepared.launchers.dlp),
        "chunk_scan_bwd_param": cast(
            Callable[[], None],
            prepared.launchers.param_scan,
        ),
    }

    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    tc_dtype = _tc_input_dtype(cfg.dtype)
    bytes_u = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_b = _shape_bytes((cfg.batch, bc_groups, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_d_out = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_u_prev = _shape_bytes((cfg.batch, cfg.heads, cfg.P), tc_dtype)
    bytes_b_prev = _shape_bytes((cfg.batch, bc_groups, D), tc_dtype)
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
            effective_bytes=bytes_d_out
            + bytes_b
            + bytes_m
            + _tensor_bytes(outputs.chunk_start_grad),
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
            + bytes_u_prev
            + bytes_b_prev
            + _tensor_bytes(
                outputs.value_grad_chunk,
                outputs.value_boundary_grad,
                outputs.logprefix_grad,
                outputs.transition_grad,
                outputs.transition_grad,
            )
            + _scaled_tensor_bytes(
                outputs.key_grad_chunk,
                bc_groups=bc_groups,
                heads=cfg.heads,
            )
            + _scaled_tensor_bytes(
                outputs.key_boundary_grad,
                bc_groups=bc_groups,
                heads=cfg.heads,
            ),
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
            + bytes_u_prev
            + bytes_b_prev
            + _tensor_bytes(
                outputs.value_grad_chunk,
                outputs.value_boundary_grad,
                outputs.logprefix_grad,
                outputs.transition_grad,
                outputs.transition_grad,
            )
            + _scaled_tensor_bytes(
                outputs.key_grad_chunk,
                bc_groups=bc_groups,
                heads=cfg.heads,
            )
            + _scaled_tensor_bytes(
                outputs.key_boundary_grad,
                bc_groups=bc_groups,
                heads=cfg.heads,
            ),
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
            + bytes_u_prev
            + bytes_b_prev
            + _tensor_bytes(
                chunk_starts,
                outputs.logprefix_grad,
                outputs.rotation_grad,
            )
            + _scaled_tensor_bytes(
                outputs.query_grad_chunk,
                bc_groups=bc_groups,
                heads=cfg.heads,
            )
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
            + bytes_u_prev
            + bytes_b_prev
            + _tensor_bytes(outputs.logprefix_grad),
            launch=launchers["chunk_scan_bwd_dlp"],
            prepare=_noop,
        ),
        "chunk_scan_bwd_param": KernelRunner(
            name="chunk_scan_bwd_param",
            effective_bytes=bytes_m
            + bytes_k
            + bytes_dphase
            + _tensor_bytes(
                outputs.logprefix_grad,
                outputs.rotation_grad,
                outputs.transition_grad,
                outputs.transition_grad,
                outputs.transition_grad,
                outputs.tap_prev_grad,
                outputs.tap_curr_grad,
            ),
            launch=launchers["chunk_scan_bwd_param"],
            prepare=prepare_param,
            note="integrated stage in chunk_scan_bwd",
        ),
    }


def build_kernel_runners(
    *,
    v2_cfg: V2KernelPerfConfig | None = None,
    scanprep_cfg: ScanPrepPerfConfig | None = None,
) -> list[KernelRunner]:
    if v2_cfg is None:
        v2_cfg = V2KernelPerfConfig(
            batch=DEFAULT_V2_BATCH,
            heads=DEFAULT_V2_HEADS,
            T=DEFAULT_V2_T,
            N=DEFAULT_V2_N,
            P=DEFAULT_V2_P,
            chunk_size=DEFAULT_V2_CHUNK,
            bc_groups=DEFAULT_V2_BC_GROUPS,
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
    v2_cfg: V2KernelPerfConfig | None = None,
    scanprep_cfg: ScanPrepPerfConfig | None = None,
) -> KernelRunner:
    if v2_cfg is None:
        v2_cfg = V2KernelPerfConfig(
            batch=DEFAULT_V2_BATCH,
            heads=DEFAULT_V2_HEADS,
            T=DEFAULT_V2_T,
            N=DEFAULT_V2_N,
            P=DEFAULT_V2_P,
            chunk_size=DEFAULT_V2_CHUNK,
            bc_groups=DEFAULT_V2_BC_GROUPS,
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
