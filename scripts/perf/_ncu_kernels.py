from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import sys
from typing import Callable, cast

from cuda.bindings import driver as cuda  # pyright: ignore[reportAttributeAccessIssue]
import cutlass
import cutlass.cute as cute
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import dtype_from_str  # noqa: E402
from _training import DEFAULT_TRAINING_PERF_CONFIG  # noqa: E402
from slinoss.layers import SLinOSSMLPConfig, SLinOSSScanPrep  # noqa: E402
from slinoss.ops.block.cute.activation import (  # noqa: E402
    FfnActivationBwdFused,
    FfnActivationFwdFused,
)
import slinoss.ops.block.cute.common as block_cute_common  # noqa: E402
from slinoss.ops.block.cute.common import ActivationKind  # noqa: E402
from slinoss.ops.block.cute.norm import (  # noqa: E402
    FfnRmsNormBwdFused,
    FfnRmsNormFwdFused,
)
from slinoss.ops.mixer.cute.bwd import MixerTailRowwiseBwdFused  # noqa: E402
import slinoss.ops.mixer.cute.common as mixer_cute_common  # noqa: E402
from slinoss.ops.mixer.cute.fwd import MixerTailRowwiseFwdFused  # noqa: E402
from slinoss.ops.scanprep.cute.common import (  # noqa: E402
    SCANPREP_PARAM_DIM,
    make_fake_tensor_arg,
)
from slinoss.ops.scanprep.cute.kernels.bwd import ScanPrepBwdFused  # noqa: E402
from slinoss.ops.scanprep.cute.kernels.fwd import ScanPrepFwdFused  # noqa: E402
from slinoss.ops.v2x2ssd.cute.kernels.bwd import (  # noqa: E402
    BackwardProblemShape,
    _increment_bwd_tensor_specs,
    _scan_bwd_tensor_specs,
    _state_bwd_tensor_specs,
    _get_compiled_v2x2ssd_bwd_kernel,
    _make_backward_compile_artifacts_from_runtime_artifacts,
    _make_backward_runtime_artifacts,
    _make_static_tensor_spec_view,
    _make_tvm_ffi_runtime_and_compile_args_from_specs,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.common import _TileConfig  # noqa: E402
from slinoss.ops.v2x2ssd.cute.kernels.bwd.boundary import (  # noqa: E402
    BwdBoundaryAmpere,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.dcdr import BwdDCDRAmpere  # noqa: E402
from slinoss.ops.v2x2ssd.cute.kernels.bwd.db import (  # noqa: E402
    BwdDBAmpere,
    BwdDBIncrementAccumulatorAmpere,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.du import (  # noqa: E402
    BwdDUAmpere,
    BwdDUIncrementAccumulatorAmpere,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.dz0 import BwdDZ0Ampere  # noqa: E402
from slinoss.ops.v2x2ssd.cute.kernels.bwd.param import (  # noqa: E402
    BwdParamIncrementAccumulatorAmpere,
    BwdParamScanAmpere,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import (  # noqa: E402
    BwdStatePassingAmpere,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd import v2x2ssd_fwd_cute  # noqa: E402


DEFAULT_V2_BATCH = int(DEFAULT_TRAINING_PERF_CONFIG.batch_size)
DEFAULT_V2_HEADS = int(DEFAULT_TRAINING_PERF_CONFIG.n_heads)
DEFAULT_V2_T = int(DEFAULT_TRAINING_PERF_CONFIG.seq_len)
DEFAULT_V2_N = int(DEFAULT_TRAINING_PERF_CONFIG.mixer.d_state)
DEFAULT_V2_P = int(DEFAULT_TRAINING_PERF_CONFIG.mixer.d_head)
DEFAULT_V2_CHUNK = int(DEFAULT_TRAINING_PERF_CONFIG.mixer.chunk_size)
DEFAULT_V2_DTYPE = DEFAULT_TRAINING_PERF_CONFIG.dtype
DEFAULT_V2_BC_GROUPS = int(
    getattr(DEFAULT_TRAINING_PERF_CONFIG, "resolved_bc_groups", DEFAULT_V2_HEADS)
)
_DEFAULT_FFN_CONFIG = DEFAULT_TRAINING_PERF_CONFIG.ffn or SLinOSSMLPConfig()
DEFAULT_FFN_D_MODEL = int(DEFAULT_TRAINING_PERF_CONFIG.d_model)
DEFAULT_FFN_HIDDEN_DIM = int(
    _DEFAULT_FFN_CONFIG.resolve_hidden_dim(DEFAULT_FFN_D_MODEL)
)
DEFAULT_FFN_KIND = _DEFAULT_FFN_CONFIG.kind
DEFAULT_FFN_EPS = float(DEFAULT_TRAINING_PERF_CONFIG.norm_eps)


@dataclass(frozen=True)
class V2KernelPerfConfig:
    batch: int = DEFAULT_V2_BATCH
    heads: int = DEFAULT_V2_HEADS
    T: int = DEFAULT_V2_T
    N: int = DEFAULT_V2_N
    P: int = DEFAULT_V2_P
    chunk_size: int = DEFAULT_V2_CHUNK
    bc_groups: int | None = None
    dtype: torch.dtype = DEFAULT_TRAINING_PERF_CONFIG.dtype
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
    dtype: torch.dtype = DEFAULT_TRAINING_PERF_CONFIG.dtype
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


@dataclass(frozen=True)
class MixerTailPerfConfig:
    batch: int = DEFAULT_V2_BATCH
    heads: int = DEFAULT_V2_HEADS
    T: int = DEFAULT_V2_T
    P: int = DEFAULT_V2_P
    dtype: torch.dtype = DEFAULT_TRAINING_PERF_CONFIG.dtype
    d_skip_dtype: torch.dtype = torch.float32
    device: str = "cuda"
    seed: int = 0
    eps: float = 1.0e-5
    warps_per_block: int = 8

    @property
    def hidden_dim(self) -> int:
        return int(self.heads * self.P)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass(frozen=True)
class FfnPerfConfig:
    batch: int = DEFAULT_V2_BATCH
    T: int = DEFAULT_V2_T
    d_model: int = DEFAULT_FFN_D_MODEL
    hidden_dim: int = DEFAULT_FFN_HIDDEN_DIM
    kind: str = DEFAULT_FFN_KIND
    dtype: torch.dtype = DEFAULT_TRAINING_PERF_CONFIG.dtype
    device: str = "cuda"
    seed: int = 0
    eps: float = DEFAULT_FFN_EPS
    warps_per_block: int = 8

    @property
    def projected_dim(self) -> int:
        return int(self.hidden_dim if self.kind == "gelu" else 2 * self.hidden_dim)

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
    "mixer_tail_rowwise_fwd",
    "mixer_tail_rowwise_bwd",
    "ffn_norm_fwd",
    "ffn_norm_bwd",
    "ffn_activation_fwd",
    "ffn_activation_bwd",
    "v2x2ssd_fwd",
    "v2x2ssd_bwd",
    "bwd_db",
    "bwd_dcdr",
    "bwd_param_scan",
    "bwd_du",
    "bwd_dz0",
    "bwd_state_passing",
    "bwd_db_increment_accumulator",
    "bwd_boundary",
    "bwd_param_increment_accumulator",
    "bwd_du_increment_accumulator",
)


KERNEL_GROUPS = {
    "all": KERNEL_ORDER,
    "scanprep": ("scanprep_fwd", "scanprep_bwd"),
    "mixer_tail": ("mixer_tail_rowwise_fwd", "mixer_tail_rowwise_bwd"),
    "ffn": (
        "ffn_norm_fwd",
        "ffn_norm_bwd",
        "ffn_activation_fwd",
        "ffn_activation_bwd",
    ),
    "v2x2ssd": (
        "v2x2ssd_fwd",
        "v2x2ssd_bwd",
    ),
    "v2x2ssd_bwd_launches": (
        "bwd_db",
        "bwd_dcdr",
        "bwd_param_scan",
        "bwd_du",
        "bwd_dz0",
        "bwd_state_passing",
        "bwd_db_increment_accumulator",
        "bwd_boundary",
        "bwd_param_increment_accumulator",
        "bwd_du_increment_accumulator",
    ),
}


def list_kernel_names() -> tuple[str, ...]:
    return KERNEL_ORDER


def list_kernel_groups() -> dict[str, tuple[str, ...]]:
    return {name: tuple(kernels) for name, kernels in KERNEL_GROUPS.items()}


def _noop() -> None:
    return None


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
    return SLinOSSScanPrep(
        n_heads=cfg.heads,
        bc_groups=cfg.resolved_bc_groups,
        d_state=cfg.N,
        d_head=cfg.P,
        device=cfg.torch_device,
    ).to(dtype=cfg.dtype)


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
    bc_groups, _ = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
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

    compiled = cute.compile(
        ScanPrepFwdFused(
            h_size=cfg.heads,
            g_size=bc_groups,
            p_size=cfg.P,
            n_size=cfg.N,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            theta_init_min=prep.theta_init_min,
            theta_init_max=prep.theta_init_max,
            theta_mod_scale=prep.theta_mod_scale,
            alpha_min=prep.alpha_min,
            alpha_max=prep.alpha_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            pack_warps_per_block=cfg.pack_warps_per_block,
            coeff_block_size=cfg.coeff_block_size_fwd,
        ),
        make_fake_tensor_arg(value),
        make_fake_tensor_arg(bc_grouped),
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
        options="--enable-tvm-ffi",
    )

    def launch() -> None:
        compiled(
            value,
            bc_grouped,
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
            prep.alpha_bias.detach(),
            prep.theta_mod_bias.detach(),
            prep.theta_bias.detach(),
            cast(torch.Tensor, prep.theta_sign).detach(),
        )
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, cfg.P), dtype)
        + 2 * _shape_bytes((cfg.batch, bc_groups, cfg.T, 2 * cfg.N), dtype)
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2), torch.float32)
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2, 2), torch.float32)
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
    bc_groups, _ = _validate_bc_groups(cfg.heads, cfg.resolved_bc_groups)
    prep = _make_scanprep_module(cfg)

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
            theta_mod_scale=prep.theta_mod_scale,
            alpha_min=prep.alpha_min,
            alpha_max=prep.alpha_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            pack_warps_per_block=cfg.pack_warps_per_block,
            coeff_block_size=cfg.coeff_block_size_bwd,
        ),
        make_fake_tensor_arg(dU),
        make_fake_tensor_arg(bc_grouped),
        make_fake_tensor_arg(dB),
        make_fake_tensor_arg(dC),
        make_fake_tensor_arg(params_flat),
        make_fake_tensor_arg(prep.dt_bias.detach()),
        make_fake_tensor_arg(prep.alpha_bias.detach()),
        make_fake_tensor_arg(prep.theta_mod_bias.detach()),
        make_fake_tensor_arg(prep.theta_bias.detach()),
        make_fake_tensor_arg(cast(torch.Tensor, prep.theta_sign).detach()),
        make_fake_tensor_arg(dM),
        make_fake_tensor_arg(dK),
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
            bc_grouped,
            dB,
            dC,
            params_flat,
            prep.dt_bias.detach(),
            prep.alpha_bias.detach(),
            prep.theta_mod_bias.detach(),
            prep.theta_bias.detach(),
            cast(torch.Tensor, prep.theta_sign).detach(),
            dM,
            dK,
            value_grad,
            bc_grad,
            dparams,
            bias_grad,
        )

    effective_bytes = (
        _shape_bytes((cfg.batch, cfg.heads, cfg.T, cfg.P), dtype)
        + _shape_bytes(
            (cfg.batch, cfg.T, cfg.heads * SCANPREP_PARAM_DIM),
            dtype,
        )
        + _shape_bytes(
            (cfg.batch, cfg.T, bc_groups, prep.bc_param_rows, cfg.N),
            dtype,
        )
        + 2 * _shape_bytes((cfg.batch, bc_groups, cfg.T, 2 * cfg.N), dtype)
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2), torch.float32)
        + _shape_bytes((cfg.batch, cfg.heads, cfg.T, 2, 2), torch.float32)
        + _tensor_bytes(
            prep.dt_bias.detach(),
            prep.alpha_bias.detach(),
            prep.theta_mod_bias.detach(),
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


def _build_mixer_tail_rowwise_fwd_runner(cfg: MixerTailPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    scan_output = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=device,
        dtype=dtype,
    )
    gate = torch.randn(
        (cfg.batch, cfg.T, cfg.hidden_dim),
        device=device,
        dtype=dtype,
    )
    out_norm_weight = torch.randn(
        (cfg.hidden_dim,),
        device=device,
        dtype=dtype,
    )
    skip_input = torch.randn_like(scan_output)
    d_skip = torch.randn((cfg.heads,), device=device, dtype=cfg.d_skip_dtype)
    normed = torch.empty(
        (cfg.batch, cfg.T, cfg.hidden_dim),
        device=device,
        dtype=dtype,
    )

    compiled = cute.compile(
        MixerTailRowwiseFwdFused(
            h_size=cfg.heads,
            p_size=cfg.P,
            eps=cfg.eps,
            has_skip=True,
            storage_dtype=dtype,
            warps_per_block=cfg.warps_per_block,
        ),
        mixer_cute_common.make_fake_tensor_arg(scan_output),
        mixer_cute_common.make_fake_tensor_arg(gate),
        mixer_cute_common.make_fake_tensor_arg(out_norm_weight),
        mixer_cute_common.make_fake_tensor_arg(skip_input),
        mixer_cute_common.make_fake_tensor_arg(d_skip),
        mixer_cute_common.make_fake_tensor_arg(normed),
        options="--enable-tvm-ffi",
    )

    def launch() -> None:
        compiled(
            scan_output,
            gate,
            out_norm_weight,
            skip_input,
            d_skip,
            normed,
        )

    return KernelRunner(
        name="mixer_tail_rowwise_fwd",
        effective_bytes=_tensor_bytes(
            scan_output,
            gate,
            out_norm_weight,
            skip_input,
            d_skip,
            normed,
        ),
        launch=launch,
        prepare=_noop,
    )


def _build_mixer_tail_rowwise_bwd_runner(cfg: MixerTailPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    scan_output = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=device,
        dtype=dtype,
    )
    gate = torch.randn(
        (cfg.batch, cfg.T, cfg.hidden_dim),
        device=device,
        dtype=dtype,
    )
    out_norm_weight = torch.randn(
        (cfg.hidden_dim,),
        device=device,
        dtype=dtype,
    )
    skip_input = torch.randn_like(scan_output)
    d_skip = torch.randn((cfg.heads,), device=device, dtype=cfg.d_skip_dtype)
    d_normed = torch.randn(
        (cfg.batch, cfg.T, cfg.hidden_dim),
        device=device,
        dtype=dtype,
    )
    d_scan_output = torch.empty_like(scan_output)
    d_gate = torch.empty_like(gate)
    d_skip_input = torch.empty_like(scan_output)
    d_d_skip = torch.zeros((cfg.heads,), device=device, dtype=torch.float32)
    d_norm_weight = torch.zeros((cfg.hidden_dim,), device=device, dtype=torch.float32)

    compiled = cute.compile(
        MixerTailRowwiseBwdFused(
            h_size=cfg.heads,
            p_size=cfg.P,
            eps=cfg.eps,
            has_skip=True,
            warps_per_block=cfg.warps_per_block,
        ),
        mixer_cute_common.make_fake_tensor_arg(scan_output),
        mixer_cute_common.make_fake_tensor_arg(gate),
        mixer_cute_common.make_fake_tensor_arg(out_norm_weight),
        mixer_cute_common.make_fake_tensor_arg(skip_input),
        mixer_cute_common.make_fake_tensor_arg(d_skip),
        mixer_cute_common.make_fake_tensor_arg(d_normed),
        mixer_cute_common.make_fake_tensor_arg(d_scan_output),
        mixer_cute_common.make_fake_tensor_arg(d_gate),
        mixer_cute_common.make_fake_tensor_arg(d_skip_input),
        mixer_cute_common.make_fake_tensor_arg(d_d_skip),
        mixer_cute_common.make_fake_tensor_arg(d_norm_weight),
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        d_d_skip.zero_()
        d_norm_weight.zero_()

    def launch() -> None:
        compiled(
            scan_output,
            gate,
            out_norm_weight,
            skip_input,
            d_skip,
            d_normed,
            d_scan_output,
            d_gate,
            d_skip_input,
            d_d_skip,
            d_norm_weight,
        )

    return KernelRunner(
        name="mixer_tail_rowwise_bwd",
        effective_bytes=_tensor_bytes(
            scan_output,
            gate,
            out_norm_weight,
            skip_input,
            d_skip,
            d_normed,
            d_scan_output,
            d_gate,
            d_skip_input,
            d_d_skip,
            d_norm_weight,
        ),
        launch=launch,
        prepare=prepare,
    )


def _build_ffn_norm_fwd_runner(cfg: FfnPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    residual = torch.randn((cfg.batch, cfg.T, cfg.d_model), device=device, dtype=dtype)
    norm_weight = torch.randn((cfg.d_model,), device=device, dtype=dtype)
    output = torch.empty_like(residual)

    compiled = cute.compile(
        FfnRmsNormFwdFused(
            hidden_dim=cfg.d_model,
            eps=cfg.eps,
            warps_per_block=cfg.warps_per_block,
        ),
        block_cute_common.make_fake_tensor_arg(residual),
        block_cute_common.make_fake_tensor_arg(norm_weight),
        block_cute_common.make_fake_tensor_arg(output),
        options="--enable-tvm-ffi",
    )

    def launch() -> None:
        compiled(
            residual,
            norm_weight,
            output,
        )

    return KernelRunner(
        name="ffn_norm_fwd",
        effective_bytes=_tensor_bytes(residual, norm_weight, output),
        launch=launch,
        prepare=_noop,
    )


def _build_ffn_norm_bwd_runner(cfg: FfnPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    residual = torch.randn((cfg.batch, cfg.T, cfg.d_model), device=device, dtype=dtype)
    norm_weight = torch.randn((cfg.d_model,), device=device, dtype=dtype)
    d_output = torch.randn_like(residual)
    d_input = torch.empty_like(residual)
    d_weight_accum = torch.zeros((cfg.d_model,), device=device, dtype=torch.float32)

    compiled = cute.compile(
        FfnRmsNormBwdFused(
            hidden_dim=cfg.d_model,
            eps=cfg.eps,
            warps_per_block=cfg.warps_per_block,
        ),
        block_cute_common.make_fake_tensor_arg(residual),
        block_cute_common.make_fake_tensor_arg(norm_weight),
        block_cute_common.make_fake_tensor_arg(d_output),
        block_cute_common.make_fake_tensor_arg(d_input),
        block_cute_common.make_fake_tensor_arg(d_weight_accum),
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        d_weight_accum.zero_()

    def launch() -> None:
        compiled(
            residual,
            norm_weight,
            d_output,
            d_input,
            d_weight_accum,
        )

    return KernelRunner(
        name="ffn_norm_bwd",
        effective_bytes=_tensor_bytes(
            residual,
            norm_weight,
            d_output,
            d_input,
            d_weight_accum,
        ),
        launch=launch,
        prepare=prepare,
    )


def _build_ffn_activation_fwd_runner(cfg: FfnPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    projected = torch.randn(
        (cfg.batch, cfg.T, cfg.projected_dim),
        device=device,
        dtype=dtype,
    )
    hidden = torch.empty(
        (cfg.batch, cfg.T, cfg.hidden_dim),
        device=device,
        dtype=dtype,
    )

    compiled = cute.compile(
        FfnActivationFwdFused(
            hidden_dim=cfg.hidden_dim,
            kind=cast(ActivationKind, cfg.kind),
            warps_per_block=cfg.warps_per_block,
        ),
        block_cute_common.make_fake_tensor_arg(projected),
        block_cute_common.make_fake_tensor_arg(hidden),
        options="--enable-tvm-ffi",
    )

    def launch() -> None:
        compiled(
            projected,
            hidden,
        )

    return KernelRunner(
        name="ffn_activation_fwd",
        effective_bytes=_tensor_bytes(projected, hidden),
        launch=launch,
        prepare=_noop,
    )


def _build_ffn_activation_bwd_runner(cfg: FfnPerfConfig) -> KernelRunner:
    _seed_all(cfg.seed)
    device = cfg.torch_device
    dtype = cfg.dtype

    projected = torch.randn(
        (cfg.batch, cfg.T, cfg.projected_dim),
        device=device,
        dtype=dtype,
    )
    d_hidden = torch.randn(
        (cfg.batch, cfg.T, cfg.hidden_dim),
        device=device,
        dtype=dtype,
    )
    d_projected = torch.empty_like(projected)

    compiled = cute.compile(
        FfnActivationBwdFused(
            hidden_dim=cfg.hidden_dim,
            kind=cast(ActivationKind, cfg.kind),
            warps_per_block=cfg.warps_per_block,
        ),
        block_cute_common.make_fake_tensor_arg(projected),
        block_cute_common.make_fake_tensor_arg(d_hidden),
        block_cute_common.make_fake_tensor_arg(d_projected),
        options="--enable-tvm-ffi",
    )

    def launch() -> None:
        compiled(
            projected,
            d_hidden,
            d_projected,
        )

    return KernelRunner(
        name="ffn_activation_bwd",
        effective_bytes=_tensor_bytes(projected, d_hidden, d_projected),
        launch=launch,
        prepare=_noop,
    )


def _build_v2x2ssd_forward_runner(cfg: V2KernelPerfConfig) -> KernelRunner:
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

    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    tc_dtype = _tc_input_dtype(cfg.dtype)

    bytes_u = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_bc = _shape_bytes((cfg.batch, bc_groups, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_u_prev = _shape_bytes((cfg.batch, cfg.heads, cfg.P), tc_dtype)
    bytes_b_prev = _shape_bytes((cfg.batch, bc_groups, D), tc_dtype)
    bytes_state = _shape_bytes((cfg.batch, cfg.heads, cfg.P, D), torch.float32)
    bytes_output = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), torch.float32)

    def launch() -> None:
        v2x2ssd_fwd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=cfg.chunk_size,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            return_final_state=True,
            return_intermediates=True,
        )

    launch()

    return KernelRunner(
        name="v2x2ssd_fwd",
        effective_bytes=(
            bytes_u
            + bytes_bc
            + bytes_bc
            + bytes_m
            + bytes_k
            + bytes_u_prev
            + bytes_b_prev
            + bytes_state
            + bytes_output
        ),
        launch=launch,
        prepare=_noop,
    )


def _compute_forward_intermediates_for_bwd(
    cfg: V2KernelPerfConfig,
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    initial_states: torch.Tensor,
    B_prev: torch.Tensor,
    U_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _output, m_chunk, chunk_starts = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        v2x2ssd_fwd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=cfg.chunk_size,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            return_final_state=False,
            return_intermediates=True,
        ),
    )
    return m_chunk, chunk_starts


def _build_v2x2ssd_bwd_runtime_artifacts(cfg: V2KernelPerfConfig):
    _seed_all(cfg.seed)
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B_grouped"]
    C = tensors["C_grouped"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev_grouped"]
    U_prev = tensors["U_prev"]
    m_chunk, chunk_starts = _compute_forward_intermediates_for_bwd(
        cfg,
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    d_out = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    d_final = torch.randn_like(initial_states)
    return _make_backward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
        scan_num_threads_du=128,
        scan_num_threads_db=128,
        scan_num_threads_dcdr=128,
        scan_num_threads_param=32,
        state_num_threads=128,
        state_pairs_per_thread=8,
        B_prev=B_prev,
        U_prev=U_prev,
        initial_states=initial_states,
        d_final_state=d_final,
        validate_runtime_contract=True,
    )


def _reset_v2x2ssd_bwd_runtime_artifacts(runtime_artifacts) -> None:
    for tensor in runtime_artifacts.runtime_args[15:]:
        tensor.zero_()


def _build_v2x2ssd_backward_runner(cfg: V2KernelPerfConfig) -> KernelRunner:
    runtime_artifacts = _build_v2x2ssd_bwd_runtime_artifacts(cfg)
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    compiled = cast(
        Callable[..., None],
        _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts),
    )

    def launch() -> None:
        compiled(*runtime_artifacts.runtime_args)

    _reset_v2x2ssd_bwd_runtime_artifacts(runtime_artifacts)
    launch()
    torch.cuda.synchronize(cfg.torch_device)

    return KernelRunner(
        name="v2x2ssd_bwd",
        effective_bytes=_tensor_bytes(*runtime_artifacts.runtime_args),
        launch=launch,
        prepare=lambda: _reset_v2x2ssd_bwd_runtime_artifacts(runtime_artifacts),
    )


def _compile_full_v2x2ssd_bwd(runtime_artifacts) -> Callable[..., None]:
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    return cast(
        Callable[..., None],
        _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts),
    )


def _prime_v2x2ssd_bwd_dependencies(
    runtime_artifacts,
    *,
    device: torch.device,
) -> None:
    _reset_v2x2ssd_bwd_runtime_artifacts(runtime_artifacts)
    _compile_full_v2x2ssd_bwd(runtime_artifacts)(*runtime_artifacts.runtime_args)
    torch.cuda.synchronize(device)


def _make_bwd_db_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        _n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    scan_num_threads_db = int(launch_cfg[1])
    scan_specs = _scan_bwd_tensor_specs(problem_shape)
    U_scan_spec = scan_specs[0]
    BC_scan_spec = scan_specs[1]
    M_scan_spec = scan_specs[2]
    K_scan_spec = scan_specs[3]
    U_prev_scan_spec = scan_specs[5]
    B_prev_scan_spec = scan_specs[6]
    dM_scan_scratch_spec = scan_specs[8]
    dBC_scan_spec = scan_specs[17]
    dB_prev_scan_spec = scan_specs[19]

    @cute.jit
    def _bwd_db_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        C_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        d_out_t: cute.Tensor,
        U_prev_t: cute.Tensor,
        B_prev_t: cute.Tensor,
        dB_t: cute.Tensor,
        dB_prev_t: cute.Tensor,
        dM_previous_t: cute.Tensor,
        dM_current_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        U_view = _make_static_tensor_spec_view(U_t, U_scan_spec)
        B_view = _make_static_tensor_spec_view(B_t, BC_scan_spec)
        C_view = _make_static_tensor_spec_view(C_t, BC_scan_spec)
        M_view = _make_static_tensor_spec_view(M_t, M_scan_spec)
        K_view = _make_static_tensor_spec_view(K_t, K_scan_spec)
        d_out_view = _make_static_tensor_spec_view(d_out_t, U_scan_spec)
        U_prev_view = _make_static_tensor_spec_view(U_prev_t, U_prev_scan_spec)
        B_prev_view = _make_static_tensor_spec_view(B_prev_t, B_prev_scan_spec)
        dB_view = _make_static_tensor_spec_view(dB_t, dBC_scan_spec)
        dB_prev_view = _make_static_tensor_spec_view(dB_prev_t, dB_prev_scan_spec)
        dM_previous_view = _make_static_tensor_spec_view(
            dM_previous_t, dM_scan_scratch_spec
        )
        dM_current_view = _make_static_tensor_spec_view(
            dM_current_t, dM_scan_scratch_spec
        )
        dtype = cast(type[cutlass.Numeric], U_view.element_type)
        kernel = BwdDBAmpere(
            dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_db,
        )
        kernel.call_on_stream(
            U_view,
            B_view,
            C_view,
            M_view,
            K_view,
            d_out_view,
            U_prev_view,
            B_prev_view,
            dB_view,
            dB_prev_view,
            dM_previous_view,
            dM_current_view,
            stream,
        )

    return _bwd_db_host_wrapper


def _build_v2x2ssd_bwd_db_runner(cfg: V2KernelPerfConfig) -> KernelRunner:
    runtime_artifacts = _build_v2x2ssd_bwd_runtime_artifacts(cfg)
    scan_specs = _scan_bwd_tensor_specs(runtime_artifacts.problem_shape)
    db_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[0],
                runtime_artifacts.tc_dtype,
                scan_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[1],
                runtime_artifacts.tc_dtype,
                scan_specs[1],
            ),
            (
                runtime_artifacts.runtime_args[2],
                runtime_artifacts.tc_dtype,
                scan_specs[1],
            ),
            (runtime_artifacts.runtime_args[3], torch.float32, scan_specs[2]),
            (runtime_artifacts.runtime_args[4], torch.float32, scan_specs[3]),
            (
                runtime_artifacts.runtime_args[7],
                runtime_artifacts.tc_dtype,
                scan_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[10],
                runtime_artifacts.tc_dtype,
                scan_specs[5],
            ),
            (
                runtime_artifacts.runtime_args[11],
                runtime_artifacts.tc_dtype,
                scan_specs[6],
            ),
            (
                runtime_artifacts.runtime_args[21],
                runtime_artifacts.tc_dtype,
                scan_specs[17],
            ),
            (
                runtime_artifacts.runtime_args[22],
                runtime_artifacts.tc_dtype,
                scan_specs[19],
            ),
            (runtime_artifacts.runtime_args[26], torch.float32, scan_specs[8]),
            (runtime_artifacts.runtime_args[27], torch.float32, scan_specs[8]),
        )
    )
    compiled = cute.compile(
        _make_bwd_db_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
            launch_cfg=runtime_artifacts.launch_cfg,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.outputs.dB_scan.zero_()
        runtime_artifacts.outputs.dB_prev_scan.zero_()
        runtime_artifacts.runtime_args[26].zero_()
        runtime_artifacts.runtime_args[27].zero_()

    prepare()
    compiled(*db_args)
    torch.cuda.synchronize(cfg.torch_device)

    return KernelRunner(
        name="bwd_db",
        effective_bytes=_tensor_bytes(*db_args),
        launch=lambda compiled=compiled, args=db_args: compiled(*args),
        prepare=prepare,
        note="top-level v2x2ssd backward DB launch",
    )


def _make_bwd_dcdr_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        _n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    scan_num_threads_dcdr = int(launch_cfg[2])
    scan_specs = _scan_bwd_tensor_specs(problem_shape)
    U_scan_spec = scan_specs[0]
    BC_scan_spec = scan_specs[1]
    M_scan_spec = scan_specs[2]
    K_scan_spec = scan_specs[3]
    chunk_starts_scan_spec = scan_specs[4]
    U_prev_scan_spec = scan_specs[5]
    B_prev_scan_spec = scan_specs[6]
    dlogp_scan_spec = scan_specs[7]
    dR_scan_spec = scan_specs[9]
    dBC_scan_spec = scan_specs[17]

    @cute.jit
    def _bwd_dcdr_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        C_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        d_out_t: cute.Tensor,
        U_prev_t: cute.Tensor,
        B_prev_t: cute.Tensor,
        chunk_starts_t: cute.Tensor,
        dC_t: cute.Tensor,
        dlogp_t: cute.Tensor,
        dR_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        U_view = _make_static_tensor_spec_view(U_t, U_scan_spec)
        B_view = _make_static_tensor_spec_view(B_t, BC_scan_spec)
        C_view = _make_static_tensor_spec_view(C_t, BC_scan_spec)
        M_view = _make_static_tensor_spec_view(M_t, M_scan_spec)
        K_view = _make_static_tensor_spec_view(K_t, K_scan_spec)
        d_out_view = _make_static_tensor_spec_view(d_out_t, U_scan_spec)
        U_prev_view = _make_static_tensor_spec_view(U_prev_t, U_prev_scan_spec)
        B_prev_view = _make_static_tensor_spec_view(B_prev_t, B_prev_scan_spec)
        chunk_starts_view = _make_static_tensor_spec_view(
            chunk_starts_t, chunk_starts_scan_spec
        )
        dC_view = _make_static_tensor_spec_view(dC_t, dBC_scan_spec)
        dlogp_view = _make_static_tensor_spec_view(dlogp_t, dlogp_scan_spec)
        dR_view = _make_static_tensor_spec_view(dR_t, dR_scan_spec)
        dtype = cast(type[cutlass.Numeric], U_view.element_type)
        kernel = BwdDCDRAmpere(
            dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_dcdr,
        )
        kernel.call_on_stream(
            U_view,
            B_view,
            C_view,
            M_view,
            K_view,
            d_out_view,
            U_prev_view,
            B_prev_view,
            chunk_starts_view,
            dC_view,
            dlogp_view,
            dR_view,
            stream,
        )

    return _bwd_dcdr_host_wrapper


def _build_v2x2ssd_bwd_dcdr_runner(cfg: V2KernelPerfConfig) -> KernelRunner:
    runtime_artifacts = _build_v2x2ssd_bwd_runtime_artifacts(cfg)
    scan_specs = _scan_bwd_tensor_specs(runtime_artifacts.problem_shape)
    dcdr_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[0],
                runtime_artifacts.tc_dtype,
                scan_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[1],
                runtime_artifacts.tc_dtype,
                scan_specs[1],
            ),
            (
                runtime_artifacts.runtime_args[2],
                runtime_artifacts.tc_dtype,
                scan_specs[1],
            ),
            (runtime_artifacts.runtime_args[3], torch.float32, scan_specs[2]),
            (runtime_artifacts.runtime_args[4], torch.float32, scan_specs[3]),
            (
                runtime_artifacts.runtime_args[7],
                runtime_artifacts.tc_dtype,
                scan_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[10],
                runtime_artifacts.tc_dtype,
                scan_specs[5],
            ),
            (
                runtime_artifacts.runtime_args[11],
                runtime_artifacts.tc_dtype,
                scan_specs[6],
            ),
            (runtime_artifacts.runtime_args[9], torch.float32, scan_specs[4]),
            (
                runtime_artifacts.runtime_args[23],
                runtime_artifacts.tc_dtype,
                scan_specs[17],
            ),
            (runtime_artifacts.runtime_args[24], torch.float32, scan_specs[7]),
            (runtime_artifacts.runtime_args[25], torch.float32, scan_specs[9]),
        )
    )
    compiled = cute.compile(
        _make_bwd_dcdr_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
            launch_cfg=runtime_artifacts.launch_cfg,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.outputs.dC_scan.zero_()
        runtime_artifacts.runtime_args[24].zero_()
        runtime_artifacts.runtime_args[25].zero_()

    prepare()
    compiled(*dcdr_args)
    torch.cuda.synchronize(cfg.torch_device)

    return KernelRunner(
        name="bwd_dcdr",
        effective_bytes=_tensor_bytes(*dcdr_args),
        launch=lambda compiled=compiled, args=dcdr_args: compiled(*args),
        prepare=prepare,
        note="top-level v2x2ssd backward DCDR launch",
    )


def _make_bwd_param_scan_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg,
):
    (
        _batch_size,
        _heads,
        _bc_groups,
        _padded_time,
        _P,
        _D,
        _n_chunks,
        chunk_size,
        _,
    ) = problem_shape
    scan_num_threads_param = int(launch_cfg[3])
    scan_specs = _scan_bwd_tensor_specs(problem_shape)
    M_scan_spec = scan_specs[2]
    K_scan_spec = scan_specs[3]
    dlogp_param_spec = scan_specs[11]
    dR_param_spec = scan_specs[12]
    d_param_scan_spec = scan_specs[10]

    @cute.jit
    def _bwd_param_scan_host_wrapper(
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        dlogp_t: cute.Tensor,
        dM_previous_t: cute.Tensor,
        dM_current_t: cute.Tensor,
        dR_t: cute.Tensor,
        dM_t: cute.Tensor,
        dK_previous_t: cute.Tensor,
        dK_current_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M_view = _make_static_tensor_spec_view(M_t, M_scan_spec)
        K_view = _make_static_tensor_spec_view(K_t, K_scan_spec)
        dlogp_view = _make_static_tensor_spec_view(dlogp_t, dlogp_param_spec)
        dM_previous_view = _make_static_tensor_spec_view(
            dM_previous_t, d_param_scan_spec
        )
        dM_current_view = _make_static_tensor_spec_view(dM_current_t, d_param_scan_spec)
        dR_view = _make_static_tensor_spec_view(dR_t, dR_param_spec)
        dM_view = _make_static_tensor_spec_view(dM_t, d_param_scan_spec)
        dK_previous_view = _make_static_tensor_spec_view(
            dK_previous_t, d_param_scan_spec
        )
        dK_current_view = _make_static_tensor_spec_view(dK_current_t, d_param_scan_spec)
        kernel = BwdParamScanAmpere(
            chunk_size=chunk_size,
            num_threads=scan_num_threads_param,
        )
        kernel.call_on_stream(
            M_view,
            K_view,
            dlogp_view,
            dM_previous_view,
            dM_current_view,
            dR_view,
            dM_view,
            dK_previous_view,
            dK_current_view,
            stream,
        )

    return _bwd_param_scan_host_wrapper


def _build_v2x2ssd_bwd_param_scan_runner(cfg: V2KernelPerfConfig) -> KernelRunner:
    runtime_artifacts = _build_v2x2ssd_bwd_runtime_artifacts(cfg)
    _prime_v2x2ssd_bwd_dependencies(runtime_artifacts, device=cfg.torch_device)
    scan_specs = _scan_bwd_tensor_specs(runtime_artifacts.problem_shape)
    param_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (runtime_artifacts.runtime_args[3], torch.float32, scan_specs[2]),
            (runtime_artifacts.runtime_args[4], torch.float32, scan_specs[3]),
            (runtime_artifacts.runtime_args[24], torch.float32, scan_specs[11]),
            (runtime_artifacts.runtime_args[26], torch.float32, scan_specs[10]),
            (runtime_artifacts.runtime_args[27], torch.float32, scan_specs[10]),
            (runtime_artifacts.runtime_args[25], torch.float32, scan_specs[12]),
            (runtime_artifacts.runtime_args[28], torch.float32, scan_specs[10]),
            (runtime_artifacts.runtime_args[29], torch.float32, scan_specs[10]),
            (runtime_artifacts.runtime_args[30], torch.float32, scan_specs[10]),
        )
    )
    compiled = cute.compile(
        _make_bwd_param_scan_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
            launch_cfg=runtime_artifacts.launch_cfg,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.outputs.dM_scan.zero_()
        runtime_artifacts.outputs.dK_scan.zero_()

    prepare()
    compiled(*param_args)
    torch.cuda.synchronize(cfg.torch_device)

    return KernelRunner(
        name="bwd_param_scan",
        effective_bytes=_tensor_bytes(*param_args),
        launch=lambda compiled=compiled, args=param_args: compiled(*args),
        prepare=prepare,
        note="top-level v2x2ssd backward parameter scan launch",
    )


def _make_bwd_du_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        _n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    scan_num_threads_du = int(launch_cfg[0])
    scan_specs = _scan_bwd_tensor_specs(problem_shape)
    U_scan_spec = scan_specs[0]
    BC_scan_spec = scan_specs[1]
    M_scan_spec = scan_specs[2]
    K_scan_spec = scan_specs[3]
    U_prev_scan_spec = scan_specs[5]
    B_prev_scan_spec = scan_specs[6]
    dU_prev_scan_spec = scan_specs[18]

    @cute.jit
    def _bwd_du_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        C_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        d_out_t: cute.Tensor,
        U_prev_t: cute.Tensor,
        B_prev_t: cute.Tensor,
        dU_t: cute.Tensor,
        dU_prev_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        U_view = _make_static_tensor_spec_view(U_t, U_scan_spec)
        B_view = _make_static_tensor_spec_view(B_t, BC_scan_spec)
        C_view = _make_static_tensor_spec_view(C_t, BC_scan_spec)
        M_view = _make_static_tensor_spec_view(M_t, M_scan_spec)
        K_view = _make_static_tensor_spec_view(K_t, K_scan_spec)
        d_out_view = _make_static_tensor_spec_view(d_out_t, U_scan_spec)
        U_prev_view = _make_static_tensor_spec_view(U_prev_t, U_prev_scan_spec)
        B_prev_view = _make_static_tensor_spec_view(B_prev_t, B_prev_scan_spec)
        dU_view = _make_static_tensor_spec_view(dU_t, U_scan_spec)
        dU_prev_view = _make_static_tensor_spec_view(dU_prev_t, dU_prev_scan_spec)
        dtype = cast(type[cutlass.Numeric], U_view.element_type)
        kernel = BwdDUAmpere(
            dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_du,
        )
        kernel.call_on_stream(
            U_view,
            B_view,
            C_view,
            M_view,
            K_view,
            d_out_view,
            U_prev_view,
            B_prev_view,
            dU_view,
            dU_prev_view,
            stream,
        )

    return _bwd_du_host_wrapper


def _build_v2x2ssd_bwd_du_runner(cfg: V2KernelPerfConfig) -> KernelRunner:
    runtime_artifacts = _build_v2x2ssd_bwd_runtime_artifacts(cfg)
    scan_specs = _scan_bwd_tensor_specs(runtime_artifacts.problem_shape)
    du_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[0],
                runtime_artifacts.tc_dtype,
                scan_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[1],
                runtime_artifacts.tc_dtype,
                scan_specs[1],
            ),
            (
                runtime_artifacts.runtime_args[2],
                runtime_artifacts.tc_dtype,
                scan_specs[1],
            ),
            (runtime_artifacts.runtime_args[3], torch.float32, scan_specs[2]),
            (runtime_artifacts.runtime_args[4], torch.float32, scan_specs[3]),
            (
                runtime_artifacts.runtime_args[7],
                runtime_artifacts.tc_dtype,
                scan_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[10],
                runtime_artifacts.tc_dtype,
                scan_specs[5],
            ),
            (
                runtime_artifacts.runtime_args[11],
                runtime_artifacts.tc_dtype,
                scan_specs[6],
            ),
            (
                runtime_artifacts.runtime_args[19],
                runtime_artifacts.tc_dtype,
                scan_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[20],
                runtime_artifacts.tc_dtype,
                scan_specs[18],
            ),
        )
    )
    compiled = cute.compile(
        _make_bwd_du_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
            launch_cfg=runtime_artifacts.launch_cfg,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.outputs.dU_scan.zero_()
        runtime_artifacts.outputs.dU_prev_scan.zero_()

    prepare()
    compiled(*du_args)
    torch.cuda.synchronize(cfg.torch_device)

    return KernelRunner(
        name="bwd_du",
        effective_bytes=_tensor_bytes(*du_args),
        launch=lambda compiled=compiled, args=du_args: compiled(*args),
        prepare=prepare,
        note="top-level v2x2ssd backward DU launch",
    )


def _make_bwd_dz0_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        _P,
        _D,
        _n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    dz0_cta_tiler = launch_cfg[11]
    scan_specs = _scan_bwd_tensor_specs(problem_shape)
    d_out_dz0_spec = scan_specs[13]
    C_dz0_spec = scan_specs[14]
    M_dz0_spec = scan_specs[15]
    d_chunk_starts_scan_spec = scan_specs[16]

    @cute.jit
    def _bwd_dz0_host_wrapper(
        d_out_t: cute.Tensor,
        C_t: cute.Tensor,
        M_t: cute.Tensor,
        d_chunk_starts_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        d_out_view = _make_static_tensor_spec_view(d_out_t, d_out_dz0_spec)
        C_view = _make_static_tensor_spec_view(C_t, C_dz0_spec)
        M_view = _make_static_tensor_spec_view(M_t, M_dz0_spec)
        d_chunk_starts_view = _make_static_tensor_spec_view(
            d_chunk_starts_t, d_chunk_starts_scan_spec
        )
        dtype = cast(type[cutlass.Numeric], C_view.element_type)
        kernel = BwdDZ0Ampere(
            dtype,
            chunk_size=chunk_size,
            heads=heads,
            bc_groups=bc_groups,
            cta_tiler=dz0_cta_tiler,
        )
        kernel.call_on_stream(
            d_out_view,
            C_view,
            M_view,
            d_chunk_starts_view,
            stream,
        )

    return _bwd_dz0_host_wrapper


def _build_v2x2ssd_bwd_dz0_runner(cfg: V2KernelPerfConfig) -> KernelRunner:
    runtime_artifacts = _build_v2x2ssd_bwd_runtime_artifacts(cfg)
    scan_specs = _scan_bwd_tensor_specs(runtime_artifacts.problem_shape)
    dz0_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[7],
                runtime_artifacts.tc_dtype,
                scan_specs[13],
            ),
            (
                runtime_artifacts.runtime_args[2],
                runtime_artifacts.tc_dtype,
                scan_specs[14],
            ),
            (runtime_artifacts.runtime_args[3], torch.float32, scan_specs[15]),
            (runtime_artifacts.runtime_args[15], torch.float32, scan_specs[16]),
        )
    )
    compiled = cute.compile(
        _make_bwd_dz0_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
            launch_cfg=runtime_artifacts.launch_cfg,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.runtime_args[15].zero_()

    prepare()
    compiled(*dz0_args)
    torch.cuda.synchronize(cfg.torch_device)

    return KernelRunner(
        name="bwd_dz0",
        effective_bytes=_tensor_bytes(*dz0_args),
        launch=lambda compiled=compiled, args=dz0_args: compiled(*args),
        prepare=prepare,
        note="top-level v2x2ssd backward DZ0 launch",
    )


def _make_bwd_state_passing_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg,
):
    (
        _batch_size,
        _heads,
        _bc_groups,
        _padded_time,
        _P,
        _D,
        _n_chunks,
        _chunk_size,
        _n_d_tiles,
    ) = problem_shape
    (
        _scan_num_threads_du,
        _scan_num_threads_db,
        _scan_num_threads_dcdr,
        _scan_num_threads_param,
        state_num_threads,
        state_pairs_per_thread,
        state_copy_bits_starts,
        state_copy_bits_dstarts,
        state_copy_bits_dinc,
        state_copy_bits_initial,
        state_copy_bits_final,
        _dz0_cta_tiler,
    ) = launch_cfg
    state_specs = _state_bwd_tensor_specs(problem_shape)
    chunk_starts_state_spec = state_specs[0]
    chunk_multiplier_state_spec = state_specs[1]
    final_state_spec = state_specs[2]

    @cute.jit
    def _bwd_state_passing_host_wrapper(
        chunk_starts_t: cute.Tensor,
        d_chunk_starts_t: cute.Tensor,
        d_final_state_t: cute.Tensor,
        chunk_multiplier_t: cute.Tensor,
        d_increment_t: cute.Tensor,
        d_chunk_multiplier_t: cute.Tensor,
        d_initial_state_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        chunk_starts_view = _make_static_tensor_spec_view(
            chunk_starts_t, chunk_starts_state_spec
        )
        d_chunk_starts_view = _make_static_tensor_spec_view(
            d_chunk_starts_t, chunk_starts_state_spec
        )
        d_final_state_view = _make_static_tensor_spec_view(
            d_final_state_t, final_state_spec
        )
        chunk_multiplier_view = _make_static_tensor_spec_view(
            chunk_multiplier_t, chunk_multiplier_state_spec
        )
        d_increment_view = _make_static_tensor_spec_view(
            d_increment_t, chunk_starts_state_spec
        )
        d_chunk_multiplier_view = _make_static_tensor_spec_view(
            d_chunk_multiplier_t, chunk_multiplier_state_spec
        )
        d_initial_state_view = _make_static_tensor_spec_view(
            d_initial_state_t, final_state_spec
        )
        kernel = BwdStatePassingAmpere(
            _TileConfig(
                num_threads=state_num_threads,
                pairs_per_thread=state_pairs_per_thread,
            ),
            copy_bits_starts=state_copy_bits_starts,
            copy_bits_dstarts=state_copy_bits_dstarts,
            copy_bits_dinc=state_copy_bits_dinc,
            copy_bits_initial=state_copy_bits_initial,
            copy_bits_final=state_copy_bits_final,
        )
        kernel.call_on_stream(
            chunk_starts_view,
            d_chunk_starts_view,
            d_final_state_view,
            chunk_multiplier_view,
            d_increment_view,
            d_chunk_multiplier_view,
            d_initial_state_view,
            stream,
        )

    return _bwd_state_passing_host_wrapper


def _build_v2x2ssd_bwd_state_passing_runner(
    cfg: V2KernelPerfConfig,
) -> KernelRunner:
    runtime_artifacts = _build_v2x2ssd_bwd_runtime_artifacts(cfg)
    _prime_v2x2ssd_bwd_dependencies(runtime_artifacts, device=cfg.torch_device)
    state_specs = _state_bwd_tensor_specs(runtime_artifacts.problem_shape)
    state_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (runtime_artifacts.runtime_args[9], torch.float32, state_specs[0]),
            (runtime_artifacts.runtime_args[15], torch.float32, state_specs[0]),
            (runtime_artifacts.runtime_args[12], torch.float32, state_specs[2]),
            (runtime_artifacts.runtime_args[8], torch.float32, state_specs[1]),
            (
                runtime_artifacts.runtime_args[16],
                runtime_artifacts.tc_dtype,
                state_specs[0],
            ),
            (runtime_artifacts.runtime_args[18], torch.float32, state_specs[1]),
            (runtime_artifacts.runtime_args[17], torch.float32, state_specs[2]),
        )
    )
    compiled = cute.compile(
        _make_bwd_state_passing_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
            launch_cfg=runtime_artifacts.launch_cfg,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.runtime_args[16].zero_()
        runtime_artifacts.outputs.d_chunk_multiplier.zero_()
        runtime_artifacts.outputs.d_initial_state.zero_()

    prepare()
    compiled(*state_args)
    torch.cuda.synchronize(cfg.torch_device)

    return KernelRunner(
        name="bwd_state_passing",
        effective_bytes=_tensor_bytes(*state_args),
        launch=lambda compiled=compiled, args=state_args: compiled(*args),
        prepare=prepare,
        note="top-level v2x2ssd backward state passing launch",
    )


def _make_bwd_boundary_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    increment_specs = _increment_bwd_tensor_specs(problem_shape)
    M_increment_spec = increment_specs[3]
    K_increment_spec = increment_specs[4]
    d_boundary_spec = increment_specs[7]
    U_prev_chunks_spec = increment_specs[8]
    B_prev_chunks_spec = increment_specs[9]
    dMp0_spec = increment_specs[11]
    dB_prev_boundary_spec = increment_specs[15]

    @cute.jit
    def _bwd_boundary_host_wrapper(
        d_increment_t: cute.Tensor,
        B_prev_chunks_t: cute.Tensor,
        U_prev_chunks_t: cute.Tensor,
        M_t: cute.Tensor,
        K_previous_t: cute.Tensor,
        dU_prev_scan_t: cute.Tensor,
        dB_prev_scan_t: cute.Tensor,
        dMp0_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        d_boundary_view = _make_static_tensor_spec_view(
            d_increment_t,
            d_boundary_spec,
        )
        B_prev_chunks_view = _make_static_tensor_spec_view(
            B_prev_chunks_t,
            B_prev_chunks_spec,
        )
        U_prev_chunks_view = _make_static_tensor_spec_view(
            U_prev_chunks_t,
            U_prev_chunks_spec,
        )
        M_increment_view = _make_static_tensor_spec_view(M_t, M_increment_spec)
        K_previous_increment_view = _make_static_tensor_spec_view(
            K_previous_t,
            K_increment_spec,
        )
        dU_prev_increment_view = _make_static_tensor_spec_view(
            dU_prev_scan_t,
            U_prev_chunks_spec,
        )
        dB_prev_boundary_view = _make_static_tensor_spec_view(
            dB_prev_scan_t,
            dB_prev_boundary_spec,
        )
        dMp0_view = _make_static_tensor_spec_view(dMp0_t, dMp0_spec)
        dtype = cast(type[cutlass.Numeric], d_boundary_view.element_type)
        kernel = BwdBoundaryAmpere(
            dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )
        kernel.call_on_stream(
            d_boundary_view,
            B_prev_chunks_view,
            U_prev_chunks_view,
            M_increment_view,
            K_previous_increment_view,
            dU_prev_increment_view,
            dB_prev_boundary_view,
            dMp0_view,
            stream,
        )

    return _bwd_boundary_host_wrapper


def _build_v2x2ssd_bwd_boundary_runner(
    cfg: V2KernelPerfConfig,
) -> KernelRunner:
    _seed_all(cfg.seed)
    bc_groups, _heads_per_group = _validate_bc_groups(
        cfg.heads,
        cfg.resolved_bc_groups,
    )
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B_grouped"]
    C = tensors["C_grouped"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev_grouped"]
    U_prev = tensors["U_prev"]

    m_chunk, chunk_starts = _compute_forward_intermediates_for_bwd(
        cfg,
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    d_out = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    d_final = torch.randn_like(initial_states)
    runtime_artifacts = _make_backward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
        scan_num_threads_du=128,
        scan_num_threads_db=128,
        scan_num_threads_dcdr=128,
        scan_num_threads_param=32,
        state_num_threads=128,
        state_pairs_per_thread=8,
        B_prev=B_prev,
        U_prev=U_prev,
        initial_states=initial_states,
        d_final_state=d_final,
        validate_runtime_contract=True,
    )
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    full_bwd = cast(
        Callable[..., None],
        _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts),
    )
    full_bwd(*runtime_artifacts.runtime_args)
    torch.cuda.synchronize(cfg.torch_device)

    increment_specs = _increment_bwd_tensor_specs(runtime_artifacts.problem_shape)
    boundary_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[16],
                runtime_artifacts.tc_dtype,
                increment_specs[7],
            ),
            (
                runtime_artifacts.runtime_args[14],
                runtime_artifacts.tc_dtype,
                increment_specs[9],
            ),
            (
                runtime_artifacts.runtime_args[13],
                runtime_artifacts.tc_dtype,
                increment_specs[8],
            ),
            (
                runtime_artifacts.runtime_args[3],
                torch.float32,
                increment_specs[3],
            ),
            (
                runtime_artifacts.runtime_args[5],
                torch.float32,
                increment_specs[4],
            ),
            (
                runtime_artifacts.runtime_args[20],
                runtime_artifacts.tc_dtype,
                increment_specs[8],
            ),
            (
                runtime_artifacts.runtime_args[22],
                runtime_artifacts.tc_dtype,
                increment_specs[15],
            ),
            (
                runtime_artifacts.runtime_args[33],
                torch.float32,
                increment_specs[11],
            ),
        )
    )
    compiled = cute.compile(
        _make_bwd_boundary_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    d_mp0 = runtime_artifacts.runtime_args[33]

    def prepare() -> None:
        runtime_artifacts.outputs.dU_prev_scan.zero_()
        runtime_artifacts.outputs.dB_prev_scan.zero_()
        d_mp0.zero_()

    return KernelRunner(
        name="bwd_boundary",
        effective_bytes=_tensor_bytes(runtime_artifacts.runtime_args[16])
        + _tensor_bytes(runtime_artifacts.runtime_args[14])
        + _tensor_bytes(runtime_artifacts.runtime_args[13])
        + _tensor_bytes(runtime_artifacts.runtime_args[3])
        + _tensor_bytes(runtime_artifacts.runtime_args[5])
        + 2 * _tensor_bytes(runtime_artifacts.outputs.dU_prev_scan)
        + 2 * _tensor_bytes(runtime_artifacts.outputs.dB_prev_scan)
        + _tensor_bytes(d_mp0),
        launch=lambda compiled=compiled, args=boundary_args: compiled(*args),
        prepare=prepare,
        note="top-level bwd boundary accumulator",
    )


def _make_bwd_db_increment_accumulator_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    scan_specs = _scan_bwd_tensor_specs(problem_shape)
    increment_specs = _increment_bwd_tensor_specs(problem_shape)
    dBC_scan_spec = scan_specs[17]
    U_increment_spec = increment_specs[0]
    B_increment_spec = increment_specs[2]
    M_increment_spec = increment_specs[3]
    K_increment_spec = increment_specs[4]
    d_increment_dp_spec = increment_specs[6]
    dM_sum_part_spec = increment_specs[10]

    @cute.jit
    def _bwd_db_increment_accumulator_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        M_t: cute.Tensor,
        K_previous_t: cute.Tensor,
        K_current_t: cute.Tensor,
        d_increment_t: cute.Tensor,
        dB_scan_t: cute.Tensor,
        dM_sum_part_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        U_increment_view = _make_static_tensor_spec_view(U_t, U_increment_spec)
        B_increment_view = _make_static_tensor_spec_view(B_t, B_increment_spec)
        M_increment_view = _make_static_tensor_spec_view(M_t, M_increment_spec)
        K_previous_increment_view = _make_static_tensor_spec_view(
            K_previous_t,
            K_increment_spec,
        )
        K_current_increment_view = _make_static_tensor_spec_view(
            K_current_t,
            K_increment_spec,
        )
        d_increment_dp_view = _make_static_tensor_spec_view(
            d_increment_t,
            d_increment_dp_spec,
        )
        dB_scan_view = _make_static_tensor_spec_view(dB_scan_t, dBC_scan_spec)
        dM_sum_part_view = _make_static_tensor_spec_view(
            dM_sum_part_t,
            dM_sum_part_spec,
        )
        dtype = cast(type[cutlass.Numeric], U_increment_view.element_type)
        kernel = BwdDBIncrementAccumulatorAmpere(
            dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )
        kernel.call_on_stream(
            U_increment_view,
            B_increment_view,
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            d_increment_dp_view,
            dB_scan_view,
            dM_sum_part_view,
            stream,
        )

    return _bwd_db_increment_accumulator_host_wrapper


def _build_v2x2ssd_bwd_db_increment_accumulator_runner(
    cfg: V2KernelPerfConfig,
) -> KernelRunner:
    _seed_all(cfg.seed)
    bc_groups, _heads_per_group = _validate_bc_groups(
        cfg.heads,
        cfg.resolved_bc_groups,
    )
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B_grouped"]
    C = tensors["C_grouped"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev_grouped"]
    U_prev = tensors["U_prev"]

    m_chunk, chunk_starts = _compute_forward_intermediates_for_bwd(
        cfg,
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    d_out = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    d_final = torch.randn_like(initial_states)
    runtime_artifacts = _make_backward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
        scan_num_threads_du=128,
        scan_num_threads_db=128,
        scan_num_threads_dcdr=128,
        scan_num_threads_param=32,
        state_num_threads=128,
        state_pairs_per_thread=8,
        B_prev=B_prev,
        U_prev=U_prev,
        initial_states=initial_states,
        d_final_state=d_final,
        validate_runtime_contract=True,
    )
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    full_bwd = cast(
        Callable[..., None],
        _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts),
    )
    full_bwd(*runtime_artifacts.runtime_args)
    torch.cuda.synchronize(cfg.torch_device)

    scan_specs = _scan_bwd_tensor_specs(runtime_artifacts.problem_shape)
    increment_specs = _increment_bwd_tensor_specs(runtime_artifacts.problem_shape)
    accumulator_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[0],
                runtime_artifacts.tc_dtype,
                increment_specs[0],
            ),
            (
                runtime_artifacts.runtime_args[1],
                runtime_artifacts.tc_dtype,
                increment_specs[2],
            ),
            (
                runtime_artifacts.runtime_args[3],
                torch.float32,
                increment_specs[3],
            ),
            (
                runtime_artifacts.runtime_args[5],
                torch.float32,
                increment_specs[4],
            ),
            (
                runtime_artifacts.runtime_args[6],
                torch.float32,
                increment_specs[4],
            ),
            (
                runtime_artifacts.runtime_args[16],
                runtime_artifacts.tc_dtype,
                increment_specs[6],
            ),
            (
                runtime_artifacts.runtime_args[21],
                runtime_artifacts.tc_dtype,
                scan_specs[17],
            ),
            (
                runtime_artifacts.runtime_args[32],
                torch.float32,
                increment_specs[10],
            ),
        )
    )
    compiled = cute.compile(
        _make_bwd_db_increment_accumulator_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    d_msum_part = runtime_artifacts.runtime_args[32]

    def prepare() -> None:
        runtime_artifacts.outputs.dB_scan.zero_()
        d_msum_part.zero_()

    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    n_chunks = _n_chunks(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    tc_dtype = _tc_input_dtype(cfg.dtype)
    bytes_u = _shape_bytes((cfg.batch, cfg.heads, T_pad, cfg.P), tc_dtype)
    bytes_b = _shape_bytes((cfg.batch, bc_groups, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_d_inc = _shape_bytes(
        (cfg.batch, cfg.heads, n_chunks, cfg.P, D),
        torch.float32,
    )

    return KernelRunner(
        name="bwd_db_increment_accumulator",
        effective_bytes=bytes_u
        + bytes_b
        + bytes_m
        + bytes_k
        + bytes_d_inc
        + 2 * _tensor_bytes(runtime_artifacts.outputs.dB_scan)
        + _tensor_bytes(d_msum_part),
        launch=lambda compiled=compiled, args=accumulator_args: compiled(*args),
        prepare=prepare,
        note="top-level bwd DB increment accumulator",
    )


def _make_bwd_du_increment_accumulator_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    increment_specs = _increment_bwd_tensor_specs(problem_shape)
    dU_increment_spec = increment_specs[1]
    B_increment_spec = increment_specs[2]
    M_increment_spec = increment_specs[3]
    K_increment_spec = increment_specs[4]
    d_increment_spec = increment_specs[5]

    @cute.jit
    def _bwd_du_increment_accumulator_host_wrapper(
        d_increment_t: cute.Tensor,
        B_t: cute.Tensor,
        M_t: cute.Tensor,
        K_previous_t: cute.Tensor,
        K_current_t: cute.Tensor,
        dU_scan_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        d_increment_view = _make_static_tensor_spec_view(
            d_increment_t,
            d_increment_spec,
        )
        B_increment_view = _make_static_tensor_spec_view(B_t, B_increment_spec)
        M_increment_view = _make_static_tensor_spec_view(M_t, M_increment_spec)
        K_previous_increment_view = _make_static_tensor_spec_view(
            K_previous_t,
            K_increment_spec,
        )
        K_current_increment_view = _make_static_tensor_spec_view(
            K_current_t,
            K_increment_spec,
        )
        dU_increment_view = _make_static_tensor_spec_view(
            dU_scan_t,
            dU_increment_spec,
        )
        dtype = cast(type[cutlass.Numeric], B_increment_view.element_type)
        kernel = BwdDUIncrementAccumulatorAmpere(
            dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )
        kernel.call_on_stream(
            d_increment_view,
            B_increment_view,
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            dU_increment_view,
            stream,
        )

    return _bwd_du_increment_accumulator_host_wrapper


def _build_v2x2ssd_bwd_du_increment_accumulator_runner(
    cfg: V2KernelPerfConfig,
) -> KernelRunner:
    _seed_all(cfg.seed)
    bc_groups, _heads_per_group = _validate_bc_groups(
        cfg.heads,
        cfg.resolved_bc_groups,
    )
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B_grouped"]
    C = tensors["C_grouped"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev_grouped"]
    U_prev = tensors["U_prev"]

    m_chunk, chunk_starts = _compute_forward_intermediates_for_bwd(
        cfg,
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    d_out = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    d_final = torch.randn_like(initial_states)
    runtime_artifacts = _make_backward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
        scan_num_threads_du=128,
        scan_num_threads_db=128,
        scan_num_threads_dcdr=128,
        scan_num_threads_param=32,
        state_num_threads=128,
        state_pairs_per_thread=8,
        B_prev=B_prev,
        U_prev=U_prev,
        initial_states=initial_states,
        d_final_state=d_final,
        validate_runtime_contract=True,
    )
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    full_bwd = cast(
        Callable[..., None],
        _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts),
    )
    full_bwd(*runtime_artifacts.runtime_args)
    torch.cuda.synchronize(cfg.torch_device)

    increment_specs = _increment_bwd_tensor_specs(runtime_artifacts.problem_shape)
    accumulator_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[16],
                runtime_artifacts.tc_dtype,
                increment_specs[5],
            ),
            (
                runtime_artifacts.runtime_args[1],
                runtime_artifacts.tc_dtype,
                increment_specs[2],
            ),
            (
                runtime_artifacts.runtime_args[3],
                torch.float32,
                increment_specs[3],
            ),
            (
                runtime_artifacts.runtime_args[5],
                torch.float32,
                increment_specs[4],
            ),
            (
                runtime_artifacts.runtime_args[6],
                torch.float32,
                increment_specs[4],
            ),
            (
                runtime_artifacts.runtime_args[19],
                runtime_artifacts.tc_dtype,
                increment_specs[1],
            ),
        )
    )
    compiled = cute.compile(
        _make_bwd_du_increment_accumulator_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.outputs.dU_scan.zero_()

    T_pad = _t_pad(cfg.T, cfg.chunk_size)
    n_chunks = _n_chunks(cfg.T, cfg.chunk_size)
    D = B.shape[-1]
    tc_dtype = _tc_input_dtype(cfg.dtype)
    bytes_b = _shape_bytes((cfg.batch, bc_groups, T_pad, D), tc_dtype)
    bytes_m = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2), torch.float32)
    bytes_k = _shape_bytes((cfg.batch, cfg.heads, T_pad, 2, 2), torch.float32)
    bytes_d_inc = _shape_bytes(
        (cfg.batch, cfg.heads, n_chunks, cfg.P, D),
        torch.float32,
    )

    return KernelRunner(
        name="bwd_du_increment_accumulator",
        effective_bytes=bytes_d_inc
        + bytes_b
        + bytes_m
        + bytes_k
        + 2 * _tensor_bytes(runtime_artifacts.outputs.dU_scan),
        launch=lambda compiled=compiled, args=accumulator_args: compiled(*args),
        prepare=prepare,
        note="top-level bwd DU increment accumulator",
    )


def _make_bwd_param_increment_accumulator_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
):
    (
        _batch_size,
        _heads,
        _bc_groups,
        _padded_time,
        _P,
        _D,
        _n_chunks,
        chunk_size,
        n_d_tiles,
    ) = problem_shape
    increment_specs = _increment_bwd_tensor_specs(problem_shape)
    M_increment_spec = increment_specs[3]
    K_increment_spec = increment_specs[4]
    dM_sum_part_spec = increment_specs[10]
    dMp0_spec = increment_specs[11]
    d_chunk_multiplier_increment_spec = increment_specs[12]
    d_param_increment_spec = increment_specs[13]

    @cute.jit
    def _bwd_param_increment_accumulator_host_wrapper(
        M_t: cute.Tensor,
        K_previous_t: cute.Tensor,
        K_current_t: cute.Tensor,
        dM_sum_part_t: cute.Tensor,
        dMp0_t: cute.Tensor,
        d_chunk_multiplier_t: cute.Tensor,
        dM_scan_t: cute.Tensor,
        dK_previous_scan_t: cute.Tensor,
        dK_current_scan_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M_increment_view = _make_static_tensor_spec_view(M_t, M_increment_spec)
        K_previous_increment_view = _make_static_tensor_spec_view(
            K_previous_t,
            K_increment_spec,
        )
        K_current_increment_view = _make_static_tensor_spec_view(
            K_current_t,
            K_increment_spec,
        )
        dM_sum_part_view = _make_static_tensor_spec_view(
            dM_sum_part_t,
            dM_sum_part_spec,
        )
        dMp0_view = _make_static_tensor_spec_view(dMp0_t, dMp0_spec)
        d_chunk_multiplier_increment_view = _make_static_tensor_spec_view(
            d_chunk_multiplier_t,
            d_chunk_multiplier_increment_spec,
        )
        dM_increment_view = _make_static_tensor_spec_view(
            dM_scan_t,
            d_param_increment_spec,
        )
        dK_previous_increment_view = _make_static_tensor_spec_view(
            dK_previous_scan_t,
            d_param_increment_spec,
        )
        dK_current_increment_view = _make_static_tensor_spec_view(
            dK_current_scan_t,
            d_param_increment_spec,
        )
        kernel = BwdParamIncrementAccumulatorAmpere(
            chunk_size=chunk_size,
            n_d_tiles=n_d_tiles,
        )
        kernel.call_on_stream(
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            dM_sum_part_view,
            dMp0_view,
            d_chunk_multiplier_increment_view,
            dM_increment_view,
            dK_previous_increment_view,
            dK_current_increment_view,
            stream,
        )

    return _bwd_param_increment_accumulator_host_wrapper


def _build_v2x2ssd_bwd_param_increment_accumulator_runner(
    cfg: V2KernelPerfConfig,
) -> KernelRunner:
    _seed_all(cfg.seed)
    tensors = _build_grouped_v2_inputs(cfg)
    U = tensors["U"]
    M = tensors["M"]
    K = tensors["K"]
    B = tensors["B_grouped"]
    C = tensors["C_grouped"]
    initial_states = tensors["initial_states"].to(dtype=torch.float32).contiguous()
    B_prev = tensors["B_prev_grouped"]
    U_prev = tensors["U_prev"]

    m_chunk, chunk_starts = _compute_forward_intermediates_for_bwd(
        cfg,
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    d_out = torch.randn(
        (cfg.batch, cfg.heads, cfg.T, cfg.P),
        device=cfg.torch_device,
        dtype=torch.float32,
    )
    d_final = torch.randn_like(initial_states)
    runtime_artifacts = _make_backward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=cfg.chunk_size,
        compute_dtype=torch.float32,
        scan_num_threads_du=128,
        scan_num_threads_db=128,
        scan_num_threads_dcdr=128,
        scan_num_threads_param=32,
        state_num_threads=128,
        state_pairs_per_thread=8,
        B_prev=B_prev,
        U_prev=U_prev,
        initial_states=initial_states,
        d_final_state=d_final,
        validate_runtime_contract=True,
    )
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    full_bwd = cast(
        Callable[..., None],
        _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts),
    )
    full_bwd(*runtime_artifacts.runtime_args)
    torch.cuda.synchronize(cfg.torch_device)

    increment_specs = _increment_bwd_tensor_specs(runtime_artifacts.problem_shape)
    accumulator_args, _alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (
                runtime_artifacts.runtime_args[3],
                torch.float32,
                increment_specs[3],
            ),
            (
                runtime_artifacts.runtime_args[5],
                torch.float32,
                increment_specs[4],
            ),
            (
                runtime_artifacts.runtime_args[6],
                torch.float32,
                increment_specs[4],
            ),
            (
                runtime_artifacts.runtime_args[32],
                torch.float32,
                increment_specs[10],
            ),
            (
                runtime_artifacts.runtime_args[33],
                torch.float32,
                increment_specs[11],
            ),
            (
                runtime_artifacts.runtime_args[18],
                torch.float32,
                increment_specs[12],
            ),
            (
                runtime_artifacts.runtime_args[28],
                torch.float32,
                increment_specs[13],
            ),
            (
                runtime_artifacts.runtime_args[29],
                torch.float32,
                increment_specs[13],
            ),
            (
                runtime_artifacts.runtime_args[30],
                torch.float32,
                increment_specs[13],
            ),
        )
    )
    compiled = cute.compile(
        _make_bwd_param_increment_accumulator_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
        ),
        *compile_args,
        options="--enable-tvm-ffi",
    )

    def prepare() -> None:
        runtime_artifacts.outputs.dM_scan.zero_()
        runtime_artifacts.outputs.dK_scan.zero_()

    bytes_m = _tensor_bytes(M)
    bytes_k = _tensor_bytes(K)
    bytes_d_msum = _tensor_bytes(runtime_artifacts.runtime_args[32])
    bytes_d_mp0 = _tensor_bytes(runtime_artifacts.runtime_args[33])
    bytes_d_chunk_multiplier = _tensor_bytes(runtime_artifacts.runtime_args[18])

    return KernelRunner(
        name="bwd_param_increment_accumulator",
        effective_bytes=bytes_m
        + bytes_k
        + bytes_d_msum
        + bytes_d_mp0
        + bytes_d_chunk_multiplier
        + 2 * _tensor_bytes(runtime_artifacts.outputs.dM_scan)
        + 2 * _tensor_bytes(runtime_artifacts.outputs.dK_scan),
        launch=lambda compiled=compiled, args=accumulator_args: compiled(*args),
        prepare=prepare,
        note="top-level bwd parameter increment accumulator",
    )


def build_kernel_runners(
    *,
    v2_cfg: V2KernelPerfConfig | None = None,
    scanprep_cfg: ScanPrepPerfConfig | None = None,
    mixer_tail_cfg: MixerTailPerfConfig | None = None,
    ffn_cfg: FfnPerfConfig | None = None,
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
    if mixer_tail_cfg is None:
        mixer_tail_cfg = MixerTailPerfConfig(
            batch=v2_cfg.batch,
            heads=v2_cfg.heads,
            T=v2_cfg.T,
            P=v2_cfg.P,
            dtype=v2_cfg.dtype,
            device=v2_cfg.device,
            seed=v2_cfg.seed,
        )
    if ffn_cfg is None:
        ffn_cfg = FfnPerfConfig(
            batch=v2_cfg.batch,
            T=v2_cfg.T,
            dtype=v2_cfg.dtype,
            device=v2_cfg.device,
            seed=v2_cfg.seed,
        )

    runners: dict[str, KernelRunner] = {}
    runners.update(
        {
            "scanprep_fwd": _build_scanprep_fwd_runner(scanprep_cfg),
            "scanprep_bwd": _build_scanprep_bwd_runner(scanprep_cfg),
            "mixer_tail_rowwise_fwd": _build_mixer_tail_rowwise_fwd_runner(
                mixer_tail_cfg
            ),
            "mixer_tail_rowwise_bwd": _build_mixer_tail_rowwise_bwd_runner(
                mixer_tail_cfg
            ),
            "ffn_norm_fwd": _build_ffn_norm_fwd_runner(ffn_cfg),
            "ffn_norm_bwd": _build_ffn_norm_bwd_runner(ffn_cfg),
            "ffn_activation_fwd": _build_ffn_activation_fwd_runner(ffn_cfg),
            "ffn_activation_bwd": _build_ffn_activation_bwd_runner(ffn_cfg),
        }
    )
    runners["v2x2ssd_fwd"] = _build_v2x2ssd_forward_runner(v2_cfg)
    runners["v2x2ssd_bwd"] = _build_v2x2ssd_backward_runner(v2_cfg)
    runners["bwd_db"] = _build_v2x2ssd_bwd_db_runner(v2_cfg)
    runners["bwd_dcdr"] = _build_v2x2ssd_bwd_dcdr_runner(v2_cfg)
    runners["bwd_param_scan"] = _build_v2x2ssd_bwd_param_scan_runner(v2_cfg)
    runners["bwd_du"] = _build_v2x2ssd_bwd_du_runner(v2_cfg)
    runners["bwd_dz0"] = _build_v2x2ssd_bwd_dz0_runner(v2_cfg)
    runners["bwd_state_passing"] = _build_v2x2ssd_bwd_state_passing_runner(v2_cfg)
    runners["bwd_db_increment_accumulator"] = (
        _build_v2x2ssd_bwd_db_increment_accumulator_runner(v2_cfg)
    )
    runners["bwd_boundary"] = _build_v2x2ssd_bwd_boundary_runner(v2_cfg)
    runners["bwd_param_increment_accumulator"] = (
        _build_v2x2ssd_bwd_param_increment_accumulator_runner(v2_cfg)
    )
    runners["bwd_du_increment_accumulator"] = (
        _build_v2x2ssd_bwd_du_increment_accumulator_runner(v2_cfg)
    )
    return [runners[name] for name in KERNEL_ORDER]


def build_kernel_runner(
    name: str,
    *,
    v2_cfg: V2KernelPerfConfig | None = None,
    scanprep_cfg: ScanPrepPerfConfig | None = None,
    mixer_tail_cfg: MixerTailPerfConfig | None = None,
    ffn_cfg: FfnPerfConfig | None = None,
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
    if mixer_tail_cfg is None:
        mixer_tail_cfg = MixerTailPerfConfig(
            batch=v2_cfg.batch,
            heads=v2_cfg.heads,
            T=v2_cfg.T,
            P=v2_cfg.P,
            dtype=v2_cfg.dtype,
            device=v2_cfg.device,
            seed=v2_cfg.seed,
        )
    if ffn_cfg is None:
        ffn_cfg = FfnPerfConfig(
            batch=v2_cfg.batch,
            T=v2_cfg.T,
            dtype=v2_cfg.dtype,
            device=v2_cfg.device,
            seed=v2_cfg.seed,
        )

    if name == "scanprep_fwd":
        return _build_scanprep_fwd_runner(scanprep_cfg)
    if name == "scanprep_bwd":
        return _build_scanprep_bwd_runner(scanprep_cfg)
    if name == "mixer_tail_rowwise_fwd":
        return _build_mixer_tail_rowwise_fwd_runner(mixer_tail_cfg)
    if name == "mixer_tail_rowwise_bwd":
        return _build_mixer_tail_rowwise_bwd_runner(mixer_tail_cfg)
    if name == "ffn_norm_fwd":
        return _build_ffn_norm_fwd_runner(ffn_cfg)
    if name == "ffn_norm_bwd":
        return _build_ffn_norm_bwd_runner(ffn_cfg)
    if name == "ffn_activation_fwd":
        return _build_ffn_activation_fwd_runner(ffn_cfg)
    if name == "ffn_activation_bwd":
        return _build_ffn_activation_bwd_runner(ffn_cfg)
    if name == "v2x2ssd_fwd":
        return _build_v2x2ssd_forward_runner(v2_cfg)
    if name == "v2x2ssd_bwd":
        return _build_v2x2ssd_backward_runner(v2_cfg)
    if name == "bwd_db":
        return _build_v2x2ssd_bwd_db_runner(v2_cfg)
    if name == "bwd_dcdr":
        return _build_v2x2ssd_bwd_dcdr_runner(v2_cfg)
    if name == "bwd_param_scan":
        return _build_v2x2ssd_bwd_param_scan_runner(v2_cfg)
    if name == "bwd_du":
        return _build_v2x2ssd_bwd_du_runner(v2_cfg)
    if name == "bwd_dz0":
        return _build_v2x2ssd_bwd_dz0_runner(v2_cfg)
    if name == "bwd_state_passing":
        return _build_v2x2ssd_bwd_state_passing_runner(v2_cfg)
    if name == "bwd_db_increment_accumulator":
        return _build_v2x2ssd_bwd_db_increment_accumulator_runner(v2_cfg)
    if name == "bwd_boundary":
        return _build_v2x2ssd_bwd_boundary_runner(v2_cfg)
    if name == "bwd_param_increment_accumulator":
        return _build_v2x2ssd_bwd_param_increment_accumulator_runner(v2_cfg)
    if name == "bwd_du_increment_accumulator":
        return _build_v2x2ssd_bwd_du_increment_accumulator_runner(v2_cfg)
    raise KeyError(f"Unknown kernel runner: {name}")
