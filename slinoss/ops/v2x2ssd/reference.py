"""Reference implementations for the SLinOSS v2x2 scan operator.

The scan maintains a 2x2 state per state lane, identified with a complex scalar
``z = x + i v``. The canonical tensor contract is CUDA-oriented:

- time-major / token-major storage: time is the third dimension
- the hot state axis is flattened as ``D = 2N``
- flattened complex vectors use interleaved pair order
  ``[re0, im0, re1, im1, ...]``

Canonical layouts
-----------------
- ``U``: ``(batch, heads, T, P)``
- ``M``: ``(batch, heads, T, 2)`` packed complex ``(re, im)``
- ``K``: ``(batch, heads, T, 2, 2)`` with tap ``0=prev`` and ``1=curr``,
  each packed as ``(re, im)``
- ``B, C``: ``(batch, groups, T, 2N)`` flattened interleaved complex vectors
  shared across contiguous head groups, where ``groups`` divides ``heads``
- State ``z``: ``(batch, heads, P, 2N)``
- Output ``Y``: ``(batch, heads, T, P)``

Streaming semantics
-------------------
Optional grouped ``B_prev`` and per-head ``U_prev`` provide time-0 "previous"
values:

- ``B_prev_seq[t] = B_prev`` if ``t == 0`` else ``B[..., t-1]``
- ``U_prev_seq[t] = U_prev`` if ``t == 0`` else ``U[..., t-1]``

This file provides three forward references:

- ``v2x2ssm``: sequential oracle (Python loop over ``T``)
- ``v2x2ssd_ref``: chunked mathematical reference
- ``v2x2ssd``: 3-stage kernel-shaped decomposition

Numerical contract
------------------
The chunked references assume the standard SLinOSS operating region: packed
complex transitions are finite and strictly nonzero. This matches the model
parameterization used in the paper, where the contraction radius is bounded away
from zero. Exact-zero transitions remain well-defined in the sequential oracle
``v2x2ssm``, but are treated as a contract violation for the chunked paths.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _promote_real_dtypes(dtypes: Sequence[torch.dtype]) -> torch.dtype:
    if not dtypes:
        raise ValueError("Expected at least one dtype.")
    out = dtypes[0]
    for dt in dtypes[1:]:
        out = torch.promote_types(out, dt)
    return out


def _resolve_dtypes(
    *,
    input_dtypes: Sequence[torch.dtype],
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
    default_output_dtype: torch.dtype,
) -> tuple[torch.dtype, torch.dtype]:
    if compute_dtype is None:
        promoted = _promote_real_dtypes(list(input_dtypes))
        rdtype = (
            torch.float32 if promoted in (torch.float16, torch.bfloat16) else promoted
        )
    else:
        rdtype = compute_dtype

    if rdtype not in (torch.float32, torch.float64):
        raise ValueError(f"compute_dtype must be float32 or float64, got {rdtype}.")

    odtype = default_output_dtype if output_dtype is None else output_dtype
    if odtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"output_dtype must be a real floating dtype, got {odtype}.")
    return rdtype, odtype


def _complex_dtype_from_real(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise ValueError(
        f"Complex arithmetic requires float32 or float64 compute, got {real_dtype}."
    )


def _check_contiguous(name: str, x: torch.Tensor) -> None:
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous; got strides {x.stride()}.")


def _check_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values.")


def _check_reference_inputs_finite(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> None:
    for name, tensor in (("U", U), ("M", M), ("K", K), ("B", B), ("C", C)):
        _check_finite(name, tensor)

    if initial_states is not None:
        _check_finite("initial_states", initial_states)
    if B_prev is not None:
        _check_finite("B_prev", B_prev)
    if U_prev is not None:
        _check_finite("U_prev", U_prev)


def _resolve_heads_per_group(*, n_heads: int, bc_groups: int, name: str) -> int:
    if bc_groups <= 0:
        raise ValueError(f"{name} must have a positive group dimension.")
    if n_heads % bc_groups != 0:
        raise ValueError(
            f"{name} group dimension must divide heads. Got heads={n_heads}, groups={bc_groups}."
        )
    return n_heads // bc_groups


def _expand_grouped_heads(
    x: torch.Tensor,
    *,
    n_heads: int,
    name: str,
) -> torch.Tensor:
    heads_per_group = _resolve_heads_per_group(
        n_heads=n_heads,
        bc_groups=int(x.shape[1]),
        name=name,
    )
    if heads_per_group == 1:
        return x
    return x.repeat_interleave(heads_per_group, dim=1).contiguous()


def _validate_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    """Validates inputs for the canonical v2x2 scan API."""
    tensors = (U, M, K, B, C)
    if not all(torch.is_floating_point(t) for t in tensors):
        raise ValueError("U, M, K, B, and C must be floating-point tensors.")

    device = U.device
    if not all(t.device == device for t in tensors):
        raise ValueError(
            f"All inputs must be on the same device. Got {[t.device for t in tensors]}."
        )

    for name, tensor in (("U", U), ("M", M), ("K", K), ("B", B), ("C", C)):
        _check_contiguous(name, tensor)

    if U.ndim != 4:
        raise ValueError(f"U must be (batch,heads,T,P). Got {tuple(U.shape)}.")
    if M.ndim != 4 or M.shape[-1] != 2:
        raise ValueError(f"M must be (batch,heads,T,2). Got {tuple(M.shape)}.")
    if K.ndim != 5 or K.shape[-2:] != (2, 2):
        raise ValueError(f"K must be (batch,heads,T,2,2). Got {tuple(K.shape)}.")
    if B.ndim != 4:
        raise ValueError(f"B must be (batch,groups,T,2N). Got {tuple(B.shape)}.")
    if C.ndim != 4:
        raise ValueError(f"C must be (batch,groups,T,2N). Got {tuple(C.shape)}.")

    batch_size, n_heads, T, P = map(int, U.shape)
    if M.shape[:3] != (batch_size, n_heads, T):
        raise ValueError("Leading (batch,heads,T) dims of M must match U.")
    if K.shape[:3] != (batch_size, n_heads, T):
        raise ValueError("Leading (batch,heads,T) dims of K must match U.")
    if B.shape[0] != batch_size or B.shape[2] != T:
        raise ValueError("B must match U on batch/time dims.")
    if C.shape[0] != batch_size or C.shape[2] != T:
        raise ValueError("C must match U on batch/time dims.")

    D = int(B.shape[-1])
    if D % 2 != 0:
        raise ValueError(f"B/C 2N dim must be divisible by 2. Got {D}.")
    bc_groups = int(B.shape[1])
    _resolve_heads_per_group(n_heads=n_heads, bc_groups=bc_groups, name="B/C")
    if tuple(C.shape) != (batch_size, bc_groups, T, D):
        raise ValueError("C must match grouped B exactly.")
    N = D // 2

    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")

    if initial_states is not None:
        if initial_states.shape != (batch_size, n_heads, P, D):
            raise ValueError(
                "initial_states must be (batch,heads,P,2N) "
                f"={(batch_size, n_heads, P, D)}. Got {tuple(initial_states.shape)}."
            )
        _check_contiguous("initial_states", initial_states)
        if initial_states.device != device:
            raise ValueError("initial_states must be on the same device as inputs.")

    if B_prev is not None:
        if B_prev.shape != (batch_size, bc_groups, D):
            raise ValueError(
                f"B_prev must be (batch,groups,2N)={(batch_size, bc_groups, D)}. Got {tuple(B_prev.shape)}."
            )
        if U_prev is None or U_prev.shape != (batch_size, n_heads, P):
            raise ValueError(
                f"U_prev must be (batch,heads,P)={(batch_size, n_heads, P)}. "
                f"Got {None if U_prev is None else tuple(U_prev.shape)}."
            )
        _check_contiguous("B_prev", B_prev)
        _check_contiguous("U_prev", U_prev)
        if B_prev.device != device or U_prev.device != device:
            raise ValueError("B_prev and U_prev must be on the same device as inputs.")

    return batch_size, n_heads, T, N, P


def _validate_chunk_increment_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    T: int,
) -> tuple[int, int, int, int, int]:
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")

    for name, tensor in (("U", U), ("M", M), ("K", K), ("B", B)):
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be floating-point.")
        _check_contiguous(name, tensor)

    if U.ndim != 4:
        raise ValueError(f"U must be (batch,heads,T,P). Got {tuple(U.shape)}.")
    if M.ndim != 4 or M.shape[-1] != 2:
        raise ValueError(f"M must be (batch,heads,T,2). Got {tuple(M.shape)}.")
    if K.ndim != 5 or K.shape[-2:] != (2, 2):
        raise ValueError(f"K must be (batch,heads,T,2,2). Got {tuple(K.shape)}.")
    if B.ndim != 4:
        raise ValueError(f"B must be (batch,groups,T,2N). Got {tuple(B.shape)}.")
    if int(U.shape[2]) != int(T):
        raise ValueError(f"T={T} must match U.shape[2]={int(U.shape[2])}.")

    batch_size, n_heads, _, P = map(int, U.shape)
    if M.shape[:3] != (batch_size, n_heads, int(T)):
        raise ValueError("M leading dims must match U.")
    if K.shape[:3] != (batch_size, n_heads, int(T)):
        raise ValueError("K leading dims must match U.")
    if B.shape[0] != batch_size or B.shape[2] != int(T):
        raise ValueError("B must match U on batch/time dims.")

    D = int(B.shape[-1])
    if D % 2 != 0:
        raise ValueError(f"B 2N dim must be divisible by 2. Got {D}.")
    bc_groups = int(B.shape[1])
    _resolve_heads_per_group(n_heads=n_heads, bc_groups=bc_groups, name="B")
    N = D // 2

    if B_prev is not None:
        if tuple(B_prev.shape) != (batch_size, bc_groups, D):
            raise ValueError(
                f"B_prev must be (batch,groups,2N)={(batch_size, bc_groups, D)}. Got {tuple(B_prev.shape)}."
            )
        if U_prev is None or tuple(U_prev.shape) != (batch_size, n_heads, P):
            raise ValueError(
                f"U_prev must be (batch,heads,P)={(batch_size, n_heads, P)}. "
                f"Got {None if U_prev is None else tuple(U_prev.shape)}."
            )
        _check_contiguous("B_prev", B_prev)
        _check_contiguous("U_prev", U_prev)

    return batch_size, n_heads, int(T), N, P


def _validate_state_passing_inputs(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    initial_states: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if not torch.is_floating_point(inc) or not torch.is_floating_point(m_chunk):
        raise ValueError("inc and m_chunk must be floating-point tensors.")
    _check_contiguous("inc", inc)
    _check_contiguous("m_chunk", m_chunk)

    if inc.ndim != 5:
        raise ValueError(
            f"inc must be (batch,heads,chunks,P,2N). Got {tuple(inc.shape)}."
        )
    if m_chunk.ndim != 4 or m_chunk.shape[-1] != 2:
        raise ValueError(
            f"m_chunk must be (batch,heads,chunks,2). Got {tuple(m_chunk.shape)}."
        )

    batch_size, n_heads, n_chunks, P, D = map(int, inc.shape)
    if D % 2 != 0:
        raise ValueError(
            f"inc last dim must be divisible by 2 (flattened 2N). Got {D}."
        )
    if tuple(m_chunk.shape[:3]) != (batch_size, n_heads, n_chunks):
        raise ValueError("m_chunk leading dims must match inc.")
    N = D // 2

    if initial_states is not None:
        if initial_states.shape != (batch_size, n_heads, P, D):
            raise ValueError(
                "initial_states must be (batch,heads,P,2N) "
                f"={(batch_size, n_heads, P, D)}. Got {tuple(initial_states.shape)}."
            )
        _check_contiguous("initial_states", initial_states)

    return batch_size, n_heads, n_chunks, N, P


def _validate_chunk_scan_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    T: int,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    batch_size, n_heads, T_chk, N, P = _validate_inputs(
        U,
        M,
        K,
        B,
        C,
        initial_states=None,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    if T_chk != int(T):
        raise ValueError(f"T={T} must match U.shape[2]={T_chk}.")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    D = int(B.shape[-1])
    pad = (-int(T)) % int(chunk_size)
    n_chunks = (int(T) + pad) // int(chunk_size)
    if chunk_starts.shape != (batch_size, n_heads, n_chunks, P, D):
        raise ValueError(
            "chunk_starts must be (batch,heads,chunks,P,2N) "
            f"={(batch_size, n_heads, n_chunks, P, D)}. Got {tuple(chunk_starts.shape)}."
        )
    _check_contiguous("chunk_starts", chunk_starts)
    return batch_size, n_heads, int(T), N, P, n_chunks


def _as_complex_pairs(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"{name} must be float32/float64 after casting. Got {x.dtype}."
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"{name} last dim must be divisible by 2. Got {tuple(x.shape)}."
        )
    if not x.is_contiguous():
        x = x.contiguous()
    N = int(x.shape[-1]) // 2
    return torch.view_as_complex(x.reshape(*x.shape[:-1], N, 2))


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _to_complex_scalar(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.shape[-1] != 2:
        raise ValueError(f"{name} must have last dim 2. Got {tuple(x.shape)}.")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"{name} must be float32/float64 after casting. Got {x.dtype}."
        )
    if not x.is_contiguous():
        x = x.contiguous()
    return torch.view_as_complex(x)


def _to_complex_taps(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have trailing dims (2,2). Got {tuple(x.shape)}.")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"{name} must be float32/float64 after casting. Got {x.dtype}."
        )
    if not x.is_contiguous():
        x = x.contiguous()
    return torch.view_as_complex(x)


def _resolve_prev0(
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    *,
    batch_size: int,
    n_heads: int,
    D: int,
    P: int,
    device: torch.device,
    real_dtype: torch.dtype,
    complex_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if B_prev is None:
        b_prev0 = torch.zeros(
            (batch_size, n_heads, D // 2), device=device, dtype=complex_dtype
        )
        u_prev0 = torch.zeros((batch_size, n_heads, P), device=device, dtype=real_dtype)
        return b_prev0, u_prev0

    if U_prev is None:
        raise ValueError("U_prev must be provided when B_prev is provided.")

    b_prev0 = _as_complex_pairs(B_prev.to(dtype=real_dtype), name="B_prev").to(
        dtype=complex_dtype
    )
    u_prev0 = U_prev.to(dtype=real_dtype)
    return b_prev0, u_prev0


def _resolve_initial_state(
    initial_states: torch.Tensor | None,
    *,
    batch_size: int,
    n_heads: int,
    P: int,
    D: int,
    device: torch.device,
    real_dtype: torch.dtype,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    if initial_states is None:
        return torch.zeros(
            (batch_size, n_heads, P, D // 2), device=device, dtype=complex_dtype
        )
    return _as_complex_pairs(
        initial_states.to(dtype=real_dtype), name="initial_states"
    ).to(dtype=complex_dtype)


def _propagate_chunk_states(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagates chunk start states with direct products and no log-domain algebra."""
    batch_size, n_heads, n_chunks, P, N = map(int, inc.shape)
    if n_chunks == 0:
        empty = inc.new_empty((batch_size, n_heads, 0, P, N))
        return empty, initial_state

    starts: list[torch.Tensor] = []
    z = initial_state
    for c in range(n_chunks):
        starts.append(z)
        z = m_chunk[:, :, c].unsqueeze(-1).unsqueeze(-1) * z + inc[:, :, c]

    return torch.stack(starts, dim=2), z


def _normalize_unit_complex(z: torch.Tensor) -> torch.Tensor:
    mag = torch.abs(z)
    eps = torch.finfo(mag.dtype).tiny
    return z / mag.clamp_min(eps)


def _chunked_transition_prefix_parts(
    m_blk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns stable prefix parts for the chunked SLinOSS operating region.

    The fast chunked paths must never form a segment ratio as ``prefix_t *
    inv_prefix_s`` inside a dot product. We instead split each transition into a
    log-magnitude scan and a unit-phase scan, then form each ratio directly from
    their difference/product. Exact-zero transitions are rejected here because
    they are outside the model's intended operating region and make
    ``log(|m_t|)`` ill-defined. Callers are expected to satisfy the general
    finite-input contract before reaching this fast path.
    """
    mag = torch.abs(m_blk)
    if bool((mag == 0).any()):
        raise ValueError(
            "M must be strictly nonzero in chunked scan paths. Exact-zero "
            "transitions are outside the SLinOSS operating region; use v2x2ssm "
            "if you need arbitrary zero-transition behavior."
        )

    logprefix = torch.cumsum(torch.log(mag), dim=-1)
    phase = m_blk / mag
    phase_prefix = _normalize_unit_complex(torch.cumprod(phase, dim=-1))
    prefix = torch.exp(logprefix).to(dtype=m_blk.dtype) * phase_prefix
    return logprefix, phase_prefix, prefix


def _chunk_increment_core(
    m_blk: torch.Tensor,
    beta_prev_blk: torch.Tensor,
    beta_curr_blk: torch.Tensor,
    u_prev_blk: torch.Tensor,
    u_curr_blk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes per-chunk affine increments and net transitions by reverse scan."""
    batch_size, n_heads, n_chunks, L, N = map(int, beta_prev_blk.shape)
    P = int(u_curr_blk.shape[-1])
    device = m_blk.device
    dtype = m_blk.dtype

    inc = torch.zeros((batch_size, n_heads, n_chunks, P, N), device=device, dtype=dtype)
    m_suf = torch.ones((batch_size, n_heads, n_chunks), device=device, dtype=dtype)
    for t in range(L - 1, -1, -1):
        bprev_decay = m_suf.unsqueeze(-1) * beta_prev_blk[..., t, :]
        bcurr_decay = m_suf.unsqueeze(-1) * beta_curr_blk[..., t, :]
        inc = inc + u_prev_blk[..., t, :].unsqueeze(-1) * bprev_decay.unsqueeze(-2)
        inc = inc + u_curr_blk[..., t, :].unsqueeze(-1) * bcurr_decay.unsqueeze(-2)
        m_suf = m_suf * m_blk[..., t]

    return inc, m_suf


def _segment_scales(
    logprefix: torch.Tensor,
    phase_prefix: torch.Tensor,
    *,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Builds lower-triangular segment ratios from prefix parts.

    This helper exists only for the mathematical reference. The staged path uses
    the same ratio construction in a tiled loop to stay closer to the intended
    kernel structure.
    """
    L = int(logprefix.shape[-1])
    tril = torch.tril(torch.ones((L, L), device=logprefix.device, dtype=torch.bool))
    logdiff = (logprefix[..., :, None] - logprefix[..., None, :]).masked_fill(
        ~tril, 0.0
    )
    phase_ratio = phase_prefix[..., :, None] * torch.conj(phase_prefix[..., None, :])
    seg = phase_ratio * torch.exp(logdiff).to(dtype=complex_dtype)
    return seg.masked_fill(~tril, 0.0)


def _pad_time_full(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    real_dtype: torch.dtype,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
]:
    batch_size, n_heads, T, P = map(int, U.shape)
    bc_groups = int(B.shape[1])
    D = int(B.shape[-1])
    device = U.device
    pad = (-T) % int(chunk_size)

    U_f = U.to(dtype=real_dtype)
    M_f = M.to(dtype=real_dtype)
    K_f = K.to(dtype=real_dtype)
    B_f = B.to(dtype=real_dtype)
    C_f = C.to(dtype=real_dtype)

    if pad:
        U_f = torch.cat(
            [
                U_f,
                torch.zeros(
                    (batch_size, n_heads, pad, P), device=device, dtype=real_dtype
                ),
            ],
            dim=2,
        )
        M_pad = torch.zeros(
            (batch_size, n_heads, pad, 2), device=device, dtype=real_dtype
        )
        M_pad[..., 0] = 1.0
        M_f = torch.cat([M_f, M_pad], dim=2)
        K_f = torch.cat(
            [
                K_f,
                torch.zeros(
                    (batch_size, n_heads, pad, 2, 2), device=device, dtype=real_dtype
                ),
            ],
            dim=2,
        )
        B_f = torch.cat(
            [
                B_f,
                torch.zeros(
                    (batch_size, bc_groups, pad, D), device=device, dtype=real_dtype
                ),
            ],
            dim=2,
        )
        C_f = torch.cat(
            [
                C_f,
                torch.zeros(
                    (batch_size, bc_groups, pad, D), device=device, dtype=real_dtype
                ),
            ],
            dim=2,
        )

    T_pad = int(U_f.shape[2])
    n_chunks = T_pad // int(chunk_size)
    return U_f, M_f, K_f, B_f, C_f, T_pad, n_chunks


def _pad_time_partial(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    real_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    batch_size, n_heads, T, P = map(int, U.shape)
    bc_groups = int(B.shape[1])
    D = int(B.shape[-1])
    device = U.device
    pad = (-T) % int(chunk_size)

    U_f = U.to(dtype=real_dtype)
    M_f = M.to(dtype=real_dtype)
    K_f = K.to(dtype=real_dtype)
    B_f = B.to(dtype=real_dtype)

    if pad:
        U_f = torch.cat(
            [
                U_f,
                torch.zeros(
                    (batch_size, n_heads, pad, P), device=device, dtype=real_dtype
                ),
            ],
            dim=2,
        )
        M_pad = torch.zeros(
            (batch_size, n_heads, pad, 2), device=device, dtype=real_dtype
        )
        M_pad[..., 0] = 1.0
        M_f = torch.cat([M_f, M_pad], dim=2)
        K_f = torch.cat(
            [
                K_f,
                torch.zeros(
                    (batch_size, n_heads, pad, 2, 2), device=device, dtype=real_dtype
                ),
            ],
            dim=2,
        )
        B_f = torch.cat(
            [
                B_f,
                torch.zeros(
                    (batch_size, bc_groups, pad, D), device=device, dtype=real_dtype
                ),
            ],
            dim=2,
        )

    T_pad = int(U_f.shape[2])
    n_chunks = T_pad // int(chunk_size)
    return U_f, M_f, K_f, B_f, T_pad, n_chunks


def _resolve_empty_outputs(
    *,
    batch_size: int,
    n_heads: int,
    bc_groups: int,
    P: int,
    D: int,
    device: torch.device,
    output_dtype: torch.dtype,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Y = torch.empty((batch_size, n_heads, 0, P), device=device, dtype=output_dtype)
    if initial_states is None:
        final_state = torch.zeros(
            (batch_size, n_heads, P, D), device=device, dtype=output_dtype
        )
    else:
        final_state = initial_states.to(dtype=output_dtype).contiguous()

    if B_prev is None:
        B_last = torch.zeros(
            (batch_size, bc_groups, D), device=device, dtype=output_dtype
        )
        U_last = torch.zeros(
            (batch_size, n_heads, P), device=device, dtype=output_dtype
        )
    else:
        B_last = B_prev.to(dtype=output_dtype).contiguous()
        U_last = U_prev.to(dtype=output_dtype).contiguous()  # type: ignore[union-attr]

    return Y, final_state, B_last, U_last


def v2x2ssm(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential scan oracle."""
    batch_size, n_heads, T, N, P = _validate_inputs(
        U, M, K, B, C, initial_states, B_prev, U_prev
    )
    _check_reference_inputs_finite(U, M, K, B, C, initial_states, B_prev, U_prev)
    D = 2 * N

    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=U.dtype,
    )
    device = U.device
    cplx_dtype = _complex_dtype_from_real(rdtype)
    bc_groups = int(B.shape[1])

    if T == 0:
        return _resolve_empty_outputs(
            batch_size=batch_size,
            n_heads=n_heads,
            bc_groups=bc_groups,
            P=P,
            D=D,
            device=device,
            output_dtype=odtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
        )

    U_r = U.to(dtype=rdtype)
    M_r = M.to(dtype=rdtype)
    K_r = K.to(dtype=rdtype)
    B_r = _expand_grouped_heads(B.to(dtype=rdtype), n_heads=n_heads, name="B")
    C_r = _expand_grouped_heads(C.to(dtype=rdtype), n_heads=n_heads, name="C")
    B_prev_r = (
        None
        if B_prev is None
        else _expand_grouped_heads(
            B_prev.to(dtype=rdtype), n_heads=n_heads, name="B_prev"
        )
    )

    m = _to_complex_scalar(M_r, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_r, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]

    b_t = _as_complex_pairs(B_r, name="B").to(dtype=cplx_dtype)
    c_conj = torch.conj(_as_complex_pairs(C_r, name="C").to(dtype=cplx_dtype))

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev_r,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    b_prev_seq = torch.cat([b_prev0.unsqueeze(2), b_t[:, :, :-1]], dim=2)
    u_prev_seq = torch.cat([u_prev0.unsqueeze(2), U_r[:, :, :-1]], dim=2)

    beta_prev = k_prev.unsqueeze(-1) * b_prev_seq
    beta_curr = k_curr.unsqueeze(-1) * b_t

    z = _resolve_initial_state(
        initial_states,
        batch_size=batch_size,
        n_heads=n_heads,
        P=P,
        D=D,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    Y = torch.empty((batch_size, n_heads, T, P), device=device, dtype=rdtype)
    for t in range(T):
        drive = u_prev_seq[:, :, t].to(dtype=cplx_dtype).unsqueeze(-1) * beta_prev[
            :, :, t
        ].unsqueeze(-2) + U_r[:, :, t].to(dtype=cplx_dtype).unsqueeze(-1) * beta_curr[
            :, :, t
        ].unsqueeze(-2)
        z = m[:, :, t].unsqueeze(-1).unsqueeze(-1) * z + drive
        Y[:, :, t] = (c_conj[:, :, t].unsqueeze(-2) * z).sum(dim=-1).real

    return (
        Y.to(dtype=odtype).contiguous(),
        _pack_complex_pairs(z, real_dtype=odtype),
        B[:, :, -1, :].to(dtype=odtype).contiguous(),
        U[:, :, -1, :].to(dtype=odtype).contiguous(),
    )


def v2x2ssd_ref(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked mathematical reference."""
    batch_size, n_heads, T, N, P = _validate_inputs(
        U, M, K, B, C, initial_states, B_prev, U_prev
    )
    _check_reference_inputs_finite(U, M, K, B, C, initial_states, B_prev, U_prev)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")

    D = 2 * N
    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=U.dtype,
    )
    device = U.device
    cplx_dtype = _complex_dtype_from_real(rdtype)
    bc_groups = int(B.shape[1])

    if T == 0:
        return _resolve_empty_outputs(
            batch_size=batch_size,
            n_heads=n_heads,
            bc_groups=bc_groups,
            P=P,
            D=D,
            device=device,
            output_dtype=odtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
        )

    B_last = B[:, :, -1, :].to(dtype=odtype).contiguous()
    U_last = U[:, :, -1, :].to(dtype=odtype).contiguous()

    U_f, M_f, K_f, B_f, C_f, T_pad, n_chunks = _pad_time_full(
        U, M, K, B, C, chunk_size=chunk_size, real_dtype=rdtype
    )
    B_f = _expand_grouped_heads(B_f, n_heads=n_heads, name="B")
    C_f = _expand_grouped_heads(C_f, n_heads=n_heads, name="C")
    B_prev_f = (
        None
        if B_prev is None
        else _expand_grouped_heads(
            B_prev.to(dtype=rdtype), n_heads=n_heads, name="B_prev"
        )
    )
    L = int(chunk_size)

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_f, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]
    b_t = _as_complex_pairs(B_f, name="B").to(dtype=cplx_dtype)
    c_conj = torch.conj(_as_complex_pairs(C_f, name="C").to(dtype=cplx_dtype))

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev_f,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    b_prev_seq = torch.cat([b_prev0.unsqueeze(2), b_t[:, :, :-1]], dim=2)
    u_prev_seq = torch.cat([u_prev0.unsqueeze(2), U_f[:, :, :-1]], dim=2)
    beta_prev = k_prev.unsqueeze(-1) * b_prev_seq
    beta_curr = k_curr.unsqueeze(-1) * b_t

    z0 = _resolve_initial_state(
        initial_states,
        batch_size=batch_size,
        n_heads=n_heads,
        P=P,
        D=D,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    c_blk = c_conj.reshape(batch_size, n_heads, n_chunks, L, N)
    bprev_blk = beta_prev.reshape(batch_size, n_heads, n_chunks, L, N)
    bcurr_blk = beta_curr.reshape(batch_size, n_heads, n_chunks, L, N)
    u_prev_blk = u_prev_seq.reshape(batch_size, n_heads, n_chunks, L, P).to(
        dtype=cplx_dtype
    )
    u_curr_blk = U_f.reshape(batch_size, n_heads, n_chunks, L, P).to(dtype=cplx_dtype)

    inc, m_chunk = _chunk_increment_core(
        m_blk, bprev_blk, bcurr_blk, u_prev_blk, u_curr_blk
    )

    chunk_starts, z_final = _propagate_chunk_states(inc, m_chunk, initial_state=z0)
    logprefix, phase_prefix, prefix = _chunked_transition_prefix_parts(m_blk)
    seg = _segment_scales(logprefix, phase_prefix, complex_dtype=cplx_dtype)

    y_diag = torch.einsum(
        "bhctn,bhcsn,bhcts,bhcsp->bhctp",
        c_blk,
        bprev_blk,
        seg,
        u_prev_blk,
    ) + torch.einsum(
        "bhctn,bhcsn,bhcts,bhcsp->bhctp",
        c_blk,
        bcurr_blk,
        seg,
        u_curr_blk,
    )

    y_off = torch.einsum(
        "bhctn,bhcpn->bhctp", c_blk * prefix.unsqueeze(-1), chunk_starts
    )
    Y_pad = (y_diag + y_off).real.reshape(batch_size, n_heads, T_pad, P)

    return (
        Y_pad[:, :, :T].to(dtype=odtype).contiguous(),
        _pack_complex_pairs(z_final, real_dtype=odtype),
        B_last,
        U_last,
    )


def chunk_increment(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    T: int,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes per-chunk affine increments and chunk transitions.

    Returns:
      - ``inc``: ``(batch, heads, chunks, P, 2N)``
      - ``m_chunk``: ``(batch, heads, chunks, 2)``
    """
    batch_size, n_heads, _, N, P = _validate_chunk_increment_inputs(
        U, M, K, B, B_prev, U_prev, T
    )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")

    D = 2 * N
    rdtype, _ = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = U.device

    U_f, M_f, K_f, B_f, _, n_chunks = _pad_time_partial(
        U, M, K, B, chunk_size=chunk_size, real_dtype=rdtype
    )
    B_f = _expand_grouped_heads(B_f, n_heads=n_heads, name="B")
    B_prev_f = (
        None
        if B_prev is None
        else _expand_grouped_heads(
            B_prev.to(dtype=rdtype), n_heads=n_heads, name="B_prev"
        )
    )
    L = int(chunk_size)

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_f, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]
    b_t = _as_complex_pairs(B_f, name="B").to(dtype=cplx_dtype)

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev_f,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    b_prev_seq = torch.cat([b_prev0.unsqueeze(2), b_t[:, :, :-1]], dim=2)
    u_prev_seq = torch.cat([u_prev0.unsqueeze(2), U_f[:, :, :-1]], dim=2)
    beta_prev = k_prev.unsqueeze(-1) * b_prev_seq
    beta_curr = k_curr.unsqueeze(-1) * b_t

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    bprev_blk = beta_prev.reshape(batch_size, n_heads, n_chunks, L, N)
    bcurr_blk = beta_curr.reshape(batch_size, n_heads, n_chunks, L, N)
    u_prev_blk = u_prev_seq.reshape(batch_size, n_heads, n_chunks, L, P).to(
        dtype=cplx_dtype
    )
    u_curr_blk = U_f.reshape(batch_size, n_heads, n_chunks, L, P).to(dtype=cplx_dtype)

    inc, m_chunk = _chunk_increment_core(
        m_blk, bprev_blk, bcurr_blk, u_prev_blk, u_curr_blk
    )
    return (
        _pack_complex_pairs(inc, real_dtype=rdtype),
        torch.view_as_real(m_chunk).to(dtype=rdtype).contiguous(),
    )


def state_passing(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagates state across chunk boundaries."""
    batch_size, n_heads, n_chunks, N, P = _validate_state_passing_inputs(
        inc, m_chunk, initial_states
    )
    D = 2 * N

    rdtype, _ = _resolve_dtypes(
        input_dtypes=[inc.dtype, m_chunk.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = inc.device

    inc_c = _as_complex_pairs(inc.to(dtype=rdtype), name="inc").to(dtype=cplx_dtype)
    m_c = _to_complex_scalar(m_chunk.to(dtype=rdtype), name="m_chunk").to(
        dtype=cplx_dtype
    )
    z = _resolve_initial_state(
        initial_states,
        batch_size=batch_size,
        n_heads=n_heads,
        P=P,
        D=D,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    chunk_starts, z = _propagate_chunk_states(inc_c, m_c, initial_state=z)
    return (
        _pack_complex_pairs(chunk_starts, real_dtype=rdtype),
        _pack_complex_pairs(z, real_dtype=rdtype),
    )


def chunk_scan(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    T: int,
    chunk_size: int,
    output_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes within-chunk outputs from chunk start states."""
    batch_size, n_heads, _, N, P, n_chunks = _validate_chunk_scan_inputs(
        U, M, K, B, C, chunk_starts, B_prev, U_prev, T, chunk_size
    )
    D = 2 * N

    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype, chunk_starts.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=output_dtype,
    )
    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = U.device

    U_f, M_f, K_f, B_f, C_f, T_pad, _ = _pad_time_full(
        U, M, K, B, C, chunk_size=chunk_size, real_dtype=rdtype
    )
    B_f = _expand_grouped_heads(B_f, n_heads=n_heads, name="B")
    C_f = _expand_grouped_heads(C_f, n_heads=n_heads, name="C")
    B_prev_f = (
        None
        if B_prev is None
        else _expand_grouped_heads(
            B_prev.to(dtype=rdtype), n_heads=n_heads, name="B_prev"
        )
    )
    L = int(chunk_size)

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_f, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]
    b_t = _as_complex_pairs(B_f, name="B").to(dtype=cplx_dtype)
    c_conj = torch.conj(_as_complex_pairs(C_f, name="C").to(dtype=cplx_dtype))
    z0 = _as_complex_pairs(chunk_starts.to(dtype=rdtype), name="chunk_starts").to(
        dtype=cplx_dtype
    )

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev_f,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    b_prev_seq = torch.cat([b_prev0.unsqueeze(2), b_t[:, :, :-1]], dim=2)
    beta_prev = k_prev.unsqueeze(-1) * b_prev_seq
    beta_curr = k_curr.unsqueeze(-1) * b_t

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    u_curr_blk = U_f.reshape(batch_size, n_heads, n_chunks, L, P)
    u_prev_boundary = torch.empty(
        (batch_size, n_heads, n_chunks, P), device=device, dtype=rdtype
    )
    u_prev_boundary[:, :, 0] = u_prev0
    if n_chunks > 1:
        u_prev_boundary[:, :, 1:] = u_curr_blk[:, :, :-1, -1, :]
    c_blk = c_conj.reshape(batch_size, n_heads, n_chunks, L, N)
    bprev_blk = beta_prev.reshape(batch_size, n_heads, n_chunks, L, N)
    bcurr_blk = beta_curr.reshape(batch_size, n_heads, n_chunks, L, N)

    logprefix, phase_prefix, prefix = _chunked_transition_prefix_parts(m_blk)
    BHC = batch_size * n_heads * n_chunks
    c_flat = c_blk.reshape(BHC, L, N)
    bprev_flat = bprev_blk.reshape(BHC, L, N)
    bcurr_flat = bcurr_blk.reshape(BHC, L, N)
    logprefix_flat = logprefix.reshape(BHC, L)
    phase_prefix_flat = phase_prefix.reshape(BHC, L)
    u_curr_flat = u_curr_blk.reshape(BHC, L, P)
    u_boundary_flat = u_prev_boundary.reshape(BHC, P)
    z0_flat = z0.reshape(BHC, P, N)

    prefix_flat = prefix.reshape(BHC, L)
    q_off_flat = c_flat * prefix_flat.unsqueeze(-1)
    y_off_flat = torch.bmm(q_off_flat, z0_flat.transpose(1, 2)).real

    y_diag_flat = torch.zeros((BHC, L, P), device=device, dtype=rdtype)
    t_idx = torch.arange(L, device=device).unsqueeze(1)
    s_tile = 16
    lp_t = logprefix_flat.unsqueeze(-1)
    phase_t = phase_prefix_flat.unsqueeze(-1)

    for s0 in range(0, L, s_tile):
        s1 = min(L, s0 + s_tile)
        s_idx = torch.arange(s0, s1, device=device).unsqueeze(0)
        causal = s_idx <= t_idx
        causal_mask = causal.unsqueeze(0)

        lp_s = logprefix_flat[:, s0:s1].unsqueeze(1)
        phase_s = torch.conj(phase_prefix_flat[:, s0:s1]).unsqueeze(1)
        # Form the segment ratio directly. This is the v3-style fix for the
        # unstable ``prefix_t * inv_prefix_s`` factorization.
        scale = (
            torch.exp((lp_t - lp_s).masked_fill(~causal_mask, 0.0)).to(dtype=cplx_dtype)
            * phase_t
            * phase_s
        )

        kprev_tile = bprev_flat[:, s0:s1, :]
        scores_prev = torch.bmm(c_flat, kprev_tile.transpose(1, 2))
        scores_prev = (scores_prev * scale).real
        scores_prev = scores_prev.masked_fill(~causal_mask, 0.0)

        if s0 == 0:
            if s1 == 1:
                vprev_blk = u_boundary_flat.unsqueeze(1)
            else:
                vprev_blk = torch.cat(
                    [u_boundary_flat.unsqueeze(1), u_curr_flat[:, : s1 - 1, :]],
                    dim=1,
                )
        else:
            vprev_blk = u_curr_flat[:, s0 - 1 : s1 - 1, :]
        y_diag_flat = y_diag_flat + torch.bmm(scores_prev, vprev_blk)

        kcurr_tile = bcurr_flat[:, s0:s1, :]
        scores_curr = torch.bmm(c_flat, kcurr_tile.transpose(1, 2))
        scores_curr = (scores_curr * scale).real
        scores_curr = scores_curr.masked_fill(~causal_mask, 0.0)
        y_diag_flat = y_diag_flat + torch.bmm(scores_curr, u_curr_flat[:, s0:s1, :])

    Y_pad = (y_diag_flat + y_off_flat).reshape(batch_size, n_heads, T_pad, P)
    return Y_pad[:, :, :T].to(dtype=odtype).contiguous()


def v2x2ssd(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the scan via a 3-stage, kernel-shaped decomposition."""
    batch_size, n_heads, T, N, P = _validate_inputs(
        U, M, K, B, C, initial_states, B_prev, U_prev
    )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")

    D = 2 * N
    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=U.dtype,
    )
    device = U.device
    bc_groups = int(B.shape[1])

    if T == 0:
        return _resolve_empty_outputs(
            batch_size=batch_size,
            n_heads=n_heads,
            bc_groups=bc_groups,
            P=P,
            D=D,
            device=device,
            output_dtype=odtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
        )

    B_last = B[:, :, -1, :].to(dtype=odtype).contiguous()
    U_last = U[:, :, -1, :].to(dtype=odtype).contiguous()

    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=rdtype,
    )
    chunk_starts, final_state = state_passing(
        inc,
        m_chunk,
        initial_states=initial_states,
        compute_dtype=rdtype,
    )
    Y = chunk_scan(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        output_dtype=odtype,
        compute_dtype=rdtype,
    )

    return Y, final_state.to(dtype=odtype).contiguous(), B_last, U_last


__all__ = [
    "v2x2ssm",
    "v2x2ssd_ref",
    "chunk_increment",
    "state_passing",
    "chunk_scan",
    "v2x2ssd",
]
