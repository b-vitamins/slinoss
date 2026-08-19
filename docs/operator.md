# The SO(3) scan operator

## Per-step map

The state `z` carries `N` independent 3-vectors per row. One token contributes a
transition quaternion

    m_t = exp(ls_t) * q(w_t)

whose unit part is the quaternion exponential of the rotation vector `w_t` and
whose norm is `exp(ls_t)`. It acts on a 3-vector by conjugation:

    m v conj(m) = exp(2*ls) * R(q(w)) v

The rotation is exact SO(3) for any `w`, and the scale enters squared. Every
decay in the operator is therefore `exp(2*(...))`, and the log-scale prefix
appears doubled wherever a segment decay is formed.

One quaternion acts on all `N` lanes of a row identically. The lanes are
independent; the transition is shared. That is the whole reason the operator
reduces to dense GEMMs.

## Tap parameterization

Forcing is two-tap first-order hold: the input at a step enters through the
current and the previous token. Each tap acts on a 3-vector as

    K(v) = kr * v + g * (w . v) w + h * (w x v)

a polynomial in `w`, analytic at `w = 0`.

The axis normal form `k_par v_par + k_re v_perp + k_im (a x v)` with `a = w/|w|`
is the same map in different coordinates, related by `k_re = kr`,
`k_par = kr + g*|w|^2`, `k_im = h*|w|`. It is not used: it is singular at the
origin, costs an `rsqrt` and a clamp, and turns well-definedness into a
whole-tensor validity check. The polynomial chart makes that condition
structural.

## Chunk factorization

The sequence is cut into chunks of `L` tokens. Inside a chunk, two prefixes over
the tokens carry everything:

    lp_t  = sum_{s<=t} ls_s            log-scale prefix, scalar
    Q_t   = q(w_t) (*) ... (*) q(w_0)  quaternion prefix, unit

Both are recomputed by whichever kernel needs them and never reach global
memory. `Q` is renormalized once per chunk after the scan.

From the prefixes, three 3x3 matrices per token compose the rotation with each
tap: `An` for the current tap, `Ap` for the previous tap, and `Ac = R(Q_t)`
alone for the readout. One rowwise change of basis by these matrices puts every
vector in the chunk-local frame, after which each contraction is a dense real
GEMM.

The chunk output splits into a diagonal part and an offset part:

    diagonal: mask(t,r) = exp(2*(lp_t - lp_r)),  r <= t, applied to
              S = Cr @ Bn^T, then Y_diag = S @ U
    offset:   Y_off = exp(2*lp_t) * (Cr @ zstart^T)

The chunk increment and the inter-chunk recurrence are

    wgt_r      = exp(2*(lp_{L-1} - lp_r))
    inc_local  = (U * wgt)^T @ Bn  +  (U_shift * wgt)^T @ Bp
    zstart_c   = s_c
    s_{c+1}    = R(Q_{L-1}) (exp(2*lp_{L-1}) s_c + inc_local_c)

The state passing over chunks is the only serial dimension left.

## Numerical invariants

Guaranteed by parameterization, then asserted by tests. A clamp, an epsilon, or
a branch added to work around one of these is a sign the parameterization is
wrong, not the kernel.

1. `ls <= 0`. The chunk-local log-scale prefix is monotone non-increasing and
   every decay factor lies in `(0,1]`. Overflow is unreachable; underflow is
   graceful and correct.
2. `|w| <= w_max < pi`. The quaternion exponential is a single branchless
   minimax polynomial accurate to float32 epsilon over the whole reachable
   domain. Average active threads per warp stays at 32.00.
3. A segment decay is never factored as `exp(2*lp_t) * exp(-2*lp_s)`. It is
   formed as `exp(2*(lp_t - lp_s))` from the log difference. Underflow times
   overflow is how NaN gets in.
4. `trans`, `K`, the per-step quaternions, both chunk-local prefixes, the 3x3
   table, and `z` are float32 everywhere, including under autocast. Only `U`,
   `B`, `C`, `Y`, the score matrix, and GEMM operands are low precision.
5. Quaternion prefix products are renormalized once per chunk after the scan.
   Rotation error enters the rotation matrix squared; unit-norm drift is not
   tolerated. The projection in the backward is the adjoint of this and is not
   optional either.
6. The score decay mask is applied to the float32 accumulator after the GEMM,
   never folded into a bfloat16 operand.

## Tensor contracts

Time-major, contiguous. A backend does not transpose or repack an input to suit
a kernel; a kernel that wants a different layout is rewritten.

| tensor  | shape         | dtype           |
|---------|---------------|-----------------|
| `U`     | `(B,H,T,P)`   | bf16/fp16/fp32  |
| `trans` | `(B,H,T,4)`   | fp32            |
| `K`     | `(B,H,T,2,4)` | fp32            |
| `B`     | `(B,G,T,3N)`  | bf16/fp16/fp32  |
| `C`     | `(B,G,T,3N)`  | bf16/fp16/fp32  |
| `z`     | `(B,H,P,3N)`  | fp32            |
| `Y`     | `(B,H,T,P)`   | bf16/fp16/fp32  |

`trans` packs `(w_x, w_y, w_z, ls)`. `K` packs per tap `(kr, g, h, 0)` with tap
index `0` = previous and `1` = current; lane 3 is a hard zero, present for
float4 alignment.

The trailing `3N` is `N` 3-vectors in lane-major order: element `3n+i` is
component `i` of 3-vector `n`.

`B` and `C` are grouped: head `h` reads group `h // (H // G)`, so `G` divides
`H`, and `G == H` is the ungrouped case rather than a separate signature. The
broadcast from groups to heads has the cross-head `dB` reduction as its
pullback, so no kernel forms that reduction by hand.

`N` is a multiple of 16, so `3N` is a multiple of 48 and therefore of 16. Every
contraction is MMA-k friendly with no padding. The fix for a shape that does not
fit is the shape constraint, not a padding path. Every shape multiple lives in
`slinoss/config.py` with its reason.

The one exception: a GEMM's `M` mode is free, because `P` is a row count and not
a contraction extent. `M` is rounded up to `MMA_TILE_M` inside the shared tile,
the pad rows are zero-filled, and the store is predicated. Never a padded
tensor.
