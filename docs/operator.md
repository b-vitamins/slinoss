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

The taps are not free. First-order hold is the exact integral of the step against
a forcing that is linear between the two tokens, so with the per-step generator
`L = 2*ls*I + [w]_x`, whose exponential is the transition above,

    K_prev = int_0^1 r exp(L r) dr       = phi_1(L) - phi_2(L)
    K_curr = int_0^1 (1 - r) exp(L r) dr = phi_2(L)

where `phi_k(x) = sum_n x^n / (n+k)!`. `L` is a scalar plus a skew part, hence
normal, with eigenvalue `p = 2*ls` on `w` and `z = p + i*|w|` on the plane across
it. The chart above is exactly those eigenvalues: `kr = Re f(z)`,
`g = (f(p) - kr) / |w|^2`, `h = Im f(z) / |w|` for `f` either moment. So the
projection carries four columns per head and the taps carry none: no tap
parameter, no tap initialization, no trapezoidal or other approximation.

Both `phi` come from the recurrence `phi_{k+1} = (phi_k - 1/k!)/x` off
`phi_0 = exp(x)`, which costs one complex division per order and loses `eps/|x|`.
Inside `|z|^2 < 1` the series is summed instead, at 20 terms in float64 and 12 on
the device. The chart's own two divisions are by `|w|^2` and `|w|`, floored: `g`
and `h` are ill-conditioned as `|w|` falls while `g w w^T` and `h [w]_x` are not,
so the chart holds an absolute accuracy in that corner and the operator sees a
relative one.

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

## One-tap form

Reindexing the previous-tap column onto its own token collapses the two taps to
one. Both taps then read the same operand `b_{r-1}`, the same weight `u_{r-1}`,
and the same mask, so one table column carries both:

    Afuse_r    = Ap_r + exp(2*ls_r) * An_{r-1},   Afuse_0 = Ap_0
    Bfuse_r    = Afuse_r b_{r-1}
    diagonal:  Y_diag = (mask * (Cr @ Bfuse^T)) @ U_shift + dnow * U
    increment: inc_local = (U_shift * wgt)^T @ Bfuse + outer(U_{L-1}, Bn_{L-1})

with `dnow_t = <Cr_t, Bn_t>`. The mask is unchanged and the table width is
unchanged -- the column that held `Ap` holds `Afuse` -- and a chunk costs four
dense GEMMs instead of seven.

What the reindex leaves over is rank-one, not a GEMM: row `t = r-1` falls outside
the mask, and slot `L` falls outside the chunk. The diagonal leftover is
rotation-free, because `R(Q_t)` is orthogonal and the two frames cancel:
`<Cr_t, Bn_t>` equals `<c_t, K_curr(b_t)>`, the same dot product with the tap
applied in the token's own frame, measured to 6.9e-16 relative. It needs neither
the quaternion prefix nor the chunk-local frame and is orderable before the scan.

`Afuse_0` takes no `An` term. The previous chunk's `An_{L-1}` is in the previous
chunk's frame and its contribution already arrives through `zstart`, so injecting
it is wrong rather than redundant: `y` moves by 2.3e-01.

Padding is asymmetric here. Under two taps a zero-padded token is an exact no-op,
since `w = 0` and `ls = 0` give the identity transition and a zero tap kills the
forcing. That contract is retracted for this path: slot `n = T mod L` of a ragged
tail chunk carries the last real token's now-tap. `U` and `B` are zero past the
tail, `U_shift` and `B_shift` are built by shifting the padded sequence rather
than by padding the shift, and the pad token's table row is materialized rather
than predicated away. Padding the shift leaves `y` at roundoff and moves `state`
by O(1), because a pad column enters rows `t >= n` alone and the tail slice
discards those.

## Adjoint

The backward saves the chunk boundary and nothing else derived. Every other
chunk-local intermediate is rematerialized by the forward's own code, so the
recompute cannot drift from the forward, and the held boundary is the forward's own
output rather than a second expression of it. It is read-only in the backward, and
the gradients are bit-identical either way.

The pieces mirror the forward. The diagonal term transposes the decay mask. The
reverse chunk recurrence carries a `(B,H,P,3N)` accumulator and is the only
serial dimension in the backward, as state passing is in the forward.

Two-tap forcing makes the chunk boundary two-sided: the first token of a chunk
owes a `u_{t-1}` and a `b_{t-1}` cotangent to the last token of the chunk before
it. Those carries are one per chunk boundary, and at the sequence start they are
the operator's `du_prev` and `db_prev`.

The log-scale cotangent is a reverse cumulative sum over the chunk of a
per-token quantity assembled from the diagonal, increment, offset, and
chunk-transition terms. The split across those terms follows which stage holds
which operand, so no stage can derive its own half from another's.

## Numerical invariants

Guaranteed by parameterization, then asserted by tests. A clamp, an epsilon, or
a branch added to work around one of these is a sign the parameterization is
wrong, not the kernel.

1. `-LS_MAX_MAG <= ls <= 0`. The chunk-local log-scale prefix is monotone
   non-increasing and every decay factor lies in `(0,1]`. Overflow is unreachable;
   underflow is graceful and correct. Bounded below as well, which no kernel reads
   and no other invariant needs: the floor is a lifetime floor, so no token
   annihilates a row, and it is the decay's counterpart of
   `|w| <= 2*w_max < 2*pi`.
   Both bounds are two-sided and neither is reached.
2. `|w| <= 2*w_max < 2*pi`. The quaternion exponential is a single branchless
   polynomial accurate to float32 epsilon over the whole reachable domain. The
   chart reaches `|w| = w_max`, including the canonical SO(3) half turn at the
   default scale, at finite raw radius. Average active threads per warp stays at
   32.00.
3. A segment decay is never factored as `exp(2*lp_t) * exp(-2*lp_s)`. It is
   formed as `exp(2*(lp_t - lp_s))` from the log difference. Underflow times
   overflow is how NaN gets in. The one-tap form multiplies the mask by a second
   factor, so the now-tap's effective decay is `mask(t,r) * exp(2*ls_r)`: both lie
   in `(0,1]` by invariant 1 and neither comes from a reciprocal, so the product
   is a decay and the rule stands.
4. `trans`, `K`, the per-step quaternions, both chunk-local prefixes, and the 3x3
   table are float32 everywhere, including under autocast. Only `U`, `B`, `C`,
   `Y`, the score matrix, and GEMM operands are low precision. `z` is float32 in
   the recurrence that produces it and in the state the operator returns; the
   chunk-start copy the next GEMM reads is stored at the operand dtype, because
   that GEMM narrows it on the way into shared memory either way.
5. Quaternion prefix products are renormalized once per chunk after the scan.
   Rotation error enters the rotation matrix squared; unit-norm drift is not
   tolerated. The projection in the backward is the adjoint of this and is not
   optional either.
6. The score decay mask is applied to the float32 accumulator after the GEMM. It
   is indexed by both the row and the column, so folding it into an operand means
   splitting it as `exp(2*lp_t) * exp(-2*lp_r)`, which invariant 3 refuses: what
   this bounds is the factorization, not the operand dtype. A single-index factor
   is a different quantity. The one-tap form folds `exp(2*ls_r)` into the float32
   table column `Afuse` and thence into a bfloat16 operand, at no measurable cost:
   against the two-tap form the operand-rounding error ratio scatters 0.6x to 1.9x
   over ten shapes and five seeds, with no shape holding a sign.

## Tensor contracts

Time-major. A backend does not transpose or repack an input to suit a kernel; a
kernel that wants a different layout is rewritten.

Contiguous, except `B` and `C`. Those two are column bands of the mixer's fused
projection, so their token stride is the projection width and not `3N`. The
requirement on them is pitched instead: unit stride on the trailing axis,
non-overlapping rows, and a base address and pitch that both land on a boundary. A
contiguous buffer meets that rule at a pitch equal to its row width, so the two
layouts are one contract and a standalone caller needs no change. `3N` is a
multiple of 48, so the shape constraint already carries the alignment. The rule
holds for the cotangents too: `dB` and `dC` are written into the band the caller
names, at that band's pitch, never gathered into a contiguous buffer and copied
back.

The boundary is a 32-byte sector where the pitch exceeds the row width and 16
bytes where it does not. A band row starting mid-sector fetches one sector it
discards, and no bandwidth counter attributes it; a contiguous row shares its last
sector with the next row and wastes nothing. The producer pads to the stricter of
the two. Band order in the projection is value, gate, `B`, `C`, parameters, then
padding: the three activation widths are sector multiples already, so putting the
one free width last keeps every offset on a sector with no padding between bands.

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
float4 alignment, and its cotangent is exactly zero rather than within a
tolerance.

The trailing `3N` is `N` 3-vectors in lane-major order: element `3n+i` is
component `i` of 3-vector `n`.

The `z` row is the state the operator takes and returns. The `(B,H,C,P,3N)`
chunk-start state and its cotangent are not on that boundary and are not float32:
they carry the activation dtype, per invariant 4.

At one token the operator takes `z` and advances it in place instead of returning
it, and `b_prev` and `u_prev` with it. There is nothing to factor at that extent:
a single-token chunk has no prefix to form and no chunk-start copy, so the
activation-dtype `zstart` of invariant 4 does not exist and the float32 row is
read once and written once. `slinoss/ops/decode/` is that boundary and it is the
same map, asserted against both `T`-token implementations in float64. A caller
that copies any of the three carries out of it is reading and writing a buffer
onto itself, which is the traffic the in-place signature exists to delete.
`docs/decode.md` holds that stage's fusion boundary, its roofline account and its
drift table.

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
