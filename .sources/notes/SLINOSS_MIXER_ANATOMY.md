# SLinOSS mixer anatomy and capability diagnosis

**Status:** current for the source committed with this note on 2026-09-06. This
file is part of the mixer contract: any change to `SLinOSSMixerConfig`,
`SLinOSSConfig`, `SLinOSSMixer`, `MixerState`, `scanprep`, `so3ssd`, or the mixer
tail must update the anatomy, initialization, counts, and diagnosis here in the
same change.

**Authoritative code checkpoint:**
`a8ea238741c44361da87bfb42b653f16f0e8f4f7`. This checkpoint preserves the
benchmarked mixer, recurrence, initialization, and default runtime layout choices.
Its only post-closure executable change makes the offline chunk-layout census
accept an explicit shared-memory carveout instead of silently querying whichever
GPU runs the report.

This note has three evidence labels:

- **Code fact** means directly established by the current implementation.
- **Experimental fact** means established by a named run or a paper ablation.
- **Diagnosis** means the smallest mechanistic inference consistent with those
  facts. It is not silently promoted to an experimental result.

The scope is the sequence mixer itself. The residual block, SwiGLU channel mixer,
embedding, final norm, and vocabulary head are described at the boundary but are
not included in the mixer parameter counts.

## 1. Symbols and legal shapes

| Symbol | Meaning | Constraint |
| --- | --- | --- |
| `B` | batch size | positive |
| `T` | sequence length | positive |
| `D` | residual width, `d_model` | positive |
| `E` | expanded mixer width, `round(expand * D)` | divisible by `P` |
| `P` | rows per head, `d_head` | positive multiple of 16 |
| `H` | number of heads, `E / P` | divisible by `G` |
| `G` | number of B/C groups, `n_groups` | positive divisor of `H` |
| `S` | state width, `d_state = 3N` | positive multiple of 48 |
| `N` | independent 3-vector lanes per row, `S / 3` | positive multiple of 16 |
| `W` | causal-convolution width, `d_conv` | positive |
| `L` | scan chunk size | power of two in `[16, 128]` |

Only B and C are grouped. Head `h` reads group
`floor(h / (H/G))`. U, the transition, first-order-hold taps, recurrent state,
skip, and norm remain head-specific. At fixed `E` and `S`, changing `P` changes
`H` but not the number of recurrent state scalars `HPS = ES`; changing `G`
increases B/C projection parameters and cache without changing `ES`.

The fused input projection computes

```text
A = align_up(4H, 16)
F = 2E + 2GS + 4H       useful projected features
Q = 2E + 2GS + A        aligned activation stride
```

and the following contiguous column bands:

```text
x [B,T,D]
  |
  `-- in_proj [F,D] -> aligned buffer [Q] -------------------------------.
       value [E] | gate [E] | B [GS] | C [GS] | transition [4H] | pad   |
           |           |          |        |             |               |
       causal       tail gate   optional B/C        scanprep             |
       conv+SiLU                causal conv                               |
           |                         |                                    |
       U [B,H,T,P]       B,C [B,G,T,S]     (w,ls), FOH taps              |
           `------------------------ so3ssd scan -------------------------'
                                      |
                              y [B,H,T,P]
                                      |
                       skip + head RMSNorm + SiLU gate
                                      |
                              tail [B,T,E]
                                      |
                              out_proj [D,E]
                                      |
                                output [B,T,D]
```

The alignment tail has `A - 4H` storage columns but no projection rows. The
GEMM writes its `F` useful outputs directly into a buffer whose token stride is
`Q`; kernels retain their sector-aligned pitch without dead parameters or
optimizer state.

## 2. Exact recurrent operator

For token `t` and head `h`, scanprep maps four raw scalars into a rotation vector
`w_{h,t}` and log-scale `ls_{h,t}`. Let `R(w)` be the corresponding SO(3)
rotation, repeated on all `N` three-dimensional lanes, and let
`a = exp(2 ls)`. For each recurrent row `p`, the mathematical reference is

```text
q_t = quat_exp(w_t)

z_t = a_t R(q_t) z_{t-1}
    + outer(u_{t-1}, Kprev_t b_{t-1})
    + outer(u_t,     Kcurr_t b_t)

y_t = <c_t, z_t>
```

with the full shapes

```text
Z_t       [B,H,P,S]
U_t       [B,H,P]
B_t, C_t  [B,G,S]
y_t       [B,H,P]
```

`Kprev_t` and `Kcurr_t` are not learned parameters. They are exact
first-order-hold moments of the same generator
`2 ls_t I + [w_t]_x`, evaluated in closed form. The two taps make the update a
three-term discretization using both the previous and current forcing operands.

The transition is a scaled rotation in each 3-vector lane. It is orthogonal up
to the scalar contraction and is noncommutative across tokens. It does **not**
mix different 3-vector lanes, and it is the same rotation for all `P*N` vectors
inside a head at a token.

The recurrence is affine/linear in `Z_{t-1}` conditional on the token-derived
operands. It is nonlinear as a sequence-to-sequence neural layer and its
transition products are noncommutative, but it is **not a nonlinear recurrence
in its carried state**. In particular, B/C and the transition depend on the
current token representation, not on `Z_{t-1}`. This distinction is central to
the capability diagnosis in section 10.

The parallel scan composes `(scale, quaternion, affine increment)` in an
associative semidirect product. Training therefore has `O(T)` work and
`O(log T)` parallel depth while token-by-token decode carries constant-size
state.

## 3. Token-dependent quantities

Every learned tensor is static between tokens; "token-dependent" below means the
effective operand produced from the token changes with `t`.

| Effective quantity | Shape per sequence | Token-dependent? | Source |
| --- | ---: | :---: | --- |
| raw value | `[B,T,E]` | yes | input-projection value band |
| `U` after depthwise conv and SiLU | `[B,H,T,P]` | yes, local | value band + `conv_weight/bias` |
| output gate | `[B,T,E]` | yes | input-projection gate band |
| `B` write direction | `[B,G,T,S]` | yes, optionally local | B band + optional key conv |
| `C` read direction | `[B,G,T,S]` | yes, optionally local | C band + optional key conv |
| transition displacement | `[B,H,T,4]` | yes | transition band |
| transition operating point | `[H,4]` | no | `transition_bias` |
| SO(3) transition and scale | `[B,H,T,4]` | yes | operating point + displacement + bounded maps |
| FOH taps | `[B,H,T,2,4]` packed | yes | deterministic functions of transition |
| recurrent state | `[B,H,P,S]` | history-dependent | scan output/carry; not a parameter |
| direct skip | `[H]` | no | `d_skip` |
| head RMS gain | `[H,P]` | no | `norm_weight` |

The initial state is a fixed, nonpersistent cyclic buffer rather than a learned
parameter. There is no learned tap tensor, step counter, or separate B/C/value
projection. One fused projection supplies all token operands.

## 4. Transition parameterization

The token row is added directly to the static head embedding:

```text
raw_w  = transition_bias[h,0:3] + token_band[b,t,h,0:3]
raw_ls = transition_bias[h,3]   + token_band[b,t,h,3]
```

The physical rotation vector is

```text
w = w_chart * raw_w / sqrt(1 + ||raw_w||^2 / 4)
```

where `w_chart = 3.141592502593994`, the largest float32 strictly below pi, is a
fixed operator constant rather than a constructor knob. The reachable radius is
strictly below `2*w_chart`, so every canonical SO(3) rotation, including a
half-turn, is an interior finite-parameter point. The derivative at the origin
is `w_chart`; there is no inner `tanh` or head-radius multiplier.

The physical log-scale is

```text
ls = -0.25 * sigmoid(raw_ls)
a  = exp(2 ls)
```

so every finite token has `a in (exp(-0.5), 1)`. A token cannot expand or
annihilate recurrent state. A horizon of `h` tokens is represented by
`ls = -0.5/h`, hence `a = exp(-1/h)`.

The rotation and decay coordinates are independently learnable. They share four
projection columns per head only as storage; the maps do not tie phase to
retention.

## 5. Learnable parameter inventory

The table describes one `SLinOSSMixer`. The default constructor has
`bias=False`, `conv_bias=True`, and `key_conv=True`.

| Parameter | Shape | Count | Effective role | Initialization | Precision / decay contract |
| --- | ---: | ---: | --- | --- | --- |
| `in_proj.weight` | `[F,D]` | `FD` | all token operands | PyTorch linear draw; B/C rows rescaled to row norm `1/sqrt(S)`; transition rows zero | module dtype; ordinary decay except harness policy |
| `in_proj.bias` | `[F]` | `F` if `bias` | same bands | PyTorch default; B/C and transition rows zero | module dtype; absent by default |
| `conv_weight` | `[E,W]` | `EW` | depthwise causal value convolution | uniform `[-1/sqrt(W), +1/sqrt(W)]` | module dtype |
| `conv_bias` | `[E]` | `E` if enabled | value-conv bias | zero | module dtype; enabled by default |
| `key_weight` | `[2GS,W]` | `2GSW` if enabled | independent depthwise causal conv on B and C channels | exact delta: all zero except current-token tap = 1 | module dtype; enabled by default |
| `transition_bias` | `[H,4]` | `4H` | static head operating points for rotation and decay | inverted period/horizon lattice | pinned fp32; marked no-weight-decay |
| `d_skip` | `[H]` | `H` | direct `U` gain | one | pinned fp32; marked no-weight-decay |
| `norm_weight` | `[H,P]` | `HP=E` | per-row RMSNorm gain | one | module dtype |
| `out_proj.weight` | `[D,E]` | `DE` | mixer output projection | live PyTorch linear draw; scaled by `1/sqrt(2*n_layers)` inside `SLinOSSBlock` | module dtype |
| `out_proj.bias` | `[D]` | `D` if `bias` | output bias | zero | module dtype; absent by default |

For a whitened unit-RMS input, the B/C row rescaling gives each projected
S-vector expected squared norm one. After the optional B/C convolution, every
realized vector is L2-normalized over S in fp32 (fp64 for the double oracle) and
stored back into its existing projection band. The scan therefore receives unit
B/C vectors without another activation-sized buffer. The normalization pullback
is applied before the B/C-convolution or fused-projection pullback.

The exact count is

```text
FD + [bias]F + EW + [conv_bias]E + [key_conv]2GSW
   + 4H + H + E + DE + [bias]D.
```

At the defaults this simplifies to

```text
FD + EW + 2E + 2GSW + 5H + DE.
```

No-weight-decay markings are declarations by the module. MAD, LM, and state
tracking honor them; state tracking additionally preserves upstream's exemption
for the token embedding. MQAR intentionally applies its published decay to every
parameter. The distinction is explicit in each harness rather than inferred from
the spelling of a mixer leaf.

### Concrete instantiated counts

| Configuration | `F` / `Q-F` | Mixer parameters | Core recurrent scalars `ES` | Comment |
| --- | ---: | ---: | ---: | --- |
| `D128 E256 P64 H4 G1 S144 W4` | `816 / 0` | `139,924` | `36,864` | current H4/G1 MAD/state geometry |
| `D128 E256 P32 H8 G1 S144 W4` | `832 / 0` | `141,992` | `36,864` | H8/G1 changes transport heads without buying B/C groups |
| `D128 E256 P32 H8 G4 S144 W4` | `1,696 / 0` | `256,040` | `36,864` | H8/G4 changes addressing capacity and parameters, not core state size |
| `D128 E256 P32 H8 G8 S144 W4` | `2,848 / 0` | `408,104` | `36,864` | one independent B/C system per head; core state remains fixed |
| `D320 E640 P64 H10 G1 S96 W4` | `1,512 / 8` | `693,298` | `61,440` | current small LM-probe geometry; aligned storage has no parameter rows |

For the first row the individual counts are: input projection 104,448; value
conv 1,024; conv bias 256; key conv 1,152; transition bias 16; skip 4;
norm 256; output projection 32,768.

## 6. Runtime state

The module stores a nonpersistent fp32 cyclic state `[H,P,S]`. Row `(h,p)` is a
unit vector at column `(h*P+p) mod S`. Whole-sequence execution expands that
buffer across the batch as `z0`; decode allocation and reset reproduce the same
state. It is deterministic, fixed, absent from checkpoints and parameter counts,
and casts only between fp32 and the fp64 oracle state.

Incremental inference allocates five mutable buffers per layer and batch item:

| Buffer | Shape | Dtype | Purpose |
| --- | ---: | --- | --- |
| `conv` | `[B,W-1,E]` | activation | value-conv history |
| `keys` | `[B,W-1,2GS]` | activation | B/C-conv history; absent when key conv is off |
| `ssm` | `[B,H,P,S]` | fp32 (fp64 oracle exception) | recurrent state |
| `b_prev` | `[B,G,S]` | activation | previous FOH write direction |
| `u_prev` | `[B,H,P]` | activation | previous FOH input |

Per batch item this is

```text
(W-1)E + [key_conv](W-1)2GS + HPS + GS + HP
```

scalars. It is 38,896 scalars at H4/G1 or H8/G1 and 41,920 at H8/G4 for the
D128 geometry above when key convolution is enabled. There is no step counter.

## 7. Initialization, exactly

### 7.1 Static transition bank

Rotation period and decay horizon use separate fixed physical bands:

```text
period  in [4, 256] tokens
horizon in [4, 4096] tokens
```

Neither sequence length nor a constructor reach/span knob is consulted. At four
or more heads, the two log grids form a nearly square boustrophedon lattice.
Two- and three-head models receive independent endpoint-covering grids rather
than collapsing one axis. A one-head model uses the geometric midpoint of each
band: horizon 128, period 32. Rotation axes use a deterministic spherical
Fibonacci set.

For H=4 the four `(horizon, period)` corners are approximately

```text
(4,4), (4096,4), (4096,256), (4,256).
```

The desired physical period and horizon are inverted through the bounded maps,
so the initialization specifies the operator itself, not merely raw parameter
values. The slowest decay mode retains `exp(-T/4096)` across `T` tokens: 93.94%
at `T=256` and 77.88% at `T=1024`.

### 7.2 Token paths and residual boundary

After ordinary framework resets, the constructor makes these deliberate edits:

- value and gate projection rows remain live random;
- B and C projection rows remain live but are balanced to expected unit vector
  norm on whitened input;
- realized B and C vectors are L2-normalized after their optional convolution;
- all four token transition-displacement rows are zero;
- alignment-pad rows are zero;
- the B/C convolution begins as the identity/delta;
- `d_skip` and `norm_weight` are one;
- recurrent state begins at the deterministic cyclic basis;
- `out_proj` begins live under its framework draw.

An `SLinOSSMixer` cannot know the depth of an external scaffold. It therefore
keeps the live output draw unchanged. `SLinOSSBlock`, which does know stack
depth, scales both its mixer and FFN output matrices by
`1/sqrt(2*n_layers)` and zeroes the FFN output bias. This places residual-branch
variance ownership at the scaffold boundary instead of encoding stack depth in
a standalone mixer configuration.

The initialized model has both an immediately observable transport carrier and
live token forcing. Its first backward reaches B, C, token-transition,
static-transition, convolution, tail, and output paths; there is no zero-output
gradient blackout.

### 7.3 Repairs made by the cleanup campaign

- Every tensor is initialized once; block and stack constructors no longer
  recursively reset children.
- Every legal head count receives both a meaningful phase and retention scale;
  low-head models no longer collapse to the fast decay endpoint.
- Rotation reach and initialization coverage are fixed operator invariants;
  `w_max` and `init_span` are no longer benchmark knobs.
- B and C enter the scan with unit runtime norm, so their magnitude no longer
  changes address, write strength and read scale simultaneously.
- The tail is `RMSNorm(scan + skip) * silu(gate)`, so gate magnitude survives.
- LM no longer trains the transition and skip parameters at an unexplained
  one-tenth learning rate. Static transition and skip parameters retain their
  explicit no-weight-decay declaration.

These repairs do not implement the innovation-correction recurrence in section
12. They remove avoidable cold starts and self-cancellation while retaining the
existing additive SO(3) scan.

## 8. Block and stack boundary

`SLinOSSBlock` adds two fp32 RMSNorm gains of shape `[D]` and a SwiGLU FFN with
three matrices. It applies

```text
pre-norm -> mixer -> residual add/pre-norm -> SwiGLU FFN
```

with the residual stream accumulated in fp32. `SLinOSSStack` optionally adds an
unpadded token embedding, final fp32 norm gain, and an untied padded vocabulary
head. Both residual branch output matrices use `1/sqrt(2*n_layers)` scaling;
there is no weight tying. None of those surrounding parameters is included in
section 5's mixer counts.

## 9. Historical differential diagnosis

The selective-copy winner was commit `109a56b` as a complete configuration; the
pre-cleanup state-tracking winner entered at `2e75c89`. The repaired source keeps
the state winner's reachable operator while removing its input-path cold start:

| Axis | MAD winner `109a56b` | Pre-cleanup state winner | Repaired source |
| --- | --- | --- | --- |
| rotation chart | radius below pi | full finite SO(3) reach | full finite SO(3) reach |
| token rotation drive | radius-anchored `tanh` | direct additive displacement | direct additive displacement |
| recurrent `z0` | zero | fixed cyclic basis | fixed cyclic basis |
| B/C projection | both live random | B zero, C live random | both live; balanced rows and unit runtime vectors |
| output projection | live random | zero | live; depth-scaled by native block |
| B/C short conv | absent | identity-initialized, learnable | identity-initialized, learnable |
| token transition band | zero | zero | zero |
| period / horizon | both 4..4096 | both 4..4096 | period 4..256; horizon 4..4096 |
| tail gate | before head RMS norm | before head RMS norm | after head RMS norm |

Historical point estimates reported for the complete `109a56b` configuration
were 94.71% and 92.28% selective copy on two seeds. They do not isolate any one
row of this table. Prior paired work did isolate a large z0 effect in one MAD
configuration: cyclic z0 + live B scored 57.41%, while zero z0 + the same live B
scored 90.82%. Conversely, transplanting the old anchored drive was neither
sufficient for selective copy (about 89.01%) nor compatible with A5 (about
7.15%). Those facts reject both simplistic stories, "tanh caused MAD" and "live
B alone caused MAD."

The defensible historical diagnosis is: the state winner moved toward
homogeneous group action (full chart, cyclic carrier, quiet token transition)
while moving away from input-driven memory (zero B and output). The repaired
source keeps the full chart and carrier required by A5 while making B, C and
output live and bounded. That is a source fact, not yet a transferred performance
claim; section 14 records only runs actually made on this source.

One historical branch also paired adjacent transition heads, labelled D00/D10 in
the experiment record. Each pair duplicated one initialized horizon, period, and
rotation axis. Its even head used the unrestricted coordinate
`raw = bias + token_band`; its odd head first used
`raw = bias + |bias|*token_band`, and a later variant bounded that relative term
with `tanh`. This was a bespoke optimization hedge, not B/C grouping and not a
standard ingredient inherited from the compared literature. It has no isolated
paying ablation: the radius-driven arm alone failed A5, paired arms solved A5, and
an unrestricted homogeneous chart also solves A5. Pairing additionally halves
the number of distinct initialized spectral points. It is absent from repaired
master, where every head has its own lattice point and uses the same unrestricted
coordinate.

Two older implementations sharpen the initialization diagnosis. Production
SLinOSS at `slinoss-old` commit `5eb1e26` defaulted to `G=H`, left its fused B/C
rows and output projection live under framework initialization, and normalized
each realized complex B/C vector to unit RMS (L2 norm `sqrt(S)`). Its recurrent
state was zero. The research SP2SSD mixer at `c2a9064` also used live Xavier B/C
and output projections, but initialized its actual correction precision to
`softplus(-8) ~= 0.000335`, effectively freezing the update. Commit `3a5e775`
replaced that dead corner with `phi_init=1` and live gentle modulation. Thus
projection liveness alone is insufficient: the operator-level write/correction
coefficient and the number of independent address systems must also be live.

## 10. Winning ingredients found in `.sources`

This section distinguishes task demands from architectural brands. Numbers are
used only where a source contains a causally informative ablation.

### 10.1 Finite-group and formal state tracking

The winners all make the token transition capable of representing the relevant
finite group:

- **PD-SSM** uses a token-selected column-one-hot/monomial transition
  `P(x)D(x)`. This family is closed under composition and can encode arbitrary
  finite-state transitions exactly. Its initial carrier is one-hot.
- **DeltaProduct / Gated DeltaNet `[-1,1]`** composes generalized Householder
  factors. Extending the coefficient to allow reflections/negative eigenvalues
  is load-bearing; products of reflections represent rotations and
  permutations.
- **Mamba-3** adds data-dependent complex rotations. In its reported state
  ablation, full Mamba-3 scores 100/98.51/87.75 on parity and two arithmetic
  tasks, while standard RoPE scores 1.56/20.70/2.62 and no RoPE
  2.27/1.49/0.72. The rotation must be input-dependent and compositional.
- **KLA** obtains history-dependent Möbius/precision transitions which compose
  through 2x2 matrices and solve A5 at constant depth.
- **Selective RoPE** independently shows that adding input-dependent rotation
  to a decaying GLA enables parity/state tracking while improving recall.

**Ingredient:** a token-selective transition family with group closure,
non-positive-real spectrum where required, and a reachable representation of
the target group. SLinOSS's full SO(3) chart and noncommutative quaternion scan
directly supply this for A5. The fixed cyclic carrier makes the homogeneous
action immediately observable; the controlled current-source ablation in
section 14 establishes that this carrier is load-bearing for the mixer.

### 10.2 Selective copy: GDN2's actual guts

GDN2's state is edited by

```text
S_t = (I - k_t (b_t * k_t)^T) Diag(alpha_t) S_{t-1}
    + k_t (w_t * v_t)^T.
```

Its relevant implementation facts are:

- q and k are L2-normalized in the recurrent kernel;
- q, k, and v have live short-convolutional projections;
- decay is channelwise and token-dependent;
- `b_t` is an independent channelwise **erase** gate on the key axis;
- `w_t` is an independent channelwise **write** gate on the value axis;
- both gates begin live: zero-centered logits give sigmoid values near 0.5;
- all linears, including output, begin live under Xavier-uniform gain
  `2^-2.5`;
- the released ablation says the erase gate supplies most of GDN2's gain.

The small Xavier gain is not directly transplantable as a B/C mechanism. GDN2
applies it to every linear and puts q/k through a nonlinear short-convolution path
before exact L2 normalization. Repaired SLinOSS instead has a linear,
identity-initialized B/C path before exact L2 normalization and initializes each
raw address to expected norm one. Multiplying only those rows by a scalar leaves
the forward address unchanged (away from epsilon) while multiplying the
normalization VJP by the reciprocal scalar. Thus `2^-2.5` on SLinOSS B/C alone
would chiefly be a roughly `2^2.5 = 5.66` times angular-gradient preconditioner,
not a smaller write or a GDN2-style operator repair.

The user's measured GDN2 selective-copy result is about 95%. Its mechanism is
not raw capacity. A normalized key identifies one address; the delta term erases
or corrects the state only in that addressed direction while preserving
unrelated associations; the independent value-axis write decides what to
commit. That is lossless selective editing under collisions.

Repaired SLinOSS L2-normalizes realized B/C vectors but retains an additive
outer-product write with no state-dependent erase/correction. Its homogeneous
transition globally rotates/contracts every lane in a head. It can append
information, transport it, and globally forget it; it cannot directly say
"replace the value at this content address while leaving the rest alone."

**Ingredient:** normalized content addressing plus a state-aware delta/erase
update, with live balanced read, write, and output paths.

### 10.3 Compression and fuzzy recall: KLA/KLA+

KLA maintains content together with a scalar/diagonal precision state. Process
noise makes the precision update a Möbius recurrence and feeds accumulated
confidence back into the mean's forget/write gate. Removing only process noise
changes the MAD scores as follows:

| Task | Full KLA | `p=0` | Delta |
| --- | ---: | ---: | ---: |
| Compression | 85.03 | 72.91 | -12.12 |
| Fuzzy Recall | 45.70 | 43.21 | -2.49 |
| Selective Copy | 90.67 | 75.73 | -14.94 |

Memorization and exact/noisy context recall stay approximately saturated. This
is unusually clean causal evidence: the hard MAD tasks benefit from a gate that
depends on accumulated history/confidence, not just on the present token.
Nonzero process noise also prevents precision from growing without bound. KLA+
improves compression to 88.87 and selective copy to 91.45 by propagating the
predictive distribution into probabilistic decoding.

**Ingredient:** state-dependent confidence/normalization that controls how much
new evidence overwrites accumulated memory. Compression particularly rewards a
well-conditioned sufficient statistic; fuzzy recall adds local token grouping
before that statistic is addressed.

### 10.4 Easy/saturated MAD tasks

The original MAD definitions explain why Memorization, Context Recall, and Noisy
Recall separate architectures less sharply:

- Memorization is a fixed dictionary and can live largely in model weights.
- Context Recall requires in-context key/value storage and lookup but not
  selective ordered overwriting under arbitrary blank runs.
- Noisy Recall uses a fixed noise vocabulary, so a feed-forward classifier can
  learn what to ignore before ordinary recall.

Many large-state or content-addressed mixers saturate these tasks. They establish
liveness, basic write/read, and noise filtering, but they do not identify the
missing selective-edit mechanism. The discriminating MAD axes are Compression,
Fuzzy Recall, and Selective Copy, exactly as the KLA process-noise ablation shows.

### 10.5 Language modelling: Mamba-3

Mamba-3 combines five relevant ingredients:

1. data-dependent complex rotation for state tracking;
2. stable token-dependent decay;
3. exponential-trapezoidal previous/current writes;
4. RMS-normalized B/C with learned head/channel biases initialized positive;
5. live input, read, write, and output paths, with optional low-rank MIMO.

Its 440M ablations are specific:

- no B/C bias and no trapezoid: 16.68 perplexity;
- trapezoid without B/C bias: 16.49;
- both: 15.72;
- adding the old short conv: 15.85;
- B/C bias initialized at zero: 16.57; initialized at trainable one: 15.72;
- neither B nor C bias: 16.52; B-only: 16.68; C-only: 15.98; both: 15.69.

The important lesson is balance and normalization, not "turn B on." An
asymmetric B-only intervention is empirically the worst bias configuration in
Mamba-3. Positive, normalized, paired B/C paths are synergistic. SLinOSS has the
mathematically stronger SO(3) transition and a first-order-hold previous/current
update. The pre-cleanup state winner initialized B dead, C live, and output
dead; the repaired source makes all three paths live and L2-normalizes both B
and C at runtime.

**Ingredient:** live, normalized, balanced read/write/output paths around stable
selective dynamics. MIMO rank is a secondary capacity/throughput choice, not the
first causal repair.

### 10.6 UEA and long time series: LinOSS, D-LinOSS, spectral SSMs

- LinOSS's strength comes from stable forced harmonic oscillators: a broad
  oscillatory basis transports long-range phase without exploding.
- D-LinOSS improves it by learning damping independently from frequency and
  initializing eigenvalues over the full angular range in a near-unit ring
  `[0.9,1.0]`.
- autocorrelation-based SSM analysis finds that a small fraction of exactly
  nondecaying modes improves long-memory benchmarks, but too many hurt; it also
  ties frequency separation to optimization conditioning.
- DFouT likewise emphasizes separate damping/frequency coverage, uniform
  discrete spectral coverage, and phase synchronization rather than a single
  timescale diagonal.

**Ingredient:** a well-covered, near-conservative oscillatory spectrum with
damping independent of frequency and a variance-controlled input forcing path.
SLinOSS already has the right stable transport family and a two-dimensional
period/horizon lattice. The repaired low-head allocation covers both physical
axes instead of collapsing to the fast endpoint. Its remaining risks are the
fixed 4096-token horizon across regimes and unconstrained forcing after
initialization—not lack of an oscillator.

### 10.7 MQAR and associative recall

DeltaNet, GDN2, and KLA all use normalized content addresses plus a correction
or confidence mechanism. KLA exceeds 95% at long-context MQAR in its reported
single-block setting. Group sharing matters here because one B/C group makes all
heads use the same address basis, but increasing G merely supplies more address
bases; it does not create selective correction.

**Ingredient:** normalized query/key geometry and collision-aware memory
updates. More groups can expose the mechanism at more addresses, but cannot
substitute for it.

## 11. Capacity and comparison fairness

The current D128/P64/G1/S144 mixer carries 36,864 recurrent scalars. In the KLA
paper's D128/state-8 geometry, the effective filter state is 2,048 scalars. The
original MAD protocol normalizes fixed-state architectures to total state 4,096.
Thus this SLinOSS geometry has 18x KLA's stated effective state and 9x the
original MAD target before counting decode convolution carries.

The current mixer alone has 139,924 parameters at that geometry. The local KLA
paper reconstruction's mixer has 87,681. Under the same local `kla-paper-v2`
scaffold with vocabulary 32, the complete models are approximately 279,444 and
227,201 parameters respectively. H8/G4 keeps the 36,864 core state scalars but
raises the SLinOSS mixer to 256,040 parameters.

Therefore:

- historical wins against published table values demonstrate useful capability;
- they do **not** establish iso-state or iso-parameter dominance;
- an H/G increase may improve performance by buying addressing parameters, but
  is not causal evidence for a better recurrence;
- paper-quality claims need both the in-tree reproducible profile and an
  explicitly matched state/parameter faceoff.

The in-tree profiles now label this honestly. `legacy-hybrid` is a historical
MAD-Lab/KLA-protocol hybrid, and `kla-paper-v2` is a locked textual reconstruction
that is not published-table eligible because the released KLA repository and
paper do not define one executable matching configuration.

## 12. Mechanistic verdict

### Established

SLinOSS is unusually strong at **transport and state algebra**:

- full finite SO(3) reach, including half-turns;
- noncommutative composition under an exact associative scan;
- stable nonexpansive dynamics;
- independent phase and decay coordinates;
- exact first-order-hold forcing;
- a fixed cyclic carrier plus live, input-driven write/read paths.

The cleanup repair removed the known initialization self-owns:

- B and C start live and variance-balanced rather than asymmetrically live/dead;
- B and C are unit runtime vectors rather than magnitude-coupled addresses;
- output starts live, with depth scaling owned by the native block;
- the cyclic state makes the homogeneous SO(3) action observable immediately;
- the output gate follows RMS normalization and therefore retains amplitude;
- decay horizon and rotation period use separate physical spectra.

It remains weakly parameterized for **state editing**:

- additive writes do not inspect the previous state;
- there is no targeted erase, innovation, or confidence feedback;
- G=1 shares one B/C address system across all heads;
- its generous raw state size can mask these deficits on easier tasks.

That is the real tension. It is not "SO(3) helps A5 but hurts MAD." Rotation and
decay are complementary and are present in other joint winners. The conflict is
between **blind additive writes** and the **normalized, state-aware edits**
required by selective memory and LM. The cleanup campaign fixes cold starts and
scale cancellation; it deliberately does not claim to have added the missing
editing operation.

### The simplest credible common denominator

The source evidence supports one coherent mixer, not task-specific init:

1. Keep the current full-reach SO(3) transition, stable decay, and independent
   period/horizon bank.
2. Keep the fixed cyclic carrier established by the controlled A5 comparison;
   normalized token forcing must coexist with it rather than replace it.
3. Replace blind additive writing with a normalized innovation/delta update. In
   orientation compatible with a `[P,S]` state, the minimal form is

   ```text
   Zbar_t = a_t R_t Z_{t-1}
   Bhat_t = normalize(B_t)
   Chat_t = normalize(C_t)
   old_t  = Zbar_t (erase_t * Bhat_t)
   Z_t    = Zbar_t - outer(old_t, Bhat_t)
                     + outer(write_t * U_t, Bhat_t)
   y_t    = Z_t Chat_t
   ```

   This preserves SO(3) transport while adding GDN2/DeltaNet's collision-aware,
   independently gated erase and write.
4. Keep B, C, value and output live and variance-balanced, with zero token
   transition deltas so the explicit SO(3) bank remains the initial transport.
5. Scale write magnitude from recurrence variance (for example by the
   contraction-dependent stationary-variance factor), not by benchmark name or
   an arbitrary A5/SC knob.

This is a general-purpose recurrence: the SO(3) homogeneous term supplies group
composition and oscillatory time-series transport; the innovation term supplies
selective memory, compression, MQAR, and LM. It is also conceptually simple
enough for an ICLR paper: **parallel noncommutative transport plus normalized
delta correction**.

There is an engineering consequence: multiplying a rotation by a rank-one
state-dependent correction no longer closes as the existing four-number
quaternion-plus-scale scan. Preserving parallel training requires a rotating-frame
chunk/WY factorization or an equivalent associative summary. This is a real
operator/kernel change, not an initialization-only patch, and should be scoped
honestly.

### If architectural differentiation remains necessary

The defensible differentiation is in constructor **shape/topology**, not task
conditionals or per-branch initialization:

- `d_head` and `n_groups` set the number and sharing of content-address systems;
- optional short B/C convolution sets local motif formation;
- optional MIMO rank trades parameters and decode arithmetic intensity.

The recurrence, maps, normalization, and initialization law should stay shared.
A fixed heads-per-group or parameters-per-state scaling rule is publishable;
"H8/G4 for this benchmark because it won" is not. UEA may reasonably use a
different state/head geometry from a language model just as model width differs,
but not a different hidden initialization story.

### What is not supported

- Restoring the old anchored `tanh` as the global fix.
- Claiming live B alone is sufficient.
- Treating H/G sweeps as causal diagnosis.
- Keeping zero B and zero output as a universal initialization for LM and then
  compensating with task-specific settings.
- Claiming published-table dominance from the current unmatched MAD geometry.

## 13. Evidence map

Primary local sources used for the diagnosis:

- Current implementation: `slinoss/config.py`, `slinoss/mixer.py`,
  `slinoss/state.py`, `slinoss/ops/scanprep/reference.py`,
  `slinoss/ops/so3ssd/reference.py`, `slinoss/ops/mixer/reference.py`.
- GDN2 implementation and stated ablations:
  `.sources/code/GatedDeltaNet-2/lit_gpt/gdn2.py` and its `README.md`.
- KLA equations, MAD table, and process-noise ablation:
  `.sources/papers/shaj2026kalman/gauss_icml2026.tex` and
  `.sources/notes/shaj2026kalman.md`.
- Mamba-3 recurrence, initialization, LM and state-tracking ablations:
  `.sources/notes/lahoti2026mamba3.md` and its mirrored code/paper.
- State-tracking mechanisms: `.sources/notes/terzic2025pdssm.md`,
  `.sources/notes/grazzi2025unlocking.md`, and
  `.sources/notes/siems2025deltaproduct.md`.
- Rotation/decay bridge: `.sources/notes/movahedi2026selective.md`.
- MAD task definitions and comparison rules:
  `.sources/papers/poli2024mechanistic/sections/3_mad.tex` and appendices.
- Oscillatory/time-series mechanisms: `.sources/notes/rusch2024linoss.md`,
  `.sources/notes/boyer2025dlinoss.md`,
  `.sources/notes/liu2025autocorrelation.md`, and
  `.sources/notes/solozabal2025uncovering.md`.

## 14. Performance-record boundary

The current source establishes what master constructs; it does not turn older
branch runs into current-tip measurements. The durable record presently supports:

- a historical paired free/stable H8/G4 configuration improved from 99.70% to
  100.00% and solved at 10k with a perfect held-out tail, but no matched
  paired-versus-homogeneous ablation attributes that result to pairing;
- exact current master, which has no parity pairing, reached 95.43%/91.65% at
  5k, 94.72%/85.75% at the 10k learning-rate peak, and recovered to 99.97%
  overall / 99.84% tail at 15k on cyclic H8/G4 with two layers
  (`solved=True`, loss 0.0015, 528,012 total model parameters);
- changing only that H8/G4 initial state to zero collapsed the matched 5k point
  to 4.56% overall and 3.24% tail, a controlled -90.87/-88.41 point effect;
- exact current master at cyclic H8/G8 with two layers reached 68.68%/45.41%
  at 5k, then 100.00% overall and 100.00% tail at 10k (`solved=True`, loss
  0.0001, 832,140 total model parameters);
- exact current master at cyclic H8/G8/P32 with one layer reached 99.9786%
  overall and 100.00% on the longest held-out tail at 20k (loss 0.00243,
  423,780 total / 408,104 mixer parameters);
- historical `109a56b` selective-copy point estimates of 94.71% and 92.28%;
- the causal z0 and anchored-drive results recorded in section 9;
- exact repaired-source MAD point estimates under `kla-paper-v2`, seed 12345:
  memorization 100.0000%, context recall 99.9986%, compression 83.7451%, fuzzy
  recall 41.5355%, and selective copy 72.7246%; noisy recall remains pending;
- a matched 100-step, 13,107,200-token LM probe at 2,048-token context:
  40,501,976-parameter SLinOSS reached validation loss 8.556816 at 42,741.9
  tokens/s, while 40,525,424-parameter Mamba3 reached 7.470294 at 65,381.6
  tokens/s. SLinOSS's logged pre-clip gradient norm exceeded one million.

Accordingly, the repaired source is a measurement-established **A5
state-tracking winner** at both H8/G4 and H8/G8. Its selective-copy result and
matched LM loss/throughput decisively reject joint-crusher status; the remaining
state and noisy-recall rows are still running.
That wording must remain until one exact current-tip configuration is evaluated
under the locked harnesses and fair state/parameter disclosures.
