# SLinOSS mixer anatomy and capability diagnosis

**Status:** current for `slinoss/` tree object
`e769a12bacd24fea0d9dfee707b6aded9f95ea24` (the mixer source at repository
commit `1b7b995526311309001f037c2c4175e3068d0888`) on 2026-09-05. This file is
part of the mixer contract: any change to
`SLinOSSConfig`, `SLinOSSMixer`, `MixerState`, `scanprep`, `so3ssd`, or the mixer
tail must update the anatomy, initialization, counts, and diagnosis here in the
same change.

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

The fused input projection has width

```text
A = align_up(4H, 16)
Q = 2E + 2GS + A
```

and the following contiguous column bands:

```text
x [B,T,D]
  |
  `-- in_proj [Q,D] -----------------------------------------------------.
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
                       skip + SiLU gate + head RMSNorm
                                      |
                              tail [B,T,E]
                                      |
                              out_proj [D,E]
                                      |
                                output [B,T,D]
```

The alignment tail has `A - 4H` projected columns. Those rows are stored and
count as parameters, but no semantic consumer reads them and their cotangent is
zero.

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
| transition operating point | `[H,4]` | no | `transition_embedding.weight` |
| SO(3) transition and scale | `[B,H,T,4]` | yes | operating point + displacement + bounded maps |
| FOH taps | `[B,H,T,2,4]` packed | yes | deterministic functions of transition |
| recurrent state | `[B,H,P,S]` | history-dependent | scan output/carry; not a parameter |
| direct skip | `[H]` | no | `d_skip` |
| head RMS gain | `[H,P]` | no | `norm_weight` |

There is no learned initial state, no learned tap tensor, no step counter, and no
separate B/C/value projection. One fused projection supplies all token operands.

## 4. Transition parameterization

The token row is added directly to the static head embedding:

```text
raw_w  = transition_embedding[h,0:3] + token_band[b,t,h,0:3]
raw_ls = transition_embedding[h,3]   + token_band[b,t,h,3]
```

The physical rotation vector is

```text
w = w_max * raw_w / sqrt(1 + ||raw_w||^2 / 4)
```

where the default `w_max = 3.141592502593994` is the largest float32 strictly
below pi. The reachable radius is strictly below `2*w_max`, so every canonical
SO(3) rotation, including a half-turn, is an interior finite-parameter point.
The derivative at the origin is `w_max`; there is no inner `tanh` or
head-radius multiplier.

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
| `in_proj.weight` | `[Q,D]` | `QD` | all token operands | PyTorch linear Kaiming-uniform, then B, transition, and pad rows zeroed | module dtype; ordinary decay except harness policy |
| `in_proj.bias` | `[Q]` | `Q` if `bias` | same bands | PyTorch default, then B, transition, pad zeroed | module dtype; absent by default |
| `conv_weight` | `[E,W]` | `EW` | depthwise causal value convolution | uniform `[-1/sqrt(W), +1/sqrt(W)]` | module dtype |
| `conv_bias` | `[E]` | `E` if enabled | value-conv bias | zero | module dtype; enabled by default |
| `key_weight` | `[2GS,W]` | `2GSW` if enabled | independent depthwise causal conv on B and C channels | exact delta: all zero except current-token tap = 1 | module dtype; enabled by default |
| `transition_embedding.weight` (`param_bias`) | `[H,4]` | `4H` | static head operating points for rotation and decay | inverted period/horizon lattice | pinned fp32; marked no-weight-decay |
| `d_skip` | `[H]` | `H` | direct `U` gain | one | pinned fp32; marked no-weight-decay |
| `norm_weight` | `[H,P]` | `HP=E` | per-row RMSNorm gain | one | module dtype |
| `out_proj.weight` | `[D,E]` | `DE` | mixer output projection | **zero** after consuming PyTorch reset | module dtype |
| `out_proj.bias` | `[D]` | `D` if `bias` | output bias | zero | module dtype; absent by default |

The exact count is

```text
QD + [bias]Q + EW + [conv_bias]E + [key_conv]2GSW
   + 4H + H + E + DE + [bias]D.
```

At the defaults this simplifies to

```text
QD + EW + 2E + 2GSW + 5H + DE.
```

No-weight-decay markings are declarations by the module, not a universal
optimizer law. The MAD and LM harnesses honor `_no_weight_decay`. The official
state-tracking-compatible grouping follows the upstream name rule: names
containing `embedding` get embedding decay and everything else gets ordinary
decay. Consequently the transition embedding is exempt there, while `d_skip`
is in the ordinary group despite its module flag. This is explicit harness
behavior, not silent fallback.

### Concrete instantiated counts

| Configuration | `Q` / pad | Mixer parameters | Core recurrent scalars `ES` | Comment |
| --- | ---: | ---: | ---: | --- |
| `D128 E256 P64 H4 G1 S144 W4` | `816 / 0` | `139,924` | `36,864` | current H4/G1 MAD/state geometry |
| `D128 E256 P32 H8 G4 S144 W4` | `1,696 / 0` | `256,040` | `36,864` | H8/G4 changes addressing capacity and parameters, not core state size |
| `D320 E640 P64 H10 G1 S96 W4` | `1,520 / 8` | `695,858` | `61,440` | current small LM-probe geometry; 2,560 pad-weight scalars |

For the first row the individual counts are: input projection 104,448; value
conv 1,024; conv bias 256; key conv 1,152; transition embedding 16; skip 4;
norm 256; output projection 32,768.

## 6. Fixed and runtime state

`initial_state` is a non-persistent fp32 buffer of shape `[H,P,S]`, not a
parameter. Every row contains one unit coordinate:

```text
initial_state[h,p,(h*P+p) mod S] = 1
```

This deterministic cyclic oscillator basis gives the homogeneous rotation a
nonzero carrier before any token write. It encodes no task label and draws no
randomness, but it is an architectural prior: output can be produced by rotating
and reading this carrier even when B is zero.

Incremental inference allocates five mutable buffers per layer and batch item:

| Buffer | Shape | Dtype | Purpose |
| --- | ---: | --- | --- |
| `conv` | `[B,W-1,E]` | activation | value-conv history |
| `keys` | `[B,W-1,2GS]` | activation | B/C-conv history; allocated even if key conv is off |
| `ssm` | `[B,H,P,S]` | fp32 (fp64 oracle exception) | recurrent state |
| `b_prev` | `[B,G,S]` | activation | previous FOH write direction |
| `u_prev` | `[B,H,P]` | activation | previous FOH input |

Per batch item this is

```text
(W-1)E + (W-1)2GS + HPS + GS + HP
```

scalars. It is 38,896 scalars at H4/G1 and 41,920 at H8/G4 for the D128
geometry above. The key cache is present to keep state shape independent of the
`key_conv` flag. There is no step counter.

## 7. Initialization, exactly

### 7.1 Static transition bank

The initialized fast endpoint is

```text
fast = 2*pi / (0.5*w_max) ~= 4 tokens
```

and the slow endpoint is the explicit constructor field `init_span`, 4096 by
default. Sequence length is not consulted. Horizons and periods are two
independent log grids over `[fast, init_span]`, laid out as a nearly square
boustrophedon lattice across heads. Rotation axes use a deterministic spherical
Fibonacci set.

For H=4 the four `(horizon, period)` corners are approximately

```text
(4,4), (4096,4), (4096,4096), (4,4096).
```

The desired physical period and horizon are inverted through the bounded maps,
so the initialization specifies the operator itself, not merely raw parameter
values. The slowest mode retains `exp(-T/4096)` across `T` tokens: 93.94% at
`T=256` and 77.88% at `T=1024`.

### 7.2 Token paths and residual boundary

After ordinary framework resets, the constructor makes these deliberate edits:

- value, gate, and C projection rows remain live random;
- B projection rows are zero;
- all four token transition-displacement rows are zero;
- alignment-pad rows are zero;
- the B/C convolution begins as the identity/delta;
- `d_skip` and `norm_weight` are one;
- the recurrent state is the cyclic basis;
- the entire output projection is zero.

Thus the state-writing term is initially zero, while an autonomous rotated
carrier and its C read are live internally. The mixer output is nevertheless
exactly zero because `out_proj` is zero, so a residual block begins as an exact
identity.

This has two optimization consequences that must not be euphemized:

1. On the first backward pass, `d(out_proj.weight)` can be nonzero, but the
   gradient into the mixer tail is multiplied by the zero output weight. Every
   earlier mixer parameter therefore receives zero gradient on optimizer step
   one. This is a one-step cold start.
2. The B parameter is trainable and can receive gradient after the output path
   becomes live, but the model begins with no token-dependent state writes. The
   transition can be trained against the nonzero cyclic carrier before B becomes
   useful. The initialization therefore strongly favors learning autonomous
   transport before input-driven memory.

These are code facts. Whether a one-step delay alone materially changes a long
run is an empirical question; the broader autonomous-versus-input-driven bias is
structural.

## 8. Block and stack boundary

`SLinOSSBlock` adds two fp32 RMSNorm gains of shape `[D]` and a SwiGLU FFN with
three matrices. It applies

```text
pre-norm -> mixer -> residual add/pre-norm -> SwiGLU FFN
```

with the residual stream accumulated in fp32. `SLinOSSStack` optionally adds an
unpadded token embedding, final fp32 norm gain, and an untied padded vocabulary
head. There is no depth scaling and no weight tying. None of those parameters is
included in section 5's mixer counts.

## 9. What changed between the historical MAD winner and current master

The selective-copy winner was commit `109a56b` as a complete configuration; the
state-tracking winner entered at `2e75c89` and is the basis of current master.
The difference is a bundle, not one `tanh` switch:

| Axis | Historical MAD winner `109a56b` | Current master | Mechanistic effect |
| --- | --- | --- | --- |
| rotation chart | radius asymptotes below pi | radius asymptotes below 2pi; half-turn finite | master removes a real A5/group reachability defect |
| token rotation drive | `bias + ||bias||*tanh(delta)` | `bias + delta`, then one outer radial map | old drive preserves each head's initialized scale but suppresses slow-head gradients/reach; master is globally reachable |
| recurrent `z0` | zero | fixed cyclic nonzero basis | changes input-driven memory into a live autonomous carrier |
| B/write projection | live framework random | zero | master starts with no input-driven state update |
| output projection | live framework random | zero | master starts as residual no-op and blocks all inner gradients on step one |
| B/C short conv | absent | present but exact identity initially | no forward difference at initialization; adds learnable local motifs |
| token transition band | zero | zero | same static transition bank at initialization |
| period/horizon endpoint | hard-coded 4096 | explicit constructor `init_span=4096` | same default physics; master prevents a harness length from changing it silently |

Historical point estimates reported for the complete `109a56b` configuration
were 94.71% and 92.28% selective copy on two seeds. They do not isolate any one
row of this table. Prior paired work did isolate a large z0 effect in one MAD
configuration: cyclic z0 + live B scored 57.41%, while zero z0 + the same live B
scored 90.82%. Conversely, transplanting the old anchored drive was neither
sufficient for selective copy (about 89.01%) nor compatible with A5 (about
7.15%). Those facts reject both simplistic stories, "tanh caused MAD" and "live
B alone caused MAD."

The defensible differential diagnosis is: the state winner intentionally moved
the model toward homogeneous group action (full chart, nonzero carrier, quiet
token transition), while simultaneously moving initialization away from
input-driven memory (zero B and zero output). That bundle favors state algebra
and creates an avoidable conflict with write-heavy tasks. The chart itself is
resolved: retaining full finite SO(3) reach is mandatory. The unresolved design
question is the correct state-editing rule and balanced initialization, not chart
reach or head-count roulette.

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
directly supply this for A5. The cyclic carrier makes the homogeneous action
immediately observable, but a nonzero carrier could also be written from input;
the theory does not require that it be a fixed constructor buffer.

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

The user's measured GDN2 selective-copy result is about 95%. Its mechanism is
not raw capacity. A normalized key identifies one address; the delta term erases
or corrects the state only in that addressed direction while preserving
unrelated associations; the independent value-axis write decides what to
commit. That is lossless selective editing under collisions.

Current SLinOSS has an additive outer-product write and a token read, but no key
normalization and no state-dependent erase/correction. Its homogeneous
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
Mamba-3. Positive, normalized, paired B/C paths are synergistic. SLinOSS already
has the mathematically stronger SO(3) transition and a first-order-hold
previous/current update, but master initializes B dead, C live, and output dead.
That is the opposite of the source-backed LM regime.

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
period/horizon lattice. Its remaining risks are coarse allocation at low H,
fixed 4096 scaling across regimes, and unnormalized forcing—not lack of an
oscillator.

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

SLinOSS master is unusually strong at **transport and state algebra**:

- full finite SO(3) reach, including half-turns;
- noncommutative composition under an exact associative scan;
- stable nonexpansive dynamics;
- independent phase and decay coordinates;
- exact first-order-hold forcing;
- a live autonomous carrier at initialization.

It is weakly parameterized and badly initialized for **state editing**:

- additive writes do not inspect the previous state;
- B/C are not normalized content addresses;
- there is no targeted erase, innovation, or confidence feedback;
- master begins with B dead, C live, output dead, and z0 live;
- G=1 shares one B/C address system across all heads;
- its generous raw state size can mask these deficits on easier tasks.

That is the real tension. It is not "SO(3) helps A5 but hurts MAD." Rotation and
decay are complementary and are present in other joint winners. The conflict is
between an **autonomous-carrier initialization plus blind additive writes** and
the **live, normalized, state-aware edits** required by selective memory and LM.

### The simplest credible common denominator

The source evidence supports one coherent mixer, not task-specific init:

1. Keep the current full-reach SO(3) transition, stable decay, and independent
   period/horizon bank.
2. Seed state from a live, normalized token write rather than relying on a fixed
   task-independent carrier. A zero runtime state is then compatible with
   memory tasks and does not remove group expressivity after the first symbol.
3. Replace blind additive writing with a normalized innovation/delta update. In
   orientation compatible with a `[P,S]` state, the minimal form is

   ```text
   Zbar_t = a_t R_t Z_{t-1}
   k_t    = normalize(B_t)
   old_t  = Zbar_t k_t
   Z_t    = Zbar_t + beta_t outer(v_t - old_t, k_t)
   y_t    = Z_t normalize(C_t)
   ```

   This preserves SO(3) transport while adding GDN2/DeltaNet's missing
   collision-aware correction. A channelwise erase/write generalization is
   available if the scalar beta proves insufficient, but it is not the first
   form to ship.
4. Initialize B, C, value, and output live and variance-balanced. Normalize B/C
   and initialize any B/C offsets as a paired positive scheme, not B-only. Keep
   token transition deltas zero so the explicit SO(3) bank remains the initial
   transport.
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

- the state-tracking winner's paired free/stable H8/G4 result improved from
  99.70% to 100.00% and solved at 10k with a perfect held-out tail;
- historical `109a56b` selective-copy point estimates of 94.71% and 92.28%;
- the causal z0 and anchored-drive results recorded in section 9;
- no clean, complete, matched all-six-MAD plus LM result table at the current
  master tip yet.

Accordingly, current master is a source-established **state-tracking design
winner**, not yet a measurement-established joint MAD/LM crusher. That wording
must remain until one exact current-tip configuration is evaluated under the
locked harnesses and fair state/parameter disclosures.
