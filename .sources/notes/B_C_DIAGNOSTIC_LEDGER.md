# B/C diagnostic ledger

## 2026-09-07 — R10 prior-art fidelity check

**Base:** R10 transition tangent (`1/sqrt(D)`), current SO(3) operator and
deterministic cyclic `z0`; diagnostic source `67215c0`.

| Arm | Exact B/C contract | Result | Decision |
| --- | --- | --- | --- |
| rejected `f175391` | default fan-in projection + scalar-coordinate RMS; no affine or positive bias; B/C convolution retained | grad `1.39e5` at step 20 | Invalid prior-art attribution: matches neither source. |
| `r10-old-bc` | bias-free default fan-in projection; no B/C convolution; positive amplitude + spherical direction; RMS of `N` SO(3)-lane magnitudes | abort step 21; loss `7.571794`; grad `1211.687` | Faithful 3D lift, but incompatible with the current unscaled write taps. |
| `r10-mamba3-bc` | bias-free default fan-in projection; no B/C convolution; scalar RMSNorm with learned gain one; group broadcast; learned per-head B/C bias one | abort step 20; loss `7.285688`; grad `1111.806` | Faithful Mamba3 B/C frontend, but not stable under the current recurrence. |

Source facts:

- Old SLinOSS normalizes `N` complex-lane magnitudes, so the packed-real norm is
  `sqrt(N)`. The faithful SO(3) analogue also has norm `sqrt(N)`, not
  `sqrt(3N)`.
- Mamba3 normalizes `S` scalar coordinates, then adds learned all-ones B/C
  biases per head. Its initial norm is approximately `sqrt(2S)`.
- Current R10 uses L2 norm one. At `S=96=3N`, the observed mean norms were
  `1.000`, `5.657`, and `13.833` for R10, old-lift, and Mamba3 respectively.
- Current SO(3) initial FOH tap norms are `0.31–0.61`. Old SLinOSS's taps carry
  `dt in [0.005,0.1]`; Mamba3's previous/current writes also carry `dt` and a
  trapezoid gate. Their RMS-sized B/C and small write coefficients are coupled.
- The tail hid the internal mismatch: initial mixer-output RMS was `0.1834`
  (old lift) and `0.1815` (Mamba3 lift). By step 9, `transition_bias` gradient
  norm was already `90.075` and `25.128`, versus B/C projection norms
  `2.591/3.967` and `0.077/0.174`.

**Conclusion:** RMS normalization alone was never the transferable mechanism.
Do not transplant prior-art B/C magnitude without its write-scale convention.
No candidate from this node is eligible for master or A5.

Receipts:

- `../receipts/lm-prior-bc-67215c0/r10-old-bc.json` — SHA-256
  `88ab82d2e36b2cf05d68bbb1612f073394398bc2ff1dff334631513143bfa517`
- `../receipts/lm-prior-bc-67215c0/r10-mamba3-bc.json` — SHA-256
  `fa446518be1ce3816a7c840e467a32368e0842de2622ab15cf33491896441ba1`

## 2026-09-07 — LM harness integrity and transition causality

**Comparison contract:** seed 0, 12 layers, width 320, sequence length 2048,
13,107,200 training tokens, token batch 131,072, fp32 compute with a bf16 token
embedding, AdamW `(0.8, 0.95)`, peak hidden LR
`0.0030983866769659337`, and global clipping at 3.0. These are paired bounded
screens, not the full published Mamba-3 pretraining run.

### Harness findings

- The registered `mamba3` arm constructs `fla.layers.mamba3.Mamba3`, not
  `mamba_ssm.modules.mamba3.Mamba3`.
- The two installed classes have identical parameter names, shapes, and seeded
  initial tensors, but the FLA port maps `dd_A` through negative softplus. The
  official implementation uses its heavy-tail activation. Replacing only that
  method in the FLA class reproduces the official implementation's entire
  100-step trajectory, validation loss, maximum gradient, and final parameter
  metrics exactly. The existing `mamba3` row is therefore not an official
  Mamba-3 result.
- Corpus bytes, digests, vocabulary bounds, next-token shift, logical-vocabulary
  cross entropy, and held-out traversal were checked directly and by the 35 LM
  data/train/loss tests. No token or target misalignment was found.
- The KLA paper specifies a `0.1x` learning-rate multiplier for its SSM group.
  Commit `ffcdcb0` removed that multiplier from `GroupPolicy` and changed the
  test to require the full hidden LR. Current `master` therefore does not
  implement that published optimizer contract. In addition, SLinOSS's
  token-transition rows live inside the fused `in_proj.weight`, so the current
  whole-parameter grouping cannot assign those rows an SSM rate at all.
- The common LM scaffold omits the official Mamba stack's `1/sqrt(depth)`
  reinitialization of `out_proj.weight`. Applying that change alone did not
  explain the main result: official Mamba-3 worsened from `6.395432` to
  `6.616573` validation loss and remained gradient-stable in either case.

### Executable controls

| Arm | Status | Validation loss | Maximum pre-clip gradient | Clipped steps |
| --- | ---: | ---: | ---: | ---: |
| official Mamba-3 SISO | complete | 6.395432 | 10.9166 | 5/100 |
| registered FLA port, softplus `A` | complete | 7.406182 | 214.291 | 55/100 |
| FLA port, only `A` changed to official heavy-tail | complete | 6.395432 | 10.9166 | 5/100 |
| ordinary R10 | complete | 6.578479 | 677.891 | 94/100 |
| R10 + raw Mamba-3 B/C frontend | abort at step 20 | -- | 1111.806 | -- |
| R10 + unit-magnitude Mamba-3 B/C | abort at step 37 | -- | 1041.020 | -- |
| previous arm, B/C `+1` biases removed | abort at step 21 | -- | 1291.635 | -- |
| exact R10 with only B/C convolution removed | abort at step 25 | -- | 1703.410 | -- |
| unit-magnitude Mamba-3 B/C with convolution retained | abort at step 50 | -- | 1047.475 | -- |
| unit-magnitude Mamba-3 B/C, static transition frozen | complete | 6.309995 | 879.812 | 78/100 |
| unit-magnitude Mamba-3 B/C, token transition frozen | complete | 5.431634 | 5.2693 | 3/100 |

The official and heavy-tail-FLA rows have byte-identical step arrays, losses,
clipping metrics, and final parameter metrics (their throughput differs by
implementation). The two freeze rows differ only in which transition source can learn:
the per-head static operating point or the input-dependent transition projection.

### Localized cause

- Raw Mamba-3 B/C vectors are not portable by themselves. Mamba-3 multiplies
  writes by learned timestep/trapezoid factors: initial effective
  `gamma * C^T B` RMS is about `0.46--2.87` across layers. The raw transplant
  omits those factors and exposes `C^T K B` RMS about `45--47`, with write-only
  scan read RMS about `800--1036` before the mixer tail. This is one real scale
  error, but unit magnitude does not cure the later runaway.
- B/C bias removal is not a cure. It reduces the step-9 gradient from `34.87`
  to `4.36`, then the run still crosses `1291` at step 21.
- B/C convolution is a causal confound in the earlier transplant runs. Removing
  only that convolution from ordinary R10 leaves the initial function unchanged
  but changes a completing run into a step-25 abort. Retaining it delays the
  unit-B/C failure but does not eliminate it.
- In an ordinary R10 step-9 backward, heads initialized at horizon 4096 account
  for `99.055%` of static rotation-gradient energy and `98.857%` of token
  rotation-gradient energy. Decay-coordinate gradients are negligible, and all
  twenty largest headwise transition gradients belong to horizon-4096 heads.
- The decisive intervention is the transition-source cross-control. With the
  static transition frozen and token-dependent rotations live, the run still
  clips 78 times and reaches gradient `879.812`. With token-dependent rotations
  frozen and the static SO(3) transition fully learnable, maximum gradient is
  only `5.2693`, clipping falls to 3/100, and validation improves to `5.431634`.

**Verdict:** the runaway is not a generic Mamba-3 B/C failure and not a bad LM
target stream. It is a latent SLinOSS transition-conditioning failure. The
near-conservative 4096-horizon heads preserve phase sensitivity across most of
the 2048-token context; an input-dependent rotation is applied at every token,
and its shared projection accumulates those long-range phase derivatives. That
band monopolizes global clipping. Larger raw B/C writes and removal of the B/C
convolution aggravate it, but neither is the root: disabling only token-dependent
transition learning removes the runaway and substantially improves loss while
leaving the static full-reach SO(3) operator intact.

No production change follows automatically from this diagnostic. Before another
candidate can be called KLA-protocol LM, the harness must restore the declared
SSM learning-rate contract and make the fused token-transition rows separately
groupable; before another row can be called Mamba-3, it must construct the
official class or verify the FLA port's transition map against it.

Primary receipts (SHA-256):

- `../receipts/lm-integrity-2026-09-07/automation/official-mamba3-trace.json` —
  `578982d97af603a6a52a16506d1a3cdb3429c09b559403f5ddce48e93c1652de`
- `../receipts/lm-integrity-2026-09-07/sonata/fla-softplus-trace.json` —
  `a1b0a955e1520a57ecf04648fe5226b25a9e2c4a88d57ac7dde6bb951363a684`
- `../receipts/lm-integrity-2026-09-07/sonata/fla-heavy-tail.json` —
  `c782db3386f489d68acc0ec175a30b9801edd5fc3cd8438a2048c3e02e252c29`
- `../receipts/lm-integrity-2026-09-07/automation/r10-head-trace.json` —
  `a4b70eff851b4471062c74c5ea049fa9853afff622eee181fc9031d7e8a3d40a`
- `../receipts/lm-integrity-2026-09-07/automation/r10-unit-frozen-bias-automation.json` —
  `e0540ee68eb11f5508586b310eab3b591b8194aed724d123a64b8253cbc36656`
- `../receipts/lm-integrity-2026-09-07/automation/r10-unit-frozen-token-automation.json` —
  `f43933a83b958806e37573940c05f10968822495200b03563a736780549eccf1`
