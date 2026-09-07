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
