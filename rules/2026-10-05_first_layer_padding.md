# First-layer padding test (decision rule written before the results)

English translation (12 October 2026) of the internal plan written on 5 October 2026,
before any run started. After the runs, the file changed only by the addition of the
"OUTCOME" section. Applied by `oct5/analyze_padding.py`.

## Question

Does the 200-epoch joint model's higher error on the drift member of the matched fog
pairs (and the 0.064 difference that remains after BN-adapt even inside the range) come
from the zero padding of the first 3x3 convolution? The two members are related by a
scalar affine map (x_d = A + c (x_D - A)); if the padding also transforms under this map
(reflect, circular), the first layer's output is, at every position, scaled by c plus a
constant. Zero padding (the data set's mean in normalized units) breaks this relation at
the border, and after normalization the break grows roughly as (1 - c)/c: large for the
drift member, small for the diffusion member.

## Design (inference only, no retraining)

Models: the saved 200-epoch joint models, CIFAR-10 and CIFAR-100, three seeds each
(fog-joint-full-s1..3, c100-fog-joint-full-s1..3). For each model, the two members of the
four matched pairs are corrupted once; the same corrupted images are used in all
conditions.

Padding of the first convolution (conv1 only; deeper layers unchanged):
- P0 zeros (as trained)
- P1 reflect
- P2 circular
- P3 constant padding with the haze value (A = 1, normalized)
Also:
- S per-image scalar standardization (in raw pixel space, then channel normalization).
  It makes the two members identically distributed by definition.

Each condition in two forms: frozen (stored BN statistics) and BN-adapt (statistics of
the test batch, batch 200). In each condition, the sampling floor: the error difference
of the two members on the L^-1 (fog_invert) input.

## Decision rules

Difference and positive pair, as in the earlier rule: D(s) = Err(drift member) -
Err(diffusion member); a pair is positive if D has the same sign in the three seeds,
|D(s)| > |D_inv(s)| in every seed, and |mean D| > SD(D).

Code checks (if they fail, the result is not interpreted):
- K1: P0 frozen differences within 0.04 of the stored differences
  (oct3/fog-joint-full-s*/output) in every pair and seed.
- K2: no positive pair in condition S (frozen and BN-adapt).

Test 1 (BN-adapt anomaly): if no pair is positive with BN-adapt under P1 and P2, the
difference that remains after BN-adapt is attributed to the first layer's zero padding.
If a positive pair remains, the padding explanation is incomplete.

Test 2 (main result, frozen model), separately for CIFAR-10 and CIFAR-100, over the three
pairs outside the training range (v = 0.089, 0.197, 0.490):
- (a) at least 2 positive pairs under both P1 and P2: the frozen difference does not
  depend on the first layer's zero padding; the main claim is kept and the test is
  reported in an appendix.
- (b) no positive pair under either P1 or P2: the frozen difference is a zero-padding
  artefact; the paper is reframed (mechanism = padding).
- (c) any other case: inconclusive; the decision is made by retraining (joint model with
  reflect padding and with input standardization, normalization ablation).

Caveat (written in advance): the model was trained with zero padding; changing the
padding at inference changes the border statistics of both members and may raise the
raw errors. Since the comparison is paired, the difference is still meaningful, but the
raw errors are reported too. P3 is additional information and does not enter the
decision.

## OUTCOME

**CIFAR-10 (5 October 21:59, three seeds, rule unchanged, with `oct5/analyze_padding.py`):**
- K1 passed (zero padding reproduces the stored differences within 0.04). K2 passed.
- Test 1 met: with zero padding the BN-adapt difference is positive in 4 of 4 pairs
  (0.066-0.106, 0.081 inside the range); with reflect, circular and haze padding no pair
  is positive (|difference| <= 0.004). The difference left after BN-adapt comes from the
  first layer's zero padding.
- Test 2 outcome (a): in the frozen model, two of the outer pairs (v = 0.197, 0.490) are
  positive with reflect (0.060, 0.066) and with circular (0.062, 0.065); with zero padding
  0.082, 0.080. At v = 0.089 the difference falls to 0.010-0.012 and is not positive
  (0.018 with zero padding, positive). The main claim is kept; about a quarter of the
  difference comes from zero padding.
- Haze padding (information): 2 of the outer pairs positive (0.060, 0.064).
- Per-image standardization removes the difference at inference by definition
  (|difference| <= 0.003 at every pair); it raises the clean error from 0.115 to 0.135.

**CIFAR-100 (5 October 22:37, three seeds, rule unchanged):**
- K1 passed, K2 passed.
- Test 1 met: with zero padding the BN-adapt difference is positive in 4 of 4 pairs
  (0.023-0.047); with reflect, circular and haze padding no pair is positive
  (|difference| <= 0.003).
- Test 2 outcome (a): in the frozen model all three outer pairs are positive with reflect
  (0.035, 0.063, 0.036) and with circular (0.036, 0.062, 0.036); with zero padding 0.044,
  0.070, 0.040.
- Haze padding (information): 3 of the outer pairs positive.
- Clean error is almost unchanged by the padding (zeros 0.384, reflect 0.384, circular
  0.383); on CIFAR-10 too (0.115, 0.114, 0.114). Changing the padding at inference does
  not break the model; the test is valid.

Note (design of K2): since standardization makes the L^-1 input identical to the raw
input, the floor equals the difference and K2 passes automatically; the informative part
is how small the raw differences are (CIFAR-10 <= 0.003, CIFAR-100 <= 0.004).

Summary: the difference left after BN-adapt comes entirely from the first layer's zero
padding. The difference in the frozen model persists independently of the padding; the
padding explains about a quarter of it on CIFAR-10 and a tenth to a fifth on CIFAR-100;
on CIFAR-10 the v = 0.089 pair is not positive without zero padding.

*[translation note]* The interpretation that part of the frozen difference comes from
zero padding was withdrawn after the joint model was retrained with reflect padding and
kept the full difference (`2026-10-05_retraining_and_normalization.md`, outcome V1; paper,
Appendix B). An earlier draft of the paper quoted the same share, computed from the
three-seed means of the two outer pairs, as about a fifth (21.5%).
