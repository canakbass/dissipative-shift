# After 10 October: the retrained joint model and the normalization ablation

English translation (12 October 2026) of the internal plan written on 5 October 2026,
before any run started and before any code was written. After the runs, dated notes made
before the corresponding results and the "OUTCOME" section were added. Applied by
`oct10/analyze_variants.py`.

## Why

The padding test (`2026-10-05_first_layer_padding.md`) was done at inference: the padding
of the first layer of a model trained with zero padding was changed at test time. The
frozen difference persisted at the outer pairs (CIFAR-10 2/3, CIFAR-100 3/3); after
BN-adapt the difference disappeared completely. The decisive version is to train a model
whose padding transforms with the affine map from the start (reflect). The same runs
also cover the planned normalization ablation: is the difference specific to batch
normalization?

## Design

Joint model, with the same grid, data, budget and augmentation as fog-joint-full
(200 epochs, per-sample crop and flip), CIFAR-10, three seeds (1, 2, 3). Only the
architecture changes:

- V0: BN, first conv zero padding (the existing fog-joint-full results; not rerun)
- V1: BN, first conv reflect padding (other layers unchanged)
- V2: GroupNorm (32 groups) in place of every BN, zero padding
- V3: LayerNorm (GroupNorm, 1 group) in place of every BN, zero padding
- V4: BN, zero padding, per-image scalar standardization of the input (training and test)

Measurements for each model: frozen error on both members of the four matched pairs;
BN-adapt error for the models with BN (V1, V4); the L^-1 floor for both; clean test
error. No TENT in this round.

Quota: an estimated 3-4 GPU hours per model; V1-V4 x 3 seeds = 12 runs, ~45 GPU hours.
If the quota is not enough, the order is V1, then V2, V3, V4.

## Decision rules

Positive pair, as in the earlier rule: D(s) = Err(drift member) - Err(diffusion member);
same sign in the three seeds, |D(s)| > |D_inv(s)| in every seed, |mean D| > SD(D). Outer
pairs: v = 0.089, 0.197, 0.490.

Code checks (if they fail, no interpretation):
- K1: in V4 the two members have the same distribution, so |mean D| exceeds 0.01 at no
  pair.
- K2: in V1 no pair is positive with BN-adapt (the theory requires this: the first layer
  transforms with the affine map, and the batch statistics remove the scale and the
  offset).
- K3: every variant's clean error is below 0.20 (shows that training worked).

Test A (padding, decisive version), V1 frozen:
- (a) at least 2 of the outer pairs positive: the frozen difference is not a zero-padding
  artefact; the paper's main number moves to V1's difference, with V0 beside it.
- (b) none positive: the difference comes from training with zero padding; the paper is
  reframed (mechanism = padding) and the result of the inference test is replaced by this
  decisive version.
- (c) otherwise: inconclusive; both results are reported, the main number does not
  change, and it is stated plainly that the inference and the training results disagree.

Test B (is it specific to BN), V2 and V3 frozen:
- At least 2 of the outer pairs positive in both: the difference is not specific to
  batch normalization.
- None positive in either: the difference is specific to the BN architecture; this
  narrows the scope of the main claim and is written up as such.
- Otherwise: mixed; reported per variant.

V4 does not enter the decision: it removes the difference by definition; only its cost
in clean error is reported.

Reporting: all variants, all pairs, whatever the outcome.

## Analysis script (6 October 17:30, before the V1 results)

The rules were written, unchanged, into oct10/analyze_variants.py; it was tried on V0
(it reproduces the known numbers). Kernel names: V1 fog-joint-full-reflect, V2
fog-joint-full-gn32, V3 fog-joint-full-ln, V4 fog-joint-full-std. Since the V1 kernel
runs runner.py's fog_joint_full job, TENT is computed too; it does not enter the
decision.

## Implementation (11 October, before V2-V4 were run; rules unchanged)

Three jobs in runner.py: fog_joint_full_gn32 (GroupNorm(32) in place of every BatchNorm,
shortcuts included), fog_joint_full_ln (GroupNorm(1)), fog_joint_full_std (BatchNorm;
training, validation and test inputs standardized per image with standardize_images,
the L^-1 arm too). Budget, grid and augmentation as in fog_joint_full (FULL_BUDGET). In
the models without BN (V2, V3) BN-adapt is not run (null); TENT is run in none of them
(plan: no TENT in this round). Smoke test (CPU, seconds): 20 normalization layers in
every variant, the BN variant's parameter names unchanged, the earlier jobs' FULL_BUDGET
unchanged, standardized x_t and standardized L^-1 x_t differ by 1e-7.

## OUTCOME

### V1 (6 October 22:25; fog-joint-full-reflect-s1..s3, oct10/analyze_variants.py, rule unchanged)

The kernels' JOB is fog_joint_full_reflect, 200 epochs, first conv padding_mode=reflect
(checked in the kernel code). V2-V4 not yet run.

- K2 (V1, no pair positive with BN-adapt): PASSED. BN-adapt differences between -0.0047
  and +0.0047.
- K3 (clean error < 0.20): PASSED, 0.1145 (V0: 0.1147).
- Test A (V1 frozen): 2 of the outer pairs positive -> OUTCOME (a). The frozen difference
  is not a zero-padding artefact; the paper's main number moves to V1's difference, with
  V0 beside it.
  - v=0.089: +0.0188 [+0.0475 +0.0014 +0.0075], floor 0.0110 -> not positive (positive in
    V0, 0.0154)
  - v=0.197: +0.0884 [+0.1414 +0.0712 +0.0527], floor 0.0093 -> positive (V0: 0.0785)
  - v=0.490: +0.0715 [+0.0655 +0.0746 +0.0745], floor 0.0084 -> positive (V0: 0.0812)
  - v=0.042 (inside the range): +0.0000
- TENT (does not enter the decision): mean differences in V1 between -0.0057 and +0.0014
  (0.05-0.11 in V0).

Interpretation (after the result): changing the padding at inference had made the
difference about a fifth smaller (0.060/0.066); in the model trained with reflect padding
from the start the difference is as large as in the zero-padded model (0.088/0.072
against 0.079/0.081). So the statement "a fifth of the difference comes from zero
padding" was the effect of changing a trained model's padding afterwards, not a
mechanism; it must be corrected in the paper. The difference left after BN-adapt does
come from zero padding: in the reflect-trained model BN-adapt and TENT close the
difference completely.

### V2-V4 (11 October 21:50; oct10/analyze_variants.py, rule unchanged)

Each variant 3 seeds, 200 epochs; the norm label in the result rows was checked.
- K1 (V4, |mean D| > 0.01 at no pair): PASSED (largest |mean D| 0.0037).
- K3 (clean error < 0.20): V2 0.1288, V3 0.1860, V4 0.1078 -> PASSED.
- Test B: in V2 (GroupNorm 32) 0 of the outer pairs positive, in V3 (LayerNorm) 1 ->
  "Otherwise: mixed; reported per variant."
  - V2 frozen: v=0.089 +0.0077 [+0.0132 +0.0019 +0.0081]; v=0.197 +0.0163 [-0.0003
    +0.0099 +0.0394]; v=0.490 +0.0190 [-0.0002 +0.0214 +0.0357]; none positive (sign or
    floor fails).
  - V3 frozen: v=0.089 -0.0012; v=0.197 +0.0177 [+0.0222 +0.0133 +0.0177] positive;
    v=0.490 +0.0144 [+0.0268 -0.0090 +0.0255] not.
- V4 (does not enter the decision): clean error 0.1078, against 0.1147 for V0 -> no cost
  (no difference by definition; frozen and BN-adapt differences all |.| < 0.005).

Interpretation (after the result): the outer-pair difference, 0.07-0.09 in the BatchNorm
model, falls below 0.02 in the models that normalize each image and passes the rule at
one pair at most. Error levels: in the GN/LN and standardization models the drift
member's error falls to that of the diffusion member (at v=0.197: V0 0.566/0.487, V2
0.501/0.485, V4 0.481/0.482). So the large difference depends on BatchNorm's statistics
fixed in training; by the plan this narrows the scope of the main claim ("a model with
BatchNorm") and is written up as such. It also shows directly that the difference is
avoidable: the model trained on standardized input avoids it at no cost in clean error.
