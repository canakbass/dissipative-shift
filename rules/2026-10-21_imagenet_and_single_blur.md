# Pretrained ImageNet models on matched fog pairs, and the single-model blur pair

English translation (22 October 2026) of the internal plan `oct21/PLAN.md`, written and
approved on 21 October 2026 before any of its runs. After approval, only the dated
implementation correction and the "OUTCOME" section were added. Applied by
`oct21/summarize.py`; the paper's numbers come from `paper/facts.py`, which the verifier
compares with it field by field.

This document fixes, **before the runs**, the design of two experiments, their decision
rules and how the results enter the paper. It is not changed after the user's approval.
If a change is needed later, it is written in a separate, dated section and called
"later" in the paper.

Status: APPROVED by the user on 21 October 2026. The rules do not change after this date.

## Experiment 1: matched fog pairs on pretrained ImageNet models (`imnet-fog`)

**Question.** Does the CIFAR-10 finding "equal information, different error" appear in
standard, large-scale, pretrained models? If so, does the batch-normalization /
layer-normalization distinction hold there too? This answers the objections "CIFAR scale"
and "only models we trained ourselves". No training, inference only.

**Data.**
- The ImageNet-1k validation set, Kaggle dataset `sautkin/imagenet1kvalid` (1000 folders
  x 50 images).
- With a fixed seed (0), 10 images per class are drawn: 10,000 images.
- Preprocessing: short side resized to 256, centre crop 224 x 224, scaled to [0, 1]. Then
  the corruption, and last each model's own normalization.
- **Label-mapping check (stopping condition):** the clean top-1 error of ResNet-50
  (IMAGENET1K_V1) must lie in 0.20-0.28 (published value 0.239). Otherwise the folder <->
  class mapping is wrong and the run stops without writing results. The fallback data is
  then ImageNetV2 (matched-frequency, 10,000 images), with the same rule and the same check.

**Corruption.** The paper's Ornstein-Uhlenbeck fog: white haze point $A = 1$, no clipping.
- Drift regime: $a = 1$, $\sigma = 0.05$.
- Diffusion regime: $a = 0.05$, $\sigma = 0.2$.
- Pairs matched in variance, v in {0.0025, 0.005, 0.01, 0.021, 0.042, 0.089}. v = 0.042 and
  0.089 are shared with the CIFAR pairs; the smaller values were added because ImageNet
  models are more sensitive to noise than the CIFAR models.

| v | drift t | contrast beta | diffusion t | contrast beta |
|---|---|---|---|---|
| 0.0025 | 0.5493 | 0.577 | 0.0623 | 0.997 |
| 0.005 | 0.8047 | 0.447 | 0.1242 | 0.994 |
| 0.01 | 1.0986 | 0.333 | 0.2469 | 0.988 |
| 0.021 | 1.4396 | 0.237 | 0.5117 | 0.975 |
| 0.042 | 1.7719 | 0.170 | 0.9985 | 0.951 |
| 0.089 | 2.1393 | 0.118 | 2.0089 | 0.904 |

**Models** (torchvision weights, fixed by name):
- Batch normalization: `resnet50` IMAGENET1K_V1, `resnet50` IMAGENET1K_V2,
  `efficientnet_b0` IMAGENET1K_V1.
- Layer normalization: `vit_b_16` IMAGENET1K_V1, `convnext_tiny` IMAGENET1K_V1,
  `swin_t` IMAGENET1K_V1.

**Measurements.** For each model, each v and each noise seed (1, 2, 3; the same images):
- `err_drift` and `err_diff`: top-1 error on the two members.
- `err_drift_inv` and `err_diff_inv`: the same model's error on the inputs mapped back by
  $L_t^{-1}(x - c_t)$. Mapped back, the two members fall on the same distribution, so their
  difference is the **sampling floor**.
- Clean error.
- Batch-normalized models only: BN-adapt error (batches of 200, episodic), for both members.
- Gap = `err_drift - err_diff`. Floor = `|err_drift_inv - err_diff_inv|`.

**Decision rule** (the paper's matched-pair rule). A pair shows a "verified gap" if:
- (i) the gap has the same sign in all 3 seeds,
- (ii) in every seed, |gap| exceeds the floor,
- (iii) the absolute value of the 3-seed mean exceeds the standard deviation across seeds.

**Informative pair.** The rule applies only to informative pairs: the model's error on both
members is below 0.95 (chance 0.999) and at least 0.01 above the clean error. Pairs that
are not informative are reported but do not enter the decision.

**Pre-specified hypotheses and outcome classes.**
- **H1 (transfer):** at least two of the three batch-normalized models show a positive
  (drift member worse) verified gap at at least two informative pairs.
- **H2 (specific to batch normalization):** none of the three layer-normalized models shows
  a positive verified gap at two or more pairs. In addition, at its own two largest
  informative v, each layer-normalized model's mean gap is below half the mean gap of the
  batch-normalized models at the same v.
- **Outcome classes:**
  - H1 and H2 -> **"transfers at scale, specific to batch normalization"**. The paper's
    narrowing to batch normalization is supported at scale.
  - H1 without H2 -> **"transfers at scale, not specific to batch normalization"**. The gap
    is present in layer-normalized models too; the paper's claim narrowed to batch
    normalization widens, and the mechanism sentence is rewritten accordingly.
  - No H1 -> **"does not transfer to pretrained ImageNet models"**. Enters the paper as a
    limit of scope.
  - If any model shows a negative verified gap (diffusion member worse), the reversal of
    direction is reported separately.
- **Reporting:** all models, all pairs and all seeds in one table (appendix). The outcome
  class enters Section 5.7 as a paragraph. The abstract gets only one clause, according to
  the outcome class. The outcome class is reported whatever it is.
- **Note:** these models never saw fog; both members lie entirely outside their training
  data. This replaces the CIFAR distinction "beyond the training range" and is stated
  plainly in the paper.

**Compute.** One kernel per model, six kernels in all, each on a GPU with internet access
(for the torchvision weights) and the dataset attached. Estimated 0.5-1.5 hours per kernel.

## Experiment 3: the single-model blur pair (`blur-joint`)

**Question.** The 7.7-fold ratio of the blur pair was measured with a separate frozen model
for each blur. Like the joint model of the fog pairs, what is the ratio when one frozen
model is evaluated on both blurs?

**Design.**
- CIFAR-10, ResNet-18, the paper's 15-epoch recipe, 3 seeds (1, 2, 3).
- One frozen model, trained on data split equally over five points so that the total
  training budget stays the same: clean (t = 0), deterministic blur t in {0.05, 0.1},
  heat-equation blur (`blurfft_s15_a01`) t in {0.05, 0.1}.
- At the test severities of both families (0, 0.05, 0.1, 0.25, 0.5, 1, 2), the frozen
  model, BN-adapt and TENT (Adam, 1e-3) are evaluated.
- Oracles: retrained with the paper's recipe for both families at t in {1, 2} (4 oracles
  per seed).

**Primary quantity.** At t = 2, the ratio R = (error BN-adapt removes under deterministic
blur) / (error BN-adapt removes under heat-equation blur).

**Decision rule.**
- **"Replicates":** R > 1 in every seed and R >= 2 computed from the 3-seed means.
- **"Weakens":** R > 1 in every seed, but the mean R < 2.
- **"Does not replicate":** R <= 1 in at least one seed.
- Reported separately, outside the decision:
  - the difference of frozen errors between the two families at t = 2;
  - the split against the oracle (ratio of avoidable gaps and ratio of recovered
    fractions).

**Reporting.** One paragraph in Section 5.2. The caveat "a separate model for each blur" in
the abstract and the contributions:
- if the outcome is "replicates", the single-model result is given next to the main number;
- for the other two outcomes the caveat stays and the outcome is stated plainly.

## Implementation
- Code:
  - `kaggle_imnet_fog/imnet_fog.py` (new);
  - a new `blur_joint` job in `kaggle_oct3_runner/runner.py` (existing jobs untouched).
- Kernel directories under `oct21/`, built with the logic of `oct3/build_kernels.py`.
- Local smoke test:
  - imnet-fog with small randomly initialized models on 8 fake images;
  - blur-joint with a small model and little data.
  - Takes seconds; no heavy computation on the laptop.
- Results are downloaded to `oct21/<kernel>/output/`; the summary script applies the rule.
  When the results arrive, a dated "OUTCOME" section is added at the end of this document.
- The paper and the verifier are updated after the results in a separate author round,
  followed by an audit.

## Implementation correction (21 October, about 17:45, after the first kernel finished)
In the EfficientNet-B0 kernel, BN-adapt's clean error came out at 0.569. Cause: the images
are read in class order, so each batch of 200 holds only 20 classes, and the batch
statistics shift with the class. This does not match the episodic evaluation (mixed
batches) that the rule assumes. Correction: after loading, the images are shuffled once
with a fixed seed (0). The decision rule looks at frozen-model error; the frozen model's
per-image error does not depend on the order, so the rule is unaffected. Only the BN-adapt
diagnostic is affected. The three batch-normalized models (ResNet-50 V1, V2,
EfficientNet-B0) are rerun with the corrected code in kernels named `-r2`; BN-adapt values
are reported from the `-r2` runs only. The first runs are kept for frozen error, and the
two runs are compared and reported (the effect of order and of the noise draw). The
decision rule and the hypotheses did not change.

## OUTCOME (21 October 2026, 19:00; `python oct21/summarize.py`, details in `oct21/summary.json`)

**Experiment 1, outcome class: "transfers at scale, not specific to batch normalization"
(H1 held, H2 failed).**
- All six models show a positive verified gap at nearly every informative pair: the drift
  (lower-contrast) member is worse.
- All three batch-normalized models pass the H1 criterion; all three layer-normalized
  models show a positive verified gap at two or more pairs.
- The gap is larger in the layer-normalized models. At v = 0.042: ViT-B/16 +0.490, Swin-T
  +0.354, ConvNeXt-T +0.320, ResNet-50 V1 +0.102, ResNet-50 V2 +0.068, EfficientNet-B0
  +0.033.
- EfficientNet-B0 shows a verified negative gap at v = 0.089 (-0.055, both members near
  chance).
- The `-r2` reruns (shuffled batches) reproduced the frozen results: a model's gap moves
  by at most 0.005, and the rule's decision is the same at every pair.
- BN-adapt (`-r2` only, diagnostic): at small v it largely closes the gap (EfficientNet-B0
  |gap| <= 0.005 at every v; ResNet-50 <= 0.01 at v <= 0.005). At large v it remains in the
  ResNet-50s (+0.07 at v = 0.042).
- Note on interpretation: these models never saw fog. They correspond to the CIFAR
  "trained on clean data only" model, which was at chance on CIFAR. The CIFAR
  BN/GN/LN comparison, in contrast, used models trained on fog. The two results measure
  different situations.

*[translation note]* The "at most 0.005" above is the original's rounding; the largest
change is 0.0051, which the paper gives as "0.005" (nearest rounding), not as a bound.

**Experiment 3, outcome: "replicates".**
- At t = 2 the ratio of error removed by BN-adapt is 7.01, 13.53 and 5.29 per seed; 7.44
  from the 3-seed means.
- Frozen error 0.724 against 0.809; oracles 0.235 against 0.541.
- Split against the oracle: ratio of avoidable gaps 1.83, recovered fraction 0.674 against
  0.166, that is 4.07-fold.
