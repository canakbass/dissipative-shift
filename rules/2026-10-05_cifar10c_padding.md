# CIFAR-10-C: first-layer padding and test-time normalization

English translation (12 October 2026) of the internal plan written on 5 October 2026,
before any model or data was downloaded and before any result was seen. After the run,
only dated amendments made before the result and the "OUTCOME" section were added.
Applied by `bench/analyze_bench.py`.

## Hypothesis (H)

In standard pretrained CIFAR-10 models, switching the first convolution's zero padding
to reflect padding at inference lowers BN-adapt's error under CIFAR-10-C's contrast-type
corruptions. Reason: the padding test (`2026-10-05_first_layer_padding.md`); the contrast
corruption is almost an affine map before quantization. Uncertainty: in CIFAR-10-C
contrast, the centre is the image's own mean (a per-image offset, which batch statistics
cannot fully remove), and there is clipping and uint8 quantization; CIFAR-10-C fog is
not affine (an added fractal pattern).

Literature (search of 5 October): spatial effects of padding are known (Alsallakh et al.
2021; Kayhan and van Gemert 2020; in adversarial robustness, Gavrikov and Keuper 2023);
no study was found showing that test-time normalization fails under contrast-type
shifts because of padding. Majumdar and Saha (2026) report that TTA failures cluster in
brightness, fog, contrast and defocus.

## Design (inference only, CPU)

Models: chenyaofo/pytorch-cifar-models (torch.hub) CIFAR-10 weights: resnet20, resnet56,
vgg13_bn. If one cannot be loaded, the next: resnet32, resnet44, vgg11_bn. Normalization
with each model's own values.

Data: the official CIFAR-10-C (Zenodo 2535967), severities 1-5, 10000 images each, in
file order.
- Hypothesis corruptions: contrast (primary), brightness, fog (secondary).
- Controls: gaussian_noise (no affine structure), and the clean test set.

Conditions: the first convolution's padding in {zeros (as trained), reflect}; other
layers unchanged. Each: frozen (stored statistics) and BN-adapt (batch 200, batch
statistics in training mode, no gradient). Secondary: TENT (resnet20 only; Adam, lr
1e-3, one step per batch, from scratch at each severity), only on the hypothesis
corruptions.

## End point and decision rule

Gain G = Err_BNadapt(zeros) - Err_BNadapt(reflect), for one model and corruption, the mean
over the 5 severities. Paired standard error: at each severity, the standard error of the
difference in correctness on the same images, combined treating the five severities as
independent.

For each corruption separately, over the three models:
- H supported: G > 0 in all three models, G > 3 SE in at least two.
- H rejected: G <= 1 SE (negative included) in at least two models.
- Otherwise: inconclusive.
Practical size is reported separately: G >= 0.01 counts as "of meaningful size".

Controls (they limit the interpretation and do not change H on their own):
- C1 clean set: |Err(zeros) - Err(reflect)| with BN-adapt is reported for every model; if
  reflect raises the clean error by more than 0.005, this is written up as a cost.
- C2 gaussian_noise: even if H is supported for contrast, if G > 3 SE under
  gaussian_noise in at least two of the three models, the effect is written up as not
  specific to affine shifts.
- The effect of the padding change on the frozen model (G_frozen) is reported; it does
  not enter the decision.

Reporting: all models, corruptions, severities, whatever the outcome.

## Amendment (5 October 23:10, before the result)

The download of the official CIFAR-10-C from Zenodo stopped at 3.8 MB (two attempts).
The four corruptions are generated with the functions of the official generator
(hendrycks/robustness, make_cifar_c.py), in the same storage format (np.uint8,
severities stacked): bench/make_cifar10c_subset.py. Contrast and brightness are
deterministic, fog and Gaussian noise random with a fixed seed. If the official files
can be downloaded later, contrast and brightness are compared one to one and the test is
run once more on the official files.

## Preview (5 October 23:15, rule unchanged)

During the script's smoke test, a contrast transform written by hand (severity 5,
c = 0.15, towards the per-image channel mean) was applied with resnet20 to the first 2000
images of the clean test set: BN-adapt error 0.1905 with zero padding, 0.1915 with
reflect. Not an official run; the rule and the design were not changed after it.
Possible explanation (written after the result): CIFAR-10-C contrast pulls the image
towards its own mean; in normalized units this mean is close to the value of zero
padding (the data set's mean), whereas in the fog pairs the haze point was white.

## Amendment 2 (6 October, before the Kaggle result was seen)

The run started on the laptop was stopped at the user's request (only resnet20 and
resnet56 had finished; kept as a local copy). The decision run is on Kaggle (GPU), all
three models in the same kernel, with the data generated inside the kernel by the same
official functions. Fog and Gaussian noise may be drawn differently from the local data
depending on the numpy version; contrast and brightness are deterministic. The rule is
applied to the Kaggle results.

## OUTCOME (6 October 11:10, Kaggle, rule unchanged, with bench/analyze_bench.py)

H REJECTED for all three corruptions (G <= 1 SE in at least two models for each).
- Contrast, BN-adapt: G = -0.0028 (resnet20), -0.0091 (resnet56), +0.0028 (vgg13_bn; from
  severity 5 alone, +0.0136). Brightness: -0.0020, -0.0034, -0.0024. Fog: -0.0059,
  -0.0080, -0.0029.
- TENT (resnet20): contrast -0.0061, brightness -0.0012, fog -0.0034.
- Frozen (the padding change itself): |G| <= 0.002 for contrast, brightness and fog.
- C1 clean set, BN-adapt: reflect raises the error by +0.0032, +0.0062 (a cost, > 0.005),
  +0.0018.
- C2 Gaussian noise: G > 3 SE in no model; reflect hurts BN-adapt by -0.008 to -0.014.

Interpretation (after the result): changing the padding at inference has a general cost
(a train-test mismatch), and under CIFAR-10-C's contrast-type corruptions there is no
gain that exceeds it. In the fog pairs the effect was large because the haze point
(white) is far from the value of zero padding (the data set's mean); CIFAR-10-C contrast
pulls the image to its own mean, a point close to that value. The effect is specific to
corruptions that pull towards a point far from the padding value.

*[translation note]* The references in the literature paragraph are those of the
original note; Gavrikov and Keuper (2023) and Majumdar and Saha (2026) are not cited in
the paper.
