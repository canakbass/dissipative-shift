# Pre-registration -- Testing the Fog Hypothesis by Ablation (2026-09-28)

Written BEFORE running it.

## The hypothesis under test

In the paper's discussion we proposed that recovery is low in the fog family
because of the OU process's **drift term**: every class mean converges toward a
common point A. Batch-norm-level adaptation (BN-adapt, TENT, EATA) can correct
global channel statistics but cannot undo the convergence of class means, since
undoing that would require already knowing which class an image belongs to.

Until now this was a hypothesis and had not been tested.

## Design

The OU process dX = -a(X-A)dt + sigma dW has two observable effects:
- m(t) = e^{-at}: the surviving fraction of the original signal, and of the
  between-class separation
- v(t) = (sigma^2/2a)(1-m^2): the added variance

Changing `a` alone moves both, so instead of nudging one parameter we construct
**two extreme regimes**, both still exact OU semigroups so the theory is intact:

- **fog_drift** (a=1.0, sigma=0.05): the mean converges quickly and the noise is
  negligible. The loss is almost entirely from mean convergence.
- **fog_diffuse** (a=0.05, sigma=0.2): the mean barely converges (m=0.905 at
  t=2) and the variance is large. The loss is almost entirely from noise.

## Predictions (binding)

1. **fog_drift**: the oracle's error will rise only SLIGHTLY with t, since this
   regime is nearly reversible -- in the sigma->0 limit it is the old
   deterministic fog, for which R*(t)=R*(0). The frozen model will nonetheless
   degrade badly, so Delta_frozen will be large. The critical prediction:
   **TENT and BN-adapt will close very little of that gap** (recovery < 30%).
2. **fog_diffuse**: the oracle's error will rise clearly with t (genuine
   information loss). The critical prediction: **TENT and BN-adapt will close
   most of the gap** (recovery > 50%), because the added noise shifts the channel
   statistics, which is the kind of damage BN can correct.
3. The **between-class separation over within-class spread** ratio, measured in
   feature space, will collapse with t in fog_drift and fall much less in
   fog_diffuse.

If 1 and 2 both hold, the hypothesis is supported. If recovery turns out high in
fog_drift as well, the hypothesis is WRONG, and we will remove the paragraph from
the discussion rather than leaving it in as a hypothesis.

## What we will not do

No t-SNE or PCA visualization: t-SNE geometry cannot support a quantitative
claim. The scalar separation measure above is reported instead.

## Outcome (added after the fact)

Predictions 1 and 2 failed; prediction 3 is indeterminate. Three seeds throughout;
recovery fractions are ratios of three-seed means.

Prediction 1 failed in both of its parts. The drift regime's oracle rose by 0.363
over its grid, essentially the same as the diffusion regime's 0.365, not "only
slightly"; and TENT recovered 64%, 43% and 14% of the drift regime's gap at its
three held-out severities, below 30% only at the last. The premise was wrong, not
the oracle: inverting the attenuation e^{-at} multiplies the scattering noise by
e^{at}, so the drift regime's equivalent noise variance grows as e^{2at} and
reaches 0.503 at t=3 -- the same as the diffusion regime's 0.490 at t=8. The
paper derives this (Proposition 1).

Prediction 2 failed: diffusion-regime recovery was 47%, 34% and 27%, never above
50%, and per seed it was unstable (0.5%, 87% and 36% at t=2) because the gap
being recovered is close to zero there.

Prediction 3 is indeterminate. The frozen model's feature separation fell to 16%
of its clean value under drift (t=3) and to 35% under diffusion (t=8), in the
predicted direction, but at the 2x extrapolation horizon the order is reversed,
and we did not fix in advance what "much less" meant. We count it neither way.
(An earlier draft of this section called it unsupported; that overstated the
data.) One observation we did not predict: at those two end points, which have
nearly equal Bayes risk, the oracle's feature separation is nearly equal too
(0.846 and 0.849), while the frozen model's differs by a factor of two.

The mechanism hypothesis was removed from the discussion, as this file required.
