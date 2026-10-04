# Pre-registration v6 -- TENT hyperparameter sweep

**Written:** 2026-09-30. The job had been launched but NO results had been seen.

## Why

A reviewer objected that our "BN-adapt is approximately TENT" result could be a
hyperparameter artifact. The evidence suggests the objection is serious:

- We ran TENT at a single setting: SGD, lr=1e-3, momentum 0.9, one step per
  batch, batch size 200.
- TENT and EATA agree to within 0.001 everywhere. Two different methods landing
  that close suggests both are barely moving from the source model.
- TENT's own paper reports gains over statistic re-estimation. Our null result
  contradicts it, and a difference in the adaptation setting would explain that.

So "the gradient step is unnecessary" and "our gradient step was not large
enough" are currently indistinguishable. This sweep separates them.

## Setup

The frozen model is trained once, then 17 configurations are evaluated against
that same model: BN-adapt, plus TENT x {SGD, Adam} x lr in {1e-4, 1e-3, 1e-2,
1e-1} x steps per batch in {1, 4}. Three families (gauss_noise, deterministic
blur, OU fog), all severities. One seed.

**No oracle is trained:** comparing BN-adapt against TENT does not need one,
because the oracle cancels from the difference of their Deltas. That makes the
sweep cheap and keeps the comparison inside a single run.

## Predictions

1. **Main prediction:** no configuration puts TENT appreciably ahead of
   BN-adapt. Criterion: the best TENT configuration is at most 0.01 better than
   BN-adapt, averaged over held-out severities, per family.

2. **Secondary:** large learning rates (1e-1) are unstable and push the error
   above BN-adapt's.

## Refutation condition (binding)

If any configuration beats BN-adapt by more than 0.01 on average over held-out
severities, then **our conclusion that entropy minimization adds nothing to the
statistics update is wrong.** In that case:

- The BN-adapt-equals-TENT claim is withdrawn outright, not qualified with "at
  our setting".
- The main tables are regenerated with the better TENT configuration.
- The corresponding item is removed from the abstract and the contributions.

We will not narrow the sweep range to rescue the claim, nor dismiss the result by
appealing to what the standard setting is.

## What we will not do

- No new setting will be added after the fact because we dislike the result.
- No separate tuning in BN-adapt's favour; it has nothing to tune, being zero
  gradient steps and statistics only.

## Outcome (added after the fact)

The prediction failed in two of three families and the claim was withdrawn.
Averaged over held-out severities, the best configuration beat BN-adapt by 0.020
under Gaussian noise and 0.014 in the deterministic-blur control, both above the
0.01 threshold; only fog stayed inside it, at 0.007.

The diagnosis was narrower than "undertuned". Switching only the optimizer, from
SGD to Adam at the same learning rate and the same single step, moved TENT from
0.0027 ahead of BN-adapt to 0.0171 ahead under noise. The sweep's overall best
(SGD, 1e-2, four steps) added little beyond that.

Prediction 2 held. Eleven of the twelve configurations at learning rate 1e-1
were worse than BN-adapt, most of them badly (Adam at 1e-1 drove the error to
0.85-0.87, near chance); the twelfth, SGD with one step under Gaussian noise, was
0.0006 better, which is inside any reasonable noise floor. (An earlier draft of
this outcome section stated that prediction 2 had failed. That was written
without checking the data and was wrong.)

The Gaussian-noise comparison was rerun end to end with Adam, three seeds, all
quantities from one run per seed. The paper reports the withdrawal rather than
qualifying the original claim, and keeps the uncorrected columns, marked, for the
other families.
