# Pre-registration v5 -- DANN lambda sweep

**Written:** 2026-09-29, before the job was launched. No results had been seen.

## Why

A reviewer objected that the paper's claim -- "DANN failed to beat the frozen
baseline in eleven of twelve cases" -- could be a by-product of DANN being
undertuned. If so, the claim is a tuning artifact and should be withdrawn.

This sweep scales DANN's standard lambda schedule as
`lambd(p) = lambda_max * (2/(1+e^{-10p}) - 1)`
over `lambda_max in {0.1, 0.3, 1.0, 3.0}`. The value 1.0 is the setting used in
the literature and in our main run.

Scope: `gauss_noise` only, and only the four severities OUTSIDE the training
horizon (t = 0.02, 0.05, 0.1, 0.2), which is where the claim lives.

**The frozen model and the oracle are re-measured inside the same run.** The
reviewer was right to criticize cross-run comparison; every Delta in this sweep
comes from one run.

## Predictions

1. **Main prediction:** if DANN's failure is real, no `lambda_max` closes the gap
   appreciably at held-out severities. Criterion:
   `min_lambda Delta_DANN(t) > 0.5 * Delta_frozen(t)` at both t=0.1 and t=0.2.

2. **Secondary:** the largest `lambda_max` (3.0) gives the worst result, because
   aligning domains on a target whose information has been destroyed damages the
   class structure too.

3. **Secondary:** no `lambda_max` beats the oracle, i.e. `Delta_DANN(t) > 0` for
   every t and every lambda.

## Refutation condition (binding)

If any `lambda_max` gives `Delta_DANN(t) < 0.5 * Delta_frozen(t)` at t=0.1 or
t=0.2, then **the paper's DANN claim is a tuning artifact.** In that case the
"eleven of twelve" sentence is removed and replaced by what the sweep shows for a
tuned DANN. We will not narrow the sweep range or hunt for a new lambda value in
order to rescue the claim.

If prediction 2 or 3 fails, those sentences come out of the discussion; the
sweep's main result (prediction 1) is judged independently.

## What we will not do

- No extra lambda value will be added after seeing the sweep.
- Learning rate and epoch count will not be tuned separately in DANN's favour;
  every arm sees the same budget (15 epochs, LR 0.1, cosine).

## Outcome (added after the fact)

Prediction 1 held. Taking the best `lambda_max` separately at each severity --
which requires test labels, so no honest practitioner could do it -- the ratio
`min_lambda Delta_DANN / Delta_frozen` was 1.13, 0.90, 0.82 and 0.85 across the
four severities, never below the 0.5 threshold. Prediction 3 held: no setting
beat the oracle. Prediction 2 failed: the worst setting was lambda_max=3.0 at two
severities, 0.3 at one and 1.0 at another, with no consistent ordering, so we make
no claim about how the adversarial weight should be set.
