# Matched fog pairs: decision rules written before the results (2 and 3 October 2026)

English translation (12 October 2026) of the section "Decision rules written before the
results were seen" of the internal run plan for 3 October. The rest of that plan
(kernel generation, launching, per-account quotas and run times) is logistics and is
left out.

## Decision rules written before the results were seen

- **fog-common (R1). Rule written on 2 October, before any result; `oct3/summarize.py`
  applies it in code.** In a pair matched in v, the two members have *equal* Bayes risk
  by Proposition 1; the raw inputs are two different affine images of the same random
  variable X0 + N~, so any difference in one model's error between the two members comes
  by definition from the reversible part. For seed s: D_raw(s) = Err(drift member) -
  Err(diffusion member), on the raw input; D_inv(s) = the same difference for the clean
  model on the input L^-1(x - c) (the two members have the same distribution there; the
  difference is only noise sampling).
  - **Positive:** in at least one pair, (i) D_raw has the same sign in all three seeds,
    (ii) |D_raw(s)| > |D_inv(s)| in every seed, (iii) |mean D_raw| > the standard deviation
    of D_raw across seeds.
  - **Inconclusive:** no positive pair, and in every pair the raw errors of both members
    are within 0.05 of chance (error > 0.85).
  - **Negative:** none of the above.
  - **Incomplete:** if one of the three seeds is missing, no decision is made.
  - **The L^-1 arm is not evidence, it is a code check:** for the clean model, the L^-1
    errors of the two members must agree to within 4 standard errors of the difference of
    two independent binomial proportions (n = test set). If not, the code has a bug and the
    result is not interpreted.
  - **Reporting:** all four pairs are reported, with how many pass the criterion ("at
    least one" involves multiple comparisons).
- **Second test: drift_matched against diffuse_own** (two separate models whose training
  ranges are matched in v), same rule. One difference: the floor in (ii) is still the
  *clean model's* D_inv(s) for the same pair. The L^-1 difference of two different models
  is not sampling noise but a model difference; it is reported descriptively.
- **OUTCOME (3 October 06:00, applied with `oct3/summarize.py`, rule unchanged):**
  - The code check passed: the clean model's L^-1 differences are < 0.005 in every pair
    and seed.
  - Test 1 (clean model): the rule gives POSITIVE (1 of 4 pairs), but both members' errors
    are 0.89-0.90 (chance 0.90) and the difference is 0.008. Ceiling effect; not used as
    evidence.
  - Test 2 (drift_matched / diffuse_own): POSITIVE, 4 of 4 pairs, 3 of 3 seeds. At equal
    Bayes risk and equal horizon in v, the drift model's error is higher by 0.047-0.086;
    oracle differences -0.009 to 0.016. In pair 1 both points are training severities of
    their own models.
  - Outside the rule, descriptive: the original 4x reversal repeats with the regimes' own
    models in 3/3 seeds, and in 0/3 when the drift model's training range is matched in v
    (an extrapolation artefact; R1 confirmed).
- **fog-joint (started on 3 October 09:55; CIFAR-10 and CIFAR-100). Rule written on
  3 October 07:15, with no result.** A single model is trained, with the same budget, on
  the union of the two regimes' grids matched in v (t = 0; drift 1.4284, 1.7727; diffusion
  0.5, 1.0) and evaluated at both members of the four matched pairs. D_raw(s) and D_inv(s)
  are computed with this one model; since it is the same model and the same distribution,
  D_inv is only sampling noise.
  - **Positive / Inconclusive / Negative / Incomplete:** the same criteria as fog-common
    test 1.
  - **Code check:** D_inv within 4 SE in every pair and seed.
  - **Reporting:** all four pairs, with how many pass.
- **fog-joint-full (written on 3 October 17:30, with no result).** The joint model at the
  standard CIFAR budget: 200 epochs, per-sample crop and flip; everything else as in
  fog-joint (grid, budget, evaluation points). CIFAR-10 and CIFAR-100, 3 seeds each. Rule
  as in fog-joint. The clean test error is also reported (to show that the model is no
  longer undertrained).
  - **Reporting fixed in advance:** if positive, the paper's main numbers move to the
    full-budget result and the 15-epoch result stays beside it. If negative, the main claim
    is reported as "dependent on the short training budget" and the abstract and the
    contributions are changed accordingly. If partly positive (on one data set only), the
    two are reported separately.
- **fog-joint-full OUTCOME (4 October 01:45, rule unchanged):** the code check passed.
  CIFAR-10: POSITIVE, 3 of 4 pairs (v = 0.089, 0.197, 0.490: +0.015, +0.079, +0.081);
  inside the range +0.003. Clean error 0.115. CIFAR-100: POSITIVE, 3 of 4 pairs (+0.041,
  +0.072, +0.037); inside the range, +0.009 failed the floor in one seed. Clean error
  0.384. By the reporting rule, the main numbers move to the full-budget result.
- **Where the L^-1 arm is informative:** at unmatched points. At each point,
  Err(raw) - Err(L^-1) is the share of that model's error at that point that comes from
  the reversible part (for example in the 4x pair, with the regimes' own models).
- **wiener-lfl.** If witness(f32) is below the oracle, slack in the control family is
  proved without the float32 caveat (the witness is an upper bound on the R* of the stored
  data). If witness(f32) is close to witness(f64), the float32 loss is negligible. If the
  witness is still above the oracle, the pre-specified test stays "inconclusive" and 0.125
  is a valid bound only if R* is constant.
- **blur-factorial.** How much of the 7.7-fold factor the BN-adapt difference between
  periodic deterministic blur (sigma = 0, a_min = 0) and heat-equation blur (sigma = 15,
  a_min = 0.1) explains, and how much a_min alone (sigma = 0, a_min = 0.1) does.
  Attribution is made only according to this.
- **adam-main.** For heat-equation blur, fog and the control, the TENT (Adam) fractions
  replace the SGD fractions; under heat-equation blur, the DANN-TENT comparison is
  rewritten against TENT with Adam.
- **TIN.** If the normalization guard passed, the noise results and the fog pairs return
  to the main text as Section 5.7; if it did not, the kernel does not start in the first
  place.

*[translation note]* The fog-joint and fog-joint-full rules were written after the
fog-common results were known; the paper says so (Section 5.4).
