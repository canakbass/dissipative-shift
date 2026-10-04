# Pre-registration -- Oracle Tightness and Reversible Control Families (2026-09-28)

Written BEFORE running any of the tests below. The aim is to make "how close is
the oracle to R*" a central measurement of the paper rather than a footnote.

## Background

Gaussian blur, like our first fog model, is provably invertible: the Fourier
transform of a Gaussian kernel never reaches exactly zero, so deconvolution is
always possible in exact arithmetic. Between t=0.25 and t=1.0 that invertibility
survives float32 precision (the Nyquist gain falls from 0.085 to 5e-5) and is
lost numerically only around t=2.0 (gain 2.7e-9). So R*(t) should equal R*(0),
yet our retrained oracle's error rises with t, reaching 0.25 at t=2.0. That
points to the oracle proxy being a LOOSE rather than tight upper bound on R* --
something the theory predicts but we have not measured.

(The Nyquist figures above are for the idealized continuous kernel. We later
measured the discrete operator the code actually applies and the numbers differ;
see the paper.)

## Predictions, written before running

1. **Wiener deconvolution plus a clean (t=0) classifier** will give LOWER error
   than the retrained oracle at blur t=1.0 and t=2.0, because the true R*(t)
   equals R*(0) while the retrained oracle cannot reach it, being limited by
   training and representation. Concretely: error <= 0.16 at t=1.0 (the current
   oracle is 0.17) and <= 0.18 at t=2.0 (current oracle 0.25). The exact numbers
   are unknown but the DIRECTION is not: the Wiener route should be at least as
   good as the oracle everywhere.
2. **The old, pre-correction deterministic fog data** can serve as a second
   reversible control family: another case where R*(t) should be flat while the
   retrained oracle rises. Before using it we need to confirm that the noise
   clipping in those runs did not leak into the fog arm. Code inspection suggests
   it did not, since they are separate function calls.
3. **The stochastic heat equation** (frequency-dependent OU) will be validated in
   a kill-test under the same discipline: self-test within 0.02, small
   extrapolation error, and R*(t) converging to the Bayes error of the stationary
   distribution -- not to 0.5, because here the class means do not converge, only
   the variance grows.

## Binding precondition

The main CIFAR run (more seeds, longer training, all methods in one run) will NOT
start until these three tests have resolved. If the Wiener test does not come out
in the predicted direction -- that is, if the retrained oracle already beats
Wiener -- then our "the oracle is loose" hypothesis is wrong and the plan gets
revisited.

## Outcome (added after the fact)

Prediction 1 failed. It was registered for t=1.0 and t=2.0, and at both the
retrained oracle beat the Wiener route: 0.179 against 0.241 at t=1 and 0.250
against 0.507 at t=2, with the forward and inverse operators matched and the
regularizer swept. (The first implementation of this test had mismatched
operators and a fixed regularizer; both errors are documented in the paper's
corrections log. Fixing them did not change the outcome at the registered
severities.)

Addendum (later audit). We first recorded prediction 1 as failed. The test cannot
decide it. The regularizer selected at both registered severities, eps = 1e-4, was
the smallest in the grid swept (1e-4 to 1e-1). At that value the Wiener filter's
gain never exceeds 1/(2 sqrt(eps)) = 50, and it attenuates to less than half every
mode whose gain is below sqrt(eps) = 1e-2: 64% of the modes at t=1 and 82% at t=2.
The witness lost because its regularizer discarded those frequencies, which says
nothing about whether the stored images still contain them. We therefore record
the outcome as "not met; inconclusive". It is not counted as support for the
hypothesis either. Under the binding precondition above, the plan was revisited:
the like-for-like rerun puts oracle and witness on the same operator, extends the
grid to 1e-14, and adds the same witness on images stored in float64.

At t=0.25 and t=0.5, which were NOT part of the registered prediction, the Wiener
route came out ahead (0.125 against 0.130, and 0.130 against 0.147). We report
that as an unregistered observation and not as a confirmation, for three
reasons: it was not predicted at those severities; the two columns are evaluated
on different blur implementations (the Wiener column on a periodic FFT blur, the
oracle column on the sampled, reflect-padded blur of the main experiments), so
they are not the same data; and the Wiener column is a single seed, with margins
(0.005 and 0.017) no larger than the oracle's own seed-to-seed spread (0.008 and
0.019). A like-for-like rerun is planned.

Prediction 2 was not pursued: the old fog runs were discarded rather than reused.
Prediction 3 was narrowed to a composition check, since there is no closed-form
Bayes risk for the image-grid version.
