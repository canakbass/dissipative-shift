# Semigroup Scope -- A List Fixed in Advance

This file is a precondition of the Stage 1 kill-test: the scope is not discovered
by experiment afterwards, it is proved mathematically beforehand. The proof of
S(t+s) = S(t) . S(s) for each family fits on one line.

## In scope -- exact semigroups

**1. Gaussian noise.** S(t)(X) = X + N(0, 2t.I).
The sum of two independent Gaussians is Gaussian:
N(0,2t.I) * N(0,2s.I) = N(0, 2(t+s).I), so S(t).S(s) = S(t+s) exactly. This is
the standard heat semigroup on R^d.

**Implementation note.** The first
CIFAR-10-C and dSprites-C code applied `np.clip(., 0, 1)` on top of this step.
Since clip(clip(x+n1)+n2) != clip(x+n1+n2), that silently breaks the semigroup we
had proved, and it is not negligible at large t, where sigma=sqrt(2t) approaches
or exceeds the [0,1] pixel range. The fix: clipping removed entirely. Pixel values
may fall outside [0,1], which is not a problem for a network trained on
normalized inputs.

**2. Gaussian blur, parameterized by variance.** S(t)(X) = X conv K_t, with K_t a
Gaussian kernel of variance 2t. Convolving two Gaussian kernels gives a Gaussian
kernel with the summed variance, K_t conv K_s = K_{t+s}, so S(t).S(s) = S(t+s)
exactly for the continuous kernel. Our implementation samples the kernel at integer
pixels, truncates it at 4 sigma and pads by reflection; measured on 32x32 images,
composing two blurs of t=0.05 deviates from one blur of t=0.1 by 0.23 in relative
operator norm, falling to 0.014 at 0.25+0.25 and 1e-4 at 0.5+0.5. The cause is
sampling, not the boundary: a Gaussian of variance 0.1 sampled at integer pixels
has effective variance 0.013. This family is used as a control, whose role rests on
injectivity rather than composition, so the deviation does not affect it.

Note that the standard CIFAR-10-C "Gaussian Blur" severity levels do not use
this parameterization, being on a perceptual scale. Here t is the kernel variance
directly, not CIFAR-10-C's severity index.

**Correction (2026-09-29): this family is reversible, so not dissipative.**
Convolution with a Gaussian has strictly positive Fourier gain at every
frequency, so the map is injective and R*(t) = R*(0) is constant. This is the same
mistake we made with the fog model: composing correctly and losing information are
different requirements. The composition law holds, so the Blackwell inequality
still applies -- but as an equality. The theorem does not predict that R* grows.

We do not drop this family; we reclassify it as a **reversible control**. It is
the only place where R* is known exactly, and therefore the only family where we
can measure how far the oracle proxy sits from R*: the oracle's error rises by
0.125 across a family whose unavoidable error does not move. A Wiener
deconvolution test points the same way at t=0.25 and t=0.5, but as run it is not a
like-for-like comparison (different blur implementations, one seed), so it is
preliminary until rerun.

The implemented operator is injective: as a linear map on 32x32 images its
smallest singular value is 2.9e-2 at t=0.25 and 6.6e-12 at t=2, never zero. It is
badly conditioned (condition number 1.5e11 at t=2, against float32 precision of
1.2e-7), so images stored in float32 after blurring lose some high-frequency
components to rounding; the Bayes risk of the stored data may rise somewhat.


**2b. The dissipative version of blur: the stochastic heat equation.**
To get a blur that genuinely destroys information, add the diffusion term. In
Fourier space each mode is an independent OU process with rate
omega^2 = f_x^2 + f_y^2 + a_min and Var[eta_omega(t)] =
(sigma^2/2.omega^2)(1 - e^{-2.omega^2.t}). It composes exactly by
Chapman-Kolmogorov. Parameters: sigma=15, a_min=0.1. Here the oracle's error rises
from 0.119 to 0.539, against 0.125 to 0.250 in the deterministic version, where
the whole rise is oracle slack.

The a_min floor is what makes this a contraction at every frequency including DC,
so it is not simply the control plus noise: at t=2 the DC component is itself
attenuated to 0.82.

**3. Fog / Beer-Lambert attenuation -- corrected definition (2026-09).**
The first version was S(t)(X) = X.e^{-at} + A.(1-e^{-at}): a deterministic affine
map, no clipping, no noise. A later check showed that this is a bijection for every
finite t, invertible via X = (X(t)-A(1-e^{-at}))/e^{-at}. Bijections lose no
information, so R*(t)=R*(0) should have held. The Blackwell argument needs a
genuine garbling -- a stochastic kernel that destroys information -- and a
deterministic reparameterization does not supply one. This was a mathematical
error, not an implementation bug.

**The fix: an Ornstein-Uhlenbeck process.** dX = -a(X-A)dt + sigma.dW, whose
transition kernel is closed form:
X(t)|X(0) ~ N(A + (X(0)-A)e^{-at}, (sigma^2/2a)(1-e^{-2at})). This is an exact
Markov semigroup by Chapman-Kolmogorov, the standard property of any
time-homogeneous diffusion, so no separate proof is needed. It combines the old
Beer-Lambert attenuation term (the mean converges exponentially to A) and the
scattering noise (the variance grows) in one coherent process, and it does destroy
information: as t -> infinity every class converges to the same stationary
Gaussian around A, and R*(t) -> 0.5, chance level for two balanced classes. We fix
a=1, since t can be read as "a . physical time".

Closed-form validation is in `kill_test_fog.py`: a self-test (Delta(t) about zero
for the Bayes rule), a generator estimate, and extrapolation from low t out to
t=20. One correction to that file is worth recording: the generator estimate
originally took the class means from the analytic expression rather than from
samples, which made recovering `a` a rearrangement of the closed form rather than
a measurement. With the means estimated from samples it returns a_hat=0.988
against a true 1.0, and the extrapolation error grows from 0.002 to 0.005. The
looser number is the real one.

Real haze attenuates each pixel according to its scene depth, so its attenuation
depends on the image; this model attenuates every pixel equally and adds a Gaussian
scattering term. That is a simplification, but the model is a proved semigroup that
loses information.

## Out of scope -- not semigroups

**Motion blur.** Convolving a box kernel of length t with one of length s gives a
trapezoid, not a box of length t+s. The family is not closed, so
S(t).S(s) is not in {S(u) : u >= 0} under any reparameterization. Rejected before
any experiment.

What this does and does not assert: motion blur is not shown to preserve
information. What is missing is our route to establishing that it loses any --
the composition law. One could ask directly whether a length-L blur is a garbling
of a length-l blur, which in Fourier terms needs the ratio of the two transfer
functions to be a valid filter; the shorter box has zeros where the longer one
does not, so that fails in general. We did not pursue it further.

**JPEG compression.** Lossy quantization plus entropy coding, with a discrete
quality factor for "severity". Repeated compression does not satisfy
S(t+s) = S(t).S(s): the second pass re-encodes the first pass's block-quantization
artifacts differently, and there is no single continuous parameter in which to
state a composition law at all. Rejected.

## Conclusion

The CIFAR-10-C experiments use only the families below. The list was frozen before
Stage 2 began and was not turned into an algebraic surprise discovered later by
experiment. Two families were nonetheless reclassified, having passed the
composition test while turning out to be **reversible**.

**Dissipative families** (R* genuinely grows; the main results live here):
- Gaussian noise, without clipping
- Heat-equation blur (stochastic, sigma=15, a_min=0.1)
- OU fog (parameterized by a and sigma)

**Reversible controls** (R* is flat, so all measured degradation is avoidable;
used to measure the oracle's slack):
- Deterministic Gaussian blur
- Deterministic Beer-Lambert fog (the first version; its runs were discarded
  rather than reused, so only the blur control appears in the paper)

The reclassification narrows the scope rather than widening it: two families were
removed from the "dissipative corruption" claim and given a weaker, provable role
instead. dSprites-C was dropped entirely (see paper/removed/).
