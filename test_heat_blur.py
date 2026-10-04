"""Tests the semigroup property of the blur_heat family, using the operator the
experiments actually apply.

An earlier version of this check used omega^2 = fy^2+fx^2, i.e. without the a_min
floor, while the real corruption uses rate = fy^2+fx^2+a_min. Testing a different
operator is not a test, so the operator here is copied verbatim from the runners.

This check was promised in PREREG_v3 and never reported; it is reported now.
"""
import numpy as np

BLUR_HEAT_SIGMA = 15.0
BLUR_HEAT_A_MIN = 0.1


def _rate(h, w):
    fy = np.fft.fftfreq(h) * 2 * np.pi
    fx = np.fft.fftfreq(w) * 2 * np.pi
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    return FY**2 + FX**2 + BLUR_HEAT_A_MIN     # a_min included


def heat_blur(imgs, t, rng, sigma=BLUR_HEAT_SIGMA):
    """Identical to corrupt(..., 'blur_heat') in the runners."""
    if t == 0:
        return imgs.copy()
    n, c, h, w = imgs.shape
    rate = _rate(h, w)
    decay = np.exp(-rate * t)
    X = np.fft.fft2(imgs, axes=(-2, -1))
    mean = np.real(np.fft.ifft2(X * decay[None, None], axes=(-2, -1)))
    var = (sigma**2 / (2 * rate)) * (1 - decay**2)
    filt = np.sqrt(var / (h * w))
    white = rng.standard_normal((n, c, h, w))
    colored = np.real(np.fft.ifft2(np.fft.fft2(white, axes=(-2, -1)) * filt[None, None], axes=(-2, -1)))
    return (mean + colored).astype(np.float32)


rng = np.random.default_rng(0)
h = w = 32
img = rng.random((400, 1, h, w)).astype(np.float32)
rate = _rate(h, w)

OUT = {}
print("=== blur_heat semigroup check (the operator used in the code) ===")
print(f"sigma={BLUR_HEAT_SIGMA}, a_min={BLUR_HEAT_A_MIN}, {h}x{w} periodic grid\n")

# (1) Composition of the mean: S(t)S(s) = S(t+s), the deterministic part
print("(1) deterministic part: exp(-rate*t)*exp(-rate*s) == exp(-rate*(t+s))")
for t, s in [(0.25, 0.25), (0.5, 1.0), (1.0, 1.0)]:
    lhs = np.exp(-rate * t) * np.exp(-rate * s)
    rhs = np.exp(-rate * (t + s))
    print(f"  t={t}, s={s}: max|diff| = {np.abs(lhs - rhs).max():.3e}")
    OUT["det_max_abs"] = max(OUT.get("det_max_abs", 0.0), float(np.abs(lhs - rhs).max()))

# (2) Composition of the variance, by Chapman-Kolmogorov
print("\n(2) noise variance: Var(t+s) == Var(t)*exp(-2*rate*s) + Var(s)")
def var_of(t): return (BLUR_HEAT_SIGMA**2 / (2 * rate)) * (1 - np.exp(-2 * rate * t))
for t, s in [(0.25, 0.25), (0.5, 1.0), (1.0, 1.0)]:
    lhs = var_of(t + s)
    rhs = var_of(t) * np.exp(-2 * rate * s) + var_of(s)
    print(f"  t={t}, s={s}: max|diff| = {np.abs(lhs - rhs).max():.3e}  (relative {np.abs((lhs-rhs)/lhs).max():.3e})")
    OUT["var_max_rel"] = max(OUT.get("var_max_rel", 0.0), float(np.abs((lhs - rhs) / lhs).max()))

# (3) At the sample level: applying it in two steps and in one should match in
#     second moment, since the distributions should be identical
print("\n(3) samples: two-step vs one-step, empirical variance per frequency")
for t, s in [(0.25, 0.25), (0.5, 0.5)]:
    r1 = np.random.default_rng(1); r2 = np.random.default_rng(2)
    two = heat_blur(heat_blur(img, t, r1), s, r1)
    one = heat_blur(img, t + s, r2)
    p_two = (np.abs(np.fft.fft2(two, axes=(-2, -1)))**2).mean(axis=(0, 1))
    p_one = (np.abs(np.fft.fft2(one, axes=(-2, -1)))**2).mean(axis=(0, 1))
    rel = np.abs(p_two - p_one) / np.maximum(p_one, 1e-12)
    print(f"  t={t}, s={s}: power-spectrum relative difference, median={np.median(rel):.4f} "
          f"95th={np.quantile(rel, 0.95):.4f}  (n={img.shape[0]} images)")
    OUT.setdefault("ps_median_two_vs_one", {})[f"{t}+{s}"] = float(np.median(rel))
    # the sampling floor: two independent one-step samples of the same severity
    one_b = heat_blur(img, t + s, np.random.default_rng(3))
    p_b = (np.abs(np.fft.fft2(one_b, axes=(-2, -1)))**2).mean(axis=(0, 1))
    rel_b = np.abs(p_b - p_one) / np.maximum(p_one, 1e-12)
    print(f"  t={t}, s={s}: floor, one-step vs one-step (independent): median={np.median(rel_b):.4f}")
    OUT.setdefault("ps_median_one_vs_one", {})[f"{t}+{s}"] = float(np.median(rel_b))

# (4) What a_min does to DC: the source of the difference from the control
print("\n(4) DC attenuation due to a_min (it would be 1.0 for deterministic blur)")
for t in (0.25, 0.5, 1.0, 2.0):
    print(f"  t={t}: DC gain = {np.exp(-BLUR_HEAT_A_MIN*t):.4f}")

import json
OUT["n_images"] = int(img.shape[0])
json.dump(OUT, open("test_heat_blur_results.json", "w"), indent=1)
print("\nwritten: test_heat_blur_results.json")
