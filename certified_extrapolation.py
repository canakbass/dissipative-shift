"""Certified extrapolation of the Bayes risk along an Ornstein-Uhlenbeck fog family.

The fog generator (a, sigma) is estimated from unlabelled samples at low severities.
Isotropic noise only moves the trace of the covariance, so the traceless part decays
exactly as exp(-2at); the trace then gives sigma. A bootstrap rectangle for (a, sigma)
gives, by monotonicity of v(t; a, sigma) = sigma^2 (e^{2at} - 1) / (2a) in both
parameters, an interval [v_lo(t), v_hi(t)] for the equivalent noise variance, and by
Corollary 1 a bracket R*_ref(v_lo) <= R*(t) <= R*_ref(v_hi).

Part 1 is the two-dimensional setting of kill_test_fog.py, where R* is known in closed
form, so coverage and width can be measured. Writes certified_extrapolation_results.json.
"""
import json

import numpy as np
from scipy.stats import norm

D, SIGMA0 = 3.0, 0.5                     # class means (0,0) and (D,0), within-class sd
N_PER_T = 20000                          # unlabelled samples per training severity
N_BOOT = 200
N_REP = 100
ALPHA = 0.05                             # joint level of the (a, sigma) rectangle (Bonferroni)


def v_eq(t, a, s):
    return s ** 2 * np.expm1(2 * a * t) / (2 * a)


def r_star(v):
    """Bayes risk of X0 + N(0, vI) for the two-class mixture (Proposition 1)."""
    return norm.cdf(-D / (2 * np.sqrt(SIGMA0 ** 2 + v)))


def sample(t, n, a, s, rng):
    """Unlabelled fog samples at severity t; haze point midway between the means."""
    y = rng.integers(0, 2, n)
    x0 = np.stack([y * D, np.zeros(n)], 1) + SIGMA0 * rng.standard_normal((n, 2))
    A = np.array([D / 2, 0.0])
    decay = np.exp(-a * t)
    return A + (x0 - A) * decay + np.sqrt(s ** 2 / (2 * a) * (1 - decay ** 2)) * rng.standard_normal((n, 2))


def estimate(covs, ts):
    """(a, sigma) from per-severity covariances. Traceless part: T(t) = e^{-2at} T(0)."""
    ts = np.asarray(ts)
    tl = np.array([np.linalg.norm(c - np.trace(c) / 2 * np.eye(2)) for c in covs])
    a = -np.polyfit(ts, np.log(tl), 1)[0] / 2
    half_tr = np.array([np.trace(c) / 2 for c in covs])
    g = -np.expm1(-2 * a * ts[1:]) / (2 * a)          # (1 - e^{-2at}) / 2a
    resid = half_tr[1:] - np.exp(-2 * a * ts[1:]) * half_tr[0]
    s2 = max(float(np.dot(g, resid) / np.dot(g, g)), 1e-12)
    return a, np.sqrt(s2)


def rectangle(samples, ts, rng):
    est = []
    for _ in range(N_BOOT):
        covs = [np.cov(x[rng.integers(0, len(x), len(x))].T) for x in samples]
        est.append(estimate(covs, ts))
    est = np.array(est)
    q = ALPHA / 4                                      # two parameters, two sides
    return np.quantile(est[:, 0], [q, 1 - q]), np.quantile(est[:, 1], [q, 1 - q])


def run_regime(name, a, s, train_ts, test_ts, rng):
    cover, widths, bounds_ok, rects = [], [], [], []
    for _ in range(N_REP):
        samples = [sample(t, N_PER_T, a, s, rng) for t in train_ts]
        (a_lo, a_hi), (s_lo, s_hi) = rectangle(samples, train_ts, rng)
        rects.append([a_lo, a_hi, s_lo, s_hi])
        t = np.asarray(test_ts)
        v_lo, v_hi = v_eq(t, a_lo, s_lo), v_eq(t, a_hi, s_hi)
        r_true = r_star(v_eq(t, a, s))
        cover.append(bool(np.all((r_star(v_lo) <= r_true) & (r_true <= r_star(v_hi)))))
        w = np.log(v_hi / v_lo)
        widths.append(w)
        c = 2 * np.log(s_hi / s_lo)
        # W - c lies in [0, 2 t (a_hi - a_lo)] always, and above t (a_hi - a_lo) when a_lo >= 0
        lower = t * (a_hi - a_lo) if a_lo >= 0 else 0.0
        bounds_ok.append(bool(np.all((c + lower <= w + 1e-12) & (w <= c + 2 * t * (a_hi - a_lo) + 1e-12))))
    widths = np.array(widths)
    rects = np.array(rects)
    out = dict(a=a, sigma=s, train_ts=train_ts, test_ts=test_ts,
               coverage_simultaneous=float(np.mean(cover)), width_bounds_hold=float(np.mean(bounds_ok)),
               mean_log_width=dict(zip(map(str, test_ts), widths.mean(0).tolist())),
               mean_a_halfwidth=float(np.mean(rects[:, 1] - rects[:, 0]) / 2),
               frac_a_lo_positive=float(np.mean(rects[:, 0] > 0)),
               slope_last=float((widths.mean(0)[-1] - widths.mean(0)[-2]) / (test_ts[-1] - test_ts[-2])),
               two_mean_delta_a=float(2 * np.mean(rects[:, 1] - rects[:, 0])),
               mean_sigma_ratio=float(np.mean(rects[:, 3] / rects[:, 2])),
               bracket_example={str(t): [float(r_star(v_eq(t, rects[0, 0], rects[0, 2]))),
                                         float(r_star(v_eq(t, a, s))),
                                         float(r_star(v_eq(t, rects[0, 1], rects[0, 3])))] for t in test_ts})
    print(f"{name}: simultaneous coverage {out['coverage_simultaneous']:.2f}, width bounds hold in "
          f"{out['width_bounds_hold']:.2f}, a half-width {out['mean_a_halfwidth']:.4f}, a_lo > 0 in "
          f"{out['frac_a_lo_positive']:.2f}, last slope {out['slope_last']:.3f} vs 2*Delta_a {out['two_mean_delta_a']:.3f}")
    for t in test_ts:
        lo, mid, hi = out["bracket_example"][str(t)]
        print(f"   t={t:<5} log-width {out['mean_log_width'][str(t)]:.3f}   R* bracket (one replication) "
              f"[{lo:.4f}, {hi:.4f}] around {mid:.4f}")
    return out


def main():
    rng = np.random.default_rng(0)
    res = {"n_per_t": N_PER_T, "n_boot": N_BOOT, "n_rep": N_REP, "alpha": ALPHA}
    # the fog sanity check of the paper
    res["fog_check"] = run_regime("fog check (a=1, sigma=0.6)", 1.0, 0.6, [0.0, 0.1, 0.2],
                                  [0.5, 1.0, 2.0, 5.0, 10.0, 20.0], rng)
    # drift- and diffusion-dominated analogues, training ranges matched in equivalent
    # variance and test points at equal multiples of the training horizon in t
    res["drift"] = run_regime("drift-dominated (a=1, sigma=0.3)", 1.0, 0.3, [0.0, 0.25, 0.5],
                              [1.0, 2.0, 3.0], rng)
    t_d = float(np.log1p(2 * 1.0 * v_eq(0.5, 1.0, 0.3) / 0.6 ** 2 * 0.05) / (2 * 0.05))  # v-matched range
    res["diffusion"] = run_regime("diffusion-dominated (a=0.05, sigma=0.6)", 0.05, 0.6, [0.0, t_d / 2, t_d],
                                  [2 * t_d, 4 * t_d, 6 * t_d], rng)
    json.dump(res, open("certified_extrapolation_results.json", "w"), indent=1)
    print("written: certified_extrapolation_results.json")


if __name__ == "__main__":
    main()
