"""Kill-test with a gap that is known in advance and non-zero.

The symmetric kill-test cannot fail on the frozen-model side: with two balanced,
equal-covariance classes and isotropic noise, the Bayes boundary is the
perpendicular bisector at every severity, so a classifier frozen at low severity
is still Bayes-optimal at high severity and Delta(t) is zero by construction.

Here the class priors are unequal (pi0 = 0.8, pi1 = 0.2). Projected onto the
direction joining the class means (separation d), the Bayes rule thresholds at
    c*(t) = d/2 + sigma^2(t) * ln(pi0/pi1) / d,
which moves with t because sigma^2(t) = sigma0^2 + 2t grows. A classifier frozen
at low severity keeps the threshold it learned there, so at high severity it is
suboptimal by an amount we can compute in closed form. The test passes only if
the measured frozen-model gap matches that prediction, and the retrained oracle
matches R*(t).
"""
import json
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

SEED = 0
D = 3.0                      # distance between class means
SIGMA0 = 0.5
PI0, PI1 = 0.8, 0.2
TRAIN_TS = [0.0, 0.1, 0.2]
TEST_TS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
N_TRAIN = 12000              # per severity, split by the priors
N_EVAL = 200000
TOL = 0.006                  # pass/fail tolerance on each check


def sigma2(t):
    return SIGMA0**2 + 2 * t


def err_at_threshold(c, t):
    """Error of the rule 'predict 1 iff x > c' on the 1-D projection."""
    s = np.sqrt(sigma2(t))
    return PI0 * (1 - norm.cdf(c / s)) + PI1 * norm.cdf((c - D) / s)


def c_star(t):
    return D / 2 + sigma2(t) * np.log(PI0 / PI1) / D


def r_star(t):
    return err_at_threshold(c_star(t), t)


def sample(ts, n_total, rng):
    xs, ys = [], []
    for t in ts:
        n1 = int(round(n_total * PI1)); n0 = n_total - n1
        s = np.sqrt(sigma2(t))
        x0 = np.stack([rng.normal(0, s, n0), rng.normal(0, s, n0)], 1)
        x1 = np.stack([rng.normal(D, s, n1), rng.normal(0, s, n1)], 1)
        xs += [x0, x1]; ys += [np.zeros(n0), np.ones(n1)]
    return np.vstack(xs), np.concatenate(ys)


def fit(x, y):
    m = LogisticRegression(C=1e6, max_iter=2000)
    m.fit(x, y)
    return m


def threshold_of(m):
    """Where the fitted linear rule crosses zero along the first axis (at x2 = 0)."""
    w, b = m.coef_[0], m.intercept_[0]
    return -b / w[0]


def main():
    rng = np.random.default_rng(SEED)
    # The frozen model is trained on a pooled mixture of the training severities,
    # each contributing the same prior mix; its threshold is fixed from then on.
    xf, yf = sample(TRAIN_TS, N_TRAIN, rng)
    frozen = fit(xf, yf)
    c_frozen = threshold_of(frozen)
    with open("kill_test_asym_threshold.json", "w") as f:
        json.dump(dict(c_frozen=float(c_frozen), c_star_0=float(c_star(0))), f, indent=1)
    print(f"frozen threshold c_f = {c_frozen:.4f}   (c*(0) = {c_star(0):.4f}, "
          f"c* at mean training variance = "
          f"{D/2 + np.mean([sigma2(t) for t in TRAIN_TS])*np.log(PI0/PI1)/D:.4f})\n")

    hdr = (f"{'t':>5} | {'R*(t)':>7} | {'oracle':>7} {'slack':>8} | "
           f"{'pred. frozen':>12} {'meas. frozen':>12} {'pred-meas':>10} | {'Delta pred':>10}")
    print(hdr); print("-" * len(hdr))
    rows, ok = [], True
    for t in TEST_TS:
        xe, ye = sample([t], N_EVAL, rng)
        r_or = (fit(*sample([t], N_TRAIN * len(TRAIN_TS), rng)).predict(xe) != ye).mean()
        r_fr = (frozen.predict(xe) != ye).mean()
        pred_fr = err_at_threshold(c_frozen, t)       # closed-form error at the frozen threshold
        rs = r_star(t)
        slack = r_or - rs
        miss = pred_fr - r_fr
        ok &= abs(slack) < TOL and abs(miss) < TOL
        flag = "  (train)" if t in TRAIN_TS else ""
        print(f"{t:>5.1f} | {rs:>7.4f} | {r_or:>7.4f} {slack:>+8.4f} | "
              f"{pred_fr:>12.4f} {r_fr:>12.4f} {miss:>+10.4f} | {pred_fr-rs:>10.4f}{flag}")
        rows.append(dict(t=t, r_star=rs, r_oracle=r_or, oracle_slack=slack,
                         frozen_pred=pred_fr, frozen_meas=r_fr, frozen_pred_minus_meas=miss,
                         delta_pred=pred_fr - rs, delta_meas=r_fr - rs,
                         is_train_severity=t in TRAIN_TS))
    with open("kill_test_asym_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    big = max(r["delta_pred"] for r in rows)
    print(f"\nlargest predicted avoidable gap: {big:.4f}  (non-zero by design)")
    print("PASSED" if ok else "FAILED", "-- tolerance", TOL)


if __name__ == "__main__":
    main()
