"""
Kill-test addendum: the corrected Beer-Lambert fog model.

The problem: the earlier fog model
X(t) = X*e^-t + A(1-e^-t) was a deterministic affine map with no clipping and no
noise, invertible for every finite t via X = (X(t)-A(1-e^-t))/e^-t. A bijection
loses no information, so R*(t) = R*(0) should have held. The Blackwell argument
needs a garbling -- a stochastic kernel that actually destroys information --
not a bijection.

The fix: an Ornstein-Uhlenbeck process, dX = -a(X-A)dt + sigma*dW. This combines
the attenuation (the mean converges exponentially to A, the same term the old
Beer-Lambert model had) and the scattering noise (the variance grows) in one
process. The OU transition kernel is an exact Markov semigroup by
Chapman-Kolmogorov, and it does destroy information: as t->inf every class
converges to the same stationary Gaussian around A and the Bayes error goes to
chance. That is the property we need.

X(t) | X(0) ~ N( A + (X(0)-A)e^{-at}, sigma^2/(2a) * (1-e^{-2at}) )

We set a=1; t can be read as "a * physical time".
"""
import json

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

RNG_SEED = 0
MU1, MU2 = np.array([0.0, 0.0]), np.array([3.0, 0.0])
SIGMA0 = 0.5
A = np.array([1.5, 0.0])  # the common point fog pulls toward (midway between classes)
N_GEN_EST = 20000   # samples used for the generator estimate
FOG_SIGMA = 0.6           # OU diffusion coefficient (stationary std = sigma/sqrt(2))
TRAIN_TS = [0.0, 0.1, 0.2]
N_TRAIN_PER_T_PER_CLASS = 4000
N_EVAL_PER_CLASS = 20000
TEST_TS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SELF_TEST_TOL = 0.02


def d_of_t(t):
    return np.linalg.norm(MU2 - MU1) * np.exp(-t)


def var_of_t(t):
    decay2 = np.exp(-2 * t)
    return SIGMA0**2 * decay2 + (FOG_SIGMA**2 / 2) * (1 - decay2)


def r_star_closed(t, var=None):
    v = var_of_t(t) if var is None else var
    return norm.cdf(-d_of_t(t) / (2 * np.sqrt(v)))


def sample(t, n, y, rng):
    mu0 = MU1 if y == 0 else MU2
    mean = A + (mu0 - A) * np.exp(-t)
    std = np.sqrt(var_of_t(t))
    return mean + std * rng.standard_normal((n, 2))


def bayes_rule(x, t):
    mu1_t = A + (MU1 - A) * np.exp(-t)
    mu2_t = A + (MU2 - A) * np.exp(-t)
    return (np.linalg.norm(x - mu2_t, axis=1) < np.linalg.norm(x - mu1_t, axis=1)).astype(int)


def self_test(rng):
    print("=== Self-test: Bayes-optimal rule vs closed-form R*(t) (OU fog) ===")
    for t in TEST_TS:
        x0 = sample(t, 5000, 0, rng)
        x1 = sample(t, 5000, 1, rng)
        x = np.vstack([x0, x1])
        y = np.array([0] * 5000 + [1] * 5000)
        err = (bayes_rule(x, t) != y).mean()
        rstar = r_star_closed(t)
        delta = err - rstar
        status = "OK" if abs(delta) < SELF_TEST_TOL else "FAIL"
        print(f"  t={t:>5.2f}  empirical={err:.4f}  R*(t)={rstar:.4f}  Delta={delta:+.4f}  [{status}]")
        assert abs(delta) < SELF_TEST_TOL, f"SELF-TEST FAILED (t={t})"
    print(f"  expected R*=0.5 as t->inf, R*(t=20)={r_star_closed(20):.4f}  (stationary-state check)")
    print("  -> self-test passed.\n")


def estimate_generator(rng):
    print("=== Generator estimate: fit d(t) and var(t) parameters from low t ===")
    # d(t) = d0 * exp(-a*t) -> log(d(t)) = log(d0) - a*t  (linear regression)
    # The class means are estimated from samples. An earlier version
    # computed d(t) from the analytic formula A + (MU-A)e^{-t}, which fits the closed
    # form to itself: a_hat came out exact because it could not come out otherwise.
    # That was a tautology, not an estimate.
    d_obs = []
    for t in TRAIN_TS:
        s1 = sample(t, N_GEN_EST, 0, rng).mean(axis=0)
        s2 = sample(t, N_GEN_EST, 1, rng).mean(axis=0)
        d_obs.append(np.linalg.norm(s2 - s1))
    slope, intercept = np.polyfit(TRAIN_TS, np.log(d_obs), 1)
    a_hat, d0_hat = -slope, np.exp(intercept)
    print(f"  true: a=1.0, d0={np.linalg.norm(MU2-MU1):.4f}")
    print(f"  estimate: a_hat={a_hat:.4f}, d0_hat={d0_hat:.4f}")

    # for the variance: var(t) = sigma0^2 * e^{-2at} + (fog_sigma^2/2)(1-e^{-2at})
    # Rather than a nonlinear fit, note there are only two unknowns
    # (sigma0^2 and fog_sigma^2/2), so two low-t points suffice.
    var_obs = []
    for t in TRAIN_TS:
        xs = sample(t, N_GEN_EST, 0, rng)
        var_obs.append(xs.var(axis=0).mean())   # variance about the sample mean
    decay2 = np.exp(-2 * a_hat * np.array(TRAIN_TS))
    # var = sigma0^2 * decay2 + c*(1-decay2)  =>  linear regression (decay2, 1-decay2) -> var
    Xd = np.stack([decay2, 1 - decay2], axis=1)
    coef, *_ = np.linalg.lstsq(Xd, var_obs, rcond=None)
    sigma0_sq_hat, c_hat = coef
    fog_sigma_hat = np.sqrt(max(c_hat, 1e-8) * 2)
    print(f"  true: sigma0^2={SIGMA0**2:.4f}, fog_sigma={FOG_SIGMA:.4f}")
    print(f"  estimate: sigma0^2_hat={sigma0_sq_hat:.4f}, fog_sigma_hat={fog_sigma_hat:.4f}\n")
    return a_hat, d0_hat, sigma0_sq_hat, c_hat


def make_dataset(ts, n_per_class, rng):
    xs, ys = [], []
    for t in ts:
        for y in (0, 1):
            xs.append(sample(t, n_per_class, y, rng))
            ys.append(np.full(n_per_class, y))
    return np.vstack(xs), np.concatenate(ys)


def train_frozen_model(rng):
    x, y = make_dataset(TRAIN_TS, N_TRAIN_PER_T_PER_CLASS, rng)
    m = LogisticRegression(); m.fit(x, y); return m


def train_oracle(t, rng):
    """Same total budget as the frozen model, concentrated at one severity."""
    n_total = len(TRAIN_TS) * N_TRAIN_PER_T_PER_CLASS
    x, y = make_dataset([t], n_total, rng)
    m = LogisticRegression(); m.fit(x, y); return m


def run_eval(a_hat, d0_hat, sigma0_sq_hat, c_hat, rng):
    print("=== Extrapolation + oracle slack + frozen-model waste ===")
    frozen = train_frozen_model(rng)
    header = (f"{'t':>6} | {'R*(t) closed':>13} | {'R*_hat extrap':>14} | {'extrap. err':>12} | "
              f"{'R_oracle':>9} | {'slack':>10} | {'R_model':>9} | {'Delta':>9}")
    print(header)
    print("-" * len(header))
    rows = []
    for t in TEST_TS:
        r_true = r_star_closed(t)
        decay2_hat = np.exp(-2 * a_hat * t)
        d_hat = d0_hat * np.exp(-a_hat * t)
        var_hat = sigma0_sq_hat * decay2_hat + c_hat * (1 - decay2_hat)
        r_hat = norm.cdf(-d_hat / (2 * np.sqrt(var_hat)))
        err = r_hat - r_true

        xe, ye = make_dataset([t], N_EVAL_PER_CLASS, rng)
        r_model = (frozen.predict(xe) != ye).mean()
        oracle = train_oracle(t, rng)
        r_oracle = (oracle.predict(xe) != ye).mean()
        slack = r_oracle - r_true
        delta = r_model - r_true
        flag = "  (train)" if t in TRAIN_TS else ""
        print(f"{t:>6.2f} | {r_true:>13.4f} | {r_hat:>14.4f} | {err:>+12.4f} | "
              f"{r_oracle:>9.4f} | {slack:>+10.4f} | {r_model:>9.4f} | {delta:>+9.4f}{flag}")
        rows.append(dict(t=t, r_star_closed=r_true, r_star_extrapolated=r_hat,
                         extrapolation_error=err, r_oracle=r_oracle, oracle_slack=slack,
                         r_model=r_model, delta=delta, is_train_severity=t in TRAIN_TS))
    return rows


def main():
    rng = np.random.default_rng(RNG_SEED)
    self_test(rng)
    a_hat, d0_hat, sigma0_sq_hat, c_hat = estimate_generator(rng)
    with open("kill_test_fog_generator.json", "w") as f:
        json.dump(dict(a_hat=float(a_hat), fog_sigma_hat=float(np.sqrt(max(c_hat, 1e-8) * 2)),
                       n_samples=N_GEN_EST, self_test_tol=SELF_TEST_TOL), f, indent=1)
    rows = run_eval(a_hat, d0_hat, sigma0_sq_hat, c_hat, rng)
    with open("kill_test_fog_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\nresults written: kill_test_fog_results.jsonl")


if __name__ == "__main__":
    main()
