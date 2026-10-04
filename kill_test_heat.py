"""
Kill-test: the stochastic heat equation, the version of blur that genuinely
destroys information.

dX = -a(omega)*X dt + sigma dW, a separate OU process for each Fourier mode omega
(rate = omega^2, the standard heat-equation eigenvalue). Being the transition
kernel of a time-homogeneous diffusion, this is an exact Markov semigroup by
Chapman-Kolmogorov; no separate proof is needed, since it is the OU argument of
kill_test_fog.py applied mode by mode.

Simplification: instead of a real FFT over a pixel grid, a toy classification
problem with d=4 frequency modes, each with its own decay rate a_k (higher
frequency decays faster, as in real blur). The covariance stays diagonal, so the
Mahalanobis distance has a closed form.
"""
import numpy as np
from scipy.stats import norm

RNG_SEED = 0
D = 4
A_RATES = np.array([0.5, 1.0, 2.0, 4.0])  # decay rate per mode (higher frequency decays faster)
MU1 = np.array([1.5, 1.2, 0.9, 0.6])
MU2 = np.array([-1.5, -1.2, -0.9, -0.6])
SIGMA0 = 0.5
DIFF_SIGMA = np.array([0.6, 0.6, 0.6, 0.6])  # diffusion coefficient per mode
TRAIN_TS = [0.0, 0.1, 0.2]
TEST_TS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SELF_TEST_TOL = 0.02


def mean_t(mu, t):
    return mu * np.exp(-A_RATES * t)


def var_t(t):
    decay2 = np.exp(-2 * A_RATES * t)
    return SIGMA0**2 * decay2 + (DIFF_SIGMA**2 / (2 * A_RATES)) * (1 - decay2)


def r_star_closed(t):
    m1, m2 = mean_t(MU1, t), mean_t(MU2, t)
    v = var_t(t)
    delta = np.sqrt(np.sum((m1 - m2) ** 2 / v))
    return norm.cdf(-delta / 2)


def sample(t, n, y, rng):
    mu0 = MU1 if y == 0 else MU2
    mean = mean_t(mu0, t)
    std = np.sqrt(var_t(t))
    return mean + std * rng.standard_normal((n, D))


def bayes_rule(x, t):
    m1, m2 = mean_t(MU1, t), mean_t(MU2, t)
    v = var_t(t)
    # Mahalanobis distance with diagonal covariance: return 1 if closer to class 2
    d1 = np.sum((x - m1) ** 2 / v, axis=1)
    d2 = np.sum((x - m2) ** 2 / v, axis=1)
    return (d2 < d1).astype(int)


def self_test(rng):
    print("=== Self-test: Bayes-optimal rule vs closed-form R*(t) (stochastic heat equation) ===")
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
    print(f"  expected R*=0.5 as t->inf (every mode decays to zero): R*(t=20)={r_star_closed(20):.4f}")
    print("  -> self-test passed.\n")


def estimate_generator(rng):
    print("=== Generator estimate: fit a_k and sigma_k for each mode from low t ===")
    # mode by mode: log(mean_k(t)) = log(mu_k) - a_k*t, assuming mu_k>0 so the sign is fixed
    a_hat = np.zeros(D)
    for k in range(D):
        obs = [mean_t(MU1, t)[k] for t in TRAIN_TS]
        slope, intercept = np.polyfit(TRAIN_TS, np.log(np.abs(obs)), 1)
        a_hat[k] = -slope
    print(f"  true a: {A_RATES}")
    print(f"  estimate a_hat: {np.round(a_hat, 4)}")

    var_obs = np.zeros((len(TRAIN_TS), D))
    for i, t in enumerate(TRAIN_TS):
        x0 = sample(t, 20000, 0, rng) - mean_t(MU1, t)
        var_obs[i] = x0.var(axis=0)
    sigma0_sq_hat = np.zeros(D)
    c_hat = np.zeros(D)
    for k in range(D):
        decay2 = np.exp(-2 * a_hat[k] * np.array(TRAIN_TS))
        Xd = np.stack([decay2, 1 - decay2], axis=1)
        coef, *_ = np.linalg.lstsq(Xd, var_obs[:, k], rcond=None)
        sigma0_sq_hat[k], c_hat[k] = coef
    print(f"  true sigma0^2: {SIGMA0**2:.4f} (same for every mode)")
    print(f"  estimate sigma0^2_hat: {np.round(sigma0_sq_hat, 4)}\n")
    return a_hat, sigma0_sq_hat, c_hat


def run_eval(a_hat, sigma0_sq_hat, c_hat):
    print("=== Extrapolation accuracy ===")
    header = f"{'t':>6} | {'R*(t) closed':>13} | {'R*_hat(t) extrap':>17} | {'extrap. err':>12}"
    print(header)
    print("-" * len(header))
    rows = []
    for t in TEST_TS:
        r_true = r_star_closed(t)
        decay2_hat = np.exp(-2 * a_hat * t)
        m1_hat = MU1 * np.exp(-a_hat * t)
        m2_hat = MU2 * np.exp(-a_hat * t)
        var_hat = sigma0_sq_hat * decay2_hat + c_hat * (1 - decay2_hat)
        delta_hat = np.sqrt(np.sum((m1_hat - m2_hat) ** 2 / var_hat))
        r_hat = norm.cdf(-delta_hat / 2)
        err = r_hat - r_true
        print(f"{t:>6.2f} | {r_true:>13.4f} | {r_hat:>17.4f} | {err:>+12.4f}")
        rows.append(dict(t=t, r_star_closed=r_true, r_star_extrapolated=r_hat, extrapolation_error=err))
    return rows


def main():
    rng = np.random.default_rng(RNG_SEED)
    self_test(rng)
    a_hat, sigma0_sq_hat, c_hat = estimate_generator(rng)
    rows = run_eval(a_hat, sigma0_sq_hat, c_hat)
    import json
    with open("kill_test_heat_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\nresults written: kill_test_heat_results.jsonl")


if __name__ == "__main__":
    main()
