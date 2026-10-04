"""
Stage 1 kill-test: checks Bayes-risk monotonicity under a dissipative
(semigroup) distribution shift, and the extrapolation-horizon hypothesis,
against a closed form and without a GPU.

Setup: 2-D, K=2 classes, equal-covariance isotropic Gaussian mixture under the
heat semigroup S(t): sigma^2(t) = sigma0^2 + 2t. The mixture stays Gaussian
under the heat flow, so R*(t) has a closed form.

Four checks run in order; each is meaningless unless the previous one passed:
  1) Self-test        : Delta(t) ~= 0 for the Bayes-optimal rule itself
                        (is the R*(t) formula right?)
  2) Generator estimate: sigma^2(t)=a+b*t fitted by linear regression on low t,
                        then extrapolated to large t_test (extrapolation fidelity)
  3) Oracle slack     : a model retrained at each severity, compared against the
                        known R*(t) -- the check that cannot be run on real images
  4) Frozen-model waste: logistic regression frozen at low t, evaluated at large
                       Delta(t) = R_model(t) - R*(t)
"""
import json

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

RNG_SEED = 0
MU1, MU2 = np.array([0.0, 0.0]), np.array([3.0, 0.0])
SIGMA0 = 0.5
TRAIN_TS = [0.0, 0.1, 0.2]
TEST_TS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_TRAIN_PER_T_PER_CLASS = 2000
N_EVAL_PER_CLASS = 5000
SELF_TEST_TOL = 0.02


def sigma2_true(t):
    return SIGMA0**2 + 2 * t


def r_star_closed(t, sigma2=None):
    d = np.linalg.norm(MU2 - MU1)
    s2 = sigma2_true(t) if sigma2 is None else sigma2
    return norm.cdf(-d / (2 * np.sqrt(s2)))


def sample(t, n, y, rng):
    mu = MU1 if y == 0 else MU2
    return mu + np.sqrt(sigma2_true(t)) * rng.standard_normal((n, 2))


def make_dataset(ts, n_per_class, rng):
    xs, ys = [], []
    for t in ts:
        for y in (0, 1):
            xs.append(sample(t, n_per_class, y, rng))
            ys.append(np.full(n_per_class, y))
    return np.vstack(xs), np.concatenate(ys)


def bayes_rule(x):
    return (np.linalg.norm(x - MU2, axis=1) < np.linalg.norm(x - MU1, axis=1)).astype(int)


def self_test(rng):
    print("=== 1) Self-test: Bayes-optimal rule vs closed-form R*(t) ===")
    for t in TEST_TS:
        x, y = make_dataset([t], N_EVAL_PER_CLASS, rng)
        err = (bayes_rule(x) != y).mean()
        rstar = r_star_closed(t)
        delta = err - rstar
        status = "OK" if abs(delta) < SELF_TEST_TOL else "FAIL"
        print(f"  t={t:>5.2f}  empirical={err:.4f}  R*(t)={rstar:.4f}  Delta={delta:+.4f}  [{status}]")
        assert abs(delta) < SELF_TEST_TOL, (
            f"SELF-TEST FAILED (t={t}): check the R*(t) formula, Delta={delta:.4f}"
        )
    print("  -> self-test passed: the closed form for R*(t) is correct.\n")


def estimate_generator(rng):
    print("=== 2) Generator estimate: sigma^2(t) = a + b*t (linear regression on low t) ===")
    est_var = []
    for t in TRAIN_TS:
        x0 = sample(t, N_TRAIN_PER_T_PER_CLASS, 0, rng) - MU1
        x1 = sample(t, N_TRAIN_PER_T_PER_CLASS, 1, rng) - MU2
        v = np.concatenate([x0.ravel(), x1.ravel()]).var()
        est_var.append(v)
    b_hat, a_hat = np.polyfit(TRAIN_TS, est_var, 1)
    print(f"  true: a=sigma0^2={SIGMA0**2:.4f}, b=2.0")
    print(f"  estimate: a_hat={a_hat:.4f}, b_hat={b_hat:.4f}\n")
    return a_hat, b_hat


def train_frozen_model(rng):
    x, y = make_dataset(TRAIN_TS, N_TRAIN_PER_T_PER_CLASS, rng)
    model = LogisticRegression()
    model.fit(x, y)
    return model


def train_oracle(t, rng):
    """Oracle proxy: trained from scratch at a single severity, on the same total
    budget as the frozen model. This is the exact counterpart of the oracle in the
    paper's CIFAR protocol. Because R*(t) is known in closed form here, we can
    measure the oracle's slack directly -- the one thing real images do not allow."""
    n_total = len(TRAIN_TS) * N_TRAIN_PER_T_PER_CLASS   # budget matching
    x, y = make_dataset([t], n_total, rng)
    model = LogisticRegression()
    model.fit(x, y)
    return model


def run_eval(model, a_hat, b_hat, rng):
    print("=== 3) Extrapolation fidelity + oracle slack + frozen-model waste ===")
    rows = []
    header = (f"{'t':>6} | {'R*(t) closed':>13} | {'R*_hat extrap':>14} | {'extrap. err':>12} | "
              f"{'R_oracle(t)':>12} | {'slack':>10} | {'R_model(t)':>11} | {'Delta(t)':>9}")
    print(header)
    print("-" * len(header))
    for t in TEST_TS:
        x, y = make_dataset([t], N_EVAL_PER_CLASS, rng)
        r_model = (model.predict(x) != y).mean()
        oracle = train_oracle(t, rng)
        r_oracle = (oracle.predict(x) != y).mean()
        r_true = r_star_closed(t)
        r_hat = r_star_closed(t, sigma2=a_hat + b_hat * t)
        extr_err = r_hat - r_true
        slack = r_oracle - r_true          # how far the oracle proxy sits from the true R*
        delta = r_model - r_true
        train_regime = "  (train)" if t in TRAIN_TS else ""
        print(
            f"{t:>6.2f} | {r_true:>13.4f} | {r_hat:>14.4f} | {extr_err:>+12.4f} | "
            f"{r_oracle:>12.4f} | {slack:>+10.4f} | {r_model:>11.4f} | {delta:>+9.4f}{train_regime}"
        )
        rows.append(
            dict(t=t, r_star_closed=r_true, r_star_extrapolated=r_hat,
                 extrapolation_error=extr_err, r_oracle=r_oracle,
                 oracle_slack=slack, r_model=r_model, delta=delta,
                 is_train_severity=t in TRAIN_TS)
        )
    print()
    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"results written: {path}")


def try_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(no matplotlib, skipping the plot)")
        return
    ts = [r["t"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ts, [r["r_star_closed"] for r in rows], "k-", label="R*(t) closed form")
    ax[0].plot(ts, [r["r_star_extrapolated"] for r in rows], "b--", label="R*_hat(t) extrapolated")
    ax[0].plot(ts, [r["r_model"] for r in rows], "r.-", label="R_model(t)")
    ax[0].axvline(max(TRAIN_TS), color="gray", linestyle=":", label="training horizon")
    ax[0].set_xlabel("t (severity)")
    ax[0].set_ylabel("error / risk")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Bayes risk and model error")

    ax[1].plot(ts, [r["delta"] for r in rows], "r.-", label="Delta(t) = R_model - R*")
    ax[1].axhline(0, color="k", linewidth=0.5)
    ax[1].axvline(max(TRAIN_TS), color="gray", linestyle=":")
    ax[1].set_xlabel("t (severity)")
    ax[1].set_ylabel("Delta(t)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Frozen-model waste")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot written: {path}")


def main():
    rng = np.random.default_rng(RNG_SEED)
    self_test(rng)
    a_hat, b_hat = estimate_generator(rng)
    model = train_frozen_model(rng)
    rows = run_eval(model, a_hat, b_hat, rng)
    write_jsonl(rows, "kill_test_results.jsonl")
    try_plot(rows, "kill_test_delta.png")


if __name__ == "__main__":
    main()
