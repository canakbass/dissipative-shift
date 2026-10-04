"""Certified extrapolation on CIFAR-10 images, from unlabelled corrupted images.

For each fog regime of the paper, 50000 training images are split into disjoint thirds,
one per training severity (as the frozen model's budget is split), and corrupted.
Labels are never used. The fog noise is white across pixels, so every covariance
between two different pixels or channels decays exactly as exp(-2at); the per-pixel
variance then gives sigma. A bootstrap over images gives a rectangle for (a, sigma),
hence an interval for the equivalent noise variance v(t) at the held-out severities.

By Proposition 1 and Corollary 1, R*_fog(t) <= R*_noise(t_n) for any noise severity
t_n >= v_hi(t)/2, and R*_noise(t_n) is at most the true error of any classifier trained
at t_n, in particular of the noise oracle, whose test error bounds its true error up to
binomial sampling. Needs data/cifar-10-batches-py. CPU, a few minutes.
Writes certified_extrapolation_cifar_results.json.
"""
import json
import os

import numpy as np
from scipy.stats import norm
from torchvision.datasets import CIFAR10

ROOT = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 400
ALPHA = 0.05
REGIMES = {   # name: (a, sigma, training severities, held-out severities)
    "fog_main": (1.0, 0.2, [0.0, 0.1, 0.25], [0.5, 1.0, 2.0, 4.0]),
    "fog_drift": (1.0, 0.05, [0.0, 0.25, 0.5], [1.0, 2.0, 3.0]),
    "fog_diffuse": (0.05, 0.2, [0.0, 0.5, 1.0], [2.0, 4.0, 8.0]),
}
NOISE_RUNS = {1: "kaggle_cifar10_c_bnadapt/output_seed1/cifar10_c_results_bnadapt_seed1.jsonl",
              2: "kaggle_cifar10_c_bnadapt/output_seed2/cifar10_c_results_bnadapt_seed2.jsonl",
              3: "kaggle_cifar10_c_bnadapt_b/output_seed3/cifar10_c_results_bnadapt_seed3.jsonl"}
N_TEST = 10000


def v_eq(t, a, s):
    return s ** 2 * np.expm1(2 * a * t) / (2 * a)


def fog(x, t, a, s, rng):
    decay = np.exp(-a * t)
    out = x * decay + (1 - decay)                               # haze point A = 1
    var = s ** 2 / (2 * a) * (1 - decay ** 2)
    return out + np.sqrt(var) * rng.standard_normal(x.shape).astype(np.float32) if var > 0 else out


def per_image_stats(x):
    """Per-image contributions to (off-diagonal covariance, per-pixel variance), centred at
    the subset mean; the bootstrap averages these. Off-diagonal: horizontal and vertical
    neighbours and the three channel pairs at the same pixel."""
    z = x - x.mean(0, keepdims=True)
    off = np.stack([(z[:, :, :, :-1] * z[:, :, :, 1:]).mean((1, 2, 3)),
                    (z[:, :, :-1, :] * z[:, :, 1:, :]).mean((1, 2, 3)),
                    (z[:, 0] * z[:, 1]).mean((1, 2)), (z[:, 0] * z[:, 2]).mean((1, 2)),
                    (z[:, 1] * z[:, 2]).mean((1, 2))], 1).mean(1)
    var = (z ** 2).mean((1, 2, 3))
    return off.astype(np.float64), var.astype(np.float64)


def estimate(offs, vars_, ts):
    ts = np.asarray(ts)
    a = -np.polyfit(ts, np.log(offs), 1)[0] / 2
    g = -np.expm1(-2 * a * ts[1:]) / (2 * a)
    resid = vars_[1:] - np.exp(-2 * a * ts[1:]) * vars_[0]
    return a, np.sqrt(max(float(np.dot(g, resid) / np.dot(g, g)), 1e-12))


def noise_oracle_bound(tn_needed):
    """Upper bound on R*_noise at the smallest grid severity >= tn_needed: the best seed's
    oracle test error plus a one-sided binomial margin (Bonferroni over three seeds)."""
    rows = {}
    for s, p in NOISE_RUNS.items():
        for l in open(os.path.join(ROOT, p)):
            r = json.loads(l)
            if r["family"] == "gauss_noise":
                rows.setdefault(r["t"], {})[s] = r["err_oracle"]
    grid = sorted(rows)
    above = [t for t in grid if t >= tn_needed]
    if not above:
        return None, None
    tn = above[0]
    z = norm.ppf(1 - ALPHA / 3)
    ub = min(e + z * np.sqrt(e * (1 - e) / N_TEST) for e in rows[tn].values())
    return tn, float(ub)


def main():
    rng = np.random.default_rng(0)
    ds = CIFAR10(root=os.path.join(ROOT, "data"), train=True, download=False)
    x_all = np.transpose(ds.data.astype(np.float32) / 255.0, (0, 3, 1, 2))
    x_all = x_all[rng.permutation(len(x_all))]
    res = {"n_boot": N_BOOT, "alpha": ALPHA}
    fog_oracle = {}
    for s, p in NOISE_RUNS.items():
        for l in open(os.path.join(ROOT, p)):
            r = json.loads(l)
            if r["family"] == "fog_beer_lambert":
                fog_oracle.setdefault(r["t"], []).append(r["err_oracle"])
    for name, (a, sig, train_ts, test_ts) in REGIMES.items():
        n = len(x_all) // len(train_ts)
        stats = [per_image_stats(fog(x_all[i * n:(i + 1) * n], t, a, sig, rng)) for i, t in enumerate(train_ts)]
        a_hat, s_hat = estimate([o.mean() for o, _ in stats], [v.mean() for _, v in stats], train_ts)
        boot = []
        for _ in range(N_BOOT):
            idx = [rng.integers(0, len(o), len(o)) for o, _ in stats]
            boot.append(estimate([o[i].mean() for (o, _), i in zip(stats, idx)],
                                 [v[i].mean() for (_, v), i in zip(stats, idx)], train_ts))
        boot = np.array(boot)
        q = ALPHA / 4
        (a_lo, a_hi), (s_lo, s_hi) = np.quantile(boot[:, 0], [q, 1 - q]), np.quantile(boot[:, 1], [q, 1 - q])
        out = dict(a=a, sigma=sig, a_hat=float(a_hat), sigma_hat=float(s_hat), a_rect=[float(a_lo), float(a_hi)],
                   sigma_rect=[float(s_lo), float(s_hi)], points={})
        print(f"{name}: a={a} est {a_hat:.4f} [{a_lo:.4f}, {a_hi:.4f}]   sigma={sig} est {s_hat:.4f} [{s_lo:.4f}, {s_hi:.4f}]")
        for t in test_ts:
            v_true, v_lo, v_hi = v_eq(t, a, sig), v_eq(t, a_lo, s_lo), v_eq(t, a_hi, s_hi)
            tn, ub = noise_oracle_bound(v_hi / 2)
            pt = dict(v_true=float(v_true), v_lo=float(v_lo), v_hi=float(v_hi), covered=bool(v_lo <= v_true <= v_hi),
                      log_width=float(np.log(v_hi / v_lo)), noise_grid_point=tn, certified_upper=ub)
            if name == "fog_main" and t in fog_oracle:
                e = float(np.mean(fog_oracle[t]))
                pt["fog_oracle"] = e
                if ub is not None:
                    pt["certified_slack_lower"] = max(0.0, e - norm.ppf(1 - ALPHA) * np.sqrt(e * (1 - e) / N_TEST) - ub)
            out["points"][str(t)] = pt
            print(f"   t={t:<4} v true {v_true:.4f} in [{v_lo:.4f}, {v_hi:.4f}] (log-width {pt['log_width']:.3f})  "
                  f"R* <= {ub if ub is None else round(ub, 4)} (noise oracle at t={tn})"
                  + (f"  fog oracle {pt['fog_oracle']:.4f}" if "fog_oracle" in pt else ""))
        res[name] = out
    json.dump(res, open(os.path.join(ROOT, "certified_extrapolation_cifar_results.json"), "w"), indent=1)
    print("written: certified_extrapolation_cifar_results.json")


if __name__ == "__main__":
    main()
