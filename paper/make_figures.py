"""Paper figures. Okabe-Ito colorblind-safe palette, no personal paths in output.
Uses the corrected (v2fix) corruption code results: unclipped Gaussian noise,
Ornstein-Uhlenbeck fog (genuinely information-losing, not the invertible affine
map of the first draft)."""
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito palette
C_ORACLE = "#000000"
C_FROZEN = "#D55E00"
C_BNADAPT = "#56B4E9"
C_TENT = "#0072B2"
C_EATA = "#009E73"
C_DANN = "#CC79A7"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "figure.dpi": 300,
})

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs(f"{BASE}/paper/figures", exist_ok=True)

# ---------- Figure 1a: kill-test (Gaussian noise / heat semigroup) ----------
kt = [json.loads(l) for l in open(f"{BASE}/kill_test_results.jsonl")]
ts = [r["t"] for r in kt]
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
ax = axes[0]
ax.plot(ts, [r["r_star_closed"] for r in kt], color=C_ORACLE, label=r"$R^*(t)$ (closed form)")
ax.plot(ts, [r["r_star_extrapolated"] for r in kt], color=C_TENT, linestyle="--", label=r"$\hat{R}^*(t)$ (extrapolated)")
ax.plot(ts, [r["r_model"] for r in kt], color=C_FROZEN, marker="o", markersize=2.5, linestyle=":", label=r"frozen classifier")
ax.axvline(0.2, color="gray", linewidth=0.6, linestyle=":")
ax.set_xlabel(r"severity $t$")
ax.set_ylabel("error")
ax.legend(frameon=False, loc="lower right", fontsize=6.5)
ax.set_title("(a) heat semigroup (noise)")

ax = axes[1]
ax.axhline(0, color="gray", linewidth=0.6)
ax.plot(ts, [r["delta"] for r in kt], color=C_FROZEN, marker="o", markersize=2.5)
ax.axvline(0.2, color="gray", linewidth=0.6, linestyle=":")
ax.set_xlabel(r"severity $t$")
ax.set_ylabel(r"$\Delta(t)$")
ax.set_title("(b) waste of the frozen classifier")
fig.tight_layout()
fig.savefig(f"{BASE}/paper/figures/fig_killtest.pdf")
print("fig_killtest.pdf written")

# ---------- Figure 1b: kill-test fog (OU semigroup) ----------
kf = [json.loads(l) for l in open(f"{BASE}/kill_test_fog_results.jsonl")]
tsf = [r["t"] for r in kf]
fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.3))
ax.plot(tsf, [r["r_star_closed"] for r in kf], color=C_ORACLE, label=r"$R^*(t)$ closed form")
ax.plot(tsf, [r["r_star_extrapolated"] for r in kf], color=C_TENT, linestyle="--", label=r"$\hat{R}^*(t)$ extrapolated")
ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0.2, color="gray", linewidth=0.6, linestyle=":")
ax.set_xlabel(r"severity $t$")
ax.set_ylabel("Bayes risk")
ax.set_title("Ornstein-Uhlenbeck fog")
ax.legend(frameon=False, loc="lower right", fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{BASE}/paper/figures/fig_killtest_fog.pdf")
print("fig_killtest_fog.pdf written")

# ---------- Figure 2: CIFAR-10-C main result (corrected corruptions) ----------
# Oracle, frozen, bnadapt, tent and eata all come from the bnadapt runs,
# which compute all five from a single frozen model. The figure previously took
# oracle/frozen/tent/eata from the v2fix run and bnadapt from a different one; that
# mixture inflated the BN-TENT gap to as much as 0.0166, against at most 0.0049
# within a single run, and contradicted the caption's "within half a point".
# make_tables.py already had this fix; the figure had been left behind.
bn_dirs = {1: "kaggle_cifar10_c_bnadapt/output_seed1",
           2: "kaggle_cifar10_c_bnadapt/output_seed2",
           3: "kaggle_cifar10_c_bnadapt_b/output_seed3"}
bh_dirs = {1: "kaggle_cifar10_c_bnadapt_blurheat/output_seed1",
           2: "kaggle_cifar10_c_bnadapt_blurheat/output_seed2",
           3: "kaggle_blurheat_bnadapt_s3/output_seed3"}
bn_rows = []
for seed, d in bn_dirs.items():
    bn_rows += [json.loads(l) for l in open(f"{BASE}/{d}/cifar10_c_results_bnadapt_seed{seed}.jsonl")]
for seed, d in bh_dirs.items():
    p = f"{BASE}/{d}/cifar10_c_results_bnadapt_blurheat_seed{seed}.jsonl"
    if os.path.exists(p):
        bn_rows += [json.loads(l) for l in open(p)]
rows = bn_rows  # oracle/frozen/tent/eata are read from these same rows

# DANN has its own run, and that run does not retrain an oracle, so the DANN curve
# necessarily comes from a separate run. The caption says so.
dann_rows = []
for seed in [1, 2, 3]:
    dann_rows += [json.loads(l) for l in open(f"{BASE}/kaggle_cifar10_c_dann/output_seed{seed}_v2fix/cifar10_c_results_dann_seed{seed}_v2fix.jsonl")]

bh_dann = {1: "kaggle_cifar10_c_dann_blurheat/output_seed1",
           2: "kaggle_dann_blurheat_s2/output_seed2",
           3: "kaggle_dann_blurheat_s3/output_seed3"}
for seed, d in bh_dann.items():
    p = f"{BASE}/{d}/cifar10_c_results_dann_blurheat_seed{seed}.jsonl"
    if os.path.exists(p):
        dann_rows += [json.loads(l) for l in open(p)]
    else:
        print(f"WARNING (figure): DANN blur_heat seed{seed} missing")

by_key = defaultdict(lambda: defaultdict(list))
for r in rows:
    k = (r["family"], r["t"])
    by_key[k]["oracle"].append(r["err_oracle"])
    by_key[k]["frozen"].append(r["err_frozen"])
    by_key[k]["tent"].append(r["err_tent"])
    by_key[k]["eata"].append(r["err_eata"])
for r in dann_rows:
    k = (r["family"], r["t"])
    by_key[k]["dann"].append(r["err_dann"])
for r in bn_rows:
    k = (r["family"], r["t"])
    by_key[k]["bnadapt"].append(r["err_bnadapt"])

train_ts = {
    "gauss_noise": {0.0, 0.005, 0.01},
    "blur_heat": {0.0, 0.05, 0.1},
    "gauss_blur": {0.0, 0.05, 0.1},
    "fog_beer_lambert": {0.0, 0.1, 0.25},
}
titles = {
    "gauss_noise": "(a) Gaussian noise",
    "blur_heat": "(b) Heat-equation blur",
    "fog_beer_lambert": "(c) Ornstein-Uhlenbeck fog",
    "gauss_blur": "(d) Deterministic blur\n(reversible control)",
}
PANELS = ["gauss_noise", "blur_heat", "fog_beer_lambert", "gauss_blur"]

fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.7), sharey=True)
for ax, fam in zip(axes, PANELS):
    ks = sorted([k for k in by_key if k[0] == fam], key=lambda k: k[1])
    ts = [k[1] for k in ks]
    for name, color, marker in [
        ("oracle", C_ORACLE, "o"), ("frozen", C_FROZEN, "s"), ("bnadapt", C_BNADAPT, "P"),
        ("tent", C_TENT, "^"), ("eata", C_EATA, "v"), ("dann", C_DANN, "D"),
    ]:
        means = [st.mean(by_key[k][name]) for k in ks]
        stds = [st.stdev(by_key[k][name]) if len(by_key[k][name]) > 1 else 0.0 for k in ks]
        label = {"oracle": "Oracle", "frozen": "Frozen", "bnadapt": "BN-adapt"}.get(name, name.upper())
        ax.plot(ts, means, color=color, marker=marker, markersize=2.8, label=label)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(ts, lo, hi, color=color, alpha=0.12, linewidth=0)
    max_train_t = max(train_ts[fam])
    ax.axvline(max_train_t, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"severity $t$")
    ax.set_title(titles[fam])
axes[0].set_ylabel("test error")
axes[0].legend(frameon=False, loc="upper left", ncol=1, fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{BASE}/paper/figures/fig_cifar_main.pdf")
print("fig_cifar_main.pdf written")
