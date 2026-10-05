#!/usr/bin/env python3
"""Computes every quantity the paper quotes, from the raw result files.

Writes paper/facts.json. The text is written against these numbers and
verify_claims.py checks the text against them, so a number in the paper has one
source. Run from the repository root or from paper/.
"""
import json
import math
import os
import statistics as st

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F = {}


def load(path):
    return [json.loads(l) for l in open(os.path.join(BASE, path))]


def mean(xs):
    return float(st.mean(xs))


def sd(xs):
    return float(st.stdev(xs)) if len(xs) > 1 else 0.0


# ------------------------------------------------------------------ CIFAR main
BN = {1: "kaggle_cifar10_c_bnadapt/output_seed1",
      2: "kaggle_cifar10_c_bnadapt/output_seed2",
      3: "kaggle_cifar10_c_bnadapt_b/output_seed3"}
BH = {1: "kaggle_cifar10_c_bnadapt_blurheat/output_seed1",
      2: "kaggle_cifar10_c_bnadapt_blurheat/output_seed2",
      3: "kaggle_blurheat_bnadapt_s3/output_seed3"}
R = {}
for s, d in BN.items():
    for r in load(f"{d}/cifar10_c_results_bnadapt_seed{s}.jsonl"):
        R.setdefault((r["family"], r["t"]), {})[s] = r
for s, d in BH.items():
    for r in load(f"{d}/cifar10_c_results_bnadapt_blurheat_seed{s}.jsonl"):
        R.setdefault((r["family"], r["t"]), {})[s] = r


def m(fam, t, field):
    return mean([r[field] for r in R[(fam, t)].values()])


def s_(fam, t, field):
    return sd([r[field] for r in R[(fam, t)].values()])


def held_out(fam):
    return sorted(t for (f, t), v in R.items() if f == fam and not v[1]["is_train_severity"])


FAMS = ["gauss_noise", "blur_heat", "fog_beer_lambert", "gauss_blur"]

# blur pair, oracle-free
for fam in ("gauss_blur", "blur_heat"):
    F[f"{fam}_t2_frozen"] = m(fam, 2.0, "err_frozen")
    F[f"{fam}_t2_oracle"] = m(fam, 2.0, "err_oracle")
    F[f"{fam}_t2_bn_removes"] = m(fam, 2.0, "err_frozen") - m(fam, 2.0, "err_bnadapt")
    F[f"{fam}_t2_tent_removes"] = m(fam, 2.0, "err_frozen") - m(fam, 2.0, "err_tent")
F["blur_ratio_bn"] = F["gauss_blur_t2_bn_removes"] / F["blur_heat_t2_bn_removes"]
F["blur_ratio_tent"] = F["gauss_blur_t2_tent_removes"] / F["blur_heat_t2_tent_removes"]
F["blur_ratio_bn_per_seed"] = {
    s: (R[("gauss_blur", 2.0)][s]["err_frozen"] - R[("gauss_blur", 2.0)][s]["err_bnadapt"])
    / (R[("blur_heat", 2.0)][s]["err_frozen"] - R[("blur_heat", 2.0)][s]["err_bnadapt"]) for s in (1, 2, 3)}

# recovery fractions (ratio of three-seed means), SGD run
for fam in FAMS:
    recs = {t: 100 * (1 - m(fam, t, "delta_tent") / m(fam, t, "delta_frozen")) for t in held_out(fam)}
    F[f"rec_sgd_{fam}"] = recs
    F[f"rec_sgd_{fam}_min"] = min(recs.values())
    F[f"rec_sgd_{fam}_max"] = max(recs.values())
# control: slack-corrected bound at t=2
F["control_slack_t2"] = m("gauss_blur", 2.0, "err_oracle") - m("gauss_blur", 0.0, "err_oracle")
F["control_rec_t2_corrected"] = 100 * F["gauss_blur_t2_tent_removes"] / (
    m("gauss_blur", 2.0, "delta_frozen") + F["control_slack_t2"])
F["control_oracle_curve"] = {t: m("gauss_blur", t, "err_oracle") for t in sorted(t for f, t in R if f == "gauss_blur")}

# BN / TENT / EATA spread under SGD
spread = []
for fam in FAMS:
    for t in held_out(fam):
        v = [m(fam, t, k) for k in ("err_bnadapt", "err_tent", "err_eata")]
        spread.append(max(v) - min(v))
F["sgd_bn_tent_eata_spread_max"] = max(spread)
noise_spread = [max(m("gauss_noise", t, k) for k in ("err_bnadapt", "err_tent", "err_eata"))
                - min(m("gauss_noise", t, k) for k in ("err_bnadapt", "err_tent", "err_eata"))
                for t in held_out("gauss_noise")]
F["sgd_noise_spread_max"] = max(noise_spread)

# monotonicity
diss = ["gauss_noise", "blur_heat", "fog_beer_lambert"]
tot = ok = 0
for fam in diss:
    ts = sorted(t for f, t in R if f == fam)
    for s in (1, 2, 3):
        for a, b in zip(ts, ts[1:]):
            tot += 1
            ok += R[(fam, b)][s]["err_oracle"] >= R[(fam, a)][s]["err_oracle"]
F["mono_total"], F["mono_ok"] = tot, ok
ts = sorted(t for f, t in R if f == "gauss_blur")
inv_all, inv_held, inv_list = 0, [], []
for s in (1, 2, 3):
    for a, b in zip(ts, ts[1:]):
        d = R[("gauss_blur", b)][s]["err_oracle"] - R[("gauss_blur", a)][s]["err_oracle"]
        if d < 0:
            inv_all += 1
            inv_list.append((s, a, b, d))
            if a >= 0.25:
                inv_held.append((s, a, b, d))
F["control_inversions"] = inv_all
F["control_inversions_heldout"] = inv_held
F["control_inversion_list"] = inv_list
F["control_t0_seeds"] = [R[("gauss_blur", 0.0)][s]["err_oracle"] for s in (1, 2, 3)]

# ------------------------------------------------------------------ fog = noise
def v_eq(t, a, sig):
    return (sig ** 2 / (2 * a)) * (np.exp(2 * a * t) - 1)


nt = sorted(t for f, t in R if f == "gauss_noise")


def noise_interp(field, tn):
    return float(np.interp(tn, nt, [m("gauss_noise", t, field) for t in nt]))


eq = {}
for t in sorted(t for f, t in R if f == "fog_beer_lambert"):
    tn = float(v_eq(t, 1.0, 0.2) / 2)
    row = dict(t_noise=tn, fog_oracle=m("fog_beer_lambert", t, "err_oracle"))
    if tn <= max(nt):
        row["noise_oracle"] = noise_interp("err_oracle", tn)
        row["diff"] = row["fog_oracle"] - row["noise_oracle"]
    eq[t] = row
F["fog_noise_eq"] = eq
diffs = [abs(r["diff"]) for r in eq.values() if "diff" in r]
F["fog_noise_mean_absdiff"] = float(np.mean(diffs))
F["fog_noise_max_absdiff"] = float(np.max(diffs))
F["fog_noise_t0_diff"] = eq[0.0]["diff"]
# slopes of the noise oracle between grid points: decreasing slopes mean the curve is
# concave on the grid, so linear interpolation lies below it (understates the noise
# oracle and overstates fog - noise differences)
no = [m("gauss_noise", t, "err_oracle") for t in nt]
slopes = [(no[i + 1] - no[i]) / (nt[i + 1] - nt[i]) for i in range(len(nt) - 1)]
F["noise_oracle_slopes"] = slopes
F["noise_oracle_concave"] = all(slopes[i + 1] <= slopes[i] for i in range(len(slopes) - 1))
F["fog_t2_tnoise"] = eq[2.0]["t_noise"]
F["fog_t4_tnoise"] = eq[4.0]["t_noise"]
F["fog_t1_tnoise"] = eq[1.0]["t_noise"]
F["noise_grid_max"] = max(nt)

# matched-information fog vs noise, descriptive
tn = eq[1.0]["t_noise"]
F["match_fog_t1"] = dict(
    oracle=m("fog_beer_lambert", 1.0, "err_oracle"), frozen=m("fog_beer_lambert", 1.0, "err_frozen"),
    bn_removes=m("fog_beer_lambert", 1.0, "err_frozen") - m("fog_beer_lambert", 1.0, "err_bnadapt"),
    tent_rec=100 * (m("fog_beer_lambert", 1.0, "err_frozen") - m("fog_beer_lambert", 1.0, "err_tent"))
    / m("fog_beer_lambert", 1.0, "delta_frozen"),
    horizon=tn / float(v_eq(0.25, 1.0, 0.2) / 2))
nf = {k: noise_interp(k, tn) for k in ("err_oracle", "err_frozen", "err_bnadapt", "err_tent")}
F["match_noise_eq"] = dict(
    oracle=nf["err_oracle"], frozen=nf["err_frozen"], bn_removes=nf["err_frozen"] - nf["err_bnadapt"],
    tent_rec=100 * (nf["err_frozen"] - nf["err_tent"]) / (nf["err_frozen"] - nf["err_oracle"]),
    horizon=tn / 0.01)

# ------------------------------------------------------------------ fog ablation
A = {}
for s in (1, 2, 3):
    for r in load(f"kaggle_fog_ablation/output_seed{s}/cifar10_c_results_fogabl_seed{s}.jsonl"):
        A.setdefault((r["family"], r["t"]), []).append(r)
# a and sigma from the script that ran the ablation, the largest training severity from its rows
import re as _re0
_abl_src = open(os.path.join(BASE, "kaggle_fog_ablation/fog_ablation_kaggle.py")).read()
PAR = {f: (float(a_), float(s_), max(t for (g, t), rs in A.items() if g == f and rs[0]["is_train_severity"]))
       for f, a_, s_ in _re0.findall(r'"(fog_\w+)": dict\(a=([\d.]+), sigma=([\d.]+)\)', _abl_src)}
assert set(PAR) == {"fog_drift", "fog_diffuse"}, PAR


def am(fam, t, field):
    return mean([r[field] for r in A[(fam, t)]])


abl = {}
for (fam, t), rs in A.items():
    a, sig, tmax = PAR[fam]
    tn = float(v_eq(t, a, sig) / 2)
    row = dict(oracle=am(fam, t, "err_oracle"), oracle_sd=sd([r["err_oracle"] for r in rs]),
               frozen=am(fam, t, "err_frozen"), tent=am(fam, t, "err_tent"),
               v_eq=float(v_eq(t, a, sig)), t_noise=tn, horizon=t / tmax,
               is_train=rs[0]["is_train_severity"])
    row["d_frozen"] = row["frozen"] - row["oracle"]
    row["d_frozen_sd"] = sd([r["err_frozen"] - r["err_oracle"] for r in rs])
    row["tent_removes"] = row["frozen"] - row["tent"]
    row["avoidable_share"] = 100 * row["d_frozen"] / row["frozen"]
    if not row["is_train"]:
        row["rec"] = 100 * row["tent_removes"] / row["d_frozen"]
        row["rec_per_seed"] = [100 * (r["err_frozen"] - r["err_tent"]) / (r["err_frozen"] - r["err_oracle"]) for r in rs]
    if tn <= max(nt):
        row["noise_oracle"] = noise_interp("err_oracle", tn)
    abl[f"{fam}@{t}"] = row
F["ablation"] = abl
F["abl_drift_rise"] = abl["fog_drift@3.0"]["oracle"] - abl["fog_drift@0.0"]["oracle"]
F["abl_diffuse_rise"] = abl["fog_diffuse@8.0"]["oracle"] - abl["fog_diffuse@0.0"]["oracle"]
# extrapolation distance in equivalent-noise terms: v at the test point over v at the
# largest training severity (the drift regime is far further out than its t-horizon says)
F["abl_v_horizon"] = {k: r["v_eq"] / float(v_eq(PAR[k.split("@")[0]][2], *PAR[k.split("@")[0]][:2]))
                      for k, r in abl.items() if not r["is_train"]}
F["abl_v_train_max"] = {f: float(v_eq(PAR[f][2], *PAR[f][:2])) for f in PAR}
# drift training range matched in v to the diffusion range (the Oct-3 fog_common run)
F["drift_tmax_matched"] = float(np.log(1 + 2 * 1.0 * F["abl_v_train_max"]["fog_diffuse"] / 0.05 ** 2) / 2)
F["drift_decay"] = {t: float(np.exp(-t)) for t in (2.0, 3.0)}

# ------------------------------------------------------------------ DANN
D = {}
for s in (1, 2, 3):
    for r in load(f"kaggle_cifar10_c_dann/output_seed{s}_v2fix/cifar10_c_results_dann_seed{s}_v2fix.jsonl"):
        D.setdefault((r["family"], r["t"]), []).append(r["err_dann"])
for s, d in {1: "kaggle_cifar10_c_dann_blurheat/output_seed1",
             2: "kaggle_dann_blurheat_s2/output_seed2",
             3: "kaggle_dann_blurheat_s3/output_seed3"}.items():
    for r in load(f"{d}/cifar10_c_results_dann_blurheat_seed{s}.jsonl"):
        D.setdefault((r["family"], r["t"]), []).append(r["err_dann"])
V2 = {}
for s in (1, 2, 3):
    for r in load(f"kaggle_cifar10_c/output_seed{s}_v2fix/cifar10_c_results_seed{s}_v2fix.jsonl"):
        V2.setdefault((r["family"], r["t"]), []).append(r["err_tent"])
nuis = {k: abs(mean(V2[k]) - m(k[0], k[1], "err_tent")) for k in V2}
F["tent_cross_run_max"] = max(nuis[k] for k in nuis if not R[k][1]["is_train_severity"])
yard = F["tent_cross_run_max"]
dann = {}
for k in sorted(D):
    if R[k][1]["is_train_severity"]:
        continue
    diff = mean(D[k]) - m(k[0], k[1], "err_tent")
    dann[f"{k[0]}@{k[1]}"] = dict(diff=diff, resolved=abs(diff) > yard)
F["dann_vs_tent"] = dann
orig = [v for k, v in dann.items() if not k.startswith("blur_heat")]
F["dann_resolved"] = sum(v["resolved"] for v in orig)
F["dann_resolved_worse"] = sum(v["resolved"] and v["diff"] > 0 for v in orig)
F["dann_unresolved"] = sum(not v["resolved"] for v in orig)
res = [v["diff"] for v in orig if v["resolved"]]
F["dann_resolved_min"], F["dann_resolved_max"] = min(res), max(res)
bh = [v["diff"] for k, v in dann.items() if k.startswith("blur_heat")]
F["dann_blurheat_diffs"] = bh

# heat-equation blur: TENT was computed by a second, independent 3-seed run, and seed 1
# of that run exists twice (two accounts ran it at the same time). Every pairing gives
# its own yardstick and its own DANN - TENT difference; we report all of them.
BH2 = {"s1": "kaggle_cifar10_c_blurheat/output_seed1/cifar10_c_results_blurheat_seed1.jsonl",
       "s1_rerun": "kaggle_cifar10_c_blurheat/output_seed1_rerun/cifar10_c_results_blurheat_seed1.jsonl"}
BH2_23 = ["kaggle_cifar10_c_blurheat/output_seed2/cifar10_c_results_blurheat_seed2.jsonl",
          "kaggle_blurheat_s3/output_seed3/cifar10_c_results_blurheat_seed3.jsonl"]
heat_runs = {"main": {t: m("blur_heat", t, "err_tent") for t in held_out("blur_heat")}}
for tag, p1 in BH2.items():
    rows = [{r["t"]: r for r in load(p)} for p in [p1] + BH2_23]
    heat_runs[tag] = {t: mean([rr[t]["err_tent"] for rr in rows]) for t in held_out("blur_heat")}
s1a = {r["t"]: r["err_tent"] for r in load(BH2["s1"])}
s1b = {r["t"]: r["err_tent"] for r in load(BH2["s1_rerun"])}
F["heat_tent_same_seed_max"] = max(abs(s1a[t] - s1b[t]) for t in held_out("blur_heat"))
F["heat_tent_yardstick"] = {tag: max(abs(heat_runs["main"][t] - heat_runs[tag][t]) for t in held_out("blur_heat"))
                            for tag in BH2}
dh = {tag: {t: mean(D[("blur_heat", t)]) - heat_runs[tag][t] for t in held_out("blur_heat")} for tag in heat_runs}
F["dann_heat_vs_tent"] = dh
ahead = [dh[tag][t] for tag in dh for t in held_out("blur_heat") if t < 2.0]
F["dann_heat_ahead_min"], F["dann_heat_ahead_max"] = -max(ahead), -min(ahead)   # all negative: DANN ahead
assert max(ahead) < 0
F["dann_heat_t1"] = {tag: dh[tag][1.0] for tag in dh}
F["dann_heat_t2"] = {tag: dh[tag][2.0] for tag in dh}

# ------------------------------------------------------------------ lambda sweep
LS = load("kaggle_dann_lambda_sweep/output_seed1/cifar10_c_results_dann_lambda_sweep_seed1.jsonl")
F["lambda_min_ratio"] = {r["t"]: min(r["delta_dann"].values()) / r["delta_frozen"] for r in LS}
F["tent_sgd_ratio_noise"] = {t: m("gauss_noise", t, "delta_tent") / m("gauss_noise", t, "delta_frozen")
                             for t in held_out("gauss_noise")}

# ------------------------------------------------------------------ TENT sweep
SW = load("kaggle_tent_lrsweep/output_seed1/cifar10_c_results_tentsweep_seed2.jsonl")
fs = {}
for r in SW:
    fs.setdefault(r["family"], []).append(r)
sweep = {}
for fam, rs in fs.items():
    ho = [r for r in rs if not r["is_train_severity"]]
    cfgs = ho[0]["errs"].keys()
    mm = {c: mean([r["errs"][c] for r in ho]) for c in cfgs}
    tent = {c: v for c, v in mm.items() if c != "bnadapt"}
    best = min(tent, key=tent.get)
    sweep[fam] = dict(bn=mm["bnadapt"], ours=mm["tent_lr0.001_s1_sgd"], adam=mm["tent_lr0.001_s1_adam"],
                      best_cfg=best, best=tent[best],
                      gap_ours=mm["bnadapt"] - mm["tent_lr0.001_s1_sgd"],
                      gap_adam=mm["bnadapt"] - mm["tent_lr0.001_s1_adam"],
                      gap_best=mm["bnadapt"] - tent[best])
F["sweep"] = sweep

# ------------------------------------------------------------------ corrected noise run (Adam)
FX = {}
for s in (1, 2, 3):
    for r in load(f"kaggle_tent_fixed_s{s}/output/cifar10_c_results_tentfixed_seed{s}.jsonl"):
        FX.setdefault(r["t"], []).append(r)
fx = {}
for t, rs in FX.items():
    if rs[0]["is_train_severity"]:
        continue
    dfr = mean([r["delta_frozen"] for r in rs])
    fx[t] = dict(d_frozen=dfr, d_frozen_sd=sd([r["delta_frozen"] for r in rs]),
                 bn_minus_tent=mean([r["delta_bnadapt"] for r in rs]) - mean([r["delta_tent"] for r in rs]),
                 rec=100 * (1 - mean([r["delta_tent"] for r in rs]) / dfr))
F["fixed"] = fx

# ------------------------------------------------------------------ Wiener (fixed operator)
W = {r["t"]: r for r in load("kaggle_wiener_fixed/output/wiener_test_results.jsonl")}
F["wiener"] = {t: dict(naive=r["err_naive"], best=r["err_wiener_best"], best_eps=r["best_eps"],
                       oracle=m("gauss_blur", t, "err_oracle"), oracle_sd=s_("gauss_blur", t, "err_oracle"))
               for t, r in W.items()}

# ------------------------------------------------------------------ kill-tests
KT = load("kill_test_results.jsonl")
F["kt_noise_slack_max"] = max(abs(r["oracle_slack"]) for r in KT)
F["kt_noise_delta_max"] = max(abs(r["delta"]) for r in KT)
F["kt_noise_extrap_max"] = max(abs(r["extrapolation_error"]) for r in KT)
KF = load("kill_test_fog_results.jsonl")
F["kt_fog_slack_max"] = max(abs(r["oracle_slack"]) for r in KF)
F["kt_fog_extrap_worst"] = max(KF, key=lambda r: abs(r["extrapolation_error"]))["extrapolation_error"]
KA = load("kill_test_asym_results.jsonl")
F["kt_asym_slack_max"] = max(abs(r["oracle_slack"]) for r in KA)
F["kt_asym_predmiss_max"] = max(abs(r["frozen_pred_minus_meas"]) for r in KA)
F["kt_asym_gap_max"] = max(r["delta_pred"] for r in KA)

# ------------------------------------------------------------------ Tiny-ImageNet (first run)
T = {r["t"]: r for r in load("kaggle_tinyimagenet/output_v3/tin_results_tin_seed1.jsonl")}
F["tin_spread_max"] = max(max(r[k] for k in ("delta_bnadapt", "delta_tent", "delta_eata"))
                          - min(r[k] for k in ("delta_bnadapt", "delta_tent", "delta_eata"))
                          for r in T.values() if not r["is_train_severity"])


# ------------------------------------------------------------------ blur operator measurements
from scipy.ndimage import gaussian_filter1d


def blur_matrix(t, n=32, mode="reflect", trunc=4.0):
    return np.stack([gaussian_filter1d(np.eye(n)[i], np.sqrt(2 * t), mode=mode, truncate=trunc)
                     for i in range(n)], axis=1)


op = {}
for t in (0.25, 0.5, 1.0, 2.0):
    sv = np.linalg.svd(blur_matrix(t), compute_uv=False)
    op[t] = dict(min_sv_2d=float(sv.min() ** 2), cond_2d=float((sv.max() / sv.min()) ** 2))
F["blur_operator"] = op
comp = {}
for t, s2 in ((0.05, 0.05), (0.1, 0.15), (0.25, 0.25), (0.5, 0.5)):
    Dm = blur_matrix(t) @ blur_matrix(s2) - blur_matrix(t + s2)
    comp[f"{t}+{s2}"] = float(np.linalg.norm(Dm, 2) / np.linalg.norm(blur_matrix(t + s2), 2))
F["blur_composition_dev"] = comp
k = gaussian_filter1d(np.eye(32)[16], np.sqrt(2 * 0.05), mode="wrap")
F["sampled_kernel_var_t005"] = float(np.sum(k * (np.arange(32) - 16) ** 2))
F["float32_eps"] = float(np.finfo(np.float32).eps)

# heat-equation blur on the 32x32 periodic grid: omega^2 = fx^2 + fy^2 + a_min with
# f in radians per pixel, so the largest omega^2 is 2 pi^2 + a_min
HEAT_SIGMA, HEAT_AMIN = 15.0, 0.1
F["heat_omega2_max"] = 2 * np.pi ** 2 + HEAT_AMIN


def heat_noise_var(w2, t):
    return HEAT_SIGMA ** 2 / (2 * w2) * (1 - np.exp(-2 * w2 * t))


F["heat_noisevar_t2_dc"] = float(heat_noise_var(HEAT_AMIN, 2.0))
F["heat_noisevar_t2_max"] = float(heat_noise_var(F["heat_omega2_max"], 2.0))
F["heat_dc_decay_t2"] = float(np.exp(-HEAT_AMIN * 2.0))
# Wiener filter conj(H)/(|H|^2 + eps): the gain over all |H| peaks at 1/(2 sqrt(eps))
F["wiener_gain_cap"] = {str(e): 1 / (2 * np.sqrt(e)) for e in (1e-4, 1e-2)}

# heat-equation blur's equivalent covariance per mode in the unitary Fourier basis (the
# basis in which additive noise has covariance 2t I): sigma^2 (e^{2 w2 t} - 1) / (2 w2 HW).
# Corollary 1 orders it against an isotropic v I only if v <= its smallest or >= its
# largest per-mode value.
fq = np.fft.fftfreq(32) * 2 * np.pi
W2 = (fq[:, None] ** 2 + fq[None, :] ** 2 + HEAT_AMIN).ravel()
heat_eq = {}
for t in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
    st_ = HEAT_SIGMA ** 2 * np.expm1(2 * W2 * t) / (2 * W2 * 1024)
    heat_eq[t] = dict(min=float(st_.min()), max=float(st_.max()))
F["heat_eq_cov"] = heat_eq
F["heat_noisevar_t2_dc_unitary"] = F["heat_noisevar_t2_dc"] / 1024
F["heat_noisevar_t2_max_unitary"] = F["heat_noisevar_t2_max"] / 1024


# Wiener witness: modes with |H| < sqrt(eps) are attenuated to less than half
def gauss_kernel_fft(sigma, size=32):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return np.fft.fft2(np.fft.ifftshift(k / k.sum()))


F["wiener_half_suppressed_frac"] = {t: float(np.mean(np.abs(gauss_kernel_fft(np.sqrt(2 * t))) < 1e-2))
                                    for t in (0.25, 0.5, 1.0, 2.0)}

# float32 saturation: what the first Tiny-ImageNet run logged, and a local reproduction
NS = json.load(open(os.path.join(BASE, "paper/logged_norm_stats.json")))["runs"]["tiny_imagenet_seed1"]
F["tin_logged_mean"], F["tin_logged_std"] = NS["mean"][0], NS["std"][0]
F["tin_N"] = 40000 * 64 * 64
F["tin_2p24_over_N"] = 2 ** 24 / F["tin_N"]
FS = json.load(open(os.path.join(BASE, "float32_saturation_results.json")))
F["f32sat_local_mean"] = FS["mean_float32"][0]
F["f32sat_local_std_min"], F["f32sat_local_std_max"] = min(FS["std_float32"]), max(FS["std_float32"])
F["f32sat_numpy"] = FS["numpy_version"]


# ------------------------------------------------------------------ misc quantities quoted in text
# feature separation at the ends of the ablation grids
for key, (fam, t) in {"drift_end": ("fog_drift", 3.0), "diffuse_end": ("fog_diffuse", 8.0)}.items():
    F[f"sep_oracle_{key}"] = mean([r["sep_oracle"] for r in A[(fam, t)]])
    F[f"sep_frozen_{key}"] = mean([r["sep_frozen"] for r in A[(fam, t)]])
# largest oracle difference between the v2fix run and the BN-adapt run (same seeds)
V2O = {}
for s in (1, 2, 3):
    for r in load(f"kaggle_cifar10_c/output_seed{s}_v2fix/cifar10_c_results_seed{s}_v2fix.jsonl"):
        V2O[(r["family"], r["t"], s)] = r["err_oracle"]
F["oracle_cross_run_max"] = max(abs(V2O[k] - R[(k[0], k[1])][k[2]]["err_oracle"]) for k in V2O)
# largest BN-adapt vs TENT gap within one run (what the corrected figure shows)
F["within_run_bn_tent_max"] = max(abs(m(f, t, "err_bnadapt") - m(f, t, "err_tent")) for (f, t) in R)
# clean oracle error, mean over families
F["clean_oracle_mean"] = mean([m(f, 0.0, "err_oracle") for f in FAMS])


# ------------------------------------------------------------------ certified quantities (Section 4.2, 5.5)
from scipy.stats import norm as _norm

N_TEST = 10000
_z1, _z3 = _norm.ppf(0.95), _norm.ppf(1 - 0.05 / 3)


def _se(e):
    return float(np.sqrt(e * (1 - e) / N_TEST))


def _upper(errs):
    """Upper bound on a Bayes risk from three oracles: best seed plus a one-sided
    binomial margin, Bonferroni over the seeds."""
    return min(e + _z3 * _se(e) for e in errs)


def _cert(errs, ub):
    """Certified slack of each oracle: its error minus a one-sided margin, minus ub."""
    return [max(0.0, e - _z1 * _se(e) - ub) for e in errs]


ctl0 = _upper([R[("gauss_blur", 0.0)][s]["err_oracle"] for s in (1, 2, 3)])
F["cert_control_upper"] = ctl0
F["cert_control"] = {t: _cert([R[("gauss_blur", t)][s]["err_oracle"] for s in (1, 2, 3)], ctl0)
                     for t in held_out("gauss_blur")}
# joint version of the control bound: Bonferroni over the six errors behind it (the three
# t = 0 oracles that define the upper bound and the three t = 2 oracles), so the three
# per-seed bounds hold together at 95%
_z6 = _norm.ppf(1 - 0.05 / 6)
_u6 = min(e + _z6 * _se(e) for e in [R[("gauss_blur", 0.0)][s]["err_oracle"] for s in (1, 2, 3)])
F["control_slack_t2_joint"] = sorted(R[("gauss_blur", 2.0)][s]["err_oracle"] - _z6 * _se(R[("gauss_blur", 2.0)][s]["err_oracle"]) - _u6
                                     for s in (1, 2, 3))
F["control_rec_t2_corrected_joint"] = 100 * F["gauss_blur_t2_tent_removes"] / (
    m("gauss_blur", 2.0, "delta_frozen") + F["control_slack_t2_joint"][0])
# the control gap correction with margins: the smallest per-seed certified slack at t = 2
# is a lower bound for every seed, so the true gap exceeds Delta_frozen by at least it
F["control_slack_t2_margin"] = [min(F["cert_control"][2.0]), max(F["cert_control"][2.0])]
F["control_rec_t2_corrected_margin"] = 100 * F["gauss_blur_t2_tent_removes"] / (
    m("gauss_blur", 2.0, "delta_frozen") + F["control_slack_t2_margin"][0])
ngrid = sorted(t for f, t in R if f == "gauss_noise")
checks = []
for fam, (a_, s_, _) in PAR.items():
    for (f, t), rs in A.items():
        if f != fam or t == 0.0:
            continue
        above = [g for g in ngrid if g >= v_eq(t, a_, s_) / 2]
        if above:
            checks += _cert([r["err_oracle"] for r in rs], _upper([R[("gauss_noise", above[0])][s]["err_oracle"] for s in (1, 2, 3)]))
for t in (0.1, 0.25, 0.5, 1.0):
    above = [g for g in ngrid if g >= v_eq(t, 1.0, 0.2) / 2]
    checks += _cert([R[("fog_beer_lambert", t)][s]["err_oracle"] for s in (1, 2, 3)],
                    _upper([R[("gauss_noise", above[0])][s]["err_oracle"] for s in (1, 2, 3)]))
F["cert_dissipative_n"] = len(checks)
F["cert_dissipative_positive"] = sorted(c for c in checks if c > 0)
F["cert_dissipative_expected_by_chance"] = 0.05 * len(checks)

CE = json.load(open(os.path.join(BASE, "certified_extrapolation_results.json")))
F["ce_synthetic"] = CE
CC = json.load(open(os.path.join(BASE, "certified_extrapolation_cifar_results.json")))
F["ce_cifar"] = CC
F["ce_cifar_covered"] = sum(p["covered"] for r in ("fog_main", "fog_drift", "fog_diffuse") for p in CC[r]["points"].values())
F["ce_cifar_points"] = sum(len(CC[r]["points"]) for r in ("fog_main", "fog_drift", "fog_diffuse"))
fm = CC["fog_main"]
F["ce_cifar_main_slope"] = (fm["points"]["4.0"]["log_width"] - fm["points"]["2.0"]["log_width"]) / 2
F["ce_cifar_main_two_delta_a"] = 2 * (fm["a_rect"][1] - fm["a_rect"][0])


# ------------------------------------------------------------------ 3 October runs (oct3/)
import glob as _glob


def _oct3(pattern):
    rows = []
    for p in sorted(_glob.glob(os.path.join(BASE, "oct3", pattern))):
        rows += load(os.path.relpath(p, BASE))
    return rows


def _m3(xs):
    return mean(xs), (sd(xs) if len(xs) > 1 else float("nan"))


# --- like-for-like Wiener witness: same periodic operator, eps chosen on a held-out split
WL = _oct3("wiener-lfl-s*/output/wiener_lfl_seed*.jsonl")
wl = {}
for t in sorted({r["t"] for r in WL}):
    rs = [r for r in WL if r["t"] == t]
    row = dict(n=len(rs), naive=_m3([r["err_naive"] for r in rs]), oracle=_m3([r["err_oracle"] for r in rs]),
               witness=_m3([r["err_witness"] for r in rs]))
    if t > 0:
        row["witness_f64"] = _m3([r["err_witness_f64"] for r in rs])
        row["eps"] = sorted({r["eps"] for r in rs})
        row["eps_f64"] = sorted({r["eps_f64"] for r in rs})
        # certified slack per seed: oracle minus witness minus one-sided margins on both
        row["cert_slack"] = sorted(max(0.0, r["err_oracle"] - _z1 * _se(r["err_oracle"]) - r["err_witness"] - _z1 * _se(r["err_witness"]))
                                   for r in rs)
        # joint version: Bonferroni over the 12 errors behind the six bounds (oracle and
        # witness, three seeds, t = 1 and t = 2), so all six hold together at 95%
        _z12 = _norm.ppf(1 - 0.05 / 12)
        row["cert_slack_joint"] = sorted(r["err_oracle"] - _z12 * _se(r["err_oracle"]) - r["err_witness"] - _z12 * _se(r["err_witness"])
                                         for r in rs)
        row["gap"] = sorted(r["err_oracle"] - r["err_witness"] for r in rs)
        row["f32_cost"] = mean([r["err_witness"] - r["err_witness_f64"] for r in rs])
    wl[t] = row
F["wiener_lfl"] = wl
F["wiener_lfl_seeds"] = sorted({r["seed"] for r in WL})

# --- fog_common: matched-v pairs (decision rule in oct3/PLAN.md, applied by oct3/summarize.py)
FC = _oct3("fog-common-s*/output/cifar10_c_results_fog_common_seed*.jsonl")
fcb = {(r["family"], round(r["t"], 4), r["model"], r["seed"]): r for r in FC}
fc_seeds = sorted({r["seed"] for r in FC})
F["fc_seeds"] = fc_seeds
pairs = sorted({(round(r["t"], 4), r["v"]) for r in FC if r["family"] == "fog_drift" and r["model"] == "clean"})
dpts = sorted({(round(r["t"], 4), r["v"]) for r in FC if r["family"] == "fog_diffuse" and r["model"] == "clean"})
fcp = []
for td, vd in pairs:
    m = [tD for tD, vD in dpts if abs(vd - vD) / vD < 1e-3]
    if not m:
        continue
    tD = m[0]
    row = dict(t_drift=td, t_diffuse=tD, v=vd)
    for md, mD, tag in (("clean", "clean", "clean"), ("drift_matched", "diffuse_own", "matched")):
        raw = [fcb[("fog_drift", td, md, s)]["err_frozen"] - fcb[("fog_diffuse", tD, mD, s)]["err_frozen"] for s in fc_seeds]
        inv = [fcb[("fog_drift", td, "clean", s)]["err_frozen_inv"] - fcb[("fog_diffuse", tD, "clean", s)]["err_frozen_inv"] for s in fc_seeds]
        row[tag] = dict(err_drift=mean([fcb[("fog_drift", td, md, s)]["err_frozen"] for s in fc_seeds]),
                        err_diffuse=mean([fcb[("fog_diffuse", tD, mD, s)]["err_frozen"] for s in fc_seeds]),
                        d_raw=raw, d_raw_mean=mean(raw), d_raw_sd=sd(raw), d_inv_abs_max=max(abs(x) for x in inv),
                        positive=(all(x > 0 for x in raw) or all(x < 0 for x in raw)) and all(abs(a) > abs(b) for a, b in zip(raw, inv))
                        and abs(mean(raw)) > sd(raw))
        row[tag]["bn_drift"] = mean([fcb[("fog_drift", td, md, s)]["err_bnadapt"] for s in fc_seeds])
        row[tag]["bn_diffuse"] = mean([fcb[("fog_diffuse", tD, mD, s)]["err_bnadapt"] for s in fc_seeds])
    row["oracle_diff"] = mean([fcb[("fog_drift", td, "clean", s)]["err_oracle"] - fcb[("fog_diffuse", tD, "clean", s)]["err_oracle"] for s in fc_seeds])
    row["oracle_drift"] = mean([fcb[("fog_drift", td, "clean", s)]["err_oracle"] for s in fc_seeds])
    row["oracle_diffuse"] = mean([fcb[("fog_diffuse", tD, "clean", s)]["err_oracle"] for s in fc_seeds])
    row["v_horizon"] = fcb[("fog_drift", td, "drift_matched", fc_seeds[0])]["v_horizon"]
    fcp.append(row)
F["fc_pairs"] = fcp
F["fc_oracle_diff_seed_max"] = max(abs(fcb[("fog_drift", r["t_drift"], "clean", s)]["err_oracle"]
                                       - fcb[("fog_diffuse", r["t_diffuse"], "clean", s)]["err_oracle"]) for r in fcp for s in fc_seeds)
F["fc_matched_npos"] = sum(r["matched"]["positive"] for r in fcp)
F["fc_clean_npos"] = sum(r["clean"]["positive"] for r in fcp)
F["fc_clean_err_min"] = min(min(r["clean"]["err_drift"], r["clean"]["err_diffuse"]) for r in fcp)
F["fc_clean_err_max"] = max(max(r["clean"]["err_drift"], r["clean"]["err_diffuse"]) for r in fcp)
F["fc_inv_code_check_max"] = max(r["clean"]["d_inv_abs_max"] for r in fcp)
# the original 4x reversal, re-measured, with the drift regime's own model and with the v-matched one
rev = {}
for md in ("drift_own", "drift_matched"):
    d_ = [fcb[("fog_drift", 2.0, md, s)]["err_frozen"] for s in fc_seeds]
    D_ = [fcb[("fog_diffuse", 4.0, "diffuse_own", s)]["err_frozen"] for s in fc_seeds]
    rev[md] = dict(drift=mean(d_), diffuse=mean(D_), n_reversed=sum(a > b for a, b in zip(d_, D_)))
F["fc_4x"] = rev
# the clean model at the lowest drift severity, raw and with the reversible part undone
F["fc_clean_drift_t1"] = dict(raw=mean([fcb[("fog_drift", 1.0, "clean", s)]["err_frozen"] for s in fc_seeds]),
                              inv=mean([fcb[("fog_drift", 1.0, "clean", s)]["err_frozen_inv"] for s in fc_seeds]),
                              v=fcb[("fog_drift", 1.0, "clean", fc_seeds[0])]["v"])
F["fc_clean_beyond_chance_v"] = min(r["v"] for r in FC if r["model"] == "clean" and r["err_frozen_inv"] > 0.85)

# --- blur factorial on one periodic FFT operator (BN-adapt error removed at each severity)
FB = _oct3("blur-factorial-s*/output/cifar10_c_results_factorial_seed*.jsonl") + \
     _oct3("blur-factorial-extra-s*/output/cifar10_c_results_factorial_extra_seed*.jsonl")
fact = {}
for fam in sorted({r["family"] for r in FB}):
    for t in sorted({r["t"] for r in FB if r["family"] == fam}):
        rs = [r for r in FB if r["family"] == fam and r["t"] == t]
        fact[f"{fam}@{t}"] = dict(n=len(rs), bn_removes=_m3([r["err_frozen"] - r["err_bnadapt"] for r in rs]),
                                  frozen=_m3([r["err_frozen"] for r in rs]), oracle=_m3([r["err_oracle"] for r in rs]))
F["blur_factorial"] = fact


# --- TENT/EATA with Adam for the families that had only the SGD setting
AM = _oct3("adam-main-s*/output/cifar10_c_results_adam_main_seed*.jsonl")
F["adam_main_seeds"] = sorted({r["seed"] for r in AM})
FAMMAP = {"blurfft_s15_a01": "blur_heat", "gauss_blur": "gauss_blur", "fog_beer_lambert": "fog_beer_lambert"}
am = {}
for fam, key in FAMMAP.items():
    for t in sorted({r["t"] for r in AM if r["family"] == fam}):
        rs = [r for r in AM if r["family"] == fam and r["t"] == t]
        row = {k: _m3([r[k] for r in rs]) for k in ("err_oracle", "err_frozen", "err_bnadapt", "err_tent", "err_eata",
                                                     "delta_frozen", "delta_bnadapt", "delta_tent", "delta_eata")}
        row["n"] = len(rs)
        row["is_train"] = rs[0]["is_train_severity"]
        dfz = mean([r["delta_frozen"] for r in rs])
        if not row["is_train"]:
            row["rec_tent"] = 100 * mean([r["err_frozen"] - r["err_tent"] for r in rs]) / dfz
            row["bn_minus_tent"] = mean([r["err_bnadapt"] - r["err_tent"] for r in rs])
        am[f"{key}@{t}"] = row
F["adam_main"] = am
for key in FAMMAP.values():
    recs = {k: v["rec_tent"] for k, v in am.items() if k.startswith(key + "@") and "rec_tent" in v
            and v["delta_frozen"][0] > 0.05}          # fractions of gaps below 0.05 are not meaningful
    if recs:
        F[f"rec_adam_{key}_min"], F[f"rec_adam_{key}_max"] = min(recs.values()), max(recs.values())
        F[f"rec_adam_{key}"] = recs
# heat-equation blur: DANN against TENT with Adam, same seeds (oracle-free)
dh_adam = {}
for t in held_out("blur_heat"):
    tent = [r["err_tent"] for r in AM if r["family"] == "blurfft_s15_a01" and r["t"] == t]
    if len(tent) == 3:
        dh_adam[t] = mean(D[("blur_heat", t)]) - mean(tent)
F["dann_heat_vs_tent_adam"] = dh_adam

# --- one run-to-run yardstick for every family: the largest change in a method's
# three-seed mean between two independent trainings, at held-out severities. Three
# trainings exist: the main run, the earlier run with TENT at the SGD setting, and the
# Adam reruns; BN-adapt has no settings and is computed in the main run and the reruns.
_inv = {v: k for k, v in FAMMAP.items()}
_bn_cross = {}
for fam in FAMS:
    for t in held_out(fam):
        other = ([r["err_bnadapt"] for r in FX[t]] if fam == "gauss_noise" else
                 [r["err_bnadapt"] for r in AM if r["family"] == _inv[fam] and r["t"] == t])
        if len(other) == 3:
            _bn_cross[f"{fam}@{t}"] = abs(mean([r["err_bnadapt"] for r in R[(fam, t)].values()]) - mean(other))
F["bn_cross_run"] = _bn_cross
F["bn_cross_run_max_key"], F["bn_cross_run_max"] = max(_bn_cross.items(), key=lambda kv: kv[1])
F["yardstick"] = max(F["tent_cross_run_max"], F["bn_cross_run_max"], max(F["heat_tent_yardstick"].values()))
for v in dann.values():
    v["resolved_all"] = abs(v["diff"]) > F["yardstick"]
_orig = [v for k, v in dann.items() if not k.startswith("blur_heat")]
F["dann_resolved_all"] = sum(v["resolved_all"] for v in _orig)
F["dann_resolved_all_worse"] = sum(v["resolved_all"] and v["diff"] > 0 for v in _orig)
_res = [v["diff"] for v in _orig if v["resolved_all"]]
F["dann_resolved_all_min"], F["dann_resolved_all_max"] = min(_res), max(_res)
F["dann_unresolved_all_keys"] = sorted(k for k, v in dann.items() if not k.startswith("blur_heat") and not v["resolved_all"])
F["dann_heat_within_yardstick"] = (all(abs(x) <= F["yardstick"] for tag in dh for x in dh[tag].values())
                                   and all(abs(x) <= F["yardstick"] for x in dh_adam.values()))
F["dann_vs_tent"] = dann

# --- Tiny-ImageNet rerun: Adam, float64 statistics (results rebuilt from the kernel log)
TA = _oct3("tin-adam-s*/output/tin_results_tin_adam_seed*.jsonl")
F["tin_adam"] = {r["t"]: r for r in TA}


# --- fog_joint: one model trained on both regimes' v-matched grids (rule: oct3/PLAN.md, 3 Oct 07:15)
def _joint(pattern, key):
    JR = _oct3(pattern)
    if not JR:
        return
    jb = {(r["family"], round(r["t"], 4), r["seed"]): r for r in JR}
    seeds = sorted({r["seed"] for r in JR})
    out = []
    for td, tD in sorted({(round(r["t"], 4), None) for r in JR if r["family"] == "fog_drift"}):
        v = jb[("fog_drift", td, seeds[0])]["v"]
        tD = [t for (f, t, s) in jb if f == "fog_diffuse" and abs(jb[(f, t, s)]["v"] - v) / v < 1e-3][0]
        raw = [jb[("fog_drift", td, s)]["err_frozen"] - jb[("fog_diffuse", tD, s)]["err_frozen"] for s in seeds]
        inv = [jb[("fog_drift", td, s)]["err_frozen_inv"] - jb[("fog_diffuse", tD, s)]["err_frozen_inv"] for s in seeds]
        out.append(dict(v=v, t_drift=td, t_diffuse=tD, seeds=seeds,
                        err_drift=mean([jb[("fog_drift", td, s)]["err_frozen"] for s in seeds]),
                        err_diffuse=mean([jb[("fog_diffuse", tD, s)]["err_frozen"] for s in seeds]),
                        err_drift_inv=mean([jb[("fog_drift", td, s)]["err_frozen_inv"] for s in seeds]),
                        err_diffuse_inv=mean([jb[("fog_diffuse", tD, s)]["err_frozen_inv"] for s in seeds]),
                        bn_drift=mean([jb[("fog_drift", td, s)]["err_bnadapt"] for s in seeds]),
                        bn_diffuse=mean([jb[("fog_diffuse", tD, s)]["err_bnadapt"] for s in seeds]),
                        d_raw=raw, d_raw_mean=mean(raw), d_raw_sd=sd(raw) if len(raw) > 1 else float("nan"),
                        d_inv_abs_max=max(abs(x) for x in inv),
                        positive=len(seeds) == 3 and (all(x > 0 for x in raw) or all(x < 0 for x in raw))
                        and all(abs(a) > abs(b) for a, b in zip(raw, inv)) and abs(mean(raw)) > sd(raw)))
    F[key] = out
    F[key + "_npos"] = sum(r["positive"] for r in out)


_joint("fog-joint-s*/output/cifar10_c_results_fog_joint_seed*.jsonl", "fj_pairs")
_joint("c100-fog-joint-s*/output/cifar100_c_results_c100_fog_joint_seed*.jsonl", "fj100_pairs")
_joint("fog-joint-full-s*/output/cifar10_c_results_fog_joint_full_seed*.jsonl", "fjf_pairs")
_joint("c100-fog-joint-full-s*/output/cifar100_c_results_c100_fog_joint_full_seed*.jsonl", "fjf100_pairs")
for _k, _p in (("fj", "fog-joint-s*/output/cifar10_c_results_fog_joint_seed*.jsonl"),
               ("fj100", "c100-fog-joint-s*/output/cifar100_c_results_c100_fog_joint_seed*.jsonl"),
               ("fjf", "fog-joint-full-s*/output/cifar10_c_results_fog_joint_full_seed*.jsonl"),
               ("fjf100", "c100-fog-joint-full-s*/output/cifar100_c_results_c100_fog_joint_full_seed*.jsonl")):
    _cl = [r["err_frozen"] for r in _oct3(_p) if r["family"] == "clean"]
    if _cl:
        F[f"{_k}_clean_err"] = _m3(_cl)


# --- scale: CIFAR-100 and Tiny-ImageNet (matched fog pairs, blur pair)
def _fc_pairs(rows, models=("drift_matched", "diffuse_own")):
    """Matched-v pairs from a fog_common-style run: per pair, the separate-model gap
    and the oracle difference; tolerant of missing seeds."""
    if not rows:
        return None
    b = {(r["family"], round(r["t"], 4), r["model"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    dr = sorted({(round(r["t"], 4), r["v"]) for r in rows if r["family"] == "fog_drift"})
    df = sorted({(round(r["t"], 4), r["v"]) for r in rows if r["family"] == "fog_diffuse"})
    out = []
    for td, vd in dr:
        m = [tD for tD, vD in df if abs(vd - vD) / vD < 1e-3]
        if not m:
            continue
        tD = m[0]
        try:
            raw = [b[("fog_drift", td, models[0], s)]["err_frozen"] - b[("fog_diffuse", tD, models[1], s)]["err_frozen"] for s in seeds]
            orc = [b[("fog_drift", td, models[0], s)]["err_oracle"] - b[("fog_diffuse", tD, models[1], s)]["err_oracle"] for s in seeds]
        except KeyError:
            continue
        out.append(dict(v=vd, t_drift=td, t_diffuse=tD, seeds=seeds, d_raw=raw, d_raw_mean=mean(raw),
                        oracle_diff=mean(orc), oracle_diff_seed_max=max(abs(x) for x in orc),
                        err_drift=mean([b[("fog_drift", td, models[0], s)]["err_frozen"] for s in seeds]),
                        err_diffuse=mean([b[("fog_diffuse", tD, models[1], s)]["err_frozen"] for s in seeds])))
    return out


F["c100_fc_pairs"] = _fc_pairs(_oct3("c100-fog-common-s*/output/cifar100_c_results_c100_fog_common_seed*.jsonl"))
for _key, _pat in (("c100", "c100-fog-common-s*/output/cifar100_c_results_c100_fog_common_seed*.jsonl"),
                   ("tin", "tin-fog-common-s*/output/tin_results_tin_fog_common_seed*.jsonl")):
    _rr = _oct3(_pat)
    _vd = [r["v"] for r in _rr if r["family"] == "fog_diffuse"]
    _cl = [r["err_frozen"] for r in _rr if r["model"] == "clean"            # matched-pair points only
           and any(abs(r["v"] - v) / v < 1e-3 for v in _vd)]
    if _cl:
        F[f"{_key}_clean_err_min"], F[f"{_key}_clean_err_max"] = min(_cl), max(_cl)
F["tin_fc_pairs"] = _fc_pairs(_oct3("tin-fog-common-s*/output/tin_results_tin_fog_common_seed*.jsonl"))
CB = _oct3("c100-blur-s*/output/cifar100_c_results_c100_blur_seed*.jsonl")
if CB:
    cb = {}
    for fam in ("gauss_blur", "blurfft_s15_a01"):
        for t in sorted({r["t"] for r in CB if r["family"] == fam}):
            rs = [r for r in CB if r["family"] == fam and r["t"] == t]
            cb[f"{fam}@{t}"] = dict(n=len(rs), frozen=_m3([r["err_frozen"] for r in rs]), oracle=_m3([r["err_oracle"] for r in rs]),
                                    bn_removes=_m3([r["err_frozen"] - r["err_bnadapt"] for r in rs]),
                                    tent_removes=_m3([r["err_frozen"] - r["err_tent"] for r in rs]))
    F["c100_blur"] = cb
    if "gauss_blur@2.0" in cb and "blurfft_s15_a01@2.0" in cb:
        F["c100_blur_ratio_bn"] = cb["gauss_blur@2.0"]["bn_removes"][0] / cb["blurfft_s15_a01@2.0"]["bn_removes"][0]


# --- certified slack in the dissipative families, every fixed classifier as a witness
# All fog and noise points are isotropic, so by Proposition 1 and Corollary 1 the Bayes
# risk is a non-decreasing function of the equivalent variance v alone. Any fixed
# classifier's test error at a point q bounds R*(p) for every p with v(p) <= v(q).
# Witnesses: oracles, frozen models, the separate and joint models (no adaptation
# method: its predictions depend on the test batch). Bonferroni over all witnesses.
def _vfog(t, a, s_):
    return s_ ** 2 * math.expm1(2 * a * t) / (2 * a)


# the main fog family's parameters, from the script that ran it (a = 1: decay = exp(-t))
_bn_src = open(os.path.join(BASE, "kaggle_cifar10_c_bnadapt/cifar10_c_bnadapt_kaggle.py")).read()
assert "decay = np.exp(-t)" in _bn_src
_MAIN_FOG = (1.0, float(_re0.search(r"^FOG_SIGMA = ([\d.]+)", _bn_src, _re0.M).group(1)))
_pts = []
for s_ in (1, 2, 3):
    for (f, t), rs in R.items():
        if f not in ("gauss_noise", "fog_beer_lambert") or s_ not in rs:
            continue
        v = 2 * t if f == "gauss_noise" else _vfog(t, *_MAIN_FOG)
        _pts.append((f"main/{f}", "oracle", v, s_, rs[s_]["err_oracle"]))
        _pts.append((f"main/{f}/frozen", "witness", v, s_, rs[s_]["err_frozen"]))
for (f, t), rs in A.items():
    a_, s__ = PAR[f][0], PAR[f][1]
    for r in rs:
        _pts.append((f"abl/{f}", "oracle", _vfog(t, a_, s__), r["seed"], r["err_oracle"]))
        _pts.append((f"abl/{f}/frozen", "witness", _vfog(t, a_, s__), r["seed"], r["err_frozen"]))
for r in FC:
    _pts.append((f"fc/{r['family']}/{r['model']}", "witness", r["v"], r["seed"], r["err_frozen"]))
    if r["model"] == "clean":
        _pts.append((f"fc/{r['family']}", "oracle", r["v"], r["seed"], r["err_oracle"]))
for tag, pat in (("joint15", "fog-joint-s*/output/cifar10_c_results_fog_joint_seed*.jsonl"),
                 ("joint200", "fog-joint-full-s*/output/cifar10_c_results_fog_joint_full_seed*.jsonl")):
    for r in _oct3(pat):
        if r["family"] != "clean":
            _pts.append((f"{tag}/{r['family']}", "witness", r["v"], r["seed"], r["err_frozen"]))
_wit = [(src, v, e, s_) for src, k, v, s_, e in _pts]
_zK = _norm.ppf(1 - 0.05 / len(_wit))
cert_all = []
for src, k, v, s_, e in _pts:
    if k != "oracle" or v == 0:
        continue
    u, who, wv, ws = min((e2 + _zK * _se(e2), src2, v2, s2) for src2, v2, e2, s2 in _wit if v2 >= v - 1e-12)
    cert_all.append(dict(oracle=src, v=v, seed=s_, slack=e - _z1 * _se(e) - u, witness=who, witness_v=wv, witness_seed=ws))
_pos = [c for c in cert_all if c["slack"] > 0]
F["cert_all_n"], F["cert_all_pos"], F["cert_all_K"] = len(cert_all), len(_pos), len(_wit)
F["cert_all_z"] = float(_zK)
F["cert_all_expected"] = 0.05 * len(cert_all)
F["cert_all_max"] = max(c["slack"] for c in _pos)
F["cert_all_by_joint200"] = sum(c["witness"].startswith("joint200") for c in _pos)
F["cert_all_by_witness"] = {w: sum(c["witness"] == w for c in _pos) for w in sorted({c["witness"] for c in _pos})}
F["cert_all_by_15epoch"] = sum(not c["witness"].startswith("joint200") for c in _pos)   # every other witness is a 15-epoch model
_o15 = [c for c in _pos if not c["witness"].startswith("joint200")]
F["cert_15epoch_same_v"] = sum(abs(c["witness_v"] - c["v"]) <= 1e-9 * max(c["v"], 1e-12) for c in _o15)
F["cert_15epoch_larger_v"] = sum(c["witness_v"] > c["v"] * (1 + 1e-9) for c in _o15)
F["cert_15epoch_detail"] = [dict(oracle=c["oracle"], v=c["v"], seed=c["seed"], witness=c["witness"], witness_v=c["witness_v"],
                                 witness_seed=c["witness_seed"]) for c in _o15]
_v_train = min(r["v"] for r in F["fc_pairs"])   # the pair at which both members are training severities
_c042 = [c["slack"] for c in _pos if c["oracle"].startswith("fc/") and abs(c["v"] - _v_train) < 1e-6 * _v_train]
F["cert_fc_042_min"], F["cert_fc_042_max"] = min(_c042), max(_c042)

# --- contrast and raw noise variance: the joint and separate models' training grids
# (read from the result rows) and the matched points. Regime parameters come from the
# runner that produced the rows.
import re as _re
_FP = {f: (float(a), float(sg)) for f, a, sg in _re.findall(r'"(fog_\w+)": dict\(a=([\d.]+), sigma=([\d.]+)\)',
                                                           open(os.path.join(BASE, "kaggle_oct3_runner/runner.py")).read())}


def _contrast(fam, t):
    return math.exp(-_FP[fam][0] * t)


def _rawvar(fam, t):
    a, sg = _FP[fam]
    return sg * sg / (2 * a) * (1 - math.exp(-2 * a * t))


_jrows = [r for r in _oct3("fog-joint-full-s*/output/cifar10_c_results_fog_joint_full_seed*.jsonl")
          + _oct3("fog-joint-s*/output/cifar10_c_results_fog_joint_seed*.jsonl") if r["family"] != "clean"]
_grids = {tuple(map(tuple, r["model_train_grid"])) for r in _jrows}
assert len(_grids) == 1, _grids                      # every joint run used the same grid
_grid = sorted(next(iter(_grids)))
F["joint_train_contrasts"] = sorted(_contrast(f, t) for f, t in _grid)
F["joint_train_contrast_min"] = F["joint_train_contrasts"][0]
_gaps = [(b - a, a, b) for a, b in zip(F["joint_train_contrasts"], F["joint_train_contrasts"][1:])]
F["joint_train_contrast_gap"] = list(max(_gaps)[1:])   # the widest interval the model never saw
F["joint_train_rawvar_max"] = max(_rawvar(f, t) for f, t in _grid)
F["matched_contrast"] = {f"{r['v']:.3f}": dict(drift=_contrast("fog_drift", r["t_drift"]),
                                               diffuse=_contrast("fog_diffuse", r["t_diffuse"]))
                         for r in F["fj_pairs"]}
F["matched_rawvar_ratio"] = {f"{r['v']:.3f}": dict(drift=_rawvar("fog_drift", r["t_drift"]) / F["joint_train_rawvar_max"],
                                                   diffuse=_rawvar("fog_diffuse", r["t_diffuse"]) / F["joint_train_rawvar_max"])
                             for r in F["fj_pairs"]}
# the separate models' training grids (fog_common rows record them per model)
_sep = {}
for r in _oct3("fog-common-s*/output/cifar10_c_results_fog_common_seed*.jsonl"):
    if r["model"] in ("drift_matched", "diffuse_own"):
        _sep.setdefault(r["model"], set()).add(tuple(r["model_train_ts"]))
assert all(len(v) == 1 for v in _sep.values()), _sep
F["separate_train_contrasts"] = {m: sorted(_contrast("fog_drift" if m.startswith("drift") else "fog_diffuse", t)
                                           for t in next(iter(v))) for m, v in _sep.items()}

# --- adaptation does not close the matched-pair gap (joint model, 200 epochs)
for _key, _pat in (("fjf", "fog-joint-full-s*/output/cifar10_c_results_fog_joint_full_seed*.jsonl"),
                   ("fjf100", "c100-fog-joint-full-s*/output/cifar100_c_results_c100_fog_joint_full_seed*.jsonl")):
    _rr = [r for r in _oct3(_pat) if r["family"] != "clean"]
    _out = []
    for r in sorted(F[_key + "_pairs"], key=lambda q: q["v"]):
        a_ = [q for q in _rr if q["family"] == "fog_drift" and abs(q["t"] - r["t_drift"]) < 1e-3]
        b_ = [q for q in _rr if q["family"] == "fog_diffuse" and q["t"] == r["t_diffuse"]]
        _out.append(dict(v=r["v"], bn_gap=mean([q["err_bnadapt"] for q in a_]) - mean([q["err_bnadapt"] for q in b_]),
                         tent_gap=mean([q["err_tent"] for q in a_]) - mean([q["err_tent"] for q in b_]),
                         bn_drift=mean([q["err_bnadapt"] for q in a_]), frozen_drift=r["err_drift"]))
    F[_key + "_adapt"] = _out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# --- first-layer padding test (rule: oct5/PLAN.md, written before the runs): the saved
# 200-epoch joint models evaluated with the first convolution padded in five ways
PAD_KINDS = ("zeros", "reflect", "circular", "haze", "standardize")


def _pad_pairs(rows):
    out = {}
    for r in rows:
        if r["family"] != "clean":
            out.setdefault(round(r["v"], 3), {}).setdefault(r["seed"], {})[r["family"]] = r
    return out


def _pad_judge(by_seed, key):
    ss = sorted(by_seed)
    raw = [by_seed[s]["fog_drift"][f"err_{key}"] - by_seed[s]["fog_diffuse"][f"err_{key}"] for s in ss]
    inv = [by_seed[s]["fog_drift"][f"err_{key}_inv"] - by_seed[s]["fog_diffuse"][f"err_{key}_inv"] for s in ss]
    pos = (len(raw) == 3 and (all(x > 0 for x in raw) or all(x < 0 for x in raw))
           and all(abs(a) > abs(b) for a, b in zip(raw, inv)) and abs(mean(raw)) > sd(raw))
    return dict(gap=mean(raw), gaps=raw, floor=max(abs(x) for x in inv), positive=pos)


for _tag, _pat in (("c10", "padding-test-s*/output/cifar10_c_results_padding_test_seed*.jsonl"),
                   ("c100", "c100-padding-test-s*/output/cifar100_c_results_c100_padding_test_seed*.jsonl")):
    _rows = _oct3(_pat)
    if len({r["seed"] for r in _rows}) < 3:
        continue
    res = {}
    for k in PAD_KINDS:
        rk = [r for r in _rows if r["pad"] == k]
        prs = _pad_pairs(rk)
        res[k] = dict(clean_frozen=mean([r["err_frozen"] for r in rk if r["family"] == "clean"]),
                      clean_bnadapt=mean([r["err_bnadapt"] for r in rk if r["family"] == "clean"]),
                      **{key: {f"{v:.3f}": _pad_judge(prs[v], key) for v in sorted(prs)} for key in ("frozen", "bnadapt")})
    F[f"pad_{_tag}"] = res
    _out = [v for v in res["zeros"]["frozen"] if float(v) > 0.05]
    _pos2 = [v for v in _out if res["reflect"]["frozen"][v]["positive"] and res["circular"]["frozen"][v]["positive"]]
    F[f"pad_{_tag}_outer_positive"] = {k: sum(res[k]["frozen"][v]["positive"] for v in _out) for k in PAD_KINDS}
    # share of the frozen gap that the first layer's zero padding accounts for, over the
    # outer pairs that pass under both commuting paddings
    _z = sum(res["zeros"]["frozen"][v]["gap"] for v in _pos2)
    _c = mean([sum(res[k]["frozen"][v]["gap"] for v in _pos2) for k in ("reflect", "circular")])
    F[f"pad_{_tag}_zero_share"] = 1 - _c / _z
    F[f"pad_{_tag}_commuting_outer"] = [min(res[k]["frozen"][v]["gap"] for k in ("reflect", "circular") for v in _pos2),
                                        max(res[k]["frozen"][v]["gap"] for k in ("reflect", "circular") for v in _pos2)]
    F[f"pad_{_tag}_bn_commuting_max"] = max(abs(res[k]["bnadapt"][v]["gap"]) for k in ("reflect", "circular", "haze")
                                            for v in res[k]["bnadapt"])
    F[f"pad_{_tag}_bn_zeros"] = [min(res["zeros"]["bnadapt"][v]["gap"] for v in res["zeros"]["bnadapt"]),
                                 max(res["zeros"]["bnadapt"][v]["gap"] for v in res["zeros"]["bnadapt"])]


out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.json")
json.dump(clean(F), open(out, "w"), indent=1, sort_keys=True)
print(f"{len(F)} top-level facts written to {out}")
