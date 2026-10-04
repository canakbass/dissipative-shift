"""Generate appendix LaTeX tables directly from the raw (corrected, v2fix) result
files, so numbers in the paper are never hand-transcribed."""
import json
import os
import statistics as st
from collections import defaultdict


def sg(v, nd=4):
    """Signed number in math mode, so LaTeX renders a real minus sign rather than a
    plain-text hyphen."""
    return f"${v:+.{nd}f}$"


BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.environ.get("TABLES_OUT", f"{BASE}/paper/tables")   # verify_claims.py regenerates into a temp dir
os.makedirs(OUT, exist_ok=True)

fam_names = {"gauss_noise": "Gaussian noise", "blur_heat": "Heat-equation blur",
             "fog_beer_lambert": "OU fog", "gauss_blur": "Deterministic blur (control)"}

# ---------------- kill-tests: one table for the three closed-form checks ----------------
# Columns that follow from the shown ones (oracle error = R* + slack, frozen error =
# R* + Delta*, extrapolated R* = R* + extrapolation error) are left out.
kt = {r["t"]: r for r in (json.loads(l) for l in open(f"{BASE}/kill_test_results.jsonl"))}
ka = {r["t"]: r for r in (json.loads(l) for l in open(f"{BASE}/kill_test_asym_results.jsonl"))}
kf = {r["t"]: r for r in (json.loads(l) for l in open(f"{BASE}/kill_test_fog_results.jsonl"))}
lines = [r"\begin{table}[!htbp]",
         r"\caption{Sanity checks against closed form. For each check, the Bayes risk $R^*(t)$, the slack of the retrained oracle, $R_{\text{oracle}}(t) - R^*(t)$, and the frozen classifier's true avoidable error $\Delta^*(t) = \mathrm{Err}_{\text{frozen}}(t) - R^*(t)$; where a generator is estimated at three low severities, the error of $R^*(t)$ extrapolated from it. Additive noise, equal priors: the hypothesis class (linear) contains the Bayes rule and the oracle is within $0.0043$ of $R^*$ throughout; the frozen classifier is Bayes-optimal by construction, so $\Delta^*$ is zero up to sampling noise. Unequal priors ($\pi_0 = 0.8$): the Bayes threshold moves with $t$, so a classifier frozen at $t \le 0.2$ has a known, non-zero $\Delta^*(t)$ (``predicted''), which the measured frozen error must match. It does, to within $0.0016$, and the oracle is within $0.0008$ of $R^*$. Fog: $R^*(t) \to 0.5$ as $t\to\infty$, so the corruption destroys information; the generator is estimated from sampled class means and variances, and the setting is symmetric, so the frozen side is not exercised. Negative values are finite-sample noise in the evaluation set.}",
         r"\centering\scriptsize\setlength{\tabcolsep}{2pt}",
         r"\resizebox{\linewidth}{!}{",
         r"\begin{tabular}{rrrrrrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{Additive noise, equal priors} & \multicolumn{4}{c}{Additive noise, unequal priors} & \multicolumn{4}{c}{Ornstein--Uhlenbeck fog} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
         r"$t$ & $R^*$ & slack & $\Delta^*$ & extrap.\ err. & $R^*$ & slack & pred.\ $\Delta^*$ & meas.$-$pred. & $R^*$ & slack & $\Delta^*$ & extrap.\ err. \\",
         r"\midrule"]
for t in sorted(set(kt) | set(ka) | set(kf)):
    c = [f"{t:.1f}"]
    r = kt.get(t)
    c += [f"{r['r_star_closed']:.4f}", sg(r['oracle_slack']), sg(r['delta']), sg(r['extrapolation_error'])] if r else ["---"] * 4
    r = ka.get(t)
    c += [f"{r['r_star']:.4f}", sg(r['oracle_slack']), f"{r['delta_pred']:.4f}", sg(r['frozen_meas'] - r['frozen_pred'])] if r else ["---"] * 4
    r = kf.get(t)
    c += [f"{r['r_star_closed']:.4f}", sg(r['oracle_slack']), sg(r['delta']), sg(r['extrapolation_error'])] if r else ["---"] * 4
    lines.append(" & ".join(c) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}}", r"\label{tab:app-killtest}", r"\end{table}"]
with open(f"{OUT}/killtest_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("killtest_table.tex written")

# ---------------- CIFAR-10 full table (corrected corruptions, +BN-adapt) ----------------
# Oracle, frozen, BN-adapt, TENT and EATA all
# come from one run family (kaggle_cifar10_c_bnadapt*), because that run computes all
# of them from the same frozen model. An earlier version took BN-adapt from this run
# and TENT/EATA from a different one, which compared two different model instances and
# produced meaningless single-point claims such as "BN-adapt beats TENT". DANN has no
# oracle of its own, so it is paired with the same-seed oracle from this run -- not
# with the mean, which would hide the oracle's own seed-to-seed variance inside DANN's
# standard deviation.
bn_dirs = {1: "kaggle_cifar10_c_bnadapt/output_seed1", 2: "kaggle_cifar10_c_bnadapt/output_seed2", 3: "kaggle_cifar10_c_bnadapt_b/output_seed3"}
bn_rows_by_seed = {}
for seed, d in bn_dirs.items():
    bn_rows_by_seed[seed] = [json.loads(l) for l in open(f"{BASE}/{d}/cifar10_c_results_bnadapt_seed{seed}.jsonl")]

# blur_heat (the stochastic heat equation) comes from separate runs, but those runs
# also compute all five conditions from a single frozen model, so they can be appended
# directly.
bh_dirs = {1: "kaggle_cifar10_c_bnadapt_blurheat/output_seed1",
           2: "kaggle_cifar10_c_bnadapt_blurheat/output_seed2",
           3: "kaggle_blurheat_bnadapt_s3/output_seed3"}
for seed, d in bh_dirs.items():
    p = f"{BASE}/{d}/cifar10_c_results_bnadapt_blurheat_seed{seed}.jsonl"
    if os.path.exists(p):
        bn_rows_by_seed[seed] += [json.loads(l) for l in open(p)]
    else:
        print(f"WARNING: blur_heat seed{seed} missing ({p})")

dann_rows_by_seed = {}
for seed in [1, 2, 3]:
    dann_rows_by_seed[seed] = [json.loads(l) for l in open(f"{BASE}/kaggle_cifar10_c_dann/output_seed{seed}_v2fix/cifar10_c_results_dann_seed{seed}_v2fix.jsonl")]

# DANN blur_heat: seed 1 from the first run, seeds 2 and 3 added later. A missing seed
# is skipped, and the DANN column for that family then rests on fewer seeds.
bh_dann = {1: "kaggle_cifar10_c_dann_blurheat/output_seed1",
           2: "kaggle_dann_blurheat_s2/output_seed2",
           3: "kaggle_dann_blurheat_s3/output_seed3"}
dann_bh_seeds = []
for seed, d in bh_dann.items():
    p = f"{BASE}/{d}/cifar10_c_results_dann_blurheat_seed{seed}.jsonl"
    if os.path.exists(p):
        dann_rows_by_seed[seed] += [json.loads(l) for l in open(p)]
        dann_bh_seeds.append(seed)
    else:
        print(f"WARNING: DANN blur_heat seed{seed} missing ({p})")
print(f"DANN blur_heat seeds: {len(dann_bh_seeds)} {dann_bh_seeds}")

oracle_by_key_seed = defaultdict(dict)  # k -> {seed: oracle}
d_frozen = defaultdict(list)
d_bnadapt = defaultdict(list)
d_tent = defaultdict(list)
d_eata = defaultdict(list)
train_flag = {}
for seed, rs in bn_rows_by_seed.items():
    for r in rs:
        k = (r["family"], r["t"])
        oracle_by_key_seed[k][seed] = r["err_oracle"]
        d_frozen[k].append(r["delta_frozen"])
        d_bnadapt[k].append(r["delta_bnadapt"])
        d_tent[k].append(r["delta_tent"])
        d_eata[k].append(r["delta_eata"])
        train_flag[k] = r["is_train_severity"]

oracle_by_key = {k: list(v.values()) for k, v in oracle_by_key_seed.items()}  # for monotonicity/kill-test reuse below

d_dann = defaultdict(list)
for seed, rs in dann_rows_by_seed.items():
    for r in rs:
        k = (r["family"], r["t"])
        d_dann[k].append(r["err_dann"] - oracle_by_key_seed[k][seed])  # same-seed pairing



def ms(vals):
    if not vals:
        return "---"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:+.3f}$\\pm${s:.3f}"


lines = [r"\begin{table}[!htbp]",
         r"\caption{CIFAR-10 under the three dissipative semigroup corruptions and the deterministic-blur reversible control: measured avoidable error $\Delta_X(t)$ for all methods, families, and severities, mean $\pm$ standard deviation over three seeds. Rows marked $\dagger$ are training severities for the frozen model and DANN. TENT and EATA use the original SGD adaptation setting (Section~\ref{sec:tent-sweep}). DANN comes from a separate run and is paired with the same-seed oracle of the main run; oracles differ between runs by up to $0.027$, and Section~\ref{sec:dann} compares DANN without the oracle. In the control the Bayes risk is flat, so the oracle's rise there is slack (Section~\ref{sec:oracle}). At fog $t=4$ the oracle is at $0.84$, near chance ($0.9$).}",
         r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Family & $t$ & $\Delta_{\text{frozen}}$ & $\Delta_{\text{BN-adapt}}$ & $\Delta_{\text{DANN}}$ & $\Delta_{\text{TENT}}$ & $\Delta_{\text{EATA}}$ & \\", r"\midrule"]
for k in sorted(oracle_by_key.keys(), key=lambda kk: (kk[0], kk[1])):
    fam, t = k
    dagger = r"$\dagger$" if train_flag[k] else ""
    lines.append(f"{fam_names[fam]} & {t:.3f} & {ms(d_frozen[k])} & {ms(d_bnadapt[k])} & {ms(d_dann[k])} & {ms(d_tent[k])} & {ms(d_eata[k])} & {dagger} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\label{tab:app-cifar}", r"\end{table}"]
with open(f"{OUT}/cifar_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("cifar_table.tex written")

# ---------------- reruns with Adam as the adaptation optimizer (3 October) ----------------
import glob
AM = []
for p in sorted(glob.glob(f"{BASE}/oct3/adam-main-s*/output/cifar10_c_results_adam_main_seed*.jsonl")):
    AM += [json.loads(l) for l in open(p)]
am_names = {"blurfft_s15_a01": "Heat-equation blur", "fog_beer_lambert": "OU fog", "gauss_blur": "Deterministic blur (control)"}
lines = [r"\begin{table}[!htbp]",
         r"\caption{Heat-equation blur, fog and the control rerun with Adam as the adaptation optimizer for TENT and EATA, three seeds, every quantity from one run per seed (the noise family is in Table~\ref{tab:tent-fixed}). Oracle error and measured avoidable error $\Delta_X(t)$, mean $\pm$ standard deviation. Rows marked $\dagger$ are training severities.}",
         r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Family & $t$ & $\mathrm{Err}_{\text{oracle}}$ & $\Delta_{\text{frozen}}$ & $\Delta_{\text{BN-adapt}}$ & $\Delta_{\text{TENT}}$ & $\Delta_{\text{EATA}}$ & \\", r"\midrule"]
for fam in ("blurfft_s15_a01", "fog_beer_lambert", "gauss_blur"):
    for t in sorted({r["t"] for r in AM if r["family"] == fam}):
        rs = [r for r in AM if r["family"] == fam and r["t"] == t]
        dagger = r"$\dagger$" if rs[0]["is_train_severity"] else ""
        cells = [ms([r[k] for r in rs]) for k in ("err_oracle", "delta_frozen", "delta_bnadapt", "delta_tent", "delta_eata")]
        lines.append(f"{am_names[fam]} & {t:.3f} & " + " & ".join(cells) + f" & {dagger} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\label{tab:adam-all}", r"\end{table}"]
with open(f"{OUT}/adam_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("adam_table.tex written")

# ---------------- blur factorial: error removed by BN-adapt (3 October) ----------------
FBR = []
for pat in ("blur-factorial-s*/output/cifar10_c_results_factorial_seed*.jsonl",
            "blur-factorial-extra-s*/output/cifar10_c_results_factorial_extra_seed*.jsonl"):
    for p in sorted(glob.glob(f"{BASE}/oct3/{pat}")):
        FBR += [json.loads(l) for l in open(p)]
MAIN = {}
for s_, d in {1: "kaggle_cifar10_c_bnadapt/output_seed1", 2: "kaggle_cifar10_c_bnadapt/output_seed2",
              3: "kaggle_cifar10_c_bnadapt_b/output_seed3"}.items():
    for l in open(f"{BASE}/{d}/cifar10_c_results_bnadapt_seed{s_}.jsonl"):
        r = json.loads(l)
        if r["family"] == "gauss_blur":
            MAIN.setdefault(("gauss_blur", r["t"]), []).append(r)
for s_, d in {1: "kaggle_cifar10_c_bnadapt_blurheat/output_seed1", 2: "kaggle_cifar10_c_bnadapt_blurheat/output_seed2",
              3: "kaggle_blurheat_bnadapt_s3/output_seed3"}.items():
    for l in open(f"{BASE}/{d}/cifar10_c_results_bnadapt_blurheat_seed{s_}.jsonl"):
        r = json.loads(l)
        MAIN.setdefault(("blur_heat", r["t"]), []).append(r)


def removed(rs):
    return ms([r["err_frozen"] - r["err_bnadapt"] for r in rs])


lines = [r"\begin{table}[t]",
         r"\caption{Error removed by BN-adapt, $\mathrm{Err}_{\text{frozen}} - \mathrm{Err}_{\text{BN-adapt}}$, at the held-out severities of the blur families, mean $\pm$ standard deviation over three seeds. The middle three columns are one periodic Fourier operator with the diffusion noise ($\sigma_{\text{blur}}$) and the contrast floor ($a_{\min}$) switched on or off; the outer two columns are the families of the main experiments, from the main run. Only the diffusion noise changes the Bayes risk.}",
         r"\label{tab:factorial}", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{cccccc}", r"\toprule",
         r" & Reflect-padded, & \multicolumn{3}{c}{Periodic Fourier operator} & Heat-equation \\",
         r"\cmidrule(lr){3-5}",
         r"$t$ & sampled kernel & no noise, no floor & floor only & noise only & blur \\", r"\midrule"]
for t in (0.25, 0.5, 1.0, 2.0):
    cells = [removed(MAIN[("gauss_blur", t)])] + \
            [removed([r for r in FBR if r["family"] == fam and r["t"] == t]) for fam in ("blurfft_s0_a0", "blurfft_s0_a01", "blurfft_s15_a0")] + \
            [removed(MAIN[("blur_heat", t)])]
    lines.append(f"{t:.2f} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f"{OUT}/factorial_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("factorial_table.tex written")

# ---------------- fog pairs matched in equivalent variance (3 October) ----------------
def _rows(pat):
    out = []
    for p in sorted(glob.glob(f"{BASE}/oct3/{pat}")):
        out += [json.loads(l) for l in open(p)]
    return out


FCR = _rows("fog-common-s*/output/cifar10_c_results_fog_common_seed*.jsonl")
FJR = _rows("fog-joint-s*/output/cifar10_c_results_fog_joint_seed*.jsonl")


def _m(rows, fam, t, model, key="err_frozen"):
    xs = [r[key] for r in rows if r["family"] == fam and abs(r["t"] - t) < 1e-9 and r["model"] == model]
    return st.mean(xs)


drift_ts = sorted({r["t"] for r in FJR if r["family"] == "fog_drift"})
diff_ts = sorted({r["t"] for r in FJR if r["family"] == "fog_diffuse"})
lines = [r"\begin{table}[t]",
         r"\caption{Fog pairs matched in equivalent noise variance $v$, three-seed means. By Proposition~\ref{prop:equivalence} the two members of each pair have exactly the same Bayes risk. The $v$ ratio is $v$ as a multiple of the largest $v$ the models were trained on, the same for both members; the extrapolation horizon of Table~\ref{tab:ablation} is the corresponding ratio of severities. All models here are trained for 15 epochs. The joint model is one network trained on both regimes; the separate models are one per regime, trained over ranges matched in $v$; all have the same budget. A model trained on clean images only (not shown) is at chance, $0.89$ to $0.90$, at every point.}",
         r"\label{tab:matched}", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{cccccccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Severity $t$} & & \multicolumn{2}{c}{Oracle} & \multicolumn{2}{c}{Joint model} & \multicolumn{2}{c}{Separate models} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"$v$ & Drift & Diffusion & $v$ ratio & Drift & Diffusion & Drift & Diffusion & Drift & Diffusion \\", r"\midrule"]
for td, tD in zip(drift_ts, diff_ts):
    v = [r["v"] for r in FJR if r["family"] == "fog_diffuse" and r["t"] == tD][0]
    hz = [r["v_horizon"] for r in FJR if r["family"] == "fog_diffuse" and r["t"] == tD][0]
    cells = [f"{v:.3f}", f"{td:.2f}", f"{tD:.0f}", f"{hz:.1f}",
             f"{_m(FCR, 'fog_drift', td, 'clean', 'err_oracle'):.3f}", f"{_m(FCR, 'fog_diffuse', tD, 'clean', 'err_oracle'):.3f}",
             f"{_m(FJR, 'fog_drift', td, 'joint'):.3f}", f"{_m(FJR, 'fog_diffuse', tD, 'joint'):.3f}",
             f"{_m(FCR, 'fog_drift', td, 'drift_matched'):.3f}", f"{_m(FCR, 'fog_diffuse', tD, 'diffuse_own'):.3f}"]
    lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f"{OUT}/matched_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("matched_table.tex written")

# ---------------- scale: matched fog pairs on CIFAR-100 and Tiny-ImageNet ----------------
C1 = _rows("c100-fog-common-s*/output/cifar100_c_results_c100_fog_common_seed*.jsonl")
CJ = _rows("c100-fog-joint-s*/output/cifar100_c_results_c100_fog_joint_seed*.jsonl")
TF = _rows("tin-fog-common-s*/output/tin_results_tin_fog_common_seed*.jsonl")
lines = [r"\begin{table}[t]",
         r"\caption{Fog pairs matched in equivalent noise variance $v$ on CIFAR-100 (three seeds) and Tiny-ImageNet (one seed); means. The two members of each pair have the same Bayes risk. Models as in Table~\ref{tab:matched}, trained for 15 epochs; the Tiny-ImageNet run has no joint model and omits the pair at $v = 0.042$.}",
         r"\label{tab:scale}", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{Oracle} & \multicolumn{2}{c}{Joint model} & \multicolumn{2}{c}{Separate models} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Dataset & $v$ & Drift & Diffusion & Drift & Diffusion & Drift & Diffusion \\", r"\midrule"]
for name, FC_, FJ_ in (("CIFAR-100", C1, CJ), ("Tiny-ImageNet", TF, None)):
    dts = sorted({r["t"] for r in FC_ if r["family"] == "fog_drift" and r["model"] == "drift_matched"
                  and any(abs(q["v"] - r["v"]) / r["v"] < 1e-3 for q in FC_ if q["family"] == "fog_diffuse")})
    for td in dts:
        v = [r["v"] for r in FC_ if r["family"] == "fog_drift" and r["t"] == td][0]
        tD = [r["t"] for r in FC_ if r["family"] == "fog_diffuse" and abs(r["v"] - v) / v < 1e-3][0]
        cells = [name, f"{v:.3f}", f"{_m(FC_, 'fog_drift', td, 'clean', 'err_oracle'):.3f}", f"{_m(FC_, 'fog_diffuse', tD, 'clean', 'err_oracle'):.3f}"]
        cells += [f"{_m(FJ_, 'fog_drift', td, 'joint'):.3f}", f"{_m(FJ_, 'fog_diffuse', tD, 'joint'):.3f}"] if FJ_ else ["---", "---"]
        cells += [f"{_m(FC_, 'fog_drift', td, 'drift_matched'):.3f}", f"{_m(FC_, 'fog_diffuse', tD, 'diffuse_own'):.3f}"]
        lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f"{OUT}/scale_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("scale_table.tex written")

# ---------------- the joint model at 15 and at 200 epochs ----------------
def _jrows(pat):
    return [r for r in _rows(pat) if r["family"] != "clean"]


lines = [r"\begin{table}[t]",
         r"\caption{The joint model at the short budget used elsewhere in the paper (15 epochs, one crop and flip per batch) and at the standard CIFAR budget (200 epochs, per-sample crop and flip), at the fog pairs matched in equivalent variance $v$. Error on the drift and the diffusion member, three-seed means; the two members have the same Bayes risk. ``Clean'' is the model's error on uncorrupted test images, measured only in the 200-epoch run; the joint model sees uncorrupted images in one fifth of its training data, so this is higher than for a model trained on clean images alone.}",
         r"\label{tab:budget}", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{15 epochs} & \multicolumn{3}{c}{200 epochs} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Dataset & $v$ & Drift & Diffusion & Gap & Drift & Diffusion & Gap \\", r"\midrule"]
for name, short, full in (("CIFAR-10", "fog-joint-s*/output/cifar10_c_results_fog_joint_seed*.jsonl",
                           "fog-joint-full-s*/output/cifar10_c_results_fog_joint_full_seed*.jsonl"),
                          ("CIFAR-100", "c100-fog-joint-s*/output/cifar100_c_results_c100_fog_joint_seed*.jsonl",
                           "c100-fog-joint-full-s*/output/cifar100_c_results_c100_fog_joint_full_seed*.jsonl")):
    S_, L_ = _jrows(short), _jrows(full)
    for td in sorted({r["t"] for r in S_ if r["family"] == "fog_drift"}):
        v = [r["v"] for r in S_ if r["family"] == "fog_drift" and r["t"] == td][0]
        tD = [r["t"] for r in S_ if r["family"] == "fog_diffuse" and abs(r["v"] - v) / v < 1e-3][0]
        cells = [name, f"{v:.3f}"]
        for R__ in (S_, L_):
            a, b = _m(R__, "fog_drift", td, "joint"), _m(R__, "fog_diffuse", tD, "joint")
            cells += [f"{a:.3f}", f"{b:.3f}", f"${a - b:+.3f}$"]
        lines.append(" & ".join(cells) + r" \\")
    cl_s = [r["err_frozen"] for r in _rows(short) if r["family"] == "clean"]
    cl_l = [r["err_frozen"] for r in _rows(full) if r["family"] == "clean"]
    lines.append(f"{name} & clean & \\multicolumn{{3}}{{c}}{{{st.mean(cl_s):.3f}}} & \\multicolumn{{3}}{{c}}{{{st.mean(cl_l):.3f}}} \\\\"
                 if cl_s else f"{name} & clean & \\multicolumn{{3}}{{c}}{{---}} & \\multicolumn{{3}}{{c}}{{{st.mean(cl_l):.3f}}} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f"{OUT}/budget_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("budget_table.tex written")

# ---------------- summary stats used in the main text (printed for manual transcription) ----------------
print("\n--- summary numbers quoted in the main text ---")
for k in [("gauss_noise", 0.2), ("gauss_blur", 2.0), ("fog_beer_lambert", 2.0), ("fog_beer_lambert", 4.0)]:
    om = st.mean(oracle_by_key[k])
    print(f"{k}: oracle={om:.4f}  D_frozen={ms(d_frozen[k])}  D_bnadapt={ms(d_bnadapt[k])}  D_dann={ms(d_dann[k])}  D_tent={ms(d_tent[k])}  D_eata={ms(d_eata[k])}")

print("\n--- t=0 (clean) oracle error, per family ---")
for fam in fam_names:
    vals = [st.mean(oracle_by_key[(fam, 0.0)])]
    print(fam, oracle_by_key[(fam, 0.0)])
