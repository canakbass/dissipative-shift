"""Summarizes whatever Oct-3 outputs have been downloaded into oct3/<kernel>/output/.
Missing kernels are skipped. Run:  python oct3/summarize.py
The numbers the paper quotes go through paper/facts.py afterwards; this is for reading."""
import glob
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, pattern))):
        out += [json.loads(l) for l in open(p)]
    return out


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "   ---       "
    return f"{st.mean(xs):.3f}+/-{st.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f} (n=1)"


def fog_common(tag, pattern):
    R = rows(pattern)
    if not R:
        return
    seeds = sorted({r["seed"] for r in R})
    print(f"\n=== {tag}: seeds {seeds} ===")
    by = defaultdict(list)
    for r in R:
        by[(r["family"], round(r["t"], 4), r["model"])].append(r)
    pts = sorted({(f, t) for f, t, _ in by})
    print(f"{'family':<11} {'t':>6} {'v':>7} {'model':<13} {'oracle':>13} {'frozen':>13} {'frozen_inv':>13} "
          f"{'bn':>13} {'bn_inv':>13} {'tent':>13} {'v_horizon':>9}")
    for f, t in pts:
        for m in ("clean", "drift_own", "drift_matched", "diffuse_own"):
            rs = by.get((f, t, m))
            if not rs:
                continue
            vh = rs[0]["v_horizon"]
            print(f"{f:<11} {t:>6} {rs[0]['v']:>7.4f} {m:<13} " + " ".join(
                f"{ms([r[k] for r in rs]):>13}" for k in ("err_oracle", "err_frozen", "err_frozen_inv", "err_bnadapt",
                                                         "err_bnadapt_inv", "err_tent"))
                  + (f" {vh:>9.1f}" if vh else f" {'---':>9}"))
    # the matched pairs: same v, so by Proposition 1 the same Bayes risk
    diff_ts = sorted(t for f, t in pts if f == "fog_diffuse")
    drift_ts = sorted(t for f, t in pts if f == "fog_drift")
    print("\nmatched-v pairs (drift - diffusion), per-seed mean:")
    for td in drift_ts:
        vd = by[("fog_drift", td, "clean")][0]["v"]
        match = [tD for tD in diff_ts if abs(by[("fog_diffuse", tD, "clean")][0]["v"] - vd) / vd < 1e-3]
        if not match:
            continue
        tD = match[0]
        for md, mD in (("clean", "clean"), ("drift_matched", "diffuse_own")):
            a, b = by.get(("fog_drift", td, md)), by.get(("fog_diffuse", tD, mD))
            if not a or not b:
                continue
            for k in ("err_oracle", "err_frozen", "err_frozen_inv", "err_bnadapt", "err_tent"):
                d = [x[k] - y[k] for x in a for y in b if x["seed"] == y["seed"]]
                print(f"  drift t={td} vs diffusion t={tD} (v={vd:.4f})  {md}/{mD:<13} {k:<15} {ms(d)}")


def decide_fog_common(R, chance, n_test, label):
    """The decision rule written on 2 Oct (rules/2026-10-02_matched_fog_pairs.md), before any result existed.
    Returns the outcome string; prints the per-pair evidence."""
    by = {(r["family"], round(r["t"], 4), r["model"], r["seed"]): r for r in R}
    seeds = sorted({r["seed"] for r in R})
    drift = sorted({(t, by[("fog_drift", t, "clean", s)]["v"]) for (f, t, m, s) in by if f == "fog_drift" and m == "clean"})
    diff = sorted({(t, by[("fog_diffuse", t, "clean", s)]["v"]) for (f, t, m, s) in by if f == "fog_diffuse" and m == "clean"})
    pairs = [(td, tD) for td, vd in drift for tD, vD in diff if abs(vd - vD) / vD < 1e-3]
    print(f"\n--- decision rule ({label}): {len(pairs)} matched pairs, seeds {seeds} ---")
    if len(seeds) < 3:
        print("  EKSIK: fewer than three seeds; no decision")
        return "EKSIK"
    code_ok, results = True, {}
    for test, (md, mD) in (("1: clean model", ("clean", "clean")), ("2: drift_matched vs diffuse_own", ("drift_matched", "diffuse_own"))):
        npos, nceil, nok = 0, 0, 0
        for td, tD in pairs:
            try:
                raw = [by[("fog_drift", td, md, s)]["err_frozen"] - by[("fog_diffuse", tD, mD, s)]["err_frozen"] for s in seeds]
                inv = [by[("fog_drift", td, "clean", s)]["err_frozen_inv"] - by[("fog_diffuse", tD, "clean", s)]["err_frozen_inv"] for s in seeds]
                e_d = st.mean(by[("fog_drift", td, md, s)]["err_frozen"] for s in seeds)
                e_D = st.mean(by[("fog_diffuse", tD, mD, s)]["err_frozen"] for s in seeds)
            except KeyError:
                continue
            nok += 1
            if md == "clean":   # the inverse-map arm is a code check, not evidence
                for s in seeds:
                    p = st.mean([by[("fog_drift", td, "clean", s)]["err_frozen_inv"], by[("fog_diffuse", tD, "clean", s)]["err_frozen_inv"]])
                    se = (2 * p * (1 - p) / n_test) ** 0.5
                    if abs(inv[seeds.index(s)]) > 4 * max(se, 1e-12):
                        code_ok = False
                        print(f"  CODE CHECK FAILED at drift t={td} / diffusion t={tD}, seed {s}: "
                              f"inverse-map difference {inv[seeds.index(s)]:+.4f} > 4 SE ({4 * se:.4f})")
            same_sign = all(x > 0 for x in raw) or all(x < 0 for x in raw)
            above_floor = all(abs(a) > abs(b) for a, b in zip(raw, inv))
            above_sd = abs(st.mean(raw)) > st.stdev(raw)
            positive = same_sign and above_floor and above_sd
            ceiling = e_d > chance - 0.05 and e_D > chance - 0.05
            npos += positive
            nceil += ceiling
            print(f"  test {test:<32} drift t={td:<7} diffusion t={tD:<4} D_raw={['%+.4f' % x for x in raw]} "
                  f"mean={st.mean(raw):+.4f} sd={st.stdev(raw):.4f} |D_inv|={['%.4f' % abs(x) for x in inv]} "
                  f"err={e_d:.3f}/{e_D:.3f}  sign={same_sign} floor={above_floor} sd={above_sd}"
                  f"{'  POSITIVE' if positive else ''}{'  (near chance)' if ceiling else ''}")
        if nok == 0:
            results[test] = "EKSIK"
        elif npos:
            results[test] = f"OLUMLU ({npos} of {nok} pairs)"
        elif nceil == nok:
            results[test] = "BELIRSIZ"
        else:
            results[test] = "OLUMSUZ"
    if not code_ok:
        print("  => KOD HATASI: the inverse-map arm disagrees; results are not interpreted")
        return "KOD HATASI"
    for k, v in results.items():
        print(f"  => test {k}: {v}")
    return results


def decide_joint(R, chance, n_test, label):
    """fog-joint rule (rules/2026-10-02_matched_fog_pairs.md, 3 Oct 07:15): fog-common test 1 with the joint model's
    own inverse-map difference as the sampling floor."""
    R2 = [dict(r, model="clean") for r in R]          # reuse the test-1 machinery
    for r in R:
        R2.append(dict(r, model="drift_matched" if r["family"] == "fog_drift" else "diffuse_own"))
    res = decide_fog_common(R2, chance, n_test, label + " (joint model)")
    if isinstance(res, dict):
        res = res.get("1: clean model")
        print(f"  => joint model: {res}")
    return res


def reversible_share(R, label):
    """Unmatched points: Err(raw) - Err(inverse map) is the part of that model's error the
    reversible map causes. Informative here, unlike at matched pairs."""
    by = defaultdict(list)
    for r in R:
        by[(r["family"], round(r["t"], 4), r["model"])].append(r["err_frozen"] - r["err_frozen_inv"])
    print(f"\n--- {label}: error caused by the reversible part, Err(raw) - Err(inverse map) ---")
    for (f, t, m), xs in sorted(by.items()):
        print(f"  {f:<11} t={t:<7} {m:<13} {ms(xs)}")


def wiener():
    R = rows("wiener-lfl-s[0-9]/output/wiener_lfl_seed*.jsonl")
    if not R:
        return
    print(f"\n=== Wiener like-for-like: seeds {sorted({r['seed'] for r in R})} ===")
    by = defaultdict(list)
    for r in R:
        by[r["t"]].append(r)
    for t in sorted(by):
        rs = by[t]
        print(f"t={t:<5} naive {ms([r['err_naive'] for r in rs])}  witness(f32) {ms([r['err_witness'] for r in rs])} "
              f"eps={sorted({r['eps'] for r in rs}, key=str)}  witness(f64) {ms([r.get('err_witness_f64') for r in rs])} "
              f"eps={sorted({r.get('eps_f64') for r in rs}, key=str)}  oracle {ms([r['err_oracle'] for r in rs])}")


def main_runs(tag, pattern):
    R = rows(pattern)
    if not R:
        return
    print(f"\n=== {tag}: seeds {sorted({r['seed'] for r in R})} ===")
    by = defaultdict(list)
    for r in R:
        by[(r["family"], r["t"])].append(r)
    print(f"{'family':<18} {'t':>5} {'oracle':>13} {'frozen':>13} {'bn':>13} {'tent':>13} {'eata':>13} "
          f"{'bn removes':>13} {'TENT rec%':>9}")
    for (f, t), rs in sorted(by.items()):
        rec = None
        dfz = st.mean(r["delta_frozen"] for r in rs)
        if not rs[0]["is_train_severity"] and dfz > 0:
            rec = 100 * st.mean(r["err_frozen"] - r["err_tent"] for r in rs) / dfz
        print(f"{f:<18} {t:>5} " + " ".join(f"{ms([r[k] for r in rs]):>13}" for k in
                                         ("err_oracle", "err_frozen", "err_bnadapt", "err_tent", "err_eata"))
              + f" {ms([r['err_frozen'] - r['err_bnadapt'] for r in rs]):>13}" + (f" {rec:>9.1f}" if rec is not None else ""))


if __name__ == "__main__":
    fog_common("fog_common (CIFAR-10)", "fog-common-s[0-9]/output/cifar10_c_results_fog_common_seed*.jsonl")
    FC = rows("fog-common-s[0-9]/output/cifar10_c_results_fog_common_seed*.jsonl")
    if FC:
        decide_fog_common(FC, chance=0.9, n_test=10000, label="CIFAR-10")
        reversible_share(FC, "CIFAR-10")
    FJ = [r for r in rows("fog-joint-s[0-9]/output/cifar10_c_results_fog_joint_seed*.jsonl") if r["family"] != "clean"]
    if FJ:
        decide_joint(FJ, chance=0.9, n_test=10000, label="CIFAR-10")
    for lab, pat, ch in (("CIFAR-10, 200 epochs", "fog-joint-full-s[0-9]/output/cifar10_c_results_fog_joint_full_seed*.jsonl", 0.9),
                         ("CIFAR-100, 200 epochs", "c100-fog-joint-full-s[0-9]/output/cifar100_c_results_c100_fog_joint_full_seed*.jsonl", 0.99)):
        FF = rows(pat)
        if FF:
            cl = [r["err_frozen"] for r in FF if r["family"] == "clean"]
            print(f"\n{lab}: clean test error {ms(cl)}")
            decide_joint([r for r in FF if r["family"] != "clean"], chance=ch, n_test=10000, label=lab)
    FJ100 = rows("c100-fog-joint-s[0-9]/output/cifar100_c_results_c100_fog_joint_seed*.jsonl")
    if FJ100:
        decide_joint(FJ100, chance=0.99, n_test=10000, label="CIFAR-100")
    fog_common("fog_common (CIFAR-100)", "c100-fog-common-s[0-9]/output/cifar100_c_results_c100_fog_common_seed*.jsonl")
    FH = rows("c100-fog-common-s[0-9]/output/cifar100_c_results_c100_fog_common_seed*.jsonl")
    if FH:
        decide_fog_common(FH, chance=0.99, n_test=10000, label="CIFAR-100")
        reversible_share(FH, "CIFAR-100")
    main_runs("blur pair (CIFAR-100)", "c100-blur-s[0-9]/output/cifar100_c_results_c100_blur_seed*.jsonl")
    fog_common("fog_common (Tiny-ImageNet)", "tin-fog-common-s[0-9]/output/tin_results_tin_fog_common_seed*.jsonl")
    FT = rows("tin-fog-common-s[0-9]/output/tin_results_tin_fog_common_seed*.jsonl")
    if FT:
        decide_fog_common(FT, chance=0.99, n_test=5000, label="Tiny-ImageNet")
    wiener()
    main_runs("adam_main", "adam-main-s[0-9]/output/cifar10_c_results_adam_main_seed*.jsonl")
    main_runs("blur factorial", "blur-factorial-s[0-9]/output/cifar10_c_results_factorial_seed*.jsonl")
    main_runs("blur factorial extra", "blur-factorial-extra-s[0-9]/output/cifar10_c_results_factorial_extra_seed*.jsonl")
    main_runs("Tiny-ImageNet noise (Adam)", "tin-adam-s[0-9]/output/tin_results_tin_adam_seed*.jsonl")
