"""Applies the decision rules of rules/2026-10-05_retraining_and_normalization.md
(written before the runs), unchanged, to the retrained joint models.
    python oct10/analyze_variants.py
Written on 6 Oct 2026, before any V1-V4 result existed. Reads oct3/<variant>-s*/output;
a variant with fewer than three seeds is listed as incomplete and enters no decision."""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# variant -> kernel directory prefix (job names: fog_joint_full, fog_joint_full_reflect, ...)
VARIANTS = {
    "V0": "fog-joint-full",           # BN, zero padding (existing results)
    "V1": "fog-joint-full-reflect",   # BN, reflect padding in the first convolution
    "V2": "fog-joint-full-gn32",      # GroupNorm, 32 groups
    "V3": "fog-joint-full-ln",        # GroupNorm, 1 group
    "V4": "fog-joint-full-std",       # BN, per-image scalar standardization of the input
}
WITH_BN = ("V0", "V1", "V4")
OUTER = (0.089, 0.197, 0.490)


def load(prefix):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "oct3", f"{prefix}-s[0-9]", "output", "*.jsonl"))):
        rows += [json.loads(line) for line in open(p)]
    return rows


def pairs(rows):
    """{v: {seed: (drift row, diffuse row)}}, members matched by equal v."""
    drift = [r for r in rows if r["family"] == "fog_drift"]
    diffuse = [r for r in rows if r["family"] == "fog_diffuse"]
    out = {}
    for d in drift:
        m = [r for r in diffuse if r["seed"] == d["seed"] and abs(r["v"] - d["v"]) / d["v"] < 1e-3]
        if m:
            out.setdefault(round(d["v"], 3), {})[d["seed"]] = (d, m[0])
    return out


def judge(by_seed, key):
    ss = sorted(by_seed)
    raw = [by_seed[s][0][f"err_{key}"] - by_seed[s][1][f"err_{key}"] for s in ss]
    inv = [by_seed[s][0][f"err_{key}_inv"] - by_seed[s][1][f"err_{key}_inv"] for s in ss]
    positive = (len(raw) == 3 and (all(x > 0 for x in raw) or all(x < 0 for x in raw))
                and all(abs(a) > abs(b) for a, b in zip(raw, inv)) and abs(st.mean(raw)) > st.stdev(raw))
    return raw, inv, positive


def main():
    res, clean = {}, {}
    for name, prefix in VARIANTS.items():
        rows = load(prefix)
        seeds = sorted({r["seed"] for r in rows})
        print(f"\n===== {name} ({prefix}): seeds {seeds}")
        if len(seeds) < 3:
            print("  incomplete; enters no decision")
            continue
        clean[name] = st.mean(r["err_frozen"] for r in rows if r["family"] == "clean")
        print(f"  clean error (frozen) {clean[name]:.4f}")
        P = pairs(rows)
        res[name] = {}
        for key in ("frozen", "bnadapt") if name in WITH_BN else ("frozen",):
            for v in sorted(P):
                raw, inv, pos = judge(P[v], key)
                res[name][(key, v)] = (st.mean(raw), pos)
                dr = st.mean(P[v][s][0][f"err_{key}"] for s in P[v])
                df = st.mean(P[v][s][1][f"err_{key}"] for s in P[v])
                print(f"  {key:<8} v={v:.3f}: drift {dr:.3f} diffuse {df:.3f} gap {st.mean(raw):+.4f} "
                      f"[{' '.join(f'{x:+.4f}' for x in raw)}] floor {max(abs(x) for x in inv):.4f} "
                      f"{'POSITIVE' if pos else '-'}")

    def n_outer(name, key="frozen"):
        return sum(res[name][(key, v)][1] for v in OUTER if (key, v) in res[name])

    print("\n===== code checks")
    ok = {}
    if "V4" in res:
        ok["V4"] = all(abs(g) <= 0.01 for g, _ in res["V4"].values())
        print(f"  K1 (V4: no pair with |mean D| > 0.01): {'PASS' if ok['V4'] else 'FAIL'}")
    if "V1" in res:
        ok["V1"] = not any(p for (key, _), (_, p) in res["V1"].items() if key == "bnadapt")
        print(f"  K2 (V1: no positive pair with BN-adapt): {'PASS' if ok['V1'] else 'FAIL'}")
    for name in res:
        if name == "V0":
            continue
        k3 = clean[name] < 0.20
        ok[name] = ok.get(name, True) and k3
        print(f"  K3 ({name}: clean error below 0.20): {'PASS' if k3 else 'FAIL'} ({clean[name]:.4f})")

    print("\n===== tests")
    if "V1" in res and ok["V1"]:
        n = n_outer("V1")
        outcome = "(a)" if n >= 2 else "(b)" if n == 0 else "(c)"
        print(f"  Test A (V1 frozen): {n} of 3 outer pairs positive -> outcome {outcome}")
        if "V0" in res:
            print(f"    V0 for comparison: {n_outer('V0')} of 3")
    elif "V1" in res:
        print("  Test A: code checks failed for V1; not interpreted")
    if "V2" in res and "V3" in res and ok["V2"] and ok["V3"]:
        n2, n3 = n_outer("V2"), n_outer("V3")
        outcome = ("not specific to batch normalization" if n2 >= 2 and n3 >= 2 else
                   "specific to the BN architecture" if n2 == 0 and n3 == 0 else "mixed; reported per variant")
        print(f"  Test B (V2 {n2}/3, V3 {n3}/3 outer pairs positive): {outcome}")
    if "V4" in res and ok["V4"] and "V0" in clean:
        print(f"  V4 (no decision): clean error {clean['V4']:.4f} against {clean['V0']:.4f} for V0")


if __name__ == "__main__":
    main()
