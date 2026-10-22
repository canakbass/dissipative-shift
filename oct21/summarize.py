"""Applies the pre-specified rules of oct21/PLAN.md to the downloaded results.
    python oct21/summarize.py            (prints the tables and writes oct21/summary.json)
Written before the results of experiments 1 and 3 were seen (only the first
EfficientNet-B0 log had been read, for the BN-adapt bug of PLAN.md's correction note).
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BN = ["resnet50_v1", "resnet50_v2", "efficientnet_b0"]
LN = ["vit_b_16", "convnext_tiny", "swin_t"]
V_GRID = [0.0025, 0.005, 0.01, 0.021, 0.042, 0.089]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):  # sample standard deviation, as in the paper's rule (paper/facts.py)
    return st.stdev(xs) if len(xs) > 1 else 0.0


def load(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, pattern))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def imnet(run="first"):
    """run = "first": the kernels as approved (frozen errors, the decision);
    run = "r2": the BN models rerun with shuffled batches (BN-adapt; frozen as a replication)."""
    out = {}
    for m in BN + LN:
        name = {"resnet50_v1": "resnet50-v1", "resnet50_v2": "resnet50-v2", "efficientnet_b0": "efficientnet-b0",
                "vit_b_16": "vit-b-16", "convnext_tiny": "convnext-tiny", "swin_t": "swin-t"}[m]
        d = f"imnet-fog-{name}" + ("-r2" if run == "r2" else "")
        rows = load(f"{d}/output/imnet_fog_{m}.jsonl")
        if not rows:
            continue
        clean = [r for r in rows if r["family"] == "clean"][0]
        R = {(r["family"], r["v"], r["seed"]): r for r in rows if r["family"] != "clean"}
        seeds = sorted({s for _, _, s in R})
        pairs = []
        for v in V_GRID:
            if not all((f, v, s) in R for f in ("drift", "diffuse") for s in seeds):
                continue
            raw = [R[("drift", v, s)]["err"] - R[("diffuse", v, s)]["err"] for s in seeds]
            inv = [R[("drift", v, s)]["err_inv"] - R[("diffuse", v, s)]["err_inv"] for s in seeds]
            e_d = mean([R[("drift", v, s)]["err"] for s in seeds])
            e_D = mean([R[("diffuse", v, s)]["err"] for s in seeds])
            informative = all(e < 0.95 and e > clean["err"] + 0.01 for e in (e_d, e_D))
            verified = len(seeds) == 3 and (all(x > 0 for x in raw) or all(x < 0 for x in raw)) \
                and all(abs(a) > abs(b) for a, b in zip(raw, inv)) and abs(mean(raw)) > sd(raw)
            bn = None
            if clean.get("err_bnadapt") is not None:
                bn = mean([R[("drift", v, s)]["err_bnadapt"] - R[("diffuse", v, s)]["err_bnadapt"] for s in seeds])
            pairs.append(dict(v=v, err_drift=e_d, err_diffuse=e_D, gap=mean(raw), gaps=raw, sd=sd(raw),
                              floor=max(abs(x) for x in inv), informative=informative, verified=verified,
                              positive=verified and mean(raw) > 0, negative=verified and mean(raw) < 0,
                              bn_gap=bn))
        out[m] = dict(clean=clean["err"], clean_bnadapt=clean.get("err_bnadapt"), seeds=seeds, pairs=pairs)
    return out


def decide(res):
    def n_pos(m):
        return sum(p["positive"] for p in res[m]["pairs"] if p["informative"])
    bn_hits = [m for m in BN if m in res and n_pos(m) >= 2]
    h1 = len(bn_hits) >= 2
    ln_hits = [m for m in LN if m in res and n_pos(m) >= 2]
    half = {}
    for m in LN:
        if m not in res:
            continue
        inf_v = sorted(p["v"] for p in res[m]["pairs"] if p["informative"])[-2:]
        ok = True
        for v in inf_v:
            g_ln = [p["gap"] for p in res[m]["pairs"] if p["v"] == v][0]
            g_bn = [p["gap"] for b in BN if b in res for p in res[b]["pairs"] if p["v"] == v]
            ok = ok and bool(g_bn) and g_ln < 0.5 * mean(g_bn)
        half[m] = dict(vs=inf_v, below_half=ok)
    h2 = not ln_hits and all(h["below_half"] for h in half.values()) and len(half) == 3
    neg = {m: [p["v"] for p in res[m]["pairs"] if p["negative"] and p["informative"]] for m in res}
    if not h1:
        cls = "does not transfer to pretrained ImageNet models"
    elif h2:
        cls = "transfers at scale, specific to batch normalization"
    else:
        cls = "transfers at scale, not specific to batch normalization"
    return dict(H1=h1, bn_models_with_2plus=bn_hits, H2=h2, ln_models_with_2plus=ln_hits, ln_half=half,
                negative_verified=neg, outcome=cls, complete=all(m in res for m in BN + LN))


def blur_joint():
    rows = load("blur-joint-s*/output/*.jsonl")
    R = {(r["family"], r["t"], r["seed"]): r for r in rows}
    seeds = sorted({s for _, _, s in R})
    if not seeds:
        return None
    rem = lambda f, s: R[(f, 2.0, s)]["err_frozen"] - R[(f, 2.0, s)]["err_bnadapt"]
    det, heat = "gauss_blur", "blurfft_s15_a01"
    ratios = [rem(det, s) / rem(heat, s) if rem(heat, s) != 0 else float("inf") for s in seeds]
    r_mean = mean([rem(det, s) for s in seeds]) / mean([rem(heat, s) for s in seeds])
    if len(seeds) == 3 and all(x > 1 for x in ratios) and r_mean >= 2:
        cls = "replicates"
    elif len(seeds) == 3 and all(x > 1 for x in ratios):
        cls = "weakens"
    else:
        cls = "does not replicate" if len(seeds) == 3 else "incomplete"
    fz = lambda f: mean([R[(f, 2.0, s)]["err_frozen"] for s in seeds])
    orc = lambda f: mean([R[(f, 2.0, s)]["err_oracle"] for s in seeds])
    gap = lambda f: fz(f) - orc(f)
    rec = lambda f: 1 - (mean([R[(f, 2.0, s)]["err_bnadapt"] for s in seeds]) - orc(f)) / gap(f)
    return dict(seeds=seeds, removed_det=[rem(det, s) for s in seeds], removed_heat=[rem(heat, s) for s in seeds],
                ratio_per_seed=ratios, ratio_of_means=r_mean, outcome=cls,
                frozen_det=fz(det), frozen_heat=fz(heat), oracle_det=orc(det), oracle_heat=orc(heat),
                gap_ratio=gap(det) / gap(heat), recovered_det=rec(det), recovered_heat=rec(heat),
                recovered_ratio=rec(det) / rec(heat))


def main():
    out = {}
    for run in ("first", "r2"):
        res = imnet(run)
        out[f"imnet_{run}"] = res
        print(f"\n=== ImageNet matched fog pairs ({run}) ===")
        for m, r in res.items():
            print(f"{m:<16} clean={r['clean']:.4f}" + (f" clean_bnadapt={r['clean_bnadapt']:.4f}" if r["clean_bnadapt"] is not None else ""))
            for p in r["pairs"]:
                flag = ("+" if p["positive"] else "-" if p["negative"] else ".") + ("" if p["informative"] else " (not informative)")
                bn = f" bn_gap={p['bn_gap']:+.4f}" if p["bn_gap"] is not None else ""
                print(f"   v={p['v']:<7} drift={p['err_drift']:.4f} diff={p['err_diffuse']:.4f} gap={p['gap']:+.4f} "
                      f"sd={p['sd']:.4f} floor={p['floor']:.4f} {flag}{bn}")
    dec = decide(out["imnet_first"])
    out["imnet_decision"] = dec
    print("\nDECISION (first runs):", json.dumps(dec, indent=1))
    bj = blur_joint()
    out["blur_joint"] = bj
    print("\n=== blur_joint ===\n", json.dumps(bj, indent=1))
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
