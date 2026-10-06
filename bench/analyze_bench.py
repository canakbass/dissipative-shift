"""Applies the decision rule fixed before the CIFAR-10-C padding run (the paper's
appendix on first-layer padding) to its result file.
    python bench/analyze_bench.py [results.jsonl]
The default is the file the paper's numbers come from."""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "oct6", "padding-bench", "output", "bench",
                                                          "padding_cifar10c_results.jsonl")
rows = [json.loads(line) for line in open(PATH)]
models = list(dict.fromkeys(r["model"] for r in rows))
HYP = ["contrast", "brightness", "fog"]


def summary(model, corr, mode):
    rs = [r for r in rows if r["model"] == model and r["corruption"] == corr and r["mode"] == mode]
    if len(rs) != 5:
        return None
    g = sum(r["gain"] for r in rs) / 5
    se = math.sqrt(sum(r["se"] ** 2 for r in rs)) / 5
    return dict(G=g, SE=se, z=g / se if se > 0 else float("nan"), per_sev=[r["gain"] for r in rs],
                err_zeros=sum(r["err_zeros"] for r in rs) / 5, err_reflect=sum(r["err_reflect"] for r in rs) / 5)


def verdict(corr, mode="bnadapt"):
    s = [summary(m, corr, mode) for m in models]
    s = [x for x in s if x]
    if len(s) < 3:
        return "incomplete", s
    if all(x["G"] > 0 for x in s) and sum(x["G"] > 3 * x["SE"] for x in s) >= 2:
        return "supported", s
    if sum(x["G"] <= x["SE"] for x in s) >= 2:
        return "rejected", s
    return "inconclusive", s


print(f"models: {models}")
for corr in HYP + ["gaussian_noise"]:
    for mode in ("bnadapt", "frozen", "tent"):
        for m in models:
            x = summary(m, corr, mode)
            if x:
                print(f"  {corr:<15} {mode:<8} {m:<18} G {x['G']:+.4f} (SE {x['SE']:.4f}, z {x['z']:+.1f})  "
                      f"err zeros {x['err_zeros']:.4f} reflect {x['err_reflect']:.4f}  per severity "
                      + " ".join(f"{g:+.4f}" for g in x["per_sev"]))
print()
for corr in HYP:
    v, _ = verdict(corr)
    print(f"H for {corr}: {v}")
v_noise, s_noise = verdict("gaussian_noise")
print(f"C2 gaussian_noise: G > 3 SE in {sum(x['G'] > 3 * x['SE'] for x in s_noise)} of {len(s_noise)} models")
for m in models:
    c = [r for r in rows if r["model"] == m and r["corruption"] == "clean" and r["mode"] == "bnadapt"]
    if c:
        d = c[0]["err_reflect"] - c[0]["err_zeros"]
        print(f"C1 clean, {m}: BN-adapt error zeros {c[0]['err_zeros']:.4f}, reflect {c[0]['err_reflect']:.4f} "
              f"(reflect {'costs' if d > 0.005 else 'within 0.005 of zeros,'} {d:+.4f})")
