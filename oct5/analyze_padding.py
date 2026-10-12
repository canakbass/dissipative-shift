"""Applies the decision rules of rules/2026-10-05_first_layer_padding.md (written
before the runs), unchanged, to the padding-test results.
    python oct5/analyze_padding.py
Reads oct3/padding-test-s[0-9]/output and oct3/c100-padding-test-s[0-9]/output, and for check K1
the stored 200-epoch joint-model results in oct3/(c100-)fog-joint-full-s[0-9]/output."""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ("zeros", "reflect", "circular", "haze", "standardize")


def load(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "oct3", pattern))):
        rows += [json.loads(line) for line in open(p)]
    return rows


def pairs(rows):
    """{v: {seed: (drift row, diffuse row)}} for one padding kind."""
    out = {}
    for r in rows:
        if r["family"] == "clean":
            continue
        out.setdefault(round(r["v"], 3), {}).setdefault(r["seed"], {})[r["family"]] = r
    return {v: {s: (d["fog_drift"], d["fog_diffuse"]) for s, d in by_s.items()} for v, by_s in out.items()}


def judge(by_seed, key):
    raw = [by_seed[s][0][f"err_{key}"] - by_seed[s][1][f"err_{key}"] for s in sorted(by_seed)]
    inv = [by_seed[s][0][f"err_{key}_inv"] - by_seed[s][1][f"err_{key}_inv"] for s in sorted(by_seed)]
    positive = (len(raw) == 3 and (all(x > 0 for x in raw) or all(x < 0 for x in raw))
                and all(abs(a) > abs(b) for a, b in zip(raw, inv)) and abs(st.mean(raw)) > st.stdev(raw))
    return raw, inv, positive


def main():
    for ds, pat, stored_pat in (("CIFAR-10", "padding-test-s[0-9]/output/*.jsonl", "fog-joint-full-s[0-9]/output/*.jsonl"),
                                ("CIFAR-100", "c100-padding-test-s[0-9]/output/*.jsonl", "c100-fog-joint-full-s[0-9]/output/*.jsonl")):
        rows = load(pat)
        seeds = sorted({r["seed"] for r in rows})
        print(f"\n===== {ds}: seeds {seeds}")
        if len(seeds) < 3:
            print("  incomplete; no decision")
            continue
        res = {k: pairs([r for r in rows if r["pad"] == k]) for k in KINDS}
        for k in KINDS:
            clean = [r for r in rows if r["pad"] == k and r["family"] == "clean"]
            print(f"  clean error, pad={k:<11} frozen {st.mean(r['err_frozen'] for r in clean):.3f}  "
                  f"bn {st.mean(r['err_bnadapt'] for r in clean):.3f}")
        verdict = {}
        for key in ("frozen", "bnadapt"):
            for k in KINDS:
                line = []
                npos_out = 0
                for v in sorted(res[k]):
                    raw, inv, pos = judge(res[k][v], key)
                    dr = st.mean(res[k][v][s][0][f"err_{key}"] for s in res[k][v])
                    df = st.mean(res[k][v][s][1][f"err_{key}"] for s in res[k][v])
                    line.append(f"v={v}: {dr:.3f}/{df:.3f} gap {st.mean(raw):+.3f} [{' '.join(f'{x:+.3f}' for x in raw)}]"
                                f" floor {max(abs(x) for x in inv):.3f} {'POS' if pos else '-'}")
                    if v > 0.05 and pos:
                        npos_out += 1
                    verdict.setdefault((key, k), []).append((v, pos))
                print(f"  {key:<8} pad={k:<11} " + "\n" + "\n".join("      " + x for x in line))
        # K1: zeros/frozen against the stored results
        stored = pairs([dict(r, pad="zeros") for r in load(stored_pat) if r["family"] != "clean"])
        k1 = all(abs((res["zeros"][v][s][0]["err_frozen"] - res["zeros"][v][s][1]["err_frozen"])
                     - (stored[v][s][0]["err_frozen"] - stored[v][s][1]["err_frozen"])) <= 0.04
                 for v in res["zeros"] for s in res["zeros"][v])
        k2 = not any(p for key in ("frozen", "bnadapt") for _, p in verdict[(key, "standardize")])
        print(f"\n  K1 (zero padding reproduces the stored gaps within 0.04): {'PASS' if k1 else 'FAIL'}")
        print(f"  K2 (standardization leaves no positive pair): {'PASS' if k2 else 'FAIL'}")
        if not (k1 and k2):
            print("  code checks failed: results are not interpreted")
            continue
        t1 = not any(p for k in ("reflect", "circular") for _, p in verdict[("bnadapt", k)])
        print(f"  Test 1 (BN-adapt, reflect and circular, no positive pair): "
              f"{'met: the BN-adapt gap is the first layer' + chr(39) + 's zero padding' if t1 else 'not met'}")
        out = {k: sum(p for v, p in verdict[("frozen", k)] if v > 0.05) for k in ("reflect", "circular")}
        if all(n >= 2 for n in out.values()):
            t2 = "(a) frozen gap does not depend on zero padding; main claim kept"
        elif all(n == 0 for n in out.values()):
            t2 = "(b) frozen gap is a zero-padding artefact; reframe"
        else:
            t2 = "(c) inconclusive; decide by retraining"
        print(f"  Test 2 (frozen, outer pairs positive: reflect {out['reflect']}, circular {out['circular']}): {t2}")
        haze = sum(p for v, p in verdict[("frozen", "haze")] if v > 0.05)
        print(f"  (haze padding, informational: outer pairs positive {haze})")


if __name__ == "__main__":
    main()
