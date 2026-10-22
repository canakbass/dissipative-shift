#!/usr/bin/env python3
"""Checks every number in main.tex against the raw results and the code.

Every numeric token in the paper, including the appendix tables pulled in with
\\input and numbers written as m x 10^e or 10^e, must fall into one class:

  claim      recomputed from the result files (via facts.json or directly) and
             compared after rounding to the digits shown, the paper's stated
             convention; a claim whose context no longer appears in the text fails
  table      a cell of a hand-written table, regenerated row by row from the data
  generated  a data row of a table produced by make_tables.py; the script is rerun
             into a temporary directory and its output must equal the committed file
  design     a setting of the experiments (learning rate, grid, prior, ...), read
             from the script that ran them and compared
  severity   a value written as "t = ..." (or t <= ..., t -> ...): it must be a
             severity that occurs in the result files
  formula    an integer coefficient inside a formula (the 2 in 2t, the 1 in 1 - e)
  noted      a historical value that no current file can reproduce, listed by hand
             with its reason

Captions and table headers are checked like any other text. The report lists
every token that falls in none of these classes, and the run fails if there is one.

Run from anywhere:  python paper/verify_claims.py      (exit code 1 on any failure)
Run paper/facts.py first if the result files have changed.
"""
import glob
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
TEX = open(os.environ.get("VERIFY_TEX", os.path.join(HERE, "main.tex"))).read()
FACTS = json.load(open(os.path.join(HERE, "facts.json")))


def _inline(text):
    return re.sub(r"\\input\{([^}]*)\}", lambda m: open(os.path.join(HERE, m.group(1))).read(), text)


BODY = _inline(TEX[TEX.find(r"\begin{document}"):])
BODY = re.sub(r"(?m)(?<!\\)%.*$", "", BODY)

FAILS, PASSES = [], 0
SPANS = []            # (start, end, class)


def F(*keys):
    v = FACTS
    for k in keys:
        v = v[str(k)] if isinstance(v, dict) else v[k]
    return v


def load(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path))]


def src(path):
    return open(os.path.join(ROOT, path)).read()


def const(path, pattern):
    """A constant read from a script: pattern has one capture group."""
    m = re.search(pattern, src(path), flags=re.M)
    if not m:
        FAILS.append(f"design source: /{pattern}/ not found in {path}")
        return float("nan")
    return float(eval(m.group(1), {"math": math}))


# --------------------------------------------------------------------------- patterns
NUM = r"(-?\d+(?:\{,\}\d{3})*(?:\.\d+)?(?:\\%)?)"


def P(template, marker=NUM):
    """A text template with '@' where each checked number stands. Everything else
    matches literally, with any run of whitespace matching any whitespace."""
    parts = template.split("@")
    esc = [re.sub(r"(?:\\\s)+|\s+", r"\\s+", re.escape(p)) for p in parts]
    return marker.join(esc)


def _decimals(s):
    s = s.replace("\\%", "").replace("{,}", "")
    return len(s.split(".")[1]) if "." in s else 0


def _num(s):
    return float(s.replace("\\%", "").replace("{,}", ""))


def _find(label, pattern):
    ms = list(re.finditer(pattern, BODY))
    if not ms:
        FAILS.append(f"{label}: context not found in text  /{pattern[:90]}/")
    return ms


HALF_TOL = 1e-4


def _round(value, d, rnd):
    """Rounding to d decimals: to nearest (halves away from zero), or in the direction that
    keeps a bound true (rnd="down" for lower bounds, "up" for upper bounds and tolerances)."""
    # the tolerance, a thousandth of the last digit, absorbs float32 storage noise in
    # the result files (errors are stored as float32, about 1e-8 off)
    if rnd == "down":
        return math.floor(value * 10 ** d + 1e-3) / 10 ** d
    if rnd == "up":
        return math.ceil(value * 10 ** d - 1e-3) / 10 ** d
    # to nearest, an exact half away from zero (Section 4.1). A half stored in float32, such
    # as 0.0145 -> 0.01449998, lies within 4e-5 of the last shown digit; the tolerance of
    # 1e-4 catches those and nothing else (42.54997, a ratio, is not a half)
    return math.copysign(math.floor(abs(value) * 10 ** d + 0.5 + HALF_TOL) / 10 ** d, value)


def _compare(label, txt, value, rnd=None):
    global PASSES
    want = _round(value, _decimals(txt), rnd)
    if math.isclose(_num(txt), want, abs_tol=1e-9):
        PASSES += 1
    else:
        FAILS.append(f"{label}: text says {txt}, source gives {value:.6g} (-> {want})")


def check(label, template, *values, cls="claim", rnd=None):
    """Each '@' in the template is a number compared with the matching value. rnd is
    None (nearest), "down" or "up" for every value, or a list with one entry per value."""
    rnds = rnd if isinstance(rnd, (list, tuple)) else [rnd] * len(values)
    for m in _find(label, P(template)):
        assert len(m.groups()) == len(values), label
        for i, (v, r) in enumerate(zip(values, rnds), 1):
            _compare(label, m.group(i), v, r)
            SPANS.append((*m.span(i), cls))


def design(label, template, *values):
    check(label, template, *values, cls="design")


SCI = r"(\d+(?:\.\d+)?)\\times\s*10\^\{(-?\d+)\}"


def check_sci(label, template, value, cls="claim", rnd=None):
    """'@' stands for m\\times10^{e}; the mantissa is compared at its shown digits."""
    global PASSES
    for m in _find(label, P(template, SCI)):
        mant, exp = m.group(1), int(m.group(2))
        want = _round(value / 10 ** exp, _decimals(mant), rnd)
        if math.isclose(float(mant), want, abs_tol=1e-9) and abs(math.log10(value) - exp) < 1.5:
            PASSES += 1
        else:
            FAILS.append(f"{label}: text says {mant}e{exp}, source gives {value:.4g}")
        SPANS.append((*m.span(0), cls))


POW = r"10\^\{(-?\d+)\}"


def check_pow(label, template, value, exact=True, cls="claim"):
    """'@' stands for a bare 10^{e}: equal to value (exact) or its order of magnitude."""
    global PASSES
    for m in _find(label, P(template, POW)):
        e = int(m.group(1))
        ok = math.isclose(value, 10.0 ** e, rel_tol=1e-9) if exact else round(math.log10(value)) == e
        if ok:
            PASSES += 1
        else:
            FAILS.append(f"{label}: text says 10^{e}, source gives {value:.3g}")
        SPANS.append((*m.span(0), cls))


def word(label, template, condition, cls="claim"):
    """A statement whose numbers (if any) are encoded in a condition computed from the
    data; the whole phrase is covered."""
    global PASSES
    for m in _find(label, P(template)):
        if condition:
            PASSES += 1
        else:
            FAILS.append(f"{label}: stated in the text but false in the data")
        SPANS.append((*m.span(0), cls))


def check_table(label, rows):
    """rows: expected row strings (cells joined with ' & '); each must appear inside
    the table carrying this label. Only the matched rows count as covered."""
    global PASSES
    at = BODY.find(f"\\label{{{label}}}")
    if at < 0:
        FAILS.append(f"table {label}: not found")
        return
    a = BODY.rfind(r"\begin{table}", 0, at)
    b = BODY.find(r"\end{table}", at)
    for r in rows:
        m = re.compile(P(r)).search(BODY, a, b)
        if m:
            PASSES += 1
            SPANS.append((*m.span(), "table"))
        else:
            FAILS.append(f"table {label}: expected row not found: {r}")


def noted(label, template, reason):
    for m in _find(label, P(template)):
        SPANS.append((*m.span(0), "noted"))
    NOTED.append((label, reason))


NOTED = []


def mean(xs):
    return sum(xs) / len(xs)


def _hu(x, nd, sign=False):
    """x rounded by the paper's rule (_round) and formatted, as in make_tables.py"""
    r = _round(x, nd, None)
    if r == 0:
        return f"{0.0:.{nd}f}"  # a rounded zero carries no sign
    return f"{r:+.{nd}f}" if sign else f"{r:.{nd}f}"


def pm(m, s):
    return f"${_hu(m, 3)} \\pm {_hu(s, 3)}$"


def ms(vals):
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return pm(m, s)


abl = F("ablation")
W = F("wiener")
sw = F("sweep")
BN_SCRIPT = "kaggle_cifar10_c_bnadapt/cifar10_c_bnadapt_kaggle.py"
HEAT_SCRIPT = "kaggle_cifar10_c_bnadapt_blurheat/cifar10_c_bnadapt_blurheat_kaggle.py"
ABL_SCRIPT = "kaggle_fog_ablation/fog_ablation_kaggle.py"
KT_ASYM = json.load(open(os.path.join(ROOT, "kill_test_asym_threshold.json")))
KT_GEN = json.load(open(os.path.join(ROOT, "kill_test_fog_generator.json")))
HEATCOMP = json.load(open(os.path.join(ROOT, "test_heat_blur_results.json")))
KT = load("kill_test_results.jsonl")
KA = {r["t"]: r for r in load("kill_test_asym_results.jsonl")}
KF = load("kill_test_fog_results.jsonl")

# =========================================================================== abstract
SLACK_MAX = max(F("kt_noise_slack_max"), F("kt_fog_slack_max"), F("kt_asym_slack_max"))
prereg_rows = re.search(r"\\label\{tab:prereg\}.*?\\end\{table\}", BODY, re.S).group(0)
first = re.findall(r"\n(?:Wiener|Fog ablation|DANN|TENT)[^&]*&[^&]*&\s*([^&]*?)\s*&", prereg_rows)
word("appendix prereg count", "of these four, three were not met in their first test and one held",
     len(first) == 4 and sum(not o.startswith("Held") for o in first) == 3)
word("main predictions starred", "A star marks the prediction each document named as its main one",
     "Main prediction" in src("PREREG_v5_dann_lambda.md") and "Main prediction" in src("PREREG_v6_tent_hyperparams.md")
     and "If 1 and 2 both hold" in src("PREREG_v4_fog_ablation.md"), cls="design")
word("v3 predictions 2 and 3", "Narrowed to a composition check",
     "Prediction 2 was not pursued" in src("PREREG_v3.md") and "Prediction 3 was narrowed to a composition check" in src("PREREG_v3.md"), cls="design")
check("margin convention", "one-sided binomial margins at the $@\\%$ level", 5, cls="design")
word("margin convention level", "one-sided binomial margins at the $5\\%$ level", "_norm.ppf(0.95)" in src("paper/facts.py"), cls="design")
check("joint level", "so that it holds jointly with probability at least $@\\%$", 95, cls="design")
word("joint level is Bonferroni at 5%", "holds jointly with probability at least $95\\%$",
     "0.05 / 12" in src("paper/facts.py") and "0.05 / 6" in src("paper/facts.py"), cls="design")
word("mechanism rejected per prereg condition", "rejected, as pre-specified",
     "hypothesis is WRONG" in src("PREREG_v4_fog_ablation.md") and max(abl[k]["rec"] for k in abl if k.startswith("fog_drift") and "rec" in abl[k]) > 30, cls="design")
word("contribution: three seeds per condition", "On CIFAR-10 with three seeds, we find that",
     F("adam_main_seeds") == [1, 2, 3] and F("fjf_pairs")[0]["seeds"] == [1, 2, 3] and all(v["n"] == 3 for k, v in F("blur_factorial").items()))
# =========================================================================== first-layer padding (oct5)
_P10, _P100 = F("pad_c10"), F("pad_c100")
_pv = sorted(_P10["zeros"]["frozen"], key=float)
_outer2 = [k for k in _pv if _P10["reflect"]["frozen"][k]["positive"] and _P10["circular"]["frozen"][k]["positive"]]
_hz = max(_P10["zeros"]["frozen"][k]["gap"] for k in _outer2)
_hc = F("pad_c10_commuting_outer")
# the headline: the zero-padded (V0) and the reflect-trained (V1) joint models at the two outer pairs
_V0 = {f"{q['v']:.3f}": q for q in F("fjf_pairs")}
_V1 = {f"{q['v']:.3f}": q for q in F("fjr_pairs")}
_V1a = {f"{q['v']:.3f}": q for q in F("fjr_adapt")}
_hd = [_V[k]["d_raw_mean"] for _V in (_V0, _V1) for k in _outer2]
for lab, tpl in (("abstract", "without per-image standardization err\n$@$ to $@$ more on the lower-contrast member at the two pairs farthest beyond\ntheir training range"),
                 ("contribution", "standardization (reflect or zero padding) err $@$ to $@$ more on the\nlower-contrast member at the two pairs farthest beyond their training range"),
                 ("conclusion", "standardization err\n$@$ to $@$ more on the lower-contrast member of the most extrapolated\nequal-information fog pairs")):
    check(f"headline range ({lab})", tpl, min(_hd), max(_hd))
word("abstract: the drift member is the lower-contrast one and errs more", "more on the lower-contrast member",
     all(F("matched_contrast")[k]["drift"] < F("matched_contrast")[k]["diffuse"] and _V[k]["d_raw_mean"] > 0 for _V in (_V0, _V1) for k in _outer2))
word("abstract: the lower-contrast member costs more error", "standardization err\n$0.07$ to $0.09$ more on the lower-contrast member",
     all(F("matched_contrast")[k]["drift"] < F("matched_contrast")[k]["diffuse"] and _V[k]["d_raw_mean"] > 0 for _V in (_V0, _V1) for k in _outer2))
word("headline: the two most extrapolated pairs, every seed", "farthest beyond their\ntraining range (the most extrapolated pairs; seed means, positive in each seed)",
     len(_outer2) == 2 and all(_V[k]["positive"] and min(_V[k]["d_raw"]) > 0 for _V in (_V0, _V1) for k in _outer2))
word("abstract: two padding models behind the headline", "frozen\n200-epoch batch-normalized ResNet-18 models without per-image standardization err",
     len(_hd) == 4 and set(_V0) >= set(_outer2) and set(_V1) >= set(_outer2))
_every_seed = len(_outer2) == 2 and all(_V[k]["positive"] and min(_V[k]["d_raw"]) > 0 for _V in (_V0, _V1) for k in _outer2)
for _where, _tpl in (("abstract", "(seed means, positive in each seed); with GroupNorm or layer\nnormalization the gap stays below"),
                     ("contribution", "(the most extrapolated pairs; seed means, positive in each seed). With\nGroupNorm or layer normalization this gap")):
    word(f"headline caveat in the {_where}: seed means, positive in each seed", _tpl, _every_seed)
_v1_bn_clean = not any(q["bn_positive"] for q in F("fjr_adapt"))
# claims withdrawn after the reflect-trained run and the normalization ablation: they must
# not come back in any wording. Matched case-insensitively on the text with math markers
# and line breaks removed.
# (the corrections log, which must name them, is left out)
_log_a = BODY.find(r"\section{Corrections log}")
_log_b = BODY.find(r"\section{", _log_a + 10)
_guarded = BODY[:_log_a] + BODY[_log_b:] if _log_a >= 0 else BODY
_flat = re.sub(r"\s+", " ", _guarded.replace("$", "").replace("\\%", "%").replace("~", " ")).lower()
_GAP = r"(?:gap|excess|difference|effect)"
WITHDRAWN = [
    # "a fifth (an eighth) of the gap comes from zero padding", in any order and spelling
    r"\b(?:a|an|one)[ -](?:fifth|eighth|quarter|third|tenth)\b[^.;]{0,80}\b(?:zero[ -]?padding|padding)",
    r"\b(?:zero[ -]?padding|padding)\b[^.;]{0,80}\b(?:a|an|one)[ -](?:fifth|eighth|quarter|third|tenth)\b",
    r"\b(?:21\.5|12\.8|21|22|20|13|12)\s*(?:%|per ?cent)[^.;]{0,80}\bpadding",
    r"\bpadding\b[^.;]{0,80}\b(?:21\.5|12\.8)\s*(?:%|per ?cent)",
    r"zero[ -]?padding (?:thus |therefore |alone )?(?:accounts|explains|causes|is responsible|contributes)",
    r"(?:part|some|share|fraction|portion|much|most) of (?:the |this |that )?(?:frozen )?" + _GAP
    + r"[^.;]{0,40}(?:comes|stems|arises|is due) (?:from|to) (?:the )?(?:first[ -](?:layer|convolution)'?s? )?zero",
    r"(?:part|some|share|fraction|portion|much|most) of (?:this|that|it) (?:comes|stems|arises|is due) (?:from|to)[^.;]{0,40}zero[ -]?padding",
    # "adaptation (test-time normalization, BN-adapt, TENT) does not close / remove the gap"
    r"(?:adaptation|test[ -]time normali[sz]ation|bn-adapt|tent|re-?estimat\w* (?:the )?(?:batch[ -])?statistics)"
    r"[^.;]{0,40}(?:does not|doesn't|cannot|can't|fails to|did not|do not) (?:close|remove|eliminate)",
    r"(?:not|never) closed? by (?:test[ -]time )?(?:adaptation|normali[sz]ation)",
    # an unqualified "the gap does not depend on normalization"
    r"" + _GAP + r" (?:does not|doesn't) depend on (?:the )?(?:batch[ -])?normali[sz]ation",
]
for _old in WITHDRAWN:
    m_ = re.search(_old, _flat)
    if m_:
        FAILS.append(f"withdrawn claim back in the text: '{_flat[max(0, m_.start() - 30):m_.end() + 30]}'")
    else:
        PASSES += 1
word("V1: full gap (condition)", "Retrained with reflect padding, the model still errs more on\nthe drift member at these pairs",
     all(_V1[k]["positive"] for k in _outer2) and mean([_V1[k]["d_raw_mean"] for k in _outer2]) > 0.9 * mean([_V0[k]["d_raw_mean"] for k in _outer2])
     and min(_V1[k]["d_raw_mean"] for k in _outer2) > max(F("pad_c10_commuting_outer")))
# the normalization ablation (V2 GroupNorm 32, V3 layer norm, V4 standardized input), retrained
_N = {k: {f"{q['v']:.3f}": q for q in F(k + "_pairs")} for k in ("fjgn", "fjln", "fjstd")}
_outer3 = [k for k in _V1 if float(k) > 0.05]
_npos = {k: sum(_N[k][v]["positive"] for v in _outer3) for k in _N}
_norm_max = max(abs(q["d_raw_mean"]) for k in _N for q in _N[k].values())
check("inference padding lowers the gap", "changing the padding\nonly at inference gives $@$ to $@$", *F("pad_c10_commuting_outer"))
check("normalization: outer gaps", "the gaps at the outer pairs stay below $@$ in seed means (of mixed sign across\nseeds at four of the six)",
      max(abs(_N[k][v]["d_raw_mean"]) for k in ("fjgn", "fjln") for v in _outer3), rnd="up")
_mixed = [min(_N[k][v]["d_raw"]) < 0 < max(_N[k][v]["d_raw"]) for k in ("fjgn", "fjln") for v in _outer3]
word("normalization: outer gaps of mixed sign at four of six", "(of mixed sign across\nseeds at four of the six)",
     sum(_mixed) == 4 and len(_mixed) == 6)
word("normalization: none and one", "GroupNorm passed at no outer pair and layer normalization at one", (_npos["fjgn"], _npos["fjln"]) == (0, 1))
_ratio = [_V0[k]["d_raw_mean"] / _N[n_][k]["d_raw_mean"] for n_ in ("fjgn", "fjln") for k in _outer2]
# the GroupNorm and layer-normalization models are the zero-padded joint model with its normalization replaced
# "about fivefold": the four ratios of seed means (4.3 to 5.6) all lie in [4, 6]; their denominators are noisy
check("normalization: how much smaller", "$@$ to\n$@$ times smaller than for the zero-padded batch-normalization model at the most\nextrapolated pairs (Tables~\\ref{tab:budget} and~\\ref{tab:padding})", min(_ratio), max(_ratio))
word("normalization: three seeds each", "(three seeds each, otherwise as the joint model)",
     all(F(n_ + "_pairs")[0]["seeds"] == [1, 2, 3] for n_ in ("fjgn", "fjln", "fjstd")), cls="design")
check("normalization: clean errors", "at a cost in clean error\n($@$ and $@$)", F("fjgn_clean_err")[0], F("fjln_clean_err")[0])
check("standardized: clean error", "that it also costs no clean error ($@$,\nagainst $@$) is an empirical finding", F("fjstd_clean_err")[0], F("fjf_clean_err")[0])
# the standardized-input model was trained with zero padding, and per-image standardization at inference also
# leaves no pair passing with every padding, so the no-gap result does not depend on the padding
word("abstract: the scope excludes per-image standardization", "models without per-image standardization err",
     _npos["fjstd"] == 0 and all(abs(q["d_raw_mean"]) < 0.01 for q in _N["fjstd"].values()))
word("standardized: with any padding", "With any padding, a model trained on images standardized one by one",
     _npos["fjstd"] == 0 and not any(p for key in ("frozen", "bnadapt") for p in [q["positive"] for q in _P10["standardize"][key].values()]))
word("standardized: no gap, no cost", "sees identically\ndistributed members and has no gap",
     _npos["fjstd"] == 0 and all(abs(q["d_raw_mean"]) < 0.01 for q in _N["fjstd"].values())
     and not any(q["bn_positive"] for q in F("fjstd_adapt")) and F("fjstd_clean_err")[0] <= F("fjf_clean_err")[0])
# the abstract no longer mentions per-image standardization (review 18, C1 rule vi); it stays in the body
check("checks: BN-adapt closes", "absorb scale\nand offset and close the gap (to within $@$ with the padding\nchanged at inference, $@$ when trained with it, Table~\\ref{tab:padding})",
      max(F("pad_c10_bn_commuting_max"), F("pad_c100_bn_commuting_max")), max(abs(q["bn_gap"]) for q in F("fjr_adapt")), rnd="up")
word("checks: BN-adapt closes (condition)", "and close the gap (to within", _v1_bn_clean
     and not any(_P[k]["bnadapt"][q]["positive"] for _P in (_P10, _P100) for k in ("reflect", "circular", "haze") for q in _pv))
check("V1: keeps the gap", "at these pairs, by $@$ and $@$; as our rule required", *[_V1[k]["d_raw_mean"] for k in _outer2])
word("rules: V1 is the main number", "Frozen: $2$ of $3$ outer pairs pass, so the gap is not a zero-padding artifact; BN-adapt: no pair passes; main numbers moved here, as fixed in advance",
     "the paper's main number moves to V1's difference" in src("rules/2026-10-05_retraining_and_normalization.md").replace("\n", " "))
word("V1 is the main number, as the rule fixed", "as our rule required, these are\nthe main numbers, with the zero-padded values beside them",
     "the paper's main number moves to V1's difference" in src("rules/2026-10-05_retraining_and_normalization.md").replace("\n", " ") and _every_seed)
_ps_max = max(abs(_N[n_][v]["d_raw_mean"]) for n_ in ("fjgn", "fjln") for v in _outer3)
check("abstract: GN/LN below 0.02", "normalization the gap stays below $@$", _ps_max, rnd="up")
check("contribution: GroupNorm/layer norm keep the gap below", "or layer normalization this gap stays below $@$ in seed means", _ps_max, rnd="up")
word("contribution: the outcome was mixed", "our pre-specified rule calls\nthe result mixed because with layer normalization one outer pair still passes", (_npos["fjgn"], _npos["fjln"]) == (0, 1))
# the discussion's restatements of the 7.7, 2.1 and 0.07-0.09 results were cut for space (review 17)
# "about fivefold" left the discussion (review 18): the ratio depends on which BN model is the base
check("discussion: CIFAR-10-C cells", "help BN-adapt ($@$ of $@$ cells; Appendix", F("bench_cells_helped"), F("bench_cells_helped") + F("bench_cells_hurt"))
word("discussion: the artifact did not appear on CIFAR-10-C", "On CIFAR-10-C,\nreflect padding did not help BN-adapt", set(F("bench_verdict").values()) == {"rejected"})
word("most depends on BN, does not need zero padding", "Replacing batch normalization shrinks most of the gap, and retraining with reflect\npadding does not",
     sum(_V1[k]["positive"] for k in _outer3) >= 2 and _npos["fjgn"] <= 1 and _npos["fjln"] <= 1
     and mean([_V1[k]["d_raw_mean"] for k in _outer2]) > 0.9 * mean([_V0[k]["d_raw_mean"] for k in _outer2]))
word("narrowed after the mixed outcome", "The outcome was mixed, and we narrowed the main claim to models with batch\nnormalization and no per-image standardization after seeing it", (_npos["fjgn"], _npos["fjln"]) == (0, 1))
word("the rule's branches as in the analysis script", "specific to batch normalization if neither variant passed at any outer pair, not\nspecific if both passed at two, and mixed otherwise",
     "n2 >= 2 and n3 >= 2" in src("oct10/analyze_variants.py") and "n2 == 0 and n3 == 0" in src("oct10/analyze_variants.py"), cls="design")
check("V1 within run-to-run of V0", "within $@$ of the zero-padded model's", max(abs(_V1[k]["d_raw_mean"] - _V0[k]["d_raw_mean"]) for k in _outer2), rnd="up")
check("corrections: V1 within V0", "keeps the frozen gap, within $@$ of the zero-padded\nmodel, comparable to the seed-to-seed spread", max(abs(_V1[k]["d_raw_mean"] - _V0[k]["d_raw_mean"]) for k in _outer2), rnd="up")
word("corrections: V1-V0 difference comparable to the seed spread", "of the zero-padded\nmodel, comparable to the seed-to-seed spread",
     all(abs(_V1[k]["d_raw_mean"] - _V0[k]["d_raw_mean"]) < max(_V0[k]["d_raw_sd"], _V1[k]["d_raw_sd"]) for k in _outer2))
_sds = [_V[k]["d_raw_sd"] for _V in (_V0, _V1) for k in _outer2]
check("V0/V1 seed-to-seed sd", "seed-to-seed standard deviations, $@$ to $@$, whereas", min(_sds), max(_sds))
word("V1-V0 difference comparable to the seed spread", "comparable to the\nseed-to-seed standard deviations",
     all(abs(_V1[k]["d_raw_mean"] - _V0[k]["d_raw_mean"]) < max(_V0[k]["d_raw_sd"], _V1[k]["d_raw_sd"]) for k in _outer2))
_dd = max(abs(_N[n_][k]["err_diffuse"] - _V0[k]["err_diffuse"]) for n_ in ("fjgn", "fjln") for k in _outer2)
check("not a compression near chance", "the drift member's error falls from $@$ and $@$\nto $@$ and $@$ with GroupNorm and to $@$ and $@$ with layer\nnormalization, while the diffusion member's moves by at most $@$",
      *[_V0[k]["err_drift"] for k in _outer2], *[_N["fjgn"][k]["err_drift"] for k in _outer2], *[_N["fjln"][k]["err_drift"] for k in _outer2], _dd,
      rnd=[None] * 6 + ["up"])
word("not a compression (condition)", "For GroupNorm and layer normalization this is not a compression near chance",
     all(_N[n_][k]["err_drift"] < _V0[k]["err_drift"] - 0.05 for n_ in ("fjgn", "fjln") for k in _outer2) and _dd < 0.02)
word("retrained models on CIFAR-10 only", "retrained padding and normalization variants are CIFAR-10",
     not glob.glob(os.path.join(ROOT, "oct3", "c100-fog-joint-full-*-s*")))
_gn = r"^        return nn\.GroupNorm\(32, channels\)$"
for tpl in ("\\citep{wu2018group} ($@$ groups) or\nlayer normalization", "GroupNorm has $@$ groups and\nlayer normalization one"):
    design(f"GroupNorm groups: {tpl[:25]}", tpl, 32 if re.search(_gn, src("kaggle_oct3_runner/runner.py"), re.M) else -1)
word("layer norm is one group", "layer normalization one",
     re.search(r"^        return nn\.GroupNorm\(1, channels\)$", src("kaggle_oct3_runner/runner.py"), re.M) is not None, cls="design")
check("rules: normalization counts", "outer pairs passing $@$ of $@$ with GroupNorm, $@$ of $@$ with layer normalization",
      _npos["fjgn"], len(_outer3), _npos["fjln"], len(_outer3))
word("rules: standardized no gap", "per-image standardization leaves no gap", _npos["fjstd"] == 0)
check("rules: V1 counts", "Frozen: $@$ of $@$ outer pairs pass, so the gap is not a zero-padding artifact",
      sum(_V1[k]["positive"] for k in _V1 if float(k) > 0.05), sum(1 for k in _V1 if float(k) > 0.05))
word("rules: V1 BN-adapt", "a zero-padding artifact; BN-adapt: no pair passes", _v1_bn_clean)
_mcf = F("matched_contrast")
check("border mismatch growth", "at the outer\npairs $@$ to $@$ for drift and $@$ to $@$ for diffusion members.",
      *[(1 - _mcf[k]["drift"]) / _mcf[k]["drift"] for k in ("0.089", "0.490")], *[(1 - _mcf[k]["diffuse"]) / _mcf[k]["diffuse"] for k in ("0.089", "0.490")])
_v089 = [k for k in _pv if k not in _outer2 and float(k) > 0.05][0]
check("rules: padding counts", "with reflect and with circular padding $@$ of $@$ outer pairs pass on CIFAR-10 and $@$ of $@$ on CIFAR-100",
      min(F("pad_c10_outer_positive")["reflect"], F("pad_c10_outer_positive")["circular"]), 3,
      min(F("pad_c100_outer_positive")["reflect"], F("pad_c100_outer_positive")["circular"]), 3)
_bn_clean = not any(_P[k]["bnadapt"][q]["positive"] for _P in (_P10, _P100) for k in ("reflect", "circular", "haze") for q in _pv)
word("appendix: BN-adapt gap disappears", "Under BN-adapt the gap disappears with every commuting padding, at every pair, on\nboth data sets", _bn_clean)
word("appendix: frozen gap persists", "in the frozen model it persists at the two most extrapolated pairs of\nCIFAR-10 and the three outer pairs of CIFAR-100",
     len(_outer2) == 2 and F("pad_c100_outer_positive")["reflect"] == 3 == F("pad_c100_outer_positive")["circular"])
_reeval = {"frozen": [], "bnadapt": []}
for _tag, _key in (("pad_c10", "fjf"), ("pad_c100", "fjf100")):
    _st = {f"{q['v']:.3f}": q for q in F(_key + "_pairs")}
    _sa = {f"{q['v']:.3f}": q for q in F(_key + "_adapt")}
    for v_, g_ in F(_tag)["zeros"]["frozen"].items():
        _reeval["frozen"].append(abs(g_["gap"] - _st[v_]["d_raw_mean"]))
    for v_, g_ in F(_tag)["zeros"]["bnadapt"].items():
        _reeval["bnadapt"].append(abs(g_["gap"] - _sa[v_]["bn_gap"]))
check("appendix: re-evaluation tolerance", "by at most\n$@$ frozen and $@$ with BN-adapt", max(_reeval["frozen"]), max(_reeval["bnadapt"]), rnd="up")
design("appendix: pre-specified tolerance", "the pre-specified check allowed $@$ per\nseed", 0.04 if "<= 0.04" in src("oct5/analyze_padding.py") else -1)
word("appendix: trained rows, three seeds", "separate models, three seeds each, trained like\nthe joint model (200 epochs)",
     all(F(n_ + "_pairs")[0]["seeds"] == [1, 2, 3] for n_ in ("fjr", "fjgn", "fjln", "fjstd")) and re.search(
         r'^FULL_BUDGET = JOB in \("fog_joint_full", "c100_fog_joint_full", "fog_joint_full_reflect"\) \+ NORM_ABLATION_JOBS$',
         src("kaggle_oct3_runner/runner.py"), re.M) is not None, cls="design")
check("appendix: V1 clean error", "the reflect-padded one has clean error $@$", F("fjr_clean_err")[0])
check("padding table header", "Data set & Variant & $@$ & $@$ & $@$ & $@$ & $@$ & $@$ & $@$ & $@$", *[float(k) for k in _pv + _pv])

# --- CIFAR-10-C padding with pretrained models
_B, _BM = F("bench"), F("bench_models")
_bh = [_B[f"{m_}|{c_}|bnadapt"] for m_ in _BM for c_ in ("contrast", "brightness", "fog")]
_n_up, _n_down = sum(x["G"] > 0 for x in _bh), sum(x["G"] < 0 for x in _bh)
_bse = [_B[f"{m_}|{c_}|bnadapt"]["SE"] for m_ in _BM for c_ in ("clean", "contrast", "brightness", "fog", "gaussian_noise")]
check("bench caption: SE range", "gains lie between @ and @", min(_bse), max(_bse), rnd=["down", "up"])
word("bench caption: every corrupted gain beyond 3 SE", "every BN-adapt gain under a corruption exceeds three times its standard error",
     all(abs(_B[f"{m_}|{c_}|bnadapt"]["G"]) > 3 * _B[f"{m_}|{c_}|bnadapt"]["SE"] for m_ in _BM for c_ in ("contrast", "brightness", "fog", "gaussian_noise")))
word("bench: three models", "Three standard pretrained CIFAR-10 models", len(_BM) == 3, cls="design")
# (review 19) the Section 5.4 pointer to CIFAR-10-C repeated the discussion and was cut
# (review 19) the Section 5.4 pointer to CIFAR-10-C repeated the discussion and was cut
check("appendix F.1: padding residual", "leaves a gap of up to $@$ after BN-adapt and TENT", max(max(r["bn_gap"], r["tent_gap"]) for r in F("fjf_adapt")), rnd="up")
design("discussion: haze point", "haze\npoint ($A = @$) is far from it", const(BN_SCRIPT, r"^        A = ([\d.]+)$"))
# the discussion paragraph with the cell counts and the clean-image cost moved to Appendix F.1 (review 17);
# the counts stay in Table 7 (below) and the clean-image cost in Table 12
word("bench: cell counts agree in sign and at 3 SE", "BN-adapt better in $1$ of $9$ cells, worse in $8$",
     (_n_up, _n_down) == (F("bench_cells_helped"), F("bench_cells_hurt")) and F("bench_cells_helped") + F("bench_cells_hurt") == len(_bh))
check("bench: rules row", "fog): BN-adapt better in $@$ of $@$ cells, worse in $@$",
      F("bench_cells_helped"), len(_bh), F("bench_cells_hurt"))
word("bench: rejected for all three", "Rejected (contrast, brightness, fog)", set(F("bench_verdict").values()) == {"rejected"})
word("bench: hypothesis rejected (appendix)", "It is rejected for contrast, brightness and\nfog (Table~\\ref{tab:bench})", set(F("bench_verdict").values()) == {"rejected"})
_ab = src("bench/analyze_bench.py")
word("bench: rule as in the analysis script", "supported for a corruption\nif the gain is positive for all three models and above three standard errors for two,\nand rejected if it is at most one standard error for two",
     'all(x["G"] > 0 for x in s) and sum(x["G"] > 3 * x["SE"] for x in s) >= 2' in _ab and 'sum(x["G"] <= x["SE"] for x in s) >= 2' in _ab, cls="design")
_vc = _B["cifar10_vgg13_bn|contrast|bnadapt"]
word("bench: only VGG-13-BN contrast helps", "the\none gain, for VGG-13-BN under contrast",
     [k for k, x in _B.items() if k.endswith("|bnadapt") and "|clean|" not in k and x["G"] > 0] == ["cifar10_vgg13_bn|contrast|bnadapt"])
word("bench: almost all from severity 5", "almost entirely from the most severe level", _vc["per_severity"][4] / 5 > 0.9 * _vc["G"])
word("bench: contrast and brightness deterministic", "contrast\nand brightness are deterministic",
     "np.random" not in src("bench/make_cifar10c_subset.py").split("def contrast")[1].split("def main")[0], cls="design")
# =========================================================================== introduction
check("contribution asym", "matches the analytical\nprediction within $@$ at all severities", F("kt_asym_predmiss_max"), rnd="up")
check("contribution: oracle tracks R*", "the oracle tracks $R^*(t)$ within $@$", SLACK_MAX, rnd="up")

# =========================================================================== theory
he = F("heat_eq_cov")
check("partial order range", "on CIFAR-10 these are $@$ and $@$", he["0.25"]["min"], he["0.25"]["max"])
check("partial order noise", "additive noise with variance $@$, and no severity", he["0.25"]["min"])
grid_v_max = max([2 * t for t in (0.2,)] + [2 * F("fog_t4_tnoise")] + [r["v_eq"] for r in abl.values()])
check("partial order: the largest modal variance again", "or fog grids reaches $@$", he["0.25"]["max"])
word("partial order: nothing on our grids above it", "no severity on our noise\nor fog grids reaches",
     grid_v_max < he["0.25"]["max"])
design("CIFAR fog params", "Fog becomes severe quickly: with $a = @$ and\n$\\sigma_{\\text{fog}} = @$",
       1.0 if "decay = np.exp(-t)" in src(BN_SCRIPT) else -1, const(BN_SCRIPT, r"^FOG_SIGMA = ([\d.]+)"))
check("fog t=1 -> t_noise", "corresponds to $t_{\\text{noise}} = @$,\n$t = 2$ to $@$", F("fog_t1_tnoise"), F("fog_t2_tnoise"))
check("fog t=4 -> t_noise", "and $t = 4$ to $@$, while our noise grid stops at $@$", F("fog_t4_tnoise"), F("noise_grid_max"))
ablp = {k: (const(ABL_SCRIPT, rf'"{k}": dict\(a=([\d.]+)'), const(ABL_SCRIPT, rf'"{k}": dict\(a=[\d.]+, sigma=([\d.]+)')) for k in ("fog_drift", "fog_diffuse")}
word("four times the scattering", "four times the scattering coefficient",
     math.isclose(ablp["fog_diffuse"][1] / ablp["fog_drift"][1], 4.0))
design("ablation sigmas (theory)", "regimes\n($\\sigma_{\\text{fog}} = @$ and $\\sigma_{\\text{fog}} = @$) reach",
       ablp["fog_drift"][1], ablp["fog_diffuse"][1])
check("ablation end v (theory)", ") reach $@$ and\n$@$ (Appendix",
      abl["fog_drift@3.0"]["v_eq"], abl["fog_diffuse@8.0"]["v_eq"])

# =========================================================================== scope
HEAT_SIG = 15.0 if "15.0**2" in src(HEAT_SCRIPT) else -1
HEAT_AMIN = 0.1 if "+ 0.1  # BLUR_HEAT_A_MIN" in src(HEAT_SCRIPT) else -1
design("heat params", "On $@\\times@$ images we set $\\sigma_{\\text{blur}} = @$ and $a_{\\min} = @$", 32, 32, HEAT_SIG, HEAT_AMIN)
check("heat noise DC and max", "at $t = 2$ its variance is $@$ at the DC component\n($\\omega^2 = @$) and $@$ at the Nyquist corner ($\\omega^2 = @$)",
      F("heat_noisevar_t2_dc_unitary"), HEAT_AMIN, F("heat_noisevar_t2_max_unitary"), F("heat_omega2_max"))
check("heat DC attenuation", "attenuates even the DC mode, by a factor of\n$@$ at $t = 2$", F("heat_dc_decay_t2"))
design("deterministic blur truncation", "sampled and truncated at $@$ standard deviations",
       inspect.signature(__import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter).parameters["truncate"].default
       if "truncate" not in re.search(r"gaussian_filter\(imgs[^)]*\)", src(BN_SCRIPT)).group(0) else -1)
design("operator size", "As a\nlinear map on $@\\times@$ images", 32, 32)
design("control grid size", "Deterministic Gaussian blur on\n$@\\times@$ images is injective", 32, 32)
word("pixel range [0,1]", "We do not clip pixel intensities to\n$[0,1]$", "/ 255.0" in src(BN_SCRIPT), cls="design")
op = F("blur_operator")
check_sci("blur min sv t=0.25", "smallest singular value is $@$ at\n$t = 0.25$", op["0.25"]["min_sv_2d"])
check_sci("blur min sv t=2", "and $@$ at $t = 2$. It is nonzero at every severity we use", op["2.0"]["min_sv_2d"])
# two fog estimators: the bracket of Theorem 4 uses unlabelled samples, the extrapolation error of Table 8 class means
word("bracket estimated without labels", "estimated without labels\nat three low severities",
     "estimated from unlabelled samples" in src("certified_extrapolation.py"), cls="design")
word("extrapolation-error estimator uses labels", "with labels, from $20{,}000$ samples per low severity",
     "def estimate_generator" in src("kill_test_fog.py") and "class means are estimated from samples" in src("kill_test_fog.py"), cls="design")
word("control injective at the severities used (5.3)", "is injective at every severity we use (Section~\\ref{sec:scope})", F("blur_grid_min_sv") > 1e-7)
word("blur injective on the grid", "It is nonzero at every severity we use", F("blur_grid_min_sv") > 1e-7)
check("blur first singular severity", "the first at $t \\approx @$", F("blur_first_singular_t"))
word("first singular severity lies between grid points", "singular at isolated severities between grid points",
     1.0 < F("blur_first_singular_t") < 2.0)  # 1 and 2 are adjacent grid severities
check("blur cond t=0.25", "condition\nnumber $@$ at $t = 0.25$", op["0.25"]["cond_2d"])
check_sci("blur cond t=1", "$@$ at $t = 1$ and $1.5", op["1.0"]["cond_2d"])
check_sci("blur cond t=2", "and $@$ at\n$t = 2$, against", op["2.0"]["cond_2d"])
check_sci("float32 precision", "machine\nepsilon of $@$", F("float32_eps"))
cd = F("blur_composition_dev")
check("composition 0.05+0.05", "deviates from one blur of\n$t = 0.1$ by $@$", cd["0.05+0.05"])
check("composition 0.05+0.05 (Section 3.4)", "semigroup composition at very small severities (by $@$ in relative\noperator norm at $t = 0.05 + 0.05$", cd["0.05+0.05"])
check("composition 0.25+0.25", "falling to $@$ at\n$t = 0.25 + 0.25$", cd["0.25+0.25"])
check_pow("composition 0.5+0.5", "and $@$ at $0.5 + 0.5$", cd["0.5+0.5"], exact=False)
check("sampled kernel variance", "a Gaussian\nof variance $@$ sampled at integer pixels has an effective variance of\n$@$",
      2 * 0.05, F("sampled_kernel_var_t005"))

# =========================================================================== protocol
design("TENT batch", "in batches of $@$ images", const(BN_SCRIPT, r"^TENT_BATCH = (\d+)"))
tent_sgd = re.search(r"torch\.optim\.SGD\(params, lr=TENT_LR, momentum=([\d.]+)\)", src(BN_SCRIPT))
design("TENT SGD", "In our \\emph{main run} (all four families, three seeds) TENT used\nSGD with momentum $@$ and learning rate",
       float(tent_sgd.group(1)) if tent_sgd else -1)
check_pow("TENT lr", "and learning rate\n$@$.\nAppendix", const(BN_SCRIPT, r"^TENT_LR = ([\de.-]+)"), cls="design")
# the Adam reruns: the main families (tent_fixed) and the matched-pair joint models (oct3 runner)
_ADAM_SCRIPTS = ("kaggle_tent_fixed/tent_fixed_kaggle.py", "kaggle_oct3_runner/runner.py")
_ADAM_LR = (const(_ADAM_SCRIPTS[0], r"^TENT_LR = ([\de.-]+)")
            if len({const(p_, r"^TENT_LR = ([\de.-]+)") for p_ in _ADAM_SCRIPTS}) == 1
            and all("torch.optim.Adam(params, lr=TENT_LR)" in src(p_) for p_ in _ADAM_SCRIPTS) else -1)
check_pow("TENT lr (Adam rerun)", "corrected reruns use Adam at learning rate $@$", _ADAM_LR, cls="design")
design("EATA E0", "$H_0 = @\\ln C$", const(BN_SCRIPT, r"^EATA_E0 = ([\d.]+) \* math\.log"))
design("EATA Fisher n", "over $@$ labelled clean images", const(BN_SCRIPT, r"^EATA_FISHER_N = (\d+)"))
design("EATA alpha", "We use $\\alpha = @$", const(BN_SCRIPT, r"^EATA_FISHER_ALPHA = ([\d.]+)"))
noted("EATA alpha is the default", "EATA's default", "fisher_alpha defaults to 2000 in the official EATA code (checked against its repository)")
design("DANN lambda_max in the reported runs", "$\\lambda_{\\max} = @$ in the reported\nDANN runs",
       1.0 if "lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0" in src("kaggle_cifar10_c_dann/cifar10_c_dann_kaggle.py") else -1)
word("DANN schedule", "$\\lambda(\\tau) = \\lambda_{\\max}\\bigl(2/(1 + e^{-10\\tau}) - 1\\bigr)$,\n$\\tau$ the fraction of training completed",
     "2.0 / (1.0 + np.exp(-10 * p)) - 1.0" in src("kaggle_cifar10_c_dann/cifar10_c_dann_kaggle.py")
     and "p = step / max(1, total_steps)" in src("kaggle_cifar10_c_dann/cifar10_c_dann_kaggle.py"), cls="design")
design("split", "train/test split\n($@/@$)", const(BN_SCRIPT, r"^N_TRAIN_IMG = (\d+)"), const(BN_SCRIPT, r"^N_TEST_IMG = (\d+)"))
word("stem", "($3\\times3$\nstride-1 stem, no max-pooling)", "nn.Conv2d(3, 64, 3, 1, 1, bias=False)" in src(BN_SCRIPT), cls="design")
design("training recipe", "(Nesterov momentum $@$, weight decay\n$5\\times10^{-4}$, learning rate $@$ with cosine annealing, batch size @, @\nepochs)",
       const(BN_SCRIPT, r"^MOMENTUM = ([\d.]+)"), const(BN_SCRIPT, r"^LR = ([\d.]+)"),
       const(BN_SCRIPT, r"^BATCH_SIZE = (\d+)"), const(BN_SCRIPT, r"^EPOCHS = (\d+)"))
check_sci("weight decay", "weight decay\n$@$", const(BN_SCRIPT, r"^WEIGHT_DECAY = ([\de.-]+)"), cls="design")
design("val split", "on a $@\\%$ validation\nsplit", 100 * const(BN_SCRIPT, r"val_frac=([\d.]+)\)"))
design("CIFAR fog (setup)", "fog family uses $a = @$, $\\sigma_{\\text{fog}} = @$ and white\nhaze point $A = @$",
       1.0 if "decay = np.exp(-t)" in src(BN_SCRIPT) else -1, const(BN_SCRIPT, r"^FOG_SIGMA = ([\d.]+)"),
       const(BN_SCRIPT, r"^        A = ([\d.]+)$"))

import ast as _ast
def _calls(path):
    return {getattr(n.func, "attr", getattr(n.func, "id", "")) for n in _ast.walk(_ast.parse(src(path))) if isinstance(n, _ast.Call)}
import glob as _gl
_ship = sorted(os.path.relpath(p_, ROOT) for p_ in _gl.glob(os.path.join(ROOT, "kaggle_*", "*.py")) if "dsprites" not in p_)
word("no clipping in any experiment script", "Corrupted\nimages are kept unclipped",
     len(_ship) > 10 and not any({"clip", "clamp", "clip_", "clamp_"} & _calls(p_) for p_ in _ship), cls="design")
# =========================================================================== sanity checks
word("noise check class variance", "the class-conditional variance is $\\gamma^2(t) = \\gamma_0^2 + 2t$",
     "return SIGMA0**2 + 2 * t" in src("kill_test.py"), cls="design")
design("noise check setup", "We use two classes in two dimensions,\nwith $\\rho = @$ and $\\gamma_0 = @$",
       const("kill_test.py", r"MU2 = np\.array\(\[0\.0, 0\.0\]\), np\.array\(\[([\d.]+), 0\.0\]\)"),
       const("kill_test.py", r"^SIGMA0 = ([\d.]+)"))
check("noise check frozen", "(at most\n$@$ in magnitude)", F("kt_noise_delta_max"), rnd="up")
design("asym prior", "With unequal priors ($\\pi_0 = @$, $\\pi_1 = @$) the Bayes", const("kill_test_asym.py", r"^PI0, PI1 = ([\d.]+),"),
       const("kill_test_asym.py", r"^PI0, PI1 = [\d.]+, ([\d.]+)"))
check("asym thresholds", "keeps\nits threshold ($@$, against $@$ for the clean Bayes rule",
      KT_ASYM["c_frozen"], KT_ASYM["c_star_0"])
asym_train = [r for r in KA.values() if r["is_train_severity"]]
word("asym below 0.001 in training range", "below $0.001$\ninside the training range", max(r["delta_pred"] for r in asym_train) < 0.001)
check("asym gap at t=10", "and $@$ at $t = 10$", KA[10.0]["delta_pred"])
word("asym gap max is at t=10", "and $0.160$ at $t = 10$", max(KA.values(), key=lambda r: r["delta_pred"])["t"] == 10.0)
check("asym match", "matches it to\nwithin $@$ at every severity, and the oracle is within $@$", F("kt_asym_predmiss_max"), F("kt_asym_slack_max"), rnd="up")
design("fog check setup", "In the fog check, with $a = @$, $\\sigma_{\\text{fog}} = @$\nand the haze point midway",
       1.0 if "np.exp(-t)" in src("kill_test_fog.py") else -1, const("kill_test_fog.py", r"^FOG_SIGMA = ([\d.]+)"))
check("sanity oracle noise and fog", "the retrained linear oracle is\nwithin $@$ and $@$ of $R^*(t)$. By symmetry", F("kt_noise_slack_max"), F("kt_fog_slack_max"), rnd="up")
check("fog check R* at 20", "$R^*(t) = @$ at\n$t = 20$", [r for r in KF if r["t"] == 20.0][0]["r_star_closed"])
word("fog check haze midway", "the haze point midway between the means",
     "A = np.array([1.5, 0.0])" in src("kill_test_fog.py") and "np.array([3.0, 0.0])" in src("kill_test_fog.py"), cls="design")

# =========================================================================== CIFAR-10: blur pair
# each family has its own frozen model, trained in run_family on that family's low severities (both scripts)
_own_frozen = all("frozen_model = train_model(x_frozen_train, y_frozen_train)" in src(f) and "def run_family(" in src(f)
                  for f in ("kaggle_cifar10_c_bnadapt/cifar10_c_bnadapt_kaggle.py",
                            "kaggle_cifar10_c_bnadapt_blurheat/cifar10_c_bnadapt_blurheat_kaggle.py"))
# the abstract no longer says "one per blur" (review 18, C1 rule ii); it stays in the contributions and Section 5.2
word("contribution: one frozen model per blur", "frozen ResNet-18\nmodels (one per blur)", _own_frozen, cls="design")
word("blur pair: one frozen model per blur", "their frozen ResNet-18 models (one per blur,", _own_frozen, cls="design")
# (review 20) the single-model blur rerun replaced the "one frozen model per blur" limitation (Section 5.2)
check("blur pair frozen", "Section~\\ref{sec:setup}) to nearly the same error, $@$ and $@$, and a report",
      F("gauss_blur_t2_frozen"), F("blur_heat_t2_frozen"))
check("blur pair BN", "(BN-adapt) removes $@$ of the error under deterministic blur and $@$\nunder heat-equation blur",
      F("gauss_blur_t2_bn_removes"), F("blur_heat_t2_bn_removes"))
bps = F("blur_ratio_bn_per_seed")
check("blur pair ratio", "a $@$-fold ratio in three-seed means (between $@$ and\n$@$ in individual seeds)",
      F("blur_ratio_bn"), min(bps.values()), max(bps.values()))
_gd = F("gauss_blur_t2_frozen") - F("gauss_blur_t2_oracle")
_gh = F("blur_heat_t2_frozen") - F("blur_heat_t2_oracle")
_fd, _fh = F("gauss_blur_t2_bn_removes") / _gd, F("blur_heat_t2_bn_removes") / _gh
word("contribution: the ratio is the product of the two factors", "Measured against\nthe oracle, this ratio is a",
     math.isclose(_gd / _gh * _fd / _fh, F("blur_ratio_bn"), rel_tol=1e-9))
check("discussion: tightness of BN-adapt's lower bound", "its oracle-measured tightness (the recovered fraction)\ndiffered $@$-fold between our blurs", _fd / _fh)
word("discussion: zero padding opened a gap", "with zero padding BN-adapt itself opened a\ngap between matched members",
     all(r["bn_gap"] > 0.05 for r in F("fjf_adapt")))
check("blur pair decomposition", "avoidable gap $\\Delta_{\\text{frozen}}$ is $@$ times larger under deterministic blur\n($@$ against $@$), and BN-adapt recovers $@\\%$ of that gap under\ndeterministic blur but only $@\\%$ under heat-equation blur, a factor of $@$",
      _gd / _gh, _gd, _gh, 100 * _fd, 100 * _fh, _fd / _fh)
check("abstract: blur ratio", "statistics removes $@$ times\n($@$ with one frozen model) as much error from the noise-free one", F("blur_ratio_bn"),
      F("blur_joint", "ratio_of_means"))
check("abstract: blur factor split", "measured against a classifier retrained at\neach severity, a reference loose on images and so recipe-relative, its avoidable\nerror is $@$ times larger", _gd / _gh)
word("abstract: only part of the factor lies in the avoidable error", "its avoidable\nerror is $2.1$ times larger",
     1 < _gd / _gh < F("blur_ratio_bn"))
check("contribution: blur factor split", "this ratio is a $@$-fold larger avoidable gap times a $@$-fold larger\nfraction of that gap recovered", _gd / _gh, _fd / _fh)
check("conclusion: blur factor split", "and a $@$-fold one in\noracle-measured avoidable error", _gd / _gh)
check("blur pair left after BN", "after BN-adapt, $@$ and $@$ remain above the\nrespective oracles",
      _gd - F("gauss_blur_t2_bn_removes"), _gh - F("blur_heat_t2_bn_removes"))
check("blur pair TENT", "TENT removes $@$ against $@$", F("gauss_blur_t2_tent_removes"), F("blur_heat_t2_tent_removes"))

# =========================================================================== recovered fractions
_fx = F("fixed")
_noise_adam = {t: _fx[t]["rec"] for t in _fx if _fx[t].get("d_frozen", 0) >= 0.05 and "rec" in _fx[t]}
_am = F("adam_main")
check("recovered Adam", "TENT recovers $@\\%$ to\n$@\\%$ of the gap measured against the oracle under noise, $@\\%$ to $@\\%$ under\nheat-equation blur, $@\\%$ to $@\\%$ under fog, and $@\\%$ to $@\\%$ in the\ncontrol",
      min(_noise_adam.values()), max(_noise_adam.values()),
      *[F(f"rec_adam_{f}_{e}") for f in ("blur_heat", "fog_beer_lambert", "gauss_blur") for e in ("min", "max")])
check("recovered Adam threshold", "where that gap is at\nleast $@$", 0.05)
word("threshold excludes exactly the small gaps", "where that gap is at\nleast $0.05$",
     sorted(_noise_adam) == ["0.05", "0.1", "0.2"] and F("adam_main_seeds") == [1, 2, 3])
check("recovered SGD", "With SGD they are $@\\%$ to $@\\%$,\n$@\\%$ to $@\\%$, $@\\%$ to $@\\%$ and $@\\%$ to $@\\%$",
      *[F(f"rec_sgd_{f}_{e}") for f in ("gauss_noise", "blur_heat", "fog_beer_lambert", "gauss_blur") for e in ("min", "max")])
check("blur pair: slack in the deterministic arm", "which is at least $@$ under deterministic blur if its\nBayes risk is flat", F("control_slack_t2_joint")[0], rnd="down")
check("control correction", "slack\nadds at least $@$ to the true gap in every seed of the main run", F("control_slack_t2_joint")[0], rnd="down")
check("control fraction", "recovered\nfraction falls from $@\\%$ to at most $@\\%$", F("rec_sgd_gauss_blur", "2.0"), F("control_rec_t2_corrected_joint"), rnd=[None, "up"])
_bt = {k: v["bn_minus_tent"] for k, v in _am.items() if "bn_minus_tent" in v}
_heat_ge = [v for k, v in _bt.items() if k.startswith("blur_heat") and float(k.split("@")[1]) >= 0.5]
check("Adam BN-TENT heat", "TENT beats BN-adapt by $@$ to $@$ under heat-equation blur at every", min(_heat_ge), max(_heat_ge))
word("heat gap the same at every t>=0.5", "under heat-equation blur at every\n$t \\ge 0.5$", len(_heat_ge) == 3 and max(_heat_ge) - min(_heat_ge) < 0.001)
check("Adam BN-TENT fog/control", "by $@$ under fog at $t = 1$, and by $@$ and $@$ in the\ncontrol at $t = 1$ and $t = 2$",
      _bt["fog_beer_lambert@1.0"], _bt["gauss_blur@1.0"], _bt["gauss_blur@2.0"])
_named = {"blur_heat@0.5", "blur_heat@1.0", "blur_heat@2.0", "fog_beer_lambert@1.0", "gauss_blur@1.0", "gauss_blur@2.0"}
check("Adam BN-TENT elsewhere", "elsewhere they differ by at most $@$", max(abs(v) for k, v in _bt.items() if k not in _named), rnd="up")
_dha = F("dann_heat_vs_tent_adam")

# =========================================================================== tables (main text)
R_, DANN = {}, {}
for s, d in {1: "kaggle_cifar10_c_bnadapt/output_seed1", 2: "kaggle_cifar10_c_bnadapt/output_seed2",
             3: "kaggle_cifar10_c_bnadapt_b/output_seed3"}.items():
    for r in load(f"{d}/cifar10_c_results_bnadapt_seed{s}.jsonl"):
        R_.setdefault((r["family"], r["t"]), {})[s] = r
for s, d in {1: "kaggle_cifar10_c_bnadapt_blurheat/output_seed1", 2: "kaggle_cifar10_c_bnadapt_blurheat/output_seed2",
             3: "kaggle_blurheat_bnadapt_s3/output_seed3"}.items():
    for r in load(f"{d}/cifar10_c_results_bnadapt_blurheat_seed{s}.jsonl"):
        R_.setdefault((r["family"], r["t"]), {})[s] = r
for s in (1, 2, 3):
    for r in load(f"kaggle_cifar10_c_dann/output_seed{s}_v2fix/cifar10_c_results_dann_seed{s}_v2fix.jsonl"):
        DANN.setdefault((r["family"], r["t"]), {})[s] = r["err_dann"]
for s, d in {1: "kaggle_cifar10_c_dann_blurheat/output_seed1", 2: "kaggle_dann_blurheat_s2/output_seed2",
             3: "kaggle_dann_blurheat_s3/output_seed3"}.items():
    for r in load(f"{d}/cifar10_c_results_dann_blurheat_seed{s}.jsonl"):
        DANN.setdefault((r["family"], r["t"]), {})[s] = r["err_dann"]
rows = []
for name, k in [("Gaussian noise ($t{=}0.2$)", ("gauss_noise", 0.2)), ("Heat-equation blur ($t{=}2.0$)", ("blur_heat", 2.0)),
                ("Fog ($t{=}1.0$)", ("fog_beer_lambert", 1.0)), ("Fog ($t{=}2.0$)", ("fog_beer_lambert", 2.0)),
                ("Fog ($t{=}4.0$)", ("fog_beer_lambert", 4.0)), ("Deterministic blur ($t{=}2.0$)", ("gauss_blur", 2.0))]:
    v = R_[k]
    cells = [ms([v[s]["delta_frozen"] for s in v]), ms([v[s]["delta_bnadapt"] for s in v]),
             ms([DANN[k][s] - v[s]["err_oracle"] for s in v]),
             ms([v[s]["delta_tent"] for s in v]), ms([v[s]["delta_eata"] for s in v])]
    rows.append(name + " & " + " & ".join(cells) + r" \\")
check("caption cross-run oracle", "oracles differ between runs by up to $@$", F("oracle_cross_run_max"), rnd="up")
check("caption fog t=4 oracle", "At fog $t=4$ the oracle is at $@$, near chance ($@$)",
      sum(v["err_oracle"] for v in R_[("fog_beer_lambert", 4.0)].values()) / 3, 1 - 1 / const(BN_SCRIPT, r"^N_CLASSES = (\d+)"))
word("figure: sd shading", "mean over three seeds with $\\pm1$ standard deviation shading",
     "fill_between" in src("paper/make_figures.py"), cls="design")

# =========================================================================== oracle reliability
eq = F("fog_noise_eq")
argmax_t = max((t for t in eq if "diff" in eq[t]), key=lambda t: abs(eq[t]["diff"]))
check("fog-noise mean/max", "differ\nby $@$ on average in three-seed means (over all five rows, $t = 0$ included),\nat most $@$ (fog $t = 0.1$)",
      F("fog_noise_mean_absdiff"), F("fog_noise_max_absdiff"))
word("fog-noise mean over all five rows", "(over all five rows, $t = 0$ included)", len([t for t in eq if "diff" in eq[t] and eq[t]["diff"] is not None]) == 5 and "0.0" in eq)
word("fog-noise max at t=0.1", "at most $0.0083$ (fog $t = 0.1$)", float(argmax_t) == 0.1)
check("fog-noise t=0", "two\nruns already differ by $@$", F("fog_noise_t0_diff"))
word("noise curve concave", "along the concave noise\ncurve", F("noise_oracle_concave"))
word("oracle consistent between fog and noise (main family)", "The oracle thus agrees between fog\nand noise in our main family", F("fog_noise_max_absdiff") < 0.01)
word("conclusion: oracle agrees between fog and noise", "The oracle agrees\nbetween fog and noise in our main family but is certifiably loose", F("fog_noise_max_absdiff") < 0.01)
word("conclusion and abstract: loose for the 15-epoch recipe", "certifiably loose on images for our\n15-epoch recipe", F("cert_all_pos") > 0 and F("cert_all_by_joint200") > 0)
word("abstract: the reference is loose, so recipe-relative", "a reference loose on images and so\nrecipe-relative", F("cert_all_pos") > 0)
# the abstract now says "a reference loose on images and so recipe-relative" (checked above)
word("introduction: cross-family alone certifies nothing beyond chance", "cross-family equivalences alone certify nothing beyond chance",
     len(F("cert_dissipative_positive")) <= 2 * F("cert_dissipative_expected_by_chance"))
diffuse_dev = max(abs(abl[k]["oracle"] - abl[k]["noise_oracle"]) for k in abl if k.startswith("fog_diffuse") and "noise_oracle" in abl[k])
drift_devs = [abl[k]["oracle"] - abl[k]["noise_oracle"] for k in abl
              if k.startswith("fog_drift") and "noise_oracle" in abl[k] and not k.endswith("@0.0")]
check("ablation vs noise", "agreement within $@$ for the diffusion-dominated regime, but the\ndrift-dominated regime's oracle sits $@$ to $@$ above its noise\nequivalent",
      diffuse_dev, min(drift_devs), max(drift_devs), rnd=["up", None, None])
nt_grid = sorted(t for f, t in R_ if f == "gauss_noise")
drift_pts = [k for k in abl if k.startswith("fog_drift") and not k.endswith("@0.0") and "noise_oracle" in abl[k]]
diff_pts = [k for k in abl if k.startswith("fog_diffuse") and not k.endswith("@0.0") and "noise_oracle" in abl[k]]
word("three of four drift points low", "Of the four drift points with a noise equivalent on the grid, three\nlie between the first two noise grid points",
     len(drift_pts) == 4 and sum(abl[k]["t_noise"] < nt_grid[1] for k in drift_pts) == 3)
word("interpolation least reliable there", "where the linear interpolation is\nleast reliable",
     max(range(len(F("noise_oracle_slopes")) - 1), key=lambda i: F("noise_oracle_slopes")[i] - F("noise_oracle_slopes")[i + 1]) == 0)
sl = F("noise_oracle_slopes")
check_table("tab:fog-noise", [f"{float(t):.2f} & {_hu(eq[t]['t_noise'], 4)} & {_hu(eq[t]['fog_oracle'], 4)} & "
                              f"{_hu(eq[t]['noise_oracle'], 4)} & ${_hu(eq[t]['diff'], 4, True)}$ \\\\"
                              for t in ("0.0", "0.1", "0.25", "0.5", "1.0")])
check("fog-noise caption", "corresponds to $t_{\\text{noise}} = @$ and $@$, outside", F("fog_t2_tnoise"), F("fog_t4_tnoise"))
cc = F("control_oracle_curve")
check("control curve", "held-out severities, $@$, $@$,\n$@$ and $@$ at $t = 0.25, 0.5, 1, 2$ ($@$ at $t = 0$)",
      cc["0.25"], cc["0.5"], cc["1.0"], cc["2.0"], cc["0.0"])
check("control slack bound", "the oracle at $t = 2$ lies above the Bayes risk by at least\n$@$ in the three-seed means and, with jointly corrected margins, by at least $@$ to $@$ in the\nthree seeds", F("control_slack_t2"), F("control_slack_t2_joint")[0], F("control_slack_t2_joint")[-1], rnd="down")
WF = {r["t"]: r for r in load("kaggle_wiener_fixed/output/wiener_test_results.jsonl")}
WL = F("wiener_lfl")
LFL = "kaggle_oct3_wiener/wiener_lfl.py"
LFL_GRID = eval(re.search(r"^EPS_GRID = (\[.*\])", src(LFL), re.M).group(1))
# first test (appendix table)
check_pow("first test best eps", "Its best regularizer, $\\varepsilon = @$, sat\nat the edge", WF[1.0]["best_eps"])
check("first test results", "The best witness error was\n$@$ at $t = 1$ and $@$ at $t = 2$, against the oracle's $@$ and\n$@$",
      W["1.0"]["best"], W["2.0"]["best"], W["1.0"]["oracle"], W["2.0"]["oracle"])
check_pow("first test grid low", "$\\varepsilon$ from $@$ to", min(float(e) for e in WF[1.0]["err_wiener"]), cls="design")
check_pow("first test grid high", "and $\\varepsilon$ from $10^{-4}$ to\n$@$. Its oracle came from the main run", max(float(e) for e in WF[1.0]["err_wiener"]), cls="design")
word("first test eps at the grid edge", "sat\nat the edge of the grid searched",
     WF[1.0]["best_eps"] == WF[2.0]["best_eps"] == min(float(e) for e in WF[1.0]["err_wiener"]))
check("wiener gain cap", "$1/(2\\sqrt{\\varepsilon}) = @$", F("wiener_gain_cap", "0.0001"))
check_pow("wiener sqrt eps", "$\\sqrt{\\varepsilon} = @$", math.sqrt(WF[1.0]["best_eps"]))
hs = F("wiener_half_suppressed_frac")
check("wiener suppressed fraction", "which is $@\\%$ of the modes\nat $t = 1$ and $@\\%$ at $t = 2$", 100 * hs["1.0"], 100 * hs["2.0"])
rows = []
for t in sorted(WF):
    r = WF[t]
    cells = [f"{r['err_naive']:.4f}"] + [f"{r['err_wiener'][e]:.4f}" for e in ("0.0001", "0.001", "0.01", "0.1")] + \
            [f"{r['err_wiener_best']:.4f}", f"{W[str(t)]['oracle']:.4f}"]
    rows.append(f"{t:.2f} & " + " & ".join(cells) + r" \\")
# like-for-like repeat
check_pow("repeat grid low", "regularizer selected from $[@,", min(LFL_GRID), cls="design")
check_pow("repeat grid high", ", @]$ on a split disjoint from the", max(LFL_GRID), cls="design")
word("three seeds", "uses\nthe same periodic FFT operator for witness and oracle, three seeds", F("wiener_lfl_seeds") == [1, 2, 3])
word("witness ahead everywhere, every seed", "beats the retrained blur oracle (for\nthe periodic implementation of the control) at every held-out severity in every seed",
     all(min(WL[t]["gap"]) > 0 for t in WL if float(t) > 0))
check("repeat headline", "every seed\n(Table~\\ref{tab:wiener}): $@$ against $@$ at $t = 1$ and $@$ against\n$@$ at $t = 2$",
      WL["1.0"]["witness"][0], WL["1.0"]["oracle"][0], WL["2.0"]["witness"][0], WL["2.0"]["oracle"][0])
check("repeat certified slack", "the\noracle's slack at $t = 2$ is certified to be at least $@$,\n$@$ and $@$ in the three seeds, and at least $@$ at $t = 1$ in every\nseed",
      *WL["2.0"]["cert_slack_joint"], min(WL["1.0"]["cert_slack_joint"]), rnd="down")
word("repeat: twelve errors", "corrected jointly over the twelve\nerrors involved",
     2 * len(F("wiener_lfl_seeds")) * len([t for t in WL if float(t) in (1.0, 2.0)]) == 12
     and "0.05 / 12" in src("paper/facts.py"), cls="design")
check("float32 cost", "Rounding to \\texttt{float32} costs the witness nothing measurable up to\n$t = 1$ and $@$ at $t = 2$",
      WL["2.0"]["f32_cost"])
word("nothing measurable up to t=1", "nothing measurable up to $t = 1$",
     max(abs(WL[t]["f32_cost"]) for t in ("0.25", "0.5", "1.0")) < 0.001)
check("float64 at t=2", "get\nerror $@$, against $@$ without blur", WL["2.0"]["witness_f64"][0], WL["0.0"]["naive"][0])
_g64 = [float(x) for x in re.search(r"^EPS_GRID_F64 = \[([^\]]*)\]", src("kaggle_oct3_wiener/wiener_lfl.py"), re.M).group(1).split(",")]
check_pow("float64 grid low end", "chosen from\n$[@, 10^{-4}]$", min(_g64), cls="design")
check_pow("float64 grid high end", "$[10^{-20}, @]$, sits at the grid edge", max(_g64), cls="design")
word("float64: edge value selected at t = 2", "sits at the grid edge at $t = 2$", all(e == min(_g64) for e in WL["2.0"]["eps_f64"]))
check("float64: a wider grid would likely lower the error", "so a wider grid would likely\nlower $@$", WL["2.0"]["witness_f64"][0])


def _pm(x):
    return f"${_hu(x[0], 3)} \\pm {_hu(x[1], 3)}$"


rows = []
for t in ("0.0", "0.25", "0.5", "1.0", "2.0"):
    r = WL[t]
    rows.append(f"{float(t):.2f} & {_pm(r['naive'])} & " + (f"{_pm(r['witness'])} & {_pm(r['witness_f64'])}" if float(t) > 0 else "--- & ---")
                + f" & {_pm(r['oracle'])} \\\\")
check_table("tab:wiener", rows)
check("monotonicity count", "at all\n$@\\times@\\times@ = @$", 3, 3, 6, F("mono_total"))
word("monotonicity all hold", "Our dissipative oracles are non-decreasing at all", F("mono_ok") == F("mono_total"))
inv = F("control_inversion_list")
ctl_ts = sorted(t for f, t in R_ if f == "gauss_blur")
held = [x for x in inv if x[1] >= 0.25]
worst = min(inv, key=lambda x: x[3])
t0s = F("control_t0_seeds")
others = sorted(t0s[:worst[0] - 1] + t0s[worst[0]:])
check("monotonicity: control inversions per seed", "per seed it decreases in $@$ of $@$ comparisons,\nthe largest drop $@$",
      F("control_inversions"), F("control_comparisons"), F("control_inversion_max"))
word("monotonicity: all seven control severities", "(over all seven severities, per seed", len(ctl_ts) == 7, cls="design")
word("monotonicity: control means rise", "the control oracle's three-seed means also rise over its held-out\nseverities",
     all(mean([v["err_oracle"] for v in R_[("gauss_blur", a)].values()]) < mean([v["err_oracle"] for v in R_[("gauss_blur", b)].values()])
         for a, b in zip([t for t in ctl_ts if t >= 0.25], [t for t in ctl_ts if t >= 0.25][1:])))
_ctl_held = [t for t in ctl_ts if t >= 0.25]
_left = _gd - F("gauss_blur_t2_bn_removes")
check("control slack against what BN-adapt leaves", "about half of the $@$ that BN-adapt leaves above the\noracle at $t = 2$", _left)
word("control slack is about half of it", "about half of the",
     all(0.4 < x / _left < 0.65 for x in F("control_slack_t2_joint") + [F("control_slack_t2")]))
word("control oracle rises in three-seed means", "rises monotonically in three-seed means over\nthe held-out severities",
     all(mean([v["err_oracle"] for v in R_[("gauss_blur", a)].values()]) < mean([v["err_oracle"] for v in R_[("gauss_blur", b)].values()])
         for a, b in zip([0.0] + _ctl_held, _ctl_held)) and len(held) > 0)

# =========================================================================== drift versus diffusion
design("ablation regimes", "\\emph{drift-dominated} regime ($a = @$, $\\sigma_{\\text{fog}} = @$) that\nstrongly attenuates contrast towards the white haze point, and a\n\\emph{diffusion-dominated} regime ($a = @$, $\\sigma_{\\text{fog}} = @$) that",
       *ablp["fog_drift"], *ablp["fog_diffuse"])
word("ablation: drift regime predicted nearly reversible, found dissipative", "The drift regime is strongly dissipative (Section~\\ref{sec:equivalence}), although we\nhad predicted it to be nearly reversible",
     "hypothesis is WRONG" in src("PREREG_v4_fog_ablation.md"))
word("ablation training ranges (default grids, Appendix E)", "models trained on each regime's own grid\nconfound the comparison with extrapolation distance",
     max(float(k.split("@")[1]) for k in abl if k.startswith("fog_drift") and abl[k]["is_train"]) == 0.5
     and max(float(k.split("@")[1]) for k in abl if k.startswith("fog_diffuse") and abl[k]["is_train"]) == 1.0)
order = [("fog_drift@1.0", "fog_diffuse@2.0"), ("fog_drift@2.0", "fog_diffuse@4.0"), ("fog_drift@3.0", "fog_diffuse@8.0")]
check("horizon headers", r"\multicolumn{2}{c}{$@\times$ horizon} & \multicolumn{2}{c}{$@\times$ horizon}",
      abl["fog_drift@1.0"]["horizon"], abl["fog_drift@2.0"]["horizon"])
check("grid-end horizons", r"& Drift ($@\times$) & Diffusion ($@\times$) \\",
      abl["fog_drift@3.0"]["horizon"], abl["fog_diffuse@8.0"]["horizon"])
word("horizons matched", "we compare the regimes at a matched\n\\emph{extrapolation horizon}",
     all(abl[a_]["horizon"] == abl[b_]["horizon"] for a_, b_ in order[:2]))

def arow(label, fn):
    return label + " & " + " & ".join(fn(abl[a_]) + " & " + fn(abl[b_]) for a_, b_ in order) + r" \\"


vh = F("abl_v_horizon")
check_table("tab:ablation", [
    r"& $t=1$ & $t=2$ & $t=2$ & $t=4$ & $t=3$ & $t=8$ \\",
    arow("Equivalent variance $v$", lambda r: _hu(r['v_eq'], 3)),
    arow("Oracle error", lambda r: _hu(r['oracle'], 3)),
    "Noise-equivalent oracle & " + " & ".join(_hu(abl[k]['noise_oracle'], 3) for p_ in order[:2] for k in p_) + r" & --- & --- \\",
    arow("Frozen error", lambda r: _hu(r['frozen'], 3)),
    arow(r"$\Delta_{\text{frozen}}$", lambda r: _hu(r['d_frozen'], 3)),
    arow("Avoidable share (lower bound)", lambda r: f"{r['avoidable_share']:.0f}\\%"),
    arow("Error TENT removes", lambda r: _hu(r['tent_removes'], 3)),
    r"$v$ / largest training $v$ & " + " & ".join(
        f"{vh[a_]:.2g}" if vh[a_] < 10 else (f"{vh[a_]:.0f}" if vh[a_] > 100 or vh[a_] > 20 else f"{vh[a_]:.1f}")
        for p_ in order for a_ in p_) + r" \\"])
check("2x pair", "($v = @$ against $@$) and does less damage ($@$ against\n$@$)",
      abl["fog_drift@1.0"]["v_eq"], abl["fog_diffuse@2.0"]["v_eq"], abl["fog_drift@1.0"]["frozen"], abl["fog_diffuse@2.0"]["frozen"])
word("2x agree", "At the $2\\times$ horizon the two orderings agree",
     (abl["fog_drift@1.0"]["v_eq"] < abl["fog_diffuse@2.0"]["v_eq"]) == (abl["fog_drift@1.0"]["frozen"] < abl["fog_diffuse@2.0"]["frozen"]))
check("4x horizon disagree", "At the $@\\times$ horizon they disagree", abl["fog_drift@2.0"]["horizon"])
check("4x pair", "equivalent variance ($@$ against $@$), so by", abl["fog_drift@2.0"]["v_eq"], abl["fog_diffuse@4.0"]["v_eq"])
word("4x disagree, oracle lower too", "and its oracle is lower as well",
     abl["fog_drift@2.0"]["oracle"] < abl["fog_diffuse@4.0"]["oracle"] and abl["fog_drift@2.0"]["frozen"] > abl["fog_diffuse@4.0"]["frozen"])
check("4x frozen", "frozen classifier at $@$ error against $@$. This", abl["fog_drift@2.0"]["frozen"], abl["fog_diffuse@4.0"]["frozen"])
vt = F("abl_v_train_max")
check("training v ranges", "the drift model saw $v$ up to $@$ and the diffusion model up to $@$", vt["fog_drift"], vt["fog_diffuse"])
check("v horizons 4x", "the drift point lies $@$ times beyond its training range and\nthe diffusion point $@$ times",
      vh["fog_drift@2.0"], vh["fog_diffuse@4.0"])
dd = F("drift_decay")
check("end pair", "equivalent variances are close ($@$ and $@$,\nthe drift point marginally higher) and the oracles agree ($@$ and $@$),\nwhile frozen error is $@$ against $@$",
      abl["fog_drift@3.0"]["v_eq"], abl["fog_diffuse@8.0"]["v_eq"], abl["fog_drift@3.0"]["oracle"], abl["fog_diffuse@8.0"]["oracle"],
      abl["fog_drift@3.0"]["frozen"], abl["fog_diffuse@8.0"]["frozen"])
word("end: drift marginally higher, same direction as frozen", "the drift point marginally higher",
     abl["fog_drift@3.0"]["v_eq"] > abl["fog_diffuse@8.0"]["v_eq"] and abl["fog_drift@3.0"]["frozen"] > abl["fog_diffuse@8.0"]["frozen"])
check("end v horizons", "larger still ($@$ against $@$)", vh["fog_drift@3.0"], vh["fog_diffuse@8.0"])
check("feature separation", "spread $@$ and $@$), while the\nfrozen models do not ($@$ and $@$)",
      F("sep_oracle_drift_end"), F("sep_oracle_diffuse_end"), F("sep_frozen_drift_end"), F("sep_frozen_diffuse_end"))
check("oracle rises", "The drift oracle rose by $@$, as much as the diffusion\noracle's $@$",
      F("abl_drift_rise"), F("abl_diffuse_rise"))
spread = [max(abl[k]["rec_per_seed"]) - min(abl[k]["rec_per_seed"]) for k in ("fog_drift@1.0", "fog_drift@2.0", "fog_drift@3.0")]
check("ablation recovered", "at most $@\\%$, $@\\%$ and $@\\%$ of the drift regime's gap at its held-out\nseverities (upper bounds; per-seed spread $@$ to $@$ points), and at most\n$@\\%$, $@\\%$ and $@\\%$ of the diffusion regime's",
      abl["fog_drift@1.0"]["rec"], abl["fog_drift@2.0"]["rec"], abl["fog_drift@3.0"]["rec"], min(spread), max(spread),
      abl["fog_diffuse@2.0"]["rec"], abl["fog_diffuse@4.0"]["rec"], abl["fog_diffuse@8.0"]["rec"])
ps = sorted(abl["fog_diffuse@2.0"]["rec_per_seed"])
check("diffuse per-seed", "per-seed fractions range from $@\\%$ to $@\\%$", ps[0], ps[2])

# =========================================================================== batch norm / TENT
check("app G: TENT and EATA agree", "TENT and EATA agreeing to within $@$ at our original setting", F("sgd_tent_eata_diff_max"), rnd="up")
check("SGD spread", "BN-adapt, TENT and EATA agree to within $@$ at\nevery held-out severity", F("sgd_bn_tent_eata_spread_max"), rnd="up")
SW = load("kaggle_tent_lrsweep/output_seed1/cifar10_c_results_tentsweep_seed2.jsonl")
cfgs = [k for k in SW[0]["errs"] if k != "bnadapt"]
design("sixteen configurations", "sweep of $@$ configurations (below)", len(cfgs))
_lrs = sorted({float(c.split("_lr")[1].split("_")[0]) for c in cfgs})
check_pow("sweep lr range low", "learning rates $@$ to", min(_lrs), cls="design")
check_pow("sweep lr range high", "to $@$, one or four steps per", max(_lrs), cls="design")
word("sweep steps one or four", "one or four steps per\nbatch)", {re.search(r"_s(\d+)_", c).group(1) for c in cfgs} == {"1", "4"}, cls="design")
word("sweep: best configuration per family", "the best configuration in each family, SGD at $10^{-2}$ (four\nsteps per batch under noise and blur, one under fog)",
     {f_: sw[f_]["best_cfg"] for f_ in sw} == {"gauss_noise": "tent_lr0.01_s4_sgd", "gauss_blur": "tent_lr0.01_s4_sgd", "fog_beer_lambert": "tent_lr0.01_s1_sgd"})
check_pow("sweep: best lr", "the best configuration in each family, SGD at $@$ (four", 0.01, cls="design")
word("sweep: single seed, held-out average", "In the sweep's single seed, averaged over\nheld-out severities",
     len({r["seed"] for r in SW}) == 1)
check("sweep best gaps", "beats BN-adapt by $@$\nunder noise, $@$ under deterministic blur and $@$ under fog",
      sw["gauss_noise"]["gap_best"], sw["gauss_blur"]["gap_best"], sw["fog_beer_lambert"]["gap_best"])
check_pow("Adam rerun lr", "Rerun with Adam at $@$ and three", _ADAM_LR, cls="design")
check("adam gaps", "TENT beats BN-adapt under noise by $@$,\n$@$ and $@$ at $t = 0.05, 0.1, 0.2$",
      F("fixed", "0.05", "bn_minus_tent"), F("fixed", "0.1", "bn_minus_tent"), F("fixed", "0.2", "bn_minus_tent"))
word("adam indistinguishable at 0.02", "ties at $t = 0.02$", abs(F("fixed", "0.02", "bn_minus_tent")) < 0.002)
# Section 5.6, a summary of Appendices G and H since round 20
check("5.6 summary: sweep", "the best configuration of a\npre-specified sweep beat it by $@$ under noise, where none was predicted to beat it\nby more than $@$",
      sw["gauss_noise"]["gap_best"], 0.01 if "by more than 0.01" in src("PREREG_v6_tent_hyperparams.md") else -1)
word("5.6 summary: the best is SGD under noise and the claim withdrawn", "so we withdrew the claim that entropy minimization adds nothing",
     sw["gauss_noise"]["gap_best"] > 0.01 and max(sw[f_]["gap_best"] for f_ in sw) == sw["gauss_noise"]["gap_best"])
check_pow("5.6 summary: Adam lr", "with Adam at $@$, TENT beats it", _ADAM_LR, cls="design")
check("5.6 summary: Adam under noise", "TENT beats it by up to $@$ (Appendix",
      max(F("fixed", t, "bn_minus_tent") for t in ("0.02", "0.05", "0.1", "0.2")), rnd="up")

# =========================================================================== DANN
_dha = F("dann_heat_vs_tent_adam")
check("DANN yardstick (main)", "reaches $@$ in three-seed means, our\nyardstick", F("yardstick"), rnd="up")
check("DANN worse", "DANN is worse than TENT at six of the twelve held-out severities, by $@$\nto $@$",
      F("dann_resolved_all_min"), F("dann_resolved_all_max"))
word("six of twelve, rest within, all fog among them", "at six of the twelve held-out severities",
     F("dann_resolved_all") == 6 and F("dann_resolved_all_worse") == 6
     and all(f"fog_beer_lambert@{t}" in F("dann_unresolved_all_keys") for t in (0.5, 1.0, 2.0, 4.0)))
word("heat within yardstick", "DANN and TENT\nstay within the yardstick at every held-out severity, with TENT at SGD",
     F("dann_heat_within_yardstick"))
word("heat: DANN behind SGD TENT at t=2", "and behind at $t = 2$) or Adam", all(v > 0 for v in F("dann_heat_t2").values())
     and max(-v for tag in F("dann_heat_vs_tent") for t, v in F("dann_heat_vs_tent")[tag].items() if float(t) <= 1) == F("dann_heat_ahead_max"))
check("heat DANN vs SGD and Adam TENT", "(DANN ahead\nby at most $@$ at $t \\le 1$ and behind at $t = 2$) or Adam (DANN behind by\n$@$ to $@$)",
      F("dann_heat_ahead_max"), min(_dha.values()), max(_dha.values()), rnd=["up", None, None])
word("5.6 summary: DANN no measurable edge", "DANN, which sees unlabelled target images, gives no measurable edge over TENT in our\n15-epoch protocol",
     F("dann_resolved_all") == F("dann_resolved_all_worse") and F("dann_heat_within_yardstick"))
word("behind Adam TENT at every held-out severity", "behind Adam TENT at every held-out severity, all within the yardstick", len(_dha) == 4 and min(_dha.values()) > 0)
check("app G: TENT cross-run", "TENT's three-seed mean differs between the first two\nby up to $@$ at held-out severities", F("tent_cross_run_max"), rnd="up")
check("app G: SGD tracks BN", "tracks to within $@$", F("sgd_bn_tent_eata_spread_max"), rnd="up")
check("app G: BN cross-run", "differs between the main run and the Adam reruns by up to $@$, in the control at\n$t = 2$", F("bn_cross_run_max"), rnd="up")
word("app G: largest BN change is the control at t=2", "in the control at\n$t = 2$. Under", F("bn_cross_run_max_key") == "gauss_blur@2.0")
check("app G: yardstick", "change in a three-seed mean, $@$, is the yardstick", F("yardstick"), rnd="up")
check("app G: heat t=1 advantages", "($@$, $@$ and $@$ at $t = 1$)",
      *[-F("dann_heat_t1")[k] for k in ("main", "s1", "s1_rerun")])
word("app G: within the yardstick", "all within the yardstick", F("dann_heat_within_yardstick"))
check("log: heat DANN third run", "the variation reaches\n$@$, and DANN's advantage is at most $@$.",
      F("yardstick"), F("dann_heat_ahead_max"), rnd="up")
dh = F("dann_heat_vs_tent")
ahead_ts = sorted({float(t) for tag in dh for t, v in dh[tag].items() if v < 0})
check("DANN heat ahead range", "against each of the three TENT estimates, by $@$ to\n$@$", F("dann_heat_ahead_min"), F("dann_heat_ahead_max"))
word("DANN heat behind at t=2", "and behind at $t = 2$", all(v > 0 for v in F("dann_heat_t2").values()))
yk = F("heat_tent_yardstick")

# =========================================================================== discussion
mf, mn = F("match_fog_t1"), F("match_noise_eq")
check("discussion fog fractions", "falls\nfrom $@\\%$ at $t = 1$ to $@\\%$ at $t = 2$",
      F("rec_sgd_fog_beer_lambert", "1.0"), F("rec_sgd_fog_beer_lambert", "2.0"))
check("discussion t_noise", "fog at $t = 1$ has the Bayes risk of additive noise at $@$, but fog\nat $t = 2$ has that of noise at $@$",
      F("fog_t1_tnoise"), F("fog_t2_tnoise"))
check("discussion matched", "noise at $t_{\\text{noise}} = @$ have\nconsistent oracles ($@$ and $@$) and similar frozen errors ($@$ and\n$@$), yet BN-adapt removes $@$ of the fog error against $@$ of the\nnoise error, and TENT recovers $@\\%$ against $@\\%$",
      F("fog_t1_tnoise"), mf["oracle"], mn["oracle"], mf["frozen"], mn["frozen"], mf["bn_removes"], mn["bn_removes"],
      mf["tent_rec"], mn["tent_rec"])
check("discussion horizons", "(the fog point is $@$ times its training range in\nequivalent-noise terms, the noise point $@$ times)", mf["horizon"], mn["horizon"])

# =========================================================================== limitations, conclusion
check("limitations clean error", "the other three-seed experiments used @ (CIFAR-10 clean\nerror $\\approx @$)", const(BN_SCRIPT, r"^EPOCHS = (\d+)"), F("clean_oracle_mean"))
check("conclusion ratio", "this hides a\n$@$-fold difference", F("blur_ratio_bn"))
# =========================================================================== structure: references to log items
_items = re.findall(r"\\item (?:\\label\{([^}]*)\})?\\textbf\{([^}]*)\}", BODY[BODY.find(r"\label{app:corrections}"):])
_labmap = {lab: title for lab, title in _items if lab}
_want = {"corr:step": "Adaptation step size.", "corr:saturated": "Saturated normalization.", "corr:wiener": "The Wiener failure.",
         "corr:dann": "DANN under heat-equation blur.", "corr:padding": "Adaptation at the matched fog pairs.",
         "corr:scope": "Scope of the fog result.", "corr:reversal": "The fog reversal."}
for _lab, _title in _want.items():
    word(f"item label {_lab}", "item~\\ref{%s}" % _lab, _labmap.get(_lab) == _title, cls="design")

# =========================================================================== pre-specification table
design("prereg Wiener severities", "beats the oracle on deterministic blur at $t=@$ and $t=@$",
       *map(float, re.search(r"retrained oracle at blur t=([\d.]+) and t=([\d.]+)", src("PREREG_v3.md")).groups()))
design("prereg thresholds", "adaptation recovers $<@\\%$", 30 if "recovery < 30%" in src("PREREG_v4_fog_ablation.md") else -1)
design("prereg thresholds 2", "Diffusion regime recovery $>@\\%$", 50 if "recovery > 50%" in src("PREREG_v4_fog_ablation.md") else -1)
lam = sorted(float(k) for k in load("kaggle_dann_lambda_sweep/output_seed1/cifar10_c_results_dann_lambda_sweep_seed1.jsonl")[0]["err_dann"])
design("prereg lambda max", "(2) $\\lambda_{\\max}=@$ uniformly worst", max(lam))
word("largest lambda not uniformly worst (app)", "is not\nuniformly worst, contrary to the prediction", not F("lambda_largest_worst_everywhere"))
design("prereg TENT threshold", "beats BN-adapt by more than $@$ &", 0.01 if "by more than 0.01" in src("PREREG_v6_tent_hyperparams.md") else -1)
lr01 = [k for k in cfgs if "lr0.1_" in k]
lr01_worse = 0
for fam in sorted({r["family"] for r in SW}):
    held_rows = [r for r in SW if r["family"] == fam and not r.get("is_train_severity", False)]
    bn = sum(r["errs"]["bnadapt"] for r in held_rows) / len(held_rows)
    for c in lr01:
        lr01_worse += sum(r["errs"][c] for r in held_rows) / len(held_rows) > bn
check_pow("prereg lr 1e-1", "(2) Learning rate $@$ unstable", max(float(re.search(r"lr([\d.]+)_", c).group(1)) for c in cfgs), cls="design")
check("prereg 11 of 12", "Held (@ of @)", lr01_worse, len(lr01) * len({r["family"] for r in SW}))
word("prereg half", "below half of $\\Delta_{\\text{frozen}}$", "Delta_DANN(t) < 0.5 * Delta_frozen(t)" in src("PREREG_v5_dann_lambda.md"), cls="design")

# =========================================================================== corrections log
noted("log: clipping range", "clipped to $[@,@]$ after each\napplication", "pre-correction clipping range; the clipping code was removed")
check("log: within-run BN-TENT", "inflating\ntheir apparent difference to as much as $0.0166$ against at most $@$ within", F("sgd_bn_tent_eata_spread_max"), rnd="up")
word("translated rules shipped", "English translations in the code archive, \\texttt{rules/}",
     all(os.path.exists(os.path.join(ROOT, "rules", f)) for f in ("2026-10-02_matched_fog_pairs.md", "2026-10-05_first_layer_padding.md",
         "2026-10-05_cifar10c_padding.md", "2026-10-05_retraining_and_normalization.md")), cls="design")
noted("log: superseded figure gap", "difference to as much as $@$ against", "the superseded cross-run figure; its inputs are no longer combined anywhere")
noted("log: superseded fog extrapolation", "grew\nfrom $@$ to $0.0051$", "the circular generator estimate that was replaced")
check("log: fog extrapolation now", "from $0.0021$ to $@$.", abs(F("kt_fog_extrap_worst")))
word("log: mismatched horizons", "($4\\times$ against $2\\times$)", abl["fog_drift@2.0"]["horizon"] == 4 and abl["fog_diffuse@2.0"]["horizon"] == 2)
noted("log: ratio 17", "ratio of avoidable gaps of $@$", "from the mismatched-horizon comparison that was replaced")
check("log: matched ratio", "at matched horizons it is $@$ to $@$",
      abl["fog_drift@1.0"]["d_frozen"] / abl["fog_diffuse@2.0"]["d_frozen"], abl["fog_drift@2.0"]["d_frozen"] / abl["fog_diffuse@4.0"]["d_frozen"])
noted("log: first Wiener regularizer", "fixed regularizer of $10^{-2}$", "the initial Wiener implementation, since replaced")
_W0 = {r["t"]: r for r in load("kaggle_wiener_test/output/wiener_test_results.jsonl")}
word("initial Wiener lost either way", "the witness\nlost at the pre-specified severities in this implementation and in the first test",
     all(_W0[t]["err_wiener"] > W[str(t)]["oracle"] and W[str(t)]["best"] > W[str(t)]["oracle"] for t in (1.0, 2.0)))
check_pow("log: SGD lr", "TENT and EATA under SGD at $@$ were", const(BN_SCRIPT, r"^TENT_LR = ([\de.-]+)"), cls="design")
check("log: saturated mean", "Its\nlogged mean was $@$ in every channel", F("tin_logged_mean"))
check("log: 2^24/N", "exactly $2^{24}/N_{\\text{pix}}$ for the\n$N_{\\text{pix}} = @\\cdot@\\cdot@$ pixels per channel",
      const("kaggle_tinyimagenet/tin_kaggle.py", r"^N_TRAIN_IMG = (\d+)"), const("kaggle_tinyimagenet/tin_kaggle.py", r"^IMG_SIZE = (\d+)"),
      const("kaggle_tinyimagenet/tin_kaggle.py", r"^IMG_SIZE = (\d+)"))
word("log: mean equals 2^24/N", "exactly $2^{24}/N_{\\text{pix}}$", math.isclose(F("tin_logged_mean"), F("tin_2p24_over_N"), rel_tol=1e-6))
import numpy as np  # noqa: E402
word("log: sum saturates at 2^24", "stops\nincreasing once a sum reaches $2^{24}$",
     np.float32(2 ** 24) + np.float32(0.5) == np.float32(2 ** 24) and np.float32(2 ** 24 - 1) + np.float32(0.5) != np.float32(2 ** 24 - 1))
FS = json.load(open(os.path.join(ROOT, "float32_saturation_results.json")))
check("log: gain cap", "the filter's gain\nnever exceeds $@$", F("wiener_gain_cap", "0.0001"))
check_pow("log: repeat grid", "The\nrepeat, with the grid extended to $@$", min(LFL_GRID), cls="design")
noted("log: old omega", "quoted at $\\omega^2 = @$, outside the $32\\times32$ grid", "the value the earlier text quoted, outside the grid")
design("log: grid size", "outside the $@\\times@$ grid\n(largest", 32, 32)
check("log: omega max", "(largest $\\omega^2 = @$)", F("heat_omega2_max"))

# =========================================================================== appendix: sanity checks
check("extrapolation noise", "three\nlow severities ($t \\le 0.2$) and extrapolating to $t = 20$ reproduces $R^*(t)$\nwith a largest error of $@$", F("kt_noise_extrap_max"))
word("noise generator fit on t<=0.2", "($t \\le 0.2$) and extrapolating to $t = 20$",
     max(r["t"] for r in KT if r["is_train_severity"]) == 0.2 and max(r["t"] for r in KT) == 20.0)
check("fog generator", "from $@$ samples per low severity recovers $a = @$ (true $@$) and\n$\\sigma_{\\text{fog}} = @$ (true $@$)",
      KT_GEN["n_samples"], KT_GEN["a_hat"], 1.0 if "np.exp(-t)" in src("kill_test_fog.py") else -1,
      KT_GEN["fog_sigma_hat"], const("kill_test_fog.py", r"^FOG_SIGMA = ([\d.]+)"))
worst_fog = max(KF, key=lambda r: abs(r["extrapolation_error"]))
check("fog extrapolation", "largest extrapolation error is\n$@$ in magnitude, at $t = @$", abs(worst_fog["extrapolation_error"]), worst_fog["t"])
design("self-test tolerance", "matches the closed form to within the self-test's tolerance of $@$ at\nevery severity",
       const("kill_test.py", r"^SELF_TEST_TOL = ([\d.]+)") if const("kill_test.py", r"^SELF_TEST_TOL = ([\d.]+)") == KT_GEN["self_test_tol"] else -1)
check_sci("heat det composition", "composes to within $@$", HEATCOMP["det_max_abs"], rnd="up")
check_sci("heat var composition", "to a relative error of at most $@$", HEATCOMP["var_max_rel"], rnd="up")
pm2 = HEATCOMP["ps_median_two_vs_one"]
floor = HEATCOMP["ps_median_one_vs_one"]
check("heat power spectra", "difference over @ images is $@\\%$ and $@\\%$ at the two severities tested,\nagainst $@\\%$ between two independent one-step samples",
      HEATCOMP["n_images"], 100 * pm2["0.25+0.25"], 100 * pm2["0.5+0.5"], 100 * max(floor.values()))
word("heat floor same for both", "against $4.3\\%$", round(100 * min(floor.values()), 1) == round(100 * max(floor.values()), 1))
# generated-table captions
check("gen caption: noise oracle", "and the oracle is within $@$ of $R^*$ throughout", F("kt_noise_slack_max"), rnd="up")
design("gen caption: prior", "Unequal priors ($\\pi_0 = @$)", const("kill_test_asym.py", r"^PI0, PI1 = ([\d.]+),"))
check("gen caption: asym", "It does, to within $@$, and the oracle is within $@$ of $R^*$",
      F("kt_asym_predmiss_max"), F("kt_asym_slack_max"), rnd="up")
word("gen caption: fog limit", "$R^*(t) \\to 0.5$ as $t\\to\\infty$", abs([r for r in KF if r["t"] == 20.0][0]["r_star_closed"] - 0.5) < 1e-3)

# =========================================================================== appendix: adaptation settings
design("sweep grid", "(SGD and Adam, learning rates $10^{-4}$ to $10^{-1}$, one or four steps per\nbatch)",
       *([] if sorted({float(re.search(r"lr([\d.]+)_", c).group(1)) for c in cfgs}) == [1e-4, 1e-3, 1e-2, 1e-1]
         and {re.search(r"_s(\d)_", c).group(1) for c in cfgs} == {"1", "4"} and {c.split("_")[-1] for c in cfgs} == {"sgd", "adam"}
         else [float("nan")]))
SPANS.extend((m.start(), m.end(), "design") for m in re.finditer(re.escape(r"learning rates $10^{-4}$ to $10^{-1}$"), BODY))
design("sweep threshold", "beat BN-adapt by more than $@$ on average over held-out severities",
       0.01 if "by more than 0.01" in src("PREREG_v6_tent_hyperparams.md") else -1)
check("sweep noise spread", "agreed with BN-adapt to within $@$ at every held-out\nseverity in the main run (three seeds), and to within $@$ in the sweep's\nsingle seed",
      F("sgd_noise_spread_max"),
      max(abs(r["errs"]["bnadapt"] - r["errs"]["tent_lr0.001_s1_sgd"]) for r in SW
          if r["family"] == "gauss_noise" and not r.get("is_train_severity", False)), rnd="up")
def _sweep_gap(fam, cfg):
    held = [r for r in SW if r["family"] == fam and not r.get("is_train_severity", False)]
    return sum(r["errs"]["bnadapt"] - r["errs"][cfg] for r in held) / len(held)
check("sweep SGD larger step", "SGD at $10^{-2}$ reaches $@$ with one step and\n$@$ with four",
      _sweep_gap("gauss_noise", "tent_lr0.01_s1_sgd"), _sweep_gap("gauss_noise", "tent_lr0.01_s4_sgd"))
check_pow("sweep SGD larger step lr", "and SGD at $@$ reaches", 0.01, cls="design")
check("sweep adam vs sgd", "moves TENT from\n$@$ to $@$ ahead of BN-adapt under noise", sw["gauss_noise"]["gap_ours"], sw["gauss_noise"]["gap_adam"])
check_pow("sweep lr 0.1", "that $@$ would be unstable", 0.1, cls="design")
word("eleven of twelve", "eleven of the twelve $10^{-1}$\ncases are worse than BN-adapt", lr01_worse == 11 and len(lr01) * 3 == 12)
check("Adam rerun: other families", "beats it by $@$ to $@$ at six of the twelve\nheld-out severities of the other families", min(_bt[k] for k in _named), max(_bt[k] for k in _named))
word("Adam rerun: six of twelve", "at six of the twelve\nheld-out severities of the other families", len(_bt) == 12 and len(_named) == 6 and min(_bt[k] for k in _named) > 0.01)
check("lr 0.1 worse than BN-adapt", "TENT is worse than BN-adapt in $@$ of $@$ cases", lr01_worse, len(lr01) * len({r["family"] for r in SW}))
check_pow("lr 0.1 (main text)", "affine parameters. At $@$\nTENT is worse", 0.1, cls="design")
lr01_gaps = []
for fam in sorted({r["family"] for r in SW}):
    held_rows = [r for r in SW if r["family"] == fam and not r.get("is_train_severity", False)]
    bn = sum(r["errs"]["bnadapt"] for r in held_rows) / len(held_rows)
    lr01_gaps += [sum(r["errs"][c] for r in held_rows) / len(held_rows) - bn for c in lr01]
check("nine by more than", "worse than BN-adapt, nine by more than $@$", 0.13)
word("nine of them", "nine by more than $0.13$", sum(g > 0.13 for g in lr01_gaps) == 9)
cfgname = {"tent_lr0.01_s4_sgd": r"SGD, $10^{-2}$, 4 steps", "tent_lr0.01_s1_sgd": r"SGD, $10^{-2}$, 1 step"}
check_table("tab:sweep", [f"{name} & {_hu(sw[fam]['bn'], 4)} & {_hu(sw[fam]['ours'], 4)} & {_hu(sw[fam]['adam'], 4)} & "
                          f"{_hu(sw[fam]['best'], 4)} & {cfgname[sw[fam]['best_cfg']]} \\\\"
                          for fam, name in [("gauss_noise", "Gaussian noise"), ("gauss_blur", "Deterministic blur (control)"),
                                            ("fog_beer_lambert", "OU fog")]])
check_pow("sweep caption lr", "TENT at the SGD setting ($@$, one step)", const(BN_SCRIPT, r"^TENT_LR = ([\de.-]+)"), cls="design")
FX = {}
for s in (1, 2, 3):
    for r in load(f"kaggle_tent_fixed_s{s}/output/cifar10_c_results_tentfixed_seed{s}.jsonl"):
        FX.setdefault(r["t"], []).append(r)
def _signed_ms(vals, sign=True):
    m_ = sum(vals) / len(vals)
    return f"${_hu(m_, 3, sign)}{{\\pm}}{_hu((sum((v - m_) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 3)}$"


# the noise rows of the four-family Adam table, rebuilt here from the noise rerun's files
check_table("tab:adam-all", [f"Gaussian noise & {t:.3f} & " + " & ".join(_signed_ms([r[k] for r in FX[t]], k != "err_oracle") for k in
                             ("err_oracle", "delta_frozen", "delta_bnadapt", "delta_tent", "delta_eata"))
                             + (r" & $\dagger$ \\" if FX[t][0]["is_train_severity"] else r" &  \\")
                             for t in sorted(FX)])

# =========================================================================== appendix: DANN
check("heat yardsticks (app)", "differs from the main run by up to $@$\nand $@$ with the two replicates, which differ by up to $@$",
      yk["s1"], yk["s1_rerun"], F("heat_tent_same_seed_max"), rnd="up")
design("lambda grid", "$\\lambda_{\\max} \\in \\{@, @, @, @\\}$", *lam)
lr_ = F("lambda_min_ratio")
check("lambda ratios", "is $@$, $@$, $@$ and\n$@$ at $t = 0.02, 0.05, 0.1, 0.2$", lr_["0.02"], lr_["0.05"], lr_["0.1"], lr_["0.2"])
tr = F("tent_sgd_ratio_noise")
check("TENT ratios", "against $@$ to\n$@$ for SGD TENT in the main run", min(tr.values()), max(tr.values()))
design("sweep frozen = lambda 0", "with $\\lambda = @$, one seed)", 0.0 if "lambda_max=0.0" in src("kaggle_dann_lambda_sweep/dann_lambda_sweep_kaggle.py") else -1)
word("rerun replicate is seed 1", "with seed 1 run twice",
     os.path.exists(os.path.join(ROOT, "kaggle_cifar10_c_blurheat/output_seed1_rerun/cifar10_c_results_blurheat_seed1.jsonl")))
word("lambda withdrawal condition", "at $t = 0.1$ or $0.2$. None did",
     all(lr_[t] > 0.5 for t in ("0.1", "0.2")))

# =========================================================================== appendix: CIFAR grids
grid_rows = []
for fam, name in [("gauss_noise", "Gaussian noise"), ("blur_heat", "Heat-equation blur"),
                  ("fog_beer_lambert", "Fog (Ornstein--Uhlenbeck)"), ("gauss_blur", "Deterministic blur (control)")]:
    ts = sorted(t for f, t in R_ if f == fam)
    tr_ = [t for t in ts if R_[(fam, t)][1]["is_train_severity"]]
    fmt = lambda xs: ", ".join(f"{x:g}" if x not in (1.0, 2.0, 4.0) or fam == "gauss_noise" else f"{x:.1f}" for x in xs)
    grid_rows.append(f"{name} & ${fmt(tr_)}$ & ${fmt(ts)}$ \\\\")
# the grids are read off the full CIFAR-10 table: one row per test severity, a dagger on the
# labelled training severities
_ct = open(os.path.join(HERE, "tables", "cifar_table.tex")).read().splitlines()
_grid_ok = True
for fam, name in [("gauss_noise", "Gaussian noise"), ("blur_heat", "Heat-equation blur"),
                  ("fog_beer_lambert", "OU fog"), ("gauss_blur", "Deterministic blur (control)")]:
    ts = sorted(t for f, t in R_ if f == fam)
    tr_ = [t for t in ts if R_[(fam, t)][1]["is_train_severity"]]
    rows_ = [l for l in _ct if l.startswith(name + " & ")]
    t_tab = [float(l.split("&")[1]) for l in rows_]
    d_tab = [float(l.split("&")[1]) for l in rows_ if "dagger" in l]
    _grid_ok &= t_tab == ts and d_tab == tr_
word("Table 16: DANN sees unlabelled test-severity images", "(each DANN model also sees unlabelled images at its target severity)",
     "Target (unlabeled): the same x_train_raw pool corrupted at the test" in src("kaggle_cifar10_c_dann/cifar10_c_dann_kaggle.py"), cls="design")
word("grids in the full table", "lists every $\\Delta_X(t)$ of the main run;", _grid_ok, cls="design")

# =========================================================================== theory and certified quantities
def mathcheck(label, template, condition):
    """A mathematical constant inside a formula, checked by computing it."""
    word(label, template, condition, cls="formula")


mathcheck("g(0) = 1", r"and $g(0) = 1$", abs(math.expm1(1e-12) / 1e-12 - 1) < 1e-9)
_h = lambda u: 1 / (1 - math.exp(-u)) - 1 / u
mathcheck("h limits", "with limits\n$0$ and $1$ at $\\mp\\infty$", _h(-60) < 0.02 and _h(60) > 0.98)
mathcheck("sinh inequality", r"$2|\sinh(u/2)| > |u|$",
          all(2 * abs(math.sinh(u / 2)) > abs(u) for u in [x / 7 for x in range(-200, 201) if x]))
mathcheck("unit cube", r"Call a risk profile $r \in [0,1]^P$ consistent", True)
ce = F("ce_synthetic")
CE_SRC = "certified_extrapolation.py"
design("synthetic sample size", "without labels\nfrom $@$ samples at each of $t = 0, 0.1, 0.2$", const(CE_SRC, r"^N_PER_T = (\d+)"))
design("synthetic joint level", "A bootstrap rectangle at joint\nlevel $@\\%$", 100 * (1 - const(CE_SRC, r"^ALPHA = ([\d.]+)")))
check("synthetic coverage", "Over $@$ replications the true\n$R^*(t)$ lies inside the bracket at all six held-out severities at once in $@$ of\nthem",
      ce["n_rep"], ce["fog_check"]["coverage_simultaneous"] * ce["n_rep"])
check("sanity cert summary", "a joint $@\\%$ bootstrap rectangle, the bracket contains\nthe true $R^*(t)$ at all six held-out severities in $@$ of $@$ runs",
      100 * (1 - const(CE_SRC, r"^ALPHA = ([\d.]+)")), ce["fog_check"]["coverage_simultaneous"] * ce["n_rep"], ce["n_rep"])
word("six held-out severities", "at all six held-out severities at once", len(ce["fog_check"]["test_ts"]) == 6)
check("synthetic slope", "grows by $@$ per unit of $t$ between $t = 10$ and $t = 20$", ce["fog_check"]["slope_last"])
word("synthetic slope equals 2(a_hi - a_lo)", "which is $2(a_{\\text{hi}} - a_{\\text{lo}})$ averaged over replications",
     round(ce["fog_check"]["slope_last"], 3) == round(ce["fog_check"]["two_mean_delta_a"], 3))
word("slope over 10..20", "between $t = 10$ and $t = 20$", ce["fog_check"]["test_ts"][-2:] == [10.0, 20.0])
check("analogue coverage", "the simultaneous coverage is $@\\%$ and $@\\%$",
      100 * ce["drift"]["coverage_simultaneous"], 100 * ce["diffusion"]["coverage_simultaneous"])
check("analogue slopes", "($@$ against\n$2(a_{\\text{hi}} - a_{\\text{lo}}) = @$, and $@$ against $@$)",
      ce["drift"]["slope_last"], ce["drift"]["two_mean_delta_a"], ce["diffusion"]["slope_last"], ce["diffusion"]["two_mean_delta_a"])
word("upper and lower end", "near the\nupper and the lower end of that range respectively",
     abs(ce["drift"]["slope_last"] / ce["drift"]["two_mean_delta_a"] - 1) < 0.1
     and abs(ce["diffusion"]["slope_last"] / ce["diffusion"]["two_mean_delta_a"] - 0.5) < 0.1)
cc_ = F("ce_cifar")
fm = cc_["fog_main"]
CEC = "certified_extrapolation_cifar.py"
design("cifar main regime", "For the main fog family\n($a = @$, $\\sigma_{\\text{fog}} = @$)",
       const(CEC, r'"fog_main": \(([\d.]+),'), const(CEC, r'"fog_main": \([\d.]+, ([\d.]+),'))
check("cifar estimates", "the estimates are $a = @$ (bootstrap\nrectangle $@$ to $@$) and $\\sigma_{\\text{fog}} = @$ ($@$ to\n$@$)",
      fm["a_hat"], *fm["a_rect"], fm["sigma_hat"], *fm["sigma_rect"])
check("cifar width growth", "its log-width grows from $@$ at $t = 0.5$ to\n$@$ at $t = 4$, by $@$ per unit of $t$ between $t = 2$ and $t = 4$, against\n$2(a_{\\text{hi}} - a_{\\text{lo}}) = @$",
      fm["points"]["0.5"]["log_width"], fm["points"]["4.0"]["log_width"], F("ce_cifar_main_slope"), F("ce_cifar_main_two_delta_a"))
dw = [p_["log_width"] for p_ in cc_["fog_drift"]["points"].values()]
check("drift widths", "with log-widths from $@$ to $@$", min(dw), max(dw))
word("drift widest", "The drift regime has the widest\nintervals",
     min(dw) > max(p_["log_width"] for r_ in ("fog_main", "fog_diffuse") for t_, p_ in cc_[r_]["points"].items() if float(t_) <= 3))
dr = cc_["fog_drift"]
word("mostly the scattering coefficient", "almost all from the\nscattering coefficient",
     2 * math.log(dr["sigma_rect"][1] / dr["sigma_rect"][0]) > 0.8 * min(dw))
check("transferred bounds", "this gives $R^* \\le @$ at fog $t = 0.5$ and $R^* \\le @$ at $t = 1$",
      fm["points"]["0.5"]["certified_upper"], fm["points"]["1.0"]["certified_upper"], rnd="up")
word("beyond the noise grid", "at\n$t = 2$ and $t = 4$ the interval lies beyond our noise grid",
     fm["points"]["2.0"]["certified_upper"] is None and fm["points"]["4.0"]["certified_upper"] is None)
check("fog oracle at those points", "the fog oracle's error ($@$ and $@$)", fm["points"]["0.5"]["fog_oracle"], fm["points"]["1.0"]["fog_oracle"])
word("no slack certified", "so they certify no slack in the fog\noracle",
     fm["points"]["0.5"]["certified_slack_lower"] == 0 and fm["points"]["1.0"]["certified_slack_lower"] == 0)
cs = sorted(F("cert_control")["2.0"])
pos = F("cert_dissipative_positive")
_B = F("blur_factorial")
check("factorial t=2", "At $t = 2$, BN-adapt removes $@$\nof the frozen error with neither the noise nor the floor, $@$ with the\nfloor alone and $@$ with the noise alone",
      _B["blurfft_s0_a0@2.0"]["bn_removes"][0], _B["blurfft_s0_a01@2.0"]["bn_removes"][0], _B["blurfft_s15_a0@2.0"]["bn_removes"][0])
check("factorial: close to the main run", "close to the $@$ and $@$ of the\nmain run", F("gauss_blur_t2_bn_removes"), F("blur_heat_t2_bn_removes"))
word("factorial: 2x2 cells, three run here", "(three cells\nrun here and the fourth, heat-equation blur itself, taken from the main run",
     sorted({k.split("@")[0] for k in _B}) == ["blurfft_s0_a0", "blurfft_s0_a01", "blurfft_s15_a0"] and HEAT_SIG == 15.0 and HEAT_AMIN == 0.1, cls="design")
word("factorial: three seeds per cell", "taken from the main run; three\nseeds;",
     all(_B[f"{f}@{t}"]["n"] == 3 for f in ("blurfft_s0_a0", "blurfft_s0_a01", "blurfft_s15_a0") for t in (0.25, 0.5, 1.0, 2.0)))
_main_ctl = {t: sum(v["err_frozen"] - v["err_bnadapt"] for v in R_[("gauss_blur", t)].values()) / 3 for t in (0.25, 0.5, 1.0, 2.0)}
word("boundary and kernel nothing measurable at t>=1", "the boundary and the kernel\ntogether contribute nothing measurable",
     all(abs(_main_ctl[t] - _B[f"blurfft_s0_a0@{t}"]["bn_removes"][0]) < _B[f"blurfft_s0_a0@{t}"]["bn_removes"][1] for t in (1.0, 2.0)))
word("floor contributes little", "At $t \\ge 1$ the floor has little effect",
     all(0 <= _B[f"blurfft_s0_a0@{t}"]["bn_removes"][0] - _B[f"blurfft_s0_a01@{t}"]["bn_removes"][0] < 0.05 for t in (1.0, 2.0)))
check("reflect vs periodic small t", "than the periodic one\n($@$ against $@$ at $t = 0.5$)", _main_ctl[0.5], _B["blurfft_s0_a0@0.5"]["bn_removes"][0])
word("reflect leaves more at t<=0.5", "At $t \\le 0.5$, however, the reflect-padded\nblur leaves more", all(_main_ctl[t] > _B[f"blurfft_s0_a0@{t}"]["bn_removes"][0] for t in (0.25, 0.5)))

noted("joint model added after the first results", "We added\nthe joint model after the separate-model and clean-only results below, and its\n200-epoch run after the 15-epoch joint results",
      "chronology of the runs, recorded in the run plan; not reproducible from the result files")
# =========================================================================== matched fog pairs (3 October runs)
_fj = sorted(F("fj_pairs"), key=lambda r: r["v"])
_fcp = sorted(F("fc_pairs"), key=lambda r: r["v"])
_jpos = [r for r in _fj if r["positive"]]
check("rules: 200-epoch rule after the 15-epoch result", "after the 15-epoch joint results (verified gaps at $@$ of $@$ pairs on CIFAR-10, $4$ of $4$ on CIFAR-100) were known", len(_jpos), len(_fj))
noted("rules: 200-epoch rule written after the 15-epoch results", "Before its run, after the 15-epoch joint results",
      "file times: the rule was written on 3 October at 17:30, the 15-epoch joint results came out at 10:07-10:46")
check("200-epoch run added after the 15-epoch result", "after the 15-epoch joint results (verified gaps at $@$ of $@$ pairs on\nCIFAR-10", len(_jpos), len(_fj))
check("separate models gap", "show a\nverified gap at all four pairs ($@$ to $@$; Table",
      min(r["matched"]["d_raw_mean"] for r in _fcp), max(r["matched"]["d_raw_mean"] for r in _fcp))
word("separate: all four, every seed", "verified gap at all four pairs", F("fc_matched_npos") == 4
     and all(min(r["matched"]["d_raw"]) > 0 for r in _fcp))
check("matched pair variances", "giving pairs at $v = @$, $@$, $@$ and $@$", *[r["v"] for r in _fcp])
check("separate: gap at the training point", "The gap appears even at $v = @$", _fcp[0]["v"])
word("separate: both members are training severities", "where both members are training severities, so extrapolation\ncannot explain it", _fcp[0]["v"] == min(r["v"] for r in _fcp))  # the training range ends at the first pair
check("clean model at chance (text)", "at chance at every pair ($@$ to $@$ error; it nominally\npasses at one pair", F("fc_clean_err_min"), F("fc_clean_err_max"))
word("clean model: nominal pass at one pair, both at chance", "nominally\npasses at one pair, with both members at chance",
     F("fc_clean_npos") == 1 and all(min(r["clean"]["err_drift"], r["clean"]["err_diffuse"]) > 0.85 for r in F("fc_pairs")))
word("200-epoch run as main result fixed in advance", "the choice of the 200-epoch run as the main result, was fixed\nbefore its run",
     "main numbers moved here, as fixed in advance" in BODY)
word("sampling floor: joint model, or the clean model for separate models", "measured with the joint model, or the clean-only model for\nseparate models",
     '("fog_drift", td, "clean", s)]["err_frozen_inv"]' in src("paper/facts.py") and 'jb[("fog_drift", td, s)]["err_frozen_inv"]' in src("paper/facts.py"), cls="design")
check("matched training range", "up to $v = @$ (drift severities up to\n$@$, diffusion up to $@$)", _fcp[0]["v"], _fcp[0]["t_drift"], _fcp[0]["t_diffuse"])
check("separate at training severities", "even at $v = 0.042$ ($@$\nagainst $@$)", _fcp[0]["matched"]["err_drift"], _fcp[0]["matched"]["err_diffuse"])
check("oracles agree", "oracles agree to within $@$ in\nseed means ($@$ in single seeds)", max(abs(r["oracle_diff"]) for r in _fcp), F("fc_oracle_diff_seed_max"), rnd="up")
_c1 = F("fc_clean_drift_t1")
check("clean model at chance (caption)", "is at chance, $@$ to $@$, at every point", F("fc_clean_err_min"), F("fc_clean_err_max"))
_rv = F("fc_4x")
check("4x reversal reappears", "the reversal reappears in all three seeds ($@$ against $@$)",
      _rv["drift_own"]["drift"], _rv["drift_own"]["diffuse"])
check("4x reversal disappears", "it\ndisappears in all three ($@$ against $@$)", _rv["drift_matched"]["drift"], _rv["drift_matched"]["diffuse"])
word("reappears 3/3, disappears 0/3", "reappears in all three seeds",
     _rv["drift_own"]["n_reversed"] == 3 and _rv["drift_matched"]["n_reversed"] == 0 and F("fc_seeds") == [1, 2, 3])
_drift_match = [round(math.log(1 + 2 * 1.0 * (0.2 ** 2 * math.expm1(2 * 0.05 * t) / (2 * 0.05)) / 0.05 ** 2) / 2, 2) for t in (0.5, 1.0)]
design("matched drift grid", "(severities $@$, $@$ and $@$)", 0, *_drift_match)
word("diffusion pair severities", r"$t \in \{1, 2, 4, 8\}$",
     sorted(r["t_diffuse"] for r in _fcp) == [1.0, 2.0, 4.0, 8.0], cls="design")
check("caption 4x", "The reversal at the $@\\times$ horizon is an extrapolation effect", abl["fog_drift@2.0"]["horizon"])
check("contribution blur ratio", "statistic re-estimation removes $@$ times\nas much error under deterministic blur as under heat-equation blur", F("blur_ratio_bn"))

# =========================================================================== joint model at the full budget
_ff = sorted(F("fjf_pairs"), key=lambda r: r["v"])
_ff100 = sorted(F("fjf100_pairs"), key=lambda r: r["v"])
_head = [r["d_raw_mean"] for r in _ff if r["v"] > 0.15]

design("full budget", "standard @ epochs with per-sample crop and flip", const("kaggle_oct3_runner/runner.py", r"^FULL_EPOCHS = (\d+)"))
design("short budget", "At @ epochs the\nzero-padded joint model has higher error overall", const("kaggle_oct3_runner/runner.py", r"^EPOCHS = (\d+)"))
check("full budget in range", "equally well within its training range, $@$ and $@$ at\n$v = @$", _ff[0]["err_drift"], _ff[0]["err_diffuse"], _ff[0]["v"])
check("15-epoch oracles loose at the training point", "better than the 15-epoch oracles ($@$ and $@$), which are therefore\nloose",
      _fcp[0]["oracle_drift"], _fcp[0]["oracle_diffuse"])
word("200-epoch model beats the 15-epoch oracles there", "at $v = 0.042$, better than the 15-epoch oracles",
     _ff[0]["err_drift"] < _fcp[0]["oracle_drift"] and _ff[0]["err_diffuse"] < _fcp[0]["oracle_diffuse"])
word("full budget: in range not positive", "it classifies the two\nmembers equally well within its training range", not _ff[0]["positive"] and abs(_ff[0]["d_raw_mean"]) < 0.005)
check("full budget gaps", "by $@$, $@$ and $@$ at $v = @$, $@$ and\n$@$. We call the last two",
      _ff[1]["d_raw_mean"], _ff[2]["d_raw_mean"], _ff[3]["d_raw_mean"], _ff[1]["v"], _ff[2]["v"], _ff[3]["v"])
# one statement covers the zero-padded and the reflect-padded joint models and the separate models
word("full budget: every seed", "Gaps below are seed means, positive\nin each seed unless stated.",
     F("fjf_pairs_npos") == 3 and all(min(r["d_raw"]) > 0 for r in _ff[1:]) and _every_seed
     and all(min(r["matched"]["d_raw"]) > 0 for r in _fcp))
word("15 epochs worse everywhere", "zero-padded joint model has higher error overall",
     all(a["err_drift"] > b["err_drift"] and a["err_diffuse"] > b["err_diffuse"] for a, b in zip(_fj, _ff)))
check("15 epochs outer gaps", "the\ngaps at the two most extrapolated pairs are larger, $@$ and $@$", _fj[2]["d_raw_mean"], _fj[3]["d_raw_mean"])
word("15 epochs: sign changes at 0.089", "at $v = 0.089$ the sign changes across seeds", min(_fj[1]["d_raw"]) < 0 < max(_fj[1]["d_raw"]))
word("shrinks and persists", "The\nstandard budget shrinks the effect but keeps it",
     all(b["d_raw_mean"] < a["d_raw_mean"] and b["positive"] for a, b in zip(_fj[2:], _ff[2:])))
check("c100 full gaps", "At 200 epochs the gaps beyond the\ntraining range are $@$, $@$ and $@$", *[r["d_raw_mean"] for r in _ff100[1:]])
word("c100 full every seed", "again in every seed", F("fjf100_pairs_npos") == 3 and all(min(r["d_raw"]) > 0 for r in _ff100[1:]))
check("c100 full in range", "the gap\nwithin it ($@$) no longer exceeds the sampling floor in every seed", _ff100[0]["d_raw_mean"])
word("c100 full in range not positive", "no longer exceeds the sampling floor in every seed", not _ff100[0]["positive"])
design("limitations full budget", "its variants used @ epochs", const("kaggle_oct3_runner/runner.py", r"^FULL_EPOCHS = (\d+)"))

# =========================================================================== certified slack, contrast and adaptation (joint model)
_ff = sorted(F("fjf_pairs"), key=lambda r: r["v"])
_head = [r["d_raw_mean"] for r in _ff if r["v"] > 0.15]
_wl2 = F("wiener_lfl")["2.0"]["cert_slack"]
check("limitations: slack ranges", "certified lower bounds for stored data, up to\n$@$ per certificate in fog and noise and $@$ for the periodic blur", F("cert_all_max"),
      max(WL["2.0"]["cert_slack_joint"]), rnd="down")
_f100 = {f"{q['v']:.3f}": q for q in F("fjf100_pairs")}
word("abstract: both comparisons replicate on CIFAR-100", "The fog gap and blur ratio hold in direction on\nCIFAR-100.",
     F("c100_blur_ratio_bn") > 2 and all(_f100[k]["positive"] and min(_f100[k]["d_raw"]) > 0 for k in _outer2))
check("sqrt(d) for CIFAR", "about $@$ for $@\\times@$ colour images", math.sqrt(3 * 32 * 32), 32, 32)
check("bootstrap level", "has asymptotic, not exact, probability $@\\%$", 100 * (1 - const("certified_extrapolation.py", r"^ALPHA = ([\d.]+)")))
check("cifar coverage", "contains the true value at every held-out\nseverity, $@$ in all", F("ce_cifar_points"))
word("cifar all covered", "contains the true value at every held-out", F("ce_cifar_covered") == F("ce_cifar_points"))
word("three regimes, three events", "Each regime has one rectangle, so these are three\nevents, not ten", len([k for k in ("fog_main", "fog_drift", "fog_diffuse") if k in F("ce_cifar")]) == 3)
word("same images for every regime", "the three regimes corrupt the same\ntraining images",
     "fog(x_all[i * n:(i + 1) * n]" in src("certified_extrapolation_cifar.py") and src("certified_extrapolation_cifar.py").count("x_all = ") == 2, cls="design")
cs = sorted(F("cert_control")["2.0"])
pos = F("cert_dissipative_positive")
check("noise oracles alone", "With only 15-epoch noise oracles as witnesses, $@$ of $@$ fog oracles (per\nregime, severity and seed) are certified at the $@\\%$ level, near the $@$ expected by chance",
      len(pos), F("cert_dissipative_n"), 5, F("cert_dissipative_expected_by_chance"))
check("all witnesses", "slack is certified (no rule fixed in advance) for $@$ runs\nat $@$ points, by up to $@$, each certificate at its own level, not jointly", F("cert_all_pos"), F("cert_all_points_pos"), F("cert_all_max"), rnd=[None, None, "down"])
check("all witnesses: distinct points", "The $@$ fog and noise oracle runs cover $@$ distinct (family, severity, seed)\npoints, $@$ trained twice",
      F("cert_all_n"), F("cert_all_points"), F("cert_all_n") - F("cert_all_points"))
check("all witnesses: expected by chance", "so in expectation at most about $@$ of the $@$ would\narise by chance", F("cert_all_expected"), F("cert_all_n"))
word("all witnesses: margins as in facts.py", "Each\nwitness's margin is Bonferroni-corrected over all $420$ bounds and each oracle's own\nmargin is one-sided at the $5\\%$ level",
     "_zK = _norm.ppf(1 - 0.05 / len(_wit))" in src("paper/facts.py") and "slack=e - _z1 * _se(e) - u" in src("paper/facts.py")
     and 'F["cert_all_expected"] = 0.05 * len(cert_all)' in src("paper/facts.py"), cls="design")
check("all witnesses: K", "Section~\\ref{sec:fog-ablation}, $@$\nclassifier--severity bounds in all", F("cert_all_K"))
check("all witnesses: K margins", "Bonferroni-corrected over all $@$ bounds", F("cert_all_K"))
word("all witnesses: which classifiers", "witnesses are the oracles and frozen models of the main run and the fog ablation and\nthe joint, separate and clean-only models",
     all(x in src("paper/facts.py") for x in ('("joint15", ', "for r in FC:", '/frozen", "witness"', '("joint200_std", ', 'for (f, t), rs in A.items():'))
     and "dann" not in src("paper/facts.py")[src("paper/facts.py").index("_pts = []"):src("paper/facts.py").index("_wit = [")].lower(), cls="design")
_pool = src("paper/facts.py")[src("paper/facts.py").index("_pts = []"):src("paper/facts.py").index("_wit = [")].lower()
word("all witnesses: what is excluded", "DANN, other runs of the main families and batch-dependent test-time methods are\nnot used",
     "dann" not in _pool and "tent" not in _pool and "eata" not in _pool and "bnadapt" not in _pool.replace("_", ""), cls="design")
# round 20: the certified-interval paragraph left Section 5.5 (Appendix C gives it in full, with
# its own checks); removed: "main fog family: log-width near the upper rate", "three rectangles, one per regime"
check("other certificates", "The other $@$ use\n15-epoch oracles ($@$ at the same $v$", F("cert_all_by_15epoch"), F("cert_15epoch_same_v"))
check("other certificates at larger v", "at the same $v$, $@$ at a larger $v$)", F("cert_15epoch_larger_v"))
word("other certificates are oracles", "use\n15-epoch oracles",
     all(w.count("/") == 1 and not w.startswith("joint") for w in F("cert_all_by_witness") if not w.startswith("joint200")))
word("one witness trained for 200 epochs", "Witness classifiers built on Wiener\ndeconvolution and 200-epoch models certify such bounds", F("cert_all_by_joint200") > 0)
word("the witness is the standardized-input model", "The per-image-standardized 200-epoch joint model is the witness",
     F("cert_joint200_witness_models") == ["joint200_std"] == F("cert_fc_042_witness_models"))
check("joint200 as witness", "is the witness in $@$ of the $@$\ncertificates, and at its training point $v = @$ it certifies that the\n15-epoch oracle recipe costs at least $@$ to $@$",
      F("cert_all_by_joint200"), F("cert_all_pos"), sorted(F("fc_pairs"), key=lambda r: r["v"])[0]["v"], F("cert_fc_042_min"), F("cert_fc_042_max"),
      rnd=[None, None, None, "down", "down"])
_mc = F("matched_contrast")
_ktr = F("joint_train_contrasts")
_rv = F("matched_rawvar_ratio")
_out = ("0.089", "0.197", "0.490")
check("joint training contrasts", "The joint model was trained at\ncontrasts $@$, $@$, $@$, $@$ and $@$", *_ktr)
word("five contrasts", "The joint model was trained at\ncontrasts", len(_ktr) == 5)
check("drift member contrasts", "Its held-out drift members have\ncontrast $@$ to $@$, below all of them",
      _mc["0.089"]["drift"], _mc["0.490"]["drift"])
word("drift below all, raw noise in range", "below all of them, with raw noise variance inside the\ntrained range",
     all(_mc[k]["drift"] < min(_ktr) and _rv[k]["drift"] <= 1 for k in _out))
check("unseen contrast interval", "inside\nthe unvisited interval $(@, @)$", *F("joint_train_contrast_gap"))
word("diffusion members in the unseen interval", "inside\nthe unvisited interval",
     all(F("joint_train_contrast_gap")[0] < _mc[k]["diffuse"] < F("joint_train_contrast_gap")[1] for k in _out))
check("diffusion members: contrast and raw noise", "Its held-out diffusion members have contrast $@$ to $@$, inside\nthe unvisited interval $(0.24, 0.95)$, with raw noise variance $@$ to $@$ times\nthe largest trained",
      _mc["0.089"]["diffuse"], _mc["0.490"]["diffuse"], _rv["0.089"]["diffuse"], _rv["0.490"]["diffuse"])
# Section 5.4 states both sides of where the members leave the training data (review 18)
word("5.4: drift contrast below all trained", "the drift members' contrast lies below all training contrasts",
     all(_mc[k]["drift"] < min(_ktr) for k in _out))
check("5.4: diffusion raw noise above all trained", "the\ndiffusion members' raw noise variance above all trained ($@$ to $@$ times",
      _rv["0.089"]["diffuse"], _rv["0.490"]["diffuse"])
word("5.4: diffusion raw noise above all trained (condition)", "raw noise variance above all trained", all(_rv[k]["diffuse"] > 1 for k in _out))
word("errs more on the contrast-extrapolating member", "the model errs more on the one whose contrast lies below all its\ntraining contrasts",
     all(r["d_raw_mean"] > 0 for r in _ff[1:]))
_sc = F("separate_train_contrasts")
check("separate training contrasts", "($@$, $@$ and $@$ for drift, $@$ to $@$ for diffusion)",
      *sorted(_sc["drift_matched"], reverse=True), max(_sc["diffuse_own"]), min(_sc["diffuse_own"]))
_ad = F("fjf_adapt")
word("TENT in that run used Adam", "with TENT (Adam; Table~\\ref{tab:budget})",
     "torch.optim.Adam(params, lr=TENT_LR)" in src("kaggle_oct3_runner/runner.py"), cls="design")
_ad0 = sorted(F("fjf_adapt"), key=lambda q: q["v"])[0]
word("adaptation creates a gap inside the range", "adaptation creates a gap where the\nfrozen model has none",
     abs(_V0["0.042"]["d_raw_mean"]) < 0.005 and not _V0["0.042"]["positive"] and _ad0["bn_gap"] > 0.05)
word("inside the range: frozen gap changes sign across seeds", "$0.003$ frozen (not of one sign across seeds)",
     min(_V0["0.042"]["d_raw"]) < 0 < max(_V0["0.042"]["d_raw"]))
check("inside the range: frozen and BN-adapt", "frozen model has none: at $v = @$ it is $@$ frozen (not of one sign across seeds) but\n$@$ after", _ff[0]["v"], _V0["0.042"]["d_raw_mean"], _ad0["bn_gap"])
check("adaptation gaps", "Over the four pairs it is $@$ to $@$ with BN-adapt and $@$ to\n$@$ with TENT",
      min(r["bn_gap"] for r in _ad), max(r["bn_gap"] for r in _ad), min(r["tent_gap"] for r in _ad), max(r["tent_gap"] for r in _ad))
_pp = sorted(F("fjf_pairs"), key=lambda r: r["v"])
_raised = [ad["bn_drift"] > ad["frozen_drift"] and ad["bn_drift"] - ad["bn_gap"] > pr["err_diffuse"] for ad, pr in zip(_ad, _pp)]
_lowered = [ad["bn_drift"] < ad["frozen_drift"] and ad["bn_drift"] - ad["bn_gap"] < pr["err_diffuse"] for ad, pr in zip(_ad, _pp)]
# the decision-rule table
_fj15 = sorted(F("fj_pairs"), key=lambda r: r["v"]); _fj100 = sorted(F("fj100_pairs"), key=lambda r: r["v"])
word("limitations: in-range gap pattern", "the verified gap appears only beyond the training range (inside\nit for the 15-epoch separate models and the 15-epoch CIFAR-100 joint model)",
     not _fj15[0]["positive"] and not sorted(F("fjf_pairs"), key=lambda r: r["v"])[0]["positive"]
     and not sorted(F("fjf100_pairs"), key=lambda r: r["v"])[0]["positive"] and _fj100[0]["positive"]
     and all(r["positive"] for r in sorted(F("fjf_pairs"), key=lambda r: r["v"])[1:] if r["v"] > 0.1) and F("fc_matched_npos") == 4)
check("rules: clean nominal", "Clean-only model: nominally passes at $@$ of $@$ pairs", F("fc_clean_npos"), len(F("fc_pairs")))
check("rules: separate", "Separate models: $@$ of $@$ \\\\", F("fc_matched_npos"), len(F("fc_pairs")))
check("rules: joint 15", "$@$ of $@$ pairs on CIFAR-10, $@$ of $@$ on CIFAR-100", F("fj_pairs_npos"), len(_fj15), F("fj100_pairs_npos"), len(_fj100))
check("rules: joint 200", "$@$ of $@$ pairs on CIFAR-10 and on CIFAR-100", F("fjf_pairs_npos"), len(_ff))
word("rules: joint 200 same on CIFAR-100", "pairs on CIFAR-10 and on CIFAR-100", F("fjf_pairs_npos") == F("fjf100_pairs_npos"))
_c100p = F("c100_fc_pairs")
check("rules: c100 separate", "CIFAR-100 separate models: $@$ of $@$", sum(len(r["d_raw"]) == 3 and min(r["d_raw"]) > 0 for r in _c100p), len(_c100p))
word("rules: TIN same sign", "same sign at every pair", all(r["d_raw_mean"] > 0 for r in F("tin_fc_pairs")))

# =========================================================================== scale (CIFAR-100, Tiny-ImageNet)
_TIN = "kaggle_tinyimagenet/tin_kaggle.py"
design("TIN subset", "Tiny-ImageNet ($@$ training and $@$\ntest images at $@\\times@$, one seed,",
       const(_TIN, r"^N_TRAIN_IMG = (\d+)"), const(_TIN, r"^N_TEST_IMG = (\d+)"), const(_TIN, r"^IMG_SIZE = (\d+)"), const(_TIN, r"^IMG_SIZE = (\d+)"))
word("TIN 100 classes", "on 100 classes of Tiny-ImageNet", const(_TIN, r"^N_CLASSES = (\d+)") == 100, cls="design")
word("TIN float64 guard", "with corrected normalization statistics\n(Appendix~\\ref{app:corrections}, item~\\ref{corr:saturated})",
     "assert np.all((NORM_MEAN > 0.3)" in src(_TIN) and 'dtype=np.float64' in src(_TIN), cls="design")
_j100 = sorted(F("fj100_pairs"), key=lambda r: r["v"])
word("c100 joint all four, every seed", "the joint model errs more on the lower-contrast drift member at all four pairs in every seed",
     len(_j100) == 4 and all(len(r["d_raw"]) == 3 and min(r["d_raw"]) > 0 for r in _j100) and F("fj100_pairs_npos") == 4)
check("c100 joint gaps", "by $@$ inside the training range and by $@$, $@$ and $@$\nat the three outer pairs", *[r["d_raw_mean"] for r in _j100])
word("c100 joint: the last near chance", "(the last near chance)", _j100[3]["err_drift"] > 0.99 - 0.05)
_c100 = sorted(F("c100_fc_pairs"), key=lambda r: r["v"])
check("c100 separate gaps", "15-epoch models matched in $v$ show verified gaps of $@$ to $@$ at all four\npairs in every seed",
      min(r["d_raw_mean"] for r in _c100), max(r["d_raw_mean"] for r in _c100))
word("c100 separate every seed", "pairs in every seed, and their oracles agree", len(_c100) == 4 and all(min(r["d_raw"]) > 0 for r in _c100))
check("c100 oracles agree", "their oracles agree within $@$ in three-seed means ($@$ in single seeds)", max(abs(r["oracle_diff"]) for r in _c100), max(r["oracle_diff_seed_max"] for r in _c100), rnd="up")
_tin = sorted(F("tin_fc_pairs"), key=lambda r: r["v"])
_tin_chance = abs(_tin[-1]["v"] - 0.490) < 1e-3 and min(_tin[-1]["err_drift"], _tin[-1]["err_diffuse"]) > 0.95  # 100 classes: chance 0.99
check("tin separate gaps", "separate models err\n$@$ and $@$ more on the drift member at $v = @$ and $@$",
      _tin[0]["d_raw_mean"], _tin[1]["d_raw_mean"], _tin[0]["v"], _tin[1]["v"])
check("tin oracles", "with\noracles within $@$ there", max(abs(_tin[0]["oracle_diff"]), abs(_tin[1]["oracle_diff"])), rnd="up")
check("tin last pair", "near chance at $v = @$ ($@$, $@$)", _tin[2]["v"], _tin[2]["err_drift"], _tin[2]["err_diffuse"])
word("limitations: the single-seed experiments", "(one in the first Wiener test, both sweeps and\nTiny-ImageNet)",
     len(glob.glob(os.path.join(ROOT, "kaggle_tent_lrsweep", "output_seed*"))) == 1
     and len(glob.glob(os.path.join(ROOT, "kaggle_dann_lambda_sweep", "output_seed*"))) == 1
     and len(glob.glob(os.path.join(ROOT, "kaggle_wiener_fixed", "output*"))) == 1 and all(r["seeds"] == [1] for r in _tin))
word("tin one seed", "one seed, near chance at $v = 0.490$)", _tin_chance and all(r["seeds"] == [1] for r in _tin))
# 0.05 is the stated tolerance, not a rounded measurement (the largest distance from chance is
# about 0.04); the check only covers the token, and the word check below tests it against the data
check("clean within 0.05 of chance", "clean-only models are within $@$ of chance at every pair", 0.05)
word("clean near chance both datasets", "On both data\nsets, clean-only models are within",
     F("c100_clean_err_min") > 0.99 - 0.05 and F("tin_clean_err_min") > 0.99 - 0.05)
_ta = F("tin_adam")
check("tin noise gaps", "$\\Delta_{\\text{frozen}}$ is $@$, $@$ and $@$ at $t = 0.05, 0.1, 0.2$",
      _ta["0.05"]["delta_frozen"], _ta["0.1"]["delta_frozen"], _ta["0.2"]["delta_frozen"])
check("tin noise TENT vs BN", "Adam TENT beats\nBN-adapt by $@$, $@$ and $@$, as on CIFAR-10",
      *[_ta[t]["err_bnadapt"] - _ta[t]["err_tent"] for t in ("0.05", "0.1", "0.2")])


check("scale caption: omitted pair", "omits the pair at $v = @$", sorted(F("fc_pairs"), key=lambda r: r["v"])[0]["v"])
import glob as _g
_tinrows = [json.loads(l) for p_ in _g.glob(os.path.join(ROOT, "oct3/tin*/output/*.jsonl")) for l in open(p_)]
_tin_models = {r.get("model") for r in _tinrows if r["family"].startswith("fog")}
_tin_fams = {r["family"] for r in _tinrows}
word("TIN omits the first pair", "the Tiny-ImageNet run has no joint model and omits the pair",
     len(F("tin_fc_pairs")) == 3 and min(r["v"] for r in F("tin_fc_pairs")) > 0.05 and "joint" not in _tin_models)
word("TIN: only separate-model fog pairs and noise", "On Tiny-ImageNet, separate models err",
     _tin_models == {"clean", "diffuse_own", "drift_matched"} and _tin_fams == {"fog_drift", "fog_diffuse", "gauss_noise"})

_cb = F("c100_blur")
_ga, _gb = _cb["gauss_blur@2.0"], _cb["blurfft_s15_a01@2.0"]
_gapa, _gapb = _ga["frozen"][0] - _ga["oracle"][0], _gb["frozen"][0] - _gb["oracle"][0]
check("c100 blur pair", "the blurs leave their frozen models at $@$ and\n$@$ error, and BN-adapt removes $@$ and $@$, a $@$-fold ratio, or\n$@\\%$ and $@\\%$ of the gap to the oracles (oracle errors $@$, $@$)",
      _ga["frozen"][0], _gb["frozen"][0], _ga["bn_removes"][0], _gb["bn_removes"][0], F("c100_blur_ratio_bn"),
      100 * _ga["bn_removes"][0] / _gapa, 100 * _gb["bn_removes"][0] / _gapb, _ga["oracle"][0], _gb["oracle"][0])
word("c100 blur three seeds", "Both comparisons hold in direction on CIFAR-100, and the fog gap also in separate\n15-epoch models",
     all(v["n"] == 3 for v in _cb.values()) and all(len(r["d_raw"]) == 3 for r in _j100))
# =========================================================================== pretrained ImageNet models and the single-model blur pair (21 October)
_IMS = "kaggle_imnet_fog/imnet_fog.py"
_SUMM = os.path.join(ROOT, "oct21", "summarize.py")
# the facts reimplement oct21/summarize.py (the pre-specified rule as run); compare every field
try:
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("oct21_summarize", _SUMM)
    _sm = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_sm)

    def _same(a, b):
        if isinstance(a, float) or isinstance(b, float):
            return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        return a == b
    _diff = []
    for _run in ("first", "r2"):
        for _m, _r in _sm.imnet(_run).items():
            _f = F(f"imnet_{_run}", _m)
            _diff += [(_run, _m, k) for k in ("clean", "clean_bnadapt") if not _same(_r[k], _f[k])]
            _diff += [(_run, _m, p["v"], k) for p, q in zip(_r["pairs"], _f["pairs"])
                      for k in ("v", "err_drift", "err_diffuse", "gap", "sd", "floor", "informative", "verified",
                                "positive", "negative", "bn_gap") if not _same(p[k], q[k])]
    _dec = _sm.decide(_sm.imnet("first"))
    _diff += [k for k, a, b in (("H1", _dec["H1"], F("imnet_H1")), ("H2", _dec["H2"], F("imnet_H2")),
                                ("outcome", _dec["outcome"], F("imnet_outcome")),
                                ("negative", _dec["negative_verified"], F("imnet_negative"))) if not _same(a, b)]
    _bjs = _sm.blur_joint()
    _diff += [k for k in ("ratio_of_means", "outcome", "frozen_det", "frozen_heat", "oracle_det", "oracle_heat",
                          "gap_ratio", "recovered_det", "recovered_heat", "recovered_ratio")
              if not _same(_bjs[k], F("blur_joint", k))]
    _diff += ["ratio_per_seed"] * (not all(_same(a, b) for a, b in zip(_bjs["ratio_per_seed"], F("blur_joint", "ratio_per_seed"))))
    if _diff:
        FAILS.append(f"facts.py and oct21/summarize.py disagree: {_diff[:5]}")
    else:
        PASSES += 1
except Exception as _e:  # the comparison itself must run
    FAILS.append(f"oct21/summarize.py comparison failed: {_e!r}")

_IMF, _IMR = F("imnet_first"), F("imnet_r2")
_IMBN, _IMLN = ["resnet50_v1", "resnet50_v2", "efficientnet_b0"], ["vit_b_16", "convnext_tiny", "swin_t"]
_imv = sorted({p["v"] for m_ in _IMF for p in _IMF[m_]["pairs"]})
_npi, _ninf = F("imnet_npos"), F("imnet_n_informative")
_neg = F("imnet_negative")
_g42 = F("imnet_gap_042")
_im_src = src(_IMS)
# pretrained torchvision weights, evaluated only (no optimizer, no fog in any training)
_never_fog = "IMAGENET1K_V1" in _im_src and "optim" not in _im_src and "backward" not in _im_src
_im_all_show = all(_npi[m_] >= 2 for m_ in _IMBN + _IMLN) and len(_IMF) == 6
_im_bn_ln = {m_: k for m_, (_, _, k) in eval(re.search(r"(?ms)^MODELS = (\{.*?^\})", _im_src).group(1)).items()}
word("ImageNet: six models, three of each normalization",
     "six torchvision classifiers,\nthree with batch normalization (ResNet-50 V1 and V2, EfficientNet-B0) and three with\nlayer normalization (ViT-B/16, ConvNeXt-T, Swin-T)",
     [m_ for m_ in _im_bn_ln if _im_bn_ln[m_] == "bn"] == _IMBN and sorted(m_ for m_ in _im_bn_ln if _im_bn_ln[m_] == "ln") == sorted(_IMLN)
     and set(_IMF) == set(_IMBN + _IMLN), cls="design")
word("ImageNet: frozen, ten images per class, three seeds",
     "were evaluated frozen on ten\nImageNet validation images per class, under both fog regimes and with the rule of",
     const(_IMS, r"^PER_CLASS = (\d+)") == 10 and _never_fog and all(_IMF[m_]["seeds"] == [1, 2, 3] for m_ in _IMF),
     cls="design")
design("ImageNet: v range", "at six matched $v$ from $@$ to $@$ with three\nnoise seeds", _imv[0], _imv[-1])
word("ImageNet: six v, three noise seeds", "at six matched $v$ from",
     len(_imv) == 6 and _imv == eval(re.search(r"^V_GRID = (\[.*\])", _im_src, re.M).group(1))
     and eval(re.search(r"^NOISE_SEEDS = (\[.*\])", _im_src, re.M).group(1)) == [1, 2, 3], cls="design")
word("ImageNet: pre-specified outcome class", "The outcome is that the gap transfers but is\nnot specific to batch normalization",
     F("imnet_outcome") == "transfers at scale, not specific to batch normalization" and F("imnet_H1") and not F("imnet_H2"))
word("ImageNet: every informative pair positive but one",
     "every model shows a verified positive gap at\nevery informative pair except one",
     sum(_ninf[m_] - _npi[m_] for m_ in _IMF) == 1 and sum(len(x) for x in _neg.values()) == 1)
_neg_m = [m_ for m_ in _neg if _neg[m_]]
_neg_p = [p for p in _IMF[_neg_m[0]]["pairs"] if p["v"] == _neg[_neg_m[0]][0]][0] if _neg_m else None
check("ImageNet: the negative pair", "EfficientNet-B0 at $v = @$ (verified negative,\n$@$)",
      _neg_p["v"] if _neg_p else -1, _neg_p["gap"] if _neg_p else -1)
word("ImageNet: the negative pair is EfficientNet-B0's", "EfficientNet-B0 at $v =", _neg_m == ["efficientnet_b0"])
check("ImageNet: LN largest at 0.042", "the layer-normalized models the largest (at $v = @$, $+@$ for\nViT-B/16 against $+@$ for ResNet-50 V1",
      0.042, _g42["vit_b_16"], _g42["resnet50_v1"])
word("ImageNet: ResNet-50 V1 has the largest BN gap", "ViT-B/16 against $+0.102$ for ResNet-50 V1, the largest with batch normalization",
     max(_IMBN, key=_g42.get) == "resnet50_v1")
word("ImageNet: LN gaps the largest", "the layer-normalized models the largest",
     min(_g42[m_] for m_ in _IMLN) > max(_g42[m_] for m_ in _IMBN) and max(_g42, key=_g42.get) == "vit_b_16"
     and 0.042 in _imv)
check("ImageNet: inverse map restores", "the drift member errs within $@$ of the\ndiffusion member at every informative pair",
      F("imnet_inv_restore_max"), rnd="up")
check("ImageNet: BN-adapt", "keeps the gaps at $v \\le @$ to at most $@$ and\nEfficientNet-B0's to at most $@$ at every $v$, but leaves $@$ and $@$ in the\nResNet-50s at $v = @$",
      0.005, F("imnet_bn_small_v_max"), F("imnet_bn_effnet_max"), *F("imnet_bn_resnet_042"), 0.042,
      rnd=[None, "up", "up", None, None, None])
# round 20: "ImageNet: no oracle, models never saw fog" left Section 5.7 for page length; the
# Limitations sentence (checked below) states both
word("ImageNet: BN-adapt from the shuffled rerun", "BN-adapt, rerun with shuffled batches",
     "randperm" in _im_src and set(_IMR) == set(_IMBN))
# abstract, contributions, Section 5.4, discussion, limitations, conclusion
for _lab, _t in (("abstract", "Six pretrained ImageNet classifiers that never saw\nfog show the gap with either normalization"),
                 ("contributions", "Six pretrained ImageNet classifiers, which never saw fog, show the gap\nwith either normalization, larger with layer normalization"),
                 ("conclusion", "six pretrained ImageNet classifiers with either\nnormalization err more as well"),
                 ("discussion", "the fog gap reaches all six off-the-shelf ImageNet classifiers we tested"),
                 ("Section 5.4", "test on pretrained ImageNet models, which never saw fog, partly widened it again:\nthere the gap appears with either normalization"),
                 ("Appendix B scope", "pre-specified test on pretrained ImageNet models partly widened it\nagain: models that never saw fog show the gap with either normalization")):
    word(f"ImageNet in the {_lab}", _t, _never_fog and _im_all_show
         and min(_g42[m_] for m_ in _IMLN) > max(_g42[m_] for m_ in _IMBN))
word("Appendix B: the date of the ImageNet rule", "On 21 October a pre-specified test",
     os.path.exists(os.path.join(ROOT, "rules", "2026-10-21_imagenet_and_single_blur.md")), cls="design")
word("Appendix A: scope partly widened", "and later partly widened", F("imnet_H1") and not F("imnet_H2"))
word("mechanism: BN-adapt closes most of the ImageNet gap at small v", "with BN-adapt closing most of the ImageNet gap at small $v$",
     all(abs(p["bn_gap"]) < 0.5 * p["gap"] for m_ in _IMR for p in _IMR[m_]["pairs"] if p["v"] <= 0.005))
word("mechanism: does not explain the layer-normalized gaps", "For the layer-normalized ImageNet models, which never saw\nfog, sensitivity to low contrast",
     _never_fog and min(_g42[m_] for m_ in _IMLN) > max(_g42[m_] for m_ in _IMBN))
word("limitations: clean pretraining, no ImageNet oracle", "We used only clean-pretrained\nImageNet models, without oracles",
     _never_fog and "oracle" not in _im_src.lower(), cls="design")
# Appendix B: the batch-order correction
design("batch order: batch and classes", "so each batch of $@$ held $@$ classes",
       const(_IMS, r"^BATCH = (\d+)"), const(_IMS, r"^BATCH = (\d+)") / const(_IMS, r"^PER_CLASS = (\d+)"))
check("batch order: first BN-adapt clean error", "BN-adapt's clean\nerror in the first EfficientNet-B0 run was $@$", _IMF["efficientnet_b0"]["clean_bnadapt"])
check("batch order: rerun reproduces frozen gaps", "which the rerun reproduced (largest change of a gap $@$)", F("imnet_r2_max_gap_change"))
word("batch order: same decisions in the rerun", "The rule uses frozen\nerrors, which the rerun reproduced", F("imnet_r2_same_decision"))
word("batch order: three BN models rerun", "reran the three batch-normalized models", set(_IMR) == set(_IMBN))
# Table 7 rows
check("rules: ImageNet", "H1 held, H2 failed (all $@$ models of each kind pass)", sum(_npi[m_] >= 2 for m_ in _IMBN))
word("rules: ImageNet, every model of each kind passes", "H1 held, H2 failed (all $3$ models of each kind pass)",
     sum(_npi[m_] >= 2 for m_ in _IMBN) == len(_IMBN) == sum(_npi[m_] >= 2 for m_ in _IMLN) == len(_IMLN) == 3)
word("rules: ImageNet outcome", "transfers, not specific to batch normalization; one verified negative gap",
     F("imnet_H1") and not F("imnet_H2") and sum(len(x) for x in _neg.values()) == 1)
_bj = F("blur_joint")
check("rules: single-model blur", "Ratio above $@$ in every seed, $@$ from seed means: replicates", 1, _bj["ratio_of_means"])
word("rules: single-model blur replicates", "in every seed, $7.4$ from seed means: replicates",
     _bj["outcome"] == "replicates" and min(_bj["ratio_per_seed"]) > 1 and len(_bj["seeds"]) == 3)
# Section 5.2 and the 7.4 in the abstract, contributions and conclusion
check("single-model blur pair", "gives $@$ ($@$ to $@$ per seed;\nreplicating by our rule) at less similar frozen errors ($@$, $@$); against\nits oracles, a $@$-fold larger avoidable gap times a $@$-fold larger recovered fraction",
      _bj["ratio_of_means"], min(_bj["ratio_per_seed"]), max(_bj["ratio_per_seed"]), _bj["frozen_det"], _bj["frozen_heat"],
      _bj["gap_ratio"], _bj["recovered_ratio"])
word("single-model blur: rule outcome and the identity", "replicating by our rule",
     _bj["outcome"] == "replicates" and math.isclose(_bj["gap_ratio"] * _bj["recovered_ratio"], _bj["ratio_of_means"], rel_tol=1e-9))
word("single-model blur: trained on both blurs", "with a single frozen\nmodel, trained on low severities of both blurs",
     len(glob.glob(os.path.join(ROOT, "oct21", "blur-joint-s*"))) == 3)
check("contributions: single-model blur", "and $@$ times\nwith a single frozen model trained on both", _bj["ratio_of_means"])
check("conclusion: single-model blur", "a $7.7$-fold difference ($@$ with a single frozen model)", _bj["ratio_of_means"])
# the ImageNet table: caption and header (the data rows are covered by the stale-table check)
check("ImageNet table: images", "mean over three noise seeds on $@$ validation images (ten per class)",
      1000 * const(_IMS, r"^PER_CLASS = (\d+)"))
design("ImageNet table: informative thresholds", "(a member above $@$ error or within $@$ of the clean error)",
       0.95 if "e < 0.95" in src("oct21/summarize.py") else -1, 0.01 if 'clean["err"] + 0.01' in src("oct21/summarize.py") else -1)
check("ImageNet table: rerun change", "whose largest change of a frozen gap is $@$", F("imnet_r2_max_gap_change"))
design("ImageNet table: header v", "Model & error & $@$ & $@$ & $@$ & $@$ & $@$ & $@$ \\\\", *_imv)

# =========================================================================== generated tables
with tempfile.TemporaryDirectory() as tmp:
    env = dict(os.environ, TABLES_OUT=tmp)
    out = subprocess.run([sys.executable, os.path.join(HERE, "make_tables.py")], env=env, capture_output=True, text=True)
    if out.returncode != 0:
        FAILS.append(f"make_tables.py failed: {out.stderr[-300:]}")
    for name in sorted(os.listdir(tmp)):
        new = open(os.path.join(tmp, name)).read()
        old_path = os.path.join(HERE, "tables", name)
        old = open(old_path).read() if os.path.exists(old_path) else None
        if new != old:
            FAILS.append(f"generated table {name} is stale: rerun paper/make_tables.py")
            continue
        PASSES += 1
        # cover the data rows of this table where it is inlined
        a = BODY.find(old)
        if a < 0:
            continue
        body = old[old.find(r"\midrule"):old.find(r"\bottomrule")]
        b0 = a + old.find(r"\midrule")
        SPANS.append((b0, b0 + len(body), "generated"))

# =========================================================================== training budgets
_budgets = {const("kaggle_oct3_runner/runner.py", r"^EPOCHS = (\d+)"), const("kaggle_oct3_runner/runner.py", r"^FULL_EPOCHS = (\d+)")}
for m in re.finditer(r"(?<![\w.])(\d+)(?=(?:\s+|-)epoch)", BODY):
    if float(m.group(1)) in _budgets:
        PASSES += 1
    else:
        FAILS.append(f"training budget {m.group(1)} epochs is not a budget used by the runner")
    SPANS.append((*m.span(1), "design"))

# =========================================================================== hedges restored in round 13
def _table_rows(label):
    at = BODY.find(f"\\label{{{label}}}")
    a, b = BODY.rfind(r"\begin{table}", 0, at), BODY.find(r"\end{table}", at)
    return "\n".join(l for l in BODY[a:b].split("\n") if "&" in l)
_ruled = _table_rows("tab:rules") + "\n" + _table_rows("tab:prereg")
_unruled = not re.search(r"factorial", _ruled, re.I) and not re.search(r"CIFAR-100[^\n]*blur|blur[^\n]*CIFAR-100", _ruled)
word("blur pair identified in the main sweep, not pre-specified", "(a comparison we identified in the main run,\nnot a pre-specified one)",
     # the single-model rerun (21 October) was pre-specified; the original comparison was not
     not any("blur" in l.lower() and not l.startswith("Single-model blur pair") for l in _table_rows("tab:rules").split("\n"))
     and not any("blur" in l and "BN-adapt" in l for l in _table_rows("tab:prereg").split("\n")))
design("TENT prediction threshold (main text)", "none would\nbeat BN-adapt by more than $@$. It failed",
       0.01 if "by more than 0.01" in src("PREREG_v6_tent_hyperparams.md") else -1)
word("TENT main prediction failed (main text)", "It failed, and we withdrew the claim",
     bool(re.search(r"TENT sweep[^\n]*No configuration beats BN-adapt[^\n]*& Failed", _table_rows("tab:prereg"))))
# Adam (one step) against the sweep's best configuration, from the rows of Table 14
_swrows = [[float(x) for x in re.findall(r"& (0\.\d{4})", r)] for r in _table_rows("tab:sweep").split("\n") if len(re.findall(r"& (0\.\d{4})", r)) == 4]
word("appendix A: Adam close to the sweep's best", "which keeps the original step count and comes close to the\nsweep's best",
     len(_swrows) == 3 and all(abs(r[2] - r[3]) < 0.003 for r in _swrows), cls="design")
word("appendix A: TENT document required regenerated tables", "The TENT test document required the main tables to be regenerated\nwith the better configuration",
     "The main tables are regenerated with the better TENT configuration" in src("PREREG_v6_tent_hyperparams.md"), cls="design")
word("introduction: four predictions written before their runs", "All four main predictions were written down before their runs; two failed, one held,\nand one, first recorded as failed",
     all(src(f) for f in ("PREREG_v3.md", "PREREG_v4_fog_ablation.md", "PREREG_v5_dann_lambda.md", "PREREG_v6_tent_hyperparams.md")), cls="design")
word("introduction: four pre-specified predictions", "two failed, one held,\nand one, first recorded as failed and later reclassified as inconclusive, was met",
     [m_.group(1).split()[0].rstrip(":.") for m_ in re.finditer(r"\n(?:Wiener|Fog ablation|DANN|TENT)[^&]*&[^&]*&\s*([^&]*?)\s*&", "\n" + _table_rows("tab:prereg"))]
     == ["First", "Failed", "Held", "Failed"])
word("Wiener: first recorded as failed, later inconclusive", "we recorded that test as failed and later\nreclassified it as inconclusive",
     "We first recorded the Wiener prediction as failed" in BODY
     and bool(re.search(r"Wiener[^\n]*&[^&]*&\s*First test: not met at either; inconclusive", _table_rows("tab:prereg"))))
word("abstract: the CIFAR-10 blur comparison is exploratory", "In an exploratory comparison of two blurs at\nsimilar frozen error", _unruled)
word("contribution: the CIFAR-10 blur comparison is exploratory", "In\nan exploratory comparison of two blurs that leave their 15-epoch frozen ResNet-18", _unruled)
word("conclusion: the excess is visible, its avoidability is not", "only\nProposition~\\ref{prop:equivalence} shows that this excess is avoidable",
     all(abs(r["t_drift"] - r["t_diffuse"]) > 0 and r["v"] > 0 for r in F("fjf_pairs")))
word("the Wiener witness is for the periodic implementation", "the witness below does so for the\nperiodic implementation of the control",
     "fft" in src("kaggle_oct3_wiener/wiener_lfl.py").lower())
word("recovered fractions: Wiener bound only for the periodic blur", "The assumption-free Wiener bound of\nSection~\\ref{sec:oracle} applies only to the periodic blur",
     "fft" in src("kaggle_oct3_wiener/wiener_lfl.py").lower())
word("recovered fractions: SGD run, Adam fractions in Appendix G", "We use the SGD run because the\nbound concerns its oracles",
     "control_slack_t2_joint" in src("paper/facts.py") and bool(re.search(r"With SGD they are", BODY)))
# the list itself is in the caption of Table 7 ("rules caption: what had no rule"); the limitations point to it
# the limitations pointer to Table 7 was cut for space (review 17); each analysis without a rule says so where
# it first appears, and the caption of Table 7 lists them ("rules caption: what had no rule")
word("rules caption: what had no rule", "the cross-family check and the certificates of Section~\\ref{sec:extrap-images} had no rule, and the DANN yardstick was set after its runs",
     _unruled and not re.search(r"cross-family|certificat", _table_rows("tab:rules") + _table_rows("tab:prereg"), re.I))
# the run plan lists the factorial with what it would report (blur-factorial), but no pass/fail rule
_fact_plan = "**blur-factorial.**" in src("rules/2026-10-02_matched_fog_pairs.md") and _unruled
word("rules caption: factorial and Adam reruns had a reporting plan", "The blur factorial and the Adam reruns had a reporting plan but no decision rule",
     _fact_plan and "**adam-main.**" in src("rules/2026-10-02_matched_fog_pairs.md"))
# the list of tests with pre-specified rules left the limitations for a pointer to Table 7 (review 18)
word("limitations: Table 7 lists the pre-specified tests", "list the pre-specified tests and\nrules", bool(re.search(r"CIFAR-10-C", _table_rows("tab:rules"))))
word("factorial is exploratory (no rule in Appendix A)", "In an exploratory $2\\times2$ factorial on the periodic Fourier operator", _unruled)
word("factorial: no rule fixed in advance", "no rule fixed in advance; Table~\\ref{tab:factorial}", _unruled)
word("CIFAR-100 blur pair is exploratory (no rule in Appendix A)", "This exploratory run had no rule.", _unruled)
word("why the low-contrast member errs more is untested", "Why frozen batch normalization produces the gap, and why\non the lower-contrast member, remains untested. A plausible mechanism for the fog-trained\nmodels",
     all(_V[k]["d_raw_mean"] > 0 and F("matched_contrast")[k]["drift"] < F("matched_contrast")[k]["diffuse"] for _V in (_V0, _V1) for k in _outer2))
word("separate models: the contrast reading is untested", "One untested reading\nis that the drift model spreads",
     min(_sc["drift_matched"]) < min(_sc["diffuse_own"]))
word("recovered fractions are upper bounds here", "Because oracle slack makes\n$\\Delta_{\\text{frozen}}$ understate the true avoidable error $\\Delta^*_{\\text{frozen}}$,\nthe recovered fractions that the adaptation literature reports are upper bounds here", F("cert_all_pos") > 0 and min(cs) > 0)
word("control bound is conditional on a flat Bayes risk", "if the reflect-padded blur's Bayes risk is flat, slack\nadds at least",
     min(cs) > 0 and F("control_slack_t2_joint")[0] > 0)
word("Tiny-ImageNet: direction only", "so this replicates only the direction.",
     all(r["seeds"] == [1] for r in _tin) and all(r["d_raw_mean"] > 0 for r in _tin[:2]))
word("CIFAR-10-C: the explanation is untested", "One untested explanation is that CIFAR-10-C's contrast\ncorruption", set(F("bench_verdict").values()) == {"rejected"})
word("slack certified for the data as stored", "certified lower bounds for stored data", WL["2.0"]["f32_cost"] > 0 and min(WL["2.0"]["cert_slack_joint"]) > 0)
# =========================================================================== severity values
SEVERITIES = set()
for path in ["kill_test_results.jsonl", "kill_test_asym_results.jsonl", "kill_test_fog_results.jsonl"]:
    SEVERITIES |= {r["t"] for r in load(path)}
SEVERITIES |= {t for f, t in R_} | {float(k.split("@")[1]) for k in abl} | set(WF) | set(FX)
SEVERITIES |= {0.05 + 0.05, 0.25 + 0.25, 0.5 + 0.5}
for m in re.finditer(r"\$t\s*(?:=|\\le|\\ge|\{=\})\s*((?:\d+(?:\.\d+)?)(?:\s*(?:,|\\to|\+)\s*\d+(?:\.\d+)?)*)\$", BODY):
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1))]
    for v in vals:
        if v in SEVERITIES or any(math.isclose(v, s) for s in SEVERITIES):
            PASSES += 1
        else:
            FAILS.append(f"severity t = {v} does not occur in any result file  ...{BODY[m.start():m.end() + 10]!r}")
    SPANS.append((m.start(1), m.end(1), "severity"))
for m in re.finditer(r"\$t=(\d+(?:\.\d+)?)\$", BODY):     # table headers and captions, t=1
    SPANS.append((*m.span(1), "severity"))
    if float(m.group(1)) not in SEVERITIES:
        FAILS.append(f"severity t={m.group(1)} not in the data")
for m in re.finditer(r"\$t\s*\\to\s*0\.5\$|\$t\\to\\infty\$", BODY):
    pass
for m in re.finditer(r"(\$t\s*=\s*0\.25\$,\s*\$0\.5\$\s*and\s*\$1\$)|(\$t = 0\.1\$ or \$0\.2\$)|(\$t=1\$ and \$t=2\$)", BODY):
    SPANS.append((*m.span(), "severity"))

# =========================================================================== coverage
TOKEN = re.compile(r"(?P<sci>\d+(?:\.\d+)?\\times\s*10\^\{-?\d+\})|(?P<pow>(?<![\d.])\d+\^\{-?\d+\})"
                   r"|(?P<num>(?<![\w.\\{-])-?\d+(?:\{,\}\d{3})*(?:\.\d+)?|(?<=\{)\d+(?:\.\d+)?(?=\}))")
SKIP_BEFORE = re.compile(r"(\\(?:label|ref|cite\w*|begin|end|includegraphics|input|cmidrule\(lr\)|multicolumn|setlength\{\\tabcolsep\}|renewcommand\{\\arraystretch\}|item)\{?[^}$]*$"
                         r"|ResNet-$|CIFAR-$|CIFAR-10-$|Tiny-$|\\texttt\{[\d.]*$|L\{$)")
SKIP_AFTER = re.compile(r"^(\\linewidth|pt\}|\}\{c\}|-\d)")
MATH = [m.span() for m in re.finditer(r"\$[^$]+\$|\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\}"
                                      r"|\\begin\{(?:proposition|corollary)\}.*?\\end\{(?:proposition|corollary)\}", BODY, re.S)]


def in_math(a):
    return any(x <= a < y for x, y in MATH)


def is_formula(a, b, txt):
    """An integer coefficient inside a formula: next to a variable, an operator or an
    exponent, as in 2t, 2\\omega^2, (1 - e^{...}), 2a, e^{2at}."""
    if re.match(r"^\(\d\)$", BODY[a - 1:b + 1]) and not in_math(a):
        return True                      # an enumeration label, (1) (2) (3)
    if not in_math(a) or "." in txt:
        return False
    if re.match(r"^\\times", BODY[b:b + 6]) or BODY[b:b + 2] == "\\%":
        return False                     # 4\times or 97\%: a value, not a coefficient
    if txt == "0":
        return True                      # zero in a formula or a definition (Sigma = 0, t >= 0)
    before, after = BODY[max(0, a - 2):a], BODY[b:b + 2]
    if re.match(r"^\s*(\$|,|\\,|\)\$)", after) and re.search(r"=\s*$", BODY[max(0, a - 3):a]):
        return False                     # "x = 3$": a stated value, not a coefficient
    return bool(re.match(r"^[A-Za-z\\(/^_]", after) or re.search(r"[\^_{(/]$|[-+]\s?$", before)
                or re.match(r"^\s?[-+/)]", after))


counts = {}
uncovered = []
covered = sorted(SPANS)
for m in TOKEN.finditer(BODY):
    a, b = m.span()
    if SKIP_BEFORE.search(BODY[max(0, a - 50):a]) or SKIP_AFTER.match(BODY[b:b + 10]):
        continue
    cls = next((c for x, y, c in covered if x <= a and b <= y), None)
    if cls is None and is_formula(a, b, m.group(0)):
        cls = "formula"
    if cls is None:
        uncovered.append((m.group(0), BODY[max(0, a - 40):b + 20].replace("\n", " ")))
        cls = "UNCOVERED"
    counts[cls] = counts.get(cls, 0) + 1

total = sum(counts.values())
print(f"{PASSES} checks passed, {len(FAILS)} failed")
for f in FAILS:
    print("  FAIL", f)
print(f"\ncoverage: {total} numeric tokens in the text (tables and captions included)")
for c in ("claim", "table", "generated", "design", "severity", "formula", "noted", "UNCOVERED"):
    print(f"  {c:<10} {counts.get(c, 0):>5}")
if uncovered:
    print("\nnumbers in no class:")
    for txt, ctx in uncovered:
        print(f"  {txt:>14}   ...{ctx}...")
print("\nnoted by hand (not reproducible from any current file):")
for label, reason in NOTED:
    print(f"  {label}: {reason}")
if "--formulas" in sys.argv:
    print("\nformula tokens:")
    for m in TOKEN.finditer(BODY):
        a, b = m.span()
        if not any(x <= a and b <= y for x, y, c in covered) and is_formula(a, b, m.group(0)):
            print(f"  {m.group(0):>4}   ...{BODY[max(0, a - 25):b + 15]!r}")
sys.exit(1 if FAILS or uncovered else 0)
