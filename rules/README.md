# Decision rules fixed before the runs

The paper calls a prediction or decision rule *pre-specified* when it was written down
in an internal document before the run it concerns (Section 4.1 of the paper). Those
documents were written in Turkish and carry no external timestamp. This directory holds
English translations, made on 12 October 2026 after all runs, so that the rules can be
read next to the scripts that apply them. Each file keeps the original's structure:
the rule as written before the run, notes added before the run, and the outcome added
after it. Kaggle account names, quotas and launch instructions are left out; nothing
else is changed. Remarks added during translation are marked *[translation note]*.

| File | Written | Applied by | Paper |
| --- | --- | --- | --- |
| `2026-10-02_matched_fog_pairs.md` | 2 and 3 October | `oct3/summarize.py`, `paper/facts.py` | Section 5.4, Table 7 |
| `2026-10-05_first_layer_padding.md` | 5 October | `oct5/analyze_padding.py` | Appendix F, Table 7 |
| `2026-10-05_cifar10c_padding.md` | 5 October | `bench/analyze_bench.py` | Appendix F.1, Table 7 |
| `2026-10-05_retraining_and_normalization.md` | 5 October | `oct10/analyze_variants.py` | Section 5.4, Appendix F, Table 7 |
| `2026-10-21_imagenet_and_single_blur.md` | 21 October (translated 22 October) | `oct21/summarize.py`, `paper/facts.py` | Sections 5.2 and 5.7, Appendix B, Table 7 |

The earlier pre-specification documents (`PREREG_*.md` at the repository root) were
written in English and are kept as written.
