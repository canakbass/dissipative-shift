# Separating Avoidable and Unavoidable Error Under Dissipative Distribution Shift

Code, result files and checking scripts for the paper of the same title
(Can Akbaş, Manisa Celal Bayar University).

## What is here

| Path | Contents |
| --- | --- |
| `kaggle_*/` | One training and evaluation script per experiment, run as a Kaggle GPU notebook, and the result files it wrote (`output*/`). |
| `kaggle_oct3_runner/runner.py` | The later runs: matched fog pairs, the joint model at 15 and 200 epochs, the Adam reruns, the blur factorial, CIFAR-100 and Tiny-ImageNet. Their results are in `oct3/*/output/`. |
| `kaggle_oct3_wiener/wiener_lfl.py` | The like-for-like Wiener witness. |
| `kill_test*.py`, `test_heat_blur.py` | The synthetic checks with closed-form Bayes risk, and the composition check of the heat-equation operator (CPU, minutes). |
| `certified_extrapolation*.py` | Certified extrapolation of the equivalent noise, synthetic and on CIFAR-10 (CPU). |
| `float32_saturation_check.py` | The float32 accumulation check described in the corrections log. |
| `paper/` | LaTeX source of the paper, the scripts that build its tables and figures, `facts.py` and `verify_claims.py`. |
| `PREREG_*.md` | Pre-registration documents, kept as written; each outcome was added after its run. |
| `SEMIGROUP_SCOPE.md` | Notes on which corruptions form a semigroup, and why. |

## Checking the numbers in the paper

```
pip install -r requirements.txt
python paper/facts.py           # raw result files -> paper/facts.json
python paper/verify_claims.py   # checks every number in paper/main.tex; exit code 1 on any mismatch
```

`verify_claims.py` classifies every numeric token in the paper, tables and captions
included, as a value recomputed from the result files, a cell of a regenerated
table, a setting read from the script that ran the experiment, a severity, a formula
coefficient, or a historical value listed by hand with its reason; it fails if any
number falls in none of these classes. `make_tables.py` and `make_figures.py`
rebuild the tables and figures from the same files.

## Rerunning the experiments

The GPU scripts were run on Kaggle notebooks with two T4 GPUs. They take their seed
from the `RUN_SEED` environment variable, except the two Wiener scripts, which set
`SEED` at the top; `kaggle_oct3_runner/runner.py` runs the job named by the `JOB`
line at its top (one of the keys of `JOBS`). Each script downloads CIFAR-10,
CIFAR-100 or Tiny-ImageNet itself; paths under `/kaggle/working` can be changed at
the top of each script. `certified_extrapolation_cifar.py` expects CIFAR-10 in
`data/cifar-10-batches-py`.

## License

MIT; see `LICENSE`.

## Paper source

The LaTeX source of the paper, with its generated tables and figures, will be added
here when the paper is accepted. Until then `paper/verify_claims.py`, which reads
`paper/main.tex`, cannot be run from this repository; `paper/facts.py`,
`paper/make_tables.py` and `paper/make_figures.py` can.
