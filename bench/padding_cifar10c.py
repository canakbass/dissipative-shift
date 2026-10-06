"""First-layer padding and test-time normalization on CIFAR-10-C, with standard
pretrained CIFAR-10 models (chenyaofo/pytorch-cifar-models). Inference only; uses a
GPU when one is available.

For each model, corruption and severity, the same images are evaluated with the first
convolution padded with zeros (as trained) and with reflect padding, frozen and with
BN-adapt (batch statistics, batch 200); TENT (Adam, lr 1e-3, one step per batch, reset
at each severity) for the first model on the hypothesis corruptions. Per-image
correctness is compared between the two paddings, so the difference in error comes with
a paired standard error. The decision rule was written before this ran.

    THREADS=8 python bench/padding_cifar10c.py
Needs data/CIFAR-10-C/{contrast,brightness,fog,gaussian_noise,labels}.npy and
data/cifar-10-batches-py. Writes bench/padding_cifar10c_results.jsonl.
"""
import copy
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("THREADS", "8")))
torch.manual_seed(0)
ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C10C = os.path.join(ROOT, "data", "CIFAR-10-C")
OUT = os.path.join(ROOT, "bench", "padding_cifar10c_results.jsonl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = os.environ.get("MODELS", "cifar10_resnet20,cifar10_resnet56,cifar10_vgg13_bn").split(",")
FALLBACK = ["cifar10_resnet32", "cifar10_resnet44", "cifar10_vgg11_bn"]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)    # the models' own normalization
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
CORRUPTIONS = ["contrast", "brightness", "fog", "gaussian_noise"]
TENT_CORRUPTIONS = ["contrast", "brightness", "fog"]
BATCH = 200
TENT_LR = 1e-3


def load_model(name):
    return torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True, trust_repo=True, verbose=False).to(DEVICE).eval()


def first_conv(model):
    return next(m for m in model.modules() if isinstance(m, nn.Conv2d))


def with_padding(model, kind):
    m = copy.deepcopy(model)
    if kind != "zeros":
        first_conv(m).padding_mode = kind
    return m


def to_input(x_uint8):
    x = torch.from_numpy(x_uint8).float().div_(255.0).permute(0, 3, 1, 2)
    return ((x - MEAN) / STD).to(DEVICE)


@torch.no_grad()
def correct_frozen(model, x, y):
    model.eval()
    out = [model(x[i:i + BATCH]).argmax(1) for i in range(0, len(x), BATCH)]
    return (torch.cat(out).cpu() == y).numpy()


def _batch_stat_model(model):
    m = copy.deepcopy(model)
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.track_running_stats = False
            mod.running_mean, mod.running_var = None, None
    return m


@torch.no_grad()
def correct_bnadapt(model, x, y):
    m = _batch_stat_model(model)
    out = [m(x[i:i + BATCH]).argmax(1) for i in range(0, len(x), BATCH)]
    return (torch.cat(out).cpu() == y).numpy()


def correct_tent(model, x, y):
    m = _batch_stat_model(model)
    params = []
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)
            params += [mod.weight, mod.bias]
    opt = torch.optim.Adam(params, lr=TENT_LR)
    preds = []
    for i in range(0, len(x), BATCH):
        logits = m(x[i:i + BATCH])
        preds.append(logits.argmax(1).detach())        # predict, then adapt on the batch
        p = logits.softmax(1)
        loss = -(p * logits.log_softmax(1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (torch.cat(preds).cpu() == y).numpy()


def paired(c0, c1):
    """Errors with zero (c0) and reflect (c1) padding and the paired standard error of
    their difference, from the images on which the two disagree."""
    n = len(c0)
    a = int(np.sum(c0 & ~c1))      # right with zero padding only
    b = int(np.sum(~c0 & c1))      # right with reflect padding only
    se = np.sqrt(max(a + b - (a - b) ** 2 / n, 0.0)) / n
    return dict(err_zeros=float(1 - c0.mean()), err_reflect=float(1 - c1.mean()),
                gain=float((b - a) / n), se=float(se), n=n, zeros_only=a, reflect_only=b)


def clean_test():
    d = pickle.load(open(os.path.join(ROOT, "data", "cifar-10-batches-py", "test_batch"), "rb"), encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.array(d[b"labels"], dtype=np.int64)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    labels = np.load(os.path.join(C10C, "labels.npy")).astype(np.int64)
    data = {c: np.load(os.path.join(C10C, f"{c}.npy"), mmap_mode="r") for c in CORRUPTIONS}
    xc, yc = clean_test()
    done = set()
    if os.path.exists(OUT):
        done = {(r["model"], r["corruption"], r["severity"], r["mode"]) for r in map(json.loads, open(OUT))}
    models, pool = [], list(MODELS) + FALLBACK
    while len(models) < len(MODELS) and pool:
        name = pool.pop(0)
        try:
            models.append((name, load_model(name)))
        except Exception as e:      # a model that cannot be loaded is replaced by the next one
            print(f"cannot load {name}: {e}", flush=True)
            if name in MODELS:
                pool = [f for f in pool if f not in MODELS] + [f for f in pool if f in MODELS]
    for k, (name, model) in enumerate(models):
        print(f"model {name}: first conv {first_conv(model)}", flush=True)
        variants = {kind: with_padding(model, kind) for kind in ("zeros", "reflect")}
        jobs = [("clean", 0, xc, yc)] + [(c, s, np.asarray(data[c][(s - 1) * 10000:s * 10000]),
                                          labels[(s - 1) * 10000:s * 10000])
                                         for c in CORRUPTIONS for s in range(1, 6)]
        for corr, sev, x_np, y_np in jobs:
            modes = ["frozen", "bnadapt"] + (["tent"] if k == 0 and corr in TENT_CORRUPTIONS else [])
            x, y = to_input(np.ascontiguousarray(x_np)), torch.from_numpy(y_np)
            for mode in modes:
                if (name, corr, sev, mode) in done:
                    continue
                t0 = time.time()
                fn = {"frozen": correct_frozen, "bnadapt": correct_bnadapt, "tent": correct_tent}[mode]
                if mode == "tent":
                    torch.manual_seed(0)
                c = {kind: fn(variants[kind], x, y) for kind in ("zeros", "reflect")}
                row = dict(model=name, corruption=corr, severity=sev, mode=mode, **paired(c["zeros"], c["reflect"]),
                           seconds=round(time.time() - t0, 1))
                with open(OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{name:<18} {corr:<15} s={sev} {mode:<8} zeros {row['err_zeros']:.4f} reflect "
                      f"{row['err_reflect']:.4f} gain {row['gain']:+.4f} (se {row['se']:.4f}) {row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
