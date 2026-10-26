"""Matched fog pairs on pretrained ImageNet models (oct21/PLAN.md, experiment 1).

Inference only. For one torchvision model (MODEL below; build_kernels.py sets it per
kernel), evaluates 10,000 ImageNet validation images (10 per class, fixed seed 0)
under the paper's two Ornstein-Uhlenbeck fog regimes at equivalent-variance-matched
severities. The two members of a pair have the same Bayes risk (Proposition 1), so any
error difference is avoidable. Each corrupted batch is also mapped back by
L_t^{-1}(x - c_t); the two inverted members are identically distributed, and their
error difference is the sampling floor of the decision rule. Three noise seeds.
"""
import copy
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.transforms import v2 as T

MODEL = "efficientnet_b0"
SMOKE = os.environ.get("SMOKE") == "1"  # local CPU smoke test: random weights, fake images

MODELS = {  # name: (torchvision architecture, weights, normalization kind)
    "resnet50_v1": ("resnet50", "IMAGENET1K_V1", "bn"),
    "resnet50_v2": ("resnet50", "IMAGENET1K_V2", "bn"),
    "efficientnet_b0": ("efficientnet_b0", "IMAGENET1K_V1", "bn"),
    "vit_b_16": ("vit_b_16", "IMAGENET1K_V1", "ln"),
    "convnext_tiny": ("convnext_tiny", "IMAGENET1K_V1", "ln"),
    "swin_t": ("swin_t", "IMAGENET1K_V1", "ln"),
}
FOG = {"drift": dict(a=1.0, sigma=0.05), "diffuse": dict(a=0.05, sigma=0.2)}
A = 1.0                                     # white haze point, as in the paper
V_GRID = [0.0025, 0.005, 0.01, 0.021, 0.042, 0.089]
NOISE_SEEDS = [1, 2, 3]
PER_CLASS = 10
BATCH = 200                                 # also the BN-adapt batch, as in the paper
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(__file__), "smoke_out")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if SMOKE:
    V_GRID, NOISE_SEEDS = V_GRID[:1], NOISE_SEEDS[:1]


def fog_v(fam, t):
    a, s = FOG[fam]["a"], FOG[fam]["sigma"]
    return s ** 2 / (2 * a) * (math.exp(2 * a * t) - 1)


def fog_t_for_v(fam, v):
    a, s = FOG[fam]["a"], FOG[fam]["sigma"]
    return math.log(1 + 2 * a * v / s ** 2) / (2 * a)


def find_root():
    """The folder holding the 1000 class folders 00000 .. 00999 of the attached dataset."""
    for d, subdirs, _ in os.walk("/kaggle/input"):
        names = [s for s in subdirs if s.isdigit()]
        if len(names) >= 1000:
            return d
    raise FileNotFoundError("no folder with 1000 class subfolders under /kaggle/input")


def load_images():
    if SMOKE:
        g = torch.Generator().manual_seed(0)
        return (torch.rand(8, 3, 224, 224, generator=g) * 255).to(torch.uint8), torch.arange(8)
    root = find_root()
    classes = sorted(s for s in os.listdir(root) if s.isdigit())
    assert len(classes) == 1000, len(classes)
    rng = np.random.default_rng(0)
    files, labels = [], []
    for c in classes:
        fs = sorted(glob.glob(os.path.join(root, c, "*")))
        pick = rng.choice(len(fs), PER_CLASS, replace=False)
        files += [fs[i] for i in sorted(pick)]
        labels += [int(c)] * PER_CLASS
    tf = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.PILToTensor()])

    def one(f):
        with Image.open(f) as im:
            return tf(im.convert("RGB"))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(one, files))
    print(f"  loaded {len(imgs)} images from {root} ({time.time()-t0:.1f}s)", flush=True)
    return torch.stack(imgs), torch.tensor(labels)


def build_model():
    arch, wname, kind = MODELS[MODEL]
    weights = None if SMOKE else torchvision.models.get_model_weights(arch)[wname]
    model = torchvision.models.get_model(arch, weights=weights).to(DEVICE).eval()
    return model, kind


def bn_adapt_copy(model):
    """BN-adapt: every BatchNorm normalizes with the current test batch's statistics;
    no gradient step, every other module stays in eval mode (dropout, stochastic depth)."""
    m2 = copy.deepcopy(model).eval()
    for m in m2.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.track_running_stats = False
            m.running_mean, m.running_var = None, None
    return m2


@torch.no_grad()
def predict(model, x01):
    x = (x01 - MEAN.to(x01.device)) / STD.to(x01.device)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
        return model(x).argmax(1)


@torch.no_grad()
def evaluate(models, imgs, labels, fam=None, t=0.0, seed=0):
    """Correctness per image for each model in `models` on the raw corrupted images and,
    for fog, on the same draw mapped back by L_t^{-1}(x - c_t)."""
    out = {k: [] for k in models}
    out_inv = {k: [] for k in models}
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed * 1_000_003 + round(t * 1e6) + (7 if fam == "drift" else 0)))
    if fam is not None:
        a, s = FOG[fam]["a"], FOG[fam]["sigma"]
        beta = math.exp(-a * t)
        std = math.sqrt(s ** 2 / (2 * a) * (1 - beta ** 2))
    for i in range(0, imgs.shape[0], BATCH):
        x = imgs[i:i + BATCH].to(DEVICE).float() / 255.0
        y = labels[i:i + BATCH].to(DEVICE)
        if fam is not None:
            noise = torch.randn(x.shape, generator=gen, device=DEVICE)
            x = A + (x - A) * beta + std * noise          # no clipping, as in the paper
            x_inv = A + (x - A) / beta
        for k, m in models.items():
            out[k].append((predict(m, x) == y).cpu())
            if fam is not None:
                out_inv[k].append((predict(m, x_inv) == y).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    res_inv = {k: torch.cat(v).numpy() for k, v in out_inv.items()} if fam is not None else {}
    return res, res_inv


def main():
    print(f"device: {DEVICE}, model: {MODEL}, smoke: {SMOKE}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    imgs, labels = load_images()
    model, kind = build_model()
    models = {"frozen": model}
    if kind == "bn":
        models["bnadapt"] = bn_adapt_copy(model)
    rows, bits = [], {}
    t0 = time.time()
    clean, _ = evaluate(models, imgs, labels)
    err_clean = {k: 1 - v.mean() for k, v in clean.items()}
    for k, v in clean.items():
        bits[f"clean_{k}"] = v
    rows.append(dict(job="imnet_fog", model=MODEL, norm=kind, family="clean", t=0.0, v=0.0, seed=0,
                     n=int(imgs.shape[0]), err=float(err_clean["frozen"]),
                     err_bnadapt=float(err_clean["bnadapt"]) if kind == "bn" else None))
    print(f"  clean error {err_clean['frozen']:.4f}" + (f", BN-adapt {err_clean['bnadapt']:.4f}" if kind == "bn" else ""), flush=True)
    # stop conditions (PLAN.md): the label mapping must be right before anything is written
    if not SMOKE:
        if MODEL == "resnet50_v1" and not 0.20 <= err_clean["frozen"] <= 0.28:
            sys.exit(f"ABORT: ResNet-50 V1 clean error {err_clean['frozen']:.4f} outside 0.20-0.28; label mapping wrong")
        if err_clean["frozen"] > 0.5:
            sys.exit(f"ABORT: clean error {err_clean['frozen']:.4f} > 0.5; label mapping or preprocessing wrong")
    for seed in NOISE_SEEDS:
        for v in V_GRID:
            for fam in ("drift", "diffuse"):
                t = fog_t_for_v(fam, v)
                res, res_inv = evaluate(models, imgs, labels, fam, t, seed)
                key = f"{fam}_v{v}_s{seed}"
                for k in res:
                    bits[f"{key}_{k}"] = res[k]
                    bits[f"{key}_{k}_inv"] = res_inv[k]
                r = dict(job="imnet_fog", model=MODEL, norm=kind, family=fam, t=t, v=v, seed=seed,
                         beta=math.exp(-FOG[fam]["a"] * t), n=int(imgs.shape[0]),
                         err=float(1 - res["frozen"].mean()), err_inv=float(1 - res_inv["frozen"].mean()),
                         err_bnadapt=float(1 - res["bnadapt"].mean()) if kind == "bn" else None,
                         err_bnadapt_inv=float(1 - res_inv["bnadapt"].mean()) if kind == "bn" else None)
                rows.append(r)
                print(f"  seed {seed} v={v:<7} {fam:<8} t={t:.4f} beta={r['beta']:.3f} err={r['err']:.4f} "
                      f"err_inv={r['err_inv']:.4f}" + (f" bn={r['err_bnadapt']:.4f}" if kind == "bn" else "")
                      + f"  ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, f"imnet_fog_{MODEL}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez_compressed(os.path.join(OUT, f"imnet_fog_{MODEL}_correct.npz"), labels=labels.numpy(), **bits)
    print(f"done ({time.time()-t0:.0f}s): {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
