"""
Stage 3 -- CIFAR-10-C (Kaggle GPU kernel): repeat the Stage 1/2 protocol with a
real ResNet-18 plus TENT and EATA. Part of a multi-seed run: a single-seed
version showed small but ambiguous negative Deltas inside the training horizon,
and three seeds separate GPU/SGD noise from a real effect. The seed is passed
through the RUN_SEED environment variable and each run is tagged seed_N.

Four arms are compared, all starting from the same frozen (source) model:
  - Err_oracle(t)  : a ResNet-18 trained separately AT that severity and tested at
                     the same severity (the empirical ceiling standing in for R*(t))
  - Err_frozen(t)  : trained at low severities and then frozen, never adapted
  - Err_tent(t)    : TENT (Wang et al. 2021) -- BatchNorm affine parameters only,
                     unlabeled entropy minimization, restarted from scratch at each
                     severity (episodic)
  - Err_eata(t)    : EATA (Niu et al. 2022) -- TENT plus (a) filtering out
                     high-entropy samples (E0 = 0.4*ln(K)), and (b) a
                     Fisher-weighted regularizer penalizing movement away from the
                     source BN parameters.
                     Simplification: the original paper's redundancy/diversity
                     filter (cosine similarity of a new prediction against recently
                     seen ones) is omitted. The two main components are kept, the
                     third is not.

  Delta_X(t) = Err_X(t) - Err_oracle(t)   -> the waste of method X

The frozen/source model's total training budget is held equal to the oracle's.
Without that, the frozen model beats the oracle simply by seeing more data, and
Delta(t) comes out spuriously negative inside the training horizon.
The semigroup scope is fixed in SEMIGROUP_SCOPE.md.
"""
import copy
import json
import math
import os
import time

# A full run died with a CUDA OOM after ~5000 batches: it could not allocate
# 2.44 GiB while 13.35 GiB was already in use. Not a leak but fragmentation, which
# PyTorch's own guidance covers. The source was augment_batch slicing and stacking
# 128 small tensors per batch. This must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.datasets import CIFAR10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/kaggle/working/cifar10"
N_TRAIN_IMG = 50000  # the whole official CIFAR-10 training split
N_TEST_IMG = 10000   # the whole official CIFAR-10 test split (disjoint, no leakage)
EPOCHS = 15  # calibration: ~45 s/epoch on this GPU. 24 models x 40 epochs would be
             # ~12 h, past the session and quota limits; 15 epochs brings it to ~4.5 h
BATCH_SIZE = 128
LR = 0.1             # SGD + momentum + cosine: a stronger CIFAR recipe than Adam at 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
TENT_LR = 1e-3
TENT_BATCH = 200
SEED = int(os.environ.get("RUN_SEED", "2"))
RUN_TAG = f"blurheat_seed{SEED}"
N_CLASSES = 10
EATA_E0 = 0.4 * math.log(N_CLASSES)  # Niu et al. 2022, the standard threshold
EATA_FISHER_ALPHA = 2000.0           # the CIFAR-10 default in EATA's reference code
EATA_FISHER_N = 2000                 # labeled source images used for the Fisher estimate
NORM_MEAN, NORM_STD = None, None  # (3,) per channel, fixed from the t=0 pool

# blur_heat only: gauss_noise and fog_beer_lambert are already corrected and
# correct (see cifar10_c_kaggle.py), so there is no need to rerun them here.
FAMILIES = {
    "blur_heat": dict(
        train_ts=[0.0, 0.05, 0.1],
        test_ts=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    ),
}


def _to_nchw01(imgs_uint8):
    return np.transpose(imgs_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2))


def load_cifar():
    """Uses the official train/test split; the two are disjoint, so no leakage."""
    train_ds = CIFAR10(root=DATA_ROOT, train=True, download=True)
    test_ds = CIFAR10(root=DATA_ROOT, train=False, download=True)
    x_train = _to_nchw01(train_ds.data)
    y_train = np.array(train_ds.targets, dtype=np.int64)
    x_test = _to_nchw01(test_ds.data)
    y_test = np.array(test_ds.targets, dtype=np.int64)
    return x_train, y_train, x_test, y_test


FOG_SIGMA = 0.2  # OU diffusion coefficient; stationary std = FOG_SIGMA/sqrt(2)

# blur_heat: the damped stochastic heat equation,
# dX = -(omega^2 + A_MIN)*X dt + sigma dW. Each Fourier mode is a separate OU
# process, which makes this an exact Markov semigroup (Chapman-Kolmogorov, the
# standard property of a time-homogeneous diffusion). Adding A_MIN > 0 keeps the DC
# mode (omega=0) from diffusing without bound, so it converges to a stationary
# distribution as the fog model does.
#
# How this differs from deterministic heat-kernel blur (the gauss_blur family): not
# only high frequencies but all frequencies, DC included, eventually decay to zero,
# and noise is added, so information is genuinely destroyed. The gauss_blur family
# is retained not as dissipative but as a reversible control, used to measure how
# far the oracle proxy sits from the Bayes risk.
BLUR_HEAT_SIGMA = 15.0
BLUR_HEAT_A_MIN = 0.1


def _blur_heat_rate(h, w):
    fy = np.fft.fftfreq(h) * 2 * np.pi
    fx = np.fft.fftfreq(w) * 2 * np.pi
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    return FY**2 + FX**2 + BLUR_HEAT_A_MIN


def corrupt(imgs, t, family, rng):
    """An earlier version clipped noise to
    [0,1] and modelled fog as a deterministic affine map with no noise. Clipping
    breaks the semigroup property S(t+s)=S(t).S(s), since
    clip(clip(x+n1)+n2) != clip(x+n1+n2); and the deterministic affine fog map is a
    bijection, invertible for every finite t, so it loses no information and does
    not satisfy the premise of the Blackwell argument, which needs a genuine
    garbling. This version: (1) no clipping anywhere, (2) fog is an
    Ornstein-Uhlenbeck process -- exponential attenuation together with scattering
    noise -- which is an exact Markov semigroup and does destroy information,
    converging to a stationary distribution as t->inf
    (see kill_test_fog.py, validated against the closed form)."""
    if family == "gauss_noise":
        return imgs + rng.normal(0, np.sqrt(2 * t), imgs.shape).astype(np.float32) if t > 0 else imgs.copy()
    if family == "gauss_blur":
        sigma = np.sqrt(2 * t)
        if sigma == 0:
            return imgs.copy()
        # sigma=0 on the N and C axes of (N,C,H,W) -> spatial blur only, in one
        # vectorized call. A Python loop would be the bottleneck at 50k images.
        return gaussian_filter(imgs, sigma=(0, 0, sigma, sigma), mode="reflect")
    if family == "fog_beer_lambert":
        A = 1.0
        decay = np.exp(-t)
        mean = imgs * decay + A * (1 - decay)
        var = (FOG_SIGMA**2 / 2) * (1 - decay**2)
        if var <= 0:
            return mean
        return mean + rng.normal(0, np.sqrt(var), imgs.shape).astype(np.float32)
    if family == "blur_heat":
        n, c, h, w = imgs.shape
        rate = _blur_heat_rate(h, w)
        decay = np.exp(-rate * t)
        X = np.fft.fft2(imgs, axes=(-2, -1))
        mean = np.real(np.fft.ifft2(X * decay[None, None], axes=(-2, -1))).astype(np.float32)
        if t <= 0:
            return mean
        var = (BLUR_HEAT_SIGMA**2 / (2 * rate)) * (1 - decay**2)
        filt = np.sqrt(var / (h * w))
        white = rng.standard_normal((n, c, h, w)).astype(np.float32)
        white_fft = np.fft.fft2(white, axes=(-2, -1))
        colored = np.real(np.fft.ifft2(white_fft * filt[None, None], axes=(-2, -1))).astype(np.float32)
        return mean + colored
    raise ValueError(family)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """Standard CIFAR ResNet-18: 3x3 stride-1 stem, no max-pool (unlike the ImageNet version)."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, n_classes)

    def _make_layer(self, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def augment_batch(x):
    """Standard CIFAR recipe: 4px reflect pad + random 32x32 crop + horizontal flip.
    Training only; not applied to val or test.

    A per-sample Python loop (128 slices plus a stack, every batch) caused CUDA
    memory fragmentation and an OOM after ~5000 batches: 2.44 GiB could not be
    allocated while 13.35 GiB was in use. Drawing one shared crop and flip per
    batch removes the fragmentation source and is faster -- pad, slice, optional
    flip: three operations instead of 128 small tensors. The cost is a little
    augmentation diversity within a batch."""
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    top = int(torch.randint(0, 9, (1,)))
    left = int(torch.randint(0, 9, (1,)))
    out = xp[:, :, top:top + 32, left:left + 32]
    if torch.rand(1).item() < 0.5:
        out = out.flip(-1)
    return out


def to_tensor(x, y):
    mean = NORM_MEAN.reshape(1, 3, 1, 1)
    std = NORM_STD.reshape(1, 3, 1, 1)
    x = (x - mean) / std
    return torch.from_numpy(x).float().to(DEVICE), torch.from_numpy(y).long().to(DEVICE)


def train_model(x, y, epochs=EPOCHS, seed=SEED, val_frac=0.15):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = ResNet18().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                           weight_decay=WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    x_t, y_t = to_tensor(x, y)
    x_tr, y_tr = x_t[train_idx], y_t[train_idx]
    x_val, y_val = x_t[val_idx], y_t[val_idx]
    n_tr = x_tr.shape[0]

    best_val_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        tperm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH_SIZE):
            b = tperm[i:i + BATCH_SIZE]
            xb = augment_batch(x_tr[b])
            opt.zero_grad()
            loss = loss_fn(model(xb), y_tr[b])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val).argmax(1) == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    del x_t, y_t, x_tr, y_tr, x_val, y_val, opt, sched, best_state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def eval_error(model, x, y):
    model.eval()
    x_t, y_t = to_tensor(x, y)
    pred = model(x_t).argmax(dim=1)
    return (pred != y_t).float().mean().item()


def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * logits.log_softmax(1)).sum(1)


def configure_tent(source_model):
    """TENT (Wang et al. 2021): only the BatchNorm affine parameters (weight, bias)
    are updated, by unlabeled entropy minimization. Each batch's own statistics are
    used instead of the BN running statistics (track_running_stats is turned off)."""
    model = copy.deepcopy(source_model).to(DEVICE)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean, m.running_var = None, None
            for p in m.parameters():
                p.requires_grad_(True)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    params = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()]
    opt = torch.optim.SGD(params, lr=TENT_LR, momentum=0.9)
    return model, opt


def tent_eval_error(source_model, x, y, batch_size=TENT_BATCH):
    """Episodic: restarts from the source model at each severity, so no state leaks across severities."""
    model, opt = configure_tent(source_model)
    x_t, y_t = to_tensor(x, y)
    n = x_t.shape[0]
    n_correct = 0
    for i in range(0, n, batch_size):
        xb, yb = x_t[i:i + batch_size], y_t[i:i + batch_size]
        out = model(xb)
        loss = softmax_entropy(out).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        n_correct += (out.argmax(1) == yb).sum().item()
    err = 1 - n_correct / n
    del model, opt, x_t, y_t
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return err


def compute_fisher(source_model, x_fisher, y_fisher, batch_size=200):
    """For EATA's Fisher-weighted regularizer: the mean squared gradient of the BN
    affine parameters on labeled, uncorrupted source data. A large Fisher value means
    the parameter matters for the source task, so moving away from it at test time
    should cost more. One pass, following EATA's reference recipe."""
    model = copy.deepcopy(source_model).to(DEVICE)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad_(True)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    params = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()]
    source_vals = [p.detach().clone() for p in params]
    fisher = [torch.zeros_like(p) for p in params]
    loss_fn = nn.CrossEntropyLoss()
    x_t, y_t = to_tensor(x_fisher, y_fisher)
    n = x_t.shape[0]
    for i in range(0, n, batch_size):
        xb, yb = x_t[i:i + batch_size], y_t[i:i + batch_size]
        model.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        for f, p in zip(fisher, params):
            f += (p.grad.detach() ** 2) * xb.shape[0]
    fisher = [f / n for f in fisher]
    del model, x_t, y_t
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return fisher, source_vals


def configure_eata(source_model):
    model = copy.deepcopy(source_model).to(DEVICE)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean, m.running_var = None, None
            for p in m.parameters():
                p.requires_grad_(True)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    params = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()]
    opt = torch.optim.SGD(params, lr=TENT_LR, momentum=0.9)
    return model, opt, params


def eata_eval_error(source_model, fisher, source_vals, x, y, batch_size=TENT_BATCH):
    """Episodic, same protocol as TENT: samples above the entropy threshold E0 are
    excluded from the adaptation loss (they still count toward predictions and
    accuracy, just not toward the gradient); the rest are optimized on entropy plus
    the Fisher-weighted anchor term."""
    model, opt, params = configure_eata(source_model)
    x_t, y_t = to_tensor(x, y)
    n = x_t.shape[0]
    n_correct = 0
    for i in range(0, n, batch_size):
        xb, yb = x_t[i:i + batch_size], y_t[i:i + batch_size]
        out = model(xb)
        ent = softmax_entropy(out)
        mask = ent < EATA_E0
        if mask.any():
            adapt_loss = ent[mask].mean()
            anchor_loss = sum(
                (f * (p - sv) ** 2).sum() for f, p, sv in zip(fisher, params, source_vals)
            )
            total_loss = adapt_loss + EATA_FISHER_ALPHA * anchor_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        n_correct += (out.argmax(1) == yb).sum().item()
    err = 1 - n_correct / n
    del model, opt, x_t, y_t
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return err


def run_family(name, cfg, x_train_raw, y_train, x_eval_raw, y_eval, rng):
    print(f"\n########## Family: {name} ##########", flush=True)
    train_ts, test_ts = cfg["train_ts"], cfg["test_ts"]

    t0 = time.time()
    n_per_ts = N_TRAIN_IMG // len(train_ts)
    chunks, chunk_labels = [], []
    for i, t in enumerate(train_ts):
        sl = slice(i * n_per_ts, (i + 1) * n_per_ts)
        chunks.append(corrupt(x_train_raw[sl], t, name, rng))
        chunk_labels.append(y_train[sl])
    x_frozen_train = np.concatenate(chunks)
    y_frozen_train = np.concatenate(chunk_labels)
    frozen_model = train_model(x_frozen_train, y_frozen_train)
    print(f"  source/frozen model trained ({time.time()-t0:.1f}s, train_ts={train_ts})", flush=True)

    t_fisher = time.time()
    fisher, source_vals = compute_fisher(
        frozen_model, x_train_raw[:EATA_FISHER_N], y_train[:EATA_FISHER_N]
    )
    print(f"  EATA Fisher estimate done ({time.time()-t_fisher:.1f}s, n={EATA_FISHER_N})", flush=True)

    rows = []
    header = (f"{'t':>6} | {'Err_oracle':>10} | {'Err_frozen':>10} | {'Err_tent':>9} | {'Err_eata':>9} | "
              f"{'D_frozen':>9} | {'D_tent':>9} | {'D_eata':>9}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for t in test_ts:
        x_eval_t = corrupt(x_eval_raw, t, name, rng)

        x_oracle_train_t = corrupt(x_train_raw, t, name, rng)
        oracle_model = train_model(x_oracle_train_t, y_train, seed=SEED + 1)
        err_oracle = eval_error(oracle_model, x_eval_t, y_eval)

        err_frozen = eval_error(frozen_model, x_eval_t, y_eval)
        err_tent = tent_eval_error(frozen_model, x_eval_t, y_eval)
        err_eata = eata_eval_error(frozen_model, fisher, source_vals, x_eval_t, y_eval)
        del oracle_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        d_frozen = err_frozen - err_oracle
        d_tent = err_tent - err_oracle
        d_eata = err_eata - err_oracle
        flag = "  (training horizon)" if t in train_ts else ""
        print(f"{t:>6.3f} | {err_oracle:>10.4f} | {err_frozen:>10.4f} | {err_tent:>9.4f} | {err_eata:>9.4f} | "
              f"{d_frozen:>+9.4f} | {d_tent:>+9.4f} | {d_eata:>+9.4f}{flag}", flush=True)
        rows.append(dict(family=name, t=t, seed=SEED, err_oracle=err_oracle, err_frozen=err_frozen,
                          err_tent=err_tent, err_eata=err_eata, delta_frozen=d_frozen,
                          delta_tent=d_tent, delta_eata=d_eata, is_train_severity=t in train_ts))
    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nresults written: {path}", flush=True)


def try_plot(all_rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(no matplotlib, skipping the plot)", flush=True)
        return
    families = sorted(set(r["family"] for r in all_rows))
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4), squeeze=False)
    for ax, fam in zip(axes[0], families):
        rs = [r for r in all_rows if r["family"] == fam]
        ts = [r["t"] for r in rs]
        ax.plot(ts, [r["err_oracle"] for r in rs], "k-", label="Err_oracle (floor)")
        ax.plot(ts, [r["err_frozen"] for r in rs], "r.-", label="Err_frozen (no adaptation)")
        ax.plot(ts, [r["err_tent"] for r in rs], "b.-", label="Err_TENT")
        ax.plot(ts, [r["err_eata"] for r in rs], "g.-", label="Err_EATA")
        train_horizon = max(t for r, t in zip(rs, ts) if r["is_train_severity"])
        ax.axvline(train_horizon, color="gray", linestyle=":", label="training horizon")
        ax.set_title(fam)
        ax.set_xlabel("t (severity)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot written: {path}", flush=True)


def main():
    global NORM_MEAN, NORM_STD
    print(f"device: {DEVICE}, RUN_TAG: {RUN_TAG} (set via the RUN_SEED environment variable)", flush=True)
    rng = np.random.default_rng(SEED)
    x_train, y_train, x_test, y_test = load_cifar()
    perm = rng.permutation(x_train.shape[0])  # so the frozen chunks are not class-imbalanced
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"  train={x_train.shape[0]} test={x_test.shape[0]}, {x_train.shape[2]}x{x_train.shape[3]}, "
          f"10 classes (CIFAR-10, official split)", flush=True)
    NORM_MEAN = x_train.mean(axis=(0, 2, 3))
    NORM_STD = x_train.std(axis=(0, 2, 3))
    print(f"  normalization (fixed from the t=0 training pool, per channel): mean={NORM_MEAN} std={NORM_STD}", flush=True)

    all_rows = []
    t_start = time.time()
    for name, cfg in FAMILIES.items():
        all_rows.extend(run_family(name, cfg, x_train, y_train, x_test, y_test, rng))
    print(f"\ntotal time: {time.time()-t_start:.1f}s", flush=True)

    write_jsonl(all_rows, f"/kaggle/working/cifar10_c_results_{RUN_TAG}.jsonl")
    try_plot(all_rows, f"/kaggle/working/cifar10_c_delta_{RUN_TAG}.png")


if __name__ == "__main__":
    main()
