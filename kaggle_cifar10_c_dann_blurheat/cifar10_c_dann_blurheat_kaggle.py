"""
Stage 3 addendum: a DANN (Ganin & Lempitsky 2015) comparison on CIFAR-10-C.

Note on method: DANN is not a test-time adaptation method like
TENT or EATA. It is a domain-adaptation method that needs access to unlabeled
images from the target severity during training -- privileged information the
test-time methods do not get. So Delta_dann(t) is not a like-for-like comparison
against Delta_tent(t) or Delta_eata(t); DANN answers a different question ("how
much would you gain if you had unlabeled target data at training time?") from the
one TENT and EATA answer ("how much can you gain after the model is frozen and
deployed, with no training-time access to target data at all?").

Source (labeled): the same low-severity pool the frozen model uses, for a fair
baseline. Target (unlabeled): the same x_train_raw pool corrupted at the test
severity, with its labels discarded and only a domain=1 label for the domain
classifier. One simplification worth noting: the source and target image pools can
overlap. This is not label leakage, since the labels serve different purposes, but
we record it.

A separate DANN model is trained for each (family, severity) pair, mirroring the
oracle's per-severity structure.
"""
import copy
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.datasets import CIFAR10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/kaggle/working/cifar10"
N_TRAIN_IMG = 50000
N_TEST_IMG = 10000
EPOCHS = 15
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = int(os.environ.get("RUN_SEED", "1"))
RUN_TAG = f"dann_blurheat_seed{SEED}"
NORM_MEAN, NORM_STD = None, None

FAMILIES = {
    "blur_heat": dict(
        train_ts=[0.0, 0.05, 0.1],
        test_ts=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    ),
}


def _to_nchw01(imgs_uint8):
    return np.transpose(imgs_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2))


def load_cifar():
    train_ds = CIFAR10(root=DATA_ROOT, train=True, download=True)
    test_ds = CIFAR10(root=DATA_ROOT, train=False, download=True)
    x_train = _to_nchw01(train_ds.data)
    y_train = np.array(train_ds.targets, dtype=np.int64)
    x_test = _to_nchw01(test_ds.data)
    y_test = np.array(test_ds.targets, dtype=np.int64)
    return x_train, y_train, x_test, y_test


FOG_SIGMA = 0.2  # OU diffusion coefficient; stationary std = FOG_SIGMA/sqrt(2)


def corrupt(imgs, t, family, rng):
    """Clipping removed (it broke the semigroup
    property) and fog is now an Ornstein-Uhlenbeck process (an exact semigroup and a
    genuinely information-destroying kernel). Same correction as the main script; see
    the comment there and kill_test_fog.py."""
    if family == "gauss_noise":
        return imgs + rng.normal(0, np.sqrt(2 * t), imgs.shape).astype(np.float32) if t > 0 else imgs.copy()
    if family == "gauss_blur":
        sigma = np.sqrt(2 * t)
        if sigma == 0:
            return imgs.copy()
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
        fy = np.fft.fftfreq(h) * 2 * np.pi
        fx = np.fft.fftfreq(w) * 2 * np.pi
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        rate = FY**2 + FX**2 + 0.1  # BLUR_HEAT_A_MIN
        decay = np.exp(-rate * t)
        X = np.fft.fft2(imgs, axes=(-2, -1))
        mean = np.real(np.fft.ifft2(X * decay[None, None], axes=(-2, -1))).astype(np.float32)
        if t <= 0:
            return mean
        var = (15.0**2 / (2 * rate)) * (1 - decay**2)  # BLUR_HEAT_SIGMA=15.0
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
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DANNResNet18(nn.Module):
    """ResNet-18 trunk + label head + domain head behind a gradient reversal layer."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.label_head = nn.Linear(512, n_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def _make_layer(self, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x, lambd=0.0):
        feat = self.features(x)
        label_out = self.label_head(feat)
        domain_out = self.domain_head(grad_reverse(feat, lambd))
        return label_out, domain_out


def augment_batch(x):
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    top, left = int(torch.randint(0, 9, (1,))), int(torch.randint(0, 9, (1,)))
    out = xp[:, :, top:top + 32, left:left + 32]
    if torch.rand(1).item() < 0.5:
        out = out.flip(-1)
    return out


def to_tensor(x, y=None):
    mean = NORM_MEAN.reshape(1, 3, 1, 1)
    std = NORM_STD.reshape(1, 3, 1, 1)
    x = (x - mean) / std
    xt = torch.from_numpy(x).float().to(DEVICE)
    if y is None:
        return xt
    return xt, torch.from_numpy(y).long().to(DEVICE)


def train_dann(x_src, y_src, x_tgt, epochs=EPOCHS, seed=SEED, val_frac=0.15):
    """x_src/y_src: labeled source. x_tgt: unlabeled target at the same severity."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = x_src.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = DANNResNet18().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                           weight_decay=WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    x_src_t, y_src_t = to_tensor(x_src, y_src)
    x_tr, y_tr = x_src_t[train_idx], y_src_t[train_idx]
    x_val, y_val = x_src_t[val_idx], y_src_t[val_idx]
    x_tgt_t = to_tensor(x_tgt)
    n_tr, n_tgt = x_tr.shape[0], x_tgt_t.shape[0]

    best_val_acc, best_state = -1.0, None
    total_steps = epochs * (n_tr // BATCH_SIZE + 1)
    step = 0
    for ep in range(epochs):
        model.train()
        tperm = torch.randperm(n_tr, device=DEVICE)
        gperm = torch.randperm(n_tgt, device=DEVICE)
        for i in range(0, n_tr, BATCH_SIZE):
            p = step / max(1, total_steps)
            lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # DANN's standard lambda schedule
            step += 1

            b_src = tperm[i:i + BATCH_SIZE]
            b_tgt = gperm[(i % n_tgt):(i % n_tgt) + BATCH_SIZE]
            if b_tgt.shape[0] < 2:
                b_tgt = gperm[:BATCH_SIZE]
            xb_src = augment_batch(x_tr[b_src])
            xb_tgt = augment_batch(x_tgt_t[b_tgt])
            yb_src = y_tr[b_src]

            label_out, dom_src = model(xb_src, lambd)
            _, dom_tgt = model(xb_tgt, lambd)
            dom_labels_src = torch.zeros(xb_src.shape[0], dtype=torch.long, device=DEVICE)
            dom_labels_tgt = torch.ones(xb_tgt.shape[0], dtype=torch.long, device=DEVICE)

            loss = (ce(label_out, yb_src)
                    + ce(dom_src, dom_labels_src)
                    + ce(dom_tgt, dom_labels_tgt))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val)[0].argmax(1) == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    del x_src_t, y_src_t, x_tr, y_tr, x_val, y_val, x_tgt_t, opt, sched, best_state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def eval_error(model, x, y):
    model.eval()
    x_t, y_t = to_tensor(x, y)
    pred = model(x_t)[0].argmax(dim=1)
    return (pred != y_t).float().mean().item()


def run_family(name, cfg, x_train_raw, y_train, x_eval_raw, y_eval, rng):
    print(f"\n########## Family: {name} ##########", flush=True)
    train_ts, test_ts = cfg["train_ts"], cfg["test_ts"]
    n_per_ts = N_TRAIN_IMG // len(train_ts)
    src_chunks, src_labels = [], []
    for i, t in enumerate(train_ts):
        sl = slice(i * n_per_ts, (i + 1) * n_per_ts)
        src_chunks.append(corrupt(x_train_raw[sl], t, name, rng))
        src_labels.append(y_train[sl])
    x_src = np.concatenate(src_chunks)
    y_src = np.concatenate(src_labels)

    rows = []
    header = f"{'t':>6} | {'Err_dann':>9} | not"
    print(header, flush=True)
    for t in test_ts:
        x_eval_t = corrupt(x_eval_raw, t, name, rng)
        x_tgt = corrupt(x_train_raw, t, name, rng)  # unlabeled target; labels discarded
        t0 = time.time()
        model = train_dann(x_src, y_src, x_tgt, seed=SEED)
        err_dann = eval_error(model, x_eval_t, y_eval)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        flag = "  (training horizon)" if t in train_ts else ""
        print(f"{t:>6.3f} | {err_dann:>9.4f}  ({time.time()-t0:.1f}s){flag}", flush=True)
        rows.append(dict(family=name, t=t, seed=SEED, err_dann=err_dann,
                          is_train_severity=t in train_ts))
    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nresults written: {path}", flush=True)


def main():
    global NORM_MEAN, NORM_STD
    print(f"device: {DEVICE}, RUN_TAG: {RUN_TAG}", flush=True)
    rng = np.random.default_rng(SEED)
    x_train, y_train, x_test, y_test = load_cifar()
    perm = rng.permutation(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]
    NORM_MEAN = x_train.mean(axis=(0, 2, 3))
    NORM_STD = x_train.std(axis=(0, 2, 3))
    print(f"  train={x_train.shape[0]} test={x_test.shape[0]}", flush=True)

    all_rows = []
    t_start = time.time()
    for name, cfg in FAMILIES.items():
        all_rows.extend(run_family(name, cfg, x_train, y_train, x_test, y_test, rng))
    print(f"\ntotal time: {time.time()-t_start:.1f}s", flush=True)
    write_jsonl(all_rows, f"/kaggle/working/cifar10_c_results_{RUN_TAG}.jsonl")


if __name__ == "__main__":
    main()
