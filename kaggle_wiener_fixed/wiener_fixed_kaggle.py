"""
Pre-registered test (PREREG_v3.md): Gaussian blur is provably invertible, since
its Fourier transform never reaches exactly zero. If the true R*(t) equals R*(0),
then the retrained oracle's rise with t must come only from a training or
representation limit, not from lost information. We test that directly: train one
clean (t=0) classifier, invert the blurred test images by Wiener deconvolution, and
classify them with that same clean model. If the Wiener-plus-clean-model error is
lower than the oracle retrained at that severity, the oracle is demonstrably loose.

Two corrections after review. The forward blur now uses the same periodic FFT
operator the deconvolution assumes -- previously it used a truncated kernel with
reflect padding, so the inverse was undoing an operator we had not applied. And
the Wiener regularizer is swept rather than fixed at 1e-2, which suppressed every
frequency the test exists to recover.
"""
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
SEED = 1
EPS_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
TEST_TS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
NORM_MEAN, NORM_STD = None, None


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


def blur_forward(imgs, t):
    """Deterministic Gaussian blur via FFT, applied in the frequency domain so that it
    shares the exact kernel definition the deconvolution uses."""
    sigma = np.sqrt(2 * t)
    if sigma == 0:
        return imgs.copy()
    # The forward blur must be the same operator the inverse assumes.
    # An earlier version applied scipy's 4-sigma-truncated kernel with reflect
    # padding in the forward direction while the inverse assumed a periodic,
    # untruncated kernel. Part of Wiener's apparent failure came from that mismatch.
    # Both directions are now a periodic FFT.
    if sigma == 0:
        return imgs.copy()
    k_fft = gaussian_kernel_fft(sigma, imgs.shape[-1])
    X = np.fft.fft2(imgs, axes=(-2, -1))
    return np.real(np.fft.ifft2(X * k_fft[None, None], axes=(-2, -1))).astype(np.float32)


def gaussian_kernel_fft(sigma, size):
    """FFT of a Gaussian kernel centred on a size x size spatial grid, under periodic
    boundary conditions -- which is what the forward operator now also uses."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = np.fft.ifftshift(kernel)
    return np.fft.fft2(kernel)


def wiener_deconvolve(imgs, t, eps):
    """imgs: (N,C,H,W) blurred images. Applies a Wiener (regularized inverse) filter
    per image and per channel via a 2-D FFT."""
    sigma = np.sqrt(2 * t)
    if sigma == 0:
        return imgs.copy()
    n, c, h, w = imgs.shape
    k_fft = gaussian_kernel_fft(sigma, h)  # assumes h == w (CIFAR: 32x32)
    out = np.empty_like(imgs)
    denom = np.abs(k_fft) ** 2 + eps
    filt = np.conj(k_fft) / denom
    for i in range(n):
        for ch in range(c):
            Y = np.fft.fft2(imgs[i, ch])
            X_hat = np.real(np.fft.ifft2(Y * filt))
            out[i, ch] = X_hat
    return out.astype(np.float32)


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


class ResNet18(nn.Module):
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
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    top, left = int(torch.randint(0, 9, (1,))), int(torch.randint(0, 9, (1,)))
    out = xp[:, :, top:top + 32, left:left + 32]
    if torch.rand(1).item() < 0.5:
        out = out.flip(-1)
    return out


def to_tensor(x, y):
    mean = NORM_MEAN.reshape(1, 3, 1, 1)
    std = NORM_STD.reshape(1, 3, 1, 1)
    x = (x - mean) / std
    return torch.from_numpy(x).float().to(DEVICE), torch.from_numpy(y).long().to(DEVICE)


def train_clean_model(x, y, epochs=EPOCHS, seed=SEED, val_frac=0.15):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    model = ResNet18().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
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
    return model


@torch.no_grad()
def eval_error(model, x, y):
    model.eval()
    x_t, y_t = to_tensor(x, y)
    pred = model(x_t).argmax(dim=1)
    return (pred != y_t).float().mean().item()


def main():
    global NORM_MEAN, NORM_STD
    print(f"device: {DEVICE}", flush=True)
    x_train, y_train, x_test, y_test = load_cifar()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]
    x_train, y_train = x_train[:N_TRAIN_IMG], y_train[:N_TRAIN_IMG]
    NORM_MEAN = x_train.mean(axis=(0, 2, 3))
    NORM_STD = x_train.std(axis=(0, 2, 3))
    print(f"  train={x_train.shape[0]} test={x_test.shape[0]}", flush=True)

    t0 = time.time()
    clean_model = train_clean_model(x_train, y_train)
    err_clean = eval_error(clean_model, x_test, y_test)
    print(f"clean (t=0) model trained ({time.time()-t0:.1f}s), clean test error={err_clean:.4f}", flush=True)

    rows = [dict(t=0.0, err_naive=err_clean, err_wiener={str(e): err_clean for e in EPS_GRID},
                 err_wiener_best=err_clean, best_eps=None)]
    print(f"{'t':>6} | {'Err_naive':>10} | " + " | ".join(f"eps={e:<7g}" for e in EPS_GRID)
          + f" | {'best':>8}", flush=True)
    for t in TEST_TS:
        t0 = time.time()
        x_blur = blur_forward(x_test, t)
        err_naive = eval_error(clean_model, x_blur, y_test)
        errs = {}
        for e in EPS_GRID:
            errs[str(e)] = eval_error(clean_model, wiener_deconvolve(x_blur, t, e), y_test)
        best_eps = min(EPS_GRID, key=lambda e: errs[str(e)])
        print(f"{t:>6.2f} | {err_naive:>10.4f} | "
              + " | ".join(f"{errs[str(e)]:>11.4f}" for e in EPS_GRID)
              + f" | {errs[str(best_eps)]:>8.4f}  (eps={best_eps:g}, {time.time()-t0:.1f}s)", flush=True)
        rows.append(dict(t=t, err_naive=err_naive, err_wiener=errs,
                         err_wiener_best=errs[str(best_eps)], best_eps=best_eps))

    with open("/kaggle/working/wiener_test_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\nresults written: /kaggle/working/wiener_test_results.jsonl", flush=True)


if __name__ == "__main__":
    main()
