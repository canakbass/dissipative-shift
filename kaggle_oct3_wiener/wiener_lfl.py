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
N_TRAIN_IMG = 45000   # 50000 minus the selection split
N_TEST_IMG = 10000
EPOCHS = 15
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 1
# Down to 1e-14: with eps >= 1e-4 the filter's gain is capped at 1/(2 sqrt(eps)) = 50,
# so a coarse grid measures what the regularizer suppresses, not what float32 storage
# destroys. The witness error is an upper bound on the Bayes risk of the data as
# stored, so any eps that beats the oracle shows oracle slack with no float32 caveat.
EPS_GRID = [1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# The same witness on images stored in float64: the difference to the float32 witness
# is what float32 storage costs this witness.
EPS_GRID_F64 = [1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
TEST_TS = [0.25, 0.5, 1.0, 2.0]
N_SELECT = 5000   # held out from all training; used only to pick the Wiener regularizer
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


def blur_forward(imgs, t, dtype=np.float32):
    """Deterministic Gaussian blur via FFT, applied in the frequency domain so that it
    shares the exact kernel definition the deconvolution uses. The result is stored in
    `dtype` (float32 everywhere except the float64-storage witness)."""
    sigma = np.sqrt(2 * t)
    if sigma == 0:
        return imgs.astype(dtype)
    # The forward blur must be the same operator the inverse assumes.
    # An earlier version applied scipy's 4-sigma-truncated kernel with reflect
    # padding in the forward direction while the inverse assumed a periodic,
    # untruncated kernel. Part of Wiener's apparent failure came from that mismatch.
    # Both directions are now a periodic FFT.
    k_fft = gaussian_kernel_fft(sigma, imgs.shape[-1])
    X = np.fft.fft2(imgs, axes=(-2, -1))
    return np.real(np.fft.ifft2(X * k_fft[None, None], axes=(-2, -1))).astype(dtype)


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
    """Like-for-like version of the Wiener test.

    The witness (clean-trained model + Wiener deconvolution) and the oracle are
    evaluated on the same blurred test images, produced by the same periodic FFT
    operator. The oracle is retrained on that operator at each severity with the
    clean model's budget. The regularizer is chosen on a held-out selection split,
    not on the test labels.
    """
    global NORM_MEAN, NORM_STD
    print(f"device: {DEVICE}, seed {SEED}", flush=True)
    x_train, y_train, x_test, y_test = load_cifar()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]
    x_sel, y_sel = x_train[:N_SELECT], y_train[:N_SELECT]
    x_train, y_train = x_train[N_SELECT:N_SELECT + N_TRAIN_IMG], y_train[N_SELECT:N_SELECT + N_TRAIN_IMG]
    # float64 accumulation: x_train is a transposed, non-contiguous float32 view, and
    # under some numpy versions a float32 reduction over it saturates at 2^24 (the
    # Tiny-ImageNet check hit this; the earlier CIFAR runs did not). Fail fast if the
    # statistics are not the standard CIFAR-10 ones the earlier runs used.
    NORM_MEAN = x_train.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    NORM_STD = x_train.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    assert np.allclose(NORM_MEAN, [0.4914, 0.4822, 0.4465], atol=5e-3), NORM_MEAN
    assert np.allclose(NORM_STD, [0.2470, 0.2435, 0.2616], atol=5e-3), NORM_STD
    print(f"  train={x_train.shape[0]} select={x_sel.shape[0]} test={x_test.shape[0]}", flush=True)

    t0 = time.time()
    clean_model = train_clean_model(x_train, y_train, seed=SEED)
    err_clean = eval_error(clean_model, x_test, y_test)
    print(f"clean model trained ({time.time()-t0:.1f}s), clean test error={err_clean:.4f}", flush=True)

    rows = [dict(t=0.0, seed=SEED, err_naive=err_clean, err_witness=err_clean, eps=None,
                 err_oracle=err_clean, err_wiener_test={})]
    for t in TEST_TS:
        t0 = time.time()
        x_blur_te = blur_forward(x_test, t)
        x_blur_sel = blur_forward(x_sel, t)
        sel = {e: eval_error(clean_model, wiener_deconvolve(x_blur_sel, t, e), y_sel) for e in EPS_GRID}
        eps = min(EPS_GRID, key=lambda e: sel[e])
        test_all = {str(e): eval_error(clean_model, wiener_deconvolve(x_blur_te, t, e), y_test) for e in EPS_GRID}
        # float64-stored witness, regularizer chosen the same way
        x64_te, x64_sel = blur_forward(x_test, t, np.float64), blur_forward(x_sel, t, np.float64)
        sel64 = {e: eval_error(clean_model, wiener_deconvolve(x64_sel, t, e), y_sel) for e in EPS_GRID_F64}
        eps64 = min(EPS_GRID_F64, key=lambda e: sel64[e])
        test64 = {str(e): eval_error(clean_model, wiener_deconvolve(x64_te, t, e), y_test) for e in EPS_GRID_F64}
        del x64_te, x64_sel
        k_min = float(np.abs(gaussian_kernel_fft(np.sqrt(2 * t), x_test.shape[-1])).min())
        err_naive = eval_error(clean_model, x_blur_te, y_test)
        oracle = train_clean_model(blur_forward(x_train, t), y_train, seed=SEED)
        err_oracle = eval_error(oracle, x_blur_te, y_test)
        del oracle
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print(f"t={t:<5} naive={err_naive:.4f} witness={test_all[str(eps)]:.4f} (eps={eps:g}, chosen on the "
              f"selection split) witness_f64={test64[str(eps64)]:.4f} (eps={eps64:g}) "
              f"oracle={err_oracle:.4f} k_min={k_min:.2e}  ({time.time()-t0:.1f}s)", flush=True)
        rows.append(dict(t=t, seed=SEED, err_naive=err_naive, err_witness=test_all[str(eps)], eps=eps,
                         err_oracle=err_oracle, err_wiener_test=test_all,
                         err_wiener_select={str(e): v for e, v in sel.items()},
                         err_witness_f64=test64[str(eps64)], eps_f64=eps64, err_wiener_f64_test=test64,
                         err_wiener_f64_select={str(e): v for e, v in sel64.items()}, k_min=k_min))

    out = f"/kaggle/working/wiener_lfl_seed{SEED}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"results written: {out}", flush=True)


if __name__ == "__main__":
    main()
