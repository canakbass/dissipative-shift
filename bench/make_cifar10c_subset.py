"""Generates four CIFAR-10-C corruptions with the functions of the official generator
(hendrycks/robustness, ImageNet-C/create_c/make_cifar_c.py), copied below unchanged
except for numpy 2 compatibility (np.float_ -> np.float64). The released files could not
be downloaded (the Zenodo transfer stalled). Contrast and brightness are deterministic;
fog and Gaussian noise are random and use a fixed seed per corruption.

    python bench/make_cifar10c_subset.py
Writes data/CIFAR-10-C/{contrast,brightness,fog,gaussian_noise,labels}.npy in the
released layout: severities 1-5 stacked, 10000 test images each, uint8, HWC.
"""
import os
import pickle

import numpy as np
import skimage as sk
from PIL import Image

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "CIFAR-10-C")


def plasma_fractal(mapsize=32, wibbledecay=3):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def fog(x, severity=1):
    c = [(.2, 3), (.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:32, :32][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def contrast(x, severity=1):
    c = [.75, .5, .4, .3, 0.15][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.05, .1, .15, .2, .3][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(ROOT, "data", "cifar-10-batches-py", "test_batch")):
        from torchvision.datasets import CIFAR10
        CIFAR10(root=os.path.join(ROOT, "data"), train=False, download=True)
    d = pickle.load(open(os.path.join(ROOT, "data", "cifar-10-batches-py", "test_batch"), "rb"), encoding="bytes")
    images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(d[b"labels"])
    for seed, fn in enumerate((gaussian_noise, fog, brightness, contrast)):
        np.random.seed(seed)
        out = []
        for severity in range(1, 6):
            for img in images:
                out.append(np.uint8(fn(Image.fromarray(img), severity)))
        np.save(os.path.join(OUT, f"{fn.__name__}.npy"), np.array(out).astype(np.uint8))
        print(f"{fn.__name__}: {len(out)} images", flush=True)
    np.save(os.path.join(OUT, "labels.npy"), np.tile(labels, 5).astype(np.uint8))


if __name__ == "__main__":
    main()
