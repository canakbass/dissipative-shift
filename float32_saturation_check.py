"""Reproduces the saturated normalization statistics of the first Tiny-ImageNet run on
synthetic data of the same shape (40000 images, 64x64, 3 channels), laid out the same
way (uint8 HWC -> float32 /255 -> transposed to NCHW, a non-contiguous view).
Needs ~5 GB of RAM and well under a minute. Writes float32_saturation_results.json."""
import gc
import json

import numpy as np

rng = np.random.default_rng(0)
n = 40000
u8 = np.empty((n, 64, 64, 3), np.uint8)
for i in range(0, n, 4000):
    u8[i:i + 4000] = (rng.beta(2, 2.4, (4000, 64, 64, 3)) * 255).astype(np.uint8)
x = np.transpose(u8.astype(np.float32) / 255.0, (0, 3, 1, 2))
del u8
gc.collect()
N = n * 64 * 64
out = dict(numpy_version=np.__version__, n_pixels_per_channel=N, c_contiguous=bool(x.flags["C_CONTIGUOUS"]),
           two_pow_24_over_N=2**24 / N,
           mean_float32=x.mean(axis=(0, 2, 3)).tolist(),
           std_float32=x.std(axis=(0, 2, 3)).tolist(),
           mean_float64=x.mean(axis=(0, 2, 3), dtype=np.float64).tolist(),
           std_float64=x.std(axis=(0, 2, 3), dtype=np.float64).tolist())
json.dump(out, open("float32_saturation_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
