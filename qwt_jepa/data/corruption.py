"""Corruption on-the-fly (Phase B - Buoc B3).

Ap trong MIEN GOC, TRUOC QWT. Nhan `rng` (numpy Generator) de tai lap.
Khong bao gio ghi de file sach; khong tao nhieu truc tiep tren he so QWT.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Anh: [3, H, W] float32 trong [0, 1]
# --------------------------------------------------------------------------- #
def _box1d(x: np.ndarray, k: int, axis: int) -> np.ndarray:
    kernel = np.ones(k, dtype=np.float32) / k
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis, x)


def corrupt_image(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Gaussian noise + motion blur + giam sang + luong tu hoa kieu nen JPEG."""
    out = img.astype(np.float32, copy=True)

    std = float(rng.uniform(*cfg["gauss_std"]))
    if std > 0:
        out = out + rng.normal(0.0, std, out.shape).astype(np.float32)

    k_lo, k_hi = cfg["blur_kernel"]
    k = int(rng.integers(int(k_lo), int(k_hi) + 1))
    if k >= 2:
        axis = 2 if rng.random() < 0.5 else 1          # blur ngang hoac doc
        out = _box1d(out, k, axis=axis)

    out = out * float(rng.uniform(0.7, 1.1))           # giam sang / tang nhe

    q_lo, q_hi = cfg["jpeg_q"]                         # q thap -> nen manh -> it muc luong tu
    q = float(rng.uniform(q_lo, q_hi))
    levels = max(4, int(round(q / 4)))
    out = np.round(np.clip(out, 0.0, 1.0) * levels) / levels

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# IMU: [T, 6] da chuan hoa (acc | gyro)
# --------------------------------------------------------------------------- #
def corrupt_imu(u: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Gaussian theo kenh + bias (hang so theo cua so) + drift (random walk) + spike thua."""
    out = u.astype(np.float32, copy=True)
    t, c = out.shape

    std = float(rng.uniform(*cfg["gauss_std"]))
    if std > 0:
        out = out + rng.normal(0.0, std, out.shape).astype(np.float32)

    b_max = float(cfg["bias"][1])
    if b_max > 0:
        out = out + rng.uniform(-b_max, b_max, size=c).astype(np.float32)

    d_max = float(cfg["drift"][1])
    if d_max > 0:
        steps = rng.normal(0.0, d_max, size=(t, c)).astype(np.float32)
        out = out + np.cumsum(steps, axis=0) / np.sqrt(t)

    p = float(cfg["spike_prob"])
    if p > 0:
        mask = rng.random((t, c)) < p
        n = int(mask.sum())
        if n:
            out[mask] += rng.normal(0.0, 5.0 * max(std, 1e-3), size=n).astype(np.float32)

    return out.astype(np.float32)
