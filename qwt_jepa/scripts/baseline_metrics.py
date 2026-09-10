"""Do NGUONG BASELINE cho tung chi so trong log train (khong can model).

Tra loi cau hoi "PSNR bao nhieu la tot?" bang cach do trom hai moc:
  1. Neu model tra ve DUNG anh nhieu dau vao  -> nguong PHAI vuot.
  2. Neu model tra ve mau trung binh cua anh   -> san tuyet doi (vo dung).

Cac moc nay phu thuoc cau hinh `corruption` trong base.yaml, nen phai chay lai
moi khi sua muc do nhieu.

    python -m qwt_jepa.scripts.baseline_metrics --split valid --n 128
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.data.dataset import PairedNoisyCleanDataset    # noqa: E402
from qwt_jepa.data.normalize import ImuNormalizer            # noqa: E402


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return -10.0 * math.log10(max(float(((a - b) ** 2).mean()), 1e-10))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_ROOT / "qwt_jepa" / "configs" / "base.yaml"))
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=128, help="so mau lay mau thua")
    ap.add_argument("--epoch", type=int, default=0, help="seed corruption (khop set_epoch)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    nz = ImuNormalizer.from_yaml(cfg["data"]["norm_stats"])
    ds = PairedNoisyCleanDataset(
        cfg["data"]["manifest"][args.split], cfg["data"]["root"], cfg, args.split, nz, args.epoch
    )

    print(f"root     : {cfg['data']['root']}")
    print(f"manifest : {cfg['data']['manifest'][args.split]}")
    print(f"so mau   : {len(ds)} (sau khi bo dong co imu_end < window)")

    acc: dict[str, list[float]] = {k: [] for k in (
        "psnr_noisy", "psnr_mean", "limg_noisy", "limg_mean",
        "rmse_acc", "rmse_gyro", "limu_noisy",
    )}
    stride = max(1, len(ds) // args.n)
    for k in range(args.n):
        s = ds[(k * stride) % len(ds)]
        c, n = s["img_clean"].numpy(), s["img_noisy"].numpy()
        acc["psnr_noisy"].append(psnr(n, c))
        acc["psnr_mean"].append(psnr(np.full_like(c, c.mean()), c))
        acc["limg_noisy"].append(float(np.abs(n - c).mean()))
        acc["limg_mean"].append(float(np.abs(c - c.mean()).mean()))

        ic, iz = s["imu_clean"].numpy(), s["imu_noisy"].numpy()
        acc["rmse_acc"].append(float(np.sqrt(((iz[:, 0:3] - ic[:, 0:3]) ** 2).mean())))
        acc["rmse_gyro"].append(float(np.sqrt(((iz[:, 3:6] - ic[:, 3:6]) ** 2).mean())))
        acc["limu_noisy"].append(
            float(np.abs(iz[:, 0:3] - ic[:, 0:3]).mean() + np.abs(iz[:, 3:6] - ic[:, 3:6]).mean())
        )

    print(f"\n--- n={args.n}, epoch seed={args.epoch} ---")
    print(f"{'chi so':<34}{'mean':>9}{'p10':>9}{'p90':>9}")
    for k, v in acc.items():
        a = np.asarray(v)
        print(f"{k:<34}{a.mean():>9.3f}{np.percentile(a, 10):>9.3f}{np.percentile(a, 90):>9.3f}")


if __name__ == "__main__":
    main()
