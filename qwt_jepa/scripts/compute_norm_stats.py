"""Tinh mean/std acc & gyro tren split TRAIN, luu configs/norm_stats.yaml (Buoc B2).

Chay:
    python -m qwt_jepa.scripts.compute_norm_stats --config qwt_jepa/configs/base.yaml
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.data.dataset import _load_imu_txt        # noqa: E402
from qwt_jepa.data.normalize import ImuNormalizer      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_ROOT / "qwt_jepa" / "configs" / "base.yaml"))
    ap.add_argument("--max-rows", type=int, default=4000, help="so sample lay tu manifest (0 = het)")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    data_root = pathlib.Path(cfg["data"]["root"])
    manifest = pathlib.Path(cfg["data"]["manifest"]["train"])
    window = int(cfg["data"]["imu"]["window_size"])

    rows = list(csv.DictReader(open(manifest, newline="")))
    rows = [r for r in rows if int(r["imu_end_row_exclusive"]) - window >= 0]
    if args.max_rows:
        step = max(1, len(rows) // args.max_rows)
        rows = rows[::step]
    print(f"dung {len(rows)} cua so IMU de uoc luong thong ke")

    acc_chunks, gyro_chunks = [], []
    for r in rows:
        end = int(r["imu_end_row_exclusive"])
        s = end - window
        acc_chunks.append(_load_imu_txt(str(data_root / r["imu_acc_path"]))[s:end])
        gyro_chunks.append(_load_imu_txt(str(data_root / r["imu_gyro_path"]))[s:end])

    acc = np.concatenate(acc_chunks, axis=0)
    gyro = np.concatenate(gyro_chunks, axis=0)

    norm = ImuNormalizer(
        acc_mean=acc.mean(0).tolist(),
        acc_std=acc.std(0).tolist(),
        gyro_mean=gyro.mean(0).tolist(),
        gyro_std=gyro.std(0).tolist(),
    )
    out_path = pathlib.Path(cfg["data"]["norm_stats"])
    norm.save(out_path)
    print(f"da luu {out_path}")
    print(yaml.safe_dump(norm.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
