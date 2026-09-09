"""PairedNoisyCleanDataset (Phase B - Buoc B4).

Doc manifest.csv co san cua tartanair-v2-jepa. Chi luu du lieu SACH; nhieu sinh
on-the-fly trong __getitem__, co seed on dinh theo (env, traj, frame, epoch).

Cua so IMU: CAUSAL, 128 mau KET THUC tai `imu_end_row_exclusive` cua manifest
    imu_end   = int(row["imu_end_row_exclusive"])
    imu_start = imu_end - window_size
Loai cac dong co imu_start < 0.
"""

from __future__ import annotations

import csv
import functools
import pathlib
import zlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .corruption import corrupt_image, corrupt_imu
from .normalize import ImuNormalizer


@functools.lru_cache(maxsize=64)
def _load_imu_txt(path: str) -> np.ndarray:
    """Doc file acc.txt / gyro.txt (3 cot, cach nhau khoang trang) -> [N, 3] float32."""
    return np.loadtxt(path, dtype=np.float32)


def _stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


class PairedNoisyCleanDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | pathlib.Path,
        data_root: str | pathlib.Path,
        cfg: dict,
        split: str,
        normalizer: ImuNormalizer | None = None,
        epoch: int = 0,
    ):
        self.root = pathlib.Path(data_root)
        self.cfg = cfg
        self.split = split
        self.epoch = epoch
        self.normalizer = normalizer or ImuNormalizer.identity()

        icfg = cfg["data"]["image"]
        ucfg = cfg["data"]["imu"]
        self.img_size = int(icfg["train_crop"][0] if split == "train" else icfg["eval_crop"][0])
        self.window = int(ucfg["window_size"])
        self.sr = float(ucfg["sampling_rate"])
        self.corr_img_cfg = cfg["corruption"]["image"]
        self.corr_imu_cfg = cfg["corruption"]["imu"]

        env_filter = self._env_filter(cfg, split)

        self.rows: list[dict] = []
        with open(manifest_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                if env_filter is not None and r["environment"] not in env_filter:
                    continue
                end = int(r["imu_end_row_exclusive"])
                if end - self.window < 0:
                    continue
                self.rows.append(
                    {
                        "env": r["environment"],
                        "traj": f'{r["environment"]}/{r["difficulty"]}/{r["trajectory"]}',
                        "frame_idx": int(r["sample_id"].rsplit("__", 1)[-1]),
                        "image_path": r["image_path"],
                        "acc_path": r["imu_acc_path"],
                        "gyro_path": r["imu_gyro_path"],
                        "imu_end": end,
                    }
                )
        if not self.rows:
            raise RuntimeError(f"khong co dong hop le trong {manifest_csv} (split={split})")

    @staticmethod
    def _env_filter(cfg: dict, split: str):
        sp = cfg["data"].get("split", {})
        if not sp.get("enforce_env_split", False):
            return None
        key = {"train": "train_envs", "valid": "val_envs", "val": "val_envs", "test": "test_envs"}[split]
        return set(sp.get(key, []))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    # ------------------------------------------------------------------ #
    def _load_image(self, rel_path: str) -> np.ndarray:
        with Image.open(self.root / rel_path) as im:
            im = im.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0        # [H, W, 3]
        return np.ascontiguousarray(arr.transpose(2, 0, 1))   # [3, H, W]

    def _load_imu(self, row: dict) -> np.ndarray:
        end = row["imu_end"]
        start = end - self.window
        acc = _load_imu_txt(str(self.root / row["acc_path"]))[start:end]     # [128, 3]
        gyro = _load_imu_txt(str(self.root / row["gyro_path"]))[start:end]   # [128, 3]
        return np.concatenate([acc, gyro], axis=1).astype(np.float32)        # [128, 6]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        img_clean = self._load_image(row["image_path"])                     # [3, H, W] in [0,1]
        imu_clean = self.normalizer(self._load_imu(row))                    # [128, 6] chuan hoa

        rng = np.random.default_rng(
            _stable_seed(row["env"], row["traj"], row["frame_idx"], self.epoch)
        )
        img_noisy = corrupt_image(img_clean, rng, self.corr_img_cfg)
        imu_noisy = corrupt_imu(imu_clean, rng, self.corr_imu_cfg, sr=self.sr)

        return {
            "img_clean": torch.from_numpy(img_clean),
            "img_noisy": torch.from_numpy(img_noisy),
            "imu_clean": torch.from_numpy(imu_clean),
            "imu_noisy": torch.from_numpy(imu_noisy),
            "meta": {"env": row["env"], "traj": row["traj"], "frame_idx": row["frame_idx"]},
        }


def jepa_collate(samples: list[dict]) -> dict:
    out = {
        k: torch.stack([s[k] for s in samples], dim=0)
        for k in ("img_clean", "img_noisy", "imu_clean", "imu_noisy")
    }
    out["meta"] = [s["meta"] for s in samples]
    return out
