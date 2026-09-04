"""Chuan hoa acc va gyro rieng (Phase B - Buoc B2).

mean/std tinh CHI tren split train, luu ra configs/norm_stats.yaml.
Val/test dung lai dung thong ke nay.
"""

from __future__ import annotations

import pathlib

import numpy as np
import yaml


class ImuNormalizer:
    def __init__(
        self,
        acc_mean: list[float],
        acc_std: list[float],
        gyro_mean: list[float],
        gyro_std: list[float],
        eps: float = 1e-6,
    ):
        self.acc_mean = np.asarray(acc_mean, dtype=np.float32)
        self.acc_std = np.asarray(acc_std, dtype=np.float32) + eps
        self.gyro_mean = np.asarray(gyro_mean, dtype=np.float32)
        self.gyro_std = np.asarray(gyro_std, dtype=np.float32) + eps

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """[T, 6] raw -> [T, 6] chuan hoa."""
        out = u.astype(np.float32, copy=True)
        out[:, 0:3] = (out[:, 0:3] - self.acc_mean) / self.acc_std
        out[:, 3:6] = (out[:, 3:6] - self.gyro_mean) / self.gyro_std
        return out

    def denormalize(self, u: np.ndarray) -> np.ndarray:
        out = u.astype(np.float32, copy=True)
        out[:, 0:3] = out[:, 0:3] * self.acc_std + self.acc_mean
        out[:, 3:6] = out[:, 3:6] * self.gyro_std + self.gyro_mean
        return out

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "acc_mean": self.acc_mean.tolist(),
            "acc_std": (self.acc_std - 1e-6).tolist(),
            "gyro_mean": self.gyro_mean.tolist(),
            "gyro_std": (self.gyro_std - 1e-6).tolist(),
        }

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "ImuNormalizer":
        d = yaml.safe_load(pathlib.Path(path).read_text())
        return cls(d["acc_mean"], d["acc_std"], d["gyro_mean"], d["gyro_std"])

    @classmethod
    def identity(cls) -> "ImuNormalizer":
        return cls([0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1])
