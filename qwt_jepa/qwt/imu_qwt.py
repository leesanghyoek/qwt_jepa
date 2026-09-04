"""QWT/iQWT Haar 1D nhieu muc cho IMU, ban PyTorch (Phase A - Buoc A2).

6 kenh -> 2 quaternion thuan tach rieng:
    q_acc  = 0 + ax i + ay j + az k
    q_gyro = 0 + gx i + gy j + gz k

Haar 1D tren truc thoi gian, nhieu muc. Do dai le duoc dem canh (replicate),
tra ve `len` de iQWT cat lai.

API:
    coeffs = imu_qwt(u, levels=3)     # u: [B, T, 6]  -> {"acc": pyr, "gyro": pyr}
    u_rec  = imu_iqwt(coeffs)         # -> [B, T, 6]
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_SQRT2 = math.sqrt(2.0)


def _to_pure_quat(v: torch.Tensor) -> torch.Tensor:
    """[B, T, 3] -> [B, T, 4] voi phan thuc w = 0."""
    w = v.new_zeros(v.shape[0], v.shape[1], 1)
    return torch.cat([w, v], dim=-1)


def _interleave(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % a.dim()
    stacked = torch.stack((a, b), dim=dim + 1)
    shape = list(a.shape)
    shape[dim] = a.shape[dim] * 2
    return stacked.reshape(shape)


def _pad_even_1d(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Dem 1 mau (replicate) doc theo truc thoi gian neu do dai le."""
    t = x.shape[1]
    if t % 2:
        x = F.pad(x.transpose(1, 2), (0, 1), mode="replicate").transpose(1, 2)
    return x, t


def _haar1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mot muc Haar 1D tren [B, T, C] (T chan). Tra ve A (xap xi), D (chi tiet)."""
    a = (x[:, 0::2, :] + x[:, 1::2, :]) / _SQRT2
    d = (x[:, 0::2, :] - x[:, 1::2, :]) / _SQRT2
    return a, d


def _ihaar1(a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    even = (a + d) / _SQRT2
    odd = (a - d) / _SQRT2
    return _interleave(even, odd, dim=1)


def _group_qwt(q: torch.Tensor, levels: int) -> dict:
    """QWT 1D cho mot nhom (acc hoac gyro), q: [B, T, 4]."""
    pyr: dict = {"levels": int(levels)}
    cur = q
    for lvl in range(1, levels + 1):
        cur, length = _pad_even_1d(cur)
        a, d = _haar1(cur)
        pyr[lvl] = {"D": d, "len": length}
        cur = a
    pyr[levels]["A"] = cur
    return pyr


def _group_iqwt(pyr: dict, lens: dict | None = None) -> torch.Tensor:
    levels = pyr["levels"]
    cur = pyr[levels]["A"]
    for lvl in range(levels, 0, -1):
        band = pyr[lvl]
        length = lens[lvl] if lens is not None else band["len"]
        cur = _ihaar1(cur, band["D"])[:, :length, :]
    return cur


# --------------------------------------------------------------------------- #
# API chinh
# --------------------------------------------------------------------------- #
def imu_qwt(u: torch.Tensor, levels: int = 3) -> dict:
    """QWT Haar 1D nhieu muc cho IMU.

    Args:
        u: [B, T, 6] = [ax, ay, az, gx, gy, gz] (da chuan hoa).
        levels: so muc.

    Returns:
        {"acc": pyr_acc, "gyro": pyr_gyro}, moi pyr:
          {"levels": L,
           1: {"D", "len"}, ..., L: {"A", "D", "len"}}
        moi dai co shape [B, t_l, 4].
    """
    if u.dim() != 3 or u.shape[-1] != 6:
        raise ValueError(f"can [B, T, 6], nhan {tuple(u.shape)}")
    acc = _to_pure_quat(u[..., 0:3])
    gyro = _to_pure_quat(u[..., 3:6])
    return {"acc": _group_qwt(acc, levels), "gyro": _group_qwt(gyro, levels)}


def imu_iqwt(coeffs: dict, lens: dict | None = None) -> torch.Tensor:
    """Nghich dao imu_qwt. Tra ve [B, T, 6] (ghep acc | gyro, lay phan vector)."""
    acc = _group_iqwt(coeffs["acc"], lens)
    gyro = _group_iqwt(coeffs["gyro"], lens)
    return torch.cat([acc[..., 1:4], gyro[..., 1:4]], dim=-1)
