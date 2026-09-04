"""QWT/iQWT Haar 2D nhieu muc cho anh, ban PyTorch (Phase A - Buoc A1).

Chuyen tu testQWT.py (numpy) sang torch, chay tren batch [B, 3, H, W] va GPU.

Moi pixel RGB -> quaternion thuan  q = 0 + R i + G j + B k  ->  bien doi Haar
dong thoi tren 4 thanh phan (w, x, y, z). Bo loc Haar he so thuc nen w = 0
duoc giu nguyen qua moi muc.

Kim tu thap he so:
  - Muc 1..L-1 : giu {LH, HL, HH}, phan ra tiep dai LL.
  - Muc L      : giu ca {LL, LH, HL, HH}.

API:
    coeffs = image_qwt(q, levels=3)   # q: [B, 4, H, W]
    q_rec  = image_iqwt(coeffs)       # -> [B, 4, H, W]
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# RGB <-> quaternion thuan
# --------------------------------------------------------------------------- #
def rgb_to_quat(x: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] -> [B, 4, H, W] voi kenh 0 (phan thuc w) = 0."""
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"can [B, 3, H, W], nhan {tuple(x.shape)}")
    w = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.cat([w, x], dim=1)


def quat_to_rgb(q: torch.Tensor) -> torch.Tensor:
    """[B, 4, H, W] -> [B, 3, H, W], lay phan vector (i, j, k)."""
    return q[:, 1:4]


# --------------------------------------------------------------------------- #
# Tien ich
# --------------------------------------------------------------------------- #
def _interleave(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    """Xen ke a, b doc theo `dim`: ket qua[..., 2k] = a[..., k], [..., 2k+1] = b[..., k]."""
    dim = dim % a.dim()
    stacked = torch.stack((a, b), dim=dim + 1)
    shape = list(a.shape)
    shape[dim] = a.shape[dim] * 2
    return stacked.reshape(shape)


def _pad_even_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Dem canh (replicate) cho H, W chan. Tra ve kich thuoc goc de iQWT cat lai."""
    h, w = x.shape[-2:]
    ph, pw = h % 2, w % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


def _haar2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mot muc Haar 2D tren [B, C, H, W] (H, W chan). Tra ve LL, LH, HL, HH."""
    lo_x = (x[..., 0::2] + x[..., 1::2]) / _SQRT2           # thap theo x  [B, C, H, W/2]
    hi_x = (x[..., 0::2] - x[..., 1::2]) / _SQRT2           # cao theo x
    ll = (lo_x[..., 0::2, :] + lo_x[..., 1::2, :]) / _SQRT2  # [B, C, H/2, W/2]
    lh = (lo_x[..., 0::2, :] - lo_x[..., 1::2, :]) / _SQRT2
    hl = (hi_x[..., 0::2, :] + hi_x[..., 1::2, :]) / _SQRT2
    hh = (hi_x[..., 0::2, :] - hi_x[..., 1::2, :]) / _SQRT2
    return ll, lh, hl, hh


def _ihaar2(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Nghich dao mot muc Haar 2D. Tra ve [B, C, 2H, 2W]."""
    lo_x_e = (ll + lh) / _SQRT2
    lo_x_o = (ll - lh) / _SQRT2
    hi_x_e = (hl + hh) / _SQRT2
    hi_x_o = (hl - hh) / _SQRT2
    lo_x = _interleave(lo_x_e, lo_x_o, dim=-2)   # ghep lai theo y
    hi_x = _interleave(hi_x_e, hi_x_o, dim=-2)
    x_e = (lo_x + hi_x) / _SQRT2
    x_o = (lo_x - hi_x) / _SQRT2
    return _interleave(x_e, x_o, dim=-1)         # ghep lai theo x


# --------------------------------------------------------------------------- #
# API chinh
# --------------------------------------------------------------------------- #
def image_qwt(q: torch.Tensor, levels: int = 3) -> dict:
    """QWT Haar 2D nhieu muc.

    Args:
        q: [B, 4, H, W] quaternion thuan (dung rgb_to_quat truoc).
        levels: so muc phan ra.

    Returns:
        dict:
          {"levels": L,
           1: {"LH", "HL", "HH", "shape": (h, w)},
           ...,
           L: {"LL", "LH", "HL", "HH", "shape": (h, w)}}
        moi dai co shape [B, 4, h_l, w_l]; "shape" la kich thuoc dau vao cua muc do.
    """
    if q.dim() != 4 or q.shape[1] != 4:
        raise ValueError(f"can [B, 4, H, W], nhan {tuple(q.shape)}")

    pyr: dict = {"levels": int(levels)}
    cur = q
    for lvl in range(1, levels + 1):
        cur, shape = _pad_even_2d(cur)
        ll, lh, hl, hh = _haar2(cur)
        pyr[lvl] = {"LH": lh, "HL": hl, "HH": hh, "shape": shape}
        cur = ll
    pyr[levels]["LL"] = cur
    return pyr


def image_iqwt(pyr: dict, shapes: dict | None = None) -> torch.Tensor:
    """Nghich dao image_qwt.

    Args:
        pyr: dict tra ve boi image_qwt, hoac dict do recon head tao (LL chi o muc sau cung).
        shapes: {level: (h, w)} de cat lai; neu None thi lay pyr[level]["shape"].

    Returns:
        [B, 4, H, W].
    """
    levels = pyr["levels"]
    cur = pyr[levels]["LL"]
    for lvl in range(levels, 0, -1):
        band = pyr[lvl]
        shape = shapes[lvl] if shapes is not None else band["shape"]
        cur = _ihaar2(cur, band["LH"], band["HL"], band["HH"])
        cur = cur[..., : shape[0], : shape[1]]
    return cur
