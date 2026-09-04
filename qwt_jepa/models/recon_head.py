"""Hai reconstruction head + iQWT (Phase E - Buoc E1).

Tu embedding tai tung token -> du doan he so QWT sach cua tung dai -> iQWT ve
mien goc.

  Image head : emb token anh  -> he so QWT sach -> image_iqwt -> [B, 3, 256, 256]
  IMU head   : emb token IMU  -> he so QWT sach -> imu_iqwt   -> [B, 128, 6]

Head chi la vai lop Linear; phan "lam sach" chu yeu do latent JEPA ganh.
Chi du doan phan vector (i, j, k); phan thuc w gan 0 truoc khi iQWT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..qwt.image_qwt import image_iqwt
from ..qwt.imu_qwt import imu_iqwt
from .layout import TokenLayout
from .tokenizer import unpatchify


def _pad_w0_img(v: torch.Tensor) -> torch.Tensor:
    """[B, 3, h, w] -> [B, 4, h, w] voi kenh w = 0."""
    w = v.new_zeros(v.shape[0], 1, v.shape[2], v.shape[3])
    return torch.cat([w, v], dim=1)


def _pad_w0_seq(v: torch.Tensor) -> torch.Tensor:
    """[B, L, 3] -> [B, L, 4] voi w = 0."""
    w = v.new_zeros(v.shape[0], v.shape[1], 1)
    return torch.cat([w, v], dim=-1)


class ImageHead(nn.Module):
    def __init__(self, cfg: dict, layout: TokenLayout):
        super().__init__()
        d_model = int(cfg["model"]["d_model"])
        self.p = layout.patch
        self.levels = layout.levels_image
        self.shapes = layout.image_shapes
        self.proj = nn.Linear(d_model, 3 * self.p * self.p)

    def forward(self, emb_full: torch.Tensor, layout: TokenLayout) -> torch.Tensor:
        pyr: dict = {"levels": self.levels}
        for e in layout.image_entries:
            tok = emb_full[:, e.start:e.end]                      # [B, gh*gw, d]
            band = self.proj(tok)                                 # [B, gh*gw, 3p^2]
            band = unpatchify(band, e.gh, e.gw, self.p)           # [B, 3, h, w]
            pyr.setdefault(e.level, {})[e.band] = _pad_w0_img(band)
        q = image_iqwt(pyr, shapes=self.shapes)                   # [B, 4, H, W]
        return q[:, 1:4]                                          # [B, 3, H, W]


class ImuHead(nn.Module):
    def __init__(self, cfg: dict, layout: TokenLayout):
        super().__init__()
        d_model = int(cfg["model"]["d_model"])
        self.levels = layout.levels_imu
        self.lens = layout.imu_shapes
        self.proj = nn.Linear(d_model, 3)

    def forward(self, emb_full: torch.Tensor, layout: TokenLayout) -> torch.Tensor:
        coeffs: dict = {
            "acc": {"levels": self.levels},
            "gyro": {"levels": self.levels},
        }
        for e in layout.imu_entries:
            tok = emb_full[:, e.start:e.end]              # [B, L, d]
            v = _pad_w0_seq(self.proj(tok))               # [B, L, 4]
            coeffs[e.group].setdefault(e.level, {})[e.band] = v
        return imu_iqwt(coeffs, lens=self.lens)           # [B, T, 6]
