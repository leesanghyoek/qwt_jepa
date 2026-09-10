"""QWT coeffs -> token d_model (Phase C - Buoc C1).

`build_tokens(qwt_image, qwt_imu) -> tokens [B, N, d_model]` dung chung cho ca
nhanh noisy (context) va nhanh clean (target).

Moi token = 1 patch p x p cua mot dai anh, hoac 1 buoc thoi gian cua mot dai IMU.
Chi dung phan vector (i, j, k) cua he so quaternion (phan thuc w luon 0 voi Haar).

Cong 3 embedding: modality (image / imu-acc / imu-gyro), scale/subband, va
positional (bang hoc tuyet doi theo vi tri token).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layout import TokenLayout


def patchify(band: torch.Tensor, gh: int, gw: int, p: int) -> torch.Tensor:
    """[B, 3, gh*p, gw*p] -> [B, gh*gw, 3*p*p]."""
    b = band.shape[0]
    x = band.reshape(b, 3, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(b, gh * gw, 3 * p * p)


def unpatchify(x: torch.Tensor, gh: int, gw: int, p: int) -> torch.Tensor:
    """[B, gh*gw, 3*p*p] -> [B, 3, gh*p, gw*p]."""
    b = x.shape[0]
    x = x.reshape(b, gh, gw, 3, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(b, 3, gh * p, gw * p)


class Tokenizer(nn.Module):
    def __init__(self, cfg: dict, layout: TokenLayout):
        super().__init__()
        self.layout = layout
        d_model = int(cfg["model"]["d_model"])
        p = layout.patch

        self.image_proj = nn.Linear(3 * p * p, d_model)
        self.imu_proj = nn.Linear(3, d_model)
        self.modality_emb = nn.Embedding(3, d_model)
        self.scale_emb = nn.Embedding(layout.num_scales, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, layout.n_tokens, d_model))

        self.register_buffer("mod_ids", layout.modality, persistent=False)
        self.register_buffer("scale_ids", layout.scale_id, persistent=False)

        # Bien do he so QWT chenh ~100 lan giua cac dai (L3 LL std 2.31 vs L1 HH
        # std 0.024). image_proj / imu_proj la MOT Linear dung chung, nen neu khong
        # chuan hoa thi he so dai min cho activation nho xiu -> encoder khong nhin
        # thay chi tiet -> anh tai tao mo. Buffer nen di theo checkpoint.
        # Mac dinh 1.0 = khong chuan hoa; train.py goi set_band_scales() de nap so that.
        self.register_buffer("img_band_scale", torch.ones(len(layout.image_entries)))
        self.register_buffer("imu_band_scale", torch.ones(len(layout.imu_entries)))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.scale_emb.weight, std=0.02)

    def forward(self, qwt_image: dict, qwt_imu: dict) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        for i, e in enumerate(self.layout.image_entries):
            band = qwt_image[e.level][e.band][:, 1:4]          # [B, 3, h, w] (phan vector)
            band = band / self.img_band_scale[i]               # dua moi dai ve std ~1
            patches = patchify(band, e.gh, e.gw, self.layout.patch)
            parts.append(self.image_proj(patches))            # [B, gh*gw, d]

        for i, e in enumerate(self.layout.imu_entries):
            band = qwt_imu[e.group][e.level][e.band][:, :, 1:4]  # [B, L, 3]
            band = band / self.imu_band_scale[i]
            parts.append(self.imu_proj(band))                    # [B, L, d]

        tok = torch.cat(parts, dim=1)                            # [B, N, d]
        tok = tok + self.pos_embed
        tok = tok + self.modality_emb(self.mod_ids).unsqueeze(0)
        tok = tok + self.scale_emb(self.scale_ids).unsqueeze(0)
        return tok
