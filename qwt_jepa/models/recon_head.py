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
from .tokenizer import patchify, unpatchify


class _GatedHead(nn.Module):
    """Phan dung chung cua hai recon head: he so nhan CO CHAN."""

    def _gain(self, g: torch.Tensor) -> torch.Tensor:
        """1 + gate_max * tanh(g). Bang 1.0 dung khi g = 0 -> khoi tao van la phep copy.

        Vi sao phai chan: `1 + g` khong chan lam model PHAN KY. Do that (head_lr_mult 10):
            e0 PSNR 17.50 net 1.393 | e1 10.68 net 2.368 | e2 13.62 net 1.829
        net > 1 nghia la anh ra NET HON ca anh sach - model bia tan so cao de trong net,
        va L_img tren TRAIN tang 0.044 -> 0.25. He so nhan khong chan la phep NHAN, nen
        sai so nho o g bi nhan len theo bien do he so - dai LL bien do lon thi mot chut
        lech cung thanh loi to. tanh giu he so trong [1-gate_max, 1+gate_max].
        gate_max = 1.0 -> he so trong [0, 2]: van du cho khu nhoe (can > 1) va co he so
        (can < 1), nhung khong the phong to vo han.
        """
        gm = float(getattr(self, "gate_max", 1.0))
        return 1.0 + gm * torch.tanh(g) if gm > 0 else 1.0 + g


def _pad_w0_img(v: torch.Tensor) -> torch.Tensor:
    """[B, 3, h, w] -> [B, 4, h, w] voi kenh w = 0."""
    w = v.new_zeros(v.shape[0], 1, v.shape[2], v.shape[3])
    return torch.cat([w, v], dim=1)


def _pad_w0_seq(v: torch.Tensor) -> torch.Tensor:
    """[B, L, 3] -> [B, L, 4] voi w = 0."""
    w = v.new_zeros(v.shape[0], v.shape[1], 1)
    return torch.cat([w, v], dim=-1)


class ImageHead(_GatedHead):
    def __init__(self, cfg: dict, layout: TokenLayout):
        super().__init__()
        d_model = int(cfg["model"]["d_model"])
        self.p = layout.patch
        self.levels = layout.levels_image
        self.shapes = layout.image_shapes
        # skip: noi HE SO QWT THO (da chuan hoa) cua chinh dai do vao dau vao head.
        # Chi tiet min khong phai song sot qua 12 lop attention toan cuc nua.
        # PHAI la he so tho chu KHONG phai dau ra tokenizer: tokenizer la
        # Linear(3*p*p -> d_model) = nen 768 xuong 384, da mat mot nua thong tin,
        # nen noi no vao thi head van khong the dung lai anh dau vao.
        # Voi he so tho thi nghiem tam thuong la "copy anh nhieu" (~21.99 dB tren du
        # lieu Kaggle) - cao hon han cho model dang dung - va loss day tiep tu do len.
        self.skip = bool(cfg["model"].get("recon_skip", True)) and bool(
            cfg["model"].get("recon_from_full", True)
        )
        d_patch = 3 * self.p * self.p
        # gate: head xuat HE SO NHAN cho he so nhieu thay vi chi cong them.
        # Khu nhieu wavelet ve ban chat la CO he so: w_hat = w * g(w, ngu canh) - mot
        # phep NHAN phu thuoc du lieu. Mot lop Linear chi cong duoc, khong nhan duoc,
        # nen khong bieu dien noi phep co. Va vi self.proj dung chung cho ca 10 dai
        # anh, duong di cua he so tho la MOT ma tran duy nhat - khong the cho L1 HH
        # he so 0.3 con L3 LL he so 1.05.
        # Do tran PSNR tren du lieu that (24 mau valid):
        #   copy nguyen anh nhieu                 19.51 dB
        #   1 he so nhan / moi dai                20.66 dB  (+1.15)
        #   1 he so nhan / moi TOKEN              21.82 dB  (+2.31)
        # KHONG dung sigmoid: 44% sai so nam o dai L3 LL la thieu sang, can cho phep
        # he so LON HON 1 de tang sang.
        self.gate = bool(cfg["model"].get("recon_gate", True)) and self.skip
        self.proj = nn.Linear(
            d_model + (d_patch if self.skip else 0), d_patch * (2 if self.gate else 1)
        )
        if self.gate:
            # khoi tao 0 => gate=0, delta=0 => dau ra DUNG BANG anh nhieu.
            # Model bat dau ngay tai san va chi co the di len.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        # Doi xung voi Tokenizer: head du doan he so DA CHUAN HOA roi nhan lai scale.
        self.register_buffer("band_scale", torch.ones(len(layout.image_entries)))
        self.gate_abs = 0.0          # chi de theo doi, khong tham gia tinh toan
        self.gate_max = float(cfg["model"].get("recon_gate_max", 1.0))

    def forward(
        self,
        emb_full: torch.Tensor,
        layout: TokenLayout,
        bands_out: dict | None = None,
        skip_qwt: dict | None = None,
    ) -> torch.Tensor:
        """`bands_out` (tuy chon): nhan he so DA CHUAN HOA cua tung dai, de tinh
        loss can bang giua cac dai (xem losses.band_balanced_loss).
        `skip_qwt` (tuy chon): thap QWT cua dau vao NHIEU, noi vao lam skip."""
        pyr: dict = {"levels": self.levels}
        for i, e in enumerate(layout.image_entries):
            tok = emb_full[:, e.start:e.end]                      # [B, gh*gw, d]
            raw_p = None
            if skip_qwt is not None:
                raw = skip_qwt[e.level][e.band][:, 1:4] / self.band_scale[i]
                raw_p = patchify(raw, e.gh, e.gw, self.p)
                tok = torch.cat([tok, raw_p], dim=-1)
            norm = self.proj(tok)                                 # he so da chuan hoa
            if self.gate:
                g, d = norm.chunk(2, dim=-1)
                # Theo doi bien do gate: proj khoi tao bang 0 nen luc dau gate = 0
                # (dau ra = anh vao). Neu sau nhieu epoch gate VAN ~0 thi head chua
                # roi diem xuat phat - la van de toc do hoc, khong phai hoc sai.
                self.gate_abs = float(g.detach().abs().mean())
                norm = raw_p * self._gain(g) + d
            if bands_out is not None:
                bands_out[(e.level, e.band)] = norm
            band = norm * self.band_scale[i]                      # [B, gh*gw, 3p^2]
            band = unpatchify(band, e.gh, e.gw, self.p)           # [B, 3, h, w]
            pyr.setdefault(e.level, {})[e.band] = _pad_w0_img(band)
        q = image_iqwt(pyr, shapes=self.shapes)                   # [B, 4, H, W]
        return q[:, 1:4]                                          # [B, 3, H, W]


class ImuHead(_GatedHead):
    def __init__(self, cfg: dict, layout: TokenLayout):
        super().__init__()
        d_model = int(cfg["model"]["d_model"])
        self.levels = layout.levels_imu
        self.lens = layout.imu_shapes
        self.skip = bool(cfg["model"].get("recon_skip", True)) and bool(
            cfg["model"].get("recon_from_full", True)
        )
        self.gate = bool(cfg["model"].get("recon_gate", True)) and self.skip
        self.proj = nn.Linear(d_model + (3 if self.skip else 0), 6 if self.gate else 3)
        if self.gate:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.register_buffer("band_scale", torch.ones(len(layout.imu_entries)))
        self.gate_abs = 0.0          # chi de theo doi, khong tham gia tinh toan
        self.gate_max = float(cfg["model"].get("recon_gate_max", 1.0))

    def forward(
        self,
        emb_full: torch.Tensor,
        layout: TokenLayout,
        bands_out: dict | None = None,
        skip_qwt: dict | None = None,
    ) -> torch.Tensor:
        coeffs: dict = {
            "acc": {"levels": self.levels},
            "gyro": {"levels": self.levels},
        }
        for i, e in enumerate(layout.imu_entries):
            tok = emb_full[:, e.start:e.end]              # [B, L, d]
            raw = None
            if skip_qwt is not None:
                raw = skip_qwt[e.group][e.level][e.band][:, :, 1:4] / self.band_scale[i]
                tok = torch.cat([tok, raw], dim=-1)
            norm = self.proj(tok)                         # he so da chuan hoa
            if self.gate:
                g, d = norm.chunk(2, dim=-1)
                self.gate_abs = float(g.detach().abs().mean())
                norm = raw * self._gain(g) + d
            if bands_out is not None:
                bands_out[(e.group, e.level, e.band)] = norm
            v = _pad_w0_seq(norm * self.band_scale[i])    # [B, L, 4]
            coeffs[e.group].setdefault(e.level, {})[e.band] = v
        return imu_iqwt(coeffs, lens=self.lens)           # [B, T, 6]
