"""Ghep noi: context / target / EMA / predictor / heads (Phase D - Buoc D4).

    nhanh context : du lieu NOISY  -> QWT -> tokens -> context_encoder -> predictor -> z_pred
    nhanh target  : du lieu CLEAN  -> QWT -> tokens -> target_encoder (EMA, stop-grad) -> z_tgt
    reconstruction: emb tai moi token (z_ctx cho token thay, z_pred cho token mask)
                    -> 2 head -> iQWT -> anh sach / IMU sach du doan
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..qwt.image_qwt import image_qwt, rgb_to_quat
from ..qwt.imu_qwt import imu_qwt
from ..train.masking import MaskIndices, sample_masks
from .encoder import Encoder
from .layout import build_layout
from .predictor import Predictor
from .recon_head import ImageHead, ImuHead
from .tokenizer import Tokenizer


class QwtJepa(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.layout = build_layout(cfg)

        d_model = int(cfg["model"]["d_model"])
        enc = cfg["model"]["encoder"]

        self.tokenizer = Tokenizer(cfg, self.layout)
        self.context_encoder = Encoder(d_model, int(enc["depth"]), int(enc["heads"]))

        # Nhanh target phai la ban EMA HOAN CHINH: tokenizer + encoder.
        # Neu dung chung `self.tokenizer` cho ca hai nhanh thi du co torch.no_grad()
        # o nhanh target, optimizer van cap nhat tokenizer tu nhanh context moi buoc
        # -> z_tgt dich chuyen theo huong optimizer muon -> collapse (z_std tut dan).
        self.target_tokenizer = copy.deepcopy(self.tokenizer)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_tokenizer.parameters():
            p.requires_grad_(False)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(cfg, self.layout.n_tokens)
        self.image_head = ImageHead(cfg, self.layout)
        self.imu_head = ImuHead(cfg, self.layout)

        # Token khong nhin thay va khong nam trong target -> dung hang so hoc duoc.
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.missing_token, std=0.02)

        self.levels_image = int(cfg["data"]["image"]["qwt_levels"])
        self.levels_imu = int(cfg["data"]["imu"]["qwt_levels"])
        self.recon_from_full = bool(cfg["model"].get("recon_from_full", True))
        self.recon_skip = bool(cfg["model"].get("recon_skip", True)) and self.recon_from_full

    # ------------------------------------------------------------------ #
    def _heads(self, emb, q_img, q_imu, img_bands=None, imu_bands=None):
        """Chay hai recon head. Thap QWT NHIEU duoc noi vao lam skip connection."""
        si, su = (q_img, q_imu) if self.recon_skip else (None, None)
        return (
            self.image_head(emb, self.layout, img_bands, si),
            self.imu_head(emb, self.layout, imu_bands, su),
        )

    def reconstruct(self, img_noisy: torch.Tensor, imu_noisy: torch.Tensor):
        """DUONG TRIEN KHAI: toan bo token nhieu -> encoder -> 2 head.

        Khong che gi, khong dung predictor, khong dung missing_token. Day la duong
        DUY NHAT nen dung khi danh gia hoac trien khai - dung tu ghep lai, vi no
        phai khop chinh xac voi nhanh tai tao luc train (ke ca skip connection).
        """
        q_img, q_imu = self._qwt_all(img_noisy, imu_noisy)
        tok = self.tokenizer(q_img, q_imu)
        return self._heads(self.context_encoder(tok), q_img, q_imu)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _clean_bands_img(self, q_img: dict) -> dict:
        """He so anh SACH, chuan hoa y het duong du doan -> lam target."""
        from .tokenizer import patchify

        out = {}
        for i, e in enumerate(self.layout.image_entries):
            band = q_img[e.level][e.band][:, 1:4] / self.image_head.band_scale[i]
            out[(e.level, e.band)] = patchify(band, e.gh, e.gw, self.layout.patch)
        return out

    @torch.no_grad()
    def _clean_bands_imu(self, q_imu: dict) -> dict:
        out = {}
        for i, e in enumerate(self.layout.imu_entries):
            band = q_imu[e.group][e.level][e.band][:, :, 1:4] / self.imu_head.band_scale[i]
            out[(e.group, e.level, e.band)] = band
        return out

    @torch.no_grad()
    def set_band_scales(self, scales: dict) -> None:
        """Nap bien do tung dai QWT vao tokenizer + 2 head (xem data/band_stats.py).

        Phai goi TRUOC khi load_state_dict luc resume: cac gia tri nay la buffer nen
        di theo checkpoint, ban trong checkpoint moi la ban dung.
        """
        from ..data.band_stats import scales_to_tensors

        img, imu = scales_to_tensors(scales, self.layout)
        for mod, buf, val in (
            (self.tokenizer, "img_band_scale", img),
            (self.tokenizer, "imu_band_scale", imu),
            (self.target_tokenizer, "img_band_scale", img),
            (self.target_tokenizer, "imu_band_scale", imu),
            (self.image_head, "band_scale", img),
            (self.imu_head, "band_scale", imu),
        ):
            getattr(mod, buf).copy_(val.to(getattr(mod, buf).device))

    # ------------------------------------------------------------------ #
    def _qwt_all(self, img: torch.Tensor, imu: torch.Tensor) -> tuple[dict, dict]:
        q_img = image_qwt(rgb_to_quat(img), self.levels_image)
        q_imu = imu_qwt(imu, self.levels_imu)
        return q_img, q_imu

    def _assemble_full(
        self,
        z_ctx: torch.Tensor,
        z_pred: torch.Tensor,
        mask: MaskIndices,
        batch_size: int,
    ) -> torch.Tensor:
        d = z_ctx.shape[-1]
        emb = self.missing_token.to(z_ctx.dtype).expand(batch_size, self.layout.n_tokens, d).clone()
        emb[:, mask.context_index] = z_ctx.to(emb.dtype)
        emb[:, mask.target_index] = z_pred.to(emb.dtype)
        return emb

    # ------------------------------------------------------------------ #
    def forward(self, batch: dict, generator: torch.Generator | None = None) -> dict:
        img_n, imu_n = batch["img_noisy"], batch["imu_noisy"]
        img_c, imu_c = batch["img_clean"], batch["imu_clean"]
        b = img_n.shape[0]
        device = img_n.device

        # ---- mask (dung chung ca batch) ----
        mask = sample_masks(self.layout, self.cfg, generator)
        mask = MaskIndices(
            context_index=mask.context_index.to(device),
            target_index=mask.target_index.to(device),
            context_index_full=mask.context_index_full.to(device),
        )

        # ---- nhanh context: NOISY ----
        q_img_n, q_imu_n = self._qwt_all(img_n, imu_n)
        tok_c = self.tokenizer(q_img_n, q_imu_n)                      # [B, N, d]
        z_ctx = self.context_encoder(tok_c[:, mask.context_index])    # [B, n_ctx, d]
        z_pred = self.predictor(z_ctx, mask.context_index, mask.target_index)

        # ---- nhanh target: CLEAN, stop-grad ----
        with torch.no_grad():
            q_img_c, q_imu_c = self._qwt_all(img_c, imu_c)
            tok_t = self.target_tokenizer(q_img_c, q_imu_c)
            z_all = self.target_encoder(tok_t)
            z_tgt = z_all[:, mask.target_index]

        # ---- reconstruction ----
        # LUOT 2: cho encoder xem TOAN BO 512 token nhieu, khong che gi.
        # Day la CUNG mot self.context_encoder o tren, chi la goi lan thu hai voi dau
        # vao khac - khong phai model thu hai, so tham so khong doi.
        # Vi sao can luot rieng:
        #   - Duong cu cho head an emb_full = 30% token that + 41% token predictor doan
        #     + 30% hang so missing_token. Do that: predictor thua ca meo doan theo vi
        #     tri (ti so L_jepa/pos ~1.1) nen 41% do la rac -> head chi co 30% thong tin
        #     that de dung lai ca buc anh -> mo, va thua anh nhieu dau vao 4.3 dB.
        #   - context_encoder luc train chua bao gio thay du 512 token (luon ~152),
        #     nhung luc trien khai lai nap ca 512 -> lech train/test.
        # KHONG gop hai luot lam mot: attention toan cuc se lam token context nhin thay
        # token target -> L_jepa thanh gian lan.
        img_bands: dict = {}
        imu_bands: dict = {}
        if self.recon_from_full:
            emb_rec = self.context_encoder(tok_c)                    # [B, 512, d]
        else:
            emb_rec = self._assemble_full(z_ctx, z_pred, mask, b)    # duong cu
        img_rec, imu_rec = self._heads(emb_rec, q_img_n, q_imu_n, img_bands, imu_bands)
        with torch.no_grad():
            img_bands_tgt = self._clean_bands_img(q_img_c)
            imu_bands_tgt = self._clean_bands_imu(q_imu_c)

        return {
            "z_pred": z_pred,
            "z_tgt": z_tgt,
            "z_ctx": z_ctx,          # can cho variance_loss (chong collapse)
            "img_rec": img_rec,
            "imu_rec": imu_rec,
            # he so QWT DA CHUAN HOA - de tinh loss can bang giua cac dai. Loss anh
            # tren mien pixel bi dai L3 LL nuot (94% nang luong) nen mot minh no
            # khong bao gio day model tai tao chi tiet min -> anh ra bi mo.
            "img_bands": img_bands,
            "imu_bands": imu_bands,
            "img_bands_tgt": img_bands_tgt,
            "imu_bands_tgt": imu_bands_tgt,
            "mask": mask,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def ema_update(self, m: float) -> None:
        """target = m * target + (1 - m) * online. Goi sau optimizer.step().

        Cap nhat CA HAI cap (tokenizer, encoder) - bo sot tokenizer la duong tat
        dan thang toi representation collapse.
        """
        for tgt, src in (
            (self.target_tokenizer, self.tokenizer),
            (self.target_encoder, self.context_encoder),
        ):
            for pt, pc in zip(tgt.parameters(), src.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)
            for bt, bc in zip(tgt.buffers(), src.buffers()):
                bt.copy_(bc)
