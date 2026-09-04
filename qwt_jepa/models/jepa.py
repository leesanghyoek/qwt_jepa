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
        self.target_encoder = copy.deepcopy(self.context_encoder)
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
            tok_t = self.tokenizer(q_img_c, q_imu_c)
            z_all = self.target_encoder(tok_t)
            z_tgt = z_all[:, mask.target_index]

        # ---- reconstruction ----
        emb_full = self._assemble_full(z_ctx, z_pred, mask, b)
        img_rec = self.image_head(emb_full, self.layout)             # [B, 3, H, W]
        imu_rec = self.imu_head(emb_full, self.layout)               # [B, T, 6]

        return {
            "z_pred": z_pred,
            "z_tgt": z_tgt,
            "img_rec": img_rec,
            "imu_rec": imu_rec,
            "mask": mask,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def ema_update(self, m: float) -> None:
        """Cap nhat target_encoder = m * target + (1 - m) * context. Goi sau optimizer.step()."""
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)
        for bt, bc in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            bt.copy_(bc)
