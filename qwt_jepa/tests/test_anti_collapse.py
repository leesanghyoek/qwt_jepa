"""Chot lai hai bat bien da tung lam model collapse (xem LOG_TRAIN_GIAI_THICH.md).

1. Nhanh target phai duoc CACH LY hoan toan khoi optimizer. Truoc day
   `self.tokenizer` dung chung cho ca hai nhanh: du nhanh target chay trong
   torch.no_grad(), optimizer van cap nhat tokenizer tu nhanh context moi buoc,
   nen z_tgt dich chuyen theo huong optimizer muon -> z_std tut -> L_jepa ve 0
   ma khong hoc duoc gi.

2. Dai THO (anh LL, imu A) phai luon nam trong context. Chi 36/512 token nhung
   mang gan het do sang / bo cuc; de chung bi vut thi tran PSNR tut tu ~25 dB
   xuong ~17.5 dB, tuc THAP HON ca anh nhieu dau vao (~20.9 dB).
"""

from __future__ import annotations

import copy
import pathlib

import torch
import yaml

from ..models.jepa import QwtJepa
from ..models.layout import build_layout
from ..train.losses import total_loss, variance_loss
from ..train.masking import coarse_tokens, sample_masks

_CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def _cfg() -> dict:
    cfg = yaml.safe_load(_CFG.read_text())
    cfg["model"]["encoder"]["depth"] = 2        # nho lai cho test chay nhanh
    cfg["model"]["predictor"]["depth"] = 1
    return cfg


def _batch(cfg: dict, b: int = 2) -> dict:
    h = int(cfg["data"]["image"]["train_crop"][0])
    t = int(cfg["data"]["imu"]["window_size"])
    c = int(cfg["data"]["imu"]["channels"])
    g = torch.Generator().manual_seed(0)
    return {
        "img_clean": torch.rand(b, 3, h, h, generator=g),
        "img_noisy": torch.rand(b, 3, h, h, generator=g),
        "imu_clean": torch.randn(b, t, c, generator=g),
        "imu_noisy": torch.randn(b, t, c, generator=g),
    }


def test_target_branch_frozen_and_ema_only():
    """target_tokenizer: khong co gradient, khong doi sau optimizer.step(), chi doi qua EMA."""
    cfg = _cfg()
    model = QwtJepa(cfg)

    assert model.target_tokenizer is not model.tokenizer, "target_tokenizer phai la ban RIENG"
    assert not any(p.requires_grad for p in model.target_tokenizer.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())

    before = copy.deepcopy(model.target_tokenizer.image_proj.weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    loss, _ = total_loss(model(_batch(cfg)), _batch(cfg), 1.0, 1.0, 1.0, 1.0)
    loss.backward()
    assert all(p.grad is None for p in model.target_tokenizer.parameters()), \
        "gradient khong duoc chay vao nhanh target"

    opt.step()
    assert torch.equal(before, model.target_tokenizer.image_proj.weight), \
        "optimizer.step() khong duoc dong vao nhanh target"

    model.ema_update(0.9)
    assert not torch.equal(before, model.target_tokenizer.image_proj.weight), \
        "ema_update() phai cap nhat CA tokenizer, khong chi encoder"


def test_coarse_bands_always_in_context():
    cfg = _cfg()
    layout = build_layout(cfg)
    protect = coarse_tokens(layout)
    assert protect, "phai tim thay token dai tho (anh LL + imu A)"

    for _ in range(200):
        mask = sample_masks(layout, cfg, None)
        kept = set(mask.context_index.tolist())
        target = set(mask.target_index.tolist())
        assert protect <= kept, "dai tho bi vut khoi context"
        assert not (protect & target), "dai tho khong duoc lam target"


def test_variance_loss_phat_dung_chieu():
    z_deu = torch.randn(64, 32, 16)                  # da da dang -> gan nhu khong bi phat
    z_sap = torch.zeros(64, 32, 16) + torch.randn(1, 1, 16) * 0.01   # gan nhu hang so
    assert variance_loss(z_deu, gamma=1.0) < 0.1
    assert variance_loss(z_sap, gamma=1.0) > 0.9
