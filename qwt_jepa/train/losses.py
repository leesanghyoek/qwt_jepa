"""Loss cot loi + recon + chi so chong collapse (Phase F - Buoc F1/F2, phan model-facing).

    L_jepa  = smooth_l1(z_pred, stop_grad(z_tgt))
    L_img   = charbonnier(img_rec, img_clean)
    L_imu   = l1(acc) + l1(gyro)
    L_var   = hinge(gamma - std(z_ctx))            <- chong collapse
    L_total = L_jepa + lambda_img * L_img + lambda_imu * L_imu + lambda_var * L_var

Khong negative, khong contrastive. L_var la rang buoc mot chieu kieu VICReg: chi
phat khi bieu dien BOT da dang, khong ep gi khi da du da dang.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def jepa_loss(z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(z_pred, z_tgt.detach())


@torch.no_grad()
def sharpness(x: torch.Tensor) -> torch.Tensor:
    """Bien do gradient trung binh - thuoc do DO NET. x: [B, C, H, W]."""
    return ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
            + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean())


@torch.no_grad()
def content_std(z: torch.Tensor) -> torch.Tensor:
    """Do lech chuan theo BATCH tai TUNG vi tri token, roi trung binh. z: [B, T, D].

    Do dung cai ta can: "doi anh dau vao thi bieu dien doi bao nhieu".

    KHONG dung std gop ca (batch, token). Do la cai bay: bieu dien co the dat
    std gop ~1.0 chi bang cach cho cac VI TRI token khac nhau, trong khi doi anh
    thi gan nhu khong doi gi. Do thuc te o trang thai do: std gop 0.995 nhung std
    theo noi dung chi 0.131 - va predictor bi danh bai boi mot baseline chi doan
    theo vi tri token, khong nhin anh (0.0092 so voi 0.0134).
    """
    if z.shape[0] < 2:
        return z.new_zeros(())
    return torch.sqrt(z.float().var(dim=0) + 1e-4).mean()


def variance_loss(z: torch.Tensor, gamma: float = 0.5, eps: float = 1e-4) -> torch.Tensor:
    """Phat khi do lech chuan THEO NOI DUNG tut duoi `gamma`. z: [B, T, D].

    Luc doi khang truc tiep voi ca hai kieu collapse:
      - collapse hoan toan : moi token ve cung mot vector
      - collapse vi tri    : bieu dien chi con phu thuoc vi tri token, doi anh
                             khong doi gi -> L_jepa ve 0 ma khong hoc duoc gi

    Chi ap cho z_ctx (dau ra context_encoder, da qua LayerNorm nen scale bi chan).
    KHONG ap cho z_pred: predictor ket thuc bang Linear khong chuan hoa, model se
    "lach" bang cach phong to scale dau ra thay vi tang do da dang that su.
    """
    if z.shape[0] < 2:                       # can it nhat 2 mau de co phuong sai
        return z.new_zeros(())
    std = torch.sqrt(z.float().var(dim=0) + eps)      # [T, D] - std theo BATCH
    return F.relu(gamma - std).mean()


@torch.no_grad()
def jepa_loss_position_only(z_tgt: torch.Tensor) -> torch.Tensor:
    """L_jepa ma mot ke gian lan dat duoc khi CHI doan theo vi tri token.

    Lay trung binh z_tgt theo batch tai tung vi tri roi dung chinh no lam du doan
    - tuc la bo qua hoan toan anh dau vao. Model that PHAI thap hon dang ke con so
    nay; neu khong, JEPA khong hoc duoc gi ve noi dung.
    """
    if z_tgt.shape[0] < 2:
        return z_tgt.new_zeros(())
    return F.smooth_l1_loss(z_tgt.mean(dim=0, keepdim=True).expand_as(z_tgt), z_tgt)


def band_balanced_loss(pred: dict, tgt: dict) -> torch.Tensor:
    """Charbonnier tren he so QWT DA CHUAN HOA, MOI DAI TRONG SO NGANG NHAU.

    Vi sao can: L_img tinh tren mien pixel, ma sai so o dai L3 LL gay loi pixel lon
    gap ~100 lan sai so o dai L1. Do that tren tap valid:

        dai        token   % token   % nang luong
        L3 LL          4      1.6%        94.10%
        L1 LH+HL+HH  192     75.0%         1.03%

    Nen duoi L_img mot minh, "lam mo" la nghiem TOI UU - model chi can dung cai anh
    thu nho 32x32 la da duoc ~19.7 dB. Loss nay chia deu trong so cho tung dai nen
    chi tiet min moi co gradient dang ke.
    """
    if not pred:
        return torch.zeros(())
    terms = [charbonnier(pred[k], tgt[k].to(pred[k].dtype)) for k in pred]
    return torch.stack(terms).mean()


def imu_recon_loss(rec: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    l_acc = F.l1_loss(rec[..., 0:3], clean[..., 0:3])
    l_gyro = F.l1_loss(rec[..., 3:6], clean[..., 3:6])
    return l_acc + l_gyro


def total_loss(
    out: dict,
    batch: dict,
    lambda_jepa: float = 1.0,
    lambda_img: float = 1.0,
    lambda_imu: float = 1.0,
    lambda_var: float = 0.0,
    var_gamma: float = 1.0,
    lambda_band: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    l_jepa = jepa_loss(out["z_pred"], out["z_tgt"])
    l_img = charbonnier(out["img_rec"], batch["img_clean"])
    l_imu = imu_recon_loss(out["imu_rec"], batch["imu_clean"])
    total = lambda_jepa * l_jepa + lambda_img * l_img + lambda_imu * l_imu

    l_band = torch.zeros((), device=l_jepa.device)
    if lambda_band > 0 and "img_bands" in out:
        l_band = (
            band_balanced_loss(out["img_bands"], out["img_bands_tgt"])
            + band_balanced_loss(out["imu_bands"], out["imu_bands_tgt"])
        )
        total = total + lambda_band * l_band

    l_var = torch.zeros((), device=l_jepa.device)
    if lambda_var > 0 and "z_ctx" in out:
        l_var = variance_loss(out["z_ctx"], var_gamma)
        total = total + lambda_var * l_var

    logs = {
        "L_total": float(total.detach()),
        "L_jepa": float(l_jepa.detach()),
        "L_img": float(l_img.detach()),
        "L_imu": float(l_imu.detach()),
        "L_band": float(l_band.detach()),
        "gate": float(out.get("gate_abs", 0.0)),
        "L_var": float(l_var.detach()),
        # std THEO NOI DUNG (doi anh), khong phai std gop ca (batch, token)
        "ztgt_std": float(content_std(out["z_tgt"])),
        "zctx_std": float(content_std(out["z_ctx"])) if "z_ctx" in out else 0.0,
        # nguong gian lan: L_jepa dat duoc khi chi doan theo vi tri token
        "L_jepa_pos": float(jepa_loss_position_only(out["z_tgt"])),
    }
    return total, logs


@torch.no_grad()
def ztgt_std(z_tgt: torch.Tensor) -> float:
    """std cua z_tgt theo (batch, token) tren tung chieu, roi lay trung binh.

    Tut ve ~0 la dau hieu representation collapse.
    """
    return float(z_tgt.reshape(-1, z_tgt.shape[-1]).std(dim=0).mean())
