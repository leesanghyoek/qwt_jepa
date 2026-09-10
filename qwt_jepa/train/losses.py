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


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Phat khi do lech chuan theo TUNG CHIEU (tinh qua batch va token) tut duoi `gamma`.

    Day la luc doi khang truc tiep voi representation collapse: khong the lam
    L_jepa -> 0 bang cach cho encoder xuat gan nhu cung mot vector nua.

    Chi ap cho z_ctx (dau ra context_encoder, da qua LayerNorm nen scale bi chan).
    KHONG ap cho z_pred: predictor ket thuc bang Linear khong chuan hoa, model se
    "lach" bang cach phong to scale dau ra thay vi tang do da dang that su.
    """
    z = z.float().reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def imu_recon_loss(rec: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    l_acc = F.l1_loss(rec[..., 0:3], clean[..., 0:3])
    l_gyro = F.l1_loss(rec[..., 3:6], clean[..., 3:6])
    return l_acc + l_gyro


def total_loss(
    out: dict,
    batch: dict,
    lambda_img: float = 1.0,
    lambda_imu: float = 1.0,
    lambda_var: float = 0.0,
    var_gamma: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    l_jepa = jepa_loss(out["z_pred"], out["z_tgt"])
    l_img = charbonnier(out["img_rec"], batch["img_clean"])
    l_imu = imu_recon_loss(out["imu_rec"], batch["imu_clean"])
    total = l_jepa + lambda_img * l_img + lambda_imu * l_imu

    l_var = torch.zeros((), device=l_jepa.device)
    if lambda_var > 0 and "z_ctx" in out:
        l_var = variance_loss(out["z_ctx"], var_gamma)
        total = total + lambda_var * l_var

    logs = {
        "L_total": float(total.detach()),
        "L_jepa": float(l_jepa.detach()),
        "L_img": float(l_img.detach()),
        "L_imu": float(l_imu.detach()),
        "L_var": float(l_var.detach()),
        "ztgt_std": ztgt_std(out["z_tgt"]),
        "zctx_std": ztgt_std(out["z_ctx"]) if "z_ctx" in out else 0.0,
    }
    return total, logs


@torch.no_grad()
def ztgt_std(z_tgt: torch.Tensor) -> float:
    """std cua z_tgt theo (batch, token) tren tung chieu, roi lay trung binh.

    Tut ve ~0 la dau hieu representation collapse.
    """
    return float(z_tgt.reshape(-1, z_tgt.shape[-1]).std(dim=0).mean())
