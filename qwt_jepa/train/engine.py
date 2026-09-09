"""1 epoch train / eval (Phase F - Buoc F3)."""

from __future__ import annotations

import math
import time

import torch

from . import ema_momentum
from .losses import total_loss


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _autocast(cfg: dict, device: torch.device):
    amp = str(cfg["train"].get("amp", "none")).lower()
    if device.type == "cuda" and amp in ("bf16", "bfloat16"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and amp in ("fp16", "float16"):
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def make_scaler(cfg: dict, device: torch.device):
    """GradScaler CHI can cho fp16.

    fp16 co dai so mu rat hep: gradient nho hon ~6e-5 bi lam tron ve 0 (underflow) va
    lop do ngung hoc am tham - khong bao loi gi. GradScaler nhan loss len truoc khi
    backward roi chia lai, keo gradient ve vung bieu dien duoc.
    bf16 co dai so mu bang fp32 nen KHONG can scaler.
    """
    on = device.type == "cuda" and str(cfg["train"].get("amp", "none")).lower() in ("fp16", "float16")
    return torch.amp.GradScaler("cuda", enabled=on)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    cfg: dict,
    device: torch.device,
    step: int,
    total_steps: int,
    epoch: int,
    scaler=None,
) -> int:
    model.train()
    lam = cfg["train"]["loss"]
    ema_cfg = cfg["train"]["ema"]
    grad_clip = float(cfg["train"].get("grad_clip", 0) or 0)
    log_every = int(cfg["train"].get("log_every", 50))

    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = _to_device(batch, device)

        with _autocast(cfg, device):
            out = model(batch)
            loss, logs = total_loss(out, batch, lam["lambda_img"], lam["lambda_imu"])

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)     # phai go scale TRUOC khi clip theo norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        m = ema_momentum(step, total_steps, ema_cfg["start"], ema_cfg["end"])
        model.ema_update(m)
        step += 1

        if i % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            ips = (i + 1) * loader.batch_size / (time.time() - t0)
            print(
                f"e{epoch} {i:4d}/{len(loader)} step {step:6d} | "
                f"L {logs['L_total']:.4f} (jepa {logs['L_jepa']:.4f} "
                f"img {logs['L_img']:.4f} imu {logs['L_imu']:.4f}) | "
                f"z_std {logs['ztgt_std']:.3f} | lr {lr:.2e} m {m:.4f} | {ips:.1f} im/s",
                flush=True,
            )
    return step


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: torch.device, max_batches: int = 0) -> dict:
    model.eval()
    lam = cfg["train"]["loss"]
    n = 0
    agg = {"L_jepa": 0.0, "psnr": 0.0, "rmse_acc": 0.0, "rmse_gyro": 0.0}

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = _to_device(batch, device)
        with _autocast(cfg, device):
            out = model(batch)
        _, logs = total_loss(out, batch, lam["lambda_img"], lam["lambda_imu"])

        b = batch["img_clean"].shape[0]
        mse = torch.mean((out["img_rec"].float() - batch["img_clean"].float()) ** 2).item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))
        rmse_acc = torch.sqrt(
            torch.mean((out["imu_rec"][..., 0:3].float() - batch["imu_clean"][..., 0:3].float()) ** 2)
        ).item()
        rmse_gyro = torch.sqrt(
            torch.mean((out["imu_rec"][..., 3:6].float() - batch["imu_clean"][..., 3:6].float()) ** 2)
        ).item()

        agg["L_jepa"] += logs["L_jepa"] * b
        agg["psnr"] += psnr * b
        agg["rmse_acc"] += rmse_acc * b
        agg["rmse_gyro"] += rmse_gyro * b
        n += b

    return {k: v / max(n, 1) for k, v in agg.items()}
