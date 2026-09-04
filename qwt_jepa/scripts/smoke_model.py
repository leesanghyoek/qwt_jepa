"""Smoke test cho QwtJepa: forward 1 batch ngau nhien, in shape, backward, EMA.

Chay:
    python -m qwt_jepa.scripts.smoke_model
hoac:
    python qwt_jepa/scripts/smoke_model.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.models.jepa import QwtJepa                       # noqa: E402
from qwt_jepa.train import ema_momentum                        # noqa: E402
from qwt_jepa.train.losses import total_loss                   # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg_path = _ROOT / "qwt_jepa" / "configs" / "base.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    # Cho nhe de chay CPU nhanh; bo 2 dong duoi de dung dung config.
    cfg["model"]["encoder"]["depth"] = 4

    model = QwtJepa(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"tham so train duoc : {n_params/1e6:.2f} M")
    print(f"so token N         : {model.layout.n_tokens}")
    print(f"so scale/subband   : {model.layout.num_scales}")

    b = 2
    hw = cfg["data"]["image"]["train_crop"][0]
    t = cfg["data"]["imu"]["window_size"]
    batch = {
        "img_noisy": torch.rand(b, 3, hw, hw),
        "img_clean": torch.rand(b, 3, hw, hw),
        "imu_noisy": torch.randn(b, t, 6),
        "imu_clean": torch.randn(b, t, 6),
    }

    out = model(batch)
    print("\n--- shape dau ra ---")
    print("z_pred  :", tuple(out["z_pred"].shape))
    print("z_tgt   :", tuple(out["z_tgt"].shape))
    print("img_rec :", tuple(out["img_rec"].shape))
    print("imu_rec :", tuple(out["imu_rec"].shape))
    assert out["z_pred"].shape == out["z_tgt"].shape
    assert out["img_rec"].shape == batch["img_clean"].shape
    assert out["imu_rec"].shape == batch["imu_clean"].shape

    lam = cfg["train"]["loss"]
    loss, logs = total_loss(out, batch, lam["lambda_img"], lam["lambda_imu"])
    print("\n--- loss ---")
    for k, v in logs.items():
        print(f"{k:10s}: {v:.5f}")

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"grad_norm : {float(grad_norm):.4f}")

    # target_encoder khong duoc co gradient
    assert all(p.grad is None for p in model.target_encoder.parameters())

    model.ema_update(ema_momentum(step=0, total_steps=100000))

    for name, tensor in out.items():
        if torch.is_tensor(tensor):
            assert torch.isfinite(tensor).all(), f"{name} co NaN/Inf"
    assert torch.isfinite(loss).all()

    print("\nOK - forward / loss / backward / EMA chay khong loi.")


if __name__ == "__main__":
    main()
