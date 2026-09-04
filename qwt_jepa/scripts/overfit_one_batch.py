"""Phase G - overfit 1 batch: train ~200 step tren dung 1 batch de bat loi
mask / loss / iQWT / EMA. L_total phai giam manh, std(z_tgt) khong sup ve 0.

Chay:
    python -m qwt_jepa.scripts.overfit_one_batch --steps 200
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.models.jepa import QwtJepa                       # noqa: E402
from qwt_jepa.train.losses import total_loss                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--depth", type=int, default=4, help="ha depth encoder cho nhanh")
    ap.add_argument("--mask-seed", type=int, default=123, help="co dinh mask -> overfit sach")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = yaml.safe_load((_ROOT / "qwt_jepa" / "configs" / "base.yaml").read_text())
    if args.depth:
        cfg["model"]["encoder"]["depth"] = args.depth
    cfg["masking"]["context_keep_ratio"] = 1.0   # thay het context de recon day du

    model = QwtJepa(cfg)
    hw = cfg["data"]["image"]["train_crop"][0]
    t = cfg["data"]["imu"]["window_size"]
    b = args.batch
    batch = {
        "img_noisy": torch.rand(b, 3, hw, hw),
        "img_clean": torch.rand(b, 3, hw, hw),
        "imu_noisy": torch.randn(b, t, 6),
        "imu_clean": torch.randn(b, t, 6),
    }

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lam = cfg["train"]["loss"]

    for step in range(args.steps):
        gen = torch.Generator().manual_seed(args.mask_seed) if args.mask_seed is not None else None
        out = model(batch, generator=gen)
        loss, logs = total_loss(out, batch, lam["lambda_img"], lam["lambda_imu"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.ema_update(0.996)
        if step % 20 == 0 or step == args.steps - 1:
            print(
                f"step {step:3d}  L_total {logs['L_total']:.4f}  "
                f"L_jepa {logs['L_jepa']:.4f}  L_img {logs['L_img']:.4f}  "
                f"L_imu {logs['L_imu']:.4f}  z_std {logs['ztgt_std']:.3f}"
            )


if __name__ == "__main__":
    main()
