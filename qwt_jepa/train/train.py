"""Entrypoint train (Phase F - Buoc F3).

    python -m qwt_jepa.train.train --config qwt_jepa/configs/base.yaml

Truoc do chay 1 lan:
    python -m qwt_jepa.scripts.compute_norm_stats --config qwt_jepa/configs/base.yaml
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Subset

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.data.band_stats import compute_band_scales                 # noqa: E402
from qwt_jepa.data.dataset import PairedNoisyCleanDataset, jepa_collate   # noqa: E402
from qwt_jepa.data.normalize import ImuNormalizer                        # noqa: E402
from qwt_jepa.models.jepa import QwtJepa                                  # noqa: E402
from qwt_jepa.train.engine import evaluate, make_scaler, train_one_epoch  # noqa: E402


def build_loaders(cfg: dict, normalizer: ImuNormalizer):
    tcfg = cfg["train"]
    root = cfg["data"]["root"]
    man = cfg["data"]["manifest"]

    train_ds = PairedNoisyCleanDataset(man["train"], root, cfg, "train", normalizer)
    val_ds = PairedNoisyCleanDataset(man["valid"], root, cfg, "valid", normalizer)

    # Manifest xep theo trajectory, nen shuffle=False + limit_val_batches lam eval chi
    # doc phan DAU tap valid: do that tren 7 env local, 3200 mau dau chi phu 5 env -
    # RetroOffice va WaterMillDay khong bao gio duoc danh gia. Tren Kaggle (14 env,
    # doc 25% tap valid) con lech nang hon. Ngoai ra moi batch la 32 khung LIEN TIEP
    # cung quy dao nen content_std va moc pos-only deu bi lech.
    # Xao mot lan bang hoan vi CO DINH: vua phu deu moi truong, vua giu nguyen thu tu
    # giua cac epoch de so sanh duoc.
    perm = torch.randperm(
        len(val_ds), generator=torch.Generator().manual_seed(int(tcfg.get("val_perm_seed", 0)))
    ).tolist()
    val_ds = Subset(val_ds, perm)

    common = dict(
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=jepa_collate,
        persistent_workers=int(tcfg.get("num_workers", 4)) > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False, drop_last=False, **common
    )
    return train_ds, train_loader, val_loader


def _improved(cur: float, best: float, mode: str, min_delta: float) -> bool:
    """cur co tot hon best qua nguong min_delta khong? (mode: 'min' hoac 'max')"""
    if mode == "max":
        return cur > best + min_delta
    return cur < best - min_delta


def build_optimizer(model, cfg: dict):
    """AdamW voi nhom rieng cho 2 recon head.

    Head duoc khoi tao BANG 0 (de dau ra bat dau dung bang anh vao) nen no phai bo len
    tu con so khong, trong khi phan con lai cua model khoi tao ngau nhien va chi can
    tinh chinh. Voi cung mot lr va limit_train_batches 300 thi sau 7 epoch (2100 buoc)
    head gan nhu chua roi diem xuat phat - quan sat duoc: L_img dung o 0.045 dung bang
    moc "copy dau vao" 0.042, va RMSE imu dung im o chu so thu tu qua 7 epoch.
    head_lr_mult cho head hoc nhanh hon phan con lai. Theo doi cot `gate` trong log:
    van ~0 sau vai epoch = van chua du nhanh.
    """
    ocfg = cfg["train"]["optimizer"]
    lr = float(ocfg["lr"])
    mult = float(ocfg.get("head_lr_mult", 1.0))
    head, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_recon = n.startswith(("image_head.", "imu_head.", "recon_in.", "recon_dec.", "recon_out."))
        (head if is_recon else rest).append(p)
    groups = [{"params": rest, "lr": lr}]
    if head:
        groups.append({"params": head, "lr": lr * mult})
    return torch.optim.AdamW(
        groups, lr=lr, weight_decay=float(ocfg["weight_decay"]), betas=(0.9, 0.95)
    ), mult, len(head)


def build_scheduler(optimizer, cfg: dict, total_steps: int):
    warmup = int(cfg["train"]["optimizer"].get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_ROOT / "qwt_jepa" / "configs" / "base.yaml"))
    ap.add_argument("--out", default=str(_ROOT / "qwt_jepa" / "runs" / "base"))
    ap.add_argument("--resume", default="")
    ap.add_argument("--freeze-backbone", type=int, default=-1,
                    help="GIAI DOAN 2: dong bang tokenizer+encoder+predictor, chi hoc 2 "
                         "recon head. -1 = dung train.freeze_backbone trong config.")
    ap.add_argument("--epochs", type=int, default=0, help="ghi de train.epochs (0 = dung config)")
    ap.add_argument("--batch-size", type=int, default=0, help="ghi de train.batch_size")
    ap.add_argument("--num-workers", type=int, default=-1, help="ghi de train.num_workers")
    ap.add_argument("--encoder-depth", type=int, default=0, help="ghi de model.encoder.depth")
    ap.add_argument("--limit-train-batches", type=int, default=0, help="debug: cat ngan 1 epoch")
    ap.add_argument("--limit-val-batches", type=int, default=0,
                    help="debug: cat ngan vong eval (0 = dung train.limit_val_batches)")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers >= 0:
        cfg["train"]["num_workers"] = args.num_workers
    if args.encoder_depth:
        cfg["model"]["encoder"]["depth"] = args.encoder_depth
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    norm_path = pathlib.Path(cfg["data"]["norm_stats"])
    if norm_path.exists():
        normalizer = ImuNormalizer.from_yaml(norm_path)
        print(f"norm stats: {norm_path}")
    else:
        normalizer = ImuNormalizer.identity()
        print(f"[canh bao] chua co {norm_path} -> dung identity. "
              f"Chay compute_norm_stats truoc.")

    train_ds, train_loader, val_loader = build_loaders(cfg, normalizer)
    print(f"train {len(train_ds)} sample | {len(train_loader)} batch/epoch | device {device}")

    model = QwtJepa(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"QwtJepa: {n_par/1e6:.2f}M params train | N tokens {model.layout.n_tokens}")

    # Bien do he so QWT chenh ~100 lan giua cac dai -> phai chuan hoa, neu khong
    # encoder gan nhu khong nhin thay chi tiet min va anh tai tao bi mo.
    # Do tren vai batch train SACH; ket qua la buffer nen di theo checkpoint.
    if bool(cfg["model"].get("band_norm", True)):
        scales = compute_band_scales(train_loader, model._qwt_all, model.layout, n_batches=8)
        model.set_band_scales(scales)
        lo = min(scales, key=scales.get)
        hi = max(scales, key=scales.get)
        print(f"band_norm: ON  | dai nho nhat {lo} {scales[lo]:.4f} | "
              f"lon nhat {hi} {scales[hi]:.4f} | ti le {scales[hi]/scales[lo]:.0f}x")
    else:
        print("band_norm: OFF")

    fb = args.freeze_backbone if args.freeze_backbone >= 0 else int(
        cfg["train"].get("freeze_backbone", 0))
    if fb:
        n_fz = model.freeze_backbone()
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze_backbone: ON  ({n_fz} tensor dong bang, con {n_tr/1e6:.2f}M hoc)")

    optimizer, _mult, _nh = build_optimizer(model, cfg)
    if _mult != 1.0:
        print(f"head_lr_mult: {_mult}  ({_nh} tensor cua 2 recon head hoc nhanh hon x{_mult})")

    tcfg = cfg["train"]
    lim_train = args.limit_train_batches or int(tcfg.get("limit_train_batches", 0) or 0)
    lim_val = args.limit_val_batches or int(tcfg.get("limit_val_batches", 0) or 0)

    es = tcfg.get("early_stop") or {}
    es_metric = str(es.get("metric", "L_jepa"))
    es_mode = str(es.get("mode", "min")).lower()
    es_patience = int(es.get("patience", 0) or 0)          # 0 = tat early stop
    es_min_delta = float(es.get("min_delta", 0.0) or 0.0)

    steps_per_epoch = lim_train or len(train_loader)
    total_steps = cfg["train"]["epochs"] * steps_per_epoch
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    scaler = make_scaler(cfg, device)
    start_epoch, step = 0, 0
    best = -math.inf if es_mode == "max" else math.inf
    es_bad = 0
    if args.resume and pathlib.Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        _stale = (
            f"\n{args.resume} khong khop voi model hien tai.\n"
            "Neu day la checkpoint TRUOC khi sua collapse thi khong dung lai duoc: chung deu\n"
            "da collapse (z_std tut dan, L_jepa ~0.01, PSNR ~10 dB).\n"
            "Bo --resume de train lai tu dau. Xem LOG_TRAIN_GIAI_THICH.md muc 9."
        )
        try:
            missing, _ = model.load_state_dict(ck["model"], strict=False)
        except RuntimeError as e:                      # lech shape (doi config kien truc)
            sys.exit(f"{_stale}\n\nChi tiet: {e}")
        if missing:
            sys.exit(f"{_stale}\n\nThieu key: {missing[:5]}")
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        best = ck.get("best", best)
        es_bad = int(ck.get("es_bad", 0))
        print(f"resume tu {args.resume}: epoch {start_epoch}, step {step}, "
              f"best {best:.4f}, es_bad {es_bad}/{es_patience}")

    print(f"train: {steps_per_epoch} batch/epoch | epochs {cfg['train']['epochs']} | "
          f"early_stop {es_metric} ({es_mode}) patience {es_patience or 'off'}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_ds.set_epoch(epoch)
        loader = train_loader
        if lim_train:
            from itertools import islice

            class _Cut:
                batch_size = train_loader.batch_size

                def __iter__(self_):
                    return islice(iter(train_loader), lim_train)

                def __len__(self_):
                    return lim_train

            loader = _Cut()

        step = train_one_epoch(
            model, loader, optimizer, scheduler, cfg, device, step, total_steps, epoch, scaler
        )
        metrics = evaluate(model, val_loader, cfg, device, max_batches=lim_val)
        print(
            f"[eval e{epoch}] L_jepa {metrics['L_jepa']:.4f}"
            f"/{metrics['L_jepa_pos']:.4f}pos | z_std {metrics['zctx_std']:.3f} | "
            f"PSNR {metrics['psnr']:.2f} dB | net {metrics['sharp']:.3f} | "
            f"RMSE acc {metrics['rmse_acc']:.4f} gyro {metrics['rmse_gyro']:.4f}",
            flush=True,
        )

        cur = metrics[es_metric]
        hit_best = _improved(cur, best, es_mode, es_min_delta)
        if hit_best:
            best, es_bad = cur, 0
        else:
            es_bad += 1

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "step": step,
            "best": best,
            "es_bad": es_bad,
            "cfg": cfg,
            "norm_stats": normalizer.to_dict(),
            "val_metrics": metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if hit_best:
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> best.pt ({es_metric} {best:.4f})")
        elif es_patience:
            print(f"  early-stop: {es_bad}/{es_patience} epoch khong cai thien "
                  f"({es_metric} {cur:.4f} vs best {best:.4f})")

        if es_patience and es_bad >= es_patience:
            print(f"dung som o epoch {epoch} ({es_metric} khong cai thien {es_patience} epoch).")
            break

    print("xong.")


if __name__ == "__main__":
    main()
