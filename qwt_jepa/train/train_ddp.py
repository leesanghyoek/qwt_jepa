"""Train co ho tro DDP - chay 1 GPU hoac nhieu GPU (vd Kaggle T4 x2).

1 GPU (tuong duong train.py):
    python qwt_jepa/train/train_ddp.py \
        --config qwt_jepa/configs/kaggle.yaml --out /kaggle/working/runs/base \
        --batch-size 32 --epochs 100 --num-workers 4

2 GPU:
    torchrun --standalone --nproc_per_node=2 qwt_jepa/train/train_ddp.py \
        --config qwt_jepa/configs/kaggle.yaml --out /kaggle/working/runs/base \
        --batch-size 32 --epochs 100 --num-workers 2

LUU Y:
- --batch-size la batch MOI GPU. Batch hieu dung = batch_size * so_gpu.
- Khi --resume phai giu NGUYEN so_gpu + batch-size + epochs (total_steps, lich LR/EMA
  phu thuoc ca 3). Checkpoint co luu "world_size" de canh bao neu lech.
- Log in ra giong het train.py (chi rank 0 in) nen wrapper doc log van chay.
- eval chay tren rank 0 voi full val set -> so lieu chinh xac; cac rank khac cho o barrier.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.data.dataset import PairedNoisyCleanDataset, jepa_collate   # noqa: E402
from qwt_jepa.data.normalize import ImuNormalizer                        # noqa: E402
from qwt_jepa.models.jepa import QwtJepa                                  # noqa: E402
from qwt_jepa.train import ema_momentum                                   # noqa: E402
from qwt_jepa.train.losses import total_loss                             # noqa: E402
from qwt_jepa.train.engine import make_scaler                            # noqa: E402
from qwt_jepa.train.train import _improved, build_scheduler             # noqa: E402


def ddp_setup() -> tuple[bool, int, int, int]:
    """Khoi tao process group neu chay duoi torchrun. Tra (is_dist, rank, world, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # timeout rong hon mac dinh 10' -> eval/luu checkpoint cham khong lam watchdog giet job
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_ROOT / "qwt_jepa" / "configs" / "base.yaml"))
    ap.add_argument("--out", default=str(_ROOT / "qwt_jepa" / "runs" / "base"))
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0, help="batch MOI GPU")
    ap.add_argument("--num-workers", type=int, default=-1)
    ap.add_argument("--encoder-depth", type=int, default=0)
    ap.add_argument("--limit-train-batches", type=int, default=0)
    ap.add_argument("--find-unused", type=int, default=1, help="DDP find_unused_parameters (1/0)")
    args = ap.parse_args()

    is_dist, rank, world, local = ddp_setup()
    is_main = rank == 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    def log(*a, **k):
        if is_main:
            print(*a, **k)

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
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)   # cung seed tren moi rank -> init model giong nhau

    norm_path = pathlib.Path(cfg["data"]["norm_stats"])
    if norm_path.exists():
        normalizer = ImuNormalizer.from_yaml(norm_path)
        log(f"norm stats: {norm_path}")
    else:
        normalizer = ImuNormalizer.identity()
        log(f"[canh bao] chua co {norm_path} -> dung identity. Chay compute_norm_stats truoc.")

    root = cfg["data"]["root"]
    man = cfg["data"]["manifest"]
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 4))

    train_ds = PairedNoisyCleanDataset(man["train"], root, cfg, "train", normalizer)
    val_ds = PairedNoisyCleanDataset(man["valid"], root, cfg, "valid", normalizer)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        if is_dist else None
    )
    common = dict(
        num_workers=nw, pin_memory=True, collate_fn=jepa_collate, persistent_workers=nw > 0
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=train_sampler,
        shuffle=(train_sampler is None), drop_last=True, **common,
    )
    # eval: chia deu val cho cac rank -> nhanh gap so_gpu lan, khong rank nao cho lau
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        if is_dist else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, sampler=val_sampler, shuffle=False, drop_last=False, **common
    )

    core = QwtJepa(cfg).to(device)
    n_par = sum(p.numel() for p in core.parameters() if p.requires_grad)
    log(f"QwtJepa: {n_par / 1e6:.2f}M params train | N tokens {core.layout.n_tokens} | "
        f"world {world} | batch/gpu {bs} | batch hieu dung {bs * world}")

    ocfg = cfg["train"]["optimizer"]
    optimizer = torch.optim.AdamW(
        (p for p in core.parameters() if p.requires_grad),
        lr=float(ocfg["lr"]), weight_decay=float(ocfg["weight_decay"]), betas=(0.9, 0.95),
    )

    tcfg = cfg["train"]
    lim_train = args.limit_train_batches or int(tcfg.get("limit_train_batches", 0) or 0)
    lim_val = int(tcfg.get("limit_val_batches", 0) or 0)

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
        core.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        best = ck.get("best", best)
        es_bad = int(ck.get("es_bad", 0))
        if ck.get("world_size", 1) != world:
            log(f"[canh bao] resume world_size {ck.get('world_size', 1)} -> {world}: "
                f"lich LR/EMA se lech. Nen giu nguyen so GPU moi phien.")
        log(f"resume tu {args.resume}: epoch {start_epoch}, step {step}, "
            f"best {best:.4f}, es_bad {es_bad}/{es_patience}")

    model = (
        DDP(core, device_ids=[local] if torch.cuda.is_available() else None,
            find_unused_parameters=bool(args.find_unused))
        if is_dist else core
    )
    log(f"train {len(train_ds)} sample | {steps_per_epoch} batch/epoch | device {device}")

    lam = cfg["train"]["loss"]
    ema_cfg = cfg["train"]["ema"]
    grad_clip = float(cfg["train"].get("grad_clip", 0) or 0)
    log_every = int(cfg["train"].get("log_every", 50))
    amp = str(cfg["train"].get("amp", "none")).lower()

    def autocast():
        if device.type == "cuda" and amp in ("bf16", "bfloat16"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and amp in ("fp16", "float16"):
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast("cpu", enabled=False)

    def to_dev(batch: dict) -> dict:
        return {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    @torch.no_grad()
    def evaluate_dist() -> dict:
        """Moi rank eval phan val cua minh -> all_reduce tong -> chia. MOI rank deu goi."""
        core.eval()
        # [sum L_jepa*b, sum mse*b, sum rmse_acc*b, sum rmse_gyro*b, n]
        acc = torch.zeros(5, dtype=torch.float64, device=device)
        for bi, batch in enumerate(val_loader):
            if lim_val and bi >= lim_val:
                break
            batch = to_dev(batch)
            with autocast():
                out = core(batch)
            _, logs = total_loss(out, batch, lam["lambda_img"], lam["lambda_imu"])
            b = batch["img_clean"].shape[0]
            mse = torch.mean((out["img_rec"].float() - batch["img_clean"].float()) ** 2)
            rmse_acc = torch.sqrt(torch.mean(
                (out["imu_rec"][..., 0:3].float() - batch["imu_clean"][..., 0:3].float()) ** 2
            ))
            rmse_gyro = torch.sqrt(torch.mean(
                (out["imu_rec"][..., 3:6].float() - batch["imu_clean"][..., 3:6].float()) ** 2
            ))
            acc[0] += logs["L_jepa"] * b
            acc[1] += mse.double() * b
            acc[2] += rmse_acc.double() * b
            acc[3] += rmse_gyro.double() * b
            acc[4] += b
        if is_dist:
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        n = max(acc[4].item(), 1.0)
        mse_mean = acc[1].item() / n
        return {
            "L_jepa": acc[0].item() / n,
            "psnr": -10.0 * math.log10(max(mse_mean, 1e-10)),
            "rmse_acc": acc[2].item() / n,
            "rmse_gyro": acc[3].item() / n,
        }

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_ds.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        core.train()

        t0 = time.time()
        seen = 0
        it = iter(train_loader)
        for i in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = to_dev(batch)
            with autocast():
                out = model(batch)
                loss, logs = total_loss(out, batch, lam["lambda_img"], lam["lambda_imu"])

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)   # go scale TRUOC khi clip theo norm
                    torch.nn.utils.clip_grad_norm_(core.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(core.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()

            m = ema_momentum(step, total_steps, ema_cfg["start"], ema_cfg["end"])
            core.ema_update(m)
            step += 1
            seen += bs

            if is_main and i % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                ips = seen * world / (time.time() - t0)
                print(
                    f"e{epoch} {i:4d}/{steps_per_epoch} step {step:6d} | "
                    f"L {logs['L_total']:.4f} (jepa {logs['L_jepa']:.4f} "
                    f"img {logs['L_img']:.4f} imu {logs['L_imu']:.4f}) | "
                    f"z_std {logs['ztgt_std']:.3f} | lr {lr:.2e} m {m:.4f} | {ips:.1f} im/s",
                    flush=True,
                )

        metrics = evaluate_dist()   # MOI rank cung goi (all_reduce ben trong -> metrics giong nhau)

        # metrics/best/es_bad giong het nhau tren moi rank -> quyet dinh early-stop
        # deterministic, khong can broadcast.
        cur = metrics[es_metric]
        hit_best = _improved(cur, best, es_mode, es_min_delta)
        if hit_best:
            best, es_bad = cur, 0
        else:
            es_bad += 1

        if is_main:
            print(
                f"[eval e{epoch}] L_jepa {metrics['L_jepa']:.4f} | PSNR {metrics['psnr']:.2f} dB | "
                f"RMSE acc {metrics['rmse_acc']:.4f} gyro {metrics['rmse_gyro']:.4f}",
                flush=True,
            )
            ckpt = {
                "model": core.state_dict(),
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
                "world_size": world,
            }
            torch.save(ckpt, out_dir / "last.pt")
            if hit_best:
                torch.save(ckpt, out_dir / "best.pt")
                print(f"  -> best.pt ({es_metric} {best:.4f})", flush=True)
            elif es_patience:
                print(f"  early-stop: {es_bad}/{es_patience} epoch khong cai thien "
                      f"({es_metric} {cur:.4f} vs best {best:.4f})", flush=True)

        if is_dist:
            dist.barrier()

        if es_patience and es_bad >= es_patience:
            log(f"dung som o epoch {epoch} ({es_metric} khong cai thien {es_patience} epoch).")
            break

    log("xong.")
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
