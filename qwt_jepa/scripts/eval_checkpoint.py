"""Kiem tra mot checkpoint da train (best.pt / last.pt).

Tra loi ba cau hoi, theo dung thu tu quan trong:

  1. Model co THAT SU lam sach du lieu khong?
     -> PSNR / SSIM / RMSE cua model so voi chinh DAU VAO NHIEU tren cung sample.
        Khong vuot duoc dau vao nhieu = model vo dung, du loss dep den may.
  2. Bieu dien JEPA co bi collapse khong?
     -> z_std (theo NOI DUNG) va ti so L_jepa / L_jepa_pos-only.
        Xem LOG_TRAIN_GIAI_THICH.md muc 9-10.
  3. Sai o dau?
     -> phan bo theo environment, percentile, anh minh hoa, do thi IMU.

Do o HAI che do:
  jepa : giong het luc train - co masking, token target di qua predictor.
         So sanh duoc voi dong [eval eN] trong log train.
  full : che do TRIEN KHAI - toan bo token anh/imu nhieu vao encoder, khong che
         gi ca, roi tai tao. Day moi la chat luong khu nhieu that cua model.

Chay:
    python -m qwt_jepa.scripts.eval_checkpoint --ckpt qwt_jepa/runs/base/best.pt \
        --split test --n 256 --by-env

Tap sample: MAC DINH boc NGAU NHIEN (--sample random, seed moi moi lan chay), va truoc
khi boc thi LOAI moi dong trung voi manifest train/valid - tuc la trung voi du lieu da
chay tren Kaggle (--exclude-splits train,valid, doi chieu theo ca trajectory). Manifest
chuan cua tartanair-v2-jepa da chia roi theo trajectory nen thuong loai 0 dong; buoc nay
la luoi an toan, va no in ra so dong bi loai de kiem chung.

    --sample balanced   ngau nhien nhung chia deu so sample cho moi environment
    --sample stride     rai deu tat dinh (kieu cu, de so sanh lai dung cung tap)
    --sample-seed 12345 chay lai y het mot lan truoc (seed cua lan do nam trong
                        command.txt / summary.json / cot `sample` cua index.csv)

Ket qua: MOI LAN CHAY tao mot thu muc con moi, ten la dau thoi gian luc chay,
khong lan nao de len lan nao:

    qwt_jepa/runs/eval_runs/
        index.csv                     <- 1 dong / 1 lan chay: so chinh cua moi lan
        latest -> 2026-09-10_16-20-05 <- symlink toi lan moi nhat
        2026-09-10_16-20-05/
            command.txt               <- lenh da chay + gio chay
            summary.json  per_sample.csv
            qualitative_images.png  error_maps.png
            qualitative_imu.png     imu_rmse_per_channel.png  psnr_hist.png

Doi thu muc cha bang --out-root; ep dung mot thu muc co dinh bang --save-dir
(luc do khong tao thu muc theo thoi gian va khong ghi index.csv).

Config: mac dinh dung dung `cfg` luu trong checkpoint (kien truc phai khop).
Chi cac duong dan du lieu duoc lay tu --config local, vi checkpoint co the train
tren may khac (Kaggle) voi duong dan khac.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.data.dataset import PairedNoisyCleanDataset, jepa_collate      # noqa: E402
from qwt_jepa.data.normalize import ImuNormalizer                            # noqa: E402
from qwt_jepa.models.jepa import QwtJepa                                     # noqa: E402
from qwt_jepa.train.losses import (                                          # noqa: E402
    content_std,
    jepa_loss,
    jepa_loss_position_only,
)

# Cac khoa trong cfg["data"] chi la DUONG DAN -> duoc phep lay tu config local.
_PATH_KEYS = ("root", "jepa_root", "manifest", "norm_stats")
# Cac khoa quyet dinh KIEN TRUC (so token, kich thuoc) -> bat buoc lay tu checkpoint.
_ARCH_KEYS = (("data", "image"), ("data", "imu"), ("model",))


# --------------------------------------------------------------------------- #
# tien ich

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return -10.0 * math.log10(max(float(((a - b) ** 2).mean()), 1e-10))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: [3, H, W] trong [0, 1]."""
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            a.transpose(1, 2, 0), b.transpose(1, 2, 0),
            channel_axis=2, data_range=1.0,
        )
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def _get(d: dict, path: tuple):
    for k in path:
        d = d.get(k, {})
    return d


def merge_cfg(cfg_ck: dict, cfg_local: dict | None) -> dict:
    """Kien truc + corruption tu checkpoint; duong dan du lieu tu config local."""
    cfg = json.loads(json.dumps(cfg_ck))          # deep copy qua JSON (cfg toan scalar/list)
    if cfg_local is None:
        return cfg

    for k in _PATH_KEYS:
        if k in cfg_local.get("data", {}):
            cfg["data"][k] = cfg_local["data"][k]
    if "split" in cfg_local.get("data", {}):
        cfg["data"]["split"] = cfg_local["data"]["split"]

    for path in _ARCH_KEYS:
        a, b = _get(cfg_ck, path), _get(cfg_local, path)
        if a != b:
            print(f"[chu y] {'.'.join(path)} trong config local KHAC checkpoint -> "
                  f"dung ban cua checkpoint (neu khong se lech shape).")
    if cfg_ck.get("corruption") != cfg_local.get("corruption"):
        print("[chu y] khoi `corruption` local khac checkpoint -> dung ban cua checkpoint, "
              "de nhieu luc test giong het luc train.")
    return cfg


def _manifest_keys(path: str | pathlib.Path, level: str) -> set[str]:
    """Doc mot manifest -> tap khoa dung de doi chieu trung lap.

    level="traj"   : ca trajectory (chat hon - hai frame lien tiep cung mot trajectory
                     gan nhu la mot anh, tinh la trung).
    level="sample" : dung tung frame.
    """
    keys: set[str] = set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            traj = f'{r["environment"]}/{r["difficulty"]}/{r["trajectory"]}'
            keys.add(traj if level == "traj"
                     else f'{traj}#{int(r["sample_id"].rsplit("__", 1)[-1])}')
    return keys


def _row_key(row: dict, level: str) -> str:
    """Cung dinh dang khoa nhu _manifest_keys, nhung tinh tu row cua dataset."""
    return row["traj"] if level == "traj" else f'{row["traj"]}#{row["frame_idx"]}'


def build_dataset(cfg: dict, split: str, normalizer, epoch: int, n: int,
                  mode: str = "random", seed: int | None = None,
                  exclude_splits: tuple[str, ...] = ("train", "valid"),
                  exclude_level: str = "traj"):
    """Boc <n> sample de eval, KHONG dinh vao du lieu model da nhin thay luc train.

    Hai buoc, dung thu tu do:

      1. LOC : moi dong co khoa trung voi manifest cua `exclude_splits` (train/valid -
               nhung gi da chay tren Kaggle) bi vut truoc khi boc. Voi manifest chuan
               cua tartanair-v2-jepa ba split da roi nhau theo trajectory nen thuong loc
               0 dong; day la luoi an toan cho manifest tu che, hoac khi lan train tren
               Kaggle chia split khac voi manifest dang co o may nay.
      2. BOC : mode="random"   - ngau nhien deu tren toan bo phan con lai (mac dinh)
               mode="balanced" - ngau nhien nhung chia deu so sample cho moi environment
               mode="stride"   - rai deu theo chi so, tat dinh (kieu cu, de so sanh lai
                                 dung cung tap voi cac lan chay truoc)

    Tra ve (dataset da cat, tong so dong cua split, thong tin buoc boc).
    """
    ds = PairedNoisyCleanDataset(
        cfg["data"]["manifest"][split], cfg["data"]["root"], cfg, split, normalizer, epoch
    )
    n_total = len(ds)

    banned: set[str] = set()
    for s in exclude_splits:
        if s == split:
            continue
        mpath = cfg["data"]["manifest"].get(s)
        if not mpath or not pathlib.Path(mpath).exists():
            print(f"[chu y] khong doc duoc manifest '{s}' ({mpath}) -> bo qua buoc chong "
                  f"trung voi split nay.")
            continue
        banned |= _manifest_keys(mpath, exclude_level)

    keep = [i for i, r in enumerate(ds.rows) if _row_key(r, exclude_level) not in banned]
    n_drop = n_total - len(keep)
    if not keep:
        raise RuntimeError(
            f"split '{split}' trung HOAN TOAN voi {list(exclude_splits)} theo "
            f"{exclude_level} - khong con dong nao de eval."
        )

    rng = np.random.default_rng(seed)
    if not n or n >= len(keep):
        idx = list(keep)
    elif mode == "stride":
        stride = len(keep) / n
        idx = [keep[int(i * stride)] for i in range(n)]
    elif mode == "balanced":
        by_env: dict[str, list[int]] = {}
        for i in keep:
            by_env.setdefault(ds.rows[i]["env"], []).append(i)
        envs = sorted(by_env)
        quota = {e: n // len(envs) for e in envs}
        for e in envs[: n % len(envs)]:            # chia phan du cho vai env dau
            quota[e] += 1
        idx = []
        for e in envs:
            pool = by_env[e]
            k = min(quota[e], len(pool))
            idx += [int(i) for i in rng.choice(pool, size=k, replace=False)]
        if len(idx) < n:                           # env nao do it dong hon quota -> bu them
            rest = [i for i in keep if i not in set(idx)]
            k = min(n - len(idx), len(rest))
            idx += [int(i) for i in rng.choice(rest, size=k, replace=False)]
    else:
        idx = [int(i) for i in rng.choice(keep, size=n, replace=False)]

    idx = sorted(set(idx))
    info = {
        "mode": mode,
        "seed": seed,
        "n_split_rows": n_total,
        "n_eligible": len(keep),
        "n_excluded": n_drop,
        "exclude_splits": list(exclude_splits),
        "exclude_level": exclude_level,
    }
    return Subset(ds, idx), n_total, info


# --------------------------------------------------------------------------- #
# hai che do chay

@torch.no_grad()
def forward_full(model: QwtJepa, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Che do TRIEN KHAI: khong che gi, toan bo token nhieu -> encoder -> 2 head.

    Khong dung predictor, khong dung missing_token: do dung chat luong khu nhieu
    ma model dat duoc khi duoc nhin toan bo dau vao.
    """
    # Uy quyen cho QwtJepa.reconstruct de LUON khop voi nhanh tai tao luc train
    # (ke ca skip connection). Tu ghep lai o day tung lam lech khi kien truc doi.
    return model.reconstruct(batch["img_noisy"], batch["imu_noisy"])


@torch.no_grad()
def run_eval(model, loader, cfg, device, use_amp: bool) -> tuple[list[dict], dict]:
    """Tra ve (rows theo tung sample, agg cac chi so muc batch)."""
    model.eval()
    gen = torch.Generator().manual_seed(int(cfg["train"].get("eval_mask_seed", 1234)))

    def ac():
        if use_amp and device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    rows: list[dict] = []
    jepa_sum = {"L_jepa": 0.0, "L_jepa_pos": 0.0, "zctx_std": 0.0, "ztgt_std": 0.0, "n": 0}

    for batch in loader:
        meta = batch["meta"]
        dev_batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        with ac():
            out = model(dev_batch, generator=gen)            # che do jepa (co masking)
            img_full, imu_full = forward_full(model, dev_batch)

        b = dev_batch["img_clean"].shape[0]
        jepa_sum["L_jepa"] += float(jepa_loss(out["z_pred"], out["z_tgt"])) * b
        jepa_sum["L_jepa_pos"] += float(jepa_loss_position_only(out["z_tgt"])) * b
        jepa_sum["zctx_std"] += float(content_std(out["z_ctx"])) * b
        jepa_sum["ztgt_std"] += float(content_std(out["z_tgt"])) * b
        jepa_sum["n"] += b

        clean = dev_batch["img_clean"].float().cpu().numpy()
        noisy = dev_batch["img_noisy"].float().cpu().numpy()
        rec_f = img_full.float().cpu().numpy()
        rec_j = out["img_rec"].float().cpu().numpy()
        ic = dev_batch["imu_clean"].float().cpu().numpy()
        iz = dev_batch["imu_noisy"].float().cpu().numpy()
        ir_f = imu_full.float().cpu().numpy()
        ir_j = out["imu_rec"].float().cpu().numpy()

        for i in range(b):
            c, nz_, rf, rj = clean[i], noisy[i], rec_f[i], rec_j[i]
            rfc, rjc = np.clip(rf, 0.0, 1.0), np.clip(rj, 0.0, 1.0)
            rows.append({
                "env": meta[i]["env"],
                "traj": meta[i]["traj"],
                "frame": meta[i]["frame_idx"],
                # --- anh ---
                "psnr_full": psnr(rf, c),               # raw, so sanh duoc voi log train
                "psnr_full_clip": psnr(rfc, c),         # sau khi chan ve [0,1] (thuc te)
                "psnr_jepa": psnr(rj, c),
                "psnr_noisy": psnr(nz_, c),             # nguong PHAI vuot
                "psnr_mean": psnr(np.full_like(c, c.mean()), c),   # san vo dung
                "ssim_full": ssim(rfc, c),
                "ssim_jepa": ssim(np.clip(rj, 0, 1), c),
                "ssim_noisy": ssim(np.clip(nz_, 0, 1), c),
                "bright_err_noisy": float(nz_.mean() - c.mean()),  # bu sang con lech bao nhieu
                "bright_err_full": float(rfc.mean() - c.mean()),
                # --- imu ---
                "rmse_acc_full": rmse(ir_f[i][:, 0:3], ic[i][:, 0:3]),
                "rmse_gyro_full": rmse(ir_f[i][:, 3:6], ic[i][:, 3:6]),
                "rmse_acc_jepa": rmse(ir_j[i][:, 0:3], ic[i][:, 0:3]),
                "rmse_gyro_jepa": rmse(ir_j[i][:, 3:6], ic[i][:, 3:6]),
                "rmse_acc_noisy": rmse(iz[i][:, 0:3], ic[i][:, 0:3]),
                "rmse_gyro_noisy": rmse(iz[i][:, 3:6], ic[i][:, 3:6]),
                "rmse_acc_zero": rmse(np.zeros_like(ic[i][:, 0:3]), ic[i][:, 0:3]),
                "rmse_gyro_zero": rmse(np.zeros_like(ic[i][:, 3:6]), ic[i][:, 3:6]),
                # tung kenh rieng: kenh nao model lam sach duoc, kenh nao lam hong
                **{f"ch{c}_full": rmse(ir_f[i][:, c], ic[i][:, c]) for c in range(6)},
                **{f"ch{c}_noisy": rmse(iz[i][:, c], ic[i][:, c]) for c in range(6)},
            })

        print(f"  {len(rows)} sample...", end="\r", flush=True)

    n = max(jepa_sum.pop("n"), 1)
    return rows, {k: v / n for k, v in jepa_sum.items()}


# --------------------------------------------------------------------------- #
# bao cao

def col(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in rows], dtype=np.float64)


def table(rows: list[dict], groups: list[tuple[str, str, list[tuple[str, str]]]]) -> None:
    for title, unit, items in groups:
        print(f"\n{title}")
        print(f"  {'':<26}{'mean':>9}{'p10':>9}{'p90':>9}   {unit}")
        for label, key in items:
            a = col(rows, key)
            print(f"  {label:<26}{a.mean():>9.3f}{np.percentile(a, 10):>9.3f}"
                  f"{np.percentile(a, 90):>9.3f}")


def verdict(rows: list[dict], jepa: dict) -> list[tuple[bool, str]]:
    """Cac phep thu pass/fail - moi phep thu chong dung mot kieu tu lua."""
    out = []
    d_psnr = col(rows, "psnr_full").mean() - col(rows, "psnr_noisy").mean()
    win = float((col(rows, "psnr_full") > col(rows, "psnr_noisy")).mean())
    d_acc = col(rows, "rmse_acc_noisy").mean() - col(rows, "rmse_acc_full").mean()
    d_gyro = col(rows, "rmse_gyro_noisy").mean() - col(rows, "rmse_gyro_full").mean()
    ratio = jepa["L_jepa"] / max(jepa["L_jepa_pos"], 1e-9)

    out.append((d_psnr > 0,
                f"ANH   : PSNR model - PSNR anh nhieu = {d_psnr:+.2f} dB "
                f"(thang o {win*100:.0f}% sample)"))
    out.append((col(rows, "psnr_full").mean() > col(rows, "psnr_mean").mean(),
                f"ANH   : vuot san 'doan mau trung binh' "
                f"({col(rows,'psnr_full').mean():.2f} vs {col(rows,'psnr_mean').mean():.2f} dB)"))
    out.append((col(rows, "ssim_full").mean() > col(rows, "ssim_noisy").mean(),
                f"ANH   : SSIM {col(rows,'ssim_full').mean():.3f} vs anh nhieu "
                f"{col(rows,'ssim_noisy').mean():.3f}"))
    out.append((d_acc > 0, f"IMU   : RMSE acc  giam {d_acc:+.4f} so voi imu nhieu"))
    out.append((d_gyro > 0, f"IMU   : RMSE gyro giam {d_gyro:+.4f} so voi imu nhieu"))
    out.append((jepa["zctx_std"] > 0.3,
                f"JEPA  : z_std (noi dung) {jepa['zctx_std']:.3f} - tut ve ~0.1 la collapse"))
    out.append((ratio < 1.0,
                f"JEPA  : L_jepa/pos-only = {ratio:.2f} - >= 1.0 nghia la khong dung anh"))
    return out


# --------------------------------------------------------------------------- #
# hinh minh hoa

def save_figs(rows, ds, model, cfg, device, save_dir: pathlib.Path, k: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = col(rows, "psnr_full")
    order = np.argsort(p)
    picks = [int(order[0]), int(order[len(order) // 4]), int(order[len(order) // 2]),
             int(order[-1])][:k]

    base = ds.dataset if isinstance(ds, Subset) else ds
    idx_map = ds.indices if isinstance(ds, Subset) else list(range(len(ds)))

    # ---- luoi anh: nhieu | model (full) | model (jepa) | sach ----
    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, si in enumerate(picks):
        s = base[idx_map[si]]
        b = jepa_collate([s])
        b = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in b.items()}
        with torch.no_grad():
            gen = torch.Generator().manual_seed(int(cfg["train"].get("eval_mask_seed", 1234)))
            out = model(b, generator=gen)
            img_f, _ = forward_full(model, b)
        panes = [
            (b["img_noisy"][0], f"nhieu (dau vao)\nPSNR {rows[si]['psnr_noisy']:.2f} dB"),
            (img_f[0], f"model - che do full\nPSNR {rows[si]['psnr_full']:.2f} dB"),
            (out["img_rec"][0], f"model - che do jepa\nPSNR {rows[si]['psnr_jepa']:.2f} dB"),
            (b["img_clean"][0], f"sach (goc)\n{rows[si]['env']} #{rows[si]['frame']}"),
        ]
        for c, (t, title) in enumerate(panes):
            axes[r, c].imshow(np.clip(t.float().cpu().numpy().transpose(1, 2, 0), 0, 1))
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("Tu tren xuong: sample te nhat -> tot nhat (theo PSNR che do full)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_dir / "qualitative_images.png", dpi=110)
    plt.close(fig)

    # ---- IMU: 6 kenh cua sample trung vi ----
    si = picks[len(picks) // 2]
    s = base[idx_map[si]]
    b = jepa_collate([s])
    b = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in b.items()}
    with torch.no_grad():
        _, imu_f = forward_full(model, b)
    ic = b["imu_clean"][0].float().cpu().numpy()
    iz = b["imu_noisy"][0].float().cpu().numpy()
    ir = imu_f[0].float().cpu().numpy()
    names = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for ch in range(6):
        axes[ch].plot(iz[:, ch], color="0.65", lw=1.6, label="nhieu")
        axes[ch].plot(ir[:, ch], color="tab:red", lw=1.1, label="model")
        axes[ch].plot(ic[:, ch], color="tab:blue", lw=1.1, ls="--", label="sach")
        # Ba duong thuong chong len nhau -> ghi RMSE ngay tren truc de doc duoc so.
        axes[ch].set_ylabel(
            f"{names[ch]}\nrmse {rmse(ir[:, ch], ic[:, ch]):.3f}"
            f" / nhieu {rmse(iz[:, ch], ic[:, ch]):.3f}",
            fontsize=8,
        )
        if ch == 0:
            axes[ch].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("mau (100 Hz)")
    fig.suptitle(f"IMU (da chuan hoa) - {rows[si]['env']} #{rows[si]['frame']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_dir / "qualitative_imu.png", dpi=110)
    plt.close(fig)

    # ---- ban do sai so: |model - sach| so voi |nhieu - sach| ----
    fig, axes = plt.subplots(len(picks), 5, figsize=(16, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, si in enumerate(picks):
        s_ = base[idx_map[si]]
        b = jepa_collate([s_])
        b = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in b.items()}
        with torch.no_grad():
            img_f, _ = forward_full(model, b)
        cl = b["img_clean"][0].float().cpu().numpy()
        nz_ = b["img_noisy"][0].float().cpu().numpy()
        rf = np.clip(img_f[0].float().cpu().numpy(), 0, 1)
        e_noisy = np.abs(nz_ - cl).mean(axis=0)
        e_model = np.abs(rf - cl).mean(axis=0)
        vmax = float(max(e_noisy.max(), e_model.max()))
        d = rows[si]["psnr_full"] - rows[si]["psnr_noisy"]
        panes = [
            (nz_.transpose(1, 2, 0), "nhieu (dau vao)", None),
            (rf.transpose(1, 2, 0), "model", None),
            (cl.transpose(1, 2, 0), "sach", None),
            (e_noisy, f"sai so cua ANH NHIEU\nmean {e_noisy.mean():.4f}", vmax),
            (e_model, f"sai so cua MODEL\nmean {e_model.mean():.4f}", vmax),
        ]
        for c, (im, title, vm) in enumerate(panes):
            if vm is None:
                axes[r, c].imshow(np.clip(im, 0, 1))
            else:
                axes[r, c].imshow(im, cmap="inferno", vmin=0, vmax=vm)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(f"{d:+.2f} dB")
        axes[r, 4].set_title(panes[4][1] + f"\n[{d:+.2f} dB so voi anh nhieu]", fontsize=9)
    fig.suptitle("Hai cot phai: sang = sai nhieu. Model phai TOI HON anh nhieu moi la khu duoc.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_dir / "error_maps.png", dpi=110)
    plt.close(fig)

    # ---- IMU: RMSE tung kenh, truoc / sau, tren TOAN BO tap eval ----
    names6 = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
    before = [col(rows, f"ch{c}_noisy").mean() for c in range(6)]
    after = [col(rows, f"ch{c}_full").mean() for c in range(6)]
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.19, before, 0.38, label="imu nhieu (dau vao)", color="0.6")
    ax.bar(x + 0.19, after, 0.38, label="model (che do full)", color="tab:red")
    for i in range(6):
        d = (before[i] - after[i]) / max(before[i], 1e-9) * 100
        ax.text(x[i], max(before[i], after[i]) * 1.02, f"{d:+.0f}%",
                ha="center", fontsize=9, color="tab:green" if d > 0 else "tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(names6)
    ax.set_ylabel("RMSE (da chuan hoa) - thap hon = tot hon")
    ax.set_title(f"IMU: cot do THAP HON cot xam thi model moi lam sach duoc (n={len(rows)})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "imu_rmse_per_channel.png", dpi=110)
    plt.close(fig)

    # ---- histogram PSNR: model vs anh nhieu ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(min(col(rows, "psnr_noisy").min(), p.min()),
                       max(col(rows, "psnr_noisy").max(), p.max()), 40)
    ax.hist(col(rows, "psnr_noisy"), bins=bins, alpha=0.55, label="anh nhieu (dau vao)")
    ax.hist(p, bins=bins, alpha=0.55, label="model (che do full)")
    ax.axvline(col(rows, "psnr_mean").mean(), color="k", ls="--", lw=1,
               label="san: doan mau trung binh")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("so sample")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "psnr_hist.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(pathlib.Path.home() / "Downloads" / "best21.pt"))
    ap.add_argument("--config", default=str(_ROOT / "qwt_jepa" / "configs" / "base.yaml"),
                    help="chi lay duong dan du lieu; kien truc luon theo checkpoint")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=256, help="so sample lay tu split (0 = full)")
    ap.add_argument("--sample", default="random", choices=["random", "balanced", "stride"],
                    help="random = boc ngau nhien (mac dinh); balanced = ngau nhien nhung "
                         "chia deu cho moi environment; stride = rai deu tat dinh (kieu cu)")
    ap.add_argument("--sample-seed", type=int, default=-1,
                    help="seed boc mau; -1 = moi lan chay mot tap khac. Seed thuc su dung "
                         "duoc ghi vao command.txt/summary.json de chay lai y het")
    ap.add_argument("--exclude-splits", default="train,valid",
                    help="cac split model DA nhin thay luc train -> loai khoi tap eval "
                         "truoc khi boc. '' = tat luoi chong trung")
    ap.add_argument("--exclude-level", default="traj", choices=["traj", "sample"],
                    help="doi chieu trung theo ca trajectory (chat, mac dinh) hay tung frame")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epoch-seed", type=int, default=0, help="seed corruption (nhu set_epoch)")
    ap.add_argument("--amp", action="store_true", help="bf16 khi eval (nhanh hon, so hoi le)")
    ap.add_argument("--out-root", default=str(_ROOT / "qwt_jepa" / "runs" / "eval_runs"),
                    help="thu muc CHA; moi lan chay tao mot thu muc con dat ten theo thoi diem chay")
    ap.add_argument("--save-dir", default="",
                    help="ep dung dung mot thu muc (bo qua --out-root, ghi de neu da co)")
    ap.add_argument("--tag", default="", help="hau to them vao ten thu muc, vd --tag epoch8")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--by-env", action="store_true", help="tach ket qua theo environment")
    args = ap.parse_args()

    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"khong thay checkpoint: {ckpt_path}")
    # -1 = ngau nhien that. Chot seed NGAY BAY GIO va ghi lai, de mot lan chay bat ky
    # van lap lai duoc y nguyen tap sample da dung (--sample-seed <so da ghi>).
    sample_seed = (int(np.random.SeedSequence().entropy % (2**31))
                   if args.sample_seed < 0 else args.sample_seed)
    exclude_splits = tuple(x.strip() for x in args.exclude_splits.split(",") if x.strip())
    # Moi lan chay = mot thu muc con moi, ten la dau thoi gian luc chay. Khong lan nao
    # de len lan nao, nen so sanh duoc cac checkpoint / cac lan sua code voi nhau.
    started = time.localtime()
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", started)
    if args.tag:
        stamp = f"{stamp}_{args.tag}"
    out_root = pathlib.Path(args.out_root)
    save_dir = pathlib.Path(args.save_dir) if args.save_dir else out_root / stamp
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.save_dir:
        # `latest` luon tro toi lan chay moi nhat - do la duong dan on dinh de mo nhanh.
        latest = out_root / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        try:
            latest.symlink_to(stamp)
        except OSError:                      # he thong file khong cho symlink -> bo qua
            pass
    (save_dir / "command.txt").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', started)}\n"
        f"{sys.executable} -m qwt_jepa.scripts.eval_checkpoint "
        + " ".join(sys.argv[1:]) + "\n"
        f"# chay lai dung tap sample nay: --sample {args.sample} --sample-seed {sample_seed}\n"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("=" * 78)
    print(f"checkpoint : {ckpt_path}  ({ckpt_path.stat().st_size/1e6:.0f} MB)")
    print(f"epoch      : {ck.get('epoch')}   step: {ck.get('step')}   best: {ck.get('best')}")
    if ck.get("val_metrics"):
        vm = ck["val_metrics"]
        print("val luc luu : " + "  ".join(f"{k} {v:.4f}" for k, v in vm.items()
                                           if isinstance(v, (int, float))))
    print("=" * 78)

    cfg_local = None
    if args.config and pathlib.Path(args.config).exists():
        cfg_local = yaml.safe_load(pathlib.Path(args.config).read_text())
    cfg = merge_cfg(ck["cfg"], cfg_local)

    if ck.get("norm_stats"):
        normalizer = ImuNormalizer(**ck["norm_stats"])       # dung dung thong ke luc train
        print("norm stats : lay tu checkpoint")
    else:
        normalizer = ImuNormalizer.from_yaml(cfg["data"]["norm_stats"])
        print(f"norm stats : {cfg['data']['norm_stats']}")

    ds, n_total, samp = build_dataset(
        cfg, args.split, normalizer, args.epoch_seed, args.n,
        mode=args.sample, seed=sample_seed,
        exclude_splits=exclude_splits, exclude_level=args.exclude_level,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=jepa_collate,
    )
    print(f"du lieu    : {args.split} - {len(ds)}/{samp['n_eligible']} sample du dieu kien "
          f"(split co {n_total}) | root {cfg['data']['root']}")
    print(f"boc mau    : {samp['mode']} | seed {sample_seed} | loai trung voi "
          f"{exclude_splits if exclude_splits else '(tat)'} theo {args.exclude_level} "
          f"-> vut {samp['n_excluded']} dong")
    # Manifest chuan chia theo trajectory, nen cung mot environment van xuat hien o ca
    # train lan test. Noi ro de khong doc nham ket qua thanh "moi truong hoan toan la".
    if not cfg["data"].get("split", {}).get("enforce_env_split", False):
        print("             (split theo TRAJECTORY: trajectory khong trung train, nhung "
              "environment thi co - dat enforce_env_split=true neu muon env la hoan toan)")

    model = QwtJepa(cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[canh bao] state_dict lech: thieu {len(missing)} key, thua {len(unexpected)}.")
        if missing:
            print(f"           vd thieu: {missing[:5]}")
        if any(k.startswith("target_") for k in missing):
            print("           -> checkpoint truoc ban va collapse. Ket qua khong dang tin.")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model      : {n_par/1e6:.2f}M params | {model.layout.n_tokens} token | {device}")
    print(f"eval       : {'bf16' if args.amp else 'fp32'} | "
          f"mask seed {cfg['train'].get('eval_mask_seed', 1234)} | "
          f"corruption seed epoch={args.epoch_seed}\n")

    rows, jepa = run_eval(model, loader, cfg, device, args.amp)
    print(f"  {len(rows)} sample. xong.        ")

    # ------------------------------------------------------------------ bao cao
    print("\n" + "=" * 78)
    print(f"KET QUA - {args.split}, n={len(rows)}")
    print("=" * 78)
    table(rows, [
        ("ANH - PSNR (cao hon = tot hon)", "dB", [
            ("model (full, khong che)", "psnr_full"),
            ("model (full, da clip)", "psnr_full_clip"),
            ("model (jepa, co masking)", "psnr_jepa"),
            ("[moc] anh nhieu dau vao", "psnr_noisy"),
            ("[san] doan mau trung binh", "psnr_mean"),
        ]),
        ("ANH - SSIM (cao hon = tot hon)", "0..1", [
            ("model (full)", "ssim_full"),
            ("model (jepa)", "ssim_jepa"),
            ("[moc] anh nhieu dau vao", "ssim_noisy"),
        ]),
        ("ANH - lech do sang so voi anh sach", "|.| nho hon = tot", [
            ("anh nhieu dau vao", "bright_err_noisy"),
            ("model (full)", "bright_err_full"),
        ]),
        ("IMU - RMSE acc (thap hon = tot hon)", "da chuan hoa", [
            ("model (full)", "rmse_acc_full"),
            ("model (jepa)", "rmse_acc_jepa"),
            ("[moc] imu nhieu dau vao", "rmse_acc_noisy"),
            ("[san] tra ve toan 0", "rmse_acc_zero"),
        ]),
        ("IMU - RMSE gyro (thap hon = tot hon)", "da chuan hoa", [
            ("model (full)", "rmse_gyro_full"),
            ("model (jepa)", "rmse_gyro_jepa"),
            ("[moc] imu nhieu dau vao", "rmse_gyro_noisy"),
            ("[san] tra ve toan 0", "rmse_gyro_zero"),
        ]),
    ])

    print(f"\nJEPA (che do co masking, giong log train)")
    print(f"  L_jepa                  {jepa['L_jepa']:.4f}")
    print(f"  L_jepa pos-only         {jepa['L_jepa_pos']:.4f}   (moc gian lan: chi doan "
          f"theo vi tri token)")
    print(f"  ti so                   {jepa['L_jepa']/max(jepa['L_jepa_pos'],1e-9):.2f}"
          f"   (< 1.0 = co dung thong tin anh)")
    print(f"  z_ctx std (noi dung)    {jepa['zctx_std']:.3f}")
    print(f"  z_tgt std (noi dung)    {jepa['ztgt_std']:.3f}")

    if args.by_env:
        envs = sorted({r["env"] for r in rows})
        print(f"\nTHEO ENVIRONMENT")
        print(f"  {'env':<24}{'n':>5}{'PSNR':>9}{'anh nhieu':>11}{'delta':>8}{'SSIM':>8}")
        for e in envs:
            sub = [r for r in rows if r["env"] == e]
            a, b = col(sub, "psnr_full").mean(), col(sub, "psnr_noisy").mean()
            print(f"  {e:<24}{len(sub):>5}{a:>9.2f}{b:>11.2f}{a-b:>+8.2f}"
                  f"{col(sub,'ssim_full').mean():>8.3f}")

    print("\n" + "=" * 78)
    print("KET LUAN")
    print("=" * 78)
    checks = verdict(rows, jepa)
    for ok, msg in checks:
        print(f"  [{'OK  ' if ok else 'HONG'}] {msg}")
    n_bad = sum(1 for ok, _ in checks if not ok)
    print(f"\n  {len(checks)-n_bad}/{len(checks)} phep thu dat.")
    if n_bad:
        print("  Phep thu hong dau tien la thu can sua truoc - xem "
              "LOG_TRAIN_GIAI_THICH.md muc 9-10.")

    # ------------------------------------------------------------------ luu
    with open(save_dir / "per_sample.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "run": stamp,
        "started": time.strftime("%Y-%m-%d %H:%M:%S", started),
        "ckpt": str(ckpt_path),
        "split": args.split,
        "n": len(rows),
        "sampling": samp,
        "epoch": ck.get("epoch"),
        "step": ck.get("step"),
        "jepa": jepa,
        "mean": {k: float(col(rows, k).mean()) for k in rows[0] if k not in ("env", "traj")},
        "checks": [{"ok": bool(ok), "msg": m} for ok, m in checks],
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_figs:
        save_figs(rows, ds, model, cfg, device, save_dir, k=4)

    # Mot dong / mot lan chay, gom o thu muc cha -> nhin duoc xu huong qua cac lan.
    if not args.save_dir:
        index = out_root / "index.csv"
        line = {
            "run": stamp,
            "ckpt": ckpt_path.name,
            "ckpt_epoch": ck.get("epoch"),
            "split": args.split,
            "n": len(rows),
            "sample": f"{samp['mode']}/{sample_seed}",
            "psnr_full": round(float(col(rows, "psnr_full").mean()), 3),
            "psnr_noisy": round(float(col(rows, "psnr_noisy").mean()), 3),
            "delta_psnr": round(float((col(rows, "psnr_full") - col(rows, "psnr_noisy")).mean()), 3),
            "ssim_full": round(float(col(rows, "ssim_full").mean()), 4),
            "rmse_acc_full": round(float(col(rows, "rmse_acc_full").mean()), 4),
            "rmse_gyro_full": round(float(col(rows, "rmse_gyro_full").mean()), 4),
            "jepa_ratio": round(jepa["L_jepa"] / max(jepa["L_jepa_pos"], 1e-9), 3),
            "zctx_std": round(jepa["zctx_std"], 3),
            "checks_ok": f"{len(checks)-n_bad}/{len(checks)}",
        }
        # index.csv cu co the thieu cot moi (vd `sample`) -> ghi lai ca file voi header
        # moi, cac lan chay cu de trong. Neu chi append thi cac cot se lech nhau.
        old_rows = []
        if index.exists():
            with open(index, newline="") as fh:
                old_rows = list(csv.DictReader(fh))
        fields = list(line.keys())
        stale = old_rows and list(old_rows[0].keys()) != fields
        if stale:
            with open(index, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
                w.writeheader()
                w.writerows(old_rows)
        with open(index, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if not old_rows and not index.stat().st_size:
                w.writeheader()
            w.writerow(line)
        print(f"\nda ghi 1 dong vao {index}")

    print(f"\nda luu vao {save_dir}/:")
    for f in sorted(save_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
