"""Minh hoa bo lam nhieu: anh + IMU TRUOC va SAU corruption.

MAC DINH (che do `dataset`) script lay 1 sample THAT qua chinh
`PairedNoisyCleanDataset.__getitem__` - tuc la DUNG code path ma train.py dung, nen
cap (sach, nhieu) hien ra la y het cai model nhin thay:

  - dung anh cua manifest, resize theo data.image.train_crop
  - dung cua so IMU CAUSAL 128 mau ket thuc tai frame do (imu_end_row_exclusive)
  - dung normalizer tu data.norm_stats (acc/gyro chuan hoa rieng)
  - dung seed on dinh _stable_seed(env, traj, frame_idx, epoch)
  - dung corrupt_image / corrupt_imu voi khoi `corruption:` trong config

Doi `--epoch` se ra realization nhieu khac (giong luc train sang epoch moi).

Che do `adhoc` (--image ...) danh cho anh bat ky ngoai dataset: van goi dung
corrupt_image / corrupt_imu, nhung cua so IMU va seed do nguoi dung chon.

Hien thi: mac dinh ve THANG ra cua so matplotlib (block den khi dong), khong ghi file.
  --save [PATH] : cung ghi PNG (mac dinh ./corruption_demo.png)
  --no-window   : khong mo cua so, chi ghi PNG + mo bang trinh xem anh
Trong Jupyter / VSCode Interactive (%matplotlib inline) hinh hien inline ngay.

Vi du:

    # 1 sample ngau nhien tu split valid - dung y het luc train
    python qwt_jepa/scripts/demo_corruption.py

    # chon dung sample so 100 cua split train, epoch 3
    python qwt_jepa/scripts/demo_corruption.py --split train --index 100 --epoch 3

    # anh bat ky ngoai dataset (IMU tu sinh neu khong dua --acc/--gyro)
    python qwt_jepa/scripts/demo_corruption.py --image mona_lisa_500.jpg

LUU Y: che do dataset can torch (vi import qwt_jepa.data.dataset). Tren may nay dung:
    /home/buidinhkhoi/anaconda3/bin/python qwt_jepa/scripts/demo_corruption.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import yaml
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DATA = _ROOT / "qwt_jepa" / "data"

CH_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _load_standalone(mod_name: str):
    """Nap 1 module trong qwt_jepa/data/ theo duong dan file - tranh keo torch qua
    qwt_jepa/data/__init__.py (corruption.py va normalize.py chi can numpy)."""
    spec = importlib.util.spec_from_file_location(mod_name, _DATA / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_corruption = _load_standalone("corruption")
_normalize = _load_standalone("normalize")
corrupt_image = _corruption.corrupt_image
corrupt_imu = _corruption.corrupt_imu
ImuNormalizer = _normalize.ImuNormalizer


# --------------------------------------------------------------------------- #
# Che do DATASET - dung chinh PairedNoisyCleanDataset (giong het luc train)
# --------------------------------------------------------------------------- #
def sample_from_dataset(cfg: dict, split: str, index: int, epoch: int) -> dict:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    try:
        from qwt_jepa.data.dataset import PairedNoisyCleanDataset, _stable_seed
    except ImportError as e:                                   # thuong la thieu torch
        raise SystemExit(
            f"khong import duoc qwt_jepa.data.dataset ({e}).\n"
            f"Che do dataset can torch - thu:\n"
            f"    /home/buidinhkhoi/anaconda3/bin/python {sys.argv[0]} ...\n"
            f"hoac dung che do adhoc: --image <file anh>"
        )

    key = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}[split]
    manifest = cfg["data"]["manifest"][key]

    norm_path = pathlib.Path(cfg["data"]["norm_stats"])
    if norm_path.exists():
        normalizer = ImuNormalizer.from_yaml(norm_path)
        print(f"norm_stats : {norm_path}")
    else:
        normalizer = ImuNormalizer.identity()
        print(f"[canh bao] khong thay {norm_path} -> IMU khong chuan hoa (identity)")

    ds = PairedNoisyCleanDataset(manifest, cfg["data"]["root"], cfg, split, normalizer,
                                 epoch=epoch)
    if index < 0:
        index = int(np.random.default_rng().integers(len(ds)))
    if not 0 <= index < len(ds):
        raise SystemExit(f"--index {index} ngoai pham vi [0, {len(ds)})")

    s = ds[index]
    m = s["meta"]
    seed = _stable_seed(m["env"], m["traj"], m["frame_idx"], epoch)
    print(f"manifest   : {manifest}")
    print(f"sample     : #{index}/{len(ds)} (split {split}, epoch {epoch})")
    print(f"             env {m['env']} | traj {m['traj']} | frame {m['frame_idx']}")
    print(f"seed nhieu : {seed}  = crc32('{m['env']}|{m['traj']}|{m['frame_idx']}|{epoch}')")
    return {
        "img_clean": s["img_clean"].numpy(),
        "img_noisy": s["img_noisy"].numpy(),
        "imu_clean": s["imu_clean"].numpy(),
        "imu_noisy": s["imu_noisy"].numpy(),
        "title": f"{m['env']} / {m['traj'].split('/')[-1]} / frame {m['frame_idx']}"
                 f"  (sample #{index}, epoch {epoch})",
        "imu_src": f"cua so causal 128 mau ket thuc tai frame {m['frame_idx']} (da chuan hoa)",
    }


# --------------------------------------------------------------------------- #
# Che do ADHOC - anh bat ky ngoai dataset
# --------------------------------------------------------------------------- #
def load_image(path: str, size: int) -> np.ndarray:
    """File anh bat ky -> [3, size, size] float32 trong [0, 1] (giong dataset._load_image)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0             # [H, W, 3]
    return np.ascontiguousarray(arr.transpose(2, 0, 1))        # [3, H, W]


def load_imu_window(acc_path: str, gyro_path: str, window: int, start: int | None) -> np.ndarray:
    """acc.txt + gyro.txt (moi file [N, 3]) -> 1 cua so [window, 6] = [acc | gyro]."""
    acc = np.loadtxt(acc_path, dtype=np.float32)
    gyro = np.loadtxt(gyro_path, dtype=np.float32)
    n = min(len(acc), len(gyro))
    if start is None:
        start = max(0, n - window)                             # cua so cuoi cung (causal)
    end = start + window
    if end > n:
        raise SystemExit(f"cua so [{start}:{end}] vuot qua {n} mau IMU co san")
    return np.concatenate([acc[start:end], gyro[start:end]], axis=1).astype(np.float32)


def synth_imu_window(window: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """IMU tu sinh [window, 6] khi khong co file cam bien - vai sin + rung nhe."""
    t = np.arange(window, dtype=np.float32) / sr
    freqs = [1.5, 0.8, 2.3, 3.1, 1.1, 0.6]
    amps = [0.35, 0.25, 0.20, 0.12, 0.10, 0.08]
    phases = rng.uniform(0, 2 * np.pi, size=6)
    cols = [a * np.sin(2 * np.pi * f * t + p) + 0.01 * rng.standard_normal(window)
            for f, a, p in zip(freqs, amps, phases)]
    return np.stack(cols, axis=1).astype(np.float32)           # [window, 6]


def sample_adhoc(cfg: dict, args, img_size: int, window: int, sr: int) -> dict:
    img_clean = load_image(args.image, img_size)

    if args.acc and args.gyro:
        imu_raw = load_imu_window(args.acc, args.gyro, window,
                                  None if args.imu_start < 0 else args.imu_start)
        imu_src = f"{pathlib.Path(args.acc).name} / {pathlib.Path(args.gyro).name}"
    else:
        imu_raw = synth_imu_window(window, sr, np.random.default_rng(args.seed + 999))
        imu_src = "IMU tu sinh (sin + rung)"

    normalizer = ImuNormalizer.identity()
    if args.normalize:
        norm_path = pathlib.Path(cfg["data"]["norm_stats"])
        if norm_path.exists():
            normalizer = ImuNormalizer.from_yaml(norm_path)
            print(f"norm_stats : {norm_path}")
        else:
            print(f"[canh bao] khong thay {norm_path} -> khong chuan hoa (identity)")
    imu_clean = normalizer(imu_raw)

    # dung 1 rng cho ca hai, dung thu tu nhu dataset.__getitem__
    rng = np.random.default_rng(args.seed)
    img_noisy = corrupt_image(img_clean, rng, cfg["corruption"]["image"])
    imu_noisy = corrupt_imu(imu_clean, rng, cfg["corruption"]["imu"])

    print(f"anh        : {args.image}")
    print(f"seed nhieu : {args.seed} (adhoc, khong theo _stable_seed cua dataset)")
    return {
        "img_clean": img_clean, "img_noisy": img_noisy,
        "imu_clean": imu_clean, "imu_noisy": imu_noisy,
        "title": f"adhoc: {pathlib.Path(args.image).name}  (seed {args.seed})",
        "imu_src": imu_src + ("" if args.normalize else " - don vi goc, chua chuan hoa"),
    }


# --------------------------------------------------------------------------- #
def open_in_viewer(path: pathlib.Path) -> None:
    """Mo file bang trinh xem anh cua he dieu hanh (xdg-open / open / startfile)."""
    p = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", p])
        elif sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        else:
            opener = shutil.which("xdg-open") or shutil.which("eog") or shutil.which("feh")
            if not opener:
                print(f"  (khong co trinh xem anh; mo thu cong: {p})")
                return
            subprocess.Popen([opener, p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  -> da mo {path.name} bang trinh xem anh")
    except Exception as e:                                     # noqa: BLE001
        print(f"  (khong tu mo duoc: {e}; mo thu cong: {p})")


def build_figure(plt, s: dict, sr: int, cfg: dict):
    img_c, img_n = s["img_clean"], s["img_noisy"]
    imu_c, imu_n = s["imu_clean"], s["imu_noisy"]
    window = imu_c.shape[0]
    resid = imu_n - imu_c
    rmse = np.sqrt(np.mean(resid ** 2, axis=0))

    img_mse = float(np.mean((img_n - img_c) ** 2))
    img_psnr = -10.0 * np.log10(max(img_mse, 1e-10))

    fig = plt.figure(figsize=(13.5, 13.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.35, 1, 1, 0.85])

    def imshow(ax, chw, title):
        ax.imshow(np.clip(chw.transpose(1, 2, 0), 0, 1))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    imshow(fig.add_subplot(gs[0, 0]), img_c, "anh SACH (target)")
    imshow(fig.add_subplot(gs[0, 1]), img_n, "anh NHIEU (input context)")
    ax_d = fig.add_subplot(gs[0, 2])
    im = ax_d.imshow(np.abs(img_n - img_c).mean(axis=0), cmap="magma")
    ax_d.set_title(f"|nhieu - sach|   PSNR {img_psnr:.2f} dB", fontsize=10)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    fig.colorbar(im, ax=ax_d, fraction=0.046)

    tt = np.arange(window) / sr
    for i, name in enumerate(CH_NAMES):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        ax.plot(tt, imu_c[:, i], label="sach", lw=1.5, color="#1b7837")
        ax.plot(tt, imu_n[:, i], label="nhieu", lw=1.0, color="#d6604d", alpha=0.9)
        ax.set_title(f"{name}   RMSE {rmse[i]:.4f}", fontsize=9)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    # hang 4: sai lech (nhieu - sach) cho acc / gyro + RMSE tung kenh
    for j, (lo, hi, lab) in enumerate([(0, 3, "acc"), (3, 6, "gyro")]):
        ax = fig.add_subplot(gs[3, j])
        for i in range(lo, hi):
            ax.plot(tt, resid[:, i], lw=1.0, label=CH_NAMES[i])
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_title(f"sai lech {lab}: nhieu - sach  (bias + drift + spike)", fontsize=9)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=3, loc="best")

    ax_b = fig.add_subplot(gs[3, 2])
    colors = ["#1b7837"] * 3 + ["#4575b4"] * 3
    ax_b.bar(CH_NAMES, rmse, color=colors)
    ax_b.set_title("RMSE tung kenh IMU", fontsize=9)
    ax_b.tick_params(axis="x", rotation=45, labelsize=8)
    ax_b.tick_params(axis="y", labelsize=8)
    ax_b.grid(alpha=0.25, axis="y")

    ci, cu = cfg["corruption"]["image"], cfg["corruption"]["imu"]
    fig.suptitle(
        "TRUOC / SAU bo lam nhieu (qwt_jepa/data/corruption.py)\n"
        f"{s['title']}\n"
        f"image: gauss{ci['gauss_std']} blur{ci['blur_kernel']} jpeg_q{ci['jpeg_q']}   |   "
        f"imu: gauss{cu['gauss_std']} bias{cu['bias']} drift{cu['drift']} "
        f"spike_prob {cu['spike_prob']}",
        fontsize=11,
    )
    return fig, img_mse, img_psnr, rmse, resid


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(_ROOT / "qwt_jepa" / "configs" / "base.yaml"))
    # --- che do dataset (mac dinh) ---
    ap.add_argument("--split", default="valid", choices=["train", "valid", "val", "test"],
                    help="split lay sample (che do dataset)")
    ap.add_argument("--index", type=int, default=-1, help="chi so sample; -1 = ngau nhien")
    ap.add_argument("--epoch", type=int, default=0,
                    help="epoch dung cho seed nhieu (giong train_ds.set_epoch)")
    # --- che do adhoc ---
    ap.add_argument("--image", default="", help="CHE DO ADHOC: file anh bat ky ngoai dataset")
    ap.add_argument("--acc", default="", help="adhoc: acc.txt [N,3]")
    ap.add_argument("--gyro", default="", help="adhoc: gyro.txt [N,3]")
    ap.add_argument("--imu-start", type=int, default=-1,
                    help="adhoc: chi so bat dau cua so IMU (-1 = cua so cuoi)")
    ap.add_argument("--normalize", action="store_true", help="adhoc: chuan hoa IMU bang norm_stats")
    ap.add_argument("--seed", type=int, default=0, help="adhoc: seed rng")
    # --- hien thi ---
    ap.add_argument("--save", nargs="?", const="corruption_demo.png", default=None, metavar="PATH",
                    help="CUNG ghi PNG (mac dinh ./corruption_demo.png)")
    ap.add_argument("--no-window", action="store_true",
                    help="khong mo cua so; chi ghi PNG + mo bang trinh xem anh")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    img_size = int(cfg["data"]["image"]["train_crop"][0])
    window = int(cfg["data"]["imu"]["window_size"])
    sr = int(cfg["data"]["imu"]["sampling_rate"])

    print(f"config     : {args.config}")
    if args.image:
        print("che do     : ADHOC (anh ngoai dataset)")
        s = sample_adhoc(cfg, args, img_size, window, sr)
    else:
        print("che do     : DATASET (dung PairedNoisyCleanDataset.__getitem__ - y het luc train)")
        s = sample_from_dataset(cfg, args.split, args.index, args.epoch)

    # ---- ve ----
    want_window = not args.no_window
    try:
        import matplotlib
        if not want_window:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("can matplotlib de ve: pip install matplotlib")

    fig, img_mse, img_psnr, rmse, resid = build_figure(plt, s, sr, cfg)

    print(f"\nanh   [3,{img_size},{img_size}]  MSE {img_mse:.5f} | PSNR {img_psnr:.2f} dB | "
          f"max|diff| {np.abs(s['img_noisy'] - s['img_clean']).max():.3f}")
    print(f"IMU   [{window},6] @ {sr} Hz   {s['imu_src']}")
    for i, name in enumerate(CH_NAMES):
        print(f"      {name:7s} RMSE {rmse[i]:.4f} | bias(mean) {resid[:, i].mean():+.4f} | "
              f"max|diff| {np.abs(resid[:, i]).max():.4f}")

    def _save(path: pathlib.Path) -> None:
        fig.savefig(path, dpi=120)
        print(f"\nluu -> {path}")

    saved: pathlib.Path | None = None
    if args.save:
        saved = pathlib.Path(args.save).resolve()
        _save(saved)

    backend = plt.get_backend().lower()
    if any(k in backend for k in ("inline", "nbagg", "ipympl", "widget")):
        plt.show()                                             # Jupyter -> inline, xong
        return

    if not want_window:
        if saved is None:
            saved = pathlib.Path("corruption_demo.png").resolve()
            _save(saved)
        open_in_viewer(saved)
        return

    import time

    t0 = time.time()
    try:
        plt.show()                                             # cua so GUI, block den khi dong
    except Exception as e:                                     # noqa: BLE001
        print(f"[canh bao] loi mo cua so matplotlib: {e}")
    if time.time() - t0 < 1.0:
        print("[canh bao] khong mo duoc cua so truc tiep -> ghi PNG va mo trinh xem anh. "
              "Trong Jupyter/VSCode Interactive dung '%matplotlib inline'.")
        if saved is None:
            saved = pathlib.Path("corruption_demo.png").resolve()
            _save(saved)
        open_in_viewer(saved)


if __name__ == "__main__":
    main()
