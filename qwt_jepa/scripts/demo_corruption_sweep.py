"""Quet (ablation) tung tham so trong khoi `corruption:` - xem no anh huong the nao.

Moi HANG doi DUNG MOT tham so, cac tham so con lai bi vo hieu (dat 1e-9 de van di
qua cung nhanh code -> cung so lan boc rng -> so sanh trong 1 hang la cong bang).
Panel duoc danh dau [<- base.yaml] la gia tri dang dung trong config hien tai.

    python qwt_jepa/scripts/demo_corruption_sweep.py                  # sample valid ngau nhien
    python qwt_jepa/scripts/demo_corruption_sweep.py --index 100      # chon dung sample
    python qwt_jepa/scripts/demo_corruption_sweep.py --save-dir .     # ghi 2 file PNG

Can torch (che do dataset):  /home/buidinhkhoi/anaconda3/bin/python ...
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_HERE = pathlib.Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("_dc", _HERE / "demo_corruption.py")
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)

CH = dc.CH_NAMES


# --- config rut gon: chi bat 1 hieu ung, cac cai khac ~0 nhung VAN di qua nhanh code ---
def cfg_img(gauss=1e-9, blur=0, q=95):
    return {"gauss_std": [gauss, gauss], "blur_kernel": [blur, blur], "jpeg_q": [q, q]}


def cfg_imu(gauss=1e-9, bias=1e-9, drift=1e-9, spike=1e-9):
    return {"gauss_std": [gauss, gauss], "bias": [0.0, bias],
            "drift": [0.0, drift], "spike_prob": spike}


IMG_ROWS = [
    ("gauss_std  -  nhieu hat (cam bien anh sang yeu / ISO cao)",
     [0.02, 0.05, 0.08, 0.14, 0.20], lambda v: cfg_img(gauss=v), "{:.2f}"),
    ("blur_kernel  -  nhoe chuyen dong 1 chieu (rung tay / phoi sang lau)",
     [2, 4, 7, 10, 13], lambda v: cfg_img(blur=v), "{:d}"),
    ("jpeg_q  -  luong tu hoa kieu nen (q THAP = nhieu banding hon)",
     [95, 60, 40, 25, 12], lambda v: cfg_img(q=v), "{:d}"),
]

IMU_ROWS = [
    ("gauss_std  -  nhieu trang tung mau (noise floor cua cam bien)",
     [0.01, 0.03, 0.06, 0.10, 0.15], lambda v: cfg_imu(gauss=v), "{:.3f}"),
    ("bias  -  lech HANG SO suot ca cua so (offset chua hieu chuan)",
     [0.01, 0.03, 0.05, 0.10, 0.15], lambda v: cfg_imu(bias=v), "{:.3f}"),
    ("drift  -  troi CHAM kieu random walk (nhiet do / tich luy sai so)",
     [0.01, 0.02, 0.035, 0.06, 0.10], lambda v: cfg_imu(drift=v), "{:.3f}"),
    ("spike_prob  -  xung dot bien; bien do = 5 x gauss_std (o day ghim gauss_std=0.05)",
     [0.005, 0.015, 0.03, 0.06, 0.12], lambda v: cfg_imu(gauss=0.05, spike=v), "{:.3f}"),
]


def _mark(v, cur) -> str:
    try:
        return "   [<- base.yaml]" if abs(float(v) - float(cur)) < 1e-9 else ""
    except (TypeError, ValueError):
        return ""


def fig_image(plt, img_clean, cfg, seed):
    ci = cfg["corruption"]["image"]
    cur = [ci["gauss_std"][1], ci["blur_kernel"][1], ci["jpeg_q"][0]]
    fig, axes = plt.subplots(len(IMG_ROWS), 6, figsize=(17, 9.2), constrained_layout=True)
    for r, (label, vals, mk, fmt) in enumerate(IMG_ROWS):
        ax = axes[r][0]
        ax.imshow(img_clean.transpose(1, 2, 0))
        ax.set_title("SACH (goc)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"hang {r + 1}", fontsize=9)
        for c, v in enumerate(vals):
            out = dc.corrupt_image(img_clean, np.random.default_rng(seed), mk(v))
            mse = float(np.mean((out - img_clean) ** 2))
            psnr = -10.0 * np.log10(max(mse, 1e-10))
            ax = axes[r][c + 1]
            ax.imshow(out.transpose(1, 2, 0))
            ax.set_title(f"{fmt.format(v)}   PSNR {psnr:.1f} dB{_mark(v, cur[r])}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r][0].text(0.0, 1.22, label, transform=axes[r][0].transAxes,
                        fontsize=10.5, fontweight="bold")
    fig.suptitle("ANH: doi 1 tham so moi hang, cac tham so khac tat "
                 "(qwt_jepa/data/corruption.py)", fontsize=13)
    return fig


def fig_imu(plt, imu_clean, cfg, seed, ch, sr):
    cu = cfg["corruption"]["imu"]
    cur = [cu["gauss_std"][1], cu["bias"][1], cu["drift"][1], cu["spike_prob"]]
    i = CH.index(ch)
    t = np.arange(imu_clean.shape[0]) / sr
    fig, axes = plt.subplots(len(IMU_ROWS), 5, figsize=(18, 11.5),
                             constrained_layout=True, sharex=True)
    for r, (label, vals, mk, fmt) in enumerate(IMU_ROWS):
        for c, v in enumerate(vals):
            out = dc.corrupt_imu(imu_clean, np.random.default_rng(seed), mk(v))
            rmse = float(np.sqrt(np.mean((out[:, i] - imu_clean[:, i]) ** 2)))
            ax = axes[r][c]
            ax.plot(t, imu_clean[:, i], lw=1.6, color="#1b7837", label="sach")
            ax.plot(t, out[:, i], lw=1.0, color="#d6604d", alpha=0.9, label="nhieu")
            ax.set_title(f"{fmt.format(v)}   RMSE {rmse:.4f}{_mark(v, cur[r])}", fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
            if r == len(IMU_ROWS) - 1:
                ax.set_xlabel("t (s)", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")
        axes[r][0].text(0.0, 1.18, label, transform=axes[r][0].transAxes,
                        fontsize=10.5, fontweight="bold")
    fig.suptitle(f"IMU (kenh {ch}): doi 1 tham so moi hang, cac tham so khac tat", fontsize=13)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_ROOT / "qwt_jepa" / "configs" / "base.yaml"))
    ap.add_argument("--split", default="valid", choices=["train", "valid", "val", "test"])
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--image", default="", help="che do adhoc: anh bat ky")
    ap.add_argument("--channel", default="acc_x", choices=CH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None, metavar="DIR",
                    help="ghi corruption_sweep_image.png / _imu.png vao DIR")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    sr = int(cfg["data"]["imu"]["sampling_rate"])

    if args.image:
        size = int(cfg["data"]["image"]["train_crop"][0])
        win = int(cfg["data"]["imu"]["window_size"])
        img_clean = dc.load_image(args.image, size)
        imu_clean = dc.synth_imu_window(win, sr, np.random.default_rng(args.seed + 999))
        print(f"che do ADHOC: {args.image}")
    else:
        s = dc.sample_from_dataset(cfg, args.split, args.index, args.epoch)
        img_clean, imu_clean = s["img_clean"], s["imu_clean"]

    want_window = not args.no_window
    import matplotlib
    if not want_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1 = fig_image(plt, img_clean, cfg, args.seed)
    f2 = fig_imu(plt, imu_clean, cfg, args.seed, args.channel, sr)

    if args.save_dir:
        d = pathlib.Path(args.save_dir).resolve()
        d.mkdir(parents=True, exist_ok=True)
        for f, n in [(f1, "corruption_sweep_image.png"), (f2, "corruption_sweep_imu.png")]:
            f.savefig(d / n, dpi=110)
            print(f"luu -> {d / n}")

    if want_window:
        plt.show()


if __name__ == "__main__":
    main()
