"""Do bien do he so QWT cua TUNG dai (Phase G).

Van de: bien do he so chenh nhau ~100 lan giua cac dai. Do tren tap valid:

    dai        token   % token   % nang luong   std he so
    L3 LL          4      1.6%        94.10%      2.3130
    L1 LH         64     25.0%         0.54%      0.0625
    L1 HH         64     25.0%         0.08%      0.0242

`Tokenizer.image_proj` la MOT nn.Linear dung chung cho moi dai, nen he so L1
(std 0.04) di qua no cho activation nho xiu so voi L3 LL (std 2.31) -> encoder
gan nhu khong nhin thay chi tiet min -> anh tai tao bi mo.

Chia moi dai cho std rieng cua no truoc khi vao Linear (va nhan lai o recon head)
dua tat ca ve cung thang do.

Cac he so nay la thong ke cua DU LIEU nen phai do tren tap train, giong norm_stats.
"""

from __future__ import annotations

import torch

_IMG_KEY = "L{level}/{band}"
_IMU_KEY = "{group}/L{level}/{band}"


def image_key(level: int, band: str) -> str:
    return _IMG_KEY.format(level=level, band=band)


def imu_key(group: str, level: int, band: str) -> str:
    return _IMU_KEY.format(group=group, level=level, band=band)


@torch.no_grad()
def compute_band_scales(loader, model_qwt, layout, n_batches: int = 8) -> dict:
    """std cua phan vector (i, j, k) cho tung dai, do tren du lieu SACH.

    `model_qwt` la ham (img, imu) -> (qwt_image, qwt_imu) - dung QwtJepa._qwt_all
    de chac chan khop voi luc train.
    """
    acc: dict[str, list[float]] = {}

    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        q_img, q_imu = model_qwt(batch["img_clean"], batch["imu_clean"])
        for e in layout.image_entries:
            v = q_img[e.level][e.band][:, 1:4]
            acc.setdefault(image_key(e.level, e.band), []).append(float(v.std()))
        for e in layout.imu_entries:
            v = q_imu[e.group][e.level][e.band][:, :, 1:4]
            acc.setdefault(imu_key(e.group, e.level, e.band), []).append(float(v.std()))

    # trung binh cac batch; chan duoi de khong bao gio chia cho ~0
    return {k: max(sum(v) / len(v), 1e-3) for k, v in acc.items()}


def scales_to_tensors(scales: dict, layout) -> tuple[torch.Tensor, torch.Tensor]:
    """dict -> hai tensor xep dung thu tu layout.image_entries / imu_entries."""
    img = torch.tensor(
        [scales[image_key(e.level, e.band)] for e in layout.image_entries],
        dtype=torch.float32,
    )
    imu = torch.tensor(
        [scales[imu_key(e.group, e.level, e.band)] for e in layout.imu_entries],
        dtype=torch.float32,
    )
    return img, imu
