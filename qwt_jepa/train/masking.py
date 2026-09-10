"""Sinh context mask + target blocks kieu I-JEPA (Phase D - Buoc D3).

KHONG che pixel ngau nhien. Thay vao do:
  - Anh : chon vai vung chu nhat lien tuc tren luoi khong gian (chuan hoa [0,1]^2),
          map cung mot vung do sang MOI dai / MOI muc -> target la patch roi vao vung.
  - IMU : chon vai doan thoi gian lien tuc (chuan hoa [0,1]), map sang moi dai IMU.
  - Context = phan token con lai, co the lay mau thua theo `context_keep_ratio`.

Cho phep masking cheo modality mot cach tu nhien: target co the gom ca token anh
lan IMU, context cung vay (bat/tat bang cross_modal_masking - khi tat thi target
chi lay tu 1 phia).

Mask dung chung cho ca batch (index 1D) de batch hoa don gian - chap nhan duoc
cho prototype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..models.layout import TokenLayout


@dataclass
class MaskIndices:
    context_index: torch.Tensor        # LongTensor [n_ctx]  (da lay mau thua) -> vao encoder
    target_index: torch.Tensor         # LongTensor [n_tgt]
    context_index_full: torch.Tensor   # LongTensor            (toan bo token khong phai target)


def _rand(lo: float, hi: float, gen: torch.Generator | None) -> float:
    return lo + (hi - lo) * float(torch.rand(1, generator=gen).item())


def coarse_tokens(layout: TokenLayout) -> set[int]:
    """Token cua dai THO nhat: anh = LL, imu = A.

    Chi la vai token (anh: 4 / 512) nhung mang gan het do sang va bo cuc. Neu bi
    vut va thay bang `missing_token` thi recon head khong con duong nao dung lai
    anh: do thuc te cho thay tran PSNR (he so hoan hao, chi mat token bi vut)
    tut tu ~25 dB xuong ~17.5 dB, tuc la THAP HON ca anh nhieu dau vao (20.9 dB).
    Vi vay luon giu chung trong context, khong bao gio lam target.
    """
    out: set[int] = set()
    for e in layout.image_entries:
        if e.band == "LL":
            out.update(range(e.start, e.end))
    for e in layout.imu_entries:
        if e.band == "A":
            out.update(range(e.start, e.end))
    return out


def _image_target_tokens(
    layout: TokenLayout,
    n_blocks: int,
    scale_range: tuple[float, float],
    aspect_range: tuple[float, float],
    gen: torch.Generator | None,
) -> set[int]:
    rects: list[tuple[float, float, float, float]] = []
    for _ in range(n_blocks):
        area = _rand(scale_range[0], scale_range[1], gen)
        log_ar = _rand(math.log(aspect_range[0]), math.log(aspect_range[1]), gen)
        ar = math.exp(log_ar)
        w = min(1.0, math.sqrt(area * ar))
        h = min(1.0, math.sqrt(area / ar))
        x0 = _rand(0.0, 1.0 - w, gen)
        y0 = _rand(0.0, 1.0 - h, gen)
        rects.append((x0, y0, x0 + w, y0 + h))

    tokens: set[int] = set()
    for e in layout.image_entries:
        for iy in range(e.gh):
            ny = (iy + 0.5) / e.gh
            for ix in range(e.gw):
                nx = (ix + 0.5) / e.gw
                for (x0, y0, x1, y1) in rects:
                    if x0 <= nx <= x1 and y0 <= ny <= y1:
                        tokens.add(e.start + iy * e.gw + ix)
                        break
    return tokens


def _imu_target_tokens(
    layout: TokenLayout,
    n_segments: int,
    len_range: tuple[float, float],
    gen: torch.Generator | None,
) -> set[int]:
    segs: list[tuple[float, float]] = []
    for _ in range(n_segments):
        length = _rand(len_range[0], len_range[1], gen)
        t0 = _rand(0.0, 1.0 - length, gen)
        segs.append((t0, t0 + length))

    tokens: set[int] = set()
    for e in layout.imu_entries:
        for i in range(e.length):
            nt = (i + 0.5) / e.length
            for (t0, t1) in segs:
                if t0 <= nt <= t1:
                    tokens.add(e.start + i)
                    break
    return tokens


def sample_masks(
    layout: TokenLayout,
    cfg: dict,
    generator: torch.Generator | None = None,
) -> MaskIndices:
    m = cfg["masking"]
    cross_modal = bool(cfg["model"].get("cross_modal_masking", True))

    img_tgt = _image_target_tokens(
        layout,
        int(m["image_target_blocks"]),
        tuple(m["image_target_scale"]),
        tuple(m.get("image_target_aspect", (0.75, 1.5))),
        generator,
    )
    imu_tgt = _imu_target_tokens(
        layout,
        int(m["imu_target_segments"]),
        tuple(m["imu_target_len"]),
        generator,
    )

    if cross_modal:
        target = img_tgt | imu_tgt
    else:
        # chon ngau nhien 1 phia lam target moi lan goi
        target = img_tgt if torch.rand(1, generator=generator).item() < 0.5 else imu_tgt

    all_idx = set(range(layout.n_tokens))
    protect = coarse_tokens(layout) if bool(m.get("protect_coarse", True)) else set()
    target = target - protect
    target = target or {next(iter(all_idx - protect))}
    context_full = sorted(all_idx - target)
    if not context_full:                       # an toan: chua bao gio de trong
        context_full = [sorted(target)[0]]
        target = target - {context_full[0]}

    # Lay mau thua trong nhom KHONG duoc bao ve, roi ghep nguyen nhom bao ve vao.
    # Tong so token giu lai van xap xi keep_ratio * len(context_full) nhu cu.
    keep_ratio = float(m["context_keep_ratio"])
    droppable = [i for i in context_full if i not in protect]
    n_protected = len(context_full) - len(droppable)
    n_keep = max(0, min(len(droppable), int(round(len(context_full) * keep_ratio)) - n_protected))
    perm = torch.randperm(len(droppable), generator=generator)[:n_keep]
    context_kept = sorted(protect.intersection(context_full).union(
        droppable[i] for i in perm.tolist()
    ))
    if not context_kept:                       # an toan: context khong duoc rong
        context_kept = [context_full[0]]

    return MaskIndices(
        context_index=torch.tensor(context_kept, dtype=torch.long),
        target_index=torch.tensor(sorted(target), dtype=torch.long),
        context_index_full=torch.tensor(context_full, dtype=torch.long),
    )
