"""Bo cuc token (Phase C).

`TokenLayout` mo ta thu tu / vi tri co dinh cua token sau khi noi tat ca dai QWT
cua ca hai modality lai. Bo cuc chi phu thuoc config (kich thuoc anh, patch, levels;
do dai IMU, levels) nen tinh 1 lan roi tokenizer / masking / recon head dung chung.

Thu tu token:  [ tat ca token ANH ]  roi  [ tat ca token IMU (acc roi gyro) ]

Voi anh, moi (level, band) chia thanh luoi patch p x p.
Voi IMU, moi (group, level, band) lay tung buoc thoi gian lam 1 token.

Band theo muc:
  anh : level < L -> [LH, HL, HH] ; level == L -> [LL, LH, HL, HH]
  imu : level < L -> [D]          ; level == L -> [A, D]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

MOD_IMAGE, MOD_IMU_ACC, MOD_IMU_GYRO = 0, 1, 2


@dataclass
class Entry:
    kind: str          # "image" | "imu"
    level: int
    band: str
    start: int
    end: int
    group: str = ""    # "" | "acc" | "gyro"
    gh: int = 0        # anh: so patch theo chieu cao
    gw: int = 0        # anh: so patch theo chieu rong
    length: int = 0    # imu: so buoc thoi gian


def _image_bands(level: int, levels: int) -> list[str]:
    return (["LL"] if level == levels else []) + ["LH", "HL", "HH"]


def _imu_bands(level: int, levels: int) -> list[str]:
    return (["A"] if level == levels else []) + ["D"]


@dataclass
class TokenLayout:
    entries: list[Entry]
    modality: torch.Tensor          # LongTensor [N]
    scale_id: torch.Tensor          # LongTensor [N]
    n_tokens: int
    num_scales: int
    patch: int
    levels_image: int
    levels_imu: int
    image_shapes: dict              # {level: (h, w)} kich thuoc dau vao moi muc (anh)
    imu_shapes: dict                # {level: t}      do dai dau vao moi muc (imu)
    image_entries: list[Entry] = field(default_factory=list)
    imu_entries: list[Entry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.image_entries = [e for e in self.entries if e.kind == "image"]
        self.imu_entries = [e for e in self.entries if e.kind == "imu"]


def build_layout(cfg: dict) -> TokenLayout:
    img = cfg["data"]["image"]
    imu = cfg["data"]["imu"]
    patch = int(cfg["model"]["token_patch_image"])

    img_size = int(img["train_crop"][0])
    levels_image = int(img["qwt_levels"])
    t_size = int(imu["window_size"])
    levels_imu = int(imu["qwt_levels"])

    # Kich thuoc dau vao moi muc.
    image_shapes: dict = {}
    s = img_size
    for lvl in range(1, levels_image + 1):
        image_shapes[lvl] = (s, s)
        s = (s + 1) // 2
    imu_shapes: dict = {}
    t = t_size
    for lvl in range(1, levels_imu + 1):
        imu_shapes[lvl] = t
        t = (t + 1) // 2

    entries: list[Entry] = []
    scale_reg: dict = {}

    def scale_of(key: tuple) -> int:
        if key not in scale_reg:
            scale_reg[key] = len(scale_reg)
        return scale_reg[key]

    idx = 0
    mod_list: list[int] = []
    scale_list: list[int] = []

    # ---- token ANH ----
    for lvl in range(1, levels_image + 1):
        band_h = (image_shapes[lvl][0] + 1) // 2
        if band_h % patch != 0:
            raise ValueError(
                f"muc {lvl}: dai anh {band_h} khong chia het cho patch {patch}"
            )
        gh = gw = band_h // patch
        for band in _image_bands(lvl, levels_image):
            cnt = gh * gw
            entries.append(Entry("image", lvl, band, idx, idx + cnt, gh=gh, gw=gw))
            sid = scale_of(("image", lvl, band))
            mod_list += [MOD_IMAGE] * cnt
            scale_list += [sid] * cnt
            idx += cnt

    # ---- token IMU ----
    for group in ("acc", "gyro"):
        mod = MOD_IMU_ACC if group == "acc" else MOD_IMU_GYRO
        for lvl in range(1, levels_imu + 1):
            band_len = (imu_shapes[lvl] + 1) // 2
            for band in _imu_bands(lvl, levels_imu):
                entries.append(
                    Entry("imu", lvl, band, idx, idx + band_len, group=group, length=band_len)
                )
                sid = scale_of(("imu", group, lvl, band))
                mod_list += [mod] * band_len
                scale_list += [sid] * band_len
                idx += band_len

    return TokenLayout(
        entries=entries,
        modality=torch.tensor(mod_list, dtype=torch.long),
        scale_id=torch.tensor(scale_list, dtype=torch.long),
        n_tokens=idx,
        num_scales=len(scale_reg),
        patch=patch,
        levels_image=levels_image,
        levels_imu=levels_imu,
        image_shapes=image_shapes,
        imu_shapes=imu_shapes,
    )
