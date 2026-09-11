"""Corruption on-the-fly (Phase B - Buoc B3).

Ap trong MIEN GOC, TRUOC QWT. Nhan `rng` (numpy Generator) de tai lap.
Khong bao gio ghi de file sach; khong tao nhieu truc tiep tren he so QWT.

ANH - mo phong theo dung duong di VAT LY cua anh sang qua mot camera that:
    1. motion blur   xay ra luc PHOI SANG, o quang hoc, TRUOC khi anh sang toi sensor
    2. exposure      nhan trong mien LINEAR-LIGHT (khong phai tren gia tri sRGB), clip highlight
    3. shot noise    nhieu photon (Poisson): phuong sai TI LE voi tin hieu -> vung sang nhieu hat hon
    4. read noise    nhieu mach doc: Gaussian hang so, khong phu thuoc do sang
    5. chroma noise  dom mau tan so thap, dac trung anh ISO cao
    6. JPEG THAT     8x8 DCT + chroma subsampling qua PIL -> blocking + ringing quanh canh

  Nhieu phai nam SAU blur. Ban cu lam nguoc (nhieu -> blur) nen hat nhieu bi lam min,
  trong nhu grain da qua xu ly chu khong phai nhieu sensor.

IMU - mo hinh sai so quan tinh tieu chuan (kieu Allan variance):
    NHAN:  scale factor error + misalignment (dac tinh co dinh cua tung con chip)
    CONG:  white noise + turn-on bias + bias instability (random walk) + spike
"""

from __future__ import annotations

import functools
import io

import numpy as np
from PIL import Image

def _sev(rng, rr, k: float, invert: bool = False, u=None, couple: float = 0.0) -> float:
    """Boc CUONG DO nhieu, lech ve phia NHE.

    Thuc te da so khung hinh chi hoi nhieu, thinh thoang moi gap khung that te (thieu sang,
    rung manh, nen vo). Phan phoi DEU cho ra so khung 'te' bang so khung 'nhe' - khong giong
    du lieu that, va bat model danh qua nhieu suc cho truong hop cuc doan.

    Dung u^k (k>1) don mau ve phia `lo`; k=1 tro lai dung phan phoi deu. Cach nay xu ly duoc
    ca lo=0 (khac voi log-uniform, von khong dinh nghia tai 0).
    `invert=True` cho tham so NGUOC CHIEU nhu jpeg_q, noi 'nhe' nam o dau `hi`.
    """
    lo, hi = float(rr[0]), float(rr[1])
    v = float(rng.random()) ** k
    if u is not None and couple > 0.0:
        # Tron voi bien AN chung `u` -> cac hieu ung di CUNG NHAU thay vi doc lap.
        v = couple * float(u) + (1.0 - couple) * v
    return hi - (hi - lo) * v if invert else lo + (hi - lo) * v


def _hit(rng, cfg: dict, key: str, default: float = 1.0) -> bool:
    """Hieu ung nay CO XAY RA voi sample nay khong?

    Tach XAC SUAT XAY RA khoi CUONG DO. Truoc day moi sample deu dinh moi hieu ung, chi
    khac nhau o muc do - nen khong bao gio co khung anh that su NET, khong bao gio co cua
    so IMU that su SACH. Thuc te thi co: phan lon khung hinh khong nhoe, phan lon cua so
    khong bi lech bias dang ke. Model can thay ca hai loai de biet khi nao KHONG nen sua.
    """
    return float(rng.random()) < float(cfg.get(key, default))


_GAMMA = 2.0          # xap xi sRGB (that su ~2.2). Dung 2.0 de encode/decode chi la
                      # binh phuong / can bac hai - nhanh hon np.power(x, 2.2) nhieu lan,
                      # sai khac khong dang ke cho muc dich mo phong nhieu.


# --------------------------------------------------------------------------- #
# Anh: [3, H, W] float32 trong [0, 1]
# --------------------------------------------------------------------------- #
def _motion_blur(x: np.ndarray, length: int, angle_deg: float, rng) -> np.ndarray:
    """PSF theo DUONG DI THUC cua camera trong luc phoi sang.

    Rung tay khong bao gio la mot doan thang deu: quy dao ngoan ngoeo (random walk quanh
    huong chinh) va toc do thay doi (cham o hai dau, nhanh o giua, hoac nguoc lai) nen
    cac diem tren vet nhoe co trong so KHAC nhau. PSF duong thang trong so deu tao ra
    vet nhoe qua 'sach' - mat nhin ra ngay la do may tinh sinh.
    """
    if length < 2:
        return x
    ang = np.deg2rad(angle_deg)
    dy, dx = float(np.sin(ang)), float(np.cos(ang))
    jit = 0.35 * length                                   # do ngoan ngoeo cua quy dao
    wy = np.cumsum(rng.normal(0.0, jit / max(length, 1), length))
    wx = np.cumsum(rng.normal(0.0, jit / max(length, 1), length))
    wts = rng.uniform(0.4, 1.0, length)                   # toc do khong deu -> trong so lech
    wts /= wts.sum()
    offs = []
    for i in range(length):
        t = i - (length - 1) / 2.0
        offs.append((int(round(t * dy + wy[i])), int(round(t * dx + wx[i]))))
    m = max(1, max(max(abs(a), abs(b)) for a, b in offs))
    xp = np.pad(x, ((0, 0), (m, m), (m, m)), mode="edge")
    h, w = x.shape[1], x.shape[2]
    acc = np.zeros_like(x)
    for (sy, sx), wt in zip(offs, wts):
        acc += np.float32(wt) * xp[:, m + sy:m + sy + h, m + sx:m + sx + w]
    return acc


def _smooth_up(small: np.ndarray, h: int, w: int) -> np.ndarray:
    """Phong to muot bang noi suy BICUBIC. Truoc day dung np.repeat -> ra o vuong
    16x16 canh sac, la van nhan tao de nhan ra nhat trong ca pipeline."""
    out = np.empty((small.shape[0], h, w), np.float32)
    for i in range(small.shape[0]):
        out[i] = np.asarray(
            Image.fromarray(small[i], mode="F").resize((w, h), Image.BICUBIC), dtype=np.float32
        )
    return out


def _chroma_noise(shape: tuple, amp: float, rng: np.random.Generator, block: int = 16) -> np.ndarray:
    """Nhieu MAU tan so thap: sinh o do phan giai thap roi phong to MUOT -> dam may
    mau loang lo, khong con luoi o vuong."""
    c, h, w = shape
    small = rng.normal(0.0, amp, size=(c, h // block + 2, w // block + 2)).astype(np.float32)
    return _smooth_up(small, h, w)


def _chroma_denoise(rgb: np.ndarray, k: int = 1) -> np.ndarray:
    """Buoc LOC MAU cua ISP that: mat nguoi rat it nhay voi chi tiet mau, nen moi ISP
    deu lam min manh kenh chroma sau demosaic. Buoc nay dong thoi xoa van co ro ban co
    (checkerboard) ma demosaic song tuyen tinh de lai - nguon 'bam' ro nhat cua anh."""
    y = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).astype(np.float32)
    ch = rgb - y[None]
    for _ in range(k):
        ch = _sep3(ch, _S_BLUR)
    return (y[None] + ch).astype(np.float32)


def _jpeg(x: np.ndarray, quality: int, oy: int = 0, ox: int = 0) -> np.ndarray:
    """Nen JPEG THAT (8x8 DCT + chroma subsampling), khong phai posterize deu.

    `oy`,`ox` (0..7) doi PHA luoi 8x8 so voi noi dung anh. Neu luoi luon trung goc anh
    thi ranh gioi block luon roi vao dung nhung hang/cot co dinh -> model nho vi tri
    ranh gioi thay vi hoc khu artifact.
    """
    if oy or ox:
        x = np.pad(x, ((0, 0), (oy, 0), (ox, 0)), mode="edge")
    u8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        out = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    out = np.ascontiguousarray(out.transpose(2, 0, 1))
    return out[:, oy:, ox:] if (oy or ox) else out


_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0
_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0
# LUU Y: _K_G va _K_RB co tong = 1 va = 4 (khong chuan hoa) vi trong demosaic chung
# duoc chia cho `nrm`. Dung lam bo loc lam min thi phai dung ban CHUAN HOA nay:
_K_BLUR = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0


def _conv3(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Chap 3x3 tren [C,H,W] bang cong don dich chuyen; bien lap kieu `edge`."""
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = x.shape[1], x.shape[2]
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            c = k[dy, dx]
            if c:
                out += c * xp[:, dy:dy + h, dx:dx + w]
    return out


def _sep3(x: np.ndarray, k) -> np.ndarray:
    """Chap 3x3 TACH DUOC bang 2 lan quet 1-D (6 phep thay vi 9).
    Ap dung cho kernel dang [a,b,a] outer [a,b,a] - tuc _K_RB va _K_BLUR."""
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = x.shape[1], x.shape[2]
    t = k[0] * xp[:, :, 0:w] + k[1] * xp[:, :, 1:w + 1] + k[2] * xp[:, :, 2:w + 2]
    return k[0] * t[:, 0:h] + k[1] * t[:, 1:h + 1] + k[2] * t[:, 2:h + 2]


_S_RB = (np.float32(0.5), np.float32(1.0), np.float32(0.5))    # outer -> _K_RB (tong 4)
_S_BLUR = (np.float32(0.25), np.float32(0.5), np.float32(0.25))  # outer -> _K_BLUR (tong 1)


def _mosaic(lin: np.ndarray, py: int = 0, px: int = 0) -> np.ndarray:
    """RGB [3,H,W] -> 1 kenh Bayer [H,W]: moi pixel chi do DUOC 1 mau, dung nhu cam bien
    that dang sau color filter array.

    `py`,`px` (0/1) chon PHA cua luoi CFA (RGGB / GRBG / GBRG / BGGR). Neu luon co dinh
    RGGB thi vi tri pixel nao duoc do that / pixel nao la noi suy LUON GIONG NHAU -
    model hoc thuoc ban do do va 'go' demosaic bang tri nho, chu khong hoc khu nhieu.
    """
    h, w = lin.shape[1], lin.shape[2]
    qy, qx = py ^ 1, px ^ 1
    b = np.empty((h, w), np.float32)
    b[py::2, px::2] = lin[0, py::2, px::2]      # R
    b[py::2, qx::2] = lin[1, py::2, qx::2]      # G
    b[qy::2, px::2] = lin[1, qy::2, px::2]      # G
    b[qy::2, qx::2] = lin[2, qy::2, qx::2]      # B
    return b


@functools.lru_cache(maxsize=8)
def _cfa(h: int, w: int, py: int = 0, px: int = 0):
    """Mat na CFA va he so chuan hoa - chi phu thuoc kich thuoc anh nen cache lai
    (truoc day tinh lai moi anh, ton mot nua thoi gian demosaic)."""
    qy, qx = py ^ 1, px ^ 1
    m = np.zeros((3, h, w), np.float32)
    m[0, py::2, px::2] = 1.0
    m[1, py::2, qx::2] = 1.0
    m[1, qy::2, px::2] = 1.0
    m[2, qy::2, qx::2] = 1.0
    nrm = np.stack([_sep3(m[0:1], _S_RB)[0], _conv3(m[1:2], _K_G)[0], _sep3(m[2:3], _S_RB)[0]])
    return m, np.maximum(nrm, 1e-6)


def _demosaic(bayer: np.ndarray, py: int = 0, px: int = 0) -> np.ndarray:
    """Noi suy song tuyen tinh Bayer RGGB -> RGB.

    CHINH buoc nay sinh ra dau van tay cua anh camera that ma cong nhieu RGB doc lap
    khong bao gio co: 'zipper' rang cua o canh xien, vien mau (color fringing), va
    nhieu mau BAM DINH KHONG GIAN (moi pixel mau duoc noi suy tu hang xom -> nhieu
    cua chung tuong quan voi nhau, thanh tung dom, khong con doc lap tung pixel)."""
    m, nrm = _cfa(bayer.shape[0], bayer.shape[1], py, px)
    known = m * bayer[None]
    est = np.stack([_sep3(known[0:1], _S_RB)[0], _conv3(known[1:2], _K_G)[0],
                    _sep3(known[2:3], _S_RB)[0]])
    return np.where(m > 0, known, est / nrm).astype(np.float32)   # giu gia tri do duoc


def _illum(shape: tuple, rng, depth: float, block: int = 5) -> np.ndarray:
    """Truong chieu sang KHONG DEU theo khong gian.

    Khac vignette: vignette la dac tinh QUANG HOC (xuyen tam, luon toi o goc), con day
    la do CANH VAT - den o mot phia, vat che bong, may troi qua, cua so hat sang. Vung
    sang nhat giu nguyen, vung toi nhat bi giam toi `depth`. Sinh o do phan giai rat
    thap (block x block) roi phong to bicubic nen la vet sang toi loang, khong phai o vuong.
    """
    c, h, w = shape
    g = rng.random((1, block, block)).astype(np.float32)
    g = _smooth_up(g, h, w)
    g -= g.min()
    g /= max(float(g.max()), 1e-6)                          # chuan ve [0, 1]
    return np.repeat(1.0 - depth * g, c, axis=0)


def _vignette(shape: tuple, strength: float, cy: float = 0.0, cx: float = 0.0,
              aspect: float = 1.0, power: float = 1.0) -> np.ndarray:
    """Lens shading: ria anh toi hon tam (cos^4 xap xi bang da thuc r^2).

    Tam quang hoc LECH khoi tam anh (cy,cx), hoi bat doi xung (aspect), va DANG profile
    doi theo `power` (r^2 -> r^4).

    Luu y: moi ong kinh that deu toi goc, nen bo han thanh phan xuyen tam la KHONG dung
    vat ly - va lech tam thoi cung khong du (trung binh cua nhieu vignette lech tam VAN la
    mot truong xuyen tam). Thu phai bien thien la DO MANH va DANG profile, de model hoc
    mot HO ham chu khong hoc thuoc dung mot truong roi chia lai.
    """
    h, w = shape[1], shape[2]
    yy = (np.linspace(-1.0, 1.0, h, dtype=np.float32) - cy)[:, None]
    xx = (np.linspace(-1.0, 1.0, w, dtype=np.float32) - cx)[None, :]
    r2 = yy * yy * aspect + xx * xx / aspect
    return (1.0 - strength * np.power(r2 * 0.5, power)).astype(np.float32)


def blur_from_gyro(gyro: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[int, float]:
    """Suy do dai + huong motion blur TU GYRO. Tra (length_px, angle_deg).

    Vi sao: nhoe chuyen dong CHINH LA tich phan chuyen dong camera trong thoi gian phoi
    sang, ma gyro do dung thu do. Quay quanh truc x (ngang) -> anh truot doc; quanh truc
    y (doc) -> anh truot ngang. Dich chuyen tren anh = omega * T_phoi_sang * f.

    Truoc day `angle` boc uniform(0, 180) va `length` boc ngau nhien - HOAN TOAN doc lap
    voi IMU. Nghia la khong co quan he nao de model hoc, va toan bo y tuong "dung IMU de
    khu nhoe" khong co co so trong du lieu. Day la cho sua dieu do.

    Do tren du lieu that (gyro |omega|): p25 0.35, p50 0.58, p90 1.69 rad/s.
    Voi f=320 px (640x640, FOV 90 do) va phoi sang 1/50 s -> nhoe 2.3-10.8 px, khop
    voi dai blur_kernel [4, 14] dang dung.
    """
    f = float(cfg.get("focal_px", 320.0))
    t_lo, t_hi = cfg.get("exposure_time", (0.01, 0.025))
    T = float(t_lo) + (float(t_hi) - float(t_lo)) * float(rng.random())

    n = max(1, int(round(T * float(cfg.get("imu_rate", 100.0)))))
    w = gyro[-n:]                                  # cua so phoi sang = sat truoc thoi diem chup
    wx, wy = float(w[:, 0].mean()), float(w[:, 1].mean())

    length = int(round(f * T * float(np.hypot(wx, wy))))
    lo, hi = cfg.get("blur_kernel", (0, 32))
    length = int(np.clip(length, 0, int(hi)))
    angle = float(np.degrees(np.arctan2(wx, wy)))  # quay quanh y -> truot ngang (goc 0)
    return length, angle


def corrupt_image(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: dict,
    gyro: np.ndarray | None = None,
) -> np.ndarray:
    x = img.astype(np.float32, copy=True)

    # 1. motion blur - quang hoc, TRUOC nhieu sensor; goc bat ky (khong chi ngang/doc)
    k = float(cfg.get("severity_skew", 1.0))       # 1.0 = phan phoi deu (nhu cu)

    # DIEU KIEN CHUP: mot bien AN duy nhat chi phoi ca DO SANG, DO NHOE va MUC NHIEU.
    #
    # Camera trong moi truong thieu sang khong co lua chon nao mien phi: phoi sang LAU
    # hon thi duoc sang nhung NHOE hon; tang ISO thi net nhung NHIEU hon; va thuong van
    # ra anh TOI hon muc can. Ba thu do luon di cung nhau trong thuc te.
    #
    # Boc doc lap (nhu truoc) sinh ra duoc anh rat toi nhung NET CANG va SACH BONG -
    # to hop khong the ton tai. Model se hoc phan bo sai.
    # `couple` = 0 tro lai doc lap hoan toan; = 1 rang buoc chat (moi thu suy tu `bad`).
    # `condition` = [san, tran] cua bien an. San > 0 dam bao dieu kien KHONG BAO GIO
    # hoan hao: neu de san = 0 thi khi bad ~ 0 MOI hieu ung cung ve muc toi thieu MOT LUC
    # (blur ~0 px, shot/read min, exposure max) -> sinh ra cap (sach, gan nhu sach) khong
    # day duoc gi. Voi couple > 0 chuyen nay xay ra tap trung chu khong rai rac nhu truoc.
    cpl = float(cfg.get("couple", 0.0))
    c_lo, c_hi = cfg.get("condition", (0.0, 1.0))
    bad = float(c_lo) + (float(c_hi) - float(c_lo)) * float(rng.random()) ** k

    length, angle = 0, float(rng.uniform(0.0, 180.0))
    if _hit(rng, cfg, "p_blur"):                   # phan con lai la khung NET hoan toan
        if bool(cfg.get("blur_from_imu", False)) and gyro is not None:
            # Nhoe suy TU GYRO -> IMU thuc su du doan duoc nhoe, va viec ghep hai
            # modality moi co co so. Xem blur_from_gyro().
            length, angle = blur_from_gyro(gyro, cfg, rng)
        else:
            length = int(round(_sev(rng, cfg.get("blur_kernel", (0, 0)), k, u=bad, couple=cpl)))
    x = _motion_blur(x, length, angle, rng)

    # 2. exposure trong mien LINEAR-LIGHT (nhan tren sRGB la sai vat ly)
    e_lo, e_hi = cfg.get("exposure", (0.7, 1.1))
    ue = cpl * bad + (1.0 - cpl) * float(rng.random())     # dieu kien te -> phoi sang THAP
    xc = np.clip(x, 0.0, 1.0)
    lin = xc * xc * np.float32(e_hi - (e_hi - e_lo) * ue)

    # 3a. chieu sang khong deu theo khong gian (canh vat, khong phai ong kinh)
    il = _sev(rng, cfg.get("illum_depth", (0.0, 0.0)), k) if _hit(rng, cfg, "p_illum", 0.0) else 0.0
    if il > 0:
        lin = lin * _illum(lin.shape, rng, il, int(rng.integers(3, 7)))

    # 3b. vignetting (lens shading)
    v = _sev(rng, cfg.get("vignette", (0.0, 0.0)), k) if _hit(rng, cfg, "p_vignette") else 0.0
    if v > 0:                                      # tam quang hoc + do bat doi xung ngau nhien
        lin = lin * _vignette(lin.shape, v, float(rng.uniform(-0.25, 0.25)),
                              float(rng.uniform(-0.25, 0.25)), float(rng.uniform(0.75, 1.35)),
                              float(rng.uniform(0.7, 1.8)))

    # 4-6. Pipeline RAW kieu "Unprocessing" (Brooks et al., CVPR 2019):
    #      go white balance -> mosaic ve Bayer -> NHIEU o mien RAW -> demosaic -> WB lai.
    #      Nhieu phai sinh o mien RAW moi dung: sau demosaic no tro nen tuong quan
    #      khong gian giua cac pixel va giua cac kenh mau, dung nhu anh may anh that.
    a = np.float32(_sev(rng, cfg.get("shot_noise", (0.0, 0.0)), k, u=bad, couple=cpl))
    s = np.float32(_sev(rng, cfg["gauss_std"], k, u=bad, couple=cpl))
    if cfg.get("bayer", False):
        g_lo, g_hi = cfg.get("wb_gains", (1.2, 2.4))
        gr, gb = float(rng.uniform(g_lo, g_hi)), float(rng.uniform(g_lo, g_hi))
        lin[0] /= gr                                   # go WB -> gia tri RAW that cua sensor
        lin[2] /= gb                                   # (kenh R,B nho hon -> nhieu tuong doi lon hon)
        py, px = int(rng.integers(0, 2)), int(rng.integers(0, 2))   # pha CFA: RGGB/GRBG/GBRG/BGGR
        raw = _mosaic(lin, py, px)
        if a > 0 or s > 0:
            var = a * a * np.maximum(raw, 0.0) + s * s
            raw = raw + rng.standard_normal(raw.shape, dtype=np.float32) * np.sqrt(var)
        lin = _demosaic(raw, py, px)
        # so lan loc mau cua ISP cung ngau nhien: moi hang may ISP mot kieu, va de con so
        # co dinh thi buoc nay la mot toan tu tuyen tinh BIET TRUOC, model dao nguoc duoc.
        cd = cfg.get("chroma_denoise", 1)
        cd = int(rng.integers(int(cd[0]), int(cd[1]) + 1)) if isinstance(cd, (list, tuple)) else int(cd)
        if cd > 0:
            lin = _chroma_denoise(lin, cd)
        lin[0] *= gr                                   # ap lai WB nhu ISP
        lin[2] *= gb
    elif a > 0 or s > 0:
        var = a * a * np.maximum(lin, 0.0) + s * s
        lin += rng.standard_normal(lin.shape, dtype=np.float32) * np.sqrt(var)

    x = np.sqrt(np.clip(lin, 0.0, 1.0))                    # clip = chay sang vung highlight

    # 5. chroma noise
    cn = _sev(rng, cfg.get("chroma_noise", (0.0, 0.0)), k) if _hit(rng, cfg, "p_chroma") else 0.0
    if cn > 0:                                     # co hat dom cung ngau nhien
        x = x + _chroma_noise(x.shape, cn, rng, block=int(rng.integers(8, 33)))

    # 6. JPEG that - buoc cuoi, giong duong di cua anh ra khoi ISP
    # jpeg_q NGUOC CHIEU: q cao = anh dep, nen 'nhe' nam o dau hi -> invert=True
    if _hit(rng, cfg, "p_jpeg"):
        x = _jpeg(x, int(round(_sev(rng, cfg["jpeg_q"], k, invert=True))),
                  int(rng.integers(0, 8)), int(rng.integers(0, 8)))

    return np.clip(x, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# IMU: [T, 6] da chuan hoa (acc | gyro)
# --------------------------------------------------------------------------- #
def _vibration(t: int, c: int, rng, amp: float, f_lo: float, f_hi: float,
               q: float, sr: float) -> np.ndarray:
    """Rung ngau nhien BANG RONG CO DINH CONG HUONG.

    Cach chuan de tong hop random vibration: sinh white noise roi NAN PHO bang ham
    truyen bac 2 |H(f)| = 1/sqrt((1-r^2)^2 + (r/Q)^2), r = f/f0. Dung mo hinh mot
    ket cau co tan so rieng f0 va he so pham chat Q - dong co, canh quat, khung xe,
    mat duong deu tao pho kieu nay. Chuan hoa lai ve RMS = amp de nut van co nghia.
    """
    f = np.fft.rfftfreq(t, d=1.0 / sr).astype(np.float32)
    f0 = float(rng.uniform(f_lo, f_hi))
    r = f / max(f0, 1e-6)
    h = 1.0 / np.sqrt((1.0 - r * r) ** 2 + (r / max(q, 1e-3)) ** 2 + 1e-12)
    # Rung den tu MOT nguon vat ly (dong co, mat duong, canh quat) lam rung ca khoi may.
    # Ba truc cua cung mot cam bien do CUNG mot chuyen dong, chi khac huong chieu, nen
    # chung TUONG QUAN manh. Sinh 6 kenh doc lap la sai vat ly va nhin rat nhan tao.
    w = rng.standard_normal((t, c + 2), dtype=np.float32)
    f_all = np.fft.irfft(np.fft.rfft(w, axis=0) * h[:, None], n=t, axis=0).astype(np.float32)
    src, ind = f_all[:, :2], f_all[:, 2:]          # 2 nguon chung (acc, gyro) + phan rieng
    out = np.empty((t, c), np.float32)
    for g, lo in enumerate((0, 3)):
        d = rng.standard_normal(3).astype(np.float32)
        d /= float(np.linalg.norm(d)) + 1e-9       # huong chieu cua rung len 3 truc
        # Ti le chung/rieng cung phai ngau nhien: de co dinh 0.85/0.35 thi cau truc
        # tuong quan giua cac truc luon y het nhau -> lai thanh mot quy luat hoc thuoc duoc.
        sh, iw = float(rng.uniform(0.55, 0.95)), float(rng.uniform(0.15, 0.55))
        out[:, lo:lo + 3] = sh * src[:, g:g + 1] * d[None, :] + iw * ind[:, lo:lo + 3]
    sd = float(out.std())
    if sd > 1e-9:
        out *= amp / sd
    # Rung that KHONG DUNG (non-stationary): xe qua o ga, dong co len/xuong ga, gio giat...
    # nang luong len xuong trong cua so. Chuan hoa ve dung RMS cho MOI cua so tao ra cam
    # giac deu deu rat gia. Nhan them mot bao hinh bien thien cham, ngau nhien.
    env = rng.standard_normal((max(t // 16, 2), 1)).astype(np.float32)
    env = np.asarray(Image.fromarray(env, mode="F").resize((1, t), Image.BICUBIC),
                     dtype=np.float32).reshape(t, 1)
    return out * (1.0 + 0.55 * env)


def _alias_tone(t: int, c: int, rng, amp: float, f_lo: float, f_hi: float,
                sr: float) -> np.ndarray:
    """Nguon rung TREN Nyquist (canh quat, dong co, banh rang) bi GAP TAN SO.

    IMU that lay mau roi rac; neu bo loc chong chong lap khong cat het, mot tone o
    f > sr/2 se hien ra trong du lieu duoi dang tan so gia |f - k*sr|. Vi du rotor
    130 Hz tren IMU 100 Hz -> xuat hien nhu rung 30 Hz, khong the loc bo o hau ky.
    O day sinh sin(2*pi*f*t) roi lay mau tai sr, aliasing tu xay ra dung vat ly.
    """
    tt = np.arange(t, dtype=np.float32) / sr
    out = np.zeros((t, c), np.float32)
    # Cung mot may moc -> cung mot tan so co ban cho ca 3 truc trong nhom, chi khac
    # huong chieu va pha. Truoc day moi kenh boc mot f0 rieng: 6 nguon rung khong lien
    # quan gi nhau, khong the xay ra tren mot thiet bi that.
    for g, base in enumerate((0, 3)):
        f0 = float(rng.uniform(f_lo, f_hi))
        d = rng.standard_normal(3).astype(np.float32)
        d /= float(np.linalg.norm(d)) + 1e-9
        for j in range(3):
            k = base + j
            # May moc that phat ra tan so co ban KEM HOA BA (bac 2, 3...), bien do giam
            # dan; tan so troi nhe theo tai va bien do bi dieu bien. Mot sin thuan bien do
            # co dinh la thu 'co quy luat' nhat co the sinh ra.
            for hmn in range(1, int(rng.integers(2, 6))):   # so hoa ba ngau nhien
                a = amp * abs(float(d[j])) * float(rng.uniform(0.25, 1.0)) / hmn
                f = f0 * hmn * (1.0 + 0.02 * float(rng.standard_normal()))
                ph = float(rng.uniform(0, 2 * np.pi))
                am = 1.0 + 0.35 * np.sin(2.0 * np.pi * float(rng.uniform(0.5, 3.0)) * tt
                                         + float(rng.uniform(0, 2 * np.pi)))
                out[:, k] += a * am * np.sin(2.0 * np.pi * f * tt + ph)
    return out


def _burst_env(t: int, c: int, rng, n_max: int, a_lo: float, a_hi: float) -> np.ndarray:
    """Bao hinh 'con nhieu': phan lon cua so yen, thinh thoang BUNG LEN mot doan.

    Nhieu IMU that khong deu theo thoi gian. Dong co len ga, mot con gio giat, banh xe
    qua o ga, nhieu dien tu tu ESC - moi thu deu tao ra tung DOAN nhieu manh xen giua
    cac doan yen. De bien do white noise phang li suot ca cua so la sai ban chat: chuoi
    that co luc sach, co luc nhieu han len.
    """
    env = np.ones((t, c), np.float32)
    for _ in range(int(rng.integers(0, n_max + 1))):
        w = int(rng.integers(max(2, t // 16), max(3, t // 3)))     # do dai con
        s0 = int(rng.integers(0, max(1, t - w)))
        ramp = np.hanning(w).astype(np.float32)                    # len/xuong muot
        ch = (rng.random(c) < 0.7).astype(np.float32)              # khong phai kenh nao cung dinh
        env[s0:s0 + w] += float(rng.uniform(a_lo, a_hi)) * ramp[:, None] * ch[None, :]
    return env


def corrupt_imu(u: np.ndarray, rng: np.random.Generator, cfg: dict,
                sr: float = 100.0) -> np.ndarray:
    out = u.astype(np.float32, copy=True)
    t, c = out.shape

    # --- sai so NHAN: dac tinh co dinh cua con chip, ap len tin hieu that ---
    s_max = float(cfg.get("scale_err", (0.0, 0.0))[1])
    if s_max > 0:                                  # he so khuech dai lech vai %
        out = out * (1.0 + rng.uniform(-s_max, s_max, size=c).astype(np.float32))

    m_max = float(cfg.get("misalign", (0.0, 0.0))[1])
    if m_max > 0:                                  # lech truc: truc X ro ri sang Y/Z
        for lo in (0, 3):                          # acc va gyro co ma tran rieng
            mm = rng.uniform(-m_max, m_max, size=(3, 3)).astype(np.float32)
            np.fill_diagonal(mm, 0.0)              # duong cheo da do scale_err lo
            out[:, lo:lo + 3] = out[:, lo:lo + 3] @ (np.eye(3, dtype=np.float32) + mm).T

    # --- RUNG DONG CO HOC: chuyen dong THAT ma sensor do duoc, tan so cao ---
    vib = np.zeros_like(out)
    ks = float(cfg.get("severity_skew", 1.0))
    v_amp = _sev(rng, cfg.get("vibration", (0.0, 0.0)), ks) if _hit(rng, cfg, "p_vibration") else 0.0
    if v_amp > 0:
        f_lo, f_hi = cfg.get("vib_freqs", (8.0, 45.0))
        vib += _vibration(t, c, rng, v_amp, f_lo, f_hi, float(cfg.get("vib_q", 6.0)), sr)

    al_amp = _sev(rng, cfg.get("alias_amp", (0.0, 0.0)), ks) if _hit(rng, cfg, "p_alias") else 0.0
    if al_amp > 0:                                 # rung tren Nyquist gap xuong bang do
        a_lo, a_hi = cfg.get("alias_freq", (55.0, 190.0))
        vib += _alias_tone(t, c, rng, al_amp, a_lo, a_hi, sr)
    out = out + vib

    # VRE (vibration rectification error): phi tuyen bac 2 cua MEMS "chinh luu" rung
    # AC thanh lech DC ti le CONG SUAT rung - loi dac trung cua gia toc ke MEMS that.
    vre_c = cfg.get("vre", 0.0)
    vre = float(rng.uniform(*vre_c)) if isinstance(vre_c, (list, tuple)) else float(vre_c)
    if vre > 0:
        out = out + (vre * np.mean(vib * vib, axis=0)).astype(np.float32)

    # --- sai so CONG (dien tu) ---
    std = _sev(rng, cfg["gauss_std"], ks)          # white noise (velocity/angle random walk)
    if std > 0:
        nz = rng.normal(0.0, std, out.shape).astype(np.float32)
        b_lo, b_hi = cfg.get("burst_amp", (0.0, 0.0))
        if b_hi > 0:                               # bien do nhieu thay doi THEO THOI GIAN
            nz *= _burst_env(t, c, rng, int(cfg.get("burst_max", 3)), b_lo, b_hi)
        out = out + nz

    b_lo, b_hi = cfg["bias"]                       # turn-on bias: hang so suot ca cua so
    if b_hi > 0:
        # Boc theo TUNG KENH: thuc te thuong chi mot vai truc bi day lech han, cac truc
        # con lai gan nhu sach - chu khong phai ca 6 kenh cung lech mot the.
        # `b_lo` la muc lech TOI THIEU khi da xay ra: neu de 0 thi p_bias mat y nghia
        # (hieu ung "co xay ra" nhung cuong do ~0, tuc van khong thay gi).
        hit = (rng.random(c) < float(cfg.get("p_bias", 1.0))).astype(np.float32)
        mag = b_lo + (b_hi - b_lo) * rng.random(c) ** ks
        out = out + (hit * mag * np.sign(rng.uniform(-1.0, 1.0, size=c))).astype(np.float32)

    d_lo, d_hi = cfg["drift"]                      # bias instability: random walk cham
    if d_hi > 0 and _hit(rng, cfg, "p_drift"):
        # Boc cuong do troi theo tung cua so; `d_lo` = muc toi thieu khi da xay ra.
        d = _sev(rng, (d_lo, d_hi), ks)
        steps = rng.normal(0.0, d, size=(t, c)).astype(np.float32)
        out = out + np.cumsum(steps, axis=0) / np.sqrt(t)

    p = float(cfg["spike_prob"])                   # xung dot bien thua
    if p > 0 and _hit(rng, cfg, "p_spike"):
        mask = rng.random((t, c)) < p
        n = int(mask.sum())
        if n:
            g_lo, g_hi = cfg.get("spike_gain", (3.0, 8.0))   # truoc day co dinh 5.0x
            gain = float(rng.uniform(g_lo, g_hi))
            out[mask] += rng.normal(0.0, gain * max(std, 1e-3), size=n).astype(np.float32)

    # Cu NHAY MUC giua cua so: soc co hoc, bao hoa roi hoi phuc, buoc nhiet do... lam
    # gia tri VOT LEN roi O NGUYEN muc moi den het cua so - khac han bias (co suot ca
    # cua so) va khac spike (chi mot mau roi ve ngay).
    st_lo, st_hi = cfg.get("step", (0.0, 0.0))
    if st_hi > 0 and _hit(rng, cfg, "p_step", 0.0):
        k0 = int(rng.integers(t // 6, max(t // 6 + 1, t - t // 6)))
        hit = (rng.random(c) < 0.4).astype(np.float32)             # chi vai kenh bi
        mag = _sev(rng, (st_lo, st_hi), ks) * np.sign(rng.uniform(-1.0, 1.0, size=c))
        # Cam bien that sau mot cu soc thuong HOI PHUC mot phan (thu gian co / can bang
        # nhiet), chi con lai chut lech vinh vien. De cu nhay giu nguyen den het cua so
        # la truong hop cuc doan - va no keo TOAN BO phan phia sau lech han khoi duong
        # sach, lam bai toan gan nhu khong giai duoc cho doan do.
        tau = float(rng.uniform(*cfg.get("step_tau", (8.0, 40.0))))     # so mau
        resid = float(rng.uniform(*cfg.get("step_resid", (0.0, 0.35))))  # phan con lai
        prof = resid + (1.0 - resid) * np.exp(-np.arange(t - k0, dtype=np.float32) / tau)
        out[k0:] += (prof[:, None] * mag * hit).astype(np.float32)

    # Mat goi tin / gia tri bi treo: IMU that thinh thoang lap lai mau cu (bus ban, DMA tre)
    sl = cfg.get("stuck_len", (0, 0))
    if sl[1] > 1 and _hit(rng, cfg, "p_stuck", 0.0):
        for _ in range(int(rng.integers(1, 3))):
            w = int(rng.integers(int(sl[0]), int(sl[1]) + 1))
            s0 = int(rng.integers(0, max(1, t - w)))
            ch = int(rng.integers(0, c))
            out[s0:s0 + w, ch] = out[s0, ch]

    # --- ADC: luong tu hoa roi bao hoa toan thang. Luon la buoc CUOI CUNG. ---
    q_c = cfg.get("quant", 0.0)                    # buoc ADC khac nhau giua cac con chip
    qstep = float(rng.uniform(*q_c)) if isinstance(q_c, (list, tuple)) else float(q_c)
    if qstep > 0:
        out = np.round(out / qstep) * qstep
    fs = float(cfg.get("clip", 0.0))
    if fs > 0:
        out = np.clip(out, -fs, fs)

    return out.astype(np.float32)
