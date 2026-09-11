"""Chot lai hai bat bien da tung lam model collapse (xem LOG_TRAIN_GIAI_THICH.md).

1. Nhanh target phai duoc CACH LY hoan toan khoi optimizer. Truoc day
   `self.tokenizer` dung chung cho ca hai nhanh: du nhanh target chay trong
   torch.no_grad(), optimizer van cap nhat tokenizer tu nhanh context moi buoc,
   nen z_tgt dich chuyen theo huong optimizer muon -> z_std tut -> L_jepa ve 0
   ma khong hoc duoc gi.

2. Dai THO (anh LL, imu A) phai luon nam trong context. Chi 36/512 token nhung
   mang gan het do sang / bo cuc; de chung bi vut thi tran PSNR tut tu ~25 dB
   xuong ~17.5 dB, tuc THAP HON ca anh nhieu dau vao (~20.9 dB).
"""

from __future__ import annotations

import copy
import pathlib

import torch
import yaml

from ..models.jepa import QwtJepa
from ..models.layout import build_layout
from ..train.losses import (
    content_std,
    jepa_loss_position_only,
    total_loss,
    variance_loss,
)
from ..train.masking import coarse_tokens, sample_masks

_CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def _cfg() -> dict:
    cfg = yaml.safe_load(_CFG.read_text())
    cfg["model"]["encoder"]["depth"] = 2        # nho lai cho test chay nhanh
    cfg["model"]["predictor"]["depth"] = 1
    return cfg


def _batch(cfg: dict, b: int = 2) -> dict:
    h = int(cfg["data"]["image"]["train_crop"][0])
    t = int(cfg["data"]["imu"]["window_size"])
    c = int(cfg["data"]["imu"]["channels"])
    g = torch.Generator().manual_seed(0)
    return {
        "img_clean": torch.rand(b, 3, h, h, generator=g),
        "img_noisy": torch.rand(b, 3, h, h, generator=g),
        "imu_clean": torch.randn(b, t, c, generator=g),
        "imu_noisy": torch.randn(b, t, c, generator=g),
    }


def test_target_branch_frozen_and_ema_only():
    """target_tokenizer: khong co gradient, khong doi sau optimizer.step(), chi doi qua EMA."""
    cfg = _cfg()
    model = QwtJepa(cfg)

    assert model.target_tokenizer is not model.tokenizer, "target_tokenizer phai la ban RIENG"
    assert not any(p.requires_grad for p in model.target_tokenizer.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())

    before = copy.deepcopy(model.target_tokenizer.image_proj.weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    loss, _ = total_loss(model(_batch(cfg)), _batch(cfg), 1.0, 1.0, 1.0, 1.0)
    loss.backward()
    assert all(p.grad is None for p in model.target_tokenizer.parameters()), \
        "gradient khong duoc chay vao nhanh target"

    opt.step()
    assert torch.equal(before, model.target_tokenizer.image_proj.weight), \
        "optimizer.step() khong duoc dong vao nhanh target"

    model.ema_update(0.9)
    assert not torch.equal(before, model.target_tokenizer.image_proj.weight), \
        "ema_update() phai cap nhat CA tokenizer, khong chi encoder"


def test_coarse_bands_always_in_context():
    cfg = _cfg()
    layout = build_layout(cfg)
    protect = coarse_tokens(layout)
    assert protect, "phai tim thay token dai tho (anh LL + imu A)"

    for _ in range(200):
        mask = sample_masks(layout, cfg, None)
        kept = set(mask.context_index.tolist())
        target = set(mask.target_index.tolist())
        assert protect <= kept, "dai tho bi vut khoi context"
        assert not (protect & target), "dai tho khong duoc lam target"


def test_variance_loss_phat_dung_chieu():
    z_deu = torch.randn(64, 32, 16)                  # da da dang -> gan nhu khong bi phat
    z_sap = torch.zeros(64, 32, 16) + torch.randn(1, 1, 16) * 0.01   # gan nhu hang so
    assert variance_loss(z_deu, gamma=1.0) < 0.1
    assert variance_loss(z_sap, gamma=1.0) > 0.9


def test_variance_loss_bat_duoc_collapse_VI_TRI():
    """Bat kieu collapse thu hai: bieu dien chi phu thuoc VI TRI token, khong phu
    thuoc anh. std gop ca (batch, token) van ~1.0 nen KHONG phat hien duoc; phai do
    std theo batch tai tung vi tri.
    """
    torch.manual_seed(0)
    # moi vi tri token mot vector rieng, nhung MOI MAU trong batch deu giong het nhau
    z_vi_tri = torch.randn(1, 32, 16).expand(64, 32, 16).contiguous()

    pooled = z_vi_tri.reshape(-1, 16).std(dim=0).mean()
    assert pooled > 0.8, "std gop van cao - chinh la cai bay"
    assert content_std(z_vi_tri) < 0.05, "std theo noi dung phai ~0"
    assert variance_loss(z_vi_tri, gamma=1.0) > 0.9, "phai bi phat nang"


def test_jepa_loss_position_only_la_nguong_gian_lan():
    torch.manual_seed(0)
    z_vi_tri = torch.randn(1, 32, 16).expand(64, 32, 16).contiguous()
    # bieu dien chi theo vi tri -> ke gian lan doan chinh xac tuyet doi
    assert jepa_loss_position_only(z_vi_tri) < 1e-6
    # bieu dien co noi dung that -> ke gian lan khong the doan duoc
    assert jepa_loss_position_only(torch.randn(64, 32, 16)) > 0.1


def test_band_norm_can_bang_bien_do_cac_dai():
    """Sau khi nap band scale, MOI dai phai co std ~1 truoc khi vao Linear dung chung.

    Khong chuan hoa thi he so L1 (std ~0.024) va L3 LL (std ~2.31) chenh ~100 lan,
    encoder gan nhu khong nhin thay chi tiet min -> anh tai tao mo.
    """
    from ..models.tokenizer import patchify

    cfg = _cfg()
    model = QwtJepa(cfg)
    # PHAI dung anh co pho kieu anh THAT (nang luong don ve tan so thap). Anh nhieu
    # trang co pho phang -> moi dai da can bang san, khong co gi de chuan hoa.
    h = int(cfg["data"]["image"]["train_crop"][0])
    g = torch.Generator().manual_seed(0)
    ramp = torch.linspace(0, 1, h)
    smooth = (ramp[None, :] + ramp[:, None]) / 2                    # nen muot -> LL lon
    img = smooth.expand(4, 3, h, h) + 0.01 * torch.rand(4, 3, h, h, generator=g)
    q_img, _ = model._qwt_all(img, torch.randn(4, 128, 6, generator=g))

    def band_stds(scaled: bool) -> list[float]:
        out = []
        for i, e in enumerate(model.layout.image_entries):
            band = q_img[e.level][e.band][:, 1:4]
            if scaled:
                band = band / model.tokenizer.img_band_scale[i]
            out.append(float(band.std()))
        return out

    raw = band_stds(False)
    assert max(raw) / min(raw) > 20, "anh thu nghiem phai co bien do lech nhieu giua cac dai"

    scales = {f"L{e.level}/{e.band}": s for e, s in zip(model.layout.image_entries, raw)}
    scales |= {f"{e.group}/L{e.level}/{e.band}": 1.0 for e in model.layout.imu_entries}
    model.set_band_scales(scales)

    done = band_stds(True)
    assert all(abs(v - 1.0) < 1e-3 for v in done), f"moi dai phai ve std ~1, dang la {done}"

    # va he qua: activation vao encoder khong con chenh hang tram lan
    def act_spread() -> float:
        st = []
        for i, e in enumerate(model.layout.image_entries):
            band = q_img[e.level][e.band][:, 1:4] / model.tokenizer.img_band_scale[i]
            p = patchify(band, e.gh, e.gw, model.layout.patch)
            st.append(float(model.tokenizer.image_proj(p).std().detach()))
        return max(st) / min(st)

    assert act_spread() < max(raw) / min(raw), "chuan hoa phai lam HEP khoang cach bien do"


def test_band_scale_nap_cho_ca_nhanh_target_va_head():
    cfg = _cfg()
    model = QwtJepa(cfg)
    n_img = len(model.layout.image_entries)
    n_imu = len(model.layout.imu_entries)
    scales = {f"L{e.level}/{e.band}": 2.0 for e in model.layout.image_entries}
    scales |= {f"{e.group}/L{e.level}/{e.band}": 3.0 for e in model.layout.imu_entries}
    model.set_band_scales(scales)

    for buf, want, n in (
        (model.tokenizer.img_band_scale, 2.0, n_img),
        (model.target_tokenizer.img_band_scale, 2.0, n_img),
        (model.image_head.band_scale, 2.0, n_img),
        (model.tokenizer.imu_band_scale, 3.0, n_imu),
        (model.target_tokenizer.imu_band_scale, 3.0, n_imu),
        (model.imu_head.band_scale, 3.0, n_imu),
    ):
        assert buf.shape == (n,) and torch.allclose(buf, torch.full((n,), want))


def test_head_co_cong_khoi_tao_bang_phep_copy():
    """Voi recon_gate, model luc KHOI TAO phai tra ve DUNG dau vao nhieu.

    Do la san: model bat dau ngay tai muc "copy anh nhieu" (~21.99 dB tren du lieu
    Kaggle) va chi co the di len, thay vi mo tu 0. Gradient van phai chay duoc.
    """
    cfg = _cfg()
    if not cfg["model"].get("recon_gate", True):
        return
    model = QwtJepa(cfg)
    model.eval()
    b = _batch(cfg, b=2)

    with torch.no_grad():
        img, imu = model.reconstruct(b["img_noisy"], b["imu_noisy"])
    assert torch.allclose(img, b["img_noisy"], atol=1e-4), "anh ra phai bang anh vao"
    assert torch.allclose(imu, b["imu_noisy"], atol=1e-4), "imu ra phai bang imu vao"

    img, imu = model.reconstruct(b["img_noisy"], b["imu_noisy"])
    (img.abs().mean() + imu.abs().mean()).backward()
    assert model.image_head.proj.weight.grad.norm() > 0, "khoi tao 0 nhung van phai hoc duoc"


def test_he_so_nhan_bi_chan():
    """gain = 1 + gate_max*tanh(g) phai nam trong [1-gate_max, 1+gate_max] va bang 1 tai g=0.

    Khong chan (`1+g`) lam model phan ky: do duoc PSNR tut 25 -> 12 dB va do net len 2.4
    (anh ra net hon ca anh sach = dang bia tan so cao).
    """
    cfg = _cfg()
    model = QwtJepa(cfg)
    gm = model.image_head.gate_max
    if gm <= 0:
        return
    g = torch.linspace(-50, 50, 101)
    gain = model.image_head._gain(g)
    assert gain.min() >= 1 - gm - 1e-5 and gain.max() <= 1 + gm + 1e-5
    assert abs(float(model.image_head._gain(torch.zeros(1))) - 1.0) < 1e-6


def test_freeze_backbone_chi_con_head_hoc():
    """GIAI DOAN 2: dong bang tokenizer+encoder+predictor, chi 2 recon head con hoc."""
    cfg = _cfg()
    model = QwtJepa(cfg)
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.freeze_backbone()
    after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert after < before

    con_hoc = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert con_hoc <= {"image_head", "imu_head", "recon_in", "recon_dec", "recon_out"}, con_hoc
    for mod in (model.tokenizer, model.context_encoder, model.predictor):
        assert not any(p.requires_grad for p in mod.parameters())

    # bieu dien phai DUNG YEN: cung dau vao -> cung dau ra sau khi cap nhat head
    model.eval()
    b = _batch(cfg, b=2)
    with torch.no_grad():
        q_img, q_imu = model._qwt_all(b["img_noisy"], b["imu_noisy"])
        z1 = model.context_encoder(model.tokenizer(q_img, q_imu)).clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    img, imu = model.reconstruct(b["img_noisy"], b["imu_noisy"])
    (img.abs().mean() + imu.abs().mean()).backward()
    opt.step()
    with torch.no_grad():
        q_img, q_imu = model._qwt_all(b["img_noisy"], b["imu_noisy"])
        z2 = model.context_encoder(model.tokenizer(q_img, q_imu))
    assert torch.allclose(z1, z2), "encoder da dong bang thi bieu dien khong duoc doi"
