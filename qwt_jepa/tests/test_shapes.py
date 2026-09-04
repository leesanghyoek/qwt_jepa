"""Test C1/D/E: N on dinh theo config; z_pred.shape == z_tgt.shape; recon dung shape goc."""

from __future__ import annotations

import pathlib
import sys

import torch
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.models.jepa import QwtJepa


def _cfg():
    cfg = yaml.safe_load((_ROOT / "qwt_jepa" / "configs" / "base.yaml").read_text())
    cfg["model"]["encoder"]["depth"] = 2
    cfg["model"]["predictor"]["depth"] = 2
    return cfg


def _batch(b: int):
    return {
        "img_noisy": torch.rand(b, 3, 256, 256),
        "img_clean": torch.rand(b, 3, 256, 256),
        "imu_noisy": torch.randn(b, 128, 6),
        "imu_clean": torch.randn(b, 128, 6),
    }


def test_token_count_stable_across_batch():
    model = QwtJepa(_cfg())
    n = model.layout.n_tokens
    for b in (1, 3):
        out = model(_batch(b))
        assert model.layout.n_tokens == n
        assert out["z_pred"].shape == out["z_tgt"].shape
        assert out["z_pred"].shape[0] == b
        assert out["z_pred"].shape[-1] == int(model.cfg["model"]["d_model"])


def test_recon_shapes():
    model = QwtJepa(_cfg())
    out = model(_batch(2))
    assert tuple(out["img_rec"].shape) == (2, 3, 256, 256)
    assert tuple(out["imu_rec"].shape) == (2, 128, 6)
    assert torch.isfinite(out["img_rec"]).all()
    assert torch.isfinite(out["imu_rec"]).all()


def test_mask_partition():
    model = QwtJepa(_cfg())
    out = model(_batch(1))
    m = out["mask"]
    ctx = set(m.context_index_full.tolist())
    tgt = set(m.target_index.tolist())
    assert ctx.isdisjoint(tgt)
    assert ctx | tgt == set(range(model.layout.n_tokens))
    assert set(m.context_index.tolist()).issubset(ctx)


if __name__ == "__main__":
    test_token_count_stable_across_batch()
    test_recon_shapes()
    test_mask_partition()
    print("test_shapes: OK")
