"""Test A1/A2: iQWT o QWT ~ identity cho anh va IMU (sai so < 1e-5)."""

from __future__ import annotations

import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwt_jepa.qwt import image_iqwt, image_qwt, imu_iqwt, imu_qwt, quat_to_rgb, rgb_to_quat


def test_image_roundtrip():
    x = torch.rand(2, 3, 256, 256, dtype=torch.float64)
    rec = quat_to_rgb(image_iqwt(image_qwt(rgb_to_quat(x), levels=3)))
    assert torch.max(torch.abs(x - rec)) < 1e-5


def test_image_roundtrip_odd():
    x = torch.rand(1, 3, 130, 127, dtype=torch.float64)
    rec = quat_to_rgb(image_iqwt(image_qwt(rgb_to_quat(x), levels=3)))
    assert rec.shape == x.shape
    assert torch.max(torch.abs(x - rec)) < 1e-5


def test_imu_roundtrip():
    u = torch.randn(3, 128, 6, dtype=torch.float64)
    rec = imu_iqwt(imu_qwt(u, levels=3))
    assert rec.shape == u.shape
    assert torch.max(torch.abs(u - rec)) < 1e-5


def test_imu_roundtrip_level4_odd():
    u = torch.randn(2, 100, 6, dtype=torch.float64)
    rec = imu_iqwt(imu_qwt(u, levels=4))
    assert torch.max(torch.abs(u - rec)) < 1e-5


if __name__ == "__main__":
    test_image_roundtrip()
    test_image_roundtrip_odd()
    test_imu_roundtrip()
    test_imu_roundtrip_level4_odd()
    print("test_qwt_roundtrip: OK")
