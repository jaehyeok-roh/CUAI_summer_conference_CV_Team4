"""bd_rate 와 SyntheticLowLight 의 최소 동작 확인.

실행: python test_bd_rate_synthetic.py  (pytest 로도 실행 가능)
"""
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from dataset import SyntheticLowLight
from rd_curve import bd_rate


def test_bd_rate():
    anchor = [{"bpp": r, "psnr": q} for r, q in [(0.19, 17.0), (0.23, 17.6), (0.26, 18.1), (0.33, 18.9)]]
    cheaper = [{"bpp": p["bpp"] * 0.9, "psnr": p["psnr"]} for p in anchor]
    assert abs(bd_rate(anchor, anchor, "psnr")) < 1e-9
    assert abs(bd_rate(anchor, cheaper, "psnr") + 10) < 1e-6


def test_synthetic_low_light():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        for sub in ("low", "high"):
            os.makedirs(os.path.join(root, sub))
        high = np.random.default_rng(0).integers(40, 255, (300, 300, 3), dtype=np.uint8)
        Image.fromarray(high).save(os.path.join(root, "high", "0.png"))
        Image.fromarray((high * 0.1).astype(np.uint8)).save(os.path.join(root, "low", "0.png"))

        dark = SyntheticLowLight(root, "dark")
        low, high_t = dark[0]
        assert torch.allclose(low, high_t * dark.ratios[0], atol=1e-4)

        for mode in ("dark_awgn", "dark_pg"):
            low, high_t = SyntheticLowLight(root, mode)[0]
            assert low.shape == high_t.shape == (3, 256, 256)
            assert not torch.isnan(low).any() and 0 <= low.min() and low.max() <= 1

        try:
            SyntheticLowLight(root, "bright")
        except ValueError:
            pass
        else:
            raise AssertionError("잘못된 mode 는 ValueError 가 나야 한다")


if __name__ == "__main__":
    test_bd_rate()
    test_synthetic_low_light()
    print("OK")
