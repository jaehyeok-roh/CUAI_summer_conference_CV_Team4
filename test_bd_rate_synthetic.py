"""bd_rate 와 SyntheticLowLight 의 최소 동작 확인.

실행: python test_bd_rate_synthetic.py  (pytest 로도 실행 가능)
"""
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from compressai.zoo import bmshj2018_hyperprior

from baseline.cost import pipelines
from baseline.eval_all import codec_then, report as eval_report
from baseline.joint_finetune import forward as joint_forward, train as joint_train
from baseline.refiner import decode_all, finetune, score, to_uint8
from baseline.two_stage import evaluate_pipeline
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


def test_two_stage_pipelines():
    # 64 의 배수가 아닌 크기로 패딩/크롭과 BPP 처리를 확인한다 (가중치 다운로드 없이)
    pairs = [(torch.rand(3, 70, 100) * 0.2, torch.rand(3, 70, 100), "0.png")]
    codec = bmshj2018_hyperprior(quality=1, pretrained=False).eval()
    enhancer = torch.nn.Identity()
    only = evaluate_pipeline(pairs, enhancer, None, "E")
    assert only["bpp"] == 0
    for order in ("EC", "CE"):
        m = evaluate_pipeline(pairs, enhancer, codec, order)
        assert m["bpp"] > 0 and np.isfinite(m["psnr"]) and 0 <= m["ssim"] <= 1


def test_refiner_steps():
    # 복원 -> 재학습 -> 평가가 64 의 배수가 아닌 크기에서도 돌아가는지 확인한다 (가중치 다운로드 없이)
    lows = [torch.rand(3, 70, 100) * 0.2 for _ in range(2)]
    highs = [torch.rand(3, 70, 100) for _ in range(2)]
    images, bpp = decode_all(bmshj2018_hyperprior(quality=1, pretrained=False).eval(), lows)
    assert images[0].dtype == torch.uint8 and images[0].shape == (3, 70, 100) and bpp > 0
    enhancer = finetune(torch.nn.Conv2d(3, 3, 3, padding=1), [images], [to_uint8(h) for h in highs], iters=2, crop=32, batch=2)
    psnr, ssim = score(images, highs, enhancer)
    assert np.isfinite(psnr) and 0 <= ssim <= 1
    assert score(images, highs) != (psnr, ssim)  # enhancer 없이 평가하면 결과가 달라야 한다


def test_joint_finetune_steps():
    # 같이 학습하면(STE 포함) 코덱 가중치가 바뀌고, --freeze_codec 이면 그대로여야 한다 (가중치 다운로드 없이)
    lows = [to_uint8(torch.rand(3, 64, 96) * 0.2) for _ in range(2)]
    highs = [to_uint8(torch.rand(3, 64, 96)) for _ in range(2)]
    for freeze, ste in ((True, False), (False, False), (False, True)):
        codec = bmshj2018_hyperprior(quality=1, pretrained=False)
        before = codec.g_a[0].weight.clone()
        joint_train(codec, torch.nn.Conv2d(3, 3, 3, padding=1), lows, highs, 0.0018, iters=2, freeze_codec=freeze, ste=ste, crop=64, batch=2)
        assert torch.equal(codec.g_a[0].weight, before) == freeze

    # STE 복원은 학습 모드에서도 평가 모드 복원(반올림)과 같아야 하고, gradient 는 인코더까지 흘러야 한다
    codec = bmshj2018_hyperprior(quality=1, pretrained=False)
    x = torch.rand(1, 3, 64, 64)
    x_hat, _ = joint_forward(codec.train(), x, ste=True)
    x_hat.mean().backward()
    assert codec.g_a[0].weight.grad is not None
    with torch.no_grad():
        assert torch.allclose(x_hat, codec.eval()(x)["x_hat"], atol=1e-6)


def test_compute_cost():
    # Retinexformer 대신 3x3 conv 로: 파이프라인별로 CBAM·향상 모델 비용이 올바른 쪽에 더해지는지 확인한다
    rows = pipelines(1, torch.nn.Conv2d(3, 3, 3, padding=1), h=64, w=96, runs=1)
    std, ec, ce, ours = rows["Standard codec"], rows["Enhance -> Compress"], rows["Compress -> Enhance"], rows["Ours"]
    M = 192  # quality 1~5 의 latent 채널 수
    assert ours["receiver"]["params"] - std["receiver"]["params"] == 2 * M * (M // 16) + 2 * 7 * 7 + 1
    assert ours["edge"]["flops"] == std["edge"]["flops"]
    assert ec["edge"]["flops"] - std["edge"]["flops"] == 3 * 3 * 3 * 3 * 64 * 96
    assert ce["receiver"]["params"] - std["receiver"]["params"] == 3 * 3 * 3 * 3 + 3
    assert ec["receiver"] == std["receiver"]


def test_eval_all():
    # 출력이 8비트 값이고, BD 와 부호 검정이 알려진 차이를 그대로 되찾는지 확인한다 (가중치 다운로드 없이)
    run = codec_then(bmshj2018_hyperprior(quality=1, pretrained=False).eval(), torch.nn.Conv2d(3, 3, 3, padding=1).eval())
    y, bpp = run(torch.rand(3, 70, 100) * 0.2)
    assert y.shape == (3, 70, 100) and bpp > 0 and torch.allclose(y * 255, (y * 255).round(), atol=1e-4)

    def curve(dp):  # 5장 x 4품질. 같은 bpp 에서 PSNR 이 dp, SSIM 이 dp/50 만큼 높다
        return {q: [[f"{i}.png", 0.05 * (q + 1) * (1 + i / 10), 19 + q + i / 10 + dp, 0.6 + 0.05 * q + dp / 50] for i in range(5)] for q in range(4)}
    entry = eval_report({"Control (codec frozen)": curve(0.0), "Ours: v2 (STE)": curve(0.5)})[("Ours: v2 (STE)", "Control (codec frozen)")]
    assert abs(entry["bd_psnr"] - 0.5) < 1e-9 and abs(entry["bd_ssim"] - 0.01) < 1e-9
    assert entry["bd_rate_ssim"] < 0 and entry["wins_PSNR"] == (5, 5)


if __name__ == "__main__":
    test_bd_rate()
    test_synthetic_low_light()
    test_two_stage_pipelines()
    test_refiner_steps()
    test_joint_finetune_steps()
    test_compute_cost()
    test_eval_all()
    print("OK")
