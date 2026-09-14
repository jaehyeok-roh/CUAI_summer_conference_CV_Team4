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
from baseline.edge_finetune import train as edge_train
from baseline.eval_all import codec_then, report as eval_report
from baseline.joint_finetune import align_tone, forward as joint_forward, train as joint_train
from baseline.refiner import decode_all, finetune, score, to_uint8
from baseline.split_iat import receiver as split_receiver, train_local as split_train_local
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

    # mbt2018-mean(--codec): mean 기준 반올림까지 평가 모드 복원과 같아야 하고, 엣지 코덱 적응 학습도 돌아야 한다
    from compressai.zoo import mbt2018_mean
    codec = mbt2018_mean(quality=1, pretrained=False)
    x_hat, _ = joint_forward(codec.train(), x, ste=True)
    x_hat.mean().backward()
    assert codec.g_a[0].weight.grad is not None
    with torch.no_grad():
        assert torch.allclose(x_hat, codec.eval()(x)["x_hat"], atol=1e-6)
    before = codec.g_a[0].weight.clone()
    edge_train(torch.nn.Conv2d(3, 3, 3, padding=1), codec, lows, highs, 0.0018, iters=2, part="codec", target="aligned", crop=64, batch=2, alpha=0.5)
    assert not torch.equal(codec.g_a[0].weight, before)

    # 채널을 섞고 밝기를 바꾼(affine) 영상에 원래 영상의 색·톤을 맞추면 그 영상이 그대로 나와야 한다 (--target aligned)
    src = to_uint8(torch.rand(3, 16, 16) * 0.5 + 0.25)
    ref = to_uint8(src.flip(0).float() / 255 * 0.8 + 0.1)
    assert (align_tone(src, ref).int() - ref.int()).abs().max() <= 1


def test_edge_finetune_steps():
    # --train/--target 에 따라 향상 모델(IAT 대신 3x3 conv)과 코덱 중 학습하는 쪽만 바뀌어야 한다 (가중치 다운로드 없이)
    lows = [to_uint8(torch.rand(3, 64, 96) * 0.2) for _ in range(2)]
    highs = [to_uint8(torch.rand(3, 64, 96)) for _ in range(2)]
    for part, target in (("both", "gt"), ("codec", "gt"), ("enhancer", "gt"), ("codec", "input"), ("both", "teacher"), ("codec", "aligned")):
        enhancer, codec = torch.nn.Conv2d(3, 3, 3, padding=1), bmshj2018_hyperprior(quality=1, pretrained=False)
        before = enhancer.weight.clone(), codec.g_a[0].weight.clone()
        edge_train(enhancer, codec, lows, highs, 0.0018, iters=2, part=part, target=target, crop=64, batch=2)
        assert torch.equal(enhancer.weight, before[0]) == (part == "codec")
        assert torch.equal(codec.g_a[0].weight, before[1]) == (part == "enhancer")

    # --alpha: aligned 목표와 코덱 입력을 섞어도 코덱만 학습돼야 한다
    enhancer, codec = torch.nn.Conv2d(3, 3, 3, padding=1), bmshj2018_hyperprior(quality=1, pretrained=False)
    before = enhancer.weight.clone(), codec.g_a[0].weight.clone()
    edge_train(enhancer, codec, lows, highs, 0.0018, iters=2, part="codec", target="aligned", crop=64, batch=2, alpha=0.5)
    assert torch.equal(enhancer.weight, before[0]) and not torch.equal(codec.g_a[0].weight, before[1])

    # --target aligned: 영상마다 채널을 섞고 밝기를 바꾼(affine) 결과에 원래 영상을 맞추면 그 결과가 그대로 나와야 한다
    from baseline.edge_finetune import align_tone_batch
    src = torch.rand(2, 3, 16, 16) * 0.5 + 0.25
    ref = torch.stack([src[0].flip(0) * 0.8 + 0.1, src[1] * 1.2 - 0.05])
    assert torch.allclose(align_tone_batch(src, ref), ref, atol=1e-5)

    # --source gt 대조군은 향상 모델을 거치지 않고 정답 영상으로 코덱만 맞춰야 한다
    class NoEnhancer(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("source=gt 인데 향상 모델을 거쳤다")
    codec = bmshj2018_hyperprior(quality=1, pretrained=False)
    before = codec.g_a[0].weight.clone()
    edge_train(NoEnhancer(), codec, lows, highs, 0.0018, iters=2, part="codec", target="input", source="gt", crop=64, batch=2)
    assert not torch.equal(codec.g_a[0].weight, before)

    # --train enhancer_local 은 향상 모델의 지역 분기만 바꾸고 전역 분기와 코덱은 그대로 둬야 한다
    class TwoBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Module()
            self.net.local_net, self.net.global_net = torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return self.net.global_net(self.net.local_net(x))
    enhancer, codec = TwoBranch(), bmshj2018_hyperprior(quality=1, pretrained=False)
    before = [m.weight.clone() for m in (enhancer.net.local_net, enhancer.net.global_net, codec.g_a[0])]
    edge_train(enhancer, codec, lows, highs, 0.0018, iters=2, part="enhancer_local", crop=64, batch=2)
    assert not torch.equal(enhancer.net.local_net.weight, before[0])
    assert torch.equal(enhancer.net.global_net.weight, before[1]) and torch.equal(codec.g_a[0].weight, before[2])

    # 코덱을 고정해도(train 모드) bpp 만으로 향상 모델에 gradient 가 가야 한다: IAT 가 압축하기 좋은 출력을 배우는 경로
    enhancer, codec = torch.nn.Conv2d(3, 3, 3, padding=1), bmshj2018_hyperprior(quality=1, pretrained=False).train().requires_grad_(False)
    _, likelihoods = joint_forward(codec, enhancer(torch.rand(1, 3, 64, 64)), ste=True)
    sum(torch.log2(l).sum() for l in likelihoods.values()).backward()
    assert enhancer.weight.grad.abs().sum() > 0


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


def test_split_iat_steps():
    # 나눠 돌린 IAT(전역 파라미터를 따로 받아 적용)가 한 번에 돌린 출력과 같고, 재학습은 지역 분기만 바꿔야 한다 (IAT 대신 같은 구조의 작은 모듈)
    class Local(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 3, padding=1)

        def forward(self, x):
            mul, add = self.conv(x).chunk(2, dim=1)
            return torch.sigmoid(mul) * 2, torch.tanh(add) * 0.1

    class Global(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 10)

        def forward(self, x):
            out = self.fc(x.mean(dim=(2, 3)))
            return out[:, :1].sigmoid() + 0.5, out[:, 1:].view(-1, 3, 3) * 0.1 + torch.eye(3)

    class TinyIAT(torch.nn.Module):  # IAT_main.IAT 와 같은 forward 구조
        def __init__(self):
            super().__init__()
            self.local_net, self.global_net = Local(), Global()

        def apply_color(self, image, ccm):
            shape = image.shape
            return torch.clamp(torch.tensordot(image.view(-1, 3), ccm, dims=[[-1], [-1]]).view(shape), 1e-8, 1.0)

        def forward(self, img_low):
            mul, add = self.local_net(img_low)
            img_high = (img_low * mul + add).permute(0, 2, 3, 1)
            gamma, color = self.global_net(img_low)
            img_high = torch.stack([self.apply_color(img_high[i], color[i]) ** gamma[i] for i in range(img_high.shape[0])])
            return mul, add, img_high.permute(0, 3, 1, 2)

    net = TinyIAT()
    x = torch.rand(2, 3, 64, 64) * 0.3
    with torch.no_grad():
        assert torch.allclose(split_receiver(net, x, *net.global_net(x)), net(x)[2], atol=1e-6)

    lows = [to_uint8(torch.rand(3, 64, 96) * 0.2) for _ in range(2)]
    highs = [to_uint8(torch.rand(3, 64, 96)) for _ in range(2)]
    before = net.local_net.conv.weight.clone(), net.global_net.fc.weight.clone()
    split_train_local(net, lows, lows, highs, iters=2, crop=64, batch=2)  # 복원 영상 자리에 원본을 넣어도 학습 동작은 확인된다
    assert not torch.equal(net.local_net.conv.weight, before[0]) and torch.equal(net.global_net.fc.weight, before[1])


if __name__ == "__main__":
    test_bd_rate()
    test_synthetic_low_light()
    test_two_stage_pipelines()
    test_refiner_steps()
    test_joint_finetune_steps()
    test_edge_finetune_steps()
    test_split_iat_steps()
    test_compute_cost()
    test_eval_all()
    print("OK")
