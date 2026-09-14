"""
v2: 기존 코덱 + refiner 를 같이 미세조정한다 (enhancement-aware compression).

refiner 실험에서 '기존 코덱 + refiner' 가 'Ours + refiner' 보다 좋았다. 여기서는 그 조합에서 출발해,
코덱은 어두운 영상을 그대로 압축하되 수신 측 refiner 결과가 좋아지도록 코덱과 refiner 를 같이 학습한다.

  기본           : 코덱 + refiner 를 같이 학습 (v2)
  --ste          : 같이 학습하되 복원 경로는 평가처럼 반올림한다 (straight-through). rate 는 노이즈 근사 그대로
  --freeze_codec : 코덱은 고정하고 refiner 만 같은 조건으로 더 학습 (대조군. 추가 학습량·품질별 특화 효과를 뺀다)
  --iat_dir      : 2단계 향상. 엣지에서 고정 IAT 로 먼저 향상한 영상을 코덱에 넣고, 수신 측 refiner 가 한 번 더 향상한다.
                   IAT 는 엣지 향상 -> 압축이 v2 보다 같은 비트에서 +0.84 dB 좋았지만, 정답 대비 화질이 IAT 자체 화질(23.38 dB)에 막힌다
  --target aligned : (--iat_dir 전용) 목표를 정답 대신 '정답의 색·톤을 IAT 출력에 affine 으로 맞춘 영상'으로 둔다.
                   정답을 목표로 IAT/코덱을 학습하면 LOL 학습셋 정답의 색·톤에 과적합했다(edge_finetune.py). refiner 가 색·톤은
                   IAT 를 따르고 잡음·디테일만 고치게 한다

손실은 CompressAI 학습과 같은 λ·255²·MSE(refiner 출력, 정상조도 GT) + bpp 이고, λ 는 품질별 사전학습 값이다.
평가는 refiner.py 와 같은 경로(코덱 복원 영상을 8비트로 저장 -> refiner)이며, 학습 전(Before) 값도 같이 적는다. 평가 목표는 늘 원래 정답이다.

실행 (two_stage.py 와 같은 준비 후, 레포 루트에서):
    python baseline/joint_finetune.py --train_dir <LOL 경로>/our485 --eval_dir <LOL 경로>/eval15 \
        --refiner <refiner 실행 Output>/refiner_Standard_codec.pth --qualities 1 3 5 7 [--ste | --freeze_codec] \
        [--iat_dir <IAT>/IAT_enhance [--target aligned]]
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from compressai.zoo import bmshj2018_hyperprior

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # baseline/
from rd_curve import device, load_eval_pairs
from refiner import decode_all, sample_batch, score, to_uint8
from two_stage import load_retinexformer

LAMBDAS = {1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130, 5: 0.0250, 6: 0.0483, 7: 0.0932, 8: 0.18}  # CompressAI bmshj2018 학습 λ


def forward(codec, x, ste):
    """학습용 코덱 순전파 -> (x_hat, likelihoods). ste 면 복원 경로만 반올림(straight-through)해 평가 때 복원과 같게 한다"""
    if not ste:
        out = codec(x)
        return out["x_hat"], out["likelihoods"]
    y = codec.g_a(x)
    z_hat, z_likelihoods = codec.entropy_bottleneck(codec.h_a(torch.abs(y)))
    _, y_likelihoods = codec.gaussian_conditional(y, codec.h_s(z_hat))
    y_hat = y + (torch.round(y) - y).detach()
    return codec.g_s(y_hat), {"y": y_likelihoods, "z": z_likelihoods}


def align_tone(src, ref):
    """src 의 색·톤을 ref 에 맞춘다: 픽셀 [r, g, b, 1] -> ref 로 가는 affine(4x3) 최소제곱. uint8 CPU 텐서 (3, H, W)"""
    x = torch.cat([src.reshape(3, -1).T.double() / 255, torch.ones(src[0].numel(), 1, dtype=torch.float64)], 1)
    a = torch.linalg.lstsq(x, ref.reshape(3, -1).T.double() / 255).solution
    return to_uint8((x @ a).T.reshape(src.shape))


def train(codec, refiner, lows, highs, lmbda, iters, freeze_codec, ste=False, crop=128, batch=8, lr=1e-4):
    """lows[i] -> 코덱 -> refiner -> highs[i]. 입력은 uint8 CPU 텐서, crop 은 코덱 입력이라 64 의 배수여야 한다."""
    groups = [{"params": list(refiner.parameters()), "lr": lr}]
    if not freeze_codec:
        groups.append({"params": list(codec.parameters()), "lr": lr / 10})  # 사전학습 코덱은 작은 학습률로
    codec.train(not freeze_codec).requires_grad_(not freeze_codec)
    refiner.train()
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)
    sums = [0.0, 0.0, 0.0]
    for it in range(iters):
        x, y = sample_batch([lows], highs, crop, batch)
        x_hat, likelihoods = forward(codec, x, ste)
        bpp = sum(torch.log2(l).sum() for l in likelihoods.values()) / -(batch * crop * crop)
        mse = F.mse_loss(refiner(x_hat), y)
        loss = lmbda * 255 ** 2 * mse + bpp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        sums = [s + v.item() for s, v in zip(sums, (loss, mse, bpp))]
        if (it + 1) % 1000 == 0:  # 배치 하나는 크롭에 따라 크게 흔들려서 최근 1000 iter 평균을 찍는다
            print(f"    iter {it + 1}/{iters} (최근 1000 평균) loss {sums[0] / 1000:.4f} mse {sums[1] / 1000:.5f} bpp {sums[2] / 1000:.4f}", flush=True)
            sums = [0.0, 0.0, 0.0]
    codec.eval()
    refiner.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./LOL_Dataset/lol_dataset/our485")
    parser.add_argument("--eval_dir", default="./LOL_Dataset/lol_dataset/eval15")
    parser.add_argument("--refiner", default="./results/rd_curve/refiner_Standard_codec.pth")
    parser.add_argument("--qualities", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--freeze_codec", action="store_true")
    parser.add_argument("--ste", action="store_true")
    parser.add_argument("--iat_dir", default=None)
    parser.add_argument("--target", choices=("gt", "aligned"), default="gt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="./results/rd_curve/joint_finetune.json")
    args = parser.parse_args()
    if args.target == "aligned" and not args.iat_dir:
        parser.error("--target aligned 는 --iat_dir 와 같이 쓴다 (IAT 출력의 색·톤에 정답을 맞춘다)")

    train_pairs, test = load_eval_pairs(args.train_dir), load_eval_pairs(args.eval_dir)
    lows, highs = [to_uint8(low) for low, _, _ in train_pairs], [to_uint8(high) for _, high, _ in train_pairs]
    test_lows, test_highs = [low for low, _, _ in test], [high for _, high, _ in test]
    if args.freeze_codec:
        tag, label = "frozen", "Standard codec + refiner (codec frozen)"
    else:
        tag = "joint_ste" if args.ste else "joint"
        label = f"Joint fine-tuned codec + refiner (v2{', STE' if args.ste else ''})"
    if args.iat_dir:
        from edge_finetune import IAT  # edge_finetune 이 이 파일을 import 해서 맨 위가 아니라 여기서 불러온다
        iat = IAT(args.iat_dir).to(device).eval()
        with torch.no_grad():  # IAT 전역 분기는 영상 통계를 써서 크롭이 아니라 전체 영상에 한 번 적용한다 (edge_finetune.py 평가와 같음)
            lows = [to_uint8(iat(low.unsqueeze(0).to(device))[0]) for low, _, _ in train_pairs]
            test_lows = [to_uint8(iat(low.unsqueeze(0).to(device))[0]).float() / 255 for low in test_lows]
        if args.target == "aligned":
            highs = [align_tone(high, low) for high, low in zip(highs, lows)]
        tag = f"iat_{tag}" + ("_aligned" if args.target == "aligned" else "")
        label = "Edge IAT + " + label + (" (target: tone-aligned GT)" if args.target == "aligned" else "")
    results = {"Before": [], label: []}
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    def evaluate(name, quality, codec, refiner):
        images, bpp = decode_all(codec, test_lows)
        psnr, ssim = score(images, test_highs, refiner)
        results[name].append({"quality": quality, "bpp": bpp, "psnr": psnr, "ssim": ssim})
        print(f"  {name} q={quality}: PSNR {psnr:.2f} / SSIM {ssim:.4f} / BPP {bpp:.4f}", flush=True)

    for quality in args.qualities:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        codec = bmshj2018_hyperprior(quality=quality, pretrained=True).to(device).eval()
        refiner = load_retinexformer(args.refiner)
        evaluate("Before", quality, codec, refiner)
        print(f"[q={quality}] {label} 학습 ({args.iters} iters, lambda {LAMBDAS[quality]})", flush=True)
        train(codec, refiner, lows, highs, LAMBDAS[quality], args.iters, args.freeze_codec, args.ste)
        evaluate(label, quality, codec, refiner)
        torch.save({"codec": codec.state_dict(), "refiner": refiner.state_dict()}, os.path.join(out_dir, f"{tag}_q{quality}.pth"))
        with open(args.out, "w") as f:  # 세션이 끊겨도 끝난 품질까지는 남긴다
            json.dump(results, f, indent=2)
    print(f"저장됨: {args.out}")


if __name__ == "__main__":
    main()
