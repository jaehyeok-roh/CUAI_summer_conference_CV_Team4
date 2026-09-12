"""
v2: 기존 코덱 + refiner 를 같이 미세조정한다 (enhancement-aware compression).

refiner 실험에서 '기존 코덱 + refiner' 가 'Ours + refiner' 보다 좋았다. 여기서는 그 조합에서 출발해,
코덱은 어두운 영상을 그대로 압축하되 수신 측 refiner 결과가 좋아지도록 코덱과 refiner 를 같이 학습한다.

  기본           : 코덱 + refiner 를 같이 학습 (v2)
  --freeze_codec : 코덱은 고정하고 refiner 만 같은 조건으로 더 학습 (대조군. 추가 학습량·품질별 특화 효과를 뺀다)

손실은 CompressAI 학습과 같은 λ·255²·MSE(refiner 출력, 정상조도 GT) + bpp 이고, λ 는 품질별 사전학습 값이다.
평가는 refiner.py 와 같은 경로(코덱 복원 영상을 8비트로 저장 -> refiner)이며, 학습 전(Before) 값도 같이 적는다.

실행 (two_stage.py 와 같은 준비 후, 레포 루트에서):
    python baseline/joint_finetune.py --train_dir <LOL 경로>/our485 --eval_dir <LOL 경로>/eval15 \
        --refiner <refiner 실행 Output>/refiner_Standard_codec.pth --qualities 1 3 5 7 [--freeze_codec]
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


def train(codec, refiner, lows, highs, lmbda, iters, freeze_codec, crop=128, batch=8, lr=1e-4):
    """lows[i] -> 코덱 -> refiner -> highs[i]. 입력은 uint8 CPU 텐서, crop 은 코덱 입력이라 64 의 배수여야 한다."""
    groups = [{"params": list(refiner.parameters()), "lr": lr}]
    if not freeze_codec:
        groups.append({"params": list(codec.parameters()), "lr": lr / 10})  # 사전학습 코덱은 작은 학습률로
    codec.train(not freeze_codec).requires_grad_(not freeze_codec)
    refiner.train()
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)
    for it in range(iters):
        x, y = sample_batch([lows], highs, crop, batch)
        out = codec(x)
        bpp = sum(torch.log2(l).sum() for l in out["likelihoods"].values()) / -(batch * crop * crop)
        mse = F.mse_loss(refiner(out["x_hat"]), y)
        loss = lmbda * 255 ** 2 * mse + bpp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (it + 1) % 1000 == 0:
            print(f"    iter {it + 1}/{iters} loss {loss.item():.4f} mse {mse.item():.5f} bpp {bpp.item():.4f}", flush=True)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="./results/rd_curve/joint_finetune.json")
    args = parser.parse_args()

    train_pairs, test = load_eval_pairs(args.train_dir), load_eval_pairs(args.eval_dir)
    lows, highs = [to_uint8(low) for low, _, _ in train_pairs], [to_uint8(high) for _, high, _ in train_pairs]
    test_lows, test_highs = [low for low, _, _ in test], [high for _, high, _ in test]
    tag = "frozen" if args.freeze_codec else "joint"
    label = "Standard codec + refiner (codec frozen)" if args.freeze_codec else "Joint fine-tuned codec + refiner (v2)"
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
        train(codec, refiner, lows, highs, LAMBDAS[quality], args.iters, args.freeze_codec)
        evaluate(label, quality, codec, refiner)
        torch.save({"codec": codec.state_dict(), "refiner": refiner.state_dict()}, os.path.join(out_dir, f"{tag}_q{quality}.pth"))
        with open(args.out, "w") as f:  # 세션이 끊겨도 끝난 품질까지는 남긴다
            json.dump(results, f, indent=2)
    print(f"저장됨: {args.out}")


if __name__ == "__main__":
    main()
