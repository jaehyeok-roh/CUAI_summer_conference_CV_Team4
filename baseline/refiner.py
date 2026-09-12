"""
Retinexformer 를 코덱 출력 뒤의 후처리(refiner)로 재학습해, 앞단 코덱끼리 공정하게 비교한다.

  - Ours           : 우리 결합 코덱(rd_curve.py 의 OURS_RUNS, W&B 체크포인트)의 복원 영상 (이미 밝음)
  - Standard codec : CompressAI 사전학습 코덱으로 저조도 영상을 그대로 압축한 복원 영상 (어두움)

두 앞단 모두 같은 설정(LOL-v1 사전학습 가중치에서 시작, 같은 반복 횟수·학습률·크롭)으로
Retinexformer 를 our485 에서만 재학습하고 eval15 에서 평가한다. 앞단 코덱은 고정이라 BPP 는 그대로다.

출력 곡선:
  - Ours                                        / Ours + refiner
  - Standard codec + Retinexformer (pretrained) / Standard codec + refiner
    (앞의 것은 two_stage.py 의 Compress -> Enhance 와 같은 조건이라 검산용으로도 쓴다)

실행 (two_stage.py 와 같은 준비 + wandb 로그인 후, GPU 권장):
    python baseline/refiner.py --train_dir <LOL 경로>/our485 --eval_dir <LOL 경로>/eval15 --weights pretrained_weights/LOL_v1.pth
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
from models import HyperpriorWithCBAM
from rd_curve import OURS_RUNS, compute_psnr, compute_ssim, device, fetch_from_wandb, load_eval_pairs, load_state_dict_into, pad_to_64
from two_stage import load_retinexformer


def to_uint8(x):
    return (x.clamp(0, 1) * 255).round().byte().cpu()


@torch.no_grad()
def decode_all(codec, lows):
    """저조도 영상들을 코덱으로 압축·복원한다. (uint8 복원 영상 목록, 평균 BPP)"""
    images, bpp = [], 0.0
    for low in lows:
        x, h, w = pad_to_64(low)
        out = codec(x.unsqueeze(0).to(device))
        images.append(to_uint8(out["x_hat"][0, :, :h, :w]))
        bpp += sum(torch.log2(l.clamp(min=1e-9)).sum() / (-h * w) for l in out["likelihoods"].values()).item()
    return images, bpp / len(lows)


def sample_batch(inputs, targets, crop, batch):
    """inputs[k][i](k: quality, i: 이미지)와 targets[i] 의 같은 위치를 무작위로 자르고 뒤집은 배치. uint8 CPU 텐서 -> 0~1 float"""
    xs, ys = [], []
    for _ in range(batch):
        k, i = random.randrange(len(inputs)), random.randrange(len(targets))
        _, h, w = targets[i].shape
        top, left = random.randint(0, h - crop), random.randint(0, w - crop)
        x = inputs[k][i][:, top:top + crop, left:left + crop]
        y = targets[i][:, top:top + crop, left:left + crop]
        if random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device).float() / 255, torch.stack(ys).to(device).float() / 255


def finetune(enhancer, inputs, targets, iters, crop=128, batch=8, lr=1e-4):
    """inputs[k][i](k: quality, i: 이미지) -> targets[i] 로 enhancer 를 L1 재학습한다. 입력은 모두 uint8 CPU 텐서."""
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)
    enhancer.train()
    for it in range(iters):
        x, y = sample_batch(inputs, targets, crop, batch)
        loss = F.l1_loss(enhancer(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (it + 1) % 1000 == 0:
            print(f"    iter {it + 1}/{iters} L1 {loss.item():.4f}", flush=True)
    return enhancer.eval()


@torch.no_grad()
def score(images, highs, enhancer=None):
    """uint8 복원 영상(enhancer 가 있으면 통과시킨 뒤)과 정상조도 GT 의 평균 PSNR/SSIM"""
    psnr = ssim = 0.0
    for image, high in zip(images, highs):
        x, h, w = pad_to_64(image.float() / 255)
        x = x.unsqueeze(0).to(device)
        if enhancer is not None:
            x = enhancer(x)
        x = x[:, :, :h, :w].clamp(0, 1)
        high = high.unsqueeze(0).to(device)
        psnr += compute_psnr(x, high, data_range=1.0).item()
        ssim += compute_ssim(x, high, data_range=1.0).item()
    return psnr / len(highs), ssim / len(highs)


def load_codecs():
    """앞단별 [(quality, 코덱)]. 아직 끝나지 않은 Ours run 은 건너뛴다."""
    ours = []
    for quality, run_name, filename, cbam, match in OURS_RUNS.values():
        ckpt = fetch_from_wandb(run_name, filename, match)
        if ckpt is None:
            print(f"  {run_name}: 끝난 run 이 없어 건너뜀")
            continue
        model = HyperpriorWithCBAM(quality=quality, cbam_position=cbam, pretrained=False)
        load_state_dict_into(model, ckpt)
        ours.append((quality, model.eval().to(device)))
    standard = [(q, bmshj2018_hyperprior(quality=q, pretrained=True).eval().to(device)) for q in range(1, 9)]
    return {"Ours": ours, "Standard codec": standard}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./LOL_Dataset/lol_dataset/our485")
    parser.add_argument("--eval_dir", default="./LOL_Dataset/lol_dataset/eval15")
    parser.add_argument("--weights", default="./pretrained_weights/LOL_v1.pth")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="./results/rd_curve/refiner.json")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, test = load_eval_pairs(args.train_dir), load_eval_pairs(args.eval_dir)
    train_lows, train_highs = [low for low, _, _ in train], [to_uint8(high) for _, high, _ in train]
    test_lows, test_highs = [low for low, _, _ in test], [high for _, high, _ in test]
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for family, codecs in load_codecs().items():
        print(f"\n[{family}] quality {[q for q, _ in codecs]} 로 our485 복원 영상 만드는 중", flush=True)
        train_inputs, test_points = [], []
        for quality, codec in codecs:
            train_inputs.append(decode_all(codec, train_lows)[0])
            images, bpp = decode_all(codec, test_lows)
            test_points.append((quality, images, bpp))

        print(f"[{family}] Retinexformer 재학습 ({args.iters} iters)", flush=True)
        refiner = finetune(load_retinexformer(args.weights), train_inputs, train_highs, args.iters)
        torch.save(refiner.state_dict(), os.path.join(out_dir, f"refiner_{family.replace(' ', '_')}.pth"))

        # Ours 출력은 이미 밝아서 사전학습 Retinexformer 를 그대로 붙이지 않는다
        before = (family, None) if family == "Ours" else (f"{family} + Retinexformer (pretrained)", load_retinexformer(args.weights))
        for label, enhancer in (before, (f"{family} + refiner", refiner)):
            results[label] = []
            for quality, images, bpp in test_points:
                psnr, ssim = score(images, test_highs, enhancer)
                results[label].append({"quality": quality, "bpp": bpp, "psnr": psnr, "ssim": ssim})
                print(f"  {label} q={quality}: PSNR {psnr:.2f} / SSIM {ssim:.4f} / BPP {bpp:.4f}", flush=True)
        del train_inputs

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"저장됨: {args.out}")


if __name__ == "__main__":
    main()
