"""
후보 A: 경량 향상 모델 IAT 를 구조에 따라 엣지와 수신 측으로 나눈다 (split IAT).

엣지 : 원본 저조도 영상에서 IAT 의 전역 분기만 돌려 색 행렬(3x3)과 감마(1)를 구하고, 저조도 영상 자체를 기존 코덱으로 압축한다.
       영상 비트스트림에 전역 파라미터 10개(float16, 160비트)를 붙여 보낸다.
수신 : 복원한 저조도 영상에 지역 분기(mul, add)를 돌리고 받은 전역 파라미터를 적용한다. 지역 분기는 복원 영상 -> 정답으로 재학습한다.

근거: E->C 는 향상으로 커진 잡음에 비트를 쓴다(품질 1 에서 향상 영상 0.096 vs 저조도 영상 0.040 bpp). C->E 는 압축이 영상 통계를 망가뜨린다.
색·밝기를 정하는 전역 파라미터를 압축 전 원본에서 구해 보내면 둘 다 피할 수 있다. 또 전역 분기를 고정하고 지역 분기만 정답으로
학습하면 색·톤이 틀어지지 않았다(edge_finetune.py --train enhancer_local, 품질 3 PSNR 23.02 dB 유지).

평가 (eval_all.py 와 같은 8비트 경로, 결과는 eval_all.py 형식):
  Compress -> IAT (receiver)    : 수신 측에서 IAT 전체 (전역 파라미터도 복원 영상에서)
  Split IAT (pretrained local)  : 전역 파라미터만 원본에서, 지역 분기는 사전학습 그대로
  Split IAT (fine-tuned local)  : 제안

실행 (레포 루트에서, IAT 저장소의 IAT_enhance 폴더와 timm 필요):
    python baseline/split_iat.py --train_dir <LOL 경로>/our485 --eval_dir <LOL 경로>/eval15 --iat_dir <IAT>/IAT_enhance --qualities 5 3 7 1
"""
import argparse
import copy
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from compressai.zoo import bmshj2018_hyperprior

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # baseline/
from edge_finetune import IAT
from eval_all import compress, evaluate, q8
from rd_curve import device, load_eval_pairs
from refiner import decode_all, sample_batch, to_uint8

SIDE_BITS = 10 * 16  # 색 행렬 9개 + 감마 1개, float16


def apply_global(net, img, gamma, color):
    """IAT forward 의 전역 단계: 영상마다 색 행렬을 곱하고 감마를 적용한다. img (B,3,H,W), gamma (B,1), color (B,3,3)"""
    x = img.permute(0, 2, 3, 1)
    x = torch.stack([net.apply_color(x[i], color[i]) ** gamma[i] for i in range(x.shape[0])])
    return x.permute(0, 3, 1, 2)


def receiver(net, decoded, gamma, color):
    """수신 측: 복원한 저조도 영상에 지역 분기 -> 받은 전역 파라미터 적용"""
    mul, add = net.local_net(decoded)
    return apply_global(net, decoded * mul + add, gamma, color)


def train_local(net, decoded, lows, highs, iters, crop=256, batch=8, lr=1e-4):
    """지역 분기만 복원 영상 -> 정답으로 학습한다. 전역 파라미터는 같은 위치의 원본 크롭에서 구한다(IAT 학습처럼 크롭 단위).
    decoded, lows, highs: 같은 순서의 uint8 CPU 텐서 목록"""
    net.train()
    net.global_net.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(net.local_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)
    stacked = [[torch.cat([d, l]) for d, l in zip(decoded, lows)]]  # 복원·원본을 채널로 붙여 같은 위치를 자르고 뒤집는다
    total = 0.0
    for it in range(iters):
        x, y = sample_batch(stacked, highs, crop, batch)
        with torch.no_grad():
            gamma, color = net.global_net(x[:, 3:])
        loss = F.mse_loss(receiver(net, x[:, :3], gamma, color), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if (it + 1) % 1000 == 0:
            print(f"    iter {it + 1}/{iters} (최근 1000 평균) mse {total / 1000:.5f}", flush=True)
            total = 0.0
    net.eval()


def split_pipeline(net, codec):
    """저조도 영상 -> (엣지: 전역 파라미터 + 저조도 영상 압축) -> (수신: 복원 영상에 지역 분기 + 전역 파라미터). bpp 에 전역 파라미터 비트 포함"""
    @torch.no_grad()
    def run(low):
        decoded, bpp = compress(codec, low)
        gamma, color = net.global_net(low.unsqueeze(0).to(device))
        return q8(receiver(net, decoded.unsqueeze(0), gamma, color)[0]), bpp + SIDE_BITS / (low.shape[1] * low.shape[2])
    return run


def codec_then_iat(net, codec):
    """저조도 영상 압축 -> 수신 측에서 IAT 전체"""
    @torch.no_grad()
    def run(low):
        decoded, bpp = compress(codec, low)
        return q8(net(decoded.unsqueeze(0))[2][0]), bpp
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./LOL_Dataset/lol_dataset/our485")
    parser.add_argument("--eval_dir", default="./LOL_Dataset/lol_dataset/eval15")
    parser.add_argument("--iat_dir", default="./Illumination-Adaptive-Transformer/IAT_enhance")
    parser.add_argument("--qualities", type=int, nargs="+", default=[5, 3, 7, 1])
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="./results/rd_curve")
    args = parser.parse_args()

    train_pairs, test = load_eval_pairs(args.train_dir), load_eval_pairs(args.eval_dir)
    train_lows = [low for low, _, _ in train_pairs]
    lows, highs = [to_uint8(low) for low in train_lows], [to_uint8(high) for _, high, _ in train_pairs]
    net = IAT(args.iat_dir).net.to(device).eval()
    x = test[0][0].unsqueeze(0).to(device)
    with torch.no_grad():  # 나눠 돌려도 원래 IAT 와 출력이 같아야 한다
        assert torch.allclose(receiver(net, x, *net.global_net(x)), net(x)[2], atol=1e-5), "split IAT 가 원래 IAT 와 다르다"
    pretrained_local = copy.deepcopy(net.local_net.state_dict())
    os.makedirs(args.out_dir, exist_ok=True)
    for quality in args.qualities:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        codec = bmshj2018_hyperprior(quality=quality, pretrained=True).to(device).eval()
        net.local_net.load_state_dict(pretrained_local)
        results = evaluate({"Compress -> IAT (receiver)": [(quality, lambda: codec_then_iat(net, codec))],
                            "Split IAT (pretrained local)": [(quality, lambda: split_pipeline(net, codec))]}, test)
        decoded, bpp = decode_all(codec, train_lows)
        print(f"[q={quality}] 학습 영상 복원 평균 bpp {bpp:.4f}, 지역 분기 재학습 ({args.iters} iters)", flush=True)
        train_local(net, decoded, lows, highs, args.iters)
        results.update(evaluate({"Split IAT (fine-tuned local)": [(quality, lambda: split_pipeline(net, codec))]}, test))
        torch.save(net.local_net.state_dict(), os.path.join(args.out_dir, f"split_local_q{quality}.pth"))
        with open(os.path.join(args.out_dir, f"split_q{quality}.json"), "w") as f:  # 품질마다 따로 저장해 세션이 끊겨도 끝난 것은 남긴다
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
