"""
B: 엣지의 경량 향상 모델(IAT) + 기존 코덱을 미세조정한다.

경량 IAT 를 엣지에 둔 Enhance -> Compress 가 수신 측 공동 미세조정(v2 STE)보다 좋았다(같은 비트에서 +0.84 dB).
여기서는 그 조합에서 출발해 IAT 와 코덱을 미세조정한다. 손실은 λ·255²·MSE(복원, 목표) + bpp 이고,
λ 와 학습률 비율(코덱은 1/10)은 v2 와 같으며 복원 경로는 STE(평가처럼 반올림)다.

  --train both            : IAT + 코덱
  --train codec           : IAT 고정, 코덱만
  --train enhancer        : 코덱 고정, IAT 만
  --train enhancer_local  : 코덱 고정, IAT 의 지역 분기만 (색 행렬·감마를 정하는 전역 분기는 고정)
  --target gt       : 목표 = 정상조도 정답 (기본)
  --target input    : 목표 = 코덱 입력(IAT 출력). 표준 RD 손실로 코덱만 향상된 영상 분포에 맞춘다 (--train codec 전용)
  --target teacher  : 목표 = 학습 전(사전학습) IAT 출력. 향상 결과는 사전학습 IAT 를 따르되, IAT 가 압축하기 좋은 출력을 내도록 같이 배운다
  --target aligned  : 목표 = 정답의 색·톤을 코덱 입력(IAT 출력)에 affine 으로 맞춘 영상 (--train codec 전용, 크롭마다 맞춘다).
                      톤은 IAT 를 따르고 구조는 잡음 없는 정답이라, 코덱이 향상으로 커진 잡음을 복원 대상에서 빼도록 배운다
  --source gt       : 코덱 입력을 IAT 출력 대신 정상조도 정답 영상으로 둔다 (--train codec --target input 전용).
                      "향상된 영상에 맞춰서" 비트가 주는지, "LOL 장면에 맞춰서" 주는지 가르는 대조군
  --enhancer zerodce : IAT 대신 Zero-DCE++ (쌍 데이터 없이 학습한 초경량 향상 모델) 를 엣지 향상 모델로 쓴다.
                       결과가 IAT·LOL 학습 모델에만 해당하는지 확인하는 용도 (체크포인트·결과 이름에 _zerodce)

지금까지 LOL-v1 결과:
  정답 목표: 사전학습 IAT -> Compress 보다 비트가 늘고 PSNR 이 1.7~2.3 dB 떨어졌다(SSIM 은 비슷). 출력 밝기가 정답보다 5.7% 밝았지만
             평균 밝기를 맞춘 GT-mean PSNR 도 똑같이 떨어져, 원인은 전체 밝기가 아니라 LOL 학습셋 정답의 색·톤에 맞춰진 것이다.
  IAT 출력 목표(코덱만): 사전학습 IAT 출력 대비 같은 PSNR/SSIM 에 드는 비트 -9.4% / -10.9% (LOL-v1, 15/15 장),
                        -3.2% / -8.4% (LOL-v2-synthetic, 82/100, 93/100 장). 정답 대비로는 LOL-v1 에서만 드러난다.
  정답 영상으로 코덱만 적응(--source gt): 정답 대비 LOL-v1 에서 IAT 출력 목표와 같거나 조금 낫다 -> 이득은 향상 영상 전용이 아니라 도메인 적응이다.
  사전학습 IAT 출력 목표 공동 학습(teacher): 품질 3 에서 코덱만 적응보다 PSNR 0.4 dB 낮다.

평가는 eval_all.py 와 같은 경로(IAT 출력 8비트 -> 코덱 -> 8비트)이고, 결과는 eval_all.py 형식({이름: {품질: [[이미지, bpp, PSNR, SSIM], ...]}})이다.

실행 (레포 루트에서, IAT 저장소 https://github.com/cuiziteng/Illumination-Adaptive-Transformer 의 IAT_enhance 폴더와 timm 필요.
Zero-DCE++ 는 https://github.com/Li-Chongyi/Zero-DCE_extension 의 Zero-DCE++ 폴더):
    python baseline/edge_finetune.py --train_dir <LOL 경로>/our485 --eval_dir <LOL 경로>/eval15 --iat_dir <IAT>/IAT_enhance --train codec --target input --qualities 1 3 5
"""
import argparse
import copy
import json
import os
import random
import sys
import types

import torch
import torch.nn.functional as F
from compressai.zoo import bmshj2018_hyperprior

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # baseline/
from eval_all import compress, evaluate, q8
from joint_finetune import LAMBDAS, forward
from rd_curve import device, load_eval_pairs
from refiner import sample_batch, to_uint8

LABELS = {"both": "Joint fine-tuned IAT + codec", "codec": "IAT frozen + fine-tuned codec", "enhancer": "Fine-tuned IAT + codec frozen",
          "enhancer_local": "Fine-tuned IAT local branch + codec frozen"}
TARGETS = {"gt": "", "input": " (target: IAT output)", "teacher": " (target: pretrained IAT output)", "aligned": " (target: tone-aligned GT)"}


class IAT(torch.nn.Module):
    """IAT(BMVC 2022) LOL-v1 가중치. evaluation_lol_v1.py 와 같게 정규화 없는 0~1 입력, 출력은 (mul, add, 향상 영상) 의 세 번째"""
    def __init__(self, iat_dir):
        super().__init__()
        sys.modules.setdefault("imp", types.ModuleType("imp"))  # global_net.py 의 쓰지 않는 import imp (Python 3.12 에서 삭제됨)
        sys.path.insert(0, iat_dir)
        from model.IAT_main import IAT as Net
        self.net = Net()
        self.net.load_state_dict(torch.load(os.path.join(iat_dir, "best_Epoch_lol_v1.pth"), map_location="cpu"))

    def forward(self, x):
        return self.net(x)[2]


class ZeroDCE(torch.nn.Module):
    """Zero-DCE++ (TPAMI 2021) 공식 가중치 snapshots_Zero_DCE++/Epoch99.pth. 0~1 입력, 곡선 추정은 원래 해상도(scale_factor 1)에서 한다
    (공식 추론은 큰 영상에서 속도 때문에 12 배 줄이지만, 그러려면 영상 크기가 12 의 배수여야 해서 크롭 학습과 맞지 않는다)"""
    def __init__(self, zerodce_dir):
        super().__init__()
        sys.path.insert(0, zerodce_dir)
        from model import enhance_net_nopool
        self.net = enhance_net_nopool(1)
        self.net.load_state_dict(torch.load(os.path.join(zerodce_dir, "snapshots_Zero_DCE++", "Epoch99.pth"), map_location="cpu"))

    def forward(self, x):
        return self.net(x)[0]


def align_tone_batch(src, ref):
    """src (B, 3, H, W) 의 색·톤을 ref 에 맞춘다: 영상마다 [r, g, b, 1] -> ref 로 가는 affine(4x3) 최소제곱. 0~1 float"""
    x = torch.cat([src, torch.ones_like(src[:, :1])], 1).flatten(2).transpose(1, 2).double()
    a = torch.linalg.lstsq(x, ref.flatten(2).transpose(1, 2).double()).solution
    return (x @ a).transpose(1, 2).reshape(src.shape).float().clamp(0, 1)


def train(enhancer, codec, lows, highs, lmbda, iters, part, target="gt", source="low", crop=256, batch=8, lr=1e-4):
    """lows[i] -> enhancer -> 코덱 -> 목표(target: gt / input / teacher / aligned, 모듈 설명 참고). part: both / codec / enhancer / enhancer_local.
    source=gt 면 코덱 입력이 highs[i] 이고 enhancer 를 거치지 않는다. 입력은 uint8 CPU 텐서, crop 은 64 의 배수여야 한다."""
    teacher = copy.deepcopy(enhancer).eval().requires_grad_(False) if target == "teacher" else None  # 학습 전 enhancer
    tune_enhancer, tune_codec = part != "codec", part in ("both", "codec")
    enhancer.train(tune_enhancer).requires_grad_(tune_enhancer)  # 고정이면 eval 모드로 BatchNorm 통계도 그대로 둔다
    if part == "enhancer_local":  # 색·톤을 정하는 전역 분기(색 행렬·감마, BatchNorm 포함)는 학습 전 그대로 둔다
        enhancer.net.global_net.eval().requires_grad_(False)
    codec.train().requires_grad_(tune_codec)  # 고정이어도 train 모드: rate 가 노이즈 근사라야 bpp 의 gradient 가 IAT 까지 흐른다
    groups = [{"params": [p for p in enhancer.parameters() if p.requires_grad], "lr": lr}] if tune_enhancer else []
    if tune_codec:
        groups.append({"params": list(codec.parameters()), "lr": lr / 10})  # 사전학습 코덱은 작은 학습률로 (v2 와 같음)
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)
    sums = [0.0, 0.0, 0.0]
    for it in range(iters):
        x, y = sample_batch([lows if source == "low" else highs], highs, crop, batch)
        x_in = enhancer(x) if source == "low" else x
        x_hat, likelihoods = forward(codec, x_in, ste=True)
        bpp = sum(torch.log2(l).sum() for l in likelihoods.values()) / -(batch * crop * crop)
        if target == "gt":
            ref = y
        elif target == "input":
            ref = x_in.detach()
        elif target == "aligned":  # 정답의 구조(잡음 없음)에 IAT 출력의 색·톤. IAT 가 크롭 통계로 톤을 정해서 크롭마다 맞춘다
            with torch.no_grad():
                ref = align_tone_batch(y, x_in)
        else:  # LOL 정답의 색·톤을 새로 배우지 않게, 목표는 사전학습 IAT 출력으로 둔다
            with torch.no_grad():
                ref = teacher(x)
        mse = F.mse_loss(x_hat, ref)
        loss = lmbda * 255 ** 2 * mse + bpp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        sums = [s + v.item() for s, v in zip(sums, (loss, mse, bpp))]
        if (it + 1) % 1000 == 0:  # 배치 하나는 크롭에 따라 크게 흔들려서 최근 1000 iter 평균을 찍는다
            print(f"    iter {it + 1}/{iters} (최근 1000 평균) loss {sums[0] / 1000:.4f} mse {sums[1] / 1000:.5f} bpp {sums[2] / 1000:.4f}", flush=True)
            sums = [0.0, 0.0, 0.0]
    enhancer.eval()
    codec.eval()


def enhance_then_compress(enhancer, codec):
    """저조도 영상 -> enhancer -> 코덱. IAT 는 전역 분기가 영상 통계를 써서 패딩 없이 전체 영상에 적용한다"""
    @torch.no_grad()
    def run(low):
        return compress(codec, q8(enhancer(low.unsqueeze(0).to(device))[0]))
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./LOL_Dataset/lol_dataset/our485")
    parser.add_argument("--eval_dir", default="./LOL_Dataset/lol_dataset/eval15")
    parser.add_argument("--iat_dir", default="./Illumination-Adaptive-Transformer/IAT_enhance")
    parser.add_argument("--enhancer", choices=("iat", "zerodce"), default="iat")
    parser.add_argument("--zerodce_dir", default="./Zero-DCE_extension/Zero-DCE++")
    parser.add_argument("--train", choices=LABELS, default="both")
    parser.add_argument("--target", choices=TARGETS, default="gt")
    parser.add_argument("--source", choices=("low", "gt"), default="low")
    parser.add_argument("--qualities", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="./results/rd_curve")
    args = parser.parse_args()
    if args.target in ("input", "aligned") and args.train != "codec":
        parser.error("--target input/aligned 는 --train codec 에서만 쓴다 (향상 모델까지 학습하면 목표가 같이 움직인다. 그럴 땐 --target teacher)")
    if args.source == "gt" and (args.train, args.target) != ("codec", "input"):
        parser.error("--source gt 는 --train codec --target input 에서만 쓴다 (정답 영상 도메인에 코덱만 맞추는 대조군)")
    if args.enhancer == "zerodce" and args.train == "enhancer_local":
        parser.error("--train enhancer_local 은 지역·전역 분기가 있는 IAT 전용이다")

    train_pairs, test = load_eval_pairs(args.train_dir), load_eval_pairs(args.eval_dir)
    lows, highs = [to_uint8(low) for low, _, _ in train_pairs], [to_uint8(high) for _, high, _ in train_pairs]
    label = LABELS[args.train] + TARGETS[args.target] + (" (adapted on GT images)" if args.source == "gt" else "")
    if args.enhancer == "zerodce":
        label = label.replace("IAT", "Zero-DCE++")
    os.makedirs(args.out_dir, exist_ok=True)
    for quality in args.qualities:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        codec = bmshj2018_hyperprior(quality=quality, pretrained=True).to(device)
        enhancer = (IAT(args.iat_dir) if args.enhancer == "iat" else ZeroDCE(args.zerodce_dir)).to(device)
        print(f"[q={quality}] {label} 학습 ({args.iters} iters, lambda {LAMBDAS[quality]})", flush=True)
        train(enhancer, codec, lows, highs, LAMBDAS[quality], args.iters, args.train, args.target, args.source)
        name = (f"edge_{args.train}" + ("" if args.target == "gt" else f"_{args.target}") + ("_gtsrc" if args.source == "gt" else "")
                + ("" if args.enhancer == "iat" else f"_{args.enhancer}") + (f"_seed{args.seed}" if args.seed else "") + f"_q{quality}")
        torch.save({"codec": codec.state_dict(), "enhancer": enhancer.state_dict()}, os.path.join(args.out_dir, f"{name}.pth"))
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:  # 품질마다 따로 저장해 세션이 끊겨도 끝난 것은 남긴다
            json.dump(evaluate({label: [(quality, lambda: enhance_then_compress(enhancer, codec))]}, test), f, indent=1)


if __name__ == "__main__":
    main()
