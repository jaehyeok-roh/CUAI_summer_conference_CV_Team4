"""
2단계 baseline: 사전학습 저조도 향상 모델(Retinexformer, LOL-v1)과 사전학습 CompressAI 코덱을 이어 붙인다.

  - Enhance -> Compress : 향상한 뒤 압축 (엣지 기기에서 향상 모델까지 실행)
  - Compress -> Enhance : 저조도 영상을 그대로 압축하고 수신 측에서 향상 (엣지 기기는 코덱만 실행)
  - Enhance only        : 압축 없이 향상만 한 상한선

eval15, PSNR/SSIM, BPP(likelihood 기반) 계산은 rd_curve.py 와 동일하다.

준비 (Kaggle 셀, 레포 루트에서):
    !pip install -q einops gdown
    !wget -q https://raw.githubusercontent.com/caiyuanhao1998/Retinexformer/master/basicsr/models/archs/RetinexFormer_arch.py -O baseline/RetinexFormer_arch.py
    !gdown -q --folder https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV -O pretrained_weights

실행:
    python baseline/two_stage.py --eval_dir <LOL 경로>/eval15 --weights pretrained_weights/LOL_v1.pth
"""
import argparse
import json
import os
import sys

import torch
from compressai.zoo import bmshj2018_hyperprior

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트 (rd_curve)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # baseline/ (RetinexFormer_arch)
from rd_curve import load_eval_pairs, pad_to_64, compute_psnr, compute_ssim, device


def build_retinexformer():
    from RetinexFormer_arch import RetinexFormer
    # Retinexformer 레포 Options/RetinexFormer_LOL_v1.yml 의 network_g 설정
    return RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])


def load_retinexformer(weights):
    model = build_retinexformer()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state.get("params", state))
    return model.eval().to(device)


@torch.no_grad()
def evaluate_pipeline(pairs, enhancer, codec, order):
    """order: "E"(향상만), "EC"(향상 -> 압축), "CE"(압축 -> 향상). 평균 PSNR/SSIM/BPP 를 돌려준다."""
    total = {"psnr": 0.0, "ssim": 0.0, "bpp": 0.0}
    for low, high, _ in pairs:
        x, h, w = pad_to_64(low)
        x, high = x.unsqueeze(0).to(device), high.unsqueeze(0).to(device)
        if order == "EC":
            x = enhancer(x).clamp(0, 1)
        if order in ("EC", "CE"):
            out = codec(x)
            x = out["x_hat"].clamp(0, 1)
            total["bpp"] += sum(
                torch.log2(l.clamp(min=1e-9)).sum() / (-h * w) for l in out["likelihoods"].values()
            ).item()
        if order in ("E", "CE"):
            x = enhancer(x).clamp(0, 1)
        x = x[:, :, :h, :w]
        total["psnr"] += compute_psnr(x, high, data_range=1.0).item()
        total["ssim"] += compute_ssim(x, high, data_range=1.0).item()
    return {k: v / len(pairs) for k, v in total.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="./LOL_Dataset/lol_dataset/eval15")
    parser.add_argument("--weights", default="./pretrained_weights/LOL_v1.pth")
    parser.add_argument("--out", default="./results/rd_curve/two_stage.json")
    args = parser.parse_args()

    pairs = load_eval_pairs(args.eval_dir)
    enhancer = load_retinexformer(args.weights)

    m = evaluate_pipeline(pairs, enhancer, None, "E")
    print(f"Enhance only: PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.4f}")
    results = {"Enhance only": m, "Enhance -> Compress": [], "Compress -> Enhance": []}

    for q in range(1, 9):
        codec = bmshj2018_hyperprior(quality=q, pretrained=True).eval().to(device)
        for order, label in (("EC", "Enhance -> Compress"), ("CE", "Compress -> Enhance")):
            m = evaluate_pipeline(pairs, enhancer, codec, order)
            m["quality"] = q
            results[label].append(m)
            print(f"  {label} q={q}: PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.4f} / BPP {m['bpp']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"저장됨: {args.out}")


if __name__ == "__main__":
    main()
